import argparse
import datetime
import logging
import math
import faiss
from tqdm import tqdm
import time
from os import path as osp

import torch
import torch.nn as nn
import torch.distributed as dist

import numpy as np

from mabe.data import create_dataset_dataloader
from mabe.models import create_model
from mabe.utils import (
    MessageLogger,
    dict2str,
    get_env_info,
    get_root_logger,
    get_time_str,
    init_dist,
    init_tb_logger,
    init_wandb_logger,
    make_exp_dirs,
    parse,
    set_random_seed,
)


def parse_options():
    """
    parse options
    set distributed setting
    set ramdom seed
    set cudnn deterministic
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt", type=str, required=True, help="Path to option YAML file."
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt)

    # distributed setting
    init_dist()
    opt["rank"] = dist.get_rank()
    opt["world_size"] = dist.get_world_size()

    # random seed
    seed = opt.get("manual_seed")
    assert seed is not None, "Seed must be set."
    set_random_seed(seed + opt["rank"])

    # cudnn deterministic
    if opt["cudnn_deterministic"]:
        # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    return opt


def init_loggers(opt, prefix, log_level, use_tb_logger):
    log_file = osp.join(
        opt["path"]["log"], f"{prefix}_{opt['name']}_{get_time_str()}.log"
    )
    logger = get_root_logger(log_level=log_level, log_file=log_file, initialized=False)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize tensorboard logger and wandb logger
    tb_logger = None
    if use_tb_logger:
        tb_logger = init_tb_logger(log_dir=opt["path"]["tb_logger"])
        if opt["logger"]["use_wandb"]:
            init_wandb_logger(opt)
    return logger, tb_logger


def compute_features(eval_loader, model, opt):
    print('Computing features...')
    encoder_q = model.net.module.encoder_q.cuda()
    encoder_q.eval()
    features = torch.zeros(len(eval_loader.dataset), opt["common"]["out_emb_size"]).cuda()
    for i, data in enumerate(tqdm(eval_loader)):
        images = data["x1"].float().cuda()
        index = data["idx"]
        with torch.no_grad():
            images = images.cuda(non_blocking=True)
            feat = encoder_q(images) 
            features[index] = feat
    dist.barrier()        
    dist.all_reduce(features, op=dist.ReduceOp.SUM)     
    return features.cpu()


def run_kmeans(x, opt):
    """
    Args:
        x: data to be clustered
    """
    
    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[]}
    
    for seed, num_cluster in enumerate(opt["network"]["num_cluster"]):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = torch.distributed.get_rank()  
        index = faiss.GpuIndexFlatL2(res, d, cfg)  

        clus.train(x, index)   

        D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]
        
        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
        
        # sample-to-centroid distances for each cluster 
        Dcluster = [[] for c in range(k)]          
        for im,i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])
        
        # concentration estimation (phi)        
        density = np.zeros(k)
        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
                density[i] = d     
                
        #if cluster only has one point, use the max to estimate its concentration        
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax 

        density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
        density = opt["network"]["T"]*density/density.mean()  #scale the mean to temperature 
        
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)    

        im2cluster = torch.LongTensor(im2cluster).cuda()               
        density = torch.Tensor(density).cuda()
        
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)    
        
    return results


def main():
    opt = parse_options()
    seed = opt["manual_seed"]

    # mkdir for experiments and logger
    make_exp_dirs(opt)

    # initialize loggers
    logger, tb_logger = init_loggers(
        opt, prefix="train", log_level=logging.INFO, use_tb_logger=True
    )

    # create train, validation, test datasets and dataloaders
    train_set, train_loader, num_iter_per_epoch = create_dataset_dataloader(
        opt["datasets"]["train"], shuffle=True, seed=seed
    )
    val_set, val_loader, _ = create_dataset_dataloader(
        opt["datasets"]["val"], shuffle=False, seed=seed
    )
    test_set, test_loader, _ = create_dataset_dataloader(
        opt["datasets"]["test"], shuffle=False, seed=seed
    )

    cluster_set, cluster_loader, _ = create_dataset_dataloader(
        opt["datasets"]["cluster"], shuffle=True, seed=seed
    )

    # log training statistics
    total_iters = int(opt["train"]["total_iter"])
    total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
    logger.info(f"\n\tTotal epochs: {total_epochs}\n\tTotal iters: {total_iters}.")

    # create model
    model = create_model(opt)
    start_epoch = 0
    current_iter = 0
    cluster_result = None

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # training
    logger.info(f"Start training from epoch: {start_epoch}, iter: {current_iter}")
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    logger.info("Save model")
    model.save(0, 0)
    # logger.info("Validate")
    # model.test(val_set, val_loader)
    # model.save_result(0, 0, "val")
    # logger.info("Test")
    # model.test(test_set, test_loader)
    # model.save_result(0, 0, "test")
    for epoch in range(start_epoch, total_epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        for train_data in train_loader:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt["train"].get("warmup_iter", -1)
            )
            # clusting
            if current_iter % opt["network"]["cluster_interval"] == 0 and current_iter > opt["train"]["warmup_iter"]:
            # if current_iter > opt["train"]["warmup_iter"]:
                # compute momentum features for center-cropped images
                features = compute_features(cluster_loader, model, opt)         
                # placeholder for clustering result
                cluster_result = {'im2cluster':[],'centroids':[],'density':[]}
                for num_cluster in opt["network"]["num_cluster"]:
                    cluster_result['im2cluster'].append(torch.zeros(len(cluster_set),dtype=torch.long).cuda())
                    cluster_result['centroids'].append(torch.zeros(int(num_cluster), opt["network"]["out_emb_size"]).cuda())
                    cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda()) 

                # if dist.get_rank() == 0:
                #     features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
                #     features = features.numpy()
                #     cluster_result = run_kmeans(features, opt)  #run kmeans clustering on master node
                #     # save the clustering result
                #     # torch.save(cluster_result,os.path.join(args.exp_dir, 'clusters_%d'%epoch))  

                features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
                features = features.numpy()
                cluster_result = run_kmeans(features, opt)  #run kmeans clustering on master node

                dist.barrier()  
                # broadcast clustering result
                for k, data_list in cluster_result.items():
                    for data_tensor in data_list:           
                        dist.broadcast(data_tensor, 0, async_op=False)     



            # training
            model.feed_data(train_data, train=True)
            model.optimize_parameters_amp(current_iter)
            iter_time = time.time() - iter_time
            # log
            if current_iter % opt["logger"]["print_freq"] == 0:
                log_vars = {"epoch": epoch, "iter": current_iter}
                log_vars.update({"lrs": model.get_current_learning_rate()})
                log_vars.update({"time": iter_time, "data_time": data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)
            # save model, validation and test
            if current_iter % opt["logger"]["save_checkpoint_freq"] == 0:
                logger.info("Save model")
                model.save(epoch, current_iter)
                # logger.info("Validate")
                # model.test(val_set, val_loader)
                # model.save_result(epoch, current_iter, "val")
                # logger.info("Test")
                # model.test(test_set, test_loader)
                # model.save_result(epoch, current_iter, "test")
            data_time = time.time()
            iter_time = time.time()
        # end of iter
    # end of epoch

    logger.info("Save model")
    model.save(-1, -1)  # -1 stands for the latest
    logger.info("Validate")
    model.test(val_set, val_loader)
    model.save_result(-1, -1, "val")
    logger.info("Test")
    model.test(test_set, test_loader)
    model.save_result(-1, -1, "test")
    if tb_logger is not None:
        tb_logger.close()
    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f"End of training. Time consumed: {consumed_time}")


if __name__ == "__main__":
    main()
