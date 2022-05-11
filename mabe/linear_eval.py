import logging
import time

import numpy as np
import torch

from mabe.data import create_dataset_dataloader
from mabe.models import create_model
from mabe.train import init_loggers, parse_options
from mabe.utils import MessageLogger


def main():
    opt = parse_options()
    seed = opt["manual_seed"]
    linear_eval_opt = opt["linear_eval"]
    linear_eval_opt["rank"] = opt["rank"]
    linear_eval_opt["world_size"] = opt["world_size"]

    # initialize loggers
    logger, tb_logger = init_loggers(
        opt, prefix="linear_eval", log_level=logging.WARN, use_tb_logger=False
    )
    indexes = np.loadtxt(linear_eval_opt["meta_path"]).astype(int)

    for current_iter in range(
        linear_eval_opt["st_iter"],
        linear_eval_opt["ed_iter"] + 1,
        linear_eval_opt["linear_eval_freq"],
    ):
        ckpt_path = f"{opt['path']['models']}/net_{current_iter}.pth"
        feats_path = ckpt_path.replace(".pth", "_val_feats.npy")
        labels_path = ckpt_path.replace(".pth", "_val_labels.npy")
        feats = torch.from_numpy(np.load(feats_path)).cuda()
        labels = torch.from_numpy(np.load(labels_path)).cuda()

        num_seeds = linear_eval_opt["num_seeds"]
        num_subtasks = linear_eval_opt["num_subtasks"]
        results = torch.zeros((num_seeds, num_subtasks))
        for seed in range(num_seeds):
            index = indexes[seed]
            num = labels.shape[0]
            num_train = int(num * 0.6)
            num_val = int(num * 0.2)

            train_index = index[:num_train]
            val_index = index[num_train : num_train + num_val]
            test_index = index[num_train + num_val :]

            for subtask in range(num_subtasks):
                train_loader = [
                    {"x": feats[train_index], "label": labels[train_index, subtask]}
                ]
                val_loader = [
                    {"x": feats[val_index], "label": labels[val_index, subtask]}
                ]
                test_loader = [
                    {"x": feats[test_index], "label": labels[test_index, subtask]}
                ]

                lr_list = linear_eval_opt["lr_list"]
                lr_results = []
                for lr in lr_list:

                    # create model
                    linear_eval_opt["train"] = {
                        "optim": {
                            "type": "SGD",
                            "lr": lr,
                        }
                    }
                    model = create_model(linear_eval_opt)

                    epoch_val_results = [model.test(None, val_loader)]
                    epoch_test_results = [model.test(None, test_loader)]
                    for epoch in list(range(linear_eval_opt["total_epoch"])):
                        data_time = 0
                        data_st, epoch_st = time.time(), time.time()
                        # train
                        for train_data in train_loader:
                            data_time += time.time() - data_st
                            model.feed_data(train_data)
                            model.optimize_parameters_amp()
                            data_st = time.time()
                        epoch_time = time.time() - epoch_st
                        # val and test
                        epoch_val_results.append(model.test(None, val_loader))
                        epoch_test_results.append(model.test(None, test_loader))
                        # log
                        # logger.warn(f"epoch: {epoch}\t"
                        #             f"lr: {lr:.0e}\t"
                        #             f"time: {epoch_time:.3f}\t"
                        #             f"data_time: {data_time:.3f}")
                    epoch = np.argmax(epoch_val_results)
                    result = epoch_test_results[epoch]
                    lr_results.append(result)
                    logger.warn(f"lr: {lr:e}, epoch: {epoch}, result: {result:.5f}")
                    logger.warn(f"val: {epoch_val_results}")
                    logger.warn(f"test: {epoch_test_results}")
                idx = np.argmax(lr_results)
                lr = lr_list[idx]
                result = lr_results[idx]
                logger.warn(
                    f"seed: {seed}, subtask: {subtask}, lr: {lr:.0e}, result: {result:.4f}"
                )
                logger.warn(f"lr: {lr_list}")
                logger.warn(f"test: {lr_results}")
                results[seed][subtask] = result
        logger.warn(
            f"current_iter: {current_iter}, ave: {results.mean():.4f}, {results}"
        )


if __name__ == "__main__":
    main()
