export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4322 mabe/test.py -opt options/inference/10_s3d_20k_iter.yml &&
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4322 mabe/test.py -opt options/inference/10_s3d_40k_iter.yml &&
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4322 mabe/test.py -opt options/inference/10_s3d_70k_iter.yml