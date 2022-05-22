export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=4322 mabe/test.py -opt options/inference/06_n_views_10k_iter.yml &&
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=4322 mabe/test.py -opt options/inference/06_n_views_20k_iter.yml &&
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=4322 mabe/test.py -opt options/inference/06_n_views_30k_iter.yml
