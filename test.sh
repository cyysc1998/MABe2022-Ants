export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4322 mabe/test.py -opt options/inference/03_re_60k_iter.yml &&
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4322 mabe/test.py -opt options/inference/03_re_100k_iter.yml &&
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4322 mabe/test.py -opt options/inference/04_guassian_blur_20k_iter.yml &&
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4322 mabe/test.py -opt options/inference/04_guassian_blur_90k_iter.yml