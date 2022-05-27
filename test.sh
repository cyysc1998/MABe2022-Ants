export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4322 mabe/test.py -opt options/inference/12_2n_views_40k_iter.yml &&
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4322 mabe/test.py -opt options/inference/12_2n_views_60k_iter.yml &&
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4322 mabe/test.py -opt options/inference/12_2n_views_80k_iter.yml