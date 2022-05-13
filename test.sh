export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4322 mabe/test.py -opt options/inference/01_more_frame_70k_iter.yml &&
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4322 mabe/test.py -opt options/inference/01_more_frame_90k_iter.yml