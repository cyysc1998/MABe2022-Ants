export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=4322 mabe/test.py -opt options/29_more_aug.yml
