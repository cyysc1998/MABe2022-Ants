export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 mabe/val.py -opt options/34_all_data.yml