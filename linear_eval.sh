export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=4322 mabe/linear_eval.py -opt options/14_2nt_w.yml