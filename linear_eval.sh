export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=5432 mabe/linear_eval.py -opt options/06_n_views.yml