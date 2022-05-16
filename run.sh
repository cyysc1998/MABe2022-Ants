export PYTHONPATH=$PYTHONPATH:.
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 mabe/linear_eval.py -opt options/03_re.yml &&
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 mabe/linear_eval.py -opt options/04_guassian_blur.yml