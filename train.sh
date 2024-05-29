python3 -m torch.distributed.launch \
--nproc_per_node=1 --use_env  \
--master_port 29513 main.py \
--arch-lr 0.01 --arch-min-lr 0.001 \
--epoch 3 --batch-size 256 \
--data-path /home/datasets/imagenet/imagenet/ \
--output_dir /home/pranav/ECCV_workshop/DiffRate/learnt \
--model vit_deit_small_patch16_224 \
--target_flops 2.9