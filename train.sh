python3 main.py \
--arch-lr 0.01 --arch-min-lr 0.001 \
--epoch 3 --batch-size 256 \
--data-path /home/shivam/datasets/imagenet \
--output_dir /home/pranav/DiffRate/learnt/LSMS \
--model vit_deit_small_patch16_224 \
--target_thru 74.0 \
--target_batch_size 8