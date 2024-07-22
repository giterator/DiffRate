python3 main.py \
--arch-lr 0.01 --arch-min-lr 0.001 \
--epoch 3 --batch-size 64 \
--data-path /home/shivam/datasets/imagenet \
--output_dir /home/pranav/DiffRate/learnt/LSMS \
--model vit_large_patch16_mae \
--target_thru 75.0 \
--target_batch_size 8