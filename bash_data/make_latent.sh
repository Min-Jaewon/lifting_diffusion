export CUDA_VISIBLE_DEVICES=7

idx=0
num=17000

start=$(((idx * num)+(6500)))
stop=$(((idx + 1) * num))

python utils_data/make_latents_youhq.py \
    --root_path /mnt/dataset2/jaewon/YouHQ/YouHQ-Train-frames \
    --save_path /mnt/dataset2/jaewon/YouHQ/YouHQ-Train-latents