export CUDA_VISIBLE_DEVICES=4

idx=0
num=7000

start=$((10000 + idx * num))
stop=$((10000 + (idx + 1) * num))

python utils_data/make_embedding_youhq.py \
    --pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
    --root_path /mnt/dataset2/jaewon/YouHQ/YouHQ-Train-prompts