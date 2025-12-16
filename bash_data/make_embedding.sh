CUDA_VISIBLE_DEVICES=7
python utils_data/make_embedding_youhq.py \
    --pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
    --root_path /mnt/dataset1/m_jaewon/cvpr26/data/YouHQ-Train-prompts \
    --start_num 0 \
    --end_num 5005 