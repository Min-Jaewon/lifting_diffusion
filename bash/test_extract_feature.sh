CUDA_VISIBLE_DEVICES=5 python test/extract_feature.py \
    --pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
    --transformer_model_name_or_path="preset/models/dit4sr_q" \
    --image_path /mnt/dataset2/jaewon/YouHQ/YouHQ-Train-LQ-Ours \
    --output_dir results/ \
    --prompt_path /mnt/dataset2/jaewon/YouHQ/YouHQ-Train-prompts