CUDA_VISIBLE_DEVICES=6 python test/warping.py \
    --pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
    --transformer_model_name_or_path="preset/models/dit4sr_q" \
    --output_dir results/ \
    --prompt_path /mnt/dataset2/jaewon/YouHQ/YouHQ-Train-prompts