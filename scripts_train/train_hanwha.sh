export CUDA_VISIBLE_DEVICES=4

LR=5e-5
CHANNELS=256
WANDB_NAME="TEST"
# Final : DPT + Vision Encoder + SEA-RAFT + Custom DPT + Context DPT
accelerate launch --config_file single-gpu.yaml --main_process_port 29503 train/train_lifting_hanwha.py \
    --report_to wandb \
    --pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
    --transformer_model_name_or_path="preset/models/dit4sr_q" \
    --output_dir="./experiments/${WANDB_NAME}" \
    --root_folders="/mnt/dataset3/jaewon/YouHQ/YouHQ-Train" \
    --mixed_precision="fp16" \
    --learning_rate ${LR} \
    --train_batch_size 2 \
    --gradient_accumulation_steps=1 \
    --null_text_ratio=0.0 \
    --dataloader_num_workers=0 \
    --target_lifting_layer 0 1 2 3 4 5 6 7 \
    --checkpointing_steps 25000 \
    --checkpoints_total_limit 4 \
    --validation_steps 100000 \
    --max_train_steps 100000 \
    --target_indices 0 1 2 3 \
    --target_modules 'attn.norm_k' \
    --wandb_name ${WANDB_NAME}
