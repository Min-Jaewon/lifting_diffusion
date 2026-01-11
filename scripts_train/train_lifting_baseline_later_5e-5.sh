export CUDA_VISIBLE_DEVICES=0,1

LR=5e-5
CHANNELS=256
WANDB_NAME="[Baseline]_Later_${LR}"
# Final : DPT + Vision Encoder + SEA-RAFT + Custom DPT + Context DPT
python -m accelerate.commands.launch --config_file multi-gpu.yaml --main_process_port 29511 train/train_lifting_baseline.py \
    --report_to wandb \
    --pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
    --transformer_model_name_or_path="preset/models/dit4sr_q" \
    --output_dir="./experiments/${WANDB_NAME}" \
    --root_folders="/mnt/dataset3/jaewon/YouHQ/YouHQ-Train" \
    --trainable_modules 'control_conv' \
    --mixed_precision="fp16" \
    --learning_rate ${LR} \
    --train_batch_size 1 \
    --gradient_accumulation_steps=1 \
    --null_text_ratio=0.2 \
    --dataloader_num_workers=0 \
    --target_lifting_layer 16 17 18 19 20 21 22 23 \
    --checkpointing_steps 25000 \
    --checkpoints_total_limit 4 \
    --validation_steps 100000 \
    --max_train_steps 100000 \
    --wandb_name ${WANDB_NAME}
