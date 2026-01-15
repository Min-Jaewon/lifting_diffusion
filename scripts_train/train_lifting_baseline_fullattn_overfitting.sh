export CUDA_VISIBLE_DEVICES=7

LR=5e-5
CHANNELS=256
WANDB_NAME="[Baseline]_AddLayer_Overfitting_Video"
# Final : DPT + Vision Encoder + SEA-RAFT + Custom DPT + Context DPT
python -m accelerate.commands.launch --config_file single-gpu.yaml --main_process_port 29512 train/train_lifting_baseline_addfullattn_overfitting.py \
    --report_to wandb \
    --pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
    --transformer_model_name_or_path="preset/models/dit4sr_q_ours" \
    --output_dir="./experiments/${WANDB_NAME}" \
    --root_folders="/media/data1/jaewon/YouHQ/YouHQ-Train" \
    --mixed_precision="fp16" \
    --learning_rate ${LR} \
    --train_batch_size 1 \
    --gradient_accumulation_steps=1 \
    --null_text_ratio=0.2 \
    --dataloader_num_workers=0 \
    --checkpointing_steps 2500 \
    --checkpoints_total_limit 4 \
    --validation_steps 10000 \
    --max_train_steps 10000 \
    --wandb_name ${WANDB_NAME}
