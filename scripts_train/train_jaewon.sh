export CUDA_VISIBLE_DEVICES=5

LR=5e-5
CHANNELS=256
WANDB_NAME="[DPT+Vis_Enc/SEA-RAFT]RAFT_layer12/13/14/16_lr${LR}"

accelerate launch --config_file single-gpu.yaml --main_process_port 29502 train/train_jaewon.py \
    --report_to wandb \
    --pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
    --transformer_model_name_or_path="preset/models/dit4sr_q" \
    --output_dir="/mnt/dataset2/jaewon/experiments/${WANDB_NAME}" \
    --root_folders="/mnt/dataset2/jaewon/YouHQ/YouHQ-Train" \
    --mixed_precision="fp16" \
    --learning_rate ${LR} \
    --train_batch_size 4 \
    --gradient_accumulation_steps=1 \
    --null_text_ratio=0.0 \
    --dataloader_num_workers=0 \
    --checkpointing_steps=10000 \
    --checkpoints_total_limit 4 \
    --only_train_raft \
    --feature_upsample="dpt" \
    --conv_1x1_channels ${CHANNELS} \
    --use_sea_raft \
    --use_raft_encoder \
    --target_modules ff \
    --target_indices 12 13 14 16 \
    --validation_steps 500 \
    --max_train_steps 50000 \
    --wandb_name ${WANDB_NAME}
