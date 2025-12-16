export CUDA_VISIBLE_DEVICES=2

LR=5e-5
LAYER_IDX=12
L_WEIGHT=0.1
WANDB_NAME="ALL_bilinear_lr${LR}_layer${LAYER_IDX}_lweight${L_WEIGHT}"
PRETRAINED="RAFT_bilinear_layer${LAYER_IDX}_lr${LR}"

accelerate launch --config_file single-gpu.yaml --main_process_port 25110 train/train_jaewon.py \
    --report_to wandb \
    --pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
    --transformer_model_name_or_path="preset/models/dit4sr_q" \
    --output_dir="./experiments/${WANDB_NAME}" \
    --root_folders="/mnt/dataset2/jaewon/YouHQ/YouHQ-Train" \
    --mixed_precision="fp16" \
    --learning_rate ${LR} \
    --train_batch_size 2 \
    --gradient_accumulation_steps=1 \
    --null_text_ratio=0.2 \
    --dataloader_num_workers=0 \
    --checkpointing_steps=10000 \
    --checkpoints_total_limit 1 \
    --feature_upsample="bilinear" \
    --target_modules ff \
    --target_indices ${LAYER_IDX} \
    --validation_steps 500 \
    --stage1_raft_weight /mnt/dataset1/m_jaewon/cvpr26/DiT4SR/preset/${PRETRAINED}/checkpoint-10000/raft_weights.pt \
    --flow_loss_weight ${L_WEIGHT} \
    --max_train_steps 50000 \
    --wandb_name ${WANDB_NAME}