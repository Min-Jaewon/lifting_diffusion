#!/bin/bashbas
# export CUDA_VISIBLE_DEVICES=2

# LR=5e-5
# LAYER_IDX=11
# WANDB_NAME="OVERFITTING/TEST"

# accelerate launch --config_file single-gpu-overfitting.yaml train/train_overfitting_jaewon.py \
#     --report_to wandb \
#     --pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
#     --transformer_model_name_or_path="preset/models/dit4sr_q" \
#     --output_dir="./experiments/${WANDB_NAME}" \
#     --root_folders="/mnt/dataset2/jaewon/YouHQ/YouHQ-Train" \
#     --mixed_precision="fp16" \
#     --learning_rate ${LR} \
#     --train_batch_size 1 \
#     --gradient_accumulation_steps=1 \
#     --null_text_ratio=0.0 \
#     --dataloader_num_workers=0 \
#     --checkpointing_steps=10000 \
#     --checkpoints_total_limit 1 \
#     --only_train_raft \
#     --feature_upsample="bilinear" \
#     --target_modules ff \
#     --target_indices ${LAYER_IDX} \
#     --validation_steps 10 \
#     --wandb_name ${WANDB_NAME} \
#     --max_train_steps 500 

#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

LR=5e-5
WANDB_PREFIX="OVERFITTING"

for LAYER_IDX in {18..23}; do
    echo "=== Running layer index: ${LAYER_IDX} ==="

    accelerate launch --config_file single-gpu.yaml --main_process_port 29504 train/train_overfitting_jaewon.py \
        --report_to wandb \
        --pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
        --transformer_model_name_or_path="preset/models/dit4sr_q" \
        --output_dir="./experiments/${WANDB_NAME}" \
        --root_folders="/mnt/dataset2/jaewon/YouHQ/YouHQ-Train" \
        --mixed_precision="fp16" \
        --learning_rate ${LR} \
        --train_batch_size 1 \
        --gradient_accumulation_steps=1 \
        --null_text_ratio=0.0 \
        --dataloader_num_workers=0 \
        --checkpointing_steps=10000 \
        --checkpoints_total_limit 1 \
        --only_train_raft \
        --feature_upsample="bilinear" \
        --target_modules ff \
        --target_indices ${LAYER_IDX} \
        --validation_steps 10 \
        --wandb_name ${LAYER_IDX}_test \
        --wandb_project DiT4SR_OVERFITTING_3 \
        --max_train_steps 10

    echo "=== Finished layer ${LAYER_IDX} ==="
    echo
done
