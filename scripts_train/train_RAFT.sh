export CUDA_VISIBLE_DEVICES=5

accelerate launch --config_file multi-gpu.yaml train/train_jaewon.py \
    --report_to wandb \
    --pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
    --transformer_model_name_or_path="preset/models/dit4sr_q" \
    --output_dir="./experiments/test" \
    --root_folders="/mnt/dataset2/jaewon/YouHQ/YouHQ-Train" \
    --mixed_precision="fp16" \
    --learning_rate=5e-5 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --dataloader_num_workers=0 \
    --checkpointing_steps=10000 \
    --use_sea_raft \
    --only_train_raft \
    --checkpoints_total_limit 4 \
    --validation_steps 500 \
    --max_train_steps 50000 \
    --wandb_name="[RAFT/SEA-RAFT]Baseline"
