export CUDA_VISIBLE_DEVICES=4

LR=5e-5
CHANNELS=256
WANDB_NAME="[Final]layer_12/13/14/16_lr${LR}"
# Final : DPT + Vision Encoder + SEA-RAFT + Custom DPT + Context DPT
accelerate launch --config_file single-gpu.yaml --main_process_port 29532 test/test_jaewon_flow.py \
    --pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
    --transformer_model_name_or_path="preset/models/dit4sr_q" \
    --output_dir="/mnt/dataset2/jaewon/experiments/test_middle_layer" \
    --root_folders="/mnt/dataset2/jaewon/YouHQ/YouHQ-Val-LQ-Ours-frames" \
    --raft_model_path="/mnt/dataset2/jaewon/experiments/[Final_abl]layer_10_11_12_13_lr5e-5/checkpoint-100000/raft_weights.pt" \
    --mixed_precision="fp16" \
    --feature_upsample="dpt" \
    --use_custom_dpt \
    --use_context_dpt \
    --use_raft_encoder \
    --target_modules ff \
    --target_indices 0 1 2 3 