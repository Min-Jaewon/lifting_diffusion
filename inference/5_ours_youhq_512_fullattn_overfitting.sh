DATASET=YouHQ40-512
IDX=5
NUM=10
CHUNK=3
OVERLAP=1


WARP_START=41
WARP_END=0 # Not applying warping
FUSE_SCLAE=0.5
WARPING_MODE=sigma_scaling
CKP_POINT=100000

CUDA_VISIBLE_DEVICES=${IDX} python test/test_jaewon_overfitting.py \
    --start_idx 0 \
    --end_idx 10 \
    --pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
    --transformer_model_name_or_path="/mnt/dataset1/m_jaewon/icml26/DiT4SR/experiments/[Baseline]_LaterLayer_Overfitting_Video_AllLayer/checkpoint-5000" \
    --raft_model_path="/mnt/dataset3/jaewon/_final_weights/laeyr_0_1_2_3/checkpoint-100000/raft_weights.pt" \
    --video_path /mnt/dataset3/jaewon/eval/${DATASET}/train_videos \
    --prompt_path /mnt/dataset3/jaewon/eval/${DATASET}/train_prompts \
    --output_dir /mnt/dataset3/jaewon/eval/${DATASET}/DiT4SR_lifting_baseline_overfitting_AllLayer_5000_same_noise \
    --use_same_noise \
    --target_lifting_layer 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
    --chunk_size ${CHUNK} \
    --overlap_size ${OVERLAP}

# FUSE_SCLAE=1.0

# CUDA_VISIBLE_DEVICES=${IDX} python test/test_jaewon.py \
#     --start_idx 0 \
#     --end_idx 1 \
#     --pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
#     --transformer_model_name_or_path="preset/models/dit4sr_q" \
#     --raft_model_path="/mnt/dataset2/jaewon/_final_weights/laeyr_0_1_2_3/checkpoint-100000/raft_weights.pt" \
#     --video_path /mnt/dataset2/jaewon/eval/${DATASET}/lq_videos \
#     --prompt_path /mnt/dataset2/jaewon/eval/${DATASET}/prompts \
#     --output_dir /mnt/dataset2/jaewon/eval/${DATASET}/DiT4SR_ours_flow_abl/${CHUNK}c${OVERLAP}_${WARP_START}_${WARP_END}w_${FUSE_SCLAE}fs \
#     --chunk_size ${CHUNK} \
#     --overlap_size ${OVERLAP} \
#     --warpping_start_step ${WARP_START} \
#     --warpping_end_step ${WARP_END} \
#     --fuse_scale ${FUSE_SCLAE}



