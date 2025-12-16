DATASET=UDM10
IDX=6
NUM=10
CHUNK=8
OVERLAP=1
WARP_STEP=30

CUDA_VISIBLE_DEVICES=${IDX} python test/test_jaeeun.py \
    --start_idx 0 \
    --end_idx 10 \
    --pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
    --transformer_model_name_or_path="preset/models/dit4sr_q" \
    --raft_model_path="/mnt/dataset2/jaewon/_final_weights/laeyr_0_1_2_3/checkpoint-100000/raft_weights.pt" \
    --video_path /mnt/dataset2/jaewon/eval/${DATASET}/lq_videos \
    --prompt_path /mnt/dataset2/jaewon/eval/${DATASET}/prompts \
    --output_dir /mnt/dataset2/jaewon/eval/${DATASET}/DiT4SR_ours_flow_${CHUNK}c${OVERLAP}o_fuse0.5_decode_full_viz \
    --chunk_size ${CHUNK} \
    --overlap_size ${OVERLAP} \
    --fuse_scale 0.5 \
    --debug_viz \
    --debug_viz_dir /mnt/dataset2/jaewon/eval/${DATASET}/DiT4SR_ours_flow_${CHUNK}c${OVERLAP}o_fuse0.5_decode_full_viz/debug_viz
