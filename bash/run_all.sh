#!/bin/bash

# 사용할 GPU 목록
GPUS=(3 5)
NUM_GPUS=${#GPUS[@]}

for i in {0..9}; do
    start=$i
    end=$((i + 1))
    gpu_id=${GPUS[$((i % NUM_GPUS))]}   # 3,4,5 반복 배정

    echo "Running job $i on GPU $gpu_id (start=$start, end=$end)"

    CUDA_VISIBLE_DEVICES=$gpu_id python test/layer_abl.py \
        --start_num $start \
        --end_num $end \
        --image_path /mnt/dataset2/jaewon/YouHQ/YouHQ-Train-LQ-Ours \
        --output_dir results/ \
        > job_${start}_${end}.log 2>&1 &
done

wait
echo "✅ All jobs finished!"