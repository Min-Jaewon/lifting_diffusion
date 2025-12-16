import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# [색상 설정] 2번 와인 스타일 (빨강 계열)
COLORS = {
    'ours':    '#641E16',  # Dark Burgundy (메인 강조 - 아주 진한 빨강)
    'raft':    '#D98880',  # Dark Sienna (중간 톤)
    'retrain': '#A93226'   # Soft Red (연한 톤)
}

def compute_average_metrics(json_path):
    if not os.path.exists(json_path):
        return None

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    metric_keys = ['epe', '1px', '3px', '5px']
    accumulators = {k: [] for k in metric_keys}
    
    for video_id, frames in data.items():
        for frame_id, metrics in frames.items():
            for k in metric_keys:
                if k in metrics:
                    accumulators[k].append(metrics[k])
    
    averages = {}
    for k, v in accumulators.items():
        if v:
            averages[k] = np.mean(v)
        else:
            averages[k] = 0.0
            
    return averages

def plot_metrics_individually():
    # 1. 경로 설정
    json_source_dir = '/mnt/dataset2/jaewon/flow_dir/flow_evaluation_results_full/total'
    save_dir = '/mnt/dataset2/jaewon/flow_dir'
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Reading JSONs from: {json_source_dir}")
    print(f"Saving plots to:    {save_dir}")

    # 2. Baseline 로드
    print("Loading Baselines...")
    raft_path = os.path.join(json_source_dir, 'all_raft_results.json')
    raft_means = compute_average_metrics(raft_path)
    
    retrain_path = os.path.join(json_source_dir, 'all_retrain_raft_results.json')
    retrain_means = compute_average_metrics(retrain_path)

    # 3. Ours 로드
    num_timesteps = 40
    steps = list(range(num_timesteps)) # 0 ~ 39
    
    ours_history = {k: [] for k in ['epe', '1px', '3px', '5px']}
    
    print("Loading Ours results per step...")
    for t in tqdm(steps):
        json_path = os.path.join(json_source_dir, f'all_ours_results_step_{t:02d}.json')
        step_avg = compute_average_metrics(json_path)
        
        if step_avg:
            for k in ours_history:
                ours_history[k].append(step_avg[k])
        else:
            for k in ours_history:
                ours_history[k].append(np.nan)

    # ---------------------------------------------------------
    # [핵심 변경] X축 Tick 설정 (Step -> Timestep 변환)
    # ---------------------------------------------------------
    # 5단위로 눈금 표시 (0, 5, ..., 40)
    tick_interval = 10
    tick_indices = np.arange(0, num_timesteps + 1, tick_interval) 
    
    # Step 0 -> t=1000, Step 40 -> t=0 공식 적용
    # 라벨: 1000에서 시작해서 점점 줄어듦
    tick_labels = [int(1000 - (i * (1000 / num_timesteps))) for i in tick_indices]

    # ---------------------------------------------------------
    # 4. 그래프 그리기
    # ---------------------------------------------------------
    metrics_to_plot = ['epe', '1px', '3px', '5px']
    print("\nGenerating individual plots with Timestep axis...")
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(3, 3))
        
        if metric == 'epe':
            # 2) RAFT Baseline
            if raft_means:
                r_val = raft_means[metric]
                plt.axhline(y=r_val, color=COLORS['raft'], linestyle='--', 
                            linewidth=1.0, alpha=0.9, 
                            label=f'RAFT (Avg: {r_val:.3f})')
                
            # 3) Retrain RAFT Baseline
            if retrain_means:
                rr_val = retrain_means[metric]
                plt.axhline(y=rr_val, color=COLORS['retrain'], linestyle='--', 
                            linewidth=1.0, alpha=0.9, 
                            label=f'RAFT* (Avg: {rr_val:.3f})')
            # 1) Ours Plot
            plt.plot(steps, ours_history[metric], 
                    label='DGFlow', 
                    color=COLORS['ours'], 
                    marker='o', markersize=2, linewidth=1)
        else:
            # 1) Ours Plot
            plt.plot(steps, ours_history[metric], 
                    label='DGFlow', 
                    color=COLORS['ours'], 
                    marker='o', markersize=2, linewidth=1)
            # 2) RAFT Baseline
            if raft_means:
                r_val = raft_means[metric]
                plt.axhline(y=r_val, color=COLORS['raft'], linestyle='--', 
                            linewidth=1.0, alpha=0.9, 
                            label=f'RAFT (Avg: {r_val:.3f})')
                
            # 3) Retrain RAFT Baseline
            if retrain_means:
                rr_val = retrain_means[metric]
                plt.axhline(y=rr_val, color=COLORS['retrain'], linestyle='--', 
                            linewidth=1.0, alpha=0.9, 
                            label=f'RAFT* (Avg: {rr_val:.3f})')
            
        
        # 스타일링 및 축 설정
        # plt.title(f'Flow Evaluation: {metric.upper()}', fontsize=16, fontweight='bold', pad=15)
        
        # X축 라벨 변경 (Step -> Diffusion Timestep)
        # plt.xlabel('Diffusion Timestep ($t$)', fontsize=14, labelpad=10)
        # plt.ylabel(metric, fontsize=14, labelpad=10)
        
        # 위에서 계산한 Tick 적용 (핵심
        plt.xticks(tick_indices, tick_labels)
        
        # [수정] 1000(좌측) 글자가 잘리지 않도록 양옆에 여백(-2 ~ +2)을 추가합니다.
        # 기존: plt.xlim(0, num_timesteps)
        plt.xlim(0, num_timesteps-1) 
        
        plt.grid(True, linestyle=':', color='gray', alpha=0.4)
        
        plt.grid(True, linestyle=':', color='gray', alpha=0.4)
        plt.legend(fontsize=7, framealpha=0.8, shadow=False, loc='center right')
        plt.tick_params(axis='both', which='major', labelsize=8)
        
        # 저장
        filename = f'{metric}_visualization.png'
        save_path = os.path.join(save_dir, filename)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    plot_metrics_individually()