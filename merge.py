import os
import json
import glob
from tqdm import tqdm

def merge_json_files():
    # 1. 경로 설정
    base_dir = '/mnt/dataset2/jaewon/flow_dir/flow_evaluation_results_full/'
    save_dir = os.path.join(base_dir, 'total')
    
    # total 폴더가 없으면 생성
    os.makedirs(save_dir, exist_ok=True)

    print(f"Working Directory: {base_dir}")
    print(f"Save Directory: {save_dir}")

    # ---------------------------------------------------------
    # 2. 일반 Method 병합 (Raft, Retrain Raft, Sea Raft)
    # ---------------------------------------------------------
    # 파일명 패턴: all_raft_results_{start}_{end}.json
    base_methods = ['all_raft_results', 'all_retrain_raft_results', 'all_sea_raft_results']

    for method in base_methods:
        # 해당 패턴을 가진 모든 json 파일 검색
        search_pattern = os.path.join(base_dir, f'{method}_*.json')
        file_list = glob.glob(search_pattern)
        
        if not file_list:
            print(f"[Warning] No files found for {method}")
            continue

        merged_data = {}
        print(f"Merging {method} ({len(file_list)} files)...")
        
        for file_path in file_list:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # video_id를 key로 사용하므로 update로 병합하면 됩니다.
                merged_data.update(data)
        
        # 병합된 결과 저장 (파일명 뒤의 start_end 제거)
        output_path = os.path.join(save_dir, f'{method}.json')
        with open(output_path, 'w') as f:
            json.dump(merged_data, f)
        print(f"Saved: {output_path}")

    # ---------------------------------------------------------
    # 3. Ours Method 병합 (Timestep별로 존재)
    # ---------------------------------------------------------
    # 파일명 패턴: all_ours_results_step_{t:02d}_{start}_{end}.json
    num_timestep = 40  # 기존 코드 기준

    for t in tqdm(range(num_timestep), desc='Merging Ours Steps'):
        step_str = f'step_{t:02d}'
        method_prefix = f'all_ours_results_{step_str}'
        
        # 해당 step의 모든 범위 파일 검색
        search_pattern = os.path.join(base_dir, f'{method_prefix}_*.json')
        file_list = glob.glob(search_pattern)
        
        if not file_list:
            continue

        merged_data = {}
        for file_path in file_list:
            with open(file_path, 'r') as f:
                data = json.load(f)
                merged_data.update(data)
        
        # 병합된 결과 저장
        output_path = os.path.join(save_dir, f'{method_prefix}.json')
        with open(output_path, 'w') as f:
            json.dump(merged_data, f)

    print("\nAll merging complete!")

if __name__ == "__main__":
    merge_json_files()