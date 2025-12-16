import os
from RAFT.core.utils import flow_viz
from RAFT.raft_tools import sequence_loss
import glob
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import json

def viz(gt_img, lq_img, gt_flow, raft_flow, sea_raft_flow, our_flow, retrain_raft_flow):
    # gt_img, lq_img, lq_img, lq_img, lq_img
    # gt_flow, raft_flow, sea_raft_flow, our_flow, retrain_raft_flow: [1, 2, H, W]
    
    # gt_img : H,W,3 (uint8)
    gt_img = torch.from_numpy(gt_img).permute(2,0,1)
    lq_img = torch.from_numpy(lq_img).permute(2,0,1)
    
    gt_flo = flow_viz.flow_to_image(gt_flow[0].permute(1,2,0).cpu().numpy())
    raft_flo = flow_viz.flow_to_image(raft_flow[0].permute(1,2,0).cpu().numpy())
    sea_raft_flo = flow_viz.flow_to_image(sea_raft_flow[0].permute(1,2,0).cpu().numpy())
    our_flo = flow_viz.flow_to_image(our_flow[0].permute(1,2,0).cpu().numpy())
    retrain_raft_flo = flow_viz.flow_to_image(retrain_raft_flow[0].permute(1,2,0).cpu().numpy())

    row1 = torch.cat([gt_img, lq_img, lq_img, lq_img, lq_img], dim=2)  # [5, 3, H, W]
    row2 = torch.cat([torch.from_numpy(gt_flo).permute(2,0,1), 
                        torch.from_numpy(raft_flo).permute(2,0,1),
                        torch.from_numpy(sea_raft_flo).permute(2,0,1),
                        torch.from_numpy(our_flo).permute(2,0,1),
                        torch.from_numpy(retrain_raft_flo).permute(2,0,1)], dim=2)  # [5, 3, H, W]

    viz_image = torch.cat([row1, row2], dim=1).permute(1,2,0).cpu().numpy().astype(np.uint8)
    viz_image = Image.fromarray(viz_image)
    return viz_image
    
    

def main(save_image, target_timestep, args):
    # Case: crop image
    # Image Directories
    lq_frame = '/mnt/dataset2/jaewon/YouHQ/YouHQ-Val-LQ-Ours-frames'
    gt_frame = '/mnt/dataset2/jaewon/YouHQ/YouHQ-Val-frames'
    viz_save_dir = '/mnt/dataset2/jaewon/YouHQ/YouHQ-Val-flow-visualization-ablation'
    # Flow Directories
    gt_flow = '/mnt/dataset2/jaewon/YouHQ/YouHQ-Val-flow-sea-full'
    ours_bilinear_flow = '/mnt/dataset2/jaewon/experiments/test_bilinear/lq_flows_ours'
    ours_no_raft_encoder_flow = '/mnt/dataset2/jaewon/experiments/test_no_raft_encoder/lq_flows_ours'
    ours_middle_layer_flow = '/mnt/dataset2/jaewon/experiments/test_middle_layer/lq_flows_ours'
    ours_flow = '/mnt/dataset2/jaewon/experiments/test/lq_flows_ours'

    gt_flow_list = sorted(os.listdir(gt_flow))

    num_timestep = 40
    
    all_bilinear_results = {}
    all_no_raft_encoder_results = {}
    all_middle_layer_results = {}
    all_ours_results = {}
    
    # Save final results
    save_dir = f'/mnt/dataset2/jaewon/flow_dir/flow_evaluation_result_ablation/'
    os.makedirs(save_dir, exist_ok=True)
    
    for video_id in tqdm(gt_flow_list, desc='Evaluating videos'):
        gt_flow_path = os.path.join(gt_flow, video_id)
        gt_flow_files = sorted(glob.glob(os.path.join(gt_flow_path, 'flow_**.npy')))
        
        video_id_str = f'{int(video_id):03d}'
        
        frame_all_bilinear = {}
        frame_all_no_raft_encoder = {}
        frame_all_middle_layer = {}
        frame_all_ours = {}
        
        timestep_idx = target_timestep
        timestep_str = f'step_{timestep_idx:03d}'
        
        
        ours_bilinear_flow_path = os.path.join(ours_bilinear_flow, video_id_str, timestep_str, f'flow_{video_id_str}.npy')
        ours_no_raft_encoder_flow_path = os.path.join(ours_no_raft_encoder_flow, video_id_str, timestep_str, f'flow_{video_id_str}.npy')
        ours_middle_layer_flow_path = os.path.join(ours_middle_layer_flow, video_id_str, timestep_str, f'flow_{video_id_str}.npy')
        ours_flow_path = os.path.join(ours_flow, video_id_str, timestep_str, f'flow_{video_id_str}.npy')
        
        ours_all_bilinear_flow_tensor = torch.from_numpy(np.load(ours_bilinear_flow_path))
        ours_all_no_raft_encoder_flow_tensor = torch.from_numpy(np.load(ours_no_raft_encoder_flow_path))
        ours_all_middle_layer_flow_tensor = torch.from_numpy(np.load(ours_middle_layer_flow_path))
        ours_all_flow_tensor = torch.from_numpy(np.load(ours_flow_path))
        
        for flow_file in tqdm(gt_flow_files, desc=f'Processing video {video_id}'):
            frame_id = os.path.basename(flow_file).split('_')[1].split('.')[0]
            gt_flow_tensor = torch.from_numpy(np.load(flow_file)).permute(2,0,1).unsqueeze(0)  # [1, 2, H, W]
            
            # Load flows
            ours_bilinear_flow_tensor = ours_all_bilinear_flow_tensor[int(frame_id)].unsqueeze(0)
            ours_no_raft_encoder_flow_tensor = ours_all_no_raft_encoder_flow_tensor[int(frame_id)].unsqueeze(0)
            ours_middle_layer_flow_tensor = ours_all_middle_layer_flow_tensor[int(frame_id)].unsqueeze(0)
            ours_flow_tensor = ours_all_flow_tensor[int(frame_id)].unsqueeze(0)
            
            # _, ours_bilinear_metric = sequence_loss([ours_bilinear_flow_tensor], gt_flow_tensor, valid=ours_bilinear_flow_tensor.new_ones(gt_flow_tensor.shape[0:1]))
            # _, ours_no_raft_encoder_metric = sequence_loss([ours_no_raft_encoder_flow_tensor], gt_flow_tensor, valid=ours_no_raft_encoder_flow_tensor.new_ones(gt_flow_tensor.shape[0:1]))
            # _, ours_middle_layer_metric = sequence_loss([ours_middle_layer_flow_tensor], gt_flow_tensor, valid=ours_middle_layer_flow_tensor.new_ones(gt_flow_tensor.shape[0:1]))
            _, ours_metric = sequence_loss([ours_flow_tensor], gt_flow_tensor, valid=ours_flow_tensor.new_ones(gt_flow_tensor.shape[0:1]))

            # Accumulate results
            # frame_all_bilinear[frame_id] = ours_bilinear_metric
            # frame_all_no_raft_encoder[frame_id] = ours_no_raft_encoder_metric
            # frame_all_middle_layer[frame_id] = ours_middle_layer_metric
            frame_all_ours[frame_id] = ours_metric
            
            # Load Image
            if save_image:
                lq_frame_path = os.path.join(lq_frame, video_id+'_frames', f'frame_{frame_id}.jpg')
                gt_frame_path = os.path.join(gt_frame, video_id+'_frames', f'frame_{frame_id}.jpg')
                lq_pil = Image.open(lq_frame_path).convert('RGB')
                gt_pil = Image.open(gt_frame_path).convert('RGB')
                lq_np = np.array(lq_pil)
                gt_np = np.array(gt_pil)
                
                viz_image = viz(gt_np, lq_np, gt_flow_tensor, ours_bilinear_flow_tensor, ours_no_raft_encoder_flow_tensor, ours_middle_layer_flow_tensor, ours_flow_tensor)
                save_viz_dir = os.path.join(viz_save_dir, f'step_{target_timestep:02d}', video_id)
                os.makedirs(save_viz_dir, exist_ok=True)
                viz_image.save(os.path.join(save_viz_dir, f'frame_{frame_id}.png'))
        
        # all_bilinear_results[video_id] = frame_all_bilinear
        # all_no_raft_encoder_results[video_id] = frame_all_no_raft_encoder
        # all_middle_layer_results[video_id] = frame_all_middle_layer
        all_ours_results[video_id] = frame_all_ours

            
    # Save dict results as json
    # with open(os.path.join(save_dir, 'all_bilinear_results.json'), 'w') as f:
    #     json.dump(all_bilinear_results, f)
    # with open(os.path.join(save_dir, 'all_no_raft_encoder_results.json'), 'w') as f:
    #     json.dump(all_no_raft_encoder_results, f)
    # with open(os.path.join(save_dir, 'all_middle_layer_results.json'), 'w') as f:
    #     json.dump(all_middle_layer_results, f)
    with open(os.path.join(save_dir, 'all_ours_results.json'), 'w') as f:
        json.dump(all_ours_results, f)
            
def average(target_metric, target_timestep=19):
    # Average across frames and then videos
    bilinear_json_path = f'/mnt/dataset2/jaewon/flow_dir/flow_evaluation_result_ablation/all_bilinear_results.json'
    no_raft_encoder_json_path = f'/mnt/dataset2/jaewon/flow_dir/flow_evaluation_result_ablation/all_no_raft_encoder_results.json'
    middle_layer_json_path = f'/mnt/dataset2/jaewon/flow_dir/flow_evaluation_result_ablation/all_middle_layer_results.json'
    ours_result_json_path = f'/mnt/dataset2/jaewon/flow_dir/flow_evaluation_result_ablation/all_ours_results.json'
    
    
    # Load results from json
    with open(bilinear_json_path, 'r') as f:
        bilinear_results = json.load(f)
    with open(no_raft_encoder_json_path, 'r') as f:
        no_raft_encoder_results = json.load(f)
    with open(middle_layer_json_path, 'r') as f:
        middle_layer_results = json.load(f)
    with open(ours_result_json_path, 'r') as f:
        ours_results = json.load(f)

    bilinear_metrics = []
    no_raft_encoder_metrics = []
    middle_layer_metrics = []
    ours_metrics = []
    
    
    for video_id in bilinear_results.keys():
        frame_metrics_bilinear = []
        frame_metrics_no_raft_encoder = []
        frame_metrics_middle_layer = []
        frame_metrics_ours = []
        
        for frame_id in bilinear_results[video_id].keys():
            frame_metrics_bilinear.append(bilinear_results[video_id][frame_id][target_metric])
            frame_metrics_no_raft_encoder.append(no_raft_encoder_results[video_id][frame_id][target_metric])
            frame_metrics_middle_layer.append(middle_layer_results[video_id][frame_id][target_metric])
            frame_metrics_ours.append(ours_results[video_id][frame_id][target_metric])
        
        bilinear_metrics.append(np.mean(frame_metrics_bilinear))
        no_raft_encoder_metrics.append(np.mean(frame_metrics_no_raft_encoder))
        middle_layer_metrics.append(np.mean(frame_metrics_middle_layer))
        ours_metrics.append(np.mean(frame_metrics_ours))

    # Logging final average
    print(f'-- Average {target_metric} at step {target_timestep} across videos --')
    print(f'Bilinear:          {np.mean(bilinear_metrics):.4f}')
    print(f'No Raft Encoder:   {np.mean(no_raft_encoder_metrics):.4f}')
    print(f'Middle Layer:     {np.mean(middle_layer_metrics):.4f}')
    print(f'Ours:              {np.mean(ours_metrics):.4f}')
    print('------------------------------------------')
import multiprocessing as mp

def run_single_timestep(t):
    print(f"[Process] Running timestep {t}")
    main(save_image=True, target_timestep=t)

    target_metrics = ['epe', '1px', '3px', '5px']
    for metric in target_metrics:
        average(metric, t)

    print(f"[Process] Finished timestep {t}")
 
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--start', type=int, default=0, help='Start timestep index')
    parser.add_argument('--end', type=int, default=39, help='End timestep index')
    args = parser.parse_args()
    # main(save_image=True, target_timestep=19, args=args)
    target_metrics = ['epe', '1px', '3px', '5px']
    for metric in target_metrics:
        average(metric, 19)