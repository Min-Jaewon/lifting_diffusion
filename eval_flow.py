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
    
    

def main(save_image, args):
    # Case: crop image
    # Image Directories
    lq_frame = '/mnt/dataset2/jaewon/YouHQ/YouHQ-Val-LQ-Ours-frames'
    gt_frame = '/mnt/dataset2/jaewon/YouHQ/YouHQ-Val-frames'
    viz_save_dir = '/mnt/dataset2/jaewon/YouHQ/YouHQ-Val-flow-visualization'
    # Flow Directories
    gt_flow = '/mnt/dataset2/jaewon/YouHQ/YouHQ-Val-flow-sea-full'
    raft_flow = '/mnt/dataset2/jaewon/YouHQ/YouHQ-Val-LQ-Ours-flows-raft'
    retrain_raft_flow = '/mnt/dataset2/jaewon/YouHQ/YouHQ-Val-LQ-Ours-flows-retrain-raft'
    sea_raft_flow = '/mnt/dataset2/jaewon/YouHQ/YouHQ-Val-LQ-Ours-flow-sea-full'
    ours_flow = '/mnt/dataset2/jaewon/experiments/test/lq_flows_ours'

    gt_flow_list = sorted(os.listdir(gt_flow))

    num_timestep = 40
    
    all_raft_results = {}
    all_retrain_raft_results = {}
    all_sea_raft_results = {}
    all_ours_results = [ {} for _ in range(num_timestep)]
    
    # Save final results
    save_dir = f'./results/flow_evaluation_results/'
    os.makedirs(save_dir, exist_ok=True)
    
    for video_id in tqdm(gt_flow_list, desc='Evaluating videos'):
        gt_flow_path = os.path.join(gt_flow, video_id)
        raft_flow_path = os.path.join(raft_flow, video_id+'_frames')
        retrain_raft_flow_path = os.path.join(retrain_raft_flow, video_id+'_frames')
        sea_raft_flow_path = os.path.join(sea_raft_flow, video_id)
        
        gt_flow_files = sorted(glob.glob(os.path.join(gt_flow_path, 'flow_**.npy')))
        
        video_id_str = f'{int(video_id):03d}'
        
        frame_all_raft = {}
        frame_all_retrain_raft = {}
        frame_all_sea_raft = {}
        
        for flow_file in tqdm(gt_flow_files, desc=f'Processing video {video_id}'):
            frame_id = os.path.basename(flow_file).split('_')[1].split('.')[0]
            
            # Load flows
            gt_flow_frame_path = flow_file
            raft_flow_frame_path = os.path.join(raft_flow_path, f'flow_{frame_id}.npy')
            retrain_raft_flow_frame_path = os.path.join(retrain_raft_flow_path, f'flow_{frame_id}.npy')
            sea_raft_flow_frame_path = os.path.join(sea_raft_flow_path, f'flow_{frame_id}.npy')
            
            gt_flow_tensor = torch.from_numpy(np.load(gt_flow_frame_path)).permute(2,0,1).unsqueeze(0).float()
            raft_flow_tensor = torch.from_numpy(np.load(raft_flow_frame_path)).permute(2,0,1).unsqueeze(0).float()
            retrain_raft_flow_tensor = torch.from_numpy(np.load(retrain_raft_flow_frame_path)).permute(2,0,1).unsqueeze(0).float()
            sea_raft_flow_tensor = torch.from_numpy(np.load(sea_raft_flow_frame_path)).permute(2,0,1).unsqueeze(0).float()

            
            _, raft_metric = sequence_loss([raft_flow_tensor], gt_flow_tensor, valid=raft_flow_tensor.new_ones(gt_flow_tensor.shape[0:1]))
            _, retrain_raft_metric = sequence_loss([retrain_raft_flow_tensor], gt_flow_tensor, valid=retrain_raft_flow_tensor.new_ones(gt_flow_tensor.shape[0:1]))
            _, sea_raft_metric = sequence_loss([sea_raft_flow_tensor], gt_flow_tensor, valid=sea_raft_flow_tensor.new_ones(gt_flow_tensor.shape[0:1]))

            # Accumulate results
            frame_all_raft[frame_id] = raft_metric
            frame_all_retrain_raft[frame_id] = retrain_raft_metric
            frame_all_sea_raft[frame_id] = sea_raft_metric
            
        all_raft_results[video_id] = frame_all_raft
        all_retrain_raft_results[video_id] = frame_all_retrain_raft
        all_sea_raft_results[video_id] = frame_all_sea_raft     
        
        for target_timestep in tqdm(range(num_timestep), desc=f'Evaluating Ours for video {video_id}'):
            frame_all_ours = {}
            timestep_idx = target_timestep
            timestep_str = f'step_{timestep_idx:03d}'
            ours_flow_path = os.path.join(ours_flow, video_id_str, timestep_str, f'flow_{video_id_str}.npy')
            ours_all_flow_tensor = torch.from_numpy(np.load(ours_flow_path))
            
            for flow_file in tqdm(gt_flow_files, desc=f'Processing Ours at timestep {target_timestep}'):
                frame_id = os.path.basename(flow_file).split('_')[1].split('.')[0]
                
                # Load flows
                ours_flow_tensor = ours_all_flow_tensor[int(frame_id)].unsqueeze(0)

                _, ours_metric = sequence_loss([ours_flow_tensor], gt_flow_tensor, valid=ours_flow_tensor.new_ones(gt_flow_tensor.shape[0:1]))
                
                # Accumulate results]
                frame_all_ours[frame_id] = ours_metric
                
                # Load Image
                if save_image:
                    lq_frame_path = os.path.join(lq_frame, video_id+'_frames', f'frame_{frame_id}.jpg')
                    gt_frame_path = os.path.join(gt_frame, video_id+'_frames', f'frame_{frame_id}.jpg')
                    lq_pil = Image.open(lq_frame_path).convert('RGB')
                    gt_pil = Image.open(gt_frame_path).convert('RGB')
                    lq_np = np.array(lq_pil)
                    gt_np = np.array(gt_pil)
                    
                    viz_image = viz(gt_np, lq_np, gt_flow_tensor, raft_flow_tensor, sea_raft_flow_tensor, ours_flow_tensor, retrain_raft_flow_tensor)
                    save_viz_dir = os.path.join(viz_save_dir, f'step_{target_timestep:02d}', video_id)
                    os.makedirs(save_viz_dir, exist_ok=True)
                    viz_image.save(os.path.join(save_viz_dir, f'frame_{frame_id}.png'))
                

            all_ours_results[target_timestep][video_id] = frame_all_ours


            
    # Save dict results as json
    with open(os.path.join(save_dir, 'all_raft_results.json'), 'w') as f:
        json.dump(all_raft_results, f)
    with open(os.path.join(save_dir, 'all_retrain_raft_results.json'), 'w') as f:
        json.dump(all_retrain_raft_results, f)
    with open(os.path.join(save_dir, 'all_sea_raft_results.json'), 'w') as f:
        json.dump(all_sea_raft_results, f)
        
    for t in range(num_timestep):
        with open(os.path.join(save_dir, f'all_ours_results_step_{t:02d}.json'), 'w') as f:
            json.dump(all_ours_results[t], f)
    
def main_denoising(save_image, args):
    # Case: crop image
    device= 'cuda:0'
    # Image Directories
    lq_frame = '/mnt/dataset2/jaewon/eval/YouHQ40/lq_videos_frames'
    gt_frame = '/mnt/dataset2/jaewon/eval/YouHQ40/gt_frames'
    viz_save_dir = '/mnt/dataset2/jaewon/eval/YouHQ40/flow_eval_viz'
    # Flow Directories
    gt_flow_dir = '/mnt/dataset2/jaewon/flow_dir/denoising_gt_flow'
    raft_flow_dir = '/mnt/dataset2/jaewon/flow_dir/denoising_raft_flow'
    retrain_raft_flow_dir = '/mnt/dataset2/jaewon/flow_dir/denoising_retrain_raft_flow'
    sea_raft_flow_dir = '/mnt/dataset2/jaewon/flow_dir/denoising_sea_raft_flow'
    ours_flow = '/mnt/dataset2/jaewon/eval/YouHQ40/DiT4SR_ours_flow_4c1_0_40w_sigma_scaling_0.5fs/flows'

    gt_flow_list = sorted(os.listdir(gt_flow_dir))
    num_timestep = 40
    
    all_raft_results = {}
    all_retrain_raft_results = {}
    all_sea_raft_results = {}
    all_ours_results = [ {} for _ in range(num_timestep)]
    
    # Save final results
    save_dir = f'/mnt/dataset2/jaewon/flow_dir/flow_evaluation_results_full/'
    os.makedirs(save_dir, exist_ok=True)
    
    num_samples = len(gt_flow_list)
    
    
    for flow_file_path in tqdm(gt_flow_list, desc='Evaluating videos'):
        video_id = flow_file_path.split('.')[0]
        if int(video_id) < args.start or int(video_id) >= args.end:
            continue
        gt_flow_path = os.path.join(gt_flow_dir , flow_file_path)
        raft_flow_path = os.path.join(raft_flow_dir, flow_file_path)
        retrain_raft_flow_path = os.path.join(retrain_raft_flow_dir, flow_file_path)
        sea_raft_flow_path = os.path.join(sea_raft_flow_dir, flow_file_path)
        
        print(f'Loading flows for video {video_id}...')
        all_gt_flow = np.load(gt_flow_path)  # [Frame, 2, H, W]
        all_raft_flow = np.load(raft_flow_path)
        all_retrain_raft_flow = np.load(retrain_raft_flow_path)
        all_sea_raft_flow = np.load(sea_raft_flow_path)
        print(f'Flows loaded.')
        
        all_gt_flow_tensor = torch.from_numpy(all_gt_flow).to(device)
        all_raft_flow_tensor = torch.from_numpy(all_raft_flow).to(device)
        all_retrain_raft_flow_tensor = torch.from_numpy(all_retrain_raft_flow).to(device)
        all_sea_raft_flow_tensor = torch.from_numpy(all_sea_raft_flow).to(device)
        
        num_frames = all_gt_flow.shape[0]
        
        video_id_str = f'{int(video_id):03d}'
        ours_flow_path = os.path.join(ours_flow, video_id_str+'_forward_flows.npy')
        print(f'Loading Ours flows from {ours_flow_path}...')
        ours_all_timestep_flow_tensor = torch.from_numpy(np.load(ours_flow_path)).to(device) # [Timestep, 2, Frame, H, W]
        print(f'Ours flows loaded.')
        
        frame_all_raft = {}
        frame_all_retrain_raft = {}
        frame_all_sea_raft = {}
        121
        for frame_idx in tqdm(range(num_frames), desc=f'Processing video {video_id}'):
            frame_id = f'{frame_idx:02d}'
            # Load flows
            gt_flow_tensor = all_gt_flow_tensor[frame_idx].unsqueeze(0)
            raft_flow_tensor = all_raft_flow_tensor[frame_idx].unsqueeze(0)
            retrain_raft_flow_tensor = all_retrain_raft_flow_tensor[frame_idx].unsqueeze(0)
            sea_raft_flow_tensor = all_sea_raft_flow_tensor[frame_idx].unsqueeze(0)
            
            
            _, raft_metric = sequence_loss([raft_flow_tensor], gt_flow_tensor, valid=raft_flow_tensor.new_ones(gt_flow_tensor.shape[0:1]))
            _, retrain_raft_metric = sequence_loss([retrain_raft_flow_tensor], gt_flow_tensor, valid=retrain_raft_flow_tensor.new_ones(gt_flow_tensor.shape[0:1]))
            _, sea_raft_metric = sequence_loss([sea_raft_flow_tensor], gt_flow_tensor, valid=sea_raft_flow_tensor.new_ones(gt_flow_tensor.shape[0:1]))
            
            # Accumulate results
            frame_all_raft[frame_id] = raft_metric
            frame_all_retrain_raft[frame_id] = retrain_raft_metric
            frame_all_sea_raft[frame_id] = sea_raft_metric
            
        all_raft_results[video_id] = frame_all_raft
        all_retrain_raft_results[video_id] = frame_all_retrain_raft
        all_sea_raft_results[video_id] = frame_all_sea_raft
        
        for timestep in tqdm(range(num_timestep), desc=f'Evaluating Ours for video {video_id}'):
            frame_all_ours = {}
            target_timestep = timestep
            timestep_idx = target_timestep
            timestep_str = f'step_{timestep_idx:03d}'
            ours_all_flow_tensor = ours_all_timestep_flow_tensor[timestep].permute(1,0,2,3)  # [Frame, 2, H, W]
            
            for frame_idx in tqdm(range(num_frames), desc=f'Processing Ours at timestep {target_timestep}'):
                frame_id = f'{frame_idx:02d}'
                gt_flow_tensor = all_gt_flow_tensor[frame_idx].unsqueeze(0)
                raft_flow_tensor = all_raft_flow_tensor[frame_idx].unsqueeze(0)
                retrain_raft_flow_tensor = all_retrain_raft_flow_tensor[frame_idx].unsqueeze(0)
                sea_raft_flow_tensor = all_sea_raft_flow_tensor[frame_idx].unsqueeze(0)
                
                # Load flows
                ours_flow_tensor = ours_all_flow_tensor[int(frame_id)].unsqueeze(0)
                # Remove padded 8 pixels if any
                ours_flow_tensor = ours_flow_tensor[:,:,:gt_flow_tensor.shape[2], :gt_flow_tensor.shape[3]]
                _, ours_metric = sequence_loss([ours_flow_tensor], gt_flow_tensor, valid=ours_flow_tensor.new_ones(gt_flow_tensor.shape[0:1]))
                
                # Accumulate results
                frame_all_ours[frame_id] = ours_metric
                
                # Load Image
                if save_image:
                    lq_frame_path = os.path.join(lq_frame, video_id, f'{int(frame_id):03d}.png')
                    gt_frame_path = os.path.join(gt_frame, video_id, f'{int(frame_id):03d}.png')
                    lq_pil = Image.open(lq_frame_path).convert('RGB')
                    gt_pil = Image.open(gt_frame_path).convert('RGB')
                    lq_np = np.array(lq_pil)
                    gt_np = np.array(gt_pil)
                    
                    viz_image = viz(gt_np, lq_np, gt_flow_tensor, raft_flow_tensor, sea_raft_flow_tensor, ours_flow_tensor, retrain_raft_flow_tensor)
                    save_viz_dir = os.path.join(viz_save_dir, f'step_{target_timestep:02d}', video_id)
                    os.makedirs(save_viz_dir, exist_ok=True)
                    viz_image.save(os.path.join(save_viz_dir, f'frame_{frame_id}.png'))
                
            all_ours_results[target_timestep][video_id] = frame_all_ours

        all_gt_flow_tensor = all_gt_flow_tensor.cpu()
        all_raft_flow_tensor = all_raft_flow_tensor.cpu()
        all_retrain_raft_flow_tensor = all_retrain_raft_flow_tensor.cpu()
        all_sea_raft_flow_tensor = all_sea_raft_flow_tensor.cpu()
        ours_all_timestep_flow_tensor = ours_all_timestep_flow_tensor.cpu()
        del all_gt_flow_tensor, all_raft_flow_tensor, all_retrain_raft_flow_tensor, all_sea_raft_flow_tensor, ours_all_timestep_flow_tensor
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        
    with open(os.path.join(save_dir, f'all_raft_results_{args.start}_{args.end}.json'), 'w') as f:
        json.dump(all_raft_results, f)
    with open(os.path.join(save_dir, f'all_retrain_raft_results_{args.start}_{args.end}.json'), 'w') as f:
        json.dump(all_retrain_raft_results, f)
    with open(os.path.join(save_dir, f'all_sea_raft_results_{args.start}_{args.end}.json'), 'w') as f:
        json.dump(all_sea_raft_results, f) 
    for t in range(num_timestep):
        with open(os.path.join(save_dir, f'all_ours_results_step_{t:02d}_{args.start}_{args.end}.json'), 'w') as f:
            json.dump(all_ours_results[t], f)
            
def average(target_metric, target_timestep):
    # Average across frames and then videos
    raft_result_json_path = f'./results/flow_evaluation_results/step_{target_timestep:02d}/all_raft_results.json'
    retrain_raft_result_json_path = f'./results/flow_evaluation_results/step_{target_timestep:02d}/all_retrain_raft_results.json'
    sea_raft_result_json_path = f'./results/flow_evaluation_results/step_{target_timestep:02d}/all_sea_raft_results.json'
    ours_result_json_path = f'./results/flow_evaluation_results/step_{target_timestep:02d}/all_ours_results.json'
    
    
    
    # Load results from json
    import json
    with open(raft_result_json_path, 'r') as f:
        all_raft_results = json.load(f)
    with open(retrain_raft_result_json_path, 'r') as f:
        all_retrain_raft_results = json.load(f)
    with open(sea_raft_result_json_path, 'r') as f:
        all_sea_raft_results = json.load(f)
    with open(ours_result_json_path, 'r') as f:
        all_ours_results = json.load(f)
    
    raft_metrics = []
    retrain_raft_metrics = []
    sea_raft_metrics = []
    ours_metrics = []
    
    for video_id in all_raft_results.keys():
        frame_metrics_raft = []
        frame_metrics_retrain_raft = []
        frame_metrics_sea_raft = []
        frame_metrics_ours = []
        for frame_id in all_raft_results[video_id].keys():
            frame_metrics_raft.append(all_raft_results[video_id][frame_id][target_metric])
            frame_metrics_retrain_raft.append(all_retrain_raft_results[video_id][frame_id][target_metric])
            frame_metrics_sea_raft.append(all_sea_raft_results[video_id][frame_id][target_metric])
            frame_metrics_ours.append(all_ours_results[video_id][frame_id][target_metric])
        
        raft_metrics.append(np.mean(frame_metrics_raft))
        retrain_raft_metrics.append(np.mean(frame_metrics_retrain_raft))
        sea_raft_metrics.append(np.mean(frame_metrics_sea_raft))
        ours_metrics.append(np.mean(frame_metrics_ours))

    # Logging final average
    print(f'-- Average {target_metric} at step {target_timestep} across videos --')
    print('RAFT:', np.mean(raft_metrics))
    print('Retrain RAFT:', np.mean(retrain_raft_metrics))
    print('SEA-RAFT:', np.mean(sea_raft_metrics))
    print('Our Method:', np.mean(ours_metrics))
    print('------------------------------')
    
    with open(f'./step_{target_timestep:02d}/average_{target_metric}_results.txt', 'w') as f:
        f.write(f'-- Average {target_metric} at step {target_timestep} across videos --\n')
        f.write(f'RAFT: {np.mean(raft_metrics)}\n')
        f.write(f'Retrain RAFT: {np.mean(retrain_raft_metrics)}\n')
        f.write(f'SEA-RAFT: {np.mean(sea_raft_metrics)}\n')
        f.write(f'Our Method: {np.mean(ours_metrics)}\n')
        f.write('------------------------------\n')

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
    main_denoising(save_image=True, args=args)
    # main(save_image=True, args=args)