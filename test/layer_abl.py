import os
import sys
sys.path.append(os.getcwd())
import glob
import argparse
import numpy as np
import re
from PIL import Image

import torch

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

from pipelines.pipeline_dit4sr import StableDiffusion3ControlNetPipeline
from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix

from torchvision import transforms
import time

### Newly added
from typing import Iterable
import cv2
import re
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
from RAFT.core.corr_for_anaylsis import CorrBlock
import torch.nn.functional as F
import matplotlib.gridspec as gridspec
from torch import nn
from RAFT.core.utils import flow_viz
from einops import rearrange, repeat, reduce

logger = get_logger(__name__, log_level="INFO")

tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
MAX_FLOW = 400

def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    if flow_preds is None: # Case of uncond path
        metrics = {
            'epe': 0.0,
            '1px': 0.0,
            '3px': 0.0,
            '5px': 0.0,
        }
        
        return torch.tensor(0.0), metrics
    
    n_predictions = len(flow_preds)    
    flow_loss = 0.0
    
    
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    mag = mag.unsqueeze(1)          
    valid = (valid >= 0.5) & (mag < max_flow)
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

# Codes form ZeroCo by Hongyu An et al.
def unnormalise_and_convert_mapping_to_flow(map):
    # here map is normalised to -1;1
    # we put it back to 0,W-1, then convert it to flow
    B, C, H, W = map.size()
    mapping = torch.zeros_like(map)
    # mesh grid
    mapping[:,0,:,:] = (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0 # unormalise
    mapping[:,1,:,:] = (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0 # unormalise

    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if mapping.is_cuda:
        grid = grid.cuda()
    flow = mapping - grid
    return flow

def softmax_with_temperature(x, beta, d = 1):
    r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
    M, _ = x.max(dim=d, keepdim=True)
    x = x - M # subtract maximum value for stability
    exp_x = torch.exp(x/beta)
    exp_x_sum = exp_x.sum(dim=d, keepdim=True)
    return exp_x / exp_x_sum

def soft_argmax(corr, beta=0.02, x_normal=None, y_normal=None):
    r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
    b,_,h,w = corr.size()
    corr = softmax_with_temperature(corr, beta=beta, d=1)
    corr = corr.view(-1,h,w,h,w) # (target hxw) x (source hxw)

    grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
    x_normal = x_normal.expand(b,w)
    x_normal = x_normal.view(b,w,1,1)
    grid_x = (grid_x*x_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
    
    grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
    y_normal = y_normal.expand(b,h)
    y_normal = y_normal.view(b,h,1,1)
    grid_y = (grid_y*y_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
    
    return grid_x, grid_y
    
def read_all_frames(video_path, to_rgb=True, as_pil=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("No frames read from the video")
    # OpenCV는 BGR → 보통 RGB로 변환해서 사용
    if to_rgb:
        frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    if as_pil:
        return [Image.fromarray(frame) for frame in frames]
    return frames  # list of numpy array (H, W, 3)

def iter_frames_dirs(root: str) -> Iterable[str]:
    """
    the expected structure: root/category/unique_id/*.mp4
    """
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise NotADirectoryError(f"Root not found: {root}")

    for category in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        category_path = os.path.join(root, category)
        for unique_id in sorted(d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))):
            uid_path = os.path.join(category_path, unique_id)
            for video_id in sorted(os.listdir(uid_path)):
                if video_id.endswith(".mp4"):
                    video_path = os.path.join(uid_path, video_id)
                    yield video_path
                    
def viz(gt_img, lq_img, flow_pred, flow_gt, flow_lq):
    white_img = (np.ones_like(gt_img) * 255).astype(np.uint8)
    
    # --- 플로우 맵 ---
    flow_pred = flow_pred[0].permute(1, 2, 0).detach().cpu().numpy()
    flow_gt = flow_gt[0].permute(1, 2, 0).detach().cpu().numpy()
    flow_lq = flow_lq[0].permute(1, 2, 0).detach().cpu().numpy()
    
    flow_pred_viz = flow_viz.flow_to_image(flow_pred)
    flow_gt_viz = flow_viz.flow_to_image(flow_gt)
    flow_lq_viz = flow_viz.flow_to_image(flow_lq)
    
    # --- 가로로 이미지 이어붙이기 ---
    top_row = np.concatenate([gt_img, lq_img, white_img], axis=1)
    bottom_row = np.concatenate([flow_gt_viz,flow_pred_viz, flow_lq_viz], axis=1)
    cat_img = np.concatenate([top_row, bottom_row], axis=0)
    
    return Image.fromarray(cat_img)
    
def main(args):
    layer_abl = 'results/layer_abl'
    
    os.makedirs(layer_abl, exist_ok=True)

    gt_img_dir = '/mnt/dataset2/jaewon/YouHQ/YouHQ-Train-frames'
    
    # Prepare Input
    video_list = list(iter_frames_dirs(args.image_path))
    
    # Pick 10 videos randomly for testing
    random_seed = 42
    np.random.seed(random_seed)
    video_list = np.random.choice(video_list, size=10, replace=False).tolist()
    
    
    timesteps = [999, 973, 948, 922, 897, 871, 845, 820, 794, 768, 
                743, 717, 692, 666, 640, 615, 589, 564, 538, 512, 
                487, 461, 435, 410, 384, 359, 333, 307, 282, 256, 
                231, 205, 179, 154, 128, 102, 77, 51, 26, 0]
    
    
    device='cuda'
    
    # Save the metric for analysis across layers & timesteps
    
    # Inference loop
    for seq_id, video_path in enumerate(video_list):
        if seq_id < args.start_num:
            continue
        if args.end_num != -1 and seq_id >= args.end_num:
            break
        epe_across_layers = defaultdict(list)  # key: layer_idx, value: list of epe for each video & timestep
        _1px_across_layers = defaultdict(list)
        _3px_across_layers =defaultdict(list)
        _5px_across_layers =defaultdict(list)

        img = read_all_frames(video_path, to_rgb=True, as_pil=True)
        np.random.seed(seq_id)
        frame_idx = np.random.randint(0, len(img)-1)
        # print(f'Processing video: {video_path}, frame index: {frame_idx}-{frame_idx+1}')
        # continue
        relative_video_path = os.path.relpath(video_path, args.image_path).replace('.mp4', '')
        feature_path = os.path.join(args.output_dir, 'feature', relative_video_path)
        
        image_1 = img[frame_idx]
        image_2 = img[frame_idx+1]
        
        gt_image_1_path = os.path.join(gt_img_dir, relative_video_path+'_frames', f'frame_{frame_idx:02d}.jpg')
        gt_image_2_path = os.path.join(gt_img_dir, relative_video_path+'_frames', f'frame_{frame_idx+1:02d}.jpg')
        
        gt_image_1 = cv2.imread(gt_image_1_path)
        gt_image_1 = cv2.cvtColor(gt_image_1, cv2.COLOR_BGR2RGB)
        
        gt_image_2 = cv2.imread(gt_image_2_path)
        gt_image_2 = cv2.cvtColor(gt_image_2, cv2.COLOR_BGR2RGB)
    
        print(f'===Processing video: {relative_video_path}===')
        
        for layer_idx in range(24):
            layer = f'transformer_blocks@{layer_idx}@ff'
            
            features_1_path = os.path.join(feature_path, f'{layer}_{frame_idx:02d}.pt')
            features_2_path = os.path.join(feature_path, f'{layer}_{frame_idx+1:02d}.pt')
            
            features_1 = torch.load(features_1_path)  # (Timestep, 1024, 1536)
            features_1 = torch.stack(features_1, dim=0)  # (Timestep, 1024, 1536)
            
            features_2 = torch.load(features_2_path)  # (Timestep, 1024, 1536)
            features_2 = torch.stack(features_2, dim=0)  # (Timestep, 1024, 1536)
            
            flow_gt = np.load(os.path.join(gt_img_dir.replace('frames','Analysis'),'gt_flows', relative_video_path.split('/')[-1]+'_frames', f'flow_{frame_idx:03d}.npy'))
            flow_gt = torch.from_numpy(flow_gt).permute(2,0,1).unsqueeze(0).float().to(device)  # (1, 2, H, W)
            
            flow_lr = np.load(os.path.join(gt_img_dir.replace('frames','Analysis'), 'lq_flows',relative_video_path.split('/')[-1], f'flow_{frame_idx:03d}.npy'))
            flow_lr = torch.from_numpy(flow_lr).permute(2,0,1).unsqueeze(0).float().to(device)  # (1, 2, H, W)
            
            for t in tqdm(range(len(timesteps)), desc=f'Layer {layer_idx}'):
                # # For Debugging append random values
                # epe_across_layers[layer_idx].append(torch.rand(1).item())
                # _1px_across_layers[layer_idx].append(torch.rand(1).item())
                # _3px_across_layers[layer_idx].append(torch.rand(1).item())
                # _5px_across_layers[layer_idx].append(torch.rand(1).item())
                # continue
                
                timestep = timesteps[t]
                feat_t_1 = features_1[t]  # (1024, 1536)
                feat_t_2 = features_2[t]  # (1024, 1536)

                    
                feat_t_1_tensor = feat_t_1.reshape(32,32,-1).permute(2,0,1).unsqueeze(0)  # (1, C, 32, 32)
                feat_t_2_tensor = feat_t_2.reshape(32,32,-1).permute(2,0,1).unsqueeze(0)  # (1, C, 32, 32)
                feat_t_1_tensor = feat_t_1_tensor.float().contiguous()
                feat_t_2_tensor = feat_t_2_tensor.float().contiguous()
                # Upsample using Bilinear
                
                upsampled_feat_1 = F.interpolate(feat_t_1_tensor, size=(64,64), mode='bilinear', align_corners=False)
                upsampled_feat_2 = F.interpolate(feat_t_2_tensor, size=(64,64), mode='bilinear', align_corners=False)
                cost_volume_bilinear = CorrBlock.corr(upsampled_feat_1, upsampled_feat_2)  # (1, 64, 64, 1, 64, 64)
                cost_volume_bilinear = cost_volume_bilinear.reshape(1, -1, 64, 64)  # (1, 4096, 64, 64)
    
                feature_size = 64
                x_normal = np.linspace(-1, 1, feature_size)
                y_normal = np.linspace(-1, 1, feature_size)
                x_normal = nn.Parameter(torch.from_numpy(x_normal).float().to(device), requires_grad=False)
                y_normal = nn.Parameter(torch.from_numpy(y_normal).float().to(device), requires_grad=False)
                grid_x, grid_y = soft_argmax(cost_volume_bilinear.to(device), beta=1e-4, x_normal=x_normal, y_normal=y_normal)
                coarse_flow = torch.cat((grid_x, grid_y), dim=1)
                
                flow_est = unnormalise_and_convert_mapping_to_flow(coarse_flow) 
                flow_est = F.interpolate(flow_est, size=(512,512), mode='bilinear', align_corners=False) * 8.0  # upsample to image size
            
                # Save visualization
                viz_img = viz(image_1, gt_image_1, flow_est, flow_gt, flow_lr)
                save_path = os.path.join(layer_abl, relative_video_path, f'layer_{layer_idx}', f'timestep_{timestep:03d}.png')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                viz_img.save(save_path)
                
                # Compute metric
                _, metric = sequence_loss([flow_est], flow_gt, torch.ones_like(flow_gt[:, :1, :, :]))
                # All keys in metric have to be saved
                epe_across_layers[layer_idx].append(metric['epe'])
                _1px_across_layers[layer_idx].append(metric['1px'])
                _3px_across_layers[layer_idx].append(metric['3px'])
                _5px_across_layers[layer_idx].append(metric['5px'])
                
        
                
        # Plot the metric across layers & timesteps with 3D landscape
        for metric_name, metric_across_layers in zip(['EPE', '1px', '3px', '5px'], 
                                                     [epe_across_layers, _1px_across_layers, _3px_across_layers, _5px_across_layers]):
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            
            layer_indices = sorted(metric_across_layers.keys())
            timestep_indices = timesteps
            
            L, T = np.meshgrid(layer_indices, timestep_indices)
            Z = np.array([metric_across_layers[layer_idx] for layer_idx in layer_indices]).T  # shape (len(timesteps), len(layers))
            
            # Plot the surface with blue colormap
            ax.plot_surface(L, T, Z, cmap='viridis')
            ax.set_xlabel('Layer Index')
            ax.set_ylabel('Timestep')
            ax.set_zlabel(metric_name)
            ax.set_title(f'{metric_name} across Layers and Timesteps for {relative_video_path}')
            
            save_path = os.path.join(layer_abl+'_plot', relative_video_path, f'{metric_name}_landscape.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        
        # Save the metric values as .npy for further analysis
        save_metric_path = os.path.join(layer_abl+'_stat', relative_video_path.replace('/', '_')+'_metrics.npz')
        os.makedirs(os.path.dirname(save_metric_path), exist_ok=True)
        np.savez(save_metric_path,
                 epe=epe_across_layers,
                 _1px=_1px_across_layers,
                 _3px=_3px_across_layers,
                 _5px=_5px_across_layers)
        
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_num", type=int, default=0)
    parser.add_argument("--end_num", type=int, default=-1)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='preset/models/stable-diffusion-3.5-medium')
    parser.add_argument("--transformer_model_name_or_path", type=str, default='dit4sr_q')
    parser.add_argument("--prompt", type=str, default="") # user can add self-prompt to improve the results
    parser.add_argument("--added_prompt", type=str, default='Cinematic, hyper sharpness, highly detailed, perfect without deformations, '
                            'camera, hyper detailed photo - realistic maximum detail, 32k, Color '
                            'Grading, ultra HD, extreme meticulous detailing, skin pore detailing. ')
    parser.add_argument("--negative_prompt", type=str, default='motion blur, noisy, dotted, bokeh, pointed, '
                            'CG Style, 3D render, unreal engine, blurring, dirty, messy, '
                            'worst quality, low quality, frames, watermark, signature, jpeg artifacts, '
                            'deformed, lowres, chaotic')
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default='results')
    parser.add_argument("--mixed_precision", type=str, default="fp16") # no/fp16/bf16
    parser.add_argument("--guidance_scale", type=float, default=8.0)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224) # latent size, for 24G
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) # image size, for 13G
    parser.add_argument("--latent_tiled_size", type=int, default=64) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=24) 
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample_times", type=int, default=1)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--start_point", type=str, choices=['lr', 'noise'], default='noise') # LR Embedding Strategy, choose 'lr latent + 999 steps noise' as diffusion start point. 
    parser.add_argument("--save_prompts", action='store_true')
    parser.add_argument("--prompt_path", type=str, default='prompt_LR')
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    args = parser.parse_args()
    main(args)



