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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
from RAFT.core.corr import CorrBlock
import torch.nn.functional as F
import matplotlib.gridspec as gridspec
logger = get_logger(__name__, log_level="INFO")

tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

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
# from sklearn.decomposition import PCA # PCA는 현재 사용하지 않으므로 주석 처리
import matplotlib.pyplot as plt
from tqdm import tqdm
from RAFT.core.corr import CorrBlock
import torch.nn.functional as F
import matplotlib.gridspec as gridspec

logger = get_logger(__name__, log_level="INFO")

tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

def plot_grid_with_query(
    hr_img_query, query_x, query_y,
    hr_image_target, cost_map_dict, timesteps, 
    grid_size, output_path, layer_id, feature_type, title_prefix='t'
):
    """
    쿼리 포인트 이미지(상단 중앙), Layer ID (상단 왼쪽), Feature 설명 (상단 오른쪽),
    그리고 Cost map 그리드(하단)를 하나의 이미지로 저장합니다.
    
    상단 레이아웃은 4-2-4 컬럼 스팬을 사용합니다.
    """
    rows, cols = grid_size # (4, 10)
    
    # 텍스트를 위한 좌우 여백이 아닌, 그리드 자체의 너비를 사용하므로 figsize 수정
    fig = plt.figure(figsize=(cols * 2.5, (rows + 1) * 2.7)) # + 6 제거
    
    # --- [수정된 부분] ---
    # GridSpec: (rows+1) 행, cols (10) 열
    # 더 이상 좌우에 텍스트용 추가 열(cols+2)이 필요 없습니다.
    gs = gridspec.GridSpec(rows + 1, cols, figure=fig, 
                           height_ratios=[1.0] + [1.0] * rows, # 2.0을 1.0으로 수정
                           wspace=0.1, hspace=0.2)
    # --- [수정 완료] ---

    # 1. Layer ID 텍스트 (왼쪽 4개 컬럼 위)
    ax_layer_text = fig.add_subplot(gs[0, 0:4]) # 0, 1, 2, 3 컬럼 스팬
    ax_layer_text.text(0.5, 0.5, f'Layer : {layer_id}', 
                       horizontalalignment='center', verticalalignment='center',
                       fontsize=20, color='black', transform=ax_layer_text.transAxes)
    ax_layer_text.axis('off')

    # 2. 쿼리 이미지 그리기 (중앙 2개 컬럼 위)
    ax_query = fig.add_subplot(gs[0, 4:6]) # 4, 5 컬럼 스팬
    img_np_query = np.array(hr_img_query)
    
    marker_color = (255, 0, 0) # RGB
    cv2.circle(img_np_query, (query_x, query_y), radius=5, color=marker_color, thickness=3)
    
    ax_query.imshow(img_np_query)
    ax_query.set_title(f'Query Point (x={query_x}, y={query_y})', fontsize=12, pad=10, color='black')
    ax_query.axis('off')

    # 3. Feature Type 텍스트 (오른쪽 4개 컬럼 위)
    ax_feature_text = fig.add_subplot(gs[0, 6:10]) # 6, 7, 8, 9 컬럼 스팬
    
    if feature_type == 'org':
        feature_text_content = 'Original Feature (32x32)'
    elif feature_type == 'anyup':
        feature_text_content = 'AnyUp Feature (64x64)'
    elif feature_type == 'bilinear':
        feature_text_content = 'Bilinear Feature (64x64)'
    else:
        feature_text_content = f'{feature_type} Feature' 

    ax_feature_text.text(0.5, 0.5, feature_text_content, 
                         horizontalalignment='center', verticalalignment='center',
                         fontsize=16, color='black', transform=ax_feature_text.transAxes)
    ax_feature_text.axis('off')

    # 4. Cost Map 그리드 그리기 (하단)
    target_img_np = np.array(hr_image_target)
    H, W, _ = target_img_np.shape
    
    axes_grid = []
    for r in range(rows):
        for c in range(cols):
            # 그리드는 1행부터, 0열부터 배치 (인덱스 수정)
            axes_grid.append(fig.add_subplot(gs[r + 1, c])) # c + 1 -> c

    for i, timestep in enumerate(timesteps):
        if i >= len(axes_grid):
            break
            
        ax = axes_grid[i]
        key = f'{title_prefix}{timestep}'
        
        if key in cost_map_dict:
            cost_map = cost_map_dict[key]
            
            if cost_map.max() > cost_map.min():
                cost_map_norm = (cost_map - cost_map.min()) / (cost_map.max() - cost_map.min())
            else:
                cost_map_norm = np.zeros_like(cost_map)
                
            heatmap = cv2.applyColorMap(np.uint8(255 * cost_map_norm), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            heatmap_resized = cv2.resize(heatmap_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
            
            overlayed_img = cv2.addWeighted(target_img_np, 0.6, heatmap_resized, 0.4, 0)
            
            ax.imshow(overlayed_img)
            ax.set_title(f'{title_prefix}={timestep}', fontsize=8, color='black')
            ax.axis('off')
        else:
            ax.axis('off')
            
    plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=0.5) 
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Combined grid saved to: {output_path}")
    
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
                    
def main(args):
    org_matching = 'results/matching_vis_org'
    anyup_matching = 'results/matching_vis_anyup'
    bilinear_matching = 'results/matching_vis_bilinear'
    
    
    os.makedirs(org_matching, exist_ok=True)

    gt_img_dir = '/mnt/dataset1/m_jaewon/cvpr26/data/YouHQ-Train-frames'
    
    # Prepare Input
    video_list = list(iter_frames_dirs(args.image_path))
    video_list = video_list[:10] # for debug
    
    timesteps = [999, 973, 948, 922, 897, 871, 845, 820, 794, 768, 
                743, 717, 692, 666, 640, 615, 589, 564, 538, 512, 
                487, 461, 435, 410, 384, 359, 333, 307, 282, 256, 
                231, 205, 179, 154, 128, 102, 77, 51, 26, 0]
    
    from anyup.hubconf import anyup
    
    device='cuda'
    
    upsampler = anyup(pretrained=True, device=device).eval()
    
    
    # Inference loop
    for video_path in video_list:
        img = read_all_frames(video_path, to_rgb=True, as_pil=True)
        img = img[0]  # use only the first frame for feature extraction
        
        relative_video_path = os.path.relpath(video_path, args.image_path).replace('.mp4', '')
        gt_image_path = os.path.join(gt_img_dir, relative_video_path+'_frames', 'frame_00.jpg')
        feature_path = os.path.join(args.output_dir, 'feature', relative_video_path)
        
        gt2_image_path = os.path.join(gt_img_dir, relative_video_path+'_frames', 'frame_09.jpg')
        feature2_path = os.path.join(args.output_dir, 'feature', relative_video_path+'_10th_frame')
        
        layers = os.listdir(feature_path)
        
        print(f'===Processing video: {relative_video_path}===')
        
        for layer in layers:
            print(f'---Processing layer: {layer}---')
            layer_name = layer.split('.')[0]
            layer_id = layer_name.split('@')[1]
            
            if 'ff' not in layer_name:
                print(f'Skipping layer {layer_name} as it is not a target feature layer.')
                continue
            
            if int(layer_id) not in [0,1,2,3,10,11,12,13,20,21,22,23]:
                print(f'Skipping layer {layer_name} as it is not in the target layers.')
                continue
            
            save_path = os.path.join(org_matching, relative_video_path, f'{layer_name}_matching.png')
            anyup_save_path = os.path.join(anyup_matching, relative_video_path, f'{layer_name}_matching.png')
            bilinear_save_path = os.path.join(bilinear_matching, relative_video_path, f'{layer_name}_matching.png')
            
            
            # Target flags
            DO_ORG = True
            DO_ANYUP = True
            DO_BILINEAR = True
            
            query_x = 150
            query_y = 150
            
            if DO_ORG and os.path.exists(save_path):
                print(f'Skipping existing file: {save_path}')
                continue
            if DO_ANYUP and os.path.exists(anyup_save_path):
                print(f'Skipping existing file: {anyup_save_path}')
                continue
            if DO_BILINEAR and os.path.exists(bilinear_save_path):
                print(f'Skipping existing file: {bilinear_save_path}')
                continue
            
            
            layer_path = os.path.join(feature_path, layer)
            features = torch.load(layer_path)  # (Timestep, 1024, 1536)
            features = torch.stack(features, dim=0)  # (Timestep, 1024, 1536)
            
            features2_path = os.path.join(feature2_path, layer)
            features2 = torch.load(features2_path)  # (Timestep, 1024, 1536)
            features2 = torch.stack(features2, dim=0)  # (Timestep, 1024, 1536)
            
            
            org_cost_maps = {}
            anyup_cost_maps = {}
            bilinear_cost_maps = {}
            
            # Upsample PCA visualization for better visibility
            hr_img = Image.open(os.path.join(gt_image_path.replace('-frames', '-LQ-Ours-frames'))).convert('RGB')
            hr_tensor = transforms.ToTensor()(hr_img).unsqueeze(0).to(device)
            
            hr2_img = Image.open(os.path.join(gt2_image_path.replace('-frames', '-LQ-Ours-frames'))).convert('RGB')
            hr2_tensor = transforms.ToTensor()(hr2_img).unsqueeze(0).to(device)
            
            
            for i in tqdm(range(len(timesteps))):
                timestep = timesteps[i]
                feat_t = features[i] # (1024, 1536)
                feat_t2 = features2[i] # (1024, 1536)
                
                if DO_ORG:
                    feat_t_reshaped = feat_t.reshape(32,32,-1).permute(2,0,1).unsqueeze(0)  # (1, C, 32, 32)
                    feat_t2_reshaped  = feat_t2.reshape(32,32,-1).permute(2,0,1).unsqueeze(0)  # (1, C, 32, 32)
                    # Compute correlation between feat_t and feat_t2
                    feat_t_reshaped = feat_t_reshaped.float().contiguous()
                    feat_t2_reshaped = feat_t2_reshaped.float().contiguous()
                    cost_volume = CorrBlock.corr(feat_t_reshaped, feat_t2_reshaped)  # (1, 32, 32, 1, 32, 32)
                    target_y, target_x = query_y//16, query_x//16
                    target_cost_map = cost_volume[0, target_y, target_x, 0]  # (32, 32)
                    target_cost_map = target_cost_map.detach().cpu().numpy()
                    
                    org_cost_maps[f't{timestep}'] = target_cost_map
                    
                    
                if DO_ANYUP:
                    # Upsample using AnyUp
                    feat_t_tensor = feat_t.permute(1,0).reshape(-1, 32, 32).unsqueeze(0).to(device) 
                    feat_t2_tensor = feat_t2.permute(1,0).reshape(-1, 32, 32).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        feat_t_tensor = feat_t_tensor.to(torch.float32)
                        feat_t2_tensor = feat_t2_tensor.to(torch.float32)
                        upsampled_feat = upsampler(hr_tensor, feat_t_tensor, output_size=(64,64))
                        upsampled_feat2 = upsampler(hr2_tensor, feat_t2_tensor, output_size=(64,64))
                    
                    upsampled_feat = upsampled_feat.squeeze(0).reshape(-1,64*64).permute(1,0)  # (4096, C)
                    upsampled_feat2 = upsampled_feat2.squeeze(0).reshape(-1,64*64).permute(1,0)  # (4096, C)
                    # Compute correlation between upsampled_feat and upsampled_feat2
                    upsampled_feat = upsampled_feat.reshape(1, -1, 64, 64).permute(0,1,2,3)  # (1, C, 64, 64)
                    upsampled_feat2 = upsampled_feat2.reshape(1, -1, 64, 64).permute(0,1,2,3)  # (1, C, 64, 64)
                    cost_volume_anyup = CorrBlock.corr(upsampled_feat, upsampled_feat2)  # (1, 64, 64, 1, 64, 64)
                    target_y_anyup, target_x_anyup = query_y//8, query_x//8
                    target_cost_map_anyup = cost_volume_anyup[0, target_y_anyup, target_x_anyup, 0]
                
                    anyup_cost_maps[f't{timestep}'] = target_cost_map_anyup.detach().cpu().numpy()
                    
                if DO_BILINEAR:
                    feat_t_tensor = feat_t.reshape(32,32,-1).permute(2,0,1).unsqueeze(0)  # (1, C, 32, 32)
                    feat_t2_tensor = feat_t2.reshape(32,32,-1).permute(2,0,1).unsqueeze(0)  # (1, C, 32, 32)
                    feat_t_tensor = feat_t_tensor.float().contiguous()
                    feat_t2_tensor = feat_t2_tensor.float().contiguous()
                    # Upsample using Bilinear
                    bilinear_feat = F.interpolate(feat_t_tensor, size=(64,64), mode='bilinear', align_corners=False)
                    bilinear_feat2 = F.interpolate(feat_t2_tensor, size=(64,64), mode='bilinear', align_corners=False)
                    cost_volume_bilinear = CorrBlock.corr(bilinear_feat, bilinear_feat2)  # (1, 64, 64, 1, 64, 64)
                    target_y_bilinear, target_x_bilinear = query_y//8, query_x//8
                    target_cost_map_bilinear = cost_volume_bilinear[0, target_y_bilinear, target_x_bilinear, 0]
                    
                    bilinear_cost_maps[f't{timestep}'] = target_cost_map_bilinear.detach().cpu().numpy()
                
                        
            # Save PCA visualizations
            
            if DO_ORG:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plot_grid_with_query(
                    hr_img,
                    query_x,
                    query_y,
                    hr2_img,
                    org_cost_maps,
                    timesteps,
                    grid_size=(4, 10),
                    output_path=save_path,
                    layer_id=layer_id,
                    feature_type='org',
                    title_prefix='t'
                )
            
            if DO_ANYUP:
                os.makedirs(os.path.dirname(anyup_save_path), exist_ok=True)
                plot_grid_with_query(
                    hr_img,
                    query_x,
                    query_y,
                    hr2_img,
                    anyup_cost_maps,
                    timesteps,
                    grid_size=(4, 10),
                    output_path=anyup_save_path,
                    layer_id=layer_id,
                    feature_type='anyup',
                    title_prefix='t' 
                )
                
            if DO_BILINEAR:
                os.makedirs(os.path.dirname(bilinear_save_path), exist_ok=True)
                plot_grid_with_query(
                    hr_img,
                    query_x,
                    query_y,
                    hr2_img,
                    bilinear_cost_maps,
                    timesteps,
                    grid_size=(4, 10),
                    output_path=bilinear_save_path,
                    layer_id=layer_id,
                    feature_type='bilinear',
                    title_prefix='t'
                )
        break # for debug, process only one video
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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



