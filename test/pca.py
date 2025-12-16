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

logger = get_logger(__name__, log_level="INFO")

tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

def plot_pca_grid(pca_vis_dict, timesteps, grid_size=(2, 10), output_path='pca_grid.png', title_prefix='t'):
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.5))
    axes = axes.flatten()
    for i, timestep in enumerate(timesteps):
        key = f'{title_prefix}_{timestep}'
        if key in pca_vis_dict:
            pca_img = pca_vis_dict[key]
            axes[i].imshow(pca_img)
            axes[i].set_title(f'{title_prefix}={timestep}', fontsize=8)
            axes[i].axis('off')
        else:
            axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
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
    pca_path = 'results/pca_vis'
    anyup_pca_path = 'results/pca_vis_anyup'
    bilinear_pca_path = 'results/pca_vis_bilinear'
    
    os.makedirs(pca_path, exist_ok=True)

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
        layers = os.listdir(feature_path)
        
        print(f'===Processing video: {relative_video_path}===')
        
        for layer in layers:
            print(f'---Processing layer: {layer}---')
            layer_name = layer.split('.')[0]
            layer_id = layer_name.split('@')[1]
            
            if int(layer_id) not in [10,11,12,13]:
                print(f'Skipping layer {layer_name} as it is not in the target layers.')
                continue
            
            save_path = os.path.join(pca_path, relative_video_path, f'{layer_name}_pca_grid.png')
            anyup_save_path = os.path.join(anyup_pca_path, relative_video_path, f'{layer_name}_anyup_pca_grid.png')
            bilinear_save_path = os.path.join(bilinear_pca_path, relative_video_path, f'{layer_name}_bilinear_pca_grid.png')
            
            # Target flags
            DO_ORG = True
            DO_ANYUP = True
            DO_BILINEAR = True
            
            if DO_ORG and os.path.exists(save_path):
                print(f'PCA visualization already exists at {save_path}, skipping...')
                DO_ORG = False
            
            if DO_ANYUP and os.path.exists(anyup_save_path):
                print(f'AnyUp PCA visualization already exists at {anyup_save_path}, skipping...')
                DO_ANYUP = False
            
            if DO_BILINEAR and os.path.exists(bilinear_save_path):
                print(f'Bilinear PCA visualization already exists at {bilinear_save_path}, skipping...')
                DO_BILINEAR = False
            
            
            # if os.path.exists(save_path):
            #     print(f'PCA visualization already exists at {save_path}, skipping...')
            #     continue
            
            
            layer_path = os.path.join(feature_path, layer)
            features = torch.load(layer_path)  # (Timestep, 1024, 1536)
            features = torch.stack(features, dim=0)  # (Timestep, 1024, 1536)
            avg_feature = features.mean(dim=0).numpy()  # (1024, 1536)
            
            pca_vis_dict = {}
            anyup_vis_dict = {}
            bilinear_vis_dict = {}
            
            # Upsample PCA visualization for better visibility
            hr_img = Image.open(os.path.join(gt_image_path.replace('-frames', '-LQ-Ours-frames'))).convert('RGB')
            hr_tensor = transforms.ToTensor()(hr_img).unsqueeze(0).to(device)
            
            for i in tqdm(range(len(timesteps))):
                timestep = timesteps[i]
                feat_t = features[i].numpy()  # (1024, 1536)
                pca = PCA(n_components=3)
                
                if DO_ORG:
                    pca_result = pca.fit_transform(feat_t)  # (1024, 3)
                    pca_min = pca_result.min(axis=0)
                    pca_max = pca_result.max(axis=0)
                    pca_norm = (pca_result - pca_min) / (pca_max - pca_min + 1e-8)  # (1024, 3), normalized to [0, 1]
                    pca_img = pca_norm.reshape(32, 32, 3)  # assuming feature map size is 32x32
                    pca_img = (pca_img * 255).astype(np.uint8)
                    pca_vis_dict[f'timestep_{timestep}'] = Image.fromarray(pca_img)
                
                if DO_ANYUP:
                    # Upsample using AnyUp
                    feat_t_tensor = torch.from_numpy(feat_t).permute(1,0).reshape(-1, 32, 32).unsqueeze(0).to(device) 
                    with torch.no_grad():
                        feat_t_tensor = feat_t_tensor.to(torch.float32)
                        upsampled_feat = upsampler(hr_tensor, feat_t_tensor, output_size=(64,64))
                    
                    upsampled_feat_np = upsampled_feat.squeeze(0).reshape(-1,64*64).permute(1,0).cpu().numpy()  # (4096, C)
                    pca_upsampled_result = pca.fit_transform(upsampled_feat_np)  # (4096, 3)
                    pca_upsampled_min = pca_upsampled_result.min(axis=0)
                    pca_upsampled_max = pca_upsampled_result.max(axis=0)
                    pca_upsampled_norm = (pca_upsampled_result - pca_upsampled_min) / (pca_upsampled_max - pca_upsampled_min + 1e-8)  # (4096, 3)
                    pca_upsampled_img = pca_upsampled_norm.reshape(64, 64, 3)  # (64, 64, 3)
                    pca_upsampled_img = (pca_upsampled_img * 255).astype(np.uint8)
                    anyup_vis_dict[f'timestep_{timestep}'] = Image.fromarray(pca_upsampled_img)
                
                    if DO_BILINEAR:
                        # Upsample using Bilinear
                        bilinear_upsampled_feat = torch.nn.functional.interpolate(
                            feat_t_tensor, size=(64,64), mode='bilinear', align_corners=False
                        )
                        bilinear_upsampled_feat_np = bilinear_upsampled_feat.squeeze(0).reshape(-1,64*64).permute(1,0).cpu().numpy()  # (4096, C)
                        pca_bilinear_result = pca.fit_transform(bilinear_upsampled_feat_np)  # (4096, 3)
                        pca_bilinear_min = pca_bilinear_result.min(axis=0)
                        pca_bilinear_max = pca_bilinear_result.max(axis=0)
                        pca_bilinear_norm = (pca_bilinear_result - pca_bilinear_min) / (pca_bilinear_max - pca_bilinear_min + 1e-8)  # (4096, 3)
                        pca_bilinear_img = pca_bilinear_norm.reshape(64, 64, 3)  # (64, 64, 3)
                        pca_bilinear_img = (pca_bilinear_img * 255).astype(np.uint8)
                        bilinear_vis_dict[f'timestep_{timestep}'] = Image.fromarray(pca_bilinear_img)
                        
            # Save PCA visualizations
            
            if DO_ORG:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plot_pca_grid(pca_vis_dict, timesteps, grid_size=(4, 10), output_path=save_path, title_prefix='timestep')
            
            if DO_ANYUP:
                os.makedirs(os.path.dirname(anyup_save_path), exist_ok=True)
                plot_pca_grid(anyup_vis_dict, timesteps, grid_size=(4, 10), output_path=anyup_save_path, title_prefix='timestep')
                
            if DO_BILINEAR:
                os.makedirs(os.path.dirname(bilinear_save_path), exist_ok=True)
                plot_pca_grid(bilinear_vis_dict, timesteps, grid_size=(4, 10), output_path=bilinear_save_path, title_prefix='timestep')
        
        
    
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



