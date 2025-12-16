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

logger = get_logger(__name__, log_level="INFO")

tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

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
    the expected structure: root/category/unique_id/video_id/flow_*.npy
    """
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise NotADirectoryError(f"Root not found: {root}")
    for category in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        category_path = os.path.join(root, category)
        for unique_id in sorted(d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))):
            uid_path = os.path.join(category_path, unique_id)
            for video_id in sorted(os.listdir(uid_path)):
                video_path = os.path.join(uid_path, video_id)
                if os.path.isdir(video_path):
                    for flow_name in sorted(os.listdir(video_path)):
                        if flow_name.endswith(".npy") and flow_name.startswith("flow_"):
                            flow_path =  os.path.join(video_path, flow_name)
                            yield flow_path
def remove_focus_sentences(text):
    prohibited_words = ['focus', 'focal', 'prominent', 'close-up', 'black and white', 'blur', 'depth', 'dense', 'locate', 'position']
    parts = re.split(r'([.?!])', text)
    
    filtered_sentences = []
    i = 0
    while i < len(parts):
        sentence = parts[i]
        punctuation = parts[i+1] if (i+1 < len(parts)) else ''

        full_sentence = sentence + punctuation
        
        full_sentence_lower = full_sentence.lower()
        skip = False
        for word in prohibited_words:
            if word.lower() in full_sentence_lower:
                skip = True
                break
        
        if not skip:
            filtered_sentences.append(full_sentence)
        
        i += 2
    
    return "".join(filtered_sentences).strip()

# Copied from dreambooth sd3 example
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

# Copied from dreambooth sd3 example
def load_text_encoders(class_one, class_two, class_three, args):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


def load_dit4sr_pipeline(args, accelerator):
    
    from model_dit4sr.transformer_sd3 import SD3Transformer2DModel

    # Load scheduler, tokenizer and models.
    
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        args.transformer_model_name_or_path, subfolder="transformer"
    )
    # controlnet = SD3ControlNetModel.from_pretrained(args.controlnet_model_name_or_path, subfolder='controlnet')
    # Load the tokenizer
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
    )

    # import correct text encoder class
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )

    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
            text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three, args
        )

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)
    transformer.requires_grad_(False)

    # Get the validation pipeline
    validation_pipeline = StableDiffusion3ControlNetPipeline(
        vae=vae, text_encoder=text_encoder_one, text_encoder_2=text_encoder_two, text_encoder_3=text_encoder_three, 
        tokenizer=tokenizer_one, tokenizer_2=tokenizer_two, tokenizer_3=tokenizer_three, 
        transformer=transformer, scheduler=scheduler,
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)

    return validation_pipeline

def main(args):
    txt_path = os.path.join(args.output_dir, 'txt')
    os.makedirs(txt_path, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Setting Seed
    seed = 42
    set_seed(seed) # Fix for debugging

    # pipeline = load_dit4sr_pipeline(args, accelerator)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # Prepare Input
    flow_path = '/mnt/dataset2/jaewon/YouHQ/YouHQ-Train-flow-sea-full'
    flow_list = list(iter_frames_dirs(flow_path))
    
    img_preproc = transforms.Compose([       
                transforms.ToTensor(),
            ])
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    # Pick randomly 10 flow files for testing
    
    random_indices = np.random.choice(len(flow_list), size=10, replace=False)
    flow_list = [flow_list[i] for i in random_indices]
    
    vae.requires_grad_(False)
    vae.to(device)
    for f_path in flow_list:
        frame_idx = int(re.findall(r'flow_(\d+)\.npy', os.path.basename(f_path))[0])
        flow_dir = os.path.dirname(f_path)
        frame_dir = os.path.dirname(f_path).replace('-flow-sea-full', '-LQ-Ours-frames')
        frame_1_path = os.path.join(frame_dir+'_frames', f'frame_{frame_idx:02d}.jpg')
        frame_2_path = os.path.join(frame_dir+'_frames', f'frame_{frame_idx+1:02d}.jpg')
        
        back_f_path = os.path.join(flow_dir, f'backward_flow_{frame_idx+1:02d}.npy')
        
        rel_path = os.path.relpath(f_path, flow_path)
        video_id = os.path.dirname(rel_path).replace('/', '_')
        
        frame_1 = Image.open(frame_1_path).convert('RGB')
        frame_2 = Image.open(frame_2_path).convert('RGB')
        
        img1 = img_preproc(frame_1)
        img1 = img1.unsqueeze(0).to(device) * 2.0 - 1.0
        
        img2 = img_preproc(frame_2)
        img2 = img2.unsqueeze(0).to(device) * 2.0 - 1.0
        
        flow_forward = np.load(f_path)  # (H, W, 2), flow from frame1 to frame2
        flow_backward = np.load(back_f_path)  # (H, W, 2), flow from frame2 to frame1
        
        original_flow_forward = flow_forward.copy()
        
        flow_forward = torch.from_numpy(flow_forward).to(device)
        flow_backward = torch.from_numpy(flow_backward).to(device)
        
        # interpolate flows to match latent size
        import torch.nn.functional as F
        h,w = 64, 64  # latent size for sd3 medium
        w_f = 512
        s = 1.0*w/w_f
        
        flow_forward = F.interpolate(flow_forward.permute(2,0,1).unsqueeze(0), size=(h,w), mode='bilinear', align_corners=True) * s
        flow_backward = F.interpolate(flow_backward.permute(2,0,1).unsqueeze(0), size=(h,w), mode='bilinear', align_corners=True) * s
        
        flows = torch.stack([flow_forward, flow_backward], dim=0).squeeze(1)  # (2, H, W, 2)
        
        with torch.no_grad():
            img1_latents = vae.encode(img1).latent_dist.sample()
            img1_latents = (img1_latents - vae.config.shift_factor)  * vae.config.scaling_factor

            img2_latents = vae.encode(img2).latent_dist.sample()
            img2_latents = (img2_latents - vae.config.shift_factor) * vae.config.scaling_factor
            
        from pipelines.propagator import flow_warp, fbConsistencyCheck
        
        warped_latents = flow_warp(img1_latents, flow_forward.permute(0,2,3,1))
        valid_mask = fbConsistencyCheck(flow_forward, flow_backward, alpha1=0.01, alpha2=0.5)
        valid_latents = warped_latents * valid_mask
        
        # interpolate missing regions with img2_latents
        interpolated_latents = img1_latents * 0.5 + warped_latents * 0.5
        final_latents = interpolated_latents * valid_mask + img2_latents * (1 - valid_mask)
        
        # Visualization valid mask 1->white, 0->black
        valid_mask_img = valid_mask.squeeze(0).squeeze(0).cpu().numpy() * 255
        valid_mask_img = valid_mask_img.astype(np.uint8)
        valid_mask_img = np.stack([valid_mask_img]*3, axis=-1)
        valid_mask_img = Image.fromarray(valid_mask_img)
        
        
        # Decoding and saving
        with torch.no_grad():
            image = vae.decode(valid_latents / vae.config.scaling_factor + vae.config.shift_factor).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)

            decoded_img_1 = vae.decode(img1_latents / vae.config.scaling_factor + vae.config.shift_factor).sample
            decoded_img_1 = (decoded_img_1 / 2 + 0.5).clamp(0, 1)
            decoded_img_1 = decoded_img_1.cpu().permute(0, 2, 3, 1).numpy()[0]
            decoded_img_1 = (decoded_img_1 * 255).astype(np.uint8)
            decoded_img_1 = Image.fromarray(decoded_img_1)
            
            decoded_img_2 = vae.decode(img2_latents / vae.config.scaling_factor + vae.config.shift_factor).sample
            decoded_img_2 = (decoded_img_2 / 2 + 0.5).clamp(0, 1)
            decoded_img_2 = decoded_img_2.cpu().permute(0, 2, 3, 1).numpy()[0]
            decoded_img_2 = (decoded_img_2 * 255).astype(np.uint8)
            decoded_img_2 = Image.fromarray(decoded_img_2)

            interpolated_img = vae.decode(interpolated_latents / vae.config.scaling_factor + vae.config.shift_factor).sample
            interpolated_img = (interpolated_img / 2 + 0.5).clamp(0, 1)
            interpolated_img = interpolated_img.cpu().permute(0, 2, 3, 1).numpy()[0]
            interpolated_img = (interpolated_img * 255).astype(np.uint8)
            interpolated_img = Image.fromarray(interpolated_img)
            
            final_img = vae.decode(final_latents / vae.config.scaling_factor + vae.config.shift_factor).sample
            final_img = (final_img / 2 + 0.5).clamp(0, 1)
            final_img = final_img.cpu().permute(0, 2, 3, 1).numpy()[0]
            final_img = (final_img * 255).astype(np.uint8)
            final_img = Image.fromarray(final_img)  
            
        # Visualization forward flow
        from RAFT.core.utils import flow_viz
        flow_img = flow_viz.flow_to_image(original_flow_forward)
        flow_img = Image.fromarray(flow_img)
        
        # Make Grid with titles on each image
        # frame 1 / frame 2 / flow
        # decoded 1 / decoded 2 / warped
        # interpolated / final / mask
        import matplotlib.pyplot as plt

        # 3x3 figure 생성
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # 15인치는 대략 512×512 이미지 기준 적당
        fig.subplots_adjust(wspace=0.05, hspace=0.3)  # 이미지 간 간격

        img_list = [
            frame_1, frame_2, flow_img,
            decoded_img_1, decoded_img_2, image,
            interpolated_img, final_img, valid_mask_img
        ]

        # 3) plot
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.subplots_adjust(wspace=0.05, hspace=0.3)
        title_list = [
            f'Frame {frame_idx}', f'Frame {frame_idx+1}', 'Flow',
            'Decoded 1', 'Decoded 2', 'Warped',
            'Interp_from_img1', 'Final_for_img2', 'Mask'
        ]

        for ax, img, title in zip(axes.ravel(), img_list, title_list):
            ax.imshow(img)
            ax.set_title(title, fontsize=14, color='white', pad=5, fontweight='bold')
            ax.axis('off')

        fig.patch.set_facecolor('gray')
        # Save grid image
        os.makedirs('warping_results', exist_ok=True)
        grid_save_path = os.path.join('warping_results', f'{video_id}_frame_{frame_idx:02d}_warping.png')
        grid_save_path = grid_save_path.replace('.png', '.jpg')
        plt.savefig(grid_save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        
        
        
    
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



