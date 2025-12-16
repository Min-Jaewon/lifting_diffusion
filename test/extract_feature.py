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

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )
    # Setting Seed
    seed = 42
    set_seed(seed) # Fix for debugging

    pipeline = load_dit4sr_pipeline(args, accelerator)
    generator = torch.Generator(device=accelerator.device)
    generator.manual_seed(seed)

    # Prepare Input
    video_list = list(iter_frames_dirs(args.image_path))
    np.random.seed(42)
    video_list = np.random.choice(video_list, size=10, replace=False).tolist()
    model = pipeline.transformer
    
    relative_paths = [os.path.relpath(p, args.image_path) for p in video_list]
    prompt_path = [os.path.join(args.prompt_path, p.replace('.mp4', '.txt')) for p in relative_paths]
    
    
    # Set up hooks to extract features
    target_modules = ['ff'] # Input of 'norm2' -> Pre-AdaLN, Input of 'ff' -> Post-AdaLN
    pattern = rf"transformer_blocks\.(\d+)\.(?:{'|'.join(target_modules)})$"
    target_indices = [_ for _ in range(24)] # all layers
    target_features = defaultdict(list)
   
    def get_input_hook(name):
        def hook(module, input, output):
            # [2, 2048, 1536] -> [1024, 1536] 
            if input[0].shape[0] == 2:
                feature = input[0][0].chunk(2, dim=0)[0].detach().cpu()
            else:
                feature = input[0].detach().cpu()
            target_features[name].append(feature)
        return hook
        
    def register_hooks(model, indices=None):
        handles = []
        for name, module in model.named_modules():
            is_match = re.match(pattern, name)
            if is_match:
                block_idx = int(name.split('.')[1])
                if block_idx in indices:
                    handle = module.register_forward_hook(get_input_hook(name))
                    handles.append(handle)
                    
        return handles
    
    handles = register_hooks(model, target_indices)
    
    # Inference loop
    for i_seed, (video_path, p_path) in enumerate(zip(video_list, prompt_path)):
        imgs = read_all_frames(video_path, to_rgb=True, as_pil=True)
        np.random.seed(i_seed)
        frame_idx = np.random.randint(0, len(imgs)-1)
        for idx in [frame_idx+1, frame_idx+1]:
            img = imgs[idx]  # randomly pick one frame for feature extraction
            relative_video_path = os.path.relpath(video_path, args.image_path).replace('.mp4', '')
            with open(p_path, 'r') as f:
                prompt = f.read()
            prompt = remove_focus_sentences(prompt)
            prompt = prompt + ', ' + args.added_prompt
            negative_prompt = args.negative_prompt
            
            print(f'===Processing video: {relative_video_path}===')
            print(f'===Prompt: {prompt}===')
            
            width, height = img.size
            print(f'Image size: {width}x{height}')
            
            feature_save_dir = os.path.join(args.output_dir, f'feature')
            result_save_dir = os.path.join(args.output_dir, f'sample')
            
            os.makedirs(feature_save_dir, exist_ok=True)
            os.makedirs(result_save_dir, exist_ok=True)
            
            # Clear previous features
            target_features.clear()
            
            # Run inference
            with torch.autocast("cuda"):
                start_time = time.time()
                image = pipeline(
                    prompt=prompt,
                    control_image=img,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    height=height,
                    width=width,
                    guidance_scale=args.guidance_scale,
                    negative_prompt=negative_prompt,
                    start_point=args.start_point,
                    latent_tiled_size=args.latent_tiled_size,
                    latent_tiled_overlap=args.latent_tiled_overlap,
                    args=args,
                ).images[0]
                end_time = time.time()
                print(f'Inference time for sample {end_time - start_time:.2f} seconds')

            if args.align_method == 'nofix':
                image = image
            else:
                if args.align_method == 'wavelet':
                    image = wavelet_color_fix(image, img)
                elif args.align_method == 'adain': # default
                    image = adain_color_fix(image, img)
            
            save_image_path = os.path.join(result_save_dir, relative_video_path, f'frame_{idx:02d}.png')
            os.makedirs(os.path.dirname(save_image_path), exist_ok=True)
            image.save(save_image_path)
            
            
            # Save extracted features
            for name, feat in target_features.items():
                save_name = name.replace('.', '@')
                agg_feat = torch.stack(feat, dim=0)  # (Timestep, 1024, 1536)
                save_feat_path = os.path.join(feature_save_dir, relative_video_path, f'{save_name}_{idx:02d}.pt')
                os.makedirs(os.path.dirname(save_feat_path), exist_ok=True)
                torch.save(feat, save_feat_path)
                print(f'Saved feature: {save_feat_path}')
            break
        
        
        
    
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



