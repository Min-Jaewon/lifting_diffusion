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

from pipelines.pipeline_dit4sr_video import StableDiffusion3ControlNetPipeline
from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix

from torchvision import transforms
import time
# from cvlab12_codes.pipeline_dit4sr_video import StableDiffusion3ControlNetPipeline
from tqdm import tqdm
import RAFT.raft_tools as raft_tools
logger = get_logger(__name__, log_level="INFO")

tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

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
    
    # Load RAFT model for feature extraction
    raft = raft_tools.get_raft_model(args.raft_model_path,
                                        mixed_precision=args.mixed_precision,
                                        feature_upsample=args.feature_upsample,
                                        use_raft_encoder=args.use_raft_encoder,
                                        conv_1x1_channels=args.conv_1x1_channels,
                                        use_l2_norm=args.use_l2_norm,
                                        use_context_dpt=args.use_context_dpt,
                                        use_custom_dpt=args.use_custom_dpt,
                                        evaluation=True,)
    print("Loaded RAFT model for feature extraction.")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)
    transformer.requires_grad_(False)
    raft.requires_grad_(False)

    # Get the validation pipeline
    validation_pipeline = StableDiffusion3ControlNetPipeline(
        vae=vae, text_encoder=text_encoder_one, text_encoder_2=text_encoder_two, text_encoder_3=text_encoder_three, 
        tokenizer=tokenizer_one, tokenizer_2=tokenizer_two, tokenizer_3=tokenizer_three, 
        transformer=transformer, scheduler=scheduler, raft=raft,
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
    transformer.to(accelerator.device, dtype=weight_dtype)
    
    if args.use_2nd_device:
        vae.to('cuda:1', dtype=weight_dtype)
        raft.to('cuda:1',)
    else:
        raft.to(accelerator.device,)
        vae.to(accelerator.device, dtype=weight_dtype)

    return validation_pipeline


def main(args):
    FPS_MAP = {
        'REDS': 60,
        'UDM10' : 30,
        'VideoLQ' : 30,
        'SPMCS' : 30,
        'YouHQ40' : 30,
    }
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dit4sr")

    pipeline = load_dit4sr_pipeline(args, accelerator)
    if accelerator.is_main_process:
        for dataset_name in FPS_MAP.keys():
            if dataset_name in args.video_path:
                video_fps = FPS_MAP[dataset_name]
                print(f'Set fps for {dataset_name} : {video_fps}')
        
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator.manual_seed(args.seed)

        video_list = sorted(glob.glob(os.path.join(args.video_path, '*.mp4')))
        prompt_list = sorted(glob.glob(os.path.join(args.prompt_path, '*.txt')))
        if args.end_idx == -1:
            args.end_idx = len(video_list)
            
        passed_video_list = []
        
        for image_idx, (video_file, prompt_name) in enumerate(zip(video_list, prompt_list)):
            assert os.path.basename(video_file).split('.')[0] == os.path.basename(prompt_name).split('.')[0], f"Video and prompt file names do not match: {video_file}, {prompt_name}"
            video_basename = os.path.basename(video_file).split('.')[0]
            if image_idx < args.start_idx or image_idx >= args.end_idx:
                continue
            
            print(f'================== process {image_idx} video... ===================')
            if args.save_flow:
                args.flow_save_dir = os.path.join(args.output_dir, 'flows', f'{video_basename}')
                os.makedirs(os.path.join(args.output_dir, 'flows'), exist_ok=True)
            # Get Image as (F, 3, H, W)
            import cv2
            cap = cv2.VideoCapture(video_file)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame)
                frames.append(pil_frame)
            cap.release()
            video_length = len(frames)
            
            
            print(f'video length: {video_length} / fps: {video_fps}')
            
            with open(prompt_name, 'r') as f:
                validation_prompt = f.read()
            validation_prompt = remove_focus_sentences(validation_prompt)
            validation_prompt += ' ' + args.added_prompt # clean, extremely detailed, best quality, sharp, clean
            negative_prompt = args.negative_prompt #dirty, messy, low quality, frames, deformed, 
            
            org_width, org_height = frames[0].size
            
            padding_flag = False
            resize_flag = False
            if (org_width // 8) * (org_height // 8) <= args.latent_tiled_size **2:
                # Does not need tile inference
                # --> must be square number
                if (org_width // 8) * (org_height // 8) == int((org_width//8)*(org_height//8)**0.5) **2:
                    print(f'No tile inference for size ({org_width}, {org_height})')
                else:
                    # Resize to longer size
                    resize_flag = True
                    resize_length = max(org_width, org_height)
                    print(f'Resize from ({org_width}, {org_height}) to ({resize_length}, {resize_length}) for tile inference')
                    resized_frames = []
                    for frame in frames:
                        resized_frame = frame.resize((resize_length, resize_length), resample=Image.BICUBIC)
                        resized_frames.append(resized_frame)
                    frames = resized_frames
            
            resize_w, resize_h = frames[0].size            
            if (resize_h % 16 != 0) or (resize_w % 16 != 0):
                import torch.nn.functional as F
                padding_flag = True
                base = 16
                # Size must be multiple of 'base'using F.pad
                new_width = ((resize_w - 1) // base + 1) * base
                new_height = ((resize_h - 1) // base + 1) * base
                print(f'Padding from ({resize_w}, {resize_h}) to ({new_width}, {new_height})')
                padded_frames = []
                for frame in frames:
                    padding = F.pad(tensor_transforms(frame).unsqueeze(0), (0, new_width - resize_w, 0, new_height - resize_h), mode='reflect')
                    padded_frame = transforms.ToPILImage()(padding.squeeze(0))
                    padded_frames.append(padded_frame)
                frames = padded_frames
            
            # frames = frames[:5] # For debugging, use only 5 frames
            # frames = frames[20:23]
            # print(f'Prompt: {validation_prompt}')
            width, height = frames[0].size

            print(f'input size: {height}x{width}')
            generator.manual_seed(image_idx)
            
            # Prepare noise
            if args.use_same_noise:
                noise = torch.randn((1, 16, height//8, width//8), generator=generator, device=accelerator.device).repeat(len(frames), 1, 1, 1)
            else:
                noise = torch.randn((len(frames), 16, height//8, width//8), generator=generator, device=accelerator.device)
            total_time = 0.0
            output_images = []
            
            # print(f'Processing frame {frame_idx+1}~{frame+args.chunk_size}/{video_length}')
            # ###### DEBUGING ONLY ######
            # debugging_pil = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
            # output_images.append(debugging_pil)  # for breakpoint()
            # continue
            # ###### DEBUGING ONLY ######
            frames = torch.stack([tensor_transforms(frame).to(accelerator.device) for frame in frames], dim=0)
            
            with torch.autocast("cuda"):
                start_time = time.time()
                images = pipeline(
                        prompt=validation_prompt, control_image=frames, num_inference_steps=args.num_inference_steps, generator=generator, height=height, width=width,
                        guidance_scale=args.guidance_scale, negative_prompt=negative_prompt,
                        start_point=args.start_point, latent_tiled_size=args.latent_tiled_size, latent_tiled_overlap=args.latent_tiled_overlap, latents=noise,
                        args=args, return_dict=False,
                    )[0]
                end_time = time.time()
                total_time += (end_time - start_time)
                
            for i_frame in range(len(images)):
                image = images[i_frame]
                frame = frames[i_frame].cpu().numpy().transpose(1,2,0)
                
                if padding_flag:
                    image = image.crop((0, 0, resize_w, resize_h))
                if resize_flag:
                    image = image.resize((org_width, org_height), resample=Image.BICUBIC)
                
                if args.align_method == 'nofix':
                    image = image
                else:
                    if args.align_method == 'wavelet':
                        image = wavelet_color_fix(image, frame)
                    elif args.align_method == 'adain':
                        image = adain_color_fix(image, frame)
                
                output_images.append(image)
            
            # Save as mp4
            video_save_path = os.path.join(args.output_dir, '_mp4', f'{video_basename}.mp4')
            os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
            os.makedirs(os.path.join(args.output_dir, f'{video_basename}'), exist_ok=True)

            video_width, video_height = output_images[0].size[0], output_images[0].size[1]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_save_path, fourcc, video_fps, (video_width, video_height))
            for frame_idx in range(len(output_images)):
                output_image = output_images[frame_idx]
                output_image.save(os.path.join(args.output_dir, f'{video_basename}/{frame_idx:04d}.png'))
                
                output_frame = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)
                video_writer.write(output_frame)
                
            
            video_writer.release()
            print(f'inference time: {total_time:.3f}s')
            
            # Post-process with ffmpeg to ensure compatibility
            import subprocess
            tmp_path = video_save_path.replace('.mp4', '_tmp.mp4')
            
            subprocess.run([
                'ffmpeg', '-y', '-i', video_save_path,
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                tmp_path
            ])
            os.replace(tmp_path, video_save_path)
            print(f'Saved video to {video_save_path}')
            
            # Clean the cache
            if 'frames' in locals():
                del frames
            if 'images' in locals():
                del images
            if 'output_images' in locals():
                del output_images
            if 'noise' in locals():
                del noise
            
            if dataset_name == 'UDM10':
                # Logging time for UDM10 only
                with open(os.path.join(args.output_dir, 'udm10_times.txt'), 'a') as f:
                    f.write(f'{video_basename}: {total_time:.3f}s3\n')
            
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            gc.collect()
            torch.cuda.ipc_collect()
            # ✅ CUDA 메모리 강제 정리
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        
            
            # ✅ 메모리 상태 출력
            print(f'GPU memory after: {torch.cuda.memory_allocated() / 1024**3:.2f} GB')
            print(f'GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB\n')
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Feature Extraction Settings
    parser.add_argument("--raft_model_path", type=str, default='', help="Pretrained RAFT model path.")
    parser.add_argument("--use_raft_encoder", action="store_true", default=True, help="Whether to use RAFT encoder features.")
    parser.add_argument("--conv_1x1_channels", type=int, default=256, help="Number of output channels for 1x1 conv")
    parser.add_argument("--use_l2_norm", action="store_true", default=False, help="Whether to L2 normalize the extracted features from RAFT.")
    parser.add_argument("--use_context_dpt", action="store_true", default=True, help="Whether to use DPTHead to extract context features.")
    parser.add_argument("--use_custom_dpt", action="store_true", default=True, help="Whether to use CustomDPTHead for context features.")
    parser.add_argument("--feature_upsample", type=str, choices=['bilinear', 'conv', 'dpt'], default='dpt', help="Feature upsample method in RAFT.")
    parser.add_argument("--target_modules", nargs='*', type=str, default=['ff'], help="Target modules to extract features in DiT4SR.")
    parser.add_argument("--target_indices", nargs='*', type=int, default=[0,1,2,3], help="Target layer indices to extract features in DiT4SR.")
    parser.add_argument("--cpu_offload", action="store_true", default=False, help="Whether to offload models to CPU to save GPU memory.")
    parser.add_argument("--use_2nd_device", action="store_true", default=False, help="Whether to use 2nd GPU device for RAFT model.")
    parser.add_argument("--save_flow", action="store_true", default=False, help="Whether to save the estimated optical flow.")
    parser.add_argument("--flow_save_dir", type=str, default='results/flows', help="Directory to save the estimated optical flows.")
    # Full Attn Settings
    parser.add_argument("--target_lifting_layer", nargs="*", type=int, default=None, help="Target lifting layer indices.")
    parser.add_argument("--use_same_noise", action="store_true", default=False, help="Whether to use same noise for all frames.")
    # Warping Settings
    parser.add_argument("--warpping_start_step", type=int, default=0, help="The step to start applying flow-based warping.")
    parser.add_argument("--warpping_end_step", type=int, default=40, help="The step to end applying flow-based warping.")
    parser.add_argument("--fuse_scale", type=float, default=0.5, help="The weight for interoplation between warped latents and original latents.")
    parser.add_argument("--warping_mode", type=str, default=None,)
    # General Settings
    parser.add_argument("--chunk_size", type=int, default=4) # Regarded as batch size
    parser.add_argument("--overlap_size", type=int, default=1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
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
    parser.add_argument("--video_path", type=str, default=None)
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



