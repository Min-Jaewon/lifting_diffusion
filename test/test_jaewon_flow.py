#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import logging
import math
import os
import shutil
import sys
sys.path.append(os.getcwd())
from pathlib import Path

import accelerate
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    StableDiffusion3ControlNetPipeline,
    StableDiffusion3Pipeline,
)
from model_dit4sr.transformer_sd3 import SD3Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory, cast_training_params
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

from dataloaders.paired_dataset_sd3_latent import PairedCaptionDataset

### Newly added
import RAFT.raft_tools as raft_tools
from RAFT.core.raft import RAFT
from RAFT.core.utils import flow_viz
from dataloaders.paired_dataset_sd3_latent_jaewon import VideoPairedCaptionDataset, collate_fn
from einops import rearrange, repeat
import re
import numpy as np
import PIL.Image as Image
import wandb
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from torchvision import transforms
import os
import glob
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.0.dev0")

logger = get_logger(__name__)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--transformer_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
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
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--precondition_outputs",
        type=int,
        default=1,
        help="Flag indicating if we are preconditioning the model outputs or not as done in EDM. This affects how "
        "model `target` is calculated.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='NOTHING',
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    parser.add_argument("--root_folders",  type=str , default='' )
    parser.add_argument("--null_text_ratio", type=float, default=0.5)
    parser.add_argument('--trainable_modules', nargs='*', type=str, default=["control"])
    
    # Newly added
    parser.add_argument("--raft_model_path", type=str, default='/mnt/dataset1/m_jaewon/cvpr26/DiT4SR/RAFT/models/raft-things.pth', help="Path to pretrained RAFT model.")
    parser.add_argument("--feature_upsample", type=str, choices=['bilinear', 'conv', 'dpt'], default='dpt', help="Feature upsample method in RAFT.")
    parser.add_argument("--target_modules", nargs='*', type=str, default=['ff'], help="Target modules to extract features in DiT4SR.")
    parser.add_argument("--target_indices", nargs='*', type=int, default=[23], help="Target layer indices to extract features in DiT4SR.")
    parser.add_argument("--use_raft_encoder", action="store_true", default=False, help="Whether to use RAFT encoder features.")
    parser.add_argument("--conv_1x1_channels", type=int, default=256, help="Number of output channels for 1x1 conv")
    parser.add_argument("--use_l2_norm", action="store_true", default=False, help="Whether to L2 normalize the extracted features from RAFT.")
    parser.add_argument("--use_context_dpt", action="store_true", default=False, help="Whether to use DPTHead to extract context features.")
    parser.add_argument("--use_custom_dpt", action="store_true", default=False, help="Whether to use CustomDPTHead for context features.")
    
    # Parse the arguments
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args



# Copied from dreambooth sd3 example
def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


# Copied from dreambooth sd3 example
def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


# Copied from dreambooth sd3 example
def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
    # prompt_embeds = clip_prompt_embeds

    return prompt_embeds, pooled_prompt_embeds


def viz(lq_img, flow_pred,):
    b = 1
    results = []
    lq_img = rearrange(lq_img, '(b f) c h w -> b f c h w', f=2)
    
    for i in range(b):
        # --- LQ 이미지 ---
        lq_img_i = lq_img[i][0].permute(1, 2, 0).detach().cpu().numpy()
        lq_img_i = (lq_img_i * 0.5 + 0.5)              # [-1,1] → [0,1]
        lq_img_i = np.clip(lq_img_i, 0, 1)
        lq_img_i = (lq_img_i * 255).astype(np.uint8)
        
        
        # --- 플로우 맵 ---
        flow_pred_i = flow_pred[i].permute(1, 2, 0).detach().cpu().numpy()
        flow_pred_viz = flow_viz.flow_to_image(flow_pred_i)
        
        # --- 세로로 이어붙이기 ---
        cat_img = np.concatenate([lq_img_i, flow_pred_viz], axis=0)
        
        results.append(Image.fromarray(cat_img))

    return results  

def decode_image(vae, latents):
    # scale and decode the image latents with vae
    latents = 1 / vae.config.scaling_factor * latents + vae.config.shift_factor
    image = vae.decode(latents, return_dict=False)[0]

    # clamp the image to be between 0 and 1
    image = (image / 2 + 0.5).clamp(0, 1)

    return image        

def pred_img_grid(lq_imgs, gt_imgs, pred_imgs,):
    # lq_imgs : (b, 2, c, h, w) [0, 1]
    # gt_imgs : (b, 2, c, h, w) [0, 1]
    # pred_imgs : (b, 2, c, h, w) [0, 1]
    b = lq_imgs.shape[0]
    results = []
    for i in range(b):
        # row1 : (Frame1) LQ | Pred | GT
        # row2: (Frame2) LQ | Pred | GT
        
        lq_img1 = (lq_imgs[i,0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        lq_img2 = (lq_imgs[i,1].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        gt_img1 = (gt_imgs[i,0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        gt_img2 = (gt_imgs[i,1].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        pred_img1 = (pred_imgs[i,0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        pred_img2 = (pred_imgs[i,1].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

        # Concatenate images for grid display
        row1 = np.concatenate([lq_img1, pred_img1, gt_img1], axis=1)
        row2 = np.concatenate([lq_img2, pred_img2, gt_img2], axis=1)
        
        results.append(np.concatenate([row1, row2], axis=0))
        
    # Convert to PIL Images
    results = [Image.fromarray(img) for img in results]
    
    return results

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

def compute_text_embeddings(prompt, text_encoders, tokenizers):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, args.max_sequence_length
        )
        prompt_embeds = prompt_embeds
        pooled_prompt_embeds = pooled_prompt_embeds
    return prompt_embeds, pooled_prompt_embeds

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def main(args):
    # set accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        
    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    if args.transformer_model_name_or_path is not None:
        transformer = SD3Transformer2DModel.from_pretrained_local(
            args.transformer_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
        )
    else:
        transformer = SD3Transformer2DModel.from_pretrained_local(
            args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
        )

    
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
    
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
            text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three, args
        )
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    
    
    transformer.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)
    vae.requires_grad_(False)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
        
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    
    # Feature Extraction Hooks
    target_modules = args.target_modules # Input of 'norm2' -> Pre-AdaLN, Input of 'ff' -> Post-AdaLN
    pattern = rf"transformer_blocks\.(\d+)\.(?:{'|'.join(target_modules)})$"
    target_indices = args.target_indices
    target_features = []
    
    ## Hooks for feature extraction
    def get_input_hook(name):
        def hook(module, input, output):
            # [B, 2, 2048, 1536] -> [B, 1024, 1536] 
            feature = input[0].chunk(2, dim=1)[0]
            target_features.append(feature)
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
    
    handles = register_hooks(transformer, target_indices)
    
    raft = raft_tools.get_raft_model(args.raft_model_path,
                                     mixed_precision=args.mixed_precision,
                                     feature_upsample=args.feature_upsample,
                                     use_raft_encoder=args.use_raft_encoder,
                                     conv_1x1_channels=args.conv_1x1_channels,
                                     use_l2_norm=args.use_l2_norm,
                                     use_context_dpt=args.use_context_dpt,
                                     use_custom_dpt=args.use_custom_dpt,
                                     evaluation=True,)
                               
    raft.requires_grad_(False) 
    raft.to(accelerator.device)    


    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    free_memory()

    video_list = [os.path.join(args.root_folders, d) for d in sorted(os.listdir(args.root_folders))]
    prompt_dir = args.root_folders.replace('-LQ-Ours-frames', '-prompts')
    prompt_list = sorted(glob.glob(os.path.join(prompt_dir, '*.txt')))
    
    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]    
    
    
    img_preproc = transforms.Compose([       
            transforms.ToTensor(),
            ])
    
    timesteps_list, num_inference_steps = retrieve_timesteps(noise_scheduler_copy, 40, accelerator.device)
    # # Save timesteps_list as txt
    # for t in timesteps_list:
    #     with open('./timesteps_list.txt', 'a') as f:
    #         f.write(f'{t.item()}\n')
    for t in range(num_inference_steps):
        if t!=19:
            continue
        timestep = timesteps_list[t].unsqueeze(0).to(accelerator.device)
        for i, (vid_path, prompt_path) in tqdm(enumerate(zip(video_list, prompt_list)), total=len(video_list), desc=f'TimeStep {timestep[0].item()}'):
            assert os.path.basename(vid_path).split('_')[0] == os.path.basename(prompt_path).split('.')[0], "Video and prompt file names do not match."
            video_id = os.path.basename(vid_path).split('_')[0]
            
            # Get Video
            frames = []
            for frame_file in sorted(os.listdir(vid_path)):
                frame_path = os.path.join(vid_path, frame_file)
                frame = Image.open(frame_path).convert('RGB')
                frame = img_preproc(frame)
                frames.append(frame)
                
            # Read Prompt
            with open(prompt_path, 'r') as f:
                prompt = f.read().strip()
            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings([prompt], text_encoders, tokenizers)
            prompt_embeds = torch.cat([prompt_embeds]*2, dim=0).to(device=accelerator.device, dtype=weight_dtype)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embeds]*2, dim=0).to(device=accelerator.device, dtype=weight_dtype)
            
            flows = []
            viz_images = []
            flow_save_dir = os.path.join(args.output_dir, 'lq_flows_ours', f'{video_id}', f'step_{t:03d}')
            os.makedirs(flow_save_dir, exist_ok=True)
            for i in tqdm(range(0, len(frames)-1), desc=f'Processing video {video_id}', total=len(frames)-1):
                lr_frame1 = frames[i].unsqueeze(0)
                lr_frame2 = frames[i+1].unsqueeze(0)
                frame_pair = torch.cat([lr_frame1, lr_frame2], dim=0)
                frame_pair = frame_pair*2 - 1.0  # Scale to [-1, 1]
                frame_pair = frame_pair.to(device=accelerator.device, dtype=weight_dtype)
                
                model_input = vae.encode(frame_pair).latent_dist.sample()
                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                
                controlnet_image = model_input.clone()
                
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input, device=model_input.device, dtype=model_input.dtype)
                bsz = 1
                timesteps = timestep.repeat(2)
                
                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                # input_model_input = torch.cat([noisy_model_input, controlnet_image], dim = 1)

                with torch.no_grad():
                    target_features.clear()
                    # Predict the noise residual
                    model_pred = transformer(
                        hidden_states=noisy_model_input,
                        controlnet_image=controlnet_image,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )[0]
                    
                    # Optical flow prediction
                    diffusion_features = [feat.detach().float() for feat in target_features]
                    lr_frame1 = lr_frame1.to(device=accelerator.device)
                    lr_frame2 = lr_frame2.to(device=accelerator.device)
                    flow_preds = raft(lr_frame1, lr_frame2, diffusion_features, iters=12,)
                    flow_forward = flow_preds[-1]
                    
                viz_image = viz(frame_pair, flow_forward)[0]
                img_save_path = os.path.join(flow_save_dir, f'frame{i:02d}.png')
                viz_image.save(img_save_path)
                flows.append(flow_forward.squeeze(0))
                
                
            flows = torch.stack(flows, dim=0)  # (num_flows, 2, H, W)
            flow_save_path = os.path.join(flow_save_dir, f'flow_{video_id}.npy')
            np.save(flow_save_path, flows.cpu().numpy())
        
            

if __name__ == "__main__":
    args = parse_args()
    main(args)
