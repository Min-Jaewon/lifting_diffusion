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
from model_dit4sr.transformer_sd3_ours import SD3Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory, cast_training_params
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

from dataloaders.paired_dataset_sd3_latent import PairedCaptionDataset

### Newly added
import RAFT.raft_tools as raft_tools
from RAFT.core.raft import RAFT
from RAFT.core.utils import flow_viz
from dataloaders.paired_dataset_sd3_latent_baseline_overfitting import VideoPairedCaptionDataset, collate_fn
from einops import rearrange, repeat
from torch.utils.data import Subset
import re
import numpy as np
import PIL.Image as Image
import wandb

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
    parser.add_argument('--trainable_modules', nargs='*', type=str, default=[])
    
    # Newly added
    parser.add_argument("--raft_model_path", type=str, default='/mnt/dataset1/m_jaewon/cvpr26/DiT4SR/RAFT/models/raft-things.pth', help="Path to pretrained RAFT model.")
    parser.add_argument("--only_train_raft", action="store_true", help="Whether or not to only train RAFT model.")
    parser.add_argument("--feature_upsample", type=str, choices=['bilinear', 'conv', 'dpt'], default='bilinear', help="Feature upsample method in RAFT.")
    parser.add_argument("--target_modules", nargs='*', type=str, default=['ff'], help="Target modules to extract features in DiT4SR.")
    parser.add_argument("--target_indices", nargs='*', type=int, default=[23], help="Target layer indices to extract features in DiT4SR.")
    parser.add_argument("--wandb_project", type=str, default='DiT4SR', help="Weights and Biases project name.")
    parser.add_argument("--wandb_name", type=str, default='TEST', help="Weights and Biases experiment name.")
    parser.add_argument("--wandb_key", type=str, default='0543b2d8eeed135a1ff9e10d5da01a1006d8f13e', help="Weights and Biases API key.")
    parser.add_argument("--dataset_size", type=int, default=10000, help="Number of samples in the training dataset.")
    parser.add_argument("--stage1_raft_weight", type=str, default=None, help="Path to the pretrained RAFT weights from stage 1.")
    parser.add_argument("--flow_loss_weight", type=float, default=1.0, help="Weight for flow loss.")
    parser.add_argument("--use_raft_encoder", action="store_true", default=False, help="Whether to use RAFT encoder features.")
    parser.add_argument("--conv_1x1_channels", type=int, default=256, help="Number of output channels for 1x1 conv")
    parser.add_argument("--use_sea_raft", action="store_true", default=False, help="Whether to use SEA-RAFT for optical flow estimation.")
    parser.add_argument("--use_l2_norm", action="store_true", default=False, help="Whether to L2 normalize the extracted features from RAFT.")
    parser.add_argument("--use_context_dpt", action="store_true", default=False, help="Whether to use DPTHead to extract context features.")
    parser.add_argument("--use_custom_dpt", action="store_true", default=False, help="Whether to use CustomDPTHead for context features.")
    parser.add_argument("--target_lifting_layer", nargs="*", type=int, default=None, help="Target lifting layer indices.")
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
    
    # if not args.only_train_raft:
    #     if len(args.target_modules) == 0:
    #         raise ValueError("At least one target module must be specified when training DiT4SR.")
    #     if args.stage1_raft_weight is None:
    #         raise ValueError("Path to the pretrained RAFT weights from stage 1 must be specified when training DiT4SR.")
        

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


def viz(gt_img,lq_img, flow_pred, flow_gt, flow_lq=None):
    """
    gt_img:     (B, C, H, W), 값 [-1, 1] 범위 (보통 Normalize된 이미지)
    lq_img:     (B, C, H, W), 값 [-1, 1] 범위 (보통 Normalize된 이미지)
    flow_pred: (B, 2, H, W) 또는 (B, H, W, 2), optical flow 예측
    flow_gt:   (B, 2, H, W) 또는 (B, H, W, 2), optical flow ground truth
    flow_lq:   (B, 2, H, W) 또는 (B, H, W, 2), optical flow low-quality (입력) --- 선택적 ---
    """
    b = gt_img.shape[0] //2 # 배치 크기 절반 (두 프레임씩 묶여있음)
    results = []
    gt_img = rearrange(gt_img, '(b f) c h w -> b f c h w', f=3)
    lq_img = rearrange(lq_img, '(b f) c h w -> b f c h w', f=3)
    
    for i in range(b):
        # --- 원본 이미지 ---
        gt_img_i = gt_img[i][0].permute(1, 2, 0).detach().cpu().numpy() # 첫 번째 프레임 선택
        # gt_img_i = (gt_img_i * 0.5 + 0.5)              # [-1,1] → [0,1]
        gt_img_i = np.clip(gt_img_i, 0, 1)
        gt_img_i = (gt_img_i * 255).astype(np.uint8)
        
        # --- LQ 이미지 ---
        lq_img_i = lq_img[i][0].permute(1, 2, 0).detach().cpu().numpy()
        # lq_img_i = (lq_img_i * 0.5 + 0.5)              # [-1,1] → [0,1]
        lq_img_i = np.clip(lq_img_i, 0, 1)
        lq_img_i = (lq_img_i * 255).astype(np.uint8)
        
        # --- Place Holder 용 White Image ---
        white_img = (np.ones_like(gt_img_i) * 255).astype(np.uint8)
        
        # --- 플로우 맵 ---
        flow_pred_i = flow_pred[i].permute(1, 2, 0).detach().cpu().numpy()
        flow_gt_i = flow_gt[i].permute(1, 2, 0).detach().cpu().numpy()
        flow_lq_i = flow_lq[i].permute(1, 2, 0).detach().cpu().numpy() if flow_lq is not None else white_img
        
        flow_pred_viz = flow_viz.flow_to_image(flow_pred_i)
        flow_gt_viz = flow_viz.flow_to_image(flow_gt_i)
        flow_lq_viz = flow_viz.flow_to_image(flow_lq_i) if flow_lq is not None else white_img
        
        # --- 가로로 이미지 이어붙이기 ---
        top_row = np.concatenate([gt_img_i, lq_img_i, white_img], axis=1)
        bottom_row = np.concatenate([flow_gt_viz,flow_pred_viz, flow_lq_viz], axis=1)
        cat_img = np.concatenate([top_row, bottom_row], axis=0)
        
        results.append(Image.fromarray(cat_img))

    return results  

@torch.no_grad()
def validation_loop(args, accelerator, transformer, raft, val_dataloader, weight_dtype,
                    noise_scheduler_copy, get_sigmas, target_features, vae=None):
    # Set model to eval mode
    transformer.eval()
    raft.eval()
    
    device = accelerator.device
    
    total_loss = 0.0
    diff_total_loss = 0.0
    flow_total_loss = 0.0
    
    epe_list, px1_list, px3_list, px5_list = [], [], [], []
    img_dict = {}
    
    n_seen = 0
    generator = torch.Generator(device=device)
    for b_idx, batch in enumerate(tqdm(val_dataloader, desc="Validation")):
        with torch.no_grad():
            model_input = batch["pixel_values"].to(device, dtype=weight_dtype)
            controlnet_image = batch["conditioning_pixel_values"].to(device, dtype=weight_dtype)
            generator.manual_seed(b_idx)
            noise = torch.randn(model_input.shape, device=model_input.device, dtype=model_input.dtype, generator=generator)

            
            bsz = model_input.shape[0] // 2
            # Sample a random timestep for each image
            # for weighting schemes where we sample timesteps non-uniformly
            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=bsz,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=args.mode_scale,
            )
            u = repeat(u, 'b -> (b 2)') # since each data point has two frames
            indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
            
            # Add noise according to flow matching.
            # zt = (1 - texp) * x + texp * z1
            sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
            noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
            
            # Get the text embedding for conditioning
            # prompts = compute_text_embeddings(batch, text_encoders, tokenizers)
            prompt_embeds = batch["prompt_embeds"].to(dtype=model_input.dtype)
            pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(dtype=model_input.dtype)
            
            if batch['is_null'][0]:
                prompt_embeds = torch.load(os.path.join(args.root_folders+'-prompt_embeds', batch['video_id'][0]+'.pt')).to(dtype=model_input.dtype).unsqueeze(0)
                pooled_prompt_embeds = torch.load(os.path.join(args.root_folders+'-pooled_prompt_embeds', batch['video_id'][0]+'.pt')).to(dtype=model_input.dtype).unsqueeze(0)    
                print('Null -> Orginal Prompt Embeds Loaded in Validation Loop')
            
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
            
            diffusion_features = target_features
            lr_image = batch["lq_image"].to(dtype=model_input.dtype)
            lr_image1 = rearrange(lr_image, '(b f) c h w -> b f c h w', f=3)[:,0,:,:,:] # take the first frame
            lr_image2 = rearrange(lr_image, '(b f) c h w -> b f c h w', f=3)[:,1,:,:,:] # take the second frame
            
            flow_preds = raft(lr_image1, lr_image2, diffusion_features, iters=12,)
            flow_gt = batch['gt_flow'].to(device, dtype=model_input.dtype)
            
            flow_loss, metrics = raft_tools.sequence_loss(
                flow_preds,
                flow_gt,
                torch.ones_like(flow_gt[:, :1, :, :]),
                0.8,
            )
            
            epe_list.append(metrics['epe'])
            px1_list.append(metrics['1px'])
            px3_list.append(metrics['3px'])
            px5_list.append(metrics['5px']) 
            
            # Preconditioning of the model outputs.
            if args.precondition_outputs:
                model_pred = model_pred * (-sigmas) + noisy_model_input

            # these weighting schemes use a uniform timestep sampling
            # and instead post-weight the loss
            weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

            # flow matching loss
            if args.precondition_outputs:
                target = model_input
            else:
                target = noise - model_input

            # Compute regular loss.
            diff_loss = torch.mean(
                (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )
            diff_loss = diff_loss.mean()

            
            if not args.only_train_raft:
                # add flow loss to total loss
                loss = diff_loss + args.flow_loss_weight * flow_loss
            else:
                loss = flow_loss
            
            total_loss += loss.item()
            flow_total_loss += flow_loss.item()
            diff_total_loss += diff_loss.item()

            n_seen += 1
            
            lq_flow = batch['lq_flow']
            pil_image = viz(
                batch['gt_image'],
                batch['lq_image'],
                flow_preds[-1],
                batch['gt_flow'],
                flow_lq=lq_flow,
            )[0]
            
            if vae is not None:
                model_pred_img = decode_image(vae, model_pred)
                model_pred_img = rearrange(model_pred_img, '(b f) c h w -> b f c h w', f=3)
                gt_img = rearrange(batch['gt_image'], '(b f) c h w -> b f c h w', f=3)
                lq_img = rearrange(batch['lq_image'], '(b f) c h w -> b f c h w', f=3)
                
                timesteps = rearrange(timesteps, '(b f) -> b f', f=3)[:,0]
                
                img_logs = pred_img_grid(
                    lq_imgs=lq_img,
                    gt_imgs=gt_img,
                    pred_imgs=model_pred_img,
                )
                img_log = img_logs[0]
            
            video_id = batch['video_id'][0]
            frame_idx = batch['frame_idx'][0]
            caption = f'{video_id}_frame_{frame_idx-1:02d}'
            img_dict[f'val/flow_viz_{b_idx:02d}'] = wandb.Image(pil_image, caption=caption)
            
            if vae is not None:
                img_dict[f'val/pred_img_{b_idx:02d}'] = wandb.Image(img_log, caption=caption + f'_step{timesteps[0].item()}')

    log_dict = {}
    if n_seen > 0:
        log_dict['val/total_loss'] = total_loss / n_seen
        log_dict['val/flow_loss'] = flow_total_loss / n_seen
        if not args.only_train_raft:
            log_dict['val/diffusion_loss'] = diff_total_loss / n_seen
    log_dict.update({
        'val/epe': float(np.mean(epe_list)),
        'val/1px': float(np.mean(px1_list)),
        'val/3px': float(np.mean(px3_list)),
        'val/5px': float(np.mean(px5_list)),
    })
    log_dict.update(img_dict)
    
    transformer.train()
    raft.train()
            
    return log_dict 

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

def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

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
    transformer.requires_grad_(False)
    for i in range(13, 24):
        block = transformer.transformer_blocks[i]
        # attn2 or tempor
        module = None
        if hasattr(block, 'attn2'):
            module = block.attn2
    
        if module is not None:
            for name, param in module.named_parameters():
                param.zero_()
                param.requires_grad = True
            print(f'transformer block {i} attn2set to trainable with zero initialization.')
        else:
            raise ValueError(f'transformer block {i} has no attn2 module.')
    
    # Load VAE
    if not args.only_train_raft:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
        )
        vae.requires_grad_(False)
    
    # # Feature Extraction Hooks
    # target_modules = args.target_modules # Input of 'norm2' -> Pre-AdaLN, Input of 'ff' -> Post-AdaLN
    # pattern = rf"transformer_blocks\.(\d+)\.(?:{'|'.join(target_modules)})$"
    # target_indices = args.target_indices
    # target_features = []
    
    # ## Hooks for feature extraction
    # def get_input_hook(name):
    #     def hook(module, input, output):
    #         # [B, 2, 2048, 1536] -> [B, 1024, 1536] 
    #         feature = input[0].chunk(2, dim=1)[0]
    #         target_features.append(feature)
    #     return hook
        
    # def register_hooks(model, indices=None):
    #     handles = []
    #     for name, module in model.named_modules():
    #         is_match = re.match(pattern, name)
    #         if is_match:
    #             block_idx = int(name.split('.')[1])
    #             if block_idx in indices:
    #                 handle = module.register_forward_hook(get_input_hook(name))
    #                 handles.append(handle)
                    
    #     return handles
    
    # handles = register_hooks(transformer, target_indices)
    
    # # Load RAFT model for optical flow extraction
    # if args.stage1_raft_weight is not None:
    #     raft = raft_tools.get_raft_model(args.stage1_raft_weight,
    #                                      mixed_precision=args.mixed_precision,
    #                                      feature_upsample=args.feature_upsample,
    #                                      use_raft_encoder=args.use_raft_encoder,
    #                                      conv_1x1_channels=args.conv_1x1_channels,
    #                                      use_l2_norm=args.use_l2_norm,
    #                                      use_context_dpt=args.use_context_dpt,
    #                                      use_custom_dpt=args.use_custom_dpt,)
    #     logger.info(f'Loaded pretrained RAFT weights from {args.stage1_raft_weight} for stage 2 training.')
    # else:
    #     raft = raft_tools.get_raft_model(args.raft_model_path,
    #                                  mixed_precision=args.mixed_precision,
    #                                  feature_upsample=args.feature_upsample,
    #                                  use_raft_encoder=args.use_raft_encoder,
    #                                  conv_1x1_channels=args.conv_1x1_channels,
    #                                  use_l2_norm=args.use_l2_norm,
    #                                  use_context_dpt=args.use_context_dpt,
    #                                  use_custom_dpt=args.use_custom_dpt,)
                               
    # Set which modules to train
    # raft.requires_grad_(True) # Default: train RAFT model
    # if not args.only_train_raft:
        # release the cross-attention part in the unet.
    # if args.target_lifting_layer is not None:
    #     for layer in args.target_lifting_layer:
    #         args.trainable_modules.append(f'transformer_blocks.{layer}.attn')
    transformer.target_lifting_layer = args.target_lifting_layer
    for name, params in transformer.named_parameters():
        # print(name)
        # if name.endswith(tuple(args.trainable_modules)):
        if any(trainable_modules in name for trainable_modules in tuple(args.trainable_modules)):
            print(f'{name} in <transformer> will be optimized.' )
            # for params in module.parameters():
            params.requires_grad = True
            
    # raft_parameters = list(filter(lambda p: p.requires_grad, raft.parameters()))
    transformer_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # num_trainable_params = sum(p.numel() for p in raft_parameters) + sum(p.numel() for p in transformer_parameters)
    num_trainable_params = sum(p.numel() for p in transformer_parameters)
    
    # if args.only_train_raft:
    #     logger.info(f"** Only training RAFT model ***")
    # else:
    #     logger.info(f"Training both RAFT and Transformer model")
    logger.info(f"Training Transformer model")
    # logger.info(f"Number of RAFT trainable parameters: {sum(p.numel() for p in raft_parameters)}")
    logger.info(f"Number of Transformer trainable parameters: {sum(p.numel() for p in transformer_parameters)}")
    logger.info(f"Total number of trainable parameters: {num_trainable_params}")

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]
                    
                    sub_dir = "transformer"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))
                    # except:
                    #     model_name = 'raft'
                    #     torch.save(model.state_dict(), os.path.join(output_dir, f"{model_name}_weights.pt"))

                    i -= 1

        def load_model_hook(models, input_dir):
            # while len(models) > 0:
                # pop models so that they are not loaded again
            model = models.pop()

            load_model = SD3Transformer2DModel.from_pretrained(input_dir, subfolder="transformer") # , low_cpu_mem_usage=False, ignore_mismatched_sizes=True


            # load diffusers style into model
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)


    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    # params_to_optimize = controlnet.parameters()
    params_to_optimize = transformer.parameters()
    # params_to_optimize = raft_parameters + transformer_parameters
    # transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    # params_to_optimize = [transformer_parameters_with_lr]
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    ) #TODO: Separate optimizers for RAFT and Transformer with different learning rates?

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, transformer and text_encoder to device and cast to weight_dtype
    transformer.to(accelerator.device, dtype=weight_dtype)
    # raft.to(accelerator.device, dtype=torch.float32)  # keep RAFT in full precision
    
    if not args.only_train_raft:
        vae.to(accelerator.device)

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    train_dataset = VideoPairedCaptionDataset(root_folder=args.root_folders, 
                                         null_text_ratio=args.null_text_ratio,
                                         use_sea_raft = args.use_sea_raft,)
    # train_dataset = Subset(train_dataset, list(range(10000)))
    val_dataset = Subset(train_dataset, list(range(0, 3)))
    
    # val_dataset = VideoPairedCaptionDataset(root_folder=args.root_folders, 
    #                                    use_sea_raft = args.use_sea_raft,
    #                                    null_text_ratio=0.0,
    #                                    val=True,)
    

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        wandb.login(key=args.wandb_key)
        accelerator.init_trackers(project_name=args.wandb_project,
                                  config=vars(args),
                                  init_kwargs={"wandb": {
                                            "entity": "cvlab-diffusion-restoration",
                                            "name": args.wandb_name,}
                                            }
                                  )

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

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

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Convert images to latent space
                model_input = batch["pixel_values"].to(dtype=weight_dtype)
                # with torch.no_grad():
                #     model_input = vae.encode(pixel_values).latent_dist.sample()
                #     model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                # controlnet(s) inference
                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                # with torch.no_grad():
                #     controlnet_image = vae.encode(controlnet_image).latent_dist.sample()
                #     controlnet_image = (controlnet_image - vae.config.shift_factor)  * vae.config.scaling_factor
                # image_embedding = controlnet_image.view(controlnet_image.shape[0], 16, -1)
                # pad_tensor = torch.zeros(controlnet_image.shape[0], 77 - image_embedding.shape[1], 4096).to(image_embedding.device, dtype=weight_dtype)
                # image_embedding = torch.cat([image_embedding, pad_tensor], dim=1)
                
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0] // 3
                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                u = repeat(u, 'b -> (b 3)') # since each data point has two frames
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                # input_model_input = torch.cat([noisy_model_input, controlnet_image], dim = 1)

                # Get the text embedding for conditioning
                # prompts = compute_text_embeddings(batch, text_encoders, tokenizers)
                prompt_embeds = batch["prompt_embeds"].to(dtype=model_input.dtype)
                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(dtype=model_input.dtype)
                # prompt_embeds = torch.cat([prompt_embeds, image_embedding], dim=-2)
                
                # target_features.clear()
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
                # diffusion_features = target_features
                lr_image = batch["lq_image"].to(dtype=model_input.dtype)
                lr_image1 = rearrange(lr_image, '(b f) c h w -> b f c h w', f=3)[:,0,:,:,:] # take the first frame
                lr_image2 = rearrange(lr_image, '(b f) c h w -> b f c h w', f=3)[:,1,:,:,:] # take the second frame
                # flow_preds = raft(lr_image1, lr_image2, diffusion_features, iters=12,)
                # flow_gt = batch['gt_flow'].to(dtype=model_input.dtype)
                # flow_loss, metrics = raft_tools.sequence_loss(
                #     flow_preds,
                #     flow_gt,
                #     torch.ones_like(flow_gt[:, :1, :, :]),
                #     0.8,
                # )
                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                # Preconditioning of the model outputs.
                if args.precondition_outputs:
                    model_pred = model_pred * (-sigmas) + noisy_model_input

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                if args.precondition_outputs:
                    target = model_input
                else:
                    target = noise - model_input

                # Compute regular loss.
                diff_loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                diff_loss = diff_loss.mean()

                
                # if not args.only_train_raft:
                #     # add flow loss to total loss
                #     loss = diff_loss + args.flow_loss_weight * flow_loss
                # else:
                #     loss = flow_loss
                loss = diff_loss
                    
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    # params_to_clip = controlnet.parameters()
                    params_to_clip = transformer_parameters
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    accelerator.log(
                        {
                            'train/total_loss': loss.detach().item(),
                            'train/lr': lr_scheduler.get_last_lr()[0],
                        }
                    )
                    # if args.only_train_raft:
                    #     accelerator.log(
                    #         {
                    #             'train/total_loss': loss.detach().item(),
                    #             'train/flow_loss': flow_loss.detach().item(),
                    #             'train/lr': lr_scheduler.get_last_lr()[0],
                    #         }
                    #     )
                    # else:
                    #     accelerator.log(
                    #         {
                    #             'train/total_loss': loss.detach().item(),
                    #             'train/flow_loss': flow_loss.detach().item(),
                    #             'train/diff_loss': diff_loss.detach().item(),
                    #             'train/lr': lr_scheduler.get_last_lr()[0],
                    #         }
                    #     )

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                continue
                if global_step % args.validation_steps == 0 or global_step == 1:
                    
                    # Logging Training Flow Visualization
                    flow_lq = batch['lq_flow']
                    
                    pil_images = viz(batch['gt_image'],
                                        batch['lq_image'],
                                        flow_preds[-1],
                                        batch['gt_flow'],
                                        flow_lq=flow_lq,)
                    for idx, pil_image in enumerate(pil_images):
                        video_id = batch['video_id'][idx]
                        frame_idx = batch['frame_idx'][idx]
                        caption = f'{video_id}_frame{frame_idx-1:02d}'
                        null_flag = batch['is_null'][idx]
                        tag = '_null_prompt' if null_flag else ''
                        caption += tag
                        accelerator.log(
                            {f'train/flow_viz_{idx}': wandb.Image(pil_image, caption=caption)}, step=global_step
                        )

                    # Decode sample images for logging
                    if not args.only_train_raft:
                        model_pred_image = decode_image(vae, model_pred.detach())
                        model_pred_image = rearrange(model_pred_image, '(b f) c h w -> b f c h w', f=3)
                        
                        gt_image = rearrange(batch['gt_image'], '(b f) c h w -> b f c h w', f=3)
                        lq_image = rearrange(batch['lq_image'], '(b f) c h w -> b f c h w', f=3)
                        
                        timesteps = rearrange(timesteps, '(b f) -> b f', f=3)[:,0]
                        img_logs = pred_img_grid(lq_image, gt_image, model_pred_image)
                        
                        for idx, img_log in enumerate(img_logs):
                            video_id = batch['video_id'][idx]
                            frame_idx = batch['frame_idx'][idx]
                            timestep = timesteps[idx].item()
                            caption = f'{video_id}_frame{frame_idx-1:02d}_step{timestep}'
                            null_flag = batch['is_null'][idx]
                            tag = '_null_prompt' if null_flag else ''
                            caption += tag
                            accelerator.log(
                                {f'train/pred_img_{idx}': wandb.Image(img_log, caption=caption)}, step=global_step
                            )
                    
                    # Validation step
                        logs = validation_loop(args, accelerator, transformer, raft, val_dataloader, weight_dtype,
                                            noise_scheduler_copy, get_sigmas, target_features, vae)
                    else:
                        logs = validation_loop(args, accelerator, transformer, raft, val_dataloader, weight_dtype,
                                            noise_scheduler_copy, get_sigmas, target_features, None)
                        
                    if accelerator.is_main_process:
                        accelerator.log(logs, step=global_step)

            # logs = {"loss": loss.detach().d(), "lr": lr_scheduler.get_last_lr()[0]}
            # progress_bar.set_postfix(**logs)
            # accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
