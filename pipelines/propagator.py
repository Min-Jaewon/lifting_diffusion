import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair, _single
import math

import torchvision
import warnings
from einops import rearrange

def get_sigma_fade_scale(sigma_t, sigma_fade_start, sigma_fade_end):
    """
    sigma_t 값에 따라 1.0에서 0.0으로 감소하는 fade_scale을 계산합니다.
    
    Args:
        sigma_t (float): 현재 타임스텝의 노이즈 레벨 (sigma).
        sigma_fade_start (float): Fading이 시작되는 높은 sigma 값 (e.g., 0.8).
        sigma_fade_end (float): Fading이 완료되는(scale=0.0) 낮은 sigma 값 (e.g., 0.1).
    """
    if sigma_fade_start <= sigma_fade_end:
        # 오류 방지
        return 0.0 if sigma_t < sigma_fade_end else 1.0

    # (sigma_t - sigma_fade_end) / (sigma_fade_start - sigma_fade_end)
    # sigma_t == sigma_fade_start -> 1.0
    # sigma_t == sigma_fade_end -> 0.0
    progress = (sigma_t - sigma_fade_end) / (sigma_fade_start - sigma_fade_end)
    
    # torch.clamp를 사용하여 [0.0, 1.0] 범위로 제한
    fade_scale = torch.clamp(progress, 0.0, 1.0)
    
    return fade_scale.item() if isinstance(progress, torch.Tensor) else float(fade_scale)

@torch.no_grad()
def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, 
                        grid_y), 2)  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    vgrid_scaled = vgrid_scaled.to(x)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    return output

def length_sq(x):
    return torch.sum(torch.square(x), dim=1, keepdim=True)

def fbConsistencyCheck(flow_fw, flow_bw, alpha1=0.01, alpha2=0.5): 
    flow_bw_warped = flow_warp(flow_bw, flow_fw.permute(0, 2, 3, 1))  # wb(wf(x))
    flow_diff_fw = flow_fw + flow_bw_warped  # wf + wb(wf(x))

    mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)  # |wf| + |wb(wf(x))|
    occ_thresh_fw = alpha1 * mag_sq_fw + alpha2

    # fb_valid_fw = (length_sq(flow_diff_fw) < occ_thresh_fw).float()
    fb_valid_fw = (length_sq(flow_diff_fw) < occ_thresh_fw).to(flow_fw)
    return fb_valid_fw


def apply_flow_warping(x, flows_forward, flows_backward, interpolation='bilinear', mode='fuse', fuse_scale=0.5,
            alpha1=0.01, alpha2=0.5):
    """
    x shape : [b, c, t, h, w]
    return [b, c, t, h, w]
    """
    base_module = ['backward_prop', 'forward_prop']
    # For backward warping
    # pred_flows_forward for backward feature propagation
    # pred_flows_backward for forward feature propagation

    b, c, t, h, w = x.shape
    _, _, t_f, h_f, w_f = flows_forward.shape
    
    if (t_f == t - 1) and (h_f == h) and (w_f == w):
        pass
    else:
        w_f = flows_forward.shape[-1]
        s = 1.0*w/w_f
        flows_forward = F.interpolate(flows_forward, (t-1, h, w), mode='area') * s
        flows_backward = F.interpolate(flows_backward, (t-1, h, w), mode='area') * s

    feats = {}
    feats['input'] = [x[:, :, i, :, :] for i in range(0, t)]

    cache_list = ['input'] +  base_module
    for p_i, module_name in enumerate(base_module):
        feats[module_name] = []

        if 'backward' in module_name:
            frame_idx = range(0, t)
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx
            flows_for_prop = flows_forward
            flows_for_check = flows_backward
        else:
            frame_idx = range(0, t)
            flow_idx = range(-1, t - 1)
            flows_for_prop = flows_backward
            flows_for_check = flows_forward

        for i, idx in enumerate(frame_idx):
            feat_current = feats[cache_list[p_i]][idx]

            if i == 0:
                feat_prop = feat_current
            else:
                flow_prop = flows_for_prop[:, :, flow_idx[i], :, :]
                flow_check = flows_for_check[:, :, flow_idx[i], :, :]
                flow_vaild_mask = fbConsistencyCheck(flow_prop, flow_check, alpha1, alpha2)

                feat_warped = flow_warp(feat_prop, flow_prop.permute(0, 2, 3, 1), interpolation)
                if mode == 'fuse': # choice 1: blur
                    feat_warped = feat_warped * fuse_scale + feat_current * (1-fuse_scale)
                elif mode == 'copy': # choice 2: alignment
                    feat_warped = feat_warped 
                feat_prop = flow_vaild_mask * feat_warped + (1-flow_vaild_mask) * feat_current
                    
            feats[module_name].append(feat_prop)

        # end for
        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

    outputs_b = torch.stack(feats['backward_prop'], dim=2) # bcthw
    outputs_f = torch.stack(feats['forward_prop'], dim=2)  # bcthw
    
    outputs = outputs_f
    
    return outputs



