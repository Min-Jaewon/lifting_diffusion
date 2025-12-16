import sys
import os
import argparse
import torch
import numpy as np
import cv2
from tqdm import tqdm

from raft import RAFT, RAFT_Diff
from RAFT.core.utils.utils import InputPadder
from RAFT.core.utils import flow_viz

from PIL import Image
from RAFT.core.corr import CorrBlock

DEVICE = 'cuda'
MAX_FLOW = 400

def load_image(img):
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None]

def viz(img, flo, frame_num, save_dir):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()

    # Unnormalize and save the image
    img = (img+ 1.0) / 2
    img = (img * 255).astype(np.uint8)
    
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    os.makedirs(save_dir, exist_ok=True)


    cv2.imwrite(os.path.join(save_dir, f'frame_{frame_num:02d}.jpg'), img_flo[:, :, [2,1,0]])
    np.save(os.path.join(save_dir, f'flow_{frame_num:02d}.npy'), flo)
    
def get_raft_model(model_path,
                   # Model parameters
                   small=False,
                   mixed_precision=None, 
                   alternate_corr=False,
                   feature_upsample=None,
                   use_raft_encoder=False,
                   conv_1x1_channels=256,
                   is_flow_extractor=False,
                   use_l2_norm=False,
                   use_context_dpt=False,
                   use_custom_dpt=False,
                   evaluation=False):

    
    # Initialize the RAFT model with the specified parameters
    if mixed_precision == 'fp16':
        use_mixed_precision = True
    else:
        use_mixed_precision = False
    
    args = argparse.Namespace(
        model=model_path,
        small=small,
        mixed_precision=use_mixed_precision,
        feature_upsample=feature_upsample,
        use_raft_encoder=use_raft_encoder,
        conv_1x1_channels=conv_1x1_channels,
        use_l2_norm=use_l2_norm,
        use_context_dpt=use_context_dpt,
        use_custom_dpt=use_custom_dpt,
    )
    
    if is_flow_extractor:
        model = RAFT(args)
    else:
        model = RAFT_Diff(args)
            
    state_dict = torch.load(args.model, map_location="cpu")
    
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    
    # Remove 'module.' prefix if it exists in the state_dict keys
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    if use_context_dpt and not evaluation:
        gru_layers = [
            "update_block.gru.convz1",
            "update_block.gru.convr1",
            "update_block.gru.convq1",
            "update_block.gru.convz2",
            "update_block.gru.convr2",
            "update_block.gru.convq2",
        ]
        gru_keys = [k for k in new_state_dict.keys() if any(layer in k for layer in gru_layers)]
        for key in gru_keys:
            if 'bias' in key:
                continue
            weight = new_state_dict[key]
            zero_init = torch.zeros_like(weight)[:, :128, :, :]  # Assuming DPT context features have 128 channels
            new_state_dict[key] = torch.concat([zero_init, weight], dim=1)
            print(f'Zero initialized weights for {key} to accommodate DPT context features.')
            
    model.load_state_dict(new_state_dict, strict=is_flow_extractor) 
    
    return model

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

