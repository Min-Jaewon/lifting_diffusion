import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock, CorrBlock_Diffusion
from RAFT.core.utils.utils import bilinear_sampler, coords_grid, upflow8
from einops import rearrange
from dpt import DPTHead
from custom_dpt import CustomDPTHead

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        # Already normalized images
        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim
        
        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])        
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)


        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions

class RAFT_Diff(nn.Module):
    def __init__(self, args):
        super(RAFT_Diff, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim, use_context_dpt=args.use_context_dpt,)

        
        if args.feature_upsample == 'dpt':
            # Upsample using DPTHead
            if args.use_custom_dpt:
                self.upsampler = CustomDPTHead(dim_in=1536, features=args.conv_1x1_channels, feature_only=True)
            else:
                self.upsampler = DPTHead(dim_in=1536, features=args.conv_1x1_channels, feature_only=True)
            
            
        elif args.feature_upsample == 'bilinear':
            # Upsample and use 1x1 conv to reduce channel dim
            self.upsampler = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(1536, args.conv_1x1_channels, kernel_size=1, stride=1, padding=0),
            )
        else:
            raise ValueError(f"Unknown feature_upsample type: {args.feature_upsample}")
        
        self.use_raft_encoder = args.use_raft_encoder
        self.use_l2_norm = args.use_l2_norm
        if args.use_context_dpt:
            if args.use_custom_dpt:
                self.dit_context_dpt = CustomDPTHead(dim_in=1536, features=128, feature_only=True)
            else:
                self.dit_context_dpt = DPTHead(dim_in=1536, features=128, feature_only=True)
            
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    
    def forward(self, image1, image2, features, iters=12, flow_init=None, upsample=True, test_mode=False,):
        """ Estimate optical flow between pair of frames """
        H, W = image1.shape[-2:]
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        
        hdim = self.hidden_dim
        cdim = self.context_dim
        
        # Reshape features: (2B, 1024, 1536) -> (B, 1024 ,1536) + (B, 1024, 1536)
        feat1 =[]
        feat2 =[]
        
        if not test_mode:
            for layer_feature in features:
                reshaped_features = rearrange(layer_feature, '(b f) c d -> b f c d', f=2)
                feat1.append(reshaped_features[:,0,:,:].permute(0,2,1).reshape(-1, 1536, 32,32)) # (B, 1024, 1536) -> (B, 1536, 32, 32)
                feat2.append(reshaped_features[:,1,:,:].permute(0,2,1).reshape(-1, 1536, 32,32)) # (B, 1024, 1536) -> (B, 1536, 32, 32)
        else:
            feat1 = features[0]
            feat2 = features[1]
            feat1 = [feat.permute(0,3,1,2) for feat in feat1]  # (B, 1024, 1536) -> (B, 1536, 32, 32)
            feat2 = [feat.permute(0,3,1,2) for feat in feat2]  # (B, 1024, 1536) -> (B, 1536, 32, 32)
        
        if self.args.use_context_dpt:
            # Also extract context features using DPTHead
            dit_context_feat1 = self.dit_context_dpt(feat1, image1)  # (B, 1536, 32, 32) -> (B, conv_1x1_channels, H/8, W/8)
        
        
        if self.args.feature_upsample == 'bilinear':
            feat1 = torch.cat(feat1, dim=1)  # (B, 1536, 32, 32)
            feat2 = torch.cat(feat2, dim=1)  # (B, 1536, 32, 32)
            
            # Upsample features to match 1/8 resolution
            fmap1 = self.upsampler(feat1)  # (B, 1536, 32, 32) -> (B, 128, 64, 64)
            fmap2 = self.upsampler(feat2)  # (B, 1536, 32, 32) -> (B, 128, 64, 64)
            
        elif self.args.feature_upsample == 'dpt':
            fmap1 = self.upsampler(feat1, image1)
            fmap2 = self.upsampler(feat2, image2)
        
        if self.use_l2_norm:
            fmap1 = F.normalize(fmap1, p=2, dim=1)
            fmap2 = F.normalize(fmap2, p=2, dim=1)
        
        if self.use_raft_encoder:
            # Also extract features using RAFT encoder
            with autocast(enabled=self.args.mixed_precision):
                raft_fmap1, raft_fmap2 = self.fnet([image1, image2])        
                
            raft_fmap1 = raft_fmap1.float()
            raft_fmap2 = raft_fmap2.float()
            if self.use_l2_norm:
                raft_fmap1 = F.normalize(raft_fmap1, p=2, dim=1)
                raft_fmap2 = F.normalize(raft_fmap2, p=2, dim=1)
            
            # Concatenate DiT features and RAFT encoder features
            fmap1 = torch.cat([fmap1, raft_fmap1], dim=1)  # (B, 512, H/8, W/8)
            fmap2 = torch.cat([fmap2, raft_fmap2], dim=1)  # (B, 512, H/8, W/8)
        
        
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        
        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        if self.args.use_context_dpt:
            # Concatenate DPT context features
            inp = torch.cat([dit_context_feat1, inp], dim=1)  # (B, cdim + conv_1x1_channels, H/8, W/8)
        
        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions



class BilinearConvTranspose2d(nn.ConvTranspose2d):
    """A conv transpose initialized to bilinear interpolation."""

    def __init__(self, in_planes, out_planes, kernel_size=4, stride=2, padding=1, groups=1):
        """Set up the layer.
        Parameters
        ----------
        channels: int
            The number of input and output channels
        stride: int or tuple
            The amount of upsampling to do
        groups: int
            Set to 1 for a standard convolution. Set equal to channels to
            make sure there is no cross-talk between channels.
        """
        if isinstance(stride, int):
            stride = (stride, stride)
        super().__init__(
            in_planes, out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the weight and bias."""
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.weight, 0)
        bilinear_kernel = self.bilinear_kernel(self.kernel_size)

        # for each output channel, applied bilinear_kernel on the same input channel, and 0 on the other input channels.
        for i in range(self.in_channels):
            if self.groups == 1:
                j = i
            else:
                j = 0
            self.weight.data[i, j] = bilinear_kernel

    @staticmethod
    def bilinear_kernel(kernel_size):
        """Generate a bilinear upsampling kernel."""
        num_dims = len(kernel_size)

        shape = (1,) * num_dims
        bilinear_kernel = torch.ones(*shape)

        # The bilinear kernel is separable in its spatial dimensions
        # Build up the kernel dim by dim
        for dim in range(num_dims):
            kernel = kernel_size[dim]
            factor = (kernel + 1) // 2
            if kernel % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            delta = torch.arange(0, kernel) - center
            channel_filter = (1 - torch.abs(delta / factor))

            # Apply the dim filter to the current dim
            shape = [1] * num_dims
            shape[dim] = kernel
            bilinear_kernel = bilinear_kernel * channel_filter.view(shape)
        # if kenel_size is (4,4), bilinear kernel is (4,4)
        # channel_filter is [0.25, 0.75, 0.75, 0.25]
        return bilinear_kernel
