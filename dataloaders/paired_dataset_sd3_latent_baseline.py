import glob
import os
import random

import torch
from torchvision import transforms
from torch.utils import data as data
from einops import rearrange, repeat
from PIL import Image
import numpy as np
from torch.utils.data import Subset
from tqdm import tqdm

def collate_fn(batch):
    gt_latent = torch.stack([item['pixel_values'] for item in batch])
    gt_latent = rearrange(gt_latent, 'b f c h w -> (b f) c h w')        
    lq_latent = torch.stack([item['conditioning_pixel_values'] for item in batch])
    lq_latent = rearrange(lq_latent, 'b f c h w -> (b f) c h w')
    prompt_embeds = torch.stack([item['prompt_embeds'] for item in batch])
    prompt_embeds = repeat(prompt_embeds, 'b ... -> (b 3) ...')  # Assuming 2 frames per video
    pooled_prompt_embeds = torch.stack([item['pooled_prompt_embeds'] for item in batch]) 
    pooled_prompt_embeds = repeat(pooled_prompt_embeds, 'b ... -> (b 3) ...')  # Assuming 2 frames per video 
    video_ids = [item['video_id'] for item in batch]  
    
    lq_image = torch.stack([item['lq_image'] for item in batch])  # (b, 2, C, H, W)
    lq_image = rearrange(lq_image, 'b f c h w -> (b f) c h w')
    gt_image = torch.stack([item['gt_image'] for item in batch])  # (b, 2, C, H, W)
    gt_image = rearrange(gt_image, 'b f c h w -> (b f) c h w')
    
    # gt_flow = torch.stack([item['gt_flow'] for item in batch])  # (b, 2, H, W)
    # lq_flow = torch.stack([item['lq_flow'] for item in batch])  # (b, 2, H, W)
    frame_idxs = [item['frame_idx'] for item in batch]  # List of frame indices
    
    is_null_flags = [item['is_null'] for item in batch]
    
    return {
        'pixel_values': gt_latent,
        'conditioning_pixel_values': lq_latent,
        'prompt_embeds': prompt_embeds,
        'pooled_prompt_embeds': pooled_prompt_embeds,
        'video_id': video_ids,
        'lq_image': lq_image,
        'gt_image': gt_image,
        # 'gt_flow': gt_flow,
        # 'lq_flow': lq_flow,
        'frame_idx': frame_idxs,
        'is_null': is_null_flags,
    }
    
class VideoPairedCaptionDataset(data.Dataset):
    def __init__(
            self,
            root_folder=None,
            null_text_ratio=0.2,
            use_sea_raft = False,
            val=False,
            # use_ram_encoder=False,
            # use_gt_caption=False,
            # caption_type = 'gt_caption',
    ):
        super(VideoPairedCaptionDataset, self).__init__()
        if val:
            root_folder = root_folder.replace('YouHQ-Train', 'YouHQ-Val')
        self.null_text_ratio = null_text_ratio 
        self.use_sea_raft = use_sea_raft
        self.rel_path_list = self.iter_frames_dirs(root_folder) if not val else self.iter_frames_dirs_val(root_folder)
        self.rel_path_list = self.rel_path_list
        self.root_folder = root_folder
        self.ref_image_idxs = self.get_frame_indices()
        
        self.img_preproc = transforms.Compose([       
            transforms.ToTensor(),
        ])

    def iter_frames_dirs(self, root):
        video_path_list = []
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            raise NotADirectoryError(f"Root not found: {root}")

        for category in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
            category_path = os.path.join(root, category)
            for unique_id in sorted(d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))):
                uid_path = os.path.join(category_path, unique_id)
                for video_id in sorted(os.listdir(uid_path)):
                    if video_id.endswith(".mp4"):
                        video_path = os.path.join(uid_path, video_id.replace('.mp4',''))
                        video_path_list.append(os.path.relpath(video_path, root))

        return video_path_list
    
    def iter_frames_dirs_val(self, root):
        """
        the expected structure: root/*.mp4
        """
        video_path_list = []
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            raise NotADirectoryError(f"Root not found: {root}")

        for video_id in sorted(os.listdir(root)):
            if video_id.endswith(".mp4"):
                video_path = os.path.join(root, video_id.replace('.mp4',''))
                video_path_list.append(os.path.relpath(video_path, root))

        return video_path_list
            
    def get_frame_indices(self):
        ref_image_idxs = []
        for video_path in self.rel_path_list:
            video_length = 30
            ref_image_idxs.append(random.randint(1, video_length - 2)) # Start from 1 to avoid the first frame
        
        return ref_image_idxs
    
    def __getitem__(self, index):

        rel_path = self.rel_path_list[index]
        # # Debugging
        # frame_idx = self.ref_image_idxs[index]
        # return {'video_id': rel_path, 'frame_idx': frame_idx}
        
        # gt_flow_path = self.root_folder +'-flow'
        # lq_flow_path = self.root_folder +'-LQ-Ours-flow'
        
        # if self.use_sea_raft:
        #     gt_flow_path += '-sea-full'
        #     lq_flow_path += '-sea-full'
        
        lr_image_path = self.root_folder +'-LQ-Ours-frames'
        gt_image_path = self.root_folder +'-frames'
        
        gt_path = self.root_folder +'-latents'
        lr_path = self.root_folder +'-LQ-Ours-latents'
        prompt_embeds_path = self.root_folder + '-prompt_embeds'
        pooled_prompt_embeds_path = self.root_folder + '-pooled_prompt_embeds'
        
        ref_img_idx = self.ref_image_idxs[index]
        
        # Load GT Flow
        
        # gt_flow_file_path = os.path.join(gt_flow_path, rel_path, f'flow_{ref_img_idx-1:02d}.npy')
        # gt_flow = torch.from_numpy(np.load(gt_flow_file_path))  # (H, W, 2)
        # gt_flow = gt_flow.permute(2, 0, 1)  # Convert to (H, W, 2) -> (2, H, W)
        
        # lq_flow_file_path = os.path.join(lq_flow_path, rel_path, f'flow_{ref_img_idx-1:02d}.npy')
        # lq_flow = torch.from_numpy(np.load(lq_flow_file_path))  # (H, W, 2)
        # lq_flow = lq_flow.permute(2, 0, 1)  # Convert to (H, W, 2) -> (2, H, W)
        
        # Load LR and GT images
        lq_image_prev_path = os.path.join(lr_image_path, rel_path+'_frames', f'frame_{ref_img_idx-1:02d}.jpg')
        lq_image_curr_path =  os.path.join(lr_image_path, rel_path+'_frames', f'frame_{ref_img_idx:02d}.jpg')
        lq_image_next_path =  os.path.join(lr_image_path, rel_path+'_frames', f'frame_{ref_img_idx+1:02d}.jpg')
        
        gt_image_prev_path = os.path.join(gt_image_path, rel_path+'_frames', f'frame_{ref_img_idx-1:02d}.jpg')
        gt_image_curr_path =  os.path.join(gt_image_path, rel_path+'_frames', f'frame_{ref_img_idx:02d}.jpg')
        gt_image_next_path =  os.path.join(gt_image_path, rel_path+'_frames', f'frame_{ref_img_idx+1:02d}.jpg')
        
        lq_image_prev_pil = Image.open(lq_image_prev_path).convert('RGB')
        lq_image_curr_pil = Image.open(lq_image_curr_path).convert('RGB')
        lq_image_next_pil = Image.open(lq_image_next_path).convert('RGB')
        
        gt_image_prev_pil = Image.open(gt_image_prev_path).convert('RGB')
        gt_image_curr_pil = Image.open(gt_image_curr_path).convert('RGB')
        gt_image_next_pil = Image.open(gt_image_next_path).convert('RGB')
        
        lq_image = torch.stack([
            self.img_preproc(lq_image_prev_pil),
            self.img_preproc(lq_image_curr_pil),
            self.img_preproc(lq_image_next_pil),
        ], dim=0)  # (3, C, H, W)
        
        gt_image = torch.stack([
            self.img_preproc(gt_image_prev_pil),
            self.img_preproc(gt_image_curr_pil),
            self.img_preproc(gt_image_next_pil),
        ], dim=0)  # (3, C, H, W)
        
            
        lq_prev_latent_path = os.path.join(lr_path, rel_path+'_frames', f'frame_{ref_img_idx-1:02d}.pt')
        lq_curr_latent_path =  os.path.join(lr_path, rel_path+'_frames', f'frame_{ref_img_idx:02d}.pt')
        lq_next_latent_path =  os.path.join(lr_path, rel_path+'_frames', f'frame_{ref_img_idx+1:02d}.pt')
        
        gt_prev_latent_path = os.path.join(gt_path, rel_path+'_frames', f'frame_{ref_img_idx-1:02d}.pt')
        gt_curr_latent_path =  os.path.join(gt_path, rel_path+'_frames', f'frame_{ref_img_idx:02d}.pt') 
        gt_next_latent_path =  os.path.join(gt_path, rel_path+'_frames', f'frame_{ref_img_idx+1:02d}.pt')
        
        lq_prev_latent = torch.load(lq_prev_latent_path)
        lq_curr_latent = torch.load(lq_curr_latent_path)
        lq_next_latent = torch.load(lq_next_latent_path)
        
        gt_prev_latent = torch.load(gt_prev_latent_path)
        gt_curr_latent = torch.load(gt_curr_latent_path)
        gt_next_latent = torch.load(gt_next_latent_path)
        
        lq_latent = torch.cat([lq_prev_latent, lq_curr_latent, lq_next_latent], dim=0)  # (3, C, H, W)
        gt_latent = torch.cat([gt_prev_latent, gt_curr_latent, gt_next_latent], dim=0)  # (3, C, H, W)
        
        # Load prompt embeds
        prompt_embeds_path = os.path.join(prompt_embeds_path, rel_path+'.pt')
        pooled_prompt_embeds_path = os.path.join(pooled_prompt_embeds_path, rel_path+'.pt')

        if random.random() < self.null_text_ratio:
            # print(f'Using NULL prompt embeds for {rel_path}')
            prompt_embeds = torch.load(os.path.join(prompt_embeds_path.replace(rel_path+'.pt', 'NULL_prompt_embeds.pt')))
            pooled_prompt_embeds = torch.load(os.path.join(pooled_prompt_embeds_path.replace(rel_path+'.pt', 'NULL_pooled_prompt_embeds.pt')))
            is_null = True
        else:        
            prompt_embeds = torch.load(prompt_embeds_path)
            pooled_prompt_embeds = torch.load(pooled_prompt_embeds_path)
            is_null = False

        example = dict()
        example["conditioning_pixel_values"] = lq_latent.squeeze(0)
        example["pixel_values"] = gt_latent.squeeze(0)
        example['prompt_embeds'] = prompt_embeds.squeeze(0)
        example['pooled_prompt_embeds'] = pooled_prompt_embeds.squeeze(0)
        example['video_id'] = rel_path
        example['lq_image'] = lq_image
        example['gt_image'] = gt_image
        # example['gt_flow'] = gt_flow
        # example['lq_flow'] = lq_flow
        example['frame_idx'] = ref_img_idx
        example['is_null'] = is_null
        
        return example

    def __len__(self):
        return len(self.rel_path_list)
    
if __name__ == "__main__":
    seed= 42
    torch.manual_seed(seed)
    random.seed(seed)   
    
    dataset = VideoPairedCaptionDataset(
        root_folder='/mnt/dataset3/jaewon/YouHQ/YouHQ-Train',
        null_text_ratio=0.0,
        use_sea_raft=True,
    )
    # val_ds = VideoPairedCaptionDataset(
    #     root_folder='/mnt/dataset2/jaewon/YouHQ/YouHQ-Val',
    #     null_text_ratio=0.0,
    #     use_sea_raft=True,
    #     val=True,
    # )
    # Check Loading
    dataset= Subset(dataset, list(range(9990, 10000)))  # Use a subset for quick testing
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"LR latent shape: {sample['conditioning_pixel_values'].shape}")
    print(f"GT latent shape: {sample['pixel_values'].shape}")
    print(f"Prompt embeds shape: {sample['prompt_embeds'].shape}")
    print(f"Pooled prompt embeds shape: {sample['pooled_prompt_embeds'].shape}")
    print(f"Video ID: {sample['video_id']}")
    print(f"LR image shape: {sample['lq_image'].shape}")
    print(f"GT image shape: {sample['gt_image'].shape}")
    # print(f'GT flow shape: {sample["gt_flow"].shape}')
    # print(f'LQ flow shape: {sample["lq_flow"].shape}')
    print(f'Frame index: {sample["frame_idx"]:2d}')

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
        
    # val_loader = torch.utils.data.DataLoader(
    #     val_ds,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=0,
    #     collate_fn=collate_fn,
    # )
    
    for batch in tqdm(train_loader, desc="Training Loader"):
        try:
            pass
        except Exception as e:
            print(f"Error in batch: {e}")
        continue
        # # Just print video id and frame idx
        # print(f"Video IDs in batch: {batch['video_id']}")
        # print(f'Video frame indices in batch: {batch["frame_idx"]}')
        
        print(f"Batch LR latent shape: {batch['conditioning_pixel_values'].shape}")
        print(f"Batch GT latent shape: {batch['pixel_values'].shape}")
        print(f"Batch prompt embeds shape: {batch['prompt_embeds'].shape}")
        print(f"Batch pooled prompt embeds shape: {batch['pooled_prompt_embeds'].shape}")
        print(f"Batch video IDs: {batch['video_id']}")
        print(f"Batch LR image shape: {batch['lq_image'].shape}")
        print(f"Batch GT image shape: {batch['gt_image'].shape}")
        # print(f'Batch GT flow shape: {batch["gt_flow"].shape}')
        # print(f'Batch LQ flow shape: {batch["lq_flow"].shape}')
        print(f'Batch frame indices: {batch["frame_idx"]}')
        print(f'Arrange LR latent shape: {rearrange(batch["conditioning_pixel_values"], "(b f) c h w -> b f c h w", f=3)[:,0,:,:,:].shape}')
        print(f'Null flags: {batch["is_null"]}')
        
        
        # Check same embeds in frames
        for i in range(len(batch['video_id'])):
            vid = batch['video_id'][i]
            pe1 = batch['prompt_embeds'][2*i]
            pe2 = batch['prompt_embeds'][2*i+1]
            ppe1 = batch['pooled_prompt_embeds'][2*i]
            ppe2 = batch['pooled_prompt_embeds'][2*i+1]
            assert torch.allclose(pe1, pe2), f"Prompt embeds do not match for video {vid}"
            assert torch.allclose(ppe1, ppe2), f"Pooled prompt embeds do not match for video {vid}"
        
        break