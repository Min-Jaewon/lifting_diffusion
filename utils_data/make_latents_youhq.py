import torch
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from typing import Iterable
from diffusers import AutoencoderKL

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default='preset/datasets/train_datasets/training', help='the dataset you want to tag.') # 
parser.add_argument("--save_path", type=str, default='preset/datasets/train_datasets/training', help='the dataset you want to tag.') # 
parser.add_argument("--end_num", type=int, default=-1)
parser.add_argument("--start_num", type=int, default=0)
args = parser.parse_args()

def iter_frames_dirs(root: str) -> Iterable[str]:
    """
    the expected structure: root/category/unique_id/video_id_frames/frame_**.jpg
    """
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise NotADirectoryError(f"Root not found: {root}")
    counter = 0
    for category in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        category_path = os.path.join(root, category)
        for unique_id in sorted(d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))):
            uid_path = os.path.join(category_path, unique_id)
            for video_id in sorted(os.listdir(uid_path)):
                counter +=1
                if counter <10010:
                    continue
                video_path = os.path.join(uid_path, video_id)
                if os.path.isdir(video_path):
                    for frame_name in sorted(os.listdir(video_path)):
                        if frame_name.endswith(".jpg"):
                            frame_path = os.path.join(video_path, frame_name)
                            yield frame_path
                    
def PrintInfo(x):
    if not isinstance(x,list):
        x=[x]
    for i in x:
        print('shape : {} ; dtype : {} ; max : {} ; min : {}'.format(i.shape,i.dtype,i.max(),i.min())  )

img_preproc = transforms.Compose([       
            transforms.ToTensor(),
        ])
img_afterproc = transforms.Compose([ 
            transforms.ToPILImage(),
        ])

video_folder = args.root_path
save_folder = args.save_path
os.makedirs(save_folder, exist_ok=True)
video_name_list = list(iter_frames_dirs(video_folder))

video_name_list = [
    '/mnt/dataset2/jaewon/YouHQ/YouHQ-Train-frames/food/0Fs-4GiNxQ8/106560_106619_01_frames/frame_23.jpg',
    '/mnt/dataset2/jaewon/YouHQ/YouHQ-Train-frames/distant/pGQ7Km9gMpg/026400_026459_01_frames/frame_01.jpg',
    '/mnt/dataset2/jaewon/YouHQ/YouHQ-Train-frames/nature/LK0p8CLZnMA/091200_091319_00_frames/frame_11.jpg',
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vae = AutoencoderKL.from_pretrained('preset/models/stable-diffusion-3.5-medium', subfolder="vae", revision=None)
vae.requires_grad_(False)
vae.to(device)

if args.end_num == -1:
    img_name_list = video_name_list[args.start_num:]
else:
    img_name_list = video_name_list[args.start_num:args.end_num]
    
for img_name in tqdm(img_name_list):
    img = Image.open(img_name).convert('RGB')
    save_path = os.path.join(save_folder, os.path.relpath(img_name, video_folder))
    save_path = save_path.replace('.jpg', '.pt')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if os.path.exists(save_path):
        continue
    
    img = img_preproc(img)
    img = img * 2.0 - 1.0
    img = img.unsqueeze(0)
    img = img.to(device)
    with torch.no_grad():
        latents = vae.encode(img).latent_dist.sample()
        latents = (latents - vae.config.shift_factor)  * vae.config.scaling_factor
    torch.save(latents.clone().cpu(), save_path)