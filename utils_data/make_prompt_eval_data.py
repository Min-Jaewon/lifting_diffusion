import torch
import cv2
from PIL import Image
import os
from tqdm import tqdm
import re
import sys
sys.path.append(os.getcwd())
from llava.llm_agent import LLavaAgent
from CKPT_PTH import LLAVA_MODEL_PATH
from typing import Iterable

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--video_dir", type=str, default='/mnt/dataset2/jaewon/eval/', help='the dataset you want to tag.') # 
parser.add_argument('--name', type=str, default=None, help='the dataset you want to tag.') #
parser.add_argument("--save_dir", type=str, default='/mnt/dataset2/jaewon/eval/', help='the dataset you want to tag.') # 
parser.add_argument("--stop_num", type=int, default=-1)
parser.add_argument("--start_num", type=int, default=0)
args = parser.parse_args()

def read_first_frame(video_path, to_rgb=True, as_pil=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    ok, frame = cap.read()  # 첫 프레임
    cap.release()
    if not ok:
        raise RuntimeError("Failed to read the first frame")

    # OpenCV는 BGR → 보통 RGB로 변환해서 사용
    if to_rgb:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if as_pil:
        return Image.fromarray(frame)
    return frame  # numpy array (H, W, 3)

def iter_frames_dirs(root: str) -> Iterable[str]:
    """
    the expected structure: root/video_id.mp4
    """
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise NotADirectoryError(f"Root not found: {root}")

    for video_id in sorted(os.listdir(root)):
        if video_id.endswith(".mp4"):
            video_path = os.path.join(root, video_id)
            yield video_path
                    
                        
def remove_focus_sentences(text):
    # 使用正则表达式按照 . ? ! 分割，并且保留分隔符本身
    # re.split(pattern, string) 如果 pattern 中带有捕获组()，分隔符也会保留在结果列表中
    prohibited_words = ['focus', 'focal', 'prominent', 'close-up', 'black and white', 'blur', 'depth', 'dense', 'locate', 'position']
    parts = re.split(r'([.?!])', text)
    
    filtered_sentences = []
    i = 0
    while i < len(parts):
        # sentence 可能是句子主体，punctuation 是该句子结尾的标点
        sentence = parts[i]
        punctuation = parts[i+1] if (i+1 < len(parts)) else ''

        # 组合为完整句子，避免漏掉结尾标点
        full_sentence = sentence + punctuation
        
        full_sentence_lower = full_sentence.lower()
        skip = False
        for word in prohibited_words:
            if word.lower() in full_sentence_lower:
                skip = True
                break
        
        # 如果该句子不包含任何禁用词，则保留
        if not skip:
            filtered_sentences.append(full_sentence)
        
        # 跳过已经处理的句子和标点
        i += 2
    
    # 根据需要选择如何重新拼接；这里去掉多余空格并直接拼接
    return "".join(filtered_sentences).strip()

@torch.no_grad()
def process_llava(
    input_image):
    llama_prompt = llava_agent.gen_image_caption([input_image])[0]
    llama_prompt = remove_focus_sentences(llama_prompt)
    return llama_prompt

def PrintInfo(x):
    if not isinstance(x,list):
        x=[x]
    for i in x:
        print('shape : {} ; dtype : {} ; max : {} ; min : {}'.format(i.shape,i.dtype,i.max(),i.min())  )

video_folder = os.path.join(args.video_dir, args.name, 'gt_videos/_mp4')
prompt_save_folder = os.path.join(args.save_dir, args.name, 'prompts')
video_name_list = list(iter_frames_dirs(video_folder))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
llava_agent = LLavaAgent(LLAVA_MODEL_PATH, device, load_8bit=True, load_4bit=False)

if args.stop_num == -1:
    video_name_list = video_name_list[args.start_num:]
else:
    video_name_list = video_name_list[args.start_num:args.stop_num]

for video_name in tqdm(video_name_list):
    # Open mp4 and get first frame with PIL
    video = read_first_frame(video_name, to_rgb=True, as_pil=True)
    video_name = video_name.split('/')[-1].split('.')[0] + '.txt'
    video_save_path = os.path.join(prompt_save_folder, video_name)
    os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
    if os.path.exists(video_save_path):
        continue
    prompt = process_llava(video)
    with open(video_save_path, 'w', encoding="utf-8") as f:
        f.write(prompt)