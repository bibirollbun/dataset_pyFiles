%%capture
!pip install --upgrade -qqq uv
try: import numpy; get_numpy = f"numpy=={numpy.__version__}"
except: get_numpy = "numpy"
!uv pip install -qqq \
    "torch>=2.8.0" "triton>=3.4.0" {get_numpy} torchvision bitsandbytes \
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
    "unsloth[base] @ git+https://github.com/unslothai/unsloth" \
    git+https://github.com/huggingface/transformers \
    git+https://github.com/triton-lang/triton.git@05b2c186c1b6c9a08375389d5efe9cb4c401c075#subdirectory=python/triton_kernels


import torch
import os, csv, queue, threading, time
import numpy as np
from PIL import Image
from transformers import SamModel, SamProcessor, AutoModel
import cv2
import requests
import json
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Dict, Any, List
from tqdm import tqdm


data_root = "/kaggle/input/openvivqa/"
image_folder = data_root + "test-images"
data_file = data_root + "test-annotations.json"

data_root, image_folder, data_file


@dataclass
class DAMConfig:
    model_path: str = 'nvidia/DAM-3B-Self-Contained'
    conv_mode: str = 'v1'
    prompt_mode: str = 'full+focal_crop'
    dtype: str = 'float16'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_dam(cfg: DAMConfig):
    dtype = torch.float16 if cfg.dtype == 'float16' else torch.bfloat16
    model = AutoModel.from_pretrained(
        cfg.model_path,
        trust_remote_code=True,
        torch_dtype=dtype
    ).to(cfg.device)
    dam = model.init_dam(conv_mode=cfg.conv_mode, prompt_mode=cfg.prompt_mode)
    return dam

def dam_describe_full_image(dam, img: Image.Image, prompt: str, **gen_kwargs) -> str:
    # Make a full-image mask (L mode, 255 = foreground)
    mask = Image.new('L', img.size, 255)
    out = dam.get_description(
        img,
        mask,
        prompt,
        streaming=False,
        temperature=gen_kwargs.get('temperature', 0.2),
        top_p=gen_kwargs.get('top_p', 0.5),
        num_beams=gen_kwargs.get('num_beams', 1),
        max_new_tokens=gen_kwargs.get('max_new_tokens', 512),
    )
    if isinstance(out, str):
        return out
    # If generator/iterable of tokens
    return ''.join(list(out))


VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
WRITE_EVERY = 25            # ghi/flush mỗi 25 dòng để nhanh hơn (đổi tuỳ ý)
PREFETCH_WORKERS = 8        # số thread nạp ảnh (I/O)
# ============================

def _iter_image_files(image_folder):
    for n in sorted(os.listdir(image_folder)):
        ext = os.path.splitext(n)[1].lower()
        if ext in VALID_EXTS:
            yield n

def run(
    image_folder: str,
    csv_path: str = "captions.csv",
    prompt: str = (
        "<image>\nDescribe the entire image in rich, precise detail focusing on"
        " objects, text, layout, and visual attributes relevant for VQA."
    ),
    temperature: float = 0.2,
    top_p: float = 0.5,
    num_beams: int = 1,
    max_new_tokens: int = 512,
):
    dam = load_dam(DAMConfig())

    # Resume: đọc file đã có
    processed = set()
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("file_name"):
                    processed.add(row["file_name"].strip())

    # Mở CSV ở chế độ append; ghi header nếu cần
    need_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    f = open(csv_path, "a", encoding="utf-8", newline="")
    writer = csv.writer(f)
    if need_header:
        writer.writerow(["file_name", "caption"])
        f.flush()

    # Hàng đợi ảnh đã nạp sẵn (PIL Image) để mô hình xử lý
    q = queue.Queue(maxsize=64)

    all_files = [n for n in _iter_image_files(image_folder) if n not in processed]
    stop_token = object()

    def loader():
        for name in all_files:
            path = os.path.join(image_folder, name)
            try:
                img = Image.open(path)  # .convert("RGB") chuyển sau (khi cần)
                q.put((name, img), block=True)
            except Exception as e:
                print(f"Skip {name}: {e}")
        q.put((stop_token, None))

    # Khởi động thread
    threads = []
    for _ in range(PREFETCH_WORKERS):
        t = threading.Thread(target=loader, daemon=True)
        t.start()
        threads.append(t)
        break

    written_since_flush = 0
    pbar = tqdm(total=len(all_files), desc="Captioning (fast)")

    while True:
        name, img = q.get()
        if name is stop_token:
            break

        try:
            img = img.convert("RGB")
        except Exception:
            pass

        caption = dam_describe_full_image(
            dam, img, prompt,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        ).strip()

        writer.writerow([name, caption])
        written_since_flush += 1

        if written_since_flush >= WRITE_EVERY:
            f.flush()
            written_since_flush = 0

        pbar.update(1)

    # Flush nốt
    f.flush()
    f.close()
    pbar.close()


run(image_folder)


# def dam_caption_max_detail(dam, img, *, global_first=True, 
#                            temperature=0.2, top_p=0.5, num_beams=1,
#                            max_new_tokens=512):
#     from PIL import Image
#     mask_full = Image.new('L', img.size, 255)
    
    
#     global_prompt = (
#     "<image>\nDescribe the entire image exhaustively. Include objects, fine textures,"
#     " exact visible text (quoted), layout/spatial relations, and counts."
#     " Only describe what is visually present; if uncertain, say 'uncertain'."
#     )
    
    
#     # 1) Global pass
#     global_cap = dam.get_description(
#     img, mask_full, global_prompt,
#     streaming=False,
#     temperature=temperature,
#     top_p=top_p,
#     num_beams=num_beams,
#     max_new_tokens=max_new_tokens,
#     )
    
    
#     # 2) (Optional) Region sweeps (example: 2x2 grid)
#     W, H = img.size
#     grid_caps = []
#     for gy in range(2):
#     for gx in range(2):
#     x0, y0 = gx * W // 2, gy * H // 2
#     x1, y1 = (gx + 1) * W // 2, (gy + 1) * H // 2
#     mask = Image.new('L', (W, H), 0)
#     # fill the tile area as foreground
#     tile = Image.new('L', (x1 - x0, y1 - y0), 255)
#     mask.paste(tile, (x0, y0))
    
    
#     local_prompt = (
#     "<image>\nDescribe the masked region in meticulous detail."
#     " Focus only on this region, mention tiny markings and exact visible text."
#     )
#     cap = dam.get_description(
#     img, mask, local_prompt,
#     streaming=False,
#     temperature=temperature,
#     top_p=top_p,
#     num_beams=num_beams,
#     max_new_tokens=max_new_tokens // 2,
#     )
#     grid_caps.append(cap)
    
    
#     # 3) Merge (simple concat; replace with dedup logic if needed)
#     merged = str(global_cap).strip()
#     if grid_caps:
#     merged += "\n\nLocal details:\n- " + "\n- ".join([str(c).strip() for c in grid_caps])
#     return merged

# -----------------------------
# Main runner
# -----------------------------


