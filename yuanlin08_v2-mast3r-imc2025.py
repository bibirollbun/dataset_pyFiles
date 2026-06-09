import sys


class CONFIG:
    # DEBUG Settings
    DRY_RUN = False
    DRY_RUN_MAX_IMAGES = 10

    # Pipeline settings
    NUM_CORES = 2
    MAST3R_MIN_PAIR = 15
    MATCH_CONF_TH = 1.001


!pip install torch torchvision torchaudio --no-index --find-links=/kaggle/input/mast3r-fix/mast3r-wheels


!pip install faiss-gpu-cu12 --no-index --find-links=/kaggle/input/mast3r-fix/mast3r-wheels


# 离线安装所有依赖（不联网）
!pip install --no-index --find-links=/kaggle/input/mast3r-fix/mast3r-wheels \
    -r /kaggle/input/mast3r-fix/mast3r/requirements.txt \
    -r /kaggle/input/mast3r-fix/mast3r/dust3r/requirements.txt \
    -r /kaggle/input/mast3r-fix/mast3r/dust3r/requirements_optional.txt


!pip install --no-index /kaggle/input/imc2024-packages-lightglue-rerun-kornia/* --no-deps


!pip install --no-index /kaggle/input/pycolmap3-11/pycolmap-3.11.1-cp311-cp311-manylinux_2_28_x86_64.whl --no-deps


# !pip install kornia_rs
# !pip install pycolmap


# 加入源码主目录（包含 mast3r, dust3r 等子目录）
sys.path.insert(0, "/kaggle/input/mast3r-fix/mast3r")
sys.path.insert(0, '/kaggle/input/mast3r-fix/mast3r/asmk')
sys.path.insert(0, '/kaggle/input/mast3r-fix/mast3r/dust3r/croco/models/curope')


# # %cd /kaggle/working/mast3r
# !PYTHONPATH="/kaggle/input/mast3r-fix/mast3r/asmk:$PYTHONPATH" python3 /kaggle/input/mast3r-fix/mast3r/demo.py --weights /kaggle/input/mast3r-fix/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth --share


!rm -rf /kaggle/working/visualization_output
!rm -rf /kaggle/working/temp
!rm -rf /kaggle/working/result


import random
import os
import numpy as np
import torch
import dataclasses

def seed_everything(seed: int = 42):
    """Set seed for reproducibility across random, numpy, torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()


# !pip uninstall -y pycolmap
# !pip install --no-cache-dir pycolmap==3.11.1 --index-url https://pypi.org/simple



import pycolmap
!pip show pycolmap






import pycolmap
import os
print(os.listdir(os.path.dirname(pycolmap.__file__)))



import sys
import os
from tqdm import tqdm
from time import time, sleep
import gc
import numpy as np
import h5py
import dataclasses
import pandas as pd
from IPython.display import clear_output
from collections import defaultdict
from copy import deepcopy
from PIL import Image

import cv2
import torch
import torch.nn.functional as F

import torch
from transformers import AutoImageProcessor, AutoModel


# IMPORTANT Utilities: importing data into colmap and competition metric
import pycolmap


sys.path.append('/kaggle/input/pycolmap3-11-imc-utils')
# from database import *
from h5_to_db import *
import metric
from pycolmap import verify_matches, TwoViewGeometryOptions

from fastprogress import progress_bar


# 主线程先 preload，确保子线程里不会挂
_ = verify_matches
_ = TwoViewGeometryOptions()


from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs, extract_correspondences_nonsym

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images


!rm -rf /kaggle/working/result



# Configuration
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Automatically use GPU if available
print(f"Using device: {device}")

schedule = 'cosine' # These seem to be unused in the provided snippet, but keep for context
lr = 0.01
niter = 300
local_model_path = "/kaggle/input/mast3r-fix/mast3r/checkpoints/"
local_model_directory = "/kaggle/input/mast3r-fix/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
retrival_model_dir = '/kaggle/input/mast3r-fix/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth'

# Now, we manually call `load_model` as suggested by `mast3r/model.py`'s `from_pretrained` logic
from mast3r.model import load_model # Assuming load_model is defined in mast3r/model.py or accessible

print(f"Loading model from local path: {local_model_directory}")
mast3r_model = load_model(local_model_directory, device=device) # Pass device to load_model
print("Model loaded successfully.")


def transform_keypoints_to_original(
    kpts_crop: np.ndarray,
    original_size: tuple[int, int],#H,W
    size_param: int = 512, # The 'size' parameter (e.g., 224, 512) used in load_images
    square_ok: bool = False
) -> np.ndarray:
    """
    Transforms keypoint coordinates from a DUST3R-processed (resized and cropped)
    image back to the original image's coordinate system.

    Args:
        kpts_crop: A NumPy array of shape (N, 2) where N is the number of keypoints,
                   and each row is (x, y) coordinate on the processed image.
        original_size: A tuple (original_width, original_height) of the original image.
        resized_crop_size: A tuple (processed_width, processed_height) of the
                           image after resizing and cropping (i.e., the dimensions
                           of the input image to DUST3R). This is W2, H2 from the
                           load_images function.
        size_param: The 'size' parameter (e.g., 224, 512) used in the
                    original load_images function.
        square_ok: The 'square_ok' parameter used in the original load_images function.

    Returns:
        A NumPy array of shape (N, 2) with the transformed keypoint coordinates
        on the original image.
    """
    # print(f"original_size: {original_size}")
    original_height, original_width = original_size
    original_height = float(original_height)
    original_width = float(original_width)

    # --- 1. Determine the dimensions after resizing but *before* cropping (W_res, H_res) ---
    # This logic mirrors the _resize_pil_image call in load_images
    if size_param == 224:
        # Target long side is used for resizing.
        target_long_side = round(size_param * max(original_width / original_height, original_height / original_width))
        if original_width >= original_height:
            W_res = target_long_side
            H_res = round(original_height * (target_long_side / original_width))
        else:
            H_res = target_long_side
            W_res = round(original_width * (target_long_side / original_height))
    else:
        # Long side is resized to size_param.
        if original_width >= original_height:
            W_res = size_param
            H_res = round(original_height * (size_param / original_width))
        else:
            H_res = size_param
            W_res = round(original_width * (size_param / original_height))

    # print(f"H_res, W_res: {H_res}_{W_res}")

    # --- 2. Calculate the cropping offsets used during processing ---
    cx, cy = W_res // 2, H_res // 2

    if size_param == 224:
        half = min(cx, cy)
        crop_left = cx - half
        crop_top = cy - half
    else:
        halfw = ((2 * cx) // 16) * 8
        halfh = ((2 * cy) // 16) * 8
        if not square_ok and W_res == H_res:
            halfh = round(3 * halfw / 4)
        
        crop_left = cx - halfw
        crop_top = cy - halfh

    

    # --- 4. Reverse the Resizing ---
    # Determine the actual scaling factor applied during the initial resize
    if original_width >= original_height:
        scale_factor = size_param / original_width
    else:
        scale_factor = size_param / original_height
    # --- 3. Reverse the Cropping ---
    # Add the crop offsets to the keypoints from the cropped image
    # print(crop_left, crop_top)
    kpts_resized = kpts_crop.astype(float) # Ensure float for accurate division
    kpts_resized[:, 0] = kpts_resized[:, 0] + crop_left
    kpts_resized[:, 1] = kpts_resized[:, 1] + crop_top


    
    # Divide by the scale factor to get original coordinates
    kpts_original = kpts_resized/ scale_factor 

    return kpts_original


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def draw_matches_on_original_images(img_path1, img_path2, matches_im0, matches_im1, save_path, n_viz=100):
    """
    画出两张原图上的匹配点连线，并保存图像

    Args:
        img_path1: str, 原图1路径
        img_path2: str, 原图2路径
        matches_im0: (N, 2) numpy array，图1上的匹配点坐标（在原图上）
        matches_im1: (N, 2) numpy array，图2上的匹配点坐标（在原图上）
        save_path: str, 输出图像路径
        n_viz: int, 可视化前n个匹配（默认100）
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 读取图像
    img0 = cv2.imread(img_path1)
    img1 = cv2.imread(img_path2)
    key1 = os.path.basename(img_path1)
    key2 = os.path.basename(img_path2)


    if img0 is None or img1 is None:
        print(f"Error: Cannot load {img_path1} or {img_path2}")
        return

    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    H0, W0 = img0.shape[:2]
    H1, W1 = img1.shape[:2]
    canvas_h = max(H0, H1)
    canvas = np.zeros((canvas_h, W0 + W1, 3), dtype=np.uint8)
    canvas[:H0, :W0] = img0
    canvas[:H1, W0:] = img1

    # 可视化子集
    num_matches = min(len(matches_im0), n_viz)
    idxs = np.round(np.linspace(0, len(matches_im0) - 1, num_matches)).astype(int)
    cmap = plt.get_cmap('rainbow')

    for i, idx in enumerate(idxs):
        (x0, y0) = matches_im0[idx]
        (x1, y1) = matches_im1[idx]
        color = tuple((np.array(cmap(i / n_viz))[:3] * 255).astype(int).tolist())

        pt1 = (int(round(x0)), int(round(y0)))
        pt2 = (int(round(x1 + W0)), int(round(y1)))

        cv2.line(canvas, pt1, pt2, color, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, pt1, 2, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, pt2, 2, color, -1, lineType=cv2.LINE_AA)

    # 保存
    cv2.imwrite(f'{save_path}/{key1}_{key2}.jpg', cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    # print(f"Saved match debug image to {save_path}")



img_path1 = '/kaggle/input/image-matching-challenge-2025/train/ETs/et_et003.png'
img_path2 = '/kaggle/input/image-matching-challenge-2025/train/ETs/et_et006.png'
images = load_images([img_path1, img_path2], size=512)
output = inference([tuple(images)], mast3r_model, device, batch_size=1, verbose=True)

# at this stage, you have the raw dust3r predictions
view1, pred1 = output['view1'], output['pred1']
view2, pred2 = output['view2'], output['pred2']

desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

# find 2D-2D matches between the two images
# matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
#                                                device=device, dist='dot', block_size=2**13)
conf1, conf2 = pred1['desc_conf'].squeeze(0).detach(), pred2['desc_conf'].squeeze(0).detach()
print(f"desc1 shape: {desc1.shape}")
print(f"conf1 shape: {conf1.shape}")
corres = extract_correspondences_nonsym(desc1, desc2, conf1, conf2,
                                        device=device, subsample=8, pixel_tol=5)
print(f"corres[0].shape: {corres[0].shape}")
print(f"corres[2].shape: {corres[2].shape}")
score = corres[2]

print("conf min:",score.min().item())
print("conf max:", score.max().item())
print("conf mean:", score.float().mean().item())
print("conf std:", score.float().std().item())
print("conf median:", score.float().median().item())

mask = score >= CONFIG.MATCH_CONF_TH
matches_im0 = corres[0][mask].cpu().numpy()
matches_im1 = corres[1][mask].cpu().numpy()
print(f"matches_im0.shape: {matches_im0.shape}")

# matches_im0 = corres[0].cpu().numpy()
# matches_im1 = corres[1].cpu().numpy()

# ignore small border around the edge
H0, W0 = view1['true_shape'][0]
valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
    matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

H1, W1 = view2['true_shape'][0]
valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
    matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

valid_matches = valid_matches_im0 & valid_matches_im1
matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

# visualize a few matches
import numpy as np
import torch
import torchvision.transforms.functional
from matplotlib import pyplot as pl

n_viz = 100
num_matches = matches_im0.shape[0]
match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

viz_imgs = []
for i, view in enumerate([view1, view2]):
    rgb_tensor = view['img'] * image_std + image_mean
    viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
print(H0,W0,H1,W1)
img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
img = np.concatenate((img0, img1), axis=1)
pl.figure()
pl.imshow(img)
cmap = pl.get_cmap('jet')
for i in range(n_viz):
    (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
    pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
pl.show(block=True)

img0 = cv2.imread(img_path1)
img1 = cv2.imread(img_path2)
H0, W0 = img0.shape[:2]
H1, W1 = img1.shape[:2]
viz_matches_im0_org = transform_keypoints_to_original(viz_matches_im0, (H0, W0))
viz_matches_im1_org = transform_keypoints_to_original(viz_matches_im1, (H1, W1))

out_org_dir= os.path.join("/kaggle/working", "temp")
os.makedirs(out_org_dir, exist_ok=True)

draw_matches_on_original_images(img_path1, img_path2, viz_matches_im0_org, viz_matches_im1_org, out_org_dir, n_viz=100)


def get_img_pairs_exhaustive(img_fnames):
    index_pairs = []
    for i in range(len(img_fnames)):
        for j in range(i+1, len(img_fnames)):
            index_pairs.append((i,j))
    return index_pairs


import os
import torch
import PIL
import numpy as np # Ensure numpy is imported for checking np.ndarray
from PIL import Image
from mast3r.retrieval.processor import Retriever
from mast3r.image_pairs import make_pairs
from mast3r.model import AsymmetricMASt3R


def get_image_list(images_path):
    """
    Scans the specified path for all image files and returns their relative paths.
    Skips unidentifiable or corrupt image files.
    """
    file_list = [os.path.relpath(os.path.join(dirpath, filename), images_path)
                 for dirpath, _, filenames in os.walk(images_path)
                 for filename in filenames]
    file_list = sorted(file_list)
    image_list = []
    for filename in file_list:
        try:
            with Image.open(os.path.join(images_path, filename)) as im:
                im.verify()  # Verify image file integrity
                image_list.append(filename)
        except (OSError, PIL.UnidentifiedImageError):
            print(f'Skipping invalid image file: {filename}')
    return image_list


def make_pair_with_mast3r_return_pairs(
    image_dir: str,
    weights_path: str,  # Path to the AsymmetricMASt3R model weights
    retrieval_model_path: str,  # Path to the retrieval model (e.g., "trainingfree.pth")
    scene_graph: str = 'retrieval-20-40',
    device: str = 'cuda'
):
    """
    Generates image pairs using MASt3R + ASMK retrieval, returning a list of pairs.

    Args:
        image_dir (str): Path to the directory containing images.
        weights_path (str): Path to the AsymmetricMASt3R model weights.
        retrieval_model_path (str): Path to the retrieval model (e.g., "trainingfree.pth").
        scene_graph (str, optional): String defining the scene graph construction strategy. 
                                     Defaults to 'retrieval-20-1-10-1'.
        device (str, optional): PyTorch device to use ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        sorted_pairs: List[Tuple[str, str]], where each tuple contains 
                      the relative paths of the paired images (img1, img2).
    """
    print("🖼️ Scanning images...")
    imgs = get_image_list(image_dir)
    imgs_fp = [os.path.join(image_dir, f) for f in imgs]

    if not imgs:
        print("⚠️ No valid images found in the directory. Returning empty pairs.")
        return []

    print(f"⚙️ Loading backbone model from {weights_path}...")
    backbone = AsymmetricMASt3R.from_pretrained(weights_path).to(device).eval()

    # print("🔍 Running ASMK retrieval...")
    retriever = Retriever(retrieval_model_path, backbone=backbone)
    
    with torch.no_grad():
        sim_matrix_np = retriever(imgs_fp) 
        
    # Cleanup GPU cache
    del retriever
    del backbone 
    torch.cuda.empty_cache()

    # print("🧮 Generating image pairs...")
    # make_pairs 应该期望一个 PyTorch 张量作为 sim_mat
    raw_pairs = make_pairs(imgs, scene_graph, prefilter=None, symmetrize=True, sim_mat=sim_matrix_np)
    # print(raw_pairs)
    
    sorted_pairs = sorted(set(tuple(sorted([a, b])) for a, b in raw_pairs))

    print(f"✅ Generated {len(sorted_pairs)} unique image pairs.")
    return sorted_pairs


import kornia as K
import kornia.feature as KF
# --- Helper function for image loading (if not already defined) ---
def load_torch_image(fname, device=torch.device('cpu')):
    img = K.io.load_image(fname, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    return img


# Must Use efficientnet global descriptor to get matching shortlists.
def get_global_desc(fnames, device = torch.device('cpu')):
    processor = AutoImageProcessor.from_pretrained('/kaggle/input/dinov2/pytorch/base/1')
    model = AutoModel.from_pretrained('/kaggle/input/dinov2/pytorch/base/1')
    model = model.eval()
    model = model.to(device)
    global_descs_dinov2 = []
    for i, img_fname_full in tqdm(enumerate(fnames),total= len(fnames)):
        key = os.path.splitext(os.path.basename(img_fname_full))[0]
        timg = load_torch_image(img_fname_full)
        with torch.inference_mode():
            inputs = processor(images=timg, return_tensors="pt", do_rescale=False).to(device)
            outputs = model(**inputs)
            dino_mac = F.normalize(outputs.last_hidden_state[:,1:].max(dim=1)[0], dim=1, p=2)
        global_descs_dinov2.append(dino_mac.detach().cpu())
    global_descs_dinov2 = torch.cat(global_descs_dinov2, dim=0)
    return global_descs_dinov2


def get_img_pairs_exhaustive(img_fnames):
    index_pairs = []
    for i in range(len(img_fnames)):
        for j in range(i+1, len(img_fnames)):
            index_pairs.append((i,j))
    return index_pairs


def get_image_pairs_shortlist_org(fnames,
                              sim_th = 0.6, # should be strict
                              min_pairs = 60,
                              exhaustive_if_less = 20,
                              device=torch.device('cpu')):
    num_imgs = len(fnames)
    if num_imgs <= exhaustive_if_less:
        return get_img_pairs_exhaustive(fnames)
    descs = get_global_desc(fnames, device=device)
    dm = torch.cdist(descs, descs, p=2).detach().cpu().numpy()

    
    mask = dm <= sim_th
    total = 0
    matching_list = []
    ar = np.arange(num_imgs)
    already_there_set = []
    for st_idx in range(num_imgs-1):
        mask_idx = mask[st_idx]
        to_match = ar[mask_idx]
        if len(to_match) < min_pairs:
            to_match = np.argsort(dm[st_idx])[:min_pairs]  
        for idx in to_match:
            if st_idx == idx:
                continue
            if dm[st_idx, idx] < 10000:
                matching_list.append(tuple(sorted((st_idx, idx.item()))))
                total+=1
    matching_list = sorted(list(set(matching_list)))
    return matching_list


import pycolmap
print(f"pycolmap version: {pycolmap.__version__}")



# Collect vital info from the dataset

@dataclasses.dataclass
class Prediction:
    image_id: str | None  # A unique identifier for the row -- unused otherwise. Used only on the hidden test set.
    dataset: str
    filename: str
    cluster_index: int | None = None
    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None

# Set is_train=True to run the notebook on the training data.
# Set is_train=False if submitting an entry to the competition (test data is hidden, and different from what you see on the "test" folder).
is_train = False
data_dir = '/kaggle/input/image-matching-challenge-2025'
workdir = '/kaggle/working/result/'
os.makedirs(workdir, exist_ok=True)

if is_train:
    sample_submission_csv = os.path.join(data_dir, 'train_labels.csv')
else:
    sample_submission_csv = os.path.join(data_dir, 'sample_submission.csv')

samples = {}
competition_data = pd.read_csv(sample_submission_csv)
for _, row in competition_data.iterrows():
    # Note: For the test data, the "scene" column has no meaning, and the rotation_matrix and translation_vector columns are random.
    if row.dataset not in samples:
        samples[row.dataset] = []
    samples[row.dataset].append(
        Prediction(
            image_id=None if is_train else row.image_id,
            dataset=row.dataset,
            filename=row.image
        )
    )

for dataset in samples:
    print(f'Dataset "{dataset}" -> num_images={len(samples[dataset])}')


import multiprocessing


import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2


def save_match_viz_image(
    key1,
    key2,
    view1,
    view2,
    matches_im0,
    matches_im1,
    feature_dir,
    n_viz: int = 100
):
    """
    Save a visual match image for a pair of images using descriptor matches.

    Parameters:
        key1, key2: str
            Base filenames of the matched image pair (used for naming the output).
        view1, view2: dict
            MASt3R inference outputs containing 'img' and 'true_shape'.
        matches_im0, matches_im1: np.ndarray of shape (N, 2)
            Coordinates of matched keypoints in image 0 and image 1.
        feature_dir: str
            Path to save the visualized match image.
        n_viz: int
            Number of matches to visualize (default: 100).
    """
    if matches_im0.shape[0] == 0:
        return  # nothing to draw

    n_viz = min(n_viz, matches_im0.shape[0])
    idx = np.round(np.linspace(0, matches_im0.shape[0] - 1, n_viz)).astype(int)
    viz_matches_im0 = matches_im0[idx]
    viz_matches_im1 = matches_im1[idx]

    image_mean = torch.tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
    image_std = torch.tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)

    viz_imgs = []
    for view in [view1, view2]:
        rgb_tensor = view['img'].cpu() * image_std + image_mean
        rgb_np = rgb_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
        viz_imgs.append((rgb_np * 255).astype(np.uint8))

    H0, W0 = viz_imgs[0].shape[:2]
    H1, W1 = viz_imgs[1].shape[:2]
    img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant')
    img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant')
    img = np.concatenate((img0, img1), axis=1)

    cmap = plt.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        color = tuple(int(c * 255) for c in cmap(i / (n_viz - 1))[:3])
        cv2.line(img, (int(x0), int(y0)), (int(x1 + W0), int(y1)), color, thickness=1)
        cv2.circle(img, (int(x0), int(y0)), radius=2, color=color, thickness=-1)
        cv2.circle(img, (int(x1 + W0), int(y1)), radius=2, color=color, thickness=-1)

    out_dir = os.path.join(feature_dir, "debug_vis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{key1}-{key2}.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"[Match Debug] Saved to {out_path}")



from typing import Dict, Tuple
from collections import defaultdict
import numpy as np

def unify_keypoints_and_matches(
    out_match: Dict[str, Dict[str, np.ndarray]],
    round_digits: int = 1
) -> Tuple[
    Dict[str, np.ndarray],                    # global_keypoints[img] = (N, 2)
    Dict[Tuple[str, str], np.ndarray]         # global_matches[(img1, img2)] = (M, 2) (global IDs)
]:
    # Step 1: 收集每张图所有坐标
    keypoints_per_image = defaultdict(list)

    for img1, subdict in out_match.items():
        for img2, match in subdict.items():
            pts1 = np.round(match[:, :2], decimals=round_digits)
            pts2 = np.round(match[:, 2:], decimals=round_digits)
            keypoints_per_image[img1].append(pts1)
            keypoints_per_image[img2].append(pts2)

    # Step 2: 为每张图构建 unique keypoints 和坐标 -> id 映射
    global_keypoints = {}
    coord_to_id = {}

    for img, kpt_list in keypoints_per_image.items():
        all_pts = np.concatenate(kpt_list, axis=0)
        all_pts = np.round(all_pts, decimals=round_digits)
        unique_pts, inverse = np.unique(all_pts, axis=0, return_inverse=True)
        global_keypoints[img] = unique_pts

        # 建立映射 coord → id
        coord_to_id[img] = {tuple(pt): idx for idx, pt in enumerate(unique_pts)}

    # Step 3: 将匹配对转换为 global_id 匹配
    global_matches = {}

    for img1, subdict in out_match.items():
        for img2, match in subdict.items():
            pts1 = np.round(match[:, :2], decimals=round_digits)
            pts2 = np.round(match[:, 2:], decimals=round_digits)

            ids1 = np.array([coord_to_id[img1][tuple(pt)] for pt in pts1])
            ids2 = np.array([coord_to_id[img2][tuple(pt)] for pt in pts2])

            global_matches[(img1, img2)] = np.stack([ids1, ids2], axis=1)

    return global_keypoints, global_matches



import os
import h5py
import numpy as np

def save_unified_keypoints_and_matches(global_keypoints, global_matches, feature_dir, lock=None):
    os.makedirs(feature_dir, exist_ok=True)
    save_kpts_file = os.path.join(feature_dir, 'keypoints.h5')
    save_matches_file = os.path.join(feature_dir, 'matches.h5')
    save_matches_txt_file = os.path.join(feature_dir, 'pairs.txt')

    # 删除旧文件
    for f in [save_kpts_file, save_matches_file, save_matches_txt_file]:
        if os.path.exists(f):
            os.remove(f)

    # 保存关键点坐标
    with h5py.File(save_kpts_file, 'w') as f_kp:
        for img_name, kpts in global_keypoints.items():
            f_kp[img_name] = kpts
        #     f_kp.create_dataset(img_name, data=np.asarray(kpts, dtype=np.float32))
        # f_kp.flush()
    print(f"✅ Saved keypoints to {save_kpts_file}")

    # 保存匹配
    with h5py.File(save_matches_file, 'w') as f_match:
        for (img1, img2), match in global_matches.items():
            group = f_match.require_group(img1)
            if len(match) >= CONFIG.MAST3R_MIN_PAIR:
            # match = np.asarray(match, dtype=np.int32)
                group.create_dataset(img2, data=match)
        # f_match.flush()
    print(f"✅ Saved matches to {save_matches_file}")
    
    with h5py.File(save_matches_file, 'r') as f, open(save_matches_txt_file, 'w') as fout:
        for k1 in f.keys():
            group = f[k1]
            for k2 in group.keys():
                fout.write(f"{k1} {k2}\n")
    print(f"✅ Saved matches to {save_matches_txt_file}")


def match_with_mast3r_and_save(index_pairs, image_list, feature_dir, model, device, lock):
    os.makedirs(feature_dir, exist_ok=True)
    cache = {}
    unique_keypoints = defaultdict(list)
    out_match = defaultdict(dict)
    local_cache = {}
    
    out_dir = os.path.join(feature_dir, "pair_res")
    out_org_dir= os.path.join(feature_dir, "pair_res_onorg")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_org_dir, exist_ok=True)
    
    for idx1, idx2 in tqdm(index_pairs):
        name1, name2 = image_list[idx1], image_list[idx2]
        key1, key2 = os.path.basename(name1), os.path.basename(name2)

        # Only re-run inference for key1 if not in cache
        images = load_images([name1, name2], size=512, verbose=False)
        output = inference([tuple(images)], mast3r_model, device, batch_size=1, verbose=False)
        
        # at this stage, you have the raw dust3r predictions
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']
        
        desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

        # matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8, device=device)
        # print(f"get pair for {key1}_{key2}, {len(matches_im0)}")

        conf1, conf2 = pred1['desc_conf'].squeeze(0).detach(), pred2['desc_conf'].squeeze(0).detach()
        corres = extract_correspondences_nonsym(desc1, desc2, conf1, conf2,
                                        device=device, subsample=8, pixel_tol=5)
        score = corres[2]
        mask = score >= CONFIG.MATCH_CONF_TH
        matches_im0 = corres[0][mask].cpu().numpy()
        matches_im1 = corres[1][mask].cpu().numpy()

        # matches_im0 = corres[0].cpu().numpy()
        # matches_im1 = corres[1].cpu().numpy()

        if len(matches_im0) < CONFIG.MAST3R_MIN_PAIR:
            continue

        H0, W0 = view1['true_shape'][0].tolist()
        H1, W1 = view2['true_shape'][0].tolist()
        
        valid0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < W0 - 3) & (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < H0 - 3)
        valid1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < W1 - 3) & (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < H1 - 3)
        valid = valid0 & valid1

        matches_im0 = matches_im0[valid]
        matches_im1 = matches_im1[valid]
        if len(matches_im0) < CONFIG.MAST3R_MIN_PAIR:
            continue
        # print("transform_keypoints_to_original begin")
        # print(f"{key1}_{key2}: {len(matches_im0)} matches")
        img0 = cv2.imread(name1)
        img1 = cv2.imread(name2)
        H0, W0 = img0.shape[:2]
        H1, W1 = img1.shape[:2]
        matches_im0_org = transform_keypoints_to_original(matches_im0, (H0, W0))
        matches_im1_org = transform_keypoints_to_original(matches_im1, (H1, W1))

        # matches_im0_org = transform_keypoints_to_original(matches_im0, view1['true_shape'][0].tolist())
        # matches_im1_org = transform_keypoints_to_original(matches_im1, view2['true_shape'][0].tolist())
        # print("transform_keypoints_to_original end")

        unique_keypoints[key1].append(matches_im0_org)
        unique_keypoints[key2].append(matches_im1_org)
        out_match[key1][key2] = np.concatenate([matches_im0_org, matches_im1_org], axis=1)
        if False:
            save_match_viz_image(key1, key2, view1, view2, matches_im0, matches_im1, out_dir)
        if False:
            draw_matches_on_original_images(name1, name2, matches_im0_org, matches_im1_org, out_org_dir)
    
    # print("out of loop")

    keypoints_unified = {}
    keypoints_id_map = {}
    out_match_unified = defaultdict(dict)

    global_keypoints, global_matches = unify_keypoints_and_matches(out_match)
    # print("points and matches unified")

    save_unified_keypoints_and_matches(global_keypoints, global_matches, feature_dir, lock)
    # print(f"Saved keypoints and matches to {feature_dir}")



import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_keypoints_and_matches(images_input, unified_kp_path, remapped_matches_path, feature_dir='visualization_output'):
    output_dir = os.path.join(feature_dir, 'visualization_output')
    os.makedirs(output_dir, exist_ok=True)

    # Load images and determine image_keys for HDF5 lookup
    if isinstance(images_input[0], str):
        loaded_images = [cv2.imread(img_path) for img_path in images_input]
        image_keys = [os.path.basename(img_path) for img_path in images_input]
    else:
        loaded_images = images_input
        # If images_input are already arrays, you need to provide the corresponding keys
        # This part is crucial: image_keys MUST align with the HDF5 keys
        image_keys = image_keys_in_h5 # Use the predefined list for the dummy case

    # Load unified keypoints
    keypoints_data = {}
    with h5py.File(unified_kp_path, 'r') as f_kp:
        for img_name_raw in f_kp.keys():
            img_name = img_name_raw.decode('utf-8') if isinstance(img_name_raw, bytes) else img_name_raw
            keypoints_data[img_name] = f_kp[img_name_raw][()] # Access with raw key if bytes

    # Load remapped matches - CORRECTED LOGIC
    # Store (img1_key, img2_key) directly with matches for robust iteration
    matches_data_pairs = [] # Will store (img1_key, img2_key, matches_array)
    with h5py.File(remapped_matches_path, 'r') as f_matches:
        print("\n--- Loading remapped matches from HDF5 ---")
        for img1_group_key_candidate in tqdm(f_matches.keys(), desc="Loading matches"):
            img1_key = img1_group_key_candidate.decode('utf-8') if isinstance(img1_group_key_candidate, bytes) else img1_group_key_candidate

            img1_group = f_matches[img1_group_key_candidate] # Access with raw key

            if isinstance(img1_group, h5py.Group):
                for img2_dataset_key_candidate in img1_group.keys():
                    img2_key = img2_dataset_key_candidate.decode('utf-8') if isinstance(img2_dataset_key_candidate, bytes) else img2_dataset_key_candidate

                    try:
                        matches_array = img1_group[img2_dataset_key_candidate][()]
                        matches_data_pairs.append((img1_key, img2_key, matches_array))
                    except Exception as e:
                        print(f"Error loading matches for pair ({img1_key}, {img2_key}): {e}")
            else:
                print(f"Warning: Expected '{img1_key}' to be a group, but found {type(img1_group)}. Skipping its contents.")


    # --- Drawing Keypoints ---
    print("\n--- Drawing Keypoints ---")
    for i, img_key in enumerate(image_keys):
        if img_key in keypoints_data:
            img = loaded_images[i].copy()
            kpts = keypoints_data[img_key]

            for kp in kpts:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(img, (x, y), 3, (0, 255, 0), -1) # Green circle for keypoint

            output_kp_path = os.path.join(output_dir, f"keypoints_{img_key}")
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(output_kp_path, img)
            print(f"Keypoints drawn on {img_key}, saved to {output_kp_path}")
        else:
            print(f"No keypoints found for {img_key} in unified keypoints file.")

    # --- Drawing Matches ---
    print("\n--- Drawing Matches ---")
    # Iterate through the (img1_key, img2_key, matches) tuples directly
    for img_name1, img_name2, matches in matches_data_pairs:
        # We no longer need to split img_pair_key, as we have img_name1 and img_name2 directly

        # Find the actual image objects and their keypoints using image_keys list
        try:
            img1_idx = image_keys.index(img_name1)
            img2_idx = image_keys.index(img_name2)
        except ValueError:
            print(f"Skipping matches for {img_name1}-{img_name2}: One or both image names not found in the provided 'images' list/keys.")
            continue

        img1 = loaded_images[img1_idx].copy()
        img2 = loaded_images[img2_idx].copy()

        kpts1 = keypoints_data.get(img_name1)
        kpts2 = keypoints_data.get(img_name2)

        if kpts1 is None or kpts2 is None:
            print(f"Skipping matches for {img_name1}-{img_name2}: keypoints not found for one or both images in unified keypoints.")
            continue
        if len(matches) == 0:
            print(f"No matches to draw for {img_name1}-{img_name2}.")
            continue

        # Ensure images are 3 channels for drawing lines
        if len(img1.shape) == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if len(img2.shape) == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        # Create a concatenated image for drawing matches
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        max_h = max(h1, h2)
        matched_img = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)
        matched_img[0:h1, 0:w1] = img1
        matched_img[0:h2, w1:w1+w2] = img2

        num_matches_to_draw = min(len(matches), 200) # Draw up to 200 matches to avoid clutter, adjust as needed

        for i in np.linspace(0, len(matches) - 1, num_matches_to_draw, dtype=int):
            match = matches[i]
            kp1_idx, kp2_idx = int(match[0]), int(match[1])

            # Bounds check for keypoint indices
            if kp1_idx >= len(kpts1) or kp2_idx >= len(kpts2):
                # print(f"Warning: Match index out of bounds for {img_name1}-{img_name2}. Skipping match {kp1_idx}-{kp2_idx}.")
                continue

            pt1 = tuple(map(int, kpts1[kp1_idx][:2]))
            pt2 = tuple(map(int, kpts2[kp2_idx][:2]))

            # Draw circles on the concatenated image
            cv2.circle(matched_img, pt1, 5, (0, 0, 255), 2) # Red circle on img1 side
            cv2.circle(matched_img, (pt2[0] + w1, pt2[1]), 5, (255, 0, 0), 2) # Blue circle on img2 side

            # Draw a line connecting the matched keypoints
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.line(matched_img, pt1, (pt2[0] + w1, pt2[1]), color, 1)

        output_match_path = os.path.join(output_dir, f"matches_{img_name1}_{img_name2}.png")
        cv2.imwrite(output_match_path, matched_img)
        print(f"Matches drawn between {img_name1} and {img_name2}, saved to {output_match_path}")


# Example call (replace with your actual 'images' list)
# If your 'images' are file paths:
# images_file_paths = ['path/to/your/image1.jpg', 'path/to/your/image2.jpg', ...]
# draw_keypoints_and_matches(images_file_paths, unified_kp_path, remapped_matches_path)

# If your 'images' are loaded numpy arrays (as in the dummy example above):
# draw_keypoints_and_matches(images, unified_kp_path, remapped_matches_path)


import os
import gc
import time
import numpy as np
import concurrent.futures
import multiprocessing
from pathlib import Path
from time import sleep, time
# from pycolmap import verify_matches, TwoViewGeometryOptions

def run_verify_matches_safe(database_path, pairs_path, max_retries=5):
    def _safe_verify():
        verify_matches(
            database_path=database_path,
            pairs_path=pairs_path,
            options=TwoViewGeometryOptions()
        )

    for attempt in range(max_retries):
        print(f"🔁 Attempt {attempt + 1} to run verify_matches")
        proc = multiprocessing.Process(target=_safe_verify)
        proc.start()
        proc.join()

        if proc.exitcode in [0, 1]:
            print("✅ verify_matches succeeded")
            return
        else:
            print(f"⚠️ verify_matches crashed with code {proc.exitcode}")
    raise RuntimeError("❌ verify_matches failed after multiple retries.")

def reconstruct_from_db(feature_dir, img_dir):
    result = {}
    local_timings = {'RANSAC': [], 'Reconstruction': []}

    database_path = f'{feature_dir}/colmap.db'
    pairs_txt = f'{feature_dir}/pairs.txt'
    if os.path.isfile(database_path):
        os.remove(database_path)
    gc.collect()
    sleep(1)

    import_into_colmap(img_dir, feature_dir=feature_dir, database_path=database_path)
    sleep(1)
    print(f"import {database_path} done!")
    output_path = f'{feature_dir}/colmap_rec'
    os.makedirs(output_path, exist_ok=True)
    print("colmap database")

    t = time()
    run_verify_matches_safe(database_path, pairs_txt)
    print("verify matching done!!!!")
    local_timings['RANSAC'].append(time() - t)
    print(f'RANSAC in {local_timings["RANSAC"][-1]:.4f} sec')

    t = time()
    mapper_options = pycolmap.IncrementalPipelineOptions()
    mapper_options.min_model_size = 3
    mapper_options.max_num_models = 5
    maps = pycolmap.incremental_mapping(database_path=database_path, image_path=img_dir, 
                                        output_path=output_path, options=mapper_options)
    print(maps)
    for map_index, rec in maps.items():
        result[map_index]={}
        for img_id, image in rec.images.items():
            result[map_index][image.name] = {
                'R': image.cam_from_world.rotation.matrix().tolist(),
                't': image.cam_from_world.translation.tolist()
            }
    local_timings['Reconstruction'].append(time() - t)
    print(f'Reconstruction done in {local_timings["Reconstruction"][-1]:.4f} sec')

    return result, local_timings

def run_one_dataset(dataset, predictions, data_dir, workdir, is_train, model, device, lock=None):
    timings = {
        "shortlisting": [],
        "feature_matching": [],
        "RANSAC": [],
        "Reconstruction": [],
    }

    try:
        images_dir = os.path.join(data_dir, 'train' if is_train else 'test', dataset)
        images = [os.path.join(images_dir, p.filename) for p in predictions]

        print(f'Processing dataset "{dataset}": {len(images)} images')
        filename_to_index = {p.filename: idx for idx, p in enumerate(predictions)}
        feature_dir = os.path.join(workdir, 'featureout', dataset)
        os.makedirs(feature_dir, exist_ok=True)

        t = time()
        # index_pairs = get_image_pairs_shortlist_org(
        #     images, sim_th=0.2, min_pairs=10,
        #     exhaustive_if_less=20, device=device)

        index_img_pairs = make_pair_with_mast3r_return_pairs(image_dir=images_dir,
                      weights_path = local_model_directory,  # Path to the AsymmetricMASt3R model weights (e.g., "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric" or local path)
                      retrieval_model_path = retrival_model_dir,  # Path to the retrieval model (e.g., "trainingfree.pth")
                      device = device)

        indexed_pairs = []
        for filename1, filename2 in index_img_pairs:
            try:
                idx1 = filename_to_index[filename1]
                idx2 = filename_to_index[filename2]
                indexed_pairs.append((idx1, idx2))
            except KeyError as e:
                print(f"Warning: Filename not found in mapping: {e}. Skipping pair ({filename1}, {filename2}).")

        timings['shortlisting'].append(time() - t)
        print(f'Shortlisting done: {len(indexed_pairs)} pairs')
        gc.collect()

        t = time()
        match_with_mast3r_and_save(indexed_pairs, images, feature_dir, model, device, lock)
        timings['feature_matching'].append(time() - t)
        print(f'MASt3R matching done in {time() - t:.2f} sec')
        gc.collect()

        maps, local_timings = reconstruct_from_db(feature_dir, images_dir)
        # print(maps)
        
        # Sort map clusters by number of registered images (ascending)
        sorted_map_items = sorted(maps.items(), key=lambda x: len(x[1]))
        # print(sorted_map_items)
        
        registered = 0
        for new_cluster_idx, (original_map_index, cur_map) in enumerate(sorted_map_items):
            for image_name, pose in cur_map.items():
                idx = filename_to_index[image_name]
                pred = predictions[idx]
                pred.cluster_index = new_cluster_idx  # use the sorted order
                pred.rotation = np.array(pose['R'])
                pred.translation = np.array(pose['t'])
                registered += 1

        mapping_result_str = f"Dataset {dataset} -> Registered {registered} / {len(images)} images with {len(maps)} clusters"
        return mapping_result_str, timings

    except Exception as e:
        print(f"Error in dataset {dataset}: {e}")
        return f"Dataset \"{dataset}\" -> Failed!", timings

def run_mast3r_pipeline(samples, data_dir, workdir, is_train, model, device):
    max_images = None
    datasets_to_process = ['ETs'] if is_train else list(samples.keys())

    overall_timings = {
        "shortlisting": [],
        "feature_matching": [],
        "RANSAC": [],
        "Reconstruction": [],
    }
    mapping_result_strs = []
    lock = multiprocessing.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for dataset, predictions in samples.items():
            if datasets_to_process and dataset not in datasets_to_process:
                print(f"Skipping {dataset}")
                continue
            futures.append(executor.submit(run_one_dataset, dataset, predictions, data_dir, workdir, is_train, model, device, lock))

        for future in concurrent.futures.as_completed(futures):
            result_str, timings = future.result()
            mapping_result_strs.append(result_str)
            for k in timings:
                overall_timings[k].extend(timings[k])

    print('\nResults')
    for s in mapping_result_strs:
        print(s)


run_mast3r_pipeline(samples, data_dir, workdir, is_train, mast3r_model, device)


array_to_str = lambda array: ';'.join([f"{x:.09f}" for x in array])
none_to_str = lambda n: ';'.join(['nan'] * n)
submission_file = '/kaggle/working/submission.csv'
with open(submission_file, 'w') as f:
    if is_train:
        f.write('dataset,scene,image,rotation_matrix,translation_vector\n')
        for dataset, predictions in samples.items():
            for prediction in predictions:
                cluster_name = 'outliers' if prediction.cluster_index is None else f'cluster{prediction.cluster_index}'

                # ✅ `rotation` is a list of lists, flatten it
                if prediction.rotation is None:
                    rotation_str = none_to_str(9)
                else:
                    rotation_flat =  prediction.rotation.flatten()  # flatten 3x3 list -> 9 elems
                    rotation_str = array_to_str(rotation_flat)

                # ✅ `translation` is a flat list
                if prediction.translation is None:
                    translation_str = none_to_str(3)
                else:
                    translation_str = array_to_str(prediction.translation)

                f.write(f'{prediction.dataset},{cluster_name},{prediction.filename},{rotation_str},{translation_str}\n')
    else:
        f.write('image_id,dataset,scene,image,rotation_matrix,translation_vector\n')
        for dataset, predictions in samples.items():
            for prediction in predictions:
                cluster_name = 'outliers' if prediction.cluster_index is None else f'cluster{prediction.cluster_index}'

                if prediction.rotation is None:
                    rotation_str = none_to_str(9)
                else:
                    rotation_flat =  prediction.rotation.flatten()
                    rotation_str = array_to_str(rotation_flat)

                if prediction.translation is None:
                    translation_str = none_to_str(3)
                else:
                    translation_str = array_to_str(prediction.translation)

                f.write(f'{prediction.image_id},{prediction.dataset},{cluster_name},{prediction.filename},{rotation_str},{translation_str}\n')

# Preview the output
!head {submission_file}



# Definitely Compute results if running on the training set.
# Do not do this when submitting a notebook for scoring. All you have to do is save your submission to /kaggle/working/submission.csv.

if is_train:
    t = time()
    final_score, dataset_scores = metric.score(
        gt_csv='/kaggle/input/image-matching-challenge-2025/train_labels.csv',
        user_csv=submission_file,
        thresholds_csv='/kaggle/input/image-matching-challenge-2025/train_thresholds.csv',
        mask_csv=None if is_train else os.path.join(data_dir, 'mask.csv'),
        inl_cf=0,
        strict_cf=-1,
        verbose=True,
    )
    print(f'Computed metric in: {time() - t:.02f} sec.')

