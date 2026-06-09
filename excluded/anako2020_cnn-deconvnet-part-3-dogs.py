2+2+2-9-99


import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # avoid PIL truncation errors on some JPEGs

import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.dpi": 120,           # sharper inline figures
    "savefig.dpi": 120,
    "figure.facecolor": "white", # white bg (matches our saved grids)
    "axes.facecolor": "white",
    "axes.grid": False,
})



# ==== CENTRALIZED VISUALIZATION STYLING CONFIGURATION ====
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from scipy.ndimage import gaussian_filter

class VisualizationConfig:
    """Centralized configuration for all visualizations"""
    # Color scheme (easy to switch between light/dark themes)
    FIGURE_BG = 'black'           # 'white' for light theme, 'black' for dark theme
    TEXT_COLOR = 'white'          # 'black' for light theme, 'white' for dark theme
    
    # Deconv styling parameters
    STANDARD_ENHANCE_FACTOR = 2.0
    DECONV_SATURATION = 0.75
    DECONV_GAMMA = 0.92
    DECONV_BLUR_SIGMA = 0.6
    DECONV_GRAY_BLEND = 0.12

def style_figure(fig):
    """Apply consistent styling to matplotlib figures"""
    fig.patch.set_facecolor(VisualizationConfig.FIGURE_BG)

def style_axis(ax):
    """Apply consistent styling to matplotlib axes"""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color(VisualizationConfig.FIGURE_BG)
    ax.set_facecolor(VisualizationConfig.FIGURE_BG)

def apply_deconv_styling(img_rgb):
    """Apply paper-style processing to deconvolutional reconstructions"""
    def _robust_rescale(img, lo=1, hi=99, out_lo=0.12, out_hi=0.88):
        p_lo = np.percentile(img, lo)
        p_hi = np.percentile(img, hi)
        if p_hi <= p_lo:
            return np.clip(img, 0, 1)
        x = (img - p_lo) / (p_hi - p_lo)
        x = np.clip(x, 0, 1)
        return out_lo + x * (out_hi - out_lo)
    
    x = img_rgb.astype(np.float32)
    # Robust per-crop contrast
    x = _robust_rescale(x, lo=1, hi=99, out_lo=0.12, out_hi=0.88)
    # Light blur (apply channel-wise)
    x = np.stack([gaussian_filter(x[..., i], VisualizationConfig.DECONV_BLUR_SIGMA) for i in range(3)], axis=-1)
    # Desaturate in HSV
    hsv = rgb_to_hsv(np.clip(x, 0, 1))
    hsv[..., 1] *= VisualizationConfig.DECONV_SATURATION
    x = hsv_to_rgb(hsv)
    # Mild gamma
    x = np.clip(x, 0, 1) ** VisualizationConfig.DECONV_GAMMA
    # Blend to mid-gray
    x = (1 - VisualizationConfig.DECONV_GRAY_BLEND) * x + VisualizationConfig.DECONV_GRAY_BLEND * 0.5
    return np.clip(x, 0, 1)



# ==== paths & sanity ====
import os, glob

# path to images
val_path     = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val"
labels_path  = "/kaggle/input/imagenet-object-localization-challenge/LOC_val_solution.csv"
mapping_path = "/kaggle/input/imagenet-object-localization-challenge/LOC_synset_mapping.txt"

# unify naming 
VAL_DIR     = val_path
LABELS_CSV  = labels_path
SYNSET_MAP  = mapping_path
VAL_ANN_DIR = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Annotations/CLS-LOC/val"  # XMLs

# checks
assert os.path.exists(VAL_DIR),     f"VAL_DIR missing: {VAL_DIR}"
assert os.path.exists(SYNSET_MAP),  f"SYNSET_MAP missing: {SYNSET_MAP}"

# labels CSV may not be strictly needed (we use XMLs), so don't hard-fail if absent:
print("LABELS_CSV present:", os.path.exists(LABELS_CSV))

n_imgs = len(glob.glob(os.path.join(VAL_DIR, "*.JPEG")))
print(f"VAL_DIR OK → {VAL_DIR}  • JPEGs: {n_imgs}")
print("VAL_ANN_DIR present:", os.path.exists(VAL_ANN_DIR))


# tiny helper: load synset→text mapping (we’ll reuse later)
def load_synset_map(path=SYNSET_MAP):
    m = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip(): continue
            sid, txt = line.strip().split(" ", 1)
            m[sid] = txt
    return m

SYN2TXT = load_synset_map()
print("Mapping entries:", len(SYN2TXT))
for k in list(SYN2TXT.keys())[:3]:
    print("  ", k, "→", SYN2TXT[k])


5+9


# find synset IDs by keywords (case-insensitive)
import re

SYNC_MAP = "/kaggle/input/imagenet-object-localization-challenge/LOC_synset_mapping.txt"

def find_synsets(keywords):
    kw = [re.compile(rf"\b{re.escape(k)}\b", re.I) for k in keywords]
    hits = []
    with open(SYNC_MAP, "r") as f:
        for line in f:
            syn, name_all = line.strip().split(" ", 1)
            name = name_all.split(",")[0].strip().lower()  # primary label
            if any(p.search(name) for p in kw):
                hits.append((syn, name))
    print(f"Found {len(hits)} matches:")
    for syn, name in hits:
        print(f"  {syn:>10}  {name}")
    return hits

# EXAMPLES — change these to what you actually need:
_ = find_synsets(["golden retriever", "chihuahua", "siberian husky", "pug", 'dog','puppy','sheepdog','sighthound','hound','greyhound','wolfhound',
            'terrier','retriever','shepherd','spaniel','setter','pointer','mastiff','bulldog','poodle','husky','beagle','collie',])



import torch
from torchvision.models import alexnet

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained model
model = alexnet(weights="DEFAULT").to(device) 
model.eval()


#SYNSETS = {'n02091032', 'n02116738'}  # my picks - choosing dogs
SYNSETS = {'n02091032', 'n02116738','n02088364', 'n02097658'}  # my picks - choosing dogs



# === STEP 0: Filter by hardcoded synsets (robust) ===
import os
import pandas as pd
from PIL import Image  # optional; only used if you uncomment the loader

# --- required globals expected: VAL_DIR, LABELS_CSV, SYNSET_MAP, SYNSETS ---
assert isinstance(SYNSETS, (set, list, tuple)), "SYNSETS must be a set/list/tuple of synset ids like {'n02084071', ...}"
SYNSETS = set(SYNSETS)

assert os.path.isdir(VAL_DIR),         f"VAL_DIR not found: {VAL_DIR}"
assert os.path.isfile(LABELS_CSV),     f"labels csv missing: {LABELS_CSV}"
assert os.path.isfile(SYNSET_MAP),     f"synset map missing: {SYNSET_MAP}"

# --- load full synset->name map, then slice to our targets ---
synset_full = {}
with open(SYNSET_MAP, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            syn_id, name = parts[0], parts[1].split(",")[0]
            synset_full[syn_id] = name

syn_names = {s: synset_full.get(s, "?") for s in SYNSETS}
missing = SYNSETS - set(synset_full.keys())

print("Target synsets:", ", ".join(sorted(SYNSETS)))
print("Names:", ", ".join(f"{s}→{syn_names.get(s, '?')}" for s in sorted(SYNSETS)))
if missing:
    print("⚠ Missing in mapping:", sorted(missing))

# --- load labels csv & detect label column ---
df = pd.read_csv(LABELS_CSV)

# prefer common column names; fall back to "the first non-ImageId column" if needed
cands = ["PredictionString", "Label", "Labels", "GroundTruth", "ground_truth", "target", "synset"]
label_col = None
for c in cands:
    if c in df.columns:
        label_col = c
        break
if label_col is None:
    other_cols = [c for c in df.columns if c.lower() != "imageid"]
    if len(other_cols) == 1:
        label_col = other_cols[0]
    else:
        raise ValueError(f"Could not find a label column. Columns: {list(df.columns)}")

assert "ImageId" in df.columns, "CSV must contain an 'ImageId' column"

df = df[["ImageId", label_col]].copy()
# robust synset extraction: take the first token if space-separated; handle NaNs
df[label_col] = df[label_col].astype(str).fillna("")
df["Synset"] = df[label_col].str.strip().str.split().str[0]

before = len(df)
df = df.drop_duplicates("ImageId")
print(f"Rows: {before} → {len(df)} unique ImageId")

# --- filter by our hardcoded synsets ---
df = df[df["Synset"].isin(SYNSETS)].copy()
print(f"Matched images (by synset): {len(df)}")

# --- build image paths (+ small fallback for extension case) ---
def first_existing_path(image_id: str):
    # primary Imagenet val extension is '.JPEG'
    p1 = os.path.join(VAL_DIR, f"{image_id}.JPEG")
    if os.path.exists(p1): 
        return p1
    # occasional alt extensions
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(VAL_DIR, f"{image_id}{ext}")
        if os.path.exists(p):
            return p
    return p1  # default (even if missing), so we can log it

df["ImagePath"] = df["ImageId"].apply(first_existing_path)
exists = df["ImagePath"].apply(os.path.exists)
print(f"On disk: {int(exists.sum())} exist | {int((~exists).sum())} missing")

# --- final selection list (for downstream use) ---
selected_images = [
    {"image_id": iid, "image_path": path, "synset": syn, "label": syn_names.get(syn, syn)}
    for iid, syn, path, ok in zip(df["ImageId"], df["Synset"], df["ImagePath"], exists)
    if ok
]

print(f"Selected files: {len(selected_images)}")
for i, it in enumerate(selected_images[:5], 1):
    print(f"  {i}. {it['image_id']} → {os.path.basename(it['image_path'])} ({it['label']})")

# (optional) actually load them now
# images = [Image.open(d["image_path"]).convert("RGB") for d in selected_images]
# print(f"Loaded {len(images)} images.)")



# === Visualize first 20 selected dog images ===
import math
from PIL import Image
import matplotlib.pyplot as plt

assert len(selected_images) > 0, "selected_images is empty"

N = min(20, len(selected_images))
rows = math.ceil(N / 5)
cols = min(5, N)

plt.figure(figsize=(4*cols, 3.5*rows))

for i, item in enumerate(selected_images[:N], 1):
    path  = item["image_path"]
    label = item.get("label", item.get("synset", ""))
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[skip] {path}: {e}")
        continue

    ax = plt.subplot(rows, cols, i)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"{label}\n{item['image_id']}", fontsize=10)

plt.tight_layout()
plt.show()
print(f"Displayed {N} images.")



# === STEP 1: Mine Conv3 feature maps + capture pool switches (pool1, pool2) ===
import time, traceback
from collections import OrderedDict
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet, AlexNet_Weights

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights   = AlexNet_Weights.DEFAULT
transform = weights.transforms()  # Resize(256) → CenterCrop(224) → ToTensor → Normalize

# if model not defined yet, create it
try:
    model
except NameError:
    model = alexnet(weights=weights).to(device).eval()
else:
    model.eval()

# --- constants for AlexNet feature indices ---
CONV1_IDX, POOL1_IDX = 0, 2
CONV2_IDX, POOL2_IDX = 3, 5
CONV3_IDX           = 6  # <- TARGET
# (POOL5 at 12 is not needed when reconstructing from conv3 down to input)

LAYER_NAME = "conv3"     # storage key

class ExactFeatureExtractor:
    """
    Extracts:
      - Conv3 feature map (pre-ReLU output of features[6])
      - MaxPool switches for pool1 (features[2]) and pool2 (features[5])
    Notes:
      * Recomputes max_pool2d with return_indices=True inside the hook so we
        don't have to rebuild the model with return_indices=True.
      * All tensors are cloned to CPU to save VRAM.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.feature_maps = OrderedDict()
        self.pooling_indices = OrderedDict()
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def hook_pool(layer_name):
            def fn(module, inputs, output):
                x = inputs[0]  # tensor BEFORE this pooling layer (i.e., after ReLU)
                # recompute with identical hparams to get indices
                y, idx = F.max_pool2d(
                    x,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    return_indices=True
                )
                self.pooling_indices[layer_name] = {
                    "indices": idx.detach().cpu().clone(),   # [N,C,Hout,Wout]
                    "out_shape": tuple(y.shape),             # (N,C,Hout,Wout)
                    "kernel": module.kernel_size,
                    "stride": module.stride,
                    "pad": module.padding,
                }
            return fn

        def hook_conv3():
            def fn(module, inputs, output):
                # store conv3 feature map (pre-ReLU); shape [N,384,13,13]
                self.feature_maps["conv3"] = output.detach().cpu().clone()
            return fn

        # attach only what we need
        self.hooks.append(self.model.features[POOL1_IDX].register_forward_hook(hook_pool("pool1")))
        self.hooks.append(self.model.features[POOL2_IDX].register_forward_hook(hook_pool("pool2")))
        self.hooks.append(self.model.features[CONV3_IDX].register_forward_hook(hook_conv3()))
        print("Registered hooks for: pool1, pool2, conv3")

    def extract_features(self, x1: torch.Tensor):
        self.feature_maps.clear()
        self.pooling_indices.clear()
        with torch.no_grad():
            _ = self.model(x1)
        # sanity
        fmap = self.feature_maps.get("conv3", None)
        assert fmap is not None, "conv3 feature map not captured"
        assert fmap.ndim == 4 and fmap.shape[1] == 384 and fmap.shape[2:] == (13, 13), \
            f"Unexpected conv3 fmap shape: {tuple(fmap.shape)} (expected [N,384,13,13])"
        assert "pool1" in self.pooling_indices and "pool2" in self.pooling_indices, "pool switches missing"
        return (self.feature_maps.copy(), self.pooling_indices.copy())

    def cleanup(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

class ProgressiveProcessor:
    """Processes the selected images and stores Conv3 maps + pool switches with progress."""
    def __init__(self, model, feature_extractor: ExactFeatureExtractor):
        self.model = model
        self.feature_extractor = feature_extractor
        self.validation_data = {}

    def process_images_with_progress(self, selected_images, layer_name=LAYER_NAME):
        n = len(selected_images)
        print(f"\n=== STEP 1: PROCESSING {n} IMAGES FOR {layer_name.upper()} ===")

        store = {
            "feature_maps": [],     # list of torch.Tensor [1,384,13,13] on CPU
            "pool_switches": [],    # list of dicts: {"pool1": {...}, "pool2": {...}}
            "image_paths": [],
            "image_info": [],
            "processing_times": [],
        }
        self.validation_data[layer_name] = store

        t0 = time.time()
        ok = 0
        for i, info in enumerate(selected_images):
            s = time.time()
            img_path = info["image_path"]
            img_id = info["image_id"]
            print(f"[{i+1}/{n}] {img_id} ({info['label']})", end=" ")

            try:
                img = Image.open(img_path).convert("RGB")
                x = transform(img).unsqueeze(0).to(device, non_blocking=True)

                fmaps, pools = self.feature_extractor.extract_features(x)

                # keep only what we need, on CPU
                store["feature_maps"].append(fmaps["conv3"])  # already CPU
                store["pool_switches"].append({
                    "pool1": pools["pool1"],
                    "pool2": pools["pool2"],
                })
                store["image_paths"].append(img_path)
                store["image_info"].append(info)

                dt = time.time() - s
                store["processing_times"].append(dt)
                ok += 1
                print(f"✓ ({dt:.1f}s)")

                if (i+1) % 10 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed/(i+1) * (n - (i+1))
                    print(f"  Progress: {ok}/{n} ({100*ok/n:.0f}%) | ETA: {eta:.0f}s")

            except Exception as e:
                print(f"✗ ERROR: {e}")
                traceback.print_exc(limit=1)

        total = time.time() - t0
        avg = total/ok if ok else 0.0
        print("\n=== STEP 1 COMPLETE ===")
        print(f"Processed: {ok}/{n} images | Total: {total:.1f}s | Avg: {avg:.2f}s/img")

        if ok:
            shp = store["feature_maps"][0].shape
            p1s = store["pool_switches"][0]["pool1"]["indices"].shape
            p2s = store["pool_switches"][0]["pool2"]["indices"].shape
            print(f"Conv3 fmap sample: {tuple(shp)} (expect (1,384,13,13))")
            print(f"Pool1 idx sample:  {tuple(p1s)} (expect (1,64,27,27))")
            print(f"Pool2 idx sample:  {tuple(p2s)} (expect (1,192,13,13))")

        return ok

# ---- Initialize / reset extractor (avoid stale hooks) ----
print("Initializing feature extractor for Conv3...")
try:
    feature_extractor.cleanup()
except:
    pass
feature_extractor = ExactFeatureExtractor(model)

# ---- Run STEP 1 ----
progressive_processor = ProgressiveProcessor(model, feature_extractor)
num_processed = progressive_processor.process_images_with_progress(selected_images, LAYER_NAME)

# ---- Quick peek at what we stored ----
vd = progressive_processor.validation_data[LAYER_NAME]
print("Stored:", len(vd["feature_maps"]), "feature maps,", len(vd["pool_switches"]), "pool-switch blobs")
print("Sample fmap dtype/shape:", vd["feature_maps"][0].dtype, vd["feature_maps"][0].shape)

print(f"\nREADY FOR STEP 2: {num_processed} images processed for {LAYER_NAME} analysis")



# 1) feature-map shape for step-1 storage
print(vd['feature_maps'][0].shape)   # should be torch.Size([1, 384, 13, 13])

# 2) we are targeting conv3 indices
print(model.features[6])             # conv3 layer

# 3) weights really tied
print((projector.deconv3.weight - model.features[6].weight).abs().max().item())
# should be 0.0



# ==== STEP 2: FIND GLOBALLY STRONGEST ACTIVATIONS (Conv3) — PER-IMAGE TOP-K ====
# Mines top-K *images* per channel (unique ImageId), and for each image records the
# strongest (y,x) location inside that image. Keeps the same downstream structure.

import torch
from typing import Dict, List

class ActivationAnalyzer:
    """Step 2: Mine top-K activations per channel (and rank strongest channels)."""
    def __init__(self, progressive_processor):
        self.processor = progressive_processor
        self.results: Dict[str, Dict] = {}  # layer_name -> dict with per_channel_topk, shapes, etc.

    def find_strongest_activations(
        self,
        layer_name: str = "conv3",
        topk: int = 9,
        post_relu: bool = True,
        mode: str = "per_image",  # "per_image" (unique images) | "per_pixel" (old behavior)
        verbose: bool = True,
    ):
        """
        Returns: dict[channel_idx] -> list of top-K hits:
            {image_idx, y, x, value, image_path, image_info, channel, rank}
        Notes:
            * mode="per_image" (default): for each channel, take the max over (y,x) per image,
              then pick the top-K images. This enforces unique ImageId per channel.
            * mode="per_pixel": original behavior (may repeat the same image several times).
        """
        if layer_name not in self.processor.validation_data:
            print(f"No data found for {layer_name}")
            return {}

        data = self.processor.validation_data[layer_name]
        fmaps: List[torch.Tensor] = data["feature_maps"]  # each [1,384,13,13] on CPU
        N = len(fmaps)
        assert N > 0, "No feature maps available from Step 1"

        # [N,384,13,13] (drop per-sample batch dim)
        F = torch.stack([fm[0] for fm in fmaps], dim=0)
        assert F.shape[1] == 384 and F.shape[2:] == (13, 13), f"Unexpected Conv3 shape: {tuple(F.shape)}"
        if post_relu:
            F = torch.clamp(F, min=0)

        N, C, H, W = F.shape
        if verbose:
            print(f"\n=== STEP 2: FINDING STRONGEST ACTIVATIONS IN {layer_name.upper()} ===")
            print(f"Tensor: {N} images × {C} channels × {H}×{W} = {N*C*H*W} positions")
            print(f"Post-ReLU: {post_relu} | Mode: {mode} | Top-K per channel: K={topk}")

        per_channel_topk: Dict[int, List[dict]] = {}

        if mode == "per_image":
            # For each (image, channel), take the max over spatial positions
            F_flat = F.view(N, C, -1)                       # [N,C,169]
            vmax, arg_flat = F_flat.max(dim=2)              # [N,C], [N,C] (arg in [0..168])
            # For each channel, pick top-K images by vmax
            # Transpose so we can topk per-channel (rows = channels)
            vmax_ch = vmax.transpose(0, 1).contiguous()     # [C,N]
            K = min(topk, N)
            vals, img_idxs = torch.topk(vmax_ch, k=K, dim=1, largest=True, sorted=True)  # [C,K], [C,K]

            for ch in range(C):
                hits: List[dict] = []
                for k in range(K):
                    img_idx = int(img_idxs[ch, k].item())
                    v = float(vals[ch, k].item())
                    flat = int(arg_flat[img_idx, ch].item())
                    y = flat // W
                    x = flat % W
                    hits.append({
                        "channel": ch,
                        "rank": k + 1,
                        "value": v,
                        "image_idx": img_idx,
                        "y": int(y),
                        "x": int(x),
                        "image_path": data["image_paths"][img_idx],
                        "image_info": data["image_info"][img_idx],
                    })
                per_channel_topk[ch] = hits
                if verbose and (ch % 64 == 0):
                    h0 = hits[0]
                    print(f"  ch {ch:3d} top={h0['value']:.3f} @ img#{h0['image_idx']} ({h0['y']},{h0['x']})")

        elif mode == "per_pixel":
            # Original behavior: flatten over all images & spatial positions per channel
            F_ch = F.permute(1, 0, 2, 3).contiguous().view(C, -1)  # [C, N*169]
            K = min(topk, F_ch.shape[1])
            vals, idxs = torch.topk(F_ch, k=K, dim=1, largest=True, sorted=True)  # [C,K]
            HW = H * W
            for ch in range(C):
                hits: List[dict] = []
                vrow, irow = vals[ch], idxs[ch]
                for k in range(K):
                    flat = int(irow[k].item())   # 0..(N*H*W-1)
                    img_idx = flat // HW
                    pos = flat % HW
                    y = int(pos // W)
                    x = int(pos % W)
                    hits.append({
                        "channel": ch,
                        "rank": k + 1,
                        "value": float(vrow[k].item()),
                        "image_idx": img_idx,
                        "y": y,
                        "x": x,
                        "image_path": data["image_paths"][img_idx],
                        "image_info": data["image_info"][img_idx],
                    })
                per_channel_topk[ch] = hits
                if verbose and (ch % 64 == 0):
                    h0 = hits[0]
                    print(f"  ch {ch:3d} top={h0['value']:.3f} @ img#{h0['image_idx']} ({h0['y']},{h0['x']})")
        else:
            raise ValueError("mode must be 'per_image' or 'per_pixel'")

        # strongest channel ranking by their top-1
        strongest = [
            {"channel": ch, "max_value": per_channel_topk[ch][0]["value"], "top_hit": per_channel_topk[ch][0]}
            for ch in range(C)
        ]
        strongest.sort(key=lambda d: d["max_value"], reverse=True)

        self.results[layer_name] = {
            "F_shape": tuple(F.shape),
            "per_channel_topk": per_channel_topk,
            "strongest_channels": strongest,
            "topk": topk,
            "post_relu": post_relu,
            "mode": mode,
        }

        if verbose:
            print(f"Analysis complete for {C} channels. Example top channel: "
                  f"ch {strongest[0]['channel']} → {strongest[0]['max_value']:.3f}")

        return per_channel_topk

    def get_top_channels(self, layer_name: str = "conv3", top_n: int = 5):
        """Return the top-N channels ranked by their top-1 activation value."""
        if layer_name not in self.results:
            print(f"No analysis found for {layer_name}. Run find_strongest_activations first.")
            return []
        strongest = self.results[layer_name]["strongest_channels"]
        out = strongest[:top_n]
        print(f"\nTop {len(out)} strongest channels in {layer_name}:")
        for i, row in enumerate(out, 1):
            ch = row["channel"]; mv = row["max_value"]
            lbl = row["top_hit"]["image_info"]["label"]
            print(f"  {i}. ch {ch:3d}: {mv:.3f}  (strongest in {lbl})")
        return out


# --- Execute Step 2 (per-image unique) ---
print("Starting activation analysis for Conv3…")
analyzer = ActivationAnalyzer(progressive_processor)

# Mine top-K per channel using per-image max (unique ImageId per channel)
channel_topk = analyzer.find_strongest_activations('conv3', topk=9, post_relu=True, mode="per_image", verbose=True)

# Rank strongest channels overall
top_channels = analyzer.get_top_channels('conv3', top_n=5)

print(f"\nStep 2 Complete: per-channel topK mined for {len(channel_topk)} channels")
print("Ready for Step 3: reconstruct top-9 per chosen channel")



# ==== STEP 3: SELECT TOP 2 STRONGEST CHANNELS AND THEIR TOP 9 IMAGES ====

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle

SHOW_RF_BOX = True  # set False to disable

# use the same 224x224 model crop so RF math matches
try:
    _display_tf
except NameError:
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
    _display_tf = Compose([Resize(256), CenterCrop(224), ToTensor()])

class TopChannelSelector:
    """Step 3: Pick the strongest channels and grab their top-N hits (images & positions)."""
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.selected_channels_data = {}

    def find_and_select_top_channels(self, layer_name='conv3', num_channels=2, top_images=9):
        """
        Uses analyzer.get_top_channels() and analyzer.results[layer]['per_channel_topk'].
        Stores for each chosen channel:
          - rank, channel_idx, max_activation
          - top_hits: list[dict] (≤ top_images) with keys: image_idx, y, x, value, image_path, image_info
          - all_hits: full per-channel list from Step 2
        """
        if layer_name not in self.analyzer.results:
            print(f"No analysis found for {layer_name}. Run Step 2 first.")
            return {}

        print(f"\n=== STEP 3: FINDING TOP {num_channels} STRONGEST CHANNELS ({layer_name}) ===")
        strongest = self.analyzer.get_top_channels(layer_name, top_n=num_channels)
        per_ch = self.analyzer.results[layer_name]["per_channel_topk"]

        selected = {}
        for rank, row in enumerate(strongest, 1):
            ch = row["channel"]
            max_val = row["max_value"]
            all_hits = per_ch[ch]
            top_hits = all_hits[:top_images]

            selected[ch] = {
                "rank": rank,
                "channel_idx": ch,
                "max_activation": max_val,
                "top_hits": top_hits,
                "all_hits": all_hits,
            }

            print(f"\nChannel {ch} (Rank #{rank})")
            print(f"  Max activation: {max_val:.3f}")
            print(f"  Top {len(top_hits)} images:")
            for i, hit in enumerate(top_hits, 1):
                lbl = hit["image_info"]["label"]
                loc = (hit["y"], hit["x"])
                print(f"    {i}. {lbl:25s}  act={hit['value']:.3f}  loc={loc}")

        self.selected_channels_data[layer_name] = selected
        return selected

    def visualize_top_channels(self, layer_name='conv3', save_prefix="conv3_channel"):
        """Show 3×3 grids of the original images that gave top activations per chosen channel."""
        if layer_name not in self.selected_channels_data:
            print(f"No selected channels data for {layer_name}.")
            return

        selected = self.selected_channels_data[layer_name]
        print(f"\n=== VISUALIZING TOP {len(selected)} CHANNELS ({layer_name}) ===")

        for ch, data in selected.items():
            rank = data["rank"]
            max_activation = data["max_activation"]
            hits = data["top_hits"]

            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            fig.suptitle(
                f'{layer_name.upper()} Channel {ch} (Rank #{rank}) — Top 9 Strongest Activations\n'
                f'Max Activation: {max_activation:.3f}',
                fontsize=16, y=0.98
            )

            for i in range(9):
                row_idx, col_idx = divmod(i, 3)
                ax = axes[row_idx, col_idx]
                if i < len(hits):
                    hit = hits[i]
                    try:
                        img = Image.open(hit["image_path"]).convert("RGB")
                        img224 = _display_tf(img).permute(1, 2, 0).numpy()  # [H,W,C] in [0,1]
                        ax.imshow(img224)

                        if SHOW_RF_BOX:
                            # conv3 RF mapping (same constants as Step 4)
                            RF_SIZE, STRIDE_TO_INPUT, START = 99, 16, 7
                            y, x = hit["y"], hit["x"]
                            cy, cx = START + y * STRIDE_TO_INPUT, START + x * STRIDE_TO_INPUT
                            rf_radius = RF_SIZE // 2  # <- renamed to avoid shadowing row_idx
                            y0, y1 = max(0, cy - rf_radius), min(224, cy + rf_radius + (RF_SIZE % 2))
                            x0, x1 = max(0, cx - rf_radius), min(224, cx + rf_radius + (RF_SIZE % 2))
                            rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                                             fill=False, linewidth=2, edgecolor="red")
                            ax.add_patch(rect)

                    except Exception as e:
                        ax.text(0.5, 0.5, f'Error\n{e}', ha='center', va='center', fontsize=10)
                    ax.axis("off")
                else:
                    ax.axis("off")

            plt.tight_layout()
            plt.subplots_adjust(top=0.93)
            save_path = f"{save_prefix}_{ch}_rank{rank}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"  ✓ Saved visualization to {save_path}")
            plt.show()

    def get_summary(self, layer_name='conv3'):
        """Print a tiny summary per selected channel."""
        if layer_name not in self.selected_channels_data:
            print(f"No data for {layer_name}")
            return

        selected = self.selected_channels_data[layer_name]
        print(f"\n=== SUMMARY: TOP {len(selected)} CHANNELS ({layer_name}) ===")
        for ch, data in selected.items():
            hits = data["top_hits"]
            if not hits:
                continue
            max_act = data["max_activation"]
            min_top = hits[-1]["value"]
            print(f"\nChannel {ch} (Rank #{data['rank']}):")
            print(f"  Max activation: {max_act:.3f}")
            print(f"  Activation range (top9): {min_top:.3f} → {max_act:.3f}")
            # simple label distro
            distro = {}
            for h in hits:
                lbl = h["image_info"]["label"]
                distro[lbl] = distro.get(lbl, 0) + 1
            print("  Label distribution:")
            for lbl, cnt in sorted(distro.items(), key=lambda x: x[1], reverse=True):
                print(f"    {lbl}: {cnt}")

    def save_channel_data(self, layer_name='conv3', filename="selected_channels_data.txt"):
        """Save detailed channel data to file (robust against copy/paste breaks)."""
        if layer_name not in self.selected_channels_data:
            print(f"No data for {layer_name}")
            return

        selected = self.selected_channels_data[layer_name]
        with open(filename, "w", encoding="utf-8") as f:
            header = f"TOP {len(selected)} STRONGEST CHANNELS — {layer_name.upper()}\n"
            f.write(header)
            f.write("=" * 64 + "\n\n")

            for ch, data in selected.items():
                rank = data["rank"]
                max_act = data["max_activation"]
                f.write(f"CHANNEL {ch} (RANK #{rank})\n")
                f.write(f"Max Activation: {max_act:.3f}\n\n")
                f.write("Top 9 Images:\n")

                for i, h in enumerate(data["top_hits"], 1):
                    image_id = h["image_info"]["image_id"]
                    label    = h["image_info"]["label"]
                    value    = h["value"]
                    y, x     = h["y"], h["x"]
                    line = (
                        f"  {i:2d}. {image_id:<25s} | {label:<25s} | "
                        f"Act: {value:6.3f} | Loc: ({y},{x})\n"
                    )
                    f.write(line)

                f.write("\n" + "-" * 64 + "\n\n")

        print(f"✓ Saved detailed channel data to {filename}")



22/3


2*2


# === EXECUTE STEP 3 (Conv3) ===
# assumes STEP 2 already ran and filled analyzer.results['conv3']

# If needed, (re)mine activations first:
if 'conv3' not in getattr(analyzer, 'results', {}):
    analyzer.find_strongest_activations('conv3', topk=9, verbose=True)

# Re-init to make sure we’re using the patched class definition
selector = TopChannelSelector(analyzer)

# 1) pick strongest channels & their top-9 hits
selected_channels = selector.find_and_select_top_channels('conv3', num_channels=2, top_images=9)

# 2) visualize the 3×3 grids of originals
selector.visualize_top_channels('conv3', save_prefix="conv3_top_channel")

# 3) tiny text summary
selector.get_summary('conv3')

# 4) write details to disk
selector.save_channel_data('conv3', "top_2_conv3_channels_analysis.txt")

# (optional) sanity: confirm file saved
import os
print("Saved file exists:", os.path.isfile("top_2_conv3_channels_analysis.txt"))



# ==== VISIBILITY BOOST UTILS (add to STEP 4) ====
import numpy as np
import matplotlib.cm as cm

def norm01_percentile(t: torch.Tensor, lo=1.0, hi=99.5, eps=1e-8):
    """Robust [0,1] scaling by percentiles."""
    flat = t.flatten()
    lo_v = torch.quantile(flat, lo/100.0)
    hi_v = torch.quantile(flat, hi/100.0)
    x = torch.clamp(t - lo_v, 0)
    x = x / max(hi_v - lo_v, eps)
    return torch.clamp(x, 0, 1)

def boost_rgb(recon_chw: torch.Tensor, p_hi=99.5, gamma=0.6, sat=1.6, per_channel=True):
    """
    recon_chw: [3,H,W] in [0,1] (your projector output)
    Stronger contrast + color pop.
    """
    x = recon_chw.clone()
    if per_channel:
        for c in range(3):
            x[c] = norm01_percentile(x[c], lo=1.0, hi=p_hi)
    else:
        x = norm01_percentile(x, lo=1.0, hi=p_hi)

    # simple saturation boost: move away from per-pixel mean
    npimg = x.numpy().transpose(1,2,0)
    m = npimg.mean(axis=2, keepdims=True)
    npimg = np.clip(m + sat*(npimg - m), 0, 1)
    # gamma for mid-tone lift
    npimg = np.clip(npimg, 0, 1) ** gamma
    return npimg  # HWC [0,1]

def recon_to_heatmap(recon_chw: torch.Tensor, p_hi=99.5, gamma=0.6, cmap_name="magma"):
    """
    Turn 3ch recon into intensity heatmap (energy), robust normalized.
    """
    E = torch.sqrt((recon_chw ** 2).sum(0))          # [H,W]
    E = norm01_percentile(E, lo=1.0, hi=p_hi) ** gamma
    cmap = cm.get_cmap(cmap_name)
    hm = cmap(E.numpy())[:, :, :3]                   # HWC, drop alpha
    return hm

# ---- tweak your panel builder to use these (replace the plotting part) ----
# Add a knob up top:
VIS_MODE = "rgb_boost"           # "rgb_boost" | "heatmap" | "heatmap_overlay"
HEAT_ALPHA = 0.40                # blend for overlay

def build_panel_for_channel(ch, analyzer, progressive_processor, layer_name='conv3', topk=9, save_path=None):
    per_ch = analyzer.results[layer_name]["per_channel_topk"]
    vd = progressive_processor.validation_data[layer_name]
    hits = per_ch[ch][:min(topk, len(per_ch[ch]))]

    fig = plt.figure(figsize=(2.6*len(hits), 5.4))
    plt.suptitle(f"{layer_name.upper()} ch {ch} — top-{len(hits)} reconstructions", y=0.98, fontsize=14)

    for i, hit in enumerate(hits):
        img_idx = hit["image_idx"]
        y, x    = hit["y"], hit["x"]
        val     = hit["value"]
        path    = hit["image_path"]

        sparse = make_sparse_conv3(ch, y, x, val)
        pool_switches = vd["pool_switches"][img_idx]
        recon = projector(sparse, pool_switches)              # [3,224,224] in [0,1]

        # --- choose visualization style for TOP row ---
        if VIS_MODE == "rgb_boost":
            vis_top = boost_rgb(recon, p_hi=99.5, gamma=0.6, sat=1.6, per_channel=True)
        else:
            heat = recon_to_heatmap(recon, p_hi=99.0, gamma=0.7, cmap_name="magma")
            if VIS_MODE == "heatmap":
                vis_top = heat
            elif VIS_MODE == "heatmap_overlay":
                _, patch = display_crop(path, y, x)           # 224x224 RF patch in [0,1]
                vis_top = np.clip((1-HEAT_ALPHA)*patch + HEAT_ALPHA*heat, 0, 1)
            else:
                vis_top = boost_rgb(recon)  # fallback

        # bottom row still the RF patch (you can also overlay here if you want)
        _, patch = display_crop(path, y, x)

        # --- plot ---
        ax = plt.subplot(2, len(hits), 1 + i)
        ax.imshow(vis_top)
        ax.set_title(f"#{i+1} val={val:.2f} @({y},{x})", fontsize=9)
        ax.axis("off")

        ax = plt.subplot(2, len(hits), len(hits) + 1 + i)
        ax.imshow(patch)
        ax.set_title(vd["image_info"][img_idx]["label"], fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    if save_path is None:
        save_path = f"conv3_ch{ch}_recons_top{len(hits)}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print("✓ Saved:", save_path)



# ==== STEP 4: RECONSTRUCT (deconv) TOP-9 HITS FROM CONV3 BACK TO INPUT (paper-ish, neutral gray) ====
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor

# --- assume `model`, `device`, `analyzer`, `progressive_processor` already exist ---
try:
    model
    device
except NameError:
    from torchvision.models import alexnet, AlexNet_Weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = alexnet(weights=AlexNet_Weights.DEFAULT).to(device).eval()

# --- Display pipeline (same spatial crop as the model, but no normalization) ---
_display_tf = Compose([Resize(256), CenterCrop(224), ToTensor()])

# --- Conv3 receptive field mapper (AlexNet): stride_to_input=16, RF=99 ---
RF_SIZE, STRIDE_TO_INPUT, START = 99, 16, 7
def conv3_rf_box(y, x, H=224, W=224, rf=RF_SIZE, jump=STRIDE_TO_INPUT, start=START):
    cy = start + y*jump
    cx = start + x*jump
    r  = rf // 2
    y0, y1 = max(0, cy - r), min(H, cy + r + (rf % 2))
    x0, x1 = max(0, cx - r), min(W, cx + r + (rf % 2))
    return int(y0), int(y1), int(x0), int(x1)

# ---- RF crop helper (torch-only; avoids PIL float issues) ----
def _safe_box(y0, y1, x0, x1, H=224, W=224, min_hw=2):
    if (y1 - y0) < min_hw:
        cy = (y0 + y1) // 2
        y0 = max(0, cy - 1); y1 = min(H, y0 + min_hw)
    if (x1 - x0) < min_hw:
        cx = (x0 + x1) // 2
        x0 = max(0, cx - 1); x1 = min(W, x0 + min_hw)
    return y0, y1, x0, x1

def crop_rf_from_recon(recon_chw: torch.Tensor, y: int, x: int) -> torch.Tensor:
    """Crop [3,224,224] to conv3 RF around (y,x) and upscale back to 224."""
    y0, y1, x0, x1 = conv3_rf_box(y, x, 224, 224)
    y0, y1, x0, x1 = _safe_box(y0, y1, x0, x1, 224, 224, min_hw=2)
    crop = recon_chw[:, y0:y1, x0:x1].unsqueeze(0)  # [1,3,h,w]
    up   = F.interpolate(crop, size=(224,224), mode="bilinear", align_corners=False)
    return up[0]  # [3,224,224]

# --- light, per-channel percentile contrast + neutral gray background ---
def _perc01(ch: torch.Tensor, lo=1.0, hi=99.0, eps=1e-8):
    ql = torch.quantile(ch.flatten(), lo/100.0)
    qh = torch.quantile(ch.flatten(), hi/100.0)
    x  = (ch - ql).clamp(min=0) / max((qh - ql).item(), eps)
    return x.clamp(0,1)

def boost01_chw(x_chw: torch.Tensor, lo=1.0, hi=99.0, gamma=0.9):
    y = x_chw.clone()
    for c in range(min(3, y.shape[0])):
        y[c] = _perc01(y[c], lo=lo, hi=hi)
    y = y.clamp(0,1) ** gamma
    return y.numpy().transpose(1,2,0)  # HWC

def neutral_gray_hwc(img_hwc: np.ndarray, target: float = 0.5) -> np.ndarray:
    """
    Shift per-channel means to neutral gray (0.5) to remove any color cast in the background.
    """
    m = img_hwc.mean(axis=(0,1), keepdims=True)  # [1,1,3]
    return np.clip(img_hwc - (m - target), 0.0, 1.0)

# --- Build mirrored deconv tail for Conv3 -> input ---
class DeconvProjectorConv3(nn.Module):
    """
    conv3T -> ReLU -> unpool2 -> conv2T -> ReLU -> unpool1 -> conv1T
    Uses real pool switches captured per-image in Step 1 (pool2, pool1).
    """
    def __init__(self, model):
        super().__init__()
        conv1 = model.features[0]
        conv2 = model.features[3]
        conv3 = model.features[6]

        self.deconv3 = nn.ConvTranspose2d(384, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(192, 64,  kernel_size=5, stride=1, padding=2, bias=False)
        # (55-1)*4 - 2*2 + 11 + 1 = 224
        self.deconv1 = nn.ConvTranspose2d(64, 3,    kernel_size=11, stride=4, padding=2, output_padding=1, bias=False)

        with torch.no_grad():
            self.deconv3.weight.copy_(conv3.weight)
            self.deconv2.weight.copy_(conv2.weight)
            self.deconv1.weight.copy_(conv1.weight)

        self.unpool2 = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=0)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=0)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, sparse_conv3, pool_switches):
        x = self.deconv3(sparse_conv3); x = self.relu(x)               # -> [1,192,13,13]
        p2 = pool_switches["pool2"]
        x = self.unpool2(x.to(device), p2["indices"].to(device))       # -> [1,192,27,27]
        x = self.deconv2(x); x = self.relu(x)                          # -> [1,64,27,27]
        p1 = pool_switches["pool1"]
        x = self.unpool1(x, p1["indices"].to(device))                  # -> [1,64,55,55]
        x = self.deconv1(x)                                            # -> [1,3,224,224]
        # simple [0,1] norm for display stability
        x = x.detach().cpu()[0]
        x = x - x.min()
        if x.max() > 0: x = x / x.max()
        return x  # [3,224,224] in [0,1]

projector = DeconvProjectorConv3(model).to(device).eval()

# --- Helpers (kept for self-contained Step 4) ---
def make_sparse_conv3(ch, y, x, val, H=13, W=13, C=384):
    s = torch.zeros((1, C, H, W), dtype=torch.float32, device=device)
    s[0, ch, y, x] = float(val)
    return s

def to_hwc01(t):
    return t.numpy().transpose(1,2,0)  # CHW -> HWC

def display_crop(img_path, y, x):
    pil = Image.open(img_path).convert("RGB")
    disp = _display_tf(pil)  # [3,224,224] in [0,1]
    npimg = disp.numpy().transpose(1,2,0)
    y0,y1,x0,x1 = conv3_rf_box(y, x, 224, 224)
    patch = npimg[y0:y1, x0:x1, :]
    patch = np.array(Image.fromarray((patch*255).astype(np.uint8)).resize((224,224), Image.NEAREST))/255.0
    return npimg, patch

# --- Panel builder: RF-cropped recon (top), RF input patch (bottom), neutral gray ---
def build_panel_for_channel(ch, analyzer, progressive_processor, layer_name='conv3', topk=9, save_path=None):
    per_ch = analyzer.results[layer_name]["per_channel_topk"]
    vd = progressive_processor.validation_data[layer_name]
    hits = per_ch[ch][:min(topk, len(per_ch[ch]))]

    fig = plt.figure(figsize=(2.6*len(hits), 5.4))
    plt.suptitle(f"{layer_name.upper()} ch {ch} — top-{len(hits)} reconstructions (RF-cropped)", y=0.98, fontsize=14)

    for i, hit in enumerate(hits):
        img_idx = hit["image_idx"]
        y, x    = hit["y"], hit["x"]
        val     = hit["value"]
        path    = hit["image_path"]

        # sparse seed at conv3
        sparse = make_sparse_conv3(ch, y, x, val)

        # reconstruct with that IMAGE'S switches
        pool_switches = vd["pool_switches"][img_idx]
        recon = projector(sparse, pool_switches)          # [3,224,224] in [0,1]

        # --- paper-ish: RF-crop + mild percentile contrast + neutral gray ---
        recon_crop = crop_rf_from_recon(recon, y, x)      # focus on motif
        recon_img  = boost01_chw(recon_crop, lo=1.0, hi=99.0, gamma=0.9)  # HWC [0,1]
        recon_img  = neutral_gray_hwc(recon_img, target=0.5)              # remove color cast

        # original RF patch (bottom row)
        _, patch = display_crop(path, y, x)

        # top row: recon
        ax = plt.subplot(2, len(hits), 1 + i)
        ax.imshow(recon_img)
        ax.set_title(f"#{i+1} val={val:.2f} @({y},{x})", fontsize=9)
        ax.axis("off")

        # bottom row: RF patch
        ax = plt.subplot(2, len(hits), len(hits) + 1 + i)
        ax.imshow(patch)
        ax.set_title(vd["image_info"][img_idx]["label"], fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    if save_path is None:
        save_path = f"conv3_ch{ch}_recons_top{len(hits)}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print("✓ Saved:", save_path)



# === EXECUTE STEP 4 ===
sel = selector.selected_channels_data['conv3']
for ch in list(sel.keys()):
    build_panel_for_channel(
        ch, analyzer, progressive_processor,
        layer_name='conv3', topk=9,
        save_path=f"conv3_ch{ch}_recons_top9.png"
    )



11/2


hit   = analyzer.results['conv3']['per_channel_topk'][ch][0]
fm    = progressive_processor.validation_data['conv3']['feature_maps'][hit['image_idx']]
assert fm.shape == (1, 384, 13, 13), "Not a conv3 fmap"
y,x,val = hit['y'], hit['x'], hit['value']
assert 0 <= y < 13 and 0 <= x < 13, "Coords not conv3 grid"
print("using projector:", type(projector).__name__)  # must be DeconvProjectorConv3



vd = progressive_processor.validation_data['conv3']
sw  = vd['pool_switches'][hit['image_idx']]
assert 'pool2' in sw and 'pool1' in sw, "missing switches"
print(sw['pool2']['indices'].shape, sw['pool1']['indices'].shape)  # expect (1,192,13,13) and (1,64,27,27)



# ==== STEP 5: Deconv-only mosaics for CONV3 (same viz modes as Layer 2) ====
# Reuses: projector(sparse, pool_switches), make_sparse_conv3, display_crop
# Input deps expected from Steps 1–4: analyzer, progressive_processor, selector

import torch, numpy as np, matplotlib.pyplot as plt
import matplotlib.cm as cm

# --- L3 seeding mode ---
USE_FULLMAP = True   # <— turn ON for texture-y L3 (whole channel map)
LOCAL_K     = 0      # try 3 or 5 to keep only a k×k neighborhood around (y,x)


# ---------- robust normalization + styles ----------
def _perc01(ch: torch.Tensor, lo=1.0, hi=99.0, eps=1e-8):
    ql = torch.quantile(ch.flatten(), lo/100.0)
    qh = torch.quantile(ch.flatten(), hi/100.0)
    x  = (ch - ql).clamp(min=0) / max((qh - ql).item(), eps)
    return x.clamp(0, 1)

def boost01_chw(x_chw: torch.Tensor, lo=1.0, hi=99.0, gamma=0.9):
    y = x_chw.clone()
    C = min(3, y.shape[0])
    for c in range(C):                      # per-channel robust contrast
        y[c] = _perc01(y[c], lo=lo, hi=hi)
    y = y.clamp(0, 1) ** gamma
    return y.numpy().transpose(1, 2, 0)     # HWC in [0,1]

def neutral_gray_hwc(img_hwc: np.ndarray, target: float = 0.5) -> np.ndarray:
    m = img_hwc.mean(axis=(0,1), keepdims=True)    # [1,1,3]
    return np.clip(img_hwc - (m - target), 0.0, 1.0)

def recon_to_heatmap(recon_chw: torch.Tensor, p_hi=99.0, gamma=0.7, cmap_name="magma"):
    # magnitude -> [0,1] with robust percentiles
    E = torch.sqrt((recon_chw ** 2).sum(0))            # [H,W]
    flat = E.flatten()
    lo_v = torch.quantile(flat, 0.01)
    hi_v = torch.quantile(flat, p_hi/100.0)
    Em = (E - lo_v).clamp(min=0) / max((hi_v - lo_v).item(), 1e-8)
    Em = Em.clamp(0, 1) ** gamma
    return cm.get_cmap(cmap_name)(Em.numpy())[:, :, :3]  # HWC

# ---------- knobs (same modes as Layer 2) ----------
VIS_MODE   = "rgb_boost"         # "rgb_boost" | "heatmap" | "heatmap_overlay"
HEAT_ALPHA = 0.40                # blend for overlay
RFCROP     = True                # focus on conv3 receptive field (paper-ish)
SAVE_PREFIX = "conv3"            # file name prefix

# auto bottom row: show RF patch unless we already overlay it on top
BOTTOM_ROW = (VIS_MODE != "heatmap_overlay")

# ---------- RF crop helper (use Step 4 if present; else fallback) ----------
try:
    crop_rf_from_recon
except NameError:
    def crop_rf_from_recon(recon_chw: torch.Tensor, y: int, x: int) -> torch.Tensor:
        # conv3 RF ≈99, stride ≈16, center offset ≈7
        cy, cx = 7 + 16*y, 7 + 16*x
        r = 99 // 2
        y0, y1 = max(0, cy - r), min(224, cy + r + (99 % 2))
        x0, x1 = max(0, cx - r), min(224, cx + r + (99 % 2))
        crop = recon_chw[:, y0:y1, x0:x1].unsqueeze(0)
        return torch.nn.functional.interpolate(crop, size=(224,224), mode="bilinear", align_corners=False)[0]

# ---------- main: export a panel for one channel ----------
def export_deconv_panel(ch, analyzer, progressive_processor,
                        layer_name='conv3', topk=9, save_path=None):
    """
    Panel for one conv3 channel with Layer-2 viz modes:
      - rgb_boost: boosted RGB deconv on neutral gray
      - heatmap: magnitude colormap
      - heatmap_overlay: heatmap blended over the RF input patch
    Top row = chosen viz; optional bottom row = RF input patch.
    """
    assert hasattr(analyzer, "results") and layer_name in analyzer.results, "Run Step 2 first."
    per_ch = analyzer.results[layer_name]["per_channel_topk"]
    vd     = progressive_processor.validation_data[layer_name]
    hits   = per_ch[ch][:min(topk, len(per_ch[ch]))]

    rows = 2 if BOTTOM_ROW else 1
    fig = plt.figure(figsize=(2.6*len(hits), 2.6*rows))
    plt.suptitle(f"{layer_name.upper()} ch {ch} — top-{len(hits)} deconvs ({VIS_MODE})",
                 y=0.98, fontsize=14)

    for i, h in enumerate(hits):
        img_idx, y, x, val = h["image_idx"], h["y"], h["x"], h["value"]
        path = h["image_path"]

        # 1) sparse seed at conv3 & reconstruct with THIS image's switches
        # sparse = make_sparse_conv3(ch, y, x, val)

        C, H, W = 384, 13, 13
        sparse = torch.zeros((1, C, H, W), dtype=torch.float32, device=device)
        
        if USE_FULLMAP:
            # keep the *entire* conv3 channel map for this image (ReLU’d)
            fmap = vd['feature_maps'][img_idx][0, ch].detach().to(device)  # [13,13]
            fmap = torch.nn.functional.relu(fmap)
        
            if LOCAL_K and LOCAL_K > 1:
                # optional: restrict to a LOCAL_K×LOCAL_K patch centered at the max location
                y0 = max(0, y - LOCAL_K//2); y1 = min(H, y + LOCAL_K//2 + 1)
                x0 = max(0, x - LOCAL_K//2); x1 = min(W, x + LOCAL_K//2 + 1)
                mask = torch.zeros_like(fmap); mask[y0:y1, x0:x1] = 1.0
                fmap = fmap * mask
        
            sparse[0, ch] = fmap
        else:
            # fallback: the old one-pixel seed
            sparse[0, ch, y, x] = float(val)

        pool_switches = vd["pool_switches"][img_idx]
        recon = projector(sparse, pool_switches)            # [3,224,224] in [0,1]

        # 2) RF crop (fills tile with motif)
        recon_crop = crop_rf_from_recon(recon, y, x) if RFCROP else recon

        # 3) choose viz
        if VIS_MODE == "rgb_boost":
            tile = boost01_chw(recon_crop, lo=1.0, hi=99.0, gamma=0.9)
            tile = neutral_gray_hwc(tile, target=0.5)
        elif VIS_MODE == "heatmap":
            tile = recon_to_heatmap(recon_crop, p_hi=99.0, gamma=0.7, cmap_name="magma")
        elif VIS_MODE == "heatmap_overlay":
            heat = recon_to_heatmap(recon_crop, p_hi=99.0, gamma=0.7, cmap_name="magma")
            _, patch = display_crop(path, y, x)             # [224,224,3] in [0,1]
            tile = np.clip((1-HEAT_ALPHA)*patch + HEAT_ALPHA*heat, 0, 1)
        else:
            tile = boost01_chw(recon_crop, lo=1.0, hi=99.0, gamma=0.9)
            tile = neutral_gray_hwc(tile, target=0.5)

        # 4) plot top row
        ax = plt.subplot(rows, len(hits), 1 + i)
        ax.imshow(tile)
        ax.set_title(f"#{i+1} val={val:.2f} @({y},{x})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")

        # 5) optional bottom row: RF patch (plain)
        if rows == 2:
            _, patch = display_crop(path, y, x)
            ax = plt.subplot(rows, len(hits), len(hits) + 1 + i)
            ax.imshow(patch)
            ax.set_title(vd["image_info"][img_idx]["label"], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")

    plt.tight_layout()
    if save_path is None:
        save_path = f"{SAVE_PREFIX}_ch{ch}_deconv_{VIS_MODE}_top{len(hits)}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print("✓ Saved:", save_path)

# ==== EXECUTE STEP 5 ====
print("=== STEP 5: exporting conv3 deconv panels with Layer-2 viz modes ===")
sel = selector.selected_channels_data['conv3']  # from Step 3

for ch in list(sel.keys()):
    export_deconv_panel(
        ch, analyzer, progressive_processor,
        layer_name='conv3', topk=9,
        save_path=f"{SAVE_PREFIX}_ch{ch}_deconv_{VIS_MODE}_top9.png"
    )

print("✓ STEP 5 complete.")



# ==== STEP 5 EXTRAS: same viz types as Layer 2 + RED RF BOXES ====
# 1) single pair: original vs deconv (for a chosen hit)
# 2) detailed sheet: rows = top-K hits, cols = [Original | Deconv | Enhanced | Meta]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import torch
from PIL import Image

# ------- reuse helpers from Step 5 / Step 4 (boost + neutral gray) -------
def _perc01(ch: torch.Tensor, lo=1.0, hi=99.0, eps=1e-8):
    ql = torch.quantile(ch.flatten(), lo/100.0)
    qh = torch.quantile(ch.flatten(), hi/100.0)
    x  = (ch - ql).clamp(min=0) / max((qh - ql).item(), eps)
    return x.clamp(0,1)

def boost01_chw(x_chw: torch.Tensor, lo=1.0, hi=99.0, gamma=0.9):
    y = x_chw.clone()
    for c in range(min(3, y.shape[0])):
        y[c] = _perc01(y[c], lo=lo, hi=hi)
    y = y.clamp(0,1) ** gamma
    return y.numpy().transpose(1,2,0)

def neutral_gray_hwc(img_hwc: np.ndarray, target: float = 0.5) -> np.ndarray:
    m = img_hwc.mean(axis=(0,1), keepdims=True)
    return np.clip(img_hwc - (m - target), 0.0, 1.0)

def recon_to_heatmap(recon_chw: torch.Tensor, p_hi=99.0, gamma=0.7, cmap_name="magma"):
    E = torch.sqrt((recon_chw ** 2).sum(0))
    flat = E.flatten()
    lo_v = torch.quantile(flat, 0.01)
    hi_v = torch.quantile(flat, p_hi/100.0)
    Em = (E - lo_v).clamp(min=0) / max((hi_v - lo_v).item(), 1e-8)
    Em = Em.clamp(0,1) ** gamma
    return cm.get_cmap(cmap_name)(Em.numpy())[:, :, :3]

# model crop (224×224) from path
def load_model_crop(img_path: str) -> np.ndarray:
    pil = Image.open(img_path).convert("RGB")
    t   = _display_tf(pil)      # [3,224,224] 0..1
    return t.numpy().transpose(1,2,0)

# ---- RF box utils (uses conv3 mapping if present, else a safe fallback) ----
def _rf_box_pixels(y, x, H=224, W=224):
    try:
        y0,y1,x0,x1 = conv3_rf_box(y, x, H, W)  # from Step 4
    except NameError:
        RF_SIZE, STRIDE, START = 99, 16, 7
        cy = START + y*STRIDE; cx = START + x*STRIDE
        r  = RF_SIZE // 2
        y0, y1 = max(0, cy - r), min(H, cy + r + (RF_SIZE % 2))
        x0, x1 = max(0, cx - r), min(W, cx + r + (RF_SIZE % 2))
    return int(y0), int(y1), int(x0), int(x1)

def draw_rf_box(ax, y, x, color='red', lw=2):
    y0,y1,x0,x1 = _rf_box_pixels(y, x)
    rect = patches.Rectangle((x0, y0), x1-x0, y1-y0,
                             linewidth=lw, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

# optional RF crop (uses helper from Step 4 if present)
def maybe_rf_crop(recon_chw: torch.Tensor, y: int, x: int, use_rf: bool=True) -> torch.Tensor:
    if not use_rf:
        return recon_chw
    return crop_rf_from_recon(recon_chw, y, x)  # defined in Step 4 cell

# =========================================================
# 1) Single pair (orig vs deconv) for one hit (rank=0 is top hit)
def render_pair_for_channel(ch, analyzer, progressive_processor,
                            layer_name='conv3', rank=0, rf_crop=False,
                            style='rgb_boost', save_path=None, box_lw=2):
    per_ch = analyzer.results[layer_name]["per_channel_topk"]
    vd     = progressive_processor.validation_data[layer_name]
    hit    = per_ch[ch][rank]
    img_idx, y, x, val, path = hit["image_idx"], hit["y"], hit["x"], hit["value"], hit["image_path"]

    # deconv
    sparse = make_sparse_conv3(ch, y, x, val)
    recon  = projector(sparse, vd["pool_switches"][img_idx])  # [3,224,224]
    recon  = maybe_rf_crop(recon, y, x, use_rf=rf_crop)

    if style == 'rgb_boost':
        vis = neutral_gray_hwc(boost01_chw(recon, lo=1.0, hi=99.0, gamma=0.9), 0.5)
    elif style == 'heatmap':
        vis = recon_to_heatmap(recon)
    else:
        vis = boost01_chw(recon)

    # original (full model crop)
    orig = load_model_crop(path)

    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), facecolor="black")
    ax1.imshow(orig)
    draw_rf_box(ax1, y, x, lw=box_lw)                          # << red RF box on original
    ax1.set_title(f"Original Image\n{vd['image_info'][img_idx]['label']}", color="white")
    ax1.axis("off")

    ax2.imshow(vis)
    # If we didn't RF-crop the deconv, also draw the RF box on the deconv side
    if not rf_crop:
        draw_rf_box(ax2, y, x, lw=box_lw)
    ax2.set_title(f"Deconv Reconstruction\nActivation: {val:.2f}", color="white")
    ax2.axis("off")

    plt.suptitle(f"Deconvolutional Visualization - {vd['image_info'][img_idx]['image_id']}",
                 color="white", fontsize=14)
    plt.tight_layout()

    if save_path is None:
        save_path = f"{layer_name}_ch{ch}_pair_rank{rank+1}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.show()
    print("✓ Saved:", save_path)

# =========================================================
# 2) Detailed sheet for a channel (rows = topK hits)
def export_detailed_sheet_for_channel(ch, analyzer, progressive_processor,
                                      layer_name='conv3', topk=9,
                                      enhance_gain=2.0, rf_crop=False,
                                      save_path=None, box_lw=2):
    """
    Columns:
      [ Original | Deconv Reconstruction | Enhanced (gain ×) | Meta ]
    Enhanced does a centered gain around 0.5 (paper-ish punch).
    """
    per_ch = analyzer.results[layer_name]["per_channel_topk"]
    vd     = progressive_processor.validation_data[layer_name]
    hits   = per_ch[ch][:min(topk, len(per_ch[ch]))]

    nrows, ncols = len(hits), 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 1.4*nrows), facecolor="black")
    if nrows == 1: axes = np.expand_dims(axes, 0)

    for r, h in enumerate(hits):
        img_idx, y, x, val, path = h["image_idx"], h["y"], h["x"], h["value"], h["image_path"]

        # original
        orig = load_model_crop(path)
        axes[r,0].imshow(orig)
        axes[r,0].set_title("Original Image", color="white", fontsize=9)
        draw_rf_box(axes[r,0], y, x, lw=box_lw)                # << red RF box
        axes[r,0].axis("off")

        # deconv
        sparse = make_sparse_conv3(ch, y, x, val)
        recon  = projector(sparse, vd["pool_switches"][img_idx])
        if rf_crop:
            recon = maybe_rf_crop(recon, y, x, use_rf=True)

        deconv = neutral_gray_hwc(boost01_chw(recon, lo=1.0, hi=99.0, gamma=0.9), 0.5)
        axes[r,1].imshow(deconv)
        if not rf_crop:
            draw_rf_box(axes[r,1], y, x, lw=box_lw)            # << red RF box on deconv if full frame
        axes[r,1].set_title("Deconv Reconstruction", color="white", fontsize=9)
        axes[r,1].axis("off")

        # enhanced (centered gain)
        m = 0.5
        enhanced = np.clip(m + enhance_gain*(deconv - m), 0, 1)
        axes[r,2].imshow(enhanced)
        axes[r,2].set_title(f"Enhanced ({enhance_gain:.1f}×)", color="white", fontsize=9)
        axes[r,2].axis("off")

        # meta
        axm = axes[r,3]
        axm.axis("off"); axm.set_facecolor("black")
        axm.text(0.02, 0.85, f"Image: {vd['image_info'][img_idx]['image_id']}", color="white", fontsize=8, transform=axm.transAxes)
        axm.text(0.02, 0.65, f"Label: {vd['image_info'][img_idx]['label']}",   color="white", fontsize=8, transform=axm.transAxes)
        axm.text(0.02, 0.45, f"Activation: {val:.2f}",                         color="white", fontsize=8, transform=axm.transAxes)
        axm.text(0.02, 0.25, f"Rank: {r+1}",                                   color="white", fontsize=8, transform=axm.transAxes)

    plt.suptitle(f"Detailed Deconvolutional Analysis — {layer_name.upper()} Channel {ch}",
                 color="white", fontsize=12)
    plt.tight_layout()

    if save_path is None:
        save_path = f"{layer_name}_ch{ch}_detailed_sheet_top{len(hits)}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.show()
    print("✓ Saved:", save_path)

# ====== EXAMPLES (run after Step 5 selection) ======
print("== Extra L3 visualizations ==")
sel = selector.selected_channels_data['conv3']  # from Step 3

for ch in list(sel.keys()):
    # A) single pair for the top hit (flip rf_crop=True to focus on the RF tile)
    render_pair_for_channel(
        ch, analyzer, progressive_processor,
        layer_name='conv3', rank=0, rf_crop=False,
        style='rgb_boost',
        save_path=f"conv3_ch{ch}_pair_top1.png",
        box_lw=2
    )

    # B) detailed sheet (like your layer-2 atlas)
    export_detailed_sheet_for_channel(
        ch, analyzer, progressive_processor,
        layer_name='conv3', topk=9, enhance_gain=2.0, rf_crop=False,
        save_path=f"conv3_ch{ch}_detailed_sheet_top9.png",
        box_lw=2
    )

print("✓ extras exported.")



# ==== EXTRA: Dual-crops panel (Deconv crops vs Original crops) for a Conv3 channel ====
import numpy as np
import matplotlib.pyplot as plt

# tiny visibility helpers (same vibe as before)
def _perc01(ch, lo=1.0, hi=99.0, eps=1e-8):
    ql = torch.quantile(ch.flatten(), lo/100.0)
    qh = torch.quantile(ch.flatten(), hi/100.0)
    y  = (ch - ql).clamp(min=0) / max((qh - ql).item(), eps)
    return y.clamp(0,1)

def boost01_chw(x_chw: torch.Tensor, lo=1.0, hi=99.0, gamma=0.9):
    y = x_chw.clone()
    for c in range(min(3, y.shape[0])):
        y[c] = _perc01(y[c], lo=lo, hi=hi)
    y = y.clamp(0,1) ** gamma
    return y.cpu().numpy().transpose(1,2,0)

def neutral_gray_hwc(img_hwc: np.ndarray, target: float = 0.5) -> np.ndarray:
    m = img_hwc.mean(axis=(0,1), keepdims=True)
    return np.clip(img_hwc - (m - target), 0.0, 1.0)

def render_dual_crops_for_channel(
    ch: int,
    analyzer,
    progressive_processor,
    layer_name: str = "conv3",
    n: int = 6,
    rf_crop: bool = True,
    facecolor: str = "black",
    save_path: str | None = None,
):
    """
    Makes a 2×n figure:
      top    = RF-cropped deconv reconstructions (with visibility boost)
      bottom = matching RF crops from the original images
    """
    per_ch = analyzer.results[layer_name]["per_channel_topk"]
    vd     = progressive_processor.validation_data[layer_name]
    hits   = per_ch[ch][:min(n, len(per_ch[ch]))]

    # prepare figure
    fig, axes = plt.subplots(2, len(hits), figsize=(len(hits)*2.6, 4.2), facecolor=facecolor)
    if len(hits) == 1:
        axes = np.expand_dims(axes, axis=1)

    # row headers
    axes[0,0].set_title("Deconv crops", color="white" if facecolor=="black" else "black", fontsize=10, loc="left", pad=12)
    axes[1,0].set_title("Original crops", color="white" if facecolor=="black" else "black", fontsize=10, loc="left", pad=12)
    # left channel label (rotated)
    fig.text(0.005, 0.5, f"Channel {ch}", va="center", rotation=90,
             color="white" if facecolor=="black" else "black", fontsize=11)

    for i, h in enumerate(hits):
        img_idx, y, x, val, path = h["image_idx"], h["y"], h["x"], h["value"], h["image_path"]

        # --- Deconv crop ---
        sparse = make_sparse_conv3(ch, y, x, val)                   # one-hot (scaled by val)
        pool_switches = vd["pool_switches"][img_idx]
        recon = projector(sparse, pool_switches)                    # [3,224,224] (CHW, ~0..1)
        recon_crop = crop_rf_from_recon(recon, y, x) if rf_crop else recon

        vis = boost01_chw(recon_crop, lo=1.0, hi=99.0, gamma=0.9)
        vis = neutral_gray_hwc(vis, target=0.5)

        ax = axes[0, i]
        ax.imshow(vis)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_axis_off()

        # --- Original RF crop ---
        _, patch = display_crop(path, y, x)                         # [224,224,3] in [0,1]
        ax = axes[1, i]
        ax.imshow(patch)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_axis_off()

    plt.tight_layout(pad=1.2, rect=(0.02, 0.02, 0.98, 0.95))
    if save_path is None:
        save_path = f"{layer_name}_tight_dual_crops_ch{ch}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=facecolor)
    print(f"✓ Saved: {save_path}  •  channel={ch}  •  n={len(hits)}")
    plt.show()

# ---- example usage (for the channels you currently have selected in Step 3) ----
sel = selector.selected_channels_data["conv3"]  # from your Step 3 selector
for ch in list(sel.keys()):
    render_dual_crops_for_channel(
        ch, analyzer, progressive_processor,
        layer_name="conv3", n=6, rf_crop=True,
        facecolor="black",
        save_path=f"conv3_tight_dual_crops_ch{ch}.png"
    )


