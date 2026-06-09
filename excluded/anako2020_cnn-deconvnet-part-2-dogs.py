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



# ==== VISUALIZATION COLOR SCHEME CONFIGURATION ====
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from scipy.ndimage import gaussian_filter

class VisualizationConfig:
    """Centralized configuration for all visualization styling"""
    
    # Color scheme settings
    FIGURE_BG = 'black'  # or 'white' for light theme
    TEXT_COLOR = 'white'  # or 'black' for light theme
    SPINE_COLOR = 'black'
    SPINE_WIDTH = 2
    
    # Deconvolutional enhancement parameters
    DECONV_ENHANCE = {
        'saturation': 0.75,      # reduce saturation for calmer look
        'gamma': 0.92,           # slight gamma correction
        'blur_sigma': 0.6,       # gaussian blur to reduce checkerboard
        'gray_blend': 0.12,      # blend toward mid-gray
        'contrast_lo': 1,        # robust contrast percentiles
        'contrast_hi': 99,
        'out_lo': 0.12,          # output range
        'out_hi': 0.88
    }
    
    # Standard enhancement multiplier
    STANDARD_ENHANCE_FACTOR = 2.0

# Global styling functions
def robust_rescale(img, config=VisualizationConfig.DECONV_ENHANCE):
    """Rescale per-crop using robust percentiles"""
    p_lo = np.percentile(img, config['contrast_lo'])
    p_hi = np.percentile(img, config['contrast_hi'])
    if p_hi <= p_lo:
        return np.clip(img, 0, 1)
    x = (img - p_lo) / (p_hi - p_lo)
    x = np.clip(x, 0, 1)
    return config['out_lo'] + x * (config['out_hi'] - config['out_lo'])

def apply_deconv_styling(img_rgb, config=VisualizationConfig.DECONV_ENHANCE):
    """Apply paper-style enhancement to deconv crops"""
    x = img_rgb.astype(np.float32)
    
    # Robust contrast
    x = robust_rescale(x, config)
    
    # Light blur (channel-wise)
    x = np.stack([gaussian_filter(x[..., i], config['blur_sigma']) for i in range(3)], axis=-1)
    
    # Desaturate in HSV
    hsv = rgb_to_hsv(np.clip(x, 0, 1))
    hsv[..., 1] *= config['saturation']
    x = hsv_to_rgb(hsv)
    
    # Mild gamma
    x = np.clip(x, 0, 1) ** config['gamma']
    
    # Blend to mid-gray
    x = (1 - config['gray_blend']) * x + config['gray_blend'] * 0.5
    
    return np.clip(x, 0, 1)

def style_axis(ax, config=VisualizationConfig):
    """Apply consistent axis styling"""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(config.SPINE_WIDTH)
        spine.set_color(config.SPINE_COLOR)
    ax.set_facecolor(config.FIGURE_BG)

def style_figure(fig, config=VisualizationConfig):
    """Apply consistent figure styling"""
    fig.patch.set_facecolor(config.FIGURE_BG)


# #To change themes, modify the VisualizationConfig class:
# # For light theme:
# VisualizationConfig.FIGURE_BG = 'white'
# VisualizationConfig.TEXT_COLOR = 'black'

# # For dark theme:
# VisualizationConfig.FIGURE_BG = 'black'  
# VisualizationConfig.TEXT_COLOR = 'white'


# ==== Step 0.a â€” paths & sanity ====
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
print(f"VAL_DIR OK â†’ {VAL_DIR}  â€¢ JPEGs: {n_imgs}")
print("VAL_ANN_DIR present:", os.path.exists(VAL_ANN_DIR))


# tiny helper: load synsetâ†’text mapping (weâ€™ll reuse later)
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
    print("  ", k, "â†’", SYN2TXT[k])


# One-off: find synset IDs by keywords (case-insensitive)
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

# EXAMPLES â€” change these to what you actually need:
_ = find_synsets(["golden retriever", "chihuahua", "siberian husky", "pug", 'dog','puppy','sheepdog','sighthound','hound','greyhound','wolfhound',
            'terrier','retriever','shepherd','spaniel','setter','pointer','mastiff','bulldog','poodle','husky','beagle','collie',])



import torch
from torchvision.models import alexnet

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained model
model = alexnet(weights="DEFAULT").to(device) 
model.eval()


SYNSETS = {'n02091032', 'n02116738'}  # my picks - choosing dogs


# === STEP 0c: Filter by hardcoded synsets (standalone) ===
import os, pandas as pd
from PIL import Image

# --- config (edit paths if needed) ---
#VAL_DIR     = VAL_DIR  # e.g. "/kaggle/input/.../ILSVRC/Data/CLS-LOC/val"
#LABELS_CSV  = "/kaggle/input/imagenet-object-localization-challenge/LOC_val_solution.csv"
#SYNSET_MAP  = "/kaggle/input/imagenet-object-localization-challenge/LOC_synset_mapping.txt"

# --- sanity on paths ---
assert os.path.isdir(VAL_DIR), f"VAL_DIR not found: {VAL_DIR}"
assert os.path.exists(LABELS_CSV), f"labels csv missing: {LABELS_CSV}"
assert os.path.exists(SYNSET_MAP), f"synset map missing: {SYNSET_MAP}"
assert os.path.exists(SYNSET_MAP), f"synset map missing: {SYNSET_MAP}"


# --- lookup names for status line ---
syn_names = {}
with open(SYNSET_MAP, "r") as f:
    for line in f:
        parts = line.strip().split(" ", 1)
        if len(parts) == 2 and parts[0] in SYNSETS:
            syn_names[parts[0]] = parts[1].split(",")[0]

missing = SYNSETS - set(syn_names.keys())
print("Target synsets:", ", ".join(sorted(SYNSETS)))
print("Names:", ", ".join(f"{s}â†’{syn_names.get(s, '?')}" for s in sorted(SYNSETS)))
if missing:
    print("âš  Missing in mapping:", sorted(missing))

# --- load labels, extract synset, dedup by ImageId ---
df = pd.read_csv(LABELS_CSV)
sec = next(c for c in df.columns if c.lower() != "imageid")  # usually 'PredictionString' or 'Label'
df = df[["ImageId", sec]].copy()
df["Synset"] = df[sec].astype(str).str.split().str[0]
before = len(df)
df = df.drop_duplicates("ImageId")
print(f"Rows: {before} â†’ {len(df)} unique ImageId")

# --- filter by hardcoded synsets ---
df = df[df["Synset"].isin(SYNSETS)].copy()
print(f"Matched images: {len(df)}")

# --- build paths + existence check ---
df["ImagePath"] = df["ImageId"].apply(lambda i: os.path.join(VAL_DIR, f"{i}.JPEG"))
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
    print(f"  {i}. {it['image_id']} â†’ {os.path.basename(it['image_path'])} ({it['label']})")

# (optional) actually load them now
# images = [Image.open(d["image_path"]).convert("RGB") for d in selected_images]
# print(f"Loaded {len(images)} images.")



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



# ==== DEFINE ExactFeatureExtractor CLASS FIRST ====
# Step 1 - extract features map
# Process validation set (50 dog images instead of 50,000)

# Extracting conv2 feature maps and pooling indices from 50 images
# Storting all activation tensors and switches

import time
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models import alexnet, AlexNet_Weights

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights   = AlexNet_Weights.DEFAULT
transform = weights.transforms()  # Resize(256) â†’ CenterCrop(224) â†’ ToTensor â†’ Normalize

# if model not defined yet, create it
try:
    model
except NameError:
    model = alexnet(weights=weights).to(device).eval()


class ExactFeatureExtractor:
    """Extract features with exact pooling indices like Zeiler & Fergus paper"""
    def __init__(self, model):
        self.model = model
        self.feature_maps = OrderedDict()
        self.pooling_indices = OrderedDict()
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture both feature maps and compute pooling indices"""
        alexnet_layer_map = {
            0: 'conv1',   1: 'relu1',   2: 'pool1',
            3: 'conv2',   4: 'relu2',   5: 'pool2', 
            6: 'conv3',   7: 'relu3',
            8: 'conv4',   9: 'relu4',
            10: 'conv5',  11: 'relu5',  12: 'pool5'
        }
        
        def make_hook(name, layer_idx):
            def hook_fn(module, input, output):
                self.feature_maps[name] = output.clone()
                
                if isinstance(module, nn.MaxPool2d):
                    input_tensor = input[0]
                    _, indices = F.max_pool2d(
                        input_tensor,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        return_indices=True
                    )
                    self.pooling_indices[name] = indices.clone()
            return hook_fn
        
        layer_names = []
        for idx, layer in enumerate(self.model.features):
            if idx in alexnet_layer_map:
                layer_name = alexnet_layer_map[idx]
                hook = layer.register_forward_hook(make_hook(layer_name, idx))
                self.hooks.append(hook)
                layer_names.append(layer_name)
        
        print("Registered hooks for layers:", layer_names)
    
    def extract_features(self, input_tensor):
        """Extract features and indices"""
        self.feature_maps.clear()
        self.pooling_indices.clear()
        
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        return self.feature_maps.copy(), self.pooling_indices.copy()
    
    def cleanup(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()

class ProgressiveProcessor:
    """Step 1: Process selected images with detailed progress tracking"""
    
    def __init__(self, model, feature_extractor):
        self.model = model
        self.feature_extractor = feature_extractor
        self.validation_data = {}
        
    def process_images_with_progress(self, selected_images, layer_name='conv2'):
        """Process images with detailed progress tracking"""
        num_images = len(selected_images)
        print(f"\n=== STEP 1: PROCESSING {num_images} DOG IMAGES FOR {layer_name.upper()} ===")
        
        # Initialize storage
        self.validation_data[layer_name] = {
            'feature_maps': [],
            'image_paths': [],
            'image_info': [],
            'processing_times': []
        }
        
        total_start_time = time.time()
        successful_count = 0
        
        for i, img_info in enumerate(selected_images):
            img_start_time = time.time()
            img_path = img_info['image_path']
            img_id = img_info['image_id']
            
            print(f"[{i+1}/{num_images}] {img_id} ({img_info['label']})", end=" ")
            
            try:
                # Load and process image
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                # Extract features
                feature_maps, pooling_indices = self.feature_extractor.extract_features(img_tensor)
                
                # Store results
                self.validation_data[layer_name]['feature_maps'].append(feature_maps[layer_name].clone())
                self.validation_data[layer_name]['image_paths'].append(img_path)
                self.validation_data[layer_name]['image_info'].append(img_info)
                
                img_time = time.time() - img_start_time
                self.validation_data[layer_name]['processing_times'].append(img_time)
                
                successful_count += 1
                print(f"âœ“ ({img_time:.1f}s)")
                
                # Progress summary every 10 images
                if (i + 1) % 10 == 0:
                    total_elapsed = time.time() - total_start_time
                    avg_time = total_elapsed / (i + 1)
                    eta = avg_time * (num_images - i - 1)
                    print(f"  Progress: {successful_count}/{num_images} ({100*successful_count/num_images:.0f}%) | ETA: {eta:.0f}s")
                
            except Exception as e:
                print(f"âœ— ERROR: {e}")
                continue
        
        # Final summary
        total_time = time.time() - total_start_time
        avg_time = total_time / successful_count if successful_count > 0 else 0
        
        print(f"\n=== STEP 1 COMPLETE ===")
        print(f"Successfully processed: {successful_count}/{num_images} dog images")
        print(f"Total time: {total_time:.1f} seconds ({avg_time:.1f}s per image)")
        
        if successful_count > 0:
            sample_shape = self.validation_data[layer_name]['feature_maps'][0].shape
            print(f"Feature map shape: {sample_shape}")
        
        return successful_count

# Initialize feature extractor
print("Initializing feature extractor...")
try:
    feature_extractor.cleanup()
except:
    pass

feature_extractor = ExactFeatureExtractor(model)

# Execute Step 1
progressive_processor = ProgressiveProcessor(model, feature_extractor)
num_processed = progressive_processor.process_images_with_progress(selected_images, 'conv2')

# What did we store per image?
vd = progressive_processor.validation_data['conv2']
print("Stored items:", len(vd['feature_maps']), "feature maps")
print("Sample fmap:", vd['feature_maps'][0].dtype, vd['feature_maps'][0].shape)

print(f"\nREADY FOR STEP 2: {num_processed} images processed for conv2 analysis")


# ==== STEP 2: FIND GLOBALLY STRONGEST ACTIVATIONS ====

# Analyzing all 192 conv2 channels across 50 images
# Finding maximum activation value and location for each channel
# Ranking all activations to identify strongest channels

class ActivationAnalyzer:
    """Step 2: Find globally strongest activations across all processed images"""
    
    def __init__(self, progressive_processor):
        self.processor = progressive_processor
        self.global_activations = {}
        
    def find_strongest_activations(self, layer_name='conv2', verbose=True):
        """Find strongest activations across all images for each channel"""
        if layer_name not in self.processor.validation_data:
            print(f"No data found for {layer_name}")
            return {}
            
        data = self.processor.validation_data[layer_name]
        feature_maps = data['feature_maps']
        num_images = len(feature_maps)
        num_channels = feature_maps[0].shape[1]  # 192 for conv2
        
        if verbose:
            print(f"\n=== STEP 2: FINDING STRONGEST ACTIVATIONS IN {layer_name.upper()} ===")
            print(f"Analyzing {num_images} images Ã— {num_channels} channels = {num_images * num_channels} activations")
        
        # For each channel, find strongest activations across all images
        channel_results = {}
        
        for channel_idx in range(num_channels):
            if verbose and channel_idx % 50 == 0:
                print(f"Processing channel {channel_idx+1}/{num_channels}")
                
            channel_activations = []
            
            # Go through all 50 images for this channel
            for img_idx, feature_map in enumerate(feature_maps):
                channel_map = feature_map[0, channel_idx]  # Shape: [27, 27]
                max_val = torch.max(channel_map).item()
                
                # Find location of max value
                max_locations = (channel_map == max_val).nonzero(as_tuple=False)
                if len(max_locations) > 0:
                    location = max_locations[0]  # Take first if multiple
                    y, x = location[0].item(), location[1].item()
                    
                    channel_activations.append({
                        'image_idx': img_idx,
                        'image_info': data['image_info'][img_idx],
                        'image_path': data['image_paths'][img_idx],
                        'max_value': max_val,
                        'location': (y, x),
                    })
            
            # Sort by activation strength (strongest first)
            channel_activations.sort(key=lambda x: x['max_value'], reverse=True)
            channel_results[channel_idx] = channel_activations
        
        self.global_activations[layer_name] = channel_results
        
        if verbose:
            print(f"Analysis complete for all {num_channels} channels")
            
        return channel_results
    
    def get_top_channels(self, layer_name='conv2', top_n=5):
        """Get the channels with highest maximum activations"""
        if layer_name not in self.global_activations:
            print(f"No analysis found for {layer_name}")
            return []
            
        channel_results = self.global_activations[layer_name]
        
        # Find the maximum activation for each channel
        channel_max_values = []
        for channel_idx, activations in channel_results.items():
            if activations:  # Make sure we have activations for this channel
                max_activation = activations[0]['max_value']  # Already sorted
                channel_max_values.append({
                    'channel': channel_idx,
                    'max_value': max_activation,
                    'activations': activations
                })
        
        # Sort channels by their maximum activation values
        channel_max_values.sort(key=lambda x: x['max_value'], reverse=True)
        
        print(f"\nTop {top_n} strongest channels in {layer_name}:")
        for i, channel_data in enumerate(channel_max_values[:top_n]):
            channel = channel_data['channel']
            max_val = channel_data['max_value']
            best_image = channel_data['activations'][0]['image_info']['label']
            print(f"  {i+1}. Channel {channel}: {max_val:.3f} (strongest in {best_image})")
            
        return channel_max_values[:top_n]

# Execute Step 2
print("Starting activation analysis...")
analyzer = ActivationAnalyzer(progressive_processor)

# Find strongest activations across all 50 dog images
channel_results = analyzer.find_strongest_activations('conv2', verbose=True)

# Get top 5 strongest channels
top_channels = analyzer.get_top_channels('conv2', top_n=5)

print(f"\nStep 2 Complete: Found strongest activations across {len(channel_results)} channels")
print("Ready for Step 3: Select top 9 images per channel")


# ==== STEP 3: SELECT TOP 2 STRONGEST CHANNELS AND THEIR TOP 9 IMAGES ====

import matplotlib.pyplot as plt
from PIL import Image

class TopChannelSelector:
    """Step 3: Find 2 strongest channels and select top 9 images for each"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.selected_channels_data = {}
        
    def find_and_select_top_channels(self, layer_name='conv2', num_channels=2, top_images=9):
        """Find the strongest channels and select their top activating images"""
        if layer_name not in self.analyzer.global_activations:
            print(f"No analysis found for {layer_name}")
            return {}
            
        print(f"\n=== STEP 3: FINDING TOP {num_channels} STRONGEST CHANNELS ===")
        
        # Get top channels globally
        top_channels = self.analyzer.get_top_channels(layer_name, top_n=num_channels)
        
        if not top_channels:
            print("No channels found!")
            return {}
        
        # For each top channel, get its top 9 images
        selected_data = {}
        
        for rank, channel_data in enumerate(top_channels, 1):
            channel_idx = channel_data['channel']
            max_activation = channel_data['max_value']
            all_activations = channel_data['activations']
            
            # Select top 9 images for this channel
            top_9_images = all_activations[:top_images]
            
            selected_data[channel_idx] = {
                'rank': rank,
                'channel_idx': channel_idx,
                'max_activation': max_activation,
                'top_9_images': top_9_images,
                'all_activations': all_activations
            }
            
            print(f"\nChannel {channel_idx} (Rank #{rank}):")
            print(f"  Max activation: {max_activation:.3f}")
            print(f"  Top {top_images} images selected:")
            
            for i, img_data in enumerate(top_9_images, 1):
                img_info = img_data['image_info']
                activation = img_data['max_value']
                location = img_data['location']
                print(f"    {i}. {img_info['image_id']}: {img_info['label']} "
                      f"(act: {activation:.3f}, loc: {location})")
        
        self.selected_channels_data[layer_name] = selected_data
        return selected_data
    
    def visualize_top_channels(self, layer_name='conv2', save_prefix="conv2_channel"):
        """Visualize top 9 images for each of the strongest channels"""
        if layer_name not in self.selected_channels_data:
            print(f"No selected channels data for {layer_name}")
            return
            
        selected_data = self.selected_channels_data[layer_name]
        
        print(f"\n=== VISUALIZING TOP {len(selected_data)} CHANNELS ===")
        
        for channel_idx, data in selected_data.items():
            rank = data['rank']
            max_activation = data['max_activation']
            top_9_images = data['top_9_images']
            
            print(f"\nVisualizing Channel {channel_idx} (Rank #{rank}, Max Act: {max_activation:.3f})")
            
            # Create 3x3 visualization
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            fig.suptitle(f'Conv2 Channel {channel_idx} (Rank #{rank}) - Top 9 Strongest Activations\n'
                        f'Max Activation: {max_activation:.3f}', 
                        fontsize=16, y=0.98)
            
            for i in range(9):
                row, col = i // 3, i % 3
                ax = axes[row, col]
                
                if i < len(top_9_images):
                    img_data = top_9_images[i]
                    img_path = img_data['image_path']
                    img_info = img_data['image_info']
                    activation = img_data['max_value']
                    location = img_data['location']
                    
                    try:
                        # Load and display image
                        img = Image.open(img_path).convert('RGB')
                        ax.imshow(img)
                        ax.set_title(f"{i+1}. {img_info['label']}\n"
                                   f"Act: {activation:.2f}, Loc: {location}", 
                                   fontsize=11)
                        ax.axis('off')
                        
                    except Exception as e:
                        ax.text(0.5, 0.5, f'Error loading image\n{e}', 
                               ha='center', va='center', fontsize=10)
                        ax.set_title(f"{i+1}. Error", fontsize=11)
                        ax.axis('off')
                else:
                    ax.axis('off')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)
            
            # Save visualization
            save_path = f"{save_prefix}_{channel_idx}_rank{rank}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"  âœ“ Saved visualization to {save_path}")
            
            plt.show()
    
    def get_summary(self, layer_name='conv2'):
        """Get summary of selected channels and their patterns"""
        if layer_name not in self.selected_channels_data:
            print(f"No data for {layer_name}")
            return
            
        selected_data = self.selected_channels_data[layer_name]
        
        print(f"\n=== SUMMARY: TOP {len(selected_data)} CHANNELS ANALYSIS ===")
        
        for channel_idx, data in selected_data.items():
            rank = data['rank']
            max_activation = data['max_activation']
            top_9_images = data['top_9_images']
            
            print(f"\nChannel {channel_idx} (Rank #{rank}):")
            print(f"  Max activation: {max_activation:.3f}")
            print(f"  Activation range: {top_9_images[-1]['max_value']:.3f} - {max_activation:.3f}")
            
            # Count breed distribution
            breed_counts = {}
            for img_data in top_9_images:
                breed = img_data['image_info']['label']
                breed_counts[breed] = breed_counts.get(breed, 0) + 1
            
            print(f"  Breed distribution:")
            for breed, count in sorted(breed_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"    {breed}: {count}")
    
    def save_channel_data(self, layer_name='conv2', filename="selected_channels_data.txt"):
        """Save detailed channel data to file"""
        if layer_name not in self.selected_channels_data:
            print(f"No data for {layer_name}")
            return
            
        selected_data = self.selected_channels_data[layer_name]
        
        with open(filename, 'w') as f:
            f.write(f"TOP {len(selected_data)} STRONGEST CHANNELS - {layer_name.upper()}\n")
            f.write("=" * 60 + "\n\n")
            
            for channel_idx, data in selected_data.items():
                rank = data['rank']
                max_activation = data['max_activation']
                top_9_images = data['top_9_images']
                
                f.write(f"CHANNEL {channel_idx} (RANK #{rank})\n")
                f.write(f"Max Activation: {max_activation:.3f}\n\n")
                
                f.write("Top 9 Images:\n")
                for i, img_data in enumerate(top_9_images, 1):
                    img_info = img_data['image_info']
                    activation = img_data['max_value']
                    location = img_data['location']
                    f.write(f"  {i:2d}. {img_info['image_id']:25s} | {img_info['label']:25s} | "
                           f"Act: {activation:6.3f} | Loc: {location}\n")
                
                f.write("\n" + "-" * 60 + "\n\n")
        
        print(f"âœ“ Saved detailed channel data to {filename}")

# ==== EXECUTE STEP 3 ====
print("=== STEP 3: SELECTING TOP 2 STRONGEST CHANNELS ===")

# Initialize the selector
selector = TopChannelSelector(analyzer)

# Find top 2 channels and select their top 9 images
selected_channels = selector.find_and_select_top_channels('conv2', num_channels=2, top_images=9)

# Visualize both channels
selector.visualize_top_channels('conv2', save_prefix="conv2_top_channel")

# Show summary analysis
selector.get_summary('conv2')

# Save detailed data
selector.save_channel_data('conv2', "top_2_channels_analysis.txt")

print(f"\nâœ… STEP 3 COMPLETE!")
print(f"âœ… Found top 2 strongest channels: {list(selected_channels.keys())}")
print(f"âœ… Selected 9 strongest activating images for each channel")
print(f"âœ… Created visualizations and saved analysis")
print(f"âœ… Ready for Step 4: Deconvolutional reconstruction")


# ==== STEP 4: CREATE ISOLATED ACTIVATIONS ====
# For each selected image, create sparse activation maps by zeroing out all other feature maps
# Keep only the target channel with its full spatial extent

# This directly implements the Zeiler & Fergus approach: 
# "Zero out all other feature maps except map f, keep the entire spatial extent of feature map f (not just single pixels)"

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

class IsolatedActivationCreator:
    """Step 4: Create sparse activations for deconvolutional reconstruction"""
    
    def __init__(self, feature_extractor, selector, progressive_processor):
        self.feature_extractor = feature_extractor
        self.selector = selector
        self.processor = progressive_processor
        self.isolated_activations = {}
        
    def create_isolated_activations_for_channel(self, layer_name='conv2', channel_idx=None):
        """Create isolated activations for all top 9 images of a specific channel"""
        if layer_name not in self.selector.selected_channels_data:
            print(f"No selected channels data for {layer_name}")
            return None
            
        if channel_idx not in self.selector.selected_channels_data[layer_name]:
            print(f"Channel {channel_idx} not found in selected data")
            return None
            
        channel_data = self.selector.selected_channels_data[layer_name][channel_idx]
        top_9_images = channel_data['top_9_images']
        
        print(f"\n=== STEP 4: CREATING ISOLATED ACTIVATIONS FOR CHANNEL {channel_idx} ===")
        print(f"Processing {len(top_9_images)} images for channel {channel_idx}")
        
        # Initialize storage for this channel
        if layer_name not in self.isolated_activations:
            self.isolated_activations[layer_name] = {}
        
        self.isolated_activations[layer_name][channel_idx] = {
            'images_data': [],
            'sparse_activations': [],
            'pooling_switches': [],
            'original_activations': []
        }
        
        # Process each of the top 9 images
        for rank, img_data in enumerate(top_9_images, 1):
            img_idx = img_data['image_idx']
            img_info = img_data['image_info']
            img_path = img_data['image_path']
            max_activation = img_data['max_value']
            location = img_data['location']
            
            print(f"  [{rank}/9] Processing {img_info['image_id']} ({img_info['label']})")
            print(f"         Max activation: {max_activation:.3f} at location {location}")
            
            try:
                # Load and process the image to get fresh activations and switches
                img = Image.open(img_path).convert('RGB')
                
                # Use the same transform as in Step 1
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                
                img_tensor = transform(img).unsqueeze(0).to(next(self.feature_extractor.model.parameters()).device)
                
                # Extract features and pooling switches
                feature_maps, pooling_indices = self.feature_extractor.extract_features(img_tensor)
                
                # Get the activation tensor for our target layer
                original_activation = feature_maps[layer_name].clone()  # Shape: [1, 192, 27, 27]
                
                # Create sparse activation - zero out all channels except target channel
                sparse_activation = torch.zeros_like(original_activation)
                sparse_activation[0, channel_idx, :, :] = original_activation[0, channel_idx, :, :]
                
                # Verify the max activation matches what we found earlier
                actual_max = torch.max(sparse_activation[0, channel_idx, :, :]).item()
                
                print(f"         Sparse activation created - max value: {actual_max:.3f}")
                
                # Store all the data we'll need for deconvolution
                isolated_data = {
                    'rank': rank,
                    'image_idx': img_idx,
                    'image_info': img_info,
                    'image_path': img_path,
                    'image_tensor': img_tensor.clone(),
                    'original_activation': original_activation.clone(),
                    'sparse_activation': sparse_activation.clone(),
                    'pooling_switches': {k: v.clone() for k, v in pooling_indices.items()},
                    'max_activation': actual_max,
                    'max_location': location,
                    'all_feature_maps': {k: v.clone() for k, v in feature_maps.items()}
                }
                
                self.isolated_activations[layer_name][channel_idx]['images_data'].append(isolated_data)
                
                print(f"         âœ“ Isolated activation stored for reconstruction")
                
            except Exception as e:
                print(f"         âœ— ERROR processing image: {e}")
                continue
        
        num_successful = len(self.isolated_activations[layer_name][channel_idx]['images_data'])
        print(f"\nâœ“ Channel {channel_idx}: {num_successful}/9 isolated activations created")
        
        return self.isolated_activations[layer_name][channel_idx]
    
    def create_all_isolated_activations(self, layer_name='conv2'):
        """Create isolated activations for all selected channels"""
        if layer_name not in self.selector.selected_channels_data:
            print(f"No selected channels data for {layer_name}")
            return
            
        selected_channels = list(self.selector.selected_channels_data[layer_name].keys())
        print(f"\n=== CREATING ISOLATED ACTIVATIONS FOR ALL {len(selected_channels)} CHANNELS ===")
        
        results = {}
        for channel_idx in selected_channels:
            channel_result = self.create_isolated_activations_for_channel(layer_name, channel_idx)
            results[channel_idx] = channel_result
        
        return results
    
    def verify_isolated_activations(self, layer_name='conv2', channel_idx=None):
        """Verify the isolated activations were created correctly"""
        if (layer_name not in self.isolated_activations or 
            channel_idx not in self.isolated_activations[layer_name]):
            print(f"No isolated activations found for {layer_name} channel {channel_idx}")
            return
            
        channel_data = self.isolated_activations[layer_name][channel_idx]
        images_data = channel_data['images_data']
        
        print(f"\n=== VERIFICATION: CHANNEL {channel_idx} ISOLATED ACTIVATIONS ===")
        print(f"Successfully created: {len(images_data)} sparse activations")
        
        for i, data in enumerate(images_data, 1):
            sparse_act = data['sparse_activation']
            original_act = data['original_activation']
            
            # Check dimensions
            print(f"  Image {i}: Sparse activation shape: {sparse_act.shape}")
            
            # Verify only target channel is non-zero
            non_zero_channels = torch.sum(torch.sum(torch.sum(sparse_act[0], dim=2), dim=1) != 0)
            print(f"           Non-zero channels: {non_zero_channels.item()} (should be 1)")
            
            # Check max activation
            sparse_max = torch.max(sparse_act).item()
            original_max = torch.max(original_act[0, channel_idx]).item()
            print(f"           Max activation: sparse={sparse_max:.3f}, original={original_max:.3f}")
            
            # Verify switches are stored
            switches_stored = len(data['pooling_switches'])
            print(f"           Pooling switches stored: {switches_stored} layers")
    
    def get_isolated_data_summary(self, layer_name='conv2'):
        """Get summary of all isolated activations created"""
        if layer_name not in self.isolated_activations:
            print(f"No isolated activations for {layer_name}")
            return
            
        print(f"\n=== ISOLATED ACTIVATIONS SUMMARY ({layer_name.upper()}) ===")
        
        for channel_idx, channel_data in self.isolated_activations[layer_name].items():
            num_images = len(channel_data['images_data'])
            print(f"Channel {channel_idx}: {num_images} isolated activations ready for deconvolution")
            
            if num_images > 0:
                # Show activation range for this channel
                activations = [data['max_activation'] for data in channel_data['images_data']]
                min_act, max_act = min(activations), max(activations)
                print(f"  Activation range: {min_act:.3f} - {max_act:.3f}")
                
                # Show what pooling layers we have switches for
                sample_switches = channel_data['images_data'][0]['pooling_switches']
                switch_layers = list(sample_switches.keys())
                print(f"  Pooling switches available: {switch_layers}")

# ==== EXECUTE STEP 4 ====
print("=== STEP 4: CREATING ISOLATED ACTIVATIONS ===")

# Initialize the isolated activation creator
activation_creator = IsolatedActivationCreator(feature_extractor, selector, progressive_processor)

# Create isolated activations for all selected channels (11 and 180)
isolated_results = activation_creator.create_all_isolated_activations('conv2')

# Verify the results
activation_creator.verify_isolated_activations('conv2', 11)
activation_creator.verify_isolated_activations('conv2', 180)

# Show summary
activation_creator.get_isolated_data_summary('conv2')

print(f"\nâœ… STEP 4 COMPLETE!")
print(f"âœ… Created isolated activations for channels {list(isolated_results.keys())}")
print(f"âœ… All data ready for deconvolutional reconstruction (Step 5)")
print(f"âœ… Each sparse activation has only 1 non-zero channel with full spatial extent")
print(f"âœ… Pooling switches stored for accurate unpooling")


# ==== STEP 5: DECONVOLUTIONAL NETWORK RECONSTRUCTION ====
# Project sparse activations back to pixel space using deconvnet
# Start from AFTER pool2, not from conv2 activations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class DeconvolutionalNetwork:
    """Step 5: Deconvolutional network to project activations back to pixel space"""
    
    def __init__(self, original_model):
        self.original_model = original_model
        self.device = next(original_model.parameters()).device
        
        # Extract the convolutional layers we need to reverse
        self.layer_info = {}
        
        # Map AlexNet feature layers
        alexnet_layer_map = {
            0: ('conv1', nn.Conv2d),   2: ('pool1', nn.MaxPool2d),
            3: ('conv2', nn.Conv2d),   5: ('pool2', nn.MaxPool2d),
            6: ('conv3', nn.Conv2d),   8: ('conv4', nn.Conv2d),
            10: ('conv5', nn.Conv2d),  12: ('pool5', nn.MaxPool2d)
        }
        
        # Store layer information for reconstruction
        for idx, layer in enumerate(original_model.features):
            if idx in alexnet_layer_map:
                layer_name, layer_type = alexnet_layer_map[idx]
                self.layer_info[layer_name] = {
                    'layer': layer,
                    'type': layer_type,
                    'index': idx
                }
        
        print("Deconvolutional network initialized")
        print(f"Available layers for reconstruction: {list(self.layer_info.keys())}")
    
    def get_conv2_pooled_activation(self, sparse_activation):
        """
        Apply pool2 to the conv2 sparse activation to match the expected input shape for unpooling
        
        The issue: We stored conv2 activations BEFORE pool2, but we need pool2 switches
        Solution: Apply pool2 to get the correct shape, then unpool
        """
        pool2_layer = self.layer_info['pool2']['layer']
        
        # Apply pool2 to get the shape that matches our stored switches
        pooled_activation, _ = F.max_pool2d(
            sparse_activation,
            kernel_size=pool2_layer.kernel_size,
            stride=pool2_layer.stride,
            padding=pool2_layer.padding,
            return_indices=True
        )
        
        return pooled_activation
    
    def reconstruct_from_conv2(self, sparse_activation, pooling_switches):
        """
        Reconstruct input from conv2 activation using deconvnet
        
        Corrected approach:
        1. Start with conv2 activation [1, 192, 27, 27]
        2. Apply pool2 to get [1, 192, 13, 13] (matches switch dimensions)
        3. Unpool using stored switches back to conv2 size
        4. Continue reconstruction path
        """
        print(f"\n=== DECONVOLUTIONAL RECONSTRUCTION FROM CONV2 ===")
        print(f"Input sparse activation shape: {sparse_activation.shape}")
        
        # Step 0: Apply pool2 to match the switch dimensions
        print("Step 0: Applying pool2 to match switch dimensions...")
        pooled_activation = self.get_conv2_pooled_activation(sparse_activation)
        print(f"   After pooling: {pooled_activation.shape}")
        
        current_activation = pooled_activation
        
        # Step 1: Reverse pool2 (unpool using stored switches)
        print("Step 1: Unpooling pool2...")
        if 'pool2' in pooling_switches:
            pool2_switches = pooling_switches['pool2']
            pool2_layer = self.layer_info['pool2']['layer']
            
            print(f"   Switch shape: {pool2_switches.shape}")
            print(f"   Current activation shape: {current_activation.shape}")
            
            # Unpool using max_unpool2d
            unpooled = F.max_unpool2d(
                current_activation,
                pool2_switches,
                kernel_size=pool2_layer.kernel_size,
                stride=pool2_layer.stride,
                padding=pool2_layer.padding
            )
            current_activation = unpooled
            print(f"   After unpooling: {current_activation.shape}")
        
        # Step 2: Reverse ReLU2 (apply ReLU - deconv paper keeps ReLU)
        print("Step 2: Applying ReLU...")
        current_activation = F.relu(current_activation)
        print(f"   After ReLU: {current_activation.shape}")
        
        # Step 3: Reverse conv2 (transposed convolution)
        print("Step 3: Transposed conv2...")
        conv2_layer = self.layer_info['conv2']['layer']
        
        print(f"   Original conv2: {conv2_layer.in_channels} -> {conv2_layer.out_channels}")
        print(f"   Current activation channels: {current_activation.shape[1]}")
        
        # Create transposed convolution - input/output channels swapped from original
        transposed_conv2 = nn.ConvTranspose2d(
            in_channels=conv2_layer.out_channels,  # 192 (conv2 output becomes deconv input)
            out_channels=conv2_layer.in_channels,  # 64 (conv2 input becomes deconv output) 
            kernel_size=conv2_layer.kernel_size,
            stride=conv2_layer.stride,
            padding=conv2_layer.padding,
            bias=False
        ).to(self.device)
        
        # Use the same weights but transposed for deconvolution
        with torch.no_grad():
            # Original conv2 weight: [192, 64, 5, 5] (out_channels, in_channels, H, W)
            # For deconv we need: [192, 64, 5, 5] (in_channels, out_channels, H, W)
            # No transpose needed - just copy directly
            transposed_conv2.weight.data = conv2_layer.weight.data
        
        current_activation = transposed_conv2(current_activation)
        print(f"   After transposed conv2: {current_activation.shape}")
        
        # Step 4: Reverse pool1 (unpool using stored switches)
        print("Step 4: Unpooling pool1...")
        if 'pool1' in pooling_switches:
            pool1_switches = pooling_switches['pool1']
            pool1_layer = self.layer_info['pool1']['layer']
            
            unpooled = F.max_unpool2d(
                current_activation,
                pool1_switches,
                kernel_size=pool1_layer.kernel_size,
                stride=pool1_layer.stride,
                padding=pool1_layer.padding
            )
            current_activation = unpooled
            print(f"   After unpooling: {current_activation.shape}")
        
        # Step 5: Reverse ReLU1
        print("Step 5: Applying ReLU...")
        current_activation = F.relu(current_activation)
        print(f"   After ReLU: {current_activation.shape}")
        
        # Step 6: Reverse conv1 (transposed convolution)
        print("Step 6: Transposed conv1...")
        conv1_layer = self.layer_info['conv1']['layer']
        
        print(f"   Original conv1: {conv1_layer.in_channels} -> {conv1_layer.out_channels}")
        print(f"   Current activation channels: {current_activation.shape[1]}")
        
        transposed_conv1 = nn.ConvTranspose2d(
            in_channels=conv1_layer.out_channels,  # 64 (conv1 output becomes deconv input)
            out_channels=conv1_layer.in_channels,  # 3 (conv1 input becomes deconv output)
            kernel_size=conv1_layer.kernel_size,
            stride=conv1_layer.stride,
            padding=conv1_layer.padding,
            bias=False
        ).to(self.device)
        
        # Use the same weights for deconvolution
        with torch.no_grad():
            # Original conv1 weight: [64, 3, 11, 11] (out_channels, in_channels, H, W)
            # For deconv we need: [64, 3, 11, 11] (in_channels, out_channels, H, W)
            # No transpose needed - just copy directly
            transposed_conv1.weight.data = conv1_layer.weight.data
        
        reconstruction = transposed_conv1(current_activation)
        print(f"   Final reconstruction shape: {reconstruction.shape}")
        
        return reconstruction
    
    def reconstruct_all_images_for_channel(self, layer_name, channel_idx, activation_creator):
        """Reconstruct all 9 images for a specific channel"""
        if (layer_name not in activation_creator.isolated_activations or
            channel_idx not in activation_creator.isolated_activations[layer_name]):
            print(f"No isolated activations found for {layer_name} channel {channel_idx}")
            return []
        
        channel_data = activation_creator.isolated_activations[layer_name][channel_idx]
        images_data = channel_data['images_data']
        
        print(f"\n=== RECONSTRUCTING ALL IMAGES FOR CHANNEL {channel_idx} ===")
        print(f"Processing {len(images_data)} images...")
        
        reconstructions = []
        
        for i, img_data in enumerate(images_data, 1):
            print(f"\n[{i}/{len(images_data)}] Reconstructing {img_data['image_info']['image_id']}")
            
            try:
                sparse_activation = img_data['sparse_activation']
                pooling_switches = img_data['pooling_switches']
                
                if layer_name == 'conv2':
                    reconstruction = self.reconstruct_from_conv2(sparse_activation, pooling_switches)
                else:
                    print(f"Reconstruction from {layer_name} not implemented yet")
                    continue
                
                # Store reconstruction data
                reconstruction_data = {
                    'rank': i,
                    'image_info': img_data['image_info'],
                    'original_image_tensor': img_data['image_tensor'],
                    'sparse_activation': sparse_activation,
                    'reconstruction': reconstruction.clone(),
                    'max_activation': img_data['max_activation']
                }
                
                reconstructions.append(reconstruction_data)
                print(f"âœ“ Reconstruction complete for rank {i}")
                
            except Exception as e:
                print(f"âœ— Error reconstructing image {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\nâœ“ Channel {channel_idx}: {len(reconstructions)}/{len(images_data)} reconstructions completed")
        return reconstructions
    
    def tensor_to_image(self, tensor):
        """Convert tensor to displayable image (centralized function)"""
        tensor = tensor.cpu().detach()
        
        # Denormalize ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        denorm = tensor.squeeze(0) * std + mean
        denorm = torch.clamp(denorm, 0, 1)
        
        # Convert to numpy and transpose
        img_np = denorm.numpy().transpose(1, 2, 0)
        return img_np
    
    def visualize_reconstruction(self, reconstruction_data, save_path=None):
        """Visualize original image vs reconstruction with consistent styling"""
        original_tensor = reconstruction_data['original_image_tensor']
        reconstruction_tensor = reconstruction_data['reconstruction']
        image_info = reconstruction_data['image_info']
        max_activation = reconstruction_data['max_activation']
        
        # Create figure with consistent styling
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        style_figure(fig)
        
        # Original image
        original_img = self.tensor_to_image(original_tensor)
        ax1.imshow(original_img)
        ax1.set_title(f"Original Image\n{image_info['label']}", 
                      color=VisualizationConfig.TEXT_COLOR)
        style_axis(ax1)
        
        # Reconstruction - apply styling
        reconstruction_img = self.tensor_to_image(reconstruction_tensor)
        styled_recon = apply_deconv_styling(reconstruction_img)
        ax2.imshow(styled_recon)
        ax2.set_title(f"Deconv Reconstruction\nActivation: {max_activation:.2f}",
                      color=VisualizationConfig.TEXT_COLOR)
        style_axis(ax2)
        
        plt.suptitle(f"Deconvolutional Visualization - {image_info['image_id']}", 
                     fontsize=14, color=VisualizationConfig.TEXT_COLOR)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                       facecolor=VisualizationConfig.FIGURE_BG)
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        return fig

# ==== EXECUTE STEP 5 (DYNAMIC CHANNELS) ====
print("=== STEP 5: DECONVOLUTIONAL RECONSTRUCTION (DYNAMIC) ===")

# Initialize deconvolutional network
deconv_net = DeconvolutionalNetwork(model)

# Pick how many channels to visualize end-to-end
TOP_K_CHANNELS = 2  # change to whatever (e.g., 2 for "paper-style two rows")

# Derive the strongest channels from Step 2 results
if 'top_channels' in globals() and top_channels:
    highest_signal_channels = [c['channel'] for c in top_channels[:TOP_K_CHANNELS]]
else:
    # Fallback: query analyzer now
    top_list = analyzer.get_top_channels('conv2', top_n=TOP_K_CHANNELS)
    highest_signal_channels = [c['channel'] for c in top_list]

print("Selected highest-signal channels:", highest_signal_channels)

# Quick smoke test on the first channel only
test_ch = highest_signal_channels[0]
print(f"Testing reconstruction with Channel {test_ch}, first image...")
test_recons = deconv_net.reconstruct_all_images_for_channel('conv2', test_ch, activation_creator)

if test_recons:
    print(f"\nâœ“ Test successful! Got {len(test_recons)} reconstruction(s)")
    
    # Optional: peek with styled visualization
    if len(test_recons) > 0:
        print("\nVisualizing first reconstruction with styling...")
        deconv_net.visualize_reconstruction(test_recons[0], "test_reconstruction_styled.png")
    
    # Full reconstruction for all selected channels
    reconstructions_by_channel = {}
    for ch in highest_signal_channels:
        print(f"\nReconstructing full set for Channel {ch}...")
        reconstructions_by_channel[ch] = deconv_net.reconstruct_all_images_for_channel(
            'conv2', ch, activation_creator
        )
    
    print("\nâœ… STEP 5 COMPLETE!")
    for ch, recs in reconstructions_by_channel.items():
        print(f"âœ… Channel {ch}: {len(recs)} reconstructions")
    
    # Keep backward-compatibility variables if Step 6 still expects them
    if len(highest_signal_channels) >= 1:
        channelA = highest_signal_channels[0]
        channelA_all = reconstructions_by_channel[channelA]
    if len(highest_signal_channels) >= 2:
        channelB = highest_signal_channels[1]
        channelB_all = reconstructions_by_channel[channelB]
        
else:
    print("â�Œ Test failed, debugging needed")

print("âœ… Deconvolutional visualizations show what input patterns trigger each feature map")
print("âœ… Ready for Step 6: Create final visualization tiles")


# ==== STEP 6: CREATE FINAL ZEILER & FERGUS STYLE VISUALIZATION (DYNAMIC) ====
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec

class ZeilerFergusVisualizer:
    """Step 6: Create final visualization matching Figure 2 from the paper (dynamic channels)"""
    def __init__(self, deconv_net):
        self.deconv_net = deconv_net

    def tensor_to_image(self, tensor, denormalize=True):
        """Convert tensor to displayable image with centralized styling"""
        t = tensor.detach().cpu()
        if denormalize:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            t = torch.clamp(t.squeeze(0) * std + mean, 0, 1)
        else:
            t = torch.clamp(t.squeeze(0), 0, 1)
        return t.numpy().transpose(1, 2, 0)

    def create_zeiler_fergus_visualization(self, reconstructions_by_channel, channels,
                                           num_show=9, save_path="zeiler_fergus_visualization.png"):
        """
        Paper-style: for each channel (row), show a 3x3 grid of (reconstruction, original) pairs.
        Uses up to the first 2 channels to match the paper layout.
        """
        print("\n=== STEP 6: CREATING ZEILER & FERGUS STYLE VISUALIZATION ===")
        assert len(channels) >= 1, "No channels provided to visualize."
        channels = channels[:2]
        rows = len(channels)
        
        fig = plt.figure(figsize=(20, 6 * rows))
        style_figure(fig)

        for r, ch in enumerate(channels, start=1):
            recs = (reconstructions_by_channel.get(ch) or [])[:num_show]
            print(f"Creating Channel {ch} visualization with {len(recs)} items...")
            self._create_channel_visualization(fig, recs, channel_idx=ch,
                                               subplot_rows=rows, subplot_start=r)

        fig.suptitle('Deconvolutional Network Visualization - Conv2 Feature Maps\n'
                     'Reproducing Zeiler & Fergus "Visualizing and Understanding Convolutional Networks"',
                     fontsize=16, fontweight='bold', y=0.98, 
                     color=VisualizationConfig.TEXT_COLOR)
        
        fig.text(0.02, 0.5, 'Conv2\nLayer', rotation=90, fontsize=14,
                 fontweight='bold', ha='center', va='center',
                 color=VisualizationConfig.TEXT_COLOR)

        plt.tight_layout(rect=[0.03, 0.02, 0.97, 0.95])
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor=VisualizationConfig.FIGURE_BG, edgecolor='none')
        print(f"âœ“ Final visualization saved to {save_path}")
        plt.show()
        return fig

    def _create_channel_visualization(self, fig, reconstructions, channel_idx,
                                      subplot_rows, subplot_start):
        """Single channel block (3x3) with recon + original pairs."""
        num_show = len(reconstructions)
        for i, rd in enumerate(reconstructions):
            row_in_channel = i // 3
            col = i % 3

            # Reconstruction tile with styling
            ax = fig.add_subplot(subplot_rows * 3, 6,
                                 (row_in_channel * 6) + (col * 2) + 1 + (subplot_start - 1) * 18)
            
            recon_img = self.tensor_to_image(rd['reconstruction'])
            styled_recon = apply_deconv_styling(recon_img)
            ax.imshow(styled_recon)
            ax.set_title(f'Rank {i+1}\nReconst.', fontsize=10, fontweight='bold',
                        color=VisualizationConfig.TEXT_COLOR)
            style_axis(ax)

            # Original tile
            ax2 = fig.add_subplot(subplot_rows * 3, 6,
                                  (row_in_channel * 6) + (col * 2) + 2 + (subplot_start - 1) * 18)
            ax2.imshow(self.tensor_to_image(rd['original_image_tensor']))
            act = rd['max_activation']
            lbl = rd['image_info']['label']
            ax2.set_title(f'Original\n{lbl[:12]}...\nAct: {act:.1f}', fontsize=9,
                         color=VisualizationConfig.TEXT_COLOR)
            style_axis(ax2)

        y_pos = 0.75 if subplot_start == 1 else 0.25
        fig.text(0.5, y_pos + 0.15, f'Channel {channel_idx} - Top {num_show} Activations',
                 fontsize=14, fontweight='bold', ha='center',
                 color=VisualizationConfig.TEXT_COLOR)

    def create_paper_style_grid(self, reconstructions_by_channel, channels,
                                top_show=9, save_path="paper_style_grid.png"):
        """
        Compact grid like the paper:
        Left 3 columns = reconstructions, Right 3 columns = originals (top 3 each).
        Shows up to 2 channels (rows).
        """
        print("\n=== CREATING PAPER-STYLE GRID LAYOUT ===")
        assert len(channels) >= 1, "No channels provided to visualize."
        channels = channels[:2]

        fig, axes = plt.subplots(len(channels), 6, figsize=(18, 6 * len(channels)))
        style_figure(fig)
        
        if len(channels) == 1:
            axes = np.expand_dims(axes, axis=0)

        for r, ch in enumerate(channels):
            recs = (reconstructions_by_channel.get(ch) or [])[:top_show]
            print(f"Processing Channel {ch}...")

            # Left: 3 reconstructions with styling
            for i in range(3):
                ax = axes[r, i]
                style_axis(ax)
                if i < len(recs):
                    recon_img = self.tensor_to_image(recs[i]['reconstruction'])
                    styled_recon = apply_deconv_styling(recon_img)
                    ax.imshow(styled_recon)
                    ax.set_title(f'Ch{ch} R{i+1}\nReconst', fontsize=10, fontweight='bold',
                                color=VisualizationConfig.TEXT_COLOR)

            # Right: 3 originals
            for i in range(3):
                ax = axes[r, i + 3]
                style_axis(ax)
                if i < len(recs):
                    img = self.tensor_to_image(recs[i]['original_image_tensor'])
                    act = recs[i]['max_activation']
                    lbl = recs[i]['image_info']['label'].replace(' ', '\n')[:15]
                    ax.imshow(img)
                    ax.set_title(f'{lbl}\nAct: {act:.1f}', fontsize=9,
                                color=VisualizationConfig.TEXT_COLOR)

        fig.text(0.5, 0.95, 'Deconvolutional Network Feature Visualization',
                 fontsize=16, fontweight='bold', ha='center',
                 color=VisualizationConfig.TEXT_COLOR)
        fig.text(0.25, 0.02, 'Reconstructions', fontsize=12, fontweight='bold', ha='center',
                 color=VisualizationConfig.TEXT_COLOR)
        fig.text(0.75, 0.02, 'Original Images', fontsize=12, fontweight='bold', ha='center',
                 color=VisualizationConfig.TEXT_COLOR)

        line = plt.Line2D([0.5, 0.5], [0.05, 0.92], color='gray', linestyle='--',
                          alpha=0.5, transform=fig.transFigure)
        fig.add_artist(line)

        plt.tight_layout(rect=[0, 0.05, 1, 0.92])
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor=VisualizationConfig.FIGURE_BG)
        print(f"âœ“ Paper-style grid saved to {save_path}")
        plt.show()
        return fig

    def create_detailed_comparison(self, reconstructions_by_channel, channels,
                                   top_k=6, save_path="detailed_comparison.png"):
        """
        Detailed side-by-side comparisons. Shows top_k per channel (default 6).
        Supports any number of channels; defaults to first two for a compact page.
        """
        print("\n=== CREATING DETAILED CHANNEL COMPARISON ===")
        assert len(channels) >= 1, "No channels provided to visualize."
        channels = channels[:2]

        rows_total = top_k * len(channels)
        fig, axes = plt.subplots(rows_total, 4, figsize=(16, 4 * rows_total))
        style_figure(fig)
        
        if rows_total == 1:
            axes = np.expand_dims(axes, axis=0)

        for block, ch in enumerate(channels):
            recs = (reconstructions_by_channel.get(ch) or [])[:top_k]
            start_row = block * top_k
            title = f"Channel {ch}"
            print(f"Creating detailed view for {title}...")

            for i, rd in enumerate(recs):
                row = start_row + i

                # Original
                axes[row, 0].imshow(self.tensor_to_image(rd['original_image_tensor']))
                axes[row, 0].set_title('Original Image', fontsize=10, fontweight='bold',
                                      color=VisualizationConfig.TEXT_COLOR)
                style_axis(axes[row, 0])

                # Reconstruction with styling
                recon_img = self.tensor_to_image(rd['reconstruction'])
                styled_recon = apply_deconv_styling(recon_img)
                axes[row, 1].imshow(styled_recon)
                axes[row, 1].set_title('Deconv Reconstruction', fontsize=10, fontweight='bold',
                                      color=VisualizationConfig.TEXT_COLOR)
                style_axis(axes[row, 1])

                # Enhanced with config
                enhanced_factor = VisualizationConfig.STANDARD_ENHANCE_FACTOR
                enh = np.clip(styled_recon * enhanced_factor, 0, 1)
                axes[row, 2].imshow(enh)
                axes[row, 2].set_title(f'Enhanced ({enhanced_factor:.1f}x)', fontsize=10, fontweight='bold',
                                      color=VisualizationConfig.TEXT_COLOR)
                style_axis(axes[row, 2])

                # Info panel
                info = rd.get('image_info', {})
                axes[row, 3].text(0.1, 0.8, f"Image: {info.get('image_id', '?')}",
                                  transform=axes[row, 3].transAxes, fontsize=8,
                                  color=VisualizationConfig.TEXT_COLOR)
                axes[row, 3].text(0.1, 0.6, f"Label: {info.get('label', '?')}",
                                  transform=axes[row, 3].transAxes, fontsize=8,
                                  color=VisualizationConfig.TEXT_COLOR)
                axes[row, 3].text(0.1, 0.4, f"Activation: {rd['max_activation']:.3f}",
                                  transform=axes[row, 3].transAxes, fontsize=8, fontweight='bold',
                                  color=VisualizationConfig.TEXT_COLOR)
                axes[row, 3].text(0.1, 0.2, f"Rank: {i+1}/{top_k}",
                                  transform=axes[row, 3].transAxes, fontsize=8,
                                  color=VisualizationConfig.TEXT_COLOR)
                style_axis(axes[row, 3])

            # Section label
            block_center_y = 1.0 - ((start_row + top_k/2) / rows_total)
            fig.text(0.02, block_center_y, title, rotation=90,
                     fontsize=12, fontweight='bold', ha='center', va='center',
                     color=VisualizationConfig.TEXT_COLOR)

        plt.suptitle('Detailed Deconvolutional Analysis - Conv2 Feature Maps',
                     fontsize=16, fontweight='bold', color=VisualizationConfig.TEXT_COLOR)
        plt.tight_layout(rect=[0.05, 0.02, 1, 0.95])
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor=VisualizationConfig.FIGURE_BG)
        print(f"âœ“ Detailed comparison saved to {save_path}")
        plt.show()
        return fig



# ==== EXECUTE STEP 6 ====
print("=== STEP 6: CREATING FINAL VISUALIZATIONS ===")

visualizer = ZeilerFergusVisualizer(deconv_net)

# From Step 5:
# - highest_signal_channels: list[int], in descending â€œstrengthâ€� order
# - reconstructions_by_channel: dict[int -> list[reconstruction_data]]
channels = highest_signal_channels[:2]  # keep two rows like the paper
print("Channels to visualize:", channels)
print({ch: len(reconstructions_by_channel.get(ch, [])) for ch in channels})
assert all(len(reconstructions_by_channel.get(ch, [])) > 0 for ch in channels), \
    "One of the channels has 0 reconstructions."

# 1) Main Z&F-style figure (3x3 per channel)
main_fig = visualizer.create_zeiler_fergus_visualization(
    reconstructions_by_channel, channels,
    num_show=9,
    save_path="zeiler_fergus_conv2_visualization.png"
)

# 2) Compact paper-style grid (3 recon + 3 originals per channel)
grid_fig = visualizer.create_paper_style_grid(
    reconstructions_by_channel, channels,
    top_show=9,
    save_path="conv2_paper_style_grid.png"
)

# 3) Detailed comparison (top-6 per channel)
detail_fig = visualizer.create_detailed_comparison(
    reconstructions_by_channel, channels,
    top_k=6,
    save_path="conv2_detailed_comparison.png"
)

print("\nâœ… STEP 6 COMPLETE!")
print("âœ… Created three different visualizations:")
print("   1. Main Zeiler & Fergus style: zeiler_fergus_conv2_visualization.png")
print("   2. Paper-style grid: conv2_paper_style_grid.png")
print("   3. Detailed comparison: conv2_detailed_comparison.png")

print("\n=== VISUALIZATION SUMMARY ===")
print("Layer analyzed: Conv2")
print(f"Channels visualized: {len(channels)} (strongest globally)")
print("Images per channel (main fig): 9 (top activations)")
print(f"Total reconstructions shown: {sum(len(reconstructions_by_channel.get(ch, [])[:9]) for ch in channels)}")
print("Deconv path: Conv2 â†’ (pool2â†©) â†’ Conv1 â†’ Input (224x224)")


# ==== SPATIAL MAPPING VISUALIZATION WITH 64x64 SQUARES (DYNAMIC, STYLED) ====
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class SpatialMappingVisualizer:
    """Visualize spatial correspondence between original images and deconv reconstructions"""
    def __init__(self, deconv_net):
        self.deconv_net = deconv_net

    def tensor_to_image(self, tensor, denormalize=True):
        """Convert tensor to displayable image (consistent with other classes)"""
        t = tensor.detach().cpu()
        if denormalize:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            t = torch.clamp(t.squeeze(0) * std + mean, 0, 1)
        else:
            t = torch.clamp(t.squeeze(0), 0, 1)
        return t.numpy().transpose(1, 2, 0)

    def calculate_receptive_field_mapping(self, conv2_location,
                                          conv2_size=(27, 27), input_size=(224, 224)):
        """Map a conv2 (y,x) position to input-image center coordinates (AlexNet conv1â†’pool1â†’conv2)."""
        stride_total = 8   # 4 (conv1) * 2 (pool1)
        rf_size = 51       # combined RF at conv2
        y, x = conv2_location
        cx = x * stride_total + rf_size // 2
        cy = y * stride_total + rf_size // 2
        return cy, cx

    def find_max_activation_location(self, sparse_activation, channel_idx):
        """Find spatial argmax in [1, 192, 27, 27] for a given channel."""
        ch_map = sparse_activation[0, channel_idx]  # [27,27]
        max_val = torch.max(ch_map).item()
        locs = (ch_map == max_val).nonzero(as_tuple=False)
        if len(locs) > 0:
            y, x = locs[0].tolist()
            return y, x, max_val
        return 13, 13, 0.0  # fallback center

    def create_spatial_mapping_visualization(self, reconstructions_by_channel, channels,
                                             num_show=6, gap_rows=2,
                                             save_path="spatial_mapping_with_squares.png"):
        """
        Dynamic: dict[channel] -> list[reconstruction_data], and a list of channels.
        Renders up to 2 channels, each with `num_show` items.
        `gap_rows` adds blank spacer rows between channel blocks.
        """
        print("\n=== SPATIAL MAPPING VISUALIZATION ===")
        assert len(channels) >= 1, "No channels provided to visualize."
        channels = channels[:2]

        rows_per_block = num_show
        rows_total = rows_per_block * len(channels) + max(0, len(channels) - 1) * gap_rows

        fig, axes = plt.subplots(rows_total, 6, figsize=(18, max(1, rows_total) * 3.2))
        style_figure(fig)
        
        if rows_total == 1:
            axes = np.expand_dims(axes, 0)  # ensure 2D indexing

        for block, ch in enumerate(channels):
            recs = (reconstructions_by_channel.get(ch) or [])[:num_show]
            print(f"\nProcessing Channel {ch} (items: {len(recs)})...")
            start_row = block * (rows_per_block + gap_rows)

            for i, rd in enumerate(recs):
                row = start_row + i

                # Tensors/images
                orig_img = self.tensor_to_image(rd['original_image_tensor'])
                recon_img = self.tensor_to_image(rd['reconstruction'])
                sparse = rd['sparse_activation']

                # Locate max in conv2 for this channel
                max_y, max_x, max_val = self.find_max_activation_location(sparse, ch)
                print(f"  Rank {i+1}: max {max_val:.3f} at conv2 ({max_y}, {max_x})")

                # Map to input coords + clamp 64x64 crop
                cy, cx = self.calculate_receptive_field_mapping((max_y, max_x))
                sq = 64
                top  = int(np.clip(cy - sq // 2, 0, 224 - sq))
                left = int(np.clip(cx - sq // 2, 0, 224 - sq))

                # 0: original + red square
                axes[row, 0].imshow(orig_img)
                axes[row, 0].set_title(f'Ch{ch} Rank {i+1}\nOriginal', fontsize=10, fontweight='bold',
                                      color=VisualizationConfig.TEXT_COLOR)
                axes[row, 0].add_patch(patches.Rectangle((left, top), sq, sq,
                                                         linewidth=2, edgecolor='red', facecolor='none'))
                style_axis(axes[row, 0])

                # 1: reconstruction + same square with styling
                styled_recon = apply_deconv_styling(recon_img)
                axes[row, 1].imshow(styled_recon)
                axes[row, 1].set_title(f'Deconv\nAct: {max_val:.1f}', fontsize=10, fontweight='bold',
                                      color=VisualizationConfig.TEXT_COLOR)
                axes[row, 1].add_patch(patches.Rectangle((left, top), sq, sq,
                                                         linewidth=2, edgecolor='red', facecolor='none'))
                style_axis(axes[row, 1])

                # 2: original crop
                axes[row, 2].imshow(orig_img[top:top+sq, left:left+sq])
                axes[row, 2].set_title('Original\nCrop 64Ã—64', fontsize=10, fontweight='bold',
                                      color=VisualizationConfig.TEXT_COLOR)
                style_axis(axes[row, 2])

                # 3: reconstruction crop with styling
                rc = styled_recon[top:top+sq, left:left+sq]
                axes[row, 3].imshow(rc)
                axes[row, 3].set_title('Deconv\nCrop 64Ã—64', fontsize=10, fontweight='bold',
                                      color=VisualizationConfig.TEXT_COLOR)
                style_axis(axes[row, 3])

                # 4: enhanced crop using config
                enhanced_factor = VisualizationConfig.STANDARD_ENHANCE_FACTOR
                enh = np.clip(rc * enhanced_factor, 0, 1)
                axes[row, 4].imshow(enh)
                axes[row, 4].set_title(f'Enhanced\nCrop ({enhanced_factor:.1f}Ã—)', fontsize=10, fontweight='bold',
                                      color=VisualizationConfig.TEXT_COLOR)
                style_axis(axes[row, 4])

                # 5: info panel
                info = rd.get('image_info', {})
                info_text = (
                    f"Image: {info.get('image_id','?')}\n"
                    f"Label: {info.get('label','?')}\n"
                    f"Conv2: ({max_y},{max_x})\n"
                    f"Input: ({cy},{cx})\n"
                    f"Crop: [{top}:{top+sq}, {left}:{left+sq}]"
                )
                axes[row, 5].text(0.1, 0.5, info_text, transform=axes[row, 5].transAxes,
                                  fontsize=8, va='center', family='monospace',
                                  color=VisualizationConfig.TEXT_COLOR)
                axes[row, 5].set_title('Coordinates', fontsize=10, fontweight='bold',
                                      color=VisualizationConfig.TEXT_COLOR)
                style_axis(axes[row, 5])

            # Blank spacer rows between channel blocks
            for gr in range(gap_rows):
                gap_row = start_row + rows_per_block + gr
                if gap_row < rows_total:
                    for c in range(6):
                        style_axis(axes[gap_row, c])

            # Side label at the vertical center of this block
            center_row = start_row + rows_per_block / 2 - 0.5
            y_norm = 1.0 - (center_row + 0.5) / rows_total
            fig.text(0.02, y_norm, f'Channel {ch}', rotation=90, fontsize=12,
                     fontweight='bold', ha='center', va='center',
                     color=VisualizationConfig.TEXT_COLOR)

        plt.suptitle('Spatial Mapping: Conv2 Activations â†’ Input Regions\n'
                     'Red squares show corresponding 64Ã—64 regions',
                     fontsize=14, fontweight='bold', color=VisualizationConfig.TEXT_COLOR)
        plt.tight_layout(rect=[0.05, 0.02, 1, 0.94])
        fig.subplots_adjust(hspace=0.8)  # extra vertical space between rows
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor=VisualizationConfig.FIGURE_BG)
        print(f"âœ“ Spatial mapping visualization saved to {save_path}")
        plt.show()
        return fig

    def create_receptive_field_diagram(self):
        """Create a diagram showing the receptive field mapping"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        style_figure(fig)

        # Conv2 feature map
        conv2_grid = np.zeros((27, 27))
        conv2_grid[13, 13] = 1  # center
        ax1.imshow(conv2_grid, cmap='Blues')
        ax1.set_title('Conv2 Feature Map\n27Ã—27 spatial resolution', fontweight='bold',
                     color=VisualizationConfig.TEXT_COLOR)
        ax1.set_xlabel('X coordinate', color=VisualizationConfig.TEXT_COLOR)
        ax1.set_ylabel('Y coordinate', color=VisualizationConfig.TEXT_COLOR)
        for i in range(28):
            ax1.axhline(i - 0.5, color='gray', alpha=0.3, linewidth=0.5)
            ax1.axvline(i - 0.5, color='gray', alpha=0.3, linewidth=0.5)
        ax1.add_patch(patches.Rectangle((12.5, 12.5), 1, 1,
                                        linewidth=1, edgecolor='red', facecolor='red', alpha=0.5))
        ax1.text(13, 13, '(13,13)', ha='center', va='center', fontweight='bold', color='white')
        style_axis(ax1)

        # Input image
        input_grid = np.zeros((224, 224, 3))
        ax2.imshow(input_grid)
        ax2.set_title('Input Image\n224Ã—224 pixels', fontweight='bold',
                     color=VisualizationConfig.TEXT_COLOR)
        ax2.set_xlabel('X coordinate', color=VisualizationConfig.TEXT_COLOR)
        ax2.set_ylabel('Y coordinate', color=VisualizationConfig.TEXT_COLOR)

        # RF + crop at center
        cy, cx = self.calculate_receptive_field_mapping((13, 13))
        rf_size = 51
        rf_top, rf_left = cy - rf_size // 2, cx - rf_size // 2
        ax2.add_patch(patches.Rectangle((rf_left, rf_top), rf_size, rf_size,
                                        linewidth=1, edgecolor='red', facecolor='none'))
        ax2.text(cx, cy, f'RF: {rf_size}Ã—{rf_size}', ha='center', va='center',
                 fontweight='bold', color='blue',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        crop = 64
        ct, cl = cy - crop // 2, cx - crop // 2
        ax2.add_patch(patches.Rectangle((cl, ct), crop, crop,
                                        linewidth=2, edgecolor='red', facecolor='none'))
        ax2.text(cx, ct - 10, f'Crop: {crop}Ã—{crop}', ha='center', va='top',
                 fontweight='bold', color='red')
        style_axis(ax2)

        plt.tight_layout()
        plt.show()

        print("Receptive field mapping:")
        print(f"  Conv2 position (13,13) â†’ input center ({cy},{cx})")
        print(f"  Receptive field size: {rf_size}Ã—{rf_size}")
        print(f"  Crop region: {crop}Ã—{crop}")

# ==== TIGHT DUAL CROPS â€” SINGLE CHANNEL FIG (PAPER-STYLED DECONV) ====
def make_tight_dual_crops_single_channel(reconstructions_by_channel,
                                         channel, num_show=6, crop_size=64,
                                         save_path=None):
    """
    One figure per channel:
      2 rows Ã— 6 cols â€” left 3 cols = DECONV crops, right 3 cols = ORIGINAL crops.
    """
    recs = (reconstructions_by_channel.get(channel) or [])[:num_show]
    assert len(recs) > 0, f"No reconstructions for channel {channel}"

    def _find_max_loc(sparse_activation, ch):
        m = sparse_activation[0, ch]  # [27, 27]
        vmax = torch.max(m).item()
        ys, xs = (m == vmax).nonzero(as_tuple=True)
        y = int(ys[0]) if len(ys) else 13
        x = int(xs[0]) if len(xs) else 13
        return y, x, vmax

    def _map_conv2_to_input(yx, stride_total=8, rf_size=51):
        y, x = yx
        return int(y * stride_total + rf_size // 2), int(x * stride_total + rf_size // 2)

    def _crop_coords(rd, ch, crop_size):
        my, mx, _ = _find_max_loc(rd['sparse_activation'], ch)
        cy, cx = _map_conv2_to_input((my, mx))
        top  = int(np.clip(cy - crop_size // 2, 0, 224 - crop_size))
        left = int(np.clip(cx - crop_size // 2, 0, 224 - crop_size))
        return top, left

    def _tensor_to_img(t, denorm=True):
        t = t.detach().cpu()
        if denorm:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            t = torch.clamp(t.squeeze(0) * std + mean, 0, 1)
        else:
            t = torch.clamp(t.squeeze(0), 0, 1)
        return t.numpy().transpose(1, 2, 0)

    rows, cols = 2, 6
    fig = plt.figure(figsize=(cols*2.4, rows*2.4))
    style_figure(fig)
    
    gs = gridspec.GridSpec(rows, cols, figure=fig,
                           left=0.02, right=0.98, top=0.9, bottom=0.06,
                           wspace=0.02, hspace=0.08)

    for i, rd in enumerate(recs):
        r = i // 3
        c = i % 3

        top, left = _crop_coords(rd, channel, crop_size)
        orig   = _tensor_to_img(rd['original_image_tensor'], True)
        deconv = _tensor_to_img(rd['reconstruction'], True)

        orig_crop   = orig[top:top+crop_size,   left:left+crop_size]
        deconv_crop = deconv[top:top+crop_size, left:left+crop_size]
        
        # Apply centralized styling only to deconv crop
        deconv_crop = apply_deconv_styling(deconv_crop)

        # Deconv crop (left half)
        axL = fig.add_subplot(gs[r, c])
        axL.imshow(deconv_crop)
        style_axis(axL)

        # Original crop (right half)
        axR = fig.add_subplot(gs[r, c+3])
        axR.imshow(orig_crop)
        style_axis(axR)

    fig.text(0.25, 0.95, 'Deconv crops', color=VisualizationConfig.TEXT_COLOR,
             ha='center', va='top', fontsize=11, fontweight='bold')
    fig.text(0.75, 0.95, 'Original crops', color=VisualizationConfig.TEXT_COLOR,
             ha='center', va='top', fontsize=11, fontweight='bold')
    fig.text(0.01, 0.5, f'Channel {channel}', color=VisualizationConfig.TEXT_COLOR,
             rotation=90, va='center', ha='left', fontsize=11, fontweight='bold')

    if save_path is None:
        save_path = f'conv2_tight_dual_crops_ch{channel}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
               facecolor=VisualizationConfig.FIGURE_BG)
    print(f"âœ“ Saved: {save_path}  â€¢  channel={channel}  â€¢  n={len(recs)}")
    plt.show()
    return fig

# ==== EXECUTE SPATIAL MAPPING VISUALIZATION ====
print("=== SPATIAL MAPPING VISUALIZATION ===")
spatial_viz = SpatialMappingVisualizer(deconv_net)

print("Creating receptive field mapping diagram...")
spatial_viz.create_receptive_field_diagram()

print("\nCreating spatial mapping visualization...")
channels = highest_signal_channels[:2]  # same two you visualize elsewhere
spatial_fig = spatial_viz.create_spatial_mapping_visualization(
    reconstructions_by_channel, channels,
    num_show=6,          # show 6 per channel
    gap_rows=2,          # add two blank spacer rows between channels
    save_path="spatial_mapping_with_squares.png"
)

print("\n=== Run per-channel tight dual crops figs ===")
two_channels = highest_signal_channels[:2]
print("Rendering single-channel figs for:", two_channels)
for ch in two_channels:
    make_tight_dual_crops_single_channel(
        reconstructions_by_channel, ch,
        num_show=6, crop_size=64,
        save_path=f'conv2_tight_dual_crops_ch{ch}.png'
    )

print("\nâœ… SPATIAL MAPPING COMPLETE!")
print("âœ… Red squares show corresponding regions in original and reconstruction")
print("âœ… Cropped patches visualize what the feature truly latched onto")
print("âœ… Matches the Zeiler & Fergus-style spatial correspondence")


# ==== TIGHT DUAL CROPS â€” SINGLE CHANNEL FIG (PAPER-STYLED DECONV) ====
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec

# --- helpers (now using centralized functions) -------------------------

def _tensor_to_img(t, denorm=True):
    """Use centralized tensor conversion (keeping for local compatibility)"""
    t = t.detach().cpu()
    if denorm:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        t = torch.clamp(t.squeeze(0) * std + mean, 0, 1)
    else:
        t = torch.clamp(t.squeeze(0), 0, 1)
    return t.numpy().transpose(1, 2, 0)

def _find_max_loc(sparse_activation, ch):
    m = sparse_activation[0, ch]  # [27, 27]
    vmax = torch.max(m).item()
    ys, xs = (m == vmax).nonzero(as_tuple=True)
    y = int(ys[0]) if len(ys) else 13
    x = int(xs[0]) if len(xs) else 13
    return y, x, vmax

def _map_conv2_to_input(yx, stride_total=8, rf_size=51):
    y, x = yx
    return int(y * stride_total + rf_size // 2), int(x * stride_total + rf_size // 2)

def _crop_coords(rd, ch, crop_size):
    my, mx, _ = _find_max_loc(rd['sparse_activation'], ch)
    cy, cx = _map_conv2_to_input((my, mx))
    top  = int(np.clip(cy - crop_size // 2, 0, 224 - crop_size))
    left = int(np.clip(cx - crop_size // 2, 0, 224 - crop_size))
    return top, left

# --- main figure (now using centralized styling) ----------------------

def make_tight_dual_crops_single_channel(reconstructions_by_channel,
                                         channel, num_show=6, crop_size=64,
                                         save_path=None):
    """
    One figure per channel:
      2 rows Ã— 6 cols â€” left 3 cols = DECONV crops, right 3 cols = ORIGINAL crops.
    Now uses centralized styling configuration.
    """
    recs = (reconstructions_by_channel.get(channel) or [])[:num_show]
    assert len(recs) > 0, f"No reconstructions for channel {channel}"

    rows, cols = 2, 6
    fig = plt.figure(figsize=(cols*2.4, rows*2.4))
    style_figure(fig)  # Apply centralized figure styling
    
    gs = gridspec.GridSpec(rows, cols, figure=fig,
                           left=0.02, right=0.98, top=0.9, bottom=0.06,
                           wspace=0.02, hspace=0.08)

    for i, rd in enumerate(recs):
        r = i // 3
        c = i % 3

        top, left = _crop_coords(rd, channel, crop_size)
        orig   = _tensor_to_img(rd['original_image_tensor'], True)
        deconv = _tensor_to_img(rd['reconstruction'], True)

        orig_crop   = orig[top:top+crop_size,   left:left+crop_size]
        deconv_crop = deconv[top:top+crop_size, left:left+crop_size]
        
        # Apply centralized deconv styling
        deconv_crop = apply_deconv_styling(deconv_crop)

        # Deconv crop (left half)
        axL = fig.add_subplot(gs[r, c])
        axL.imshow(deconv_crop)
        style_axis(axL)  # Apply centralized axis styling

        # Original crop (right half)
        axR = fig.add_subplot(gs[r, c+3])
        axR.imshow(orig_crop)
        style_axis(axR)  # Apply centralized axis styling

    # Use centralized text color configuration
    fig.text(0.25, 0.95, 'Deconv crops', color=VisualizationConfig.TEXT_COLOR,
             ha='center', va='top', fontsize=11, fontweight='bold')
    fig.text(0.75, 0.95, 'Original crops', color=VisualizationConfig.TEXT_COLOR,
             ha='center', va='top', fontsize=11, fontweight='bold')
    fig.text(0.01, 0.5, f'Channel {channel}', color=VisualizationConfig.TEXT_COLOR,
             rotation=90, va='center', ha='left', fontsize=11, fontweight='bold')

    if save_path is None:
        save_path = f'conv2_tight_dual_crops_ch{channel}.png'
    
    # Use centralized background color for saving
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
               facecolor=VisualizationConfig.FIGURE_BG)
    print(f"âœ“ Saved: {save_path}  â€¢  channel={channel}  â€¢  n={len(recs)}")
    plt.show()
    return fig

# === Execute Step 6 Visualizations ===
print("\n=== STEP 6: CREATING ALL FINAL VISUALIZATIONS ===")

# Initialize visualizers
zeiler_viz = ZeilerFergusVisualizer(deconv_net)
spatial_viz = SpatialMappingVisualizer(deconv_net)

# 1. Create Zeiler & Fergus style visualization
print("\n1. Creating Zeiler & Fergus style visualization...")
zeiler_fig = zeiler_viz.create_zeiler_fergus_visualization(
    reconstructions_by_channel, highest_signal_channels,
    num_show=9, save_path="zeiler_fergus_conv2_visualization.png"
)

# 2. Create paper-style grid
print("\n2. Creating paper-style grid layout...")
grid_fig = zeiler_viz.create_paper_style_grid(
    reconstructions_by_channel, highest_signal_channels,
    top_show=9, save_path="paper_style_grid_conv2.png"
)

# 3. Create detailed comparison
print("\n3. Creating detailed channel comparison...")
detailed_fig = zeiler_viz.create_detailed_comparison(
    reconstructions_by_channel, highest_signal_channels,
    top_k=6, save_path="detailed_comparison_conv2.png"
)

# 4. Create spatial mapping visualization
print("\n4. Creating spatial mapping visualization...")
print("Creating receptive field mapping diagram...")
spatial_viz.create_receptive_field_diagram()

print("\nCreating spatial mapping visualization...")
spatial_fig = spatial_viz.create_spatial_mapping_visualization(
    reconstructions_by_channel, highest_signal_channels,
    num_show=6, gap_rows=2, save_path="spatial_mapping_conv2.png"
)

# 5. Create tight dual crops per channel
print("\n5. Creating tight dual crops visualizations...")
two_channels = highest_signal_channels[:2]
print("Rendering single-channel figs for:", two_channels)
for ch in two_channels:
    make_tight_dual_crops_single_channel(
        reconstructions_by_channel, ch,
        num_show=6, crop_size=64,
        save_path=f'conv2_tight_dual_crops_ch{ch}.png'
    )

print("\nâœ… STEP 6 COMPLETE!")
print("âœ… Created comprehensive conv2 feature visualizations")
print("âœ… Generated multiple visualization styles:")
print("   â€¢ Zeiler & Fergus paper-style layout")
print("   â€¢ Paper-style grid comparison")
print("   â€¢ Detailed side-by-side analysis")
print("   â€¢ Spatial mapping with activation regions")
print("   â€¢ Tight dual crops per channel")
print("âœ… All visualizations use consistent styling from VisualizationConfig")
print("âœ… High-quality visualizations saved as PNG files")

print("\nğŸ�¯ CONV2 FEATURE VISUALIZATION PIPELINE COMPLETE!")
print("You now have:")
print("   â€¢ Deconvolutional reconstructions showing what patterns trigger conv2 features")
print("   â€¢ Spatial correspondence between activations and input regions")
print("   â€¢ Multiple visualization formats for different analysis needs")
print("   â€¢ Consistent, publication-ready styling")
print("\nThe Zeiler & Fergus methodology successfully reveals the learned visual representations in conv2!")

