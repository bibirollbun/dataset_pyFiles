!pip install cairosvg


import pandas as pd
import numpy as np
from wordcloud import WordCloud
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import os
import random
import textwrap
from PIL import Image
from sklearn.mixture import GaussianMixture
import cv2
import time
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.color import label2rgb
import cairosvg
import io
import gc
import re

INPUT_FOLDER_PATH = '/kaggle/input/flickr-image-dataset/flickr30k_images/'

IMAGES_PATH = INPUT_FOLDER_PATH + 'flickr30k_images/' # contains jpg images like "1000092795.jpg"
DESCRIPTIONS_PATH = INPUT_FOLDER_PATH + 'results.csv'

descriptions_df = pd.read_csv(DESCRIPTIONS_PATH, sep='|')
descriptions_df.dropna()
print("There are", descriptions_df.shape[0], "descriptions in this dataset.")
descriptions_df.head()


all_comments = " ".join(descriptions_df[' comment'].astype(str).values)

wordcloud = WordCloud(
    width=800, 
    height=600, 
    background_color='white'
).generate(all_comments)

plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Image Descriptions", fontsize=16)
plt.show()


IMAGES_PATH = '/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images/'

sample_df = descriptions_df.sample(n=5, random_state=42)  # pick any number of rows you like

for idx, row in sample_df.iterrows():
    image_name = row['image_name']
    comment = row[' comment']

    img_path = os.path.join(IMAGES_PATH, image_name)
    
    img = Image.open(img_path)

    width, height = img.size

    # Print out the info
    print(f"Image: {image_name}")
    print(f"Resolution: {width} x {height}")
    print(f"Comment: {comment}")
    print("-"*50)
    
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.title(comment)
    plt.axis('off')
    plt.show()


people_keywords = {"man","woman","men","women","people","person","girl","boy","child","kid","couple","bride",
                   "groom","player","players","children","boys","girls","kids"}
animals_keywords = {"cat","dog","bird","horse","cow","sheep","goat","lion","tiger","elephant","mouse","monkey",
                    "gorilla","bear","pig","duck","goose","whale","fish","cats","dogs","ducks","birds","horses","animal"}
colors_keywords = {"red","blue","green","yellow","orange","purple","pink","black","brown","grey","gray","white"}

def classify_comment(comment: str) -> str:
    """Classify a comment into one of the themes based on keywords."""
    text = str(comment).lower()
    if any(kw in text for kw in people_keywords):
        return "people"
    elif any(kw in text for kw in animals_keywords):
        return "animals"
    elif any(kw in text for kw in colors_keywords):
        return "colors"
    else:
        return "other"

descriptions_df['theme'] = descriptions_df[' comment'].apply(classify_comment)

theme_counts = descriptions_df['theme'].value_counts()/descriptions_df.shape[0]*100
theme_counts = theme_counts.sort_values(ascending=False)

plt.figure(figsize=(8, 6))
theme_counts.plot(kind='bar', rot=0, color=['#0099cc', '#99cc00', '#cc9900', '#999999'])
plt.title("Theme Distribution in Descriptions", fontsize=14)
plt.xlabel("Theme", fontsize=12)
plt.ylabel("Count of Descriptions (%)", fontsize=12)
plt.show()


categories = ["people", "animals", "colors", "other"]

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))

for row_idx, cat in enumerate(categories):
    cat_df = descriptions_df[descriptions_df['theme'] == cat]
    sample_rows = cat_df.sample(n=4, random_state=42)

    for col_idx, (df_index, row) in enumerate(sample_rows.iterrows()):
        image_path = os.path.join(IMAGES_PATH, row['image_name'])
        
        img = Image.open(image_path)

        axes[row_idx, col_idx].imshow(img)
        axes[row_idx, col_idx].axis('off')

        wrapped_comment = textwrap.fill(row[' comment'], width=30)
        axes[row_idx, col_idx].set_title(wrapped_comment, fontsize=10)

plt.tight_layout()
plt.show()


animal_counts = {animal: 0 for animal in animals_keywords}

for comment in descriptions_df[' comment']:
    text = str(comment).lower()
    for animal in animals_keywords:
        if animal in text:
            animal_counts[animal] += 1

animal_counts_df = pd.DataFrame(animal_counts.items(), columns=["animal", "count"])
animal_counts_df = animal_counts_df.sort_values("count", ascending=False)

plt.figure(figsize=(10, 6))
bars = plt.bar(animal_counts_df["animal"], animal_counts_df["count"])
plt.xlabel("Animal")
plt.ylabel("Number of Descriptions Mentioning Animal (% whole dataset)")
plt.title("Animal Mentions in Descriptions")
plt.xticks(rotation=45)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,   
        height,                              
        f"{height / descriptions_df.shape[0] * 100:.2f}",      
        ha='center', va='bottom',           
        fontsize=9
    )

plt.tight_layout()
plt.show()


# Pick one random row
sample_row = descriptions_df.sample(n=1, random_state=42).iloc[0]
image_name = sample_row['image_name']
image_path = os.path.join(IMAGES_PATH, image_name)
comment = sample_row[' comment']

print(f"Randomly selected image: {image_name}")
print(f"Description: {comment}")

# Open the image
img = Image.open(image_path)
img_np = np.array(img)  # convert PIL image to a NumPy array (height x width x channels)
plt.imshow(img_np)


# ========================
# Helper Functions
# ========================

def compute_entropy(image):
    """
    Compute the Shannon entropy (in bits) of an image.
    If the image is color, it is first converted to grayscale.
    """
    if image.ndim == 3:
        # Convert to grayscale using luminosity weights
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        gray = image
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return entropy

def apply_gmm(image, n_components):
    """
    Apply Gaussian Mixture on the image's pixels (preserving colors).
    """
    h, w, c = image.shape
    pixels = image.reshape(-1, c)
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(pixels)
    labels = gmm.predict(pixels)
    cluster_centers = gmm.means_
    compressed_pixels = cluster_centers[labels].reshape(h, w, c)
    compressed_pixels = np.clip(compressed_pixels, 0, 255).astype(np.uint8)
    return compressed_pixels

def apply_opening(image, kernel):
    """Apply morphological opening on each channel."""
    opened = np.empty_like(image)
    for i in range(image.shape[-1]):
        opened[..., i] = cv2.morphologyEx(image[..., i], cv2.MORPH_OPEN, kernel)
    return opened

def apply_closing(image, kernel):
    """Apply morphological closing on each channel."""
    closed = np.empty_like(image)
    for i in range(image.shape[-1]):
        closed[..., i] = cv2.morphologyEx(image[..., i], cv2.MORPH_CLOSE, kernel)
    return closed

# ------------------------
# New Helper Functions
# ------------------------

def downscale_upscale(image, downscale_factor=2, resample=Image.NEAREST):
    """
    Downscale the image by 'downscale_factor' then upscale back to the original size.
    By default, uses nearest-neighbor to preserve blocky shapes and reduce detail.
    """
    pil_image = Image.fromarray(image)
    width, height = pil_image.size

    # Downscale
    new_width = max(1, width // downscale_factor)
    new_height = max(1, height // downscale_factor)
    pil_small = pil_image.resize((new_width, new_height), resample=Image.BILINEAR)

    # Upscale back to original size
    pil_restored = pil_small.resize((width, height), resample=resample)
    return np.array(pil_restored)

def downscale(image, downscale_factor=2, resample=Image.NEAREST):
    """
    Downscale the image by 'downscale_factor' then upscale back to the original size.
    By default, uses nearest-neighbor to preserve blocky shapes and reduce detail.
    """
    pil_image = Image.fromarray(image)
    width, height = pil_image.size

    # Downscale
    new_width = max(1, width // downscale_factor)
    new_height = max(1, height // downscale_factor)
    pil_small = pil_image.resize((new_width, new_height), resample=Image.BILINEAR)

    return np.array(pil_restored)

def pil_color_quantize(image, num_colors=16):
    """
    Quantize using Pillow's built-in method (median-cut or similar).
    """
    pil_image = Image.fromarray(image)
    # Convert to 'P' (palettized) with num_colors
    quantized = pil_image.quantize(colors=num_colors, method=0, kmeans=0)
    # Convert back to 'RGB'
    quantized_rgb = quantized.convert("RGB")
    return np.array(quantized_rgb)

def bilateral_smoothing(image, d=9, sigma_color=75, sigma_space=75):
    """
    Apply a bilateral filter to preserve edges while smoothing within regions.
    - d: Diameter of each pixel neighborhood.
    - sigma_color: Filter sigma in the color space.
    - sigma_space: Filter sigma in the coordinate space.
    """
    # OpenCV's bilateralFilter can process a 3-channel image in one go (BGR).
    # But we have an RGB image, so we either convert to BGR or just apply to each channel.
    # We'll do a simple approach: assume 'image' is in RGB and convert to BGR for bilateralFilter.
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    bgr_smooth = cv2.bilateralFilter(bgr, d, sigma_color, sigma_space)
    # Convert back to RGB
    rgb_smooth = cv2.cvtColor(bgr_smooth, cv2.COLOR_BGR2RGB)
    return rgb_smooth

def superpixel_simplify(image, n_segments=200, compactness=10):
    """
    Partition image into superpixels (SLIC) and replace each region with its average color.
    """
    # SLIC typically expects float in [0,1]
    image_float = image.astype(np.float32) / 255.0
    segments = slic(image_float, n_segments=n_segments, compactness=compactness, start_label=1)
    # Re-color each segment by its average RGB
    out = label2rgb(segments, image_float, kind='avg')
    # Convert back to uint8
    out = (out * 255).astype(np.uint8)
    return out

entropy_original = compute_entropy(img_np)
print(f"Original image entropy: {entropy_original:.2f} bits")

# Define a kernel for morphological ops (if used)
kernel = np.ones((3, 3), np.uint8)

# ========================
# Define Pipelines
# ========================

pipelines = {
    # ------------------------------------------------------
    # 1. GMM pipeline
    # ------------------------------------------------------
    "O+C+Gaussian 8": lambda im: apply_gmm(apply_opening(apply_closing(im, kernel), kernel), 8),
    "Downscale4+O+C+Gaussian 8": lambda im: apply_gmm(apply_opening(apply_closing(downscale_upscale(im, downscale_factor=4, resample=Image.NEAREST), kernel), kernel), 8),
    
    # ------------------------------------------------------
    # 2. Downscale+Upscale
    # ------------------------------------------------------
    "Downscale2+Upscale2(NN)": lambda im: downscale_upscale(im, downscale_factor=2, resample=Image.NEAREST),
    "Downscale4+Upscale4(NN)": lambda im: downscale_upscale(im, downscale_factor=4, resample=Image.NEAREST),
    
    # ------------------------------------------------------
    # 3. Pillow Color Quantization
    # ------------------------------------------------------
    "PIL Quantize 16": lambda im: pil_color_quantize(im, num_colors=16),
    "PIL Quantize 8":  lambda im: pil_color_quantize(im, num_colors=8),
    
    # ------------------------------------------------------
    # 4. Bilateral Smoothing + Color Quant (combo)
    # ------------------------------------------------------
    "Bilateral + PIL Quant16": lambda im: pil_color_quantize(bilateral_smoothing(im), 16),
    
    # ------------------------------------------------------
    # 5. Superpixel Simplify
    # ------------------------------------------------------
    "Superpixel(1600)": lambda im: superpixel_simplify(im, n_segments=1600, compactness=10),
    "Superpixel(800)": lambda im: superpixel_simplify(im, n_segments=800, compactness=10),
    "Superpixel(400)": lambda im: superpixel_simplify(im, n_segments=400, compactness=10),
    "Superpixel(200)": lambda im: superpixel_simplify(im, n_segments=200, compactness=10),
    "Superpixel(100)": lambda im: superpixel_simplify(im, n_segments=100, compactness=10),
}

# Containers for results
pipeline_names = []
execution_times = []
entropies = []

# ========================
# Process Each Pipeline, Display and Save Metrics
# ========================

for name, pipeline in pipelines.items():
    print(f"Processing pipeline: {name}")
    start_time = time.perf_counter()
    processed = pipeline(img_np)
    elapsed = time.perf_counter() - start_time
    ent = compute_entropy(processed)
    
    pipeline_names.append(name)
    execution_times.append(elapsed)
    entropies.append(ent)
    
    # Show the processed image
    plt.figure(figsize=(6, 6))
    plt.imshow(processed)
    plt.title(f"{name}\nTime: {elapsed:.2f} s, Entropy: {ent:.2f} bits")
    plt.axis("off")
    plt.show()

# ========================
# Plot Comparison of Execution Time and Entropy
# ========================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Bar plot for execution times
bars1 = ax1.bar(pipeline_names, execution_times, color='#3399ff')
ax1.set_xlabel("Pipeline")
ax1.set_ylabel("Execution Time (s)")
ax1.set_title("Execution Time per Pipeline")
ax1.tick_params(axis='x', rotation=45)
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height + 0.005, f"{height:.2f}", 
             ha='center', va='bottom')

# Bar plot for entropies
bars2 = ax2.bar(pipeline_names, entropies, color='#ff9933')
ax2.set_xlabel("Pipeline")
ax2.set_ylabel("Entropy (bits)")
ax2.set_title("Image Entropy per Pipeline")
ax2.tick_params(axis='x', rotation=45)
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{height:.2f}", 
             ha='center', va='bottom')

plt.tight_layout()
plt.show()


# Sample 16 images from the dataset
sample_df = descriptions_df.sample(n=16, random_state=42)

def test_superpixel(seg, comp, keep_original=False, apply_OC=False):
    # Set up a 4x4 grid for plotting
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()
    
    for idx, (_, row) in enumerate(tqdm(sample_df.iterrows(), total=16)):
        image_name = row['image_name']
        image_path = os.path.join(IMAGES_PATH, image_name)
        
        # Load the image
        im = Image.open(image_path)
        im_np = np.array(im)
        
        # Drop alpha channel if present
        if im_np.ndim == 3 and im_np.shape[-1] == 4:
            im_np = im_np[..., :3]

        if keep_original:
            simplified_im = im_np
        else:
            # Apply superpixel simplification
            simplified_im = superpixel_simplify(im_np, n_segments=seg, compactness=comp)
            
        if apply_OC:
            simplified_im = apply_closing(apply_opening(simplified_im, kernel), kernel)
        
        # Get the description from the DataFrame. Adjust for column name with a possible leading space.
        description = row[' comment'] if ' comment' in row else row['comment']
        wrapped_description = textwrap.fill(description, width=30)
        
        # Plot the simplified image with the description as its title
        axes[idx].imshow(simplified_im)
        axes[idx].set_title(wrapped_description, fontsize=8)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

test_superpixel(100, 10, keep_original=True)


test_superpixel(800, 10)


test_superpixel(1600, 10)


test_superpixel(1600, 5)


test_superpixel(1600, 20)


test_superpixel(1600, 20, apply_OC=True)


# --------------------------------------------------------------------
# 1) Core function for single-image vectorization
# --------------------------------------------------------------------
def _bitmap_to_svg_layered_core(
    image, 
    max_size_bytes=10000, 
    resize=True, 
    target_size=(384, 384), 
    adaptive_fill=True, 
    num_colors=None,
    background_subdivisions=5
):
    """
    Processes a single image -> SVG (the "core" pipeline).
    Also adds a grid of background rectangles covering the entire image,
    using 'background_subdivisions' to decide how many rectangles across
    and down.
    """
    
    # --- HELPER FUNCTIONS ---
    def compress_hex_color(hex_color):
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        # compress if possible (like #112233 -> #123)
        if (r % 17 == 0) and (g % 17 == 0) and (b % 17 == 0):
            return f'#{r//17:x}{g//17:x}{b//17:x}'
        return hex_color

    def ensure_rgb(np_img):
        """
        Convert np_img to 3-channel RGB if it's grayscale or has an alpha channel.
        """
        if np_img.ndim == 2:
            # Grayscale => replicate channel
            np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
        elif np_img.shape[2] == 4:
            # Drop alpha channel
            np_img = np_img[..., :3]
        elif np_img.shape[2] == 1:
            # Single channel => replicate
            np_img = np.concatenate([np_img]*3, axis=2)
        return np_img

    def extract_features_by_scale(img_np, num_colors=16):
        img_np = ensure_rgb(img_np)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        
        # KMeans color quantization
        pixels = img_np.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            data=pixels, K=num_colors, bestLabels=None, 
            criteria=criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS
        )
        
        palette = centers.astype(np.uint8)
        quantized = palette[labels.flatten()].reshape(img_np.shape)
        
        # For storing final features
        hierarchical_features = []
        
        # Sort color clusters by frequency
        unique_labels, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(-counts)
        sorted_colors = [palette[i] for i in sorted_indices]
        
        center_x, center_y = width / 2, height / 2
        
        # For each sorted color
        for color in sorted_colors:
            color_mask = cv2.inRange(quantized, color, color)
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            hex_color = compress_hex_color(f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}')
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 20:
                    continue
                m = cv2.moments(contour)
                if m["m00"] == 0:
                    continue
                cx = m["m10"] / m["m00"]
                cy = m["m01"] / m["m00"]
                
                # Dist from center (normalized)
                dist_from_center = np.hypot(
                    (cx - center_x) / width,
                    (cy - center_y) / height
                )
                
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Build points string
                pts_str = " ".join([f"{pt[0][0]:.1f},{pt[0][1]:.1f}" for pt in approx])
                
                # Importance
                importance = (
                    area * (1 - dist_from_center) * 
                    (1 / (len(approx) + 1))
                )
                
                hierarchical_features.append({
                    'points': pts_str,
                    'color': hex_color,
                    'area': area,
                    'importance': importance,
                })
        
        hierarchical_features.sort(key=lambda x: x['importance'], reverse=True)
        return hierarchical_features

    def simplify_polygon(points_str, simplification_level):
        """
        Different simplification levels:
            0 -> no change
            1 -> round coords to .1f
            2 -> round coords to integer
            3 -> keep ~half the points (or to integer if too few)
        """
        if simplification_level == 0:
            return points_str
        
        coords = points_str.strip().split()
        if simplification_level == 1:
            return " ".join([
                f"{float(c.split(',')[0]):.1f},{float(c.split(',')[1]):.1f}"
                for c in coords
            ])
        elif simplification_level == 2:
            return " ".join([
                f"{round(float(c.split(',')[0]))},{round(float(c.split(',')[1]))}"
                for c in coords
            ])
        elif simplification_level == 3:
            if len(coords) <= 4:
                # Just round to int
                return " ".join([
                    f"{round(float(c.split(',')[0]))},{round(float(c.split(',')[1]))}"
                    for c in coords
                ])
            else:
                # Keep half the points
                step = min(2, len(coords) // 3)
                reduced = [coords[i] for i in range(0, len(coords), step)]
                if len(reduced) < 3:
                    reduced = coords[:3]
                if coords[-1] not in reduced:
                    reduced.append(coords[-1])
                return " ".join([
                    f"{round(float(c.split(',')[0]))},{round(float(c.split(',')[1]))}"
                    for c in reduced
                ])
        return points_str

    def average_color_in_subregion(np_img, x_start, y_start, w, h):
        """
        Return a 3-channel average (R,G,B), dropping alpha or
        converting grayscale if necessary.
        """
        sub = np_img[y_start:y_start+h, x_start:x_start+w]

        # ensure sub is NxMx3
        sub = ensure_rgb(sub)

        # shape is (h, w, 3)
        mean_vals = sub.mean(axis=(0,1))
        # mean_vals is shape (3,) => (R,G,B)
        return mean_vals
    
    # ---------------- MAIN PIPELINE ---------------
    
    # 1) Possibly pick num_colors if not set
    if num_colors is None:
        if resize:
            pixel_count = target_size[0] * target_size[1]
        else:
            pixel_count = image.size[0] * image.size[1]
        if pixel_count < 65536:
            num_colors = 8
        elif pixel_count < 262144:
            num_colors = 12
        else:
            num_colors = 16

    # 2) Resize if needed
    if resize:
        original_size = image.size
        image = image.resize(target_size, Image.LANCZOS)
    else:
        original_size = image.size

    # Convert to np
    img_np = np.array(image)
    # ensure 3-channel RGB
    img_np = ensure_rgb(img_np)

    height, width = img_np.shape[:2]

    # Build SVG header
    orig_w, orig_h = original_size
    svg_header = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{orig_w}" height="{orig_h}" '
        f'viewBox="0 0 {width} {height}">\n'
    )

    # NxN background squares
    n = background_subdivisions
    cell_w = width / n
    cell_h = height / n
    bg_rects = []

    for row in range(n):
        for col in range(n):
            x0 = int(col * cell_w)
            y0 = int(row * cell_h)
            if col == n - 1:
                w_sub = width - x0
            else:
                w_sub = int(cell_w)
            if row == n - 1:
                h_sub = height - y0
            else:
                h_sub = int(cell_h)

            avg_col = average_color_in_subregion(img_np, x0, y0, w_sub, h_sub)
            # avg_col is (r,g,b)
            r, g, b = avg_col
            hex_col = compress_hex_color(f'#{int(r):02x}{int(g):02x}{int(b):02x}')
            bg_rects.append(
                f'<rect x="{x0}" y="{y0}" width="{w_sub}" height="{h_sub}" fill="{hex_col}"/>\n'
            )

    svg_base = svg_header + "".join(bg_rects)
    svg_footer = '</svg>'
    base_size = len((svg_base + svg_footer).encode('utf-8'))

    # Extract features
    features = extract_features_by_scale(img_np, num_colors=num_colors)

    # If not adaptive fill, just add polygons until out of space
    if not adaptive_fill:
        svg = svg_base
        for feat in features:
            poly_tag = f'<polygon points="{feat["points"]}" fill="{feat["color"]}" />\n'
            trial_size = len((svg + poly_tag + svg_footer).encode('utf-8'))
            if trial_size > max_size_bytes:
                break
            svg += poly_tag
        svg += svg_footer
        return svg

    # Otherwise do adaptive fill
    def size_of_polygon(points_str, fill_str):
        return len(f'<polygon points="{points_str}" fill="{fill_str}" />\n'.encode('utf-8'))

    def polygon_tag(points_str, fill_str):
        return f'<polygon points="{points_str}" fill="{fill_str}" />\n'

    # Precompute size at each simplification level
    feature_sizes = []
    for feat in features:
        pts0 = feat["points"]
        c = feat["color"]

        s0 = size_of_polygon(pts0, c)
        pts1 = simplify_polygon(pts0, 1)
        s1 = size_of_polygon(pts1, c)
        pts2 = simplify_polygon(pts0, 2)
        s2 = size_of_polygon(pts2, c)
        pts3 = simplify_polygon(pts0, 3)
        s3 = size_of_polygon(pts3, c)

        feature_sizes.append({
            'original': s0,
            'level1': s1,
            'level2': s2,
            'level3': s3
        })

    svg = svg_base
    bytes_used = base_size
    added_features = set()

    # Pass 1: original polygons
    for i, feat in enumerate(features):
        s0 = feature_sizes[i]['original']
        if bytes_used + s0 <= max_size_bytes:
            svg += polygon_tag(feat["points"], feat["color"])
            bytes_used += s0
            added_features.add(i)

    # Pass 2: simplified polygons
    for level in range(1, 4):
        key = f'level{level}'
        for i, feat in enumerate(features):
            if i in added_features:
                continue
            sz = feature_sizes[i][key]
            if bytes_used + sz <= max_size_bytes:
                spoints = simplify_polygon(feat["points"], level)
                svg += polygon_tag(spoints, feat["color"])
                bytes_used += sz
                added_features.add(i)

    svg += svg_footer

    final_size = len(svg.encode('utf-8'))
    if final_size > max_size_bytes:
        # fallback
        fallback_svg = (
            svg_header
            + f'<rect width="{width}" height="{height}" fill="#fff"/>\n'
            + '</svg>'
        )
        return fallback_svg

    return svg


# --------------------------------------------------------------------
# 2) Extended function with iteration_passes (and forced pass2=>sep=1)
# --------------------------------------------------------------------
def bitmap_to_svg_layered(
    image,
    max_size_bytes=10000,
    resize=True,
    target_size=(384, 384),
    adaptive_fill=True,
    num_colors=None,
    separation_factor=2,
    background_subdivisions=5,
    iteration_passes=1
):
    """
    Extended version that:
      1) Can tile the image (separation_factor>1) on the FIRST pass only,
      2) Adds NxN background rectangles,
      3) And can apply multiple passes (iteration_passes),
         re-rasterizing the last pass's SVG result for the next pass.
      4) Forces separation_factor=1 for the second+ pass to avoid repeating tiling.
    """

    def svg_to_png(svg_str):
        # Convert SVG -> PNG in memory
        png_data = cairosvg.svg2png(bytestring=svg_str.encode('utf-8'))
        # Convert PNG bytes -> PIL Image
        return Image.open(io.BytesIO(png_data))
    
    current_image = image

    for pass_idx in range(iteration_passes):
        # Force separation_factor=1 on second+ passes
        if pass_idx == 0:
            local_sep_factor = separation_factor
        else:
            local_sep_factor = 1

        # If local_sep_factor<=1 => single pass
        if local_sep_factor <= 1:
            final_svg = _bitmap_to_svg_layered_core(
                current_image,
                max_size_bytes=max_size_bytes,
                resize=resize,
                target_size=target_size,
                adaptive_fill=adaptive_fill,
                num_colors=num_colors,
                background_subdivisions=background_subdivisions
            )
        else:
            # Tiling approach (first pass only)
            orig_w, orig_h = current_image.size
            tile_w = orig_w // local_sep_factor
            tile_h = orig_h // local_sep_factor

            final_svg_w = orig_w * local_sep_factor
            final_svg_h = orig_h * local_sep_factor

            final_svg_header = (
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{final_svg_w}" height="{final_svg_h}" '
                f'viewBox="0 0 {final_svg_w} {final_svg_h}">\n'
            )
            # build NxN background squares for entire mosaic
            # (just like in _bitmap_to_svg_layered_core)
            big_image = current_image.resize((final_svg_w, final_svg_h), Image.LANCZOS)
            big_np = np.array(big_image)

            # local helper
            def compress_hex_color(hex_color):
                r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
                if (r % 17 == 0) and (g % 17 == 0) and (b % 17 == 0):
                    return f'#{r//17:x}{g//17:x}{b//17:x}'
                return hex_color

            def ensure_rgb(np_img):
                if np_img.ndim == 2:
                    np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
                elif np_img.shape[2] == 4:
                    np_img = np_img[..., :3]
                elif np_img.shape[2] == 1:
                    np_img = np.concatenate([np_img]*3, axis=2)
                return np_img

            big_np = ensure_rgb(big_np)
            n = background_subdivisions
            cell_w = final_svg_w / n
            cell_h = final_svg_h / n

            def average_color_in_subregion(np_img, x_start, y_start, w, h):
                sub = np_img[y_start:y_start+h, x_start:x_start+w]
                sub = ensure_rgb(sub)
                mean_vals = sub.mean(axis=(0,1))
                return mean_vals

            bg_rects = []
            for row in range(n):
                for col in range(n):
                    x0 = int(col * cell_w)
                    y0 = int(row * cell_h)
                    if col == n - 1:
                        w_sub = final_svg_w - x0
                    else:
                        w_sub = int(cell_w)
                    if row == n - 1:
                        h_sub = final_svg_h - y0
                    else:
                        h_sub = int(cell_h)

                    avg_col = average_color_in_subregion(big_np, x0, y0, w_sub, h_sub)
                    r, g, b = avg_col
                    color_hex = compress_hex_color(f'#{int(r):02x}{int(g):02x}{int(b):02x}')
                    bg_rects.append(
                        f'<rect x="{x0}" y="{y0}" width="{w_sub}" height="{h_sub}" fill="{color_hex}"/>\n'
                    )

            final_svg_footer = '</svg>'
            combined_polygons = [final_svg_header] + bg_rects

            # We'll parse polygons from each tile’s individual SVG
            polygon_pattern = re.compile(
                r'<polygon\s+points="([^"]+)"\s+fill="([^"]+)"\s*/>'
            )

            # For each tile
            for row in range(local_sep_factor):
                for col in range(local_sep_factor):
                    left = col * tile_w
                    top = row * tile_h
                    right = left + tile_w
                    bottom = top + tile_h
                    tile = current_image.crop((left, top, right, bottom))

                    tile_upscaled = tile.resize((orig_w, orig_h), Image.LANCZOS)

                    # Single pass on each tile
                    tile_svg = _bitmap_to_svg_layered_core(
                        tile_upscaled,
                        max_size_bytes=max_size_bytes,
                        resize=False,
                        target_size=target_size,
                        adaptive_fill=adaptive_fill,
                        num_colors=num_colors,
                        background_subdivisions=background_subdivisions
                    )

                    # Extract polygons
                    matches = polygon_pattern.findall(tile_svg)
                    x_offset = col * orig_w
                    y_offset = row * orig_h

                    for (points_str, fill_color) in matches:
                        pts = points_str.strip().split()
                        new_pts = []
                        for p in pts:
                            xs, ys = p.split(',')
                            x = float(xs) + x_offset
                            y = float(ys) + y_offset
                            new_pts.append(f"{x:.1f},{y:.1f}")
                        offset_points_str = " ".join(new_pts)
                        combined_polygons.append(
                            f'<polygon points="{offset_points_str}" fill="{fill_color}" />\n'
                        )

            combined_polygons.append(final_svg_footer)
            final_svg = "".join(combined_polygons)

        # Print size info
        svg_size = len(final_svg.encode('utf-8'))
        print(f"[Pass {pass_idx+1}/{iteration_passes}] separation_factor={local_sep_factor}, SVG size = {svg_size} bytes")

        # If not the last pass, rasterize for next iteration
        if pass_idx < iteration_passes - 1:
            rebitmap = svg_to_png(final_svg)
            current_image = rebitmap

    # Return final SVG
    return final_svg


def test_superpixel_to_svg(sample_df, seg, comp, keep_original=False, apply_OC=False, max_images=16,
                           iteration_passes=1, separation_factor=2):
    """
    Process up to 'max_images' from 'sample_df' in a 4x4 grid:
      1) Superpixel simplification (unless keep_original=True).
      2) Optional morphological Opening+Closing.
      3) Convert final result into an SVG using 'bitmap_to_svg_layered'.
      4) Render the SVG as a PNG and display in a subplot, titled by the sample's description.
    
    Returns:
        dict: { image_name: svg_string } for each processed image
    """
    
    # We'll process only 'max_images' from the sample_df (commonly 16 for a 4x4 grid)
    df_subset = sample_df.head(max_images)

    # Set up a 4x4 grid for plotting
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    axes = axes.ravel()

    svgs = {}  # Will store {image_name: svg_code}

    for idx, (_, row) in enumerate(tqdm(df_subset.iterrows(), total=len(df_subset))):
        image_name = row['image_name']
        image_path = os.path.join(IMAGES_PATH, image_name)

        # Load the image
        im = Image.open(image_path)
        im_np = np.array(im)

        # Drop alpha channel if present
        if im_np.ndim == 3 and im_np.shape[-1] == 4:
            im_np = im_np[..., :3]

        # 1) Superpixel simplification
        if not keep_original:
            simplified_im = superpixel_simplify(im_np, n_segments=seg, compactness=comp)
        else:
            simplified_im = im_np

        # 2) Optional morphological Opening + Closing
        if apply_OC:
            opened = apply_opening(simplified_im, kernel)
            closed = apply_closing(opened, kernel)
            final_im = closed
        else:
            final_im = simplified_im

        # 3) Convert final preprocessed result to SVG
        final_pil = Image.fromarray(final_im)
        svg_code = bitmap_to_svg_layered(final_pil, max_size_bytes=10000, resize=False,
                                         iteration_passes=iteration_passes, separation_factor=separation_factor)
        svgs[image_name] = svg_code

        # 4) Render the SVG in the subplot
        #    Convert the SVG string to a PNG in memory and then load it into Pillow.
        if idx < rows*cols:  # Ensure we have a subplot slot
            # Convert SVG string to PNG bytes
            png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
            # Load into PIL
            png_image = Image.open(io.BytesIO(png_data))

            # Get the description
            description = row[' comment'] if ' comment' in row else row['comment']
            wrapped_description = textwrap.fill(description, width=30)

            axes[idx].imshow(png_image)
            axes[idx].set_title(wrapped_description, fontsize=8)
            axes[idx].axis('off')

    # Adjust layout and show
    plt.tight_layout()
    plt.show()

    # Return the dictionary of SVGs
    return svgs


svgs = test_superpixel_to_svg(sample_df, seg=1600, comp=20, apply_OC=False)
for img_name, svg_code in svgs.items():
    print(f"{img_name} -> {len(svg_code.encode('utf-8'))} bytes in SVG")


svgs = test_superpixel_to_svg(sample_df, seg=1600, comp=20, apply_OC=False, iteration_passes=2)
for img_name, svg_code in svgs.items():
    print(f"{img_name} -> {len(svg_code.encode('utf-8'))} bytes in SVG")


svgs_from_originals = test_superpixel_to_svg(sample_df, seg=1600, comp=20, apply_OC=False, keep_original=True)
for img_name, svg_code in svgs_from_originals.items():
    print(f"{img_name} -> {len(svg_code.encode('utf-8'))} bytes in SVG")

