import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import json

DATA_PATH = "/kaggle/input/deepfake-detection-challenge/train_sample_videos"

with open("/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json", "r") as f:
    meta = json.load(f)

df = pd.DataFrame(meta).T.reset_index()
df.rename(columns={'index':'filename'}, inplace=True)
df.head()



def get_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frames / fps if fps > 0 else 0
    cap.release()
    return width, height, fps, duration
    
df_sample = df.sample(100, random_state=42).reset_index(drop=True)
# df_sample.head()

meta_data = []
for i, row in df_sample.iterrows():
    video_file = os.path.join(DATA_PATH, row["filename"])
    info = get_video_metadata(video_file)
    if info:
        width, height, fps, duration = info
        meta_data.append([row["filename"], row["label"], width, height, fps, duration])

df_meta = pd.DataFrame(meta_data, columns=["filename", "label", "width", "height", "fps", "duration"])
df_meta.head()



# Class distribution
import pandas as pd
import matplotlib.pyplot as plt


metadata = pd.read_json("/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json").T
metadata.head()

class_counts = metadata['label'].value_counts()

# Plot
class_counts.plot(kind='bar', color=['red', 'green'])
plt.title("Class Distribution (Real vs Fake)")
plt.ylabel("Count")
plt.xlabel("Class")
plt.show()



import cv2
import numpy as np
from tqdm import tqdm


video_paths = metadata.index  

video_stats = []

for video in tqdm(video_paths):
    path = f"/kaggle/input/deepfake-detection-challenge/train_sample_videos/{video}"
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        continue
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    bitrate = (frame_count * width * height) / duration if duration > 0 else 0
    
    video_stats.append([video, metadata.loc[video]['label'], width, height, fps, duration, bitrate])
    cap.release()

df_stats = pd.DataFrame(video_stats, columns=["video", "label", "width", "height", "fps", "duration", "bitrate"])
df_stats.head()



from scipy.stats import ttest_ind

features = ["width", "height", "fps", "duration", "bitrate"]

results = {}
for feature in features:
    real_vals = df_stats[df_stats["label"] == "REAL"][feature]
    fake_vals = df_stats[df_stats["label"] == "FAKE"][feature]
    stat, pval = ttest_ind(real_vals, fake_vals, equal_var=False, nan_policy='omit')
    results[feature] = {"real_mean": real_vals.mean(), "fake_mean": fake_vals.mean(), "p-value": pval}

results_df = pd.DataFrame(results).T
results_df



import cv2
import os

def extract_frames(video_path, out_dir, num_frames=10):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(frame_count // num_frames, 1)
    i, saved = 0, 0
    while cap.isOpened() and saved < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            frame_path = os.path.join(out_dir, f"frame_{saved}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        i += 1
    cap.release()


extract_frames("/kaggle/input/deepfake-detection-challenge/train_sample_videos/akzbnazxtz.mp4",
               "frames_real", num_frames=5)
extract_frames("/kaggle/input/deepfake-detection-challenge/train_sample_videos/aapnvogymq.mp4",
               "frames_fake", num_frames=5)



!pip install mahotas


# Feature engineering
import numpy as np
from skimage.feature import hog
from skimage import color
import mahotas
import matplotlib.pyplot as plt

def compute_features(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Haralick Texture Features
    haralick = mahotas.features.haralick(gray).mean(axis=0)

    # HOG Features
    hog_features, hog_img = hog(gray, visualize=True, block_norm='L2-Hys')

    # Color Histogram (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, [8,8,8], [0,180,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()

    return {"haralick": haralick,
            "hog": hog_features,
            "hog_img": hog_img,
            "hist": hist}



import glob

real_frame = glob.glob("frames_real/*.jpg")[0]
fake_frame = glob.glob("frames_fake/*.jpg")[0]

real_feats = compute_features(real_frame)
fake_feats = compute_features(fake_frame)

fig, axs = plt.subplots(1,2, figsize=(12,6))
axs[0].imshow(real_feats["hog_img"], cmap="gray")
axs[0].set_title("HOG (Real)")
axs[1].imshow(fake_feats["hog_img"], cmap="gray")
axs[1].set_title("HOG (Fake)")
plt.show()

# Compare Color Histograms
plt.plot(real_feats["hist"][:50], label="Real", color="blue")
plt.plot(fake_feats["hist"][:50], label="Fake", color="red")
plt.legend()
plt.title("Color Histogram (HSV)")
plt.show()

features = range(len(real_feats["haralick"]))
plt.plot(features, real_feats["haralick"], label="Real", marker="o")
plt.plot(features, fake_feats["haralick"], label="Fake", marker="x")
plt.legend()
plt.title("Haralick Texture Features")
plt.show()



#  Outlier Detection
import pandas as pd
import numpy as np
import cv2
import os
import glob

# Step 1: Collect metadata
def get_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Prevent division by zero
    if fps <= 0:
        fps = np.nan
    
    duration = frame_count / fps if fps and fps > 0 else np.nan
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    
    return {"video": os.path.basename(video_path),
            "fps": fps, "duration": duration,
            "width": width, "height": height}

video_list = glob.glob("/kaggle/input/deepfake-detection-challenge/train_sample_videos/*.mp4")[:100]


metadata = pd.DataFrame([get_video_metadata(v) for v in video_list])

# Drop rows with missing fps/duration
metadata = metadata.dropna(subset=["fps", "duration"])


# Step 2: Safe Z-score 
def safe_zscore(series):
    """Return Z-scores safely even if std=0"""
    if series.std() == 0 or pd.isna(series.std()):
        return pd.Series([0]*len(series), index=series.index)
    return (series - series.mean()) / series.std()

# Apply to each numeric column
for col in ["fps", "duration", "width", "height"]:
    metadata[f"{col}_z"] = safe_zscore(metadata[col])


# Step 3: Outlier Detection
outliers = metadata[
    (metadata["fps_z"].abs() > 2) |
    (metadata["duration_z"].abs() > 2) |
    (metadata["width_z"].abs() > 2) |
    (metadata["height_z"].abs() > 2)
]

metadata, outliers



# Temporal anomaly
def temporal_difference(video_path, num_frames=50):
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    diffs = []
    count = 0
    while cap.isOpened() and count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            diff = np.mean(cv2.absdiff(gray, prev_frame))
            diffs.append(diff)
        prev_frame = gray
        count += 1
    cap.release()
    return diffs

real_diffs = temporal_difference("/kaggle/input/deepfake-detection-challenge/train_sample_videos/abarnvbtwb.mp4")
fake_diffs = temporal_difference("/kaggle/input/deepfake-detection-challenge/train_sample_videos/aagfhgtpmv.mp4")

plt.plot(real_diffs, label="Real")
plt.plot(fake_diffs, label="Fake")
plt.legend()
plt.title("Temporal Frame Differences")
plt.show()



!pip install mahotas



import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import mahotas
from skimage.feature import hog, greycomatrix, greycoprops
from skimage.color import rgb2gray
from mahotas import features as mh_features 
from sklearn.decomposition import PCA

# 1. Extract frames from videos
def extract_frames(video_path, frame_interval=15, out_dir="frames"):
    """
    Extracts every Nth frame from a video.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    success = True
    
    while success:
        success, frame = cap.read()
        if not success:
            break
        if count % frame_interval == 0:
            frame_path = os.path.join(out_dir, f"{os.path.basename(video_path)}_{count}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
        count += 1
    
    cap.release()
    return frames


# 2. Feature Computation

def compute_haralick(image_gray):
    """
    Compute Haralick features from grayscale image.
    """
    return mh_features.haralick(image_gray).mean(axis=0)  # mean over directions

def compute_hog(image_gray):
    """
    Compute HOG features.
    """
    hog_features, hog_image = hog(image_gray, 
                                  orientations=9, 
                                  pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), 
                                  visualize=True,
                                  block_norm='L2-Hys')
    return hog_features, hog_image

def compute_color_hist(image):
    """
    Compute normalized color histogram (RGB).
    """
    chans = cv2.split(image)
    hist_features = []
    for chan in chans:
        hist = cv2.calcHist([chan], [0], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist)
    return np.array(hist_features)


# 3. Example Usage
real_video = "/kaggle/input/deepfake-detection-challenge/train_sample_videos/akzbnazxtz.mp4"
fake_video = "/kaggle/input/deepfake-detection-challenge/train_sample_videos/adohikbdaz.mp4"

real_frames = extract_frames(real_video, frame_interval=15, out_dir="frames/real")
fake_frames = extract_frames(fake_video, frame_interval=15, out_dir="frames/fake")

# Compute features for first frame of each
sample_real = cv2.imread(real_frames[0])
sample_fake = cv2.imread(fake_frames[0])

gray_real = cv2.cvtColor(sample_real, cv2.COLOR_BGR2GRAY)
gray_fake = cv2.cvtColor(sample_fake, cv2.COLOR_BGR2GRAY)

# Haralick
haralick_real = compute_haralick(gray_real)
haralick_fake = compute_haralick(gray_fake)

# HOG
hog_real, hog_img_real = compute_hog(gray_real)
hog_fake, hog_img_fake = compute_hog(gray_fake)

# Color Histogram
color_hist_real = compute_color_hist(sample_real)
color_hist_fake = compute_color_hist(sample_fake)

# 4. Visualization
plt.figure(figsize=(20,15))

# HOG visualization
plt.subplot(1,2,1)
plt.imshow(hog_img_real, cmap="gray")
plt.title("HOG - Real Frame")

plt.subplot(1,2,2)
plt.imshow(hog_img_fake, cmap="gray")
plt.title("HOG - Fake Frame")
plt.show()

# Color histograms
plt.figure(figsize=(10,5))
plt.plot(color_hist_real, label="Real")
plt.plot(color_hist_fake, label="Fake")
plt.title("Color Histogram Comparison")
plt.legend()
plt.show()

# Print Haralick features for inspection
print("Haralick (Real):", haralick_real[:5])   # show first 5 values
print("Haralick (Fake):", haralick_fake[:5])

# Simple comparison plot (first 5 Haralick features)
plt.figure(figsize=(8,5))
plt.plot(haralick_real[:5], marker='o', label="Real")
plt.plot(haralick_fake[:5], marker='o', label="Fake")
plt.title("Haralick Texture Features")
plt.xlabel("Feature Index")
plt.ylabel("Value")
plt.legend()
plt.show()




# PCA clusters
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.color import rgb2gray
import mahotas
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Feature Computation Functions
def compute_haralick(image_gray):
    return mahotas.features.haralick(image_gray).mean(axis=0)

def compute_hog(image_gray):
    features, _ = hog(image_gray, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    return features

def compute_color_hist(image):
    chans = cv2.split(image)
    hist_features = []
    for chan in chans:
        hist = cv2.calcHist([chan], [0], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist)
    return np.array(hist_features)

# Process Multiple Videos
def extract_features_from_video(video_path, label, frame_interval=20, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    features = []
    labels = []
    count, extracted = 0, 0
    
    while cap.isOpened() and extracted < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Compute features
            haralick = compute_haralick(gray)
            hog_feat = compute_hog(gray)
            color_hist = compute_color_hist(frame)
            # Concatenate all
            combined = np.concatenate([haralick, hog_feat, color_hist])
            features.append(combined)
            labels.append(label)
            extracted += 1
        count += 1
    
    cap.release()
    return features, labels

# Example Usage with PCA
real_videos = ["/kaggle/input/deepfake-detection-challenge/train_sample_videos/akzbnazxtz.mp4",
               "/kaggle/input/deepfake-detection-challenge/train_sample_videos/aelfnikyqj.mp4",
              "/kaggle/input/deepfake-detection-challenge/train_sample_videos/ahqqqilsxt.mp4",
              "/kaggle/input/deepfake-detection-challenge/train_sample_videos/ajqslcypsw.mp4"]
fake_videos = ["/kaggle/input/deepfake-detection-challenge/train_sample_videos/aelzhcnwgf.mp4", 
               "/kaggle/input/deepfake-detection-challenge/train_sample_videos/adylbeequz.mp4",
              "/kaggle/input/deepfake-detection-challenge/train_sample_videos/acxwigylke.mp4",
              "/kaggle/input/deepfake-detection-challenge/train_sample_videos/acxnxvbsxk.mp4"]

X, y = [], []

# Extract from real videos
for rv in real_videos:
    feats, labs = extract_features_from_video(rv, label="Real")
    X.extend(feats)
    y.extend(labs)

# Extract from fake videos
for fv in fake_videos:
    feats, labs = extract_features_from_video(fv, label="Fake")
    X.extend(feats)
    y.extend(labs)

X = np.array(X)

# Standardize before PCA
X_scaled = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# PCA clusters
plt.figure(figsize=(8,6))
for label, color in zip(["Real", "Fake"], ["blue", "red"]):
    mask = np.array(y) == label
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, alpha=0.7, c=color)

plt.title("PCA Clustering of Real vs Fake Features")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()



import cv2
import numpy as np
import matplotlib.pyplot as plt

# Optical Flow Function
def compute_optical_flow_stats(video_path, max_frames=200):
    cap = cv2.VideoCapture(video_path)
    
    # Read the first frame
    ret, frame1 = cap.read()
    if not ret:
        print("Error: Cannot read video")
        return
    
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    magnitudes = []
    variances = []
    
    frame_count = 0
    
    while True:
        ret, frame2 = cap.read()
        if not ret or frame_count >= max_frames:
            break
        
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Compute dense optical flow (Farnebäck)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Extract flow magnitude and angle
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        
        # Compute statistics
        magnitudes.append(np.mean(mag))          # average speed
        variances.append(np.var(mag))            # variance in speed (irregularity)
        
        prev_gray = gray
        frame_count += 1
    
    cap.release()
    
    return magnitudes, variances

# Example Usage
real_video = "/kaggle/input/deepfake-detection-challenge/train_sample_videos/ahqqqilsxt.mp4"
fake_video = "/kaggle/input/deepfake-detection-challenge/train_sample_videos/aelzhcnwgf.mp4"

real_mag, real_var = compute_optical_flow_stats(real_video)
fake_mag, fake_var = compute_optical_flow_stats(fake_video)

# Plot Statistics
plt.figure(figsize=(12,5))

# Mean flow magnitude
plt.subplot(1,2,1)
plt.plot(real_mag, label="Real", color="blue")
plt.plot(fake_mag, label="Fake", color="red")
plt.title("Optical Flow Mean Magnitude over Frames")
plt.xlabel("Frame Index")
plt.ylabel("Mean Magnitude")
plt.legend()

# Variance of flow magnitude
plt.subplot(1,2,2)
plt.plot(real_var, label="Real", color="blue")
plt.plot(fake_var, label="Fake", color="red")
plt.title("Optical Flow Variance over Frames")
plt.xlabel("Frame Index")
plt.ylabel("Variance of Magnitude")
plt.legend()

plt.tight_layout()
plt.show()



# Isolation forest
import os, glob, cv2, json, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


VIDEO_DIR = "/kaggle/input/deepfake-detection-challenge/train_sample_videos"
EXTS = ("*.mp4", "*.mov", "*.avi")

def extract_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or np.nan
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
    duration = frames / fps if fps and fps > 0 else np.nan
    cap.release()
    return {
        "filename": os.path.basename(video_path),
        "fps": fps, "duration": duration,
        "width": width, "height": height
    }

video_files = []
for ext in EXTS:
    video_files.extend(glob.glob(os.path.join(VIDEO_DIR, ext)))

rows = [extract_metadata(v) for v in video_files if extract_metadata(v) is not None]
df = pd.DataFrame(rows)
df["resolution_pixels"] = df["width"] * df["height"]

# Clean data: replace NaN/inf/0 with median
for col in ["fps", "duration", "resolution_pixels"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col].replace([np.inf, -np.inf, 0], np.nan, inplace=True)
    df[col].fillna(df[col].median(), inplace=True)

# Isolation Forest
features = ["fps", "duration", "resolution_pixels"]
X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso = IsolationForest(contamination=0.05, random_state=42, behaviour="new")
df["iso_label"] = iso.fit_predict(X_scaled)  # 1 = normal, -1 = outlier
df["iso_score"] = iso.decision_function(X_scaled)

# Show flagged outliers
print(df[df["iso_label"] == -1][["filename", "fps", "duration", "resolution_pixels", "iso_score"]].head())

# Visualization: FPS vs Duration
plt.figure(figsize=(8,6))
colors = df["iso_label"].map({1:"blue",-1:"red"})
plt.scatter(df["fps"], df["duration"], c=colors, s=60, edgecolor="k", alpha=0.7)
plt.xlabel("FPS")
plt.ylabel("Duration (s)")
plt.title("Isolation Forest: FPS vs Duration (Red = Outlier)")
plt.show()


