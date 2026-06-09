pip install kaggle kagglehub


import kagglehub

path = kagglehub.competition_download("image-matching-challenge-2025")
print("âœ… Dataset downloaded at:", path)



import os

# Change this path to your competition data root after unzip
base_dir = kagglehub.competition_download("image-matching-challenge-2025")

print(f"ğŸ“‚ Listing all files under: {base_dir}\n")

for root, dirs, files in os.walk(base_dir):
    level = root.replace(base_dir, "").count(os.sep)
    indent = " " * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 4 * (level + 1)
    for f in files:
        print(f"{subindent}{f}")



import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# === Paths ===
# BASE_PATH = "/kaggle/input/image-matching-challenge-2025"
train_dirs = [os.path.join(base_dir, "train/ETs"), os.path.join(base_dir, "train/stairs")]
test_dirs = [os.path.join(base_dir, "test/ETs"), os.path.join(base_dir, "test/stairs")]

# === Helper functions ===
def load_images_from_folder(folder, max_images=20):
    images = []
    for fname in sorted(os.listdir(folder))[:max_images]:
        if fname.lower().endswith(".png"):
            path = os.path.join(folder, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append((fname, img))
    return images

def match_images(img1, img2, sift, flann):
    # Detect and compute features
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0, 0, 0, []

    # FLANN matching
    matches = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]

    return len(kp1), len(kp2), len(good), good

def evaluate_folder(train_imgs, test_imgs, sift, flann):
    results = []
    for tname, timg in test_imgs:
        best_good = 0
        best_train_name = None
        for trname, trimg in train_imgs:
            kp1, kp2, good_count, _ = match_images(trimg, timg, sift, flann)
            if good_count > best_good:
                best_good = good_count
                best_train_name = trname
        results.append((tname, best_train_name, best_good))
    return results

# === Initialize feature extractor & matcher ===
sift = cv2.SIFT_create()
FLANN_INDEX_KDTREE = 1
flann = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_KDTREE, trees=5), dict(checks=50))

# === Process each category ===
final_eval = {}
for category, (train_path, test_path) in zip(["ETs", "stairs"], zip(train_dirs, test_dirs)):
    print(f"\nğŸ”¹ Processing category: {category}")
    train_imgs = load_images_from_folder(train_path)
    test_imgs = load_images_from_folder(test_path)
    
    results = evaluate_folder(train_imgs, test_imgs, sift, flann)

    # Aggregate results
    total_good = np.mean([r[2] for r in results])
    precision = np.mean([1 if r[2] > 10 else 0 for r in results])
    recall = np.mean([1 if r[2] > 5 else 0 for r in results])
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    final_eval[category] = {
        "Avg Good Matches": round(total_good, 2),
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1-Score": round(f1, 2)
    }

    print(f"  âœ… Avg Good Matches: {total_good:.2f}")
    print(f"  âœ… Precision: {precision:.2f}")
    print(f"  âœ… Recall: {recall:.2f}")
    print(f"  âœ… F1-Score: {f1:.2f}")

# === Display summary ===
import pandas as pd
df = pd.DataFrame(final_eval).T
print("\nğŸ“Š Final Evaluation Summary:\n")
display(df)

# === Optional: visualize a sample match ===
sample_train = train_imgs[0][1]
sample_test = test_imgs[0][1]
kp1, kp2, good_matches = None, None, None

kp1, des1 = sift.detectAndCompute(sample_train, None)
kp2, des2 = sift.detectAndCompute(sample_test, None)
matches = flann.knnMatch(des1, des2, k=2)
good = [m for m, n in matches if m.distance < 0.7 * n.distance]
match_vis = cv2.drawMatches(sample_train, kp1, sample_test, kp2, good[:20], None, flags=2)

plt.figure(figsize=(12,6))
plt.imshow(match_vis)
plt.title("Sample Keypoint Matches")
plt.axis("off")
plt.show()



import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === Paths ===
# base_dir = "/kaggle/input/image-matching-challenge-2025"  # Change as needed
# output_dir = "./output_matches"
# os.makedirs(output_dir, exist_ok=True)

train_dirs = [os.path.join(base_dir, "train/ETs"), os.path.join(base_dir, "train/stairs")]
test_dirs = [os.path.join(base_dir, "test/ETs"), os.path.join(base_dir, "test/stairs")]

# === Helper functions ===
def load_images_from_folder(folder, max_images=20):
    images = []
    for fname in sorted(os.listdir(folder))[:max_images]:
        if fname.lower().endswith(".png"):
            path = os.path.join(folder, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append((fname, img))
    return images

def match_images(img1, img2, detector, matcher, ratio=0.7, use_knn=True):
    # Detect keypoints and descriptors
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return kp1, kp2, 0, [], kp1, kp2

    # Match descriptors
    if use_knn:
        matches = matcher.knnMatch(des1, des2, k=2)
        good = []
        for m,n in matches:
            if m.distance < ratio * n.distance:
                good.append(m)
    else:
        matches = matcher.match(des1, des2)
        good = sorted(matches, key=lambda x: x.distance)

    good_count = len(good)
    return kp1, kp2, good_count, good, kp1, kp2

def evaluate_folder(train_imgs, test_imgs, detector, matcher, method_name):
    results = []
    for tname, timg in test_imgs:
        best_good = 0
        best_train_name = None
        for trname, trimg in train_imgs:
            kp1, kp2, good_count, _, _, _ = match_images(trimg, timg, detector, matcher)
            if good_count > best_good:
                best_good = good_count
                best_train_name = trname
        results.append((tname, best_train_name, best_good))
    return results

def compute_metrics(results):
    total_good = np.mean([r[2] for r in results])
    precision = np.mean([1 if r[2] > 10 else 0 for r in results])
    recall = np.mean([1 if r[2] > 5 else 0 for r in results])
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return round(total_good, 2), round(precision, 2), round(recall, 2), round(f1, 2)

def visualize_match(train_imgs, test_imgs, detector, matcher, title, ratio=0.7, use_knn=True):
    timg = test_imgs[0][1]
    trimg = train_imgs[0][1]
    kp1, kp2, good_count, good, kpA, kpB = match_images(trimg, timg, detector, matcher, ratio, use_knn)
    vis = cv2.drawMatches(trimg, kpA, timg, kpB, good[:25], None, flags=2)
    plt.figure(figsize=(12,6))
    plt.imshow(vis, cmap='gray')
    plt.title(f"{title} | Good Matches: {good_count}")
    plt.axis("off")
    plt.show()


# === Initialize detectors and matchers ===

# SIFT + FLANN (float descriptors)
sift = cv2.SIFT_create()
FLANN_INDEX_KDTREE = 1
flann_sift = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_KDTREE, trees=5), dict(checks=50))

# ORB + BFMatcher (binary descriptors)
orb = cv2.ORB_create(nfeatures=1000)
bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# === Main Evaluation ===
final_eval = []

for category, (train_path, test_path) in zip(["ETs", "stairs"], zip(train_dirs, test_dirs)):
    print(f"\nğŸ”¹ Category: {category}")
    train_imgs = load_images_from_folder(train_path)
    test_imgs = load_images_from_folder(test_path)

    # --- SIFT + FLANN ---
    sift_results = evaluate_folder(train_imgs, test_imgs, sift, flann_sift, "SIFT")
    s_avg, s_prec, s_rec, s_f1 = compute_metrics(sift_results)

    # --- ORB + BFMatcher ---
    orb_results = evaluate_folder(train_imgs, test_imgs, orb, bf_orb, "ORB")
    o_avg, o_prec, o_rec, o_f1 = compute_metrics(orb_results)

    final_eval.append({
        "Category": category,
        "SIFT AvgGood": s_avg, "SIFT Precision": s_prec, "SIFT Recall": s_rec, "SIFT F1": s_f1,
        "ORB AvgGood": o_avg, "ORB Precision": o_prec, "ORB Recall": o_rec, "ORB F1": o_f1
    })

# === Display summary ===
df = pd.DataFrame(final_eval)
print("\nğŸ“Š Final Evaluation Summary:\n")
display(df)

# === Visualize Matches ===
for category, (train_path, test_path) in zip(["ETs", "stairs"], zip(train_dirs, test_dirs)):
    train_imgs = load_images_from_folder(train_path)
    test_imgs = load_images_from_folder(test_path)
    print(f"\nğŸ–¼ Visual Comparisons for {category}")
    visualize_match(train_imgs, test_imgs, sift, flann_sift, f"SIFT + FLANN ({category})")
    visualize_match(train_imgs, test_imgs, orb, bf_orb, f"ORB + BFMatcher ({category})", use_knn=False)



!git clone https://github.com/cvg/LightGlue.git


%cd LightGlue


!python -m pip install -e .


# ============================================================
# 0ï¸�âƒ£ Setup â€” imports and device
# ============================================================
import sys
import os
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# ============================================================
# 1ï¸�âƒ£ Add LightGlue repo path from left panel
# ============================================================
# Suppose you attached LightGlue repo via left panel:
LIGHTGLUE_PATH = "/kaggle/working/LightGlue"  # <-- update based on your input
sys.path.append(LIGHTGLUE_PATH)

# Now import LightGlue
from lightglue import LightGlue, SuperPoint
# from lightglue.utils import match_images


# ============================================================
# 2ï¸�âƒ£ Dataset paths
# ============================================================
BASE_PATH = "/kaggle/input/image-matching-challenge-2025"
TRAIN_DIRS = [os.path.join(BASE_PATH, "train/ETs"),
              os.path.join(BASE_PATH, "train/stairs")]
TEST_DIRS  = [os.path.join(BASE_PATH, "test/ETs"),
              os.path.join(BASE_PATH, "test/stairs")]

# ============================================================
# 3ï¸�âƒ£ Load images helper
# ============================================================
def load_images(folder, max_images=10):
    images = []
    for fname in sorted(os.listdir(folder))[:max_images]:
        if fname.lower().endswith(".png"):
            path = os.path.join(folder, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append((fname, img))
    return images

# ============================================================
# 4ï¸�âƒ£ Initialize models
# ============================================================
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)

# ============================================================
# 5ï¸�âƒ£ Compute LightGlue matches
# ============================================================
def compute_lightglue_matches(img1, img2):
    t1 = torch.from_numpy(img1)[None, None].float().to(device) / 255.
    t2 = torch.from_numpy(img2)[None, None].float().to(device) / 255.

    feats1 = extractor({"image": t1})
    feats2 = extractor({"image": t2})

    matches = matcher({"image0": feats1, "image1": feats2})
    # number of valid matches
    num_matches = (matches["matches0"][0] >= 0).sum().item()
    return num_matches, matches, feats1, feats2

# ============================================================
# 6ï¸�âƒ£ Evaluate category
# ============================================================
def evaluate_category(train_path, test_path, name, max_images=10):
    train_imgs = load_images(train_path, max_images)
    test_imgs = load_images(test_path, max_images)
    results = []

    for tname, timg in tqdm(test_imgs, desc=f"Processing {name}"):
        best_good = 0
        best_train = None
        for trname, trimg in train_imgs:
            try:
                good, _, _, _ = compute_lightglue_matches(trimg, timg)
                if good > best_good:
                    best_good = good
                    best_train = trname
            except Exception as e:
                print(f"âš ï¸� Error matching {trname} -> {tname}: {e}")
                continue
        results.append((tname, best_train, best_good))

    # Metrics
    avg_matches = np.mean([r[2] for r in results])
    precision = np.mean([1 if r[2] > 30 else 0 for r in results])
    recall = np.mean([1 if r[2] > 15 else 0 for r in results])
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    summary = {
        "Category": name,
        "Avg Matches": float(round(avg_matches, 2)),
        "Precision": float(round(precision, 2)),
        "Recall": float(round(recall, 2)),
        "F1": float(round(f1, 2))
    }
    return summary, results

for cat, (train_p, test_p) in zip(["ETs", "stairs"], zip(TRAIN_DIRS, TEST_DIRS)):
    summary, _ = evaluate_category(train_p, test_p, cat, max_images=5)
    
    # Convert any NumPy scalars to Python scalars
    summary_clean = {}
    for k, v in summary.items():
        if isinstance(v, np.ndarray):
            summary_clean[k] = v.item() if v.size == 1 else str(v)
        elif isinstance(v, np.generic):
            summary_clean[k] = v.item()
        else:
            summary_clean[k] = v
    
    print(f"\nğŸ“Š LightGlue Evaluation Summary for {cat}:")
    for key, val in summary_clean.items():
        print(f"{key}: {val}")






# ============================================================
# 8ï¸�âƒ£ Visualize matches
# ============================================================
def visualize_lightglue_matches(img1, img2):
    _, matches, feats1, feats2 = compute_lightglue_matches(img1, img2)

    mkpts0 = feats1["keypoints"][0][matches["matches0"][0] >= 0].cpu().numpy()
    mkpts1 = feats2["keypoints"][0][matches["matches1"][0] >= 0].cpu().numpy()

    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    concat = np.concatenate([img1_color, img2_color], axis=1)

    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        cv2.circle(concat, (int(x0), int(y0)), 3, (0,255,0), -1)
        cv2.circle(concat, (int(x1)+img1.shape[1], int(y1)), 3, (0,255,0), -1)
        cv2.line(concat, (int(x0), int(y0)), (int(x1)+img1.shape[1], int(y1)), (255,0,0), 1)

    plt.figure(figsize=(12,6))
    plt.imshow(concat)
    plt.axis("off")
    plt.show()


# Helper to safely get first image from a folder
def get_first_image(folder):
    imgs = load_images(folder, max_images=1)
    if len(imgs) == 0:
        print(f"âš ï¸� No images found in {folder}")
        return None
    return imgs[0][1]

# Example visualizations
train_img = get_first_image(TRAIN_DIRS[0])
test_img  = get_first_image(TEST_DIRS[0])
if train_img is not None and test_img is not None:
    print("ğŸ–¼ Sample matches - ETs")
    visualize_lightglue_matches(train_img, test_img)

train_img = get_first_image(TRAIN_DIRS[1])
test_img  = get_first_image(TEST_DIRS[1])
if train_img is not None and test_img is not None:
    print("ğŸ–¼ Sample matches - Stairs")
    visualize_lightglue_matches(train_img, test_img)



import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from lightglue import LightGlue, SuperPoint

# ============================================================
# 1ï¸�âƒ£ Device
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# ============================================================
# 2ï¸�âƒ£ Dataset paths
# ============================================================
BASE_PATH = "/kaggle/input/image-matching-challenge-2025"
TRAIN_DIRS = [os.path.join(BASE_PATH, "train/ETs"), os.path.join(BASE_PATH, "train/stairs")]
TEST_DIRS  = [os.path.join(BASE_PATH, "test/ETs"), os.path.join(BASE_PATH, "test/stairs")]

# ============================================================
# 3ï¸�âƒ£ Model initialization (SuperPoint + LightGlue)
# ============================================================
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher   = LightGlue(features="superpoint").eval().to(device)

# ============================================================
# 4ï¸�âƒ£ Load images helper
# ============================================================
def load_images(folder, max_images=10):
    images = []
    for fname in sorted(os.listdir(folder))[:max_images]:
        if fname.lower().endswith(".png"):
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append((fname, img))
    return images

# ============================================================
# 5ï¸�âƒ£ Compute LightGlue matches
# ============================================================
def compute_lightglue_matches(img1, img2):
    t1 = torch.from_numpy(img1)[None, None].float().to(device) / 255.
    t2 = torch.from_numpy(img2)[None, None].float().to(device) / 255.

    feats1 = extractor({"image": t1})
    feats2 = extractor({"image": t2})

    matches = matcher({"image0": feats1, "image1": feats2})
    num_matches = (matches["matches0"][0] >= 0).sum().item()
    return num_matches

# ============================================================
# 6ï¸�âƒ£ Evaluate a category
# ============================================================
def evaluate_category(train_path, test_path, name, max_images=10):
    train_imgs = load_images(train_path, max_images)
    test_imgs  = load_images(test_path, max_images)

    if len(train_imgs) == 0 or len(test_imgs) == 0:
        print(f"âš ï¸� No images found for {name}. Skipping...")
        return None, []

    results = []
    for tname, timg in tqdm(test_imgs, desc=f"Processing {name}"):
        best_good = 0
        best_train = None
        for trname, trimg in train_imgs:
            try:
                good = compute_lightglue_matches(trimg, timg)
                if good > best_good:
                    best_good = good
                    best_train = trname
            except Exception as e:
                continue
        results.append((tname, best_train, best_good))

    # Metrics
    avg_matches = np.mean([r[2] for r in results])
    precision   = np.mean([1 if r[2] > 30 else 0 for r in results])
    recall      = np.mean([1 if r[2] > 15 else 0 for r in results])
    f1          = 2 * (precision * recall) / (precision + recall + 1e-8)

    summary = {
        "Category": name,
        "Avg Matches": round(avg_matches, 2),
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1": round(f1, 2)
    }
    return summary, results

# ============================================================
# 7ï¸�âƒ£ Run evaluation and print summaries
# ============================================================
all_results = {}
for cat, (train_p, test_p) in zip(["ETs", "stairs"], zip(TRAIN_DIRS, TEST_DIRS)):
    summary, results = evaluate_category(train_p, test_p, cat, max_images=5)
    if summary is not None:
        all_results[cat] = {"summary": summary, "pairs": results}
        print(f"\nğŸ“Š LightGlue Evaluation Summary for {cat}:")
        for key, val in summary.items():
            print(f"{key}: {val}")

# ============================================================
# 8ï¸�âƒ£ Visualization at the END
# ============================================================
def visualize_lightglue_matches(img1, img2):
    t1 = torch.from_numpy(img1)[None, None].float().to(device) / 255.
    t2 = torch.from_numpy(img2)[None, None].float().to(device) / 255.

    feats1 = extractor({"image": t1})
    feats2 = extractor({"image": t2})

    matches = matcher({"image0": feats1, "image1": feats2})

    mkpts0 = feats1["keypoints"][0][matches["matches0"][0] >= 0].cpu().numpy()
    mkpts1 = feats2["keypoints"][0][matches["matches1"][0] >= 0].cpu().numpy()

    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    concat = np.concatenate([img1_color, img2_color], axis=1)

    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        cv2.circle(concat, (int(x0), int(y0)), 3, (0,255,0), -1)
        cv2.circle(concat, (int(x1)+img1.shape[1], int(y1)), 3, (0,255,0), -1)
        cv2.line(concat, (int(x0), int(y0)), (int(x1)+img1.shape[1], int(y1)), (255,0,0), 1)

    plt.figure(figsize=(12,6))
    plt.imshow(concat)
    plt.axis("off")
    plt.show()

# Visualize first image pair for each category
for train_p, test_p, cat in zip(TRAIN_DIRS, TEST_DIRS, ["ETs", "stairs"]):
    train_imgs = load_images(train_p, max_images=1)
    test_imgs  = load_images(test_p, max_images=1)
    if len(train_imgs) > 0 and len(test_imgs) > 0:
        print(f"ğŸ–¼ Sample matches - {cat}")
        visualize_lightglue_matches(train_imgs[0][1], test_imgs[0][1])


