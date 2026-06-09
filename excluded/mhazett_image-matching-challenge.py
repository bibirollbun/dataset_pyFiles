import pandas as pd
import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import cv2
import os
from PIL import Image
from itertools import combinations
import networkx as nx
import glob


df = pd.read_csv('/kaggle/input/image-matching-challenge-2025/train_labels.csv')
df = df.dropna()
print(df.info())
print(df.describe())
print("Null data:\n", df.isnull().sum())


df['translation_vector'] = df['translation_vector'].apply(lambda x: np.fromstring(x, sep=';') if pd.notnull(x) else np.zeros(3))

df['rotation_matrix'] = df['rotation_matrix'].apply(lambda x: np.fromstring(x, sep=';').reshape(3, 3) if pd.notnull(x) else np.eye(3))

df[['x', 'y', 'z']] = pd.DataFrame(df['translation_vector'].tolist(), index=df.index)

figs = plt.figure(figsize=(8,6))
axes = figs.add_axes(111, projection='3d')

axes.scatter(df['x'], df['y'], df['z'], c='black', s=10, alpha=0.7)
axes.set_title("3D Scatter of Camera Translation Vectors")
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_zlabel("Z")
plt.grid(True)


scenes = df['scene'].unique()
plt.figure(figsize=(9,7))
for scene in scenes:
    group = df[df['scene'] == scene]
    plt.scatter(group['x'], group['y'], label=scene, s=10)
plt.title("2D Camera Positions by Scene")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.5, 1))
plt.tight_layout()


scene_counts = df['scene'].value_counts().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
scene_counts.plot(kind='bar', color='pink')
plt.title("Scene-Based Image Counts")
plt.xlabel("Scene Type")
plt.ylabel("Image Count")
plt.xticks(rotation=90, ha='right')
plt.grid(axis='y')
plt.tight_layout()


plt.figure(figsize=(10, 5))
plt.hist(df['x'].dropna(), bins=30, color='skyblue', label='X', alpha=0.6)
plt.hist(df['y'].dropna(), bins=30, color='green', label='Y', alpha=0.6)
plt.hist(df['z'].dropna(), bins=30, color='red', label='Z', alpha=0.6)
plt.title("Distribution of Camera Translation Coordinates")
plt.xlabel("Coordinate Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)


scene_counts = df.groupby(['dataset', 'scene']).size().reset_index(name='count')
scene_counts = scene_counts.sort_values(by='count', ascending=False)

plt.figure(figsize=(14, 7))
plt.barh(scene_counts['scene'] + " (" + scene_counts['dataset'] + ")", scene_counts['count'], color='orchid')
plt.xlabel("Number of Images")
plt.title("Image Count per Scene per Dataset")
plt.tight_layout()
plt.grid(True)


scenes = df['scene'].unique()
plt.figure(figsize=(10, 8))

for scene in scenes:
    subset = df[df['scene'] == scene]
    plt.scatter(subset['x'].mean(), subset['y'].mean(), label=scene)

plt.title("2D Centroids of Scenes (XY Plane)")
plt.xlabel("X Mean")
plt.ylabel("Y Mean")
plt.legend(bbox_to_anchor=(1.5, 1))
plt.grid(True)
plt.tight_layout()


df['img_path'] = df.apply(
    lambda row: (Path("train") / row["dataset"] / row["image"]).as_posix(),
    axis=1
)

orb = cv2.ORB_create(
    nfeatures=4000,
    scaleFactor=1.2,
    nlevels=8
)
print("ORB parameters:")
print("nfeatures:", orb.getMaxFeatures())
print("scaleFactor:", orb.getScaleFactor())
print("nlevels:", orb.getNLevels())

def extract_features(image_path, detector):
    print("Trying to load image:", image_path)
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("❌ Warning: could not load image", image_path)
        return None, None

    print("Image loaded, shape:", img.shape)
    keypoints, descriptors = detector.detectAndCompute(img, None)

    print("Extracted", len(keypoints), "keypoints")
    if descriptors is not None:
        print("Descriptors shape:", descriptors.shape)
    else:
        print("No descriptors found.")

    return keypoints, descriptors

BASE_PATH = Path("/kaggle/input/image-matching-challenge-2025/train")

df['img_path'] = df.apply(
    lambda row: (BASE_PATH / row["dataset"] / row["image"]).as_posix(),
    axis=1
)

sample_path = df['img_path'].iloc[0]
print("File exists?", os.path.exists(sample_path))
print("Absolute path:", os.path.abspath(sample_path))

print("File exists?", os.path.exists(sample_path))
print("Absolute path:", os.path.abspath(sample_path))

try:
    img = Image.open(sample_path)
    plt.imshow(img)
    plt.title("Sample Image")
    plt.axis("off")
except Exception as e:
    print("PIL could not open image:", e)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

def match_descriptors(desc1, desc2, matcher, ratio=0.75, min_matches=15):
    if desc1 is None or desc2 is None:
        return 0
    matches = matcher.knnMatch(desc1, desc2, k=2)
    good = []

    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) >= min_matches:
        return len(good)
    else:
        return 0

descriptor_cache = {}

def cache_features(img_path):
    if img_path not in descriptor_cache:
        kps, descs = extract_features(img_path, orb)
        descriptor_cache[img_path] = descs
    return descriptor_cache[img_path]
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

sample_df = df.head(5)
for path1, path2 in combinations(sample_df['img_path'], 2):
    desc1 = cache_features(path1)
    desc2 = cache_features(path2)
    score = match_descriptors(desc1, desc2, matcher)

    print("Match score between\n", Path(path1).name and Path(path2).name, score)

G = nx.Graph()

sample_df = df.head(10)

for idx in sample_df.index:
    G.add_node(idx)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
descriptor_cache = {}

def cache_features(img_path):
    if img_path not in descriptor_cache:
        _, descs = extract_features(img_path, orb)
        descriptor_cache[img_path] = descs
    return descriptor_cache[img_path]

for i, j in combinations(sample_df.index, 2):
    path_i = sample_df.at[i, 'img_path']
    path_j = sample_df.at[j, 'img_path']

    desc_i = cache_features(path_i)
    desc_j = cache_features(path_j)

    score = match_descriptors(desc_i, desc_j, matcher)

if score > 20:
    G.add_edge(i, j, weight=score)
    print("Added edge: ", Path(path_i).name, " ↔ ", Path(path_j).name," with score ", str(score))

components = list(nx.connected_components(G))
filtered_components = []

for c in components:
    if len(c) >= 2:
        filtered_components.append(c)

components = filtered_components

print("Number of grouped scenes:", len(components))
scene_map = {}
for i, comp in enumerate(components):
    for node in comp:
        scene_map[node] = "scene_" + str(i)

df['pred_scene'] = df.index.map(scene_map).fillna("outlier")
print("Prediction scene:\n",df['pred_scene'].value_counts().head())


scene_counts = df['pred_scene'].value_counts().sort_values(ascending=False)

plt.figure(figsize=(13, 6))
scene_counts.plot(kind='bar', color='purple')
plt.title("Distribution of Images Across Predicted Scene Categories")
plt.xlabel("Predicted Scene Category")
plt.ylabel("Number of Images")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()


scene_id = df['pred_scene'].unique()[0]

scene_images = df[df['pred_scene'] == scene_id]['img_path'].tolist()[:5]

plt.figure(figsize=(12, 4))

for i, img_path in enumerate(scene_images):
    img = Image.open(img_path)
    plt.subplot(1, len(scene_images), i + 1)
    plt.imshow(img)
    plt.title(Path(img_path).name)
    plt.axis('off')

plt.suptitle("Sample Images from " + str(scene_id))
plt.subplots_adjust(top=0.75)
plt.tight_layout()


def match_keypoints(img1_path, img2_path, detector, matcher):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 8:
        print("Not enough good matches.")
        return None, None, None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    return pts1, pts2, good

focal_length = 1.0
principal_point = (0.0, 0.0)

def estimate_pose_from_points(pts1, pts2, focal=1.0, pp=(0.0, 0.0)):
    E, mask = cv2.findEssentialMat(pts1, pts2, focal=focal, pp=pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        print("Failed to compute essential matrix.")
        return None, None

    print("Essential matrix computed.")
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, focal=focal, pp=pp)

    return R, t

img1_path = df['img_path'].iloc[0]
img2_path = df['img_path'].iloc[1]
pts1, pts2, _ = match_keypoints(img1_path, img2_path, orb, bf)

if pts1 is not None:
    print("Matched keypoints:", len(pts1))
    R, t = estimate_pose_from_points(pts1, pts2)
    if R is not None:
        print("Rotation matrix:\n", R)
        print("Translation vector:\n", t)

scene_id = [s for s in df['pred_scene'].unique() if s != 'outlier'][0]
scene_df = df[df['pred_scene'] == scene_id].reset_index(drop=True)

pose_graph = {}
ref_img_path = scene_df['img_path'].iloc[0]
pose_graph[ref_img_path] = (np.eye(3), np.zeros((3, 1)))

for i in range(1, len(scene_df)):
    img_path = scene_df['img_path'].iloc[i]
    pts1, pts2, good = match_keypoints(ref_img_path, img_path, orb, bf)

    if pts1 is None:
        print("Could not match:", img_path)
        pose_graph[img_path] = (None, None)
        continue

    print("Matches found:", len(good))
    R, t = estimate_pose_from_points(pts1, pts2)

    if R is not None and t is not None:
        pose_graph[img_path] = (R, t)
        print("Stored pose for:", Path(img_path).name)
    else:
        pose_graph[img_path] = (None, None)
        print("Pose estimation failed for:", Path(img_path).name)

plt.figure(figsize=(14, 6))

for img_path, (R, t) in pose_graph.items():
    if t is not None:
        x, y, z = t.flatten()
        plt.scatter(x, z, c='red')
        plt.text(x, z, Path(img_path).name[:10], fontsize=19)

plt.title("Estimated Camera Translations (X vs Z)")
plt.xlabel("X (translation)")
plt.ylabel("Z (translation)")
plt.grid(True)
plt.tight_layout()


def format_matrix(mat):
    if mat is None:
        return 'nan;nan;nan;nan;nan;nan;nan;nan;nan'
    return ';'.join([str(round(x, 6)) for x in mat.flatten()])

def format_vector(vec):
    if vec is None:
        return 'nan;nan;nan'
    return ';'.join([str(round(x, 6)) for x in vec.flatten()])

df['rotation_matrix'] = df['img_path'].apply(lambda p: format_matrix(pose_graph.get(p, (None, None))[0]))
df['translation_vector'] = df['img_path'].apply(lambda p: format_vector(pose_graph.get(p, (None, None))[1]))

submission = df[['dataset', 'pred_scene', 'image', 'rotation_matrix', 'translation_vector']]
submission.columns = ['dataset', 'scene', 'image', 'rotation_matrix', 'translation_vector']

submission.to_csv('submission.csv', index=False)
print("submission.csv written!")
print(submission.head())


def draw_matches_statue(img1_path, img2_path, detector, matcher, max_matches=30):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) > max_matches:
        good_matches = good_matches[:max_matches]

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    height = max(h1, h2)
    vis = np.zeros((height, w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:] = img2

    for match in good_matches:
        pt1 = tuple(np.round(kp1[match.queryIdx].pt).astype(int))
        pt2 = tuple(np.round(kp2[match.trainIdx].pt).astype(int))
        pt2 = (pt2[0] + w1, pt2[1])

        color = (255, 51, 255)
        cv2.line(vis, pt1, pt2, color, thickness=5)

    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(14, 6))
    plt.imshow(vis)
    plt.title("Statue Image Matches")
    plt.axis('off')
    plt.tight_layout()

orb = cv2.ORB_create(nfeatures=4000, scaleFactor=1.2, nlevels=8)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

img1_path = df[df['pred_scene'] != 'outlier']['img_path'].iloc[0]
img2_path = df[df['pred_scene'] != 'outlier']['img_path'].iloc[1]

draw_matches_statue(img1_path, img2_path, orb, bf)


def draw_matches_taj_mahal(detector, matcher, max_matches=30):
    folder = Path("/kaggle/input/image-matching-challenge-2025/train/pt_sacrecoeur_trevi_tajmahal")

    image_files = sorted(list(folder.glob("*.png")) + list(folder.glob("*.jpg")))
    if len(image_files) < 2:
        print("Not enough images in the folder.")
        return

    img1_path = str(image_files[0])
    img2_path = str(image_files[1])

    print("Matching:", image_files[0].name, "↔", image_files[1].name)

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) > max_matches:
        good_matches = good_matches[:max_matches]

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    height = max(h1, h2)
    vis = np.zeros((height, w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:] = img2

    for match in good_matches:
        pt1 = tuple(np.round(kp1[match.queryIdx].pt).astype(int))
        pt2 = tuple(np.round(kp2[match.trainIdx].pt).astype(int))
        pt2 = (pt2[0] + w1, pt2[1])
        cv2.line(vis, pt1, pt2, (50, 50, 255), thickness=2)

    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(14, 6))
    plt.imshow(vis)
    plt.title("Taj Mahal Image Matches")
    plt.axis('off')
    plt.tight_layout()

draw_matches_taj_mahal(orb, bf)


def draw_matches_church(detector, matcher, max_matches=30):
    folder = Path("/kaggle/input/image-matching-challenge-2025/train/imc2023_theather_imc2024_church")

    image_files = sorted(list(folder.glob("*.png")) + list(folder.glob("*.jpg")))
    if len(image_files) < 2:
        print("Not enough images in the folder.")
        return

    img1_path = str(image_files[0])
    img2_path = str(image_files[1])

    print("Matching:", image_files[0].name, "↔", image_files[1].name)

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) > max_matches:
        good_matches = good_matches[:max_matches]

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    height = max(h1, h2)
    vis = np.zeros((height, w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:] = img2

    for match in good_matches:
        pt1 = tuple(np.round(kp1[match.queryIdx].pt).astype(int))
        pt2 = tuple(np.round(kp2[match.trainIdx].pt).astype(int))
        pt2 = (pt2[0] + w1, pt2[1])
        cv2.line(vis, pt1, pt2, (0, 0, 0), thickness=2)

    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(14, 6))
    plt.imshow(vis)
    plt.title("Church Image Matches")
    plt.axis('off')
    plt.tight_layout()

draw_matches_church(orb, bf)


def draw_matches_ets(detector, matcher, max_matches=30):
    folder = Path("/kaggle/input/image-matching-challenge-2025/train/ETs")

    image_files = sorted(list(folder.glob("*.png")) + list(folder.glob("*.jpg")))
    if len(image_files) < 2:
        print("Not enough images in the folder.")
        return

    img1_path = str(image_files[0])
    img2_path = str(image_files[1])

    print("Matching:", image_files[0].name, "↔", image_files[1].name)

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) > max_matches:
        good_matches = good_matches[:max_matches]

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    height = max(h1, h2)
    vis = np.zeros((height, w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:] = img2

    for match in good_matches:
        pt1 = tuple(np.round(kp1[match.queryIdx].pt).astype(int))
        pt2 = tuple(np.round(kp2[match.trainIdx].pt).astype(int))
        pt2 = (pt2[0] + w1, pt2[1])
        cv2.line(vis, pt1, pt2, (0, 255, 0), thickness=2)

    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(vis)
    plt.title("ETs Image Matches")
    plt.axis('off')
    plt.tight_layout()

draw_matches_ets(orb, bf)


def draw_matches_baalshamin(detector, matcher, max_matches=30):
    folder = Path("/kaggle/input/image-matching-challenge-2025/train/imc2024_dioscuri_baalshamin")

    image_files = sorted(list(folder.glob("*.png")) + list(folder.glob("*.jpg")))
    if len(image_files) < 2:
        print("Not enough images in the folder.")
        return

    img1_path = str(image_files[0])
    img2_path = str(image_files[1])

    print("Matching:", image_files[0].name, "↔", image_files[1].name)

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) > max_matches:
        good_matches = good_matches[:max_matches]

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    height = max(h1, h2)
    vis = np.zeros((height, w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:] = img2

    for match in good_matches:
        pt1 = tuple(np.round(kp1[match.queryIdx].pt).astype(int))
        pt2 = tuple(np.round(kp2[match.trainIdx].pt).astype(int))
        pt2 = (pt2[0] + w1, pt2[1])
        cv2.line(vis, pt1, pt2, (0, 0, 0), thickness=5)

    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(14, 6))
    plt.imshow(vis)
    plt.title("Baalshamin Image Matches")
    plt.axis('off')
    plt.tight_layout()

draw_matches_baalshamin(orb, bf)

