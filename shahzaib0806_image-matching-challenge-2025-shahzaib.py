import pandas as pd
import numpy as np
import os


# Path to train_labels.csv
labels_path = "/kaggle/input/image-matching-challenge-2025/train_labels.csv"

# Load the CSV into a DataFrame
df = pd.read_csv(labels_path)

# Display first 5 rows
df.head()


# Count total images
print("Total images in training set:", len(df))

# Unique datasets
print("\nUnique datasets:")
print(df['dataset'].value_counts())

# Unique scenes
print("\nUnique scenes:")
print(df['scene'].value_counts())


# Ek row select karo DataFrame se
row = df.iloc[0]

# Extract values
dataset = row['dataset']
image_name = row['image']

# Image path banao
image_path = f"/kaggle/input/image-matching-challenge-2025/train/{dataset}/{image_name}"
print("Image path:", image_path)


import cv2
import matplotlib.pyplot as plt

# Load image in BGR format (OpenCV ka default)
image = cv2.imread(image_path)

# Convert BGR to RGB (matplotlib sahi colors ke liye)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show image
plt.figure(figsize=(6, 6))
plt.imshow(image_rgb)
plt.title(image_name)
plt.axis('off')
plt.show()


# Rotation & Translation
rotation = row['rotation_matrix']
translation = row['translation_vector']

print("Rotation Matrix:", rotation)
print("Translation Vector:", translation)


# Rotation & Translation Strings
rotation_str = row['rotation_matrix']
translation_str = row['translation_vector']

# Convert strings to list of floats
rotation_list = [float(x) for x in rotation_str.split(';')]
translation_list = [float(x) for x in translation_str.split(';')]


import numpy as np

# Convert to NumPy arrays
rotation_matrix = np.array(rotation_list).reshape((3, 3))
translation_vector = np.array(translation_list).reshape((3, 1))  # column vector

# Print
print("Rotation Matrix:\n", rotation_matrix)
print("\nTranslation Vector:\n", translation_vector)


# Sabhi translation vectors ko extract karo
translations = []

for i in range(len(df)):
    t_str = df.iloc[i]['translation_vector']
    t = [float(x) for x in t_str.split(';')]
    translations.append(t)

# NumPy array bana lo (shape: [num_images, 3])
translations = np.array(translations)


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of translations
ax.scatter(translations[:, 0], translations[:, 1], translations[:, 2], c='blue', s=5)

ax.set_title("3D Image Positions (Translation Vectors)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()


camera_centers = []
camera_directions = []

for i in range(len(df)):
    # Parse translation
    t_str = df.iloc[i]['translation_vector']
    t = np.array([float(x) for x in t_str.split(';')]).reshape((3, 1))
    
    # Parse rotation
    r_str = df.iloc[i]['rotation_matrix']
    r = np.array([float(x) for x in r_str.split(';')]).reshape((3, 3))
    
    # Camera center (origin)
    camera_centers.append(t.flatten())
    
    # Camera direction vector (Z axis)
    direction = r @ np.array([0, 0, 1])  # or -r[:, 2]
    camera_directions.append(direction)


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

camera_centers = np.array(camera_centers)
camera_directions = np.array(camera_directions)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot camera positions
ax.scatter(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2], color='blue', s=5)

# Plot direction arrows
scale = 0.1  # arrow length
for i in range(len(camera_centers)):
    c = camera_centers[i]
    d = camera_directions[i]
    ax.quiver(c[0], c[1], c[2], d[0], d[1], d[2], length=scale, color='red')

ax.set_title("Camera Orientations in 3D Space")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()


grouped = df.groupby(['dataset', 'scene'])

# Show how many images are in each group
for (dataset, scene), group in grouped:
    print(f"{dataset} / {scene} => {len(group)} images")


# Outliers are usually in dataset/scene == "unknown" or with single images
outlier_candidates = df.groupby(['dataset', 'scene']).filter(lambda g: len(g) <= 2)

print(f"Possible outliers found: {len(outlier_candidates)}")
display(outlier_candidates.head())


import os

folder_path = '/kaggle/input/image-matching-challenge-2025/train/imc2024_lizard_pond'
image_files = os.listdir(folder_path)

print("Available image files:", image_files[:10])  # First 10 filenames


img1_path = '/kaggle/input/image-matching-challenge-2025/train/imc2024_lizard_pond/lizard_image_0003.png'
img2_path = '/kaggle/input/image-matching-challenge-2025/train/imc2024_lizard_pond/lizard_image_0007.png'


img1_path = '/kaggle/input/image-matching-challenge-2025/train/imc2024_lizard_pond/pond_image_0012.png'
img2_path = '/kaggle/input/image-matching-challenge-2025/train/imc2024_lizard_pond/pond_image_0021.png'



import os

base_path = "/kaggle/input/image-matching-challenge-2025/train"

print("Datasets available in /train:")
print(os.listdir(base_path))




folder_path = base_path + "/imc2024_lizard_pond"
print("Files in imc2024_lizard_pond:")
print(os.listdir(folder_path)[:10])  # First 10 files



import os

folder_path = "/kaggle/input/image-matching-challenge-2025/train/imc2024_lizard_pond"

# List actual files in the folder
files = os.listdir(folder_path)
print("Total files:", len(files))
print("First 20 files:")
for f in files[:20]:
    print(f)





import cv2
import matplotlib.pyplot as plt

img1_path = "/kaggle/input/image-matching-challenge-2025/train/imc2024_lizard_pond/lizard_00003.png"
img2_path = "/kaggle/input/image-matching-challenge-2025/train/imc2024_lizard_pond/lizard_00361.png"

# Load grayscale
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# Show images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img1, cmap='gray')
plt.title("Image 1")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img2, cmap='gray')
plt.title("Image 2")
plt.axis('off')

plt.show()



import cv2
import matplotlib.pyplot as plt

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test (Lowe's ratio test)
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Draw top 50 matches
matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Show the matches
plt.figure(figsize=(20, 10))
plt.imshow(matched_img)
plt.title("Top 50 Feature Matches")
plt.axis('off')
plt.show()


import numpy as np

# Convert keypoints to coordinates
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# Assume fx = fy = 1, cx = cy = 0 (normalized coords) â€“ or use real intrinsics if available
K = np.eye(3)  # identity intrinsic matrix (you can replace this later with real intrinsics)

# Estimate Essential Matrix
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# Recover relative pose from Essential Matrix
_, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

print("Rotation Matrix (R):\n", R)
print("Translation Vector (t):\n", t)



print("Original matches:", len(matches))
print("Inliers found by RANSAC:", np.sum(mask))


E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=2.0, prob=0.999)


import cv2

# Load grayscale images
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# Check they loaded properly
if img1 is None or img2 is None:
    raise ValueError("Images couldn't be loaded. Check file paths.")

# Create SIFT detector
sift = cv2.SIFT_create()

# Detect and compute descriptors
kp1, desc1 = sift.detectAndCompute(img1, None)
kp2, desc2 = sift.detectAndCompute(img2, None)


# Matcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(desc1, desc2, k=2)

# Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print("Good matches:", len(good_matches))



# Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.8 * n.distance:  # pehle 0.75 tha
        good_matches.append(m)



print("Total good matches:", len(good_matches))
if len(good_matches) < 8:
    print(" Not enough matches for pose estimation.")



img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)
plt.figure(figsize=(15, 10))
plt.imshow(img_matches)
plt.title("Feature Matches")
plt.axis('off')
plt.show()



img1_path = '/kaggle/input/image-matching-challenge-2025/train/imc2024_lizard_pond/lizard_00361.png'
img2_path = '/kaggle/input/image-matching-challenge-2025/train/imc2024_lizard_pond/lizard_00527.png'



orb = cv2.ORB_create(1000)
kp1, desc1 = orb.detectAndCompute(img1, None)
kp2, desc2 = orb.detectAndCompute(img2, None)


print(f"desc1: {desc1 is not None}, shape: {desc1.shape if desc1 is not None else 'None'}")
print(f"desc2: {desc2 is not None}, shape: {desc2.shape if desc2 is not None else 'None'}")
print(f"Keypoints in img1: {len(kp1)}, img2: {len(kp2)}")




if desc1 is not None and desc2 is not None:
    matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"Good matches found: {len(good_matches)}")

    if len(good_matches) > 0:
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)
        plt.figure(figsize=(15, 10))
        plt.imshow(img_matches)
        plt.title("Good Matches")
        plt.axis('off')
        plt.show()
    else:
        print(" No good matches to display.")
else:
    print(" One or both descriptors are None. Skipping matching.")



# Extract matched keypoints
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])



# Temporary guess â€” replace this with real intrinsics from dataset!
fx = fy = 1000  # Focal length in pixels
cx = 512        # Principal point x (assuming image size ~1024x768)
cy = 384        # Principal point y

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])


import os

base_path = "/kaggle/input/image-matching-challenge-2025/train/imc2024_lizard_pond"

for root, dirs, files in os.walk(base_path):
    print(f" Directory: {root}")
    for file in files:
        print(f"    ğŸ—� {file}")




fx = 1600 * 0.8  # Focal length in x (assume 80% of width)
fy = 1200 * 0.8  # Focal length in y (assume 80% of height)
cx = 1600 / 2    # Principal point x (center of image)
cy = 1200 / 2    # Principal point y (center of image)

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])



# Convert matched keypoints to normalized coordinates using K
pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches])

# Normalize points
pts1_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), K, None)
pts2_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), K, None)



print("pts1 dtype:", pts1.dtype, " shape:", pts1.shape)
print("pts1 contiguous?:", pts1.flags['C_CONTIGUOUS'])

print("pts2 dtype:", pts2.dtype, " shape:", pts2.shape)
print("pts2 contiguous?:", pts2.flags['C_CONTIGUOUS'])



pts1 = np.ascontiguousarray(pts1, dtype=np.float32)
pts2 = np.ascontiguousarray(pts2, dtype=np.float32)



print("pts1 shape:", pts1.shape, " dtype:", pts1.dtype, " contiguous?:", pts1.flags['C_CONTIGUOUS'])
print("pts2 shape:", pts2.shape, " dtype:", pts2.dtype, " contiguous?:", pts2.flags['C_CONTIGUOUS'])



print("Number of good matches:", len(good_matches))


if len(good_matches) >= 8:
    # Proceed as usual
    ...
else:
    print("Not enough matches to compute pose. Need at least 8, got", len(good_matches))



sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)



bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)



if m.distance < 0.9 * n.distance:
    good_matches.append(m)



img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None)
plt.imshow(img_matches)
plt.show()



import numpy as np
import cv2

# Assume you already have:
# K      : Intrinsic matrix
# R, t   : Output from cv2.recoverPose
# pts1, pts2 : Matching keypoints (Nx2)

# Step 1: Projection matrix for camera 1 (identity pose)
P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))  # [I | 0]

# Step 2: Projection matrix for camera 2 (recovered pose)
P2 = K @ np.hstack((R, t))  # [R | t]

# Step 3: Triangulate points
pts1 = pts1.T  # shape (2, N)
pts2 = pts2.T  # shape (2, N)

pts4d_hom = cv2.triangulatePoints(P1, P2, pts1, pts2)  # shape (4, N)
pts4d = pts4d_hom[:3] / pts4d_hom[3]  # Convert from homogeneous to 3D (shape 3xN)

# (Optional) Transpose to get Nx3 array
pts3d = pts4d.T

print("Triangulated 3D points:\n", pts3d)




import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming pts3d is of shape (N, 3)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()



pts2d_1 = np.array([[523.4, 422.1]])  # from image 1
pts2d_2 = np.array([[518.7, 419.5]])  # from image 2



import numpy as np
from scipy.optimize import least_squares

def project_points(X, R, t, K):
    X = X.reshape(-1, 3)
    X_cam = (R @ X.T + t).T
    x_proj = X_cam[:, :2] / X_cam[:, 2:]
    x_proj_h = (K[:2, :2] @ x_proj.T).T + K[:2, 2]
    return x_proj_h



def reprojection_error(params, pts3d, pts2d, K):
    rvec = params[:3]
    tvec = params[3:6]
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    projected = project_points(pts3d, R, t, K)
    return (projected - pts2d).ravel()



pts3d = pts4d[:3, :].T  # Convert from homogeneous to XYZ


import numpy as np
import cv2
from scipy.optimize import least_squares

# Step 1: Convert from homogeneous 4D to 3D
pts3d = pts4d[:3, :].T   # shape: (N, 3)

# Step 2: Use the corresponding 2D points (say, pts1 for Camera 1)
pts2d = pts1.reshape(-1, 2)  # shape: (N, 2)

# Step 3: Initial rotation vector from cv2.Rodrigues
rvec_init, _ = cv2.Rodrigues(R)
params_init = np.hstack((rvec_init.ravel(), t.ravel()))

# Step 4: Reprojection error function
def project_points(X, R, t, K):
    X = X.reshape(-1, 3)
    X_cam = (R @ X.T + t).T
    x_proj = X_cam[:, :2] / X_cam[:, 2:]
    x_proj_h = (K[:2, :2] @ x_proj.T).T + K[:2, 2]
    return x_proj_h

def reprojection_error(params, pts3d, pts2d, K):
    rvec = params[:3]
    tvec = params[3:6]
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    projected = project_points(pts3d, R, t, K)
    return (projected - pts2d).ravel()

# Step 5: Optimize
res = least_squares(reprojection_error, params_init,
                    args=(pts3d, pts2d, K))

# Step 6: Get refined pose
rvec_opt = res.x[:3]
tvec_opt = res.x[3:6]
R_opt, _ = cv2.Rodrigues(rvec_opt)
t_opt = tvec_opt.reshape(3, 1)

print("Refined Rotation:\n", R_opt)
print("Refined Translation:\n", t_opt)



import os

base_path = "/kaggle/input/image-matching-challenge-2025"
for root, dirs, files in os.walk(base_path):
    print("Directory:", root)
    for name in files:
        print("  File:", name)



scene_name = "amy_gardens"



import os

scene_name = "amy_gardens"
image_dir = f"/kaggle/input/image-matching-challenge-2025/train/{scene_name}"

# Yeh check karega kya files hain wahan
image_list = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

# Show first 5 images
print("Images mil gayi hain:", image_list[:5])



sift = cv2.SIFT_create()

# Keypoints & Descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Matching with FLANN
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio test
good_matches = []
pts1 = []
pts2 = []

for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

pts1 = np.array(pts1, dtype=np.float32)
pts2 = np.array(pts2, dtype=np.float32)

print("Good matches:", len(good_matches))



print("Essential Matrix shape:", E.shape)



print("pts1:", pts1.shape, "pts2:", pts2.shape)



mask_valid = np.isfinite(pts1).all(axis=1) & np.isfinite(pts2).all(axis=1)
pts1 = pts1[mask_valid]
pts2 = pts2[mask_valid]



E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

if E is None or E.shape != (3, 3):
    print("Invalid Essential Matrix. Skipping this image pair.")
else:
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)



E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

if E is not None and E.shape == (3, 3):
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
else:
    print("Essential matrix not valid:", None if E is None else E.shape)



if len(pts1) >= 8:
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    if E is not None and E.shape == (3, 3):
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    else:
        print("Invalid Essential Matrix shape:", None if E is None else E.shape)
else:
    print(f" Not enough matches to compute Essential Matrix. Got only {len(pts1)}")



import os
import cv2
import numpy as np

# Folder jahan images hain
scene_name = "amy_gardens"
image_dir = f"/kaggle/input/image-matching-challenge-2025/train/{scene_name}"

# Image file names set karo
img1_name = "peach_0000.png"
img2_name = "peach_0001.png"

# Load grayscale images
img1 = cv2.imread(f"{image_dir}/{img1_name}", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(f"{image_dir}/{img2_name}", cv2.IMREAD_GRAYSCALE)

# Check karo images sahi load hui ya nahi
assert img1 is not None, "Image 1 load nahi hui"
assert img2 is not None, "Image 2 load nahi hui"

# SIFT detector
sift = cv2.SIFT_create()

# Detect and compute
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN matcher
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# Ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print("Good matches found:", len(good_matches))




# Matched keypoints nikaalo
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])



# Approximate intrinsic matrix (assuming image size ~1600x1200)
K = np.array([
    [1600, 0, 800],
    [0, 1600, 600],
    [0, 0, 1]
], dtype=np.float64)



E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
if E is None or E.shape != (3, 3):
    print(" Essential Matrix compute nahi hui")
else:
    print(" Essential Matrix shape:", E.shape)



# Pose recover karo
_, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
print("Recovered Rotation:\n", R)
print("Recovered Translation:\n", t)



# Projection matrices
P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K @ np.hstack((R, t))

# Triangulation
pts4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
pts3d = (pts4d_hom / pts4d_hom[3])[:3].T  # X, Y, Z points
print("Triangulated 3D points:\n", pts3d)



# Required imports
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm  # for progress bar

# Intrinsic matrix (assuming approx calibration)
K = np.array([[1600, 0, 800],
              [0, 1600, 600],
              [0, 0, 1]], dtype=np.float64)

# Path setup
scene_name = "amy_gardens"
image_dir = f"/kaggle/input/image-matching-challenge-2025/train/{scene_name}"
image_list = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

# Use first image as reference
ref_img_name = image_list[0]
camera_poses = {ref_img_name: (np.eye(3), np.zeros((3, 1)))}  # R = I, t = 0

# Store 3D points and their image observations
points_3d_all = []
observations_all = []



# ORB Detector
orb = cv2.ORB_create(5000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Load reference image
ref_img = cv2.imread(f"{image_dir}/{ref_img_name}", cv2.IMREAD_GRAYSCALE)
kp1, des1 = orb.detectAndCompute(ref_img, None)

for next_img_name in tqdm(image_list[1:], desc="Estimating poses"):
    next_img = cv2.imread(f"{image_dir}/{next_img_name}", cv2.IMREAD_GRAYSCALE)
    kp2, des2 = orb.detectAndCompute(next_img, None)

    # Match features
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 8:
        print(f" Not enough matches between {ref_img_name} and {next_img_name}")
        continue

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Estimate Essential matrix
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    if E is None or E.shape != (3, 3):
        print(f" Essential matrix computation failed for {next_img_name}")
        continue

    # Recover pose
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    # Save pose for this image
    camera_poses[next_img_name] = (R, t)

    print(f" Pose estimated for {next_img_name}")



triangulated_points = []

# Reference pose
R1, t1 = camera_poses[ref_img_name]
P1 = K @ np.hstack((R1, t1))  # Projection matrix for reference image

for img_name in tqdm(list(camera_poses.keys())[1:], desc="Triangulating points"):
    R2, t2 = camera_poses[img_name]
    P2 = K @ np.hstack((R2, t2))  # Projection matrix for the next image

    # Load both images again
    img1 = cv2.imread(f"{image_dir}/{ref_img_name}", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(f"{image_dir}/{img_name}", cv2.IMREAD_GRAYSCALE)

    # Detect and match features
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:100]  # Top 100

    if len(matches) < 8:
        print(f" Not enough matches for triangulation with {img_name}")
        continue

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Triangulate
    pts1_hom = cv2.undistortPoints(pts1.reshape(-1,1,2), K, None)
    pts2_hom = cv2.undistortPoints(pts2.reshape(-1,1,2), K, None)

    pts_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts_3d = (pts_4d_hom[:3] / pts_4d_hom[3]).T  # Convert to Nx3

    triangulated_points.append(pts_3d)

    print(f" Triangulated {pts_3d.shape[0]} points from {ref_img_name} and {img_name}")



import cv2
import os

scene_name = "amy_gardens"
image_dir = f"/kaggle/input/image-matching-challenge-2025/train/{scene_name}"
image_list = sorted(os.listdir(image_dir))

img1_name = image_list[0]
ref_img_path = os.path.join(image_dir, img1_name)

ref_img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)

if ref_img is None:
    print(" Image load nahi hui:", ref_img_path)
else:
    print(" Image load hogayi:", ref_img.shape)



import os
import cv2

scene_name = "amy_gardens"
image_dir = f"/kaggle/input/image-matching-challenge-2025/train/{scene_name}"
print("Directory exists:", os.path.exists(image_dir))
print("Files in directory:", os.listdir(image_dir))



import os
import cv2

image_dir = "/kaggle/input/image-matching-challenge-2025/train/amy_gardens"
image_list = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

img1_name = image_list[0]
ref_img_path = os.path.join(image_dir, img1_name)

ref_img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)

if ref_img is None:
    print(" Image load nahi hui:", ref_img_path)
else:
    print(" Image load hogayi:", ref_img.shape)

    # Detect features
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(ref_img, None)
    print("Keypoints found:", len(kp1))



import numpy as np

def save_submission_file(points3D, camera_poses_dict, image_names, scene_name):
    """
    points3D: Nx3 numpy array
    camera_poses_dict: dict of {image_name: (R, t)} where R: 3x3, t: 3x1
    image_names: list of image names in order
    scene_name: e.g. "amy_gardens"
    """

    # Projection matrices P = K [R|t]
    K = np.array([
        [2759.48, 0, 1520.69],
        [0, 2764.16, 1006.81],
        [0, 0, 1]
    ])  # â†� use correct intrinsics if needed

    camera_poses = []
    for name in image_names:
        R, t = camera_poses_dict[name]
        Rt = np.hstack([R, t])
        P = K @ Rt
        camera_poses.append(P)

    camera_poses = np.stack(camera_poses)  # M x 3 x 4
    points3D = np.array(points3D)          # N x 3

    # Save as .npz
    np.savez_compressed(f"{scene_name}.npz",
                        points3D=points3D,
                        camera_poses=camera_poses,
                        image_names=np.array(image_names))

    print(f" Submission file saved: {scene_name}.npz")



camera_poses_dict = camera_poses



image_names = sorted(os.listdir(image_dir))


image_names = image_list


print("3D points collected:", len(points_3d_all))


img1_name = "peach_0000.png"
img2_name = "peach_0001.png"



if len(points_3d_all) > 0:
    points3D = np.vstack(points_3d_all)
else:
    print(" Koi 3D point triangulate nahi hua.")



import cv2
import numpy as np
import os

# Image paths
scene_name = "amy_gardens"
image_dir = f"/kaggle/input/image-matching-challenge-2025/train/{scene_name}"
img1_name = "peach_0000.png"
img2_name = "peach_0001.png"
img1_path = f"{image_dir}/{img1_name}"
img2_path = f"{image_dir}/{img2_name}"

# Load images
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# Check images loaded
print("Image 1 shape:", img1.shape)
print("Image 2 shape:", img2.shape)

# Camera intrinsics (example K)
K = np.array([[1000, 0, img1.shape[1] // 2],
              [0, 1000, img1.shape[0] // 2],
              [0, 0, 1]])

# SIFT + matching
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print("Good matches found:", len(good_matches))

# Extract matched points
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# Compute essential matrix
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# Recover pose
if E is not None and E.shape == (3, 3):
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    # Projection matrices
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, t))

    # Triangulate
    pts1_tr = pts1[mask_pose.ravel() == 1]
    pts2_tr = pts2[mask_pose.ravel() == 1]

    if pts1_tr.shape[0] >= 8:
        pts4d_hom = cv2.triangulatePoints(K @ P1, K @ P2, pts1_tr.T, pts2_tr.T)
        pts3d = (pts4d_hom[:3] / pts4d_hom[3]).T
        print(" Triangulated 3D points:", pts3d.shape)
    else:
        print(" Not enough inliers to triangulate")
else:
    print(" Essential matrix not valid")



scene_name = "amy_gardens"
image_dir = f"/kaggle/input/image-matching-challenge-2025/train/{scene_name}"
image_list = sorted(os.listdir(image_dir))
print("Images found:", image_list)



import numpy as np

K = np.array([
    [1500, 0, 960],  # fx, 0, cx
    [0, 1500, 540],  # 0, fy, cy
    [0, 0, 1]
])



import cv2
import numpy as np
import os

image_dir = "/kaggle/input/image-matching-challenge-2025/train/amy_gardens/peach_0000.png"
sift = cv2.SIFT_create()

img1 = cv2.imread(os.path.join(image_dir, "peach_0000.png"), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(os.path.join(image_dir, "peach_0001.png"), cv2.IMREAD_GRAYSCALE)

# Check if images are loaded
if img1 is None or img2 is None:
    print(" Image not loaded. Check path.")
else:
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"Good matches found: {len(good_matches)}")



import os

# Try finding peach_0000.png file
for root, dirs, files in os.walk("/kaggle/input/image-matching-challenge-2025"):
    for name in files:
        if "peach_0000" in name:
            print("Found:", os.path.join(root, name))



import os
import cv2

scene_name = "amy_gardens"
image_name = "peach_0000.png"

image_dir = f"/kaggle/input/image-matching-challenge-2025/train/{scene_name}"
img_path = os.path.join(image_dir, image_name)

img = cv2.imread(img_path)

if img is None:
    print(" Image not loaded, check path again:", img_path)
else:
    print(" Image loaded successfully:", img.shape)




img1 = cv2.imread(os.path.join(image_dir, "peach_0000.png"))
img2 = cv2.imread(os.path.join(image_dir, "peach_0001.png"))

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initialize SIFT
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Match features using FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print("Good matches found:", len(good_matches))



import os
import cv2
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path

# Scene ka naam sahi se likho (jo folder mein hai)
scene_name = "amy_gardens"
scene_path = f"/kaggle/input/image-matching-challenge-2025/train/{scene_name}"
output_path = "/kaggle/working/submission.json"

# Images load karo
image_files = sorted([f for f in os.listdir(scene_path) if f.endswith(".png")])
image_paths = [os.path.join(scene_path, f) for f in image_files]

# Dummy intrinsics (agar .npz available hai toh usse load karo)
K = np.array([[1200, 0, 512], [0, 1200, 288], [0, 0, 1]])  # Focal length and principal point

# SIFT detector
sift = cv2.SIFT_create()

def extract_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load: {img_path}")
        return None, None
    kp, des = sift.detectAndCompute(img, None)
    return kp, des

def match_features(desc1, desc2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good

def get_matched_points(kp1, kp2, matches):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2

def estimate_pose(pts1, pts2, K):
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    return R, t

def triangulate_points(R, t, pts1, pts2, K):
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))
    pts1_h = cv2.convertPointsToHomogeneous(pts1)[:, 0, :]
    pts2_h = cv2.convertPointsToHomogeneous(pts2)[:, 0, :]
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = (points_4d[:3] / points_4d[3]).T
    return points_3d

#  Loop through image pairs
results = []

for i in tqdm(range(len(image_paths) - 1)):
    img1_path = image_paths[i]
    img2_path = image_paths[i + 1]
    img1_name = Path(img1_path).name
    img2_name = Path(img2_path).name

    kp1, des1 = extract_features(img1_path)
    kp2, des2 = extract_features(img2_path)

    if des1 is None or des2 is None:
        continue

    matches = match_features(des1, des2)
    if len(matches) < 10:
        continue

    pts1, pts2 = get_matched_points(kp1, kp2, matches)

    try:
        R, t = estimate_pose(pts1, pts2, K)
        points_3d = triangulate_points(R, t, pts1, pts2, K)

        pair_data = {
            "image0": img1_name,
            "image1": img2_name,
            "qvec": [1.0, 0.0, 0.0, 0.0],  # identity quaternion (dummy)
            "tvec": [float(t[0]), float(t[1]), float(t[2])],  # translation vector
            "points3D": points_3d[:100].tolist()  # limit to first 100 points
        }
        results.append(pair_data)

    except Exception as e:
        print(f"Failed on pair {img1_name}, {img2_name}: {e}")
        continue

# Save submission file
with open(output_path, "w") as f:
    json.dump(results, f)

print(f"\n Submission JSON saved to: {output_path}")



import pandas as pd

# Example: suppose you have list of image names and their camera poses
data = {
    'image_name': ['peach_0000.png', 'peach_0001.png'],
    'q_w': [1.0, 0.98],
    'q_x': [0.0, 0.01],
    'q_y': [0.0, -0.02],
    'q_z': [0.0, 0.005],
    't_x': [0.0, 1.2],
    't_y': [0.0, -0.3],
    't_z': [0.0, 0.5],
}

df = pd.DataFrame(data)
df.to_csv("submission.csv", index=False)



df.to_csv("submission.csv", index=False)



import pandas as pd

# Example list of matches and their scores
data = {
    "pair_id": [
        "peach_valley/peach_0000-0001.png",
        "peach_valley/peach_0001-0002.png"
    ],
    "score": [
        0.8432,
        0.9134
    ]
}

df = pd.DataFrame(data)

# Save to submission.csv
df.to_csv("submission.csv", index=False)
print(" submission.csv file created successfully")



import pandas as pd

# Dummy pose data (replace with actual estimates)
data = {
    "image_name": ["peach_0000.png", "peach_0001.png"],
    "q_w": [1.0, 0.99],
    "q_x": [0.0, 0.01],
    "q_y": [0.0, 0.02],
    "q_z": [0.0, 0.03],
    "t_x": [0.0, 0.1],
    "t_y": [0.0, 0.2],
    "t_z": [0.0, 0.3],
}

df = pd.DataFrame(data)

# Save to submission.csv
df.to_csv("submission.csv", index=False)
print(" submission.csv file created successfully")


