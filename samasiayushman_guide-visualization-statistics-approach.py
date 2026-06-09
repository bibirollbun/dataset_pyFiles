# Install 
import subprocess
import sys

def silent_install(packages):
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q"] + packages,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

required_packages = [
    "open3d",
    "pycolmap",
    "numpy",
    "torch",
    "torchvision",
    "opencv-python",
    "timm"
]

# Install silently
silent_install(required_packages)

# Verify 
try:
    import open3d, pycolmap, torch, cv2, timm
    print("✅ All packages installed successfully!")
except ImportError as e:
    print(f"❌ Error: {e}")


import cv2
import matplotlib.pyplot as plt

# Detect keypoints
sift = cv2.SIFT_create()
img=cv2.imread("/kaggle/input/image-matching-challenge-2025/train/ETs/another_et_another_et001.png")
kp = sift.detect(img, None)

# Draw keypoints
img_kp = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img_kp)
plt.title(f"SIFT Keypoints: {len(kp)}")
plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt


img1 = cv2.imread("/kaggle/input/image-matching-challenge-2025/train/ETs/another_et_another_et001.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("/kaggle/input/image-matching-challenge-2025/train/ETs/another_et_another_et005.png", cv2.IMREAD_GRAYSCALE)


detector = cv2.ORB_create()  # or cv2.SIFT_create()


kp1, desc1 = detector.detectAndCompute(img1, None)
kp2, desc2 = detector.detectAndCompute(img2, None)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # For ORB
matches = bf.match(desc1, desc2)
matches = sorted(matches, key=lambda x: x.distance)[:50]  # Top 50 matches


img_matches = cv2.drawMatches(
    img1, kp1, img2, kp2, matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


plt.figure(figsize=(20, 10))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title("Top 50 Feature Matches")
plt.axis('off')
plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt


img1 = cv2.imread('/kaggle/input/image-matching-challenge-2025/train/ETs/another_et_another_et001.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/kaggle/input/image-matching-challenge-2025/train/ETs/another_et_another_et002.png', cv2.IMREAD_GRAYSCALE)


assert img1 is not None and img2 is not None, "Error loading images"


sift = cv2.SIFT_create()
kp1, desc1 = sift.detectAndCompute(img1, None)
kp2, desc2 = sift.detectAndCompute(img2, None)


bf = cv2.BFMatcher()
matches = bf.match(desc1, desc2)
matches = sorted(matches, key=lambda x: x.distance)[:50]  # Top 50 matches


pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)


F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)


def draw_epilines(img1, img2, pts1, pts2, F):
    # Convert to color for visualization
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    
    for r, pt1, pt2 in zip(lines1, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Draw epipolar line on img1
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [img1.shape[1], -(r[2] + r[0]*img1.shape[1])/r[1]])
        cv2.line(img1_color, (x0, y0), (x1, y1), color, 1)
        
        
        pt2_int = (int(pt2[0][0]), int(pt2[0][1])) 
        cv2.circle(img2_color, pt2_int, 5, color, -1)
    
    return img1_color, img2_color

# 7. Generate and display results
img1_epi, img2_epi = draw_epilines(img1, img2, pts1, pts2, F)

plt.figure(figsize=(12, 6))
plt.subplot(121); plt.imshow(cv2.cvtColor(img1_epi, cv2.COLOR_BGR2RGB)); plt.title("Epipolar Lines (Image 1)")
plt.subplot(122); plt.imshow(cv2.cvtColor(img2_epi, cv2.COLOR_BGR2RGB)); plt.title("Image 2 with Keypoints")
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, pearsonr


train_df = pd.read_csv("/kaggle/input/image-matching-challenge-2025/train_labels.csv")
threshold_df = pd.read_csv("/kaggle/input/image-matching-challenge-2025/train_thresholds.csv")


train_df.head()



scene_counts = train_df['scene'].value_counts()
print("Scene Distribution:")
print(scene_counts)


plt.figure(figsize=(10, 6))
scene_counts.plot(kind='bar', color='skyblue')
plt.title("Number of Images per Scene")
plt.xlabel("Scene")
plt.ylabel("Count")
plt.show()



def parse_matrix(matrix_str):
    return np.array([float(x) for x in matrix_str.split(";")]).reshape(3, 3)

def parse_vector(vector_str):
    return np.array([float(x) for x in vector_str.split(";")])

train_df['rotation_magnitude'] = train_df['rotation_matrix'].apply(lambda x: np.linalg.norm(parse_matrix(x)))
train_df['translation_magnitude'] = train_df['translation_vector'].apply(lambda x: np.linalg.norm(parse_vector(x)))

# Summary of transformations
print("\nSummary of Rotation and Translation Magnitudes:")
print(train_df[['rotation_magnitude', 'translation_magnitude']].describe())



plt.figure(figsize=(8, 6))
sns.histplot(train_df['translation_magnitude'], kde=True, bins=20)
plt.title("Distribution of Translation Magnitudes")
plt.xlabel("Translation Magnitude")
plt.ylabel("Frequency")
plt.show()



plt.figure(figsize=(8, 6))
sns.histplot(threshold_df['thresholds'], kde=True, bins=20)
plt.title("Distribution of Thresholds")
plt.xlabel("Threshold")
plt.ylabel("Frequency")
plt.show()





scenes = train_df['scene'].unique()
rotation_by_scene = [train_df[train_df['scene'] == scene]['rotation_magnitude'] for scene in scenes]
translation_by_scene = [train_df[train_df['scene'] == scene]['translation_magnitude'] for scene in scenes]

# ANOVA for rotation magnitude
f_stat_rot, p_value_rot = f_oneway(*rotation_by_scene)
print("\nANOVA Test for Rotation Magnitudes:")
print(f"F-statistic: {f_stat_rot}, P-value: {p_value_rot}")



f_stat_trans, p_value_trans = f_oneway(*translation_by_scene)
print("\nANOVA Test for Translation Magnitudes:")
print(f"F-statistic: {f_stat_trans}, P-value: {p_value_trans}")



correlation, p_value_corr = pearsonr(train_df['rotation_magnitude'], train_df['translation_magnitude'])
print("\nCorrelation Analysis:")
print(f"Pearson Correlation: {correlation}, P-value: {p_value_corr}")


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


merged_df = pd.merge(train_df, threshold_df, on="scene")


def extract_first_number(value):
    if isinstance(value, str):
        # Handle string values (e.g., "1.2;3.4" -> 1.2)
        return float(value.split(";")[0])
    elif isinstance(value, (int, float)):
        # Already numeric
        return float(value)
    else:
        raise ValueError(f"Unexpected value type: {type(value)}")


X = np.column_stack([
    merged_df['rotation_magnitude'].apply(extract_first_number),
    merged_df['translation_magnitude'].apply(extract_first_number)
])
y = merged_df['thresholds'].apply(extract_first_number)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

r_squared = regressor.score(X_test, y_test)
print("\nRegression Analysis:")
print(f"R-squared: {r_squared:.4f}")
print("Coefficients:", regressor.coef_)
print("Intercept:", regressor.intercept_)





#just Predicting ......
y_pred = regressor.predict(X_test)
print(y_pred)

