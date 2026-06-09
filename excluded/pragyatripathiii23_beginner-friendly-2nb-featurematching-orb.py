#LOADING THE DATASET
import os 
dataset_path = "/kaggle/input/"
os.listdir(dataset_path)


import pandas as pd

# Loading train_labels.csv file
train_path = "../input/image-matching-challenge-2025/train" 
train_labels = pd.read_csv('/kaggle/input/image-matching-challenge-2025/train_labels.csv')

# Loading train_thresholds.csv file
train_thresholds_path = "../input/image-matching-challenge-2025/train_thresholds.csv"
train_thresholds = pd.read_csv(train_thresholds_path)


import cv2  # For reading and processing images
import matplotlib.pyplot as plt  # For visualizing images
import os  # For handling file paths

#selecting a specific scene from the training dataset
scene_name = "fountain"  #The idea is that all images with the scene "fountain" are of the same physical location (the same place, from different angles or times).
scene_images = train_labels[train_labels["scene"] == scene_name]["image"].values[:2]  # selecting the first two images from that scene

# Creating two side-by-side subplots to display both images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for i, img_name in enumerate(scene_images):
    img_path = os.path.join(train_path, train_labels[train_labels["scene"] == scene_name]["dataset"].values[0], img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Loaded image in grayscale (easier for feature matching) is standard for keypoint detection â€” no color info needed.
    #feature matching is almost always done in grayscale (non-color images).
    axes[i].imshow(img, cmap="gray")
    axes[i].set_title(f"Image: {img_name}") ##show each image with its filename
    axes[i].axis("off")

plt.show()


#JUST GETTING PATH TO BOTH IMAGES SEPERATELY THROUGH THIS CODE FOR LATER ANALYSIS

# Get the dataset folder (since it's the same for all images in the scene)
dataset_folder = train_labels[train_labels["scene"] == scene_name]["dataset"].values[0]

# Construct full paths using the already selected `scene_images`
image_paths = [os.path.join(train_path, dataset_folder, img_name) for img_name in scene_images]

image1_path, image2_path = image_paths

#print("First image path:", image1_path)
#print("Second image path:", image2_path)


# Load and validate images
img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

# Optional: Resize large images to avoid long computation time
img1 = cv2.resize(img1, (800, 600))
img2 = cv2.resize(img2, (800, 600))

# FAST keypoint detection
fast = cv2.FastFeatureDetector_create()
keypoints1 = fast.detect(img1, None)
keypoints2 = fast.detect(img2, None)

# Draw keypoints
img1_with_kp = cv2.drawKeypoints(img1, keypoints1, None, color=(255, 0, 0))
img2_with_kp = cv2.drawKeypoints(img2, keypoints2, None, color=(255, 0, 0))

# Show the results
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.title("Image 1 - FAST Keypoints")
plt.imshow(img1_with_kp, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Image 2 - FAST Keypoints")
plt.imshow(img2_with_kp, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()  


import numpy as np
def rank_keypoints_by_harris(img, keypoints, top_k=20):
    # Convert image to float32 for cornerHarris
    gray = np.float32(img)

    # Compute Harris corner response
    harris_response = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    # Normalize the Harris response map (optional but useful for visualization or debugging)
    harris_response = cv2.normalize(harris_response, None, 0, 255, cv2.NORM_MINMAX)

    # Get Harris score for each keypoint
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if 0 <= y < harris_response.shape[0] and 0 <= x < harris_response.shape[1]:
            kp.response = harris_response[y, x]
        else:
            kp.response = 0  # Out of bounds safeguard

    # Sort by Harris response (descending) and return top K keypoints
    keypoints_sorted = sorted(keypoints, key=lambda x: x.response, reverse=True)
    return keypoints_sorted[:top_k]


# --- Rank and draw top 50 for both images ---
top_k = 50
top_kp1 = rank_keypoints_by_harris(img1, keypoints1, top_k)
top_kp2 = rank_keypoints_by_harris(img2, keypoints2, top_k)

# Draw top keypoints
img1_top = cv2.drawKeypoints(img1, top_kp1, None, color=(0, 255, 0))
img2_top = cv2.drawKeypoints(img2, top_kp2, None, color=(0, 255, 0))

# Plot top ranked keypoints
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.title("Image 1 - Top 50 Keypoints (Harris Ranked)")
plt.imshow(img1_top, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Image 2 - Top 50 Keypoints (Harris Ranked)")
plt.imshow(img2_top, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()


# Use ORB to compute orientation only (not descriptors)
orb = cv2.ORB_create()
keypoints1 = orb.compute(img1, keypoints1)[0]
keypoints2 = orb.compute(img2, keypoints2)[0]

# Draw keypoints with orientation arrows
img1_with_oriented_kp = cv2.drawKeypoints(img1, keypoints1, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
img2_with_oriented_kp = cv2.drawKeypoints(img2, keypoints2, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)


# Show side-by-side
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.title("Image 1 - Keypoints with Orientation")
plt.imshow(img1_with_oriented_kp, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Image 2 - Keypoints with Orientation")
plt.imshow(img2_with_oriented_kp, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()



# Use ORB's compute() to compute Rotated BRIEF descriptors
descriptors1 = orb.compute(img1, keypoints1)[1]
descriptors2 = orb.compute(img2, keypoints2)[1]

img1_with_desc = cv2.drawKeypoints(img1, keypoints1, None, color=(0, 255, 0), flags=0)
img2_with_desc = cv2.drawKeypoints(img2, keypoints2, None, color=(0, 255, 0), flags=0)


# Match descriptors using Brute-Force matcher with Hamming distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance (lower distance = better match)
matches = sorted(matches, key=lambda x: x.distance)

# Draw top 50 matches
matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=2)

# Show the matches
plt.figure(figsize=(16, 8))
plt.title("Top 50 Feature Matches (ORB + BFMatcher)")
plt.imshow(matched_img)
plt.axis('off')
plt.show()


#After matching the keypoints,we want to understand how good those matches were.
#That's where this histogram comes in!
distances = [m.distance for m in matches]
plt.hist(distances, bins=50)
plt.title("ORB Match Distances")
plt.xlabel("Hamming Distance")
plt.ylabel("Frequency")
plt.show()


#Another plot to decide a reasonable threshold for "good matches"


thresholds = list(range(20, 100, 5))
good_match_counts = [len([m for m in matches if m.distance < t]) for t in thresholds]

plt.figure(figsize=(10, 5))
plt.plot(thresholds, good_match_counts, marker='o')
plt.title('Number of Good Matches vs Distance Threshold')
plt.xlabel('Distance Threshold')
plt.ylabel('Number of Good Matches')
plt.grid(True)
plt.show()


# filter "good" matches (filtered by distance)
good_matches = [m for m in matches if m.distance < 65]  

# Print summary
print(f"Total Matches: {len(matches)}")
print(f"Good Matches (<65 distance): {len(good_matches)}")

match_quality = len(good_matches) / len(matches)
print(f"Good Match Ratio: {match_quality:.2f}")

#Around 61% of the keypoint matches are "good" â€” i.e., reliable.


#making a dummy file valid for submission

sample_submission = pd.read_csv('/kaggle/input/image-matching-challenge-2025/sample_submission.csv')
#sample_submission.head()
#checking column named to have in dummy submission

# Create dummy values
sample_submission["rotation_matrix"] = "1;0;0;0;1;0;0;0;1"  # Identity matrix as a placeholder
sample_submission["translation_vector"] = "0;0;0"  # Zero translation

# Save the dummy submission file
sample_submission.to_csv("submission.csv", index=False)

print("Dummy submission file created successfully!")

