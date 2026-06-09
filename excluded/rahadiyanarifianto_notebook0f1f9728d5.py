import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import csv
import cv2


def visualize_images_in_folder(folder_path, num_images=6, rows=2, cols=3):    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.ravel()  
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".png")]
    num_images = min(num_images, len(image_files))    
    for i in range(num_images):
        img_file = image_files[i]
        img_path = os.path.join(folder_path, img_file)       
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_file}")
            continue       
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       
        axes[i].imshow(img)
        axes[i].set_title(img_file)
        axes[i].axis('off')  
    for ax in axes[num_images:]:
        ax.axis('off')    
    plt.tight_layout()
    plt.show()


folder_path = "/kaggle/input/image-matching-challenge-2024/train/church/images/"
visualize_images_in_folder(folder_path, num_images=3, rows=1, cols=3)


#Read images in grayscale
image1 = cv2.imread('/kaggle/input/image-matching-challenge-2024/train/church/images/00009.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('/kaggle/input/image-matching-challenge-2024/test/church/images/00029.png', cv2.IMREAD_GRAYSCALE)


#Grayscale with 3 channels
image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)


sift = cv2.SIFT_create()


keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)


matcher = cv2.BFMatcher()
matches = matcher.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)
matched_image_sift = cv2.drawMatches(image1_rgb, keypoints1, image2_rgb, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


matched_keypoints1 = np.float32([keypoints1[m.queryIdx].pt for m in matches[:25]])
matched_keypoints2 = np.float32([keypoints2[m.trainIdx].pt for m in matches[:25]])


for pt1, pt2 in zip(matched_keypoints1, matched_keypoints2):
    x1, y1 = pt1
    x2, y2 = pt2
plt.figure(figsize=(12, 8))
plt.imshow(matched_image_sift)
plt.title('SIFT')
plt.axis('off')
plt.savefig("matched_image_sift.png", dpi=240, bbox_inches="tight", pad_inches=0.1)
plt.show()


akaze = cv2.AKAZE_create()


keypoints1, descriptors1 = akaze.detectAndCompute(image1, None)
keypoints2, descriptors2 = akaze.detectAndCompute(image2, None)


matcher = cv2.BFMatcher()
matches = matcher.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)
matched_image_akaze = cv2.drawMatches(image1_rgb, keypoints1, image2_rgb, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


matched_keypoints1 = np.float32([keypoints1[m.queryIdx].pt for m in matches[:25]])
matched_keypoints2 = np.float32([keypoints2[m.trainIdx].pt for m in matches[:25]])


for pt1, pt2 in zip(matched_keypoints1, matched_keypoints2):
    x1, y1 = pt1
    x2, y2 = pt2
plt.figure(figsize=(12, 8))
plt.imshow(matched_image_akaze)
plt.title('AKAZE')
plt.axis('off')
plt.savefig("matched_image_akaze.png", dpi=240, bbox_inches="tight", pad_inches=0.1)
plt.show()


clahe = cv2.createCLAHE(clipLimit=5)

c_image1 = clahe.apply(image1)
c_image2 = clahe.apply(image2)

#CLAHE images with 3 channels
#c_image1_3c = cv2.cvtColor(c_image1, cv2.COLOR_GRAY2BGR)
#c_image2_3c = cv2.cvtColor(c_image2, cv2.COLOR_GRAY2BGR)


plt.subplot(2,2,1)
plt.imshow(image1_rgb)
plt.axis("off")
plt.title("Image 1 RGB")

plt.subplot(2,2,2)
plt.imshow(c_image1)
plt.axis("off")
plt.title("Image 1 CLAHE")

plt.subplot(2,2,3)
plt.imshow(image2_rgb)
plt.axis("off")
plt.title("Image 1 RGB")

plt.subplot(2,2,4)
plt.imshow(c_image2)
plt.axis("off")
plt.title("Image 1 CLAHE")


keypoints1_sc, descriptors1_sc = sift.detectAndCompute(c_image1, None)
keypoints2_sc, descriptors2_sc = sift.detectAndCompute(c_image2, None)


matcher = cv2.BFMatcher()
matches = matcher.match(descriptors1_sc, descriptors2_sc)
matches = sorted(matches, key=lambda x: x.distance)
matched_image_sift_clahe = cv2.drawMatches(image1_rgb, keypoints1_sc, image2_rgb, keypoints2_sc, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


matched_keypoints1_sc = np.float32([keypoints1_sc[m.queryIdx].pt for m in matches[:25]])
matched_keypoints2_sc = np.float32([keypoints2_sc[m.trainIdx].pt for m in matches[:25]])


for pt1, pt2 in zip(matched_keypoints1_sc, matched_keypoints2_sc):
    x1, y1 = pt1
    x2, y2 = pt2
plt.figure(figsize=(12, 8))
plt.imshow(matched_image_sift_clahe)
plt.title('SIFT + CLAHE')
plt.axis('off')
plt.savefig('matched_image_sift_clahe.png', dpi=240, bbox_inches='tight', pad_inches=0.1)
plt.show()


keypoints1_ac, descriptors1_ac = sift.detectAndCompute(c_image1, None)
keypoints2_ac, descriptors2_ac = sift.detectAndCompute(c_image2, None)


matcher = cv2.BFMatcher()
matches = matcher.match(descriptors1_ac, descriptors2_ac)
matches = sorted(matches, key=lambda x: x.distance)
matched_image_akaze_clahe = cv2.drawMatches(image1_rgb, keypoints1_ac, image2_rgb, keypoints2_ac, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


matched_keypoints1_ac = np.float32([keypoints1_ac[m.queryIdx].pt for m in matches[:25]])
matched_keypoints2_ac = np.float32([keypoints2_ac[m.trainIdx].pt for m in matches[:25]])


for pt1, pt2 in zip(matched_keypoints1_ac, matched_keypoints2_ac):
    x1, y1 = pt1
    x2, y2 = pt2
plt.figure(figsize=(12, 8))
plt.imshow(matched_image_akaze_clahe)
plt.title('AKAZE + CLAHE')
plt.axis('off')
plt.savefig('matched_image_akaze_clahe.png', dpi=240, bbox_inches='tight', pad_inches=0.1)
plt.show()

