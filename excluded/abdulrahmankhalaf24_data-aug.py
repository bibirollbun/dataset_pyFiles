import os
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


!unzip /kaggle/input/diabetic-retinopathy-detection/trainLabels.csv.zip -d /kaggle/working/
!unzip /kaggle/input/diabetic-retinopathy-detection/sampleSubmission.csv.zip -d /kaggle/working/


train_lbl = pd.read_csv('/kaggle/working/trainLabels.csv')
train_img = glob('/kaggle/input/diabetic-retinopathy-train-unzipped/train/*.jpeg')
train_names = [os.path.basename(path).replace('.jpeg', '') for path in train_img]
train_df = pd.DataFrame({'image': train_names, 'image_path': train_img})
train_df = pd.merge(train_lbl, train_df, on='image')
train_df.head()


samplesub = pd.read_csv('/kaggle/working/sampleSubmission.csv')
test_img = glob('/kaggle/input/diabetic-retinopathy-test-unzipped/test/*.jpeg')
test_names = [os.path.basename(path).replace('.jpeg', '') for path in test_img]
test_df = pd.DataFrame({'image': test_names, 'image_path': test_img})
test_df = pd.merge(samplesub, test_df, on='image')
test_df.head()



fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))
ax = axes.flatten()

for i in range(10):
    row = train_df.sample(10).iloc[i]
    img = Image.open(row['image_path'])
    ax[i].imshow(img)
    ax[i].set_title(f"Label: {row['level']}")
    ax[i].axis('off')

plt.tight_layout()
plt.show()



# Ø§Ù„Ø¥Ø¹Ø¯Ø§Ø¯
target_class = 1                # Ø§Ù„ÙƒÙ„Ø§Ø³ Ø§Ù„Ù„ÙŠ Ø¹Ø§ÙŠØ² ØªØ²ÙˆØ¯ ØµÙˆØ±Ù‡
desired_total = 5000           # Ø§Ù„Ø¹Ø¯Ø¯ Ø§Ù„Ù†Ù‡Ø§Ø¦ÙŠ Ø§Ù„Ù…Ø·Ù„ÙˆØ¨
image_size = (224, 224)        # Ø­Ø¬Ù… Ø§Ù„ØµÙˆØ±Ø© Ø¨Ø¹Ø¯ Ø§Ù„ØªØ­Ø¬ÙŠÙ…
output_suffix = 'aug'          # Ù„Ø§Ø­Ù‚Ø© Ø§Ø³Ù… Ø§Ù„ØµÙˆØ±Ø© Ø§Ù„Ø¬Ø¯ÙŠØ¯Ø©
output_dir = '/kaggle/working/augmented_images'
os.makedirs(output_dir, exist_ok=True)

# Ø§Ù„ØªØ­ÙˆÙŠÙ„Ø§Øª Ø§Ù„Ù‡Ù†Ø¯Ø³ÙŠØ©
geo_transforms = [
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomRotation(degrees=15),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
]
resize = transforms.Resize(image_size)

# ØªØ­Ù…ÙŠÙ„ Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª
train_df = pd.read_csv('/kaggle/working/trainLabels.csv')
image_dir = '/kaggle/input/diabetic-retinopathy-train-unzipped/train/'
train_df['image_path'] = train_df['image'].apply(lambda x: os.path.join(image_dir, f"{x}.jpeg"))
target_df = train_df[train_df['level'] == target_class]

# Ø­Ø³Ø§Ø¨ Ø¹Ø¯Ø¯ Ø§Ù„ØµÙˆØ± Ø§Ù„Ù…Ø·Ù„ÙˆØ¨Ø©
current_count = len(target_df)
needed_augmented = desired_total - current_count
if needed_augmented <= 0:
    print("âœ… Ø¹Ø¯Ø¯ Ø§Ù„ØµÙˆØ± ÙƒØ§Ù�ÙŠ Ø¨Ø§Ù„Ù�Ø¹Ù„.")
else:
    augmentations_per_image = max(1, needed_augmented // current_count)
    extra = needed_augmented % current_count  # Ù†ÙˆØ²Ø¹ Ø§Ù„Ø²ÙŠØ§Ø¯Ø© Ø¹Ù„Ù‰ Ø¨Ø¹Ø¶ Ø§Ù„ØµÙˆØ±

    print(f"ğŸ§® Ø¹Ø¯Ø¯ Ø§Ù„ØµÙˆØ± Ø§Ù„Ø£ØµÙ„ÙŠØ©: {current_count}")
    print(f"ğŸ› ï¸� Ø¹Ø¯Ø¯ Ø§Ù„Ù†Ø³Ø® Ù„ÙƒÙ„ ØµÙˆØ±Ø©: {augmentations_per_image}")
    print(f"â�• Ø¹Ø¯Ø¯ Ø§Ù„ØµÙˆØ± Ø§Ù„Ø¥Ø¶Ø§Ù�ÙŠØ© Ù„Ù„ØªØ¹ÙˆÙŠØ¶: {extra}")

    # ØªÙ†Ù�ÙŠØ° Ø§Ù„ØªÙˆÙ„ÙŠØ¯
    for idx, (_, row) in enumerate(tqdm(target_df.iterrows(), total=current_count)):
        img_path = row['image_path']
        image_name = os.path.splitext(os.path.basename(img_path))[0]

        try:
            image = Image.open(img_path).convert('RGB')
            image = resize(image)

            # ØªÙˆÙ„ÙŠØ¯ Ù†Ø³Ø® Ù…Ø¹Ø²Ø²Ø©
            for i in range(augmentations_per_image):
                aug = geo_transforms[(i + idx) % len(geo_transforms)]
                aug_image = aug(image)
                aug_image_name = f"{image_name}_{output_suffix}{i}.jpeg"
                aug_image.save(os.path.join(output_dir, aug_image_name))

            # Ù„Ùˆ Ù„Ø³Ù‡ Ù�Ø§Ø¶Ù„ extra ØµÙˆØ± Ù†ÙˆÙ„Ø¯Ù‡Ø§ Ù„Ø¨Ø¹Ø¶ Ø§Ù„ØµÙˆØ±
            if idx < extra:
                aug = geo_transforms[(augmentations_per_image + idx) % len(geo_transforms)]
                aug_image = aug(image)
                aug_image_name = f"{image_name}_{output_suffix}_extra.jpeg"
                aug_image.save(os.path.join(output_dir, aug_image_name))

        except Exception as e:
            print(f"â�Œ Ø®Ø·Ø£ Ù�ÙŠ Ø§Ù„ØµÙˆØ±Ø© {img_path}: {e}")




# Ø§Ù„Ø¥Ø¹Ø¯Ø§Ø¯
target_class = 3                # Ø§Ù„ÙƒÙ„Ø§Ø³ Ø§Ù„Ù„ÙŠ Ø¹Ø§ÙŠØ² ØªØ²ÙˆØ¯ ØµÙˆØ±Ù‡
desired_total = 5000           # Ø§Ù„Ø¹Ø¯Ø¯ Ø§Ù„Ù†Ù‡Ø§Ø¦ÙŠ Ø§Ù„Ù…Ø·Ù„ÙˆØ¨
image_size = (224, 224)        # Ø­Ø¬Ù… Ø§Ù„ØµÙˆØ±Ø© Ø¨Ø¹Ø¯ Ø§Ù„ØªØ­Ø¬ÙŠÙ…
output_suffix = 'aug'          # Ù„Ø§Ø­Ù‚Ø© Ø§Ø³Ù… Ø§Ù„ØµÙˆØ±Ø© Ø§Ù„Ø¬Ø¯ÙŠØ¯Ø©
output_dir = '/kaggle/working/augmented_images'
os.makedirs(output_dir, exist_ok=True)

# Ø§Ù„ØªØ­ÙˆÙŠÙ„Ø§Øª Ø§Ù„Ù‡Ù†Ø¯Ø³ÙŠØ©
geo_transforms = [
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomRotation(degrees=15),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
]
resize = transforms.Resize(image_size)

# ØªØ­Ù…ÙŠÙ„ Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª
train_df = pd.read_csv('/kaggle/working/trainLabels.csv')
image_dir = '/kaggle/input/diabetic-retinopathy-train-unzipped/train/'
train_df['image_path'] = train_df['image'].apply(lambda x: os.path.join(image_dir, f"{x}.jpeg"))
target_df = train_df[train_df['level'] == target_class]

# Ø­Ø³Ø§Ø¨ Ø¹Ø¯Ø¯ Ø§Ù„ØµÙˆØ± Ø§Ù„Ù…Ø·Ù„ÙˆØ¨Ø©
current_count = len(target_df)
needed_augmented = desired_total - current_count
if needed_augmented <= 0:
    print("âœ… Ø¹Ø¯Ø¯ Ø§Ù„ØµÙˆØ± ÙƒØ§Ù�ÙŠ Ø¨Ø§Ù„Ù�Ø¹Ù„.")
else:
    augmentations_per_image = max(1, needed_augmented // current_count)
    extra = needed_augmented % current_count  # Ù†ÙˆØ²Ø¹ Ø§Ù„Ø²ÙŠØ§Ø¯Ø© Ø¹Ù„Ù‰ Ø¨Ø¹Ø¶ Ø§Ù„ØµÙˆØ±

    print(f"ğŸ§® Ø¹Ø¯Ø¯ Ø§Ù„ØµÙˆØ± Ø§Ù„Ø£ØµÙ„ÙŠØ©: {current_count}")
    print(f"ğŸ› ï¸� Ø¹Ø¯Ø¯ Ø§Ù„Ù†Ø³Ø® Ù„ÙƒÙ„ ØµÙˆØ±Ø©: {augmentations_per_image}")
    print(f"â�• Ø¹Ø¯Ø¯ Ø§Ù„ØµÙˆØ± Ø§Ù„Ø¥Ø¶Ø§Ù�ÙŠØ© Ù„Ù„ØªØ¹ÙˆÙŠØ¶: {extra}")

    # ØªÙ†Ù�ÙŠØ° Ø§Ù„ØªÙˆÙ„ÙŠØ¯
    for idx, (_, row) in enumerate(tqdm(target_df.iterrows(), total=current_count)):
        img_path = row['image_path']
        image_name = os.path.splitext(os.path.basename(img_path))[0]

        try:
            image = Image.open(img_path).convert('RGB')
            image = resize(image)

            # ØªÙˆÙ„ÙŠØ¯ Ù†Ø³Ø® Ù…Ø¹Ø²Ø²Ø©
            for i in range(augmentations_per_image):
                aug = geo_transforms[(i + idx) % len(geo_transforms)]
                aug_image = aug(image)
                aug_image_name = f"{image_name}_{output_suffix}{i}.jpeg"
                aug_image.save(os.path.join(output_dir, aug_image_name))

            # Ù„Ùˆ Ù„Ø³Ù‡ Ù�Ø§Ø¶Ù„ extra ØµÙˆØ± Ù†ÙˆÙ„Ø¯Ù‡Ø§ Ù„Ø¨Ø¹Ø¶ Ø§Ù„ØµÙˆØ±
            if idx < extra:
                aug = geo_transforms[(augmentations_per_image + idx) % len(geo_transforms)]
                aug_image = aug(image)
                aug_image_name = f"{image_name}_{output_suffix}_extra.jpeg"
                aug_image.save(os.path.join(output_dir, aug_image_name))

        except Exception as e:
            print(f"â�Œ Ø®Ø·Ø£ Ù�ÙŠ Ø§Ù„ØµÙˆØ±Ø© {img_path}: {e}")




# Ø§Ù„Ø¥Ø¹Ø¯Ø§Ø¯
target_class = 4                # Ø§Ù„ÙƒÙ„Ø§Ø³ Ø§Ù„Ù„ÙŠ Ø¹Ø§ÙŠØ² ØªØ²ÙˆØ¯ ØµÙˆØ±Ù‡
desired_total = 5000           # Ø§Ù„Ø¹Ø¯Ø¯ Ø§Ù„Ù†Ù‡Ø§Ø¦ÙŠ Ø§Ù„Ù…Ø·Ù„ÙˆØ¨
image_size = (224, 224)        # Ø­Ø¬Ù… Ø§Ù„ØµÙˆØ±Ø© Ø¨Ø¹Ø¯ Ø§Ù„ØªØ­Ø¬ÙŠÙ…
output_suffix = 'aug'          # Ù„Ø§Ø­Ù‚Ø© Ø§Ø³Ù… Ø§Ù„ØµÙˆØ±Ø© Ø§Ù„Ø¬Ø¯ÙŠØ¯Ø©
output_dir = '/kaggle/working/augmented_images'
os.makedirs(output_dir, exist_ok=True)

# Ø§Ù„ØªØ­ÙˆÙŠÙ„Ø§Øª Ø§Ù„Ù‡Ù†Ø¯Ø³ÙŠØ©
geo_transforms = [
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomRotation(degrees=15),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
]
resize = transforms.Resize(image_size)

# ØªØ­Ù…ÙŠÙ„ Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª
train_df = pd.read_csv('/kaggle/working/trainLabels.csv')
image_dir = '/kaggle/input/diabetic-retinopathy-train-unzipped/train/'
train_df['image_path'] = train_df['image'].apply(lambda x: os.path.join(image_dir, f"{x}.jpeg"))
target_df = train_df[train_df['level'] == target_class]

# Ø­Ø³Ø§Ø¨ Ø¹Ø¯Ø¯ Ø§Ù„ØµÙˆØ± Ø§Ù„Ù…Ø·Ù„ÙˆØ¨Ø©
current_count = len(target_df)
needed_augmented = desired_total - current_count
if needed_augmented <= 0:
    print("âœ… Ø¹Ø¯Ø¯ Ø§Ù„ØµÙˆØ± ÙƒØ§Ù�ÙŠ Ø¨Ø§Ù„Ù�Ø¹Ù„.")
else:
    augmentations_per_image = max(1, needed_augmented // current_count)
    extra = needed_augmented % current_count  # Ù†ÙˆØ²Ø¹ Ø§Ù„Ø²ÙŠØ§Ø¯Ø© Ø¹Ù„Ù‰ Ø¨Ø¹Ø¶ Ø§Ù„ØµÙˆØ±

    print(f"ğŸ§® Ø¹Ø¯Ø¯ Ø§Ù„ØµÙˆØ± Ø§Ù„Ø£ØµÙ„ÙŠØ©: {current_count}")
    print(f"ğŸ› ï¸� Ø¹Ø¯Ø¯ Ø§Ù„Ù†Ø³Ø® Ù„ÙƒÙ„ ØµÙˆØ±Ø©: {augmentations_per_image}")
    print(f"â�• Ø¹Ø¯Ø¯ Ø§Ù„ØµÙˆØ± Ø§Ù„Ø¥Ø¶Ø§Ù�ÙŠØ© Ù„Ù„ØªØ¹ÙˆÙŠØ¶: {extra}")

    # ØªÙ†Ù�ÙŠØ° Ø§Ù„ØªÙˆÙ„ÙŠØ¯
    for idx, (_, row) in enumerate(tqdm(target_df.iterrows(), total=current_count)):
        img_path = row['image_path']
        image_name = os.path.splitext(os.path.basename(img_path))[0]

        try:
            image = Image.open(img_path).convert('RGB')
            image = resize(image)

            # ØªÙˆÙ„ÙŠØ¯ Ù†Ø³Ø® Ù…Ø¹Ø²Ø²Ø©
            for i in range(augmentations_per_image):
                aug = geo_transforms[(i + idx) % len(geo_transforms)]
                aug_image = aug(image)
                aug_image_name = f"{image_name}_{output_suffix}{i}.jpeg"
                aug_image.save(os.path.join(output_dir, aug_image_name))

            # Ù„Ùˆ Ù„Ø³Ù‡ Ù�Ø§Ø¶Ù„ extra ØµÙˆØ± Ù†ÙˆÙ„Ø¯Ù‡Ø§ Ù„Ø¨Ø¹Ø¶ Ø§Ù„ØµÙˆØ±
            if idx < extra:
                aug = geo_transforms[(augmentations_per_image + idx) % len(geo_transforms)]
                aug_image = aug(image)
                aug_image_name = f"{image_name}_{output_suffix}_extra.jpeg"
                aug_image.save(os.path.join(output_dir, aug_image_name))

        except Exception as e:
            print(f"â�Œ Ø®Ø·Ø£ Ù�ÙŠ Ø§Ù„ØµÙˆØ±Ø© {img_path}: {e}")





