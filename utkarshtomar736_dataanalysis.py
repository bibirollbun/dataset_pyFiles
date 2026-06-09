import os
from pathlib import Path
import random
from PIL import Image
import matplotlib.pyplot as plt


fpath = Path("/kaggle/input/image-matching-challenge-2025")


TEST_DIR = fpath / "test"
TRAIN_DIR = fpath / "train"
SAM_SUB = pd.read_csv(fpath / "sample_submission.csv")
LABELS_TRAIN = pd.read_csv(fpath / "train_labels.csv")
THRESH_TRAIN = pd.read_csv(fpath / "train_thresholds.csv")


LABELS_TRAIN.head()


LABELS_TRAIN.info()


THRESH_TRAIN.head()


THRESH_TRAIN.info()


os.listdir(fpath)


os.listdir(TRAIN_DIR)


train_subdirs = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(TRAIN_DIR / d)]

def display_close_triplet_compact():
    random_subdir = random.choice(train_subdirs)
    subdir_path = TRAIN_DIR / random_subdir
    image_files = sorted([f for f in os.listdir(subdir_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if len(image_files) >= 3:
        start_idx = random.randint(0, len(image_files) - 3)
        selected_files = image_files[start_idx : start_idx + 3]

        plt.figure(figsize=(15, 5))
        for i, img_file in enumerate(selected_files):
            img_path = subdir_path / img_file
            try:
                img = Image.open(img_path)
                plt.subplot(1, 3, i + 1)
                plt.imshow(img)
                plt.title(f"{random_subdir}\n{img_file}")
                plt.axis('off')
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                return False
        plt.tight_layout()
        plt.show()
        return True
    else:
        print(f"Not enough images in {random_subdir} for a triplet.")
        return False

# Display one such triplet
print("Close triplet:")
display_close_triplet()




