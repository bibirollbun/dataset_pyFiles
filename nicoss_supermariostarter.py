from pathlib import Path

# Define the target directory
dataset_dir = Path("datasets/")

# Check if it exists
if dataset_dir.exists():
    print("âœ… 'datasets/' already exists. No action needed.")
else:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    print("ğŸ“� Created 'datasets/' directory.")



import os
import shutil
import random
from pathlib import Path

# --- Configuration ---
dataset_kaggle_dir = Path("/kaggle/input/dl-4-cv-competition/dataset_kaggle/dataset_kaggle")
test_set_dir = Path("/kaggle/input/dl-4-cv-competition/test_set_without_label")
output_dir = Path("datasets")

# --- Step 1: Create output folder structure ---
for split in ['train', 'val', 'test']:
    (output_dir / f"images/{split}").mkdir(parents=True, exist_ok=True)
    (output_dir / f"labels/{split}").mkdir(parents=True, exist_ok=True)

# --- Step 2: 80/20 split for train/val ---
image_files = list(dataset_kaggle_dir.rglob("*.jpg"))
random.shuffle(image_files)
split_idx = int(len(image_files) * 0.8)
train_images = image_files[:split_idx]
val_images = image_files[split_idx:]

def move_files(image_list, split):
    for img_path in image_list:
        label_path = dataset_kaggle_dir / "labels" / (img_path.stem + ".txt")
        shutil.copy(str(img_path), output_dir / f"images/{split}" / img_path.name)
        if label_path.exists():
            shutil.copy(str(label_path), output_dir / f"labels/{split}" / label_path.name)

move_files(train_images, "train")
move_files(val_images, "val")

# --- Step 3: Handle test set (images only, no labels) ---
test_image_dir = test_set_dir / "images"
test_images = list(test_image_dir.glob("*.jpg"))
for img_path in test_images:
    shutil.copy(str(img_path), output_dir / "images/test" / img_path.name)

# Optional: create empty test label files
for img_path in test_images:
    empty_label_path = output_dir / "labels/test" / (img_path.stem + ".txt")
    empty_label_path.touch()

# --- Step 4: Copy classes.txt and notes.json from kaggle dataset ---
def find_file_recursive(base_dir, filename):
    for root, _, files in os.walk(base_dir):
        if filename in files:
            return Path(root) / filename
    raise FileNotFoundError(f"{filename} not found in {base_dir}")

shutil.copy(find_file_recursive(dataset_kaggle_dir, "classes.txt"), output_dir / "classes.txt")
shutil.copy(find_file_recursive(dataset_kaggle_dir, "notes.json"), output_dir / "notes.json")

print("âœ… Dataset organized successfully.")


import os

val_path = os.path.abspath("datasets/images/val")
print(f"ğŸ“‚ Absolute path of val/: {val_path}")
print(f"ğŸ“¸ Contents of val/: {os.listdir(val_path)[:5]}")  # Display 5 files for verification


from pathlib import Path
import csv

test_images = Path("datasets/images/test")
labels_dir = Path("/kaggle/working/test_predictions/labels")
output_csv = "/kaggle/working/submission.csv"


# Fake example
example_image_name = "example_test.jpg"
example_label_path = labels_dir / (Path(example_image_name).stem + ".txt")
example_label_path.parent.mkdir(parents=True, exist_ok=True)
fake_prediction = "0 0.5 0.5 0.3 0.3\n"
example_label_path.write_text(fake_prediction)


# Create empty .txt files if they don't exist
for image_file in test_images.glob("*.jpg"):
    label_file = labels_dir / (image_file.stem + ".txt")
    if not label_file.exists():
        label_file.write_text("")

# Generate the CSV file
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Id", "PredictionString"])

    for image_file in sorted(test_images.glob("*.jpg")):
        label_file = labels_dir / (image_file.stem + ".txt")
        prediction_string = ""

        if label_file.exists():
            lines = label_file.read_text().strip().splitlines()
            if lines:
                prediction_string = " ".join(
                    " ".join(line.strip().split()[:5]) for line in lines
                )

        # Always write a row, even if it's empty
        writer.writerow([image_file.name, prediction_string if prediction_string.strip() else " "])

print("âœ… Done!")

