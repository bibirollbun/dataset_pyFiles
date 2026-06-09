# --- 1. Imports ---
import os
import zipfile
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 # For reading images
from pathlib import Path
from tqdm.notebook import tqdm # For progress bar
import csv # For CSV quoting option

# Install necessary libraries if not present
!pip install -q matplotlib pandas pillow tqdm opencv-python-headless

print("Libraries imported/checked.")


# --- 2. Configuration & Paths ---

# Define specific paths within the dataset
BASE_DATA_DIR = '/kaggle/input/tom-jerry-object-detection/Tom_and_Jerry_Kaggle_dataset/Tom_and_Jerry_Kaggle_dataset'
TRAIN_IMG_DIR = os.path.join(BASE_DATA_DIR, 'train/images')
TEST_IMG_DIR = os.path.join(BASE_DATA_DIR, 'test/images') # Path to TEST images
OUTPUT_DIR = './' # Save submission in the current Colab directory


# --- 4. Plot Specific Example Images with Dummy Boxes ---
print("\n--- Plotting Specific Training Examples (with DUMMY boxes) ---")

# Define the images and their *dummy* annotations
example_images = [
    {"filename": "0bd699d9-JerryJerryQuiteContrary19662100.jpg", "boxes": [("Tom", [0, 0, 100, 100])]},
    {"filename": "4ab5acb4-TheKarateGuard20052700.jpg", "boxes": [("Jerry", [0, 0, 100, 100])]},
    {"filename": "5c4966f0-JerryJerryQuiteContrary19663360.jpg", "boxes": [("Tom", [0, 0, 100, 100]), ("Jerry", [450, 300, 600, 450])]}
]
IMG_WIDTH_DEFAULT, IMG_HEIGHT_DEFAULT = 640, 480 # For context if needed

def plot_manual_example(img_dir, example_info):
    """Plots a single image with manually defined bounding boxes."""
    filename = example_info["filename"]
    manual_boxes = example_info["boxes"]
    img_path = os.path.join(img_dir, filename)
    if not os.path.exists(img_path):
        print(f"WARN: Example image not found: {img_path}")
        return

    try:
        I = cv2.imread(img_path)
        if I is None: print(f"WARN: Could not read image {filename}"); return
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB) # Convert for matplotlib
        plt.figure(figsize=(8, 6)); plt.imshow(I); plt.axis('off'); ax = plt.gca()
        print(f"Plotting: {filename}")
        for label_name, box in manual_boxes:
             xmin, ymin, xmax, ymax = box; w = xmax - xmin; h = ymax - ymin
             if w > 0 and h > 0:
                 rect = patches.Rectangle((xmin, ymin), w, h, lw=2, edgecolor='cyan', facecolor='none'); ax.add_patch(rect)
                 plt.text(xmin, ymin - 3, label_name, color='cyan', fontsize=9, bbox=dict(facecolor='black', alpha=0.6, pad=1))
        plt.title(f"Example: {filename} (Boxes are illustrative only)"); plt.show()
    except Exception as e:
        print(f"ERROR plotting {filename}: {e}")

# Plot the defined examples
for example in example_images:
    # Ensure the image directory exists before trying to plot
    if os.path.isdir(TRAIN_IMG_DIR):
        plot_manual_example(TRAIN_IMG_DIR, example)
    else:
        print(f"ERROR: Cannot plot examples, TRAIN_IMG_DIR not found: {TRAIN_IMG_DIR}")
        break # Stop trying to plot if dir is missing


# --- 5. Generate Random Submission File (PredictionString Format) ---
print("\n--- Generating Random Submission File (PredictionString Format) ---")

if not os.path.exists(TEST_IMG_DIR):
    print(f"ERROR: Test image directory not found: {TEST_IMG_DIR}")
    print("Cannot generate submission file.")
else:
    test_image_files = [f for f in os.listdir(TEST_IMG_DIR)
                        if os.path.isfile(os.path.join(TEST_IMG_DIR, f))
                        and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(test_image_files)} images in the test directory: {TEST_IMG_DIR}")

    submission_data = []
    submission_header = ['ImageId', 'PredictionString'] # Kaggle format
    submission_data.append(submission_header)
    IMG_WIDTH_DEFAULT, IMG_HEIGHT_DEFAULT = 640, 480 # Image dimensions

    print("Generating random predictions...")
    for filename in tqdm(test_image_files, desc="Processing Test Images"):
        image_id_stem = Path(filename).stem # Use stem (filename without ext) as ImageId
        num_boxes = random.randint(0, 3) # Generate 0 to 3 boxes per image

        prediction_parts = [] # Store parts for this image's string
        if num_boxes == 0:
            # --- WORKAROUND for empty predictions ---
            # Add a single dummy prediction with very low confidence if no boxes are generated
            label_name = random.choice(['Tom', 'Jerry'])
            conf = 0.0001 # Very low confidence
            xmin, ymin, xmax, ymax = 0, 0, 1, 1 # Tiny box at origin
            prediction_parts.append(f"{label_name} {conf:.4f} {xmin} {ymin} {xmax} {ymax}")
            # --- END WORKAROUND ---
        else:
            # --- Generate Random Boxes ---
            for _ in range(num_boxes):
                label_name = random.choice(['Tom', 'Jerry'])
                conf = round(random.uniform(0.1, 0.95), 4)
                # Generate plausible random coordinates
                box_w = random.randint(30, IMG_WIDTH_DEFAULT // 2)
                box_h = random.randint(30, IMG_HEIGHT_DEFAULT // 2)
                xmin = random.randint(0, IMG_WIDTH_DEFAULT - box_w - 1)
                ymin = random.randint(0, IMG_HEIGHT_DEFAULT - box_h - 1)
                xmax = xmin + box_w
                ymax = ymin + box_h
                # Format: Label Conf Xmin Ymin Xmax Ymax
                prediction_parts.append(f"{label_name} {conf} {xmin} {ymin} {xmax} {ymax}")

        # Join all parts for this image with spaces
        prediction_string = " ".join(prediction_parts)
        # Append the actual ImageId stem and the generated prediction string
        submission_data.append([image_id_stem, prediction_string])

    # Create DataFrame and save to CSV
    submission_df = pd.DataFrame(submission_data[1:], columns=submission_header)
    # Ensure correct column order just in case
    submission_df = submission_df[['ImageId', 'PredictionString']]
    submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')

    try:
        # Save with quoting to be extra safe, although likely not needed if dummy box used
        submission_df.to_csv(
            submission_path,
            index=False,
            quoting=csv.QUOTE_MINIMAL # Quote fields only if they contain the delimiter or quote char
            )
        print(f"\nSample submission file saved to: {submission_path}")
        print(f"Total rows generated (including header): {len(submission_data)}")
        print("\nFirst 5 rows of submission file:")
        print(submission_df.head().to_string())
        print("\nLast 5 rows of submission file:")
        print(submission_df.tail().to_string())
    except Exception as e:
        print(f"ERROR saving submission file: {e}")

print("\n--- Starter Notebook Finished ---")

