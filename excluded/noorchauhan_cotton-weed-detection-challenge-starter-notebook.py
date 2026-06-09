# Import required packages
import torch
import tlc
from pathlib import Path
import pandas as pd
from IPython.display import display

# Check environment
print("Environment Check:")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"3LC version: {tlc.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )
else:
    print("!!! No GPU detected - training will be slower on CPU")

print("\n All systems ready! Let's begin.")


# Set up file paths
WORK_DIR = Path(".")  # Current directory
DATASET_YAML = WORK_DIR / "dataset.yaml"

# Verify paths exist
print("Verifying dataset structure...")
print("=" * 50)

if not DATASET_YAML.exists():
    print(f"Could not find {DATASET_YAML}")
    print(f"Current directory: {Path.cwd()}")
    print("Please make sure dataset.yaml is in the current directory")
    raise FileNotFoundError(f"Dataset config not found: {DATASET_YAML}")

print(f"âœ… Dataset config: {DATASET_YAML}")
print(f"âœ… Working directory: {WORK_DIR.resolve()}")

# Display dataset configuration
print("\n Dataset Configuration:")
print("-" * 50)
with open(DATASET_YAML, "r") as f:
    config_content = f.read()
    print(config_content)

# Count dataset files
train_images = list((WORK_DIR / "train" / "images").glob("*.jpg"))
train_labels = list((WORK_DIR / "train" / "labels").glob("*.txt"))
val_images = list((WORK_DIR / "val" / "images").glob("*.jpg"))
val_labels = list((WORK_DIR / "val" / "labels").glob("*.txt"))
test_images = list((WORK_DIR / "test" / "images").glob("*.jpg"))

print("\n Dataset Statistics:")
print("-" * 50)
print(f"âœ… Training:   {len(train_images)} images, {len(train_labels)} labels")
print(f"âœ… Validation: {len(val_images)} images, {len(val_labels)} labels")
print(f"âœ… Test: {len(test_images)} images")


# Example images for each weed class
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

print("Finding example images for each weed class...")
print("=" * 70)

# Set up paths
TRAIN_IMAGES = WORK_DIR / "train" / "images"
TRAIN_LABELS = WORK_DIR / "train" / "labels"
CLASS_NAMES = ["Carpetweed", "Morning Glory", "Palmer Amaranth"]

# Find images containing each class
class_examples = defaultdict(list)

for label_file in TRAIN_LABELS.glob("*.txt"):
    if label_file.stat().st_size > 0:
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    image_file = TRAIN_IMAGES / f"{label_file.stem}.jpg"
                    if image_file.exists():
                        class_examples[class_id].append(image_file)

# Select one clear example per class (first occurrence)
examples_to_show = {}
for class_id in range(len(CLASS_NAMES)):
    if class_examples[class_id]:
        examples_to_show[class_id] = class_examples[class_id][0]
        print(
            f"âœ“ Found example for {CLASS_NAMES[class_id]}: {examples_to_show[class_id].name}"
        )
    else:
        print(f"!!!  No examples found for {CLASS_NAMES[class_id]}")

# Display the examples
if examples_to_show:
    print("\n" + "=" * 70)
    print("Displaying example images with bounding boxes...")
    print("=" * 70)

    fig, axes = plt.subplots(1, len(examples_to_show), figsize=(15, 5))
    if len(examples_to_show) == 1:
        axes = [axes]

    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # BGR colors for OpenCV

    for idx, (class_id, image_path) in enumerate(sorted(examples_to_show.items())):
        # Read image
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Read corresponding label
        label_file = TRAIN_LABELS / f"{image_path.stem}.txt"
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    if cls == class_id:  # Only draw boxes for the target class
                        # Convert YOLO format to pixel coordinates
                        x_center, y_center, box_w, box_h = map(float, parts[1:5])
                        x1 = int((x_center - box_w / 2) * w)
                        y1 = int((y_center - box_h / 2) * h)
                        x2 = int((x_center + box_w / 2) * w)
                        y2 = int((y_center + box_h / 2) * h)

                        # Draw bounding box
                        color = colors[class_id]
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                        # Add class label
                        label_text = f"{CLASS_NAMES[class_id]}"
                        cv2.putText(
                            img,
                            label_text,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            color,
                            2,
                        )

        # Display
        axes[idx].imshow(img)
        axes[idx].set_title(
            f"Class {class_id}: {CLASS_NAMES[class_id]}", fontsize=12, fontweight="bold"
        )
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()

    print("\nâœ… Example images displayed!")
    print("\n Pro Tip: Keep these visual characteristics in mind when:")
    print("   â€¢ Analyzing model predictions in the 3LC Dashboard")
    print("   â€¢ Identifying mislabeled or missing annotations")
    print("   â€¢ Understanding class confusion patterns")

else:
    print("\nâš ï¸�  Could not find example images for visualization")


# ============================================================================
# Create 3LC Tables from YOLO Format Dataset
# âš ï¸� RUN THIS CELL ONLY ONCE (Initial Setup)
# ============================================================================
# This cell registers your dataset with 3LC for version control and analysis.
#
# âš ï¸� IMPORTANT FOR RETRAINING:
#    - First time: Run this cell to create tables
#    - Retraining: SKIP this cell and go directly to the next cell
#                  (it loads tables independently without needing this)

# Import required packages
import tlc
from pathlib import Path

# Define constants for 3LC registration
PROJECT_NAME = "kaggle_cotton_weed_detection"
DATASET_NAME = "cotton_weed_det3"
WORK_DIR = Path(".")
DATASET_YAML = WORK_DIR / "dataset.yaml"

print("=" * 70)
print("DATA REGISTRATION")
print("=" * 70)

# ============================================================================
# IDEMPOTENCY CHECK - Safe to run multiple times
# ============================================================================
try:
    # Check if tables already exist
    existing_train = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name=f"{DATASET_NAME}-train1",
    )
    existing_val = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name=f"{DATASET_NAME}-val1",
    )

    print("\nâš ï¸�  Tables already exist!")
    print(f" Training: {len(existing_train)} samples")
    print(f" Validation: {len(existing_val)} samples")
    print("\nâœ… Using existing tables (no duplicates created)")
    print(" This cell is safe to run multiple times!")

    # Set variables for compatibility
    train_table = existing_train
    val_table = existing_val

except Exception:
    # Tables don't exist, create them
    print("\nâœ… No existing tables - creating new ones...")

    # Create training table
    print("\n Creating training table...")
    train_table = tlc.Table.from_yolo(
        dataset_yaml_file=str(DATASET_YAML),
        split="train",
        task="detect",
        dataset_name=DATASET_NAME,
        project_name=PROJECT_NAME,
        table_name=f"{DATASET_NAME}-train1",
    )

    # Create validation table
    print(" Creating validation table...")
    val_table = tlc.Table.from_yolo(
        dataset_yaml_file=str(DATASET_YAML),
        split="val",
        task="detect",
        dataset_name=DATASET_NAME,
        project_name=PROJECT_NAME,
        table_name=f"{DATASET_NAME}-val1",
    )

# Display registration results
print("\nâœ… Tables created successfully!")
print("=" * 70)
print("\n Training Table:")
print(f"   Samples: {len(train_table)}")
print(f"   URL: {train_table.url}")

print("\n Validation Table:")
print(f"   Samples: {len(val_table)}")
print(f"   URL: {val_table.url}")

print("\n" + "=" * 70)
print("âœ… Phase 1 Complete: Dataset Registered with 3LC!")
print("=" * 70)

print("\n Next Steps:")
print("  (Optional) Explore tables in Dashboard: https://dashboard.3lc.ai")


# ============================================================================
# Load Tables + Configure Training
# ============================================================================
# This cell:
#   1. Loads your registered tables (includes any Dashboard edits)
#   2. Sets up training configuration (RUN_NAME, EPOCHS, etc.)
#
# For retraining: Just modify RUN_NAME/EPOCHS and rerun this + next cell!

# Import required packages
import tlc
from tlc_ultralytics import YOLO, Settings

# ============================================================================
# STEP 1: Load Tables for Training
# ============================================================================
# Define 3LC project constants
PROJECT_NAME = "kaggle_cotton_weed_detection"
DATASET_NAME = "cotton_weed_det3"

print("=" * 70)
print("LOADING TABLES FOR TRAINING")
print("=" * 70)

try:
    # ========================================================================
    # OPTION 1: Load by Name (Recommended - Automatic Latest Version)
    # ========================================================================
    # This automatically loads the latest table version (includes Dashboard edits)

    train_table_latest = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name=f"{DATASET_NAME}-train1",
    ).latest()

    val_table_latest = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name=f"{DATASET_NAME}-val1",
    ).latest()

    print(
        f"\nâœ… Training table loaded: {len(train_table_latest)} samples (latest version)"
    )
    print(
        f"âœ… Validation table loaded: {len(val_table_latest)} samples (latest version)"
    )

    # Prepare tables dictionary for training
    tables = {"train": train_table_latest, "val": val_table_latest}

    # ========================================================================
    # OPTION 2: Load by URL (Alternative - Specific Table Version)
    # ========================================================================
    # Comment above and Uncomment below to load specific table URLs from Dashboard instead
    # Use this when you want a specific edited table version, not the latest

    """
    # Get URLs from Dashboard: Click on the Tables tab â†’ Copy URL from the spoecific table info panel to clipboard
    TRAIN_TABLE_URL = "paste_your_train_table_url_here"
    VAL_TABLE_URL = "paste_your_val_table_url_here"
    
    train_table_latest = tlc.Table.from_url(TRAIN_TABLE_URL)
    val_table_latest = tlc.Table.from_url(VAL_TABLE_URL)
    
    tables = {"train": train_table_latest, "val": val_table_latest}
    
    print(f"\nâœ… Training table loaded from URL: {len(tables['train'])} samples")
    print(f"âœ… Validation table loaded from URL: {len(tables['val'])} samples")
    """

    print("\n" + "=" * 70)
    print("âœ… Tables Ready!")
    print("=" * 70)

except Exception as e:
    print(f"\n Error loading tables: {e}")
    print("\nğŸ’¡ Troubleshooting:")
    print("   1. Make sure you ran Data Registration Cell at least once")
    print("   2. Check that PROJECT_NAME and DATASET_NAME match your setup")
    print("   3. Verify tables exist in Dashboard: https://dashboard.3lc.ai")
    raise

# ============================================================================
# STEP 2: Training Configuration
# ============================================================================

print("\n" + "=" * 70)
print("YOLOV8N TRAINING WITH 3LC TRACKING")
print("=" * 70)

# ============================================================================
# TRAINING CONSTANTS - Change these for each iteration
# ============================================================================
RUN_NAME = "yolov8n_baseline"  # Change for each run (e.g., "v2_fixed_labels")
RUN_DESCRIPTION = "Baseline YOLOv8n with default hyperparameters"

# Hyperparameters (customize these!)
EPOCHS = 5  # Number of training epochs
BATCH_SIZE = 16  # Batch size (adjust based on GPU memory)
IMAGE_SIZE = 640  # Input image size (FIXED by competition rules)
DEVICE = 0  # GPU device (0 for first GPU, 'cpu' for CPU)
WORKERS = 4  # Number of dataloader workers

# Display configuration
print("\n Training Configuration:")
print(f"   Run name: {RUN_NAME}")
print("   Model: YOLOv8n (ONLY model allowed)")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Image size: {IMAGE_SIZE} (FIXED)")
print(f"   Device: GPU {DEVICE}" if DEVICE != "cpu" else "   Device: CPU")

# Display dataset info (already loaded in STEP 1 above)
print("\n Dataset:")
print(f"   Training: {len(tables['train'])} samples")
print(f"   Validation: {len(tables['val'])} samples")

# Create 3LC Settings for run tracking
settings = Settings(
    project_name=PROJECT_NAME,
    run_name=RUN_NAME,
    run_description=RUN_DESCRIPTION,
    image_embeddings_dim=2,
)

print("\n" + "=" * 70)
print("âœ… CONFIGURATION COMPLETE!")
print("=" * 70)

print("\nğŸ’¡ Configuration Summary:")
print(f"   â€¢ Tables loaded: {len(tables['train'])} train, {len(tables['val'])} val")
print(f"   â€¢ Run name: {RUN_NAME}")
print(f"   â€¢ Training for: {EPOCHS} epochs")
print(f"   â€¢ Batch size: {BATCH_SIZE}")
print(f"   â€¢ Device: GPU {DEVICE}" if DEVICE != "cpu" else "   â€¢ Device: CPU")

print("\n Next: Run the cell below to start training!")
print("   (Review the configuration above before proceeding)")


# ============================================================================
# Train the Model
# ============================================================================
# This cell loads YOLOv8n and starts training.
# Make sure you ran the cell above first!


print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

# Load YOLOv8n pretrained model
print("\nLoading YOLOv8n pretrained weights...")
model = YOLO("yolov8n.pt")
print("âœ… Model loaded (3M parameters, 6MB size)")

# Train the model with 3LC tracking
print("\n Training in progress...")
print("=" * 70)

results = model.train(
    tables=tables,  # Use 3LC Tables
    name=RUN_NAME,  # Name for saving results (creates runs/detect/{RUN_NAME}/)
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    workers=WORKERS,
    settings=settings,  # 3LC tracking
    val=True,  # Validate during training
    # AUGMENTATION - Uncomment for better performance in later iterations:
    # mosaic=1.0,              # Mosaic augmentation - helps with scale variation
    # copy_paste=0.1,          # Copy-paste - helps with occlusion
    # mixup=0.05,              # Mixup - improves generalization
    # patience=20,             # Early stopping patience
)

print("\n" + "=" * 70)
print("âœ… TRAINING COMPLETE!")
print("=" * 70)

print("\nğŸ“� Model Weights Saved:")
print(f"   Best model: runs/detect/{RUN_NAME}/weights/best.pt")
print(f"   Last model: runs/detect/{RUN_NAME}/weights/last.pt")
print("\n Use 'best.pt' for predictions and submissions (highest validation mAP)")

print("\n Next Steps:")
print("   1. Visit 3LC Dashboard: https://dashboard.3lc.ai/")
print("   2. Open your Run to analyze model errors")
print("   3. Identify data issues:")
print("      â€¢ False negatives (missed detections)")
print("      â€¢ False positives (incorrect predictions)")
print("      â€¢ Class confusion")
print("      â€¢ Poor localization")
print("   4. Fix data issues in Dashboard")
print("   5. Retrain with improved data!")
print(
    "\nLearn more: https://docs.3lc.ai/3lc/latest/how-to/basics/open-project-table-run.html"
)


# ============================================================================
# OPTIONAL: Load Model Weights from Previous Training
# ============================================================================
# Uncomment ONE of the options below to load weights

# OPTION 1: Use the model from the training cell above (DEFAULT)
# If you just ran the training cell, the 'model' variable is already loaded
# â†’ No action needed, skip this cell!
"""
print("Current model status:")
try:
    print(f"âœ… Model loaded: {type(model).__name__}")
    print(f"  Using model from training session above")
except NameError:
    print("  No model found from training session")
    print("  You must load weights using one of the options below!")
"""
# OPTION 2: Load the LATEST trained model from runs folder
# Uncomment the ENTIRE block below to auto-load the most recent training run

"""
from tlc_ultralytics import YOLO
from pathlib import Path

# Find the most recent training run
runs_dir = Path("runs/detect")
if runs_dir.exists():
    train_dirs = sorted(runs_dir.glob("train*"), key=lambda x: x.stat().st_mtime, reverse=True)
    if train_dirs:
        latest_weights = train_dirs[0] / "weights" / "best.pt"
        if latest_weights.exists():
            print(f"\nLoading latest model: {latest_weights}")
            print(f"Using weights from Ultralytics run folder: {train_dirs[0]}")
            model = YOLO(str(latest_weights))
            print("âœ… Model loaded successfully!")
        else:
            print(f"!!! Weights not found: {latest_weights}")
    else:
        print("!!! No training runs found in runs/detect/")
else:
    print("!!! runs/detect/ directory not found")
"""

# By default, use the model from Cell 11 (training)
print("âœ“ Using model from Cell 11 training session")
print("  (To load different weights, uncomment one of the options above)")


# OPTION 3: Load SPECIFIC weights (custom path)
# Replace the path with your best model
"""
from tlc_ultralytics import YOLO

# Example paths:
# - "runs/detect/train/weights/best.pt"           # First training run
# - "runs/detect/train2/weights/best.pt"          # Second training run
# - "runs/detect/yolov8n_v3/weights/best.pt"      # Named run

CUSTOM_WEIGHTS_PATH = "runs/detect/train/weights/best.pt"

print(f"\nLoading custom weights: {CUSTOM_WEIGHTS_PATH}")
model = YOLO(CUSTOM_WEIGHTS_PATH)
print("âœ… Model loaded successfully!")
"""

# OPTION 4: Load pretrained YOLOv8n (no custom training)
# Use this if you want to test the baseline pretrained model
"""
from tlc_ultralytics import YOLO

print("\nLoading pretrained YOLOv8n (COCO weights)")
model = YOLO("yolov8n.pt")
print("âœ… Model loaded successfully!")
print("!!!  Note: This is the pretrained model, not trained on cotton weeds!")
"""

print("\n" + "=" * 70)
print("Ready to generate predictions!")
print("=" * 70)


# Import required packages
from pathlib import Path
import shutil

# Define paths and constants
WORK_DIR = Path(".")
TEST_DIR = WORK_DIR / "test" / "images"
PRED_DIR = Path("predictions")
IMAGE_SIZE = 640  # Competition requirement

# Get list of test images
test_images = list(TEST_DIR.glob("*.jpg"))

# ============================================================================
# SAFER FILE MANAGEMENT - Backup instead of delete
# ============================================================================
if PRED_DIR.exists():
    from datetime import datetime

    # Create timestamped backup instead of deleting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"predictions_backup_{timestamp}")

    print("âš ï¸�  Predictions folder exists. Creating backup...")
    print(f"   Moving to: {backup_dir}")

    shutil.move(str(PRED_DIR), str(backup_dir))

    print(f"âœ… Previous predictions backed up: {backup_dir}")
    print("   (Delete old backups manually if not needed)")
else:
    print("âœ… No existing predictions found")

print("Generating predictions on test set...")
print("=" * 50)
print(f"Test images: {TEST_DIR}")
print(f"Output directory: {PRED_DIR}")
print(f"Test set size: {len(test_images)} images")

# Run inference
print("\nRunning inference...")
test_results = model.predict(
    source=str(TEST_DIR),
    save=False,  # Don't save annotated images (faster, prevents duplication)
    save_txt=True,  # Save YOLO format predictions
    save_conf=True,  # Include confidence scores
    conf=0,  # Confidence threshold (adjust as needed)
    imgsz=IMAGE_SIZE,
    project=str(PRED_DIR.parent),
    name=PRED_DIR.name,
    exist_ok=False,  # Don't allow overwriting (ensures clean predictions)
)

print("\n----Predictions generated!")


# Import required packages
from pathlib import Path

# Define constants
CLASS_NAMES = ["Carpetweed", "Morning Glory", "Palmer Amaranth"]

# Analyze predictions
PRED_DIR = Path("predictions")  # Must match Cell 21
labels_dir = PRED_DIR / "labels"

if labels_dir.exists():
    print("Test Set Prediction Analysis:")
    print("=" * 50)

    pred_files = list(labels_dir.glob("*.txt"))

    class_counts = {i: 0 for i in range(len(CLASS_NAMES))}
    images_with_preds = 0
    total_detections = 0

    for pred_file in pred_files:
        if pred_file.stat().st_size > 0:
            images_with_preds += 1
            with open(pred_file, "r") as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 6:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
                            total_detections += 1

    print(f"Total test images: {len(test_images)}")
    print(f"Images with detections: {images_with_preds}")
    print(f"Images with no detections: {len(test_images) - images_with_preds}")
    print(f"Total detections: {total_detections}")

    print("\n Detections by class:")
    for class_id, count in class_counts.items():
        percentage = (count / total_detections * 100) if total_detections > 0 else 0
        print(f"   {CLASS_NAMES[class_id]:20s}: {count:4d} ({percentage:5.1f}%)")

    print("\n----Analysis complete!")
else:
    print("!!!!No predictions found.")


# ============================================================================
# STEP 8: Generate Kaggle Submission By running this cell of code
# ============================================================================
# Import required packages
from pathlib import Path

# Define paths
WORK_DIR = Path(".")  # Current directory
PRED_DIR = Path(
    "predictions"
)  # Prediction directory (change path if you want to convert from a different predictions folder)
TEST_DIR = (
    WORK_DIR / "test" / "images"
)  # Change path if you have the Test images stored Elsewhere


print("=" * 70)
print("GENERATING KAGGLE SUBMISSION")
print("=" * 70)

labels_dir = PRED_DIR / "labels"
output_csv = "submission.csv"

# Get all test images (deduplicate by stem to avoid duplicates from case-insensitive file systems)
test_images_dict = {}  # Use dict to automatically deduplicate by image_id (stem)
for ext in ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]:
    for img_path in TEST_DIR.glob(ext):
        image_id = img_path.stem  # filename without extension
        if image_id not in test_images_dict:
            test_images_dict[image_id] = img_path

# Convert to sorted list
test_images_list = [
    test_images_dict[img_id] for img_id in sorted(test_images_dict.keys())
]

print(f"\nâœ“ Found {len(test_images_list)} test images")
print(f"âœ“ Looking for predictions in: {labels_dir}")

# Create submission data
submission_data = []
images_with_preds = 0
images_without_preds = 0
total_boxes = 0

for img_path in test_images_list:
    image_id = img_path.stem
    pred_file = labels_dir / f"{image_id}.txt"

    # Check if prediction file exists and has content
    if pred_file.exists() and pred_file.stat().st_size > 0:
        prediction_boxes = []

        with open(pred_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()

                    # YOLO saves as: class xc yc w h conf (confidence is LAST!)
                    # Kaggle needs: class conf xc yc w h (confidence is SECOND!)
                    if len(parts) >= 6:
                        # Reorder values: move confidence from position 5 to position 1
                        class_id = parts[0]
                        conf = parts[5]  # Confidence is at the end in YOLO format
                        xc, yc, w, h = parts[1], parts[2], parts[3], parts[4]
                        box_str = f"{class_id} {conf} {xc} {yc} {w} {h}"
                        prediction_boxes.append(box_str)
                        total_boxes += 1

        if prediction_boxes:
            # Join all boxes with spaces
            prediction_string = " ".join(prediction_boxes)
            images_with_preds += 1
        else:
            prediction_string = "no box"
            images_without_preds += 1
    else:
        # No prediction file or empty file
        prediction_string = "no box"
        images_without_preds += 1

    submission_data.append(
        {"image_id": image_id, "prediction_string": prediction_string}
    )

# Create DataFrame with correct column names (lowercase!)
submission_df = pd.DataFrame(submission_data)
submission_df = submission_df[["image_id", "prediction_string"]]

# Save to CSV
submission_df.to_csv(output_csv, index=False)

# Print statistics
print("\n" + "=" * 70)
print("SUBMISSION STATISTICS")
print("=" * 70)
print(f"Total images:               {len(submission_df)}")
print(f"Images with predictions:    {images_with_preds}")
print(f"Images without predictions: {images_without_preds}")
print(f"Total bounding boxes:       {total_boxes}")
if len(submission_df) > 0:
    print(f"Average boxes per image:    {total_boxes / len(submission_df):.2f}")

# Show sample
print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)
display(submission_df.head(10))

# Validation
print("\n" + "=" * 70)
print("FORMAT VALIDATION")
print("=" * 70)

# Check format
errors = []
if list(submission_df.columns) != ["image_id", "prediction_string"]:
    errors.append(f"!!! Wrong columns: {list(submission_df.columns)}")
else:
    print("âœ“ Columns correct: image_id, prediction_string")

if len(submission_df) != len(test_images_list):
    errors.append("!!! Row count mismatch")
else:
    print(f"âœ“ Row count correct: {len(submission_df)}")

# Validate prediction format (sample first 20)
format_ok = True
for idx in range(min(20, len(submission_df))):
    pred_str = str(submission_df.iloc[idx]["prediction_string"])

    if pred_str == "no box":
        continue

    values = pred_str.split()
    if len(values) % 6 != 0:
        format_ok = False
        break

if format_ok:
    print("âœ“ All sampled predictions properly formatted (6 values per box)")
else:
    errors.append("!!! Some predictions have wrong format")

if errors:
    print("\n!!! VALIDATION FAILED:")
    for err in errors:
        print(f"  {err}")
else:
    print("\n" + "=" * 70)
print("âœ… SUBMISSION READY FOR KAGGLE!")
print("=" * 70)
print(f"\nFile: {output_csv}")
print("\n Upload 'submission.csv' to Kaggle!")
print("\n Tips:")
print("   - Check your score on the public leaderboard")
print("   - You have 3 submissions per day (use them wisely!)")
print("   - Select up to 2 final submissions for judging")

