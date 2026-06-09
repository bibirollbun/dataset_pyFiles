import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score # Added accuracy_score
import torch.nn as nn
import torchvision.models as models
from tqdm.notebook import tqdm # Use tqdm notebook version for better display in notebooks


# --- 1. Setup and Seed ---
print("--- Initializing ---")
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# Optional: Deterministic behavior (can slow down training/inference)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


# --- 2. Configuration ---
print("--- Configuring Paths and Parameters ---")
APTOS_CSV_PATH = '/kaggle/input/aptos2019-blindness-detection/test.csv'
APTOS_IMG_DIR = '/kaggle/input/aptos2019-blindness-detection/test_images/'

# === PATH TO PRE-TRAINED BINARY MODEL (OUTPUTS 0 OR 1) ===
BINARY_MODEL_PATH = '/kaggle/input/eyepacs-ddr-resnet-model/EyePacs_DDR_best_resnet_model.pth'

# === PATH TO PRE-TRAINED MULTI-CLASS MODEL (OUTPUTS 0-3 for classes 1-4) ===
MULTI_CLASS_MODEL_PATH = '/kaggle/input/multi-class-reg-model/Reg_QWK_Stop_resnet_model.pth'

BATCH_SIZE_INFERENCE = 128 # Adjust based on GPU memory if needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2 # Number of workers for DataLoader

# --- Determine Binary Model Output Size ---
BINARY_NUM_CLASSES = 1 
MULTI_NUM_CLASSES = 4 # Outputs 0, 1, 2, 3 (for original labels 1, 2, 3, 4)

print(f"Using device: {DEVICE}")
print(f"APTOS CSV path: {APTOS_CSV_PATH}")
print(f"APTOS Image dir: {APTOS_IMG_DIR}")
print(f"Binary model path: {BINARY_MODEL_PATH}")
print(f"Multi-class model path: {MULTI_CLASS_MODEL_PATH}")
print(f"Binary model configured for {BINARY_NUM_CLASSES} output classes.")


# --- 3. Transforms ---
print("--- Defining Image Transforms ---")
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
print("Inference transforms defined.")

# --- 4. Model Definition ---
print("--- Defining Model Architecture ---")
class ResNetModel(nn.Module):
    def __init__(self, model_path=None, num_classes=1, load_weights=True, pretrained_backbone=False):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained_backbone)  # Load ResNet18 backbone

        # Remove the original FC layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the last layer

        self.new_classifier = nn.Sequential( 
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes) # Output layer size depends on model type
        )
        
        if model_path and load_weights:
            if not os.path.exists(model_path):
                 print(f"ERROR: Weight file not found at {model_path}. Model will be uninitialized.")
                 return
            try:
                print(f"Loading weights from: {model_path} for {num_classes} classes")
                state_dict = torch.load(model_path, map_location=DEVICE)

                load_result = self.load_state_dict(state_dict, strict=False) 
                print(f"Weight loading result: {load_result}")
                if load_result.missing_keys:
                    print(f"Warning: Missing keys during load: {load_result.missing_keys}")
                if load_result.unexpected_keys:
                    print(f"Warning: Unexpected keys during load: {load_result.unexpected_keys}")

            except Exception as e:
                print(f"Error loading weights for {num_classes}-class model from {model_path}: {e}")
                print("Model will proceed with potentially uninitialized/partially loaded weights.")

    def forward(self, x):
        x = self.resnet(x)
        x = self.new_classifier(x)
        return x

print("ResNetModel class defined.")


# --- 5. Load Pre-trained Models ---
print("--- Loading Pre-trained Models ---")
# --- Load Binary Model ---
print("Loading Binary Model...")
binary_model = ResNetModel(model_path=BINARY_MODEL_PATH, num_classes=BINARY_NUM_CLASSES)
binary_model.to(DEVICE)
binary_model.eval()
print("Binary model loaded and set to eval mode.")

# --- Load Multi-Class Model ---
print("\nLoading Multi-Class Model...")
multi_class_model = ResNetModel(model_path=MULTI_CLASS_MODEL_PATH, num_classes=MULTI_NUM_CLASSES)
multi_class_model.to(DEVICE)
multi_class_model.eval()
print("Multi-class model loaded and set to eval mode.")

# --- 6. APTOS Dataset and DataLoader ---
print("--- Preparing APTOS Data ---")
class AptosDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, file_ext=".png"):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.file_ext = file_ext
        self.df.columns = self.df.columns.str.strip()
        print(f"Dataset initialized. Found columns: {self.df.columns.tolist()}")
        # --- MODIFICATION 1: Check only for id_code ---
        if 'id_code' not in self.df.columns:
            raise ValueError("CSV file must contain 'id_code' column.")
        # --- End MODIFICATION 1 ---

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx >= len(self.df):
             raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self.df)}")
        row = self.df.iloc[idx]
        img_id = row["id_code"]

        # --- MODIFICATION 2: Handle missing 'diagnosis' column ---
        # Assign a dummy label if 'diagnosis' is not present (e.g., in test.csv)
        if 'diagnosis' in row and pd.notna(row['diagnosis']):
            original_label = int(row["diagnosis"]) # Use real label if available
        else:
            original_label = 0 # Assign dummy label 0 for test set or missing values
        # --- End MODIFICATION 2 ---

        img_filename = f"{img_id}{self.file_ext}"
        img_path = os.path.join(self.img_dir, img_filename)
        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, IOError, FileNotFoundError) as e:
            print(f"Warning: Error loading image {img_path}: {e}. Returning None.")
            # Return None for image, but keep other info for collate_fn and potential tracking
            return None, original_label, img_id, idx
        if self.transform:
            image = self.transform(image)
        # Return image, the determined label (real or dummy), image_id, and original index
        return image, original_label, img_id, idx

# Load APTOS data (Ensure APTOS_CSV_PATH points to test.csv)
if not os.path.exists(APTOS_CSV_PATH):
    raise FileNotFoundError(f"APTOS CSV file not found at {APTOS_CSV_PATH}")
if not os.path.isdir(APTOS_IMG_DIR):
     raise FileNotFoundError(f"APTOS Image directory not found at {APTOS_IMG_DIR}")

aptos_df = pd.read_csv(APTOS_CSV_PATH)
print(f"Loaded APTOS dataframe with {len(aptos_df)} samples from {APTOS_CSV_PATH}.")

# Create dataset
aptos_dataset = AptosDataset(df=aptos_df, img_dir=APTOS_IMG_DIR, transform=inference_transforms)

# Define a collate function to handle None returns from dataset
def collate_fn_skip_none(batch):
    # Filter out items where the image (item[0]) is None
    filtered_batch = [item for item in batch if item[0] is not None]
    if not filtered_batch:
        return None # Return None if the entire batch failed to load
    # Use default collate function on the filtered batch
    return torch.utils.data.dataloader.default_collate(filtered_batch)

# Create DataLoader
aptos_loader = DataLoader(
    aptos_dataset,
    batch_size=BATCH_SIZE_INFERENCE,
    shuffle=False, # Keep shuffle=False for test set prediction order
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn_skip_none, # Use the custom collate function
    pin_memory=True if DEVICE.type == 'cuda' else False
)
print(f"APTOS DataLoader ready with {len(aptos_loader)} batches.")


# --- 7. Stage 1: Binary Classification ---
print("\n--- Starting Stage 1: Binary Classification ---")
stage1_no_dr_data = [] # List to store {'id_code': ..., 'diagnosis': 0}
stage1_dr_ids = []     # List to store id_codes predicted as DR (1)
image_ids_processed = [] # Keep track of all processed image ids
original_indices_processed = [] # Keep track of original indices if needed later
skipped_batches_stage1 = 0

with torch.no_grad():
    for batch_data in tqdm(aptos_loader, desc="Stage 1: Binary Prediction"):
        if batch_data is None:
            print("Skipping a batch due to image loading errors in Stage 1.")
            skipped_batches_stage1 += 1
            continue

        # Unpack batch data - we don't need true_labels for test set prediction
        inputs, _, img_ids_batch, original_indices_batch = batch_data
        inputs = inputs.to(DEVICE)

        outputs = binary_model(inputs)

        # --- Determine Binary Prediction (0 or 1) ---
        if BINARY_NUM_CLASSES == 2:
            probs = torch.softmax(outputs, dim=1)
            predicted_binary = torch.argmax(probs, dim=1) # 0 or 1
        elif BINARY_NUM_CLASSES == 1:
            probs = torch.sigmoid(outputs).squeeze(-1) # Ensure squeeze removes last dim if size 1
            # Handle cases where batch size is 1 after filtering
            if probs.ndim == 0:
                 probs = probs.unsqueeze(0)
            predicted_binary = (probs > 0.5).long() # 0 or 1
        else:
            raise ValueError(f"Unsupported BINARY_NUM_CLASSES: {BINARY_NUM_CLASSES}.")
        # --------------------------------------------

        predicted_binary_np = predicted_binary.cpu().numpy()
        img_ids_list_batch = list(img_ids_batch)
        original_indices_list_batch = original_indices_batch.cpu().numpy()

        # --- Populate lists based on binary prediction ---
        for i in range(len(predicted_binary_np)):
            img_id = img_ids_list_batch[i]
            prediction = predicted_binary_np[i]
            original_idx = original_indices_list_batch[i]

            image_ids_processed.append(img_id)
            original_indices_processed.append(original_idx)

            if prediction == 0:
                stage1_no_dr_data.append({'id_code': img_id, 'diagnosis': 0})
            else: # prediction == 1
                stage1_dr_ids.append(img_id)
        # -------------------------------------------------

if skipped_batches_stage1 > 0:
    print(f"\nWarning: Skipped {skipped_batches_stage1} batches in Stage 1 due to loading errors.")
    print(f"Number of images processed ({len(image_ids_processed)}) might not match original dataset size ({len(aptos_df)}).")

# --- Create DataFrame for No DR predictions and Set for DR ids ---
df_no_dr = pd.DataFrame(stage1_no_dr_data)
dr_image_ids_set = set(stage1_dr_ids) # Use a set for fast lookups in Stage 2

print(f"Binary prediction finished.")
print(f"Found {len(df_no_dr)} No DR predictions (label 0).")
print(f"Found {len(dr_image_ids_set)} DR predictions (label 1) to be processed in Stage 2.")
print(f"Total images processed in Stage 1: {len(image_ids_processed)}")
if len(df_no_dr) + len(dr_image_ids_set) != len(image_ids_processed):
     print(f"Warning: Discrepancy in counts! NoDR ({len(df_no_dr)}) + DR ({len(dr_image_ids_set)}) != Processed ({len(image_ids_processed)})")


df_no_dr


# dr_image_ids_set


# --- 8. Stage 2: Multi-Class Classification (on DR Subset) ---
print("\n--- Starting Stage 2: Multi-Class Classification ---")
stage2_results_data = [] # List to store {'id_code': ..., 'diagnosis': 1-4}
processed_ids_stage2 = set()
skipped_batches_stage2 = 0

with torch.no_grad():
    for batch_data in tqdm(aptos_loader, desc="Stage 2: Multi-Class Prediction"):
        if batch_data is None:
            print("Skipping a batch due to image loading errors in Stage 2.")
            skipped_batches_stage2 += 1
            continue

        # Unpack batch data
        inputs, _, img_ids_batch, _ = batch_data # Don't need labels or original indices here

        # --- Filter batch for images predicted as DR in Stage 1 ---
        indices_to_process_in_batch = []
        ids_to_process_in_batch = []
        inputs_to_process_list = []

        for i, img_id in enumerate(img_ids_batch):
            if img_id in dr_image_ids_set:
                indices_to_process_in_batch.append(i)
                ids_to_process_in_batch.append(img_id)
                inputs_to_process_list.append(inputs[i])
        # ---------------------------------------------------------

        # --- Proceed only if there are DR images in this batch ---
        if not inputs_to_process_list:
            continue # Skip to next batch if no DR images here

        inputs_batch_multiclass = torch.stack(inputs_to_process_list).to(DEVICE)
        # ---------------------------------------------------------

        # --- Get Multi-Class Predictions ---
        outputs_multi = multi_class_model(inputs_batch_multiclass)
        _, predicted_multi_indices = torch.max(outputs_multi, 1) # Indices 0-3
        # -----------------------------------

        # --- Remap predictions (0-3) to DR scale (1-4) ---
        predicted_multi_labels = predicted_multi_indices.cpu().numpy() + 1
        # -------------------------------------------------

        # --- Store results ---
        for i in range(len(ids_to_process_in_batch)):
            img_id = ids_to_process_in_batch[i]
            multi_label = predicted_multi_labels[i]
            stage2_results_data.append({'id_code': img_id, 'diagnosis': multi_label})
            processed_ids_stage2.add(img_id)
        # ---------------------

if skipped_batches_stage2 > 0:
    print(f"\nWarning: Skipped {skipped_batches_stage2} batches in Stage 2 due to loading errors.")

# --- Create DataFrame for Stage 2 results ---
df_stage2_results = pd.DataFrame(stage2_results_data)

print(f"Multi-class prediction finished.")
print(f"Processed {len(processed_ids_stage2)} unique images in Stage 2.")
print(f"Generated {len(df_stage2_results)} multi-class predictions (labels 1-4).")

# --- Verification ---
if len(processed_ids_stage2) != len(dr_image_ids_set):
    print(f"Warning: Mismatch! Number of unique IDs processed in Stage 2 ({len(processed_ids_stage2)}) "
          f"does not match the number of DR IDs from Stage 1 ({len(dr_image_ids_set)}).")
    missed_in_stage2 = dr_image_ids_set - processed_ids_stage2
    if missed_in_stage2:
         print(f"-> {len(missed_in_stage2)} DR IDs from Stage 1 were not processed in Stage 2 (likely due to skipped batches). Example missed IDs: {list(missed_in_stage2)[:10]}")
else:
    print("Verification successful: All DR IDs from Stage 1 were processed in Stage 2.")

if not df_stage2_results.empty:
    print("\nSample of Stage 2 Results (Multi-class):")
    print(df_stage2_results.head())
else:
    print("\nNo Stage 2 results generated (either no DR images found or all were in skipped batches).")


# --- 9. Combine Results & Visualize Prediction Distribution ---
print("\n--- Combining Stage 1 and Stage 2 Results ---")

# Concatenate the DataFrames
if not df_stage2_results.empty:
    final_results_df = pd.concat([df_no_dr, df_stage2_results], ignore_index=True)
else:
    print("Warning: Stage 2 results DataFrame is empty. Using only Stage 1 No DR results.")
    final_results_df = df_no_dr.copy()


print(f"Combined DataFrame created with {len(final_results_df)} entries.")
print("Sample of combined results:")
print(final_results_df.head())
print("\nValue counts in combined results:")
print(final_results_df['diagnosis'].value_counts().sort_index())

# --- Distribution of Final Predictions (Still Useful) ---
# This plot shows the distribution of the model's predictions on the processed test set samples.
target_names_aptos = ['No DR (0)', 'Mild DR (1)', 'Moderate DR (2)', 'Severe DR (3)', 'Proliferative DR (4)']

plt.figure(figsize=(8, 6))
# Use the 'diagnosis' column from the combined DataFrame
sns.countplot(x='diagnosis', data=final_results_df, order=np.arange(len(target_names_aptos)), palette="viridis")
plt.xticks(ticks=np.arange(len(target_names_aptos)), labels=target_names_aptos, rotation=45, ha='right')
plt.xlabel("Predicted Class (Combined Stages)")
plt.ylabel("Count")
plt.title("Distribution of Final Combined Predictions on Processed Test Set")
plt.tight_layout()
plt.grid(axis="y", linestyle='--', alpha=0.7)
plt.savefig("final_prediction_distribution_test.png")
plt.show()

print("\nFinished Combining and Visualization Section.")
# Note: Evaluation metrics like QWK, classification report, and confusion matrix
# are omitted here as we don't have ground truth labels for the test set.


# --- 10. Generate Submission File ---
import pandas as pd # Ensure pandas is imported

print("\n--- Generating Submission File ---")

# final_results_df was created in the previous cell

print(f"Number of predictions in final_results_df: {len(final_results_df)}")
print(f"Number of unique id_codes in final_results_df: {final_results_df['id_code'].nunique()}")

if len(final_results_df) != final_results_df['id_code'].nunique():
     print("Warning: Duplicate id_codes found in the combined results. Check concatenation logic.")
     # Optional: Decide how to handle duplicates, e.g., keep first
     final_results_df = final_results_df.drop_duplicates(subset='id_code', keep='first')
     print(f"Dropped duplicates, new count: {len(final_results_df)}")

# Load the sample submission file to get all required id_codes
try:
    sample_sub = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
    print(f"Sample submission length: {len(sample_sub)}")
except FileNotFoundError:
    print("Error: sample_submission.csv not found. Cannot generate submission file correctly.")
    # Handle error appropriately, maybe stop execution
    raise

# Merge predictions with the sample submission using 'id_code'
# Use a left merge to keep all IDs from the sample submission and match predictions
submission_df = pd.merge(sample_sub[['id_code']], final_results_df, on='id_code', how='left')

# Check for missing predictions (images potentially skipped during loading/processing)
missing_preds = submission_df['diagnosis'].isnull().sum()
if missing_preds > 0:
    print(f"Warning: {missing_preds} images from sample_submission were not found in the processed results.")
    # Fill missing predictions. A common strategy is to predict 0 (No DR) for missing ones.
    print("Filling missing predictions with 0 (No DR).")
    submission_df['diagnosis'] = submission_df['diagnosis'].fillna(0) # Fill NaN with 0
    # Verify NaNs are filled
    if submission_df['diagnosis'].isnull().sum() > 0:
        print("Error: Failed to fill all missing predictions!")


# Ensure the diagnosis column is integer type
# Use .astype(int) after filling NaNs
submission_df['diagnosis'] = submission_df['diagnosis'].astype(int)


print("\nFinal submission DataFrame head:")
print(submission_df.head())
print(f"\nFinal submission DataFrame length: {len(submission_df)}")

# Check if final submission length matches sample submission
if len(submission_df) != len(sample_sub):
    print(f"Error: Final submission length ({len(submission_df)}) does not match sample submission length ({len(sample_sub)})!")
else:
    print("Final submission length matches sample submission length.")

# Save the submission file
try:
    submission_df.to_csv('submission.csv', index=False)
    print("\nSubmission file 'submission.csv' created successfully.")
except Exception as e:
    print(f"\nError saving submission file: {e}")

