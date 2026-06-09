# Cell 1: Setup, Package Installation, Offline Model Weights & SRC Code from Dataset

# --- Essential Imports for Setup ---
import os
import shutil
import sys
# from kaggle_secrets import UserSecretsClient # No longer needed for GITHUB_PAT for cloning

print("===============================================================================")
print("=== Stage 1: Installing/Updating Key Packages (Minimal) ===")
print("===============================================================================")
#!pip install --upgrade timm hdbscan umap-learn Pillow tqdm --quiet 
print("Key package installation/upgrade attempt complete.")
print("-" * 60)


print("\n===============================================================================")
print("=== Stage 2: Setting up Offline Model Weights ===")
print("===============================================================================")
# *** ACTION: VERIFY AND REPLACE 'public-dino-v2-weights-slug' WITH THE ACTUAL SLUG ***
DINOV2_PUBLIC_DATASET_SLUG = "dino-small-pretrained" 
DINOV2_PUBLIC_DATASET_PATH = f"/kaggle/input/{DINOV2_PUBLIC_DATASET_SLUG}"
DINOV2_WEIGHT_FILENAME_IN_DATASET = "dinov2_vits14_pretrain.pth"
EXPECTED_DINOV2_FILENAME_IN_CACHE = "dinov2_vits14_pretrain.pth"

PYTORCH_HUB_CACHE_DIR = "/root/.cache/torch/hub/checkpoints/"
os.makedirs(PYTORCH_HUB_CACHE_DIR, exist_ok=True)

source_dino_weight_path = os.path.join(DINOV2_PUBLIC_DATASET_PATH, DINOV2_WEIGHT_FILENAME_IN_DATASET)
target_dino_weight_path = os.path.join(PYTORCH_HUB_CACHE_DIR, EXPECTED_DINOV2_FILENAME_IN_CACHE)

print(f"DINOv2 Source weight path: {source_dino_weight_path}")
if os.path.exists(source_dino_weight_path):
    if source_dino_weight_path != target_dino_weight_path:
        shutil.copyfile(source_dino_weight_path, target_dino_weight_path)
        print("DINOv2 weights copied to PyTorch Hub cache successfully.")
else:
    print(f"CRITICAL WARNING: DINOv2 weight file NOT FOUND at {source_dino_weight_path}.")

# --- ALIKED & LightGlue Weights (from your team's private Kaggle Dataset) ---
# *** ACTION: VERIFY AND REPLACE 'imc2025-team-model-weights' WITH YOUR TEAM'S ACTUAL DATASET SLUG ***
TEAM_MODEL_WEIGHTS_DATASET_SLUG = "imc2025-team-model-weights"
TEAM_MODEL_WEIGHTS_DATASET_PATH = f"/kaggle/input/{TEAM_MODEL_WEIGHTS_DATASET_SLUG}"

ALIKED_WEIGHT_FILENAME = "aliked-n16.pth" 
LIGHTGLUE_WEIGHT_FILENAME = "aliked_lightglue_v0-1_arxiv.pth" 

ALIKED_WEIGHT_KAGGLE_PATH = os.path.join(TEAM_MODEL_WEIGHTS_DATASET_PATH, ALIKED_WEIGHT_FILENAME)
LIGHTGLUE_WEIGHT_KAGGLE_PATH = os.path.join(TEAM_MODEL_WEIGHTS_DATASET_PATH, LIGHTGLUE_WEIGHT_FILENAME) if LIGHTGLUE_WEIGHT_FILENAME else None

if not os.path.exists(ALIKED_WEIGHT_KAGGLE_PATH): print(f"CRITICAL WARNING: ALIKED weight file NOT FOUND at {ALIKED_WEIGHT_KAGGLE_PATH}.")
if LIGHTGLUE_WEIGHT_KAGGLE_PATH and not os.path.exists(LIGHTGLUE_WEIGHT_KAGGLE_PATH): print(f"CRITICAL WARNING: LightGlue weight file NOT FOUND at {LIGHTGLUE_WEIGHT_KAGGLE_PATH}.")
print("-" * 60)


print("\n===============================================================================")
print("=== Stage 3: Accessing SRC Code from Kaggle Dataset ===")
print("===============================================================================")

# Cell 1: Setup, Package Installation, Offline Model Weights & SRC Code from Dataset

# ... (Your Stage 1: Pip Installs - keep as is or minimal) ...
# ... (Your Stage 2: Offline Model Weights - keep as is, ensuring DINOv2 weights are copied) ...

print("\n===============================================================================")
print("=== Stage 3: Accessing SRC Code from Kaggle Dataset ===")
print("===============================================================================")
# *** ACTION: VERIFY AND REPLACE 'imc2025-team-src-code-final' WITH YOUR SRC CODE DATASET SLUG ***
SRC_CODE_DATASET_SLUG = "imc2025-team-src-code-final" 
SRC_CODE_DATASET_PATH = f"/kaggle/input/{SRC_CODE_DATASET_SLUG}"

# Path to the 'src' directory *directly inside* your Kaggle Dataset
# Since os.listdir showed ['src'], it means src/ is directly under the dataset path
PATH_TO_SRC_IN_DATASET = os.path.join(SRC_CODE_DATASET_PATH, 'src')

if os.path.exists(PATH_TO_SRC_IN_DATASET) and os.path.isdir(PATH_TO_SRC_IN_DATASET):
    print(f"Found 'src' directory directly in Kaggle Dataset: {PATH_TO_SRC_IN_DATASET}")
    # No unzipping needed. The SRC_PATH for sys.path will be this direct path.
else:
    print(f"CRITICAL ERROR: 'src' directory NOT FOUND at {PATH_TO_SRC_IN_DATASET}.")
    print(f"         Ensure your dataset '{SRC_CODE_DATASET_SLUG}' contains an 'src' folder at its root.")
    print(f"         Contents of '{SRC_CODE_DATASET_PATH}':")
    if os.path.exists(SRC_CODE_DATASET_PATH):
        print(os.listdir(SRC_CODE_DATASET_PATH))
    else:
        print(f"         Dataset path '{SRC_CODE_DATASET_PATH}' itself not found.")
print("-" * 60)

# --- Stage 4: Setting up Python Path & Verifying Modules ---
print("\n===============================================================================")
print("=== Stage 4: Setting up Python Path & Verifying Modules ===")
print("===============================================================================")
# SRC_PATH_FROM_DATASET will now be the direct path from the Kaggle Dataset
SRC_PATH_FROM_DATASET = PATH_TO_SRC_IN_DATASET # Use the path identified in Stage 3

if os.path.exists(SRC_PATH_FROM_DATASET) and os.path.isdir(SRC_PATH_FROM_DATASET):
    # We want to add the directory *containing* your modules (data, features, etc.)
    # which is SRC_PATH_FROM_DATASET itself if it is .../input/dataset_slug/src/
    sys.path.insert(0, SRC_PATH_FROM_DATASET) 
    print(f"'{SRC_PATH_FROM_DATASET}' added to sys.path.")
    print(f"Contents of {SRC_PATH_FROM_DATASET} (should be your module folders like 'data', 'features', etc.):")
    print(os.listdir(SRC_PATH_FROM_DATASET))
else:
    print(f"CRITICAL WARNING: Source directory '{SRC_PATH_FROM_DATASET}' does not exist or is not a directory.")
    print(f"         This usually means the dataset structure is not as expected or it wasn't added correctly.")
    print(f"         Imports from 'src' (actually from its submodules) will fail.")
print("-" * 60)

# --- Verify expected module directories within src/ ---
print("Checking for expected module directories within src/:")
# Update this list based on PRs merged to the 'src' folder you uploaded to the dataset
expected_module_dirs = {
    'data': True, 
    'features': True, 
    'clustering': True, 
    'matching_strategies': True,
    'sfm': True # Set to True if Davin's refactored script is in the 'src/sfm/' you uploaded
}
all_critical_modules_found = True
if os.path.exists(SRC_PATH_FROM_DATASET) and os.path.isdir(SRC_PATH_FROM_DATASET):
    for subdir_name, is_critical_for_this_run in expected_module_dirs.items():
        path_to_check = os.path.join(SRC_PATH_FROM_DATASET, subdir_name)
        if os.path.exists(path_to_check) and os.path.isdir(path_to_check):
            print(f"  Found: '{subdir_name}/'")
        else:
            if is_critical_for_this_run:
                print(f"  CRITICAL WARNING: Expected CRITICAL module directory NOT FOUND: '{path_to_check}'")
                all_critical_modules_found = False
            else:
                print(f"  INFO: Module directory not found: '{path_to_check}' (is_critical={is_critical_for_this_run})")
    if all_critical_modules_found:
        print("All critical expected module directories appear to be found in src/.")
    else:
        print("CRITICAL WARNING: Not all essential module directories were found. Subsequent imports might fail.")
else:
    print(f"Skipping module directory check as SRC_PATH_FROM_DATASET ('{SRC_PATH_FROM_DATASET}') was not found.")
    all_critical_modules_found = False

print("-" * 60)
print("Setup cell complete. Review any CRITICAL warnings above carefully.")

PIPELINE_SETUP_OK = True
# Check if the critical src path itself was found and if critical modules were found
if not (os.path.exists(SRC_PATH_FROM_DATASET) and os.path.isdir(SRC_PATH_FROM_DATASET)) or \
   not all_critical_modules_found:
    PIPELINE_SETUP_OK = False
    print("\n!!! PIPELINE SETUP HAS CRITICAL ISSUES - SUBSEQUENT CELLS MAY FAIL OR PRODUCE INVALID RESULTS !!!")


# Cell 2: Imports & Path Definitions

print("=== Stage 2.1: Importing Core Libraries and Custom Modules ===")

# --- Standard Libraries ---
import pandas as pd
import numpy as np
import os
import sys
import torch 
import timm  
from PIL import Image 
from torchvision import transforms # Often used by timm's model transforms
from tqdm.auto import tqdm 

# --- Import Our Custom Modules ---
# These imports assume the PRs for these modules have been merged into the 
# TARGET_BRANCH (e.g., 'main') that was cloned in the previous setup cell.

# REPO_NAME and TARGET_BRANCH should be defined in Cell 1 (Setup Cell)
# If not, define them here or ensure they are passed correctly.
if 'REPO_NAME' not in globals():
    print("CRITICAL WARNING: REPO_NAME not defined from Setup Cell. Using default, but this may be incorrect.")
    REPO_NAME = "imc2025-team-maschinelles-lernen-1" # Fallback, but best to define in Cell 1

if 'TARGET_BRANCH' not in globals():
    print("CRITICAL WARNING: TARGET_BRANCH not defined from Setup Cell. Using default 'main'.")
    TARGET_BRANCH = "main" # Fallback

MODULE_BASE_PATH = f'/kaggle/working/{REPO_NAME}/src'

print(f"Attempting to import modules from: {MODULE_BASE_PATH} (cloned from branch: '{TARGET_BRANCH}')")

try:
    from data.preprocessing import load_image_pil, resize_image_maintain_aspect_ratio 
    print("- Successfully imported: data.preprocessing")
except ImportError as e:
    print(f"- WARNING: Could not import from data.preprocessing: {e}.")
    print(f"  Ensure 'src/data/preprocessing.py' exists in the cloned repo ('{TARGET_BRANCH}' branch) and its PR is merged.")

try:
    from features.global_dino_extractor import DinoV2EmbeddingExtractor 
    print("- Successfully imported: features.global_dino_extractor")
except ImportError as e:
    print(f"- WARNING: Could not import from features.global_dino_extractor: {e}.")
    print(f"  Ensure 'src/features/global_dino_extractor.py' exists and PR is merged to '{TARGET_BRANCH}'.")

try:
    from clustering.hdbscan_clusterer import run_hdbscan_clustering, load_embeddings 
    print("- Successfully imported: clustering.hdbscan_clusterer")
except ImportError as e:
    print(f"- WARNING: Could not import from clustering.hdbscan_clusterer: {e}.")
    print(f"  Ensure 'src/clustering/hdbscan_clusterer.py' exists and PR is merged to '{TARGET_BRANCH}'.")

try:
    from matching_strategies.pair_selector import select_pairs_by_embedding_similarity
    print("- Successfully imported: matching_strategies.pair_selector")
except ImportError as e:
    print(f"- WARNING: Could not import from matching_strategies.pair_selector: {e}.")
    print(f"  Ensure 'src/matching_strategies/pair_selector.py' exists and PR is merged to '{TARGET_BRANCH}'.")

# Placeholder for Davin's/Raman's SfM module - uncomment when its PR is ready and merged
# try:
#     from sfm.scene_reconstructor import reconstruct_one_scene # Example name
#     print("- Successfully imported: sfm.scene_reconstructor")
# except ImportError as e:
#     print(f"- INFO: sfm.scene_reconstructor not yet imported (PR might not be merged or module not ready): {e}")

print("-" * 60)

# --- Define Key Directory Paths ---
print("=== Stage 2.2: Defining Key Directory Paths ===")

KAGGLE_INPUT_DIR = '/kaggle/input/image-matching-challenge-2025/'
KAGGLE_WORKING_DIR = '/kaggle/working/' # Writable directory

# Define paths for saving intermediate outputs generated by this notebook
# These will be in /kaggle/working/, which is cleared after the session but available for submission output.
FEATURES_NPZ_DIR = os.path.join(KAGGLE_WORKING_DIR, 'features_output', 'dino_embeddings')
CLUSTERING_CSV_DIR = os.path.join(KAGGLE_WORKING_DIR, 'features_output', 'clustering_results')
# Path for the final submission file
SUBMISSION_DIR = KAGGLE_WORKING_DIR # submission.csv goes directly in /kaggle/working/

print(f"Competition Input Directory: {KAGGLE_INPUT_DIR}")
print(f"Notebook Working Directory: {KAGGLE_WORKING_DIR}")
print(f"Path for DINOv2 embeddings output (NPZ): {FEATURES_NPZ_DIR}")
print(f"Path for Clustering results output (CSV): {CLUSTERING_CSV_DIR}")
print(f"Path for final submission.csv: {SUBMISSION_DIR}")

# Ensure these output directories exist
os.makedirs(FEATURES_NPZ_DIR, exist_ok=True)
os.makedirs(CLUSTERING_CSV_DIR, exist_ok=True)
print("Output directories for features and clustering ensured.")
print("-" * 60)
print("Imports and Path Definitions cell complete. Review any warnings carefully.")


# Cell 3: Data Loading & Test Image Path Preparation

import pandas as pd
import os # Should already be imported

print("=== Stage 3: Loading sample_submission.csv and Preparing Test Image Paths ===")

# Initialize test_set_df to ensure it's always defined with correct columns
test_set_df = pd.DataFrame(columns=['dataset', 'image', 'full_path', 'image_id_combined'])
sample_submission_df = pd.DataFrame() # Initialize as empty
ALL_IMAGES_LISTED_IN_SAMPLE_SUB_PREPARED = False # Flag

# KAGGLE_INPUT_DIR should be defined in Cell 2 (Imports & Path Definitions)
if 'KAGGLE_INPUT_DIR' not in globals():
    print("CRITICAL ERROR: KAGGLE_INPUT_DIR not defined. Please run Cell 2 first.")
else:
    SAMPLE_SUBMISSION_PATH = os.path.join(KAGGLE_INPUT_DIR, 'sample_submission.csv')
    
    print(f"Attempting to load sample_submission.csv from: {SAMPLE_SUBMISSION_PATH}")
    if not os.path.exists(SAMPLE_SUBMISSION_PATH):
        print(f"CRITICAL ERROR: sample_submission.csv not found at {SAMPLE_SUBMISSION_PATH}")
        print("       This is essential for knowing which test images to process. Pipeline cannot effectively proceed.")
    else:
        try:
            sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
            print(f"Loaded sample_submission.csv with {len(sample_submission_df)} images to process.")

            # --- Define the function to get actual test image paths (ROBUST VERSION) ---
            def get_actual_test_image_path_robust(row):
                image_filename = str(row['image'])
                dataset_name = str(row['dataset']) # Still useful for image_id_combined and final submission

                # Common possible locations for test images in Kaggle
                # KAGGLE_INPUT_DIR is /kaggle/input/image-matching-challenge-2025/
                possible_base_dirs = [
                    os.path.join(KAGGLE_INPUT_DIR, 'test_images'), # e.g., test_images/img.png
                    os.path.join(KAGGLE_INPUT_DIR, 'test'),        # e.g., test/img.png
                    os.path.join(KAGGLE_INPUT_DIR, 'images'),      # e.g., images/img.png
                    # Fallback for sample data that might be nested like train data
                    os.path.join(KAGGLE_INPUT_DIR, 'test', dataset_name) # e.g., test/dataset_X/img.png
                ]
                
                for base_dir_option in possible_base_dirs:
                    # For flat structures, image is directly in base_dir_option
                    # For nested (last option), image is in base_dir_option (which is .../test/dataset_name)
                    # This logic assumes the last option is the only one that would use the dataset_name in path
                    if base_dir_option.endswith(dataset_name): # Check if it's the nested path attempt
                         constructed_path = os.path.join(base_dir_option, image_filename)
                    else: # Flat structure attempts
                         constructed_path = os.path.join(base_dir_option, image_filename)
                    
                    if os.path.exists(constructed_path):
                        # print(f"DEBUG: Found image {image_filename} at {constructed_path}") # Verbose, for debugging only
                        return constructed_path
                
                # If no path found after trying all options
                # print(f"Warning: Image {image_filename} (dataset {dataset_name}) not found in common test locations.")
                return None

            if not sample_submission_df.empty:
                print("\nConstructing full paths for test images using robust search...")
                sample_submission_df['full_path'] = sample_submission_df.apply(get_actual_test_image_path_robust, axis=1)
                sample_submission_df['image_id_combined'] = sample_submission_df['dataset'].astype(str) + "__" + sample_submission_df['image'].astype(str)
                
                test_set_df = sample_submission_df[['dataset', 'image', 'full_path', 'image_id_combined']].copy()
                print(f"Prepared 'test_set_df'. First 5 rows:")
                print(test_set_df.head())

                num_resolved_paths = test_set_df['full_path'].notna().sum()
                print(f"\nSuccessfully resolved paths for {num_resolved_paths} / {len(test_set_df)} images.")
                
                if num_resolved_paths < len(test_set_df):
                    print("WARNING: Some image paths could not be resolved. These images will be skipped.")
                    print("Example rows with missing full_path:")
                    print(test_set_df[test_set_df['full_path'].isna()].head())
                
                if num_resolved_paths > 0 :
                     ALL_IMAGES_LISTED_IN_SAMPLE_SUB_PREPARED = True
                else: # No paths resolved at all
                     print("CRITICAL WARNING: No image paths were resolved. Subsequent steps will have no data.")

            else: # sample_submission_df was empty
                print("sample_submission.csv was empty after loading. Cannot prepare test image paths.")
        
        except Exception as e:
            print(f"ERROR during data loading or path preparation in Cell 3: {e}")
            # Ensure test_set_df is empty if an error occurs
            test_set_df = pd.DataFrame(columns=['dataset', 'image', 'full_path', 'image_id_combined'])


if not ALL_IMAGES_LISTED_IN_SAMPLE_SUB_PREPARED:
    print("\nWARNING: Test set preparation was not fully successful. 'test_set_df' might be empty or incomplete.")

print("-" * 60)
print("Cell 3: Data Loading & Test Image Path Preparation complete. Review warnings carefully.")


# Cell 4: Global Feature Extraction (DINOv2) - Test Set

# Ensure necessary variables from previous cells are available
# KAGGLE_WORKING_DIR, DinoV2EmbeddingExtractor (class), test_set_df, FEATURES_DIR (optional, for output path)
if 'KAGGLE_WORKING_DIR' not in globals() or \
   'DinoV2EmbeddingExtractor' not in globals() or \
   'test_set_df' not in globals():
    
    print("CRITICAL ERROR: Prerequisite variables (KAGGLE_WORKING_DIR, DinoV2EmbeddingExtractor, or test_set_df) not defined.")
    print("                 Please ensure Cell 1 (Setup), Cell 2 (Imports/Paths), and Cell 3 (Data Loading) have run successfully.")
    # Create dummy to prevent crash, but this stage will effectively be skipped / produce no output
    test_embeddings = {} 
    test_embeddings_npz_path = "" # Will cause issues later if not properly defined
    DINO_EXTRACTION_SUCCESSFUL = False
else:
    DINO_EXTRACTION_SUCCESSFUL = True
    print("=== Stage 4: Initializing DINOv2 Extractor for Test Set ===")
    try:
        # Ensure you are using the desired model size. 's' for ViT-Small.
        # Model weights should be pre-loaded into cache from Kaggle Dataset in Cell 1.
        dino_extractor_test = DinoV2EmbeddingExtractor(model_size='s') 
    except Exception as e:
        print(f"ERROR: Could not initialize DinoV2EmbeddingExtractor: {e}")
        print("       Ensure model weights were copied to cache in Cell 1, or check model name.")
        dino_extractor_test = None 
        DINO_EXTRACTION_SUCCESSFUL = False

    test_embeddings = {}
    processed_count = 0
    error_processing_count = 0 # Renamed from error_count for clarity
    path_not_found_count = 0   # Renamed from not_found_count

    if dino_extractor_test and not test_set_df.empty:
        print(f"\nExtracting DINOv2 embeddings for {len(test_set_df)} listed test images...")
        
        for idx, row in tqdm(test_set_df.iterrows(), total=len(test_set_df), desc="Extracting Test Embeddings"):
            full_path = row['full_path']
            unique_id = row['image_id_combined'] # 'dataset__image'

            if pd.notna(full_path) and os.path.exists(full_path):
                # --- Optional initial resize for extremely large images ---
                # try:
                #     img_pil = load_image_pil(full_path) # Assumes load_image_pil is imported
                #     if img_pil:
                #         if max(img_pil.size) > 2500: # Example threshold
                #             # Assumes resize_image_maintain_aspect_ratio is imported
                #             img_pil = resize_image_maintain_aspect_ratio(img_pil, target_max_dimension=1280) 
                #         # How to pass PIL image to get_embedding? Modify get_embedding or save temp file.
                #         # For now, simplified: pass full_path. DINOv2's transform will handle typical inputs.
                #         embedding = dino_extractor_test.get_embedding(full_path) # If get_embedding takes path
                #         # Or if get_embedding was modified to take PIL: embedding = dino_extractor_test.get_embedding(img_pil)
                # except Exception as e_preprocess:
                #     print(f"Error during pre-resize for {unique_id}: {e_preprocess}")
                #     embedding = None # Fallback or skip
                # --- End Optional initial resize ---
                
                # Current approach: pass full_path, let DINOv2's transform handle sizing
                embedding = dino_extractor_test.get_embedding(full_path) 
                
                if embedding is not None:
                    test_embeddings[unique_id] = embedding
                    processed_count += 1
                else:
                    # Error typically printed by get_embedding, just count it
                    error_processing_count += 1
            else:
                # This print can be verbose. Keep it commented for submission unless debugging.
                # print(f"Warning: Test image path not found or invalid: {full_path} for ID {unique_id}")
                path_not_found_count += 1
        
        print(f"\n--- Test Embedding Extraction Summary ---")
        print(f"Successfully extracted embeddings for: {processed_count} images.")
        if path_not_found_count > 0:
            print(f"Paths not found or invalid for:     {path_not_found_count} images (expected during interactive runs with sample_submission).")
        if error_processing_count > 0:
            print(f"Errors during embedding extraction for: {error_processing_count} images.")

    elif test_set_df.empty:
        print("test_set_df is empty. No images to process for feature extraction.")
        DINO_EXTRACTION_SUCCESSFUL = False
    else: # dino_extractor_test is None
        print("DINOv2 Extractor not initialized. Skipping feature extraction.")
        DINO_EXTRACTION_SUCCESSFUL = False

    # Define output path for test embeddings using FEATURES_DIR from Cell 2
    if 'FEATURES_DIR' in globals():
        test_embeddings_npz_path = os.path.join(FEATURES_DIR, 'test_dino_embeddings_vits.npz') # Added model size
    else:
        print("WARNING: FEATURES_DIR not defined from Cell 2. Saving NPZ to KAGGLE_WORKING_DIR.")
        test_embeddings_npz_path = os.path.join(KAGGLE_WORKING_DIR, 'test_dino_embeddings_vits.npz')


    if test_embeddings: 
        print(f"\nSaving {len(test_embeddings)} test embeddings to {test_embeddings_npz_path}...")
        np.savez_compressed(test_embeddings_npz_path, **test_embeddings)
        print("Test embeddings saved successfully.")
    else:
        print("No test embeddings were extracted (or an error occurred). NPZ file not saved (or will be empty).")
        # Ensure an empty NPZ exists if subsequent cells expect it
        if not os.path.exists(test_embeddings_npz_path):
             try:
                 np.savez_compressed(test_embeddings_npz_path) # Save an empty NPZ
                 print(f"Saved an empty NPZ file at {test_embeddings_npz_path} as a placeholder.")
             except Exception as e_save_empty:
                 print(f"Could not save empty NPZ: {e_save_empty}")
                 DINO_EXTRACTION_SUCCESSFUL = False # Mark as failed if can't even save empty

# Final check
if not DINO_EXTRACTION_SUCCESSFUL and not test_embeddings : # If it failed AND no embeddings
    print("\nWARNING: DINOv2 feature extraction was not successful or produced no embeddings.")
    print("         Subsequent clustering steps may fail or produce no results.")
    # Define test_embeddings_npz_path as None or empty string if it truly failed and no placeholder was made.
    # This helps downstream cells check if they should even attempt to load it.
    if not os.path.exists(test_embeddings_npz_path):
        test_embeddings_npz_path = None 

print("-" * 60)
print("Cell 4: Global Feature Extraction (DINOv2) for Test Set complete.")


# Cell 5: Clustering (HDBSCAN) - Test Set

print("=== Stage 5: HDBSCAN Clustering on Test Set Embeddings ===")

# Ensure necessary variables and functions from previous cells/imports are available
# KAGGLE_WORKING_DIR, load_embeddings (func), run_hdbscan_clustering (func), pd (module), os (module)
# test_embeddings_npz_path (from previous cell if DINO extraction was successful)
# FEATURES_DIR (from cell 2, if using consistent output paths)

# Initialize to a known state
test_clustering_results_df = pd.DataFrame(columns=['image_id_combined', 'predicted_scene_label_raw']) 
CLUSTERING_SUCCESSFUL = False 

# Check prerequisites
prereqs_met = True
if 'KAGGLE_WORKING_DIR' not in globals():
    print("ERROR: KAGGLE_WORKING_DIR not defined.")
    prereqs_met = False
if 'load_embeddings' not in globals(): # Function from your src/clustering/hdbscan_clusterer.py
    print("ERROR: load_embeddings function not imported/defined.")
    prereqs_met = False
if 'run_hdbscan_clustering' not in globals(): # Function from your src/clustering/hdbscan_clusterer.py
    print("ERROR: run_hdbscan_clustering function not imported/defined.")
    prereqs_met = False
if 'pd' not in globals(): # Should be imported in Cell 2
    print("ERROR: pandas (pd) not imported.")
    prereqs_met = False
if 'os' not in globals(): # Should be imported in Cell 2
    print("ERROR: os module not imported.")
    prereqs_met = False
# Check if the embeddings NPZ path was set by the previous cell
if 'test_embeddings_npz_path' not in globals() or not test_embeddings_npz_path:
    print("ERROR: test_embeddings_npz_path not defined from previous DINOv2 extraction cell.")
    prereqs_met = False


if not prereqs_met:
    print("       Prerequisite variables/functions missing. Please run previous cells, especially Cell 2 (Imports/Paths) and Cell 4 (DINOv2 Extraction).")
    print("       Skipping clustering.")
else:
    # --- Define Parameters for Clustering (Based on your best experimental results) ---
    # These should match the parameters you found effective during your local/Colab experimentation.
    print("\nDefining clustering parameters...")
    # UMAP Parameters (if use_umap=True in run_hdbscan_clustering function)
    UMAP_N_NEIGHBORS = 15
    UMAP_N_COMPONENTS = 30 
    UMAP_MIN_DIST = 0.0
    UMAP_METRIC = 'cosine'    

    # HDBSCAN Parameters
    HDBSCAN_MIN_CLUSTER_SIZE = 5 # This is a key parameter to tune
    HDBSCAN_METRIC = 'euclidean' 
    HDBSCAN_MIN_SAMPLES = None   
    # HDBSCAN_CLUSTER_SELECTION_EPSILON = 0.0 # Optional advanced param

    # Path to the embeddings file generated in the previous step
    # test_embeddings_npz_path was defined in Cell 4
    
    print(f"Attempting to run HDBSCAN clustering on embeddings from: {test_embeddings_npz_path}")

    if os.path.exists(test_embeddings_npz_path):
        test_image_ids_clust, test_embeddings_matrix_clust = load_embeddings(test_embeddings_npz_path)
        
        if test_image_ids_clust is not None and test_embeddings_matrix_clust is not None and test_embeddings_matrix_clust.size > 0:
            print(f"Successfully loaded {len(test_image_ids_clust)} embeddings for clustering (Shape: {test_embeddings_matrix_clust.shape}).")
            
            test_cluster_labels = run_hdbscan_clustering(
                test_embeddings_matrix_clust,
                use_umap=True, # Your script default, confirm this is intended
                umap_n_neighbors=UMAP_N_NEIGHBORS, 
                umap_n_components=UMAP_N_COMPONENTS, 
                umap_min_dist=UMAP_MIN_DIST, 
                umap_metric=UMAP_METRIC,
                hdbscan_min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                hdbscan_metric=HDBSCAN_METRIC,
                hdbscan_min_samples=HDBSCAN_MIN_SAMPLES
            )
            
            if test_cluster_labels is not None:
                if len(test_image_ids_clust) == len(test_cluster_labels):
                    test_clustering_results_df = pd.DataFrame({
                        'image_id_combined': test_image_ids_clust, 
                        'predicted_scene_label_raw': test_cluster_labels
                    })
                    print("\nClustering complete for test set.")
                    print("First 5 rows of clustering results:")
                    print(test_clustering_results_df.head())
                    print("\nCluster label counts (HDBSCAN output; -1 indicates noise/outliers):")
                    if not test_clustering_results_df.empty:
                        print(test_clustering_results_df['predicted_scene_label_raw'].value_counts().sort_index())
                    else:
                        print("Clustering results DataFrame is empty.")
                    CLUSTERING_SUCCESSFUL = True
                else:
                    print(f"ERROR: Mismatch in length of image_ids ({len(test_image_ids_clust)}) and cluster_labels ({len(test_cluster_labels)}).")
            else:
                print("HDBSCAN clustering function returned None (likely failed internally) on test set.")
        else:
            print("Failed to load test embeddings or embeddings file was effectively empty. Skipping clustering.")
            if test_embeddings_matrix_clust is not None and test_embeddings_matrix_clust.size == 0:
                 print("   Reason: Embeddings matrix from NPZ was empty (e.g., no test images found or processed in previous DINOv2 step).")
    else:
        print(f"Test embeddings NPZ file not found at {test_embeddings_npz_path}. Skipping clustering.")

# Ensure test_clustering_results_df is defined even if clustering fails, for downstream cells
# This was already good, just re-confirming its placement.
if 'test_clustering_results_df' not in locals(): # Should have been defined as empty at the top if prereqs failed
    print("Defining test_clustering_results_df as empty due to critical earlier errors in this cell.")
    test_clustering_results_df = pd.DataFrame(columns=['image_id_combined', 'predicted_scene_label_raw'])
    CLUSTERING_SUCCESSFUL = False # Redundant if already false, but safe.


print("-" * 60)
print("Cell 5: Clustering (HDBSCAN) for Test Set complete.")
if CLUSTERING_SUCCESSFUL:
    print("Clustering was successful and results are in 'test_clustering_results_df'.")
else:
    print("Clustering was NOT successful or was skipped. 'test_clustering_results_df' will be empty or reflect no clusters.")
    print("Subsequent pipeline steps (Pair Selection, SfM) might not have meaningful input.")


# Cell 6: Image Pair Selection (within Clusters using DINOv2 Embeddings) - Test Set

print("=== Stage 6: Image Pair Selection within Clusters (Test Set) ===")

# Ensure necessary variables and functions are available
# KAGGLE_WORKING_DIR, test_clustering_results_df, select_pairs_by_embedding_similarity (func),
# pd, os, np modules should be defined/imported from previous cells.
# FEATURES_DIR (from cell 2, if using consistent output paths for NPZ)

# Initialize to a known state
all_selected_pairs_per_cluster_scene = {} 
PAIR_SELECTION_SUCCESSFUL = False

# Check prerequisites
prereqs_met_cell6 = True
if 'KAGGLE_WORKING_DIR' not in globals(): print("ERROR: KAGGLE_WORKING_DIR not defined."); prereqs_met_cell6 = False
if 'test_clustering_results_df' not in globals(): print("ERROR: test_clustering_results_df not defined."); prereqs_met_cell6 = False
if 'select_pairs_by_embedding_similarity' not in globals(): print("ERROR: select_pairs_by_embedding_similarity function not imported."); prereqs_met_cell6 = False
if 'pd' not in globals(): print("ERROR: pandas (pd) not imported."); prereqs_met_cell6 = False
if 'os' not in globals(): print("ERROR: os module not imported."); prereqs_met_cell6 = False
if 'np' not in globals(): print("ERROR: numpy (np) not imported."); prereqs_met_cell6 = False
if 'FEATURES_DIR' not in globals(): # Assuming FEATURES_DIR is where test_dino_embeddings.npz is
    print("WARNING: FEATURES_DIR not defined from Cell 2. Will try KAGGLE_WORKING_DIR for embeddings NPZ.")
    # Fallback if FEATURES_DIR isn't globally set from Cell 2, though it should be.
    # This assumes test_embeddings_npz_path was defined in Cell 4 relative to KAGGLE_WORKING_DIR if FEATURES_DIR was missing.
    if 'test_embeddings_npz_path' not in globals(): # If even that is missing
        print("ERROR: Path to embeddings NPZ also not found.")
        prereqs_met_cell6 = False

if not prereqs_met_cell6:
    print("       Prerequisite variables/functions missing for Cell 6. Please run previous cells.")
    print("       Skipping pair selection.")
else:
    print("\nStarting Image Pair Selection within clusters...")
    
    # --- Load the full DINOv2 test embeddings ---
    # These were generated and saved in Cell 4 (Global Feature Extraction)
    # Use FEATURES_DIR if defined in Cell 2, otherwise use KAGGLE_WORKING_DIR as a fallback
    # (This logic assumes test_embeddings_npz_path was correctly defined in Cell 4)
    if 'test_embeddings_npz_path' not in globals(): # Should have been defined in Cell 4
         npz_dir_base = FEATURES_DIR if 'FEATURES_DIR' in globals() else KAGGLE_WORKING_DIR
         test_embeddings_npz_path = os.path.join(npz_dir_base, 'test_dino_embeddings_vits.npz') # Reconstruct if needed

    all_test_embeddings_dict = {} 

    print(f"Attempting to load DINOv2 embeddings from: {test_embeddings_npz_path}")
    if os.path.exists(test_embeddings_npz_path):
        try:
            loaded_embeddings_data = np.load(test_embeddings_npz_path, allow_pickle=False) # allow_pickle=False for security if not needed
            if len(loaded_embeddings_data.files) == 0: # Check for empty NPZ
                print("Warning: Embeddings NPZ file is empty (no arrays found).")
            else:
                for key in loaded_embeddings_data.files:
                    all_test_embeddings_dict[key] = loaded_embeddings_data[key]
                print(f"Successfully loaded {len(all_test_embeddings_dict)} DINOv2 embeddings for pair selection.")
            loaded_embeddings_data.close()
        except Exception as e:
            print(f"Error loading embeddings NPZ {test_embeddings_npz_path}: {e}")
    else:
        print(f"ERROR: Test embeddings NPZ file not found at {test_embeddings_npz_path}. Cannot perform pair selection.")
    
    # Fallback for small clusters if pair selector is too restrictive
    MAX_IMAGES_FOR_ALL_PAIRS_FALLBACK = 10 
    PAIR_SELECTOR_TOP_K = 10 
    PAIR_SELECTOR_SIM_THRESHOLD = 0.7 

    if not test_clustering_results_df.empty and all_test_embeddings_dict:
        if 'dataset_name' not in test_clustering_results_df.columns:
            print("Adding 'dataset_name' column to test_clustering_results_df for processing.")
            test_clustering_results_df['dataset_name'] = test_clustering_results_df['image_id_combined'].apply(lambda x: x.split('__')[0])

        for dataset_name_iter in test_clustering_results_df['dataset_name'].unique():
            print(f"\nSelecting pairs for dataset: {dataset_name_iter}")
            dataset_clusters_df_iter = test_clustering_results_df[test_clustering_results_df['dataset_name'] == dataset_name_iter]
            
            for cluster_label_iter in sorted(dataset_clusters_df_iter['predicted_scene_label_raw'].unique()): # Sorted for consistent processing order
                if cluster_label_iter == -1: 
                    continue # Skip noise points for pair selection

                scene_key = f"{dataset_name_iter}__{cluster_label_iter}" 
                print(f"  Processing cluster {cluster_label_iter} (scene_key: {scene_key}) for pair selection...")
                
                image_ids_in_cluster_list = dataset_clusters_df_iter[
                    dataset_clusters_df_iter['predicted_scene_label_raw'] == cluster_label_iter
                ]['image_id_combined'].tolist()
                
                if len(image_ids_in_cluster_list) < 2:
                    print(f"    Cluster {cluster_label_iter} has < 2 images. No pairs to select.")
                    all_selected_pairs_per_cluster_scene[scene_key] = []
                    continue
                
                selected_pairs = select_pairs_by_embedding_similarity(
                    image_ids_in_cluster_list, 
                    all_test_embeddings_dict, 
                    top_k=PAIR_SELECTOR_TOP_K, 
                    similarity_threshold=PAIR_SELECTOR_SIM_THRESHOLD
                )
                
                if not selected_pairs and len(image_ids_in_cluster_list) <= MAX_IMAGES_FOR_ALL_PAIRS_FALLBACK:
                    print(f"    Pair selector found 0 pairs for cluster {cluster_label_iter} (size {len(image_ids_in_cluster_list)}).")
                    print(f"    Attempting all-pairs fallback as cluster size <= {MAX_IMAGES_FOR_ALL_PAIRS_FALLBACK}.")
                    from itertools import combinations # Import here as it's only for fallback
                    all_possible_pairs = [tuple(sorted(p)) for p in combinations(image_ids_in_cluster_list, 2)]
                    selected_pairs = list(set(all_possible_pairs)) 
                    print(f"    Using {len(selected_pairs)} pairs from all-pairs fallback.")
                elif not selected_pairs:
                    print(f"    Pair selector found 0 pairs for cluster {cluster_label_iter} (size {len(image_ids_in_cluster_list)}), and cluster too large for fallback. No pairs selected.")

                all_selected_pairs_per_cluster_scene[scene_key] = selected_pairs
                print(f"    Selected {len(selected_pairs)} pairs for cluster {cluster_label_iter}.")
        
        if all_selected_pairs_per_cluster_scene: # Check if any pairs were actually selected across all clusters
            PAIR_SELECTION_SUCCESSFUL = True
        print("\nPair selection process complete for all clusters.")
        
    elif test_clustering_results_df.empty:
        print("Clustering results (test_clustering_results_df) are empty. Skipping pair selection.")
    elif not all_test_embeddings_dict: # Check if dict is empty, implying no embeddings loaded
        print("DINOv2 test embeddings dictionary is empty. Skipping pair selection.")
    else: # Should not be reached if above conditions are met
        print("Unknown state, skipping pair selection.")


# Ensure all_selected_pairs_per_cluster_scene is defined for downstream cells
if 'all_selected_pairs_per_cluster_scene' not in locals():
    print("Defining all_selected_pairs_per_cluster_scene as empty due to critical earlier errors in this cell.")
    all_selected_pairs_per_cluster_scene = {}
    # PAIR_SELECTION_SUCCESSFUL should already be False if this path is taken

# Example: Print out some selected pairs
if PAIR_SELECTION_SUCCESSFUL and all_selected_pairs_per_cluster_scene:
    print("\n--- Example of Selected Pairs (first few scenes with pairs) ---")
    inspected_count = 0
    for scene_key_example, pairs_list_example in all_selected_pairs_per_cluster_scene.items():
        if inspected_count < 3 and pairs_list_example: 
            print(f"Scene Key: {scene_key_example}, Number of pairs: {len(pairs_list_example)}")
            print(f"  First up to 3 pairs: {pairs_list_example[:min(3, len(pairs_list_example))]}")
            inspected_count += 1
        elif inspected_count >=3:
            break
    if inspected_count == 0:
        print("No scenes with selected pairs to show as example (all pair lists might be empty).")
elif not all_selected_pairs_per_cluster_scene: # If dict is empty
     print("No pairs were selected across any clusters, or pair selection was skipped.")


print("-" * 60)
print("Cell 6: Image Pair Selection complete.")
if PAIR_SELECTION_SUCCESSFUL and any(all_selected_pairs_per_cluster_scene.values()): # Check if any list of pairs is non-empty
    print("Pair selection ran and found pairs for at least one cluster.")
else:
    print("Pair selection was NOT successful or found NO pairs for any cluster. Subsequent local matching will have no input.")


# Cell 7: Local Feature Matching & Structure from Motion (SfM)

print("=== Stage 7: Local Feature Matching & SfM ===")

# Initialize to a known state
all_image_final_poses = {} 
SFM_STAGE_ATTEMPTED = False # Flag to indicate if we tried to run the actual SfM
SFM_OVERALL_SUCCESS = False # Flag if at least one cluster produced poses

# Ensure necessary variables from previous cells are available
prereqs_met_cell7 = True
if 'all_selected_pairs_per_cluster_scene' not in globals(): 
    print("ERROR: 'all_selected_pairs_per_cluster_scene' not defined from Cell 6 (Pair Selection).")
    prereqs_met_cell7 = False
if 'test_clustering_results_df' not in globals() or test_clustering_results_df.empty: 
    print("ERROR: 'test_clustering_results_df' not defined or empty from Cell 5 (Clustering).")
    prereqs_met_cell7 = False
if 'test_set_df' not in globals() or test_set_df.empty: # Needed for path_lookup_for_sfm
    print("ERROR: 'test_set_df' not defined or empty from Cell 3 (Data Loading).")
    prereqs_met_cell7 = False
# KAGGLE_WORKING_DIR, ALIKED_WEIGHT_KAGGLE_PATH, LIGHTGLUE_WEIGHT_KAGGLE_PATH should be from Cell 1
if 'KAGGLE_WORKING_DIR' not in globals(): print("ERROR: KAGGLE_WORKING_DIR missing."); prereqs_met_cell7 = False
if 'ALIKED_WEIGHT_KAGGLE_PATH' not in globals(): print("ERROR: ALIKED_WEIGHT_KAGGLE_PATH missing."); prereqs_met_cell7 = False
# LIGHTGLUE_WEIGHT_KAGGLE_PATH might be optional if LightGlue loads with ALIKED features

# --- Try to import Davin's module ---
DAVIN_MODULE_READY = False
reconstruct_scene_cluster_func = None # Placeholder for Davin's function
PREPROCESSING_MODULE = None

if prereqs_met_cell7:
    try:
        from sfm.scene_reconstructor import reconstruct_scene_cluster # This is the target function
        from data import preprocessing as team_preprocessing_module # Our shared preprocessing
        
        reconstruct_scene_cluster_func = reconstruct_scene_cluster
        PREPROCESSING_MODULE = team_preprocessing_module
        print("Successfully imported 'reconstruct_scene_cluster' from 'sfm.scene_reconstructor' and 'preprocessing' module.")
        DAVIN_MODULE_READY = True
    except ImportError as e:
        print(f"WARNING: Could not import Davin's SfM module (sfm.scene_reconstructor): {e}")
        print("         Ensure the PR for this module is merged to the cloned branch and it defines 'reconstruct_scene_cluster'.")
        print("         Proceeding with placeholder (NaN) poses for SfM.")
    except Exception as e_import: # Catch other potential import-related errors
        print(f"An unexpected error occurred during SfM module import: {e_import}")
        print("         Proceeding with placeholder (NaN) poses for SfM.")


if not prereqs_met_cell7:
    print("Prerequisites missing for Cell 7. SfM stage will be fully skipped, assigning NaN poses by default if possible.")
    # Attempt to fill all_image_final_poses with NaNs if test_set_df exists from sample_submission
    if 'test_set_df' in globals() and not test_set_df.empty:
        print("Populating all poses with NaNs due to prerequisite failure...")
        for idx, row in test_set_df.iterrows():
            img_id_comb = row['image_id_combined']
            # Try to get cluster label if available, else default to outliers
            scene_lbl = "outliers"
            if 'test_clustering_results_df' in globals() and not test_clustering_results_df.empty and img_id_comb in test_clustering_results_df['image_id_combined'].values:
                raw_lbl = test_clustering_results_df.loc[test_clustering_results_df['image_id_combined'] == img_id_comb, 'predicted_scene_label_raw'].iloc[0]
                scene_lbl = f"cluster{int(raw_lbl)}" if raw_lbl != -1 else "outliers"
            
            all_image_final_poses[img_id_comb] = {
                'R_arr': np.full((3,3), np.nan), 'T_arr': np.full((3,), np.nan),
                'scene_label_final': scene_lbl, 'registered': False}
    SFM_STAGE_ATTEMPTED = False # Did not even attempt Davin's module
else:
    print("\nStarting Local Feature Matching & SfM Stage...")
    
    # Create path_lookup_for_sfm from test_set_df (has 'image_id_combined' and 'full_path')
    path_lookup_for_sfm = pd.Series(test_set_df.full_path.values, index=test_set_df.image_id_combined).to_dict()

    # Define COLMAP options (Raman's input, or sensible defaults)
    colmap_options = {
        "min_model_size": 3,      # Min images to make a model
        "max_num_models": 1,      # Try to get one best model
        "ba_global_max_num_iterations": 50, # Default 100, reduce for speed if needed
        # Add other pycolmap.IncrementalPipelineOptions based on Raman's research
    }
    
    # --- ALIKED/LightGlue models - These should be initialized ONCE in the notebook if passed as objects ---
    # If Davin's script initializes them internally using paths, this block is not needed here.
    # For now, assume Davin's script takes paths to weights and initializes them.
    # ALIKED_WEIGHT_KAGGLE_PATH and LIGHTGLUE_WEIGHT_KAGGLE_PATH are defined in Cell 1.

    processed_sfm_clusters_count = 0
    for scene_key, selected_pairs_for_scene_ids in tqdm(all_selected_pairs_per_cluster_scene.items(), desc="Processing Clusters for SfM"):
        dataset_name_sfm, cluster_label_raw_str_sfm = scene_key.split('__')
        cluster_label_raw_sfm = int(cluster_label_raw_str_sfm)
        
        # This loop iterates over keys from all_selected_pairs_per_cluster_scene,
        # which should NOT contain the noise cluster (-1) if pair_selector skips it.
        # If pair_selector could potentially output for -1, then add: if cluster_label_raw_sfm == -1: continue
        
        final_scene_label = f"cluster{cluster_label_raw_sfm}" # e.g., "cluster0"
        
        current_cluster_image_ids_combined = test_clustering_results_df[
            (test_clustering_results_df['dataset_name'] == dataset_name_sfm) &
            (test_clustering_results_df['predicted_scene_label_raw'] == cluster_label_raw_sfm)
        ]['image_id_combined'].tolist()

        print(f"  Processing {final_scene_label} (Dataset: {dataset_name_sfm}) with {len(current_cluster_image_ids_combined)} images, {len(selected_pairs_for_scene_ids)} selected pairs...")

        if not current_cluster_image_ids_combined or len(current_cluster_image_ids_combined) < 2:
            print(f"    Skipping SfM for {final_scene_label}: Not enough images in cluster ({len(current_cluster_image_ids_combined)}). Assigning NaN poses.")
            for img_id in current_cluster_image_ids_combined:
                all_image_final_poses[img_id] = {'R_arr': np.full((3,3), np.nan), 'T_arr': np.full((3,), np.nan), 
                                                   'scene_label_final': final_scene_label, 'registered': False}
            continue
        
        if not selected_pairs_for_scene_ids: # No pairs from selector (even after fallback if implemented there)
            print(f"    No pairs selected by pair_selector for {final_scene_label}. Assigning NaN poses.")
            for img_id in current_cluster_image_ids_combined:
                all_image_final_poses[img_id] = {'R_arr': np.full((3,3), np.nan), 'T_arr': np.full((3,), np.nan),
                                                   'scene_label_final': final_scene_label, 'registered': False}
            continue

        SFM_STAGE_ATTEMPTED = True # We are attempting SfM for at least one cluster

        if DAVIN_MODULE_READY and reconstruct_scene_cluster_func is not None:
            # Define a unique output directory for this cluster's SfM temporary files
            cluster_sfm_temp_dir = os.path.join(KAGGLE_WORKING_DIR, "sfm_temp", scene_key)
            # shutil.rmtree(cluster_sfm_temp_dir, ignore_errors=True) # Clean up previous run for this cluster
            # os.makedirs(cluster_sfm_temp_dir, exist_ok=True)

            print(f"    Calling Davin's SfM Module for {final_scene_label}...")
            try:
                # Call the imported function from Davin's refactored script
                cluster_poses_from_sfm = reconstruct_scene_cluster_func(
                    image_ids_in_cluster=current_cluster_image_ids_combined,
                    all_image_paths_lookup=path_lookup_for_sfm,
                    selected_image_pairs_ids=selected_pairs_for_scene_ids,
                    preprocessing_module=PREPROCESSING_MODULE, # Pass the imported module
                    aliked_weights_path=ALIKED_WEIGHT_KAGGLE_PATH, # Pass path to weights
                    lightglue_weights_path=LIGHTGLUE_WEIGHT_KAGGLE_PATH, # Pass path to weights
                    base_output_dir_for_sfm_run=cluster_sfm_temp_dir, # Temp dir for this cluster's SfM
                    target_aliked_input_size=1024, # This should be an agreed-upon hyperparameter
                    colmap_mapper_options_dict=colmap_options,
                    rotation_tta=True # Enable Rotation TTA by default, make it a param if needed
                )
                # cluster_poses_from_sfm is expected to be {'image_basename.png': {'R': R_array, 'T': T_array, 'registered': True/False}}

                # Merge results
                num_registered_this_cluster = 0
                for img_id_combined_original in current_cluster_image_ids_combined:
                    img_basename_key = img_id_combined_original.split('__')[-1] # Key used in Davin's output
                    pose_info = cluster_poses_from_sfm.get(img_basename_key)
                    
                    if pose_info and isinstance(pose_info, dict) and pose_info.get('registered', False):
                        all_image_final_poses[img_id_combined_original] = {
                            'R_arr': pose_info['R'], 
                            'T_arr': pose_info['T'],
                            'scene_label_final': final_scene_label,
                            'registered': True
                        }
                        num_registered_this_cluster +=1
                        SFM_OVERALL_SUCCESS = True # At least one image got a pose
                    else: 
                        all_image_final_poses[img_id_combined_original] = {
                            'R_arr': np.full((3,3), np.nan), 'T_arr': np.full((3,), np.nan),
                            'scene_label_final': final_scene_label,
                            'registered': False
                        }
                print(f"    SfM for {final_scene_label} registered {num_registered_this_cluster} images.")

            except Exception as e_sfm_call:
                print(f"    ERROR calling/running Davin's SfM module for {final_scene_label}: {e_sfm_call}")
                # Fallback: assign NaN poses for all images in this cluster
                for img_id in current_cluster_image_ids_combined:
                    all_image_final_poses[img_id] = {'R_arr': np.full((3,3), np.nan), 'T_arr': np.full((3,), np.nan), 
                                                       'scene_label_final': final_scene_label, 'registered': False}
        else: # Placeholder logic if Davin's module is not ready/imported
            # print(f"    Using placeholder: Assigning NaN poses for {final_scene_label}.")
            for img_id in current_cluster_image_ids_combined:
                all_image_final_poses[img_id] = {
                    'R_arr': np.full((3,3), np.nan), 'T_arr': np.full((3,), np.nan),
                    'scene_label_final': final_scene_label, 'registered': False
                }
        processed_sfm_clusters_count += 1

    # Handle original noise points from clustering (images that were never in a scene cluster)
    if 'test_clustering_results_df' in globals() and not test_clustering_results_df.empty:
        original_noise_ids = test_clustering_results_df[
            test_clustering_results_df['predicted_scene_label_raw'] == -1
        ]['image_id_combined'].tolist()
        for img_id_noise in original_noise_ids:
            if img_id_noise not in all_image_final_poses: 
                 all_image_final_poses[img_id_noise] = {
                    'R_arr': np.full((3,3), np.nan), 'T_arr': np.full((3,), np.nan), 
                    'scene_label_final': "outliers", 'registered': False
                }
        print(f"\nProcessed {len(original_noise_ids)} original noise points from clustering (marked as 'outliers').")
    
    print(f"\nFinished processing {processed_sfm_clusters_count} clusters for SfM.")


# Ensure all_image_final_poses is defined for the next cell, even if all above failed
if 'all_image_final_poses' not in locals():
    print("CRITICAL ERROR: all_image_final_poses was not defined after SfM stage. Fallback to empty dict.")
    all_image_final_poses = {}

print("-" * 60)
print("Cell 7: Local Feature Matching & SfM stage complete.")
if SFM_STAGE_ATTEMPTED and SFM_OVERALL_SUCCESS:
    num_actually_registered_total = sum(1 for pose_info in all_image_final_poses.values() if pose_info.get('registered', False))
    print(f"SfM processing attempted and at least one image was registered. Total images with poses: {num_actually_registered_total} / {len(all_image_final_poses)}.")
elif SFM_STAGE_ATTEMPTED: # Attempted but no image got registered
    print(f"SfM processing attempted but NO images were successfully registered. All poses will be NaN.")
elif 'all_image_final_poses' in locals() and all_image_final_poses: # Placeholder logic ran
    print(f"SfM placeholder logic ran. Pose information (currently NaNs) stored for {len(all_image_final_poses)} images.")
else:
    print("SfM stage was NOT successful or was skipped. 'all_image_final_poses' may be empty.")


# Cell 8: Submission File Generation

print("=== Stage 8: Generating Final submission.csv File ===")

SUBMISSION_CREATED_SUCCESSFULLY = False
FINAL_SUBMISSION_DF_COLUMNS = ['dataset', 'scene', 'image', 'rotation_matrix', 'translation_vector']

# KAGGLE_INPUT_DIR and KAGGLE_WORKING_DIR should be defined in Cell 2
# all_image_final_poses should be defined and populated (even if with NaNs) from Cell 7

if 'KAGGLE_INPUT_DIR' not in globals() or 'KAGGLE_WORKING_DIR' not in globals():
    print("CRITICAL ERROR: KAGGLE_INPUT_DIR or KAGGLE_WORKING_DIR not defined. Cannot proceed.")
else:
    submission_template_df_path = os.path.join(KAGGLE_INPUT_DIR, 'sample_submission.csv')
    
    if not os.path.exists(submission_template_df_path):
        print(f"CRITICAL ERROR: sample_submission.csv (template) not found at {submission_template_df_path}.")
        print("         Cannot generate submission. Notebook will likely fail scoring.")
    else:
        try:
            submission_template_df = pd.read_csv(submission_template_df_path)
            print(f"Loaded submission template with {len(submission_template_df)} required submission rows.")

            if submission_template_df.empty:
                print("ERROR: Submission template (sample_submission.csv) is empty.")
            elif 'all_image_final_poses' not in globals() or not isinstance(all_image_final_poses, dict):
                print("ERROR: 'all_image_final_poses' dictionary not found or not a dictionary from Cell 7.")
                print("       Defaulting all images to 'outliers' with NaN poses based on template.")
                # Create a default submission based on the template
                submission_template_df['scene'] = "outliers"
                submission_template_df['rotation_matrix'] = ";".join(["nan"] * 9)
                submission_template_df['translation_vector'] = ";".join(["nan"] * 3)
                final_submission_df = submission_template_df[FINAL_SUBMISSION_DF_COLUMNS].copy()
            else:
                # Proceed with populating from all_image_final_poses
                submission_template_df['image_id_combined'] = submission_template_df['dataset'].astype(str) + "__" + submission_template_df['image'].astype(str)

                output_pred_scenes = []
                output_rot_matrices_str = []
                output_trans_vectors_str = []

                nan_rotation_str = ";".join(["nan"] * 9)
                nan_translation_str = ";".join(["nan"] * 3)
                
                processed_image_ids = set(all_image_final_poses.keys())
                template_image_ids = set(submission_template_df['image_id_combined'].tolist())
                
                missing_from_poses_dict = template_image_ids - processed_image_ids
                extra_in_poses_dict = processed_image_ids - template_image_ids

                if extra_in_poses_dict:
                    print(f"WARNING: {len(extra_in_poses_dict)} image_ids found in 'all_image_final_poses' that are NOT in sample_submission.csv. These will be ignored.")
                    # print(f"  Example extra IDs: {list(extra_in_poses_dict)[:5]}")


                print(f"Populating submission data. {len(all_image_final_poses)} entries in all_image_final_poses.")
                
                for idx, template_row in tqdm(submission_template_df.iterrows(), total=len(submission_template_df), desc="Formatting Submission"):
                    img_id_comb = template_row['image_id_combined']
                    pose_data = all_image_final_poses.get(img_id_comb) 

                    if pose_data and isinstance(pose_data, dict):
                        scene_label = pose_data.get('scene_label_final', 'outliers') # Default to outliers
                        output_pred_scenes.append(scene_label)
                        
                        # Ensure 'registered' key exists, default to False if not (e.g. placeholder did not set it)
                        is_registered = pose_data.get('registered', False) 
                        
                        r_arr = pose_data.get('R_arr')
                        has_valid_r = isinstance(r_arr, np.ndarray) and r_arr.shape == (3,3) and not np.all(np.isnan(r_arr))
                        
                        t_arr = pose_data.get('T_arr')
                        has_valid_t = isinstance(t_arr, np.ndarray) and t_arr.size == 3 and not np.all(np.isnan(t_arr))

                        if is_registered and has_valid_r and has_valid_t : # Only output R,T if explicitly registered and valid
                            output_rot_matrices_str.append(";".join(map(str, r_arr.flatten(order='C'))))
                            output_trans_vectors_str.append(";".join(map(str, t_arr.flatten())))
                        else: # Not registered, or R/T invalid, or missing
                            output_rot_matrices_str.append(nan_rotation_str)
                            output_trans_vectors_str.append(nan_translation_str)
                            if scene_label != "outliers" and not (is_registered and has_valid_r and has_valid_t):
                                # If it was assigned to a cluster but has no valid pose, change to "outliers"
                                # This is based on the forum discussion to avoid 0.0 scores
                                output_pred_scenes[-1] = "outliers" 
                                # print(f"  DevInfo: Image {img_id_comb} in scene {scene_label} had no valid pose, changed to 'outliers'.")
                    else:
                        # This image from sample_submission.csv was not in all_image_final_poses at all.
                        # This indicates an issue upstream (e.g. never got an embedding or cluster label).
                        # print(f"Warning: No pose info for {img_id_comb} in all_image_final_poses. Defaulting to 'outliers' and NaN pose.")
                        output_pred_scenes.append("outliers") 
                        output_rot_matrices_str.append(nan_rotation_str)
                        output_trans_vectors_str.append(nan_translation_str)
                
                # Create the final DataFrame with the correct columns and order
                final_submission_df = pd.DataFrame({
                    'dataset': submission_template_df['dataset'],
                    'scene': output_pred_scenes,
                    'image': submission_template_df['image'],
                    'rotation_matrix': output_rot_matrices_str,
                    'translation_vector': output_trans_vectors_str
                })
                
                output_submission_path = os.path.join(KAGGLE_WORKING_DIR, 'submission.csv')
                final_submission_df.to_csv(output_submission_path, index=False)
                print(f"\nFinal submission file created successfully at: {output_submission_path}")
                print("First 5 rows of submission.csv:")
                print(final_submission_df.head())
                if not final_submission_df.empty:
                    print("\nScene distribution in final submission:")
                    print(final_submission_df['scene'].value_counts(dropna=False).sort_index())
                SUBMISSION_CREATED_SUCCESSFULLY = True

        except Exception as e:
            print(f"ERROR during submission file generation: {e}")
            # Attempt to create a fallback dummy if a major error occurred
            try:
                if 'submission_template_df' not in locals() or submission_template_df.empty: # if template itself failed to load
                    # Try to load it again for the dummy
                    if os.path.exists(submission_template_df_path):
                         submission_template_df = pd.read_csv(submission_template_df_path)
                    else: # Cannot even load template for dummy
                         print("Cannot create fallback, template submission CSV missing.")
                         raise RuntimeError("Cannot create any submission file.")


                print("Attempting to create fallback DUMMY submission due to error...")
                fallback_df = pd.DataFrame()
                fallback_df['dataset'] = submission_template_df['dataset']
                fallback_df['scene'] = 'outliers'
                fallback_df['image'] = submission_template_df['image']
                fallback_df['rotation_matrix'] = ";".join(["nan"] * 9)
                fallback_df['translation_vector'] = ";".join(["nan"] * 3)
                
                output_submission_path = os.path.join(KAGGLE_WORKING_DIR, 'submission.csv')
                fallback_df.to_csv(output_submission_path, index=False)
                print(f"CRITICAL WARNING: Created a fallback DUMMY submission.csv at {output_submission_path} due to an error during actual submission generation.")
            except Exception as e_fallback:
                print(f"Error creating even the fallback dummy submission: {e_fallback}")


print("-" * 60)
print("Cell 8: Submission File Generation complete.")
if SUBMISSION_CREATED_SUCCESSFULLY:
    print("Submission file was generated.")
else:
    print("Submission file was NOT generated successfully or a fallback dummy was created. Check errors above.")
    print("Ensure a 'submission.csv' file exists in /kaggle/working/ for Kaggle to pick up.")

