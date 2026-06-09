# CELL 1: Setup Environment, Clone Repo (with PAT), Override Paths (Updated & Complete)

# --- 1. Basic Setup & Environment ---
print("Setting up environment...")
import os
import sys
import gc
from pathlib import Path
import warnings
import json # Needed for handling kaggle.json secret
import subprocess

warnings.filterwarnings('ignore')

# --- PREPARE KAGGLE API KEY ---
# Define expected location for kaggle api key
KAGGLE_CONFIG_DIR = Path('/root/.kaggle')
KAGGLE_JSON_TARGET_PATH = KAGGLE_CONFIG_DIR / 'kaggle.json'
KAGGLE_SECRET_LABEL = "KAGGLE_KEY" # <<< MUST MATCH the secret label you created

print(f"Checking/Setting up Kaggle API key at {KAGGLE_JSON_TARGET_PATH}...")
kaggle_api_ready = False
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    kaggle_key_content = user_secrets.get_secret(KAGGLE_SECRET_LABEL)
    KAGGLE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(KAGGLE_JSON_TARGET_PATH, 'w') as f: f.write(kaggle_key_content)
    os.chmod(KAGGLE_JSON_TARGET_PATH, 0o600)
    print("Kaggle API key successfully copied from Secrets.")
    kaggle_api_ready = True
except FileNotFoundError: print(f"WARNING: Secret '{KAGGLE_SECRET_LABEL}' not found. Kaggle API features might fail.")
except Exception as e: print(f"WARNING: Could not setup Kaggle API key: {e}")
# --- END KAGGLE API KEY SETUP ---


# --- Install Dependencies ---
print("Installing specific dependency versions...")
!pip install -q --upgrade pip
!pip install -q ml-dtypes~=0.4.0
!pip install -q tensorboard~=2.17.0

print("Installing main packages...")
ACCELERATOR_TYPE = "GPU" # <<< CHANGE TO "GPU" IF GPU IS ENABLED
TF_PACKAGE = "tensorflow-cpu==2.17.1" if ACCELERATOR_TYPE == "CPU" else "tensorflow==2.17.1"
!pip install -q pyarrow joblib seaborn matplotlib tqdm scikit-learn {TF_PACKAGE}

print("Checking/Installing Kaggle CLI...")
# Corrected Kaggle CLI Installation Logic in Cell 1

# Check/Install Kaggle CLI (now safe as key should be present if using secrets)
print("Checking/Installing Kaggle CLI...")
kaggle_cli_installed = False
try:
    import kaggle
    print("Kaggle CLI already installed.")
    kaggle_cli_installed = True
except ImportError:
    print("Kaggle CLI not found, attempting installation...")
    # Run install command using ! on its own line
    !pip install -q kaggle

    # Try importing again AFTER installation attempt
    try:
        import kaggle
        print("Kaggle CLI installed and imported successfully.")
        kaggle_cli_installed = True
    except ImportError:
        print("ERROR: Failed to import Kaggle CLI even after installation attempt.")
        # Depending on whether download is needed, you might stop here
        # if SHOULD_DOWNLOAD: assert False, "Kaggle CLI required but failed to install/import"
    except Exception as e_inner:
         print(f"An error occurred importing Kaggle CLI after install: {e_inner}")

except Exception as e_outer:
    print(f"An error occurred during Kaggle CLI check: {e_outer}")

# Optional: Check if installation succeeded if it was attempted
# if not kaggle_cli_installed:
#    print("WARNING: Kaggle CLI might not be properly installed.")

# --- Continue with cloning GitHub repo ---
print("Cloning repository using PAT from Secrets...")
# ... rest of the cell ...


# --- 2. Clone Your GitHub Repo (Using PAT from Secrets) ---
print("Cloning repository using PAT from Secrets...")
# Secrets Labels used
GIT_USER_SECRET = "GITHUB_USER"
GIT_TOKEN_SECRET = "GITHUB_TOKEN"
# Your Repo Details
GIT_REPO = "march-mania-2025v3"  # <<< Your Repo Name

REPO_PATH_STR = f"/kaggle/working/{GIT_REPO}"
REPO_PATH = Path(REPO_PATH_STR)

# Get secrets for GitHub
try:
    if 'user_secrets' not in locals(): from kaggle_secrets import UserSecretsClient; user_secrets = UserSecretsClient() # Ensure client exists
    GIT_USERNAME = user_secrets.get_secret(GIT_USER_SECRET)
    GIT_TOKEN = user_secrets.get_secret(GIT_TOKEN_SECRET)
    secrets_loaded = True
except Exception as e:
    print(f"ERROR: Could not retrieve GitHub secrets ('{GIT_USER_SECRET}', '{GIT_TOKEN_SECRET}'): {e}")
    secrets_loaded = False
    assert False, "Failed to get GitHub secrets" # Stop execution

if secrets_loaded:
    # Check if the directory exists and remove it
    if REPO_PATH.exists() and REPO_PATH.is_dir():
        print(f"Removing existing repository directory: {REPO_PATH_STR}")
        rm_return_code = os.system(f"rm -rf '{REPO_PATH_STR}'")
        if rm_return_code == 0: print("Directory removed successfully.")
        else: print(f"Warning: Failed to remove directory (Code: {rm_return_code}).")

    # Construct authenticated URL
    clone_url = f"https://{GIT_TOKEN}@github.com/{GIT_USERNAME}/{GIT_REPO}.git"
    print(f"Cloning from authenticated URL into {REPO_PATH_STR}")
    clone_result = subprocess.run(["git", "clone", "--depth", "1", clone_url, REPO_PATH_STR], capture_output=True, text=True)

    if clone_result.returncode != 0:
        print(f"ERROR: Git clone failed!")
        print(f"Stderr: {clone_result.stderr}")
        assert False, "Git clone failed"
    else:
        print("Repository cloned successfully.")

    # Change current working directory
    if REPO_PATH.exists() and REPO_PATH.is_dir():
        %cd {REPO_PATH_STR}
        print(f"Changed directory to: {os.getcwd()}")
    else:
        print(f"ERROR: Repository directory {REPO_PATH_STR} not found after clone attempt.")
        assert False, "Failed to change directory after clone"
# --- End Clone ---


# --- 3. Install Specific Requirements (Optional) ---
print("Skipping requirements.txt installation.")


# --- 4. Add Source Code to Python Path ---
if Path.cwd() == REPO_PATH: # Check CWD before modifying path
    sys.path.insert(0, str(REPO_PATH))
    sys.path.insert(0, str(REPO_PATH / 'src'))
    print(f"Added to sys.path: {str(REPO_PATH)}, {str(REPO_PATH / 'src')}")
else:
    print("ERROR: Not in expected repo directory. Path not updated.")
    assert False, "Incorrect working directory"


# --- 5. Import Config and Utils ---
print("Importing config and utils...")
config_imported = False
try:
    import config
    from src.utils import logger, seed_everything
    print("Config and utils imported successfully.")
    config_imported = True
except Exception as e: print(f"ERROR during import: {e}")


# --- 6. Override Paths for Kaggle Environment ---
if config_imported:
    print("Overriding directory paths for Kaggle environment...")
    try:
        KAGGLE_INPUT_DIR = Path(f'/kaggle/input/{config.COMPETITION_NAME}')
        KAGGLE_WORKING_DIR = Path('/kaggle/working')
        if KAGGLE_INPUT_DIR.exists() and any(KAGGLE_INPUT_DIR.iterdir()): config.RAW_DATA_DIR = KAGGLE_INPUT_DIR; print(f"RAW_DATA_DIR -> Kaggle Input: {config.RAW_DATA_DIR}"); SHOULD_DOWNLOAD = False
        else: config.RAW_DATA_DIR = KAGGLE_WORKING_DIR / "data" / "raw"; print(f"RAW_DATA_DIR -> Working Dir: {config.RAW_DATA_DIR}"); config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True); SHOULD_DOWNLOAD = True
        config.PROCESSED_DATA_DIR = KAGGLE_WORKING_DIR / "processed"; config.MODELS_DIR = KAGGLE_WORKING_DIR / "models"; config.SUBMISSIONS_DIR = KAGGLE_WORKING_DIR / "submissions"; config.VIZ_PATH = KAGGLE_WORKING_DIR / "visualizations"
        config.DATA_CACHE_FILE = config.PROCESSED_DATA_DIR / "data_cache.pkl"; config.TEAM_STATS_FILE = config.PROCESSED_DATA_DIR / "team_stats_per_season.parquet"; config.TRAIN_DATA_FILE = config.PROCESSED_DATA_DIR / "training_matchups.parquet"; config.TEST_DATA_FILE = config.PROCESSED_DATA_DIR / f"{config.CURRENT_SEASON}_prediction_matchups.parquet"
        kaggle_sub_base = Path(config.FINAL_SUBMISSION_FILE).stem.replace('_tf_v1','') + "_kaggle" # Make name more distinct
        config.FINAL_SUBMISSION_FILE = kaggle_sub_base + ".csv"; config.OOF_PREDS_FILE = config.PROCESSED_DATA_DIR / f"oof_{kaggle_sub_base}_predictions.csv"; config.SCALER_FILE = config.MODELS_DIR / f"scaler_{kaggle_sub_base}.joblib"
        print("Creating Kaggle output directories..."); config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True); config.MODELS_DIR.mkdir(parents=True, exist_ok=True); config.SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True); config.VIZ_PATH.mkdir(parents=True, exist_ok=True)
        seed_everything(); logger.info("===== Kaggle Notebook Setup Complete ====="); print(f"Final Submission Filename: {config.FINAL_SUBMISSION_FILE}")
    except AttributeError as e: print(f"ERROR: Missing variable in config: {e}"); config_imported = False
    except Exception as e: print(f"ERROR during path override: {e}"); config_imported = False
else: print("Skipping path overriding due to import errors.")

gc.collect()
print("-" * 50)
if config_imported: print("Setup finished. Proceed to the next cell to load data.")
else: print("Setup failed. Please check errors and configuration.")
print("-" * 50)


# === Stage 1: Load Raw Data ===
import time
if config_imported: # Only run if setup was okay
    from src.data_loader import load_raw_data
    from src.utils import logger

    logger.info("\n===== Stage 1: Load Raw Data (Kaggle) =====")
    stage_start = time.time()
    # Use reload=True first time on Kaggle maybe, then False for cache
    # download_if_missing should be True if RAW_DATA_DIR is in /kaggle/working
    should_download = "/kaggle/working" in str(config.RAW_DATA_DIR)
    raw_data = load_raw_data(reload=False, download_if_missing=should_download, use_cache=True)
    if not raw_data:
        logger.error("STOPPING: Failed to load raw data.")
        # Stop execution if data loading fails
        assert False, "Failed to load raw data."
    else:
        logger.info(f"Stage 1 completed in {time.time() - stage_start:.2f} seconds.")
        print(f"Loaded {len(raw_data)} datasets.")
        gc.collect()
else:
    print("Skipping Stage 1 due to setup errors.")


# === Stage 2: Feature Engineering ===
import time
import pandas as pd
if config_imported and 'raw_data' in locals() and raw_data: # Check dependencies
    from src.features import create_all_features, create_training_matchups, create_prediction_matchups
    from src.utils import logger

    logger.info("\n===== Stage 2: Feature Engineering (Kaggle) =====")
    stage_start = time.time()

    # Create team stats (saved to config.TEAM_STATS_FILE)
    team_stats = create_all_features(raw_data)
    if team_stats.empty:
        logger.error("STOPPING: Failed to create team stats.")
        assert False, "Failed to create team stats."

    # Create training data (saved to config.TRAIN_DATA_FILE)
    train_data, train_features = create_training_matchups(raw_data, team_stats)
    if train_data.empty:
         logger.error("STOPPING: Failed to create training data.")
         assert False, "Failed to create training data."

    # Create prediction structure (saved to config.TEST_DATA_FILE)
    team_stats_for_pred = team_stats.copy()
    if config.CURRENT_SEASON not in team_stats['Season'].unique():
         latest_season = team_stats['Season'].max()
         logger.warning(f"Using season {latest_season} stats as proxy for {config.CURRENT_SEASON}.")
         team_stats_for_pred = team_stats[team_stats['Season']==latest_season].copy()
         if not team_stats_for_pred.empty: team_stats_for_pred['Season'] = config.CURRENT_SEASON
         else: logger.error("No proxy stats found!"); team_stats_for_pred = pd.DataFrame()

    if team_stats_for_pred.empty:
         logger.error("STOPPING: No stats available for prediction structure.")
         assert False, "No stats for prediction structure."

    pred_data, pred_features = create_prediction_matchups(team_stats_for_pred, raw_data)
    if pred_data.empty:
         logger.error("STOPPING: Failed to create prediction data structure.")
         assert False, "Failed to create prediction data structure."

    # Feature consistency check
    if set(train_features) != set(pred_features):
        logger.error("FATAL: Mismatch between training and prediction features!")
        assert False, "Feature mismatch detected."

    logger.info(f"Stage 2 completed in {time.time() - stage_start:.2f} seconds.")
    del team_stats, train_data, pred_data # Clean up large dataframes if possible
    gc.collect()
else:
    print("Skipping Stage 2 due to setup errors or missing data.")


# === Stage 3: Pre-Training Visualization (Kaggle) ===
import time
import pandas as pd
import matplotlib.pyplot as plt # Ensure matplotlib is imported
import seaborn as sns # Ensure seaborn is imported
if config_imported: # Check dependencies
    from src.visualize import plot_feature_distributions, plot_correlation_matrix # Import plotting functions
    from src.utils import logger
    # Define the training data file path using the (potentially overridden) config
    train_data_path = config.TRAIN_DATA_FILE

    logger.info("\n===== Stage 3: Pre-Training Visualization (Kaggle) =====")
    stage_start = time.time()

    # Check if train_data and train_features exist from previous cell
    # Otherwise, load the training data file
    if 'train_data' not in locals() or 'train_features' not in locals() or train_data.empty or not train_features:
        logger.info(f"Loading training data from {train_data_path} for visualization...")
        if train_data_path.exists():
            try:
                train_data = pd.read_parquet(train_data_path)
                # Re-identify feature names if loading from file
                train_features = sorted([col for col in train_data.columns if col.endswith('_Diff')])
                if not train_features:
                    logger.error("Loaded training data but found no '*_Diff' feature columns.")
                    train_data = pd.DataFrame() # Mark as empty if features missing
            except Exception as e:
                logger.error(f"Failed to load training data from {train_data_path}: {e}")
                train_data = pd.DataFrame() # Mark as empty on error
        else:
            logger.error(f"Training data file not found at {train_data_path}. Cannot generate pre-training plots.")
            train_data = pd.DataFrame() # Mark as empty if file missing

    # Proceed only if we have data and features
    if not train_data.empty and train_features:
        try:
            logger.info("Generating feature distribution plot...")
            # Pass the specific save path within the Kaggle working directory
            plot_feature_distributions(
                train_data,
                train_features,
                save_path=config.VIZ_PATH / "feature_distributions.png"
            )
            print(f"Distribution plot saved to {config.VIZ_PATH / 'feature_distributions.png'}")

            logger.info("Generating feature correlation matrix...")
            plot_correlation_matrix(
                train_data,
                train_features,
                save_path=config.VIZ_PATH / "correlation_matrix.png"
            )
            print(f"Correlation plot saved to {config.VIZ_PATH / 'correlation_matrix.png'}")

            # --- Display Plots Directly in Notebook Output ---
            print("\n--- Pre-Training Feature Plots ---")
            from IPython.display import Image, display

            dist_img_path = config.VIZ_PATH / "feature_distributions.png"
            if dist_img_path.exists():
                print("Feature Distributions:")
                display(Image(filename=str(dist_img_path), width=800)) # Adjust width if needed
            else:
                print("Distribution plot image not found.")

            corr_img_path = config.VIZ_PATH / "correlation_matrix.png"
            if corr_img_path.exists():
                print("\nFeature Correlation Matrix:")
                display(Image(filename=str(corr_img_path), width=800)) # Adjust width if needed
            else:
                print("Correlation plot image not found.")
            # ----------------------------------------------------

        except Exception as e:
            logger.error(f"Error during pre-training visualization: {e}", exc_info=True)
    else:
        logger.warning("Skipping pre-training visualization as training data or features are unavailable.")


    logger.info(f"Stage 3 completed in {time.time() - stage_start:.2f} seconds.")
else:
    print("Skipping Stage 3 due to setup errors.")


# === Stage 4: Model Training ===
import time
if config_imported and 'train_features' in locals(): # Check dependencies
    from src.models import train_evaluate_tf
    from src.utils import logger

    logger.info("\n===== Stage 4: Model Training (Kaggle) =====")
    stage_start = time.time()
    # train_evaluate_tf uses config variables for paths/params by default
    oof_preds, scaler, features_from_train = train_evaluate_tf()
    if oof_preds is None:
        logger.error("STOPPING: Model training failed.")
        assert False, "Model training failed."
    else:
        logger.info(f"Stage 4 completed in {time.time() - stage_start:.2f} seconds.")
        # Keep scaler and features_from_train for prediction stage
        gc.collect()
else:
    print("Skipping Stage 4 due to setup errors or missing features.")


# === Stage 5: Post-Training Visualization (Kaggle) ===
import time
import pandas as pd
import matplotlib.pyplot as plt # Ensure matplotlib is imported
import seaborn as sns # Ensure seaborn is imported
if config_imported: # Check dependencies
    from src.visualize import plot_oof_calibration, plot_oof_distribution # Import plotting functions
    from src.utils import logger
    # Define the OOF file path using the (potentially overridden) config
    oof_file_path = config.OOF_PREDS_FILE

    logger.info("\n===== Stage 5: Post-Training Visualization (Kaggle) =====")
    stage_start = time.time()

    if oof_file_path.exists():
        try:
            logger.info(f"Loading OOF predictions from: {oof_file_path}")
            oof_df = pd.read_csv(oof_file_path)

            if not oof_df.empty:
                logger.info("Generating OOF calibration plot...")
                # Pass the specific save path within the Kaggle working directory
                plot_oof_calibration(oof_df, save_path=config.VIZ_PATH / "oof_calibration_curve.png")
                print(f"Calibration plot saved to {config.VIZ_PATH / 'oof_calibration_curve.png'}")

                logger.info("Generating OOF prediction distribution plot...")
                plot_oof_distribution(oof_df, save_path=config.VIZ_PATH / "oof_prediction_distribution.png")
                print(f"Distribution plot saved to {config.VIZ_PATH / 'oof_prediction_distribution.png'}")

                # --- Display Plots Directly in Notebook Output ---
                print("\n--- OOF Plots ---")
                from IPython.display import Image, display

                cal_img_path = config.VIZ_PATH / "oof_calibration_curve.png"
                if cal_img_path.exists():
                    print("OOF Calibration Curve:")
                    display(Image(filename=str(cal_img_path))) # Display the saved image
                else:
                    print("Calibration plot image not found.")

                dist_img_path = config.VIZ_PATH / "oof_prediction_distribution.png"
                if dist_img_path.exists():
                    print("\nOOF Prediction Distribution:")
                    display(Image(filename=str(dist_img_path))) # Display the saved image
                else:
                    print("Distribution plot image not found.")
                # ----------------------------------------------------

            else:
                logger.warning("OOF predictions file is empty, skipping plots.")

        except FileNotFoundError:
            logger.warning(f"OOF predictions file not found at {oof_file_path}, skipping OOF visualization.")
        except Exception as e:
            logger.error(f"Error during post-training visualization: {e}", exc_info=True)
    else:
        logger.warning(f"OOF predictions file not found at {oof_file_path}, skipping OOF visualization.")

    logger.info(f"Stage 5 completed in {time.time() - stage_start:.2f} seconds.")
else:
    print("Skipping Stage 5 due to setup errors.")


# === Stage 6: Generate Predictions ===
import time
if config_imported and 'features_from_train' in locals(): # Check dependencies
    from src.submit import generate_tf_predictions
    from src.utils import logger

    logger.info("\n===== Stage 6: Generate Final Predictions (Kaggle) =====")
    stage_start = time.time()
    # generate_tf_predictions uses config variables for paths/params by default
    submission_df = generate_tf_predictions()
    if submission_df is None:
        logger.error("STOPPING: Prediction generation failed.")
        assert False, "Prediction generation failed."
    else:
        logger.info(f"Stage 6 completed in {time.time() - stage_start:.2f} seconds.")
        print("\n--- Submission File Head (Kaggle Run) ---")
        print(submission_df.head())
        print(f"Submission file shape: {submission_df.shape}")
        print(f"Submission file saved to: {config.SUBMISSIONS_DIR / config.FINAL_SUBMISSION_FILE}")
else:
    print("Skipping Stage 6 due to setup errors or incomplete training.")

