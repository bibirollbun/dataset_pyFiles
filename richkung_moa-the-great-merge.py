# !pip install numpy pandas scikit-learn matplotlib joblib 
# !pip install xgboost 
# !pip install iterative-stratification
# !pip install notebook


!python --version



# !pip install numpy pandas matplotlib joblib scikit-learn
# !pip install xgboost 
# !pip install iterative-stratification
# !pip install iterative-stratification
# !pip install notebook
# !pip install catboost
# !pip install lightgbm
# !pip install tensorflow


# # %% [code]
# # ğŸŒŸ MoA XGBoost Pipeline (15000 Sample + Feature Engineering + CV + Loss Plot)
# # Optimized for Kaggle environment
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import joblib
# import time
# import datetime
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from xgboost import XGBClassifier
# from xgboost import __version__ as xgb_version
# from tqdm.auto import tqdm  # For progress tracking


# # For balanced multilabel CV
# from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# # Parameters
# USE_PRETRAINED_MODELS = True  # Set to True if you have pre-trained models
# OUTPUT_SUBMISSION_PATH = "/kaggle/working/submission_xgb.csv"  # Output path within the Kaggle environment

# # Check for GPU availability on Kaggle
# import subprocess
# try:
#     gpu_info = subprocess.check_output('nvidia-smi', shell=True).decode('utf-8')
#     gpu_available = True
#     print("GPU is available on this Kaggle kernel!")
#     print(gpu_info)
# except:
#     gpu_available = False
#     print("No GPU available, will use CPU mode.")

# # 1. Load data - Kaggle paths
# print("Loading datasets...")
# train_features = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")
# train_targets_scored = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")
# train_targets_nonscored = pd.read_csv("/kaggle/input/lish-moa/train_targets_nonscored.csv")
# test_features = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")
# sample_submission = pd.read_csv("/kaggle/input/lish-moa/sample_submission.csv")


# # 2. Sample 15000 training records
# print("Sampling 15000 records for training...")
# df_sample = train_features.sample(n=15000, random_state=42).reset_index(drop=True)
# ids = df_sample['sig_id']
# Y = train_targets_scored[train_targets_scored['sig_id'].isin(ids)].reset_index(drop=True)

# df_valid = train_features[~train_features['sig_id'].isin(ids)].reset_index(drop=True)
# Y_valid = train_targets_scored[train_targets_scored['sig_id'].isin(df_valid['sig_id'])].reset_index(drop=True)


# # 3. Feature engineering
# def preprocess_features(df):
#     df = df.drop(columns=["sig_id"])
#     df = pd.get_dummies(df, columns=["cp_type", "cp_dose", "cp_time"])
#     return df

# print("Preprocessing features...")
# X = preprocess_features(df_sample)
# X_test = preprocess_features(test_features)
# X_valid = preprocess_features(df_valid)

# # Feature scaling
# scaler = StandardScaler()
# X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
# X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
# X_valid = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns)


# # 4. Set up Cross Validation
# N_SPLITS = 3
# mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, random_state=42, shuffle=True)

# models = {}
# histories = {}

# # Create directories for model storage - Kaggle paths
# model_path = "/kaggle/input/moa-all-models/weights/output_param/xgb_models/param"
# # os.makedirs(model_path, exist_ok=True)
# # os.makedirs(f"{model_path}/param", exist_ok=True)  # For model parameters
# # os.makedirs(f"{model_path}/lossplot", exist_ok=True)  # For loss plots

# # Set GPU count based on Kaggle environment (typically has 1 GPU)
# num_gpus = 1 if gpu_available else 0
# if gpu_available:
#     print(f"ğŸš€ Starting training with {num_gpus} GPU acceleration")
# else:
#     print("ğŸš€ Starting training with CPU")

# start_time = time.time()

# # Check XGBoost version
# print(f"XGBoost version: {xgb_version}")

# # Calculate total tasks for progress estimation
# target_columns = Y.columns[1:]  # Skip 'sig_id'
# total_targets = len(target_columns)
# print(f"Total targets to train: {total_targets}")

# # Track training times
# target_times = []


# # 5. Train models or load pre-trained models
# if USE_PRETRAINED_MODELS:
#     print("ğŸ”„ Using pre-trained models, loading from saved files...")
    
#     # Check if model files exist
#     model_exists = all(os.path.exists(f"{model_path}/xgb_{target}.pkl") for target in target_columns)
    
#     if not model_exists:
#         print("â�Œ Error: Not all model files found. Set USE_PRETRAINED_MODELS=False to retrain")
#         exit(1)
    
#     # Load pre-trained models
#     for i, target in tqdm(enumerate(target_columns), total=total_targets, desc='Loading models'):
#         model_file = f"{model_path}/xgb_{target}.pkl"
#         models[target] = joblib.load(model_file)
#         print(f"Model loaded: {target}")
    
#     print("âœ… All models loaded successfully")
# else:
#     # Normal training flow
#     for i, target in tqdm(enumerate(target_columns), total=total_targets, desc='Overall Progress'):
#         target_start = time.time()
        
#         print(f"\nTraining model ({i+1}/{total_targets}): {target}")
#         y_target = Y[target].values
        
#         # Use train_test_split for validation split
#         indices = range(len(X))
        
#         # Check class distribution to ensure at least 2 samples per class
#         unique_values, counts = np.unique(y_target, return_counts=True)
#         min_count = counts.min() if len(counts) > 0 else 0
        
#         # Skip stratified sampling if any class has too few samples
#         if min_count < 2 or len(unique_values) <= 1:
#             print(f"Warning: Imbalanced data for target {target}, skipping stratification")
#             stratify_data = None
#         else:
#             stratify_data = y_target
            
#         train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=stratify_data)
        
#         X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
#         y_train, y_val = y_target[train_indices], y_target[val_indices]

#         # Check for class imbalance
#         pos_rate = np.mean(y_train)
#         print(f"Positive rate: {pos_rate:.4f}")
        
#         # Adjust weight for imbalanced data
#         scale_pos_weight = 1
#         if pos_rate < 0.2:
#             scale_pos_weight = (1 - pos_rate) / pos_rate
#             print(f"Imbalanced data, adjusting weight to: {scale_pos_weight:.2f}")

#         # Current GPU selection (Kaggle typically has 1 GPU)
#         current_gpu = 0 if gpu_available else None
        
#         try:
#             # Try GPU acceleration if available
#             if gpu_available:
#                 model = XGBClassifier(
#                     n_estimators=200,
#                     learning_rate=0.05,
#                     max_depth=6,
#                     subsample=0.8,
#                     colsample_bytree=0.8,
#                     eval_metric='logloss',
#                     random_state=42,
#                     verbosity=1,
#                     tree_method="gpu_hist", # Use gpu_hist for Kaggle GPU
#                     scale_pos_weight=scale_pos_weight
#                 )
                
#                 print(f"Training model using GPU...")
#             else:
#                 # CPU configuration
#                 model = XGBClassifier(
#                     n_estimators=200,
#                     learning_rate=0.05,
#                     max_depth=6,
#                     subsample=0.8,
#                     colsample_bytree=0.8,
#                     eval_metric='logloss',
#                     random_state=42,
#                     verbosity=1,
#                     tree_method="hist",
#                     scale_pos_weight=scale_pos_weight
#                 )
                
#                 print("Training model using CPU...")
                
#             # Fit model
#             model.fit(
#                 X_train, y_train,
#                 eval_set=[(X_train, y_train), (X_val, y_val)],
#                 verbose=False  # Don't output loss in terminal
#             )
#         except Exception as e:
#             print(f"Training failed, falling back to CPU: {str(e)}")
#             # Fallback to CPU
#             model = XGBClassifier(
#                 n_estimators=200,
#                 learning_rate=0.05,
#                 max_depth=6,
#                 subsample=0.8,
#                 colsample_bytree=0.8,
#                 eval_metric='logloss',
#                 random_state=42,
#                 verbosity=0,
#                 tree_method="hist",
#                 device="cpu",
#                 scale_pos_weight=scale_pos_weight
#             )
            
#             model.fit(
#                 X_train, y_train,
#                 eval_set=[(X_train, y_train), (X_val, y_val)],
#                 verbose=False
#             )

#         # Create loss plot
#         try:
#             results = model.evals_result()
#             histories[target] = results

#             plt.figure(figsize=(10, 6))
#             plt.plot(results["validation_0"]["logloss"], label="Train")
#             plt.plot(results["validation_1"]["logloss"], label="Valid")
#             plt.title(f"Logloss Curve for {target}")
#             plt.xlabel("Iterations")
#             plt.ylabel("Logloss")
#             plt.legend()
#             plt.grid()
#             plt.tight_layout()
#             plt.savefig(f"{model_path}/lossplot/loss_{target}.png")
#             plt.close()
#         except Exception as e:
#             print(f"Failed to plot loss curve: {str(e)}")

#         # Save model
#         models[target] = model
#         joblib.dump(model, f"{model_path}/param/xgb_{target}.pkl")
        
#         # Calculate and display progress
#         target_time = time.time() - target_start
#         target_times.append(target_time)
#         avg_time_per_target = np.mean(target_times)
#         remaining_targets = total_targets - (i + 1)
#         estimated_remaining_time = avg_time_per_target * remaining_targets
        
#         # Format estimated time remaining
#         remaining_time_str = str(datetime.timedelta(seconds=int(estimated_remaining_time)))
#         completion_time = datetime.datetime.now() + datetime.timedelta(seconds=estimated_remaining_time)
        
#         print(f"Target {i+1}/{total_targets} completed! ({target})")
#         print(f"Average time per target: {avg_time_per_target:.2f} seconds")
#         print(f"Estimated time remaining: {remaining_time_str}")
#         print(f"Estimated completion time: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")


# # Final completion message
# print("\nâœ… All models training completed!")
# total_time = time.time() - start_time
# print(f"Total time: {datetime.timedelta(seconds=int(total_time))}")

# # 6. Make predictions and create submission file
# print("ğŸ”® Starting prediction...")
# predictions = []

# # Show prediction progress
# for i, target in tqdm(enumerate(target_columns), total=len(target_columns), desc='Prediction progress'):
#     model = models[target]  # Use in-memory models to avoid reloading
#     pred = model.predict_proba(X_test)[:, 1]
#     predictions.append(pred)

# predictions = np.array(predictions).T
# submission = sample_submission.copy()
# submission.iloc[:, 1:] = predictions
# submission.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
# print(f"ğŸ�‰ Submission file created: {OUTPUT_SUBMISSION_PATH}")


# predictions_V = []

# # Show prediction progress
# for i, target in tqdm(enumerate(target_columns), total=len(target_columns), desc='Validation prediction progress'):
#     model = models[target]  # Use in-memory models to avoid reloading
#     pred = model.predict_proba(X_valid)[:, 1]
#     predictions_V.append(pred)

# predictions_V = np.array(predictions_V).T

# # Create validation submission with correct sig_ids
# validation_submission = pd.DataFrame(columns=Y_valid.columns)
# validation_submission['sig_id'] = df_valid['sig_id']
# for col in target_columns:
#     validation_submission[col] = predictions_V[:, list(target_columns).index(col)]

# # Save validation predictions to a separate file
# VALIDATION_OUTPUT_PATH = "/kaggle/working/validation_predictions_xgb.csv"
# validation_submission.to_csv(VALIDATION_OUTPUT_PATH, index=False)
# print(f"ğŸ�‰ é©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆå·²å»ºç«‹: {VALIDATION_OUTPUT_PATH}")

# # æª¢è¦–é©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆçš„å‰�å¹¾è¡Œ
# print("\né©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆé �è¦½:")


# ğŸŒŸ MoA XGBoost Pipeline (15000 Sample + Feature Engineering + CV + Loss Plot)
# Optimized for Kaggle environment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from xgboost import __version__ as xgb_version
from tqdm.auto import tqdm  # For progress tracking


# For balanced multilabel CV
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# Parameters
USE_PRETRAINED_MODELS = True  # Set to True if you have pre-trained models
OUTPUT_SUBMISSION_PATH = "/kaggle/working/submission_xgb.csv"  # Output path within the Kaggle environment

# Check for GPU availability on Kaggle
import subprocess
try:
    gpu_info = subprocess.check_output('nvidia-smi', shell=True).decode('utf-8')
    gpu_available = True
    print("GPU is available on this Kaggle kernel!")
    print(gpu_info)
except:
    gpu_available = False
    print("No GPU available, will use CPU mode.")

# 1. Load data - Kaggle paths
print("Loading datasets...")
train_features = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")
train_targets_scored = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")
train_targets_nonscored = pd.read_csv("/kaggle/input/lish-moa/train_targets_nonscored.csv")
test_features = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")
sample_submission = pd.read_csv("/kaggle/input/lish-moa/sample_submission.csv")

# 2. Sample 15000 training records
print("Sampling 15000 records for training...")
df_sample = train_features.sample(n=15000, random_state=42).reset_index(drop=True)
ids = df_sample['sig_id']
Y = train_targets_scored[train_targets_scored['sig_id'].isin(ids)].reset_index(drop=True)

df_valid = train_features[~train_features['sig_id'].isin(ids)].reset_index(drop=True)
Y_valid = train_targets_scored[train_targets_scored['sig_id'].isin(df_valid['sig_id'])].reset_index(drop=True)

# 3. Feature engineering
def preprocess_features(df):
    df = df.drop(columns=["sig_id"])
    df = pd.get_dummies(df, columns=["cp_type", "cp_dose", "cp_time"])
    return df

print("Preprocessing features...")
X = preprocess_features(df_sample)
X_test = preprocess_features(test_features)
X_valid = preprocess_features(df_valid)

# Feature scaling
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
X_valid = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns)

# 4. Set up Cross Validation
N_SPLITS = 3
mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, random_state=42, shuffle=True)

models = {}
histories = {}

# Create directories for model storage - Kaggle paths
model_path = "/kaggle/input/moa-all-models/weights/output_param/xgb_models/param"
# os.makedirs(model_path, exist_ok=True)
# os.makedirs(f"{model_path}/param", exist_ok=True)  # For model parameters
# os.makedirs(f"{model_path}/lossplot", exist_ok=True)  # For loss plots

# Set GPU count based on Kaggle environment (typically has 1 GPU)
num_gpus = 1 if gpu_available else 0
if gpu_available:
    print(f"ğŸš€ Starting training with {num_gpus} GPU acceleration")
else:
    print("ğŸš€ Starting training with CPU")

start_time = time.time()

# Check XGBoost version
print(f"XGBoost version: {xgb_version}")

# Calculate total tasks for progress estimation
target_columns = Y.columns[1:]  # Skip 'sig_id'
total_targets = len(target_columns)
print(f"Total targets to train: {total_targets}")

# Track training times
target_times = []

# 5. Train models or load pre-trained models
if USE_PRETRAINED_MODELS:
    print("ğŸ”„ Using pre-trained models, loading from saved files...")
    
    # Check if model files exist
    model_exists = all(os.path.exists(f"{model_path}/xgb_{target}.pkl") for target in target_columns)
    
    if not model_exists:
        print("â�Œ Error: Not all model files found. Set USE_PRETRAINED_MODELS=False to retrain")
        exit(1)
    
    # Load pre-trained models
    for i, target in tqdm(enumerate(target_columns), total=total_targets, desc='Loading models'):
        model_file = f"{model_path}/xgb_{target}.pkl"
        models[target] = joblib.load(model_file)
        print(f"Model loaded: {target}")
    
    print("âœ… All models loaded successfully")
else:
    # Normal training flow
    for i, target in tqdm(enumerate(target_columns), total=total_targets, desc='Overall Progress'):
        target_start = time.time()
        
        print(f"\nTraining model ({i+1}/{total_targets}): {target}")
        y_target = Y[target].values
        
        # Use train_test_split for validation split
        indices = range(len(X))
        
        # Check class distribution to ensure at least 2 samples per class
        unique_values, counts = np.unique(y_target, return_counts=True)
        min_count = counts.min() if len(counts) > 0 else 0
        
        # Skip stratified sampling if any class has too few samples
        if min_count < 2 or len(unique_values) <= 1:
            print(f"Warning: Imbalanced data for target {target}, skipping stratification")
            stratify_data = None
        else:
            stratify_data = y_target
            
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=stratify_data)
        
        X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
        y_train, y_val = y_target[train_indices], y_target[val_indices]

        # Check for class imbalance
        pos_rate = np.mean(y_train)
        print(f"Positive rate: {pos_rate:.4f}")
        
        # Adjust weight for imbalanced data
        scale_pos_weight = 1
        if pos_rate < 0.2:
            scale_pos_weight = (1 - pos_rate) / pos_rate
            print(f"Imbalanced data, adjusting weight to: {scale_pos_weight:.2f}")

        # Current GPU selection (Kaggle typically has 1 GPU)
        current_gpu = 0 if gpu_available else None
        
        try:
            # Try GPU acceleration if available
            if gpu_available:
                model = XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric='logloss',
                    random_state=42,
                    verbosity=1,
                    tree_method="gpu_hist", # Use gpu_hist for Kaggle GPU
                    scale_pos_weight=scale_pos_weight
                )
                
                print(f"Training model using GPU...")
            else:
                # CPU configuration
                model = XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric='logloss',
                    random_state=42,
                    verbosity=1,
                    tree_method="hist",
                    scale_pos_weight=scale_pos_weight
                )
                
                print("Training model using CPU...")
                
            # Fit model
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False  # Don't output loss in terminal
            )
        except Exception as e:
            print(f"Training failed, falling back to CPU: {str(e)}")
            # Fallback to CPU
            model = XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss',
                random_state=42,
                verbosity=0,
                tree_method="hist",
                device="cpu",
                scale_pos_weight=scale_pos_weight
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )

        # Create loss plot
        try:
            results = model.evals_result()
            histories[target] = results

            plt.figure(figsize=(10, 6))
            plt.plot(results["validation_0"]["logloss"], label="Train")
            plt.plot(results["validation_1"]["logloss"], label="Valid")
            plt.title(f"Logloss Curve for {target}")
            plt.xlabel("Iterations")
            plt.ylabel("Logloss")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(f"{model_path}/lossplot/loss_{target}.png")
            plt.close()
        except Exception as e:
            print(f"Failed to plot loss curve: {str(e)}")

        # Save model
        models[target] = model
        joblib.dump(model, f"{model_path}/param/xgb_{target}.pkl")
        
        # Calculate and display progress
        target_time = time.time() - target_start
        target_times.append(target_time)
        avg_time_per_target = np.mean(target_times)
        remaining_targets = total_targets - (i + 1)
        estimated_remaining_time = avg_time_per_target * remaining_targets
        
        # Format estimated time remaining
        remaining_time_str = str(datetime.timedelta(seconds=int(estimated_remaining_time)))
        completion_time = datetime.datetime.now() + datetime.timedelta(seconds=estimated_remaining_time)
        
        print(f"Target {i+1}/{total_targets} completed! ({target})")
        print(f"Average time per target: {avg_time_per_target:.2f} seconds")
        print(f"Estimated time remaining: {remaining_time_str}")
        print(f"Estimated completion time: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Final completion message
print("\nâœ… All models training completed!")
total_time = time.time() - start_time
print(f"Total time: {datetime.timedelta(seconds=int(total_time))}")

# 6. Make predictions and create submission file
print("ğŸ”® Starting prediction...")
predictions = []

# Show prediction progress
for i, target in tqdm(enumerate(target_columns), total=len(target_columns), desc='Prediction progress'):
    model = models[target]  # Use in-memory models to avoid reloading
    pred = model.predict_proba(X_test)[:, 1]
    predictions.append(pred)

predictions = np.array(predictions).T
submission = sample_submission.copy()
submission.iloc[:, 1:] = predictions
submission.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
print(f"ğŸ�‰ Submission file created: {OUTPUT_SUBMISSION_PATH}")

predictions_V = []

# Show prediction progress
for i, target in tqdm(enumerate(target_columns), total=len(target_columns), desc='Validation prediction progress'):
    model = models[target]  # Use in-memory models to avoid reloading
    pred = model.predict_proba(X_valid)[:, 1]
    predictions_V.append(pred)

predictions_V = np.array(predictions_V).T

# Create validation submission with correct sig_ids
validation_submission = pd.DataFrame(columns=Y_valid.columns)
validation_submission['sig_id'] = df_valid['sig_id']
for col in target_columns:
    validation_submission[col] = predictions_V[:, list(target_columns).index(col)]

# Save validation predictions to a separate file
VALIDATION_OUTPUT_PATH = "/kaggle/working/validation_predictions_xgb.csv"
validation_submission.to_csv(VALIDATION_OUTPUT_PATH, index=False)
print(f"ğŸ�‰ é©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆå·²å»ºç«‹: {VALIDATION_OUTPUT_PATH}")

# æª¢è¦–é©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆçš„å‰�å¹¾è¡Œ
print("\né©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆé �è¦½:")


# !pip install numpy pandas matplotlib joblib tqdm iterative-stratification notebook


# !nvcc --version
# !nvidia-smi


# !pip install cudf-cu12 cuml-cu12 --extra-index-url=https://pypi.nvidia.com


# !pip install --upgrade dask distributed


# # Kaggle ç’°å¢ƒå»ºè­°ä½¿ç”¨ RAPIDS 0.20ï¼ˆè¦–æƒ…æ³�èª¿æ•´ï¼‰
# !pip install cudf-cu11 cuml-cu11 --extra-index-url=https://pypi.nvidia.com
# print("å�šå®Œäº†")



import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time
import datetime
# from cuml.svm import SVC  # GPU ç‰ˆ
# from cuml import __version__ as cuml_version
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
# import cupy as cp

# ================= è·¯å¾‘è¨­å®šå�€ =================
DATA_ROOT = "/kaggle/input/lish-moa/"
MYMODELS_ROOT = "/kaggle/working/"
CSV_ROOT = "/kaggle/working/"
MODEL_OUTPUT_ROOT = "/kaggle/input/moa-all-models/svm_cpu_models"
TRAIN_FEATURES_PATH = os.path.join(DATA_ROOT, "train_features.csv")
TRAIN_TARGETS_SCORED_PATH = os.path.join(DATA_ROOT, "train_targets_scored.csv")
TRAIN_TARGETS_NONSCORED_PATH = os.path.join(DATA_ROOT, "train_targets_nonscored.csv")
TEST_FEATURES_PATH = os.path.join(DATA_ROOT, "test_features.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
OUTPUT_SUBMISSION_PATH = os.path.join(CSV_ROOT, "submission_svm.csv")
VALIDATION_OUTPUT_PATH = os.path.join(CSV_ROOT, "validation_predictions_svm.csv")
# =============================================

USE_PRETRAINED_MODELS = True

train_features = pd.read_csv(TRAIN_FEATURES_PATH)
train_targets_scored = pd.read_csv(TRAIN_TARGETS_SCORED_PATH)
train_targets_nonscored = pd.read_csv(TRAIN_TARGETS_NONSCORED_PATH)
test_features = pd.read_csv(TEST_FEATURES_PATH)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

df_sample = train_features.sample(n=15000, random_state=42).reset_index(drop=True)
ids = df_sample['sig_id']
Y = train_targets_scored[train_targets_scored['sig_id'].isin(ids)].reset_index(drop=True)

df_valid = train_features[~train_features['sig_id'].isin(ids)].reset_index(drop=True)
Y_valid = train_targets_scored[train_targets_scored['sig_id'].isin(df_valid['sig_id'])].reset_index(drop=True)

def preprocess_features(df):
    df = df.drop(columns=["sig_id"])
    df = pd.get_dummies(df, columns=["cp_type", "cp_dose", "cp_time"])
    return df

X = preprocess_features(df_sample)
X_test = preprocess_features(test_features)
X_valid = preprocess_features(df_valid)

print("ğŸ§  ä½¿ç”¨ CPU é€²è¡Œç‰¹å¾µæ¨™æº–åŒ–")
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

N_SPLITS = 3
mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, random_state=42, shuffle=True)

models = {}
histories = {}

model_path = MODEL_OUTPUT_ROOT
# os.makedirs(model_path, exist_ok=True)
# os.makedirs(f"{model_path}/param", exist_ok=True)  # å„²å­˜æ¨¡å�‹å�ƒæ•¸çš„è³‡æ–™å¤¾
# os.makedirs(f"{model_path}/lossplot", exist_ok=True)  # å„²å­˜lossåœ–çš„è³‡æ–™å¤¾

print(f"ğŸš€ é–‹å§‹è¨“ç·´ - ä½¿ç”¨ CPU")
start_time = time.time()
# print(f"cuML ç‰ˆæœ¬: {cuml_version}")

target_columns = Y.columns[1:]
total_targets = len(target_columns)
print(f"ç¸½å…±éœ€è¦�è¨“ç·´ {total_targets} å€‹ç›®æ¨™")

target_times = []

def compute_logloss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


if USE_PRETRAINED_MODELS:
    print("ğŸ”„ ä½¿ç”¨å·²è¨“ç·´å¥½çš„æ¨¡å�‹")
    for target in tqdm(target_columns, desc='è¼‰å…¥æ¨¡å�‹'):
        model_file = f"{model_path}/param/svm_{target}.pkl"
        models[target] = joblib.load(model_file)
    print("âœ… å·²æˆ�åŠŸè¼‰å…¥æ‰€æœ‰æ¨¡å�‹")
else:
    for i, target in tqdm(enumerate(target_columns), total=total_targets, desc='æ•´é«”é€²åº¦'):
        target_start = time.time()
        print(f"\nè¨“ç·´æ¨¡å�‹ ({i+1}/{total_targets}): {target}")
        y_target = Y[target].values

        indices = range(len(X))
        unique_values, counts = np.unique(y_target, return_counts=True)
        min_count = counts.min() if len(counts) > 0 else 0
        stratify_data = y_target if min_count >= 2 and len(unique_values) > 1 else None

        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=stratify_data)
        X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
        y_train, y_val = y_target[train_indices], y_target[val_indices]

        print(f"æ­£ä¾‹æ¯”ä¾‹: {np.mean(y_train):.4f}")

        model = SVC(kernel='linear', C=1.0, probability=True)
        # å�¯ç°¡åŒ–ç‚ºï¼ˆä¸�è½‰å�‹ä¹Ÿå�¯ï¼‰ï¼š
        model.fit(X_train, y_train)
        train_proba = model.predict_proba(X_train)[:, 1]
        val_proba = model.predict_proba(X_val)[:, 1]


        train_loss = compute_logloss(np.array(y_train), np.array(train_proba))
        val_loss = compute_logloss(np.array(y_val), np.array(val_proba))
        print(f"è¨“ç·´ LogLoss: {train_loss:.4f} | é©—è­‰ LogLoss: {val_loss:.4f}")

        models[target] = model
        joblib.dump(model, f"{model_path}/param/svm_{target}.pkl")

        target_time = time.time() - target_start
        target_times.append(target_time)
        avg_time_per_target = np.mean(target_times)
        estimated_remaining_time = avg_time_per_target * (total_targets - i - 1)
        print(f"é �ä¼°å®Œæˆ�æ™‚é–“: {(datetime.datetime.now() + datetime.timedelta(seconds=estimated_remaining_time)).strftime('%Y-%m-%d %H:%M:%S')}")

print("\nâœ… æ‰€æœ‰æ¨¡å�‹è¨“ç·´å®Œæˆ�")
total_time = time.time() - start_time
print(f"ç¸½è€—æ™‚: {datetime.timedelta(seconds=int(total_time))}")

print("ğŸ”® é–‹å§‹æ¸¬è©¦é›†é �æ¸¬")
predictions = []
for target in tqdm(target_columns, desc='é �æ¸¬é€²åº¦'):
    model = models[target]
    try:
        pred = model.predict_proba(X_test)[:, 1]
    except:
        pred = np.full(len(X_test), 0.5)
    predictions.append(pred)

predictions = np.array(predictions).T
submission = sample_submission.copy()
submission[target_columns] = predictions
submission.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
print("ğŸ“„ æ��äº¤æª”æ¡ˆå„²å­˜æ–¼:", OUTPUT_SUBMISSION_PATH)

print("âœ… é–‹å§‹é©—è­‰é›†é �æ¸¬")
predictions_V = []
for target in tqdm(target_columns, desc='Validation prediction'):
    model = models[target]
    try:
        pred = model.predict_proba(X_valid)[:, 1]

    except:
        pred = np.full(len(X_valid), 0.5)
    predictions_V.append(pred)

predictions_V = np.array(predictions_V).T
validation_submission = pd.DataFrame(columns=Y_valid.columns)
validation_submission['sig_id'] = df_valid['sig_id']
for col in target_columns:
    validation_submission[col] = predictions_V[:, list(target_columns).index(col)]
validation_submission.to_csv(VALIDATION_OUTPUT_PATH, index=False)
print(f"ğŸ�‰ é©—è­‰é �æ¸¬å®Œæˆ�: {VALIDATION_OUTPUT_PATH}")
print(validation_submission.head())


# !pip install catboost


# # ğŸŒŸ MoA CatBoost Pipeline (15000 Sample + Feature Engineering + CV + Loss Plot)
# # é�©ç”¨æ–¼æœ¬æ©Ÿè™•ç�†åŸ·è¡Œï¼ŒåŒ…å�« å¤šæ ¸å¿ƒä¸¦è¡Œè™•ç�†èˆ‡è™›æ“¬ç’°å¢ƒæ�­å»º
# # è«‹å…ˆåŸ·è¡Œä¸‹åˆ— bash æŒ‡ä»¤å•Ÿå‹•è™›æ“¬ç’°å¢ƒï¼š
# # ----------------------------------------------------
# # conda create -n moa-cat python=3.10 -y
# # conda activate moa-cat
# # pip install numpy pandas scikit-learn matplotlib joblib
# # pip install catboost
# # pip install tqdm
# # pip install iterative-stratification
# # pip install notebook
# # mkdir moa-cat-project
# # cd moa-cat-project
# # jupyter notebook
# # ----------------------------------------------------

# # è«‹ç¢ºèª�ä½ å·²å°‡ä»¥ä¸‹æª”æ¡ˆæ”¾å…¥ç•¶å‰�è³‡æ–™å¤¾ï¼š
# # - train_features.csv
# # - train_targets_scored.csv
# # - train_targets_nonscored.csv
# # - test_features.csv
# # - sample_submission.csv
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import joblib
# import time
# import datetime
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from catboost import CatBoostClassifier, Pool
# from tqdm.auto import tqdm  # å¼•å…¥tqdmç”¨æ–¼é€²åº¦è¿½è¹¤
# import multiprocessing as mp  # ç”¨æ–¼å¤šæ ¸å¿ƒåŠ é€Ÿ

# # Optional: ç”¨æ–¼ balanced multilabel CV
# from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# # è¨­å®šå�ƒæ•¸ - æ˜¯å�¦ä½¿ç”¨å·²è¨“ç·´å¥½çš„æ¨¡å�‹
# USE_PRETRAINED_MODELS = True  # è¨­ç‚ºTrueæ™‚ï¼Œå°‡å¾�output/paramè³‡æ–™å¤¾è¼‰å…¥å·²è¨“ç·´çš„æ¨¡å�‹

# # ================= è·¯å¾‘è¨­å®šå�€ =================
# DATA_ROOT = "/kaggle/input/lish-moa/"
# MYMODELS_ROOT = "/kaggle/working/"
# CSV_ROOT = "/kaggle/working/"
# MODEL_OUTPUT_ROOT = "/kaggle/input/moa-all-models/weights/output_param/catb_models"
# TRAIN_FEATURES_PATH = os.path.join(DATA_ROOT, "train_features.csv")
# TRAIN_TARGETS_SCORED_PATH = os.path.join(DATA_ROOT, "train_targets_scored.csv")
# TRAIN_TARGETS_NONSCORED_PATH = os.path.join(DATA_ROOT, "train_targets_nonscored.csv")
# TEST_FEATURES_PATH = os.path.join(DATA_ROOT, "test_features.csv")
# SAMPLE_SUBMISSION_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
# OUTPUT_SUBMISSION_PATH = os.path.join(CSV_ROOT, "submission_cat.csv")
# VALIDATION_OUTPUT_PATH = os.path.join(CSV_ROOT, "validation_predictions_cat.csv")
# # =============================================

# # 1. è®€å�–è³‡æ–™
# train_features = pd.read_csv(TRAIN_FEATURES_PATH)
# train_targets_scored = pd.read_csv(TRAIN_TARGETS_SCORED_PATH)
# train_targets_nonscored = pd.read_csv(TRAIN_TARGETS_NONSCORED_PATH)
# test_features = pd.read_csv(TEST_FEATURES_PATH)
# sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

# # 2. æŠ½æ¨£ 15000 ç­†è¨“ç·´è³‡æ–™
# df_sample = train_features.sample(n=15000, random_state=42).reset_index(drop=True)
# ids = df_sample['sig_id']
# Y = train_targets_scored[train_targets_scored['sig_id'].isin(ids)].reset_index(drop=True)

# # 3. ç‰¹å¾µå·¥ç¨‹
# def preprocess_features(df):
#     cat_features = ["cp_type", "cp_dose", "cp_time"]
#     result_df = df.copy()
#     result_df = result_df.drop(columns=["sig_id"])
#     for feature in cat_features:
#         result_df[feature] = result_df[feature].astype('category').cat.codes.astype(int)
#     return result_df, cat_features

# # è™•ç�†ç‰¹å¾µ
# X, cat_features = preprocess_features(df_sample)
# X_test, _ = preprocess_features(test_features)
# df_valid = train_features[~train_features['sig_id'].isin(ids)].reset_index(drop=True)
# Y_valid = train_targets_scored[train_targets_scored['sig_id'].isin(df_valid['sig_id'])].reset_index(drop=True)
# X_valid, _ = preprocess_features(df_valid)

# # æ¨™æº–åŒ–åƒ…é‡�å°�æ•¸å€¼ç‰¹å¾µ
# scaler = StandardScaler()
# num_features = [col for col in X.columns if col not in cat_features]
# X[num_features] = scaler.fit_transform(X[num_features])
# X_test[num_features] = scaler.transform(X_test[num_features])
# X_valid[num_features] = scaler.transform(X_valid[num_features])
# # å¼·åˆ¶ cat_features æ¬„ä½�å�‹æ…‹ç‚º intï¼Œé�¿å…�è¢«æ¨™æº–åŒ–è¦†è“‹
# for feature in cat_features:
#     X[feature] = X[feature].astype(int)
#     X_test[feature] = X[feature].astype(int)
#     X_valid[feature] = X[feature].astype(int)

# # 4. Cross Validation è¨­å®šï¼ˆMultilabelStratifiedKFoldï¼‰
# N_SPLITS = 3
# mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, random_state=42, shuffle=True)

# models = {}
# histories = {}

# # å»ºç«‹å®Œæ•´è·¯å¾‘çš„å„²å­˜è³‡æ–™å¤¾
# model_path = MODEL_OUTPUT_ROOT
# # os.makedirs(model_path, exist_ok=True)
# # os.makedirs(f"{model_path}/param", exist_ok=True)  # å„²å­˜æ¨¡å�‹å�ƒæ•¸çš„è³‡æ–™å¤¾
# # os.makedirs(f"{model_path}/lossplot", exist_ok=True)  # å„²å­˜lossåœ–çš„è³‡æ–™å¤¾

# # è¨­ç½®å�¯ç”¨æ ¸å¿ƒæ•¸é‡�
# num_cores = mp.cpu_count()
# print(f"ğŸš€ é–‹å§‹è¨“ç·´ - ä½¿ç”¨ {num_cores} å€‹ CPU æ ¸å¿ƒåŠ é€Ÿ")
# start_time = time.time()

# # è¨ˆç®—ç¸½ä»»å‹™æ•¸é‡�ï¼Œç”¨æ–¼é€²åº¦ä¼°ç®—
# target_columns = Y.columns[1:]
# total_targets = len(target_columns)
# print(f"ç¸½å…±éœ€è¦�è¨“ç·´ {total_targets} å€‹ç›®æ¨™")

# # è¿½è¹¤è¨“ç·´æ™‚é–“
# target_times = []

# class CatTracker:
#     """ç”¨æ–¼è¿½è¹¤CatBoostè¨“ç·´é��ç¨‹çš„é¡�åˆ¥"""
#     def __init__(self):
#         self.train_losses = []
#         self.valid_losses = []
    
#     def add_loss(self, train_loss, valid_loss):
#         self.train_losses.append(train_loss)
#         self.valid_losses.append(valid_loss)

# def compute_logloss(y_true, y_pred):
#     """è¨ˆç®—å°�æ•¸æ��å¤±"""
#     # ç¢ºä¿�é �æ¸¬å€¼åœ¨ (0, 1) ä¹‹é–“ï¼Œé�¿å…�æ•¸å€¼å•�é¡Œ
#     y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
#     loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
#     return loss

# # GPU è‡ªå‹•å�µæ¸¬
# try:
#     import pynvml
#     pynvml.nvmlInit()
#     NUM_GPUS = pynvml.nvmlDeviceGetCount()
#     USE_GPU = NUM_GPUS > 0
#     print(f"ğŸš€ å�µæ¸¬åˆ° {NUM_GPUS} é¡† GPUï¼ŒCatBoost å°‡è‡ªå‹•åˆ†é…� GPU è¨“ç·´ï¼�")
# except Exception:
#     USE_GPU = False
#     NUM_GPUS = 0
#     print("ğŸ’» æœªå�µæ¸¬åˆ° GPUï¼Œå°‡ä½¿ç”¨ CPU é€²è¡Œ CatBoost è¨“ç·´")

# # 5. è¨“ç·´æ¨¡å�‹æˆ–è¼‰å…¥å·²æœ‰çš„æ¨¡å�‹
# if USE_PRETRAINED_MODELS:
#     print("ğŸ”„ ä½¿ç”¨å·²è¨“ç·´å¥½çš„æ¨¡å�‹ï¼Œå¾�ä¿�å­˜çš„æª”æ¡ˆä¸­è¼‰å…¥...")
    
#     # æª¢æŸ¥æ˜¯å�¦å­˜åœ¨æ¨¡å�‹æª”æ¡ˆ
#     model_exists = all(os.path.exists(f"{model_path}/param/cat_{target}.pkl") for target in target_columns)
    
#     if not model_exists:
#         print("â�Œ éŒ¯èª¤: æ‰¾ä¸�åˆ°æ‰€æœ‰éœ€è¦�çš„æ¨¡å�‹æª”æ¡ˆï¼Œè«‹è¨­å®š USE_PRETRAINED_MODELS=False é‡�æ–°è¨“ç·´")
#         exit(1)
    
#     # è¼‰å…¥å·²è¨“ç·´çš„æ¨¡å�‹
#     for i, target in tqdm(enumerate(target_columns), total=total_targets, desc='è¼‰å…¥æ¨¡å�‹'):
#         model_file = f"{model_path}/param/cat_{target}.pkl"
#         models[target] = joblib.load(model_file)
#         print(f"å·²è¼‰å…¥æ¨¡å�‹: {target}")
    
#     print("âœ… å·²æˆ�åŠŸè¼‰å…¥æ‰€æœ‰æ¨¡å�‹")
# else:
#     # æ­£å¸¸è¨“ç·´æµ�ç¨‹
#     for i, target in tqdm(enumerate(target_columns), total=total_targets, desc='æ•´é«”é€²åº¦'):
#         target_start = time.time()
        
#         print(f"\nè¨“ç·´æ¨¡å�‹ ({i+1}/{total_targets}): {target}")
#         y_target = Y[target].values
#         fold = 0
#         fold_logloss = []
        
#         # ä½¿ç”¨ train_test_split å‡½æ•¸é€²è¡Œåˆ†å‰²
#         indices = range(len(X))
        
#         # æ›´åš´æ ¼æª¢æŸ¥é¡�åˆ¥åˆ†å¸ƒï¼Œç¢ºä¿�æ¯�å€‹é¡�åˆ¥è‡³å°‘æœ‰ 2 ç­†è³‡æ–™
#         unique_values, counts = np.unique(y_target, return_counts=True)
#         min_count = counts.min() if len(counts) > 0 else 0
        
#         # å¦‚æ�œä»»ä½•é¡�åˆ¥çš„æ¨£æœ¬æ•¸å°‘æ–¼ 2 æˆ–è€…å�ªæœ‰ä¸€å€‹é¡�åˆ¥ï¼Œå‰‡ä¸�ä½¿ç”¨åˆ†å±¤æŠ½æ¨£
#         if min_count < 2 or len(unique_values) <= 1:
#             print(f"è­¦å‘Š: ç›®æ¨™ {target} çš„è³‡æ–™åˆ†å¸ƒä¸�å�‡è¡¡ï¼ŒæŸ�é¡�åˆ¥æ¨£æœ¬æ•¸é��å°‘ï¼Œå°‡ä¸�ä½¿ç”¨åˆ†å±¤æŠ½æ¨£")
#             stratify_data = None
#         else:
#             stratify_data = y_target
            
#         train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=stratify_data)
        
#         X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
#         y_train, y_val = y_target[train_indices], y_target[val_indices]

#         # æª¢æŸ¥é¡�åˆ¥æ˜¯å�¦ä¸�å¹³è¡¡
#         pos_rate = np.mean(y_train)
#         print(f"æ­£ä¾‹æ¯”ä¾‹: {pos_rate:.4f}")
        
#         # é‡�å°�ä¸�å¹³è¡¡è³‡æ–™èª¿æ•´æ¬Šé‡�
#         class_weights = None
#         if pos_rate < 0.2:
#             # ç‚ºCatBoostè¨ˆç®—é¡�åˆ¥æ¬Šé‡�ï¼ˆå��æ¯”ä¾‹æ–¼å‡ºç�¾é »ç�‡ï¼‰
#             class_weights = [(1 - pos_rate) / pos_rate] * 2  # [æ¬Šé‡�æ­£ä¾‹, æ¬Šé‡�è² ä¾‹]
#             print(f"è³‡æ–™ä¸�å¹³è¡¡ï¼Œèª¿æ•´æ¬Šé‡�ç‚º: {(1 - pos_rate) / pos_rate:.2f}")
        
#         tracker = CatTracker()
        
#         # å»ºç«‹CatBoostæ•¸æ“šæ± 
#         train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
#         val_pool = Pool(data=X_val, label=y_val, cat_features=cat_features)
        
#         try:
#             # å»ºç«‹ CatBoost åˆ†é¡�å™¨
#             cat_params = dict(
#                 iterations=200,
#                 depth=6,
#                 learning_rate=0.05,
#                 loss_function='Logloss',
#                 eval_metric='Logloss',
#                 random_seed=42,
#                 class_weights=class_weights,
#                 thread_count=-1,
#                 verbose=False
#             )
#             if USE_GPU:
#                 cat_params['task_type'] = 'GPU'
#                 cat_params['devices'] = str(i % NUM_GPUS)
#                 print(f"[GPU {cat_params['devices']}]", end=" ")
#             model = CatBoostClassifier(**cat_params)
#             # fit æ™‚æ•¸å€¼ç‰¹å¾µè½‰ float32
#             train_pool = Pool(X_train.astype(np.float32), y_train, cat_features=cat_features)
#             val_pool = Pool(X_val.astype(np.float32), y_val, cat_features=cat_features)
#             model.fit(
#                 train_pool,
#                 eval_set=val_pool,
#                 plot=False
#             )
            
#             # è¨ˆç®—è¨“ç·´é›†å’Œé©—è­‰é›†çš„æ��å¤±
#             train_proba = model.predict_proba(X_train)[:, 1]
#             val_proba = model.predict_proba(X_val)[:, 1]
            
#             train_loss = compute_logloss(y_train, train_proba)
#             val_loss = compute_logloss(y_val, val_proba)
            
#             print(f"è¨“ç·´é›† Log Loss: {train_loss:.4f} | é©—è­‰é›† Log Loss: {val_loss:.4f}")
            
#             tracker.add_loss(train_loss, val_loss)
            
#         except Exception as e:
#             print(f"CatBoost è¨“ç·´å¤±æ•—: {str(e)}")
#             # å˜—è©¦ä½¿ç”¨è¼ƒç°¡å–®çš„CatBoostæ¨¡å�‹
#             print("å˜—è©¦ä½¿ç”¨ç°¡åŒ–çš„CatBoostæ¨¡å�‹...", end=" ")
#             model = CatBoostClassifier(
#                 iterations=50,
#                 depth=4,
#                 learning_rate=0.1,
#                 loss_function='Logloss',
#                 eval_metric='Logloss',
#                 random_seed=42,
#                 class_weights=class_weights,
#                 thread_count=-1,
#                 verbose=False
#             )
            
#             model.fit(
#                 train_pool,
#                 eval_set=val_pool,
#                 plot=False
#             )
            
#             # è¨ˆç®—è¨“ç·´é›†å’Œé©—è­‰é›†çš„æ��å¤±
#             train_proba = model.predict_proba(X_train)[:, 1]
#             val_proba = model.predict_proba(X_val)[:, 1]
            
#             train_loss = compute_logloss(y_train, train_proba)
#             val_loss = compute_logloss(y_val, val_proba)
            
#             print(f"è¨“ç·´é›† Log Loss: {train_loss:.4f} | é©—è­‰é›† Log Loss: {val_loss:.4f}")
            
#             tracker.add_loss(train_loss, val_loss)

#         # ç¹ªè£½ Loss plot
#         try:
#             histories[target] = tracker
            
#             plt.figure(figsize=(10, 6))
            
#             # å˜—è©¦ç�²å�–å¹¶ç¹ªè£½è¨“ç·´é��ç¨‹ä¸­çš„æ��å¤±æ›²ç·š
#             train_log = model.get_evals_result()
#             if train_log:
#                 iterations = range(len(train_log['learn']['Logloss']))
#                 plt.plot(iterations, train_log['learn']['Logloss'], label="Train")
#                 plt.plot(iterations, train_log['validation']['Logloss'], label="Valid")
#                 plt.title(f"CatBoost Learning Curve for {target}")
#                 plt.xlabel("Iterations")
#             else:
#                 # å¦‚æ�œæ²’æœ‰è¨“ç·´æ—¥èªŒï¼Œå°±ç¹ªè£½æœ€çµ‚çµ�æ�œ
#                 plt.plot([train_loss], label="Train")
#                 plt.plot([val_loss], label="Valid")
#                 plt.title(f"Logloss for {target}")
#                 plt.xlabel("Model")
            
#             plt.ylabel("Logloss")
#             plt.legend()
#             plt.grid()
#             plt.tight_layout()
#             plt.savefig(f"{model_path}/lossplot/loss_{target}.png")
#             plt.close()
#         except Exception as e:
#             print(f"ç„¡æ³•ç¹ªè£½æ��å¤±æ›²ç·š: {str(e)}")

#         # å„²å­˜æ¨¡å�‹
#         models[target] = model
#         joblib.dump(model, f"{model_path}/param/cat_{target}.pkl")
        
#         # è¨ˆç®—ä¸¦é¡¯ç¤ºé€²åº¦
#         target_time = time.time() - target_start
#         target_times.append(target_time)
#         avg_time_per_target = np.mean(target_times)
#         remaining_targets = total_targets - (i + 1)
#         estimated_remaining_time = avg_time_per_target * remaining_targets
        
#         # è½‰æ�›é �ä¼°å‰©é¤˜æ™‚é–“ç‚ºæ›´æ˜“è®€æ ¼å¼�
#         remaining_time_str = str(datetime.timedelta(seconds=int(estimated_remaining_time)))
#         completion_time = datetime.datetime.now() + datetime.timedelta(seconds=estimated_remaining_time)
        
#         print(f"ç›®æ¨™ {i+1}/{total_targets} å·²å®Œæˆ�! ({target})")
#         print(f"å¹³å�‡æ¯�å€‹ç›®æ¨™è¨“ç·´æ™‚é–“: {avg_time_per_target:.2f} ç§’")
#         print(f"é �ä¼°å‰©é¤˜æ™‚é–“: {remaining_time_str}")
#         print(f"é �ä¼°å®Œæˆ�æ™‚é–“: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")

# # æœ€çµ‚å®Œæˆ�è¨Šæ�¯
# print("\nâœ… æ‰€æœ‰æ¨¡å�‹è¨“ç·´å·²å®Œæˆ�!")
# total_time = time.time() - start_time
# print(f"ç¸½è€—æ™‚: {datetime.timedelta(seconds=int(total_time))}")

# # 6. æ�¨è«–èˆ‡å»ºç«‹æ��äº¤æª”æ¡ˆ
# print("ğŸ”® é–‹å§‹é€²è¡Œé �æ¸¬...")
# predictions = []

# # é¡¯ç¤ºé �æ¸¬é€²åº¦
# for i, target in tqdm(enumerate(target_columns), total=len(target_columns), desc='é �æ¸¬é€²åº¦'):
#     model = models[target]  # å„ªå…ˆä½¿ç”¨è¨˜æ†¶é«”ä¸­çš„æ¨¡å�‹é�¿å…�é‡�è¤‡è¼‰å…¥
#     pred = model.predict_proba(X_test)[:, 1]
#     predictions.append(pred)

# predictions = np.array(predictions).T
# submission = sample_submission.copy()
# submission.iloc[:, 1:] = predictions
# submission.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
# print("ğŸ�‰ å·²ç”¢å‡º submission_cat.csv å�¯ç›´æ�¥ä¸Šå‚³åˆ° Kaggle")

# # æ·»åŠ é©—è­‰é›†çš„é �æ¸¬
# print("ğŸ”� é–‹å§‹é€²è¡Œé©—è­‰é›†é �æ¸¬...")
# predictions_V = []

# # Show prediction progress
# for i, target in tqdm(enumerate(target_columns), total=len(target_columns), desc='Validation prediction progress'):
#     model = models[target]  # Use in-memory models to avoid reloading
#     pred = model.predict_proba(X_valid)
#     # å�‹æ…‹è‡ªå‹•è™•ç�†ï¼šDataFrameã€�cupyã€�numpy
#     if isinstance(pred, pd.DataFrame):
#         pred = pred.iloc[:, 1].values
#     elif hasattr(pred, 'get'):
#         pred = pred.get()
#         pred = pred[:, 1]
#     else:
#         pred = pred[:, 1]
#     predictions_V.append(pred)

# predictions_V = np.array(predictions_V).T

# # Create validation submission with correct sig_ids
# validation_submission = pd.DataFrame(columns=Y_valid.columns)
# validation_submission['sig_id'] = df_valid['sig_id']
# for col in target_columns:
#     validation_submission[col] = predictions_V[:, list(target_columns).index(col)]

# # Save validation predictions to a separate file
# validation_submission.to_csv(VALIDATION_OUTPUT_PATH, index=False)
# print(f"ğŸ�‰ CatBoost é©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆå·²å»ºç«‹: {VALIDATION_OUTPUT_PATH}")

# # æª¢è¦–é©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆçš„å‰�å¹¾è¡Œ
# # print("\né©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆé �è¦½:")
# # print(validation_submission.head())

# # # æ¸¬è©¦ CatBoost æ˜¯å�¦å�¯ä»¥æˆ�åŠŸåœ¨ GPU ä¸Šè¨“ç·´
# # if USE_GPU:
# #     print("\nğŸ”� åŸ·è¡Œ CatBoost GPU æ¸¬è©¦...")
# #     try:
# #         # å»ºç«‹ä¸€å€‹å°�çš„æ¸¬è©¦æ•¸æ“šé›†
# #         X_test_gpu = np.array([[0,0],[1,1]])
# #         y_test_gpu = np.array([0,1])
        
# #         # æ¸¬è©¦ GPU è¨“ç·´
# #         test_model = CatBoostClassifier(task_type='GPU')
# #         test_model.fit(X_test_gpu, y_test_gpu, verbose=False)
# #         print("âœ… GPU æ¸¬è©¦æˆ�åŠŸï¼�æ‚¨çš„ CatBoost å·²ç¶“è¨­ç½®å¥½ä½¿ç”¨ GPU åŠ é€Ÿã€‚")
# #     except Exception as e:
# #         print(f"â�Œ GPU æ¸¬è©¦å¤±æ•—: {e}")
# #         print("å¦‚æ�œæ‚¨ç¢ºå®šæœ‰ GPUï¼Œè«‹ç¢ºèª�å·²æ­£ç¢ºå®‰è£�æ”¯æ�´ GPU çš„ CatBoost ç‰ˆæœ¬")


# !pip install lightgbm


# ğŸŒŸ MoA LightGBM Pipeline (15000 Sample + Feature Engineering + CV + Loss Plot)
# é�©ç”¨æ–¼æœ¬æ©Ÿè™•ç�†åŸ·è¡Œï¼ŒåŒ…å�« å¤šæ ¸å¿ƒä¸¦è¡Œè™•ç�†èˆ‡è™›æ“¬ç’°å¢ƒæ�­å»º
# è«‹å…ˆåŸ·è¡Œä¸‹åˆ— bash æŒ‡ä»¤å•Ÿå‹•è™›æ“¬ç’°å¢ƒï¼š
# ----------------------------------------------------
# conda create -n moa-lgbm python=3.10 -y
# conda activate moa-lgbm
# pip install numpy pandas scikit-learn matplotlib joblib
# conda install -c conda-forge lightgbm
# pip install tqdm
# pip install iterative-stratification
# pip install notebook
# mkdir moa-lgbm-project
# cd moa-lgbm-project
# jupyter notebook
# ----------------------------------------------------

# è«‹ç¢ºèª�ä½ å·²å°‡ä»¥ä¸‹æª”æ¡ˆæ”¾å…¥ç•¶å‰�è³‡æ–™å¤¾ï¼š
# - train_features.csv
# - train_targets_scored.csv
# - train_targets_nonscored.csv
# - test_features.csv
# - sample_submission.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from tqdm.auto import tqdm  # å¼•å…¥tqdmç”¨æ–¼é€²åº¦è¿½è¹¤
import multiprocessing as mp  # ç”¨æ–¼å¤šæ ¸å¿ƒåŠ é€Ÿ

# GPU è‡ªå‹•å�µæ¸¬
try:
    import pynvml
    pynvml.nvmlInit()
    NUM_GPUS = pynvml.nvmlDeviceGetCount()
    USE_GPU = NUM_GPUS > 0
    print(f"ğŸš€ å�µæ¸¬åˆ° {NUM_GPUS} é¡† GPUï¼ŒLightGBM å°‡ä½¿ç”¨ GPU åŠ é€Ÿè¨“ç·´ï¼�")
    print(f"ğŸš€ å�µæ¸¬åˆ° {NUM_GPUS} é¡† GPUï¼ŒLightGBM å°‡è‡ªå‹•åˆ†é…� GPU è¨“ç·´ï¼�")
except Exception:
    USE_GPU = False
    NUM_GPUS = 0
    print("ğŸ’» æœªå�µæ¸¬åˆ° GPUï¼Œå°‡ä½¿ç”¨ CPU é€²è¡Œ LightGBM è¨“ç·´")

# Optional: ç”¨æ–¼ balanced multilabel CV
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# è¨­å®šå�ƒæ•¸ - æ˜¯å�¦ä½¿ç”¨å·²è¨“ç·´å¥½çš„æ¨¡å�‹
USE_PRETRAINED_MODELS = True  # è¨­ç‚ºTrueæ™‚ï¼Œå°‡å¾�output/paramè³‡æ–™å¤¾è¼‰å…¥å·²è¨“ç·´çš„æ¨¡å�‹

# ================= è·¯å¾‘è¨­å®šå�€ =================
DATA_ROOT = "/kaggle/input/lish-moa/"
MYMODELS_ROOT = "/kaggle/working/"
CSV_ROOT = "/kaggle/working/"
MODEL_OUTPUT_ROOT = "/kaggle/input/moa-all-models/weights/output_param/lgbm_models"
TRAIN_FEATURES_PATH = os.path.join(DATA_ROOT, "train_features.csv")
TRAIN_TARGETS_SCORED_PATH = os.path.join(DATA_ROOT, "train_targets_scored.csv")
TRAIN_TARGETS_NONSCORED_PATH = os.path.join(DATA_ROOT, "train_targets_nonscored.csv")
TEST_FEATURES_PATH = os.path.join(DATA_ROOT, "test_features.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
OUTPUT_SUBMISSION_PATH = os.path.join(CSV_ROOT, "submission_lgbm.csv")
VALIDATION_OUTPUT_PATH = os.path.join(CSV_ROOT, "validation_predictions_lgbm.csv")
# =============================================

# 1. è®€å�–è³‡æ–™
train_features = pd.read_csv(TRAIN_FEATURES_PATH)
train_targets_scored = pd.read_csv(TRAIN_TARGETS_SCORED_PATH)
train_targets_nonscored = pd.read_csv(TRAIN_TARGETS_NONSCORED_PATH)
test_features = pd.read_csv(TEST_FEATURES_PATH)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

# 2. æŠ½æ¨£ 15000 ç­†è¨“ç·´è³‡æ–™
df_sample = train_features.sample(n=15000, random_state=42).reset_index(drop=True)
ids = df_sample['sig_id']
Y = train_targets_scored[train_targets_scored['sig_id'].isin(ids)].reset_index(drop=True)
df_valid = train_features[~train_features['sig_id'].isin(ids)].reset_index(drop=True)
Y_valid = train_targets_scored[train_targets_scored['sig_id'].isin(df_valid['sig_id'])].reset_index(drop=True)

# 3. ç‰¹å¾µå·¥ç¨‹
def preprocess_features(df):
    df = df.drop(columns=["sig_id"])
    df = pd.get_dummies(df, columns=["cp_type", "cp_dose", "cp_time"])
    return df

X = preprocess_features(df_sample)
X_test = preprocess_features(test_features)
X_valid = preprocess_features(df_valid)

# æ¨™æº–åŒ–ç‰¹å¾µ
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
X_valid = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns)

# 4. Cross Validation è¨­å®šï¼ˆMultilabelStratifiedKFoldï¼‰
N_SPLITS = 3
mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, random_state=42, shuffle=True)

models = {}
histories = {}

# å»ºç«‹å®Œæ•´è·¯å¾‘çš„å„²å­˜è³‡æ–™å¤¾
model_path = MODEL_OUTPUT_ROOT
# os.makedirs(model_path, exist_ok=True)
# os.makedirs(f"{model_path}/param", exist_ok=True)  # å„²å­˜æ¨¡å�‹å�ƒæ•¸çš„è³‡æ–™å¤¾
# os.makedirs(f"{model_path}/lossplot", exist_ok=True)  # å„²å­˜lossåœ–çš„è³‡æ–™å¤¾

# è¨­ç½®å�¯ç”¨æ ¸å¿ƒæ•¸é‡�
num_cores = mp.cpu_count()
print(f"ğŸš€ é–‹å§‹è¨“ç·´ - ä½¿ç”¨ {num_cores} å€‹ CPU æ ¸å¿ƒåŠ é€Ÿ")
start_time = time.time()

# é¡¯ç¤ºLightGBMç‰ˆæœ¬
lgb_version = lgb.__version__
print(f"LightGBM ç‰ˆæœ¬: {lgb_version}")

# è¨ˆç®—ç¸½ä»»å‹™æ•¸é‡�ï¼Œç”¨æ–¼é€²åº¦ä¼°ç®—
target_columns = Y.columns[1:]
total_targets = len(target_columns)
print(f"ç¸½å…±éœ€è¦�è¨“ç·´ {total_targets} å€‹ç›®æ¨™")

# è¿½è¹¤è¨“ç·´æ™‚é–“
target_times = []

class LGBMTracker:
    """ç”¨æ–¼è¿½è¹¤LightGBMè¨“ç·´é��ç¨‹çš„é¡�åˆ¥"""
    def __init__(self):
        self.train_losses = []
        self.valid_losses = []
    
    def add_loss(self, train_loss, valid_loss):
        self.train_losses.append(train_loss)
        self.valid_losses.append(valid_loss)

def compute_logloss(y_true, y_pred):
    """è¨ˆç®—å°�æ•¸æ��å¤±"""
    # ç¢ºä¿�é �æ¸¬å€¼åœ¨ (0, 1) ä¹‹é–“ï¼Œé�¿å…�æ•¸å€¼å•�é¡Œ
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# 5. è¨“ç·´æ¨¡å�‹æˆ–è¼‰å…¥å·²æœ‰çš„æ¨¡å�‹
if USE_PRETRAINED_MODELS:
    print("ğŸ”„ ä½¿ç”¨å·²è¨“ç·´å¥½çš„æ¨¡å�‹ï¼Œå¾�ä¿�å­˜çš„æª”æ¡ˆä¸­è¼‰å…¥...")
    
    # æª¢æŸ¥æ˜¯å�¦å­˜åœ¨æ¨¡å�‹æª”æ¡ˆ
    model_exists = all(os.path.exists(f"{model_path}/param/lgbm_{target}.pkl") for target in target_columns)
    
    if not model_exists:
        print("â�Œ éŒ¯èª¤: æ‰¾ä¸�åˆ°æ‰€æœ‰éœ€è¦�çš„æ¨¡å�‹æª”æ¡ˆï¼Œè«‹è¨­å®š USE_PRETRAINED_MODELS=False é‡�æ–°è¨“ç·´")
        exit(1)
    
    # è¼‰å…¥å·²è¨“ç·´çš„æ¨¡å�‹
    for i, target in tqdm(enumerate(target_columns), total=total_targets, desc='è¼‰å…¥æ¨¡å�‹'):
        model_file = f"{model_path}/param/lgbm_{target}.pkl"
        models[target] = joblib.load(model_file)
        print(f"å·²è¼‰å…¥æ¨¡å�‹: {target}")
    
    print("âœ… å·²æˆ�åŠŸè¼‰å…¥æ‰€æœ‰æ¨¡å�‹")
else:
    # æ­£å¸¸è¨“ç·´æµ�ç¨‹
    for i, target in tqdm(enumerate(target_columns), total=total_targets, desc='æ•´é«”é€²åº¦'):
        target_start = time.time()
        
        print(f"\nè¨“ç·´æ¨¡å�‹ ({i+1}/{total_targets}): {target}")
        y_target = Y[target].values
        fold = 0
        fold_logloss = []
        
        # ä½¿ç”¨ train_test_split å‡½æ•¸é€²è¡Œåˆ†å‰²
        indices = range(len(X))
        
        # æ›´åš´æ ¼æª¢æŸ¥é¡�åˆ¥åˆ†å¸ƒï¼Œç¢ºä¿�æ¯�å€‹é¡�åˆ¥è‡³å°‘æœ‰ 2 ç­†è³‡æ–™
        unique_values, counts = np.unique(y_target, return_counts=True)
        min_count = counts.min() if len(counts) > 0 else 0
        
        # å¦‚æ�œä»»ä½•é¡�åˆ¥çš„æ¨£æœ¬æ•¸å°‘æ–¼ 2 æˆ–è€…å�ªæœ‰ä¸€å€‹é¡�åˆ¥ï¼Œå‰‡ä¸�ä½¿ç”¨åˆ†å±¤æŠ½æ¨£
        if (min_count < 2 or len(unique_values) <= 1):
            print(f"è­¦å‘Š: ç›®æ¨™ {target} çš„è³‡æ–™åˆ†å¸ƒä¸�å�‡è¡¡ï¼ŒæŸ�é¡�åˆ¥æ¨£æœ¬æ•¸é��å°‘ï¼Œå°‡ä¸�ä½¿ç”¨åˆ†å±¤æŠ½æ¨£")
            stratify_data = None
        else:
            stratify_data = y_target
            
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=stratify_data)
        
        X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
        y_train, y_val = y_target[train_indices], y_target[val_indices]

        # æª¢æŸ¥é¡�åˆ¥æ˜¯å�¦ä¸�å¹³è¡¡
        pos_rate = np.mean(y_train)
        print(f"æ­£ä¾‹æ¯”ä¾‹: {pos_rate:.4f}")
        
        # é‡�å°�ä¸�å¹³è¡¡è³‡æ–™èª¿æ•´æ¬Šé‡�
        scale_pos_weight = 1
        if (pos_rate < 0.2):
            scale_pos_weight = (1 - pos_rate) / pos_rate
            print(f"è³‡æ–™ä¸�å¹³è¡¡ï¼Œèª¿æ•´æ¬Šé‡�ç‚º: {scale_pos_weight:.2f}")
        
        tracker = LGBMTracker()
        eval_results = {}
        
        try:
            # å»ºç«‹ LightGBM åˆ†é¡�å™¨
            print("è¨“ç·´æ¨¡å�‹ä¸­...", end=" ")
            lgbm_params = dict(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                min_data_in_leaf=1,
                min_child_samples=1,
                min_gain_to_split=0,
                random_state=42,
                n_jobs=-1
            )
            if USE_GPU:
                lgbm_params['device'] = 'gpu'
                lgbm_params['gpu_device_id'] = i % NUM_GPUS
                print(f"[GPU {lgbm_params['gpu_device_id']}]", end=" ")
            model = lgb.LGBMClassifier(**lgbm_params)
            # fit æ™‚è½‰ float32
            model.fit(
                X_train.astype(np.float32) if USE_GPU else X_train,
                y_train,
                eval_set=[(X_train.astype(np.float32) if USE_GPU else X_train, y_train), (X_val.astype(np.float32) if USE_GPU else X_val, y_val)],
                eval_metric='logloss',
                callbacks=[lgb.log_evaluation(0)],
                early_stopping_rounds=20,
                verbose=False
            )
            
            # ç�²å�–æ¯�æ¬¡è¿­ä»£çš„è©•ä¼°çµ�æ�œ
            if hasattr(model, 'evals_result_'):
                eval_results = model.evals_result_
            
            # è¨ˆç®—è¨“ç·´é›†å’Œé©—è­‰é›†çš„æ��å¤±
            train_proba = model.predict_proba(X_train)[:, 1]
            val_proba = model.predict_proba(X_val)[:, 1]
            
            train_loss = compute_logloss(y_train, train_proba)
            val_loss = compute_logloss(y_val, val_proba)
            
            print(f"è¨“ç·´é›† Log Loss: {train_loss:.4f} | é©—è­‰é›† Log Loss: {val_loss:.4f}")
            
            tracker.add_loss(train_loss, val_loss)
            
        except Exception as e:
            print(f"LightGBM è¨“ç·´å¤±æ•—: {str(e)}")
            # å˜—è©¦ä½¿ç”¨è¼ƒç°¡å–®çš„å�ƒæ•¸
            print("å˜—è©¦ä½¿ç”¨ç°¡åŒ–çš„LightGBMæ¨¡å�‹...", end=" ")
            model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                num_leaves=16,
                scale_pos_weight=scale_pos_weight,
                min_data_in_leaf=1,
                min_child_samples=1,
                min_gain_to_split=0,
                random_state=42,
                n_jobs=-1
            )
            # å�ªå‚³å¿…è¦�å�ƒæ•¸ï¼Œç§»é™¤ verbose/early_stopping_rounds
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='logloss',
                callbacks=[lgb.log_evaluation(0)]
            )
            
            # è¨ˆç®—è¨“ç·´é›†å’Œé©—è­‰é›†çš„æ��å¤±
            train_proba = model.predict_proba(X_train)[:, 1]
            val_proba = model.predict_proba(X_val)[:, 1]
            
            train_loss = compute_logloss(y_train, train_proba)
            val_loss = compute_logloss(y_val, val_proba)
            
            print(f"è¨“ç·´é›† Log Loss: {train_loss:.4f} | é©—è­‰é›† Log Loss: {val_loss:.4f}")
            
            tracker.add_loss(train_loss, val_loss)

        # # ç¹ªè£½ Loss plot
        # try:
        #     histories[target] = tracker
            
        #     plt.figure(figsize=(10, 6))
            
        #     # å˜—è©¦ç¹ªè£½è¨“ç·´é��ç¨‹ä¸­çš„æ��å¤±æ›²ç·š
        #     if eval_results and 'train' in eval_results and 'logloss' in eval_results['train']:
        #         train_curve = eval_results['train']['logloss']
        #         valid_curve = eval_results['valid']['logloss']
                
        #         plt.plot(train_curve, label="Train")
        #         plt.plot(valid_curve, label="Valid")
        #         plt.title(f"LightGBM Logloss Curve for {target}")
        #         plt.xlabel("Iterations")
        #     else:
        #         # å¦‚æ�œæ²’æœ‰è¨“ç·´é��ç¨‹çš„æ›²ç·šï¼Œå°±ç¹ªè£½æœ€çµ‚çµ�æ�œ
        #         plt.plot([train_loss], label="Train")
        #         plt.plot([val_loss], label="Valid")
        #         plt.title(f"Logloss for {target}")
        #         plt.xlabel("Model")
            
        #     plt.ylabel("Logloss")
        #     plt.legend()
        #     plt.grid()
        #     plt.tight_layout()
        #     plt.savefig(f"{model_path}/lossplot/loss_{target}.png")
        #     plt.close()
        # except Exception as e:
        #     print(f"ç„¡æ³•ç¹ªè£½æ��å¤±æ›²ç·š: {str(e)}")

        # å„²å­˜æ¨¡å�‹
        models[target] = model
        joblib.dump(model, f"{model_path}/param/lgbm_{target}.pkl")
        
        # è¨ˆç®—ä¸¦é¡¯ç¤ºé€²åº¦
        target_time = time.time() - target_start
        target_times.append(target_time)
        avg_time_per_target = np.mean(target_times)
        remaining_targets = total_targets - (i + 1)
        estimated_remaining_time = avg_time_per_target * remaining_targets
        
        # è½‰æ�›é �ä¼°å‰©é¤˜æ™‚é–“ç‚ºæ›´æ˜“è®€æ ¼å¼�
        remaining_time_str = str(datetime.timedelta(seconds=int(estimated_remaining_time)))
        completion_time = datetime.datetime.now() + datetime.timedelta(seconds=estimated_remaining_time)
        
        print(f"ç›®æ¨™ {i+1}/{total_targets} å·²å®Œæˆ�! ({target})")
        print(f"å¹³å�‡æ¯�å€‹ç›®æ¨™è¨“ç·´æ™‚é–“: {avg_time_per_target:.2f} ç§’")
        print(f"é �ä¼°å‰©é¤˜æ™‚é–“: {remaining_time_str}")
        print(f"é �ä¼°å®Œæˆ�æ™‚é–“: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")

# æœ€çµ‚å®Œæˆ�è¨Šæ�¯
print("\nâœ… æ‰€æœ‰æ¨¡å�‹è¨“ç·´å·²å®Œæˆ�!")
total_time = time.time() - start_time
print(f"ç¸½è€—æ™‚: {datetime.timedelta(seconds=int(total_time))}")

# 6. æ�¨è«–èˆ‡å»ºç«‹æ��äº¤æª”æ¡ˆ
print("ğŸ”® é–‹å§‹é€²è¡Œé �æ¸¬...")
predictions = []

# é¡¯ç¤ºé �æ¸¬é€²åº¦
for i, target in tqdm(enumerate(target_columns), total=len(target_columns), desc='é �æ¸¬é€²åº¦'):
    model = models[target]  # å„ªå…ˆä½¿ç”¨è¨˜æ†¶é«”ä¸­çš„æ¨¡å�‹é�¿å…�é‡�è¤‡è¼‰å…¥
    pred = model.predict_proba(X_test)[:, 1]
    predictions.append(pred)

predictions = np.array(predictions).T
submission = sample_submission.copy()
submission.iloc[:, 1:] = predictions
submission.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
print("ğŸ�‰ å·²ç”¢å‡º submission_lgbm.csv å�¯ç›´æ�¥ä¸Šå‚³åˆ° Kaggle")

# æ·»åŠ é©—è­‰é›†çš„é �æ¸¬
print("ğŸ”� é–‹å§‹é€²è¡Œé©—è­‰é›†é �æ¸¬...")
predictions_V = []

# Show prediction progress
for i, target in tqdm(enumerate(target_columns), total=len(target_columns), desc='Validation prediction progress'):
    model = models[target]  # Use in-memory models to avoid reloading
    pred = model.predict_proba(X_valid)
    # å�‹æ…‹è‡ªå‹•è™•ç�†ï¼šDataFrameã€�cupyã€�numpy
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 1].values
    elif hasattr(pred, 'get'):
        pred = pred.get()
        pred = pred[:, 1]
    else:
        pred = pred[:, 1]
    predictions_V.append(pred)

predictions_V = np.array(predictions_V).T

# Create validation submission with correct sig_ids
validation_submission = pd.DataFrame(columns=Y_valid.columns)
validation_submission['sig_id'] = df_valid['sig_id']
for col in target_columns:
    validation_submission[col] = predictions_V[:, list(target_columns).index(col)]

# Save validation predictions to a separate file
validation_submission.to_csv(VALIDATION_OUTPUT_PATH, index=False)
print(f"ğŸ�‰ LightGBM é©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆå·²å»ºç«‹: {VALIDATION_OUTPUT_PATH}")

# æª¢è¦–é©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆçš„å‰�å¹¾è¡Œ
# print("\né©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆé �è¦½:")
# print(validation_submission.head())

# # æ¸¬è©¦æ˜¯å�¦å�¯æˆ�åŠŸåœ¨ GPU ä¸Šè¨“ç·´
# if USE_GPU:
#     print("\nğŸ”� åŸ·è¡Œ GPU æ¸¬è©¦...")
#     try:
#         # å»ºç«‹ä¸€å€‹å°�çš„æ¸¬è©¦æ•¸æ“šé›†
#         X_test_gpu = np.array([[0,0],[1,1]])
#         y_test_gpu = np.array([0,1])
        
#         # æ¸¬è©¦ GPU è¨“ç·´
#         test_model = lgb.LGBMClassifier(device='gpu')
#         test_model.fit(X_test_gpu, y_test_gpu)
#         print("âœ… GPU æ¸¬è©¦æˆ�åŠŸï¼�æ‚¨çš„ LightGBM å·²ç¶“è¨­ç½®å¥½ä½¿ç”¨ GPU åŠ é€Ÿã€‚")
#     except Exception as e:
#         print(f"â�Œ GPU æ¸¬è©¦å¤±æ•—: {e}")
#         print("å¦‚æ�œæ‚¨ç¢ºå®šæœ‰ GPUï¼Œè«‹ç¢ºèª�å·²å®‰è£� GPU ç‰ˆæœ¬çš„ LightGBM:")
#         print("conda install -c conda-forge lightgbm cudatoolkit=11.0")
#         print("æˆ–è€…:")
#         print("pip install lightgbm --install-option=--gpu")


# !pip uninstall scikit-learn
# !pip install -U scikit-learn==1.6.1


# import sklearn
# print("Sklearn version:", sklearn.__version__)
# print("Sklearn path:", sklearn.__file__)
# # Should be 1.6.1


# ğŸŒŸ MoA Random Forest Pipeline (15000 Sample + Feature Engineering + CV + Loss Plot)
# é�©ç”¨æ–¼æœ¬æ©Ÿè™•ç�†åŸ·è¡Œï¼ŒåŒ…å�« å¤šæ ¸å¿ƒä¸¦è¡Œè™•ç�†èˆ‡è™›æ“¬ç’°å¢ƒæ�­å»º
# è«‹å…ˆåŸ·è¡Œä¸‹åˆ— bash æŒ‡ä»¤å•Ÿå‹•è™›æ“¬ç’°å¢ƒï¼š
# ----------------------------------------------------
# conda create -n moa-rf python=3.10 -y
# conda activate moa-rf
# pip install numpy pandas scikit-learn matplotlib joblib
# pip install tqdm
# pip install iterative-stratification
# pip install notebook
# mkdir moa-rf-project
# cd moa-rf-project
# jupyter notebook
# ----------------------------------------------------

# è«‹ç¢ºèª�ä½ å·²å°‡ä»¥ä¸‹æª”æ¡ˆæ”¾å…¥ç•¶å‰�è³‡æ–™å¤¾ï¼š
# - train_features.csv
# - train_targets_scored.csv
# - train_targets_nonscored.csv
# - test_features.csv
# - sample_submission.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import time
import datetime
# print(f"scikit-learn ç‰ˆæœ¬: {sklearn_version}")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import __version__ as sklearn_version
from sklearn.calibration import CalibratedClassifierCV
from tqdm.auto import tqdm  # å¼•å…¥tqdmç”¨æ–¼é€²åº¦è¿½è¹¤
import multiprocessing as mp  # ç”¨æ–¼å¤šæ ¸å¿ƒåŠ é€Ÿ

# Optional: ç”¨æ–¼ balanced multilabel CV
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# ================= è·¯å¾‘è¨­å®šå�€ =================
DATA_ROOT = "/kaggle/input/lish-moa/"
MYMODELS_ROOT = "/kaggle/working/"
CSV_ROOT = "/kaggle/working/"
MODEL_OUTPUT_ROOT = "/kaggle/input/moa-all-models/new_rf/"
TRAIN_FEATURES_PATH = os.path.join(DATA_ROOT, "train_features.csv")
TRAIN_TARGETS_SCORED_PATH = os.path.join(DATA_ROOT, "train_targets_scored.csv")
TRAIN_TARGETS_NONSCORED_PATH = os.path.join(DATA_ROOT, "train_targets_nonscored.csv")
TEST_FEATURES_PATH = os.path.join(DATA_ROOT, "test_features.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
OUTPUT_SUBMISSION_PATH = os.path.join(CSV_ROOT, "submission_rf.csv")
VALIDATION_OUTPUT_PATH = os.path.join(CSV_ROOT, "validation_predictions_rf.csv")
# =============================================
USE_PRETRAINED_MODELS = True
# 1. è®€å�–è³‡æ–™
train_features = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")
train_targets_scored = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")
train_targets_nonscored = pd.read_csv("/kaggle/input/lish-moa/train_targets_nonscored.csv")
test_features = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")
sample_submission = pd.read_csv("/kaggle/input/lish-moa/sample_submission.csv")

# 2. æŠ½æ¨£ 15000 ç­†è¨“ç·´è³‡æ–™
df_sample = train_features.sample(n=15000, random_state=42).reset_index(drop=True)
ids = df_sample['sig_id']
Y = train_targets_scored[train_targets_scored['sig_id'].isin(ids)].reset_index(drop=True)

# 3. ç‰¹å¾µå·¥ç¨‹
def preprocess_features(df):
    df = df.drop(columns=["sig_id"])
    df = pd.get_dummies(df, columns=["cp_type", "cp_dose", "cp_time"])
    return df

X = preprocess_features(df_sample)
X_test = preprocess_features(test_features)

df_valid = train_features[~train_features['sig_id'].isin(ids)].reset_index(drop=True)
Y_valid = train_targets_scored[train_targets_scored['sig_id'].isin(df_valid['sig_id'])].reset_index(drop=True)
X_valid = preprocess_features(df_valid)

# æ¨™æº–åŒ–ç‰¹å¾µ
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
X_valid = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns)

# 4. Cross Validation è¨­å®šï¼ˆMultilabelStratifiedKFoldï¼‰
N_SPLITS = 3
mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, random_state=42, shuffle=True)

models = {}
histories = {}

# å»ºç«‹å®Œæ•´è·¯å¾‘çš„å„²å­˜è³‡æ–™å¤¾
model_path = MODEL_OUTPUT_ROOT
# os.makedirs(model_path, exist_ok=True)
# os.makedirs(f"{model_path}/param", exist_ok=True)  # å„²å­˜æ¨¡å�‹å�ƒæ•¸çš„è³‡æ–™å¤¾
# os.makedirs(f"{model_path}/lossplot", exist_ok=True)  # å„²å­˜lossåœ–çš„è³‡æ–™å¤¾

# è¨­ç½®å�¯ç”¨æ ¸å¿ƒæ•¸é‡�
num_cores = mp.cpu_count()
print(f"ğŸš€ é–‹å§‹è¨“ç·´ - ä½¿ç”¨ {num_cores} å€‹ CPU æ ¸å¿ƒåŠ é€Ÿ")
start_time = time.time()

# æª¢æŸ¥ scikit-learn ç‰ˆæœ¬
print(f"scikit-learn ç‰ˆæœ¬: {sklearn_version}")

# è¨ˆç®—ç¸½ä»»å‹™æ•¸é‡�ï¼Œç”¨æ–¼é€²åº¦ä¼°ç®—
target_columns = Y.columns[1:]
total_targets = len(target_columns)
print(f"ç¸½å…±éœ€è¦�è¨“ç·´ {total_targets} å€‹ç›®æ¨™")

# è¿½è¹¤è¨“ç·´æ™‚é–“
target_times = []

class RFTracker:
    """ç”¨æ–¼è¿½è¹¤Random Forestè¨“ç·´é��ç¨‹çš„é¡�åˆ¥"""
    def __init__(self):
        self.train_losses = []
        self.valid_losses = []
    
    def add_loss(self, train_loss, valid_loss):
        self.train_losses.append(train_loss)
        self.valid_losses.append(valid_loss)

def compute_logloss(y_true, y_pred):
    """è¨ˆç®—å°�æ•¸æ��å¤±"""
    # ç¢ºä¿�é �æ¸¬å€¼åœ¨ (0, 1) ä¹‹é–“ï¼Œé�¿å…�æ•¸å€¼å•�é¡Œ
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# GPU è‡ªå‹•å�µæ¸¬
try:
    import cupy as cp
    import cuml
    from cuml.ensemble import RandomForestClassifier as cuRF
    NUM_GPUS = cp.cuda.runtime.getDeviceCount()
    USE_GPU = NUM_GPUS > 0
    print(f"ğŸš€ å�µæ¸¬åˆ° {NUM_GPUS} é¡† GPUï¼Œå°‡å„ªå…ˆä½¿ç”¨ cuML GPU RandomForestï¼�")
except Exception:
    USE_GPU = False
    NUM_GPUS = 0
    print("ğŸ’» æœªå�µæ¸¬åˆ° GPU æˆ– cuMLï¼Œå°‡ä½¿ç”¨ sklearn CPU RandomForest")

# 5. è¨“ç·´æ¨¡å�‹æˆ–è¼‰å…¥å·²æœ‰çš„æ¨¡å�‹
if USE_PRETRAINED_MODELS:
    print("ğŸ”„ ä½¿ç”¨å·²è¨“ç·´å¥½çš„æ¨¡å�‹ï¼Œå¾�ä¿�å­˜çš„æª”æ¡ˆä¸­è¼‰å…¥...")
    
    # æª¢æŸ¥æ˜¯å�¦å­˜åœ¨æ¨¡å�‹æª”æ¡ˆ
    model_exists = all(os.path.exists(f"{model_path}/rf_{target}.pkl") for target in target_columns)
    
    if not model_exists:
        print("â�Œ éŒ¯èª¤: æ‰¾ä¸�åˆ°æ‰€æœ‰éœ€è¦�çš„æ¨¡å�‹æª”æ¡ˆï¼Œè«‹è¨­å®š USE_PRETRAINED_MODELS=False é‡�æ–°è¨“ç·´")
        exit(1)
    
    try:
        # è¼‰å…¥å·²è¨“ç·´çš„æ¨¡å�‹
        for i, target in tqdm(enumerate(target_columns), total=total_targets, desc='è¼‰å…¥æ¨¡å�‹'):
            model_file = f"{model_path}/rf_{target}.pkl"
            try:
                models[target] = joblib.load(model_file)
                print(f"å·²è¼‰å…¥æ¨¡å�‹: {target}")
            except Exception as e:
                print(f"â�Œ è¼‰å…¥æ¨¡å�‹ {target} å¤±æ•—: {e}")
                print("âš ï¸� æª¢æŸ¥åˆ°æ¨¡å�‹æª”æ¡ˆèˆ‡ç›®å‰� scikit-learn ç‰ˆæœ¬ä¸�ç›¸å®¹ï¼Œå°‡è‡ªå‹•åˆ‡æ�›ç‚ºé‡�æ–°è¨“ç·´æ¨¡å¼�ã€‚")
                USE_PRETRAINED_MODELS = False
                break
        if USE_PRETRAINED_MODELS:
            print("âœ… å·²æˆ�åŠŸè¼‰å…¥æ‰€æœ‰æ¨¡å�‹")
    except Exception as e:
        print(f"â�Œ è¼‰å…¥æ¨¡å�‹æ™‚ç™¼ç”ŸéŒ¯èª¤: {e}")
        print("âš ï¸� æª¢æŸ¥åˆ°æ¨¡å�‹æª”æ¡ˆèˆ‡ç›®å‰� scikit-learn ç‰ˆæœ¬ä¸�ç›¸å®¹ï¼Œå°‡è‡ªå‹•åˆ‡æ�›ç‚ºé‡�æ–°è¨“ç·´æ¨¡å¼�ã€‚")
        USE_PRETRAINED_MODELS = False

if not USE_PRETRAINED_MODELS:
    # æ­£å¸¸è¨“ç·´æµ�ç¨‹
    for i, target in tqdm(enumerate(target_columns), total=total_targets, desc='æ•´é«”é€²åº¦'):
        target_start = time.time()
        
        print(f"\nè¨“ç·´æ¨¡å�‹ ({i+1}/{total_targets}): {target}")
        y_target = Y[target].values
        fold = 0
        fold_logloss = []
        
        # ä½¿ç”¨ train_test_split å‡½æ•¸é€²è¡Œåˆ†å‰²
        indices = np.arange(len(X))  # sklearn 1.2 éœ€è¦� array-like

        # æ›´åš´æ ¼æª¢æŸ¥é¡�åˆ¥åˆ†å¸ƒï¼Œç¢ºä¿�æ¯�å€‹é¡�åˆ¥è‡³å°‘æœ‰ 2 ç­†è³‡æ–™
        unique_values, counts = np.unique(y_target, return_counts=True)
        min_count = counts.min() if len(counts) > 0 else 0

        # å¦‚æ�œä»»ä½•é¡�åˆ¥çš„æ¨£æœ¬æ•¸å°‘æ–¼ 2 æˆ–è€…å�ªæœ‰ä¸€å€‹é¡�åˆ¥ï¼Œå‰‡ä¸�ä½¿ç”¨åˆ†å±¤æŠ½æ¨£
        if min_count < 2 or len(unique_values) <= 1:
            print(f"è­¦å‘Š: ç›®æ¨™ {target} çš„è³‡æ–™åˆ†å¸ƒä¸�å�‡è¡¡ï¼ŒæŸ�é¡�åˆ¥æ¨£æœ¬æ•¸é��å°‘ï¼Œå°‡ä¸�ä½¿ç”¨åˆ†å±¤æŠ½æ¨£")
            stratify_data = None
        else:
            stratify_data = y_target

        # sklearn 1.2: indices å¿…é ˆæ˜¯ array-like
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=stratify_data)

        X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
        y_train, y_val = y_target[train_indices], y_target[val_indices]

        # æª¢æŸ¥é¡�åˆ¥æ˜¯å�¦ä¸�å¹³è¡¡
        pos_rate = np.mean(y_train)
        print(f"æ­£ä¾‹æ¯”ä¾‹: {pos_rate:.4f}")
        
        # é‡�å°�ä¸�å¹³è¡¡è³‡æ–™èª¿æ•´æ¬Šé‡�
        class_weight = None
        if pos_rate < 0.2 or pos_rate > 0.8:
            weight_ratio = (1 - pos_rate) / pos_rate if pos_rate < 0.5 else pos_rate / (1 - pos_rate)
            class_weight = {0: 1, 1: weight_ratio} if pos_rate < 0.5 else {0: weight_ratio, 1: 1}
            print(f"è³‡æ–™ä¸�å¹³è¡¡ï¼Œèª¿æ•´æ¬Šé‡�ç‚º: {weight_ratio}")
        
        tracker = RFTracker()
        
        try:
            if USE_GPU:
                gpu_device = i % NUM_GPUS
                print(f"[GPU {gpu_device}] cuML RF è¨“ç·´ä¸­...", end=" ")
                with cp.cuda.Device(gpu_device):
                    try:
                        # èª¿æ•´å�ƒæ•¸ä»¥æ��é«˜ç©©å®šæ€§
                        model = cuRF(
                            n_estimators=100,  # æ¸›å°‘æ¨¹çš„æ•¸é‡�
                            max_depth=6,       # æ¸›å°‘æ¨¹çš„æ·±åº¦
                            n_streams=1,       # å�ªç”¨ä¸€å€‹æµ�ï¼Œå¢�åŠ ç©©å®šæ€§
                            max_features=0.8,  # é™�åˆ¶ç‰¹å¾µæ•¸é‡�
                            # max_samples=0.8,  # sklearn 1.2 ä¸�æ”¯æ�´æ­¤å�ƒæ•¸ï¼Œç§»é™¤
                            random_state=42,
                            handle=None
                        )
                        # ç¢ºä¿�è³‡æ–™æ˜¯é€£çºŒçš„è¨˜æ†¶é«”å¡Š
                        X_train_gpu = X_train.values.astype(np.float32, order='C')
                        X_val_gpu = X_val.values.astype(np.float32, order='C')
                        
                        # é‡‹æ”¾ä¸€äº›è¨˜æ†¶é«”
                        cp.get_default_memory_pool().free_all_blocks()
                        
                        # è¨“ç·´æ¨¡å�‹
                        model.fit(X_train_gpu, y_train)
                        train_proba = model.predict_proba(X_train_gpu)[:, 1]
                        val_proba = model.predict_proba(X_val_gpu)[:, 1]
                        
                        if hasattr(train_proba, 'get'):
                            train_proba = train_proba.get()
                            val_proba = val_proba.get()
                    except Exception as gpu_error:
                        print(f"\nâš ï¸� GPUè¨“ç·´å¤±æ•—: {str(gpu_error)}")
                        print("å˜—è©¦ä½¿ç”¨æ›´ç°¡å–®çš„GPUè¨­ç½®...")
                        
                        # é‡‹æ”¾è¨˜æ†¶é«”
                        cp.get_default_memory_pool().free_all_blocks()
                        
                        # å˜—è©¦ä½¿ç”¨æ›´ç°¡å–®çš„è¨­ç½®
                        model = cuRF(
                            n_estimators=50,
                            max_depth=4,
                            n_streams=1,
                            max_features=0.6,
                            # max_samples=0.6,  # sklearn 1.2 ä¸�æ”¯æ�´æ­¤å�ƒæ•¸ï¼Œç§»é™¤
                            random_state=42,
                            handle=None
                        )
                        
                        # æ¸›å°‘è³‡æ–™é‡�
                        if len(X_train) > 5000:
                            X_train_sample = X_train.sample(n=5000, random_state=42)
                            y_train_sample = y_train[X_train_sample.index]
                            X_train_gpu = X_train_sample.values.astype(np.float32, order='C')
                        else:
                            X_train_gpu = X_train.values.astype(np.float32, order='C')
                            y_train_sample = y_train
                            
                        X_val_gpu = X_val.values.astype(np.float32, order='C')
                        
                        model.fit(X_train_gpu, y_train_sample)
                        train_proba = model.predict_proba(X_train_gpu)[:, 1]
                        val_proba = model.predict_proba(X_val_gpu)[:, 1]
                        
                        if hasattr(train_proba, 'get'):
                            train_proba = train_proba.get()
                            val_proba = val_proba.get()
            else:
                print("sklearn RF è¨“ç·´ä¸­...", end=" ")
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    class_weight=class_weight,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                train_proba = model.predict_proba(X_train)[:, 1]
                val_proba = model.predict_proba(X_val)[:, 1]
            train_loss = compute_logloss(y_train, train_proba)
            val_loss = compute_logloss(y_val, val_proba)
            print(f"è¨“ç·´é›† Log Loss: {train_loss:.4f} | é©—è­‰é›† Log Loss: {val_loss:.4f}")
            tracker.add_loss(train_loss, val_loss)
            
        except Exception as e:
            print(f"Random Forest è¨“ç·´å¤±æ•—: {str(e)}")
            # å˜—è©¦ä½¿ç”¨è¼ƒç°¡å–®çš„éš¨æ©Ÿæ£®æ�—æ¨¡å�‹
            print("å˜—è©¦ä½¿ç”¨ç°¡åŒ–çš„éš¨æ©Ÿæ£®æ�—æ¨¡å�‹...", end=" ")
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=4,
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # è¨ˆç®—è¨“ç·´é›†å’Œé©—è­‰é›†çš„æ��å¤±
            train_proba = model.predict_proba(X_train)[:, 1]
            val_proba = model.predict_proba(X_val)[:, 1]
            
            train_loss = compute_logloss(y_train, train_proba)
            val_loss = compute_logloss(y_val, val_proba)
            
            print(f"è¨“ç·´é›† Log Loss: {train_loss:.4f} | é©—è­‰é›† Log Loss: {val_loss:.4f}")
            
            tracker.add_loss(train_loss, val_loss)

        # ç¹ªè£½ Loss plot
        try:
            histories[target] = tracker
            
            plt.figure(figsize=(10, 6))
            plt.plot([train_loss], label="Train")
            plt.plot([val_loss], label="Valid")
            plt.title(f"Logloss for {target}")
            plt.xlabel("Model")
            plt.ylabel("Logloss")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(f"{model_path}/lossplot/loss_{target}.png")
            plt.close()
        except Exception as e:
            print(f"ç„¡æ³•ç¹ªè£½æ��å¤±æ›²ç·š: {str(e)}")

        # å„²å­˜æ¨¡å�‹
        models[target] = model
        joblib.dump(model, f"{model_path}/param/rf_{target}.pkl")
        
        # è¨ˆç®—ä¸¦é¡¯ç¤ºé€²åº¦
        target_time = time.time() - target_start
        target_times.append(target_time)
        avg_time_per_target = np.mean(target_times)
        remaining_targets = total_targets - (i + 1)
        estimated_remaining_time = avg_time_per_target * remaining_targets
        
        # è½‰æ�›é �ä¼°å‰©é¤˜æ™‚é–“ç‚ºæ›´æ˜“è®€æ ¼å¼�
        remaining_time_str = str(datetime.timedelta(seconds=int(estimated_remaining_time)))
        completion_time = datetime.datetime.now() + datetime.timedelta(seconds=estimated_remaining_time)
        
        print(f"ç›®æ¨™ {i+1}/{total_targets} å·²å®Œæˆ�! ({target})")
        print(f"å¹³å�‡æ¯�å€‹ç›®æ¨™è¨“ç·´æ™‚é–“: {avg_time_per_target:.2f} ç§’")
        print(f"é �ä¼°å‰©é¤˜æ™‚é–“: {remaining_time_str}")
        print(f"é �ä¼°å®Œæˆ�æ™‚é–“: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")

# æœ€çµ‚å®Œæˆ�è¨Šæ�¯
print("\nâœ… æ‰€æœ‰æ¨¡å�‹è¨“ç·´å·²å®Œæˆ�!")
total_time = time.time() - start_time
print(f"ç¸½è€—æ™‚: {datetime.timedelta(seconds=int(total_time))}")

# 6. æ�¨è«–èˆ‡å»ºç«‹æ��äº¤æª”æ¡ˆ
print("ğŸ”® é–‹å§‹é€²è¡Œé �æ¸¬...")
predictions = []

# é¡¯ç¤ºé �æ¸¬é€²åº¦
for i, target in tqdm(enumerate(target_columns), total=len(target_columns), desc='é �æ¸¬é€²åº¦'):
    model = models[target]  # å„ªå…ˆä½¿ç”¨è¨˜æ†¶é«”ä¸­çš„æ¨¡å�‹é�¿å…�é‡�è¤‡è¼‰å…¥
    
    try:
        # ä½¿ç”¨ GPU æ¨¡å�‹æ™‚éœ€è¦�ç‰¹æ®Šè™•ç�†
        if USE_GPU and hasattr(model, 'predict_proba') and 'cuml' in str(type(model)):
            # æ¸…ç�† GPU è¨˜æ†¶é«”
            cp.get_default_memory_pool().free_all_blocks()
            
            # åˆ†æ‰¹é �æ¸¬ä»¥é�¿å…�è¨˜æ†¶é«”å•�é¡Œ
            BATCH_SIZE = 5000
            all_preds = []
            
            for start_idx in range(0, len(X_test), BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, len(X_test))
                X_batch = X_test.iloc[start_idx:end_idx].values.astype(np.float32)
                
                with cp.cuda.Device(i % NUM_GPUS):  # é�¸æ“‡ GPU
                    batch_pred = model.predict_proba(X_batch)[:, 1]
                    if hasattr(batch_pred, 'get'):
                        batch_pred = batch_pred.get()
                    all_preds.append(batch_pred)
            
            # å�ˆä½µæ‰€æœ‰æ‰¹æ¬¡çš„é �æ¸¬çµ�æ�œ
            pred = np.concatenate(all_preds)
        else:
            # CPU æ¨¡å�‹ç›´æ�¥é �æ¸¬
            pred = model.predict_proba(X_test)[:, 1]
            
        predictions.append(pred)
    except Exception as e:
        print(f"\nâš ï¸� ç›®æ¨™ {target} çš„é �æ¸¬å¤±æ•—: {str(e)}")
        print("å˜—è©¦ä½¿ç”¨å‚™ç”¨æ–¹æ³•...")
        
        # ä½¿ç”¨ä¿�å®ˆçš„æ–¹æ³•å†�è©¦ä¸€æ¬¡
        try:
            if USE_GPU:
                # å˜—è©¦è½‰æ�›å›� CPU æ¨¡å�‹
                from sklearn.ensemble import RandomForestClassifier
                cpu_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42,
                    n_jobs=-1
                )
                # å¾�ç�¾æœ‰æ¨¡å�‹è¨“ç·´
                cpu_model.fit(X.values, Y[target].values)
                pred = cpu_model.predict_proba(X_test)[:, 1]
            else:
                # ä½¿ç”¨å�‡å€¼ä½œç‚ºé �æ¸¬ (æœ€å¾Œçš„é�¸é …)
                mean_val = Y[target].mean()
                pred = np.full(len(X_test), mean_val)
                
            print(f"å·²ä½¿ç”¨å‚™ç”¨æ–¹æ³•ç‚ºç›®æ¨™ {target} ç”Ÿæˆ�é �æ¸¬")
            predictions.append(pred)
        except Exception as backup_error:
            print(f"å‚™ç”¨æ–¹æ³•ä¹Ÿå¤±æ•—äº†: {str(backup_error)}")
            # ä½¿ç”¨å…¨ 0.5 é �æ¸¬
            pred = np.full(len(X_test), 0.5)
            predictions.append(pred)

predictions = np.array(predictions).T
submission = sample_submission.copy()
submission.iloc[:, 1:] = predictions
submission.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
print("ğŸ�‰ å·²ç”¢å‡º submission_rf.csv å�¯ç›´æ�¥ä¸Šå‚³åˆ° Kaggle")

# æ·»åŠ é©—è­‰é›†çš„é �æ¸¬
print("ğŸ”� é–‹å§‹é€²è¡Œé©—è­‰é›†é �æ¸¬...")
predictions_V = []

# Show prediction progress
for i, target in tqdm(enumerate(target_columns), total=len(target_columns), desc='Validation prediction progress'):
    try:
        model = models[target]  # Use in-memory models to avoid reloading
        
        # ä½¿ç”¨ GPU æ¨¡å�‹æ™‚éœ€è¦�ç‰¹æ®Šè™•ç�†
        if USE_GPU and hasattr(model, 'predict_proba') and 'cuml' in str(type(model)):
            # æ¸…ç�† GPU è¨˜æ†¶é«”
            cp.get_default_memory_pool().free_all_blocks()
            
            # åˆ†æ‰¹é �æ¸¬ä»¥é�¿å…�è¨˜æ†¶é«”å•�é¡Œ
            BATCH_SIZE = 5000
            all_preds = []
            
            for start_idx in range(0, len(X_valid), BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, len(X_valid))
                X_batch = X_valid.iloc[start_idx:end_idx].values.astype(np.float32)
                
                with cp.cuda.Device(i % NUM_GPUS):  # é�¸æ“‡ GPU
                    batch_pred = model.predict_proba(X_batch)
                    if isinstance(batch_pred, pd.DataFrame):
                        batch_pred = batch_pred.iloc[:, 1].values
                    elif hasattr(batch_pred, 'get'):
                        batch_pred = batch_pred.get()
                        batch_pred = batch_pred[:, 1]
                    else:
                        batch_pred = batch_pred[:, 1]
                    all_preds.append(batch_pred)
            
            # å�ˆä½µæ‰€æœ‰æ‰¹æ¬¡çš„é �æ¸¬çµ�æ�œ
            pred = np.concatenate(all_preds)
        else:
            # CPU æ¨¡å�‹ç›´æ�¥é �æ¸¬
            pred = model.predict_proba(X_valid)
            # å�‹æ…‹è‡ªå‹•è™•ç�†ï¼šDataFrameã€�cupyã€�numpy
            if isinstance(pred, pd.DataFrame):
                pred = pred.iloc[:, 1].values
            elif hasattr(pred, 'get'):
                pred = pred.get()
                pred = pred[:, 1]
            else:
                pred = pred[:, 1]
                
        predictions_V.append(pred)
    except Exception as e:
        print(f"\nâš ï¸� é©—è­‰é›†ç›®æ¨™ {target} çš„é �æ¸¬å¤±æ•—: {str(e)}")
        print("å˜—è©¦ä½¿ç”¨å‚™ç”¨æ–¹æ³•...")
        
        try:
            # ä½¿ç”¨å¹³å�‡å€¼ä½œç‚ºé �æ¸¬
            mean_val = Y[target].mean()
            pred = np.full(len(X_valid), mean_val)
            print(f"å·²ä½¿ç”¨å¹³å�‡å€¼ {mean_val:.4f} ä½œç‚ºç›®æ¨™ {target} çš„é �æ¸¬")
            predictions_V.append(pred)
        except Exception as backup_error:
            # ç·Šæ€¥å¾Œå‚™é�¸é …
            print(f"å‚™ç”¨æ–¹æ³•ä¹Ÿå¤±æ•—äº†ï¼š{str(backup_error)}")
            pred = np.full(len(X_valid), 0.5)
            predictions_V.append(pred)

predictions_V = np.array(predictions_V).T

# Create validation submission with correct sig_ids
validation_submission = pd.DataFrame(columns=Y_valid.columns)
validation_submission['sig_id'] = df_valid['sig_id']
for col in target_columns:
    validation_submission[col] = predictions_V[:, list(target_columns).index(col)]

# Save validation predictions to a separate file

validation_submission.to_csv(VALIDATION_OUTPUT_PATH, index=False)
print(f"ğŸ�‰ RandomForest é©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆå·²å»ºç«‹: {VALIDATION_OUTPUT_PATH}")

# æª¢è¦–é©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆçš„å‰�å¹¾è¡Œ
print("\né©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆé �è¦½:")
print(validation_submission.head())


# import shutil

# shutil.make_archive('/kaggle/working/param', 'zip', '/kaggle/working/param')


# # ğŸŒŸ MoA Random Forest Pipeline (15000 Sample + Feature Engineering + CV + Loss Plot)
# # é�©ç”¨æ–¼æœ¬æ©Ÿè™•ç�†åŸ·è¡Œï¼ŒåŒ…å�« å¤šæ ¸å¿ƒä¸¦è¡Œè™•ç�†èˆ‡è™›æ“¬ç’°å¢ƒæ�­å»º
# # è«‹å…ˆåŸ·è¡Œä¸‹åˆ— bash æŒ‡ä»¤å•Ÿå‹•è™›æ“¬ç’°å¢ƒï¼š
# # ----------------------------------------------------
# # conda create -n moa-rf python=3.10 -y
# # conda activate moa-rf
# # pip install numpy pandas scikit-learn matplotlib joblib
# # pip install tqdm
# # pip install iterative-stratification
# # pip install notebook
# # mkdir moa-rf-project
# # cd moa-rf-project
# # jupyter notebook
# # ----------------------------------------------------

# # è«‹ç¢ºèª�ä½ å·²å°‡ä»¥ä¸‹æª”æ¡ˆæ”¾å…¥ç•¶å‰�è³‡æ–™å¤¾ï¼š
# # - train_features.csv
# # - train_targets_scored.csv
# # - train_targets_nonscored.csv
# # - test_features.csv
# # - sample_submission.csv
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import joblib
# import time
# import datetime
# # print(f"scikit-learn ç‰ˆæœ¬: {sklearn_version}")
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import __version__ as sklearn_version
# from sklearn.calibration import CalibratedClassifierCV
# from tqdm.auto import tqdm  # å¼•å…¥tqdmç”¨æ–¼é€²åº¦è¿½è¹¤
# import multiprocessing as mp  # ç”¨æ–¼å¤šæ ¸å¿ƒåŠ é€Ÿ

# # Optional: ç”¨æ–¼ balanced multilabel CV
# from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# # è¨­å®šå�ƒæ•¸ - æ˜¯å�¦ä½¿ç”¨å·²è¨“ç·´å¥½çš„æ¨¡å�‹
# USE_PRETRAINED_MODELS = True  # è¨­ç‚ºTrueæ™‚ï¼Œå°‡å¾�output/paramè³‡æ–™å¤¾è¼‰å…¥å·²è¨“ç·´çš„æ¨¡å�‹
# OUTPUT_SUBMISSION_PATH = "/kaggle/working/submission_rf.csv"
# VALIDATION_OUTPUT_PATH = "/kaggle/working/validation_predictions_rf.csv"

# model_path = "/kaggle/input/moa-all-models/weights/output_param/rf_models"

# # 1. è®€å�–è³‡æ–™
# train_features = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")
# train_targets_scored = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")
# train_targets_nonscored = pd.read_csv("/kaggle/input/lish-moa/train_targets_nonscored.csv")
# test_features = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")
# sample_submission = pd.read_csv("/kaggle/input/lish-moa/sample_submission.csv")

# # 2. æŠ½æ¨£ 15000 ç­†è¨“ç·´è³‡æ–™
# df_sample = train_features.sample(n=15000, random_state=42).reset_index(drop=True)
# ids = df_sample['sig_id']
# Y = train_targets_scored[train_targets_scored['sig_id'].isin(ids)].reset_index(drop=True)

# # 3. ç‰¹å¾µå·¥ç¨‹
# def preprocess_features(df):
#     df = df.drop(columns=["sig_id"])
#     df = pd.get_dummies(df, columns=["cp_type", "cp_dose", "cp_time"])
#     return df

# X = preprocess_features(df_sample)
# X_test = preprocess_features(test_features)

# df_valid = train_features[~train_features['sig_id'].isin(ids)].reset_index(drop=True)
# Y_valid = train_targets_scored[train_targets_scored['sig_id'].isin(df_valid['sig_id'])].reset_index(drop=True)
# X_valid = preprocess_features(df_valid)

# # æ¨™æº–åŒ–ç‰¹å¾µ
# scaler = StandardScaler()
# X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
# X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
# X_valid = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns)

# # 4. Cross Validation è¨­å®šï¼ˆMultilabelStratifiedKFoldï¼‰
# N_SPLITS = 3
# mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, random_state=42, shuffle=True)

# models = {}
# histories = {}

# # å»ºç«‹å®Œæ•´è·¯å¾‘çš„å„²å­˜è³‡æ–™å¤¾
# # model_path = "/media/Pluto/richkung/EE6550_ML/Final/MyModels/output/rf_models"
# # os.makedirs(model_path, exist_ok=True)
# # os.makedirs(f"{model_path}/param", exist_ok=True)  # å„²å­˜æ¨¡å�‹å�ƒæ•¸çš„è³‡æ–™å¤¾
# # os.makedirs(f"{model_path}/lossplot", exist_ok=True)  # å„²å­˜lossåœ–çš„è³‡æ–™å¤¾

# # è¨­ç½®å�¯ç”¨æ ¸å¿ƒæ•¸é‡�
# num_cores = mp.cpu_count()
# print(f"ğŸš€ é–‹å§‹è¨“ç·´ - ä½¿ç”¨ {num_cores} å€‹ CPU æ ¸å¿ƒåŠ é€Ÿ")
# start_time = time.time()

# # æª¢æŸ¥ scikit-learn ç‰ˆæœ¬
# print(f"scikit-learn ç‰ˆæœ¬: {sklearn_version}")

# # è¨ˆç®—ç¸½ä»»å‹™æ•¸é‡�ï¼Œç”¨æ–¼é€²åº¦ä¼°ç®—
# target_columns = Y.columns[1:]
# total_targets = len(target_columns)
# print(f"ç¸½å…±éœ€è¦�è¨“ç·´ {total_targets} å€‹ç›®æ¨™")

# # è¿½è¹¤è¨“ç·´æ™‚é–“
# target_times = []

# class RFTracker:
#     """ç”¨æ–¼è¿½è¹¤Random Forestè¨“ç·´é��ç¨‹çš„é¡�åˆ¥"""
#     def __init__(self):
#         self.train_losses = []
#         self.valid_losses = []
    
#     def add_loss(self, train_loss, valid_loss):
#         self.train_losses.append(train_loss)
#         self.valid_losses.append(valid_loss)

# def compute_logloss(y_true, y_pred):
#     """è¨ˆç®—å°�æ•¸æ��å¤±"""
#     # ç¢ºä¿�é �æ¸¬å€¼åœ¨ (0, 1) ä¹‹é–“ï¼Œé�¿å…�æ•¸å€¼å•�é¡Œ
#     y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
#     loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
#     return loss

# # GPU è‡ªå‹•å�µæ¸¬
# try:
#     import cupy as cp
#     import cuml
#     from cuml.ensemble import RandomForestClassifier as cuRF
#     NUM_GPUS = cp.cuda.runtime.getDeviceCount()
#     USE_GPU = NUM_GPUS > 0
#     print(f"ğŸš€ å�µæ¸¬åˆ° {NUM_GPUS} é¡† GPUï¼Œå°‡å„ªå…ˆä½¿ç”¨ cuML GPU RandomForestï¼�")
# except Exception:
#     USE_GPU = False
#     NUM_GPUS = 0
#     print("ğŸ’» æœªå�µæ¸¬åˆ° GPU æˆ– cuMLï¼Œå°‡ä½¿ç”¨ sklearn CPU RandomForest")

# # 5. è¨“ç·´æ¨¡å�‹æˆ–è¼‰å…¥å·²æœ‰çš„æ¨¡å�‹
# if USE_PRETRAINED_MODELS:
#     print("ğŸ”„ ä½¿ç”¨å·²è¨“ç·´å¥½çš„æ¨¡å�‹ï¼Œå¾�ä¿�å­˜çš„æª”æ¡ˆä¸­è¼‰å…¥...")
    
#     # æª¢æŸ¥æ˜¯å�¦å­˜åœ¨æ¨¡å�‹æª”æ¡ˆ
#     model_exists = all(os.path.exists(f"{model_path}/param/rf_{target}.pkl") for target in target_columns)
    
#     if not model_exists:
#         print("â�Œ éŒ¯èª¤: æ‰¾ä¸�åˆ°æ‰€æœ‰éœ€è¦�çš„æ¨¡å�‹æª”æ¡ˆï¼Œè«‹è¨­å®š USE_PRETRAINED_MODELS=False é‡�æ–°è¨“ç·´")
#         exit(1)
    
#     # è¼‰å…¥å·²è¨“ç·´çš„æ¨¡å�‹
#     for i, target in tqdm(enumerate(target_columns), total=total_targets, desc='è¼‰å…¥æ¨¡å�‹'):
#         model_file = f"{model_path}/param/rf_{target}.pkl"
#         models[target] = joblib.load(model_file)
#         print(f"å·²è¼‰å…¥æ¨¡å�‹: {target}")
    
#     print("âœ… å·²æˆ�åŠŸè¼‰å…¥æ‰€æœ‰æ¨¡å�‹")
# else:
#     # æ­£å¸¸è¨“ç·´æµ�ç¨‹
#     for i, target in tqdm(enumerate(target_columns), total=total_targets, desc='æ•´é«”é€²åº¦'):
#         target_start = time.time()
        
#         print(f"\nè¨“ç·´æ¨¡å�‹ ({i+1}/{total_targets}): {target}")
#         y_target = Y[target].values
#         fold = 0
#         fold_logloss = []
        
#         # ä½¿ç”¨ train_test_split å‡½æ•¸é€²è¡Œåˆ†å‰²
#         indices = range(len(X))
        
#         # æ›´åš´æ ¼æª¢æŸ¥é¡�åˆ¥åˆ†å¸ƒï¼Œç¢ºä¿�æ¯�å€‹é¡�åˆ¥è‡³å°‘æœ‰ 2 ç­†è³‡æ–™
#         unique_values, counts = np.unique(y_target, return_counts=True)
#         min_count = counts.min() if len(counts) > 0 else 0
        
#         # å¦‚æ�œä»»ä½•é¡�åˆ¥çš„æ¨£æœ¬æ•¸å°‘æ–¼ 2 æˆ–è€…å�ªæœ‰ä¸€å€‹é¡�åˆ¥ï¼Œå‰‡ä¸�ä½¿ç”¨åˆ†å±¤æŠ½æ¨£
#         if min_count < 2 or len(unique_values) <= 1:
#             print(f"è­¦å‘Š: ç›®æ¨™ {target} çš„è³‡æ–™åˆ†å¸ƒä¸�å�‡è¡¡ï¼ŒæŸ�é¡�åˆ¥æ¨£æœ¬æ•¸é��å°‘ï¼Œå°‡ä¸�ä½¿ç”¨åˆ†å±¤æŠ½æ¨£")
#             stratify_data = None
#         else:
#             stratify_data = y_target
            
#         train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=stratify_data)
        
#         X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
#         y_train, y_val = y_target[train_indices], y_target[val_indices]

#         # æª¢æŸ¥é¡�åˆ¥æ˜¯å�¦ä¸�å¹³è¡¡
#         pos_rate = np.mean(y_train)
#         print(f"æ­£ä¾‹æ¯”ä¾‹: {pos_rate:.4f}")
        
#         # é‡�å°�ä¸�å¹³è¡¡è³‡æ–™èª¿æ•´æ¬Šé‡�
#         class_weight = None
#         if pos_rate < 0.2 or pos_rate > 0.8:
#             weight_ratio = (1 - pos_rate) / pos_rate if pos_rate < 0.5 else pos_rate / (1 - pos_rate)
#             class_weight = {0: 1, 1: weight_ratio} if pos_rate < 0.5 else {0: weight_ratio, 1: 1}
#             print(f"è³‡æ–™ä¸�å¹³è¡¡ï¼Œèª¿æ•´æ¬Šé‡�ç‚º: {weight_ratio}")
        
#         tracker = RFTracker()
        
#         try:
#             if USE_GPU:
#                 gpu_device = i % NUM_GPUS
#                 print(f"[GPU {gpu_device}] cuML RF è¨“ç·´ä¸­...", end=" ")
#                 with cp.cuda.Device(gpu_device):
#                     try:
#                         # èª¿æ•´å�ƒæ•¸ä»¥æ��é«˜ç©©å®šæ€§
#                         model = cuRF(
#                             n_estimators=100,  # æ¸›å°‘æ¨¹çš„æ•¸é‡�
#                             max_depth=6,       # æ¸›å°‘æ¨¹çš„æ·±åº¦
#                             n_streams=1,       # å�ªç”¨ä¸€å€‹æµ�ï¼Œå¢�åŠ ç©©å®šæ€§
#                             max_features=0.8,  # é™�åˆ¶ç‰¹å¾µæ•¸é‡�
#                             max_samples=0.8,   # ä½¿ç”¨éƒ¨åˆ†æ¨£æœ¬
#                             random_state=42,
#                             handle=None
#                         )
#                         # ç¢ºä¿�è³‡æ–™æ˜¯é€£çºŒçš„è¨˜æ†¶é«”å¡Š
#                         X_train_gpu = X_train.values.astype(np.float32, order='C')
#                         X_val_gpu = X_val.values.astype(np.float32, order='C')
                        
#                         # é‡‹æ”¾ä¸€äº›è¨˜æ†¶é«”
#                         cp.get_default_memory_pool().free_all_blocks()
                        
#                         # è¨“ç·´æ¨¡å�‹
#                         model.fit(X_train_gpu, y_train)
#                         train_proba = model.predict_proba(X_train_gpu)[:, 1]
#                         val_proba = model.predict_proba(X_val_gpu)[:, 1]
                        
#                         if hasattr(train_proba, 'get'):
#                             train_proba = train_proba.get()
#                             val_proba = val_proba.get()
#                     except Exception as gpu_error:
#                         print(f"\nâš ï¸� GPUè¨“ç·´å¤±æ•—: {str(gpu_error)}")
#                         print("å˜—è©¦ä½¿ç”¨æ›´ç°¡å–®çš„GPUè¨­ç½®...")
                        
#                         # é‡‹æ”¾è¨˜æ†¶é«”
#                         cp.get_default_memory_pool().free_all_blocks()
                        
#                         # å˜—è©¦ä½¿ç”¨æ›´ç°¡å–®çš„è¨­ç½®
#                         model = cuRF(
#                             n_estimators=50,
#                             max_depth=4,
#                             n_streams=1,
#                             max_features=0.6,
#                             max_samples=0.6,
#                             random_state=42,
#                             handle=None
#                         )
                        
#                         # æ¸›å°‘è³‡æ–™é‡�
#                         if len(X_train) > 5000:
#                             X_train_sample = X_train.sample(n=5000, random_state=42)
#                             y_train_sample = y_train[X_train_sample.index]
#                             X_train_gpu = X_train_sample.values.astype(np.float32, order='C')
#                         else:
#                             X_train_gpu = X_train.values.astype(np.float32, order='C')
#                             y_train_sample = y_train
                            
#                         X_val_gpu = X_val.values.astype(np.float32, order='C')
                        
#                         model.fit(X_train_gpu, y_train_sample)
#                         train_proba = model.predict_proba(X_train_gpu)[:, 1]
#                         val_proba = model.predict_proba(X_val_gpu)[:, 1]
                        
#                         if hasattr(train_proba, 'get'):
#                             train_proba = train_proba.get()
#                             val_proba = val_proba.get()
#             else:
#                 print("sklearn RF è¨“ç·´ä¸­...", end=" ")
#                 model = RandomForestClassifier(
#                     n_estimators=200,
#                     max_depth=8,
#                     class_weight=class_weight,
#                     random_state=42,
#                     n_jobs=-1
#                 )
#                 model.fit(X_train, y_train)
#                 train_proba = model.predict_proba(X_train)[:, 1]
#                 val_proba = model.predict_proba(X_val)[:, 1]
#             train_loss = compute_logloss(y_train, train_proba)
#             val_loss = compute_logloss(y_val, val_proba)
#             print(f"è¨“ç·´é›† Log Loss: {train_loss:.4f} | é©—è­‰é›† Log Loss: {val_loss:.4f}")
#             tracker.add_loss(train_loss, val_loss)
            
#         except Exception as e:
#             print(f"Random Forest è¨“ç·´å¤±æ•—: {str(e)}")
#             # å˜—è©¦ä½¿ç”¨è¼ƒç°¡å–®çš„éš¨æ©Ÿæ£®æ�—æ¨¡å�‹
#             print("å˜—è©¦ä½¿ç”¨ç°¡åŒ–çš„éš¨æ©Ÿæ£®æ�—æ¨¡å�‹...", end=" ")
#             model = RandomForestClassifier(
#                 n_estimators=50,
#                 max_depth=4,
#                 class_weight=class_weight,
#                 random_state=42,
#                 n_jobs=-1
#             )
            
#             model.fit(X_train, y_train)
            
#             # è¨ˆç®—è¨“ç·´é›†å’Œé©—è­‰é›†çš„æ��å¤±
#             train_proba = model.predict_proba(X_train)[:, 1]
#             val_proba = model.predict_proba(X_val)[:, 1]
            
#             train_loss = compute_logloss(y_train, train_proba)
#             val_loss = compute_logloss(y_val, val_proba)
            
#             print(f"è¨“ç·´é›† Log Loss: {train_loss:.4f} | é©—è­‰é›† Log Loss: {val_loss:.4f}")
            
#             tracker.add_loss(train_loss, val_loss)

#         # ç¹ªè£½ Loss plot
#         try:
#             histories[target] = tracker
            
#             plt.figure(figsize=(10, 6))
#             plt.plot([train_loss], label="Train")
#             plt.plot([val_loss], label="Valid")
#             plt.title(f"Logloss for {target}")
#             plt.xlabel("Model")
#             plt.ylabel("Logloss")
#             plt.legend()
#             plt.grid()
#             plt.tight_layout()
#             plt.savefig(f"{model_path}/lossplot/loss_{target}.png")
#             plt.close()
#         except Exception as e:
#             print(f"ç„¡æ³•ç¹ªè£½æ��å¤±æ›²ç·š: {str(e)}")

#         # å„²å­˜æ¨¡å�‹
#         models[target] = model
#         joblib.dump(model, f"{model_path}/param/rf_{target}.pkl")
        
#         # è¨ˆç®—ä¸¦é¡¯ç¤ºé€²åº¦
#         target_time = time.time() - target_start
#         target_times.append(target_time)
#         avg_time_per_target = np.mean(target_times)
#         remaining_targets = total_targets - (i + 1)
#         estimated_remaining_time = avg_time_per_target * remaining_targets
        
#         # è½‰æ�›é �ä¼°å‰©é¤˜æ™‚é–“ç‚ºæ›´æ˜“è®€æ ¼å¼�
#         remaining_time_str = str(datetime.timedelta(seconds=int(estimated_remaining_time)))
#         completion_time = datetime.datetime.now() + datetime.timedelta(seconds=estimated_remaining_time)
        
#         print(f"ç›®æ¨™ {i+1}/{total_targets} å·²å®Œæˆ�! ({target})")
#         print(f"å¹³å�‡æ¯�å€‹ç›®æ¨™è¨“ç·´æ™‚é–“: {avg_time_per_target:.2f} ç§’")
#         print(f"é �ä¼°å‰©é¤˜æ™‚é–“: {remaining_time_str}")
#         print(f"é �ä¼°å®Œæˆ�æ™‚é–“: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")

# # æœ€çµ‚å®Œæˆ�è¨Šæ�¯
# print("\nâœ… æ‰€æœ‰æ¨¡å�‹è¨“ç·´å·²å®Œæˆ�!")
# total_time = time.time() - start_time
# print(f"ç¸½è€—æ™‚: {datetime.timedelta(seconds=int(total_time))}")

# # 6. æ�¨è«–èˆ‡å»ºç«‹æ��äº¤æª”æ¡ˆ
# print("ğŸ”® é–‹å§‹é€²è¡Œé �æ¸¬...")
# predictions = []

# # é¡¯ç¤ºé �æ¸¬é€²åº¦
# for i, target in tqdm(enumerate(target_columns), total=len(target_columns), desc='é �æ¸¬é€²åº¦'):
#     model = models[target]  # å„ªå…ˆä½¿ç”¨è¨˜æ†¶é«”ä¸­çš„æ¨¡å�‹é�¿å…�é‡�è¤‡è¼‰å…¥
    
#     try:
#         # ä½¿ç”¨ GPU æ¨¡å�‹æ™‚éœ€è¦�ç‰¹æ®Šè™•ç�†
#         if USE_GPU and hasattr(model, 'predict_proba') and 'cuml' in str(type(model)):
#             # æ¸…ç�† GPU è¨˜æ†¶é«”
#             cp.get_default_memory_pool().free_all_blocks()
            
#             # åˆ†æ‰¹é �æ¸¬ä»¥é�¿å…�è¨˜æ†¶é«”å•�é¡Œ
#             BATCH_SIZE = 5000
#             all_preds = []
            
#             for start_idx in range(0, len(X_test), BATCH_SIZE):
#                 end_idx = min(start_idx + BATCH_SIZE, len(X_test))
#                 X_batch = X_test.iloc[start_idx:end_idx].values.astype(np.float32)
                
#                 with cp.cuda.Device(i % NUM_GPUS):  # é�¸æ“‡ GPU
#                     batch_pred = model.predict_proba(X_batch)[:, 1]
#                     if hasattr(batch_pred, 'get'):
#                         batch_pred = batch_pred.get()
#                     all_preds.append(batch_pred)
            
#             # å�ˆä½µæ‰€æœ‰æ‰¹æ¬¡çš„é �æ¸¬çµ�æ�œ
#             pred = np.concatenate(all_preds)
#         else:
#             # CPU æ¨¡å�‹ç›´æ�¥é �æ¸¬
#             pred = model.predict_proba(X_test)[:, 1]
            
#         predictions.append(pred)
#     except Exception as e:
#         print(f"\nâš ï¸� ç›®æ¨™ {target} çš„é �æ¸¬å¤±æ•—: {str(e)}")
#         print("å˜—è©¦ä½¿ç”¨å‚™ç”¨æ–¹æ³•...")
        
#         # ä½¿ç”¨ä¿�å®ˆçš„æ–¹æ³•å†�è©¦ä¸€æ¬¡
#         try:
#             if USE_GPU:
#                 # å˜—è©¦è½‰æ�›å›� CPU æ¨¡å�‹
#                 from sklearn.ensemble import RandomForestClassifier
#                 cpu_model = RandomForestClassifier(
#                     n_estimators=100,
#                     max_depth=6,
#                     random_state=42,
#                     n_jobs=-1
#                 )
#                 # å¾�ç�¾æœ‰æ¨¡å�‹è¨“ç·´
#                 cpu_model.fit(X.values, Y[target].values)
#                 pred = cpu_model.predict_proba(X_test)[:, 1]
#             else:
#                 # ä½¿ç”¨å�‡å€¼ä½œç‚ºé �æ¸¬ (æœ€å¾Œçš„é�¸é …)
#                 mean_val = Y[target].mean()
#                 pred = np.full(len(X_test), mean_val)
                
#             print(f"å·²ä½¿ç”¨å‚™ç”¨æ–¹æ³•ç‚ºç›®æ¨™ {target} ç”Ÿæˆ�é �æ¸¬")
#             predictions.append(pred)
#         except Exception as backup_error:
#             print(f"å‚™ç”¨æ–¹æ³•ä¹Ÿå¤±æ•—äº†: {str(backup_error)}")
#             # ä½¿ç”¨å…¨ 0.5 é �æ¸¬
#             pred = np.full(len(X_test), 0.5)
#             predictions.append(pred)

# predictions = np.array(predictions).T
# submission = sample_submission.copy()
# submission.iloc[:, 1:] = predictions
# submission.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
# print("ğŸ�‰ å·²ç”¢å‡º submission_rf.csv å�¯ç›´æ�¥ä¸Šå‚³åˆ° Kaggle")

# # æ·»åŠ é©—è­‰é›†çš„é �æ¸¬
# print("ğŸ”� é–‹å§‹é€²è¡Œé©—è­‰é›†é �æ¸¬...")
# predictions_V = []

# # Show prediction progress
# for i, target in tqdm(enumerate(target_columns), total=len(target_columns), desc='Validation prediction progress'):
#     try:
#         model = models[target]  # Use in-memory models to avoid reloading
        
#         # ä½¿ç”¨ GPU æ¨¡å�‹æ™‚éœ€è¦�ç‰¹æ®Šè™•ç�†
#         if USE_GPU and hasattr(model, 'predict_proba') and 'cuml' in str(type(model)):
#             # æ¸…ç�† GPU è¨˜æ†¶é«”
#             cp.get_default_memory_pool().free_all_blocks()
            
#             # åˆ†æ‰¹é �æ¸¬ä»¥é�¿å…�è¨˜æ†¶é«”å•�é¡Œ
#             BATCH_SIZE = 5000
#             all_preds = []
            
#             for start_idx in range(0, len(X_valid), BATCH_SIZE):
#                 end_idx = min(start_idx + BATCH_SIZE, len(X_valid))
#                 X_batch = X_valid.iloc[start_idx:end_idx].values.astype(np.float32)
                
#                 with cp.cuda.Device(i % NUM_GPUS):  # é�¸æ“‡ GPU
#                     batch_pred = model.predict_proba(X_batch)
#                     if isinstance(batch_pred, pd.DataFrame):
#                         batch_pred = batch_pred.iloc[:, 1].values
#                     elif hasattr(batch_pred, 'get'):
#                         batch_pred = batch_pred.get()
#                         batch_pred = batch_pred[:, 1]
#                     else:
#                         batch_pred = batch_pred[:, 1]
#                     all_preds.append(batch_pred)
            
#             # å�ˆä½µæ‰€æœ‰æ‰¹æ¬¡çš„é �æ¸¬çµ�æ�œ
#             pred = np.concatenate(all_preds)
#         else:
#             # CPU æ¨¡å�‹ç›´æ�¥é �æ¸¬
#             pred = model.predict_proba(X_valid)
#             # å�‹æ…‹è‡ªå‹•è™•ç�†ï¼šDataFrameã€�cupyã€�numpy
#             if isinstance(pred, pd.DataFrame):
#                 pred = pred.iloc[:, 1].values
#             elif hasattr(pred, 'get'):
#                 pred = pred.get()
#                 pred = pred[:, 1]
#             else:
#                 pred = pred[:, 1]
                
#         predictions_V.append(pred)
#     except Exception as e:
#         print(f"\nâš ï¸� é©—è­‰é›†ç›®æ¨™ {target} çš„é �æ¸¬å¤±æ•—: {str(e)}")
#         print("å˜—è©¦ä½¿ç”¨å‚™ç”¨æ–¹æ³•...")
        
#         try:
#             # ä½¿ç”¨å¹³å�‡å€¼ä½œç‚ºé �æ¸¬
#             mean_val = Y[target].mean()
#             pred = np.full(len(X_valid), mean_val)
#             print(f"å·²ä½¿ç”¨å¹³å�‡å€¼ {mean_val:.4f} ä½œç‚ºç›®æ¨™ {target} çš„é �æ¸¬")
#             predictions_V.append(pred)
#         except Exception as backup_error:
#             # ç·Šæ€¥å¾Œå‚™é�¸é …
#             print(f"å‚™ç”¨æ–¹æ³•ä¹Ÿå¤±æ•—äº†ï¼š{str(backup_error)}")
#             pred = np.full(len(X_valid), 0.5)
#             predictions_V.append(pred)

# predictions_V = np.array(predictions_V).T

# # Create validation submission with correct sig_ids
# validation_submission = pd.DataFrame(columns=Y_valid.columns)
# validation_submission['sig_id'] = df_valid['sig_id']
# for col in target_columns:
#     validation_submission[col] = predictions_V[:, list(target_columns).index(col)]

# # Save validation predictions to a separate file

# validation_submission.to_csv(VALIDATION_OUTPUT_PATH, index=False)
# print(f"ğŸ�‰ RandomForest é©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆå·²å»ºç«‹: {VALIDATION_OUTPUT_PATH}")

# # æª¢è¦–é©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆçš„å‰�å¹¾è¡Œ
# print("\né©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆé �è¦½:")
# print(validation_submission.head())


# !pip install tensorflow


# ğŸŒŸ MoA Neural Network Pipeline (15000 Sample + Feature Engineering + CV + Loss Plot)
# é�©ç”¨æ–¼æœ¬æ©Ÿè™•ç�†åŸ·è¡Œï¼ŒåŒ…å�« GPUåŠ é€Ÿ èˆ‡è™›æ“¬ç’°å¢ƒæ�­å»º
# è«‹å…ˆåŸ·è¡Œä¸‹åˆ— bash æŒ‡ä»¤å•Ÿå‹•è™›æ“¬ç’°å¢ƒï¼š
# ----------------------------------------------------
# conda create -n moa-nn python=3.10 -y
# conda activate moa-nn
# pip install numpy pandas scikit-learn matplotlib joblib
# pip install tensorflow  # æˆ– pip install tensorflow-gpu å¦‚æ�œæœ‰ NVIDIA é¡¯å�¡
# pip install tqdm
# pip install iterative-stratification
# pip install notebook
# mkdir moa-nn-project
# cd moa-nn-project
# jupyter notebook
# ----------------------------------------------------

# è«‹ç¢ºèª�ä½ å·²å°‡ä»¥ä¸‹æª”æ¡ˆæ”¾å…¥ç•¶å‰�è³‡æ–™å¤¾ï¼š
# - train_features.csv
# - train_targets_scored.csv
# - train_targets_nonscored.csv
# - test_features.csv
# - sample_submission.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tqdm.auto import tqdm  # å¼•å…¥tqdmç”¨æ–¼é€²åº¦è¿½è¹¤
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Optional: ç”¨æ–¼ balanced multilabel CV
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# è¨­å®šå�ƒæ•¸ - æ˜¯å�¦ä½¿ç”¨å·²è¨“ç·´å¥½çš„æ¨¡å�‹
USE_PRETRAINED_MODELS = True  # è¨­ç‚ºTrueæ™‚ï¼Œå°‡å¾�output/paramè³‡æ–™å¤¾è¼‰å…¥å·²è¨“ç·´çš„æ¨¡å�‹

# ================= è·¯å¾‘è¨­å®šå�€ =================
DATA_ROOT = "/kaggle/input/lish-moa/"
MYMODELS_ROOT = "/kaggle/working/"
CSV_ROOT = "/kaggle/working/"
MODEL_OUTPUT_ROOT = "/kaggle/input/moa-all-models/weights/output_param/nn_models"
TRAIN_FEATURES_PATH = os.path.join(DATA_ROOT, "train_features.csv")
TRAIN_TARGETS_SCORED_PATH = os.path.join(DATA_ROOT, "train_targets_scored.csv")
TRAIN_TARGETS_NONSCORED_PATH = os.path.join(DATA_ROOT, "train_targets_nonscored.csv")
TEST_FEATURES_PATH = os.path.join(DATA_ROOT, "test_features.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
OUTPUT_SUBMISSION_PATH = os.path.join(CSV_ROOT, "submission_nn.csv")
VALIDATION_OUTPUT_PATH = os.path.join(CSV_ROOT, "validation_predictions_nn.csv")
# =============================================

# æª¢æŸ¥æ˜¯å�¦æœ‰å�¯ç”¨GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # è¨­å®šè¨˜æ†¶é«”å¢�é•·é™�åˆ¶
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"ğŸš€ ä½¿ç”¨GPUåŠ é€Ÿ: {len(gpus)}å€‹GPUå�¯ç”¨")
    except RuntimeError as e:
        print(f"GPUè¨­å®šå¤±æ•—: {e}")
else:
    print("âš ï¸� æ²’æœ‰å�¯ç”¨çš„GPUï¼Œå°‡ä½¿ç”¨CPUé€²è¡Œè¨“ç·´ï¼Œé€Ÿåº¦è¼ƒæ…¢")

# 1. è®€å�–è³‡æ–™
train_features = pd.read_csv(TRAIN_FEATURES_PATH)
train_targets_scored = pd.read_csv(TRAIN_TARGETS_SCORED_PATH)
train_targets_nonscored = pd.read_csv(TRAIN_TARGETS_NONSCORED_PATH)
test_features = pd.read_csv(TEST_FEATURES_PATH)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

# 2. æŠ½æ¨£ 15000 ç­†è¨“ç·´è³‡æ–™
df_sample = train_features.sample(n=15000, random_state=42).reset_index(drop=True)
ids = df_sample['sig_id']
Y = train_targets_scored[train_targets_scored['sig_id'].isin(ids)].reset_index(drop=True)
df_valid = train_features[~train_features['sig_id'].isin(ids)].reset_index(drop=True)
Y_valid = train_targets_scored[train_targets_scored['sig_id'].isin(df_valid['sig_id'])].reset_index(drop=True)

# 3. ç‰¹å¾µå·¥ç¨‹
def preprocess_features(df):
    df = df.drop(columns=["sig_id"])
    df = pd.get_dummies(df, columns=["cp_type", "cp_dose", "cp_time"])
    return df

X = preprocess_features(df_sample)
X_test = preprocess_features(test_features)
X_valid = preprocess_features(df_valid)
# æ¨™æº–åŒ–ç‰¹å¾µ
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns).astype(np.float32)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns).astype(np.float32)
X_valid = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns).astype(np.float32)

# 4. Cross Validation è¨­å®šï¼ˆMultilabelStratifiedKFoldï¼‰
N_SPLITS = 3
mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, random_state=42, shuffle=True)

models = {}
histories = {}

# å»ºç«‹å®Œæ•´è·¯å¾‘çš„å„²å­˜è³‡æ–™å¤¾
model_path = MODEL_OUTPUT_ROOT
# os.makedirs(model_path, exist_ok=True)
# os.makedirs(f"{model_path}/param", exist_ok=True)  # å„²å­˜æ¨¡å�‹å�ƒæ•¸çš„è³‡æ–™å¤¾
# os.makedirs(f"{model_path}/lossplot", exist_ok=True)  # å„²å­˜lossåœ–çš„è³‡æ–™å¤¾

# é¡¯ç¤ºTensorFlowç‰ˆæœ¬
tf_version = tf.__version__
print(f"TensorFlow ç‰ˆæœ¬: {tf_version}")
start_time = time.time()

# è¨ˆç®—ç¸½ä»»å‹™æ•¸é‡�ï¼Œç”¨æ–¼é€²åº¦ä¼°ç®—
target_columns = Y.columns[1:]
total_targets = len(target_columns)
print(f"ç¸½å…±éœ€è¦�è¨“ç·´ {total_targets} å€‹ç›®æ¨™")

# è¿½è¹¤è¨“ç·´æ™‚é–“
target_times = []

class NNTracker:
    """ç”¨æ–¼è¿½è¹¤ç¥�ç¶“ç¶²çµ¡è¨“ç·´é��ç¨‹çš„é¡�åˆ¥"""
    def __init__(self):
        self.history = None
    
    def add_history(self, history):
        self.history = history

def compute_logloss(y_true, y_pred):
    """è¨ˆç®—å°�æ•¸æ��å¤±"""
    # ç¢ºä¿�é �æ¸¬å€¼åœ¨ (0, 1) ä¹‹é–“ï¼Œé�¿å…�æ•¸å€¼å•�é¡Œ
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def build_model(input_dim, learning_rate=0.001):
    """å»ºç«‹ç¥�ç¶“ç¶²çµ¡æ¨¡å�‹"""
    model = Sequential([
        # è¼¸å…¥å±¤
        Dense(256, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # éš±è—�å±¤1
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # éš±è—�å±¤2
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # è¼¸å‡ºå±¤ (äºŒå…ƒåˆ†é¡�)
        Dense(1, activation='sigmoid')
    ])
    
    # ç·¨è­¯æ¨¡å�‹
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 5. è¨“ç·´æ¨¡å�‹æˆ–è¼‰å…¥å·²æœ‰çš„æ¨¡å�‹
if USE_PRETRAINED_MODELS:
    print("ğŸ”„ ä½¿ç”¨å·²è¨“ç·´å¥½çš„æ¨¡å�‹ï¼Œå¾�ä¿�å­˜çš„æª”æ¡ˆä¸­è¼‰å…¥...")
    
    # æª¢æŸ¥æ˜¯å�¦å­˜åœ¨æ¨¡å�‹æª”æ¡ˆ
    model_exists = all(os.path.exists(f"{model_path}/param/nn_{target}.h5") for target in target_columns)
    
    if not model_exists:
        print("â�Œ éŒ¯èª¤: æ‰¾ä¸�åˆ°æ‰€æœ‰éœ€è¦�çš„æ¨¡å�‹æª”æ¡ˆï¼Œè«‹è¨­å®š USE_PRETRAINED_MODELS=False é‡�æ–°è¨“ç·´")
        exit(1)
    
    # è¼‰å…¥å·²è¨“ç·´çš„æ¨¡å�‹
    for i, target in tqdm(enumerate(target_columns), total=total_targets, desc='è¼‰å…¥æ¨¡å�‹'):
        model_file = f"{model_path}/param/nn_{target}.h5"
        models[target] = load_model(model_file)
        print(f"å·²è¼‰å…¥æ¨¡å�‹: {target}")
    
    print("âœ… å·²æˆ�åŠŸè¼‰å…¥æ‰€æœ‰æ¨¡å�‹")
else:
    # æ­£å¸¸è¨“ç·´æµ�ç¨‹
    for i, target in tqdm(enumerate(target_columns), total=total_targets, desc='æ•´é«”é€²åº¦'):
        target_start = time.time()
        
        print(f"\nè¨“ç·´æ¨¡å�‹ ({i+1}/{total_targets}): {target}")
        y_target = Y[target].values
        fold = 0
        fold_logloss = []
        
        # ä½¿ç”¨ train_test_split å‡½æ•¸é€²è¡Œåˆ†å‰²
        indices = range(len(X))
        
        # æ›´åš´æ ¼æª¢æŸ¥é¡�åˆ¥åˆ†å¸ƒï¼Œç¢ºä¿�æ¯�å€‹é¡�åˆ¥è‡³å°‘æœ‰ 2 ç­†è³‡æ–™
        unique_values, counts = np.unique(y_target, return_counts=True)
        min_count = counts.min() if len(counts) > 0 else 0
        
        # å¦‚æ�œä»»ä½•é¡�åˆ¥çš„æ¨£æœ¬æ•¸å°‘æ–¼ 2 æˆ–è€…å�ªæœ‰ä¸€å€‹é¡�åˆ¥ï¼Œå‰‡ä¸�ä½¿ç”¨åˆ†å±¤æŠ½æ¨£
        if min_count < 2 or len(unique_values) <= 1:
            print(f"è­¦å‘Š: ç›®æ¨™ {target} çš„è³‡æ–™åˆ†å¸ƒä¸�å�‡è¡¡ï¼ŒæŸ�é¡�åˆ¥æ¨£æœ¬æ•¸é��å°‘ï¼Œå°‡ä¸�ä½¿ç”¨åˆ†å±¤æŠ½æ¨£")
            stratify_data = None
        else:
            stratify_data = y_target
            
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=stratify_data)
        
        X_train, X_val = X.iloc[train_indices].values, X.iloc[val_indices].values
        y_train, y_val = y_target[train_indices], y_target[val_indices]

        # æª¢æŸ¥é¡�åˆ¥æ˜¯å�¦ä¸�å¹³è¡¡
        pos_rate = np.mean(y_train)
        print(f"æ­£ä¾‹æ¯”ä¾‹: {pos_rate:.4f}")
        
        # é‡�å°�ä¸�å¹³è¡¡è³‡æ–™èª¿æ•´æ¬Šé‡�
        class_weight = None
        if pos_rate < 0.2 or pos_rate > 0.8:
            weight_ratio = (1 - pos_rate) / pos_rate if pos_rate < 0.5 else pos_rate / (1 - pos_rate)
            # ç‚ºkerasæº–å‚™é¡�åˆ¥æ¬Šé‡�
            class_weight = {0: 1, 1: weight_ratio} if pos_rate < 0.5 else {0: weight_ratio, 1: 1}
            print(f"è³‡æ–™ä¸�å¹³è¡¡ï¼Œèª¿æ•´æ¬Šé‡�ç‚º: {weight_ratio:.2f}")
        
        tracker = NNTracker()
        
        # è¨­å®šæ—©å�œå�ƒæ•¸
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        # è¨­å®šå­¸ç¿’ç�‡æ¸›å°‘ç­–ç•¥
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=0
        )
        
        try:
            # ä½¿ç”¨ç¥�ç¶“ç¶²çµ¡åˆ†é¡�å™¨
            print(f"è¨“ç·´æ¨¡å�‹ä¸­...", end=" ")
            # ç�²å�–ç‰¹å¾µæ•¸é‡�
            input_dim = X_train.shape[1]
            
            # å‰µå»ºæ¨¡å�‹
            model = build_model(input_dim)
            
            # è¨“ç·´æ¨¡å�‹
            history = model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, reduce_lr],
                class_weight=class_weight,
                verbose=0  # ä¸�é¡¯ç¤ºè¨“ç·´é€²åº¦
            )
            
            # è¨ˆç®—è¨“ç·´é›†å’Œé©—è­‰é›†çš„æ��å¤±
            train_proba = model.predict(X_train, verbose=0).flatten()
            val_proba = model.predict(X_val, verbose=0).flatten()
            
            train_loss = compute_logloss(y_train, train_proba)
            val_loss = compute_logloss(y_val, val_proba)
            
            print(f"è¨“ç·´é›† Log Loss: {train_loss:.4f} | é©—è­‰é›† Log Loss: {val_loss:.4f}")
            
            tracker.add_history(history)
            
        except Exception as e:
            print(f"ç¥�ç¶“ç¶²çµ¡ è¨“ç·´å¤±æ•—: {str(e)}")
            # å˜—è©¦ä½¿ç”¨è¼ƒç°¡å–®çš„ç¥�ç¶“ç¶²çµ¡æ¨¡å�‹
            print("å˜—è©¦ä½¿ç”¨ç°¡åŒ–çš„ç¥�ç¶“ç¶²çµ¡æ¨¡å�‹...", end=" ")
            
            # å‰µå»ºç°¡å–®æ¨¡å�‹
            model = Sequential([
                Dense(64, input_dim=X_train.shape[1], activation='relu'),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.01),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # ç°¡å–®è¨“ç·´
            history = model.fit(
                X_train, y_train,
                epochs=30,
                batch_size=64,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                class_weight=class_weight,
                verbose=0
            )
            
            # è¨ˆç®—è¨“ç·´é›†å’Œé©—è­‰é›†çš„æ��å¤±
            train_proba = model.predict(X_train, verbose=0).flatten()
            val_proba = model.predict(X_val, verbose=0).flatten()
            
            train_loss = compute_logloss(y_train, train_proba)
            val_loss = compute_logloss(y_val, val_proba)
            
            print(f"è¨“ç·´é›† Log Loss: {train_loss:.4f} | é©—è­‰é›† Log Loss: {val_loss:.4f}")
            
            tracker.add_history(history)

        # ç¹ªè£½ Loss plot
        try:
            histories[target] = tracker
            
            if hasattr(tracker, 'history') and tracker.history is not None:
                history = tracker.history.history
                
                plt.figure(figsize=(12, 5))
                
                # ç¹ªè£½æ��å¤±æ›²ç·š
                plt.subplot(1, 2, 1)
                plt.plot(history['loss'], label='Train Loss')
                plt.plot(history['val_loss'], label='Validation Loss')
                plt.title(f'Loss Curve for {target}')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                
                # ç¹ªè£½æº–ç¢ºç�‡æ›²ç·š
                plt.subplot(1, 2, 2)
                plt.plot(history['accuracy'], label='Train Accuracy')
                plt.plot(history['val_accuracy'], label='Validation Accuracy')
                plt.title('Accuracy Curve')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(f"{model_path}/lossplot/loss_{target}.png")
                plt.close()
            else:
                # å¦‚æ�œæ²’æœ‰è¨“ç·´é��ç¨‹çš„æ›²ç·šï¼Œå°±ç¹ªè£½æœ€çµ‚çµ�æ�œ
                plt.figure(figsize=(10, 6))
                plt.bar(['Train Loss', 'Valid Loss'], [train_loss, val_loss])
                plt.title(f"Final Loss for {target}")
                plt.ylabel("Log Loss")
                plt.grid(axis='y')
                plt.tight_layout()
                plt.savefig(f"{model_path}/lossplot/loss_{target}.png")
                plt.close()
                
        except Exception as e:
            print(f"ç„¡æ³•ç¹ªè£½æ��å¤±æ›²ç·š: {str(e)}")

        # å„²å­˜æ¨¡å�‹
        models[target] = model
        model.save(f"{model_path}/param/nn_{target}.h5")
        
        # è¨ˆç®—ä¸¦é¡¯ç¤ºé€²åº¦
        target_time = time.time() - target_start
        target_times.append(target_time)
        avg_time_per_target = np.mean(target_times)
        remaining_targets = total_targets - (i + 1)
        estimated_remaining_time = avg_time_per_target * remaining_targets
        
        # è½‰æ�›é �ä¼°å‰©é¤˜æ™‚é–“ç‚ºæ›´æ˜“è®€æ ¼å¼�
        remaining_time_str = str(datetime.timedelta(seconds=int(estimated_remaining_time)))
        completion_time = datetime.datetime.now() + datetime.timedelta(seconds=estimated_remaining_time)
        
        print(f"ç›®æ¨™ {i+1}/{total_targets} å·²å®Œæˆ�! ({target})")
        print(f"å¹³å�‡æ¯�å€‹ç›®æ¨™è¨“ç·´æ™‚é–“: {avg_time_per_target:.2f} ç§’")
        print(f"é �ä¼°å‰©é¤˜æ™‚é–“: {remaining_time_str}")
        print(f"é �ä¼°å®Œæˆ�æ™‚é–“: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")

# æœ€çµ‚å®Œæˆ�è¨Šæ�¯
print("\nâœ… æ‰€æœ‰æ¨¡å�‹è¨“ç·´å·²å®Œæˆ�!")
total_time = time.time() - start_time
print(f"ç¸½è€—æ™‚: {datetime.timedelta(seconds=int(total_time))}")

# 6. æ�¨è«–èˆ‡å»ºç«‹æ��äº¤æª”æ¡ˆ
print("ğŸ”® é–‹å§‹é€²è¡Œé �æ¸¬...")
predictions = []

# å°‡æ¸¬è©¦æ•¸æ“šè½‰æ�›ç‚ºnumpyæ•¸çµ„
X_test_array = X_test.values

# é¡¯ç¤ºé �æ¸¬é€²åº¦
for i, target in tqdm(enumerate(target_columns), total=len(target_columns), desc='é �æ¸¬é€²åº¦'):
    model = models[target]  # å„ªå…ˆä½¿ç”¨è¨˜æ†¶é«”ä¸­çš„æ¨¡å�‹é�¿å…�é‡�è¤‡è¼‰å…¥
    pred = model.predict(X_test_array, verbose=0).flatten()
    predictions.append(pred)

predictions = np.array(predictions).T
submission = sample_submission.copy()
submission.iloc[:, 1:] = predictions
submission.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
print("ğŸ�‰ å·²ç”¢å‡º submission_nn.csv å�¯ç›´æ�¥ä¸Šå‚³åˆ° Kaggle")

# æ·»åŠ é©—è­‰é›†çš„é �æ¸¬
print("ğŸ”� é–‹å§‹é€²è¡Œé©—è­‰é›†é �æ¸¬...")
predictions_V = []

# Show prediction progress
for i, target in tqdm(enumerate(target_columns), total=len(target_columns), desc='Validation prediction progress'):
    model = models[target]  # Use in-memory models to avoid reloading
    pred = model.predict(X_valid.values.astype(np.float32), verbose=0).flatten()  # å¼·åˆ¶è½‰å�‹ float32
    predictions_V.append(pred)

predictions_V = np.array(predictions_V).T

# Create validation submission with correct sig_ids
validation_submission = pd.DataFrame(columns=Y_valid.columns)
validation_submission['sig_id'] = df_valid['sig_id']
for col in target_columns:
    validation_submission[col] = predictions_V[:, list(target_columns).index(col)]

# Save validation predictions to a separate file
validation_submission.to_csv(VALIDATION_OUTPUT_PATH, index=False)
print(f"ğŸ�‰ NN é©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆå·²å»ºç«‹: {VALIDATION_OUTPUT_PATH}")


import pandas as pd

# è®€å�–æ¯�å€‹ submission æª”æ¡ˆ
files = [
    # "/kaggle/working/submission_cat.csv",
    "/kaggle/working/submission_lgbm.csv",
    "/kaggle/working/submission_nn.csv",
    "/kaggle/working/submission_rf.csv",
    "/kaggle/working/submission_xgb.csv",
    "/kaggle/working/submission_svm.csv"
]

# å…ˆè®€ç¬¬ä¸€å€‹ç•¶ä½œ base
df_ensemble = pd.read_csv(files[0])
df_ensemble.iloc[:, 1:] = 0  # æŠŠé �æ¸¬æ¬„è¨­ç‚º 0ï¼Œç”¨ä¾†åŠ ç¸½ç”¨

# åŠ ç¸½æ‰€æœ‰ submission çš„é �æ¸¬æ¬„ï¼ˆä¸�å�« idï¼‰
for file in files:
    df = pd.read_csv(file)
    df_ensemble.iloc[:, 1:] += df.iloc[:, 1:]

# å�–å¹³å�‡
df_ensemble.iloc[:, 1:] /= len(files)

# å„²å­˜ç‚ºæ–°çš„ submission æª”æ¡ˆ
df_ensemble.to_csv("/kaggle/working/submission.csv", index=False)

print("æˆ�åŠŸè��å�ˆæˆ� submission.csv ï½� ğŸ�¥âœ¨")


