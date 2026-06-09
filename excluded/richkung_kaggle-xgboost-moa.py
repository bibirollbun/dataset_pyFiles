import sys
sys.path.append('/kaggle/input/iterative-stratification')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# %% [code]
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
OUTPUT_SUBMISSION_PATH = "submission.csv"  # Output path within the Kaggle environment

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
model_path = "/kaggle/input/moa-xgb-model-param"
os.makedirs(model_path, exist_ok=True)
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
VALIDATION_OUTPUT_PATH = "validation_predictions.csv"
validation_submission.to_csv(VALIDATION_OUTPUT_PATH, index=False)
print(f"ğŸ�‰ é©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆå·²å»ºç«‹: {VALIDATION_OUTPUT_PATH}")

# æª¢è¦–é©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆçš„å‰�å¹¾è¡Œ
print("\né©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆé �è¦½:")
validation_submission

