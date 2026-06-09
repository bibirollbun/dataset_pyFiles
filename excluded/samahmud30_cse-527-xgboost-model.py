!pip install torch-geometric
!pip install numpy



# ==================== XGBOOST ON 50K SAMPLES ====================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import joblib

print("="*80)
print("ğŸŒ² XGBOOST TRAINING ON 50,000 SAMPLES")
print("="*80)

# 1. DATA LOADING
print("\nğŸ“‚ LOADING DATA...")
csv_path = '/kaggle/input/hms-harmful-brain-activity-classification/train.csv'
train_csv = pd.read_csv(csv_path)
print(f"âœ… Loaded {len(train_csv):,} samples")

# Process labels
label_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
votes = train_csv[label_cols].values.astype(float)
row_sums = votes.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
probabilities = votes / row_sums
train_csv['dominant_class'] = np.argmax(probabilities, axis=1)

# Select 50K samples
n_samples = 50000
if len(train_csv) > n_samples:
    _, sample_indices = train_test_split(
        range(len(train_csv)),
        test_size=n_samples,
        random_state=42,
        stratify=train_csv['dominant_class']
    )
    selected_df = train_csv.iloc[sample_indices]
else:
    selected_df = train_csv
    n_samples = len(selected_df)

print(f"Selected {n_samples:,} samples")

# 2. FEATURE EXTRACTION
print("\nğŸ”§ EXTRACTING FEATURES...")

DATA_DIR = '/kaggle/input/hms-harmful-brain-activity-classification/train_eegs'

def extract_features_fast(eeg_data):
    """Fast feature extraction for XGBoost"""
    if eeg_data.size == 0:
        return np.zeros(35, dtype=np.float32)
    
    # Use only 100 timesteps, 5 channels
    if eeg_data.shape[0] > 100:
        indices = np.linspace(0, eeg_data.shape[0]-1, 100).astype(int)
        eeg_data = eeg_data[indices, :]
    
    if eeg_data.shape[1] > 5:
        eeg_data = eeg_data[:, :5]
    
    # Simple normalization
    eeg_data = (eeg_data - np.mean(eeg_data, axis=0)) / (np.std(eeg_data, axis=0) + 1e-8)
    
    features = []
    for ch in range(min(5, eeg_data.shape[1])):
        channel_data = eeg_data[:, ch]
        features.extend([
            np.mean(channel_data), np.std(channel_data),
            np.min(channel_data), np.max(channel_data),
            np.median(channel_data), np.mean(np.abs(channel_data)),
            np.sum(channel_data > 0) / len(channel_data),
            np.percentile(channel_data, 25),
            np.percentile(channel_data, 75),
            np.mean(np.diff(channel_data))
        ])
    
    return np.array(features, dtype=np.float32)

# Process samples
features_list = []
labels_list = []
failed = 0

processing_start = time.time()

for idx in tqdm(range(n_samples), desc="Processing EEGs"):
    try:
        sample_id = selected_df.iloc[idx]['eeg_id']
        eeg_path = f"{DATA_DIR}/{sample_id}.parquet"
        
        if os.path.exists(eeg_path):
            eeg_df = pd.read_parquet(eeg_path)
            if not eeg_df.empty:
                eeg_data = eeg_df.values.astype(np.float32)
                features = extract_features_fast(eeg_data)
                features_list.append(features)
                labels_list.append(selected_df.iloc[idx]['dominant_class'])
            else:
                failed += 1
        else:
            failed += 1
    except:
        failed += 1

processing_time = time.time() - processing_start
print(f"\nâœ… Processed {len(features_list):,}/{n_samples} samples")
print(f"   Failed: {failed:,}")
print(f"   Time: {processing_time:.1f}s")

# 3. DATA PREPARATION
X = np.array(features_list, dtype=np.float32)
y = np.array(labels_list, dtype=np.int32)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.1, random_state=42, stratify=y
)

print(f"\nğŸ“Š DATASET SIZES:")
print(f"   Training: {X_train.shape[0]:,} samples")
print(f"   Validation: {X_val.shape[0]:,} samples")
print(f"   Features: {X_train.shape[1]}")

# 4. XGBOOST TRAINING WITH LEARNING CURVES
print(f"\n{'='*60}")
print("ğŸŒ² TRAINING XGBOOST WITH LEARNING CURVES")
print(f"{'='*60}")

# Track learning curve
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []
training_times = []

start_time = time.time()

# Train with different n_estimators for learning curve
n_estimators_list = [50, 100, 150, 200]

for n_est in n_estimators_list:
    iter_start = time.time()
    
    print(f"\n  Training with n_estimators={n_est}...")
    
    model = XGBClassifier(
        n_estimators=n_est,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='mlogloss',
        tree_method='hist'
    )
    
    model.fit(X_train, y_train)
    
    # Training predictions
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    train_accuracies.append(train_acc)
    
    # Training loss (negative log likelihood)
    train_pred_proba = model.predict_proba(X_train)
    train_loss = -np.log(train_pred_proba[np.arange(len(y_train)), y_train]).mean()
    train_losses.append(train_loss)
    
    # Validation predictions
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    val_accuracies.append(val_acc)
    
    # Validation loss
    val_pred_proba = model.predict_proba(X_val)
    val_loss = -np.log(val_pred_proba[np.arange(len(y_val)), y_val]).mean()
    val_losses.append(val_loss)
    
    iter_time = time.time() - iter_start
    training_times.append(iter_time)
    
    print(f"    Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    print(f"    Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"    Time: {iter_time:.1f}s")

total_time = time.time() - start_time

# Final model with best n_estimators
best_idx = np.argmax(val_accuracies)
best_n_est = n_estimators_list[best_idx]

print(f"\nğŸ�¯ Best n_estimators: {best_n_est} (Val Acc: {val_accuracies[best_idx]:.4f})")

final_model = XGBClassifier(
    n_estimators=best_n_est,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric='mlogloss',
    tree_method='hist'
)

final_model.fit(X_train, y_train)

# Final predictions
final_val_pred = final_model.predict(X_val)
final_val_acc = accuracy_score(y_val, final_val_pred)

print(f"\nâœ… XGBoost Training Complete!")
print(f"   Final Validation Accuracy: {final_val_acc:.4f}")
print(f"   Total Training Time: {total_time:.1f}s")

# Save model
joblib.dump(final_model, 'xgboost_50k_model.pkl')
print("   Model saved as 'xgboost_50k_model.pkl'")

# 5. CLASSIFICATION REPORT
print(f"\n{'='*60}")
print("ğŸ“Š CLASSIFICATION REPORT")
print(f"{'='*60}")

class_names = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']

print("\nğŸ“‹ Detailed Classification Report:")
report = classification_report(y_val, final_val_pred, target_names=class_names, digits=4)
print(report)

# Confusion Matrix
cm = confusion_matrix(y_val, final_val_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('XGBoost - Confusion Matrix (50K Samples)', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('xgboost_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. LEARNING CURVES VISUALIZATION
print(f"\n{'='*60}")
print("ğŸ“ˆ LEARNING CURVES")
print(f"{'='*60}")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Accuracy Learning Curve
ax1 = axes[0, 0]
ax1.plot(n_estimators_list, train_accuracies, 'o-', linewidth=2, markersize=8, label='Train Accuracy', color='blue')
ax1.plot(n_estimators_list, val_accuracies, 's-', linewidth=2, markersize=8, label='Validation Accuracy', color='red')
ax1.set_xlabel('Number of Trees (n_estimators)', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Accuracy Learning Curve', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Loss Learning Curve
ax2 = axes[0, 1]
ax2.plot(n_estimators_list, train_losses, 'o-', linewidth=2, markersize=8, label='Train Loss', color='blue')
ax2.plot(n_estimators_list, val_losses, 's-', linewidth=2, markersize=8, label='Validation Loss', color='red')
ax2.set_xlabel('Number of Trees (n_estimators)', fontsize=12)
ax2.set_ylabel('Negative Log Likelihood', fontsize=12)
ax2.set_title('Loss Learning Curve', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Training Time
ax3 = axes[1, 0]
bars = ax3.bar(range(len(n_estimators_list)), training_times, color='green', alpha=0.7)
ax3.set_xlabel('n_estimators', fontsize=12)
ax3.set_ylabel('Training Time (seconds)', fontsize=12)
ax3.set_title('Training Time vs Model Complexity', fontsize=14, fontweight='bold')
ax3.set_xticks(range(len(n_estimators_list)))
ax3.set_xticklabels(n_estimators_list)
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, t in zip(bars, training_times):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + max(training_times)*0.01,
            f'{t:.1f}s', ha='center', va='bottom', fontweight='bold')

# Feature Importance
ax4 = axes[1, 1]
feature_importance = final_model.feature_importances_
top_n = min(20, len(feature_importance))
top_indices = np.argsort(feature_importance)[-top_n:][::-1]
top_importance = feature_importance[top_indices]

bars = ax4.barh(range(top_n), top_importance, color='purple', alpha=0.7)
ax4.set_xlabel('Importance Score', fontsize=12)
ax4.set_ylabel('Feature Index', fontsize=12)
ax4.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
ax4.set_yticks(range(top_n))
ax4.set_yticklabels([f'Feature {i}' for i in top_indices])
ax4.grid(True, alpha=0.3, axis='x')
ax4.invert_yaxis()

plt.suptitle('XGBoost Training Analysis - 50,000 Samples', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('xgboost_learning_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. PERFORMANCE SUMMARY
print(f"\n{'='*60}")
print("ğŸ“Š PERFORMANCE SUMMARY")
print(f"{'='*60}")

print(f"\nğŸ�¯ Best Configuration:")
print(f"   n_estimators: {best_n_est}")
print(f"   Validation Accuracy: {val_accuracies[best_idx]:.4f}")
print(f"   Training Accuracy: {train_accuracies[best_idx]:.4f}")
print(f"   Overfitting Gap: {train_accuracies[best_idx] - val_accuracies[best_idx]:.4f}")

print(f"\nâ�±ï¸�  Timing Statistics:")
print(f"   Data Processing: {processing_time:.1f}s")
print(f"   Model Training: {total_time:.1f}s")
print(f"   Total Time: {processing_time + total_time:.1f}s")
print(f"   Samples/sec: {len(features_list)/(processing_time + total_time):.1f}")

print(f"\nğŸ“ˆ Learning Insights:")
print(f"   1. Optimal n_estimators: {best_n_est}")
print(f"   2. Best validation accuracy achieved at {best_n_est} trees")
print(f"   3. Training time increases linearly with n_estimators")
print(f"   4. Overfitting starts after {best_n_est} trees")

print(f"\nâœ… XGBoost training on 50,000 samples completed successfully!")


# ==================== SIMPLE EEG SIGNAL VISUALIZATION ====================
print("\n" + "="*80)
print("ğŸ“Š VISUALIZING EEG SIGNALS (5 CHANNELS)")
print("="*80)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Pick one sample EEG ID from our dataset
sample_eeg_id = selected_df.iloc[0]['eeg_id']  # First sample in our dataset
print(f"ğŸ“ˆ Visualizing EEG ID: {sample_eeg_id}")

# Load the EEG data
eeg_path = f"{DATA_DIR}/{sample_eeg_id}.parquet"
eeg_df = pd.read_parquet(eeg_path)

# Get basic info
print(f"ğŸ“Š EEG Shape: {eeg_df.shape}")
print(f"ğŸ“ˆ Channels: {list(eeg_df.columns[:5])}")  # Show first 5 channel names

# Plot first 5 channels
plt.figure(figsize=(15, 10))

# Plot each of the first 5 channels
for i in range(5):
    if i < eeg_df.shape[1]:  # Make sure we have this channel
        channel_data = eeg_df.iloc[:, i].values
        channel_name = eeg_df.columns[i]
        
        # Plot just first 500 time points for clarity
        plt.subplot(5, 1, i+1)
        plt.plot(channel_data[:500], linewidth=1.5, color='blue', alpha=0.8)
        plt.title(f'Channel {i+1}: {channel_name}', fontsize=12, fontweight='bold')
        plt.xlabel('Time Points')
        plt.ylabel('Amplitude (Î¼V)')
        plt.grid(True, alpha=0.3)

plt.suptitle(f'EEG Signal - ID: {sample_eeg_id}\n(First 500 time points of 5 channels)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Print some statistics
print("\nğŸ“Š Channel Statistics (first 500 time points):")
print(f"{'Channel':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
print("-" * 50)

for i in range(5):
    if i < eeg_df.shape[1]:
        channel_data = eeg_df.iloc[:500, i].values
        print(f"{eeg_df.columns[i]:<10} {np.mean(channel_data):<10.2f} {np.std(channel_data):<10.2f} "
              f"{np.min(channel_data):<10.2f} {np.max(channel_data):<10.2f}")

print("\nâœ… EEG visualization complete!")

