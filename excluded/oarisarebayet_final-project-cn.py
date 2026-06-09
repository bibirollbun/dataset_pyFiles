import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import warnings
import random
warnings.filterwarnings('ignore')

RANDOM_SEED = 42

# Set all random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# TensorFlow/Keras seeds (if used)
try:
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)
except ImportError:
    pass


# ML imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                              GradientBoostingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             ConfusionMatrixDisplay, precision_score, recall_score, 
                             f1_score, precision_recall_fscore_support)

# Imbalanced data handling
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

# XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("âš ï¸� XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

# LightGBM
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    print("âš ï¸� LightGBM not available. Install with: pip install lightgbm")
    LGBM_AVAILABLE = False

print("\nâœ… All imports successful")
print(f"   XGBoost: {'Available' if XGBOOST_AVAILABLE else 'Not Available'}")
print(f"   LightGBM: {'Available' if LGBM_AVAILABLE else 'Not Available'}")



data_path = "/kaggle/input/malware-classification"
labels_df = pd.read_csv(f"{data_path}/trainLabels.csv")

family_map = {
    1: "Ramnit", 2: "Lollipop", 3: "Kelihos_ver3", 4: "Vundo",
    5: "Simda", 6: "Tracur", 7: "Kelihos_ver1", 8: "Obfuscator.ACY", 9: "Gatak"
}
labels_df["Family"] = labels_df["Class"].map(family_map)

print("\nLAST 5 rows:")
print(labels_df.tail())
print(f"\nTotal samples: {len(labels_df)}")

print("\n" + "="*80)
print("CLASS DISTRIBUTION ANALYSIS")
print("="*80)

# Use seeded sampling for reproducibility
sampled_df = labels_df.groupby("Class", group_keys=False).apply(
    lambda x: x.sample(n=min(10, len(x)), random_state=RANDOM_SEED)
).reset_index(drop=True)

print("\nSample count per class (with seed={RANDOM_SEED}):")
print(sampled_df["Class"].value_counts().sort_index())

for cls in sorted(sampled_df["Class"].unique()):
    ids = sampled_df[sampled_df["Class"] == cls]["Id"].tolist()
    print(f"Class {cls} ({family_map[cls]}) Sample IDs:\n", ids[:5], "...\n")

print("\nâš ï¸� Malware family distribution (BEFORE BALANCING):")
distribution = labels_df["Family"].value_counts()
print(distribution)

# Calculate imbalance statistics
max_samples = distribution.max()
min_samples = distribution.min()
imbalance_ratio = max_samples / min_samples
print(f"\nImbalance Ratio: {imbalance_ratio:.1f}:1 (Majority:Minority)")
print(f"   Majority class: {distribution.idxmax()} ({max_samples} samples)")
print(f"   Minority class: {distribution.idxmin()} ({min_samples} samples)")

# Visualization
plt.figure(figsize=(10, 6))
sns.countplot(data=labels_df, y="Family", order=labels_df["Family"].value_counts().index)
plt.title(f"Malware Family Distribution (Original - Imbalanced)\nImbalance Ratio: {imbalance_ratio:.1f}:1")
plt.xlabel("Number of Samples")
plt.ylabel("Malware Family")
plt.tight_layout()
plt.savefig('/kaggle/working/class_distribution_original.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nDistribution analysis complete")


!mkdir -p /kaggle/working/bytes
!7z l /kaggle/input/malware-classification/train.7z | grep '.bytes' | awk '{print $NF}' | head -n 2000 > /kaggle/working/bytes/bytes_list.txt
!7z e /kaggle/input/malware-classification/train.7z -o/kaggle/working/bytes -i@/kaggle/working/bytes/bytes_list.txt

print("File extraction complete")


print("\n" + "="*80)
print("ğŸ”¬ BYTE HISTOGRAM FEATURE EXTRACTION")
print("="*80)

def extract_byte_histogram(file_path):
    """
    Extract 256-dimensional byte frequency histogram from .bytes file
    
    Args:
        file_path: Path to .bytes file containing hex representation
    
    Returns:
        numpy array of shape (256,) containing byte frequencies
    """
    try:
        with open(file_path, 'r') as file:
            hex_lines = file.readlines()
        
        bytes_list = []
        for line in hex_lines:
            parts = line.strip().split()
            bytes_seq = parts[1:]  # Skip address column
            bytes_list.extend([b for b in bytes_seq if b != '??'])
        
        # Convert hex to integers
        byte_vals = [int(b, 16) for b in bytes_list if len(b) == 2]
        
        # Create histogram (256 bins for byte values 0-255)
        hist = np.histogram(byte_vals, bins=256, range=(0, 255))[0]
        
        return hist
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return np.zeros(256)

# Extract features from all files
file_dir = '/kaggle/working/bytes'
sample_files = sorted(os.listdir(file_dir))[:2500]  # Sort for reproducibility

X = []
file_ids = []

print(f"Extracting features from {len(sample_files)} files...")
for fname in tqdm(sample_files):
    if fname.endswith('.bytes'):
        f_id = fname.replace(".bytes", "")
        hist = extract_byte_histogram(os.path.join(file_dir, fname))
        X.append(hist)
        file_ids.append(f_id)

# Create DataFrame
df_features = pd.DataFrame(X, columns=[f'byte_{i:02X}' for i in range(256)])
df_features["Id"] = file_ids
df_features = df_features.merge(labels_df, on="Id")

print("\nFeature extraction complete")
print(f"   Shape: {df_features.shape}")
print(f"   Features: 256 byte positions")
print(f"   Samples: {len(df_features)}")

print("\nFeature statistics:")
print(df_features.describe())

# Save features
df_features.to_csv("/kaggle/working/byte_histogram_features.csv", index=False)
print("\nFeatures saved to: byte_histogram_features.csv")


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 20)

# Show first 10 samples with selected features
print("\nSample Data (First 10 rows with selected features):\n")

# Select important columns to show
display_cols = ['Id', 'byte_00', 'byte_01', 'byte_02', 'byte_10', 'byte_20', 
                'byte_50', 'byte_A0', 'byte_FF', 'Class', 'Family']

df_sample = df_features[display_cols].head(10)
display(df_sample)  




# Label encoding
le = LabelEncoder()
df_features["Family_Encoded"] = le.fit_transform(df_features["Family"]).astype(int)
family_names = le.classes_

print(f"Encoded {len(family_names)} malware families:")
for idx, name in enumerate(family_names):
    print(f"   {idx}: {name}")

# Separate features and labels
X = df_features.drop(columns=["Id", "Class", "Family", "Family_Encoded"])
y = df_features["Family_Encoded"]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Label vector shape: {y.shape}")

# Use random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=RANDOM_SEED,  
    stratify=y  # Maintain class distribution
)

print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set:     {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")



print("\nTRAINING SET DISTRIBUTION (BEFORE SMOTE):")
print("-"*80)
train_dist = pd.Series(y_train).value_counts().sort_index()

train_distribution_data = []
for idx, name in enumerate(family_names):
    count = (y_train == idx).sum()
    percentage = (count / len(y_train)) * 100
    train_distribution_data.append({
        'Class': idx,
        'Family': name,
        'Count': count,
        'Percentage': percentage
    })
    print(f"  {name:15s}: {count:4d} samples ({percentage:5.2f}%)")

# Calculate imbalance ratio
max_class = train_dist.max()
min_class = train_dist.min()
imbalance_ratio = max_class / min_class
print(f"\n Training Set Imbalance Ratio: {imbalance_ratio:.1f}:1 (Majority:Minority)")
print(f"   Majority: {family_names[train_dist.idxmax()]} ({max_class} samples)")
print(f"   Minority: {family_names[train_dist.idxmin()]} ({min_class} samples)")

print("\nTEST SET DISTRIBUTION (UNCHANGED - Original Distribution):")
print("-"*80)
test_dist = pd.Series(y_test).value_counts().sort_index()

for idx, name in enumerate(family_names):
    count = (y_test == idx).sum()
    percentage = (count / len(y_test)) * 100
    print(f"  {name:15s}: {count:4d} samples ({percentage:5.2f}%)")

# Store original distribution for later comparison
original_train_dist = train_dist.copy()
original_test_dist = test_dist.copy()



print("WHY: Byte histograms are count data - perfect for chi-square!")
print("GOAL: Identify which byte positions are most discriminative")
print(f"SEED: {RANDOM_SEED} (for reproducible feature selection)")

# Select top 100 most important features
N_FEATURES = 100

# CRITICAL: Feature selection BEFORE SMOTE to avoid data leakage
chi2_selector = SelectKBest(chi2, k=N_FEATURES)
X_train_selected = chi2_selector.fit_transform(X_train, y_train)
X_test_selected = chi2_selector.transform(X_test)

print(f"\nSelected {N_FEATURES} most discriminative features out of 256")
print(f"   Training shape: {X_train.shape} â†’ {X_train_selected.shape}")
print(f"   Test shape:     {X_test.shape} â†’ {X_test_selected.shape}")

# Get feature scores
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Chi2_Score': chi2_selector.scores_,
    'Selected': chi2_selector.get_support()
}).sort_values('Chi2_Score', ascending=False)


print(feature_scores.head(20)[['Feature', 'Chi2_Score']].to_string(index=False))

# Visualize top features
plt.figure(figsize=(12, 6))
top_20 = feature_scores.head(20)
plt.barh(top_20['Feature'], top_20['Chi2_Score'], color='#3498db')
plt.xlabel('Chi-Square Score', fontsize=12)
plt.ylabel('Byte Position', fontsize=12)
plt.title('Top 20 Most Discriminative Byte Positions for Malware Classification', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('/kaggle/working/chi_square_features.png', dpi=300, bbox_inches='tight')
plt.show()

# Save selected feature names for interpretation
selected_features = feature_scores[feature_scores['Selected']]['Feature'].tolist()
feature_scores.to_csv('/kaggle/working/feature_scores.csv', index=False)

print(f"\nFeature reduction: 256 â†’ {N_FEATURES} ({(N_FEATURES/256)*100:.1f}% retained)")



print(f"WHY: {imbalance_ratio:.1f}:1 imbalance ratio is SEVERE!")
print("SMOTE: Creates synthetic samples for minority classes")
print(f"SEED: {RANDOM_SEED} (for reproducible synthetic samples)")

# Store pre-SMOTE counts
pre_smote_counts = {
    'total': len(y_train),
    'per_class': {family_names[i]: (y_train == i).sum() for i in range(len(family_names))}
}

print("\nBEFORE SMOTE:")
print("-"*80)
for idx, name in enumerate(family_names):
    count = pre_smote_counts['per_class'][name]
    percentage = (count / pre_smote_counts['total']) * 100
    print(f"  {name:15s}: {count:4d} samples ({percentage:5.2f}%)")

print(f"\nTotal: {pre_smote_counts['total']} samples")

# ========== STRATEGIC OVERSAMPLING ==========
print("\n" + "="*80)
print("STRATEGIC OVERSAMPLING (Not Extreme)")
print("="*80)

import numpy as np
from imblearn.over_sampling import SMOTE

# Calculate statistics
class_counts = np.bincount(y_train)
median_count = int(np.median(class_counts[class_counts > 0]))
q3_count = int(np.percentile(class_counts[class_counts > 0], 75))  # 75th percentile

print(f"Statistics:")
print(f"  Maximum: {max(class_counts)} (Kelihos_ver3)")
print(f"  Q3 (75th percentile): {q3_count}")
print(f"  Median: {median_count}")
print(f"  Minimum: {min(class_counts)} (Simda)")

# Define SMART target: Use Q3 (75th percentile) NOT maximum
# This balances without extreme oversampling
smart_target = q3_count
print(f"\nSmart target: {smart_target} samples per class (Q3)")
print(f"(Not {max(class_counts)} which would oversample too much)")

# SMART sampling strategy
sampling_strategy = {}
print("\nOversampling Strategy:")
print("-"*80)
for idx, name in enumerate(family_names):
    current = pre_smote_counts['per_class'][name]
    
    if current < 10:  # Extreme minority (Simda: 8)
        # Cap at reasonable multiple (6x, max 50)
        target = min(50, current * 6)
        sampling_strategy[idx] = target
        print(f"  {name:15s}: {current:3d} â†’ {target:3d} (extreme minority, 6x cap)")
    
    elif current < median_count:  # Below median
        # Bring up to median level
        target = median_count
        sampling_strategy[idx] = target
        print(f"  {name:15s}: {current:3d} â†’ {target:3d} (below median)")
    
    elif current < smart_target:  # Below Q3
        # Bring up to Q3 level
        target = smart_target
        sampling_strategy[idx] = target
        print(f"  {name:15s}: {current:3d} â†’ {target:3d} (below Q3)")
    
    else:  # Already at or above Q3
        # Keep as is (no oversampling)
        sampling_strategy[idx] = current
        print(f"  {name:15s}: {current:3d} â†’ {current:3d} (already sufficient)")

# Adaptive k_neighbors for safety
min_class_count = min(class_counts)
safe_k_neighbors = min(3, max(1, min_class_count - 2))
print(f"\nUsing k_neighbors={safe_k_neighbors} (safe for min class with {min_class_count} samples)")

# Apply SMART SMOTE
smote = SMOTE(
    random_state=RANDOM_SEED,
    k_neighbors=safe_k_neighbors,
    sampling_strategy=sampling_strategy
)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)

# Store post-SMOTE counts
post_smote_counts = {
    'total': len(y_train_resampled),
    'per_class': {family_names[i]: (y_train_resampled == i).sum() for i in range(len(family_names))}
}

print("\nAFTER STRATEGIC OVERSAMPLING:")
print("-"*80)
for idx, name in enumerate(family_names):
    count = post_smote_counts['per_class'][name]
    percentage = (count / post_smote_counts['total']) * 100
    increase = count - pre_smote_counts['per_class'][name]
    change_symbol = "+" if increase > 0 else ""
    print(f"  {name:15s}: {count:4d} samples ({percentage:5.2f}%) [{change_symbol}{increase:4d}]")

print(f"\nTotal: {post_smote_counts['total']} samples")

# Calculate changes
samples_added = post_smote_counts['total'] - pre_smote_counts['total']
increase_percentage = (samples_added / pre_smote_counts['total']) * 100

print(f"\nTraining samples: {pre_smote_counts['total']} â†’ {post_smote_counts['total']}")
print(f"Synthetic samples added: {samples_added} (+{increase_percentage:.1f}%)")
print(f"Test samples: {len(y_test)} (UNCHANGED)")

# Calculate NEW imbalance ratio
new_counts = list(post_smote_counts['per_class'].values())
new_imbalance_ratio = max(new_counts) / min(new_counts)
print(f"\nImbalance ratio improved: {imbalance_ratio:.1f}:1 â†’ {new_imbalance_ratio:.1f}:1")

# Synthetic data analysis
print(f"\nSYNTHETIC DATA ANALYSIS:")
print("-"*80)
total_synthetic = 0
for idx, name in enumerate(family_names):
    original = pre_smote_counts['per_class'][name]
    current = post_smote_counts['per_class'][name]
    synthetic = current - original
    if synthetic > 0:
        synth_percentage = (synthetic / current) * 100
        total_synthetic += synthetic
        print(f"  {name:15s}: {synthetic:3d} synthetic ({synth_percentage:5.1f}% of class)")

synth_overall_percentage = (total_synthetic / post_smote_counts['total']) * 100
print(f"\nOverall: {total_synthetic} synthetic samples ({synth_overall_percentage:.1f}% of training data)")

# Visualize SMART SMOTE effect
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Before SMOTE
before_data = pd.DataFrame({
    'Family': family_names,
    'Count': [pre_smote_counts['per_class'][name] for name in family_names],
    'Type': ['Original'] * len(family_names)
}).sort_values('Count', ascending=True)

ax1.barh(before_data['Family'], before_data['Count'], color='#e74c3c')
ax1.set_xlabel('Number of Samples', fontsize=12)
ax1.set_title(f'Before: Severely Imbalanced\nTotal: {pre_smote_counts["total"]} samples\nImbalance: {imbalance_ratio:.1f}:1', 
              fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
ax1.set_xlim(0, max(pre_smote_counts['per_class'].values()) * 1.1)

# Add value labels
for i, (family, count) in enumerate(zip(before_data['Family'], before_data['Count'])):
    ax1.text(count + 5, i, f'{count}', va='center', fontweight='bold')

# After SMART SMOTE
after_data = pd.DataFrame({
    'Family': family_names,
    'Total': [post_smote_counts['per_class'][name] for name in family_names],
    'Original': [pre_smote_counts['per_class'][name] for name in family_names]
})
after_data['Synthetic'] = after_data['Total'] - after_data['Original']
after_data = after_data.sort_values('Total', ascending=True)

# Stacked bar chart
ax2.barh(after_data['Family'], after_data['Original'], color='#3498db', label='Original')
ax2.barh(after_data['Family'], after_data['Synthetic'], left=after_data['Original'], 
         color='#2ecc71', label='Synthetic')

ax2.set_xlabel('Number of Samples', fontsize=12)
ax2.set_title(f'After: Strategically Balanced\nTotal: {post_smote_counts["total"]} samples\nImbalance: {new_imbalance_ratio:.1f}:1', 
              fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
ax2.legend(loc='lower right')

# Add value labels with breakdown
for i, (family, original, synthetic, total) in enumerate(zip(after_data['Family'], 
                                                              after_data['Original'], 
                                                              after_data['Synthetic'], 
                                                              after_data['Total'])):
    if synthetic > 0:
        label = f'{original}+{synthetic}={total}'
    else:
        label = f'{total}'
    ax2.text(total + 5, i, label, va='center', fontsize=9)

# Add median and Q3 lines
ax2.axvline(x=median_count, color='orange', linestyle='--', alpha=0.7, linewidth=1)
ax2.axvline(x=smart_target, color='red', linestyle='--', alpha=0.7, linewidth=1)
ax2.text(median_count, len(family_names)-0.5, f' Median: {median_count}', 
         color='orange', va='center', fontsize=8)
ax2.text(smart_target, len(family_names)-1.5, f' Q3 Target: {smart_target}', 
         color='red', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('/kaggle/working/strategic_smote_comparison.png', dpi=300, bbox_inches='tight')
plt.show()




models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=600,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        class_weight='balanced', 
        random_state=RANDOM_SEED,  
        n_jobs=-1
    ),
    
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=800,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight='balanced',  
        random_state=RANDOM_SEED,  
        n_jobs=-1
    ),
    
    "LogReg": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            C=2.0,
            penalty="l2",
            max_iter=1000,
            multi_class="multinomial",
            class_weight='balanced',  
            n_jobs=-1,
            random_state=RANDOM_SEED  
        ))
    ]),
    
    "SVM-RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_SEED  
        ))
    ]),
    
    "kNN-15": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(
            n_neighbors=15,
            weights="distance",
            metric="minkowski",
            p=2
        ))
    ]),
    
    "GradBoost": GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.85,
        random_state=RANDOM_SEED,  
        verbose=0
    ),
    
    "MLP-256x128": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate="adaptive",
            max_iter=400,
            early_stopping=True,
            random_state=RANDOM_SEED,  
            verbose=False
        ))
    ]),
}

# Add XGBoost if available
if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=10,  # ADDED for 10:1 imbalance
        random_state=RANDOM_SEED,  
        n_jobs=-1,
        eval_metric='mlogloss',
        verbosity=0
    )
    print(f"âœ… XGBoost added with seed={RANDOM_SEED}")

# Add LightGBM if available
if LGBM_AVAILABLE:
    models["LightGBM"] = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',  # ADDED
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1
    )
    print(f"âœ… LightGBM added with seed={RANDOM_SEED}")

# Ensemble models
ensemble_models = {
    "Ensemble_RF_ET_GB_Hard": VotingClassifier(
        estimators=[
            ('rf', models['RandomForest']),
            ('et', models['ExtraTrees']),
            ('gb', models['GradBoost'])
        ],
        voting='hard',
        n_jobs=-1
    ),
    
    "Ensemble_RF_ET_GB_Soft": VotingClassifier(
        estimators=[
            ('rf', models['RandomForest']),
            ('et', models['ExtraTrees']),
            ('gb', models['GradBoost'])
        ],
        voting='soft',
        n_jobs=-1
    ),
}

if LGBM_AVAILABLE:
    ensemble_models["Ensemble_RF_ET_LGBM_Soft"] = VotingClassifier(
        estimators=[
            ('rf', models['RandomForest']),
            ('et', models['ExtraTrees']),
            ('lgbm', models['LightGBM'])
        ],
        voting='soft',
        n_jobs=-1
    )
    
    ensemble_models["Ensemble_ALL_Soft"] = VotingClassifier(
        estimators=[
            ('rf', models['RandomForest']),
            ('et', models['ExtraTrees']),
            ('gb', models['GradBoost']),
            ('lgbm', models['LightGBM'])
        ],
        voting='soft',
        n_jobs=-1
    )

print(f"\nTotal models defined: {len(models)} individual + {len(ensemble_models)} ensemble")
print("All models configured with random_state and class weights for imbalance")


print(f"Seed: {RANDOM_SEED} - Results should be identical across runs")

all_models = {**models, **ensemble_models}
final_results = []
overfitting_analysis = []

for name, clf in all_models.items():
    print(f"\n{'='*80}")
    print(f"Training {name}...")
    print(f"{'='*80}")
    
    # Train on SMOTE-balanced selected features
    clf.fit(X_train_resampled, y_train_resampled)
    
    
    # Training set performance
    y_train_pred = clf.predict(X_train_resampled)
    train_accuracy = accuracy_score(y_train_resampled, y_train_pred)
    train_f1 = f1_score(y_train_resampled, y_train_pred, average='weighted', zero_division=0)
    
    # Test set performance
    y_test_pred = clf.predict(X_test_selected)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    # Calculate overfitting metrics
    accuracy_gap = train_accuracy - test_accuracy
    f1_gap = train_f1 - test_f1
    
    # Overfitting severity classification
    if accuracy_gap < 0.02:
        overfit_level = "âœ… No Overfitting"
    elif accuracy_gap < 0.05:
        overfit_level = "âš ï¸� Slight Overfitting"
    elif accuracy_gap < 0.10:
        overfit_level = "ğŸŸ  Moderate Overfitting"
    else:
        overfit_level = "ğŸ”´ Severe Overfitting"
    
    # Calculate all metrics for test set
    precision_macro = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    recall_macro = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    # Store results
    final_results.append({
        'Model': name,
        'Train_Accuracy': train_accuracy,
        'Test_Accuracy': test_accuracy,
        'Accuracy_Gap': accuracy_gap,
        'Train_F1': train_f1,
        'Test_F1': test_f1,
        'F1_Gap': f1_gap,
        'Precision_Macro': precision_macro,
        'Precision_Weighted': precision_weighted,
        'Recall_Macro': recall_macro,
        'Recall_Weighted': recall_weighted,
        'F1_Macro': f1_macro,
        'F1_Weighted': f1_weighted,
        'Overfit_Level': overfit_level
    })
    
    # Print performance summary
    print(f"\nğŸ“Š Performance Summary:")
    print(f"   Training Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"   Test Accuracy:      {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"   Accuracy Gap:       {accuracy_gap:.4f} ({accuracy_gap*100:.2f}%)")
    print(f"   ")
    print(f"   Training F1:        {train_f1:.4f}")
    print(f"   Test F1:            {test_f1:.4f}")
    print(f"   F1 Gap:             {f1_gap:.4f}")
    print(f"   ")
    print(f"   {overfit_level}")
    
    # Detailed overfitting analysis
    if accuracy_gap > 0.05:
        print(f"\n   âš ï¸� WARNING: Significant train-test gap detected!")
        print(f"   This model may be overfitting the training data.")

print("\n" + "="*80)
print("âœ… ALL MODELS TRAINED AND EVALUATED")
print("="*80)


# Create DataFrame from results
df_results = pd.DataFrame(final_results)

# Sort by Test F1-Score (best first)
df_results = df_results.sort_values('Test_F1', ascending=False).reset_index(drop=True)

print("="*100)
print(f"{'Rank':<6} {'Model':<25} {'Test Acc':>10} {'Test F1':>10} {'Precision':>10} {'Recall':>10} {'F1 Macro':>10}")
print("="*100)

for idx, row in df_results.iterrows():
    print(f"{idx+1:<6} {row['Model']:<25} "
          f"{row['Test_Accuracy']:>10.4f} "
          f"{row['Test_F1']:>10.4f} "
          f"{row['Precision_Weighted']:>10.4f} "
          f"{row['Recall_Weighted']:>10.4f} "
          f"{row['F1_Macro']:>10.4f}")

print("="*100)

# Add summary statistics
print(f"\n{'TOP 3 MODELS':^100}")
print("-"*100)
for i in range(min(3, len(df_results))):
    row = df_results.iloc[i]
    print(f"{i+1}. {row['Model']:<30} "
          f"Acc: {row['Test_Accuracy']:.4f} | "
          f"F1: {row['Test_F1']:.4f} | "
          f"Prec: {row['Precision_Weighted']:.4f} | "
          f"Rec: {row['Recall_Weighted']:.4f}")

# Save to CSV
df_results.to_csv('/kaggle/working/model_results_detailed.csv', index=False)


best_model_name = df_results.iloc[0]['Model']
best_model = all_models[best_model_name]


best_metrics = df_results.iloc[0]

print(f"\nOverall Metrics:")
print(f"   Test Accuracy:           {best_metrics['Test_Accuracy']:.4f} ({best_metrics['Test_Accuracy']*100:.2f}%)")
print(f"   Train Accuracy:          {best_metrics['Train_Accuracy']:.4f} ({best_metrics['Train_Accuracy']*100:.2f}%)")
print(f"   Accuracy Gap:            {best_metrics['Accuracy_Gap']:.4f} ({best_metrics['Accuracy_Gap']*100:.2f}%)")
print(f"   ")
print(f"   Precision (Weighted):    {best_metrics['Precision_Weighted']:.4f}")
print(f"   Recall (Weighted):       {best_metrics['Recall_Weighted']:.4f}")
print(f"   F1-Score (Weighted):     {best_metrics['Test_F1']:.4f}")
print(f"   F1-Score (Macro):        {best_metrics['F1_Macro']:.4f}")
print(f"   ")
print(f"   Overfitting Status:      {best_metrics['Overfit_Level']}")

# Get predictions for best model
y_pred_best = best_model.predict(X_test_selected)


print(f"Seed: {RANDOM_SEED} - Report is reproducible\n")
print(classification_report(y_test, y_pred_best, target_names=family_names, digits=4))


cm = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=family_names)
disp.plot(cmap="Blues", ax=ax, xticks_rotation=45, values_format='d')

# Add title with overfitting info
gap_percent = best_metrics['Accuracy_Gap'] * 100
title = f"Confusion Matrix - {best_model_name}\n"
title += f"Test Accuracy: {best_metrics['Test_Accuracy']*100:.2f}% | "
title += f"Train-Test Gap: {gap_percent:.2f}% | {best_metrics['Overfit_Level']}\n"
title += f"(Trained on SMOTE + Chi-Square selected features, Seed={RANDOM_SEED})"

plt.title(title, fontsize=13)
plt.tight_layout()
plt.savefig('/kaggle/working/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("Confusion matrix saved")



print(f"Best Model: {best_model_name}")
print(f"Seed: {RANDOM_SEED} - Results are reproducible\n")

per_class_data = []

for idx, name in enumerate(family_names):
    mask = (y_test == idx)

    if mask.sum() > 0:
        y_true_class = y_test[mask]
        y_pred_class = y_pred_best[mask]

        correct = (y_pred_class == y_true_class).sum()
        total = len(y_true_class)
        class_acc = correct / total

        # Calculate precision, recall, F1 for this class
        prec = precision_score(
            y_test, y_pred_best,
            labels=[idx],
            average=None,
            zero_division=0
        )[0]

        rec = recall_score(
            y_test, y_pred_best,
            labels=[idx],
            average=None,
            zero_division=0
        )[0]

        f1 = f1_score(
            y_test, y_pred_best,
            labels=[idx],
            average=None,
            zero_division=0
        )[0]

        per_class_data.append({
            'Malware_Family': name,
            'Test_Samples': total,
            'Correct': correct,
            'Incorrect': total - correct,
            'Accuracy': class_acc,
            'Precision': prec,
            'Recall': rec,
            'F1_Score': f1
        })

df_per_class = pd.DataFrame(per_class_data)

# Display results
print(f"{'Family':<20} {'Samples':>8} {'Correct':>8} {'Wrong':>6} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-"*100)

for _, row in df_per_class.iterrows():
    print(f"{row['Malware_Family']:<20} {row['Test_Samples']:>8} {row['Correct']:>8} {row['Incorrect']:>6} "
          f"{row['Accuracy']:>10.4f} {row['Precision']:>10.4f} {row['Recall']:>10.4f} {row['F1_Score']:>10.4f}")

print("="*100)

# Save per-class results
df_per_class.to_csv("/kaggle/working/per_class_performance.csv", index=False)
print("\nâœ… Per-class results saved to: per_class_performance.csv")


# Sort by recall (ability to detect)
df_per_class_sorted = df_per_class.sort_values('Recall', ascending=True)

print("\nâš ï¸� Classes with Lowest Recall (Hardest to Detect):")
print("-"*80)
for i, row in enumerate(df_per_class_sorted.head(3).itertuples(), 1):
    print(f"   {i}. {row.Malware_Family:<20} | Recall: {row.Recall:.4f} ({row.Recall*100:.2f}%) | "
          f"Missed: {row.Incorrect}/{row.Test_Samples}")

# Sort by precision (false positive rate)
df_per_class_sorted_prec = df_per_class.sort_values('Precision', ascending=True)

print("\nâš ï¸� Classes with Lowest Precision (Most False Positives):")
print("-"*80)
for i, row in enumerate(df_per_class_sorted_prec.head(3).itertuples(), 1):
    print(f"   {i}. {row.Malware_Family:<20} | Precision: {row.Precision:.4f} ({row.Precision*100:.2f}%)")

# Highlight minority class performance

print("These classes had very few training samples before SMOTE:")

minority_classes = ['Simda', 'Vundo', 'Kelihos_ver1', 'Tracur']
for name in minority_classes:
    row = df_per_class[df_per_class['Malware_Family'] == name]
    if not row.empty:
        row = row.iloc[0]
        original_train_count = pre_smote_counts['per_class'][name]
        smote_train_count = post_smote_counts['per_class'][name]
        
        print(f"\n{name}:")
        print(f"   Original training samples: {original_train_count}")
        print(f"   After SMOTE: {smote_train_count} (+{smote_train_count - original_train_count})")
        print(f"   Test samples: {row['Test_Samples']}")
        print(f"   Test Recall: {row['Recall']:.4f} ({row['Recall']*100:.2f}%)")
        print(f"   Correctly detected: {row['Correct']}/{row['Test_Samples']}")
        
        if original_train_count < 50:
            if row['Recall'] > 0.8:
                print(f"   EXCELLENT - SMOTE successfully enabled detection!")
            elif row['Recall'] > 0.5:
                print(f"   MODERATE - SMOTE helped but still challenging")
            else:
                print(f"   POOR - Class remains difficult despite SMOTE")

print("\n" + "="*80)


import shap
import matplotlib.pyplot as plt
import numpy as np

print("\n1. GLOBAL FEATURE IMPORTANCE")
print("-" * 50)

# Use RandomForest instead of ensemble for SHAP compatibility
best_model = models["RandomForest"]

# Prepare data
if hasattr(best_model, 'named_steps'):
    model_for_shap = best_model.named_steps['clf']
    X_test_shap = best_model.named_steps['scaler'].transform(X_test_selected)
else:
    model_for_shap = best_model
    X_test_shap = X_test_selected

# Get feature names
feature_names = [f"Byte_{i}" for i in range(X_test_selected.shape[1])]

# Create explainer
explainer = shap.TreeExplainer(model_for_shap)

# Compute SHAP values for samples
n_samples = min(100, len(X_test_shap))
X_sample = X_test_shap[:n_samples]
shap_values = explainer.shap_values(X_sample)

# Handle SHAP values format
if isinstance(shap_values, list):
    shap_importance = np.abs(np.array(shap_values)).mean(axis=(0, 1))
else:
    if shap_values.ndim == 3:
        shap_importance = np.abs(shap_values).mean(axis=(0, 2))
    else:
        shap_importance = np.abs(shap_values).mean(axis=0)

# Ensure we have correct number of features
if len(shap_importance) != len(feature_names):
    min_len = min(len(shap_importance), len(feature_names))
    shap_importance = shap_importance[:min_len]
    feature_names = feature_names[:min_len]

# Global importance plot
plt.figure(figsize=(12, 8))

# Top features
top_n = min(15, len(shap_importance))
top_indices = np.argsort(shap_importance)[-top_n:][::-1]
top_features = [feature_names[i] for i in top_indices]
top_importance = [shap_importance[i] for i in top_indices]

# Plot
plt.barh(range(len(top_features)), top_importance, color='skyblue')
plt.yticks(range(len(top_features)), top_features)
plt.xlabel('Mean |SHAP Value| (Impact on Prediction)', fontsize=12)
plt.title('Top 15 Most Important Byte Positions\n(Global Feature Importance)', 
         fontsize=14, fontweight='bold', pad=20)
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('/kaggle/working/shap_global_importance_simple.png', dpi=300, bbox_inches='tight')
plt.show()

print("Saved: shap_global_importance_simple.png")

print("\n2. WATERFALL PLOT - CORRECTLY CLASSIFIED SAMPLE")
print("-" * 50)

# Find a correctly classified sample
y_test_pred = best_model.predict(X_test_selected)
correct_indices = np.where(y_test_pred == y_test)[0]

if len(correct_indices) > 0:
    sample_idx = correct_indices[0]
    true_label = y_test.iloc[sample_idx] if hasattr(y_test, 'iloc') else y_test[sample_idx]
    true_class_name = family_names[true_label]
    
    print(f"Sample Index: {sample_idx}")
    print(f"True Class: {true_class_name}")
    
    # Get SHAP values for this sample
    X_single = X_test_shap[sample_idx:sample_idx+1]
    
    # Recompute SHAP for single sample
    sample_shap = explainer.shap_values(X_single)
    
    if isinstance(sample_shap, list):
        sample_shap_values = sample_shap[true_label][0]
        base_value = explainer.expected_value[true_label]
    else:
        if sample_shap.ndim == 3:
            sample_shap_values = sample_shap[0, :, true_label]
        else:
            sample_shap_values = sample_shap[0, :]
        base_value = explainer.expected_value[true_label] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    
    # Create waterfall plot
    plt.figure(figsize=(12, 6))
    
    explanation = shap.Explanation(
        values=sample_shap_values,
        base_values=base_value,
        data=X_single.flatten(),
        feature_names=feature_names
    )
    
    shap.plots.waterfall(explanation, max_display=15, show=False)
    
    plt.title(f'Correct Prediction: {true_class_name}\nFeature Contributions', 
             fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('SHAP Value (Impact on Model Output)', fontsize=11)
    plt.tight_layout()
    plt.savefig('/kaggle/working/shap_waterfall_correct.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Saved: shap_waterfall_correct.png")
    
else:
    print("No correctly classified samples found")

print("\n3. WATERFALL PLOT - MISCLASSIFIED SAMPLE")
print("-" * 50)

# Find a misclassified sample
wrong_indices = np.where(y_test_pred != y_test)[0]

if len(wrong_indices) > 0:
    sample_idx = wrong_indices[0]
    true_label = y_test.iloc[sample_idx] if hasattr(y_test, 'iloc') else y_test[sample_idx]
    pred_label = y_test_pred[sample_idx]
    
    true_class_name = family_names[true_label]
    pred_class_name = family_names[pred_label]
    
    print(f"Sample Index: {sample_idx}")
    print(f"True Class: {true_class_name}")
    print(f"Predicted Class: {pred_class_name}")
    
    # Get SHAP values for this sample
    X_single = X_test_shap[sample_idx:sample_idx+1]
    sample_shap = explainer.shap_values(X_single)
    
    if isinstance(sample_shap, list):
        sample_shap_values = sample_shap[pred_label][0]
        base_value = explainer.expected_value[pred_label]
    else:
        if sample_shap.ndim == 3:
            sample_shap_values = sample_shap[0, :, pred_label]
        else:
            sample_shap_values = sample_shap[0, :]
        base_value = explainer.expected_value[pred_label] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    
    # Create waterfall plot
    plt.figure(figsize=(12, 6))
    
    explanation = shap.Explanation(
        values=sample_shap_values,
        base_values=base_value,
        data=X_single.flatten(),
        feature_names=feature_names
    )
    
    shap.plots.waterfall(explanation, max_display=15, show=False)
    
    plt.title(f'Misclassification: {true_class_name} as {pred_class_name}\nFeature Contributions', 
             fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('SHAP Value (Impact on Model Output)', fontsize=11)
    plt.tight_layout()
    plt.savefig('/kaggle/working/shap_waterfall_wrong.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Saved: shap_waterfall_wrong.png")
    
else:
    print("All samples correctly classified")

print("\nSHAP analysis complete.")


# Simple Model Comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics_to_plot = [
    ('Test_Accuracy', 'Test Accuracy'),
    ('Precision_Weighted', 'Precision (Weighted)'),
    ('Recall_Weighted', 'Recall (Weighted)'),
    ('Test_F1', 'F1-Score (Weighted)')
]

for idx, (col_name, display_name) in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]
    
    data = df_results.sort_values(col_name, ascending=True)
    
    ax.barh(data['Model'], data[col_name], color='steelblue')
    ax.set_xlabel(display_name, fontsize=12)
    ax.set_title(f'{display_name} Comparison', fontsize=14, fontweight='bold')
    ax.set_xlim([0.7, 1.05])
    
    for i, v in enumerate(data[col_name]):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('/kaggle/working/model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Model Performance Comparison Complete")


# ============================================================================
# 16. ADVANCED EXPERIMENTS WITH REPRODUCIBILITY
# ============================================================================

print("\n" + "="*80*2)
print("="*80*2)
print("ğŸ§ª ADVANCED EXPERIMENTS - REPRODUCIBLE ALTERNATIVE APPROACHES")
print("="*80*2)
print("="*80*2)
print(f"Seed: {RANDOM_SEED} - All experiments are reproducible")

# Store baseline for comparison
baseline_acc = best_metrics['Test_Accuracy']
baseline_f1 = best_metrics['Test_F1']
baseline_gap = best_metrics['Accuracy_Gap']

experimental_results = []

# Add baseline
experimental_results.append({
    'Method': f'Baseline ({best_model_name})',
    'Train_Accuracy': best_metrics['Train_Accuracy'],
    'Test_Accuracy': baseline_acc,
    'Accuracy_Gap': baseline_gap,
    'Train_F1': best_metrics['Train_F1'],
    'Test_F1': baseline_f1,
    'F1_Gap': best_metrics['F1_Gap'],
    'Gain_vs_Baseline': 0.0,
    'Overfit_Status': best_metrics['Overfit_Level']
})

# ============================================================================
# EXPERIMENT 1: ADASYN (Adaptive Synthetic Sampling)
# ============================================================================

print("\n" + "="*80)
print("ğŸ§ª EXPERIMENT 1: ADASYN - Adaptive Synthetic Sampling")
print("="*80)
print("CONCEPT: Generates more synthetic samples for harder-to-learn examples")
print("vs SMOTE: SMOTE generates uniformly, ADASYN focuses on boundary cases")
print(f"Seed: {RANDOM_SEED}")

try:
    adasyn = ADASYN(random_state=RANDOM_SEED, n_neighbors=5)
    X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train_selected, y_train)
    
    print(f"\nTraining samples: {len(y_train)} â†’ {len(y_train_adasyn)}")
    
    # Train RandomForest with ADASYN data
    rf_adasyn = RandomForestClassifier(
        n_estimators=600, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1
    )
    
    print("Training RandomForest with ADASYN...")
    rf_adasyn.fit(X_train_adasyn, y_train_adasyn)
    
    # Training performance
    y_train_pred_adasyn = rf_adasyn.predict(X_train_adasyn)
    train_acc_adasyn = accuracy_score(y_train_adasyn, y_train_pred_adasyn)
    train_f1_adasyn = f1_score(y_train_adasyn, y_train_pred_adasyn, average='weighted')
    
    # Test performance
    y_test_pred_adasyn = rf_adasyn.predict(X_test_selected)
    test_acc_adasyn = accuracy_score(y_test, y_test_pred_adasyn)
    test_f1_adasyn = f1_score(y_test, y_test_pred_adasyn, average='weighted')
    
    # Overfitting metrics
    gap_adasyn = train_acc_adasyn - test_acc_adasyn
    f1_gap_adasyn = train_f1_adasyn - test_f1_adasyn
    
    if gap_adasyn < 0.02:
        overfit_adasyn = "âœ… No Overfitting"
    elif gap_adasyn < 0.05:
        overfit_adasyn = "âš ï¸� Slight Overfitting"
    elif gap_adasyn < 0.10:
        overfit_adasyn = "ğŸŸ  Moderate Overfitting"
    else:
        overfit_adasyn = "ğŸ”´ Severe Overfitting"
    
    print(f"\nâœ… ADASYN Results:")
    print(f"   Train Accuracy:   {train_acc_adasyn:.4f} ({train_acc_adasyn*100:.2f}%)")
    print(f"   Test Accuracy:    {test_acc_adasyn:.4f} ({test_acc_adasyn*100:.2f}%)")
    print(f"   Accuracy Gap:     {gap_adasyn:.4f} ({gap_adasyn*100:.2f}%)")
    print(f"   Test F1-Score:    {test_f1_adasyn:.4f}")
    print(f"   Overfitting:      {overfit_adasyn}")
    print(f"   vs Baseline:      {(test_acc_adasyn - baseline_acc)*100:+.2f}%")
    
    experimental_results.append({
        'Method': 'ADASYN Sampling',
        'Train_Accuracy': train_acc_adasyn,
        'Test_Accuracy': test_acc_adasyn,
        'Accuracy_Gap': gap_adasyn,
        'Train_F1': train_f1_adasyn,
        'Test_F1': test_f1_adasyn,
        'F1_Gap': f1_gap_adasyn,
        'Gain_vs_Baseline': (test_acc_adasyn - baseline_acc)*100,
        'Overfit_Status': overfit_adasyn
    })
except Exception as e:
    print(f"â�Œ ADASYN failed: {e}")

# ============================================================================
# EXPERIMENT 2: SMOTE + TOMEK LINKS (HYBRID)
# ============================================================================

print("\n" + "="*80)
print("ğŸ§ª EXPERIMENT 2: SMOTE-TOMEK (Hybrid Approach)")
print("="*80)
print("CONCEPT: SMOTE oversample + Tomek Links clean overlapping samples")
print("WHY: Removes ambiguous samples at class boundaries")
print(f"Seed: {RANDOM_SEED}")

try:
    smote_tomek = SMOTETomek(random_state=RANDOM_SEED)
    X_train_hybrid, y_train_hybrid = smote_tomek.fit_resample(X_train_selected, y_train)
    
    print(f"\nTraining samples: {len(y_train)} â†’ {len(y_train_hybrid)}")
    
    rf_hybrid = RandomForestClassifier(
        n_estimators=600, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1
    )
    
    print("Training RandomForest with SMOTE-Tomek...")
    rf_hybrid.fit(X_train_hybrid, y_train_hybrid)
    
    # Training performance
    y_train_pred_hybrid = rf_hybrid.predict(X_train_hybrid)
    train_acc_hybrid = accuracy_score(y_train_hybrid, y_train_pred_hybrid)
    train_f1_hybrid = f1_score(y_train_hybrid, y_train_pred_hybrid, average='weighted')
    
    # Test performance
    y_test_pred_hybrid = rf_hybrid.predict(X_test_selected)
    test_acc_hybrid = accuracy_score(y_test, y_test_pred_hybrid)
    test_f1_hybrid = f1_score(y_test, y_test_pred_hybrid, average='weighted')
    
    gap_hybrid = train_acc_hybrid - test_acc_hybrid
    f1_gap_hybrid = train_f1_hybrid - test_f1_hybrid
    
    if gap_hybrid < 0.02:
        overfit_hybrid = "âœ… No Overfitting"
    elif gap_hybrid < 0.05:
        overfit_hybrid = "âš ï¸� Slight Overfitting"
    elif gap_hybrid < 0.10:
        overfit_hybrid = "ğŸŸ  Moderate Overfitting"
    else:
        overfit_hybrid = "ğŸ”´ Severe Overfitting"
    
    print(f"\nâœ… SMOTE-Tomek Results:")
    print(f"   Train Accuracy:   {train_acc_hybrid:.4f} ({train_acc_hybrid*100:.2f}%)")
    print(f"   Test Accuracy:    {test_acc_hybrid:.4f} ({test_acc_hybrid*100:.2f}%)")
    print(f"   Accuracy Gap:     {gap_hybrid:.4f} ({gap_hybrid*100:.2f}%)")
    print(f"   Test F1-Score:    {test_f1_hybrid:.4f}")
    print(f"   Overfitting:      {overfit_hybrid}")
    print(f"   vs Baseline:      {(test_acc_hybrid - baseline_acc)*100:+.2f}%")
    
    experimental_results.append({
        'Method': 'SMOTE-Tomek Hybrid',
        'Train_Accuracy': train_acc_hybrid,
        'Test_Accuracy': test_acc_hybrid,
        'Accuracy_Gap': gap_hybrid,
        'Train_F1': train_f1_hybrid,
        'Test_F1': test_f1_hybrid,
        'F1_Gap': f1_gap_hybrid,
        'Gain_vs_Baseline': (test_acc_hybrid - baseline_acc)*100,
        'Overfit_Status': overfit_hybrid
    })
except Exception as e:
    print(f"â�Œ SMOTE-Tomek failed: {e}")

# ============================================================================
# EXPERIMENT 3: XGBOOST WITH CLASS WEIGHTS
# ============================================================================

print("\n" + "="*80)
print("ğŸ§ª EXPERIMENT 3: XGBoost with Sample Weighting")
print("="*80)
print("CONCEPT: Use gradient boosting with class-aware weighting")
print("WHY: XGBoost can handle imbalance natively without synthetic samples")
print(f"Seed: {RANDOM_SEED}")

if XGBOOST_AVAILABLE:
    try:
        # Calculate sample weights
        class_counts = Counter(y_train)
        max_count = max(class_counts.values())
        sample_weights = np.array([max_count / class_counts[y] for y in y_train])
        
        xgb_clf = XGBClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            eval_metric='mlogloss',
            verbosity=0
        )
        
        print("Training XGBoost with weighted samples...")
        xgb_clf.fit(X_train_selected, y_train, sample_weight=sample_weights)
        
        # Training performance
        y_train_pred_xgb = xgb_clf.predict(X_train_selected)
        train_acc_xgb = accuracy_score(y_train, y_train_pred_xgb)
        train_f1_xgb = f1_score(y_train, y_train_pred_xgb, average='weighted')
        
        # Test performance
        y_test_pred_xgb = xgb_clf.predict(X_test_selected)
        test_acc_xgb = accuracy_score(y_test, y_test_pred_xgb)
        test_f1_xgb = f1_score(y_test, y_test_pred_xgb, average='weighted')
        
        gap_xgb = train_acc_xgb - test_acc_xgb
        f1_gap_xgb = train_f1_xgb - test_f1_xgb
        
        if gap_xgb < 0.02:
            overfit_xgb = "âœ… No Overfitting"
        elif gap_xgb < 0.05:
            overfit_xgb = "âš ï¸� Slight Overfitting"
        elif gap_xgb < 0.10:
            overfit_xgb = "ğŸŸ  Moderate Overfitting"
        else:
            overfit_xgb = "ğŸ”´ Severe Overfitting"
        
        print(f"\nâœ… XGBoost Results:")
        print(f"   Train Accuracy:   {train_acc_xgb:.4f} ({train_acc_xgb*100:.2f}%)")
        print(f"   Test Accuracy:    {test_acc_xgb:.4f} ({test_acc_xgb*100:.2f}%)")
        print(f"   Accuracy Gap:     {gap_xgb:.4f} ({gap_xgb*100:.2f}%)")
        print(f"   Test F1-Score:    {test_f1_xgb:.4f}")
        print(f"   Overfitting:      {overfit_xgb}")
        print(f"   vs Baseline:      {(test_acc_xgb - baseline_acc)*100:+.2f}%")
        print(f"   Advantage:        No synthetic data needed!")
        
        experimental_results.append({
            'Method': 'XGBoost + Weights',
            'Train_Accuracy': train_acc_xgb,
            'Test_Accuracy': test_acc_xgb,
            'Accuracy_Gap': gap_xgb,
            'Train_F1': train_f1_xgb,
            'Test_F1': test_f1_xgb,
            'F1_Gap': f1_gap_xgb,
            'Gain_vs_Baseline': (test_acc_xgb - baseline_acc)*100,
            'Overfit_Status': overfit_xgb
        })
    except Exception as e:
        print(f"â�Œ XGBoost failed: {e}")
else:
    print("â�Œ XGBoost not available - skipping experiment")

print("\n" + "="*80)
print("âœ… ALL EXPERIMENTS COMPLETE")
print("="*80)


# Create DataFrame and sort by Test Accuracy
df_experiments = pd.DataFrame(experimental_results).sort_values('Test_Accuracy', ascending=False)

# Define column widths
col_rank = 6
col_method = 40
col_test_acc = 15
col_test_f1 = 10

# Print table header
print("=" * (col_rank + col_method + col_test_acc + col_test_f1 + 10))
print(f"{'Rank':<{col_rank}} {'Method':<{col_method}} {'Test Accuracy':>{col_test_acc}} {'Test F1':>{col_test_f1}}")
print("=" * (col_rank + col_method + col_test_acc + col_test_f1 + 10))

# Print table rows
for rank, row in enumerate(df_experiments.itertuples(), 1):
    print(f"{rank:<{col_rank}} {row.Method:<{col_method}} {row.Test_Accuracy*100:>{col_test_acc-1}.2f}% {row.Test_F1:>{col_test_f1}.4f}")

print("=" * (col_rank + col_method + col_test_acc + col_test_f1 + 10))

# Save experimental results
df_experiments.to_csv("/kaggle/working/experimental_results.csv", index=False)
print("\nExperimental results saved to: experimental_results.csv")

# Find best approach
best_experiment = df_experiments.iloc[0]

print("\n" + "="*80)
print("WINNER: " + best_experiment['Method'])
print("="*80)
print(f"   Test Accuracy: {best_experiment['Test_Accuracy']*100:.2f}%")
print(f"   Test F1-Score: {best_experiment['Test_F1']:.4f}")

print("STATISTICAL INSIGHTS")
print("="*80)

print(f"\nğŸ“Š Across {len(experimental_results)} approaches:")
print(f"   Average test accuracy: {df_experiments['Test_Accuracy'].mean()*100:.2f}%")
print(f"   Best test accuracy:    {df_experiments['Test_Accuracy'].max()*100:.2f}%")
print(f"   Worst test accuracy:   {df_experiments['Test_Accuracy'].min()*100:.2f}%")
print(f"   Standard deviation:    {df_experiments['Test_Accuracy'].std()*100:.2f}%")


