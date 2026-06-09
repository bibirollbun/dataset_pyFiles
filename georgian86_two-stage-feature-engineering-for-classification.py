# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os

data_path = "/kaggle/input/malware-classification"

# List available files
print(os.listdir(data_path))


import pandas as pd

# Path to the dataset
data_path = "/kaggle/input/malware-classification"

# Load the labels
labels_df = pd.read_csv(f"{data_path}/trainLabels.csv")

# Show first 5 rows
labels_df.head()


import pandas as pd

data_path = "/kaggle/input/malware-classification"
df = pd.read_csv(f"{data_path}/trainLabels.csv")
df.head(10)  # shows 10 samples with Id and Class


# Mapping class numbers to family names
family_map = {
    1: "Ramnit",
    2: "Lollipop",
    3: "Kelihos_ver3",
    4: "Vundo",
    5: "Simda",
    6: "Tracur",
    7: "Kelihos_ver1",
    8: "Obfuscator.ACY",
    9: "Gatak"
}

# Add a new column for family name
labels_df["Family"] = labels_df["Class"].map(family_map)

# Preview updated labels
labels_df.head()



import pandas as pd

# Load labels
data_path = "/kaggle/input/malware-classification"
df = pd.read_csv(f"{data_path}/trainLabels.csv")

# Optional: Map class to family names
family_map = {
    1: "Ramnit", 2: "Lollipop", 3: "Kelihos_ver3", 4: "Vundo",
    5: "Simda", 6: "Tracur", 7: "Kelihos_ver1", 8: "Obfuscator.ACY", 9: "Gatak"
}
df["Family"] = df["Class"].map(family_map)

# Smart sampling: get min(10, count) rows per class
sampled_df = df.groupby("Class", group_keys=False).apply(
    lambda x: x.sample(n=min(10, len(x)), random_state=42)
).reset_index(drop=True)

# Show count per class in the sample
print(sampled_df["Class"].value_counts())

# Preview result
sampled_df.head()



for cls in sorted(sampled_df["Class"].unique()):
    ids = sampled_df[sampled_df["Class"] == cls]["Id"].tolist()
    print(f"Class {cls} Sample IDs:\n", ids, "\n")


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.countplot(data=labels_df, y="Family", order=labels_df["Family"].value_counts().index)
plt.title("Malware Family Distribution")
plt.xlabel("Number of Samples")
plt.ylabel("Malware Family")
plt.show()

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def extract_byte_histogram(file_path):
    counts = np.zeros(256, dtype=int)
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()[1:]  # skip the address
                for byte_str in parts:
                    if byte_str != '??':
                        try:
                            byte_val = int(byte_str, 16)
                            counts[byte_val] += 1
                        except ValueError:
                            continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return counts

# Path to dataset
data_path = "/kaggle/input/malware-classification"
bytes_path = os.path.join(data_path, "train")

# Load labels
label_df = pd.read_csv(os.path.join(data_path, "trainLabels.csv"))
sample_ids = label_df["Id"].tolist()[:100]  # First 100 samples for demo
label_map = dict(zip(label_df["Id"], label_df["Class"]))

X = []
y = []
file_names = []

for file_id in tqdm(sample_ids):
    file_path = os.path.join(bytes_path, file_id + ".bytes")
    if os.path.exists(file_path):
        hist = extract_byte_histogram(file_path)
        X.append(hist)
        y.append(label_map[file_id])
        file_names.append(file_id)


# Create the feature DataFrame
df_features = pd.DataFrame(X, columns=[f'byte_{i:02X}' for i in range(256)])
df_features["label"] = y
df_features["Id"] = file_names

# Map label → malware family
family_map = {
    1: "Ramnit", 2: "Lollipop", 3: "Kelihos_ver3", 4: "Vundo",
    5: "Simda", 6: "Tracur", 7: "Kelihos_ver1", 8: "Obfuscator.ACY", 9: "Gatak"
}
df_features["Family"] = df_features["label"].map(family_map)


# Count NaN values
print("Total NaNs:", df_features.isna().sum().sum())

# Check data types
print("Data types:\n", df_features.dtypes.value_counts())

# Preview suspicious rows
print(df_features[df_features.isna().any(axis=1)].head())


# Check if any rows are all NaN or all zeros
print(df_features.isnull().sum().sum())         # Total NaNs
print((df_features.drop(columns=["label"], errors='ignore') == 0).all(axis=1).sum())  # All-zero rows

# Optionally drop NaNs
df_features = df_features.dropna()

df_features.head()
df_features.describe()


!mkdir -p /kaggle/working/bytes
!7z l /kaggle/input/malware-classification/train.7z | grep '.bytes' | awk '{print $NF}' | head -n 2000 > /kaggle/working/bytes/bytes_list.txt

# Now extract just these 2000 files
!7z e /kaggle/input/malware-classification/train.7z -o/kaggle/working/bytes -i@/kaggle/working/bytes/bytes_list.txt


import os
import numpy as np
from tqdm import tqdm

def extract_byte_histogram(file_path):
    try:
        with open(file_path, 'r') as file:
            hex_lines = file.readlines()
        bytes_list = []
        for line in hex_lines:
            parts = line.strip().split()
            bytes_seq = parts[1:]  # ignore address part
            bytes_list.extend([b for b in bytes_seq if b != '??'])
        byte_vals = [int(b, 16) for b in bytes_list if len(b) == 2]
        hist = np.histogram(byte_vals, bins=256, range=(0, 255))[0]
        return hist
    except:
        return np.zeros(256)

# Extract features for a small sample of files
file_dir = '/kaggle/working/bytes'
sample_files = os.listdir(file_dir)[:2000]  # You can increase to 1000+

X = []
file_ids = []

for fname in tqdm(sample_files):
    if fname.endswith('.bytes'):
        f_id = fname.replace(".bytes", "")
        hist = extract_byte_histogram(os.path.join(file_dir, fname))
        X.append(hist)
        file_ids.append(f_id)


df_features = pd.DataFrame(X, columns=[f'byte_{i:02X}' for i in range(256)])
df_features["Id"] = file_ids
df_features = df_features.merge(labels_df, on="Id")


df_features.describe()

# Correlation heatmap (optional)
import seaborn as sns
plt.figure(figsize=(12, 8))
sns.heatmap(df_features.drop(columns=["Id", "Class", "Family"]).corr(), cmap="viridis")
plt.title("Feature Correlation")
plt.show()


df_features.to_csv("/kaggle/working/byte_histogram_features.csv", index=False)


import numpy as np

# Create correlation matrix
corr_matrix = df_features.drop(columns=["Id", "Class", "Family"]).corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop those features
df_reduced = df_features.drop(columns=to_drop)
print(f"Dropped {len(to_drop)} highly correlated features.")


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_features["Family"] = le.fit_transform(df_features["Family"])  # You can also use df_reduced if used earlier


from sklearn.model_selection import train_test_split

X = df_reduced.drop(columns=["Id", "Class", "Family"])
y = df_reduced["Family"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Map original Class (1–9 in trainLabels.csv) to 0–8
y_all = df_reduced["Class"].values - 1   # now in [0..8]
class_names = np.array([
    "Ramnit", "Lollipop", "Kelihos_ver3", "Vundo",
    "Simda", "Tracur", "Kelihos_ver1", "Obfuscator.ACY", "Gatak"
])

# Prepare X matrix
X_all = df_reduced.drop(columns=["Id", "Class", "Family"], errors="ignore")

# Train/Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

n_classes = len(np.unique(y_train))
print("Labels range:", np.unique(y_train), "Num classes:", n_classes)



import xgboost as xgb
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob', num_class=n_classes,
    eval_metric=['mlogloss','merror'],
    n_estimators=800, learning_rate=0.05, max_depth=7,
    subsample=0.7, colsample_bytree=0.7, gamma=0.1,
    tree_method='hist', random_state=42, n_jobs=-1
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train,y_train), (X_test,y_test)],
    callbacks=[xgb.callback.EarlyStopping(
        rounds=50, metric_name='mlogloss', data_name='validation_1', save_best=True)],
    verbose=False
)

# Leaf features
X_train_leaf = xgb_model.apply(X_train)
X_test_leaf  = xgb_model.apply(X_test)



import numpy as np
k = 25
importances = xgb_model.feature_importances_
topk_idx = np.argsort(importances)[-k:]
# If X_train/X_test are DataFrames, .iloc is fine; if they are numpy arrays, index directly
X_train_sel = X_train.iloc[:, topk_idx] if hasattr(X_train, "iloc") else X_train[:, topk_idx]
X_test_sel  = X_test.iloc[:, topk_idx]  if hasattr(X_test, "iloc")  else X_test[:, topk_idx]

X_train_comb = np.hstack([X_train_sel, X_train_leaf])
X_test_comb  = np.hstack([X_test_sel,  X_test_leaf])
num_original_selected = X_train_sel.shape[1]
categorical_cols = list(range(num_original_selected, X_train_comb.shape[1]))



import lightgbm as lgb
lgbm_model = lgb.LGBMClassifier(
    objective='multiclass', num_class=n_classes,
    boosting_type='gbdt', metric=['multi_logloss','multi_error'],
    num_leaves=31, learning_rate=0.05, feature_fraction=0.9,
    n_estimators=2000, random_state=42, n_jobs=-1
)
lgbm_model.fit(
    X_train_comb, y_train,
    eval_set=[(X_train_comb,y_train), (X_test_comb,y_test)],
    eval_metric=['multi_logloss','multi_error'],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    categorical_feature=categorical_cols
)



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = lgbm_model.predict(X_test_comb)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("Confusion Matrix (Two-Stage LGBM on Combined Features)")
plt.tight_layout()
plt.show()



# =========================
# Labels 0..8 and train/test split
# =========================
import numpy as np
import pandas as pd

# Map Class 1..9 -> 0..8
y_all = df_reduced["Class"].values - 1
class_names = np.array([
    "Ramnit","Lollipop","Kelihos_ver3","Vundo",
    "Simda","Tracur","Kelihos_ver1","Obfuscator.ACY","Gatak"
])

# Feature matrix
X_all = df_reduced.drop(columns=["Id","Class","Family"], errors="ignore")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)
n_classes = len(np.unique(y_train))
print("Shapes:", X_train.shape, X_test.shape, "Classes:", n_classes)

# =========================
# Stage 1: XGBoost (multiclass)
# =========================
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=n_classes,
    eval_metric=['mlogloss','merror'],
    n_estimators=800,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.7,
    colsample_bytree=0.7,
    gamma=0.1,
    tree_method='hist',
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train,y_train),(X_test,y_test)],
    callbacks=[xgb.callback.EarlyStopping(
        rounds=50, metric_name='mlogloss', data_name='validation_1', save_best=True)],
    verbose=False
)

# =========================
# Leaf features from Stage 1
# =========================
X_train_leaf = xgb_model.apply(X_train)  # shape: (n_train, n_trees)
X_test_leaf  = xgb_model.apply(X_test)
print("Leaf features:", X_train_leaf.shape, X_test_leaf.shape)

# =========================
# Top-k original feature selection by Stage 1 importances
# =========================
k = 25
importances = xgb_model.feature_importances_
topk_idx = np.argsort(importances)[-k:]

# If X_train is DataFrame use .iloc, else index arrays
X_train_sel = X_train.iloc[:, topk_idx] if hasattr(X_train, "iloc") else X_train[:, topk_idx]
X_test_sel  = X_test.iloc[:, topk_idx]  if hasattr(X_test,  "iloc") else X_test[:,  topk_idx]

# =========================
# Stage 2: Combine top-k originals + leaf features
# =========================
import numpy as np
X_train_comb = np.hstack([X_train_sel, X_train_leaf])
X_test_comb  = np.hstack([X_test_sel,  X_test_leaf])

num_original_selected = X_train_sel.shape[1]
categorical_cols = list(range(num_original_selected, X_train_comb.shape[1]))  # treat leaves as categorical

# =========================
# LightGBM (multiclass) on combined features
# =========================
import lightgbm as lgb

lgbm_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=n_classes,
    boosting_type='gbdt',
    metric=['multi_logloss','multi_error'],
    num_leaves=31,
    learning_rate=0.05,
    feature_fraction=0.9,
    n_estimators=2000,
    random_state=42,
    n_jobs=-1
)
lgbm_model.fit(
    X_train_comb, y_train,
    eval_set=[(X_train_comb,y_train),(X_test_comb,y_test)],
    eval_metric=['multi_logloss','multi_error'],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    categorical_feature=categorical_cols
)

# =========================
# Evaluation (multi-class)
# =========================
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = lgbm_model.predict(X_test_comb)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("Confusion Matrix (Two-Stage Classification)")
plt.tight_layout(); plt.show()

# Optional: macro/micro ROC-AUC (OvR)
try:
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_auc_score, roc_curve
    from itertools import cycle

    y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
    y_proba = lgbm_model.predict_proba(X_test_comb)  # (n_samples, n_classes)
    macro_auc = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
    micro_auc = roc_auc_score(y_test_bin, y_proba, average='micro', multi_class='ovr')
    print(f"Macro ROC-AUC: {macro_auc:.4f}  |  Micro ROC-AUC: {micro_auc:.4f}")

    # Plot subset of per-class ROC curves
    max_curves = min(n_classes, 6)
    colors = cycle(['aqua','darkorange','cornflowerblue','darkgreen','crimson','gold'])
    plt.figure(figsize=(7,6))
    for i, color in zip(range(max_curves), colors):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        plt.plot(fpr, tpr, color=color, lw=2, label=class_names[i])
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('Per-class ROC (subset)')
    plt.legend(loc='lower right', fontsize=8); plt.grid(True); plt.tight_layout(); plt.show()
except Exception as e:
    print("ROC-AUC not computed:", e)

# =========================
# Feature importance for interpretability
# =========================
# Names for combined features: top-k originals + leaf features
if hasattr(X_all, "columns"):
    orig_feature_names = X_all.columns
    topk_names = list(orig_feature_names[topk_idx])
else:
    topk_names = [f"feat_{i}" for i in topk_idx]
leaf_names = [f"xgb_leaf_{i}" for i in range(X_train_leaf.shape[1])]
combined_feature_names = topk_names + leaf_names

imp_df = pd.DataFrame({
    "feature": combined_feature_names,
    "importance": lgbm_model.feature_importances_
}).sort_values("importance", ascending=False)

plt.figure(figsize=(10,8))
sns.barplot(x="importance", y="feature", data=imp_df.head(25), palette="viridis")
plt.title("Top 25 Feature Importances (Two-Stage Classification)")
plt.xlabel("Importance"); plt.ylabel("Feature")
plt.tight_layout(); plt.show()



from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

# Calibrate LightGBM probabilities using validation data
calibrator = CalibratedClassifierCV(lgbm_model, method='isotonic', cv='prefit')
calibrator.fit(X_test_comb, y_test)

# Evaluate calibrated probabilities
proba_raw = lgbm_model.predict_proba(X_test_comb)
proba_cal = calibrator.predict_proba(X_test_comb)

ll_raw = log_loss(y_test, proba_raw, labels=np.arange(n_classes))
ll_cal = log_loss(y_test, proba_cal, labels=np.arange(n_classes))
print(f"LogLoss (raw): {ll_raw:.4f} | LogLoss (calibrated): {ll_cal:.4f}")



import joblib
import json
from pathlib import Path

out_dir = Path("/kaggle/working/two_stage_cls")
out_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(xgb_model, out_dir/"stage1_xgb_model.pkl")
joblib.dump(lgbm_model, out_dir/"stage2_lgbm_model.pkl")
joblib.dump(calibrator, out_dir/"stage2_lgbm_calibrator.pkl")

# Save feature selection indices and class names
np.save(out_dir/"topk_idx.npy", topk_idx)
np.save(out_dir/"class_names.npy", class_names)

# Save basic metadata
meta = {
    "k_top_features": int(len(topk_idx)),
    "num_leaf_features": int(X_train_leaf.shape[1]),
    "num_classes": int(n_classes)
}
with open(out_dir/"metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Artifacts saved to:", out_dir)



from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for fold, (tr, va) in enumerate(skf.split(X_train_comb, y_train), 1):
    X_tr, X_va = X_train_comb[tr], X_train_comb[va]
    y_tr, y_va = y_train[tr], y_train[va]
    lgb_cv = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=n_classes,
        boosting_type='gbdt',
        metric='multi_error',
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.9,
        n_estimators=1000,
        random_state=42,
        n_jobs=-1
    )
    lgb_cv.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric='multi_error',
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        categorical_feature=categorical_cols
    )
    y_va_pred = lgb_cv.predict(X_va)
    acc = accuracy_score(y_va, y_va_pred)
    cv_scores.append(acc)
    print(f"Fold {fold} Accuracy: {acc:.4f}")

print(f"CV Accuracy: mean={np.mean(cv_scores):.4f}, std={np.std(cv_scores):.4f}")



import pandas as pd
import numpy as np

# Load labels CSV
labels_df = pd.read_csv("/kaggle/input/malware-classification/trainLabels.csv")

# Make your features DataFrame here (histograms, entropy, etc.)
# Example dummy:
df_features = pd.DataFrame(np.random.rand(len(labels_df), 256),
                           columns=[f'byte_{i:02X}' for i in range(256)])
df_features['Class'] = labels_df['Class']

# Optionally drop highly correlated features
# df_reduced = ...

# Final combined DataFrame with features + Class
df_reduced = df_features



print(df_reduced.shape)
print(df_reduced.columns[:10])


import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import xgboost as xgb, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from itertools import cycle

# ===== Prepare labels 0..8 and split =====
y_all = df_reduced["Class"].values - 1
class_names = np.array([
    "Ramnit","Lollipop","Kelihos_ver3","Vundo",
    "Simda","Tracur","Kelihos_ver1","Obfuscator.ACY","Gatak"
])
X_all = df_reduced.drop(columns=["Id","Class","Family"], errors="ignore")

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)
n_classes = len(np.unique(y_train))

# ===== Stage 1: XGBoost multiclass =====
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob', num_class=n_classes,
    eval_metric=['mlogloss','merror'], n_estimators=800,
    learning_rate=0.05, max_depth=7, subsample=0.7,
    colsample_bytree=0.7, gamma=0.1, tree_method='hist',
    random_state=42, n_jobs=-1
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train,y_train),(X_test,y_test)],
    callbacks=[xgb.callback.EarlyStopping(
        rounds=50, metric_name='mlogloss', data_name='validation_1', save_best=True)],
    verbose=False
)

# --- Stage 1 Learning Curves ---
results = xgb_model.evals_result()
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(epochs)
plt.style.use('ggplot'); plt.figure(figsize=(18,5))
plt.subplot(1,3,1)
plt.plot(x_axis, results['validation_0']['mlogloss'], label='Train mlogloss')
plt.plot(x_axis, results['validation_1']['mlogloss'], label='Val mlogloss')
plt.title('XGB Multiclass LogLoss'); plt.grid(True); plt.legend()
plt.subplot(1,3,2)
plt.plot(x_axis, [1-e for e in results['validation_0']['merror']], label='Train Acc')
plt.plot(x_axis, [1-e for e in results['validation_1']['merror']], label='Val Acc')
plt.title('XGB Accuracy'); plt.grid(True); plt.legend()
if 'auc' in results['validation_0']:
    plt.subplot(1,3,3)
    plt.plot(x_axis, results['validation_0']['auc'], label='Train mean OvR AUC')
    plt.plot(x_axis, results['validation_1']['auc'], label='Val mean OvR AUC')
    plt.title('XGB Mean OvR AUC'); plt.grid(True); plt.legend()
plt.tight_layout(); plt.show()

# ===== Stage 2: Prepare combined features =====
X_train_leaf = xgb_model.apply(X_train)
X_test_leaf  = xgb_model.apply(X_test)
k_top = 25
topk_idx = np.argsort(xgb_model.feature_importances_)[-k_top:]
X_train_sel = X_train.iloc[:, topk_idx] if hasattr(X_train,"iloc") else X_train[:, topk_idx]
X_test_sel  = X_test.iloc[:,  topk_idx] if hasattr(X_test,"iloc")  else X_test[:,  topk_idx]
X_train_comb = np.hstack([X_train_sel, X_train_leaf])
X_test_comb  = np.hstack([X_test_sel,  X_test_leaf])
categorical_cols = list(range(X_train_sel.shape[1], X_train_comb.shape[1]))

# ===== Stage 2: LightGBM multiclass =====
lgbm_model = lgb.LGBMClassifier(
    objective='multiclass', num_class=n_classes,
    boosting_type='gbdt', metric=['multi_logloss','multi_error'],
    num_leaves=31, learning_rate=0.05, feature_fraction=0.9,
    n_estimators=2000, random_state=42, n_jobs=-1
)
lgbm_model.fit(
    X_train_comb, y_train,
    eval_set=[(X_train_comb,y_train),(X_test_comb,y_test)],
    eval_metric=['multi_logloss','multi_error'],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    categorical_feature=categorical_cols
)

# ===== Evaluation =====
y_pred = lgbm_model.predict(X_test_comb)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("Confusion Matrix"); plt.tight_layout(); plt.show()

# ===== ROC-AUC (macro/micro OvR) =====
try:
    y_bin = label_binarize(y_test, classes=np.arange(n_classes))
    y_proba = lgbm_model.predict_proba(X_test_comb)
    print(f"Macro ROC-AUC: {roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr'):.4f}")
    print(f"Micro ROC-AUC: {roc_auc_score(y_bin, y_proba, average='micro', multi_class='ovr'):.4f}")
    max_curves = min(n_classes, 6)
    colors = cycle(['aqua','darkorange','cornflowerblue','darkgreen','crimson','gold'])
    plt.figure(figsize=(7,6))
    for i, color in zip(range(max_curves), colors):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        plt.plot(fpr, tpr, color=color, lw=2, label=class_names[i])
    plt.plot([0,1],[0,1],'k--'); plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title('Per-class ROC (subset)'); plt.legend(fontsize=8); plt.grid(True)
    plt.tight_layout(); plt.show()
except Exception as e:
    print("ROC-AUC not computed:", e)

# ===== Feature Importances =====
leaf_names = [f"xgb_leaf_{i}" for i in range(X_train_leaf.shape[1])]
topk_names = list(X_all.columns[topk_idx]) if hasattr(X_all,"columns") else [f"feat_{i}" for i in topk_idx]
combined_names = topk_names + leaf_names
imp_df = pd.DataFrame({
    "feature": combined_names,
    "importance": lgbm_model.feature_importances_
}).sort_values("importance", ascending=False)
plt.figure(figsize=(10,8))
sns.barplot(x="importance", y="feature", data=imp_df.head(25), palette="viridis")
plt.title("Top 25 Feature Importances"); plt.tight_layout(); plt.show()


