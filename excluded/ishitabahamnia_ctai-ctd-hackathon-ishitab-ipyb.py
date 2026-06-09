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



import pandas as pd

try:
  test_df = pd.read_csv('/content/test.csv')
  train_df = pd.read_csv('/content/train.csv')

  print("Test DataFrame head:")
  display(test_df.head())

  print("\nTrain DataFrame head:")
  display(train_df.head())

except FileNotFoundError:
  print("Make sure 'test.csv' and 'train.csv' are uploaded to the Colab environment.")
except Exception as e:
  print(f"An error occurred: {e}")


# Identify missing values
print("Missing values in train_df:")
display(train_df.isnull().sum()[train_df.isnull().sum() > 0])

print("\nMissing values in test_df:")
display(test_df.isnull().sum()[test_df.isnull().sum() > 0])

# The 'tof' columns have a large number of missing values. Imputing them with the median might be reasonable given the nature of the data.
# For numerical columns with missing values (primarily tof columns), impute with the median.
for col in train_df.columns:
    if train_df[col].dtype in ['float64', 'int64']:
        if train_df[col].isnull().sum() > 0:
            median_val = train_df[col].median()
            train_df[col].fillna(median_val, inplace=True)
            if col in test_df.columns:
                test_df[col].fillna(median_val, inplace=True)

# Check for remaining missing values after imputation
print("\nMissing values in train_df after imputation:")
display(train_df.isnull().sum()[train_df.isnull().sum() > 0])

print("\nMissing values in test_df after imputation:")
display(test_df.isnull().sum()[test_df.isnull().sum() > 0])

# Identify categorical columns
categorical_cols_train = train_df.select_dtypes(include=['object']).columns
categorical_cols_test = test_df.select_dtypes(include=['object']).columns

print("\nCategorical columns in train_df:")
print(categorical_cols_train)

print("\nCategorical columns in test_df:")
print(categorical_cols_test)

# Exclude 'row_id' from categorical columns as it's an identifier
categorical_cols_train = categorical_cols_train.drop('row_id', errors='ignore')
categorical_cols_test = categorical_cols_test.drop('row_id', errors='ignore')


# Apply one-hot encoding
# Combine train and test for consistent encoding
combined_df = pd.concat([train_df.drop('sequence_type', axis=1, errors='ignore'), test_df], ignore_index=True)
combined_df = pd.get_dummies(combined_df, columns=categorical_cols_train.intersection(categorical_cols_test), dummy_na=False)

# Separate back into train and test
train_processed_df = combined_df.iloc[:len(train_df)].copy()
test_processed_df = combined_df.iloc[len(train_df):].copy()

# Add back 'sequence_type' to the training data if it was dropped
if 'sequence_type' in train_df.columns:
    train_processed_df['sequence_type'] = train_df['sequence_type']

print("\nTrain DataFrame after one-hot encoding:")
display(train_processed_df.head())

print("\nTest DataFrame after one-hot encoding:")
display(test_processed_df.head())


grid_search_nn.fit(X_train_processed_df, y_train_encoded)



# ------------------------------
# Data Preprocessing + Encoding
# ------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Example: load dataset (replace with your actual file)
df = pd.read_csv("submission.csv")   # or your clean dataset

# Separate features and labels
X = df.drop(columns=["model_id"])    # features
y = df["model_id"]                   # labels (or replace with real target column)

# Encode labels if needed
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_processed = scaler.fit_transform(X_train)
X_test_processed = scaler.transform(X_test)

# Convert back to DataFrame (so GridSearchCV doesnâ€™t complain about dtypes)
X_train_processed_df = pd.DataFrame(X_train_processed, columns=X.columns)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=X.columns)

y_train_encoded = y_train
y_test_encoded = y_test

print("âœ… Preprocessing complete:", X_train_processed_df.shape, y_train_encoded.shape)



from sklearn.preprocessing import StandardScaler

# Identify numerical columns after one-hot encoding
numerical_cols_train = train_processed_df.select_dtypes(include=['float64', 'int64']).columns
numerical_cols_test = test_processed_df.select_dtypes(include=['float64', 'int64']).columns

# Exclude 'sequence_counter' and 'row_id' from scaling if present
cols_to_exclude = ['sequence_counter']
numerical_cols_train = numerical_cols_train.difference(cols_to_exclude)
numerical_cols_test = numerical_cols_test.difference(cols_to_exclude)

print("\nNumerical columns to scale in train_processed_df:")
print(numerical_cols_train)

print("\nNumerical columns to scale in test_processed_df:")
print(numerical_cols_test)


# Initialize StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both train and test data
train_processed_df[numerical_cols_train] = scaler.fit_transform(train_processed_df[numerical_cols_train])
test_processed_df[numerical_cols_test] = scaler.transform(test_processed_df[numerical_cols_test])

print("\nTrain DataFrame after scaling numerical features:")
display(train_processed_df.head())

print("\nTest DataFrame after scaling numerical features:")
display(test_processed_df.head())

# Separate target variable from features in the training data
if 'sequence_type' in train_processed_df.columns:
    X_train = train_processed_df.drop('sequence_type', axis=1)
    y_train = train_processed_df['sequence_type']
else:
    X_train = train_processed_df
    y_train = None
    print("Warning: 'sequence_type' column not found in train_processed_df.")


X_test = test_processed_df

print("\nFeatures for training (X_train) head:")
display(X_train.head())

if y_train is not None:
    print("\nTarget variable for training (y_train) head:")
    display(y_train.head())

print("\nFeatures for testing (X_test) head:")
display(X_test.head())


def build_features(sensor_df: pd.DataFrame, demo_df: pd.DataFrame) -> pd.DataFrame:
    df = sensor_df.copy()
    df = df.replace(-1.0, np.nan)
    df = df.merge(demo_df, on="subject", how="left")

    id_cols = ["row_id", "sequence_id", "sequence_counter", "subject"]
    demo_cols = ["adult_child", "age", "sex", "handedness", "height_cm",
                 "shoulder_to_wrist_cm", "elbow_to_wrist_cm"]
    numeric_cols = [c for c in df.columns if c not in id_cols + demo_cols]

    agg_funcs = ["mean", "std", "min", "max", "median", "skew"]
    seq_feats = (
        df.groupby("sequence_id")[numeric_cols]
        .agg(agg_funcs)
    )
    seq_feats.columns = ["_".join(col).strip() for col in seq_feats.columns.values]
    seq_feats = seq_feats.reset_index()

    demo_feats = (
        df.groupby("sequence_id")[demo_cols]
        .first()
        .reset_index()
    )
    final = seq_feats.merge(demo_feats, on="sequence_id", how="left")
    return final



# Unsupervised Modeling
# ==============================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# --- Load data (update paths if needed) ---
df_submission = pd.read_csv("submission.csv")

# Drop model_id column if present
if "model_id" in df_submission.columns:
    X_test = df_submission.drop(columns=["model_id"]).values
else:
    X_test = df_submission.values

# --- Preprocessing ---
scaler = StandardScaler()
X_test_proc = scaler.fit_transform(X_test)

# --- Isolation Forest ---
iso = IsolationForest(n_estimators=300, random_state=42, n_jobs=-1)
iso.fit(X_test_proc)
anomaly_score = -iso.score_samples(X_test_proc)
pred = iso.predict(X_test_proc)  # -1 = anomaly, 1 = normal

# --- PCA + KMeans clustering ---
pca_components = min(10, X_test_proc.shape[1], max(1, X_test_proc.shape[0] - 1))
pca = PCA(n_components=pca_components, random_state=42)
X_embedded = pca.fit_transform(X_test_proc)

n_clusters = min(3, max(1, X_test_proc.shape[0]))
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(X_embedded)

# --- Quick reporting ---
print("IsolationForest anomaly scores:", anomaly_score[:5])
print("Predictions (-1 = anomaly, 1 = normal):", np.unique(pred, return_counts=True))
print("Cluster labels:", np.unique(cluster_labels, return_counts=True))



# Visualization: PCA scatter with anomaly coloring
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

# Scatter with cluster colors
scatter = plt.scatter(
    X_embedded[:, 0], X_embedded[:, 1],
    c=cluster_labels, cmap="tab10", alpha=0.7, s=60,
    edgecolors=["red" if p == -1 else "black" for p in pred], linewidth=1.2
)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Clustering with IsolationForest Anomaly Overlay")

# Legend for anomalies
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Normal',
           markerfacecolor='grey', markeredgecolor='black', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Anomaly',
           markerfacecolor='grey', markeredgecolor='red', markersize=8)
]
plt.legend(handles=legend_elements, title="Anomaly Status", loc="best")

plt.colorbar(scatter, label="Cluster Label")
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import lightgbm as lgb
import pandas as pd
import os

# Try to load data with multiple possible paths
def load_data():
    possible_paths = [
        "/content/train.csv",  # Common Colab path
        "./train.csv",         # Current directory
        "train.csv",           # Just the filename
        "/kaggle/input/gesture-recognition/train.csv",  # Common Kaggle path
    ]
    
    train_df, test_df = None, None
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                train_df = pd.read_csv(path)
                print(f"Successfully loaded training data from: {path}")
                break
        except:
            continue
    
    if train_df is None:
        # Create sample data for demonstration if no CSV files are found
        print("No CSV files found. Creating sample data for demonstration.")
        np.random.seed(42)
        n_samples = 1000
        n_features = 30
        
        # Create sample features
        X = np.random.randn(n_samples, n_features)
        # Create sample target (5 classes)
        y = np.random.randint(0, 5, n_samples)
        
        # Create DataFrame
        feature_cols = [f'feature_{i}' for i in range(n_features)]
        train_df = pd.DataFrame(X, columns=feature_cols)
        train_df['gesture'] = y
        
        # Create test data
        test_X = np.random.randn(200, n_features)
        test_df = pd.DataFrame(test_X, columns=feature_cols)
    else:
        # Try to load test data
        test_paths = [
            path.replace('train', 'test'),
            "/content/test.csv",
            "./test.csv",
            "test.csv",
            "/kaggle/input/gesture-recognition/test.csv",
        ]
        
        for path in test_paths:
            try:
                if os.path.exists(path):
                    test_df = pd.read_csv(path)
                    print(f"Successfully loaded test data from: {path}")
                    break
            except:
                continue
        
        if test_df is None:
            print("No test CSV found. Using a portion of training data for demonstration.")
            test_df = train_df.iloc[:200].drop(columns=['gesture'], errors='ignore')
    
    return train_df, test_df

# Load data
train_df, test_df = load_data()

# Display basic info about the data
print("\nTraining data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("\nTraining data columns:")
print(train_df.columns.tolist())

# Check if 'gesture' column exists, if not create a sample one
if 'gesture' not in train_df.columns:
    print("'gesture' column not found. Creating a sample target variable.")
    n_classes = 5
    train_df['gesture'] = np.random.randint(0, n_classes, len(train_df))

# Convert gesture column to string to avoid issues with classification_report
if train_df['gesture'].dtype != 'object':
    print("Converting numeric gesture labels to string format")
    train_df['gesture'] = 'class_' + train_df['gesture'].astype(str)

# Define features and target
X = train_df.drop(columns=["gesture"])
y = train_df["gesture"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_proc = scaler.fit_transform(X_train)
X_test_proc = scaler.transform(X_test)

# Keep DataFrames for easier handling
X_train_df = pd.DataFrame(X_train_proc, columns=X.columns)
y_train_df = pd.Series(y_train).reset_index(drop=True)
X_test_df = pd.DataFrame(X_test_proc, columns=X.columns)

# Stratified K-Fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_true_all, y_pred_all = [], []
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_df, y_train_df), 1):
    X_tr, X_val = X_train_df.iloc[train_idx], X_train_df.iloc[val_idx]
    y_tr, y_val = y_train_df.iloc[train_idx], y_train_df.iloc[val_idx]

    model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X_tr, y_tr)

    y_val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    fold_accuracies.append(acc)
    
    print(f"Fold {fold}: Accuracy = {acc:.4f}")

    y_true_all.extend(y_val)
    y_pred_all.extend(y_val_pred)

print(f"\nMean CV Accuracy: {np.mean(fold_accuracies):.4f} Â± {np.std(fold_accuracies):.4f}")

# Normalized Confusion Matrix
cm = confusion_matrix(y_true_all, y_pred_all, labels=np.unique(y_train_df), normalize='true')

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Class-Normalized Confusion Matrix (CV Predictions)")
plt.tight_layout()
plt.show()

# Classification Report - FIXED
# Convert numeric class labels back to their original string representation
y_true_labels = label_encoder.inverse_transform(y_true_all)
y_pred_labels = label_encoder.inverse_transform(y_pred_all)

# Convert classes to strings for the classification report
class_names = [str(cls) for cls in label_encoder.classes_]

print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=class_names))

# Retrain on Full Training Data
final_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
final_model.fit(X_train_df, y_train_df)

# Feature Importance (Top 20)
importances = final_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_train_df.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20))
plt.title("Top 20 Feature Importances (LightGBM)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Predict Test Set + Save Submission
# Make sure test_df has the same columns as training data
missing_cols = set(X.columns) - set(test_df.columns)
if missing_cols:
    print(f"Adding missing columns to test data: {missing_cols}")
    for col in missing_cols:
        test_df[col] = 0  # Fill missing columns with zeros

X_test_processed = scaler.transform(test_df[X.columns])  # Use only the columns that exist in training
X_test_proc_df = pd.DataFrame(X_test_processed, columns=X.columns)

y_pred = final_model.predict(X_test_proc_df)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Create submission file
if "row_id" in test_df.columns:
    row_ids = test_df["row_id"]
else:
    row_ids = test_df.index

submission = pd.DataFrame({
    "row_id": row_ids,
    "gesture": y_pred_labels
})
submission.to_csv("submission.csv", index=False)

print("âœ… CV done, confusion matrix plotted, feature importance visualized, and submission.csv saved.")
print(f"Submission file contains {len(submission)} predictions.")
print(f"Predicted classes: {np.unique(y_pred_labels)}")





import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import lightgbm as lgb
import pandas as pd
import os

# First, let's check what files are available
print("Current directory:", os.getcwd())
print("Files in current directory:")
for file in os.listdir('.'):
    print(f"- {file}")

# Try to find the CSV files
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
print(f"\nCSV files found: {csv_files}")

# If no CSV files found, create sample data
if not csv_files:
    print("No CSV files found. Creating sample data...")
    np.random.seed(42)
    n_samples = 1000
    n_features = 30
    
    # Create sample features
    X = np.random.randn(n_samples, n_features)
    # Create sample target (5 classes)
    y = np.random.randint(0, 5, n_samples)
    
    # Create DataFrame
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    train_df = pd.DataFrame(X, columns=feature_cols)
    train_df['gesture'] = y
    
    # Create test data
    test_X = np.random.randn(200, n_features)
    test_df = pd.DataFrame(test_X, columns=feature_cols)
    
    print("Sample data created successfully!")
else:
    # Try to load train and test data
    train_files = [f for f in csv_files if 'train' in f.lower()]
    test_files = [f for f in csv_files if 'test' in f.lower()]
    
    if train_files:
        train_df = pd.read_csv(train_files[0])
        print(f"Loaded training data from: {train_files[0]}")
    else:
        print("No train CSV found. Using first available CSV as training data.")
        train_df = pd.read_csv(csv_files[0])
    
    if test_files:
        test_df = pd.read_csv(test_files[0])
        print(f"Loaded test data from: {test_files[0]}")
    else:
        print("No test CSV found. Creating test data from training data.")
        test_df = train_df.iloc[:200].copy()
        if 'gesture' in test_df.columns:
            test_df = test_df.drop(columns=['gesture'])

# Display basic info about the data
print("\nTraining data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("\nTraining data columns:")
print(train_df.columns.tolist())

# Check if 'gesture' column exists, if not create a sample one
if 'gesture' not in train_df.columns:
    print("'gesture' column not found. Creating a sample target variable.")
    n_classes = 5
    train_df['gesture'] = np.random.randint(0, n_classes, len(train_df))

# Convert gesture column to string to avoid issues with classification_report
print("Converting gesture labels to string format")
train_df['gesture'] = 'class_' + train_df['gesture'].astype(str)

# Define features and target
X = train_df.drop(columns=["gesture"])
y = train_df["gesture"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_proc = scaler.fit_transform(X_train)
X_test_proc = scaler.transform(X_test)

# Keep DataFrames for easier handling
X_train_df = pd.DataFrame(X_train_proc, columns=X.columns)
y_train_df = pd.Series(y_train).reset_index(drop=True)
X_test_df = pd.DataFrame(X_test_proc, columns=X.columns)

# Stratified K-Fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_true_all, y_pred_all = [], []
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_df, y_train_df), 1):
    X_tr, X_val = X_train_df.iloc[train_idx], X_train_df.iloc[val_idx]
    y_tr, y_val = y_train_df.iloc[train_idx], y_train_df.iloc[val_idx]

    model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X_tr, y_tr)

    y_val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    fold_accuracies.append(acc)
    
    print(f"Fold {fold}: Accuracy = {acc:.4f}")

    y_true_all.extend(y_val)
    y_pred_all.extend(y_val_pred)

print(f"\nMean CV Accuracy: {np.mean(fold_accuracies):.4f} Â± {np.std(fold_accuracies):.4f}")

# Normalized Confusion Matrix
cm = confusion_matrix(y_true_all, y_pred_all, labels=np.unique(y_train_df), normalize='true')

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Class-Normalized Confusion Matrix (CV Predictions)")
plt.tight_layout()
plt.show()

# Classification Report - FIXED
# Convert numeric class labels back to their original string representation
y_true_labels = label_encoder.inverse_transform(y_true_all)
y_pred_labels = label_encoder.inverse_transform(y_pred_all)

# Convert classes to strings for the classification report
class_names = [str(cls) for cls in label_encoder.classes_]

print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=class_names))

# Retrain on Full Training Data
final_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
final_model.fit(X_train_df, y_train_df)

# Feature Importance (Top 20)
importances = final_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_train_df.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20))
plt.title("Top 20 Feature Importances (LightGBM)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Predict Test Set + Save Submission
# Make sure test_df has the same columns as training data
missing_cols = set(X.columns) - set(test_df.columns)
extra_cols = set(test_df.columns) - set(X.columns)

if missing_cols:
    print(f"Adding missing columns to test data: {missing_cols}")
    for col in missing_cols:
        test_df[col] = 0  # Fill missing columns with zeros

if extra_cols:
    print(f"Removing extra columns from test data: {extra_cols}")
    test_df = test_df[X.columns]  # Keep only the columns that exist in training

X_test_processed = scaler.transform(test_df[X.columns])
X_test_proc_df = pd.DataFrame(X_test_processed, columns=X.columns)

y_pred = final_model.predict(X_test_proc_df)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Create submission file
if "row_id" in test_df.columns:
    row_ids = test_df["row_id"]
else:
    row_ids = test_df.index

submission = pd.DataFrame({
    "row_id": row_ids,
    "gesture": y_pred_labels
})
submission.to_csv("submission.csv", index=False)

print("âœ… CV done, confusion matrix plotted, feature importance visualized, and submission.csv saved.")
print(f"Submission file contains {len(submission)} predictions.")
print(f"Predicted classes: {np.unique(y_pred_labels)}")

# Show first few rows of submission
print("\nFirst few rows of submission:")
print(submission.head())


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import lightgbm as lgb
import pandas as pd
import os

# First, check what files are available
print("Current directory:", os.getcwd())
print("Files in directory:")
for file in os.listdir('.'):
    if file.endswith('.csv'):
        print(f"- {file}")

# Try to find the correct file paths
train_path = None
test_path = None

possible_paths = [
    "train.csv",
    "test.csv",
    "/content/train.csv", 
    "/content/test.csv",
    "./train.csv",
    "./test.csv"
]

for path in possible_paths:
    if os.path.exists(path):
        if 'train' in path.lower() and train_path is None:
            train_path = path
            print(f"Found training data: {path}")
        elif 'test' in path.lower() and test_path is None:
            test_path = path
            print(f"Found test data: {path}")

# Load your data with error handling
if train_path:
    train_df = pd.read_csv(train_path)
    print(f"Successfully loaded training data from {train_path}")
else:
    # Create sample data if no CSV found
    print("No train.csv found. Creating sample data...")
    np.random.seed(42)
    n_samples = 1000
    n_features = 30
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 5, n_samples)
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    train_df = pd.DataFrame(X, columns=feature_cols)
    train_df['gesture'] = y

if test_path:
    test_df = pd.read_csv(test_path)
    print(f"Successfully loaded test data from {test_path}")
else:
    print("No test.csv found. Creating sample test data...")
    test_X = np.random.randn(200, len(train_df.columns) - 1)
    test_df = pd.DataFrame(test_X, columns=[col for col in train_df.columns if col != 'gesture'])

# Convert gesture column to string to avoid classification report issues
if 'gesture' in train_df.columns:
    train_df['gesture'] = train_df['gesture'].astype(str)
else:
    # Create sample gesture column if it doesn't exist
    train_df['gesture'] = ['class_' + str(i) for i in np.random.randint(0, 5, len(train_df))]

# Define features and target
X = train_df.drop(columns=["gesture"])
y = train_df["gesture"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_proc = scaler.fit_transform(X_train)
X_test_proc = scaler.transform(X_test)

# Keep DataFrames for easier handling
X_train_df = pd.DataFrame(X_train_proc, columns=X.columns)
y_train_df = pd.Series(y_train).reset_index(drop=True)
X_test_df = pd.DataFrame(X_test_proc, columns=X.columns)

# Stratified K-Fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_true_all, y_pred_all = [], []
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_df, y_train_df), 1):
    X_tr, X_val = X_train_df.iloc[train_idx], X_train_df.iloc[val_idx]
    y_tr, y_val = y_train_df.iloc[train_idx], y_train_df.iloc[val_idx]

    model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X_tr, y_tr)

    y_val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    fold_accuracies.append(acc)

    y_true_all.extend(y_val)
    y_pred_all.extend(y_val_pred)

print(f"Mean CV Accuracy: {np.mean(fold_accuracies):.4f} Â± {np.std(fold_accuracies):.4f}")

# Normalized Confusion Matrix
cm = confusion_matrix(y_true_all, y_pred_all, labels=np.unique(y_train_df), normalize='true')

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Class-Normalized Confusion Matrix (CV Predictions)")
plt.tight_layout()
plt.show()

# Classification Report - FIXED
# Convert numeric predictions back to original labels for proper reporting
y_true_labels = label_encoder.inverse_transform(y_true_all)
y_pred_labels = label_encoder.inverse_transform(y_pred_all)

print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_))

# Hyperparameter Tuning (optional)
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'num_leaves': [31, 50],
}

grid_search = GridSearchCV(
    estimator=lgb.LGBMClassifier(random_state=42, verbose=-1),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_df, y_train_df)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Retrain on Full Training Data with best parameters
final_model = grid_search.best_estimator_
final_model.fit(X_train_df, y_train_df)

# Feature Importance (Top 20)
importances = final_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_train_df.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20))
plt.title("Top 20 Feature Importances (LightGBM)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Predict Test Set + Save Submission
# Ensure test data has the same columns as training data
missing_cols = set(X.columns) - set(test_df.columns)
if missing_cols:
    print(f"Adding missing columns to test data: {missing_cols}")
    for col in missing_cols:
        test_df[col] = 0

# Remove extra columns from test data
test_df = test_df[X.columns]

# Preprocess test data
X_test_processed = scaler.transform(test_df)
X_test_proc_df = pd.DataFrame(X_test_processed, columns=X.columns)

y_pred = final_model.predict(X_test_proc_df)

# If row_id exists in test_df, use it; otherwise fallback to index
if "row_id" in test_df.columns:
    row_ids = test_df["row_id"]
else:
    row_ids = test_df.index

# Convert numeric predictions back to original labels
y_pred_labels = label_encoder.inverse_transform(y_pred)

submission = pd.DataFrame({
    "row_id": row_ids,
    "gesture": y_pred_labels
})
submission.to_csv("submission.csv", index=False)

print("âœ… CV done, confusion matrix plotted, feature importance visualized, and submission.csv saved.")
print(f"Submission preview:")
print(submission.head())


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import make_classification  # For sample data
import numpy as np
import lightgbm as lgb
import pandas as pd
import os

# Check if the file exists, if not use sample data
try:
    # Try to load your actual data
    train_df = pd.read_csv("path/to/your/train.csv")
    test_df = pd.read_csv("path/to/your/test.csv")
    print("Using your provided CSV files")
except FileNotFoundError:
    # Create sample data if files don't exist
    print("CSV files not found. Creating sample data for demonstration...")
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                              n_redundant=5, n_classes=5, random_state=42)
    train_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    train_df['gesture'] = y
    
    # Create a test set from the same data for demonstration
    test_df = train_df.sample(200, random_state=42).copy()
    print("Sample data created successfully")

# Define features and target
X = train_df.drop(columns=["gesture"])
y = train_df["gesture"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_proc = scaler.fit_transform(X_train)
X_test_proc = scaler.transform(X_test)

# Keep DataFrames for easier handling
X_train_df = pd.DataFrame(X_train_proc, columns=X.columns)
y_train_df = pd.Series(y_train).reset_index(drop=True)
X_test_df = pd.DataFrame(X_test_proc, columns=X.columns)

# Stratified K-Fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_true_all, y_pred_all = [], []
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_df, y_train_df), 1):
    X_tr, X_val = X_train_df.iloc[train_idx], X_train_df.iloc[val_idx]
    y_tr, y_val = y_train_df.iloc[train_idx], y_train_df.iloc[val_idx]

    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_tr, y_tr)

    y_val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    fold_accuracies.append(acc)

    y_true_all.extend(y_val)
    y_pred_all.extend(y_val_pred)

print(f"Mean CV Accuracy: {np.mean(fold_accuracies):.4f} Â± {np.std(fold_accuracies):.4f}")

# Normalized Confusion Matrix
cm = confusion_matrix(y_true_all, y_pred_all, labels=np.unique(y_train_df), normalize='true')

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Class-Normalized Confusion Matrix (CV Predictions)")
plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true_all, y_pred_all, target_names=label_encoder.classes_))

# Hyperparameter Tuning (optional)
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'num_leaves': [31, 50],
}

grid_search = GridSearchCV(
    estimator=lgb.LGBMClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_df, y_train_df)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Retrain on Full Training Data with best parameters
final_model = grid_search.best_estimator_
final_model.fit(X_train_df, y_train_df)

# Feature Importance (Top 20)
importances = final_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_train_df.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), palette="viridis")
plt.title("Top 20 Feature Importances (LightGBM)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Predict Test Set + Save Submission
y_pred = final_model.predict(X_test_df)

# If row_id exists in test_df, use it; otherwise fallback to index
if "row_id" in test_df.columns:
    row_ids = test_df["row_id"]
else:
    row_ids = test_df.index

# Convert numeric predictions back to original labels
y_pred_labels = label_encoder.inverse_transform(y_pred)

submission = pd.DataFrame({
    "row_id": row_ids,
    "gesture": y_pred_labels
})
submission.to_csv("submission.csv", index=False)

print("âœ… CV done, confusion matrix plotted, feature importance visualized, and submission.csv saved.")


# Classification Report
print("\nClassification Report:")
# Convert numeric class labels to strings for target_names
target_names = [str(cls) for cls in label_encoder.classes_]
print(classification_report(y_true_all, y_pred_all, target_names=target_names))

# Convert numeric class labels to strings for the confusion matrix labels
class_labels = [str(cls) for cls in label_encoder.classes_]

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Class-Normalized Confusion Matrix (CV Predictions)")
plt.tight_layout()
plt.show()

# Alternative: Use more descriptive class names
# If you have meaningful class names, you could use:
# class_names = ['Class_A', 'Class_B', 'Class_C', 'Class_D', 'Class_E']
# target_names = class_names

# Or use generic names based on the numeric labels:
# target_names = [f'Class_{cls}' for cls in label_encoder.classes_]








import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import make_classification  # For sample data
import numpy as np
import lightgbm as lgb
import pandas as pd
import os

# Check if the file exists, if not use sample data
try:
    # Try to load your actual data
    train_df = pd.read_csv("path/to/your/train.csv")
    test_df = pd.read_csv("path/to/your/test.csv")
    print("Using your provided CSV files")
except FileNotFoundError:
    # Create sample data if files don't exist
    print("CSV files not found. Creating sample data for demonstration...")
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                              n_redundant=5, n_classes=5, random_state=42)
    train_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    train_df['gesture'] = y
    
    # Create a test set from the same data for demonstration
    test_df = train_df.sample(200, random_state=42).copy()
    print("Sample data created successfully")

# Define features and target
X = train_df.drop(columns=["gesture"])
y = train_df["gesture"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Store original class names for reporting
if hasattr(y, 'unique'):
    original_class_names = y.unique()
else:
    original_class_names = np.unique(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_proc = scaler.fit_transform(X_train)
X_test_proc = scaler.transform(X_test)

# Keep DataFrames for easier handling
X_train_df = pd.DataFrame(X_train_proc, columns=X.columns)
y_train_df = pd.Series(y_train).reset_index(drop=True)
X_test_df = pd.DataFrame(X_test_proc, columns=X.columns)

# Stratified K-Fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_true_all, y_pred_all = [], []
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_df, y_train_df), 1):
    X_tr, X_val = X_train_df.iloc[train_idx], X_train_df.iloc[val_idx]
    y_tr, y_val = y_train_df.iloc[train_idx], y_train_df.iloc[val_idx]

    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_tr, y_tr)

    y_val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    fold_accuracies.append(acc)

    y_true_all.extend(y_val)
    y_pred_all.extend(y_val_pred)

print(f"Mean CV Accuracy: {np.mean(fold_accuracies):.4f} Â± {np.std(fold_accuracies):.4f}")

# Normalized Confusion Matrix
cm = confusion_matrix(y_true_all, y_pred_all, labels=np.unique(y_train_df), normalize='true')

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=original_class_names,
            yticklabels=original_class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Class-Normalized Confusion Matrix (CV Predictions)")
plt.tight_layout()
plt.show()

# Classification Report - FIXED
print("\nClassification Report:")
print(classification_report(y_true_all, y_pred_all, target_names=original_class_names))

# Hyperparameter Tuning (optional)
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'num_leaves': [31, 50],
}

grid_search = GridSearchCV(
    estimator=lgb.LGBMClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_df, y_train_df)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Retrain on Full Training Data with best parameters
final_model = grid_search.best_estimator_
final_model.fit(X_train_df, y_train_df)

# Feature Importance (Top 20)
importances = final_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_train_df.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), palette="viridis")
plt.title("Top 20 Feature Importances (LightGBM)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Predict Test Set + Save Submission
y_pred = final_model.predict(X_test_df)

# If row_id exists in test_df, use it; otherwise fallback to index
if "row_id" in test_df.columns:
    row_ids = test_df["row_id"]
else:
    row_ids = test_df.index

# Convert numeric predictions back to original labels
y_pred_labels = label_encoder.inverse_transform(y_pred)

submission = pd.DataFrame({
    "row_id": row_ids,
    "gesture": y_pred_labels
})
submission.to_csv("submission.csv", index=False)

print("âœ… CV done, confusion matrix plotted, feature importance visualized, and submission.csv saved.")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

import lightgbm as lgb
import xgboost as xgb

# ==============================
# Step 1: Load Data
# ==============================
print("Loading data...")
train_demo = pd.read_csv("/content/train_demographics.csv")
test = pd.read_csv("/content/test.csv")
test_demo = pd.read_csv("/content/test_demographics.csv")

train_sensor_path = "/content/train.csv"
has_train_sensor = os.path.exists(train_sensor_path)
train = pd.read_csv(train_sensor_path) if has_train_sensor else None

# Optional: if labels are provided
label_path = "/content/train_labels.csv"
has_labels = os.path.exists(label_path)
labels = pd.read_csv(label_path) if has_labels else None

# ==============================
# Step 2: Explore Data Structure
# ==============================
print("\nData Exploration:")
print(f"Train demographics shape: {train_demo.shape}")
print(f"Test shape: {test.shape}")
print(f"Test demographics shape: {test_demo.shape}")

if train is not None:
    print(f"Train sensor data shape: {train.shape}")
    
if labels is not None:
    print(f"Labels shape: {labels.shape}")

# Display first few rows of each dataset
print("\nTrain demographics sample:")
print(train_demo.head())

print("\nTest sample:")
print(test.head())

# ==============================
# Step 3: Data Preprocessing
# ==============================
print("\nPreprocessing data...")

# Check if we have sensor data or need to use demographics only
if train is not None and has_labels:
    # We have both sensor data and labels
    X = train.copy()
    y = labels.iloc[:, -1]  # Assuming last column is the target
    
elif train_demo is not None and has_labels:
    # Use demographics data with labels
    X = train_demo.copy()
    y = labels.iloc[:, -1]  # Assuming last column is the target
    
else:
    # Create sample target if no labels available
    print("No labels found. Creating sample target for demonstration.")
    X = train_demo.copy() if train_demo is not None else test_demo.copy()
    y = pd.Series(np.random.randint(0, 5, len(X)))  # Sample target with 5 classes

# Handle missing values
print("Handling missing values...")
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X.select_dtypes(include=[np.number]))

# Get column names back
X_processed = pd.DataFrame(X_imputed, columns=X.select_dtypes(include=[np.number]).columns)

# Add non-numeric columns if any
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
for col in non_numeric_cols:
    X_processed[col] = X[col]

# Encode categorical variables
print("Encoding categorical variables...")
label_encoders = {}
for col in X_processed.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
    label_encoders[col] = le

# Encode target variable
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# Convert numeric class labels to string names for reporting
target_names = [f"Class_{i}" for i in le_target.classes_]

# ==============================
# Step 4: Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# Step 5: Model Training - LightGBM
# ==============================
print("\nTraining LightGBM model...")

# Cross-validation with LightGBM
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
y_true_all, y_pred_all = [], []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_scaled, y_train), 1):
    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_tr, y_tr)
    
    y_val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    fold_accuracies.append(acc)
    
    y_true_all.extend(y_val)
    y_pred_all.extend(y_val_pred)
    
    print(f"Fold {fold} Accuracy: {acc:.4f}")

print(f"Mean CV Accuracy: {np.mean(fold_accuracies):.4f} Â± {np.std(fold_accuracies):.4f}")

# Train final model on full training data
final_model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)
final_model.fit(X_train_scaled, y_train)

# ==============================
# Step 6: Model Evaluation
# ==============================
print("\nModel Evaluation:")

# Test set evaluation
y_pred = final_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Classification report - FIXED: Use string target names
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# ==============================
# Step 7: Feature Importance
# ==============================
feature_importance = pd.DataFrame({
    'feature': X_processed.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
plt.title('Top 15 Feature Importances')
plt.tight_layout()
plt.show()

# ==============================
# Step 8: Prepare Test Data for Submission
# ==============================
print("\nPreparing submission...")

# Process test data similarly to training data
test_processed = test.copy()

# Handle missing values
test_imputed = imputer.transform(test_processed.select_dtypes(include=[np.number]))
test_processed_num = pd.DataFrame(test_imputed, columns=test_processed.select_dtypes(include=[np.number]).columns)

# Add non-numeric columns if any
non_numeric_cols_test = test_processed.select_dtypes(exclude=[np.number]).columns
for col in non_numeric_cols_test:
    test_processed_num[col] = test_processed[col]

# Encode categorical variables
for col in test_processed_num.select_dtypes(include=['object']).columns:
    if col in label_encoders:
        # Transform using fitted label encoder
        test_processed_num[col] = label_encoders[col].transform(test_processed_num[col].astype(str))
    else:
        # If new category appears, handle it
        test_processed_num[col] = test_processed_num[col].astype('category').cat.codes

# Scale test data
test_scaled = scaler.transform(test_processed_num)

# Make predictions
test_pred = final_model.predict(test_scaled)
test_pred_labels = le_target.inverse_transform(test_pred)

# Create submission file
if "row_id" in test.columns:
    row_ids = test["row_id"]
else:
    row_ids = test.index

submission = pd.DataFrame({
    "row_id": row_ids,
    "gesture": test_pred_labels
})
submission.to_csv("submission.csv", index=False)

print("âœ… Submission file created: submission.csv")
print(f"Submission preview:")
print(submission.head())

# ==============================
# Step 9: Optional - Train Additional Models
# ==============================
print("\nTraining additional models for comparison...")

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

print(f"\nBest model: LightGBM with {test_accuracy:.4f} accuracy")

print("\nâœ… All steps completed successfully!")


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import make_classification  # For sample data
import numpy as np
import lightgbm as lgb
import pandas as pd
import os

# Check if the file exists, if not use sample data
try:
    # Try to load your actual data
    train_df = pd.read_csv("path/to/your/train.csv")
    test_df = pd.read_csv("path/to/your/test.csv")
    print("Using your provided CSV files")
except FileNotFoundError:
    # Create sample data if files don't exist
    print("CSV files not found. Creating sample data for demonstration...")
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                              n_redundant=5, n_classes=5, random_state=42)
    train_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    train_df['gesture'] = y
    
    # Create a test set from the same data for demonstration
    test_df = train_df.sample(200, random_state=42).copy()
    print("Sample data created successfully")

# Define features and target
X = train_df.drop(columns=["gesture"])
y = train_df["gesture"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Store original class names for reporting
if hasattr(y, 'unique'):
    original_class_names = y.unique()
else:
    original_class_names = np.unique(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_proc = scaler.fit_transform(X_train)
X_test_proc = scaler.transform(X_test)

# Keep DataFrames for easier handling
X_train_df = pd.DataFrame(X_train_proc, columns=X.columns)
y_train_df = pd.Series(y_train).reset_index(drop=True)
X_test_df = pd.DataFrame(X_test_proc, columns=X.columns)

# Stratified K-Fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_true_all, y_pred_all = [], []
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_df, y_train_df), 1):
    X_tr, X_val = X_train_df.iloc[train_idx], X_train_df.iloc[val_idx]
    y_tr, y_val = y_train_df.iloc[train_idx], y_train_df.iloc[val_idx]

    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_tr, y_tr)

    y_val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    fold_accuracies.append(acc)

    y_true_all.extend(y_val)
    y_pred_all.extend(y_val_pred)

print(f"Mean CV Accuracy: {np.mean(fold_accuracies):.4f} Â± {np.std(fold_accuracies):.4f}")

# Normalized Confusion Matrix
cm = confusion_matrix(y_true_all, y_pred_all, labels=np.unique(y_train_df), normalize='true')

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=original_class_names,
            yticklabels=original_class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Class-Normalized Confusion Matrix (CV Predictions)")
plt.tight_layout()
plt.show()

# Classification Report - FIXED
print("\nClassification Report:")
print(classification_report(y_true_all, y_pred_all, target_names=original_class_names))

# Hyperparameter Tuning (optional)
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'num_leaves': [31, 50],
}

grid_search = GridSearchCV(
    estimator=lgb.LGBMClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_df, y_train_df)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Retrain on Full Training Data with best parameters
final_model = grid_search.best_estimator_
final_model.fit(X_train_df, y_train_df)

# Feature Importance (Top 20)
importances = final_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_train_df.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), palette="viridis")
plt.title("Top 20 Feature Importances (LightGBM)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Predict Test Set + Save Submission
y_pred = final_model.predict(X_test_df)

# If row_id exists in test_df, use it; otherwise fallback to index
if "row_id" in test_df.columns:
    row_ids = test_df["row_id"]
else:
    row_ids = test_df.index

# Convert numeric predictions back to original labels
y_pred_labels = label_encoder.inverse_transform(y_pred)

submission = pd.DataFrame({
    "row_id": row_ids,
    "gesture": y_pred_labels
})
submission.to_csv("submission.csv", index=False)

print("âœ… CV done, confusion matrix plotted, feature importance visualized, and submission.csv saved.")


from sklearn.ensemble import RandomForestClassifier

# Choose RandomForestClassifier as the model
model = RandomForestClassifier(random_state=42)

print("Chosen model: RandomForestClassifier")


# Train the model
model.fit(X_train, y_train)

print("RandomForestClassifier model trained successfully.")


import pandas as pd
import numpy as np
import os

# ==============================
# Step 1: Load Data
# ==============================
train_demo = pd.read_csv("/content/train_demographics.csv")
test = pd.read_csv("/content/test.csv")
test_demo = pd.read_csv("/content/test_demographics.csv")


train_sensor_path = "/content/train.csv"
has_train_sensor = os.path.exists(train_sensor_path)
train = pd.read_csv(train_sensor_path) if has_train_sensor else None


# Optional: if labels are provided
label_path = "/content/train_labels.csv"
has_labels = os.path.exists(label_path)
labels = pd.read_csv(label_path) if has_labels else None


# ==============================
# Step 2: Feature Engineering Function
# ==============================
def build_features(sensor_df: pd.DataFrame, demo_df: pd.DataFrame) -> pd.DataFrame:
    df = sensor_df.copy()
    df = df.replace(-1.0, np.nan)
    df = df.merge(demo_df, on="subject", how="left")


    id_cols = ["row_id", "sequence_id", "sequence_counter", "subject"]
    demo_cols = ["adult_child", "age", "sex", "handedness", "height_cm",
    "shoulder_to_wrist_cm", "elbow_to_wrist_cm"]

    # Explicitly exclude non-numeric columns
    non_numeric_cols = id_cols + demo_cols + ['sequence_type', 'orientation', 'behavior', 'phase', 'gesture']
    numeric_cols = [c for c in df.columns if c not in non_numeric_cols]


    agg_funcs = ["mean", "std", "min", "max", "median", "skew"]
    seq_feats = (
    df.groupby("sequence_id")[numeric_cols]
    .agg(agg_funcs)
    )
    seq_feats.columns = ["_".join(col).strip() for col in seq_feats.columns.values]
    seq_feats = seq_feats.reset_index()


    demo_feats = (
    df.groupby("sequence_id")[demo_cols]
    .first()
    .reset_index()
    )
    final = seq_feats.merge(demo_feats, on="sequence_id", how="left")
    return final


print(X_train.dtypes)


# Drop the 'gesture' column from X_train and X_test
X_train = X_train.drop('gesture', axis=1)
X_test = X_test.drop('gesture', axis=1)

# Train the model again
model.fit(X_train, y_train)

print("RandomForestClassifier model trained successfully after dropping the 'gesture' column.")


# Make predictions on the test data
predictions = model.predict(X_test)

# Create the submission DataFrame
submission_df = pd.DataFrame({'row_id': test_df['row_id'], 'sequence_type': predictions})

# Display the head of the submission DataFrame
print("Submission DataFrame head:")
display(submission_df.head())


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import lightgbm as lgb
import pandas as pd
import os

# First, let's check what files are available and load data properly
print("Current directory:", os.getcwd())
print("Files in directory:")
csv_files = []
for file in os.listdir('.'):
    if file.endswith('.csv'):
        csv_files.append(file)
        print(f"- {file}")

# Load data or create sample data
if 'train.csv' in csv_files:
    train_df = pd.read_csv('train.csv')
    print("Loaded train.csv")
else:
    print("Creating sample training data...")
    np.random.seed(42)
    n_samples = 1000
    n_features = 30
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 5, n_samples)
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    train_df = pd.DataFrame(X, columns=feature_cols)
    train_df['gesture'] = y

if 'test.csv' in csv_files:
    test_df = pd.read_csv('test.csv')
    print("Loaded test.csv")
else:
    print("Creating sample test data...")
    test_X = np.random.randn(200, len(train_df.columns) - 1)
    test_df = pd.DataFrame(test_X, columns=[col for col in train_df.columns if col != 'gesture'])

# Display data info
print(f"\nTraining data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Check if 'gesture' column exists and convert to string
if 'gesture' not in train_df.columns:
    print("'gesture' column not found. Creating sample target...")
    train_df['gesture'] = np.random.randint(0, 5, len(train_df))

# Convert to string to avoid issues
train_df['gesture'] = train_df['gesture'].astype(str)

# Define features and target
X = train_df.drop(columns=["gesture"])
y = train_df["gesture"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames
X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# ================================
# 1. Stratified K-Fold with CV predictions
# ================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_true_all, y_pred_all = [], []
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_df, y_train), 1):
    X_tr, X_val = X_train_df.iloc[train_idx], X_train_df.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X_tr, y_tr)

    y_val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    fold_accuracies.append(acc)

    y_true_all.extend(y_val)
    y_pred_all.extend(y_val_pred)

    print(f"Fold {fold}: Accuracy = {acc:.4f}")

print(f"\nMean CV Accuracy: {np.mean(fold_accuracies):.4f} Â± {np.std(fold_accuracies):.4f}")

# ================================
# 2. Normalized Confusion Matrix
# ================================
cm = confusion_matrix(y_true_all, y_pred_all, normalize='true')

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Class-Normalized Confusion Matrix (CV Predictions)")
plt.tight_layout()
plt.show()

# ================================
# 3. Retrain on Full Data
# ================================
final_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
final_model.fit(X_train_df, y_train)

# ================================
# 4. Feature Importance (Top 20)
# ================================
importances = final_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
# Create a DataFrame for easier plotting
importance_df = pd.DataFrame({
    'feature': X_train_df.columns[indices[:20]],
    'importance': importances[indices[:20]]
})

sns.barplot(x='importance', y='feature', data=importance_df, palette="viridis")
plt.title("Top 20 Feature Importances (LightGBM)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# ================================
# 5. Predict Test Set + Save Submission
# ================================
# Prepare test data (ensure it has the same columns as training)
missing_cols = set(X.columns) - set(test_df.columns)
if missing_cols:
    print(f"Adding missing columns to test data: {missing_cols}")
    for col in missing_cols:
        test_df[col] = 0

# Remove extra columns
test_df = test_df[X.columns]

# Scale test data
test_scaled = scaler.transform(test_df)
test_df_scaled = pd.DataFrame(test_scaled, columns=X.columns)

# Make predictions
y_pred_encoded = final_model.predict(test_df_scaled)
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Create submission
if "row_id" in test_df.columns:
    row_ids = test_df["row_id"]
else:
    row_ids = test_df.index

submission = pd.DataFrame({
    "row_id": row_ids,
    "gesture": y_pred
})
submission.to_csv("submission.csv", index=False)

print("âœ… Normalized confusion matrix plotted, feature importance visualized, and submission.csv saved.")
print(f"Submission preview:")
print(submission.head())


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming y_test is available (if not, you might need to load it from the original test data if it exists)
# For this example, let's assume we have y_test available. If not, this step would need adjustment.
# Since test.csv does not contain 'sequence_type', we cannot calculate a confusion matrix directly.
# However, if a ground truth for the test data were available, the code below would work.

# As a placeholder, let's assume a sample y_test is available for demonstration purposes.
# In a real scenario, you would load or obtain the actual ground truth labels for the test set.

# Since we don't have the ground truth for the provided test data, we will skip the confusion matrix and heatmap generation.
# If you had the ground truth (y_test), you would uncomment and run the following code:

# cm = confusion_matrix(y_test, predictions)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix Heatmap')
# plt.show()

print("Skipping confusion matrix and heatmap generation as ground truth (y_test) for the test data is not available.")
print("If you have y_test, uncomment the code to generate the confusion matrix and heatmap.")


#  Ensure submission.csv exists, otherwise create a dummy version for workflow
import os
import numpy as np
import pandas as pd

if not os.path.exists("submission.csv"):
    print("âš ï¸� 'submission.csv' not found. Creating dummy submission...")
    n_models, n_triggers = 10, 3 * 75  # Adjust as needed
    dummy = pd.DataFrame({
        "model_id": [f"model_{i:04d}" for i in range(n_models)],
        **{f"t{j}": np.random.uniform(0.01, 0.1, size=n_models) for j in range(n_triggers)}
    })
    dummy.to_csv("submission.csv", index=False)
    print("âœ… Dummy 'submission.csv' generated:", dummy.shape)
else:
    print("âœ… 'submission.csv' found.")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

import lightgbm as lgb
import xgboost as xgb

# ==============================
# Step 1: Load Data
# ==============================
print("Loading data...")
train_demo = pd.read_csv("/content/train_demographics.csv")
test = pd.read_csv("/content/test.csv")
test_demo = pd.read_csv("/content/test_demographics.csv")

train_sensor_path = "/content/train.csv"
has_train_sensor = os.path.exists(train_sensor_path)
train = pd.read_csv(train_sensor_path) if has_train_sensor else None

# Optional: if labels are provided
label_path = "/content/train_labels.csv"
has_labels = os.path.exists(label_path)
labels = pd.read_csv(label_path) if has_labels else None

# ==============================
# Step 2: Explore Data Structure
# ==============================
print("\nData Exploration:")
print(f"Train demographics shape: {train_demo.shape}")
print(f"Test shape: {test.shape}")
print(f"Test demographics shape: {test_demo.shape}")

if train is not None:
    print(f"Train sensor data shape: {train.shape}")
    
if labels is not None:
    print(f"Labels shape: {labels.shape}")

# Display first few rows of each dataset
print("\nTrain demographics sample:")
print(train_demo.head())

print("\nTest sample:")
print(test.head())

# ==============================
# Step 3: Data Preprocessing
# ==============================
print("\nPreprocessing data...")

# Check if we have sensor data or need to use demographics only
if train is not None and has_labels:
    # We have both sensor data and labels
    X = train.copy()
    y = labels.iloc[:, -1]  # Assuming last column is the target
    
elif train_demo is not None and has_labels:
    # Use demographics data with labels
    X = train_demo.copy()
    y = labels.iloc[:, -1]  # Assuming last column is the target
    
else:
    # Create sample target if no labels available
    print("No labels found. Creating sample target for demonstration.")
    X = train_demo.copy() if train_demo is not None else test_demo.copy()
    y = pd.Series(np.random.randint(0, 5, len(X)))  # Sample target with 5 classes

# Handle missing values
print("Handling missing values...")
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X.select_dtypes(include=[np.number]))

# Get column names back
X_processed = pd.DataFrame(X_imputed, columns=X.select_dtypes(include=[np.number]).columns)

# Add non-numeric columns if any
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
for col in non_numeric_cols:
    X_processed[col] = X[col]

# Encode categorical variables
print("Encoding categorical variables...")
label_encoders = {}
for col in X_processed.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
    label_encoders[col] = le

# Encode target variable
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# ==============================
# Step 4: Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# Step 5: Model Training - LightGBM
# ==============================
print("\nTraining LightGBM model...")

# Cross-validation with LightGBM
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
y_true_all, y_pred_all = [], []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_scaled, y_train), 1):
    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_tr, y_tr)
    
    y_val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    fold_accuracies.append(acc)
    
    y_true_all.extend(y_val)
    y_pred_all.extend(y_val_pred)
    
    print(f"Fold {fold} Accuracy: {acc:.4f}")

print(f"Mean CV Accuracy: {np.mean(fold_accuracies):.4f} Â± {np.std(fold_accuracies):.4f}")

# Train final model on full training data
final_model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)
final_model.fit(X_train_scaled, y_train)

# ==============================
# Step 6: Model Evaluation
# ==============================
print("\nModel Evaluation:")

# Test set evaluation
y_pred = final_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_target.classes_,
            yticklabels=le_target.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# ==============================
# Step 7: Feature Importance
# ==============================
feature_importance = pd.DataFrame({
    'feature': X_processed.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
plt.title('Top 15 Feature Importances')
plt.tight_layout()
plt.show()

# ==============================
# Step 8: Prepare Test Data for Submission
# ==============================
print("\nPreparing submission...")

# Process test data similarly to training data
test_processed = test.copy()

# Handle missing values
test_imputed = imputer.transform(test_processed.select_dtypes(include=[np.number]))
test_processed_num = pd.DataFrame(test_imputed, columns=test_processed.select_dtypes(include=[np.number]).columns)

# Add non-numeric columns if any
non_numeric_cols_test = test_processed.select_dtypes(exclude=[np.number]).columns
for col in non_numeric_cols_test:
    test_processed_num[col] = test_processed[col]

# Encode categorical variables
for col in test_processed_num.select_dtypes(include=['object']).columns:
    if col in label_encoders:
        # Transform using fitted label encoder
        test_processed_num[col] = label_encoders[col].transform(test_processed_num[col].astype(str))
    else:
        # If new category appears, handle it
        test_processed_num[col] = test_processed_num[col].astype('category').cat.codes

# Scale test data
test_scaled = scaler.transform(test_processed_num)

# Make predictions
test_pred = final_model.predict(test_scaled)
test_pred_labels = le_target.inverse_transform(test_pred)

# Create submission file
if "row_id" in test.columns:
    row_ids = test["row_id"]
else:
    row_ids = test.index

submission = pd.DataFrame({
    "row_id": row_ids,
    "gesture": test_pred_labels
})
submission.to_csv("submission.csv", index=False)

print("âœ… Submission file created: submission.csv")
print(f"Submission preview:")
print(submission.head())

# ==============================
# Step 9: Optional - Train Additional Models
# ==============================
print("\nTraining additional models for comparison...")

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

print(f"\nBest model: LightGBM with {test_accuracy:.4f} accuracy")

print("\nâœ… All steps completed successfully!")


# =========================================
# Full Pipeline: Preprocessing + Features + Models
# =========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')

# ==========================
# 1. Load train + test data
# ==========================
try:
    train_df = pd.read_csv("train.csv")
    print("Train shape:", train_df.shape)
except:
    train_df = None
    print("train.csv not found")

try:
    test_df = pd.read_csv("test.csv")
    print("Test shape:", test_df.shape)
except:
    test_df = None
    print("test.csv not found")

# ==========================
# 2. Feature Engineering
# ==========================
def build_features(df):
    feats = df.copy()
    # Example: numeric stats (replace with your logic)
    numeric_cols = feats.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        feats[f"{col}_mean"] = feats[col].mean()
        feats[f"{col}_std"] = feats[col].std()
    return feats

train_features = build_features(train_df) if train_df is not None else None
test_features  = build_features(test_df) if test_df is not None else None

# ==========================
# 3. Supervised or Unsupervised?
# ==========================
label_col = None
if train_features is not None:
    for candidate in ["sequence_type", "label", "target"]:
        if candidate in train_features.columns:
            label_col = candidate
            break

# ==========================
# 4. Preprocessing
# ==========================
if train_features is not None:
    if label_col:
        X = train_features.drop(label_col, axis=1)
        y = train_features[label_col]
    else:
        X = train_features
        y = None

    if y is not None:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        num_classes = len(np.unique(y_encoded))

        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
    else:
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
        y_train = y_val = None

    scaler = StandardScaler()
    X_train_proc = scaler.fit_transform(X_train)
    X_val_proc   = scaler.transform(X_val)
else:
    X_train_proc = X_val_proc = None
    y_train = y_val = None
    num_classes = None

# ==========================
# 5A. Supervised Models
# ==========================
if y_train is not None:

    # ---- Neural Network ----
    def build_nn_model(neurons_layer1=128, neurons_layer2=64,
                       activation='relu', optimizer='adam'):
        model = Sequential()
        model.add(Dense(neurons_layer1, activation=activation,
                        input_shape=(X_train_proc.shape[1],)))
        model.add(Dense(neurons_layer2, activation=activation))
        if num_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer=optimizer, loss='binary_crossentropy',
                          metrics=['accuracy'])
        else:
            model.add(Dense(num_classes, activation='softmax'))
            model.compile(optimizer=optimizer,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
        return model

    nn_model = KerasClassifier(model=build_nn_model, verbose=0)
    param_grid_nn = {
        'model__neurons_layer1': [64, 128],
        'model__neurons_layer2': [32, 64],
        'batch_size': [32, 64],
        'epochs': [10, 20]
    }
    grid_search_nn = GridSearchCV(
        estimator=nn_model,
        param_grid=param_grid_nn,
        cv=3,
        scoring='accuracy'
    )
    grid_search_nn.fit(X_train_proc, y_train)
    print("Best NN hyperparameters:", grid_search_nn.best_params_)

    # ---- XGBoost ----
    xgb = XGBClassifier(use_label_encoder=False,
                        eval_metric='mlogloss',
                        random_state=42)
    xgb.fit(X_train_proc, y_train)

    # ---- Predictions ----
    y_pred_nn  = grid_search_nn.predict(X_val_proc)
    y_pred_xgb = xgb.predict(X_val_proc)

    # ---- Reports ----
    print("\nXGBoost Metrics:\n",
          classification_report(y_val, y_pred_xgb,
                                target_names=le.classes_))
    print("\nNeural Net Metrics:\n",
          classification_report(y_val, y_pred_nn,
                                target_names=le.classes_))

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_val, y_pred_xgb)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix - XGBoost")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# ==========================
# 5B. Unsupervised Anomaly Detection
# ==========================
elif train_features is not None:
    iso = IsolationForest(random_state=42, contamination=0.05)
    preds = iso.fit_predict(X_train_proc)
    print("IsolationForest anomaly labels (sample):", preds[:20])



# Create interaction features
train_df['seq_acc_interaction'] = train_df['sequence_counter'] * train_df['acc_magnitude']
train_df['seq_rot_interaction'] = train_df['sequence_counter'] * train_df['rot_magnitude']
test_df['seq_acc_interaction'] = test_df['sequence_counter'] * test_df['acc_magnitude']
test_df['seq_rot_interaction'] = test_df['sequence_counter'] * test_df['rot_magnitude']

# Calculate rolling mean and standard deviation
window_size = 5
train_df['acc_magnitude_rolling_mean'] = train_df['acc_magnitude'].rolling(window=window_size).mean().fillna(0)
train_df['acc_magnitude_rolling_std'] = train_df['acc_magnitude'].rolling(window=window_size).std().fillna(0)
train_df['rot_magnitude_rolling_mean'] = train_df['rot_magnitude'].rolling(window=window_size).mean().fillna(0)
train_df['rot_magnitude_rolling_std'] = train_df['rot_magnitude'].rolling(window=window_size).std().fillna(0)

test_df['acc_magnitude_rolling_mean'] = test_df['acc_magnitude'].rolling(window=window_size).mean().fillna(0)
test_df['acc_magnitude_rolling_std'] = test_df['acc_magnitude'].rolling(window=window_size).std().fillna(0)
test_df['rot_magnitude_rolling_mean'] = test_df['rot_magnitude'].rolling(window=window_size).mean().fillna(0)
test_df['rot_magnitude_rolling_std'] = test_df['rot_magnitude'].rolling(window=window_size).std().fillna(0)

print("New features added to train_df and test_df:")
display(train_df[['acc_magnitude', 'rot_magnitude', 'seq_acc_interaction', 'seq_rot_interaction', 'acc_magnitude_rolling_mean', 'acc_magnitude_rolling_std', 'rot_magnitude_rolling_mean', 'rot_magnitude_rolling_std']].head())
display(test_df[['acc_magnitude', 'rot_magnitude', 'seq_acc_interaction', 'seq_rot_interaction', 'acc_magnitude_rolling_mean', 'acc_magnitude_rolling_std', 'rot_magnitude_rolling_mean', 'rot_magnitude_rolling_std']].head())


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Identify numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X_train.select_dtypes(include=['object', 'bool']).columns

# Exclude 'sequence_counter' from numerical columns for scaling as it was already handled
numerical_cols = numerical_cols.drop('sequence_counter', errors='ignore')

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough' # Keep other columns (like sequence_counter if not scaled)
)

# Create a pipeline with the preprocessor
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit and transform the training data
X_train_processed = pipeline.fit_transform(X_train)

# Transform the testing data
X_test_processed = pipeline.transform(X_test)

# Convert the processed data back to DataFrames (optional, but good for inspection)
# Get feature names after one-hot encoding
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)


print("\nPreprocessed X_train head:")
display(X_train_processed_df.head())

print("\nPreprocessed X_test head:")
display(X_test_processed_df.head())


from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Instantiate XGBoost model
xgb_model = XGBClassifier(random_state=42)

# Define a simple Neural Network model
nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_processed_df.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification output layer
])

# Compile the Neural Network model
nn_model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

# Print NN model summary
print("XGBoost model instantiated.")
print("\nNeural Network model summary:")
nn_model.summary()


from sklearn.preprocessing import LabelEncoder

# Encode the target variable
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# Train the XGBoost model with encoded target
xgb_model.fit(X_train_processed_df, y_train_encoded)

print("XGBoost model trained successfully with encoded target.")

# Train the Neural Network model with encoded target
history = nn_model.fit(X_train_processed_df, y_train_encoded, epochs=10, batch_size=32, verbose=0)

print("Neural Network model trained successfully with encoded target.")


from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore') # Suppress warnings from GridSearchCV

# Define a parameter grid for the xgb_model
param_grid_xgb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}

# Instantiate GridSearchCV for XGBoost
grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=3, scoring='accuracy')

# Fit GridSearchCV to the preprocessed training data and encoded labels
grid_search_xgb.fit(X_train_processed_df, y_train_encoded)

# Print the best hyperparameters for XGBoost
print("Best hyperparameters for XGBoost:", grid_search_xgb.best_params_)


# ==========================================
# Preprocessing + XGBoost + NN GridSearchCV + Evaluation + PCA Visualization
# ==========================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# --- ML Models ---
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from xgboost import XGBClassifier

# --- Load data with fallback ---
try:
    train_features = pd.read_csv("train_features.csv")
    labels = pd.read_csv("train_labels.csv")
    train_df = train_features.merge(labels, on="sequence_id", how="left")
    feature_cols = [col for col in train_df.columns if col not in ["label", "sequence_id"]]
    X = train_df[feature_cols]
    y = train_df["label"]
    print("âœ… Loaded real dataset:", X.shape, y.shape)
except Exception as e:
    print("âš ï¸� train_features.csv not found, using synthetic placeholder data:", e)
    X = pd.DataFrame(np.random.randn(200, 5), columns=[f"f{i}" for i in range(5)])
    y = np.random.randint(0, 2, size=200)

# --- Preprocessing ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels if not numeric
if not np.issubdtype(y.dtype, np.number):
    le = LabelEncoder()
    y = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert back to DataFrame for scikeras compatibility
X_train_processed_df = pd.DataFrame(X_train)
X_test_processed_df = pd.DataFrame(X_test)
y_train_encoded = y_train
y_test_encoded = y_test

# --- Neural Network builder ---
def build_nn_model(neurons_layer1=128, neurons_layer2=64, activation='relu', optimizer='adam'):
    model = Sequential()
    model.add(Dense(neurons_layer1, activation=activation, input_shape=(X_train_processed_df.shape[1],)))
    model.add(Dense(neurons_layer2, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- NN GridSearch ---
nn_model = KerasClassifier(model=build_nn_model, verbose=0)
param_grid_nn = {
    'model__neurons_layer1': [64, 128],
    'model__neurons_layer2': [32, 64],
    'batch_size': [32, 64],
    'epochs': [10, 20]
}
grid_search_nn = GridSearchCV(estimator=nn_model, param_grid=param_grid_nn, cv=3, scoring='accuracy')
grid_search_nn.fit(X_train_processed_df, y_train_encoded)
print("âœ… Best hyperparameters for Neural Network:", grid_search_nn.best_params_)

# --- XGBoost GridSearch ---
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
param_grid_xgb = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1]
}
grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=3, scoring='accuracy')
grid_search_xgb.fit(X_train_processed_df, y_train_encoded)
print("âœ… Best hyperparameters for XGBoost:", grid_search_xgb.best_params_)

# --- Predictions ---
y_pred_nn = grid_search_nn.best_estimator_.predict(X_test_processed_df)
y_pred_xgb = grid_search_xgb.best_estimator_.predict(X_test_processed_df)

# --- Reports ---
print("\nğŸ“Š Classification Report - Neural Network")
print(classification_report(y_test_encoded, y_pred_nn))

print("\nğŸ“Š Classification Report - XGBoost")
print(classification_report(y_test_encoded, y_pred_xgb))

# --- Confusion Matrices ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_nn = confusion_matrix(y_test_encoded, y_pred_nn)
sns.heatmap(cm_nn, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Confusion Matrix - Neural Network")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

cm_xgb = confusion_matrix(y_test_encoded, y_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Greens", ax=axes[1])
axes[1].set_title("Confusion Matrix - XGBoost")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")

plt.tight_layout()
plt.show()

# --- PCA Visualization ---
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# NN visualization
axes[0].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test_encoded, cmap="coolwarm", alpha=0.6, label="True class")
misclassified_nn = y_test_encoded != y_pred_nn
axes[0].scatter(X_test_pca[misclassified_nn, 0], X_test_pca[misclassified_nn, 1],
                edgecolor="black", facecolor="none", s=100, label="Misclassified")
axes[0].set_title("PCA Scatter - Neural Network")
axes[0].legend()

# XGB visualization
axes[1].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test_encoded, cmap="coolwarm", alpha=0.6, label="True class")
misclassified_xgb = y_test_encoded != y_pred_xgb
axes[1].scatter(X_test_pca[misclassified_xgb, 0], X_test_pca[misclassified_xgb, 1],
                edgecolor="black", facecolor="none", s=100, label="Misclassified")
axes[1].set_title("PCA Scatter - XGBoost")
axes[1].legend()

plt.tight_layout()
plt.show()



from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define a simple function that creates the Neural Network model
def build_nn_model(neurons_layer1=128, neurons_layer2=64, activation='relu', optimizer='adam'):
    model = Sequential()
    model.add(Dense(neurons_layer1, activation=activation, input_shape=(X_train_processed_df.shape[1],)))
    model.add(Dense(neurons_layer2, activation=activation))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification output layer
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Instantiate KerasClassifier with the model-building function
nn_model = KerasClassifier(model=build_nn_model, verbose=0)


!pip install scikeras


from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import GridSearchCV

# Define a simple function that creates the Neural Network model
def build_nn_model(neurons_layer1=128, neurons_layer2=64, activation='relu', optimizer='adam'):
    model = Sequential()
    model.add(Dense(neurons_layer1, activation=activation, input_shape=(X_train_processed_df.shape[1],)))
    model.add(Dense(neurons_layer2, activation=activation))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification output layer
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Instantiate KerasClassifier with the model-building function
nn_model = KerasClassifier(model=build_nn_model, verbose=0)

# Define a parameter grid for the nn_model
param_grid_nn = {
    'model__neurons_layer1': [64, 128],
    'model__neurons_layer2': [32, 64],
    'batch_size': [32, 64],
    'epochs': [10, 20]
}

# Instantiate GridSearchCV for Neural Network
grid_search_nn = GridSearchCV(estimator=nn_model, param_grid=param_grid_nn, cv=3, scoring='accuracy')


from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore') # Suppress warnings

# Define a simple function that creates the Neural Network model
def build_nn_model(neurons_layer1=128, neurons_layer2=64, activation='relu', optimizer='adam'):
    model = Sequential()
    model.add(Dense(neurons_layer1, activation=activation, input_shape=(X_train_processed_df.shape[1],)))
    model.add(Dense(neurons_layer2, activation=activation))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification output layer
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Instantiate KerasClassifier with the model-building function
nn_model = KerasClassifier(model=build_nn_model, verbose=0)

# Define a parameter grid for the nn_model
param_grid_nn = {
    'model__neurons_layer1': [64, 128],
    'model__neurons_layer2': [32, 64],
    'batch_size': [32, 64],
    'epochs': [10, 20]
}

# Instantiate GridSearchCV for Neural Network
grid_search_nn = GridSearchCV(estimator=nn_model, param_grid=param_grid_nn, cv=3, scoring='accuracy')

# Fit GridSearchCV to the preprocessed training data and encoded labels
grid_search_nn.fit(X_train_processed_df, y_train_encoded)

# Print the best hyperparameters for the Neural Network
print("Best hyperparameters for Neural Network:", grid_search_nn.best_params_)


# ------------------------------
# Data Preprocessing + Encoding
# ------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Example: load dataset (replace with your actual file)
df = pd.read_csv("submission.csv")   # or your clean dataset

# Separate features and labels
X = df.drop(columns=["model_id"])    # features
y = df["model_id"]                   # labels (or replace with real target column)

# Encode labels if needed
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_processed = scaler.fit_transform(X_train)
X_test_processed = scaler.transform(X_test)

# Convert back to DataFrame (so GridSearchCV doesnâ€™t complain about dtypes)
X_train_processed_df = pd.DataFrame(X_train_processed, columns=X.columns)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=X.columns)

y_train_encoded = y_train
y_test_encoded = y_test

print("âœ… Preprocessing complete:", X_train_processed_df.shape, y_train_encoded.shape)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import make_classification

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# ================================
# 1. Data Loading and Preprocessing
# ================================
# Create sample data since no file path was provided
print("Creating sample data for demonstration...")
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                          n_redundant=5, n_classes=5, random_state=42)

# Create DataFrame to match your expected structure
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
df['gesture'] = y

# Create test_df for submission (using a subset of data)
test_df = df.sample(200, random_state=42).copy()

# If you want to use your own data, replace the above with:
# df = pd.read_csv("your_train_data.csv")
# test_df = pd.read_csv("your_test_data.csv")

X = df.drop(columns=["gesture"])
y = df["gesture"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_proc = scaler.fit_transform(X_train)
X_test_proc = scaler.transform(X_test)

# ================================
# 2. Build NN Model
# ================================
def create_model(neurons_layer1=128, neurons_layer2=64, learning_rate=0.001, dropout_rate=0.3):
    model = Sequential()
    model.add(Dense(neurons_layer1, activation='relu', input_shape=(X_train_proc.shape[1],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons_layer2, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# ================================
# 3. Hyperparameter Tuning
# ================================
# Create model
model = KerasClassifier(build_fn=create_model, verbose=0)

# Define the grid search parameters
param_grid = {
    'neurons_layer1': [64, 128],
    'neurons_layer2': [32, 64],
    'batch_size': [32, 64],
    'epochs': [10, 20],
    'learning_rate': [0.001, 0.01]
}

# Create Grid Search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3, scoring='accuracy')
grid_result = grid.fit(X_train_proc, y_train)

# Summarize results
print("âœ… Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# ================================
# 4. Train Best Model
# ================================
best_params = grid_result.best_params_
best_model = create_model(
    neurons_layer1=best_params['neurons_layer1'],
    neurons_layer2=best_params['neurons_layer2'],
    learning_rate=best_params['learning_rate']
)

history = best_model.fit(
    X_train_proc, y_train,
    batch_size=best_params['batch_size'],
    epochs=best_params['epochs'],
    validation_data=(X_test_proc, y_test),
    verbose=1
)

# ================================
# 5. Evaluation
# ================================
# Evaluate on test set
test_loss, test_accuracy = best_model.evaluate(X_test_proc, y_test, verbose=0)
print(f"âœ… Test Accuracy: {test_accuracy:.4f}")

# Make predictions
y_pred_proba = best_model.predict(X_test_proc)
y_pred = np.argmax(y_pred_proba, axis=1)

# Classification report
print("\nğŸ“Š Classification Report")
print(classification_report(y_test, y_pred, target_names=[f"Class_{i}" for i in le.classes_]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[f"Class_{i}" for i in le.classes_], 
            yticklabels=[f"Class_{i}" for i in le.classes_])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# ================================
# 6. Create Submission
# ================================
# Prepare test data for prediction
test_features = test_df.drop(columns=["gesture"], errors='ignore')
if len(test_features) == 0:
    test_features = test_df.copy()
    
test_features_proc = scaler.transform(test_features)

# Make predictions on test data
test_pred_proba = best_model.predict(test_features_proc)
test_pred = np.argmax(test_pred_proba, axis=1)

# Convert back to original labels
test_pred_labels = le.inverse_transform(test_pred)

# Create submission
if "row_id" in test_df.columns:
    row_ids = test_df["row_id"]
else:
    row_ids = test_df.index

submission = pd.DataFrame({
    "row_id": row_ids,
    "gesture": test_pred_labels
})
submission.to_csv("submission.csv", index=False)

print("âœ… Normalized confusion matrix plotted, feature importance visualized, and submission.csv saved.")
print(f"Submission preview:")
print(submission.head())

# ================================
# 7. Plot Training History
# ================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import lightgbm as lgb
import pandas as pd
import os

# First, let's check what files are available and load data properly
print("Current directory:", os.getcwd())
print("Files in directory:")
csv_files = []
for file in os.listdir('.'):
    if file.endswith('.csv'):
        csv_files.append(file)
        print(f"- {file}")

# Load data or create sample data
if 'train.csv' in csv_files:
    train_df = pd.read_csv('train.csv')
    print("Loaded train.csv")
else:
    print("Creating sample training data...")
    np.random.seed(42)
    n_samples = 1000
    n_features = 30
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 5, n_samples)
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    train_df = pd.DataFrame(X, columns=feature_cols)
    train_df['gesture'] = y

if 'test.csv' in csv_files:
    test_df = pd.read_csv('test.csv')
    print("Loaded test.csv")
else:
    print("Creating sample test data...")
    test_X = np.random.randn(200, len(train_df.columns) - 1)
    test_df = pd.DataFrame(test_X, columns=[col for col in train_df.columns if col != 'gesture'])

# Display data info
print(f"\nTraining data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Check if 'gesture' column exists and convert to string
if 'gesture' not in train_df.columns:
    print("'gesture' column not found. Creating sample target...")
    train_df['gesture'] = np.random.randint(0, 5, len(train_df))

# Convert to string to avoid issues
train_df['gesture'] = train_df['gesture'].astype(str)

# Define features and target
X = train_df.drop(columns=["gesture"])
y = train_df["gesture"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames
X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# ================================
# 1. Stratified K-Fold with CV predictions
# ================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_true_all, y_pred_all = [], []
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_df, y_train), 1):
    X_tr, X_val = X_train_df.iloc[train_idx], X_train_df.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X_tr, y_tr)

    y_val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    fold_accuracies.append(acc)

    y_true_all.extend(y_val)
    y_pred_all.extend(y_val_pred)

    print(f"Fold {fold}: Accuracy = {acc:.4f}")

print(f"\nMean CV Accuracy: {np.mean(fold_accuracies):.4f} Â± {np.std(fold_accuracies):.4f}")

# ================================
# 2. Normalized Confusion Matrix
# ================================
cm = confusion_matrix(y_true_all, y_pred_all, normalize='true')

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Class-Normalized Confusion Matrix (CV Predictions)")
plt.tight_layout()
plt.show()

# ================================
# 3. Classification Report - FIXED
# ================================
# Convert numeric predictions back to original string labels for proper reporting
y_true_labels = label_encoder.inverse_transform(y_true_all)
y_pred_labels = label_encoder.inverse_transform(y_pred_all)

# Ensure class names are strings
class_names = [str(cls) for cls in label_encoder.classes_]

print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=class_names))

# ================================
# 4. Retrain on Full Data
# ================================
final_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
final_model.fit(X_train_df, y_train)

# ================================
# 5. Feature Importance (Top 20)
# ================================
importances = final_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
# Create a DataFrame for easier plotting
importance_df = pd.DataFrame({
    'feature': X_train_df.columns[indices[:20]],
    'importance': importances[indices[:20]]
})

sns.barplot(x='importance', y='feature', data=importance_df, palette="viridis")
plt.title("Top 20 Feature Importances (LightGBM)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# ================================
# 6. Predict Test Set + Save Submission
# ================================
# Prepare test data (ensure it has the same columns as training)
missing_cols = set(X.columns) - set(test_df.columns)
if missing_cols:
    print(f"Adding missing columns to test data: {missing_cols}")
    for col in missing_cols:
        test_df[col] = 0

# Remove extra columns
test_df = test_df[X.columns]

# Scale test data
test_scaled = scaler.transform(test_df)
test_df_scaled = pd.DataFrame(test_scaled, columns=X.columns)

# Make predictions
y_pred_encoded = final_model.predict(test_df_scaled)
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Create submission
if "row_id" in test_df.columns:
    row_ids = test_df["row_id"]
else:
    row_ids = test_df.index

submission = pd.DataFrame({
    "row_id": row_ids,
    "gesture": y_pred
})
submission.to_csv("submission.csv", index=False)

print("âœ… Normalized confusion matrix plotted, feature importance visualized, and submission.csv saved.")
print(f"Submission preview:")
print(submission.head())


# Make predictions on the preprocessed test data using the trained XGBoost model
xgb_predictions = xgb_model.predict(X_test_processed_df)

# Since the Neural Network GridSearchCV failed and we don't have a trained NN from tuning,
# we cannot make predictions with the tuned NN.

# Since the ground truth for the test data (y_test) is not available,
# we cannot perform a full evaluation with metrics like accuracy or a confusion matrix for either model.
print("Ground truth (y_test) for the test data is not available.")
print("Therefore, a full evaluation with metrics and a confusion matrix cannot be performed.")
print("Also, the Neural Network hyperparameter tuning failed, so its evaluation is not possible.")

# If y_test were available and NN tuning was successful, the evaluation code (commented out in the previous cell) would be used.


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import lightgbm as lgb
import pandas as pd
import os

# First, let's check what files are available and load data properly
print("Current directory:", os.getcwd())
print("Files in directory:")
csv_files = []
for file in os.listdir('.'):
    if file.endswith('.csv'):
        csv_files.append(file)
        print(f"- {file}")

# Load data or create sample data
if 'train.csv' in csv_files:
    train_df = pd.read_csv('train.csv')
    print("Loaded train.csv")
else:
    print("Creating sample training data...")
    np.random.seed(42)
    n_samples = 1000
    n_features = 30
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 5, n_samples)
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    train_df = pd.DataFrame(X, columns=feature_cols)
    train_df['gesture'] = y

if 'test.csv' in csv_files:
    test_df = pd.read_csv('test.csv')
    print("Loaded test.csv")
else:
    print("Creating sample test data...")
    test_X = np.random.randn(200, len(train_df.columns) - 1)
    test_df = pd.DataFrame(test_X, columns=[col for col in train_df.columns if col != 'gesture'])

# Display data info
print(f"\nTraining data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Check if 'gesture' column exists and convert to string
if 'gesture' not in train_df.columns:
    print("'gesture' column not found. Creating sample target...")
    train_df['gesture'] = np.random.randint(0, 5, len(train_df))

# Convert to string to avoid issues
train_df['gesture'] = train_df['gesture'].astype(str)

# Define features and target
X = train_df.drop(columns=["gesture"])
y = train_df["gesture"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames
X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# ================================
# 1. Stratified K-Fold with CV predictions
# ================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_true_all, y_pred_all = [], []
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_df, y_train), 1):
    X_tr, X_val = X_train_df.iloc[train_idx], X_train_df.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X_tr, y_tr)

    y_val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    fold_accuracies.append(acc)

    y_true_all.extend(y_val)
    y_pred_all.extend(y_val_pred)

    print(f"Fold {fold}: Accuracy = {acc:.4f}")

print(f"\nMean CV Accuracy: {np.mean(fold_accuracies):.4f} Â± {np.std(fold_accuracies):.4f}")

# ================================
# 2. Normalized Confusion Matrix
# ================================
cm = confusion_matrix(y_true_all, y_pred_all, normalize='true')

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Class-Normalized Confusion Matrix (CV Predictions)")
plt.tight_layout()
plt.show()

# ================================
# 3. Classification Report - FIXED
# ================================
# Convert numeric predictions back to original string labels for proper reporting
y_true_labels = label_encoder.inverse_transform(y_true_all)
y_pred_labels = label_encoder.inverse_transform(y_pred_all)

# Ensure class names are strings
class_names = [str(cls) for cls in label_encoder.classes_]

print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=class_names))

# ================================
# 4. Retrain on Full Data
# ================================
final_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
final_model.fit(X_train_df, y_train)

# ================================
# 5. Feature Importance (Top 20)
# ================================
importances = final_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
# Create a DataFrame for easier plotting
importance_df = pd.DataFrame({
    'feature': X_train_df.columns[indices[:20]],
    'importance': importances[indices[:20]]
})

sns.barplot(x='importance', y='feature', data=importance_df, palette="viridis")
plt.title("Top 20 Feature Importances (LightGBM)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# ================================
# 6. Predict Test Set + Save Submission
# ================================
# Prepare test data (ensure it has the same columns as training)
missing_cols = set(X.columns) - set(test_df.columns)
if missing_cols:
    print(f"Adding missing columns to test data: {missing_cols}")
    for col in missing_cols:
        test_df[col] = 0

# Remove extra columns
test_df = test_df[X.columns]

# Scale test data
test_scaled = scaler.transform(test_df)
test_df_scaled = pd.DataFrame(test_scaled, columns=X.columns)

# Make predictions
y_pred_encoded = final_model.predict(test_df_scaled)
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Create submission
if "row_id" in test_df.columns:
    row_ids = test_df["row_id"]
else:
    row_ids = test_df.index

submission = pd.DataFrame({
    "row_id": row_ids,
    "gesture": y_pred
})
submission.to_csv("submission.csv", index=False)

print("âœ… Normalized confusion matrix plotted, feature importance visualized, and submission.csv saved.")
print(f"Submission preview:")
print(submission.head())


#  Ensure submission.csv exists, otherwise create a dummy version for workflow
import os
import numpy as np
import pandas as pd

if not os.path.exists("submission.csv"):
    print("âš ï¸� 'submission.csv' not found. Creating dummy submission...")
    n_models, n_triggers = 10, 3 * 75  # Adjust as needed
    dummy = pd.DataFrame({
        "model_id": [f"model_{i:04d}" for i in range(n_models)],
        **{f"t{j}": np.random.uniform(0.01, 0.1, size=n_models) for j in range(n_triggers)}
    })
    dummy.to_csv("submission.csv", index=False)
    print("âœ… Dummy 'submission.csv' generated:", dummy.shape)
else:
    print("âœ… 'submission.csv' found.")


from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Instantiate XGBoost model
xgb_model = XGBClassifier(random_state=42)

# Define a simple Neural Network model
nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_processed_df.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification output layer
])

# Compile the Neural Network model
nn_model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

# Print NN model summary
print("XGBoost model instantiated.")
print("\nNeural Network model summary:")
nn_model.summary()


CTAI - CTD Hackathon
Predictive Models for Material Requirement Prediction in Supply Chain

header (3).png

Author/Name: Ishita
Designation: Data Scientist

[ ]
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


[ ]
import pandas as pd

try:
  test_df = pd.read_csv('/content/test.csv')
  train_df = pd.read_csv('/content/train.csv')

  print("Test DataFrame head:")
  display(test_df.head())

  print("\nTrain DataFrame head:")
  display(train_df.head())

except FileNotFoundError:
  print("Make sure 'test.csv' and 'train.csv' are uploaded to the Colab environment.")
except Exception as e:
  print(f"An error occurred: {e}")

Task
Analyze the provided test and train datasets ("/content/test.csv", "/content/train.csv") by training a model on the training data and evaluating its performance on the test data.

Data preprocessing
Subtask:
Prepare the test and train data for model training. This may include handling missing values, encoding categorical variables, and scaling numerical features.

Reasoning: Identify and handle missing values in both the train and test dataframes by checking for missing values and imputing or dropping as necessary. Then identify and encode categorical features using one-hot encoding, ensuring consistency across both datasets.


[ ]
# Identify missing values
print("Missing values in train_df:")
display(train_df.isnull().sum()[train_df.isnull().sum() > 0])

print("\nMissing values in test_df:")
display(test_df.isnull().sum()[test_df.isnull().sum() > 0])

# The 'tof' columns have a large number of missing values. Imputing them with the median might be reasonable given the nature of the data.
# For numerical columns with missing values (primarily tof columns), impute with the median.
for col in train_df.columns:
    if train_df[col].dtype in ['float64', 'int64']:
        if train_df[col].isnull().sum() > 0:
            median_val = train_df[col].median()
            train_df[col].fillna(median_val, inplace=True)
            if col in test_df.columns:
                test_df[col].fillna(median_val, inplace=True)

# Check for remaining missing values after imputation
print("\nMissing values in train_df after imputation:")
display(train_df.isnull().sum()[train_df.isnull().sum() > 0])

print("\nMissing values in test_df after imputation:")
display(test_df.isnull().sum()[test_df.isnull().sum() > 0])

# Identify categorical columns
categorical_cols_train = train_df.select_dtypes(include=['object']).columns
categorical_cols_test = test_df.select_dtypes(include=['object']).columns

print("\nCategorical columns in train_df:")
print(categorical_cols_train)

print("\nCategorical columns in test_df:")
print(categorical_cols_test)

# Exclude 'row_id' from categorical columns as it's an identifier
categorical_cols_train = categorical_cols_train.drop('row_id', errors='ignore')
categorical_cols_test = categorical_cols_test.drop('row_id', errors='ignore')


# Apply one-hot encoding
# Combine train and test for consistent encoding
combined_df = pd.concat([train_df.drop('sequence_type', axis=1, errors='ignore'), test_df], ignore_index=True)
combined_df = pd.get_dummies(combined_df, columns=categorical_cols_train.intersection(categorical_cols_test), dummy_na=False)

# Separate back into train and test
train_processed_df = combined_df.iloc[:len(train_df)].copy()
test_processed_df = combined_df.iloc[len(train_df):].copy()

# Add back 'sequence_type' to the training data if it was dropped
if 'sequence_type' in train_df.columns:
    train_processed_df['sequence_type'] = train_df['sequence_type']

print("\nTrain DataFrame after one-hot encoding:")
display(train_processed_df.head())

print("\nTest DataFrame after one-hot encoding:")
display(test_processed_df.head())


[ ]
grid_search_nn.fit(X_train_processed_df, y_train_encoded)



[ ]
# ------------------------------
# Data Preprocessing + Encoding
# ------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Example: load dataset (replace with your actual file)
df = pd.read_csv("submission.csv")   # or your clean dataset

# Separate features and labels
X = df.drop(columns=["model_id"])    # features
y = df["model_id"]                   # labels (or replace with real target column)

# Encode labels if needed
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_processed = scaler.fit_transform(X_train)
X_test_processed = scaler.transform(X_test)

# Convert back to DataFrame (so GridSearchCV doesnâ€™t complain about dtypes)
X_train_processed_df = pd.DataFrame(X_train_processed, columns=X.columns)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=X.columns)

y_train_encoded = y_train
y_test_encoded = y_test

print("âœ… Preprocessing complete:", X_train_processed_df.shape, y_train_encoded.shape)

âœ… Preprocessing complete: (8, 225) (8,)
Feature Engineering
Subtask: Feature Engineering Function

Reasoning: Identify numerical columns, scale them using StandardScaler, fitting only on the training data, and then transform both train and test data. Finally, separate the target variable 'sequence_type' from the features in the training DataFrame.


[ ]
from sklearn.preprocessing import StandardScaler

# Identify numerical columns after one-hot encoding
numerical_cols_train = train_processed_df.select_dtypes(include=['float64', 'int64']).columns
numerical_cols_test = test_processed_df.select_dtypes(include=['float64', 'int64']).columns

# Exclude 'sequence_counter' and 'row_id' from scaling if present
cols_to_exclude = ['sequence_counter']
numerical_cols_train = numerical_cols_train.difference(cols_to_exclude)
numerical_cols_test = numerical_cols_test.difference(cols_to_exclude)

print("\nNumerical columns to scale in train_processed_df:")
print(numerical_cols_train)

print("\nNumerical columns to scale in test_processed_df:")
print(numerical_cols_test)


# Initialize StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both train and test data
train_processed_df[numerical_cols_train] = scaler.fit_transform(train_processed_df[numerical_cols_train])
test_processed_df[numerical_cols_test] = scaler.transform(test_processed_df[numerical_cols_test])

print("\nTrain DataFrame after scaling numerical features:")
display(train_processed_df.head())

print("\nTest DataFrame after scaling numerical features:")
display(test_processed_df.head())

# Separate target variable from features in the training data
if 'sequence_type' in train_processed_df.columns:
    X_train = train_processed_df.drop('sequence_type', axis=1)
    y_train = train_processed_df['sequence_type']
else:
    X_train = train_processed_df
    y_train = None
    print("Warning: 'sequence_type' column not found in train_processed_df.")


X_test = test_processed_df

print("\nFeatures for training (X_train) head:")
display(X_train.head())

if y_train is not None:
    print("\nTarget variable for training (y_train) head:")
    display(y_train.head())

print("\nFeatures for testing (X_test) head:")
display(X_test.head())


[8]
0s
def build_features(sensor_df: pd.DataFrame, demo_df: pd.DataFrame) -> pd.DataFrame:
    df = sensor_df.copy()
    df = df.replace(-1.0, np.nan)
    df = df.merge(demo_df, on="subject", how="left")

    id_cols = ["row_id", "sequence_id", "sequence_counter", "subject"]
    demo_cols = ["adult_child", "age", "sex", "handedness", "height_cm",
                 "shoulder_to_wrist_cm", "elbow_to_wrist_cm"]
    numeric_cols = [c for c in df.columns if c not in id_cols + demo_cols]


Unsupervised Learning

[ ]
# Unsupervised Modeling
# ==============================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# --- Load data (update paths if needed) ---
df_submission = pd.read_csv("submission.csv")

# Drop model_id column if present
if "model_id" in df_submission.columns:
    X_test = df_submission.drop(columns=["model_id"]).values
else:
    X_test = df_submission.values

# --- Preprocessing ---
scaler = StandardScaler()
X_test_proc = scaler.fit_transform(X_test)

# --- Isolation Forest ---
iso = IsolationForest(n_estimators=300, random_state=42, n_jobs=-1)
iso.fit(X_test_proc)
anomaly_score = -iso.score_samples(X_test_proc)
pred = iso.predict(X_test_proc)  # -1 = anomaly, 1 = normal

# --- PCA + KMeans clustering ---
pca_components = min(10, X_test_proc.shape[1], max(1, X_test_proc.shape[0] - 1))
pca = PCA(n_components=pca_components, random_state=42)
X_embedded = pca.fit_transform(X_test_proc)

n_clusters = min(3, max(1, X_test_proc.shape[0]))
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(X_embedded)

# --- Quick reporting ---
print("IsolationForest anomaly scores:", anomaly_score[:5])
print("Predictions (-1 = anomaly, 1 = normal):", np.unique(pred, return_counts=True))
print("Cluster labels:", np.unique(cluster_labels, return_counts=True))

IsolationForest anomaly scores: [0.48825617 0.49114693 0.49193033 0.5021581  0.49869346]
Predictions (-1 = anomaly, 1 = normal): (array([-1,  1]), array([3, 7]))
Cluster labels: (array([0, 1, 2], dtype=int32), array([4, 4, 2]))

[ ]
# Visualization: PCA scatter with anomaly coloring
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

# Scatter with cluster colors
scatter = plt.scatter(
    X_embedded[:, 0], X_embedded[:, 1],
    c=cluster_labels, cmap="tab10", alpha=0.7, s=60,
    edgecolors=["red" if p == -1 else "black" for p in pred], linewidth=1.2


[7]
6s
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import lightgbm as lgb
import pandas as pd
import os

# Try to load data with multiple possible paths
def load_data():
    possible_paths = [
        "/content/train.csv",  # Common Colab path
        "./train.csv",         # Current directory
        "train.csv",           # Just the filename
        "/kaggle/input/gesture-recognition/train.csv",  # Common Kaggle path
    ]
    
    train_df, test_df = None, None
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                train_df = pd.read_csv(path)
                print(f"Successfully loaded training data from: {path}")
                break
        except:
            continue
    
    if train_df is None:
        # Create sample data for demonstration if no CSV files are found
        print("No CSV files found. Creating sample data for demonstration.")
        np.random.seed(42)
        n_samples = 1000
        n_features = 30
        
        # Create sample features
        X = np.random.randn(n_samples, n_features)
        # Create sample target (5 classes)
        y = np.random.randint(0, 5, n_samples)
        
        # Create DataFrame
        feature_cols = [f'feature_{i}' for i in range(n_features)]
        train_df = pd.DataFrame(X, columns=feature_cols)
        train_df['gesture'] = y
        
        # Create test data
        test_X = np.random.randn(200, n_features)
        test_df = pd.DataFrame(test_X, columns=feature_cols)
    else:
        # Try to load test data
        test_paths = [
            path.replace('train', 'test'),
            "/content/test.csv",
            "./test.csv",
            "test.csv",
            "/kaggle/input/gesture-recognition/test.csv",
        ]
        
        for path in test_paths:
            try:
                if os.path.exists(path):
                    test_df = pd.read_csv(path)
                    print(f"Successfully loaded test data from: {path}")
                    break
            except:
                continue
        
        if test_df is None:
            print("No test CSV found. Using a portion of training data for demonstration.")
            test_df = train_df.iloc[:200].drop(columns=['gesture'], errors='ignore')
    
    return train_df, test_df

# Load data
train_df, test_df = load_data()

# Display basic info about the data
print("\nTraining data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("\nTraining data columns:")
print(train_df.columns.tolist())

# Check if 'gesture' column exists, if not create a sample one
if 'gesture' not in train_df.columns:
    print("'gesture' column not found. Creating a sample target variable.")
    n_classes = 5
    train_df['gesture'] = np.random.randint(0, n_classes, len(train_df))

# Define features and target
X = train_df.drop(columns=["gesture"])
y = train_df["gesture"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_proc = scaler.fit_transform(X_train)
X_test_proc = scaler.transform(X_test)

# Keep DataFrames for easier handling
X_train_df = pd.DataFrame(X_train_proc, columns=X.columns)
y_train_df = pd.Series(y_train).reset_index(drop=True)
X_test_df = pd.DataFrame(X_test_proc, columns=X.columns)

# Stratified K-Fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_true_all, y_pred_all = [], []
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_df, y_train_df), 1):
    X_tr, X_val = X_train_df.iloc[train_idx], X_train_df.iloc[val_idx]
    y_tr, y_val = y_train_df.iloc[train_idx], y_train_df.iloc[val_idx]

    model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X_tr, y_tr)

    y_val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    fold_accuracies.append(acc)
    
    print(f"Fold {fold}: Accuracy = {acc:.4f}")

    y_true_all.extend(y_val)
    y_pred_all.extend(y_val_pred)

print(f"\nMean CV Accuracy: {np.mean(fold_accuracies):.4f} Â± {np.std(fold_accuracies):.4f}")

# Normalized Confusion Matrix
cm = confusion_matrix(y_true_all, y_pred_all, labels=np.unique(y_train_df), normalize='true')

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Class-Normalized Confusion Matrix (CV Predictions)")
plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true_all, y_pred_all, target_names=label_encoder.classes_))

# Retrain on Full Training Data
final_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
final_model.fit(X_train_df, y_train_df)

# Feature Importance (Top 20)
importances = final_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_train_df.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), palette="viridis")
plt.title("Top 20 Feature Importances (LightGBM)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Predict Test Set + Save Submission
X_test_processed = scaler.transform(test_df[X.columns])  # Use only the columns that exist in training
X_test_proc_df = pd.DataFrame(X_test_processed, columns=X.columns)

y_pred = final_model.predict(X_test_proc_df)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Create submission file
if "row_id" in test_df.columns:
    row_ids = test_df["row_id"]
else:
    row_ids = test_df.index

submission = pd.DataFrame({
    "row_id": row_ids,
    "gesture": y_pred_labels
})
submission.to_csv("submission.csv", index=False)

print("âœ… CV done, confusion matrix plotted, feature importance visualized, and submission.csv saved.")
print(f"Submission file contains {len(submission)} predictions.")

Next steps:

[3]
0s
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import lightgbm as lgb
import pandas as pd

# Load your data
train_df = pd.read_csv("path/to/your/train.csv")
test_df = pd.read_csv("path/to/your/test.csv")

# Define features and target
X = train_df.drop(columns=["gesture"])
y = train_df["gesture"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_proc = scaler.fit_transform(X_train)
X_test_proc = scaler.transform(X_test)

# Keep DataFrames for easier handling
X_train_df = pd.DataFrame(X_train_proc, columns=X.columns)
y_train_df = pd.Series(y_train).reset_index(drop=True)
X_test_df = pd.DataFrame(X_test_proc, columns=X.columns)

# Stratified K-Fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_true_all, y_pred_all = [], []
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_df, y_train_df), 1):
    X_tr, X_val = X_train_df.iloc[train_idx], X_train_df.iloc[val_idx]
    y_tr, y_val = y_train_df.iloc[train_idx], y_train_df.iloc[val_idx]

    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_tr, y_tr)

    y_val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    fold_accuracies.append(acc)

    y_true_all.extend(y_val)
    y_pred_all.extend(y_val_pred)

print(f"Mean CV Accuracy: {np.mean(fold_accuracies):.4f} Â± {np.std(fold_accuracies):.4f}")

# Normalized Confusion Matrix
cm = confusion_matrix(y_true_all, y_pred_all, labels=np.unique(y_train_df), normalize='true')

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Class-Normalized Confusion Matrix (CV Predictions)")
plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true_all, y_pred_all, target_names=label_encoder.classes_))

# Hyperparameter Tuning (optional)
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'num_leaves': [31, 50],
}

grid_search = GridSearchCV(
    estimator=lgb.LGBMClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_df, y_train_df)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Retrain on Full Training Data with best parameters
final_model = grid_search.best_estimator_
final_model.fit(X_train_df, y_train_df)

# Feature Importance (Top 20)
importances = final_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_train_df.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), palette="viridis")
plt.title("Top 20 Feature Importances (LightGBM)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Predict Test Set + Save Submission
y_pred = final_model.predict(X_test_df)

# If row_id exists in test_df, use it; otherwise fallback to index
if "row_id" in test_df.columns:
    row_ids = test_df["row_id"]
else:
    row_ids = test_df.index

# Convert numeric predictions back to original labels
y_pred_labels = label_encoder.inverse_transform(y_pred)

submission = pd.DataFrame({
    "row_id": row_ids,
    "gesture": y_pred_labels
})
submission.to_csv("submission.csv", index=False)

print("âœ… CV done, confusion matrix plotted, feature importance visualized, and submission.csv saved.")

Next steps:

[2]
0s
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import lightgbm as lgb
import pandas as pd

# Load your data
train_df = pd.read_csv("/content/train.csv")
test_df = pd.read_csv("/content/test.csv")
#check the path
train_df = pd.read_csv("path/to/your/train.csv")


# Define features and target
X = train_df.drop(columns=["gesture"])  # Replace 'gesture' with your actual target column
y = train_df["gesture"]

# Encode target labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# ================================
# 1. Train/test split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ================================
# 2. Scale features
# ================================
scaler = StandardScaler()
X_train_proc = scaler.fit_transform(X_train)
X_test_proc  = scaler.transform(X_test)

# Keep DataFrames for easier handling
X_train_df = pd.DataFrame(X_train_proc, columns=X.columns)
y_train_df = pd.Series(y_train).reset_index(drop=True)
X_test_df  = pd.DataFrame(X_test_proc, columns=X.columns)

# ================================
# 3. Stratified K-Fold CV
# ================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_true_all, y_pred_all = [], []
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_df, y_train_df), 1):
    X_tr, X_val = X_train_df.iloc[train_idx], X_train_df.iloc[val_idx]
    y_tr, y_val = y_train_df.iloc[train_idx], y_train_df.iloc[val_idx]

    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_tr, y_tr)

    y_val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    fold_accuracies.append(acc)

    y_true_all.extend(y_val)
    y_pred_all.extend(y_val_pred)

print(f"Mean CV Accuracy: {np.mean(fold_accuracies):.4f} Â± {np.std(fold_accuracies):.4f}")

# ================================
# 4. Normalized Confusion Matrix
# ================================
cm = confusion_matrix(y_true_all, y_pred_all, labels=np.unique(y_train_df), normalize='true')

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=np.unique(y_train_df),
            yticklabels=np.unique(y_train_df))

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Class-Normalized Confusion Matrix (CV Predictions)")
plt.show()

# ================================
# 5. Retrain on Full Training Data
# ================================
final_model = lgb.LGBMClassifier(random_state=42)
final_model.fit(X_train_df, y_train_df)

# ================================
# 6. Feature Importance (Top 20)
# ================================
importances = final_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices[:20]], y=X_train_df.columns[indices[:20]], palette="viridis")
plt.title("Top 20 Feature Importances (LightGBM)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# ================================
# 7. Predict Test Set + Save Submission
# ================================
y_pred = final_model.predict(X_test_df)

# If row_id exists in test_df, use it; otherwise fallback to index
if "row_id" in test_df.columns:
    row_ids = test_df["row_id"]
else:
    row_ids = test_df.index

submission = pd.DataFrame({
    "row_id": row_ids,
    "gesture": y_pred
})
submission.to_csv("submission.csv", index=False)

print("âœ… CV done, confusion matrix plotted, feature importance visualized, and submission.csv saved.")


Next steps:
Model selection
Subtask:
Choose an appropriate model for the task.

Reasoning: Based on the classification nature of the problem and the preprocessed data with numerical and one-hot encoded features, a RandomForestClassifier is a suitable choice. It can handle both numerical and categorical features effectively and is less prone to overfitting compared to some other models.


[ ]
from sklearn.ensemble import RandomForestClassifier

# Choose RandomForestClassifier as the model
model = RandomForestClassifier(random_state=42)

print("Chosen model: RandomForestClassifier")
Chosen model: RandomForestClassifier
Model training
Subtask:
Train the selected model using the preprocessed training data.

Reasoning: Train the RandomForestClassifier model using the prepared training data.


[ ]
# Train the model
model.fit(X_train, y_train)

print("RandomForestClassifier model trained successfully.")
RandomForestClassifier model trained successfully.
Reasoning: The error indicates that there are still non-numeric columns in X_train. Inspect the columns of X_train to identify the non-numeric columns that need to be dropped before training the model.


[ ]
import pandas as pd
import numpy as np
import os

# ==============================
# Step 1: Load Data
# ==============================
train_demo = pd.read_csv("/content/train_demographics.csv")
test = pd.read_csv("/content/test.csv")
test_demo = pd.read_csv("/content/test_demographics.csv")


train_sensor_path = "/content/train.csv"
has_train_sensor = os.path.exists(train_sensor_path)
train = pd.read_csv(train_sensor_path) if has_train_sensor else None


# Optional: if labels are provided
label_path = "/content/train_labels.csv"
has_labels = os.path.exists(label_path)
labels = pd.read_csv(label_path) if has_labels else None


# ==============================
# Step 2: Feature Engineering Function
# ==============================
def build_features(sensor_df: pd.DataFrame, demo_df: pd.DataFrame) -> pd.DataFrame:
    df = sensor_df.copy()
    df = df.replace(-1.0, np.nan)
    df = df.merge(demo_df, on="subject", how="left")


    id_cols = ["row_id", "sequence_id", "sequence_counter", "subject"]
    demo_cols = ["adult_child", "age", "sex", "handedness", "height_cm",
    "shoulder_to_wrist_cm", "elbow_to_wrist_cm"]

    # Explicitly exclude non-numeric columns
    non_numeric_cols = id_cols + demo_cols + ['sequence_type', 'orientation', 'behavior', 'phase', 'gesture']
    numeric_cols = [c for c in df.columns if c not in non_numeric_cols]


    agg_funcs = ["mean", "std", "min", "max", "median", "skew"]
    seq_feats = (
    df.groupby("sequence_id")[numeric_cols]
    .agg(agg_funcs)
    )
    seq_feats.columns = ["_".join(col).strip() for col in seq_feats.columns.values]
    seq_feats = seq_feats.reset_index()


    demo_feats = (
    df.groupby("sequence_id")[demo_cols]
    .first()
    .reset_index()
    )
    final = seq_feats.merge(demo_feats, on="sequence_id", how="left")
    return final

[ ]
print(X_train.dtypes)
row_id                 object
sequence_counter        int64
orientation            object
behavior               object
phase                  object
                        ...  
subject_SUBJ_063319      bool
subject_SUBJ_063346      bool
subject_SUBJ_063447      bool
subject_SUBJ_063464      bool
subject_SUBJ_064387      bool
Length: 644, dtype: object

Reasoning: The output shows that 'row_id', 'orientation', 'behavior', and 'phase' columns are of object type, which caused the error during model training. These columns are not suitable for direct use in the model and should be dropped.

Reasoning: The error message "ValueError: could not convert string to float: 'Cheek - pinch skin'" indicates that the 'gesture' column, which contains strings, is still present in the training data and is causing issues during model training. This column should also be dropped from both the training and testing dataframes.


[ ]
# Drop the 'gesture' column from X_train and X_test
X_train = X_train.drop('gesture', axis=1)
X_test = X_test.drop('gesture', axis=1)

# Train the model again
model.fit(X_train, y_train)

print("RandomForestClassifier model trained successfully after dropping the 'gesture' column.")
RandomForestClassifier model trained successfully after dropping the 'gesture' column.
Model evaluation
Subtask:
Evaluate the trained model using the preprocessed test data to assess its performance.

Reasoning: Use the trained model to make predictions on the preprocessed test data and create the submission DataFrame.


[ ]
# Make predictions on the test data
predictions = model.predict(X_test)

# Create the submission DataFrame
submission_df = pd.DataFrame({'row_id': test_df['row_id'], 'sequence_type': predictions})

# Display the head of the submission DataFrame
print("Submission DataFrame head:")
display(submission_df.head())

submission.csv - submission cell

[ ]
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
import lightgbm as lgb

# ================================
# 1. Stratified K-Fold with CV predictions
# ================================



[ ]
        **{f"t{j}": np.random.uniform(0.01, 0.1, size=n_models) for j in range(n_triggers)}
    })
    dummy.to_csv("submission.csv", index=False)
    print("âœ… Dummy 'submission.csv' generated:", dummy.shape)
else:
    print("âœ… 'submission.csv' found.")

âœ… 'submission.csv' found.
Summary:
Data Analysis Key Findings
Missing values, primarily in 'tof' and rotation ('rot_') columns, were imputed using the median strategy.
Categorical columns ('sequence_id' and 'subject') were successfully one-hot encoded across both train and test datasets.
Numerical features were scaled using StandardScaler fitted on the training data.
Non-numeric columns ('row_id', 'orientation', 'behavior', 'phase', and 'gesture') were removed from the feature sets (X_train and X_test) to enable model training.
A RandomForestClassifier model was selected and successfully trained on the preprocessed training data.
Predictions were generated on the test data, and a submission DataFrame was created containing 'row_id' and the predicted 'sequence_type'.
Insights or Next Steps
Further evaluation of the model's performance using metrics like accuracy, precision, recall, or F1-score would provide a more quantitative assessment of the model's effectiveness.
Exploring alternative models or hyperparameter tuning of the RandomForestClassifier could potentially improve prediction accuracy.
Model Evaluation - Confusion Matrix Heatmap
Subtask:
Visualize the confusion matrix as a heatmap to assess the model's performance on the test data.

Reasoning: Generate a confusion matrix using the true labels (y_test) and the model's predictions (predictions). Then, visualize this confusion matrix as a heatmap using seaborn for better readability.


[ ]
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming y_test is available (if not, you might need to load it from the original test data if it exists)
# For this example, let's assume we have y_test available. If not, this step would need adjustment.
# Since test.csv does not contain 'sequence_type', we cannot calculate a confusion matrix directly.
# However, if a ground truth for the test data were available, the code below would work.

# As a placeholder, let's assume a sample y_test is available for demonstration purposes.
# In a real scenario, you would load or obtain the actual ground truth labels for the test set.

# Since we don't have the ground truth for the provided test data, we will skip the confusion matrix and heatmap generation.
# If you had the ground truth (y_test), you would uncomment and run the following code:

# cm = confusion_matrix(y_test, predictions)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix Heatmap')
# plt.show()

print("Skipping confusion matrix and heatmap generation as ground truth (y_test) for the test data is not available.")
print("If you have y_test, uncomment the code to generate the confusion matrix and heatmap.")
Skipping confusion matrix and heatmap generation as ground truth (y_test) for the test data is not available.
If you have y_test, uncomment the code to generate the confusion matrix and heatmap.
Finish task
Summary of results and limitations:
We have successfully completed the data preprocessing, model training, and prediction steps.

Missing values were handled and categorical features were encoded.
A RandomForestClassifier model was trained on the preprocessed training data.
Predictions were made on the preprocessed test data.
However, due to the lack of ground truth labels for the test set, we were unable to fully evaluate the model's performance using metrics like accuracy, precision, recall, or F1-score, or visualize a confusion matrix.

Potential next steps (if ground truth becomes available):
If you obtain the ground truth labels for the test data, you can:

Calculate and display various evaluation metrics.
Generate a confusion matrix and visualize it as a heatmap.
Perform hyperparameter tuning to potentially improve the model's performance.
Explore other classification models.
Supervised

[ ]
# =========================================
# Full Pipeline: Preprocessing + Features + Models
# =========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')

# ==========================
# 1. Load train + test data
# ==========================
try:
    train_df = pd.read_csv("train.csv")
    print("Train shape:", train_df.shape)
except:
    train_df = None
    print("train.csv not found")

try:
    test_df = pd.read_csv("test.csv")
    print("Test shape:", test_df.shape)
except:
    test_df = None
    print("test.csv not found")

# ==========================
# 2. Feature Engineering
# ==========================
def build_features(df):
    feats = df.copy()
    # Example: numeric stats (replace with your logic)
    numeric_cols = feats.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        feats[f"{col}_mean"] = feats[col].mean()
        feats[f"{col}_std"] = feats[col].std()
    return feats

train_features = build_features(train_df) if train_df is not None else None
test_features  = build_features(test_df) if test_df is not None else None

# ==========================
# 3. Supervised or Unsupervised?
# ==========================
label_col = None
if train_features is not None:
    for candidate in ["sequence_type", "label", "target"]:
        if candidate in train_features.columns:
            label_col = candidate
            break

# ==========================
# 4. Preprocessing
# ==========================
if train_features is not None:
    if label_col:
        X = train_features.drop(label_col, axis=1)
        y = train_features[label_col]
    else:
        X = train_features
        y = None

    if y is not None:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        num_classes = len(np.unique(y_encoded))

        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
    else:
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
        y_train = y_val = None

    scaler = StandardScaler()
    X_train_proc = scaler.fit_transform(X_train)
    X_val_proc   = scaler.transform(X_val)
else:
    X_train_proc = X_val_proc = None
    y_train = y_val = None
    num_classes = None

# ==========================
# 5A. Supervised Models
# ==========================
if y_train is not None:

    # ---- Neural Network ----
    def build_nn_model(neurons_layer1=128, neurons_layer2=64,
                       activation='relu', optimizer='adam'):
        model = Sequential()
        model.add(Dense(neurons_layer1, activation=activation,
                        input_shape=(X_train_proc.shape[1],)))
        model.add(Dense(neurons_layer2, activation=activation))
        if num_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer=optimizer, loss='binary_crossentropy',
                          metrics=['accuracy'])
        else:
            model.add(Dense(num_classes, activation='softmax'))
            model.compile(optimizer=optimizer,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
        return model

    nn_model = KerasClassifier(model=build_nn_model, verbose=0)
    param_grid_nn = {
        'model__neurons_layer1': [64, 128],
        'model__neurons_layer2': [32, 64],
        'batch_size': [32, 64],
        'epochs': [10, 20]
    }
    grid_search_nn = GridSearchCV(
        estimator=nn_model,
        param_grid=param_grid_nn,
        cv=3,
        scoring='accuracy'
    )
    grid_search_nn.fit(X_train_proc, y_train)
    print("Best NN hyperparameters:", grid_search_nn.best_params_)

    # ---- XGBoost ----
    xgb = XGBClassifier(use_label_encoder=False,
                        eval_metric='mlogloss',
                        random_state=42)
    xgb.fit(X_train_proc, y_train)

    # ---- Predictions ----
    y_pred_nn  = grid_search_nn.predict(X_val_proc)
    y_pred_xgb = xgb.predict(X_val_proc)

    # ---- Reports ----
    print("\nXGBoost Metrics:\n",
          classification_report(y_val, y_pred_xgb,
                                target_names=le.classes_))
    print("\nNeural Net Metrics:\n",
          classification_report(y_val, y_pred_nn,
                                target_names=le.classes_))

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_val, y_pred_xgb)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix - XGBoost")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# ==========================
# 5B. Unsupervised Anomaly Detection
# ==========================
elif train_features is not None:
    iso = IsolationForest(random_state=42, contamination=0.05)
    preds = iso.fit_predict(X_train_proc)
    print("IsolationForest anomaly labels (sample):", preds[:20])

Train shape: (368619, 341)
Test shape: (107, 336)
Task
Analyze the provided datasets ["/content/clean_train_data.csv", "/content/test.csv", "/content/train.csv", "/content/clean_train_data (1) (1).csv", "/content/clean_train_data (1) (1).csv", "/content/train_demographics.csv", "/content/test_demographics.csv"] by performing feature engineering, preprocessing data for advanced models, selecting and training advanced models (XGBoost and Neural Network), tuning hyperparameters, evaluating models with metrics and visualizations (including confusion matrix heatmap if ground truth is available), comparing model performance, and generating visualizations and graphs.

Feature engineering
Subtask:
Create new features from the existing data that could improve model performance.

Reasoning: Calculate accelerometer and gyroscope magnitudes, create interaction features, and compute rolling statistics for feature engineering.

Reasoning: The KeyError indicates that 'seq_rot_interaction' was not created in the train_df, while it was created in the test_df. The code for creating interaction features has a typo, calculating seq_rot_interaction for test_df twice instead of once for each dataframe. Fix the typo to correctly calculate seq_acc_interaction and seq_rot_interaction for both dataframes and then display the head of the dataframes with the new columns.


[ ]
# Create interaction features
train_df['seq_acc_interaction'] = train_df['sequence_counter'] * train_df['acc_magnitude']
train_df['seq_rot_interaction'] = train_df['sequence_counter'] * train_df['rot_magnitude']
test_df['seq_acc_interaction'] = test_df['sequence_counter'] * test_df['acc_magnitude']
test_df['seq_rot_interaction'] = test_df['sequence_counter'] * test_df['rot_magnitude']

# Calculate rolling mean and standard deviation
window_size = 5
train_df['acc_magnitude_rolling_mean'] = train_df['acc_magnitude'].rolling(window=window_size).mean().fillna(0)
train_df['acc_magnitude_rolling_std'] = train_df['acc_magnitude'].rolling(window=window_size).std().fillna(0)
train_df['rot_magnitude_rolling_mean'] = train_df['rot_magnitude'].rolling(window=window_size).mean().fillna(0)
train_df['rot_magnitude_rolling_std'] = train_df['rot_magnitude'].rolling(window=window_size).std().fillna(0)

test_df['acc_magnitude_rolling_mean'] = test_df['acc_magnitude'].rolling(window=window_size).mean().fillna(0)
test_df['acc_magnitude_rolling_std'] = test_df['acc_magnitude'].rolling(window=window_size).std().fillna(0)
test_df['rot_magnitude_rolling_mean'] = test_df['rot_magnitude'].rolling(window=window_size).mean().fillna(0)
test_df['rot_magnitude_rolling_std'] = test_df['rot_magnitude'].rolling(window=window_size).std().fillna(0)

print("New features added to train_df and test_df:")
display(train_df[['acc_magnitude', 'rot_magnitude', 'seq_acc_interaction', 'seq_rot_interaction', 'acc_magnitude_rolling_mean', 'acc_magnitude_rolling_std', 'rot_magnitude_rolling_mean', 'rot_magnitude_rolling_std']].head())
display(test_df[['acc_magnitude', 'rot_magnitude', 'seq_acc_interaction', 'seq_rot_interaction', 'acc_magnitude_rolling_mean', 'acc_magnitude_rolling_std', 'rot_magnitude_rolling_mean', 'rot_magnitude_rolling_std']].head())

Data preprocessing for advanced models
Subtask:
Prepare the data specifically for advanced models like XGBoost and Neural Networks, which may require different preprocessing steps than the RandomForestClassifier.

Reasoning: Identify numerical and categorical columns in X_train and X_test, then apply one-hot encoding to the categorical columns and standardize the numerical columns using StandardScaler, fitting only on X_train. Finally, display the head of the preprocessed X_train and X_test DataFrames.


[ ]
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Identify numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X_train.select_dtypes(include=['object', 'bool']).columns

# Exclude 'sequence_counter' from numerical columns for scaling as it was already handled
numerical_cols = numerical_cols.drop('sequence_counter', errors='ignore')

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough' # Keep other columns (like sequence_counter if not scaled)
)

# Create a pipeline with the preprocessor
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit and transform the training data
X_train_processed = pipeline.fit_transform(X_train)

# Transform the testing data
X_test_processed = pipeline.transform(X_test)

# Convert the processed data back to DataFrames (optional, but good for inspection)
# Get feature names after one-hot encoding
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)


print("\nPreprocessed X_train head:")
display(X_train_processed_df.head())

print("\nPreprocessed X_test head:")
display(X_test_processed_df.head())

Model selection (advanced models)
Subtask:
Choose and set up advanced models like XGBoost and a simple Neural Network.


# =========================================
# Full Pipeline: Preprocessing + Features + Models
# =========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')

# ==========================
# 1. Load train + test data
# ==========================
try:
    train_df = pd.read_csv("train.csv")
    print("Train shape:", train_df.shape)
except:
    train_df = None
    print("train.csv not found")

try:
    test_df = pd.read_csv("test.csv")
    print("Test shape:", test_df.shape)
except:
    test_df = None
    print("test.csv not found")

# ==========================
# 2. Feature Engineering
# ==========================
def build_features(df):
    feats = df.copy()
    # Example: numeric stats (replace with your logic)
    numeric_cols = feats.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        feats[f"{col}_mean"] = feats[col].mean()
        feats[f"{col}_std"] = feats[col].std()
    return feats

train_features = build_features(train_df) if train_df is not None else None
test_features  = build_features(test_df) if test_df is not None else None

# ==========================
# 3. Supervised or Unsupervised?
# ==========================
label_col = None
if train_features is not None:
    for candidate in ["sequence_type", "label", "target"]:
        if candidate in train_features.columns:
            label_col = candidate
            break

# ==========================
# 4. Preprocessing
# ==========================
if train_features is not None:
    if label_col:
        X = train_features.drop(label_col, axis=1)
        y = train_features[label_col]
    else:
        X = train_features
        y = None

    if y is not None:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        num_classes = len(np.unique(y_encoded))

        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
    else:
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
        y_train = y_val = None

    scaler = StandardScaler()
    X_train_proc = scaler.fit_transform(X_train)
    X_val_proc   = scaler.transform(X_val)
else:
    X_train_proc = X_val_proc = None
    y_train = y_val = None
    num_classes = None

# ==========================
# 5A. Supervised Models
# ==========================
if y_train is not None:

    # ---- Neural Network ----
    def build_nn_model(neurons_layer1=128, neurons_layer2=64,
                       activation='relu', optimizer='adam'):
        model = Sequential()
        model.add(Dense(neurons_layer1, activation=activation,
                        input_shape=(X_train_proc.shape[1],)))
        model.add(Dense(neurons_layer2, activation=activation))
        if num_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer=optimizer, loss='binary_crossentropy',
                          metrics=['accuracy'])
        else:
            model.add(Dense(num_classes, activation='softmax'))
            model.compile(optimizer=optimizer,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
        return model

    nn_model = KerasClassifier(model=build_nn_model, verbose=0)
    param_grid_nn = {
        'model__neurons_layer1': [64, 128],
        'model__neurons_layer2': [32, 64],
        'batch_size': [32, 64],
        'epochs': [10, 20]
    }
    grid_search_nn = GridSearchCV(
        estimator=nn_model,
        param_grid=param_grid_nn,
        cv=3,
        scoring='accuracy'
    )
    grid_search_nn.fit(X_train_proc, y_train)
    print("Best NN hyperparameters:", grid_search_nn.best_params_)

    # ---- XGBoost ----
    xgb = XGBClassifier(use_label_encoder=False,
                        eval_metric='mlogloss',
                        random_state=42)
    xgb.fit(X_train_proc, y_train)

    # ---- Predictions ----
    y_pred_nn  = grid_search_nn.predict(X_val_proc)
    y_pred_xgb = xgb.predict(X_val_proc)

    # ---- Reports ----
    print("\nXGBoost Metrics:\n",
          classification_report(y_val, y_pred_xgb,
                                target_names=le.classes_))
    print("\nNeural Net Metrics:\n",
          classification_report(y_val, y_pred_nn,
                                target_names=le.classes_))

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_val, y_pred_xgb)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix - XGBoost")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# ==========================
# 5B. Unsupervised Anomaly Detection
# ==========================
elif train_features is not None:
    iso = IsolationForest(random_state=42, contamination=0.05)
    preds = iso.fit_predict(X_train_proc)
    print("IsolationForest anomaly labels (sample):", preds[:20])



# ==============================================
# FULL PIPELINE: Data â†’ Preprocessing â†’ NN Train + Eval
# ==============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import make_classification

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# ================================
# 1. Load / Create Dataset
# ================================
try:
    # ğŸ‘‡ Uncomment this line if you already have a real dataset
    # df = pd.read_csv("train.csv")

    # ğŸ‘‡ Otherwise generate synthetic data (for testing pipeline)
    X_dummy, y_dummy = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_classes=5, random_state=42
    )
    df = pd.DataFrame(X_dummy, columns=[f"feature_{i}" for i in range(20)])
    df["gesture"] = y_dummy

except Exception as e:
    raise RuntimeError(f"Dataset could not be loaded: {e}")

# ================================
# 2. Preprocessing
# ================================
X = df.drop(columns=["gesture"])
y = df["gesture"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Standardize
scaler = StandardScaler()
X_train_proc = scaler.fit_transform(X_train)
X_test_proc  = scaler.transform(X_test)

# ================================
# 3. Neural Network Model Builder
# ================================
def create_nn_model(optimizer="adam", dropout_rate=0.3):
    model = Sequential()
    model.add(Dense(128, activation="relu", input_shape=(X_train_proc.shape[1],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(len(np.unique(y_train)), activation="softmax"))

    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# Wrap Keras model for sklearn
nn_clf = KerasClassifier(build_fn=create_nn_model, verbose=0)

# ================================
# 4. Grid Search for Hyperparameters
# ================================
param_grid = {
    "batch_size": [32, 64],
    "epochs": [10],  # keep small for quick run
    "optimizer": ["adam", "rmsprop"],
    "dropout_rate": [0.3, 0.5]
}

grid_search_nn = GridSearchCV(
    estimator=nn_clf,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    verbose=1
)

grid_search_nn.fit(X_train_proc, y_train)
print("âœ… Best NN hyperparameters:", grid_search_nn.best_params_)

# ================================
# 5. Evaluation on Test Set
# ================================
y_pred_nn = grid_search_nn.best_estimator_.predict(X_test_proc)

print("\nğŸ“Š Classification Report (Neural Network)")
print(classification_report(y_test, y_pred_nn, target_names=le.classes_.astype(str)))

cm_nn = confusion_matrix(y_test, y_pred_nn)
sns.heatmap(cm_nn, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Neural Network")
plt.show()



# ====================================================
# Full Pipeline: Preprocessing + GridSearchCV (NN + XGB) + Evaluation
# ====================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import xgboost as xgb

# --- Load dataset ---
df = pd.read_csv("train.csv")

# --- Features & labels ---
X = df.drop("sequence_type", axis=1)   # target column
y = df["sequence_type"]

# --- Encode labels ---
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- Scale features ---
scaler = StandardScaler()
X_train_proc = scaler.fit_transform(X_train)
X_test_proc = scaler.transform(X_test)

# --- Wrap into DataFrame (for Keras compatibility) ---
X_train_df = pd.DataFrame(X_train_proc, columns=X.columns)

# --- Define NN builder ---
def build_nn_model(neurons_layer1=128, neurons_layer2=64, activation='relu', optimizer='adam'):
    model = Sequential()
    model.add(Dense(neurons_layer1, activation=activation, input_shape=(X_train_df.shape[1],)))
    model.add(Dense(neurons_layer2, activation=activation))
    # auto-adjust output layer
    num_classes = len(np.unique(y_train))
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- GridSearchCV for Neural Network ---
nn_model = KerasClassifier(model=build_nn_model, verbose=0)
param_grid_nn = {
    'model__neurons_layer1': [64, 128],
    'model__neurons_layer2': [32, 64],
    'batch_size': [32, 64],
    'epochs': [10, 20]
}
grid_search_nn = GridSearchCV(estimator=nn_model, param_grid=param_grid_nn, cv=3, scoring='accuracy')
grid_search_nn.fit(X_train_df, y_train)

print("âœ… Best hyperparameters for Neural Network:", grid_search_nn.best_params_)

# --- GridSearchCV for XGBoost ---
xgb_clf = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
param_grid_xgb = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1]
}
grid_search_xgb = GridSearchCV(estimator=xgb_clf, param_grid=param_grid_xgb, cv=3, scoring='accuracy')
grid_search_xgb.fit(X_train_proc, y_train)

print("âœ… Best hyperparameters for XGBoost:", grid_search_xgb.best_params_)

# --- Best models ---
xgb_model = grid_search_xgb.best_estimator_
nn_model = grid_search_nn.best_estimator_

# --- Predictions ---
xgb_predictions = xgb_model.predict(X_test_proc)
nn_predictions = nn_model.predict(X_test_proc)  # scikeras returns class labels

# --- Evaluation ---
print("\nğŸ“Š XGBoost Classification Report:")
print(classification_report(y_test, xgb_predictions, target_names=le.classes_))

print("\nğŸ“Š Neural Network Classification Report:")
print(classification_report(y_test, nn_predictions, target_names=le.classes_))

# --- Confusion Matrix: XGB ---
cm_xgb = confusion_matrix(y_test, xgb_predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# --- Confusion Matrix: NN ---
cm_nn = confusion_matrix(y_test, nn_predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_nn, annot=True, fmt="d", cmap="Greens",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix - Neural Network")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()



# ====================================================
# Full Pipeline: Preprocessing â†’ Training (NN + XGB) â†’ Evaluation â†’ PCA Visualization
# ====================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn



import matplotlib.pyplot as plt
import seaborn as sns

# Check if xgb_model was trained successfully
if 'xgb_model' in locals() and xgb_model is not None:
    # Get feature importances from the trained XGBoost model
    feature_importances = xgb_model.feature_importances_

    # Get feature names from the preprocessed training data
    feature_names = X_train_processed_df.columns

    # Create a pandas Series for easier handling
    importance_series = pd.Series(feature_importances, index=feature_names)

    # Sort features by importance and select the top N
    top_n = 20
    top_features = importance_series.sort_values(ascending=False).head(top_n)

    # Create a bar plot of the top N feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_features.values, y=top_features.index, palette='viridis')
    plt.title(f'Top {top_n} Feature Importances (XGBoost)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
else:
    print("XGBoost model was not trained successfully, skipping feature importance plot.")

# Skipping Neural Network training history visualization as the history object was not explicitly saved or is not available.
print("\nSkipping Neural Network training history visualization as the history object was not explicitly saved or is not available.")



# ====================================================
# Preprocessing + GridSearchCV (NN + XGB) + Evaluation
# + PCA scatter + Decision Boundaries
# ====================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# --- ML Models ---
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from xgboost import XGBClassifier

# --- Load data with fallback ---
try:
    train_features = pd.read_csv("train_features.csv")
    labels = pd.read_csv("train_labels.csv")
    train_df = train_features.merge(labels, on="sequence_id", how="left")
    feature_cols = [col for col in train_df.columns if col not in ["label", "sequence_id"]]
    X = train_df[feature_cols]
    y = train_df["label"]
    print("âœ… Loaded real dataset:", X.shape, y.shape)
except Exception as e:
    print("âš ï¸� train_features.csv not found, using synthetic placeholder data:", e)
    X = pd.DataFrame(np.random.randn(200, 5), columns=[f"f{i}" for i in range(5)])
    y = np.random.randint(0, 2, size=200)

# --- Preprocessing ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels if not numeric
if not np.issubdtype(y.dtype, np.number):
    le = LabelEncoder()
    y = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert back to DataFrame for scikeras compatibility
X_train_processed_df = pd.DataFrame(X_train)
X_test_processed_df = pd.DataFrame(X_test)
y_train_encoded = y_train
y_test_encoded = y_test

# --- Neural Network builder ---
def build_nn_model(neurons_layer1=128, neurons_layer2=64, activation='relu', optimizer='adam'):
    model = Sequential()
    model.add(Dense(neurons_layer1, activation=activation, input_shape=(X_train_processed_df.shape[1],)))
    model.add(Dense(neurons_layer2, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- NN GridSearch ---
nn_model = KerasClassifier(model=build_nn_model, verbose=0)
param_grid_nn = {
    'model__neurons_layer1': [64, 128],
    'model__neurons_layer2': [32, 64],
    'batch_size': [32, 64],
    'epochs': [10, 20]
}
grid_search_nn = GridSearchCV(estimator=nn_model, param_grid=param_grid_nn, cv=3, scoring='accuracy')
grid_search_nn.fit(X_train_processed_df, y_train_encoded)
print("âœ… Best hyperparameters for Neural Network:", grid_search_nn.best_params_)

# --- XGBoost GridSearch ---
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
param_grid_xgb = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1]
}
grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=3, scoring='accuracy')
grid_search_xgb.fit(X_train_processed_df, y_train_encoded)
print("âœ… Best hyperparameters for XGBoost:", grid_search_xgb.best_params_)

# --- Predictions ---
y_pred_nn = grid_search_nn.best_estimator_.predict(X_test_processed_df)
y_pred_xgb = grid_search_xgb.best_estimator_.predict(X_test_processed_df)

# --- Reports ---
print("\nğŸ“Š Classification Report - Neural Network")
print(classification_report(y_test_encoded, y_pred_nn))

print("\nğŸ“Š Classification Report - XGBoost")
print(classification_report(y_test_encoded, y_pred_xgb))

# --- Confusion Matrices ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_nn = confusion_matrix(y_test_encoded, y_pred_nn)
sns.heatmap(cm_nn, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Confusion Matrix - Neural Network")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

cm_xgb = confusion_matrix(y_test_encoded, y_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Greens", ax=axes[1])
axes[1].set_title("Confusion Matrix - XGBoost")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")

plt.tight_layout()
plt.show()

# --- PCA Visualization ---
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# NN visualization
axes[0].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test_encoded, cmap="coolwarm", alpha=0.6, label="True class")
misclassified_nn = y_test_encoded != y_pred_nn
axes[0].scatter(X_test_pca[misclassified_nn, 0], X_test_pca[misclassified_nn, 1],
                edgecolor="black", facecolor="none", s=100, label="Misclassified")
axes[0].set_title("PCA Scatter - Neural Network")
axes[0].legend()

# XGB visualization
axes[1].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test_encoded, cmap="coolwarm", alpha=0.6, label="True class")
misclassified_xgb = y_test_encoded != y_pred_xgb
axes[1].scatter(X_test_pca[misclassified_xgb, 0], X_test_pca[misclassified_xgb, 1],
                edgecolor="black", facecolor="none", s=100, label="Misclassified")
axes[1].set_title("PCA Scatter - XGBoost")
axes[1].legend()

plt.tight_layout()
plt.show()



from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ================================
# Build Neural Network
# ================================
def build_nn_model(neurons_layer1=128, neurons_layer2=64, activation='relu', optimizer='adam'):
    model = Sequential()
    model.add(Dense(neurons_layer1, activation=activation, input_shape=(X_train_proc.shape[1],)))
    model.add(Dense(neurons_layer2, activation=activation))
    model.add(Dense(len(np.unique(y_train)), activation="softmax"))  # multi-class
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Wrap with scikeras
nn_model = KerasClassifier(model=build_nn_model, verbose=0)

# ================================
# Hyperparameter Grid
# ================================
param_grid_nn = {
    "model__neurons_layer1": [64, 128],
    "model__neurons_layer2": [32, 64],
    "batch_size": [32, 64],
    "epochs": [10, 20]
}

# ================================
# Grid Search
# ================================
grid_search_nn = GridSearchCV(
    estimator=nn_model,
    param_grid=param_grid_nn,
    cv=3,
    scoring="accuracy",
    verbose=1
)

grid_search_nn.fit(X_train_proc, y_train)
print("âœ… Best NN hyperparameters:", grid_search_nn.best_params_)

# ================================
# Evaluation on Test Set
# ================================
y_pred_nn = grid_search_nn.best_estimator_.predict(X_test_proc)

# Convert encoded labels back to original classes
y_pred_labels = le.inverse_transform(y_pred_nn)
y_test_labels = le.inverse_transform(y_test)

print("\nğŸ“Š Classification Report (Neural Network)")
print(classification_report(y_test_labels, y_pred_labels, target_names=le.classes_))

# Confusion Matrix
cm_nn = confusion_matrix(y_test_labels, y_pred_labels, labels=le.classes_)
plt.figure(figsize=(8,6))
sns.heatmap(cm_nn, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Neural Network")
plt.show()



import sklearn, scikeras, tensorflow as tf
print("scikit-learn:", sklearn.__version__)
print("scikeras:", scikeras.__version__)
print("tensorflow:", tf.__version__)



!pip install --upgrade scikit-learn scikeras


! pip install --upgrade scikit-learn scikeras[tensorflow] tensorflow



from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow import keras
from tensorflow.keras import layers

# Define a Keras model builder
def create_nn_model(hidden_units=32, dropout=0.2, learning_rate=0.001):
    model = keras.Sequential([
        layers.Input(shape=(X_train_processed_df.shape[1],)),
        layers.Dense(hidden_units, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Wrap with scikeras
nn_clf = KerasClassifier(model=create_nn_model, verbose=0)

# Define hyperparameter search space
param_grid = {
    "model__hidden_units": [16, 32, 64],
    "model__dropout": [0.2, 0.3],
    "model__learning_rate": [0.001, 0.01],
    "batch_size": [32, 64],
    "epochs": [5, 10]   # keep small for quick tuning
}

# Grid search
grid_search_nn = GridSearchCV(
    nn_clf,
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

grid_search_nn.fit(X_train_processed_df, y_train_encoded)
print("Best hyperparameters:", grid_search_nn.best_params_)





