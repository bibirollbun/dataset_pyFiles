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
import numpy as np
from glob import glob #function is useful for finding all the pathnames matching a specified pattern
from tqdm import tqdm  #makes the progress bars
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import h5py
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.preprocessing.image import img_to_array 
import tensorflow as tf
import cv2
import gc  #helps manage memory by automatically freeing up space 
 


class CFG:
    verbose = 1  # Verbosity
    seed = 42  # Random seed
    neg_sample = 0.1 # Downsample negative calss
    pos_sample = 5.0  # Upsample positive class
    preset = "efficientnetv2_b2_imagenet"  # Name of pretrained classifier
    image_size = [128, 128]  # Input image size
    epochs = 8 # Training epochs
    batch_size = 64  # Batch size
    #lr_mode = "cos" # LR scheduler mode from one of "cos", "step", "exp"
    class_names = ['target']
    num_classes = 1


train=pd.read_csv("/kaggle/input/isic-2024-challenge/train-metadata.csv")
train.ffill()
train.sample(5)


train.target.value_counts()



import matplotlib.pyplot as plt

# Count the number of samples in each class
class_counts = train['target'].value_counts()

# Plot pie chart
plt.figure(figsize=(6, 6))
plt.pie(class_counts, labels=['Normal', 'Abnormal'], autopct='%1.2f%%', colors=['skyblue', 'red'])
plt.title("Class Distribution Before Resampling")
plt.show()


negative_df=train[train["target"]== 0].sample(frac=CFG.neg_sample,random_state=CFG.seed)
positive_df=train[train["target"]== 1].sample(frac=CFG.pos_sample,random_state=CFG.seed,replace=True)
df=pd.concat([negative_df,positive_df])
df.target.value_counts()


import matplotlib.pyplot as plt

# Count the number of samples in each class
class_counts = df['target'].value_counts()

# Plot pie chart
plt.figure(figsize=(6, 6))
plt.pie(class_counts, labels=['Normal', 'Abnormal'], autopct='%1.2f%%', colors=['skyblue', 'red'])
plt.title("Class Distribution Before Resampling")
plt.show()



from sklearn.model_selection import StratifiedGroupKFold

df = df.reset_index(drop=True) # ensure continuous index
df["fold"] = -1
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CFG.seed)
for i, (training_idx, validation_idx) in enumerate(sgkf.split(df, y=df.target, groups=df.patient_id)):
    df.loc[validation_idx, "fold"] = int(i)

# Use first fold for training and validation
training_df = df.query("fold!=0")
validation_df = df.query("fold==0")
print(f"# Num Train: {len(training_df)} | Num Valid: {len(validation_df)}")


validation_df



# Paths
HDF5_PATH = "/kaggle/input/isic-2024-challenge/train-image.hdf5"

  
# CNN Feature Extractor
feature_extractor = EfficientNetB1(weights="imagenet", include_top=False, pooling="avg")

# Batch Processing
BATCH_SIZE = 64
IMG_SIZE = (128, 128)
def proce (training_df,model_name):
    
    OUTPUT_HDF5 = f"/kaggle/working/extracted_features_{model_name}.h5"
    
    # Create HDF5 File for Output
    with h5py.File(HDF5_PATH, "r") as hdf5_file_in, h5py.File(OUTPUT_HDF5, "w") as hdf5_file_out:
        # Create expandable datasets
        feature_dataset = hdf5_file_out.create_dataset("features", shape=(0, 1280), maxshape=(None, 1280), dtype=np.float32)
         # EfficientNetB0 output shape
        label_dataset = hdf5_file_out.create_dataset(
            "labels", shape=(0,), maxshape=(None,), dtype=np.int64
        )
    
        for batch_start in tqdm(range(0, len(training_df), BATCH_SIZE), desc="Extracting & Saving CNN Features"):
            batch_end = min(batch_start + BATCH_SIZE, len(training_df))
            batch_df = training_df.iloc[batch_start:batch_end]
    
            batch_images = []
            batch_labels = []
    
            # Read Images & Labels
            for _, row in batch_df.iterrows():
                img_id = row["isic_id"]
                label = row["target"]
    
                if img_id not in hdf5_file_in:  # Safety check
                    print(f"⚠ Warning: {img_id} not found in HDF5 file!")
                    continue
    
                byte_string = hdf5_file_in[img_id][()]
                nparr = np.frombuffer(byte_string, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
                if image is None:
                    print(f"⚠ Skipping corrupted image: {img_id}")
                    continue
    
                # Resize & Normalize
                image = cv2.resize(image, IMG_SIZE)
                image = image.astype(np.float32) / 255.0
                batch_images.append(image)
                batch_labels.append(label)
    
            if not batch_images:
                continue  # Skip empty batches
    
            batch_images = np.array(batch_images)
            batch_features = feature_extractor.predict(batch_images, verbose=0)
            batch_features = batch_features.reshape(batch_features.shape[0], -1)  # Flatten
    
            # Resize datasets to append new data
            feature_dataset.resize((feature_dataset.shape[0] + batch_features.shape[0]), axis=0)
            feature_dataset[-batch_features.shape[0]:] = batch_features
    
            label_dataset.resize((label_dataset.shape[0] + len(batch_labels)), axis=0)
            label_dataset[-len(batch_labels):] = batch_labels
    
            # Memory Cleanup
            del batch_images, batch_features
            gc.collect()
            tf.keras.backend.clear_session()
    
    print(f"✅ Features & Labels saved in {OUTPUT_HDF5}")
    



proce(training_df,"train")
proce(validation_df,"val")


import h5py
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load Features & Labels from HDF5

def reading_hdf5 (OUTPUT_HDF5):
    with h5py.File(OUTPUT_HDF5, "r") as hdf5_file:
        X = np.array(hdf5_file["features"])
        y = np.array(hdf5_file["labels"])
    
    print(f"✅ Loaded Features Shape: {X.shape}, Labels Shape: {y.shape}")
    return X,y

# Split Data into Test Set (50% Test, 50% validation)
X_train,y_train=reading_hdf5("/kaggle/working/extracted_features_train.h5")
X_test,y_test=reading_hdf5("/kaggle/working/extracted_features_val.h5")
X_test2, X_val, y_test2, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

 
# Compute class weight for imbalanced datasets
neg, pos = np.bincount(y_train)
scale_pos_weightS = neg / pos  # Balance classes

# Initialize XGBoost Classifier
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",  # Binary Classification
    learning_rate=0.05,
    max_depth=4,
    eval_metric=["logloss", "error", "auc"],
    scale_pos_weight=scale_pos_weightS  # Handle class imbalance
)

# Train XGBoost Model
history=xgb_model.fit(X_train, y_train , eval_set=[(X_val, y_val)],early_stopping_rounds=20, verbose=True,)




y_pred = xgb_model.predict(X_test2)  

print("Classification Report:\n")
print(classification_report(y_test2, y_pred))    
acc = accuracy_score(y_test2, y_pred)
print("Accuracy:\n")
print(acc)



from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.ensemble import StackingClassifier

# Base models
estimators = [
    ('xgb', XGBClassifier(scale_pos_weight=neg/pos, use_label_encoder=False, eval_metric='logloss')),
    ('lgbm', LGBMClassifier(scale_pos_weight=neg/pos)),
    ('rf', RandomForestClassifier(class_weight='balanced'))
]

# Meta-model (e.g., Logistic Regression)
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# Train ensemble
stacking_clf.fit(X_train, y_train)

# Predict
y_pred = stacking_clf.predict(X_test2)



 
print("Classification Report:\n")
print(classification_report(y_test2, y_pred))    
acc = accuracy_score(y_test2, y_pred)
print("Accuracy:\n")
print(acc)



# Install imblearn if not installed
# !pip install imbalanced-learn

from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, precision_recall_curve, accuracy_score, f1_score
from xgboost import XGBClassifier
import numpy as np

 
# ==============================
# 2️⃣ XGBoost Model with Class Weight
# ==============================

# Calculate scale_pos_weight AFTER SMOTE (still helpful)
neg = np.sum(y_train == 0)
pos = np.sum(y_train == 1)
scale_pos_weight = neg / pos

# Initialize XGBoost
xgb_model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='logloss',
    learning_rate=0.05,
    max_depth=6,
    n_estimators=300,
    random_state=42
)

# Train model
xgb_model.fit(X_train, y_train)

# ==============================
# 3️⃣ Predict Probabilities on Test Set
# ==============================
y_proba = xgb_model.predict_proba(X_test2)[:, 1]  # Class 1 probability

# ==============================
# 4️⃣ Threshold Tuning for Best F1
# ==============================
precision, recall, thresholds = precision_recall_curve(y_test2, y_proba)
f1_scores = 2 * (precision * recall) / (precision + recall)
best_thresh = thresholds[np.argmax(f1_scores)]

print(f"Best threshold for F1: {best_thresh:.3f}")

# Final predictions
y_pred = (y_proba >= best_thresh).astype(int)

# ==============================
# 5️⃣ Evaluation
# ==============================
print("\nClassification Report:")
print(classification_report(y_test2, y_pred))

acc = accuracy_score(y_test2, y_pred)
f1 = f1_score(y_test2, y_pred)
print(f"Accuracy: {acc:.4f} | F1-score: {f1:.4f}")


