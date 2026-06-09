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


# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")



# Load Dataset
train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv", index_col="id")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv", index_col="id")

# Shape of the training dataset
print("\n===== Shape of the Training Data =====")
print(train_df.shape)

# Info about dataset (column types, non-null counts)
print("\n===== Info About Training Data =====")
print(train_df.info())

# Statistical summary of numerical features
print("\n===== Statistical Summary =====")
print(train_df.describe())

# Preview the first few rows
print("\n===== First 5 Rows of Training Data =====")
print(train_df.head())


num_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
categorical_cols = ['Soil Type', 'Crop Type']



sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# --- Numerical Columns ---
for col in num_cols:
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(train_df[col], kde=True, ax=ax[0], color='skyblue')
    ax[0].set_title(f'Distribution of {col}', fontsize=14)
    
    sns.boxplot(x=train_df[col], ax=ax[1], color='lightgreen')
    ax[1].set_title(f'Boxplot of {col}', fontsize=14)
    
    fig.suptitle(f'Univariate Analysis - {col}', fontsize=16)
    plt.tight_layout()
    plt.show()



# --- Categorical Columns ---
for col in categorical_cols:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=train_df, x=col, order=train_df[col].value_counts().index, palette='pastel')
    plt.title(f'Countplot of {col}', fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# --- Numerical Columns vs Target ---
for col in num_cols:
    if col != 'id':  # Skip ID
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=train_df, x='Fertilizer Name', y=col, palette='Set3')
        plt.title(f'{col} vs Fertilizer Name (Boxplot)', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()



# --- Categorical Columns vs Target ---
for col in categorical_cols:
    if col != 'Fertilizer Name':  # Skip target itself
        plt.figure(figsize=(10, 5))
        sns.countplot(data=train_df, x=col, hue='Fertilizer Name', palette='Set2')
        plt.title(f'{col} vs Fertilizer Name (Countplot)', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()



# --- Grouped Mean Statistics by Categorical Columns ---
# Average NPK by Crop Type
grouped = train_df.groupby('Crop Type')[['Nitrogen', 'Phosphorous', 'Potassium']].mean().reset_index()
grouped.plot(kind='bar', x='Crop Type', stacked=False, figsize=(12, 6), colormap='viridis')
plt.title("Average NPK Levels per Crop Type", fontsize=16)
plt.ylabel("Mean Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# --- Correlation Heatmap (Numerical Features) ---
plt.figure(figsize=(10, 8))
corr_matrix = train_df[num_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Correlation Heatmap of Numerical Features", fontsize=16)
plt.tight_layout()
plt.show()



# Label Encoding for Categorical Features
label_encoders = {col: LabelEncoder().fit(train_df[col]) for col in categorical_cols}
for col in categorical_cols:
    train_df[col] = label_encoders[col].transform(train_df[col])
    test_df[col] = label_encoders[col].transform(test_df[col])

# Encode Target
target_encoder = LabelEncoder()
train_df['Fertilizer Name'] = target_encoder.fit_transform(train_df['Fertilizer Name'])

# Feature Preparation
features = train_df.drop("Fertilizer Name", axis=1)
target = train_df["Fertilizer Name"]
test_features = test_df.copy()

# Scale Numerical Features with RobustScaler
scaler = RobustScaler()
features[num_cols] = scaler.fit_transform(features[num_cols])
test_features[num_cols] = scaler.transform(test_features[num_cols])


# MAP@3 Metric
def mean_average_precision_k(true, pred, k=3):
    def apk(a, p, k):
        p = p[:k]
        score, hits = 0.0, 0
        seen = set()
        for i, val in enumerate(p):
            if val in a and val not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(val)
        return score / min(len(a), k)
    return np.mean([apk([a], p, k) for a, p in zip(true, pred)])


# XGBoost Parameters
xgb_params = {
    'max_depth': 12,
    'colsample_bytree': 0.467,
    'subsample': 0.86,
    'n_estimators': 5000,
    'learning_rate': 0.03,
    'gamma': 0.26,
    'reg_alpha': 2.7,
    'reg_lambda': 1.4,
    'early_stopping_rounds': 100,
    'objective': 'multi:softprob',
    'enable_categorical': True,
    'use_label_encoder': False,
    'eval_metric': 'mlogloss',
    'tree_method': 'gpu_hist'
}

# Cross-Validation and Training
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_validation_probs = np.zeros((len(features), len(np.unique(target))))
test_prob_matrix = np.zeros((len(test_features), len(np.unique(target))))


# Training Loop
for fold_id, (train_idx, val_idx) in enumerate(folds.split(features, target)):
    print(f"\nFold {fold_id + 1}")
    
    X_tr, X_val = features.iloc[train_idx], features.iloc[val_idx]
    y_tr, y_val = target.iloc[train_idx], target.iloc[val_idx]

    model = XGBClassifier(**xgb_params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)

    cv_validation_probs[val_idx] = model.predict_proba(X_val)
    test_prob_matrix += model.predict_proba(test_features)

    val_top3 = np.argsort(cv_validation_probs[val_idx], axis=1)[:, -3:][:, ::-1]
    fold_score = mean_average_precision_k(y_val.tolist(), val_top3)
    print(f"✔️ Fold {fold_id + 1} MAP@3: {fold_score:.5f}")



# Overall Validation MAP@3
overall_top3 = np.argsort(cv_validation_probs, axis=1)[:, -3:][:, ::-1]
final_cv_score = mean_average_precision_k(target.tolist(), overall_top3)
print(f"\n Overall MAP@3: {final_cv_score:.5f}")


# Final Predictions
test_prob_matrix /= folds.n_splits
final_top3_preds = np.argsort(test_prob_matrix, axis=1)[:, -3:][:, ::-1]
final_labels = target_encoder.inverse_transform(np.arange(len(target_encoder.classes_)))
top3_label_names = [[final_labels[idx] for idx in row] for row in final_top3_preds]




#  Submission
submission_df = pd.DataFrame({
    "id": test_df.index,
    "Fertilizer Name": [' '.join(row) for row in top3_label_names]
})
submission_df.to_csv("submission.csv", index=False)
print("Submission saved as 'submission.csv'")


