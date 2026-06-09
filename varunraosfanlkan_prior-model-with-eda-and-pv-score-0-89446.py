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


# Load packages
import pandas as pd, numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings, time
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC  # or LinearSVC




# Ignore warnings
warnings.filterwarnings("ignore")


# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
print("Train shape:", train.shape, "Test shape:", test.shape)



print("ğŸ”� First look at train data:")
display(train.head())




print("\nğŸ“� Train shape:", train.shape)
print("ğŸ“� Test shape:", test.shape)


print("\nğŸ§¼ Missing values in test:")
test.isnull().sum()


print("\nğŸ“Œ Train data types:")
train.dtypes


print("\nğŸ�¯ Target distribution (rainfall):")
print(train['rainfall'].value_counts(normalize=True).rename_axis('rainfall').reset_index(name='percentage'))


plt.figure(figsize=(6,4))
sns.countplot(x='rainfall', data=train, palette='coolwarm')
plt.title("Rainfall Target Distribution")
plt.xlabel("Rainfall (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.grid(axis='y')
plt.show()



# Add 'year_group'
train['year_group'] = train['id'] // 365


# Set feature columns
remove_cols = ['rainfall', 'id', 'bucket', 'year_group']
FEATURES = [col for col in train.columns if col not in remove_cols]
print(f"Using {len(FEATURES)} base features:", FEATURES)



# Plot distributions and rainfall relationship
for col in FEATURES:
    plt.figure(figsize=(12, 3))

    # Distribution comparison
    plt.subplot(1, 2, 1)
    sns.kdeplot(train[col], label='Train')
    sns.kdeplot(test[col], label='Test')
    plt.title(f"{col} Distribution")
    plt.legend()

    # Binned rainfall mean
    plt.subplot(1, 2, 2)
    train['bucket'], bin_edges = pd.cut(train[col], bins=10, labels=False, retbins=True)
    bucket_means = train.groupby('bucket')['rainfall'].mean()
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(bin_midpoints, bucket_means, marker='o')
    plt.title(f"Mean Rainfall vs. {col} (Binned)")
    plt.grid()

    plt.tight_layout()
    plt.show()


# Use a sample to speed up plotting if dataset is large
sampled = train.sample(frac=0.3, random_state=42)

for col in FEATURES:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=sampled, x=col, y='rainfall', alpha=0.3)
    plt.title(f'Rainfall vs {col}')
    plt.grid()
    plt.show()



plt.figure(figsize=(12, 10))
corr = train[FEATURES + ['rainfall']].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title("Feature Correlation Heatmap")
plt.show()



for col in FEATURES:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='rainfall', y=col, data=train, palette='coolwarm')
    plt.title(f'{col} by Rainfall')
    plt.grid()
    plt.tight_layout()
    plt.show()



rainfall_by_year = train.groupby('year_group')['rainfall'].mean().reset_index()

plt.figure(figsize=(8, 4))
sns.lineplot(data=rainfall_by_year, x='year_group', y='rainfall', marker='o')
plt.title("Average Rainfall by Year Group")
plt.xlabel("Year Group")
plt.ylabel("Avg Rainfall")
plt.grid()
plt.show()



# Interaction features
INTERACT = []
for i in range(len(FEATURES)):
    for j in range(i + 1, len(FEATURES)):
        col1, col2 = FEATURES[i], FEATURES[j]
        name = f"{col1}_{col2}"
        train[name] = train[col1] * train[col2]
        test[name] = test[col1] * test[col2]
        INTERACT.append(name)

print(f"Created {len(INTERACT)} interaction features.")



# Forward feature selection
ADD = []
best_auc = 0
best_oof = np.zeros(len(train))
best_pred = np.zeros(len(test))

start_time = time.time()

for k, col in enumerate(['baseline'] + INTERACT):
    if col != 'baseline':
        ADD.append(col)

    oof = np.zeros(len(train))
    preds = np.zeros(len(test))
    kf = GroupKFold(n_splits=train.year_group.nunique())

    for fold, (tr_idx, val_idx) in enumerate(kf.split(train, groups=train.year_group)):
        X_tr = train.loc[tr_idx, FEATURES + ADD].copy()
        y_tr = train.loc[tr_idx, 'rainfall']
        X_val = train.loc[val_idx, FEATURES + ADD].copy()
        y_val = train.loc[val_idx, 'rainfall']
        X_test = test[FEATURES + ADD].copy()

        # Standardize
        for col_norm in FEATURES + ADD:
            mean, std = X_tr[col_norm].mean(), X_tr[col_norm].std()
            X_tr[col_norm] = (X_tr[col_norm] - mean) / std
            X_val[col_norm] = (X_val[col_norm] - mean) / std
            X_test[col_norm] = (X_test[col_norm] - mean) / std
            X_test[col_norm] = X_test[col_norm].fillna(0)

        # Train RAPIDS SVC
        # Use SVC with probability=True
        model = SVC(C=0.1, kernel='poly', degree=1, probability=True)
        model.fit(X_tr.values, y_tr.values)

        # Predict
        oof[val_idx] = model.predict_proba(X_val.values)[:, 1]
        preds += model.predict_proba(X_test.values)[:, 1]

    preds /= kf.get_n_splits()
    auc = roc_auc_score(train['rainfall'], oof)

    if auc > best_auc:
        print(f"[{k}] âœ… New Best AUC {auc:.5f} with {col}")
        best_auc = auc
        best_oof = oof.copy()
        best_pred = preds.copy()
    else:
        print(f"[{k}] â�Œ Worse AUC {auc:.5f} with {col}")
        if col in ADD:
            ADD.remove(col)

print(f"\nâœ… Final CV AUC = {best_auc:.5f}")
print(f"Selected {len(ADD)} interaction features.")



# Save submission
sample['rainfall'] = best_pred
sample.to_csv("submission.csv", index=False)
print("Submission saved. Shape:", sample.shape)
sample.head()



# Optional: Timing
print(f"â�±ï¸� Total runtime: {time.time() - start_time:.2f} seconds")













