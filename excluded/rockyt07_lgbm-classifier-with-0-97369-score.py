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


# ================================================================
# Kaggle Notebook: Binary Classification with Bank Dataset
# ================================================================

# ---------------------------
# Imports
# ---------------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

# ---------------------------
# Data Loading
# ---------------------------
train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Missing values (train):\n", train.isna().sum().sum())
print("Missing values (test):\n", test.isna().sum().sum())

# ---------------------------
# Feature Engineering
# ---------------------------
for df in [train, test]:
    df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1)
    df['duration_squared'] = df['duration'] ** 2
    df['duration_log'] = np.log1p(df['duration'])
    df['duration_sqrt'] = np.sqrt(df['duration'])

# ---------------------------
# Separate Features and Target
# ---------------------------
X = train.drop(['id', 'y'], axis=1)
y = train['y']

test_ids = test['id']
test = test.drop(['id'], axis=1)

# ---------------------------
# Encode Categorical Features
# ---------------------------
object_cols = X.select_dtypes(include="object").columns.tolist()
print("Categorical columns:", object_cols)

encoder = LabelEncoder()
for col in object_cols:
    X[col] = encoder.fit_transform(X[col])
    test[col] = encoder.transform(test[col])

# ---------------------------
# Standard Scaling
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
test_scaled_df = pd.DataFrame(test_scaled, columns=test.columns)

# ---------------------------
# PCA (variance explained)
# ---------------------------
target_variance = 0.99
pca = PCA(target_variance)
pca.fit(X_scaled)
print(f"Number of components to achieve {target_variance:.0%} variance: {pca.n_components_}")

plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title(f'PCA: {pca.n_components_} Components to Explain {target_variance:.0%} Variance')
plt.axhline(y=target_variance, color='r', linestyle='--')
plt.axvline(x=pca.n_components_, color='g', linestyle='--')
plt.grid(True)
plt.show()

# ---------------------------
# PCA Visualization (2D)
# ---------------------------
pca2d = PCA(n_components=2)
pc2d = pca2d.fit_transform(X_scaled)
finalDf2d = pd.DataFrame(pc2d, columns=['PC1', 'PC2'])
finalDf2d['target'] = y

fig, ax = plt.subplots(figsize=(10, 7))
for target, color in zip([0, 1], ['#008080', '#FF6F61']):
    indices = finalDf2d['target'] == target
    ax.scatter(finalDf2d.loc[indices, 'PC1'], finalDf2d.loc[indices, 'PC2'],
               c=color, s=20, label=f'Class {target}')
ax.set_title('PCA - 2D Projection')
ax.legend()
plt.show()

# ---------------------------
# PCA Visualization (3D)
# ---------------------------
pca3d = PCA(n_components=3)
pc3d = pca3d.fit_transform(X_scaled)
finalDf3d = pd.DataFrame(pc3d, columns=['PC1', 'PC2', 'PC3'])
finalDf3d['target'] = y

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
for target, color in zip([0, 1], ['#008080', '#FF6F61']):
    indices = finalDf3d['target'] == target
    ax.scatter(finalDf3d.loc[indices, 'PC1'],
               finalDf3d.loc[indices, 'PC2'],
               finalDf3d.loc[indices, 'PC3'],
               c=color, s=20, label=f'Class {target}')
ax.set_title('PCA - 3D Projection')
ax.legend()
plt.show()

# ---------------------------
# Mutual Information
# ---------------------------
mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values()

print("Mutual Information Scores:\n", mi_series)

mi_series.plot(kind='barh', figsize=(10, 6), color='#FF6F61')
plt.title('Mutual Information Scores')
plt.tight_layout()
plt.show()

# ---------------------------
# LightGBM Training with Stratified K-Fold
# ---------------------------
def train_lightgbm(train, test, target):
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    y_probs = np.zeros(len(test))
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train, target)):
        print(f"\n<== Training fold {fold+1}/{n_splits} ==>")
        
        X_train, y_train = train.iloc[train_idx], target.iloc[train_idx]
        X_val, y_val = train.iloc[val_idx], target.iloc[val_idx]
        
        model = lgb.LGBMClassifier(
            n_estimators=30000,
            class_weight='balanced',
            learning_rate=0.055,
            num_leaves=100,
            max_depth=10,
            min_child_samples=8,
            subsample=0.85,
            colsample_bytree=0.5,
            reg_alpha=0.8,
            reg_lambda=0.3,
            max_bin=4851,
            random_state=2003,
            verbosity=-1,
            boosting_type='gbdt',
            metric='auc'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(300), lgb.log_evaluation(500)]
        )
        
        models.append(model)
        y_probs += model.predict_proba(test)[:, 1] / n_splits
    
    print("\nLightGBM model training complete.")
    return y_probs, models

y_probs, models = train_lightgbm(X_scaled_df, test_scaled_df, y)

# ---------------------------
# Submission
# ---------------------------
submission = pd.DataFrame({
    'id': test_ids,
    'target': y_probs
})

submission.to_csv('submission.csv', index=False)
print("Submission saved -> submission.csv")


