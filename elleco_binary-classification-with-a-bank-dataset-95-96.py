lgb_params = {
    'max_depth': 6,
    'num_leaves': 64,
    'learning_rate': 0.05,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 30,
    'lambda_l1': 1e-3,
    'lambda_l2': 1.0,
    'min_gain_to_split': 0.1,
    'random_state': 42
}



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier

# Load data
df_train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

# Preprocessing
df_train.drop(columns=['id'], inplace=True)
df_test.drop(columns=['id'], inplace=True)

# Optional: encode target variable if not already binary
y = LabelEncoder().fit_transform(df_train['y'])  # 'yes'/'no' â†’ 1/0
X = df_train.drop(columns=['y'])

# Visual check
print(f"Training shape: {X.shape}")
print(f"Class distribution:\n{pd.Series(y).value_counts(normalize=True)}")

# Categorical visualization
def stacked_bar_plot(df, feature, target='y'):
    crosstab = pd.crosstab(df[feature], df[target], normalize='index')
    crosstab.plot(kind='bar', stacked=True, figsize=(10, 5), cmap='viridis')
    plt.title(f'{feature} vs {target}', fontsize=14)
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
stacked_bar_plot(df_train, 'loan')

# Correlation-focused histograms
X.hist(bins=20, figsize=(12, 10), color='cornflowerblue', edgecolor='black')
plt.tight_layout()
plt.show()

# One-hot encoding
X_encoded = pd.get_dummies(X, drop_first=True)
X_test_encoded = pd.get_dummies(df_test, drop_first=True)
X_encoded, X_test_encoded = X_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

# LightGBM params (modified)
lgb_params = {
    'max_depth': 6,
    'num_leaves': 64,
    'learning_rate': 0.05,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 30,
    'lambda_l1': 1e-3,
    'lambda_l2': 1.0,
    'min_gain_to_split': 0.1,
    'random_state': 42
}

# Cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

val_preds = np.zeros(len(X_encoded))
val_labels = np.zeros(len(X_encoded))
test_preds = np.zeros(len(X_test_encoded))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_encoded, y)):
    X_train, X_val = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model = LGBMClassifier(**lgb_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=[])    

    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test_encoded)[:, 1]
    
    val_preds[val_idx] = val_probs
    val_labels[val_idx] = y_val
    test_preds += test_probs / n_splits

# Evaluation
roc_auc = roc_auc_score(val_labels, val_preds)
print(f"Stratified K-Fold AUC: {roc_auc:.4f}")

# ROC Curve
fpr, tpr, _ = roc_curve(val_labels, val_preds)
roc_score = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_score:.4f}', color='teal', lw=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (LGBM)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

