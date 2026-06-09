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

# Load data
train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

# Quick check
print(train.shape)
train.head()



# Check how many missing values each column has
missing_values = train.isnull().sum()

# Filter to show only columns with missing data (if any)
missing_values = missing_values[missing_values > 0]

if missing_values.empty:
    print("âœ… No missing values found.")
else:
    print("â�—Missing values detected:\n")
    print(missing_values)



import matplotlib.pyplot as plt

# Identify only feature columns
feature_cols = [col for col in train.columns if col.startswith('var_')]

# Plot histograms in batches (20 features per batch)
for i in range(0, len(feature_cols), 20):
    batch = feature_cols[i:i+20]
    train[batch].hist(
        bins=30, figsize=(20, 15), edgecolor='black'
    )
    plt.tight_layout()
    plt.suptitle(f'Distribution of Features: var_{i} to var_{i+19}', y=1.02)
    plt.show()



# Compute correlation of all features to target
correlations = train.corr(numeric_only=True)['target'].drop('target')

# Sort descending
correlations_sorted = correlations.sort_values(ascending=False)

# Show top positively correlated features
print(correlations_sorted.head(20))



selected_features = correlations_sorted[correlations_sorted.abs() > 0.01].index.tolist()

# Create correlation matrix for the selected features
feature_corr_matrix = train[selected_features].corr().abs()



import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
sns.heatmap(feature_corr_matrix, cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap of Top Features (by correlation to target)")
plt.show()



# Threshold to consider as multicollinear
threshold = 0.85

# Set to hold final selected features
final_features = []

for feature in selected_features:
    is_correlated = False
    for sel in final_features:
        if abs(train[feature].corr(train[sel])) > threshold:
            is_correlated = True
            break
    if not is_correlated:
        final_features.append(feature)

print(f"Selected features after multicollinearity check: {len(final_features)}")



import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc as calc_auc  # alias
import lightgbm as lgb
from lightgbm import early_stopping
import matplotlib.pyplot as plt

# Load dataset
train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

# Use only the selected features + target
selected_features = final_features  # your 161 features here
X = train[selected_features]
y = train['target']

# Set up cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nğŸ“‚ Fold {fold + 1}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # LightGBM dataset
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    # LightGBM parameters (tweak as needed)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 42
    }

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        callbacks=[early_stopping(stopping_rounds=100)]
    )

    # Predict and evaluate
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_val, y_pred)
    auc_scores.append(auc)

    print(f"âœ… Fold {fold + 1} AUC: {auc:.5f}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_val, y_pred)
    roc_auc_val = calc_auc(fpr, tpr)  # use alias
    plt.plot(fpr, tpr, lw=2, label=f'Fold {fold+1} (AUC = {roc_auc_val:.3f})')

# Plot chance line
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Fold')
plt.legend(loc='lower right')
plt.show()

# Final CV result
print(f"\nğŸ�¯ Mean AUC across folds: {np.mean(auc_scores):.5f}")



# Retrain on full training set
full_train = lgb.Dataset(X, y)

final_model = lgb.train(
    params,
    full_train,
    num_boost_round=100
)

# Load test data
test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
X_test = test[selected_features]

# Predict probabilities
test_preds = final_model.predict(X_test, num_iteration=final_model.best_iteration)

# Prepare submission
submission = pd.DataFrame({
    'ID_code': test['ID_code'],
    'target': test_preds
})

submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved as submission.csv")





