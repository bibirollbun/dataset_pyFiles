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


from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression


data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

target = 'Fertilizer Name'
cat_columns = [x for x in data.columns if data[x].dtype == 'object' and x!=target]
num_columns = [x for x in data.columns if data[x].dtype != 'object']

print(cat_columns, num_columns, sep = '\n')


from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

label_enc = LabelEncoder()
ordinal_enc = OrdinalEncoder(handle_unknown='error')

data[cat_columns] = ordinal_enc.fit_transform(data[cat_columns])
test[cat_columns] = ordinal_enc.transform(test[cat_columns])
data[cat_columns] = data[cat_columns].astype('category')
test[cat_columns] = test[cat_columns].astype('category')
data['Fertilizer Name'] = label_enc.fit_transform(data['Fertilizer Name'])
data['const'] = 1
test['const'] = 1


def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


X = data.drop(target, axis = 1)
y = data[target]


import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

# --- Data Preparation ---
# Ensure 'id' and the target variable are not in your feature set.
target = 'Fertilizer Name'
features = [col for col in data.columns if col not in ['id', target]]
X = data[features]
y = data[target]
X_test = test[features]

# Initialize Stratified K-Fold for cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Model Configuration for GPU ---
# 1. LightGBM with GPU support
lgb_params = {
    'device': 'gpu',
    'random_state': 42
}
lgb_model = lgb.LGBMClassifier(**lgb_params)

# 2. XGBoost with GPU support
xgb_params = {
    'tree_method': 'gpu_hist',
    'eval_metric': 'mlogloss',
    'objective': 'multi:softprob',
    'enable_categorical': True, # <--- THIS IS THE FIX
    'random_state': 42
}
xgb_model = xgb.XGBClassifier(**xgb_params)

models = [('lgb', lgb_model), ('xgb', xgb_model)]

# --- Stacking Implementation ---
# Create empty arrays to store the out-of-fold (OOF) predictions.
num_classes = y.nunique()
oof_preds = np.zeros((len(data), num_classes * len(models)))
test_preds = np.zeros((len(test), num_classes * len(models)))

print("✅ Starting training for GPU-accelerated base models...")

# Loop through each model for training and prediction
for model_idx, (model_name, model) in enumerate(models):
    print(f"--- Training {model_name} on GPU ---")

    # Perform cross-validation to generate OOF predictions
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}")

        # Define training and validation sets for this fold
        x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        x_valid, _ = X.iloc[valid_idx], y.iloc[valid_idx]

        # Train the model
        model.fit(x_train, y_train)

        # Calculate the start and end column indices for storing predictions
        start_col = model_idx * num_classes
        end_col = start_col + num_classes

        # Predict probabilities on the validation set (OOF predictions)
        oof_preds[valid_idx, start_col:end_col] = model.predict_proba(x_valid)

        # Predict probabilities on the test set and average across folds
        test_preds[:, start_col:end_col] += model.predict_proba(X_test) / skf.get_n_splits()

print("\\n✅ Base model training complete!")

# The OOF predictions are the new features for the training set
X_meta = oof_preds
# The averaged test predictions are the new features for the test set
x_test = test_preds


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_stack = np.zeros(shape = (len(data), y.nunique()))
pred_prob_stack = np.zeros(shape= (len(test), y.nunique()))
print("\nStarting 5-Fold LR training...")
for i, (train_idx, valid_idx) in enumerate(skf.split(X_meta, y)):
    lr_model = LogisticRegression(**{
        'C': 1.436289965798556,
        'tol': 0.05692806752309682,
        'penalty': 'l2',
        'solver': 'newton-cholesky',
        'max_iter': 1001,'fit_intercept':True}
        
    )

    x_train,x_valid = X_meta[train_idx], X_meta[valid_idx]
    y_train,y_valid = y.iloc[train_idx],y.iloc[valid_idx]

    lr_model.fit(x_train, y_train)

    oof_stack[valid_idx] = lr_model.predict_proba(x_valid)
    pred_prob_stack +=lr_model.predict_proba(x_test) / skf.get_n_splits()
    actual = [[label] for label in y_valid]
    top_3_preds = np.argsort(oof_stack[valid_idx], axis=1)[:, -3:][:, ::-1]
    map3_score = mapk(actual, top_3_preds)
    print(f"✅ FOLD {i+1}: MAP@3  Score: {map3_score:.5f}")


actual = [[label] for label in y]

top_3_preds_1 = np.argsort(oof_stack, axis=1)[:, -3:][:, ::-1]  
map3_score_1 = mapk(actual, top_3_preds_1)
print(f'✅ Final  MAP@3 Score: {map3_score_1:.5f}')


top_3_preds = np.argsort(pred_prob_stack, axis=1)[:, -3:][:, ::-1]
top_3_labels = label_enc.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)
df_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
submission = pd.DataFrame({
    'id': df_sub['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv('submission.csv', index=False)
print("✅ Submission file saved as 'submission.csv'")


submission.head()

