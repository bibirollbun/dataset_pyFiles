import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc, time

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')


#--- Configurations
SEED = 42
FOLDS = 5

def seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


# --- Load dataset
train_df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_df  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
orig = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')
print("train", train_df.shape, "test", test_df.shape)


cols = [col for col in train_df.columns if col not in ['id', 'diagnosed_diabetes']]
new_cols = []

for col in cols:
    # mean
    mean_map = orig.groupby(col)['diagnosed_diabetes'].mean()
    new_mean_col_name = f"orig_mean_{col}"
    mean_map.name = new_mean_col_name

    train_df = train_df.merge(mean_map, on=col, how='left')
    test_df = test_df.merge(mean_map, on=col, how='left')
    new_cols.append(new_mean_col_name)

    # count
    new_cnt_col_name = f"orig_cnt_{col}"
    cnt_map = orig.groupby(col).size().reset_index(name=new_cnt_col_name)

    train_df = train_df.merge(cnt_map, on=col, how='left')
    test_df = test_df.merge(cnt_map, on=col, how='left')
    new_cols.append(new_cnt_col_name)

for col in new_cols:
    if 'mean' in col:
        train_df[col] = train_df[col].fillna(orig['diagnosed_diabetes'].mean())
        test_df[col] = test_df[col].fillna(orig['diagnosed_diabetes'].mean())
    else:
        train_df[col] = train_df[col].fillna(0)
        test_df[col] = test_df[col].fillna(0)


# #--- handling categorical features
# from sklearn.preprocessing import OneHotEncoder

# cols = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']
# encoder = OneHotEncoder(sparse=False, drop=None, handle_unknown='ignore')

# # train data
# encoded_train = encoder.fit_transform(train_df[cols])
# encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(cols))
# train_df = pd.concat([train_df.drop(columns=cols), encoded_train_df], axis=1)

# # test data
# encoded_test = encoder.transform(test_df[cols])
# encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(cols))
# test_df = pd.concat([test_df.drop(columns=cols), encoded_test_df], axis=1)    


X = train_df.drop(columns=['id', 'diagnosed_diabetes'])
y = train_df['diagnosed_diabetes']


# CatBoost specific: specify categorical features by index
cols = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']
cat_features_indices = [X.columns.get_loc(col) for col in cols]


roc_scores = []
models = []

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"========= \nFold {fold+1}/{FOLDS}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Initialize the CatBoost model
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3,
        cat_features=cat_features_indices,
        eval_metric='AUC',
        random_seed=SEED,
        verbose=100,
        early_stopping_rounds=100
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)
    y_pred_val = model.predict_proba(X_val)[:, 1]
    
    # Calculate the score
    score = roc_auc_score(y_val, y_pred_val)
    print(f'Fold: {fold+1} AUC score: {np.mean(score):.5f}') 

    roc_scores.append(score)
    models.append(model)


print(f'\nAverage AUC Score : {np.mean(roc_scores):.5f}, +-: {np.std(roc_scores):.5f}')


test_id = test_df.id
X_test = test_df.drop(columns=["id"])
submit_score = []

for fold_, model in enumerate(models):
    # predict test data
    pred_ = model.predict_proba(X_test)[:, 1]
    submit_score.append(pred_)

# predict test data
pred = np.mean(submit_score, axis=0)


submission = pd.DataFrame({
    'id': test_df['id'],
    'diagnosed_diabetes': pred
})

# Save
submission.to_csv('submission.csv', index=False)


submission

