import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from catboost import CatBoostClassifier, Pool

import warnings, os, gc
warnings.filterwarnings('ignore')

pd.set_option('display.max_colwidth', 150)


train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test_df  = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')

print(train_df.shape, test_df.shape)
train_df.head()


display(train_df.info())
print("\nMissing (%)")
display(train_df.isna().mean().sort_values(ascending=False).head(10))


# target distribution
sns.countplot(x='rule_violation', data=train_df)
plt.title('Target distribution'); plt.show()


# text length
train_df['text_len'] = train_df['body'].str.len()
sns.histplot(train_df['text_len'], bins=50, kde=True)
plt.title('Comment length'); plt.show()


N_FOLDS = 5
TEXT_COL = 'body'
TARGET   = 'rule_violation'

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

cat_params = {
    'loss_function': 'Logloss',
    'eval_metric'  : 'AUC',
    'iterations'   : 10000,
    'learning_rate': 0.03,
    'depth'        : 6,
    'l2_leaf_reg'  : 3,
    'random_seed'  : 42,
    'early_stopping_rounds': 300,
    'verbose'      : 200,
    'task_type'    : 'CPU',        
    'text_features': [TEXT_COL],   
}


oof        = np.zeros(len(train_df))
test_pred  = np.zeros(len(test_df))

for fold, (tr_idx, val_idx) in enumerate(cv.split(train_df, train_df[TARGET])):
    print(f"\n===== Fold {fold+1}/{N_FOLDS} =====")

    X_tr  = train_df.loc[tr_idx, [TEXT_COL]]
    X_val = train_df.loc[val_idx, [TEXT_COL]]
    y_tr  = train_df.loc[tr_idx, TARGET]
    y_val = train_df.loc[val_idx, TARGET]

    train_pool = Pool(X_tr,  label=y_tr, text_features=[TEXT_COL])
    val_pool   = Pool(X_val, label=y_val, text_features=[TEXT_COL])
    test_pool  = Pool(test_df[[TEXT_COL]], text_features=[TEXT_COL])

    model = CatBoostClassifier(**cat_params)
    model.fit(train_pool, eval_set=val_pool)

    oof[val_idx]  = model.predict_proba(val_pool)[:, 1]
    test_pred    += model.predict_proba(test_pool)[:, 1] / N_FOLDS

    print("Fold AUC:", roc_auc_score(y_val, oof[val_idx]))




# Aggregate CV metrics
print("\n===== Overall CV =====")
print("OOF AUC :", roc_auc_score(train_df[TARGET], oof))
print("OOF F1  :", f1_score(train_df[TARGET], (oof > 0.5).astype(int)))


sub = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')
sub['rule_violation'] =  test_pred
sub.to_csv('submission.csv', index=False)
print("submission.csv saved.")

