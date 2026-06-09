%load_ext cudf.pandas


# %%
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# %%
import numpy as np
import pandas as pd


# %%
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv", index_col="id")

test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")


# Handle missing values for categorical features
cat_features = train.select_dtypes(include=['object', 'category']).columns.to_list()

# %%
train.head()

# %%
test.head()


# %%
from catboost import CatBoostClassifier
from catboost import Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# %%
for i, c1 in enumerate(cat_features[:-1]):
    for j, c2 in enumerate(cat_features[i+1:]):
        n = f"{c1}_{c2}"
        train[n] = train[c1].astype('str') + "_" + train[c2].astype('str')
        test[n] = test[c1].astype('str') + "_" + test[c2].astype('str')


# %%
train

# %%
y = train['loan_paid_back']
X = train.drop(columns=['loan_paid_back'])

# %%
cat_ftr = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(cat_ftr)

# Check for class imbalance
print(y.value_counts(normalize=True))


# %%
# Cross-validation setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f'\nFold {fold + 1}\n')
    
    X_train = X.iloc[trn_idx]
    y_train = y.iloc[trn_idx]
    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]
    
    train_pool = Pool(
        data=X_train, 
        label=y_train, 
        cat_features=cat_ftr
    )
    
    val_pool = Pool(
        data=X_val, 
        label=y_val, 
        cat_features=cat_ftr
    )
    
    model = CatBoostClassifier(
        iterations=10000,          
        #eval_metric='AUC',      
        auto_class_weights='Balanced',  
        learning_rate=0.03,     
        task_type="GPU",
        devices='0',
        allow_writing_files=False,
        cat_features=cat_ftr,
        #loss_function='Logloss'
    )
    
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=100,
        verbose=200
    )
    
    # OOF predictions
    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred
    fold_auc = roc_auc_score(y_val, val_pred)
    print(f'Fold {fold + 1} AUC: {fold_auc}')
    
    # Test predictions
    test_preds += model.predict_proba(test[X.columns])[:, 1] / n_splits

# Overall CV score
cv_auc = roc_auc_score(y, oof_preds)
print(f'\nCV AUC: {cv_auc}')


# %%
submit = pd.DataFrame({'id': test['id'],
                       'loan_paid_back': test_preds})

submit.to_csv('submission.csv', index=False)

