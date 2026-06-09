!pip install xgboost
!pip install catboost


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
import seaborn as sns
import gc
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from fastai.tabular.all import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fastai.tabular.model import TabularModel
from fastai.learner import Learner
from fastai.data.core import DataLoaders
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier, Pool
import torch




train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


train_df.head(10)


test_df.head()


train_df.info()
train_df.columns


check = ['Soil Type','Crop Type','Fertilizer Name']
for i in check:
    print(train_df[i].unique())


train_encoded = pd.get_dummies(train_df, columns=['Soil Type','Crop Type'], dtype='int64')
test_encoded = pd.get_dummies(test_df,columns =['Soil Type','Crop Type'],dtype='int64')
train_encoded



def corr(train_encoded):
    if 'Fertilizer Name' in train_encoded.columns:
        train_corr = train_encoded.drop(columns=['Fertilizer Name'])
    else:
        train_corr = train_encoded.copy()
    corr_matrix = train_corr.corr()
    
    plt.figure(figsize=(60, 40))
    sns.heatmap(corr_matrix, annot=True, cmap="Blues", linewidths=0.5, fmt=".2f")
    
    plt.title("Blue Correlation Heatmap")
    plt.show()
print(corr(train_encoded))


gc.collect()


def create_sum(train_encoded, test_encoded, top_ints, ohe=None):
    if ohe is None:
        ohe = top_ints

    for i in range(len(top_ints)):
        for j in range(len(ohe)):
            f1, f2 = top_ints[i], ohe[j]
            if f1 == f2:
                continue
            df_new = pd.DataFrame({
                f'{f1}_{f2}_sum': train_encoded[f1] + train_encoded[f2],
                f'{f1}_{f2}_diff': train_encoded[f1] - train_encoded[f2],
                f'{f1}_{f2}_prod': train_encoded[f1] * train_encoded[f2],
                f'{f1}_{f2}_div': train_encoded[f1] / (train_encoded[f2] + 1e-6),
            })

            train_encoded = pd.concat([train_encoded, df_new], axis=1)
            del df_new
            gc.collect()
            df_test_new = pd.DataFrame({
                f'{f1}_{f2}_sum': test_encoded[f1] + test_encoded[f2],
                f'{f1}_{f2}_diff': test_encoded[f1] - test_encoded[f2],
                f'{f1}_{f2}_prod': test_encoded[f1] * test_encoded[f2],
                f'{f1}_{f2}_div': test_encoded[f1] / (test_encoded[f2] + 1e-6),
            })

            test_encoded = pd.concat([test_encoded, df_test_new], axis=1)
            del df_test_new
            gc.collect()

    return train_encoded, test_encoded


top_ints = ['Nitrogen', 'Phosphorous', 'Potassium', 'Temparature']
train_encoded,test_encoded = create_sum(train_encoded,test_encoded,top_ints)
train_encoded


one_hot_encoded = [
    'Soil Type_Black',
    'Soil Type_Clayey',
    'Soil Type_Loamy',
    'Soil Type_Red',
    'Soil Type_Sandy',
    'Crop Type_Barley',
    'Crop Type_Cotton',
    'Crop Type_Ground Nuts',
    'Crop Type_Maize',
    'Crop Type_Millets',
    'Crop Type_Oil seeds',
    'Crop Type_Paddy',
    'Crop Type_Pulses',
    'Crop Type_Sugarcane',
    'Crop Type_Tobacco',
    'Crop Type_Wheat'
]
train_encoded,test_encoded = create_sum(train_encoded,test_encoded,top_ints,one_hot_encoded)
train_encoded_co,test_encoded_co = train_encoded,test_encoded


def lasso_test(train_encoded,test_encoded,train_df):
    le = LabelEncoder()
    y = le.fit_transform(train_df['Fertilizer Name'])
    X = train_encoded.drop(columns=['Fertilizer Name', 'label', 'id'], errors='ignore').astype('float32')
    X_test = test_encoded.drop(columns=['label', 'id'], errors='ignore').astype('float32')
    X_small = X.sample(n=30000, random_state=42)
    y_small = y[X_small.index]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_small)
    X_full_scaled = scaler.transform(X)
    test_scaled = scaler.transform(X_test)
    lasso = LogisticRegression(
    penalty='l1',
    C=0.1,
    solver='saga',
    max_iter=1000,
    multi_class='ovr',
    random_state=42
    )
    lasso.fit(X_scaled, y_small)
    mask = (lasso.coef_ != 0).any(axis=0)
    selected_cols = X.columns[mask]
    print(f"Lasso kept {len(selected_cols)} out of {X.shape[1]} features.")
    X_final = X[selected_cols]
    X_test_final = test_encoded[selected_cols].astype('float32')
    return X_final,X_test_final
train_encoded,test_encoded = lasso_test(train_encoded,test_encoded,train_df)
train_encoded



print(corr(train_encoded))


le = LabelEncoder()
X = train_encoded.drop([col for col in train_encoded.columns if col.startswith('Fertilizer Name_')], axis=1)
y = train_df['Fertilizer Name']
y_encoded = le.fit_transform(y)
X_train, X_valid, y_train, y_valid = train_test_split(
X, y_encoded, test_size=0.2, random_state=42)




def xgb_stratified(X_train, X_test_final, y_train, num_classes=None, folds=5, seed=42):
    if num_classes is None:
        num_classes = len(np.unique(y_train))

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    oof_preds = np.zeros((X_train.shape[0], num_classes))
    test_preds = np.zeros((X_test_final.shape[0], num_classes))

    dtest = xgb.DMatrix(X_test_final)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\nðŸ“¦ Fold {fold+1}/{folds}")

        X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
        X_val, y_val = X_train.iloc[valid_idx], y_train[valid_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dvalid = xgb.DMatrix(X_val, label=y_val)

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

        params = {
            'objective': 'multi:softprob',
            'num_class': num_classes,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'mlogloss',
            'seed': seed,
            'tree_method': 'gpu_hist',
            'verbosity': 1,
        }

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=1000,
            evals=watchlist,
            callbacks=[
                xgb.callback.EarlyStopping(rounds=30, min_delta=0.0001)
            ]
        )

        oof_preds[valid_idx] = model.predict(dvalid)
        test_preds += model.predict(dtest) / folds

        del X_tr, X_val, y_tr, y_val, dtrain, dvalid
        gc.collect()

    print("\n Training complete!")
    return oof_preds, test_preds


oof_preds, test_preds = xgb_stratified(X_train, test_encoded, y_train)



k = 5
top_k_preds = np.argsort(-test_preds, axis=1)[:, :k]

flat_indices = top_k_preds.ravel()
decoded_flat_preds = le.inverse_transform(flat_indices)
formatted_preds_array = decoded_flat_preds.reshape(top_k_preds.shape)
formatted_preds_list = [' '.join(map(str, row)) for row in formatted_preds_array]

submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
submission['Fertilizer Name'] = formatted_preds_list
submission.to_csv('submission.csv', index=False)
print("Submission file ready!")
submission.head()



model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='MultiClass',  
    eval_metric='Accuracy',
    verbose=100,
    random_seed=42
)

model.fit(X_train, y_train)
test_preds = model.predict_proba(test_encoded)
k = 5
top_k_preds = np.argsort(-test_preds, axis=1)[:, :k]

flat_indices = top_k_preds.ravel()
decoded_flat_preds = le.inverse_transform(flat_indices)
formatted_preds_array = decoded_flat_preds.reshape(top_k_preds.shape)
formatted_preds_list = [' '.join(map(str, row)) for row in formatted_preds_array]
submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
submission['Fertilizer Name'] = formatted_preds_list
submission.to_csv('submission.csv', index=False)
print("Submission file ready!")
submission.head()

