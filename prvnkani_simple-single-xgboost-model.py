import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
pd.options.mode.copy_on_write = True
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import itertools
import random
import warnings
warnings.simplefilter('ignore')


def feature_engineering(df):
    df.drop(columns=['id'], inplace=True)

    df['Sex'] = df['Sex'].map({'female': 1, 'male': 2})
    df['AgeSex'] = df['Age'].astype(str) + df['Sex'].astype(str)
    df['AgeSex'] = LabelEncoder().fit_transform(df['AgeSex']) + 1
    for col in ['Sex', 'Age', 'AgeSex']:
        df['CAT_' + col] = df[col].astype('category')
        
    features = ['Age', 'Weight', 'Height', 'Body_Temp', 'Heart_Rate', 'Duration', 'Sex', 'AgeSex']

    for comb in itertools.combinations(features, 2):
        df[" * ".join(comb)] = df[list(comb)].prod(axis=1)
    
    return df


df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df_orginal = pd.read_csv("/kaggle/input/orginal-dataset/calories.csv")
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

df_train = feature_engineering(df_train)
df_orginal = feature_engineering(df_orginal)
df_test = feature_engineering(df_test)

df_train.shape, df_orginal.shape, df_test.shape


seed = 13
FOLD = 8
cv = KFold(FOLD, random_state=seed, shuffle=True)
pred_test = np.zeros((250000,))

for idx_train, idx_valid in cv.split(df_train):
    print("\n")

    X_train = df_train.iloc[idx_train]
    X_train = pd.concat([X_train, df_orginal], axis=0, ignore_index=True).sample(frac=1, random_state=seed)
    X_valid = df_train.iloc[idx_valid]

    y_train = np.log1p(X_train.pop('Calories'))
    y_valid = np.log1p(X_valid.pop('Calories'))

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
    dtest = xgb.DMatrix(df_test, enable_categorical=True)

    params = {
        'eval_metric': 'rmse',
        'seed': seed,
        'max_depth': 10,
        'learning_rate': 0.003,
        'reg_alpha': 2,
        'reg_lambda': 1,
        'max_delta_step': 2,
        'subsample': 0.9,
        'colsample_bytree': 0.55,
        'enable_categorical': True,
        'device': "cuda"
    }
    
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=1000000, 
        evals=[(dtrain, 'train'), (dval, 'validation')], 
        early_stopping_rounds=50, 
        verbose_eval=2000
    )

    predictions = model.predict(dval)
    pred_test += model.predict(dtest)

pred_test /= FOLD

df_subm = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
df_subm['Calories'] = np.expm1(pred_test)
df_subm.to_csv('submission.csv', index=False)


fig, ax = plt.subplots(figsize=(24, 24))  # Adjust the figure size if needed
xgb.plot_importance(
    model,
    ax=ax,
    max_num_features=40, 
    importance_type="weight",
)
plt.title("XGB Top Feature Importances")
plt.show()

