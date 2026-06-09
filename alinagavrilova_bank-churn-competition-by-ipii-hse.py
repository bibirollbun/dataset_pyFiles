import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

from category_encoders import MEstimateEncoder, CatBoostEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import clone
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler
from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier, Pool
from catboost.utils import eval_metric
import warnings
warnings.filterwarnings("ignore")

import random
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

seed = 42
seed_everything(seed=seed)


train = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/train.csv')
test = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/test.csv')

X = train.iloc[:, :-1]
y = train.iloc[:, -1]

num_folds = 5
skf = StratifiedKFold(n_splits = num_folds, random_state = seed, shuffle = True)


def add_features(df):
    df['Active_&_HasCrCard'] = df['IsActiveMember'] * df['HasCrCard']
    df['ZeroBalance'] = np.where(df['Balance'] == 0, 1, 0)
    df['Active_&_NonZeroBalance'] = df['IsActiveMember'] * df['ZeroBalance'].replace({0:1, 1:0})
    df['HasHighSalary'] = np.where(df['EstimatedSalary'] > 176692.65, 1, 0)
    # df['HasHighSalary_&_ZeroBalance'] = df['HasHighSalary'] * df['ZeroBalance']
    return df

FeatureGenerator = FunctionTransformer(add_features)


def fit_score(estimator, cv = skf):
    val_predictions = np.zeros((len(X)))
    test_predictions = np.zeros((len(test)))
    train_scores, val_scores = [], []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):        
        X_train = X.iloc[train_idx].reset_index(drop = True)
        y_train = y.iloc[train_idx].reset_index(drop = True)        
        X_val = X.iloc[val_idx].reset_index(drop = True)
        y_val = y.iloc[val_idx].reset_index(drop = True)
        
        model = clone(estimator)
        model.fit(X_train, y_train)
        
        train_preds = model.predict_proba(X_train)[:, 1]
        val_preds = model.predict_proba(X_val)[:, 1]
        test_preds = model.predict_proba(test)[:, 1]
                  
        val_predictions[val_idx] += val_preds
        test_predictions += test_preds / cv.get_n_splits()
        
        train_scores.append(roc_auc_score(y_train, train_preds))      
        val_scores.append(roc_auc_score(y_val, val_preds))
       
    print(f'Val Score: {np.mean(val_scores):.5f} | Train Score: {np.mean(train_scores):.5f}')
    return val_scores, val_predictions, test_predictions


predict_list = pd.DataFrame()


lgb_params = {'learning_rate': 0.008221540404557054, 'max_depth': 3, 'subsample': 0.5007231782885733, 'min_child_weight': 0.2313795650596383, 'reg_lambda': 0.33413151141784225, 'reg_alpha': 0.6136071252525226}

LGB = make_pipeline(
    FeatureGenerator,
    CatBoostEncoder(cols = ['Surname', 'CreditScore', 'Age']),
    MEstimateEncoder(cols = ['Geography', 'Gender', 'NumOfProducts']),
    StandardScaler(),
    LGBMClassifier(**lgb_params, random_state = seed, n_estimators = 1000, verbose = -1)
)

_, _, predict_list['LGB'] = fit_score(LGB)


df_train = train.copy()
df_test = test.copy()

X = df_train.iloc[:, 1:-1]
y = df_train.iloc[:, -1]
df_test = df_test.iloc[:, 1:]

X = add_features(X)
df_test = add_features(df_test)

scale_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
scaler = MinMaxScaler()
X[scale_cols] = scaler.fit_transform(X[scale_cols])
df_test[scale_cols] = scaler.transform(df_test[scale_cols])


cat_features = np.where(X.dtypes != np.float64)[0]
test_predictions = np.empty((num_folds, len(df_test)))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):   
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]
    
    train_pool = Pool(X_train, y_train, cat_features = cat_features)
    val_pool = Pool(X_val, y_val, cat_features = cat_features)
    
    model = CatBoostClassifier(eval_metric = 'AUC', learning_rate = 0.02, iterations = 1000)
    model.fit(train_pool, eval_set = val_pool, verbose = 0)

    train_preds = model.predict_proba(X_train)[:, 1]
    val_preds = model.predict_proba(X_val)[:,1]
    test_preds = model.predict_proba(df_test)[:, 1]

    test_predictions[fold, :] = test_preds

    train_score = roc_auc_score(y_train, train_preds)
    val_score = roc_auc_score(y_val, val_preds)
    
    print(f'Val Score: {np.mean(val_score):.5f} | Train Score: {np.mean(train_score):.5f}')

predict_list['CB'] = test_predictions.mean(axis=0)


weights = [0.18, 0.82]
predictions = predict_list.to_numpy() @ weights

submission_df = pd.DataFrame({'id': test['id'], 'Exited': predictions})
submission_df.to_csv('submission.csv', index=False)


submission_df

