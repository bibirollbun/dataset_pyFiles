import pandas as pd; pd.set_option('display.max_columns', 100)
import numpy as np

import gc

import warnings
warnings.filterwarnings('ignore')

from tqdm.notebook import tqdm

import re

from functools import partial
from scipy.stats import kurtosis, skew, gmean, mode

import matplotlib.pyplot as plt; plt.style.use('ggplot')
import seaborn as sns
import plotly.express as px

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, FunctionTransformer, PowerTransformer, PolynomialFeatures
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import KNNImputer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV, RepeatedStratifiedKFold, cross_val_score, cross_val_predict, RepeatedKFold
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay, cohen_kappa_score, log_loss, f1_score, r2_score, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge, RidgeCV
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC, LinearSVR

from category_encoders import TargetEncoder

import ydf
from ydf import GradientBoostedTreesLearner

import xgboost as xgb

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

from sklearn.neural_network import MLPClassifier

import optuna


%%time
train = pd.read_csv('../input/playground-series-s4e10/train.csv', index_col=0)
test = pd.read_csv('../input/playground-series-s4e10/test.csv', index_col=0)

print('The dimension of the train dataset is:', train.shape)
print('The dimension of the test dataset is:', test.shape)


train.head()


test.head()


print('--- Train ---\n')
print(100*train.isnull().sum() / train.shape[0])
print('\n')
print('--- Test ---\n')
print(100*test.isnull().sum() / train.shape[0])


print(f"There are {sum(train.duplicated())} duplicated rows in the train data frame.")

print("\n")
print(f"After dropping the loan_status column, there are {sum(train.drop(columns=['loan_status']).duplicated())} duplicated rows in the train data frame.")

print("\n")
print(f"There are {sum(test.duplicated())} duplicated rows in the test data frame.")


temp_train = train.drop(columns=['loan_status'], axis=1)
temp_test = test

inner_join = pd.merge(temp_train, temp_test)
print(f"There are {inner_join.shape[0]} observations that appear in both the train and test data frames")


ax = round(train['loan_status'].value_counts(normalize=True), 2).plot(kind='barh', color='steelblue')
ax.bar_label(ax.containers[0], label_type='edge')
ax.margins(y=0.1)
plt.xlabel('Percentage');


fig, axes = plt.subplots(2, 2, figsize=(20, 15))

cmap = sns.color_palette("coolwarm", as_cmap=True)

sns.heatmap(data=round(100*pd.crosstab(train['person_home_ownership'], train['loan_status'], normalize=0), 2), annot=True, cmap=cmap, fmt='.0f', ax=axes[0, 0])
sns.heatmap(data=round(100*pd.crosstab(train['loan_intent'], train['loan_status'], normalize=0), 2), annot=True, cmap=cmap, fmt='.0f', ax=axes[0, 1])
sns.heatmap(data=round(100*pd.crosstab(train['loan_grade'], train['loan_status'], normalize=0), 2), annot=True, cmap=cmap, fmt='.0f', ax=axes[1, 0]);
sns.heatmap(data=round(100*pd.crosstab(train['cb_person_default_on_file'], train['loan_status'], normalize=0), 2), annot=True, cmap=cmap, fmt='.0f', ax=axes[1, 1]);


fig, axes = plt.subplots(1, 3, figsize=(20, 7))

sns.kdeplot(data=train, x='person_age', hue='loan_status', ax=axes[0]);
sns.kdeplot(data=train, x='person_income', hue='loan_status', ax=axes[1]);
sns.scatterplot(data=train, x='person_age', y='person_income', hue='loan_status', ax=axes[2]);


fig, axes = plt.subplots(1, 3, figsize=(20, 7))

sns.kdeplot(data=train, x='person_emp_length', hue='loan_status', ax=axes[0]);
sns.kdeplot(data=train, x='loan_amnt', hue='loan_status', ax=axes[1]);
sns.scatterplot(data=train, x='person_emp_length', y='loan_amnt', hue='loan_status', ax=axes[2]);


fig, axes = plt.subplots(1, 3, figsize=(20, 7))

sns.kdeplot(data=train, x='loan_int_rate', hue='loan_status', ax=axes[0]);
sns.kdeplot(data=train, x='loan_percent_income', hue='loan_status', ax=axes[1]);
sns.scatterplot(data=train, x='loan_int_rate', y='loan_percent_income', hue='loan_status', ax=axes[2]);


sns.kdeplot(data=train, x='cb_person_cred_hist_length', hue='loan_status');


%%time
cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

def converting_datatypes(df, cat_cols, df_train=False):
    
    for col in cat_cols:
        df[col] = df[col].astype('category')
    
    df['person_age'] = df['person_age'].astype('int32')
    df['cb_person_cred_hist_length'] = df['cb_person_cred_hist_length'].astype('int32') 
    
    if df_train==True:

        df['loan_status'] = df['loan_status'].astype('int8')
        
    return df
        
train = converting_datatypes(train, cat_cols ,df_train=True)
test = converting_datatypes(test, cat_cols)


def clip_data(df):
    
    df['person_age'] = df['person_age'].clip(None, 90)
    df['person_emp_length'] = df['person_emp_length'].clip(None, 60) 
    
    return df

train = clip_data(train)
test = clip_data(test)


%%time
X = train.drop(columns=['loan_status'], axis=1)
Y = train['loan_status']

skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)


# Converting label to dummies
train_dummies = pd.get_dummies(X[cat_cols], drop_first=True, dtype='int8')
RF_train = pd.concat([X.drop(columns=cat_cols, axis=1), train_dummies], axis=1)

test_dummies = pd.get_dummies(test[cat_cols], drop_first=True, dtype='int8')
RF_test = pd.concat([test.drop(columns=cat_cols, axis=1), test_dummies], axis=1)

rf_params = {'n_estimators': 148,
             'max_depth': 15,
             'min_samples_split': 17,
             'min_samples_leaf': 6,
             'n_jobs': -1}

scores, rf_oof_preds, rf_test_preds = list(), list(), list()
for i, (train_index, test_index) in enumerate(skf.split(RF_train, Y)):
    
    print(f"------------ Working on Fold {i} ------------")
    
    X_train, X_test = RF_train.iloc[train_index], RF_train.iloc[test_index]
    y_train, y_test = Y[train_index], Y.iloc[test_index]
    
    rf_md = RandomForestClassifier(**rf_params).fit(X_train, y_train)
    preds = rf_md.predict_proba(X_test)[:, 1]
    
    oof_preds = pd.DataFrame()
    oof_preds['y'] = y_test.values
    oof_preds['rf_preds'] = preds
    oof_preds['fold'] = i
    rf_oof_preds.append(oof_preds)
    
    score = roc_auc_score(y_test, preds)
    print(f"The oof ROC-AUC score is {score}")
    scores.append(score)
    
    test_preds = pd.DataFrame()
    test_preds['rf_preds'] = rf_md.predict_proba(RF_test)[:, 1]
    test_preds['fold'] = i
    rf_test_preds.append(test_preds)

rf_oof_score = np.mean(scores)
rf_std = np.std(scores)
print(f"The 10-fold average oof ROC-AUC score of the RandomForest model is {rf_oof_score}")
print(f"The 10-fold std oof ROC-AUC score of the RandomForest model is {rf_std}")


%%time
lgb_params = {'learning_rate': 0.09030331403653566,
              'n_estimators': 190,
              'max_depth': 15,
              'reg_alpha': 0.25370376964322267,
              'reg_lambda': 0.06912978243728862,
              'num_leaves': 38,
              'colsample_bytree': 0.4816656035216278,
              'verbose': -1,
              'n_jobs': -1}

scores, lgb_oof_preds, lgb_test_preds = list(), list(), list()
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
    
    print(f"------------ Working on Fold {i} ------------")
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y[train_index], Y.iloc[test_index]
    
    lgb_md = LGBMClassifier(**lgb_params).fit(X_train, y_train)
    preds = lgb_md.predict_proba(X_test)[:, 1]
    
    oof_preds = pd.DataFrame()
    oof_preds['y'] = y_test.values
    oof_preds['lgb_preds'] = preds
    oof_preds['fold'] = i
    lgb_oof_preds.append(oof_preds)
    
    score = roc_auc_score(y_test, preds)
    print(f"The oof ROC-AUC score is {score}")
    scores.append(score)
    
    test_preds = pd.DataFrame()
    test_preds['lgb_preds'] = lgb_md.predict_proba(test)[:, 1]
    test_preds['fold'] = i
    lgb_test_preds.append(test_preds)

lgb_oof_score = np.mean(scores)
lgb_std = np.std(scores)
print(f"The 10-fold average oof ROC-AUC score of the LGBM model is {lgb_oof_score}")
print(f"The 10-fold std oof ROC-AUC score of the LGBM model is {lgb_std}")


%%time
xgb_params = {'n_estimators': 199,
              'max_depth': 12,
              'learning_rate': 0.09304789779291263,
              'gamma': 0.2571967403496238,
              'min_child_weight': 20,
              'colsample_bytree': 0.5141737333809174,
              'n_jobs': -1,
              'enable_categorical': True}

scores, xgb_oof_preds, xgb_test_preds = list(), list(), list()
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
    
    print(f"------------ Working on Fold {i} ------------")
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y[train_index], Y.iloc[test_index]
                        
    xgb_md = XGBClassifier(**xgb_params).fit(X_train, y_train)
    preds = xgb_md.predict_proba(X_test)[:, 1]
    
    oof_preds = pd.DataFrame()
    oof_preds['y'] = y_test.values
    oof_preds['xgb_preds'] = preds
    oof_preds['fold'] = i
    xgb_oof_preds.append(oof_preds)
    
    score = roc_auc_score(y_test, preds)
    print(f"The oof ROC-AUC score is {score}")
    scores.append(score)
    
    test_preds = pd.DataFrame()
    test_preds['xgb_preds'] = xgb_md.predict_proba(test)[:, 1]
    test_preds['fold'] = i
    xgb_test_preds.append(test_preds)

xgb_oof_score = np.mean(scores)
xgb_std = np.std(scores)
print(f"The 10-fold average oof ROC-AUC score of the XGBoost model is {xgb_oof_score}")
print(f"The 10-fold std oof ROC-AUC score of the XGBoost model is {xgb_std}")


cb_params = {'loss_function': 'Logloss',
             'iterations': 195,
             'learning_rate': 0.09381481508561976,
             'depth': 11,
             'bagging_temperature': 0.0850640366226123,
             'l2_leaf_reg': 0,
             'grow_policy': 'Lossguide',
             'task_type': 'CPU'}

test_pool = Pool(data=test, cat_features=cat_cols)

scores, cat_oof_preds, cat_test_preds = list(), list(), list()
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
    
    print(f"------------ Working on Fold {i} ------------")
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    model_pool = Pool(data=X_train, label=y_train, cat_features=cat_cols)
    eval_pool = Pool(data=X_test, label=y_test, cat_features=cat_cols)
            
    cat_md = CatBoostClassifier(**cb_params).fit(model_pool, eval_set=eval_pool, verbose=0)
    preds = cat_md.predict_proba(eval_pool)[:, 1]
    
    oof_preds = pd.DataFrame()
    oof_preds['y'] = y_test.values
    oof_preds['cat_preds'] = preds
    oof_preds['fold'] = i
    cat_oof_preds.append(oof_preds)
    
    score = roc_auc_score(y_test, preds)
    print(f"The oof RMSE score is {score}")
    scores.append(score)
    
    test_preds = pd.DataFrame()
    test_preds['cat_preds'] = cat_md.predict_proba(test_pool)[:, 1]
    test_preds['fold'] = i
    cat_test_preds.append(test_preds)

cat_oof_score = np.mean(scores)  
cat_std = np.std(scores)
print(f"The 10-fold average oof ROC-AUC score of the CatBoost model is {cat_oof_score}")
print(f"The 10-fold std oof ROC-AUC score of the CatBoost model is {xgb_std}")


%%time
oof_preds = pd.concat(lgb_oof_preds)
oof_preds['xgb_preds'] = pd.concat(xgb_oof_preds)['xgb_preds']
oof_preds['cat_preds'] = pd.concat(cat_oof_preds)['cat_preds']
oof_preds['rf_preds'] = pd.concat(rf_oof_preds)['rf_preds']


def objective(trial):
    
    weights = [trial.suggest_float(f"weight{n}", 1e-5, 1) for n in range(4)]

    scores = list()
    for i in range(0, 10):
        
        x_test = oof_preds[oof_preds['fold']==i].reset_index(drop=True)
        ens_pred = (weights[0]*x_test['lgb_preds'].values +
                    weights[1]*x_test['xgb_preds'].values + 
                    weights[2]*x_test['cat_preds'].values + 
                    weights[3]*x_test['rf_preds'].values) 
        
        y_test = x_test['y']
        score = roc_auc_score(y_test, ens_pred)
        scores.append(score)
    
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=3000, n_jobs=-1)


%%time
print("Best Trial:")
best_trial = study.best_trial

print(f"  Value: {best_trial.value}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")


%%time
w = study.best_trial.params
scores = list()
for i in range(0, 10):
    
    x_train = oof_preds[oof_preds['fold']!=i].reset_index(drop=True)
    x = x_train.drop(columns=['fold', 'y'], axis=1)
    y = x_train['y']
    
    test = oof_preds[oof_preds['fold']==i].reset_index(drop=True)
    x_test = test.drop(columns=['fold', 'y'], axis=1)
    y_test = test['y']

    optuna_pred = (w['weight0']*test['lgb_preds'] + w['weight1']*test['xgb_preds'] + w['weight2']*test['cat_preds'] + w['weight3']*test['rf_preds'])
    score = roc_auc_score(y_test, optuna_pred)
    scores.append(score)

print(f"The 10-fold oof average ROC-AUC score of the Optuna Blender is {np.mean(scores)}")



results = pd.DataFrame()
results['Model'] = ['RF', 'LGBM', 'XGB', 'CatBoost', 'Optuna Blend']
results['10-fold oof ROC-AUC'] = [rf_oof_score, lgb_oof_score, xgb_oof_score, cat_oof_score, np.mean(scores)] 
print(results)


%%time
test_preds = pd.concat(lgb_test_preds)
test_preds['xgb_preds'] = pd.concat(xgb_test_preds)['xgb_preds']
test_preds['cat_preds'] = pd.concat(cat_test_preds)['cat_preds']
test_preds['rf_preds'] = pd.concat(rf_test_preds)['rf_preds']
test_preds.head()


%%time
test_pred_final = list()
for i in range(0, 10):
    
    x_train = oof_preds[oof_preds['fold']!=i].reset_index(drop=True)
    x = x_train.drop(columns=['fold', 'y'], axis=1)
    y = x_train['y']

    temp = test_preds[test_preds['fold']==i].reset_index(drop=True)
    optuna_pred = (w['weight0']*temp['lgb_preds'] + w['weight1']*temp['xgb_preds'] + w['weight2']*temp['cat_preds'] + w['weight3']*temp['rf_preds'])
    
    test_pred_final.append(optuna_pred)


%%time
submission = pd.read_csv('../input/playground-series-s4e10/sample_submission.csv')
submission['loan_status'] = np.mean(test_pred_final, axis=0)
submission.head()


%%time
submission.to_csv('baseline_sub_1.csv', index=False)

del train, test, submission, X, Y, test_pred_final, scores, w, oof_preds, test_preds
gc.collect()


%%time
train = pd.read_csv('../input/playground-series-s4e10/train.csv', index_col=0)
test = pd.read_csv('../input/playground-series-s4e10/test.csv', index_col=0)
original = pd.read_csv('../input/loan-approval-prediction/credit_risk_dataset.csv')

cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

def converting_datatypes(df, cat_cols, df_train=False):
    
    for col in cat_cols:
        df[col] = df[col].astype('category')
    
    df['person_age'] = df['person_age'].astype('int32')
    df['cb_person_cred_hist_length'] = df['cb_person_cred_hist_length'].astype('int32') 
    
    if df_train==True:

        df['loan_status'] = df['loan_status'].astype('int8')
        
    return df
        
train = converting_datatypes(train, cat_cols ,df_train=True)
test = converting_datatypes(test, cat_cols)
original = converting_datatypes(original, cat_cols)

def clip_data(df):
    
    df['person_age'] = df['person_age'].clip(None, 90)
    df['person_emp_length'] = df['person_emp_length'].clip(None, 60) 
    
    return df

train = clip_data(train)
test = clip_data(test)
original = clip_data(original)


%%time
X = train.drop('loan_status', axis=1)
X['generated'] = 1
Y = train['loan_status']

X_org = original.drop('loan_status', axis=1)
X_org['generated'] = 0
y_org = original['loan_status']

test['generated'] = 1

skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)


%%time
dummies = pd.get_dummies(X[cat_cols], drop_first=True, dtype='int8')
X_gb = pd.concat([X.drop(columns=cat_cols, axis=1), dummies], axis=1)

dummies = pd.get_dummies(test[cat_cols], drop_first=True, dtype='int8')
test_gb = pd.concat([test.drop(columns=cat_cols, axis=1), dummies], axis=1)

dummies = pd.get_dummies(X_org[cat_cols], drop_first=True, dtype='int8')
X_org_gb = pd.concat([X_org.drop(columns=cat_cols, axis=1), dummies], axis=1)

gb_params = {'n_estimators': 473,
 'max_depth': 5,
 'min_samples_split': 4,
 'min_samples_leaf': 2,
 'subsample': 0.8867385546845281,
 'random_state': 1}

scores, gb_oof_preds, gb_test_preds = list(), list(), list()
for i, (train_index, test_index) in enumerate(skf.split(X_gb, Y)):

    print(f"------------ Working on Fold {i} ------------")
            
    X_train, X_test = X_gb.iloc[train_index], X_gb.iloc[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    X_train = pd.concat([X_train, X_org_gb], axis=0).reset_index(drop=True)
    y_train = pd.concat([y_train, y_org], axis=0).reset_index(drop=True)
    dat = pd.concat([X_train, y_train], axis=1).dropna().reset_index(drop=True)
    X_train = dat.drop(columns='loan_status', axis=1)
    y_train = dat['loan_status']

    GB_md = GradientBoostingClassifier(**gb_params).fit(X_train, y_train)
    GB_pred = GB_md.predict_proba(X_test)[:, 1]

    oof_preds = pd.DataFrame()
    oof_preds['y'] = y_test.values
    oof_preds['gb_preds'] = GB_pred
    oof_preds['fold'] = i
    gb_oof_preds.append(oof_preds)

    score = roc_auc_score(y_test, GB_pred)
    scores.append(score)
    print(f"The oof ROC-AUC score is {score}")

    test_preds = pd.DataFrame()
    test_preds['gb_preds'] = GB_md.predict_proba(test_gb)[:, 1]
    test_preds['fold'] = i
    gb_test_preds.append(test_preds)

gb_oof_score = np.mean(scores)
gb_std = np.std(scores)
print(f"The 10-fold average oof ROC-AUC score of the GradientBoosting model is {gb_oof_score}")
print(f"The 10-fold std oof ROC-AUC score of the GradientBoosting model is {gb_std}")


%%time
ydf.verbose(-1)
scores, ydf_oof_preds, ydf_test_preds = list(), list(), list()
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
            
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    X_train = pd.concat([X_train, X_org], axis=0)
    y_train = pd.concat([y_train, y_org], axis=0)
    
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    ydf_md = GradientBoostedTreesLearner(label='loan_status', 
                                         num_threads=10, 
                                         num_trees=1500,
                                         max_depth=6).train(train_data)
    ydf_pred = ydf_md.predict(test_data)

    score = roc_auc_score(y_test, ydf_pred)
    print(f"The oof ROC-AUC score is {score}")
    scores.append(score)

    oof_preds = pd.DataFrame()
    oof_preds['y'] = y_test.values
    oof_preds['ydf_preds'] = ydf_pred
    oof_preds['fold'] = i
    oof_preds['index'] = test_index
    ydf_oof_preds.append(oof_preds)

    test_preds = pd.DataFrame()
    test_preds['ydf_preds'] = ydf_md.predict(test)
    test_preds['fold'] = i
    ydf_test_preds.append(test_preds)

ydf_oof_score = np.mean(scores)
ydf_std = np.std(scores)
print(f"The 10-fold average oof ROC-AUC score of the GradientBoostedTreesLearner model is {ydf_oof_score}")
print(f"The 10-fold std oof ROC-AUC score of the GradientBoostedTreesLearner model is {ydf_std}")


%%time
lgb_params = {'learning_rate': 0.09967204378010042,
              'n_estimators': 293,
              'max_depth': 11,
              'reg_alpha': 1.8071679385784074,
              'reg_lambda': 0.013057189102691127,
              'num_leaves': 35,
              'colsample_bytree': 0.4708077087365519,
              'verbose': -1,
              'n_jobs': -1}

scores, lgb_oof_preds, lgb_test_preds = list(), list(), list()
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
    
    print(f"------------ Working on Fold {i} ------------")
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y[train_index], Y.iloc[test_index]
    
    X_train = pd.concat([X_train, X_org], axis=0)
    y_train = pd.concat([y_train, y_org], axis=0)
    
    lgb_md = LGBMClassifier(**lgb_params).fit(X_train, y_train)
    preds = lgb_md.predict_proba(X_test)[:, 1]
    
    oof_preds = pd.DataFrame()
    oof_preds['y'] = y_test.values
    oof_preds['lgb_preds'] = preds
    oof_preds['fold'] = i
    lgb_oof_preds.append(oof_preds)
    
    score = roc_auc_score(y_test, preds)
    print(f"The oof ROC-AUC score is {score}")
    scores.append(score)
    
    test_preds = pd.DataFrame()
    test_preds['lgb_preds'] = lgb_md.predict_proba(test)[:, 1]
    test_preds['fold'] = i
    lgb_test_preds.append(test_preds)

lgb_oof_score = np.mean(scores)
lgb_std = np.std(scores)
print(f"The 10-fold average oof ROC-AUC score of the LGBM model is {lgb_oof_score}")
print(f"The 10-fold std oof ROC-AUC score of the LGBM model is {lgb_std}")


%%time
lgb_dart_params = {'boosting_type': 'dart',
                   'learning_rate': 0.09755475070802529,
                   'n_estimators': 1000,
                   'max_depth': 12,
                   'reg_alpha': 0.396286496452296,
                   'reg_lambda': 0.01930416713070059,
                   'num_leaves': 40,
                   'colsample_bytree': 0.5305154955728938,
                   'max_bin': 2000,
                   'verbose': -1,
                   'n_jobs': -1}


scores, lgb_dart_oof_preds, lgb_dart_test_preds = list(), list(), list()
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
    
    print(f"------------ Working on Fold {i} ------------")
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y[train_index], Y.iloc[test_index]
    
    X_train = pd.concat([X_train, X_org], axis=0)
    y_train = pd.concat([y_train, y_org], axis=0)
    
    lgb_md = LGBMClassifier(**lgb_dart_params).fit(X_train, y_train)
    preds = lgb_md.predict_proba(X_test)[:, 1]
    
    oof_preds = pd.DataFrame()
    oof_preds['y'] = y_test.values
    oof_preds['lgb_dart_preds'] = preds
    oof_preds['fold'] = i
    lgb_dart_oof_preds.append(oof_preds)
    
    score = roc_auc_score(y_test, preds)
    print(f"The oof ROC-AUC score is {score}")
    scores.append(score)
    
    test_preds = pd.DataFrame()
    test_preds['lgb_dart_preds'] = lgb_md.predict_proba(test)[:, 1]
    test_preds['fold'] = i
    lgb_dart_test_preds.append(test_preds)

lgb_dart_oof_score = np.mean(scores)
lgb_dart_std = np.std(scores)
print(f"The 10-fold average oof ROC-AUC score of the LGBM (dart) model is {lgb_dart_oof_score}")
print(f"The 10-fold std oof ROC-AUC score of the LGBM (dart) model is {lgb_dart_std}")


%%time
lgb_goss_params = {'boosting_type': 'goss',
                   'learning_rate': 0.05120160070340736,
                   'n_estimators': 1500,
                   'max_depth': 11,
                   'reg_alpha': 0.03174190176577365,
                   'reg_lambda': 0.011462670291787876,
                   'num_leaves': 22,
                   'max_bin': 2000,
                   'colsample_bytree': 0.4382312804974896,
                   'verbose': -1,
                   'n_jobs': -1}

scores, lgb_goss_oof_preds, lgb_goss_test_preds = list(), list(), list()
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
    
    print(f"------------ Working on Fold {i} ------------")
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y[train_index], Y.iloc[test_index]
    
    X_train = pd.concat([X_train, X_org], axis=0)
    y_train = pd.concat([y_train, y_org], axis=0)
    
    lgb_md = LGBMClassifier(**lgb_goss_params).fit(X_train, y_train)
    preds = lgb_md.predict_proba(X_test)[:, 1]
    
    oof_preds = pd.DataFrame()
    oof_preds['y'] = y_test.values
    oof_preds['lgb_goss_preds'] = preds
    oof_preds['fold'] = i
    lgb_goss_oof_preds.append(oof_preds)
    
    score = roc_auc_score(y_test, preds)
    print(f"The oof ROC-AUC score is {score}")
    scores.append(score)
    
    test_preds = pd.DataFrame()
    test_preds['lgb_goss_preds'] = lgb_md.predict_proba(test)[:, 1]
    test_preds['fold'] = i
    lgb_goss_test_preds.append(test_preds)

lgb_goss_oof_score = np.mean(scores)
lgb_goss_std = np.std(scores)
print(f"The 10-fold average oof ROC-AUC score of the LGBM (goss) model is {lgb_goss_oof_score}")
print(f"The 10-fold std oof ROC-AUC score of the LGBM (goss) model is {lgb_goss_std}")


%%time
xgb_params = {'n_estimators': 2000,
              'max_depth': 9,
              'learning_rate': 0.05995911629796249,
              'gamma': 0.1276126112509503,
              'min_child_weight': 24,
              'colsample_bytree': 0.5717739416944538,
              'n_jobs': -1,
              'max_bin': 3000,
              'enable_categorical': True}

scores, xgb_oof_preds, xgb_test_preds = list(), list(), list()
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
    
    print(f"------------ Working on Fold {i} ------------")
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y[train_index], Y.iloc[test_index]
    
    X_train = pd.concat([X_train, X_org], axis=0)
    y_train = pd.concat([y_train, y_org], axis=0)
                        
    xgb_md = XGBClassifier(**xgb_params).fit(X_train, y_train)
    preds = xgb_md.predict_proba(X_test)[:, 1]
    
    oof_preds = pd.DataFrame()
    oof_preds['y'] = y_test.values
    oof_preds['xgb_preds'] = preds
    oof_preds['fold'] = i
    xgb_oof_preds.append(oof_preds)
    
    score = roc_auc_score(y_test, preds)
    print(f"The oof ROC-AUC score is {score}")
    scores.append(score)
    
    test_preds = pd.DataFrame()
    test_preds['xgb_preds'] = xgb_md.predict_proba(test)[:, 1]
    test_preds['fold'] = i
    xgb_test_preds.append(test_preds)

xgb_oof_score = np.mean(scores)
xgb_std = np.std(scores)
print(f"The 10-fold average oof ROC-AUC score of the XGBoost model is {xgb_oof_score}")
print(f"The 10-fold std oof ROC-AUC score of the XGBoost model is {xgb_std}")


%%time
cat_params = {'loss_function': 'Logloss',
              'iterations': 490,
              'learning_rate': 0.09054784949573864,
              'depth': 12,
              'bagging_temperature': 0.010287865831919592,
              'l2_leaf_reg': 4,
              'grow_policy': 'Lossguide',
              'task_type': 'CPU'}

cat_cols_cat = ['person_age',
 'person_income',
 'person_home_ownership',
 'person_emp_length',
 'loan_intent',
 'loan_grade',
 'loan_amnt',
 'loan_int_rate',
 'loan_percent_income',
 'cb_person_default_on_file',
 'cb_person_cred_hist_length']

X = X.astype('str')
X_org = X_org.astype('str') 
test_pool = Pool(data=test.astype('str'), cat_features=cat_cols_cat)

scores, cat_oof_preds, cat_test_preds = list(), list(), list()
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
    
    print(f"------------ Working on Fold {i} ------------")
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    X_train = pd.concat([X_train, X_org], axis=0)
    y_train = pd.concat([y_train, y_org], axis=0)

    model_pool = Pool(data=X_train, label=y_train, cat_features=cat_cols_cat)
    eval_pool = Pool(data=X_test, label=y_test, cat_features=cat_cols_cat)
            
    cat_md = CatBoostClassifier(**cat_params).fit(model_pool, eval_set=eval_pool, verbose=0)
    preds = cat_md.predict_proba(eval_pool)[:, 1]
    
    oof_preds = pd.DataFrame()
    oof_preds['y'] = y_test.values
    oof_preds['cat_preds'] = preds
    oof_preds['fold'] = i
    cat_oof_preds.append(oof_preds)
    
    score = roc_auc_score(y_test, preds)
    print(f"The oof ROC-AUC score is {score}")
    scores.append(score)
    
    test_preds = pd.DataFrame()
    test_preds['cat_preds'] = cat_md.predict_proba(test_pool)[:, 1]
    test_preds['fold'] = i
    cat_test_preds.append(test_preds)

cat_oof_score = np.mean(scores)  
cat_std = np.std(scores)
print(f"The 10-fold average oof ROC-AUC score of the CatBoost model is {cat_oof_score}")
print(f"The 10-fold std oof ROC-AUC score of the CatBoost model is {cat_std}")


%%time
cat_params = {'loss_function': 'Logloss',
              'iterations': 1000,
              'learning_rate': 0.045,
              'depth': 7,
              'bagging_temperature': 0.25,
              'l2_leaf_reg': 0.8,
              'colsample_bylevel': 0.40,
              'min_data_in_leaf': 30,
              'random_state': 42,
              'task_type': 'CPU'}

cat_cols_cat = ['person_age',
 'person_income',
 'person_home_ownership',
 'person_emp_length',
 'loan_intent',
 'loan_grade',
 'loan_amnt',
 'loan_int_rate',
 'loan_percent_income',
 'cb_person_default_on_file',
 'cb_person_cred_hist_length']

X = X.astype('str')
X_org = X_org.astype('str') 
test_pool = Pool(data=test.astype('str'), cat_features=cat_cols_cat)

scores, cat_oof_preds_1, cat_test_preds_1 = list(), list(), list()
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
    
    print(f"------------ Working on Fold {i} ------------")
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    X_train = pd.concat([X_train, X_org], axis=0)
    y_train = pd.concat([y_train, y_org], axis=0)

    model_pool = Pool(data=X_train, label=y_train, cat_features=cat_cols_cat)
    eval_pool = Pool(data=X_test, label=y_test, cat_features=cat_cols_cat)
            
    cat_md = CatBoostClassifier(**cat_params).fit(model_pool, eval_set=eval_pool, verbose=0)
    preds = cat_md.predict_proba(X_test)[:, 1]

    score = roc_auc_score(y_test, preds)
    print(f"The oof ROC-AUC score is {score}")
    scores.append(score)
    
    oof_preds = pd.DataFrame()
    oof_preds['y'] = y_test.values
    oof_preds['cat_preds'] = preds
    oof_preds['fold'] = i
    cat_oof_preds_1.append(oof_preds)
    
    test_preds = pd.DataFrame()
    test_preds['cat_preds'] = cat_md.predict_proba(test_pool)[:, 1]
    test_preds['fold'] = i
    cat_test_preds_1.append(test_preds)

cat_oof_score_1 = np.mean(scores)  
cat_std_1 = np.std(scores)
print(f"The 10-fold average oof ROC-AUC score of the CatBoost model is {cat_oof_score_1}")
print(f"The 10-fold std oof ROC-AUC score of the CatBoost model is {cat_std_1}")


%%time
oof_preds = pd.concat(gb_oof_preds)
oof_preds['ydf_preds'] = pd.concat(ydf_oof_preds)['ydf_preds']
oof_preds['lgb_preds'] = pd.concat(lgb_oof_preds)['lgb_preds']
oof_preds['lgb_dart_preds'] = pd.concat(lgb_dart_oof_preds)['lgb_dart_preds']
oof_preds['lgb_goss_preds'] = pd.concat(lgb_goss_oof_preds)['lgb_goss_preds']
oof_preds['xgb_preds'] = pd.concat(xgb_oof_preds)['xgb_preds']
oof_preds['cat_preds'] = pd.concat(cat_oof_preds)['cat_preds']
oof_preds['cat_preds_1'] = pd.concat(cat_oof_preds_1)['cat_preds']


def objective(trial):
    
    weights = [trial.suggest_float(f"weight{n}", 1e-5, 1) for n in range(8)]

    scores = list()
    for i in range(0, 10):
        
        x_test = oof_preds[oof_preds['fold']==i].reset_index(drop=True)
        ens_pred = (weights[0]*x_test['gb_preds'].values +
                    weights[1]*x_test['ydf_preds'].values +
                    weights[2]*x_test['lgb_preds'].values +
                    weights[3]*x_test['lgb_dart_preds'].values +
                    weights[4]*x_test['lgb_goss_preds'].values +
                    weights[5]*x_test['xgb_preds'].values + 
                    weights[6]*x_test['cat_preds'].values +
                    weights[7]*x_test['cat_preds_1'].values) 
        
        y_test = x_test['y']
        score = roc_auc_score(y_test, ens_pred)
        scores.append(score)
    
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=3000, n_jobs=-1)


%%time
print("Best Trial:")
best_trial = study.best_trial

print(f"  Value: {best_trial.value}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")


%%time
w = study.best_trial.params
scores = list()
for i in range(0, 10):
    
    x_train = oof_preds[oof_preds['fold']!=i].reset_index(drop=True)
    x = x_train.drop(columns=['fold', 'y'], axis=1)
    y = x_train['y']
    
    test = oof_preds[oof_preds['fold']==i].reset_index(drop=True)
    x_test = test.drop(columns=['fold', 'y'], axis=1)
    y_test = test['y']

    optuna_pred = (w['weight0']*test['gb_preds'] + w['weight1']*test['ydf_preds'] +
                   w['weight2']*test['lgb_preds'] + w['weight3']*test['lgb_dart_preds'] + 
                   w['weight4']*test['lgb_goss_preds'] + w['weight5']*test['xgb_preds'] + 
                   w['weight6']*test['cat_preds'] + w['weight7']*test['cat_preds_1'])
    score = roc_auc_score(y_test, optuna_pred)
    scores.append(score)

print(f"The 10-fold oof average ROC-AUC score of the Optuna Blender is {np.mean(scores)}")



results = pd.DataFrame()
results['Model'] = ['GB', 'YDF', 'LGBM', 'LGBM-dart', 'LGBM-goss', 'XGB', 'CatBoost', 'CatBoost-1', 'Optuna Blend']
results['10-fold oof ROC-AUC'] = [gb_oof_score, ydf_oof_score, lgb_oof_score, lgb_dart_oof_score, lgb_goss_oof_score, xgb_oof_score, cat_oof_score, cat_oof_score_1, np.mean(scores)] 
print(results)


%%time
test_preds = pd.concat(gb_test_preds)
test_preds['ydf_preds'] = pd.concat(ydf_test_preds)['ydf_preds']
test_preds['lgb_preds'] = pd.concat(lgb_test_preds)['lgb_preds']
test_preds['lgb_dart_preds'] = pd.concat(lgb_dart_test_preds)['lgb_dart_preds']
test_preds['lgb_goss_preds'] = pd.concat(lgb_goss_test_preds)['lgb_goss_preds']
test_preds['xgb_preds'] = pd.concat(xgb_test_preds)['xgb_preds']
test_preds['cat_preds'] = pd.concat(cat_test_preds)['cat_preds']
test_preds['cat_preds_1'] = pd.concat(cat_test_preds_1)['cat_preds']
test_preds.head()


%%time
test_pred_final = list()
for i in range(0, 10):
    
    temp = test_preds[test_preds['fold']==i].reset_index(drop=True)
    optuna_pred = (w['weight0']*temp['gb_preds'] + w['weight1']*temp['ydf_preds'] + 
                   w['weight2']*temp['lgb_preds'] + w['weight3']*temp['lgb_dart_preds'] +
                   w['weight4']*temp['lgb_goss_preds'] + w['weight5']*temp['xgb_preds'] + 
                   w['weight6']*temp['cat_preds'] + w['weight7']*temp['cat_preds_1'])
    
    test_pred_final.append(optuna_pred)


%%time
optuna_preds = np.mean(test_pred_final, axis=0)
submission = pd.read_csv('../input/playground-series-s4e10/sample_submission.csv')
submission['loan_status'] = optuna_preds
submission.head()


%%time
submission.to_csv('baseline_sub_2.csv', index=False)


%%time
Ridge_scores = list()
test_pred_final = list()

for i in range(0, 10):
    
    print(f"-------- Fold {i} --------")

    x_train = oof_preds[oof_preds['fold']!=i].reset_index(drop=True)
    x = x_train.drop(columns=['fold', 'y'], axis=1)
    y = x_train['y']
    
    test = oof_preds[oof_preds['fold']==i].reset_index(drop=True)
    x_test = test.drop(columns=['fold', 'y'], axis=1)
    y_test = test['y']

    Ridge_md = Ridge(alpha=10).fit(x, y)
    Ridge_pred = Ridge_md.predict(x_test)
    Ridge_score = roc_auc_score(y_test, Ridge_pred)
    print(f"Fold {i}, Ridge ROC-AUC {Ridge_score}")
    Ridge_scores.append(Ridge_score)

    temp = test_preds[test_preds['fold']==i].drop('fold', axis=1).reset_index(drop=True)
    test_pred = Ridge_md.predict(temp)
    test_pred_final.append(test_pred)

print(f"The 10-fold oof Ridge ROC-AUC {np.mean(Ridge_scores)}")



%%time
Ridge_preds = np.mean(test_pred_final, axis=0)
submission = pd.read_csv('../input/playground-series-s4e10/sample_submission.csv')
submission['loan_status'] = Ridge_preds
submission.head()


%%time
submission.to_csv('baseline_sub_3.csv', index=False)


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold
import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras import backend as K


TARGET_NAMES = ['loan_status']
TARGET_NAME = 'loan_status'

# Load datasets
train_data = pd.read_csv('../input/playground-series-s4e10/train.csv').assign(source=0)
test_data = pd.read_csv('../input/playground-series-s4e10/test.csv').assign(source=0)
original_data = pd.read_csv('../input/loan-approval-prediction/credit_risk_dataset.csv').assign(source=1)

#these are also in the original and have a status of 1:
train_data.loc[train_data['id'].isin([37066, 48973]), 'loan_status'] = 1
original_data['id'] = np.arange(len(original_data)) - 1E6
# matched with train
original_data = original_data.query('id not in [-999794, -993298]')

def to_rank(col):
    # descritise from 0..N
    return col.fillna(-1).rank(method='dense').astype('int') - 1

def fe(df):
    cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']    
    # treat continuous as categorical ranks:
    df['cb_person_cred_hist_length'] = to_rank(df['cb_person_cred_hist_length'])
    df['loan_amnt'] = to_rank(df['loan_amnt'])
    df['person_income'] = to_rank(df['person_income'])
    df['loan_int_rate'] = to_rank(df['loan_int_rate'])
    df['person_emp_length'] = to_rank(df['person_emp_length'])
    df['loan_percent_income'] = to_rank(df['loan_percent_income'])
    df['person_age'] = to_rank(df['person_age'])
    for col in cat_cols:
        # count + rank encoding, less to more frequent:
        col_series = df[col].fillna('#NA#')
        mapping = col_series.value_counts().to_dict()
        code_as = 0
        for i, key in enumerate(reversed(mapping)):
            mapping[key] = code_as
            code_as += 1
        df[col] = col_series.map(mapping)
        df[col] = df[col].astype('int')
    return df


df_all = fe(pd.concat([train_data, test_data, original_data]))

idxs = (~df_all[TARGET_NAMES[0]].isna()) & (df_all['source'] == 0)
train_data = df_all[idxs].reset_index(drop=True)
idxs = ( df_all[TARGET_NAMES[0]].isna()) & (df_all['source'] == 0)
test_data = df_all[idxs].drop(columns=[TARGET_NAMES[0]])
original_data = df_all.query('source == 1')        

cont_features = ['cb_person_default_on_file', 'source']
cat_features = [
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'person_emp_length',
    'loan_int_rate',
    'loan_percent_income',
    'person_age',
    'person_income',
    'loan_amnt',
    'cb_person_cred_hist_length']

cat_features_card = {}
for f in cat_features:
    cat_features_card[f] = 1 + df_all[f].max()

df_all = None 

features = cat_features + cont_features

def build_model(cat_features, cont_features):

    # Define input layers
    cat_inputs = [layers.Input(shape=(1,), name=f'cat{i}') for i in range(len(cat_features))]
    cont_inputs = layers.Input(shape=(len(cont_features),))
                                
    # Embedding layers for categorical inputs
    flat_embeddings = []
    for i, f in enumerate(cat_features):
        input_dim = int(cat_features_card[f])
        output_dim = int(min(128, round(1.6 * input_dim ** .56))) # based on the fastai library

        embedding = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)(cat_inputs[i])
        if output_dim > 32:
            embedding = layers.SpatialDropout1D(.5)(embedding)
        else:
            embedding = layers.SpatialDropout1D(.3)(embedding)
        flat_embeddings.append(layers.Flatten()(embedding))
                                
    concatenated_inputs = layers.Concatenate()(flat_embeddings + [cont_inputs, ])
    concatenated_inputs_bn = layers.BatchNormalization()(concatenated_inputs)

    x = layers.Dense(256, activation='mish')(concatenated_inputs_bn) 

    for units in (128,): 
        inp = layers.Concatenate()([x, concatenated_inputs_bn])
        x = layers.Dense(units=units, activation='mish')(inp)
        x = layers.Dropout(.3)(x) 

    # output layer
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(cat_inputs + [cont_inputs], outputs)

epochs = 8
callbacks = []

def fold_logloss(y, preds):
    return log_loss(y, preds)

def fold_auc(y, preds):
    return roc_auc_score(y, preds)

# to feed data into the NN
# we feed the categoricals column by column,
# and the continuous features in one lump.
cat_idxs= []
cont_idxs = []
for f in cat_features:
    cat_idxs.append([features.index(f)])
for f in cont_features:
    cont_idxs.append(features.index(f))
    
feature_idxs = cat_idxs + [cont_idxs]

def to_nn_feed(df):
    X = df[feats].values
    result = []
    for f_idx in feature_idxs:
        # housekeeping: to feed data into the NN
        # we feed the categoricals column by column,
        # and the continuous features in one lump.
        result.append(X[:, f_idx])
    return result

def fit_fold(tr, vl, ts):

    model = build_model(cat_features, cont_features)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=3E-4),
        loss='binary_crossentropy',
        metrics=[keras.metrics.AUC()])

    model.fit(
          to_nn_feed(tr), tr[TARGET_NAME],
          validation_data=(to_nn_feed(vl), vl[TARGET_NAME]),
          batch_size=256,
          epochs=epochs,
          callbacks=callbacks,
          verbose=0
    )

    vl_pred = model.predict(to_nn_feed(vl), verbose=0, batch_size=256).flatten()
    ts_pred = model.predict(to_nn_feed(ts), verbose=0, batch_size=256).flatten()
    
    vl_metric = fold_auc(vl[TARGET_NAME], vl_pred)
    return vl_pred, ts_pred, vl_metric


%%time
feats = features
N_FOLDS = 10

keras.utils.set_random_seed(1)
scores, tf_oof_preds, tf_test_preds = list(), list(), list()
skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
for i, (train_index, test_index) in enumerate(skf.split(train_data, train_data[TARGET_NAME])):
    tr = train_data.loc[train_index]
    vl = train_data.loc[test_index]

    # add original data to the training fold only:
    vl_pred, ts_pred, vl_metric = fit_fold(pd.concat([tr, original_data]), vl, test_data)
    
    print(f"Fold {i}, the oof ROC-AUC score is {vl_metric}")
    scores.append(vl_metric)

    oof_preds_tf = pd.DataFrame()
    oof_preds_tf['y'] = vl[TARGET_NAME].values
    oof_preds_tf['tf_preds'] = vl_pred
    oof_preds_tf['fold'] = i
    oof_preds_tf['index'] = test_index
    tf_oof_preds.append(oof_preds_tf)

    test_preds_tf = pd.DataFrame()
    test_preds_tf['tf_preds'] = ts_pred
    test_preds_tf['fold'] = i
    tf_test_preds.append(test_preds_tf)

print(f"The 10-fold oof TF ROC-AUC score is {np.mean(scores)}")


%%time
oof_preds['tf_preds'] = pd.concat(tf_oof_preds)['tf_preds']
test_preds['tf_preds'] = pd.concat(tf_test_preds)['tf_preds']

oof_preds.to_csv('oof_preds.csv', index=False)
test_preds.to_csv('test_preds.csv', index=False)


%%time
LR_scores = list()
Ridge_scores = list()
mlp_scores = list()
ens_scores = list() 

test_pred_final_LR = list()
test_pred_final_Ridge = list()
test_pred_final_MLP = list()
test_pred_final_ens = list()

for i in range(0, 10):
    
    print(f"-------- Fold {i} --------")

    x_train = oof_preds[oof_preds['fold']!=i].reset_index(drop=True)
    x = x_train.drop(columns=['fold', 'y'], axis=1)
    y = x_train['y']
    
    test = oof_preds[oof_preds['fold']==i].reset_index(drop=True)
    x_test = test.drop(columns=['fold', 'y'], axis=1)
    y_test = test['y']

    LR_md = LogisticRegression(C=0.01).fit(x, y)
    LR_pred = LR_md.predict_proba(x_test)[:, 1]
    LR_score = roc_auc_score(y_test, LR_pred)
    print(f"Fold {i}, LR ROC-AUC {LR_score}")
    LR_scores.append(LR_score)

    Ridge_md = Ridge().fit(x.drop(columns=['lgb_preds', 'gb_preds'], axis=1), y)
    Ridge_pred = Ridge_md.predict(x_test.drop(columns=['lgb_preds', 'gb_preds'], axis=1))
    Ridge_score = roc_auc_score(y_test, Ridge_pred)
    print(f"Fold {i}, Ridge ROC-AUC {Ridge_score}")
    Ridge_scores.append(Ridge_score)    

    mlp_md = MLPClassifier(hidden_layer_sizes=(100, 100), 
                           activation='logistic',  
                           max_iter=1000, 
                           random_state=1).fit(x, y)
    mlp_pred = mlp_md.predict_proba(x_test)[:, 1]
    mlp_score = roc_auc_score(y_test, mlp_pred)
    print(f"Fold {i}, MLP ROC-AUC {mlp_score}")
    mlp_scores.append(mlp_score)

    ens_pred = 0.1*LR_pred + 0.5*Ridge_pred + 0.4*mlp_pred
    ens_score = roc_auc_score(y_test, ens_pred)
    print(f"Fold {i}, Ensemble ROC-AUC {ens_score}")
    ens_scores.append(ens_score)

    temp = test_preds[test_preds['fold']==i].drop('fold', axis=1).reset_index(drop=True)
    LR_test = LR_md.predict_proba(temp)[:, 1]
    test_pred_final_LR.append(LR_test)
    
    Ridge_test = Ridge_md.predict(temp.drop(columns=['lgb_preds', 'gb_preds'], axis=1))
    test_pred_final_Ridge.append(Ridge_test)
    
    MLP_test = mlp_md.predict_proba(temp)[:, 1]
    test_pred_final_MLP.append(MLP_test)
    
    test_pred = 0.1*LR_test + 0.5*Ridge_test + 0.4*MLP_test
    test_pred_final_ens.append(test_pred)

print('\n')
print(f"The 10-fold oof LR ROC-AUC {np.mean(LR_scores)}")
print(f"The 10-fold oof Ridge ROC-AUC {np.mean(Ridge_scores)}")
print(f"The 10-fold oof MLP ROC-AUC {np.mean(mlp_scores)}")
print(f"The 10-fold oof Ensemble ROC-AUC {np.mean(ens_scores)}")


%%time
submission = pd.read_csv('../input/playground-series-s4e10/sample_submission.csv')
submission['loan_status'] = np.mean(test_pred_final_LR, axis=0)
submission.to_csv('baseline_sub_4_LR.csv', index=False)

submission = pd.read_csv('../input/playground-series-s4e10/sample_submission.csv')
submission['loan_status'] = np.mean(test_pred_final_Ridge, axis=0)
submission.to_csv('baseline_sub_4_Ridge.csv', index=False)

submission = pd.read_csv('../input/playground-series-s4e10/sample_submission.csv')
submission['loan_status'] = np.mean(test_pred_final_MLP, axis=0)
submission.to_csv('baseline_sub_4_MLP.csv', index=False)

submission = pd.read_csv('../input/playground-series-s4e10/sample_submission.csv')
submission['loan_status'] = np.mean(test_pred_final_ens, axis=0)
submission.to_csv('baseline_sub_4_ensemble.csv', index=False)

