#basics
import numpy as np
import pandas as pd 
import polars as pl
import seaborn as sns
import time
import matplotlib.pyplot as plt
import missingno as msno
pd.set_option('display.max_columns', 100)

import warnings
warnings.filterwarnings("ignore")

#preprocessing
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, LabelEncoder,OneHotEncoder, OrdinalEncoder

#feature engineering
from sklearn.feature_selection import mutual_info_classif


#transformers and pipeline
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn import set_config

#algorithms
from xgboost import XGBClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from catboost import Pool
from lightgbm import LGBMClassifier
from lightgbm import early_stopping
from lightgbm import log_evaluation
from sklearn.linear_model import LogisticRegression


#model evaluation
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, auc, accuracy_score, balanced_accuracy_score
from sklearn.metrics import make_scorer, RocCurveDisplay, confusion_matrix

# Optuna and visualization tools
import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

random_state = 42


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
print('train shape = ', train.shape)
orig_cols = train.columns[1:-1]
train.tail()


test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
print('test shape = ', test.shape)
test.head()


rainfall = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")
print("rainfall shape = ", rainfall.shape)
rainfall.head()


rainfall['rainfall'] = [1 if i == 'yes' else 0 for i in rainfall['rainfall']]
rainfall.head()


del train['id']
del test['id']
rainfall.columns = rainfall.columns.str.strip()
print(rainfall.columns)
print(train.columns)
orig_cols = [i for i in orig_cols if i !='day']
print(orig_cols)


train_df = pd.concat([train, rainfall], ignore_index=True, axis = 0)

plt.figure(figsize=(10, 10))
palette_color = sns.color_palette('pastel')
explode = [0.05, 0.05]

# Plotting
train_df.groupby('rainfall')['rainfall'].count().plot.pie(
    colors=palette_color,
    explode=explode,
    autopct="%1.1f%%",
    shadow=True,  # Adding shadow for better visibility
    startangle=140,  # Start angle for better alignment
    textprops={'fontsize': 14},  # Adjust text size
    wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}  # Adding edge color and width
)

# Adding a title
plt.title('Target Distribution', fontsize=18, weight='bold')

# Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal')

# Displaying the plot
plt.show()


feature = [i for i in train_df.columns if i != 'rainfall']


train_df['winddirection'].fillna(train_df['winddirection'].median(), inplace=True)
train_df['windspeed'].fillna(train_df['windspeed'].median(), inplace=True)
test['winddirection'].fillna(test['winddirection'].median(), inplace=True)
test['windspeed'].fillna(test['windspeed'].median(), inplace=True)


mutual_info = mutual_info_classif(train_df[feature], train_df.rainfall, random_state=random_state)
mutual_info_series = pd.Series(mutual_info)
mutual_info_series.index = feature  
mutual_info_df = pd.DataFrame(mutual_info_series.sort_values(ascending=False), columns=["Numerical_Feature_MI"])
styled_mutual_info = mutual_info_df.style.background_gradient("cool")
styled_mutual_info


def data_engineering(df):    
    
    df['sin_day'] = np.sin(2 * np.pi * df['day'] / 365)
    df['cos_day'] = np.cos(2 * np.pi * df['day'] / 365)
    
    df['tan_day'] = np.tan(2 * np.pi * df['day'] / 365)
    
    df['arcsin_day'] = np.arcsin(np.clip(df['day'] / 365, -1, 1))
    df['arccos_day'] = np.arccos(np.clip(df['day'] / 365, -1, 1))
    df['arctan_day'] = np.arctan(df['day'] / 365)
    def get_season(day):
        if day >= 335 or day <= 59:  
            return '0' #winter
        elif 60 <= day <= 151:  
            return '1' #spring
        elif 152 <= day <= 243:  
            return '2'#summer
        else:  
            return '3' #autumn
    df['season'] = df['day'].apply(get_season)
    # del df['day'] 
    return df


def temp_engineering(df):
    df['temp_diff'] = df['maxtemp'] - df['mintemp']
    df['dew_temp_diff'] = df['dewpoint'] - df['temparature']
    return df


def cloud(df):
    df['humidity_windspeed'] = df['humidity'] * df['windspeed']
    df['cloud + humidity'] = df['cloud'] + df['humidity']
    df['cloud + humidity + sunshine'] = df['cloud'] + df['humidity'] + df['sunshine']
    df['cloud * sunshine'] = df['cloud'] * df['sunshine']
    df['humidity * sunshine'] = df['humidity'] * df['sunshine']
    df['cloud^2'] = df['cloud'] ** 2
    return df


train_df = cloud(train_df)
test = cloud(test)


train_df = temp_engineering(train_df)
test = temp_engineering(test)


train_df = data_engineering(train_df)
test = data_engineering(test)


catboost_params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'learning_rate': 0.09613777604618812,
        'iterations': 1000,
        'depth': 11,
        'random_strength':0,
        'l2_leaf_reg': 7.9815276045005765,
        'task_type':'GPU',
        'random_seed':42,
        'verbose':False    
    }


train_df = train_df.dropna(subset=['rainfall'])


FOLDS = 7

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

def kfold(model_name,params, data, test_data):

    if data['rainfall'].isnull().sum() > 0:
        raise ValueError("Train target variable contains NaN values.")
    
    x = data[feature].copy()
    y = data['rainfall'].copy()
    
    oof_preds = np.zeros(len(x))
    test_preds = np.zeros(len(test))
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(data)):
        print(f"Fold {fold + 1}")
    
        X_train = x.loc[train_idx].reset_index(drop=True).copy()
        y_train = y.iloc[train_idx].values
        
        X_valid = x.loc[valid_idx].reset_index(drop=True).copy()
        y_valid = y.iloc[valid_idx].values
        
        X_test = test.reset_index(drop=True).copy()

        if model_name == 'CBC':
            model = CatBoostClassifier(**params)
            train_pool = Pool(X_train, y_train)
            valid_pool = Pool(X_valid, y_valid)
            X_test_pool = Pool(X_test)
            model.fit(X=train_pool, eval_set=valid_pool, verbose=50, early_stopping_rounds=100)

            oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]
            test_preds += model.predict_proba(test)[:, 1] / FOLDS
            
        elif model_name == 'XGBC':
            model = XGBClassifier(**params, verbose=50)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=50, verbose=50)
            best_iteration = model.best_iteration
            oof_preds[valid_idx] = model.predict_proba(X_valid, iteration_range=(50, best_iteration))[:, 1]
            test_preds += model.predict_proba(test, iteration_range=(50, best_iteration))[:, 1] / FOLDS

        elif model_name == 'LGBMC':
             eval_set = [(X_valid, y_valid)]  
             model = LGBMClassifier(**params) 
             model.fit(
                 X_train, y_train,
                 eval_set=eval_set,   
                 eval_metric='auc',  
                 callbacks=[early_stopping(50)]  
             )
             best_iteration = model.best_iteration_  
             oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]
             test_preds += model.predict_proba(X_test)[:, 1] / FOLDS
        
        print("--" * 25)
    
    ras = roc_auc_score(data['rainfall'], oof_preds)
    print(f"Validation RAS: {ras}")
    return test_preds


train_df = train_df.reset_index(drop=True)
test = test.reset_index(drop=True)


test_preds1 = kfold('CBC',catboost_params, train_df, test)


# x_train, x_test, y_train, y_test = train_test_split(train_df[feature], train_df['rainfall'], random_state = 42)

# def objective(trial):
#     params = {
#         'n_estimators' : trial.suggest_int('n_estimators', 50, 500),
#         'eta' : trial.suggest_float('eta', 0.001, 0.5),
#         'alpha' : trial.suggest_float('alpha', 0.0, 1.0),
#         'subsample' : trial.suggest_float('subsample', 0.1, 1.0),
#         'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.1, 1.0),
#         'max_depth' : trial.suggest_int('max_depth', 1, 10),
#         'min_child_weight' : trial.suggest_int('min_child_weight', 1, 10),
#         'gamma' : trial.suggest_float('gamma', 0.0, 5.0),
#         'max_bin' : trial.suggest_int('max_bin', 128, 25000),
#         # 'tree_method': 'gpu_hist',
#         'eval_metric': 'auc',
#         'objective': 'binary:logistic',
#         'verbose' : 0
#     }

#     model = XGBClassifier(**params)
#     model.fit(x_train, y_train)
#     preds = model.predict_proba(x_test)[:, 1]  
#     roc_auc = roc_auc_score(y_test, preds)

#     return roc_auc
# sampler = TPESampler()
# study = optuna.create_study(sampler=sampler, direction='maximize')

# study.optimize(objective, n_trials=100)
# print("Лучшие гиперпараметры:", study.best_params)
# print("Лучшая AUC:", study.best_value)


# xgbc_par = {'n_estimators': 125,
#             'eta': 0.060077076447574214,
#             'alpha': 0.10829670110870329,
#             'subsample': 0.7809600859296395,
#             'colsample_bytree': 0.16241886071372014,
#             'max_depth': 3,
#             'min_child_weight': 9,
#             'gamma': 1.6438795194765943,
#             'max_bin': 17973,
#             'tree_method': 'gpu_hist',
#             'eval_metric': 'auc',
#             'objective': 'binary:logistic'
#            }
# test_preds2 = kfold('XGBC', xgbc_par, train_df, test)


# x_train, x_test, y_train, y_test = train_test_split(train_df[feature], train_df['rainfall'], random_state = 42)
# def objective(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 50, 500),
#         'max_depth': trial.suggest_int('max_depth', 1, 15),
#         'num_leaves': trial.suggest_int('num_leaves', 5, 400),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
#         'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
#         'random_state': random_state,
#         "eval_metric": "auc",
#         'verbose': -1,
#     }
#     model = LGBMClassifier(**params)
#     scores = cross_val_score(model, x_train, y_train, cv=5, scoring='roc_auc')
#     auc_score = scores.mean()
    
#     return auc_score

# sampler = TPESampler(seed=random_state)
# study = optuna.create_study(direction='maximize', sampler=sampler)
# study.optimize(objective, n_trials=100)
# print("Лучшие гиперпараметры:", study.best_params)
# print("Лучшая AUC:", study.best_value)


# l_par = {'n_estimators': 232,
#          'max_depth': 1,
#          'num_leaves': 303,
#          'learning_rate': 0.25101263781938554,
#          'min_child_samples': 14,
#          'subsample': 0.746858325807502,
#          'colsample_bytree': 0.9082677247457509,
#          'lambda_l1': 2.142983876868706e-06,
#          'lambda_l2': 2.487359042523012,
#          'random_state': random_state,
#          "eval_metric": "auc",
#           'verbose': -1,}

# test_preds3 = kfold('LGBMC', l_par, train_df, test)


# fig, ax = plt.subplots(figsize=(15, 20))

# xgb.plot_importance(model, max_num_features=50, importance_type='gain', ax=ax, title="Top Features Importances (XGBoost)")

# # ticks = ax.get_yticklabels()
# # feature_names = [train.columns[int(tick.get_text().replace('f', ''))] for tick in ticks]

# # ax.set_yticklabels(feature_names) 

# plt.show()


# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(train_pool)

# shap.summary_plot(
#     shap_values, 
#     train, 
#     plot_type="bar", 
#     class_names=np.unique(rainfall),
#     color='purple',
#     show=False
# )
# plt.xticks(fontsize=14)  
# plt.yticks(fontsize=14)  
# plt.xlabel('Mean Absolute SHAP Value', fontsize=14) 
# plt.title('Feature Importance by SHAP Values', fontsize=16) 
# plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5) 
# plt.show()


sub = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
sub.rainfall = test_preds1
sub.to_csv("submission.csv", index=False)
!head submission.csv


# fig, ax = plt.subplots(figsize=(10, 30))
# importance = model.feature_importances_
# # feature_names = train.drop('target', axis=1).columns
# plt.barh(train.columns, importance)
# plt.xlabel("Feature Importance")
# plt.ylabel("Feature Name")
# plt.show()


# from xgboost import Booster
# importance_gain = model.get_booster().get_score(importance_type='gain')
# importance_dict = {feature: importance for feature, importance in importance_gain.items()}
# sorted_importance_dict = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
# feature_names = train.columns.tolist()
# importance_dict_with_names = {feature_names[int(k[1:])]: v for k, v in importance_dict.items()}
# sorted_importance_dict_with_names = dict(sorted(importance_dict_with_names.items(), key=lambda item: item[1], reverse=True))
# print(sorted_importance_dict_with_names)




