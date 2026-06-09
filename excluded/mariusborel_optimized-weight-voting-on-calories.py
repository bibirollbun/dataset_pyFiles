import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use('ggplot')
# change default colormap
plt.rcParams['image.cmap'] = 'Dark2'
import seaborn as sns
import plotly_express as px
from scipy.stats import iqr
from scipy.cluster import hierarchy
import random
import warnings
warnings.simplefilter('ignore')

# Import the various sklear tools
# from sklearn.utils import resample
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.feature_selection import mutual_info_regression, SequentialFeatureSelector as sk_sfs
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score,
                                     StratifiedKFold)
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool
from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer,
                                   PowerTransformer, QuantileTransformer,
                                   RobustScaler, StandardScaler, minmax_scale,
                                   LabelEncoder, OneHotEncoder, FunctionTransformer)
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, make_scorer, mean_squared_error, r2_score, mean_squared_log_error


from sklearn.decomposition import PCA, NMF
from sklearn.pipeline import make_pipeline, Pipeline
# from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif, SelectKBest, RFE, chi2
from sklearn.model_selection import cross_val_score

from sklearn.inspection import PartialDependenceDisplay, permutation_importance

import xgboost as xgb
from xgboost import XGBRegressor, plot_importance, cv

# from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter

import umap

import optuna
from optuna.samplers import TPESampler

# facecolor = '#2f4f4f'
facecolor = '#080808'
chartcolor = '#deb887'

print('The moduls and tools are all loaded. Some are not used in this version of the notebook but had been used in previous or will be used in later versions')


def rmsle(y_true, y_pred):
    # Ensure the inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate the logarithm of the true and predicted values
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    
    # Calculate the squared differences
    squared_diff = np.square(log_true - log_pred)
    
    # Calculate the mean of the squared differences
    mean_squared_diff = np.mean(squared_diff)
    
    # Calculate the root of the mean squared differences
    rmsle_value = np.sqrt(mean_squared_diff)
    
    return rmsle_value


train_raw = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col='id')
test_raw = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col='id')
orig_raw = pd.read_csv('/kaggle/input/calories-burnt-prediction/calories.csv').drop(columns=['User_ID'])
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

target = 'Calories'

orig_raw.columns = train_raw.columns
train_raw.head(3)


plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1)
ax_1 = sns.kdeplot(train_raw, x=target, fill=True, palette=plt.cm.Set3)
ax_1.set_facecolor(facecolor)
ax_1.grid(False)
plt.subplot(1, 2, 2)
ax_2 = sns.kdeplot(train_raw, x=target, hue='Sex', fill=True, palette=['grey', chartcolor])
ax_2.set_facecolor(facecolor)
ax_2.grid(False)
plt.tight_layout()


for feat in train_raw.columns.tolist()[1:-1]:
    plt.figure(figsize=(10, 3.2))
    plt.subplot(1, 3, 1)
    ax_1 = sns.boxenplot(train_raw, x=feat, y='Sex', color=chartcolor)
    # Set background color
    ax_1.set_facecolor(facecolor)
    ax_1.grid(False)
    plt.ylabel(feat)
    plt.subplot(1, 3, 2)
    ax_2 = sns.histplot(train_raw[train_raw['Sex']=='male'][feat], color='grey', bins=30, kde=True)
    ax_2 = sns.histplot(train_raw[train_raw['Sex']=='female'][feat], color=chartcolor, bins=30, kde=True)
    # Set background color
    ax_2.set_facecolor(facecolor)
    ax_2.grid(False)
    plt.subplot(1, 3, 3)
    ax_3 = sns.scatterplot(train_raw, x=feat, y=target, hue='Sex', palette=[ 'grey', chartcolor])
    # Set background color
    ax_3.set_facecolor(facecolor)
    ax_3.grid(False)
    plt.suptitle(f'Distribution of {feat}')
    plt.tight_layout()


train_data = train_raw.copy()
train_target = train_data.pop(target)


plt.figure(figsize=(8,4))

ax = train_data.iloc[:, 1:].corrwith(train_target).sort_values(
                                        ascending=True
                                              ).plot.barh(
                                                  color=chartcolor, 
                                                  title='Features correlation with the target',
                                                         )
plt.xlim([-0.33, 1.2])
# Set background color
ax.set_facecolor(facecolor)
ax.grid(False)
for label in ax.containers:
    ax.bar_label(label,color=chartcolor, fontsize=14)
plt.show()


corr = train_raw.corr(numeric_only=True).abs()

# Mask values that is not fall between min and max value
corr_selected = corr.mask(((corr < 0.9) | (corr > 1)), float("NaN"))

plt.figure(figsize=(8, 6))
sns.heatmap(corr_selected, cmap='Greens', cbar=False, annot=True, square=True)
plt.show()


create_new_features = True

# Define the preprocessor for new features
def featEng(df):
    if create_new_features:
        df['Max_heart_rate'] = 207 - 0.7*df['Age']
        df['%_heart_rate'] = df['Heart_Rate']/df['Max_heart_rate']
        df['heart_rate_/_weight'] = df['Heart_Rate']/df['Weight']
        df['heart_rate_/_Age'] = df['Heart_Rate']/df['Age']
        df['Heart_Rate_x_Duration'] = df['Heart_Rate']*df['Duration']
        df['Heart_Rate_x_Duration__log'] = np.log1p(df['Heart_Rate']*df['Duration'])
        df['bmi'] = df['Weight']/(0.01*df['Height'])**2
        df['log_weight_x_height'] = np.log(df['Weight']*df['Height'])
        df['body_temp_x_weight'] = df['Body_Temp']*np.log(df['Weight'])
        df['Body_Temp - 37'] = df['Body_Temp'] - 37 # assuming 37 as body temp at rest
        df['heat'] = df['Weight']*df['Body_Temp - 37']
        df['work'] = df['heat']*df['Duration']/60
    #    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100], labels=[1, 2, 3, 4, 5]).astype('int')
      #  df['Age_pred_max_heart_rate'] = pd.cut(df['Age'], 
           #                                    bins=[10, 30, 35, 40, 45, 50, 55, 60, 65, 70, 90], 
         #                                      labels=[200, 190, 185, 180, 175, 170, 165, 160, 155, 150]
            #                                  ).astype('int')
      #  df['diff_heart_rate'] = df['Max_heart_rate'] - df['Age_pred_max_heart_rate']
        df['Sex'] = (df['Sex'] == 'female')*1
       # df['Temp_Weight_ratio'] = df['Body_Temp']/df['Weight']
       # df['HeartRate_BodyTemp_ratio'] = df['Heart_Rate']/df['Body_Temp']
        df = df.drop(columns=['Duration'])
    else:
        df['Sex'] = (df['Sex'] == 'female')*1
        df['Heart_Rate_x_Duration__log'] = np.log1p(df['Heart_Rate']*df['Duration'])
        df = df.drop(columns=['Duration'])
    
    return df


def DataTargetPrep(df):
    data_ = df.copy()
    data_ = featEng(data_)
    try:
        target_ = data_.pop(target)
        target_log = np.log1p(target_)
    except:
        pass

    try:
        return data_, target_, target_log
    except:
        return data_


# Should the external dataset be included?
include_orig_data = True
if include_orig_data:
    train_set = pd.concat([train_raw, orig_raw], ignore_index=True)
else:
    train_set = train_raw

# Prep the train data and target
train_prep = train_set.copy()
X_train_prep, y_train, y_train_log = DataTargetPrep(train_prep)


# Prep the test data
test_prep = test_raw.copy()
X_test_prep = DataTargetPrep(test_prep)


X_train_prep.head()


X_test_prep.head()


corr = X_train_prep.corr(numeric_only=True).abs()

# Mask values that is not fall between min and max value
corr_selected = corr.mask(((corr < 0.6) | (corr > 1)), float("NaN"))

plt.figure(figsize=(10,8))
sns.heatmap(corr_selected, cmap='Greens', cbar=False, annot=True)
plt.show()


# Calculate mutual information with a fraction of the train dataset
mi = mutual_info_regression(X_train_prep, y_train)

mi_df = pd.DataFrame({'mi':mi}, index=X_train_prep.columns)

plt.figure(figsize=(10, 8))
ax = mi_df.sort_values(by='mi').plot.barh(title='Mutual information in train dataset', color=facecolor)
ax.set_facecolor = facecolor
plt.xlim([0, 2.3])
for label in ax.containers:
    ax.bar_label(label)
plt.show()


# Separate the train set from the validation and test sets
X_tr, X_va_ts, y_tr, y_va_ts = train_test_split(X_train_prep, y_train, 
                                                train_size=0.6, random_state=81545)
# Separate the validation and test sets
X_va, X_ts, y_va, y_ts = train_test_split(X_va_ts, y_va_ts, 
                                          test_size=0.5, random_state=81545)

# Chech the size of the sets
[d.shape for d in [X_tr, y_tr, X_va, y_va, X_ts, y_ts]]


cat_params = {'iterations': 730, 
             'learning_rate': 0.09969738378973637, 
             'colsample_bylevel': 0.8121910932765485, 
             'random_strength': 0.2315038504123904, 
             'depth': 14, 
             'bootstrap_type': 'Bayesian', 
             'bagging_temperature': 0.2878598947199843}


estimators = [
              ('XGBoost', XGBRegressor(n_estimators=600, max_depth=8, verbosity=0)), 
              ('LGBM', LGBMRegressor(n_estimators=1000,max_depth=6, learning_rate=0.02, verbose=-1)), 
              ('CatBoost', CatBoostRegressor(**cat_params, verbose=0, eval_metric='RMSE')) 
]


for est_name, est in estimators:
    est.fit(X_tr, y_tr)
    pred = np.clip(est.predict(X_va), 0, orig_raw[target].max()) # clip to prevent negative values
    valid_rmsle = np.sqrt(mean_squared_log_error(y_va, pred))
    print('{} valid_rmsle: {:.6f}'.format(est_name, valid_rmsle))


def model_features_importance(model, model_name, color):
    model = model
    model.fit(X_tr, y_tr)
    model_feat_importances = pd.DataFrame({'Features':X_tr.columns, 
                                         'Importances':model.feature_importances_}
                                       ).sort_values(by='Importances', ascending=False)

    plt.figure(figsize=(8, 4))
    ax = sns.barplot(model_feat_importances, y='Features', x='Importances', color=color)
    for label in ax.containers:
        ax.bar_label(label)
    plt.title(f'Features importance in {model_name}')
    plt.show()


colors = ['green', 'orange', 'maroon', 'steelblue', chartcolor, facecolor]
for est_name, est in estimators:
    # color_ = random.shuffle(colors)
    color_ = random.choice(colors)
    model_features_importance(est, est_name, color_)


def objective(trial):
    # Sample weights for the three estimators
    weight_xgb = trial.suggest_int('weight_xgb', 2, 8, 1)
    weight_lgb = trial.suggest_int('weight_lgb', 0, 5, 1)
    weight_cat = trial.suggest_int('weight_cat', 4, 12, 1)

    # Create the Voting Regressor
    weights=[weight_xgb, weight_lgb,  weight_cat] # The weights
    
    vr = VotingRegressor(estimators=estimators, weights=weights, n_jobs=-1) # The voting regressor

    # Fit the model and calculate predictions
    vr.fit(X_tr, y_tr)
    y_pred = vr.predict(X_va)

    # Calculate mean squared error (you can use any other metric)
    score = np.sqrt(mean_squared_log_error(y_va, y_pred))
    return score


# Define the function to run optuna optimization
def Run_Pass_cat_study(n_trials=1):
    if n_trials>1:
        # Optimize using Optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get the best weights
        best_weights = study.best_params

    else:
        print('No need to run optuna, we will use the parameters obtained earlier')
        best_weights = {'weight_xgb': 2, 
                             'weight_lgb': 1, 
                             'weight_cat': 5 
                             }

    print('\n\nbest params: {}'.format(best_weights))
    return best_weights

# weights_dico = Run_Pass_cat_study(1)

weights_list = list(Run_Pass_cat_study(25).values())


# Build the voting regressor
voting_reg = VotingRegressor(estimators, 
                             weights=weights_list,
                             n_jobs=-1, 
                             verbose=0
                            )

# Fit the voting regressor
voting_reg.fit(X_train_prep, y_train)


spliter = KFold(n_splits=4, shuffle=True, random_state=15)

plt.figure(figsize=(14,8))
for f, (tr_ind, va_ind) in enumerate(spliter.split(X_train_prep, y_train), start=1):
    X_tr, X_va = X_train_prep.loc[tr_ind], X_train_prep.loc[va_ind]
    y_tr, y_va = y_train.loc[tr_ind], y_train.loc[va_ind]
    # fit and predict for the fold
    reg = voting_reg.fit(X_tr, y_tr)
    y_va_hat = reg.predict(X_va)
    y_va_hat = np.clip(y_va_hat, orig_raw[target].min(), orig_raw[target].max())
    # score the prediction for the fold
    rmsle_score = rmsle(y_va, y_va_hat)
    r2 = r2_score(y_va, y_va_hat)
    # plot a scatter plot to compare predicted vs true values
    plt.subplot(2,2,f)
    sns.scatterplot(x=y_va, y=y_va_hat)
    plt.title("Fold_{} || rmsle: {:.4f} || r2: {:.4}".format(f, rmsle_score, r2), color='maroon')
    plt.xlabel('true_values')
    plt.ylabel('predicted')
plt.tight_layout()


model = voting_reg

model.fit(X_train_prep, y_train)

pred_base = model.predict(X_test_prep)

sub_base = sample_submission.copy()

sub_base[target] = pred_base

sub_base.to_csv('submission.csv', index=False)

pd.read_csv('submission.csv')

