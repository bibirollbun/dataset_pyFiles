import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import iqr
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
# change default colormap
plt.rcParams['image.cmap'] = 'Set2'

# Import the various sklear tools
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_log_error, accuracy_score, roc_auc_score, roc_curve

# from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# from sklearn.feature_selection import SequentialFeatureSelector as sk_sfs
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score,
                                     StratifiedKFold)
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier, 
                              StackingClassifier, BaggingClassifier,VotingClassifier)
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance, cv

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras import layers

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer,
                                   PowerTransformer, QuantileTransformer, LabelEncoder,
                                   RobustScaler, StandardScaler, minmax_scale,
                                   OneHotEncoder, FunctionTransformer)

import yellowbrick
from yellowbrick.classifier import ClassificationReport, DiscriminationThreshold, confusion_matrix
from yellowbrick.regressor import PredictionError
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from yellowbrick.regressor import ResidualsPlot
from yellowbrick.datasets import load_bikeshare
from yellowbrick.regressor import PredictionError
from yellowbrick.model_selection import FeatureImportances
from yellowbrick.model_selection import FeatureImportances
from yellowbrick.classifier import (ClassificationReport, 
DiscriminationThreshold, 
confusion_matrix,
ClassPredictionError
)

import optuna
from optuna.samplers import TPESampler
import plotly.express as px

# Set the color scheme 
# my_scheem = 'crest_r'
# Define your custom color palette
my_scheem = ["gold", "#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974", "#64b5cd"]
# my_scheem = 'Dark2'
sns.set_palette(my_scheem)
# sns.color_palette('"blend:#7AB,#EDA", as_cmap=True')
rain_palette = ['gold', 'steelblue']

pd.set_option('display.max_columns', 100)
# verify the versions
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'seaborn version: {sns.__version__}')
print(f'optuna version : {optuna.__version__}')
print(f'yellowbrick version: {yellowbrick.__version__}')


target = 'rainfall'


train_00 = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id')
test_00 = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col='id')

train_00.head(3)


train_00.info()


train_00.nunique().to_frame()


# df['wind_speeddirection'] = df['windspeed']*df['winddirection']
# df = df.drop(columns=['maxtemp', 'mintemp', 'dewpoint'])


train_00['rainfall'].value_counts()


orig_00 = pd.read_csv('/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv')
orig_00.head(3)


orig_00.shape


orig_00['rainfall'].value_counts()


orig_00.columns = orig_00.columns.str.replace(' ', '')


orig_01 = orig_00[train_00.columns].copy()

orig_01['rainfall'] = orig_00['rainfall'].map({'yes':1, 'no':0})

orig_01.head(3)


orig_01['rainfall'].value_counts()


for df in [train_00, orig_00, test_00]:
    df = df.fillna(method='bfill')


features = [col for col in test_00]


# Define a function to perform the adversarial validation of two datasets
def adversarial_validation(df_1, df_2, name_1, name_2):
    adv_df_1 = df_1[features].copy()
    adv_df_2 = df_2[features].copy()


    # label the test and train data with 0 and 1 (it doesn't really matter which is which)
    adv_df_1 = adv_df_1.assign(adv=1)
    adv_df_2 = adv_df_2.assign(adv=0)


    # combine the training and test data into one big dataset
    combined = pd.concat([adv_df_1, adv_df_2], axis=0)

    # Shuffle
    combined = combined.sample(frac=1, random_state=4)

    # perform the binary classification, for example using XGboost
    X_combined = combined.drop('adv', axis=1)
    y_combined = combined.adv


    cv = StratifiedKFold(n_splits = 5,
                        shuffle = True,
                        random_state = 64)
    xgb_model = XGBClassifier(max_depth=3,
                              learning_rate = 0.1,
                              n_estimators = 100,
                              objective = 'binary:logistic',
                              random_state = 64)

    # Get the cross validation scores
    adv_scores = []
    for i, _ in enumerate(cv.split(X_combined, y_combined)):
        X_train, X_valid, y_train, y_valid = train_test_split(X_combined, 
                                                              y_combined, 
                                                              test_size=0.3)
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict_proba(X_valid)[:,1]
        score = roc_auc_score(y_valid, y_pred)
        adv_scores.append(score)

#         print(f"Fold {i+1} AUC Score: {score:.5f}")

    #Plot the roc_curve
    mean_auc = np.mean(adv_scores)
    fpr, tpr, _ = roc_curve(y_valid, y_pred)
    plt.plot(fpr, tpr, label = 'roc_curve (AUC = %0.4f)' % mean_auc)
    plt.plot([0,1], [0,1], linestyle = '--', color = 'gray', label = 'Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'roc_curve {name_1} vs {name_2}', weight='bold')
    plt.legend()


train_comb = pd.concat([train_00, orig_01])


# num_features = list(test_00.select_dtypes('number'))
plt.figure(figsize=(18,5))
plt.subplot(1,3,1)
adversarial_validation(train_00, test_00, 'train', 'test')
plt.subplot(1,3,2)
adversarial_validation(train_00, orig_01, 'train', 'original')
plt.subplot(1,3,3)
adversarial_validation(train_comb, test_00, 'train_comb', 'test')


train_00.describe().T


orig_01.describe().T


test_00.describe().T


print(f'There are {train_00["day"].nunique()} unique days in the day feature')


plt.figure(figsize=(12,3))

train_00['day'].value_counts().plot.bar(color='steelblue')
plt.xticks([]);


plt.figure(figsize=(12, 6))
sns.swarmplot(train_00, x='day', hue=target, size=7, )
plt.xticks([])
plt.show();


from matplotlib.projections import PolarAxes

train_rain = train_00[train_00[target]==1]
train_norain = train_00[train_00[target]!=1]

# Create a wind rose plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi / 2.0)

bars = ax.bar(np.deg2rad(train_rain['winddirection']), 
              train_rain['windspeed'], width=np.pi/8, 
              bottom=0.0, color='steelblue', alpha=0.3)

for bar in bars:
    bar.set_alpha(0.5)

plt.title('Wind Speed and Direction with Rain')
plt.show()


# Create a wind rose plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi / 2.0)

bars = ax.bar(np.deg2rad(train_norain['winddirection']), 
              train_norain['windspeed'], width=np.pi/8, 
              bottom=0.0, color='gold', alpha=0.3)

for bar in bars:
    bar.set_alpha(0.5)

plt.title('Wind Speed and Direction without Rain')
plt.show()


plt.figure(figsize=(8,12))
# for f, feat in enumerate(['humidity', 'temparature', 'dewpoint', 'pressure', 'windspeed', 'cloud'], start=1):
for f, feat in enumerate(test_00.columns[1:], start=1):
    plt.subplot(5,2,f)
    sns.lineplot(train_00, x='day', y=feat, hue=target)
    if f!=1:
        plt.legend([])
    plt.title(f'{feat} over days')
plt.tight_layout()
plt.show()


train_01 = train_00.copy()

# train_01['days_bin'] = pd.cut(train_00['day'], bins=12)
# train_01['days_bin']


# plt.figure(figsize=(10,8))
# for f, feat in enumerate(test_00.columns, start=1):
#     plt.subplot(3,4,f)
#     sns.violinplot(train_00, x=feat, split=True, hue=target, palette=rain_palette)
# plt.tight_layout()


plt.figure(figsize=(12,5))
for f, feat in enumerate(test_00.columns, start=1):
    plt.subplot(2,6,f)
    sns.violinplot(train_00, y=feat, x=target, palette=rain_palette)
plt.tight_layout()


plt.figure(figsize=(12,5))
for f, feat in enumerate(test_00.columns, start=1):
    plt.subplot(2,6,f)
    sns.violinplot(orig_01, y=feat, x=target, palette=rain_palette)
plt.tight_layout()


plt.figure(figsize=(12,4))
for f, feat in enumerate(test_00.columns, start=1):
    plt.subplot(2,6,f)
    sns.histplot(train_00, x=feat, palette=rain_palette, hue=target)
    if f not in [1, 7]:
        plt.ylabel('')
    if f != 6:
        plt.legend([])
plt.tight_layout()


plt.figure(figsize=(12,4))
for f, feat in enumerate(test_00.columns, start=1):
    plt.subplot(2,6,f)
    sns.boxenplot(train_00, y=feat, x=target, palette=rain_palette)
plt.tight_layout()


def target_pie(df, set):
    df[target].value_counts().plot.pie(autopct='%1.1f%%',  
                                             # cmap='GnBu_r', 
                                             figsize=(9, 6), 
                                             labels=['rain', 'no rain'],
                                             textprops={'color': 'black'},
                                             title=f'Target pie in {set}',
                                            )
    plt.ylabel('')


plt.subplot(121)
target_pie(train_00, 'train')
plt.subplot(122)
target_pie(orig_01, 'original')


feat_corr = train_00.corr().style.format('{:.3f}').background_gradient(cmap='YlGnBu')
feat_corr


# Generate a mask for the upper triangle 
mask = np.triu(np.ones_like(train_00.corr(), dtype=bool))
# Generate a custom diverging colormap 
cmap = my_scheem[:-2]

plt.figure(figsize=(11, 6.6))
sns.heatmap(np.abs(train_00.corr()), annot=True, cmap='YlGnBu', mask=mask, fmt='.3f', cbar=False)
plt.title('Absolute correlation of the features in the train set')
plt.show()


sns.pairplot(train_00, hue=target);


def pre_processor(df):
    df['dew_humidity'] = df['dewpoint']*df['humidity']
    df['temp_gap'] = df['maxtemp'] - df['mintemp']
    df['wind_speeddirection'] = df['windspeed']*df['winddirection']
    df['cloud_windspeed'] = df['cloud']*df['windspeed']
    df['cloud_to_humidity'] = df['cloud']/df['humidity']
    df['temp_to_humidity'] = df['cloud']/df['humidity']
    df['temp_to_sunshine'] = df['sunshine']/df['temparature']
    df['month'] = pd.cut(df['day'], bins=12, labels=range(1, 13)).astype('int')
    # # df['exp_sunshine'] = np.exp(df['sunshine'])
    # # df['log_day'] = np.log(df['day'])
    df['temp_previous_day'] = df['temparature'].shift(1).fillna(0)
    # df['temp_next_day'] = df['temparature'].shift(-1).fillna(0)
    df['humidity_previous_day'] = df['humidity'].shift(1).fillna(0)
    df['pressure_previous_day'] = df['pressure'].shift(1).fillna(0)
    # df['sin_day'] = np.sin(df['day'])
    # df['wind_deg'] = np.deg2rad(df['winddirection'])
    # df['sin_winddirection'] = np.sin(df['winddirection'])
    # df['tan_winddirection'] = np.tan(df['winddirection'])
    # df['day_bins'] = pd.cut(df['day'], bins=12).astype('category')
    df = df.drop(columns=['maxtemp', 'mintemp'])

    return df


train_02 = pre_processor(train_01)
test_02 = pre_processor(test_00)
train_02


X = train_02.copy()
y = X.pop(target)


# scaler = StandardScaler()
scaler = RobustScaler()
# scaler = MinMaxScaler()

features_trans = make_column_transformer(
    (scaler, test_02.select_dtypes('number').columns),
    # (scaler, ['pressure', 'temparature', 'humidity', 'cloud',
    #                   'sunshine', 'winddirection', 'windspeed']),
    # (OneHotEncoder(), ['day_bins']),
    remainder='drop', 
    sparse_threshold=0)


tr_01 = train_01.copy()
pd.DataFrame(features_trans.fit_transform(tr_01))


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600, 10),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.3, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    model = LGBMClassifier(**params, verbose=-1)
    model_pipe = make_pipeline(features_trans, PCA(), model)

    
    scores = []
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_ind, va_ind in kfold.split(X, y):
        X_tr, X_va = X.iloc[tr_ind], X.iloc[va_ind]
        y_tr, y_va = y.iloc[tr_ind], y.iloc[va_ind]

        model_pipe.fit(X_tr, y_tr)
        score = roc_auc_score(y_va, model_pipe.predict_proba(X_va)[:, 1])
        scores.append(score)
    average_score = np.mean(scores)   
    return average_score  

    # x_tr, x_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=55)
    # model_pipe.fit(x_tr, y_tr)
    # y_pred = model_pipe.predict_proba(x_va)[:, 1]
    
    # # Calculate RMSE
    # score = roc_auc_score(y_va, y_pred)
    # return score

def get_optuna_best_params(n_trials=1):
    if n_trials >1:
        study = optuna.create_study(direction='maximize',sampler=TPESampler(seed=33))
        study.optimize(lambda trial: objective(trial), n_trials=n_trials,show_progress_bar=True,n_jobs=-1, timeout=12000)
        best_params = study.best_params
    else:
        print('No need to run optuna, we will use the parameters obtained earlier')
        best_params = {'n_estimators': 130, 
                       'max_depth': 2, 
                       'num_leaves': 250, 
                       'feature_fraction': 0.7557630763041963, 
                       'bagging_fraction': 0.604841116587687, 
                       'bagging_freq': 3, 
                       'min_child_samples': 72}
        
    print('best params: {}'.format(best_params))
    return best_params


best_params = get_optuna_best_params(n_trials=200)


# X = X.drop(columns=['maxtemp', 'mintemp', 'day'])

X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.25, random_state=5)

[d.shape for d in [X_tr, X_ts, y_tr, y_ts]]


# model = LGBMClassifier(verbose=-1)
# model = HistGradientBoostingClassifier()
# model = CatBoostClassifier(n_estimators=1000, eval_fraction=0.2, verbose=200)
# model = XGBClassifier(use_label_encoder=False, eval_metric='auc', random_state=42)

model = LGBMClassifier(**best_params, verbose=-1)

model_pipe = make_pipeline(features_trans, model)

model_pipe.fit(X_tr, y_tr)


viz = ClassificationReport(model_pipe, classes=['rain', 'no rain'], cmap='YlGnBu', colorbar=False)

viz.fit(X_tr, y_tr)
viz.score(X_ts, y_ts)
viz.show()
plt.show()


viz = DiscriminationThreshold(model_pipe)
viz.fit(X_tr,y_tr)
viz.show()
plt.show()


cpe_viz = ClassPredictionError(model_pipe, classes=['rain', 'no rain'])
cpe_viz.score(X_ts, y_ts)
cpe_viz.poof();


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)
kfold

rfc_pipe = make_pipeline(features_trans, RandomForestClassifier())

for f, (train_ind, test_ind) in enumerate(kfold.split(X, y)):
    X_train, X_test = X.iloc[train_ind], X.iloc[test_ind]
    y_train, y_test = y.iloc[train_ind], y.iloc[test_ind]
    plt.figure(figsize=(4.4,4))
    model_pipe.fit(X_train, y_train)
    conf_mat = confusion_matrix(model_pipe, X_train, y_train, 
                    X_test, y_test, percent=True, fontsize=14, cmap='YlGnBu')


for f, (train_ind, test_ind) in enumerate(kfold.split(X, y), start=1):
    X_train, X_test = X.iloc[train_ind], X.iloc[test_ind]
    y_train, y_test = y.iloc[train_ind], y.iloc[test_ind]

    model_pipe.fit(X_train, y_train)
    auc_score = roc_auc_score(y_test, model_pipe.predict_proba(X_test)[:, 1])
    print('auc_score fold_{}: {:.5f}'.format(f, auc_score))


sub_00 = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
sub_01 = sub_00.copy()

pred = model_pipe.predict_proba(test_02)[:, 1]

sub_01[target] = pred

sub_01.to_csv('submission.csv', index=False)

display(sub_01.head(10))

print('The file is ready for submission')


rfc = RandomForestClassifier(n_estimators=10)
viz = FeatureImportances(rfc)
viz.fit(X, y)
# viz.fit(X.iloc[:, :-1], y)
viz.show();


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)
kfold

rfc_pipe = make_pipeline(features_trans, rfc)

for f, (train_ind, test_ind) in enumerate(kfold.split(X, y)):
    X_train, X_test = X.iloc[train_ind], X.iloc[test_ind]
    y_train, y_test = y.iloc[train_ind], y.iloc[test_ind]
    plt.figure(figsize=(4.4,4))
    rfc_pipe.fit(X_train, y_train)
    conf_mat = confusion_matrix(rfc_pipe, X_train, y_train, 
                    X_test, y_test, percent=True, fontsize=14, cmap='YlGnBu')


for f, (train_ind, test_ind) in enumerate(kfold.split(X, y), start=1):
    X_train, X_test = X.iloc[train_ind], X.iloc[test_ind]
    y_train, y_test = y.iloc[train_ind], y.iloc[test_ind]

    rfc_pipe.fit(X_train, y_train)
    auc_score = roc_auc_score(y_test, rfc_pipe.predict_proba(X_test)[:, 1])
    print('auc_score fold_{}: {:.5f}'.format(f, auc_score))


xgb = XGBClassifier(n_estimators=10)
viz = FeatureImportances(xgb)
viz.fit(X, y)
# viz.fit(X.iloc[:, :-1], y)
viz.show();


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)
kfold

xgb_pipe = make_pipeline(features_trans, XGBClassifier())

for f, (train_ind, test_ind) in enumerate(kfold.split(X, y)):
    X_train, X_test = X.iloc[train_ind], X.iloc[test_ind]
    y_train, y_test = y.iloc[train_ind], y.iloc[test_ind]
    plt.figure(figsize=(4.4,4))
    xgb_pipe.fit(X_train, y_train)
    conf_mat = confusion_matrix(xgb_pipe, X_train, y_train, 
                    X_test, y_test, percent=True, fontsize=14, cmap='YlGnBu')


for f, (train_ind, test_ind) in enumerate(kfold.split(X, y), start=1):
    X_train, X_test = X.iloc[train_ind], X.iloc[test_ind]
    y_train, y_test = y.iloc[train_ind], y.iloc[test_ind]

    xgb_pipe.fit(X_train, y_train)
    auc_score = roc_auc_score(y_test, xgb_pipe.predict_proba(X_test)[:, 1])
    print('auc_score fold_{}: {:.5f}'.format(f, auc_score))

