import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import iqr
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn')
# change default colormap
plt.rcParams['image.cmap'] = 'Dark2'

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
                                     StratifiedKFold, TimeSeriesSplit as TSS)
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
from sklearn.linear_model import Ridge, LogisticRegression
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


train_raw = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id')
test_raw = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col='id')
sub_raw = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')

display(train_raw.head(2), test_raw.head(2))


orig_raw = pd.read_csv('/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv')

# Remove empty spaces from the features names in original dataset
orig_raw.columns = orig_raw.columns.str.replace(' ', '')

# Reorder the features in original dataset to match that of competition
orig_raw = orig_raw[train_raw.columns].copy()

# Binarize the target in the original dataset
orig_raw[target] = orig_raw[target].map({'no': 0, 'yes': 1})

# fill the missing values in test and original datasets
orig_raw = orig_raw.ffill()
test_raw = test_raw.ffill()

display(orig_raw.head(2))


# Decide if features should be engineered
feat_eng = True

def df_processing(df):
    if feat_eng:
        # df['temp_gap'] = df['maxtemp'] - df['mintemp']
        # df['temp_to_gap_ratio'] = df['temparature']*df['temp_gap']
        # for feat in ['temparature', 'dewpoint', 'humidity', 'pressure', 'cloud', 'sunshine', 'windspeed', 'winddirection']:
        #     df[f'{feat}_previous_day'] = df[feat].shift(1).fillna(0)
        #     df[f'{feat}_next_day'] = df[feat].shift(-1).fillna(0)
        #     df[f'{feat}_change_overnight'] = df[feat] - df[f'{feat}_previous_day']
        # # others
        # df['dew_humidity'] = df['dewpoint']*df['humidity']
        # df['wind_speeddirection'] = df['windspeed']*df['winddirection']
        # df['cloud_windspeed'] = df['cloud']*df['windspeed']
        # df['cloud_to_humidity'] = df['cloud']/df['humidity']
        # df['temp_to_humidity'] = df['cloud']/df['humidity']
        # df['temp_to_sunshine'] = df['sunshine']/df['temparature']
        # df['month'] = pd.cut(df['day'], bins=12, labels=range(1, 13)).astype('int')
        # # df['exp_sunshine'] = np.exp(df['sunshine'])
        # # df['log_day'] = np.log(df['day'])
        # df['sin_day'] = np.sin(df['day'])
        # df['wind_deg'] = np.deg2rad(df['winddirection'])
        # df['sin_winddirection'] = np.sin(2*np.pi*df['winddirection'])
        # df['tan_winddirection'] = np.tan(2*np.pi*df['winddirection'])
        # df['day_bins'] = pd.cut(df['day'], bins=12).astype('int')
        # df['expected_day'] = df.index%365 + 1
        df['cloudtest_88'] =  (df.cloud==88).astype(int)
        df['cloudtest_90'] =  (df.cloud>90).astype(int)
        try:
            df['tan_day'] = np.tan(2*np.pi*df['expected_day']/365)
            df['month'] = pd.cut(df['expected_day'], bins=12, labels=range(1,13)).astype('int')
         #   df['cos_day']=np.cos(2*np.pi*df['expected_day']/365)
         #   df['sin_day']=np.sin(2*np.pi*df['expected_day']/365)
            # df['cos_day2']=np.cos(2*np.pi*df['expected_day']/365/2)
            pass
        except:
            pass
        
        df = df.drop(columns=['maxtemp', 'mintemp', 'dewpoint', 'day'])
    else:
        df = df
        
    X = df.copy()
    try:
        y = X.pop(target)
        return X, y
    except:
        # pass
        return X
        # pass


train_orig = pd.concat([train_raw, orig_raw], axis=0, ignore_index=True)
train_orig.shape


# Prepare the train sets
X_tr, y_tr = df_processing(train_raw)

# Prepare the original sets
X_or, y_or = df_processing(orig_raw)

# Prepare the combined train_orig sets
X_tr_or, y_tr_or = df_processing(train_orig)

# Prepare the test set
X_ts = df_processing(test_raw)
X_ts.sample(5)


X, y = X_tr, y_tr


# for model in [
#     RandomForestClassifier, 
#     HistGradientBoostingClassifier,
#     GradientBoostingClassifier, 
#     # ExtraTreesClassifier,
#     # XGBClassifier, 
#     # LGBMClassifier, 
#     GaussianNB, 
#     CatBoostClassifier, 
#     KNeighborsClassifier,
#     Ridge,
#     LogisticRegression
# ]:

#     clf = model()
#     kfold = TSS()
#     s = cross_val_score(clf, X, y, scoring='roc_auc', cv=kfold)
#     print(
#         f'{model.__name__:200} AUC: {s.mean()}'
#     )


best_params = {'C': 100, 'penalty': 'l1', 'solver': 'liblinear'}


splits = 8

# spliter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
spliter = TSS(n_splits=splits)
clf = LogisticRegression()

scores_with_prep = []

for f, (trn_ind, val_ind) in enumerate(spliter.split(X, y), start=1):
    X_trn, X_val = X.iloc[trn_ind], X.iloc[val_ind]
    y_trn, y_val = y.iloc[trn_ind], y.iloc[val_ind]

    clf.fit(X_trn, y_trn)
    y_val_hat = clf.predict_proba(X_val)[:, 1]

    score = roc_auc_score(y_val, y_val_hat)
    scores_with_prep.append(score)

    #Plot the roc_curve
    fpr, tpr, _ = roc_curve(y_val, y_val_hat)
    plt.plot(fpr, tpr, label = 'roc_auc_fold_{} : {:.6f}'.format(f, score))
    plt.plot([0,1], [0,1], linestyle = '--', color = 'gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('average auc_score with_prep: {:.6} ± {:.4}'.format(np.mean(scores_with_prep), np.std(scores_with_prep)))
    plt.legend()


# fit the model
clf = LogisticRegression().fit(X_tr, y_tr)
# predict on the test set
pred = clf.predict_proba(X_ts)[:, 1]
# build the submission dataframe
sub_df = sub_raw.copy()
sub_df[target] = pred
sub_df.head(10)


plt.subplot(121)
sub_df.rainfall.plot.hist(bins=25, color='lightblue', figsize=(10, 3), 
                          title='Histogram of pred_proba in test set')
plt.xlabel('Predicted Proba')
plt.subplot(122)
(sub_df > 0.5).value_counts().plot.pie(labels=['rain', 'no rain'], autopct='%1.1f%%', shadow=True,
                                       explode=[0.05, 0.05], colors=['lightblue', 'grey'], radius=1.3)
plt.ylabel('');


sub_df.to_csv('submission.csv', index=False)
print('The file is ready for submission!')

