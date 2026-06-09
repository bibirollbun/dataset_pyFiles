pip install tabpfn -q


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import the various sklear tools
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn .metrics import roc_curve, roc_auc_score
from sklearn.compose import make_column_transformer
# from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# from sklearn.feature_selection import SequentialFeatureSelector as sk_sfs
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, RepeatedKFold, 
RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score, StratifiedKFold, TimeSeriesSplit as TSS)
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer, minmax_scale, 
                                   PowerTransformer, QuantileTransformer, LabelEncoder,
                                   RobustScaler, StandardScaler, FunctionTransformer,
                                   LabelEncoder, OneHotEncoder, OrdinalEncoder)

import tabpfn
from tabpfn import TabPFNClassifier

# verify the versions
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'seaborn version: {sns.__version__}')
print(f'tabpfn version : {tabpfn.__version__}')


train_raw = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id')
test_raw = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col='id')
sub_raw = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')

target = 'rainfall'

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


train_raw['time'] = np.arange(len(train_raw.index)) + 1  
train_raw.tail(3)


test_raw['time'] = np.arange(len(test_raw.index)) + 2191
test_raw.head(3)


orig_raw['time'] = np.arange(len(orig_raw.index)) + 1  
orig_raw.tail(3)


sns.jointplot(train_raw, x='pressure', y='humidity', hue=target)


sns.jointplot(train_raw, x='cloud', y='humidity', hue=target)


sns.scatterplot(train_raw, x='cloud', y='humidity', hue=target)


train_orig = pd.concat([train_raw, orig_raw])


# Decide if features should be engineered
feat_eng = True

def df_processing(df):
    if feat_eng:
        df['pressure'] = df['pressure'] - 1000
        df['temp_gap'] = df['maxtemp'] - df['mintemp']
        df['temp_to_gap_ratio'] = df['temparature']*df['temp_gap']
        for feat in ['temparature', 'dewpoint', 'humidity', 'pressure', 
                     'cloud', 'sunshine', 'windspeed', 'winddirection']:
            df[f'{feat}_previous_day'] = df[feat].shift(1).fillna(0)
            df[f'{feat}_next_day'] = df[feat].shift(-1).fillna(0)
            df[f'{feat}_change_overnight'] = df[feat] - df[f'{feat}_previous_day']
        # # others
        df['dew_humidity'] = df['dewpoint']*df['humidity']
        df['wind_speeddirection'] = df['windspeed']*df['winddirection']
        df['cloud_windspeed'] = df['cloud']*df['windspeed']
        df['cloud_to_humidity'] = df['cloud']/df['humidity']
        df['temp_to_humidity'] = df['cloud']/df['humidity']
        df['temp_to_sunshine'] = df['sunshine']/df['temparature']
        df['month'] = pd.cut(df['day'], bins=12, labels=range(1, 13)).astype('int')
        # # df['exp_sunshine'] = np.exp(df['sunshine'])
        # # df['log_day'] = np.log(df['day'])
        # df['sin_day'] = np.sin(df['day'])
        # df['wind_deg'] = np.deg2rad(df['winddirection'])
        df['sin_winddirection'] = np.sin(2*np.pi*df['winddirection'])
        df['tan_winddirection'] = np.tan(2*np.pi*df['winddirection'])
        # df['day_bins'] = pd.cut(df['day'], bins=12).astype('int')
        df['expected_day'] = df.index%365 + 1
        df['cloudtest_88'] =  (df.cloud==88).astype(int)
        df['cloudtest_90'] =  (df.cloud>90).astype(int)
        try:
            df['tan_day'] = np.tan(2*np.pi*df['expected_day']/365)
            df['month'] = pd.cut(df['expected_day'], bins=12, labels=range(1,13)).astype('int')
            df['sin_day']=np.sin(2*np.pi*df['expected_day']/365)
            df['cos_day2']=np.cos(2*np.pi*df['expected_day']/365/2)
            pass
        except:
            pass
        
        df = df.drop(columns=['maxtemp', 'mintemp',  'day', 'time', 'temparature'])
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


# Prepare the train sets
X_tr, y_tr = df_processing(train_raw)

# Prepare the original sets
X_or, y_or = df_processing(orig_raw)

# # Prepare the combined train_orig sets
X_tr_or, y_tr_or = df_processing(train_orig)

# Prepare the test set
X_ts = df_processing(test_raw)
X_ts.sample(5)


from sklearn.cluster import KMeans

X_std = StandardScaler().fit_transform(X_tr)

km = KMeans(4, random_state=42)

km.fit(X_std)


target = 'rainfall'

X_km = km.predict(X_std)

X_km


pd.Series(X_km).value_counts()


X_tr['cloudtest_90'].mean()


try:
    plt.subplot(211)
    train_raw['day'].plot(figsize=(10,4), title='day', color='maroon')
    plt.subplot(212)
    X_tr['expected_day'].plot(figsize=(10,4), title='expected day')
    plt.tight_layout()
except:
    pass


# Decide if original data should be used
use_orig = True
if use_orig:
    X = X_tr_or
    y = y_tr_or
else:
    X = X_tr
    y = y_tr   





features_trans = make_column_transformer(
    (RobustScaler(), X_tr.columns.to_list()),
    remainder='drop', 
    sparse_threshold=0)


splits = 6

# spliter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
spliter = TSS(n_splits=splits)
tab_clf = TabPFNClassifier(n_estimators=5, 
                           softmax_temperature=10, 
                           balance_probabilities=False,
                           average_before_softmax=True)

scores_with_prep = []

for f, (trn_ind, val_ind) in enumerate(spliter.split(X, y), start=1):
    X_trn, X_val = X.iloc[trn_ind], X.iloc[val_ind]
    y_trn, y_val = y.iloc[trn_ind], y.iloc[val_ind]

    clf = make_pipeline(features_trans, tab_clf).fit(X_trn, y_trn)
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


scores_no_prep = []

y_true, y_hat, y_hat_proba = list(), list(), list()

for f, (trn_ind, val_ind) in enumerate(spliter.split(X, y), start=1):
    X_trn, X_val = X.iloc[trn_ind], X.iloc[val_ind]
    y_trn, y_val = y.iloc[trn_ind], y.iloc[val_ind]

    clf = tab_clf.fit(X_trn, y_trn)
    y_val_hat = clf.predict_proba(X_val)[:, 1]

    score = roc_auc_score(y_val, y_val_hat)
    scores_no_prep.append(score)

    #Plot the roc_curve
    fpr, tpr, _ = roc_curve(y_val, y_val_hat)
    plt.plot(fpr, tpr, label = 'roc_auc_fold_{} : {:.6f}'.format(f, score))
    plt.plot([0,1], [0,1], linestyle = '--', color = 'gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('average auc_score no_prep: {:.6} ± {:.4}'.format(np.mean(scores_no_prep), np.std(scores_no_prep)))
    plt.legend()


pd.Series(scores_with_prep).plot(marker='o', figsize=(8, 3), color='maroon', legend=True, label='with prep')
pd.Series(scores_no_prep).plot(marker='o', figsize=(8, 3), color='steelblue', legend=True, label='no prep')
plt.ylim(0.8, 1);


# Decide if features prep should be used
use_prep = False

# Build the model
if use_prep:
    clf_final = make_pipeline(features_trans, TabPFNClassifier())
else:
    clf_final = TabPFNClassifier()

# Fit the model   
clf_final.fit(X, y)


clf_final.fit(X, y)

pred = clf_final.predict_proba(X_ts)[:, 1]

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

