pip install tabpfn -q


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
# change default colormap
plt.rcParams['image.cmap'] = 'Dark2'

# Import the various sklear tools
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn .metrics import accuracy_score, make_scorer, confusion_matrix, roc_auc_score
from sklearn.compose import make_column_transformer
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
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer, minmax_scale, 
                                   PowerTransformer, QuantileTransformer, LabelEncoder,
                                   RobustScaler, StandardScaler, FunctionTransformer,
                                   LabelEncoder, OneHotEncoder, OrdinalEncoder)
from tabpfn import TabPFNClassifier
import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_contour
from optuna.visualization import plot_slice
import plotly.express as px

rainfall_colors = ['lightblue', 'gold']
rainfall_colors_r = ['gold', 'lightblue']
seed = 32

pd.set_option('display.max_columns', 100)
# verify the versions
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'seaborn version: {sns.__version__}')
print(f'optuna version : {optuna.__version__}')


target = 'rainfall'


train_0 = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id')
test_0 = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col='id')
orig_0 = pd.read_csv('/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv')

display(train_0.sample(3), orig_0.sample(3), test_0.sample(3))


orig_0.columns = orig_0.columns.str.replace(' ', '')

orig_0 = orig_0[train_0.columns].copy()


for df in [train_0, orig_0, test_0]:
    print(df.isnull().sum())


# fill the missing values
orig_0 = orig_0.fillna(method='bfill')
test_0 = test_0.fillna(method='bfill')


orig_0[target] = orig_0[target].map({'no': 0, 'yes': 1})


for df_name, df in [('train', train_0), ('test', test_0), ('original', orig_0)]:
    nunb_of_duplicates = df.duplicated().sum()
    if nunb_of_duplicates != 0:
        print(f'{df_name} dataset has {nunb_of_duplicates} duplicates.')
    else:
        print(f'{df_name} dataset has no duplicates')


plt.figure(figsize=(10, 4))

plt.subplot(121)
train_0[target].value_counts().plot.pie(autopct='%.2f%%',
                                              colors=rainfall_colors,
                                              explode=[0.1, 0.1], radius=1.2)
plt.ylabel('')
plt.subplot(122)
ax = sns.countplot(train_0, x=target, palette=rainfall_colors_r)
for label in ax.containers:
    ax.bar_label(label, fontsize=10)
plt.suptitle('rain in train set')
plt.ylabel('')
plt.show()


plt.figure(figsize=(10, 4))

plt.subplot(121)
orig_0[target].value_counts().plot.pie(autopct='%.2f%%',
                                              colors=rainfall_colors,
                                              explode=[0.1, 0.1], radius=1.2)
plt.ylabel('')
plt.subplot(122)
ax = sns.countplot(orig_0, x=target, palette=rainfall_colors_r)
for label in ax.containers:
    ax.bar_label(label, fontsize=10)
plt.suptitle('Rain in original set')
plt.ylabel('')
plt.show()


from matplotlib.projections import PolarAxes
from matplotlib.colors import to_rgba

train_rain = train_0[train_0[target]==1]
train_norain = train_0[train_0[target]!=1]

# Define base colors and adjust saturation using alpha (transparency)
base_colors = ['steelblue', 'green','yellow']
adjusted_colors = [to_rgba(color, alpha=0.3) for color in base_colors]  # Adjust alpha to control saturation

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 6))

# First wind rose plot (rain)
ax1 = axes[0]
ax1.set_theta_direction(-1)
ax1.set_theta_offset(np.pi / 2.0)
bars1 = ax1.bar(
    np.deg2rad(train_rain['winddirection']),
    train_rain['windspeed'],
    width=np.pi/8,
    bottom=0.0,
    color=adjusted_colors  # Adjust color
)
ax1.set_title('Wind Speed and Direction with Rain')

# Second wind rose plot (no rain)
ax2 = axes[1]
ax2.set_theta_direction(-1)
ax2.set_theta_offset(np.pi / 2.0)
bars2 = ax2.bar(
    np.deg2rad(train_norain['winddirection']),
    train_norain['windspeed'],
    width=np.pi/8,
    bottom=0.0,
    color= adjusted_colors # Adjust color
)
ax2.set_title('Wind Speed and Direction without Rain');

# Display the plots


roll_columns = ['sunshine', 'cloud', 'humidity', 'dewpoint', 'windspeed', 'temparature', 'pressure']
n = 28

# Concatenate train and test to align indexes
train_1, test_1 = train_0.copy(), test_0.copy()
train_1['source'] = 'Train'  # Add source columns for later differentiation
test_1['source'] = 'Test'
combined = pd.concat([train_1, test_1]).reset_index(drop=True)

for col in roll_columns:
    # Calculate rolling mean for the combined dataset
    combined[f'rolling_{col}'] = combined[col].rolling(
        window=n,
        center=True,
        min_periods=(n + 1) // 2
    ).mean()

    # Separate train and test for plotting
    train_rolling = combined[combined['source'] == 'Train']
    test_rolling = combined[combined['source'] == 'Test']

    # Create a single plot with both train and test
    fig, ax = plt.subplots(figsize=(10, 4))
    train_rolling[f'rolling_{col}'].plot.line(ax=ax, color='green', label='Train')
    test_rolling[f'rolling_{col}'].plot.line(ax=ax, color='lightgreen', label='Test')

    # Customize plot
    title = f'Rolling {col} over {n} days'
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.show()


plt.figure(figsize=(10,12))

for f, feat in enumerate(test_0.columns[1:], start=1):
    plt.subplot(5,2,f)
    sns.lineplot(train_0, x='day', y=feat, hue=target, palette=rainfall_colors)
    if f!=1:
        plt.legend([])
    plt.title(f'{feat} over days', fontsize=12)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,12))

for f, feat in enumerate(test_0.columns[1:], start=1):
    plt.subplot(5,2,f)
    sns.lineplot(orig_0, x='day', y=feat, hue=target, palette=rainfall_colors)
    if f!=1:
        plt.legend([])
    plt.title(f'{feat} over days', fontsize=12)
plt.tight_layout()
plt.show()


sns.pairplot(orig_0, hue=target, palette=rainfall_colors_r, height=1.2, hue_order=[0, 1]);


# Decide if features should be engineered
feat_eng = True
n = 3

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
            df[f'rolling_{feat}'] = df[feat].rolling(window=n,center=True,min_periods=(n + 1) // 2).mean()
        # # # others
        df['dew_humidity'] = df['dewpoint']*df['humidity']
        df['wind_speeddirection'] = df['windspeed']*df['winddirection']
        df['cloud_windspeed'] = df['cloud']*df['windspeed']
        df['cloud_to_humidity'] = df['cloud']/df['humidity']
        df['temp_to_humidity'] = df['cloud']/df['humidity']
        df['temp_to_sunshine'] = df['sunshine']/df['temparature']
        df['month'] = pd.cut(df['day'], bins=12, labels=range(1, 13)).astype('int')
        # # # df['exp_sunshine'] = np.exp(df['sunshine'])
        # # df['log_day'] = np.log(df['day'])
        # df['sin_day'] = np.sin(df['day'])
        # df['wind_deg'] = np.deg2rad(df['winddirection'])
        # df['sin_winddirection'] = np.sin(2*np.pi*df['winddirection'])
        # df['tan_winddirection'] = np.tan(2*np.pi*df['winddirection'])
        # df['day_bins'] = pd.cut(df['day'], bins=12).astype('int')
        df['expected_day'] = df.index%365 + 1
        df['cloudtest_88'] =  (df.cloud==88).astype(int)
        df['cloudtest_90'] =  (df.cloud>90).astype(int)
        try:
            df['tan_day'] = np.tan(2*np.pi*df['expected_day']/365)
            df['month'] = pd.cut(df['expected_day'], bins=12, labels=range(1,13)).astype('int')
            df['sin_day']=np.sin(2*np.pi*df['expected_day']/365)
            df['cos_day2']=np.cos(2*np.pi*df['expected_day']/365/2)
            df['cos_winddirection'] = np.cos(2*np.pi*df['winddirection'])

            pass
        except:
            pass
        
    df = df.drop(columns=['maxtemp', 'mintemp', 'dewpoint'])
    X = df.copy()
    try:
        y = X.pop(target)
        return X, y
    except:
        pass
        return X
        # pass


train_orig = pd.concat([train_0, orig_0], axis=0, ignore_index=True)
train_orig.shape


# Prepare the train sets
X_tr, y_tr = df_processing(train_0)

# Prepare the original sets
X_or, y_or = df_processing(orig_0)

# Prepare the combined train_orig sets
X_tr_or, y_tr_or = df_processing(train_orig)

# Prepare the test set
X_ts = df_processing(test_0)
X_ts.sample(5)


features_trans = make_column_transformer(
    (StandardScaler(), X_tr.select_dtypes('number').columns.tolist()),
    (OneHotEncoder(), X_tr.select_dtypes(exclude='number').columns.tolist()),
    remainder='drop', 
    sparse_threshold=0)


X_prep = X_tr.copy()

x_prep = features_trans.fit_transform(X_prep)

X_prep.sample(4)


X_train, X_val, y_train, y_val = train_test_split(X_tr_or, y_tr_or, test_size=0.25, shuffle=False)

[d.shape for d in [X_train, X_val, y_train, y_val]]


lgb_params = {'n_estimators': 130, 
               'max_depth': 2, 
               'num_leaves': 250, 
               'feature_fraction': 0.7557630763041963, 
               'bagging_fraction': 0.604841116587687, 
               'bagging_freq': 3, 
               'min_child_samples': 72}

cat_params = {'iterations': 270, 
              'learning_rate': 0.16211513182629325, 
              'objective': 'CrossEntropy', 
              'colsample_bylevel': 0.8069812365614417, 
              'random_strength': 0.2312285611887174, 
              'depth': 4, 'boosting_type': 'Ordered', 
              'bootstrap_type': 'Bayesian', 
              'bagging_temperature': 0.6887719860711248}

xgb_params = {'n_estimators': 190, 
               'learning_rate': 0.017792963423540194, 
               'max_depth': 6, 
               'subsample': 0.2579692108675591, 
               'colsample_bytree': 0.2487767930540334, 
               'min_child_weight': 4}


tss = True
n_splits = 6

# List of some base models to be exploited
Models = [
          ('lgb_b',make_pipeline(features_trans, LGBMClassifier(**lgb_params,verbose=-1))),
          ('cat',make_pipeline(features_trans, CatBoostClassifier(**cat_params, verbose=False))),
          ('xgb',make_pipeline(features_trans,XGBClassifier(**xgb_params))),
          ('hgb', make_pipeline(features_trans, HistGradientBoostingClassifier())),
          ('rfc', make_pipeline(features_trans, RandomForestClassifier())),
          ('etc', make_pipeline(features_trans, ExtraTreesClassifier())),
          ('tap', TabPFNClassifier()),
          ('lgb',make_pipeline(features_trans, LGBMClassifier(verbose=-1)))
         ]

'''Dataset without any new columns'''
scores = [] # Empty cross validation score list
models = [] # Empty list of models
if tss:
    my_spliter = TSS(n_splits=n_splits)
else:
    my_spliter = KFold(n_splits=n_splits, shuffle=True, random_state=42)


for model_name, model in Models:
    
    # Cross validation
    cv_score = cross_val_score(model, 
                               X=X_or, 
                               y=y_or, 
                               cv=my_spliter,
                               scoring='roc_auc'
                              )
    
    scores.append(cv_score) # Add the scores to the scores list
    models.append(model_name) # Add the model to the list of models
    scores_df = pd.DataFrame(scores, 
                             # columns=['cv1', 'cv2', 'cv3', 'cv4', 'cv5'],
                             columns=[f'fold_{f}' for f in range(1, n_splits + 1)],
                             index=models) # Get the acores into a data frame

scores_df['avg_score'] = scores_df.mean(axis=1)
scores_df['std_score'] = scores_df.std(axis=1)
scores_df = scores_df.sort_values(by='avg_score', ascending=False)


display(
        (scores_df.style.background_gradient(cmap='Blues', axis=0)
         # .highlight_min(axis=0, color='yellow')
         .format('{:.5f}')
         .set_properties(**{'font-size': '12pt', 'weight': 'bold'}))
       )


my_palette=['green', 'steelblue', 'maroon', 'blue', 'orange', 'violet']

plt.figure(figsize=(8, 3.6))
sns.lineplot(scores_df.iloc[:, :-2].T, palette=my_palette, marker='o')
plt.ylabel('Scores')
plt.legend(loc='right', bbox_to_anchor=(1.2, 0.7),
          fancybox=True, shadow=True, ncol=1)
plt.show()


plt.figure(figsize=(6,3))
sns.lineplot(scores_df.iloc[:, :-1], palette=my_palette, marker='o')
plt.ylabel('Scores')
plt.legend(loc='right', bbox_to_anchor=(1.3, 0.7),
          fancybox=True, shadow=True, ncol=1)
plt.show()


estimators = [
    ('rfc', make_pipeline(features_trans, RandomForestClassifier())),
    ('etc', make_pipeline(features_trans, ExtraTreesClassifier())),
    ('hgb', make_pipeline(features_trans, HistGradientBoostingClassifier())),
    ('cat', make_pipeline(features_trans, CatBoostClassifier())),
    # ('lgb_b',make_pipeline(features_trans, LGBMClassifier(**lgb_params, verbose=-1))),
    ('cat_b',make_pipeline(features_trans, CatBoostClassifier(**cat_params, verbose=False))),
    # ('xgb',make_pipeline(features_trans,XGBClassifier(**xgb_params))),
    ('tap', TabPFNClassifier())
]

sc_1 = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            n_jobs=-1
    )

sc_1


my_spliter = TSS(n_splits=n_splits)

y_true, y_hat, y_hat_proba = list(), list(), list()
X, y = X_tr, y_tr

for f, (train_idx, test_idx) in enumerate(my_spliter.split(X), start=1):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    sc_1.fit(X_train, y_train)

    preds = sc_1.predict(X_test)
    pred_proba = sc_1.predict_proba(X_test)[:,1]

    y_true.extend(y_test)
    y_hat.extend(preds)
    y_hat_proba.extend(pred_proba)

oof_accu = accuracy_score(y_true, y_hat)
oof_auc = roc_auc_score(y_true, y_hat_proba)

print('oof_score: {:.5f}\nauc_score: {:.5f}'.format(oof_accu, oof_auc))


estimators = [
    ('rfc', RandomForestClassifier()),
    ('etc', ExtraTreesClassifier()),
    # ('hgb', HistGradientBoostingClassifier()),
    # ('cat', CatBoostClassifier()),
    ('lgb', LGBMClassifier(**lgb_params,verbose=-1)),
    ('cat_b', CatBoostClassifier(**cat_params, verbose=False)),
    # ('xgb', XGBClassifier(**xgb_params)),
]

sc_2 = make_pipeline(features_trans,
    StackingClassifier(
            estimators=estimators, stack_method='predict_proba',
            final_estimator=LogisticRegression(),
            n_jobs=-1
    ))

sc_2


y_true, y_hat, y_hat_proba = list(), list(), list()

for f, (train_idx, test_idx) in enumerate(my_spliter.split(X), start=1):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    sc_2.fit(X_train, y_train)

    preds = sc_2.predict(X_test)
    pred_proba = sc_2.predict_proba(X_test)[:,1]

    y_true.extend(y_test)
    y_hat.extend(preds)
    y_hat_proba.extend(pred_proba)

oof_accu = accuracy_score(y_true, y_hat)
oof_auc = roc_auc_score(y_true, y_hat_proba)

print('oof_score: {:.5f}\nauc_score: {:.5f}'.format(oof_accu, oof_auc))


final_model = sc_1

final_model.fit(X, y)


pred_or = final_model.predict(X_or)

plt.figure(figsize=(8,4))
conf_matrix = confusion_matrix(y_or, pred_or)
plt.subplot(121)
sns.heatmap(conf_matrix, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.subplot(122)
conf_matrix_norm = confusion_matrix(y_or, pred_or, normalize='true')*100
sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cbar=False, cmap='Greens')
plt.show()


pred_test_sc = final_model.predict_proba(X_ts)[:, 1]

sub_0 = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')

sub_pred_sc = sub_0.copy()

sub_pred_sc[target] = pred_test_sc

sub_pred_sc[target] = sub_pred_sc[target].astype('float')

sub_pred_sc.head(10)


plt.figure(figsize=(8, 6))

plt.subplot(211)
ax = sns.histplot(sub_pred_sc, x=target, bins=2, color='lightblue')
for label in ax.containers:
    ax.bar_label(label, fontsize=10)
plt.title('Rainfall in test set', fontsize=14)
plt.ylabel('')
plt.xlabel('')

plt.subplot(212)
ax = sns.histplot(sub_pred_sc, x=target, bins=20, color='lightblue')
for label in ax.containers:
    ax.bar_label(label, fontsize=10)
plt.title('Rainfall probability as predicted in test set', fontsize=14)
plt.ylabel('')
plt.tight_layout()
plt.show()


sub_pred_sc.to_csv('submission.csv', index=False)
print('The file is ready for submission.')


ctab = pd.crosstab(train_0[target], train_0['month'], normalize='columns')
plt.figure(figsize=(16,5))
ax = sns.heatmap(ctab, annot=True,fmt='.2%', cbar=False,
                linecolor='grey', linewidth=0.5, cmap='Blues')
plt.title('% of rainy days by month')
plt.tight_layout()

