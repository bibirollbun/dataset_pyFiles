# Setup & Read File

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

import optuna
from optuna.pruners import SuccessiveHalvingPruner

sns.set(style = 'darkgrid')

file_str = '/kaggle/input/playground-series-s5e8/'
df_train = pd.read_csv(file_str + 'train.csv', index_col = 'id')
df_test = pd.read_csv(file_str + 'test.csv', index_col = 'id')
df = pd.concat([df_train, df_test])
for col in df.select_dtypes('object'):
    df[col] = df[col].astype('category')
df_train = df.loc[df_train.index]
df_test = df.loc[df_test.index]
df_test.pop('y')

rd_seed = 1


# Shape of Datasets

print('Train Data Shape:', df_train.shape)
print('Test Data Shape:', df_test.shape)


# Train Data Overview

df_train.info()
df_train.sample(3)


# Test Data Overview

df_test.info()
df_test.sample(3)


# Change Datatypes

df_train['y'] = df_train['y'].astype('int')
df['y'] = df['y'].fillna(0)
df['y'] = df['y'].astype('int')

yes_no = {'yes': 1, 'no': 0}
for col in ['default', 'housing', 'loan']:
    df_train[col] = df_train[col].map(yes_no).astype('int')
    df_test[col] = df_test[col].map(yes_no).astype('int')
    df[col] = df[col].map(yes_no).astype('int')

months = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
          'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
df_train['month'] = df_train['month'].map(months).astype('int')
df_test['month'] = df_test['month'].map(months).astype('int')
df['month'] = df['month'].map(months).astype('int')


# Score Dataset & Get Baseline Score

def score_dataset(X, y, model = XGBRegressor(random_state = rd_seed)):
    for col in X.select_dtypes('category'):
        X[col] = X[col].cat.codes
    scores = cross_val_score(model, X, y, cv = 5, scoring = 'roc_auc')
    return scores.mean()

X = df_train.copy()
y = X.pop('y')
baseline_score = score_dataset(X, y)
print('Baseline Score: {:.5f} Roc Auc'.format(baseline_score))


# Get MI Scores

def get_mi_scores(X, y):
    for col in X.select_dtypes(['object', 'category']):
        X[col], _ = X[col].factorize()
    mi_scores = mutual_info_regression(X, y, random_state = rd_seed)
    mi_scores = pd.Series(mi_scores, name = 'MI Scores', index = X.columns)
    mi_scores = mi_scores.sort_values(ascending = False)
    return mi_scores

X = df_train.copy()
y = X.pop('y')
scores = get_mi_scores(X, y)
print(scores)


# Plot MI Scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending = True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title('Mutual Information Scores')

plot_mi_scores(scores)


# Get Score after Dropping

X = df_train.copy()
y = X.pop('y')
X.drop('default', axis = 1)
score = score_dataset(X, y)
print('Score after Dropping: {:.5f} Roc Auc'.format(score))


# Get Correlation Heatmaps

def get_corr_heatmap(df, ax, title):
    df = df.copy()
    df = df.select_dtypes('int')
    sns.heatmap(df.corr(), ax = ax, annot = True, square = True, cmap = 'coolwarm', annot_kws = {'size': 12})
    ax.set_title(title, size = 15)

fig, axs = plt.subplots(nrows = 2, figsize = (20, 20))
get_corr_heatmap(df_train, axs[0], 'Training Set Correlations')
get_corr_heatmap(df_test, axs[1], 'Test Set Correlations')


# Combine pdays and previous & Get MI Scores of Combined Features 

combined_features = pd.DataFrame()
X = df_train.copy()
y = X.pop('y')
combined_features['add_pdays_prev'] = X.previous + X.pdays
combined_features['sub_pdays_prev'] = X.previous - X.pdays
combined_features['mul_pdays_prev'] = X.previous * X.pdays
combined_features['div_pdays_prev'] = X.previous / X.pdays
combined_features['div_pdays_prev'] = combined_features['div_pdays_prev'].fillna(0)
get_mi_scores(combined_features, y)


# Mathematical Transformation

def math_transform(X):
    X = X.copy()
    X_new = pd.DataFrame()
    X_new['add_pdays_prev'] = X.previous + X.pdays
    # X_new['memory'] = pd.Series()
    # for id in X.index:
    #     num = X.loc[id, 'pdays']
    #     if num == -1:
    #         X_new.loc[id, 'memory'] = 0
    #     elif num == 0:
    #         X_new.loc[id, 'memory'] = 100
    #     else:
    #         X_new.loc[id, 'memory'] = 184 / (math.log(num * 1440) ** 1.25 + 1.84)
    X_new['avgtime'] = X.duration / X.campaign
    X_new['calldiff'] = X.previous - X_new.avgtime
    # print(get_mi_scores(X_new, y))
    return X_new


# Concatenate

def concatenate(X):
    X = X.copy()
    X_new = pd.DataFrame()
    # date
    day_counts = {1: 0, 2: 31, 3: 59, 4: 90, 5: 120, 6: 151,
                  7: 181, 8: 212, 9: 243, 10: 273, 11: 303, 12: 334}
    X_new['date'] = X.month.map(day_counts) + X.day
    # personalstat
    for col in ['job', 'marital', 'education']:
        X[col], _ = X[col].factorize()
    X_new['personalstat'] = X.job + X.marital * 12
    # X_new['personalstat'] = X_new['personalstat'] + X.education * 48
    # loanstat
    X_new['loanstat'] = X.housing + X.loan * 2
    # print(get_mi_scores(X_new, y))
    return X_new


# Group Transformation

def group_feature(X, X_new, feature_1, feature_2, strategy):
    feature_name = 'group_' + feature_1 + '_' + feature_2 + '_' + strategy
    X_new[feature_name] = X.groupby(feature_1, observed = False)[feature_2].transform(strategy)

def group_transform(X):
    X = X.copy()
    X_new = pd.DataFrame()
    group_feature(X, X_new, 'age', 'housing', 'mean')
    group_feature(X, X_new, 'month', 'housing', 'mean')
    group_feature(X, X_new, 'contact', 'duration', 'median')
    group_feature(X, X_new, 'job', 'balance', 'mean')
    group_feature(X, X_new, 'poutcome', 'duration', 'median')
    # print(get_mi_scores(X_new, y))
    return X_new


# Bundle Them Together

def manual_creation(X):
    X_new = math_transform(X)
    X_new = X_new.join(concatenate(X))
    X_new = X_new.join(group_transform(X))
    return X_new


# Score Dataset with New Features

X = df_train.copy()
X = X.join(manual_creation(X))
y = X.pop('y')

score = score_dataset(X, y)
print('Score: {:.5f} Roc Auc'.format(score))


# K-Means Clustering

def clustering(X, features, n_clusters = 20):
    X = X.copy()
    X_scaled = X.loc[:, features]
    # Standardize
    X_scaled = (X_scaled - X_scaled.mean(axis = 0)) / X_scaled.std(axis = 0)
    kmeans = KMeans(n_clusters = n_clusters, n_init = 10, max_iter = 300, random_state = rd_seed)
    X_new = pd.DataFrame(kmeans.fit_transform(X_scaled))
    X_new.columns = ['centroid_' + str(i + 1) for i in range(n_clusters)]
    X_new['cluster'] = kmeans.fit_predict(X_scaled)
    # print(get_mi_scores(X_new, y))
    return X_new


# Score Dataset with New Features

kmeans_features = ['avgtime', 'duration', 'calldiff']
X = df_train.copy()
X = X.join(manual_creation(X))
X = X.join(clustering(X, kmeans_features))
y = X.pop('y')

score = score_dataset(X, y)
print('Score: {:.5f} Roc Auc'.format(score))


# Apply Principal Component Analysis

def apply_pca(X, features):
    X = X.copy().loc[:, features]
    X_scaled = (X - X.mean(axis = 0)) / X.std(axis = 0)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    columns = ['pc_' + str(i + 1) for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns = columns)
    # Create Loadings
    loadings = pd.DataFrame(pca.components_.T, columns = columns, index = X_scaled.columns)
    return pca, X_pca, loadings


# Examine Loadings

pca_features = ['avgtime', 'duration', 'calldiff']
X = df_train.copy()
X = X.join(manual_creation(X))
X = X.join(clustering(X, kmeans_features))
pca, X_pca, loadings = apply_pca(X, pca_features)
print(loadings)
X = X.join(X_pca)


# Create New Features Inspired by PCA

def pca_inspired(X):
    X_new = pd.DataFrame()
    # X_new['pc_1_inspired'] = X.avgtime + X.duration - 2 * X.calldiff
    # X_new['pc_2_inspired'] = X.calldiff - X.avgtime + 5 * X.duration
    X_new['pc_3_inspired'] = X.avgtime * X.calldiff
    # print(get_mi_scores(X_new, y))
    return X_new


# Score Dataset with New Features

pca_features = ['avgtime', 'duration', 'calldiff']
X = df_train.copy()
X = X.join(manual_creation(X))
X = X.join(clustering(X, kmeans_features))
pca, X_pca, loadings = apply_pca(X, pca_features)
X = X.join(X_pca)
X = X.join(pca_inspired(X))
y = X.pop('y')

score = score_dataset(X, y)
print('Score: {:.5f} Roc Auc'.format(score))


# Get Final Features

def create_features(df_train, df_test = None):
    X = df_train.copy()
    y = X.pop('y')

    # When we need to create features in test data
    if df_test is not None:
        X_test = df_test.copy()
        X = pd.concat([X, X_test])

    # Transformations & Clustering & PCA
    kmeans_features = ['avgtime', 'duration', 'calldiff']
    pca_features = ['avgtime', 'duration', 'calldiff']
    X = X.join(manual_creation(X))
    X = X.join(clustering(X, kmeans_features))
    pca, X_pca, loadings = apply_pca(X, pca_features)
    X = X.join(X_pca)
    X = X.join(pca_inspired(X))

    # Split Train and Test
    if df_test is not None:
        X_test = X.loc[df_test.index, :]
        X.drop(X_test.index, inplace = True)
    
    # One-hot Encoding
    # X_cate = X.select_dtypes('category')
    # encoder = OneHotEncoder(handle_unknown = 'ignore', sparse = False)
    # X_oh = pd.DataFrame(encoder.fit_transform(X_cate))
    # X.drop(X_cate, axis = 1, inplace = True)
    # X = X.join(X_oh)
    # if df_test is not None:
    #     X_test_cate = X_test.select_dtypes('category')
    #     X_test_oh = pd.DataFrame(encoder.transform(X_cate))
    #     X_test.drop(X_cate, axis = 1, inplace = True)
    #     X_test = X_test.join(X_test_oh)

    if df_test is not None:
        return X, X_test
    else:
        return X


# Get Hyperparameters & Train Model

X = create_features(df_train)
y = df_train.loc[:, 'y']

def objective(trial):
    xgb_model = XGBClassifier(
        max_depth = trial.suggest_int('max_depth', 2, 10),
        learning_rate = trial.suggest_float('learning_rate', 1e-2, 1e-1, log = True),
        n_estimators = trial.suggest_int('n_estimators', 1000, 4000),
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10),
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.2, 1.0),
        subsample = trial.suggest_float('subsample', 0.2, 1.0),
        reg_alpha = trial.suggest_float('reg_alpha', 1e-4, 1e2, log = True),
        reg_lambda = trial.suggest_float('reg_lambda', 1e-4, 1e2, log = True),
        enable_categorical = True,
        early_stopping_rounds = 50,
        eval_metric = 'auc',
        callbacks = [
            EarlyStopping(rounds = 50)
        ],
        n_jobs = 4,
        random_state = rd_seed
    )
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = rd_seed)
    xgb_model.fit(
        X_train, y_train,
        eval_set = [(X_valid, y_valid)],
        verbose = False
    )
    return xgb_model.best_score

# study = optuna.create_study(direction = 'maximize', pruner = SuccessiveHalvingPruner())
# study.optimize(objective, n_trials = 20)

xgb_params = {
    'max_depth': 7,
    'learning_rate': 0.04803523704024523,
    'n_estimators': 8000,
    'min_child_weight': 7,
    'colsample_bytree': 0.3583806501381003,
    'subsample': 0.9846293307245002,
    'reg_alpha': 13.600428517119232,
    'reg_lambda': 0.00033439299405024547,
    'enable_categorical': True,
    'early_stopping_rounds': 55,
    'eval_metric': 'auc',
    'callbacks': [
        EarlyStopping(rounds = 55)
    ],
    'n_jobs': 4,
    'random_state': rd_seed
}
xgb_model_best = XGBRegressor(**xgb_params)


# Get Predictions and Output!

X, X_test = create_features(df_train, df_test)
y = df_train.loc[:, 'y']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = rd_seed)

xgb_model_best.fit(
    X_train, y_train,
    eval_set = [(X_valid, y_valid)],
    verbose = False
)

preds_valid = xgb_model_best.predict(X_valid)
score = roc_auc_score(y_valid, preds_valid)
print('Score: {:.5f} Roc Auc'.format(score))

preds_test = xgb_model_best.predict(X_test)
output = pd.DataFrame({'id': X_test.index, 'y': preds_test})
output.to_csv('submission.csv', index = False)

