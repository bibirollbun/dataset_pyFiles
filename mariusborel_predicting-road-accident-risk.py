import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
# from ydata_profiling import ProfileReport

from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, BaggingRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, plot_importance, cv
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, SelectKBest, RFE, chi2
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score,
                                     StratifiedKFold)
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer, minmax_scale, 
                                   PowerTransformer, QuantileTransformer, LabelEncoder,
                                   RobustScaler, StandardScaler, FunctionTransformer,
                                   LabelEncoder, OneHotEncoder, OrdinalEncoder)
from sklearn.compose import make_column_transformer

from sklearn.pipeline import make_pipeline

import optuna
from optuna.samplers import TPESampler

from yellowbrick.regressor import ResidualsPlot, PredictionError

import shap

import warnings
warnings.filterwarnings('ignore')

# Set Seaborn theme with dark grid
sns.set_theme(style="darkgrid", palette="Accent_r", font_scale=0.8)

# Update matplotlib parameters for dark background and white labels
plt.rcParams.update({
    'axes.facecolor': '#222222', 
    'figure.facecolor': '#222222', 
    'text.color': '#ff8c00',   
    'axes.labelcolor': '#82e0aa',    
    'xtick.color': '#82e0aa',      
    'ytick.color': '#82e0aa',        
    'grid.color': '#444444',         
    'axes.edgecolor': 'white'        
})

# verify the versions
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'seaborn version: {sns.__version__}')
print(f'optuna version : {optuna.__version__}')


# Set Seaborn theme with dark grid and brighter palette
sns.set_theme(style="darkgrid", palette="Accent", font_scale=0.9)

# Update matplotlib parameters for brighter dark theme
plt.rcParams.update({
    'axes.facecolor': '#333333',       # Slightly lighter than #222222
    'figure.facecolor': '#333333',
    'text.color': '#ffd700',           # Bright gold for better contrast
    'axes.labelcolor': '#ffd700',      # Softer mint green
    'xtick.color': '#ffd700',
    'ytick.color': '#ffd700',
    'grid.color': '#666666',           # Lighter grid lines
    'axes.edgecolor': '#dddddd'        # Light gray edges
})



train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv', index_col='id')
orig = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_2k.csv')

target = 'accident_risk'

train.head()


num_feats = test.select_dtypes(include='number').columns.tolist()
cat_feats = test.select_dtypes(exclude='number').columns.tolist()


# Create the figure and GridSpec layout
fig = plt.figure(figsize=(14, 24))
gs = GridSpec(8, 3, width_ratios=[4, 2, 3])

# Heatmap of correlations
ax0 = fig.add_subplot(gs[:2, 0])
corr_ = train.corr(numeric_only=True)
sns.heatmap(corr_, mask=corr_<0.01,
            annot=True, fmt='.2f',cmap="Accent_r", 
            cbar=False, ax=ax0)

# Distribution of numeric features
for f, num_feat in enumerate(num_feats, start=3):   
    ax2 = fig.add_subplot(gs[f, 0])
    sns.violinplot(train, y=num_feat)

# Distribution of the target
ax1 = fig.add_subplot(gs[-1:, 0])
sns.histplot(train, x=target, bins=30, kde=True, color='gold')


# Distribution of the target vs cat features
for f, cat_feat in enumerate(cat_feats):
    ax2 = fig.add_subplot(gs[f, 1])
    train[cat_feat].value_counts().plot.pie(autopct='%.2f%%',
                                            wedgeprops={'width':.75, 'edgecolor':'grey'}, 
                                            textprops={'color': 'grey'})
    plt.ylabel('')
    plt.title(f'{cat_feat}', color='gold')
    if f!=7:
        plt.xticks([])


# Distribution of the target vs cat features
for f, cat_feat in enumerate(cat_feats):
    ax2 = fig.add_subplot(gs[f, 2])
    sns.kdeplot(train, x=target, hue=cat_feat, fill=True)
    plt.ylabel('')
    if f==0:
        plt.title(f'{target} within cat_features')
    if f!=7:
        plt.xticks([])

plt.show()


def feat_eng(df):
#     # for feat in num_feats:
#     #     feat_range = df[feat].max() - df[feat].min()
#     #     df[feat] = df[feat]/feat_range
    
#     # df['visibility_1'] = df['lighting']+'_'+df['time_of_day']
#     # df['visibility_2'] = df['lighting']+'_'+df['weather']
#     # df['visibility_3'] = df['weather']+'_'+df['time_of_day']
#     # df['curvature*num_lanes'] = df['curvature']*df['num_lanes']
#     # df['speed_limit*num_lanes/curvature'] = df['speed_limit']*df['curvature']/df['num_lanes']

#     # df['made_feat_1'] = df['road_type']+'_'+df['weather']
#     # df['made_feat_2'] = df['time_of_day']+'_'+df['weather']
#     # df['made_feat_3'] = df['lighting']+'_'+df['weather']

#     # try:
#     #     df = df[['speed_limit', 'curvature', 'lighting', 'weather', 'num_reported_accidents', 'made_feat_1', 'made_feat_2', 'made_feat_3', target]]
#     # except:
#     #     df = df[['speed_limit', 'curvature', 'lighting', 'weather', 'num_reported_accidents', 'made_feat_1', 'made_feat_2', 'made_feat_3']]
    
#     return df


# def create_features(df):
#     # Copy dataframe
#     df = df.copy()

    # Polynomial features
    df['curvature_squared'] = df['curvature'] ** 2
    df['curvature_cubed'] = df['curvature'] ** 3
    df['speed_squared'] = df['speed_limit'] ** 2

    # Binned features
    df['curvature_bin'] = pd.cut(df['curvature'], bins=[0, 0.3, 0.6, 1.0], labels=[0, 1, 2])
    df['speed_category'] = pd.cut(df['speed_limit'], bins=[0, 30, 50, 100], labels=[0, 1, 2])

    # Interaction features
    df['speed_curvature'] = df['speed_limit'] * df['curvature']
    df['lanes_curvature'] = df['num_lanes'] * df['curvature']
    df['speed_lanes'] = df['speed_limit'] * df['num_lanes']
    df['accidents_curvature'] = df['num_reported_accidents'] * df['curvature']
    df['accidents_speed'] = df['num_reported_accidents'] * df['speed_limit']

    # Risk combinations
    df['high_risk_combo'] = ((df['curvature'] > 0.5) & (df['speed_limit'] >= 60)).astype(int)
    df['weather_lighting_risk'] = (
        ((df['weather'] == 'foggy') | (df['weather'] == 'rainy')) &
        ((df['lighting'] == 'dim') | (df['lighting'] == 'night'))
    ).astype(int)

    # Derived categorical indicators
    df['is_night'] = (df['lighting'] == 'night').astype(int)
    df['is_bad_weather'] = df['weather'].isin(['foggy', 'rainy']).astype(int)
    df['is_highway'] = (df['road_type'] == 'highway').astype(int)
    df['is_urban'] = (df['road_type'] == 'urban').astype(int)

    # Time-based and holiday proxies
    df['is_peak_time'] = df['time_of_day'].isin(['morning', 'evening']).astype(int)
    df['is_weekend'] = df['holiday'].astype(int)

    # Safety and danger scores
    df['safety_score'] = (
        df['road_signs_present'].astype(int) * 2 +
        (df['lighting'] == 'daylight').astype(int) +
        (df['weather'] == 'clear').astype(int)
    )

    df['danger_score'] = (
        (df['curvature'] > 0.6).astype(int) +
        (df['speed_limit'] >= 60).astype(int) +
        df['is_bad_weather'] +
        df['is_night'] +
        (df['num_reported_accidents'] >= 2).astype(int)
    )

    # Ratio and intensity features
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
    df['risk_intensity'] = df['curvature'] * df['speed_limit'] / 50

    return df

train = feat_eng(train)
test = feat_eng(test)
orig = feat_eng(orig)


orig.info()


feats_to_switch_type = ['num_lanes', 'speed_limit', 'num_reported_accidents']

def change_dtype(df):
    df[feats_to_switch_type] = df[feats_to_switch_type].astype('object')
    
    return df


# train = change_dtype(train)

# orig = change_dtype(orig)

# test = change_dtype(test)

# test.info()


use_original = False

if use_original:
    train_ = pd.concat([train, orig], ignore_index=True)
else:
    train_ = train

X = train_.copy()
y = X.pop(target)


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=4)

[d.shape for d in [X_train, X_valid]]


num_feats = test.select_dtypes(include='number').columns.tolist()
cat_feats = test.select_dtypes(exclude='number').columns.tolist()


scaler = MinMaxScaler()
# encoder = OneHotEncoder()
# encoder = LabelEncoder()
encoder = OrdinalEncoder()

features_trans = make_column_transformer(
    (scaler, num_feats),
    (encoder, cat_feats),
    remainder='drop', 
    sparse_threshold=0
)


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "num_leaves": trial.suggest_int("num_leaves", 4, 128),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "objective": "regression",
       # "device": "gpu"  # GPU acceleration
    }

    model = make_pipeline(features_trans, LGBMRegressor(**params, verbose=-1))
    # scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    # return -scores.mean()

    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    
    return rmse


def Run_Pass_lgbm_study(n_trials=1):
    if n_trials > 1:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=72000, show_progress_bar=True)
        best_study_params = study.best_params

        print(f"Number of finished trials: {len(study.trials)}")
        trial = study.best_trial
        print(f"Best trial RMSE score: {trial.value:.6f}")
    else:
        print("No need to run Optuna, we will use the parameters obtained earlier.")
        
        best_study_params = {'n_estimators': 820, 
                             'learning_rate': 0.1878878051782611, 
                             'num_leaves': 53, 
                             'min_child_samples': 15, 
                             'colsample_bytree': 0.7936807570341036, 
                             'subsample': 0.9503899454448166, 
                             'reg_alpha': 0.09891486220931912, 
                             'reg_lambda': 0.28218192889232424, 
                             'boosting_type': 'dart'}
    
    print(f"Best parameters: {best_study_params}")
    return best_study_params

lgbm_best_params = Run_Pass_lgbm_study(n_trials=50)


model = make_pipeline(features_trans, LGBMRegressor(**lgbm_best_params, verbose=-1))
model.fit(X, y)


feature_names = features_trans.get_feature_names_out()
feature_names.tolist()


# Data preptreatment
X_train_prep = X_train.copy()
X_train_prep = features_trans.fit_transform(X_train_prep)


# Calculate mutual information with a fraction of the train dataset
mi = mutual_info_regression(X_train_prep, y_train)

mi_df = pd.DataFrame({'mi':mi*100}, index=feature_names)

plt.figure(figsize=(10, 8))
ax = mi_df.sort_values(by='mi').plot.barh(title='Mutual information in train dataset')
# plt.xlim([0, 2.3])
for label in ax.containers:
    ax.bar_label(label)
plt.show()


# Extract fitted model
lgb_model = model.named_steps['lgbmregressor']
explainer = shap.Explainer(lgb_model, feature_names=feature_names.tolist())
shap_values = explainer(features_trans.fit_transform(X))

shap.plots.beeswarm(shap_values, axis_color='#ffd700', max_display=15)


fig, ax = plt.subplots(figsize=(8.4, 5))
resid = ResidualsPlot(
    model, 
    train_color='darkgreen',
    test_color='#ffd700',
    # hist=False,
    # qqplot=True
)

resid.fit(X_train, y_train)
resid.score(X_valid, y_valid)
resid.poof();


# n_splits= 6
# my_spliter = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# color = 90
# for f, (tr_ind, va_ind) in enumerate(my_spliter.split(X, y), 1):
#     color+=1
#     print(20*f'\033[{color}m:\033[0m')
#     reg = make_pipeline(features_trans, LGBMRegressor(**lgbm_best_params, verbose=-1))
#     X_tr, X_va = X.iloc[tr_ind], X.iloc[va_ind]
#     y_tr, y_va = y[tr_ind], y[va_ind]
#     print(f'\033[{color}mFitting Fold_{f}\033[0m')
    
#     reg.fit(X_tr, y_tr)

#     preds = reg.predict(X_va)
#     rmse = np.sqrt(mean_squared_error(y_va, preds))
#     print('\033[{}m ===> rmse: {:.5}\n\033[0m'.format(color, rmse))


model.fit(X, y)

subm = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

y_hat_test = model.predict(test)

subm[target] = y_hat_test


sns.histplot(subm, x=target, kde=True, bins=50, palette='Set1', color='gold')
plt.title(f'Distribution of predicted {target} for test data', fontsize=12)
plt.show()


subm.to_csv('submission.csv', index=False)

