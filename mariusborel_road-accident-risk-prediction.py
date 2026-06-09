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

# Set Seaborn theme with dark grid and brighter palette
my_palette = 'tab20'
sns.set_theme(style="darkgrid", palette=my_palette, font_scale=0.9)

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
# verify the versions
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'seaborn version: {sns.__version__}')
print(f'optuna version : {optuna.__version__}')


train_raw = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col='id')
test_raw = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv', index_col='id')
orig_100k_raw = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')
orig_10k_raw = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_10k.csv')

target = 'accident_risk'

train_raw.head()


# Drop duplicate rows
train = train_raw.drop_duplicates(subset=test_raw.columns.tolist())
orig_10k = orig_10k_raw.drop_duplicates(subset=test_raw.columns.tolist())
orig_100k = orig_100k_raw.drop_duplicates(subset=test_raw.columns.tolist())
test = test_raw.copy()


def percentage_of_duplicates(df_raw, df):
    return (len(df_raw) - len(df))*100/len(df_raw)


# Duplicates in train data
print('{:.2f} % of the rows in the train set are duplicates and have been droped.'.format(percentage_of_duplicates(train_raw, train)))


num_feats = test.select_dtypes(include='number').columns.tolist()
cat_feats = test.select_dtypes(exclude='number').columns.tolist()


plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', 
       labelweight='light', 
       labelsize='small',
       titleweight='bold', 
       titlesize=16, titlepad=10)


# Create the figure and GridSpec layout
fig = plt.figure(figsize=(8, 6))
gs = GridSpec(3, 4, width_ratios=[3, 3, 3, 3], height_ratios=[3, 0.1, 4])

# Distribution of numeric features
for f, num_feat in enumerate(num_feats, start=0):   
    ax2 = fig.add_subplot(gs[0, f])
    sns.violinplot(train, x=num_feat, color="#C4B454")
    plt.xlabel(num_feat, color='#C4B454', fontsize=12)

# Distribution of the target
ax1 = fig.add_subplot(gs[2:, :])
sns.histplot(train, x=target, bins=30, kde=True, color='#8A9A5B')
plt.title(f'Distribution of {target} in train data', color='#8A9A5B')

plt.tight_layout()


# Create the figure and GridSpec layout
fig = plt.figure(figsize=(10, 22))
gs = GridSpec(8, 2, width_ratios=[4, 2])

# Distribution of the target vs cat features
for f, cat_feat in enumerate(cat_feats):
    ax2 = fig.add_subplot(gs[f, 0])
    train[cat_feat].value_counts().plot.pie(autopct='%.2f%%', cmap=my_palette,
                                            wedgeprops={'width':.75, 'edgecolor':'grey'}, 
                                            textprops={'color': '#009E60'})
    plt.ylabel('')
    plt.title(f'{cat_feat}', size=12, color='#8A9A5B')
    if f!=7:
        plt.xticks([])


# Distribution of the target vs cat features
for f, cat_feat in enumerate(cat_feats):
    ax2 = fig.add_subplot(gs[f, 1])
    sns.kdeplot(train, x=target, 
                hue=cat_feat, 
                palette=my_palette, 
                fill=True)
    plt.title(f'{cat_feat}', size=12, color='#8A9A5B')
    plt.ylabel('')
    # if f==0:
    #     plt.title(f'{target} within cat_features')
    if f!=7:
        plt.xlabel('')

plt.show()


plt.figure(figsize=(7.8, 6))
corr_ = np.abs(train.corr(numeric_only=True))
sns.heatmap(corr_, 
            mask=corr_<0.1,
            annot=True, fmt='.2f',
            cmap=f'{my_palette}_r',
            cbar=False)
plt.title('Correlation of the Features', color='#8A9A5B')
plt.show()


feats_to_switch_type = ['num_lanes', 'speed_limit', 'num_reported_accidents']

def change_dtype(df):
    df[feats_to_switch_type] = df[feats_to_switch_type].astype('object')
    
    return df


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


# scaler = MinMaxScaler()
scaler = StandardScaler()
encoder = OneHotEncoder(drop='first')
# encoder = LabelEncoder()
# encoder = OrdinalEncoder()

features_trans = make_column_transformer(
    (scaler, num_feats),
    (encoder, cat_feats),
    remainder='drop', 
    sparse_threshold=0
)


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "num_leaves": trial.suggest_int("num_leaves", 4, 128),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1.0),
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "objective": "regression",
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        "max_bin": trial.suggest_int("max_bin", 100, 255),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 2.0),
        "importance_type": trial.suggest_categorical("importance_type", ["split", "gain"]),
        "device": "gpu"  # Uncomment if GPU is available
    }

    model = make_pipeline(
        features_trans,
        LGBMRegressor(**params, verbose=-1)
    )

    if cv_scorer:
        # Cross-validation (recommended)
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
        return -scores.mean()
    else:
        # Alternatively
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
        
        best_study_params = Best parameters: {'n_estimators': 540, 
                                              'learning_rate': 0.7379564665630974, 
                                              'max_depth': 9, 
                                              'num_leaves': 77,
                                              'min_child_samples': 20, 
                                              'colsample_bytree': 0.9826598299772908, 
                                              'subsample': 0.5629935268451077, 
                                              'reg_alpha': 0.35762936199895834, 
                                              'reg_lambda': 0.23717215724594404, 
                                              'boosting_type': 'dart', 
                                              'min_split_gain': 0.0009686342910453663, 
                                              'max_bin': 216,
                                              'scale_pos_weight': 0.66801181586939, 
                                              'importance_type': 'split'}
    
    print(f"\nBest parameters: {best_study_params}")
    return best_study_params


# Decide how optuna is scored
cv_scorer=False

# Run the optimization
lgbm_best_params = Run_Pass_lgbm_study(n_trials=200)


model = make_pipeline(features_trans, LGBMRegressor(**lgbm_best_params, verbose=-1))
model.fit(X, y)


# Get the features names after preprocessing
feature_names = features_trans.get_feature_names_out()

# Data preptreatment
X_train_prep = X_train.copy()
X_train_prep = features_trans.fit_transform(X_train_prep)


# Calculate mutual information with a fraction of the train dataset
mi = mutual_info_regression(X_train_prep, y_train)

mi_df = pd.DataFrame({'mi':mi*100}, index=feature_names)

ax = mi_df.sort_values(by='mi').plot.barh(title='Mutual information in train dataset', color='#8A9A5B', figsize=(8, 5))
plt.xlim([0, 32])
for label in ax.containers:
    ax.bar_label(label)
plt.show()


splits = 5
seed = 4

spliter = KFold(n_splits=splits, shuffle=True, random_state=seed)

for f, (trn_ind, val_ind) in enumerate(spliter.split(X, y), start=1):

    X_trn, X_val = X.iloc[trn_ind], X.iloc[val_ind]
    y_trn, y_val = y.iloc[trn_ind], y.iloc[val_ind]

    model = make_pipeline(features_trans, LGBMRegressor(**lgbm_best_params, verbose=-1))
    clf = model.fit(X_trn, y_trn)

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print('\033[{}m\nâ–·â–·â–· Fold_{}: RMSE = {:.6f}\033[0m'.format(90+f, f, rmse))


model.fit(X, y)

subm = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

y_hat_test = model.predict(test)

subm[target] = y_hat_test


sns.histplot(subm, x=target, kde=True, bins=50, palette='Set1', color='green')
plt.title(f'Distribution of predicted {target} for test data', fontsize=12)
plt.show()


subm.to_csv('submission.csv', index=False)
print('\nğŸ�¾ğŸ�¾ğŸ�¾ The submission file is ready!ğŸ�…ğŸª„ğŸª„ğŸª„')

