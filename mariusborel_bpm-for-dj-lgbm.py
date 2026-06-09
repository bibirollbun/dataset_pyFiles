import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from ydata_profiling import ProfileReport
from itertools import combinations

from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, BaggingRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, plot_importance, cv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score,
                                     StratifiedKFold)
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer, minmax_scale, 
                                   PowerTransformer, QuantileTransformer, LabelEncoder,
                                   RobustScaler, StandardScaler, FunctionTransformer,
                                   LabelEncoder, OneHotEncoder, OrdinalEncoder)
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


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv', index_col='id')

ext_train = pd.read_csv('/kaggle/input/bpm-prediction-challenge/Train.csv')
ext_test = pd.read_csv('/kaggle/input/bpm-prediction-challenge/Test.csv')

target = 'BeatsPerMinute'

train.head()


train.shape


# train = train.query("BeatsPerMinute>120 or BeatsPerMinute<118")

# train.shape


features = train.columns.tolist()[:-1]
features


train.info()


# train_profile_report = ProfileReport(train, title='profile report of test data')

# train_profile_report


columns = train.columns.tolist()


for col in train.columns.tolist():
    plt.figure(figsize=(8, 3))
    plt.subplot(121)
    sns.histplot(train, x=col, kde=True, bins=25)
    plt.subplot(122)
    sns.scatterplot(train, x=col, y=target)
    plt.tight_layout()
    plt.show()


# for col1, col2 in combinations(columns, 2):
#     sns.jointplot(data=train, x=col1, y=col2)
#     plt.show()


# sns.pairplot(train.sample(100), hue=target)
# plt.show()


train_corr = train.corr()

sns.heatmap(train_corr, annot=True, fmt='.2f', cmap='Oranges')
plt.show()


for col in columns:
    print(f'{col} var: {train[col].var()}')


# sns.pairplot(train)
# plt.show()


score_features = train.loc[:, train.columns.str.contains('Score')].columns.tolist()


from itertools import combinations

def data_prep(df):
    df = df.copy()
    #df['TrackDurationMs_transformed'] = df['TrackDurationMs']/60000 - 12
    for feat in test.columns.tolist():
        df[feat] = np.abs(df[feat])
        feat_var = df[feat].max()
        df[feat] = df[feat]/feat_var
    # # Add a new features that is the sum of all composition features
    # df['sum_Scores'] = np.sum(df[score_features], axis=1)
    # for feat in score_features:
    #     df[f'{feat}_ratio']  = df[feat]/df['sum_Scores']
    #     df[f'{feat}_ratio_unitime'] = df[f'{feat}_ratio']/df['TrackDurationMs_transformed']
    # Use combinations to avoid duplicates and self-division
    # for col1, col2 in combinations(features, 2):
    #     df[f'sin_{col1}'] = np.sin(2*np.pi*df[col1])
    #     df[f'cos_{col1}'] = np.cos(2*np.pi*df[col1])
    #     df[f"{col1}_div_{col2}"] = df[col1] / df[col2]
    #     df[f'log_{col1}'] = np.log(np.abs(df[col1]))
        # df[f'square_{col1}'] = df[col1]**2
        # df[f'cube_{col1}'] = df[col1]**2//df[col2]
    # df = df.drop(columns=['TrackDurationMs'])

    return df


train = pd.concat([train, ext_train], ignore_index=True)

train = data_prep(train)
test = data_prep(test)

ext_train = data_prep(ext_train)

train.head()


X = train.copy()
y = X.pop(target)


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

[d.shape for d in [X_train, X_valid]]


X_ext = ext_train.copy()
y_ext = X_ext.pop(target)

X_ext_train, X_ext_valid, y_ext_train, y_ext_valid = train_test_split(X_ext, y_ext, test_size=0.2, random_state=42)

[d.shape for d in [X_ext_train, X_ext_valid]]


scaler = StandardScaler()

def objective(trial):
    lgb_param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.1, 1),
        "num_leaves": trial.suggest_int("num_leaves", 5, 256),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 10),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 1.0),
        "objective": trial.suggest_categorical("objective", ["regression", "regression_l1"]),
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"])
    }

    model = make_pipeline(scaler, LGBMRegressor(**lgb_param_grid, verbose=-1))
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

def Run_Pass_lgbm_study(n_trials=1):
    if n_trials > 1:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=36000, show_progress_bar=True)
        best_study_params = study.best_params

        print(f"Number of finished trials: {len(study.trials)}")
        trial = study.best_trial
        print(f"Best trial RMSE score: {trial.value:.6f}")
    else:
        print("No need to run Optuna, we will use the parameters obtained earlier.")
        
        best_study_params = {'n_estimators': 930, 
                             'learning_rate': 0.4506466371411728, 
                             'num_leaves': 2, 
                             'min_child_samples': 26, 
                             'colsample_bytree': 0.6800294403838644, 
                             'subsample': 0.7530807934237363, 
                             'reg_alpha': 0.13859467915971238, 
                             'reg_lambda': 0.3162818199177214, 
                             'objective': 'regression', 
                             'boosting_type': 'dart'}

    print(f"Best parameters: {best_study_params}")
    return best_study_params

lgbm_best_params = Run_Pass_lgbm_study(n_trials=100)


# reg = RandomForestRegressor(n_estimators=500)
reg = LGBMRegressor(**lgbm_best_params, verbose=-1)
scaler = StandardScaler()

reg_pipe = make_pipeline(scaler, reg)

reg_pipe.fit(X_train, y_train)


explainer = shap.Explainer(reg)

shap_values = explainer(X)

# shap.summary_plot(s_values, features=X, feature_names=X.columns, plot_type='bar', axis_color='yellow')


shap.summary_plot(shap_values.values, features=X, feature_names=X.columns, plot_type='violin', axis_color='gold')


s_values = shap_values.values


y_hat_valid = reg_pipe.predict(X_valid)


rmse_valid = np.sqrt(mean_squared_error(y_valid, y_hat_valid))
rmse_valid


plt.figure(figsize=(8, 4))
plt.subplot(121)
sns.histplot(y_valid.values, kde=True, bins=25)
sns.histplot(y_hat_valid, kde=True, bins=25)
plt.xlabel(target)
plt.subplot(122)
sns.regplot(x=y_valid, y=y_hat_valid)
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.suptitle(f'Compare Predicted and true values: RMSE {rmse_valid.round(6)}')
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(8.4, 5))
resid = ResidualsPlot(
    reg_pipe, 
    train_color='darkgreen',
    test_color='darkorange',
    # hist=False,
    # qqplot=True
)
resid.fit(X_train, y_train)
resid.score(X_valid, y_valid)
resid.poof();


fig, ax = plt.subplots(figsize=(8.4, 5))
resid = ResidualsPlot(
    reg_pipe, 
    train_color='darkgreen',
    test_color='darkorange',
    # hist=False,
    # qqplot=True
)
resid.fit(X_ext_train, y_ext_train)
resid.score(X_ext_valid, y_ext_valid)
resid.poof();


reg_pipe.fit(X, y)

subm = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')

y_hat_test = reg_pipe.predict(test)

subm[target] = y_hat_test


sns.histplot(subm, x=target, kde=True, bins=50)
plt.title(f'Distribution of predicted {target} for test data', fontsize=12)
plt.show()


subm.to_csv('submission.csv', index=False)

