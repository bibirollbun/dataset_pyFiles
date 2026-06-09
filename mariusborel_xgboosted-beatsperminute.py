# import the required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, 
RepeatedKFold,RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score,StratifiedKFold)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer, minmax_scale, 
                                   PowerTransformer, QuantileTransformer, LabelEncoder,
                                   RobustScaler, StandardScaler, FunctionTransformer,
                                   LabelEncoder, OneHotEncoder, OrdinalEncoder)

import optuna
from optuna.samplers import TPESampler

import warnings
warnings.filterwarnings('ignore')

# Print versions
print(f'numpy version: {np.__version__}')
print(f'numpy version: {pd.__version__}')
print(f'numpy version: {sns.__version__}')


# Load the train dataset
train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv', index_col='id')
# Load the test dataset
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv', index_col='id')

# Load the externa train dataset
ext_train = pd.read_csv('/kaggle/input/bpm-prediction-challenge/Train.csv')

target = 'BeatsPerMinute'

features = test.columns.tolist()


# Preview the dataset
train.head()


# Check the shape of the dataset
train.shape


def descriptor(df):
    num_described = df.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
    num_described['n_unique'] = df.nunique() 
    num_described['missing'] = df.isna().sum()
    num_described['%_missing'] = df.isna().sum()/df.shape[0]
    num_described['dtype'] = df.dtypes
    return num_described

descriptor(train)


plt.figure(figsize=(10,5))
for i, feat in enumerate(list(train)):
    plt.subplot(2, 5, i+1)
    sns.kdeplot(train, x=feat, fill=True, alpha=0.5, color='grey')
    if i!=0:
        plt.ylabel('')
plt.tight_layout()


plt.figure(figsize=(10, 3.6))
for i, feat in enumerate(list(train)):
    plt.subplot(2, 5, i+1)
    sns.boxenplot(train, x=feat, color='grey')
    if i!= 0:
        plt.ylabel('')
plt.tight_layout()
plt.show()

print('Some few outliers can be seen for some of the features')


feat_corr_target = train.iloc[:, :-1].corrwith(train[target])
print(feat_corr_target.sort_values(ascending=False))

plt.figure(figsize=(8,4))
ax = feat_corr_target.sort_values().plot.barh(color='grey')
# for corr in ax.containers:
#     ax.bar_label(corr)
plt.title('Correlation of features with the target')
plt.xlabel('Correlation with Strength')
plt.tight_layout()
plt.show()


mutual_correlation = train.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(mutual_correlation, 
            annot=True, fmt='.3f', 
            cmap='gist_gray_r', 
            cbar=False, 
            # square=True
           )
plt.tight_layout()


# creat a list of score related features
score_features = train.loc[:, train.columns.str.contains('Score')].columns.tolist()

# Define a function to preprocess the data
def data_preparator(df):
    df = df.copy()
    
    # df['sum_Scores'] = np.sum(df[score_features], axis=1)
    # for feat in score_features:
    #     df[f'{feat}_ratio']  = df[feat]/df['sum_Scores']
    # df['TrackDuration_Tr'] = df['TrackDurationMs']/60000
    # df = df.drop(columns=['TrackDurationMs'])

    return df  


train = data_preparator(train)
test = data_preparator(test)


# Separate the data from the target
df_data = train.copy()
df_target = df_data.pop(target)


[d.shape for d in [df_data, df_target]]


# Add a new features that is the sum of all composition features
df_data['sum_Scores'] = np.sum(df_data[score_features], axis=1)

# Preview
df_data.sample(5)


# Calculate the ratio of each element in the composition
# for feat in list(df_data.iloc[:, :-1]):
for feat in score_features:
    df_data[f'{feat}_ratio']  = df_data[feat]/df_data['sum_Scores']
    # df_data = df_data.drop(columns=feat)
        
df_data.sample(5)


# correlation of data with target. 
data_corr_target = np.abs(df_data.corrwith(df_target)) # absolute value for ease to compare
data_corr_target.sort_values().plot.barh(color='grey')
plt.xlabel('Correlation with Strength')
plt.title(f'Absolute correlation of features/ratios with {target}')
plt.show()


comp_mutual_correlation = df_data.corr()

plt.figure(figsize=(9, 6))
sns.heatmap(comp_mutual_correlation, annot=True, fmt='.2f', cmap='gist_gray', cbar=False)
plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import mean_squared_error as mse, r2_score as r2_s
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest, RFE, chi2


# Split the datasets
X_train, X_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.25, random_state=5368)

# Check the shapes
[d.shape for d in [X_train, X_test, y_train, y_test]]


scaler = RobustScaler()

def objective(trial):
    xgb_param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 5000, 10000, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1),
        "num_leaves": trial.suggest_int("num_leaves", 2, 512),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 0.05),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 0.05),
    }

    model = make_pipeline(scaler, XGBRegressor(**xgb_param_grid, verbose=-1))
    
    X_train, X_val, y_train, y_val = train_test_split(df_data, df_target, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

def Run_Pass_xgb_study(n_trials=1):
    if n_trials > 1:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=36000, show_progress_bar=True)
        best_study_params = study.best_params

        print(f"Number of finished trials: {len(study.trials)}")
        trial = study.best_trial
        print(f"Best trial RMSE score: {trial.value:.6f}")
    else:
        print("No need to run Optuna, we will use the parameters obtained earlier.")
        
        best_study_params = {'n_estimators': 1030, 
                             'learning_rate': 0.4506466371411728, 
                             'num_leaves': 2, 
                             'min_child_samples': 26, 
                             'colsample_bytree': 0.6800294403838644, 
                             'subsample': 0.7530807934237363, 
                             'reg_alpha': 0.13859467915971238, 
                             'reg_lambda': 0.3162818199177214
                            }

    print(f"Best parameters: {best_study_params}")
    return best_study_params

xgb_best_params = Run_Pass_xgb_study(n_trials=100)


xgb_pipe = make_pipeline(scaler, XGBRegressor(**xgb_best_params)).fit(X_train, y_train)


n_splits = 9
scaler = StandardScaler()

spliter = KFold(n_splits=n_splits, random_state=42578, shuffle=True)

X, y = df_data, df_target

for f, (tr_ind, val_ind) in enumerate(spliter.split(X, y), start=1):
    X_tr, X_va = X.iloc[tr_ind], X.iloc[val_ind]
    # y_tr, y_va = y.iloc[tr_ind], y.iloc[val_ind]
    y_tr, y_va = y[tr_ind], y[val_ind]

    reg_pipe = make_pipeline(scaler, XGBRegressor(**xgb_best_params))

    reg_pipe.fit(X_tr, y_tr)
    y_va_hat = reg_pipe.predict(X_va)
    rmse = np.sqrt(mean_squared_error(y_va_hat, y_va))

    print(f'•••> Fold_{f} rmse: {rmse.round(6)} ✓')


test_predictions = xgb_pipe.predict(test)


submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')

# submission[target] = np.expm1(test_predictions)

from scipy.special import inv_boxcox

y_pred = xgb_pipe.predict(test)

submission[target] = y_pred


sns.histplot(submission, x=target, bins=50, kde=True, color='green')
sns.histplot(train, x=target, bins=50, kde=True, color='orange')
plt.show()


submission.to_csv('submission.csv', index=False)

