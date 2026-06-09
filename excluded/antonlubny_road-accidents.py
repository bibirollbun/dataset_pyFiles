# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn import metrics

from xgboost import XGBRegressor
from xgboost import plot_importance
import xgboost as xgb

pd.set_option('display.max_columns', None)


pip install --upgrade xgboost scikit-learn


df=pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
X_test=pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
X=df.copy()
y=X.pop('accident_risk')
test_ids=X_test['id']

target = df.columns.tolist()[-1]


# creating Score table to compare scores
Scoresy=pd.DataFrame()
Scoresy['experiment_name'] = pd.Series()

## Defining two helper functions: first for scoring, and second for saving score to table
def get_scored(X,y):    
    model = XGBRegressor(enable_categorical=True,) #tree_method='hist', device='cuda')
    score = cross_val_score(model,X,y,cv=5,scoring='neg_mean_squared_error')
    print(f'RMSE on 5 folds {np.sqrt(-1*score.mean())}')
    return score

def save_score(score, Scoresy, name):
    Scoresy = pd.concat([Scoresy, pd.DataFrame(np.sqrt(-1*score)).T],ignore_index=True)
    Scoresy['mean']= Scoresy[[0,1,2,3,4]].mean(axis=1)
    Scoresy['std'] = Scoresy[[0,1,2,3,4]].std(axis=1)
    Scoresy.iloc[-1, 0]=name
    return Scoresy


## Defining two helper functions: first for scoring, and second for saving score to table
Scoresy_m2 = pd.DataFrame()

def add_row(df, **kwargs):
    new_row = pd.DataFrame([kwargs])
    return pd.concat([df, new_row], ignore_index=True)

def get_scored_m2(X,y):
    # Prepare DMatrix for XGBoost since i'm going to use .cv()
    dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)

    xgb_params = {
        'max_depth': 11, 'learning_rate': 0.011,
        'subsample': 0.8, 'colsample_bytree': 0.82,
        'min_child_weight': 3, 'gamma': 0.011,
        'reg_alpha': 0.12, 'reg_lambda': 0.4,
        'max_delta_step': 1, 'colsample_bylevel': 0.86,
        'colsample_bynode': 0.88, 'scale_pos_weight': 0.36,
        'max_bin': 512, 'tree_method': 'hist', "device":"cuda",
        'eval_metric': 'rmse', 'random_state': 42,
    }
    
    # Run cross-validation
    cv_results = xgb.cv(
        params=xgb_params,
        dtrain=dtrain,
        nfold=5,
        num_boost_round=2000,
        metrics='rmse',
        verbose_eval=100,
        early_stopping_rounds=50
    )
    
    print(cv_results.tail())
    
    # Extract best boosting round
    best_round = cv_results['test-rmse-mean'].idxmin()
    best_rmse = cv_results['test-rmse-mean'][best_round]
    print(f"Best round: {best_round}, Best CV RMSE: {best_rmse:.7f}")
    return best_rmse, best_round, cv_results


# turning 'objects' dtype to 'category'
for col in X.columns:
    if X[col].dtype == 'object':
         X[col] = X[col].astype('category')


X.info()


# 'Heavy weighted Baaseline'
best_rmse, best_round, cv_result =  get_scored_m2(X,y)
Scoresy_m2 = add_row(Scoresy_m2, name='baseline', best_round=best_round, best_rmse=best_rmse)
Scoresy_m2


# Let's create first light weited Baseline!
Score = get_scored(X,y)
Scoresy = save_score(Score, Scoresy, 'Baseline')
Scoresy


sns.color_palette("viridis", as_cmap=True)


df_frac = df.sample(n =20000)


sns.catplot(data = df, x='road_type', y='accident_risk', kind="box",height=5, aspect=2, hue='num_lanes',palette="viridis")


sns.stripplot(data=df_frac, x='accident_risk' ,y='curvature', hue= 'public_road')


sns.barplot(data = df, y='accident_risk', x='weather', palette="viridis")


sns.barplot(data = df, y='accident_risk', x='time_of_day',hue='lighting', palette="viridis")


def preprocess_first_round(df):
    df = df.drop(columns=['id'], errors='ignore')
    df['num_lanes_cat']=df['num_lanes']
    df['num_lanes_cat']=df['num_lanes_cat'].astype('category')
    df['weather'] = df['weather'].map({'clear': 0, 'foggy': 1, 'rainy':2})
    df['weather_quality'] = df['weather'].astype(int)
    df['lighting']=df['lighting'].map({'night':0, 'dim':1, 'daylight' : 2})
    df['lighting_quality'] = df['lighting'].astype(int)
    # bool_cats=['road_signs_present','public_road', 'holiday', 'school_season']
    # df[bool_cats] = df[bool_cats] .astype(int)
    
    df['lanes_speed_lightning_feature']=(
        df['num_lanes'].astype(str) + '_' +
        df['speed_limit'].astype(str)  + '_' + 
        df['lighting'].astype(str)
    )
    
    df['lanes_speed_lightning_feature'] = df['lanes_speed_lightning_feature'].astype('category')
    df['speed_curvature_ratio'] = df['curvature']*df['speed_limit']**2
    for col in df.columns:
        if df[col].dtype == 'object':
             df[col] = df[col].astype('category')
    
    return df


X = preprocess_first_round(X)

X_test = preprocess_first_round(X_test)


def create_frequency_features(train_df, test_df, cols, num, cat):
    """
    Add frequency and binning features to the dataset.
    
    - For each column, create <col>_freq = how often each value appears in train data.
    - For numeric columns, split values into 5 and 10 quantile bins (groups) to show rank or range.
    """
    train, test = train_df.copy(), test_df.copy()

    for col in cols:
        # Frequency encoding: how common each value is
        freq = train[col].value_counts(normalize=True)
        train[f"{col}_freq"] = train[col].map(freq).astype('float')
        test[f"{col}_freq"] = test[col].map(freq).astype('float').fillna(train[f"{col}_freq"].mean())

        # Binning: group numeric values into quantiles
        if col in num:
            for q in [5, 10, 15]:
                try:
                    train[f"{col}_bin{q}"], bins = pd.qcut(train[col], q=q, labels=False, retbins=True, duplicates="drop")
                    test[f"{col}_bin{q}"] = pd.cut(test[col], bins=bins, labels=False, include_lowest=True)
                except Exception:
                    train[f"{col}_bin{q}"] = test[f"{col}_bin{q}"] = 0

    new_num = train.drop(columns=cat).columns.tolist()
    return train, test, new_num


# Identify feature
cols = X.columns.tolist()

# Categorical features
cat = [col for col in cols if X[col].dtype in ["object","category"]]

# Numerical features
num = [col for col in cols if X[col].dtype not in ["object","category","bool"]]

# Creating new features based on the frequency of numerical features
X, X_test, new_num = create_frequency_features(X, X_test, cols, num, cat)


'''To be honest i didn't figured out what is going on here - since this is same author as
binning/freq feature engenering, so i decided to leave it.
'''
# Mapping a column
map_col = "num_reported_accidents"
map_num_reported = {0:0, 1:0, 2:0, 3:2, 4:4, 5:3, 6:1, 7:0}
X[map_col] = X[map_col].map(map_num_reported)
X_test[map_col] = X_test[map_col].map(map_num_reported)


X.drop_duplicates(inplace=True)
y = y.iloc[X.index]

X[cat], X_test[cat] = X[cat].astype("category"), X_test[cat].astype("category")


best_rmse, best_round, cv_result =  get_scored_m2(X,y)

Scoresy_m2 = add_row(
    Scoresy_m2,
    name='before_feature_reduction_experement',
    best_round=best_round,
    best_rmse=best_rmse
)


Scoresy_m2


Score = get_scored(X,y)
Scoresy = save_score(Score, Scoresy, 'before_feature_reduction_experement')
Scoresy 


toxic=[]
neutral = []

tol = 1e-6 

for f in X.columns:
    X_reduced = X.copy()
    
    X_reduced.pop(f)
    X_reduced.drop(columns=toxic, errors='ignore', inplace=True)
    Score = get_scored(X_reduced,y)
    
    mean_score = np.sqrt(-1*Score.mean())
    best_mean = Scoresy['mean'].min() if not Scoresy.empty else 0
    
    if (best_mean - mean_score) > 1e-6:  
        toxic.append(f)
        print(f'{f} to toxic')
    elif abs(mean_score - best_mean) <= tol:
        toxic.append(f)
        neutral.append(f)
        print(f'{f} to neutral')
    Scoresy = save_score(Score, Scoresy, f'poped_{f}')
    


 X_reduced = X.copy()
 X_reduced.drop(columns=toxic, errors='ignore', inplace=True)
 X_test.drop(columns=toxic, errors='ignore', inplace=True)


for feature in toxic: 
    if feature not in neutral:
        print(feature)


best_rmse, best_round, cv_result =  get_scored_m2(X_reduced,y)

Scoresy_m2 = add_row(
    Scoresy_m2,
    name='after_cusotm_filtering',
    best_round=best_round,
    best_rmse=best_rmse
)


X_toxic = X.copy()
X_toxic.drop(columns=toxic, inplace=True)


Score = get_scored(X_toxic,y)
Scoresy = save_score(Score, Scoresy, 'after_cusotm_filtering_but_neutral included')



best_rmse, best_round, cv_result =  get_scored_m2(X_toxic,y)

Scoresy_m2 = add_row(
    Scoresy_m2,
    name='after_cusotm_filtering_but_neutral included',
    best_round=best_round,
    best_rmse=best_rmse
)



sns.relplot(y=Scoresy['mean'], x=Scoresy.index,kind="line").set_axis_labels('Reduction iteration', 'RMSE')



Scoresy_m2.style.background_gradient(subset=['best_rmse'], cmap='coolwarm')


Scoresy.style.background_gradient(subset=['mean'], cmap='coolwarm')


# pip install optuna-integration[xgboost]


# import optuna
# import xgboost as xgb
# # from optuna.integration import XGBoostPruningCallback


# dtrain = xgb.DMatrix(X_reduced, label=y, enable_categorical=True)


# def objective(trial):
#     params = {
#         'objective': 'reg:squarederror',
#         'eval_metric': 'rmse',
#         'tree_method': 'hist',
#         'device': 'cuda',
#         # 'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
#         'booster': 'gbtree',
#         # 'lambda': trial.suggest_float('lambda', 0.0, 10.0, step=0.01),
#         # 'alpha': trial.suggest_float('alpha', 0.0, 10.0, step=0.01),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.75, 0.85, step=0.01),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.01),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.03, step=0.001),
#         'max_depth': trial.suggest_int('max_depth', 9, 12),
#         'min_child_weight': trial.suggest_int('min_child_weight', 2, 4),
#         'gamma': trial.suggest_float('gamma', 0.01, 0.02, step=0.001),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0.3, 0.5, step=0.01),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 0.15, step=0.01),
#         'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.82, 0.9, step=0.01),
#         'colsample_bynode': trial.suggest_float('colsample_bynode', 0.85, 0.9, step=0.01),
#         'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.34, 0.38, step=0.01),
#         # 'max_bin': trial.suggest_categorical('max_bin', [256, 512])
#         # 'max_bin': 512,
#         'random_state': 42,
#         'max_delta_step': 1
#     }

#     # if params['booster'] == 'dart':
#     #     params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
#     #     params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
#     #     params['rate_drop'] = trial.suggest_float('rate_drop', 0.0, 0.3)
#     #     params['skip_drop'] = trial.suggest_float('skip_drop', 0.0, 0.3)

#     # cv adds "test-" prefix automatically
#     # pruning_callback = XGBoostPruningCallback(trial, "test-rmse")

#     cv_results = xgb.cv(
#         params,
#         dtrain,
#         num_boost_round=3500,
#         nfold=5,
#         metrics = "rmse",
#         stratified=False,
#         early_stopping_rounds=50,
#         verbose_eval=False,
#         # callbacks=[pruning_callback],
#     )

#     best_rmse = cv_results['test-rmse-mean'].min()
#     best_iter = int(cv_results['test-rmse-mean'].idxmin())
#     trial.set_user_attr('best_iteration', best_iter)

#     return best_rmse

# # Настройка Optuna
# sampler = optuna.samplers.TPESampler(seed=42)
# # sampler = optuna.samplers.CmaEsSampler(seed=42)
# # pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
# study = optuna.create_study(direction='minimize', sampler=sampler, study_name='xgb_cv_opt',
#                            storage="sqlite:///optuna_xgb_TPESampler3.db", load_if_exists=True)
# study.optimize(objective, n_trials=50, timeout=2000000, show_progress_bar=True,)

# # Сохраняем результаты
# study.trials_dataframe().to_csv("optuna_xgb_cv_trials.csv", index=False)
# print(f"Best params: {study.best_params}")
# print(f"Best RMSE: {study.best_value:.5f}")
# print(f"Best iteration: {study.best_trial.user_attrs['best_iteration']}")


# study.best_params)


# Final model and submission , parametrs from Optuna


parametrs = {
    'colsample_bytree': 0.83,
    'subsample': 0.85,
    'learning_rate': 0.016,
    'max_depth': 11,
    'min_child_weight': 3,
    'gamma': 0.012,
    'reg_lambda': 0.43,
    'reg_alpha': 0.11,
    'colsample_bylevel': 0.8799999999999999,
    'colsample_bynode': 0.85,
    'scale_pos_weight': 0.38,
    'n_estimators': 1234,
    'eval_metric': 'rmse',
    'random_state': 42,
    'tree_method': 'hist',
    "device":"cuda",
}


XGB_model = XGBRegressor(**parametrs, enable_categorical=True,)
XGB_model.fit(X_reduced,y)
pred = XGB_model.predict(X_test)


test_ids


submission = pd.DataFrame({
    'id': test_ids,
    'accident_risk': pred
})

# Save submission
submission.to_csv("submission.csv", index=False)




