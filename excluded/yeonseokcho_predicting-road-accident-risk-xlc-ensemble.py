import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
print(sample_submission.shape)
sample_submission.head()


test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
print(test.shape)
test.head()


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
print(train.shape)
train.head()


train.describe(include="all")


train.info()


# convert Boolean to integer 

binary_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
for col in binary_cols:
    train[col] = train[col].astype(int)

for col in binary_cols:
    test[col] = test[col].astype(int)

print(train.shape, test.shape)
train.head()


train["lighting"].value_counts()


# Replace categorical values with target means

categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
encoding_maps = {}

for col in categorical_cols:
    target_mean_map = train.groupby(col)['accident_risk'].mean() # target mean per category
    encoding_maps[col] = target_mean_map
    train[f'{col}_encoded'] = train[col].map(target_mean_map)
    train.drop(col, axis=1, inplace=True)

for col in categorical_cols:
    test[f'{col}_encoded'] = test[col].map(encoding_maps[col])
    test.drop(col, axis=1, inplace=True)
    
print(train.shape, test.shape)
train.head()


train["lighting_encoded"].value_counts()


# correlation with target
train_corr = train.drop('id', axis=1).corr()

sorted_features = train_corr['accident_risk'].abs().sort_values(ascending=False).index
sorted_corr_matrix = train_corr.loc[sorted_features, sorted_features]

plt.figure(figsize=(10, 8))
sns.set(font_scale=0.9)
hm = sns.heatmap(sorted_corr_matrix, 
                 annot=True,         
                 fmt='.3f',          
                 cmap='coolwarm',    
                 square=False, 
                 cbar=False)       
hm.xaxis.tick_top()
plt.xticks(rotation=30, ha='left')
plt.show()

print(train_corr['accident_risk'].sort_values(ascending=False))

# High accident Risk are from Road Curvature and Poor Lighting


# scatter ploting
def plot_scatter(ax, df, x_col, y_col, alpha=0.5, point_size=2, line_color='red'):
    ax.scatter(df[x_col], df[y_col], alpha=alpha, s=point_size)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
    ax.grid(True)

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
plot_scatter(axes[0], train, "curvature", "accident_risk")
plot_scatter(axes[1], train, "lighting_encoded", "accident_risk")
plt.tight_layout()
plt.show() 


# scatter ploting
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
plot_scatter(axes[0], train, "speed_limit", "accident_risk")
plot_scatter(axes[1], train, "num_reported_accidents", "accident_risk")
plt.tight_layout()
plt.show() 


train_en = train.copy()
train_en['weather_X_lighting'] = train_en['weather_encoded'] * train_en['lighting_encoded']


test_en = test.copy()
test_en['weather_X_lighting'] = test_en['weather_encoded'] * test_en['lighting_encoded']

print(test_en.shape, test_en.shape)
train_en.head()


# correlation with target
train_en_corr = train_en.drop('id', axis=1).corr()

sorted_features = train_en_corr['accident_risk'].abs().sort_values(ascending=False).index
sorted_corr_matrix = train_en_corr.loc[sorted_features, sorted_features]

plt.figure(figsize=(10, 8))
sns.set(font_scale=0.9)
hm = sns.heatmap(sorted_corr_matrix, 
                 annot=True,         
                 fmt='.3f',          
                 cmap='coolwarm',    
                 square=False, 
                 cbar=False)       
hm.xaxis.tick_top()
plt.xticks(rotation=30, ha='left')
plt.show()

print(train_en_corr['accident_risk'].sort_values(ascending=False))

# High accident Risk are from Road Curvature and Poor Lighting


# Cleaning data
features_to_drop = ['road_type_encoded', 
                    #'road_signs_present', 
                    #'school_season'
                   ]

train_en_cleaned = train_en.drop(columns=features_to_drop)
test_en_cleaned = test_en.drop(columns=features_to_drop)

print(train_en_cleaned.shape, test_en_cleaned.shape)
train_en_cleaned.head()


# selecting features set
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def lgbm_features_set(df, features_set):
    
    X = df.drop(['id', 'accident_risk'], axis=1)
    y = df['accident_risk']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lgbm = lgb.LGBMRegressor(random_state=42, n_estimators=1400, learning_rate=0.05, num_leaves=31)
    lgbm.fit(X_train, y_train,
             eval_set=[(X_val, y_val)],
             eval_metric='rmse',
             callbacks=[lgb.early_stopping(10, verbose=False)])
    
    preds = lgbm.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    
    print(f"LGBM with {features_set} - RMSE: {rmse:.6f}\n")
    return rmse

results = {}
results['Original Features'] = lgbm_features_set(train, "Original Features")
results['Generated Features'] = lgbm_features_set(train_en, "Enlareged Features")
results['Cleaned Features'] = lgbm_features_set(train_en_cleaned, "Cleaned Features")

#LGBM with Original Features - RMSE: 0.056261
#LGBM with Enlareged Features - RMSE: 0.056219
#LGBM with Cleaned Features - RMSE: 0.056190


# features and target data
X = train_en_cleaned.drop(['id', 'accident_risk'], axis=1)
y = train_en_cleaned['accident_risk']

# tuning and evaluation data set
X_tune, X_eval, y_tune, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data Set for Hyperparameter Tuning: {X_tune.shape, y_tune.shape}")
print(f"Data Set for Final Evaluation: {X_eval.shape, y_eval.shape}")


from sklearn.model_selection import RandomizedSearchCV
import optuna
from scipy.stats import randint, uniform

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective_lgb(trial):
    params_lgb = {
        'objective': 'regression_l1',
        'metric': 'rmse',
        'n_estimators': 1000,  # Fix n_estimators to a large value
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        #'min_child_samples': trial.suggest_int('min_child_samples', 20, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1,
    }
    
    X_train, X_val, y_train, y_val = train_test_split(X_tune, y_tune, test_size=0.2, random_state=42)

    model_lgb = lgb.LGBMRegressor(**params_lgb)
    model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse', 
                  callbacks=[lgb.early_stopping(10, verbose=False)])
    
    preds_lgb = model_lgb.predict(X_val)
    rmse = mean_squared_error(y_val, preds_lgb, squared=False)
        
    return rmse

# Hyperparameter Tuning using Optuna
study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(objective_lgb, n_trials=10, show_progress_bar=True)

best_params_lgb = study_lgb.best_trial.params
print("\nBest Parameters found:", best_params_lgb)
print(f"Best RMSE during tuning: {study_lgb.best_trial.value:.6f}")


# Training the LGB Model with Best Parameters
best_params_lgb['n_estimators'] = 1000 
best_params_lgb['verbose'] = -1 
best_params_lgb['seed'] = 42

model_lgb_tune = lgb.LGBMRegressor(**best_params_lgb)
model_lgb_tune.fit(X_tune, y_tune, 
                    eval_set=[(X_eval, y_eval)], eval_metric='rmse',
                    callbacks=[lgb.early_stopping(10, verbose=False)])

# Evaluating Model Performance
preds_lgb_tune = model_lgb_tune.predict(X_tune)
rmse_lgb_tune = mean_squared_error(y_tune, preds_lgb_tune, squared=False)

preds_lgb_eval = model_lgb_tune.predict(X_eval)
rmse_lgb_eval = mean_squared_error(y_eval, preds_lgb_eval, squared=False)

print(f"lgb RMSE with y_tune set: {rmse_lgb_tune:.6f}")
print(f"lgb RMSE with y_eval set: {rmse_lgb_eval:.6f}")

# lgb RMSE with y_tune set: 0.055457
# lgb RMSE with y_eval set: 0.056160


from sklearn.model_selection import RandomizedSearchCV
import optuna
from scipy.stats import randint, uniform
import xgboost as xgb

def objective_xgb(trial):
    params_xgb = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 1300, 
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        #'num_leaves': trial.suggest_int('num_leaves', 20, 80),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        #'min_child_samples': trial.suggest_int('min_child_samples', 20, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),     
        'random_state': 42,
        'n_jobs': -1
    }
    
    X_train, X_val, y_train, y_val = train_test_split(X_tune, y_tune, test_size=0.2, random_state=42)

    model_xgb = xgb.XGBRegressor(**params_xgb)
    model_xgb.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
    
    preds_xgb = model_xgb.predict(X_val)
    rmse = mean_squared_error(y_val, preds_xgb, squared=False)
        
    return rmse

# Hyperparameter Tuning using Optuna
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=30, show_progress_bar=True)

best_params_xgb = study_xgb.best_trial.params
print("\nBest Parameters found:", best_params_xgb)
print(f"Best RMSE during tuning: {study_xgb.best_trial.value:.6f}")


# Training the XGBoost model with Best Parameters
best_params_xgb.update({'n_estimators': 1300, 'random_state': 42, 'n_jobs': -1})

model_xgb_tune = xgb.XGBRegressor(**best_params_xgb) #tuned model
model_xgb_tune.fit(X_tune, y_tune, eval_set=[(X_eval, y_eval)], early_stopping_rounds=10, verbose=False)

# Evaluating Model Performance
preds_xgb_tune = model_xgb_tune.predict(X_tune)
rmse_xgb_tune = mean_squared_error(y_tune, preds_xgb_tune, squared=False)

preds_xgb_eval = model_xgb_tune.predict(X_eval)
rmse_xgb_eval = mean_squared_error(y_eval, preds_xgb_eval, squared=False)

print(f"xgb RMSE with y_tune set: {rmse_xgb_tune:.6f}")
print(f"xgb RMSE with y_eval set: {rmse_xgb_eval:.6f}")

#xgb RMSE with y_tune set: 0.055281
#xgb RMSE with y_eval set: 0.056142


import catboost as cat

def objective_cat(trial):
    params_cat = {
        'objective': 'RMSE',
        'iterations': 1500,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 3, 12),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'random_seed': 42,
        'verbose': 0,
        'thread_count': -1,
        'bootstrap_type': 'Bayesian',
    }
    
    X_train, X_val, y_train, y_val = train_test_split(X_tune, y_tune, test_size=0.2, random_state=42)

    model_cat = cat.CatBoostRegressor(**params_cat)
    model_cat.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
    
    preds_cat = model_cat.predict(X_val)
    rmse = mean_squared_error(y_val, preds_cat, squared=False)
    
    return rmse

# Hyperparameter Tuning using Optuna
study_cat = optuna.create_study(direction='minimize')
study_cat.optimize(objective_cat, n_trials=30, show_progress_bar=True)

best_params_cat = study_cat.best_trial.params
print("\nBest CatBoost Parameters found:", best_params_cat)
print(f"Best CV RMSE during tuning: {study_cat.best_trial.value:.6f}")


# Training the CatBoost model with Best Parameters

best_params_cat.update({
    'objective': 'RMSE',
    'iterations': 1500,
    'random_seed': 42,
    'verbose': 0,
    'thread_count': -1,
    'bootstrap_type': 'Bayesian',
})

model_cat_tune = cat.CatBoostRegressor(**best_params_cat)
model_cat_tune.fit(X_tune, y_tune, eval_set=[(X_eval, y_eval)], early_stopping_rounds=10, verbose=0)

# Evaluating Model Performance
preds_cat_tune = model_cat_tune.predict(X_tune)
rmse_cat_tune = mean_squared_error(y_tune, preds_cat_tune, squared=False)

preds_cat_eval = model_cat_tune.predict(X_eval)
rmse_car_eval = mean_squared_error(y_eval, preds_cat_eval, squared=False)

print(f"cat RMSE with y_tune set: {rmse_cat_tune:.6f}")
print(f"cat RMSE with y_eval set: {rmse_car_eval:.6f}")

# cat RMSE with y_tune set: 0.055730
# cat RMSE with y_eval set: 0.056294


rmse_best_weighted = float('inf')
weights_best = {}

# LGBM weight from 0 to 1.0 with 0.01 increments
for weight_lgb in np.arange(0, 1.0, 0.002):
    for weight_xgb in np.arange(0, 1.0 - weight_lgb, 0.002):
        weight_cat = 1.0 - weight_lgb - weight_xgb
        
        if weight_cat < 0:
            continue

        preds_ensemble_weighted = (weight_lgb * preds_lgb_eval +
                                   weight_xgb * preds_xgb_eval +
                                   weight_cat * preds_cat_eval)
        
        rmse_current = mean_squared_error(y_eval, preds_ensemble_weighted, squared=False)
        
        if rmse_current < rmse_best_weighted:
            rmse_best_weighted = rmse_current
            weights_best = {
                'LGBM': round(weight_lgb, 6),
                'XGBoost': round(weight_xgb, 6),
                'CatBoost': round(weight_cat, 6)
            }

for model_name, weight in weights_best.items():
    print(f" - {model_name}: {weight:.4f}")

# performance
preds_ensemble_weighted = (weights_best['LGBM'] * preds_lgb_eval + 
                           weights_best['XGBoost'] * preds_xgb_eval +
                           weights_best['CatBoost'] * preds_cat_eval)

rmse_ensemble_weighted_eval = mean_squared_error(y_eval, preds_ensemble_weighted, squared=False)
print(f"\nVoting ensemble RMSE with y_eval: {rmse_ensemble_weighted_eval:.8f}")

"""
 - LGBM: 0.3520
 - XGBoost: 0.6460
 - CatBoost: 0.0020

Voting ensemble RMSE with y_eval: 0.05613396
"""


from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

models_base = [
    ('lgbm', model_lgb_tune),
    ('xgb', model_xgb_tune),
    ('catboost', model_cat_tune)
]

# Meta Model
model_meta = LinearRegression()

# Stacking Regressor
model_stacking = StackingRegressor(
    estimators=models_base,
    final_estimator=model_meta,
    cv=5, 
    n_jobs=-1
)
# Stacking Model Training
model_stacking.fit(X_tune, y_tune)

# prediction & evaluation
preds_stacking_tune = model_stacking.predict(X_tune) 
rmse_stacking_tune = mean_squared_error(y_tune, preds_stacking_tune, squared=False)

preds_stacking_eval = model_stacking.predict(X_eval) 
rmse_stacking_eval = mean_squared_error(y_eval, preds_stacking_eval, squared=False)

print(f"Stacking ensemble RMSE with y_tune: {rmse_stacking_tune:.6f}")
print(f"Stacking ensemble RMSE with y_eval: {rmse_stacking_eval:.6f}")

# Stacking ensemble RMSE with y_tune: 0.055124
# Stacking ensemble RMSE with y_eval: 0.056161


test_feature = test_en_cleaned.drop(['id'], axis=1)
test_feature.head()


preds_lgb_test = model_lgb_tune.predict(test_feature)
preds_xgb_test = model_xgb_tune.predict(test_feature)
preds_cat_test = model_cat_tune.predict(test_feature)

test_pred = (weights_best['LGBM'] * preds_lgb_test + 
             weights_best['XGBoost'] * preds_xgb_test + 
             weights_best['CatBoost'] * preds_cat_test)

test_pred


submission = pd.DataFrame({'id': test.id, 'accident_risk': test_pred})
print(submission.shape)
submission.head()


submission.to_csv('submission.csv', index=False)


submission = pd.read_csv('/kaggle/working/submission.csv')
submission.head()

