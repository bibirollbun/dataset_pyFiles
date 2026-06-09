!pip install autogluon.tabular
!pip install "ray>=2.10.0,<2.45.0"


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import make_scorer, mean_squared_error
import lightgbm as lgb
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV, KFold
from catboost import CatBoostRegressor
import xgboost as xgb




import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
df_sub = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


test_ids = df_test['id']


print(f"df Train shape {df.shape}")
print(f"df Test shape {df_test.shape}")


df.columns


#Droping "id" column
df=df.drop("id", axis=1)
df_test=df_test.drop("id", axis=1)


df.head()


#Check Null Value
df.isna().sum().sum()


#Check Test Null Value 
df_test.isna().sum().sum()


#Check Duplicate Rows
df.duplicated().sum()


#Drop Duplicate Rows
df = df.drop_duplicates()
df.duplicated().sum()


# numerical features correlation
# plt.figure(figsize=(8, 6))
# correlation_matrix = df[num_cols + ['accident_risk']].corr()
# sns.heatmap(correlation_matrix, annot=True, fmt='.2f', 
#             linewidths=1, color="blue")
# plt.show()


df['weather_lighting'] = df['weather'].astype(str) + '_' + df['lighting'].astype(str)
df_test['weather_lighting'] = df_test['weather'].astype(str) + '_' + df_test['lighting'].astype(str)

df['speed_squared'] = df['speed_limit'] ** 2
df_test['speed_squared'] = df_test['speed_limit'] ** 2

df['curvature_squared'] = df['curvature'] ** 2
df_test['curvature_squared'] = df_test['curvature'] ** 2

df['meta_curvature'] = 0.3 * df['curvature']
df['meta_night'] = 0.2 * (df['lighting'] == 'night').astype(int)
df['meta_weather'] = 0.1 * (df['weather'] != 'clear').astype(int)
df['meta_speed'] = 0.2 * (df['speed_limit'] >= 60).astype(int)
df['meta_accidents'] = 0.1 * (df['num_reported_accidents'] > 2).astype(int)

df_test['meta_curvature'] = 0.3 * df_test['curvature']
df_test['meta_night'] = 0.2 * (df_test['lighting'] == 'night').astype(int)
df_test['meta_weather'] = 0.1 * (df_test['weather'] != 'clear').astype(int)
df_test['meta_speed'] = 0.2 * (df_test['speed_limit'] >= 60).astype(int)
df_test['meta_accidents'] = 0.1 * (df_test['num_reported_accidents'] > 2).astype(int)


df['log_curvature'] = np.log1p(df['curvature'])
df_test['log_curvature'] = np.log1p(df_test['curvature'])


def f(X):
    return \
    0.3 * X["curvature"] + \
    0.2 * (X["lighting"] == "night").astype(int) + \
    0.1 * (X["weather"] != "clear").astype(int) + \
    0.2 * (X["speed_limit"] >= 60).astype(int) + \
    0.1 * (X["num_reported_accidents"] > 2).astype(int)


    
df['meta'] = f(df)
df_test['meta'] = f(df_test)

#Very good FE


df.head()


df_test
df
# X= df.drop('accident_risk', axis =1)
# y= df['accident_risk']
# X_test = df_test

# X_train= X
# y_train= y

y = df['accident_risk']
X = df.drop(['accident_risk','id'], axis=1, errors='ignore')
X_test = df_test.drop('id', axis=1, errors='ignore')

# AutoGluon expects target in the same dataframe
train_ag = X.copy()
train_ag['accident_risk'] = y


from autogluon.tabular import TabularDataset, TabularPredictor

train_ag = TabularDataset(train_ag)
X_test = TabularDataset(X_test)
target = 'accident_risk'

hyperparameters = {
    "GBM": {},        # LightGBM
    "XGB": {},        # XGBoost
    "CAT": {},        # CatBoost
    # Use FASTAI (or NN_TORCH) for neural nets which can use GPU.
    # FASTAI is commonly available in autogluon; using ag_args_fit to request GPU.
    # "NN_TORCH": {"ag_args_fit": {"num_gpus": 1}},
    # "FASTAI": {"ag_args_fit": {"num_gpus": 1}},
    # "XT": {},         # ExtraTrees
    # "RF": {},         # RandomForest
}

predictor_main = TabularPredictor(label=target, eval_metric ='rmse', 
                            problem_type="regression").fit(train_ag, 
                                                           presets='best_quality',
                                                           # presets = 'extreme',
                                                           # auto_stack = True,
                                                           hyperparameters=hyperparameters,
                                                           time_limit=60*60*8.5,
                                                           verbosity=3,
                                                           # excluded_model_types=['KNN'],
                                                           ag_args_fit={'num_gpus': 2}
                                                      )



# # Lighgbm best param from random search cv
# param_lgb = {
    
#     'n_estimators': 2700,
#     'learning_rate': 0.01,
#     'num_leaves': 99,
#     'max_depth': 13,
#     'min_child_samples': 10,
#     'min_child_weight': 0.002,
#     'subsample': 0.60,
#     'subsample_freq': 1,
#     'colsample_bytree': 0.83,
#     'reg_alpha': 0.01,
#     'reg_lambda':  0.70,
#     'min_split_gain':  0.004,
#     'feature_fraction': 0.9 , 

 
# }

# # catboost best param from random search cv
# param_cat = {
#      'bagging_temperature' : 0.20,
#      'border_count'        : 178,
#      'depth'               : 8,
#      'iterations'          : 1600,
#      'l2_leaf_reg'         : 4,
#      'learning_rate'       : 0.04,
#      'random_strength'    : 0.32,
     
# }

# # xgboost best param from random search cv
# param_xgb = {
#               'n_estimators': 1251,
#               'learning_rate': 0.0074,
#               'max_depth': 9,
#               'min_child_weight': 3,
#               'subsample': 0.72,
#               'colsample_bytree': 0.74,
#               'colsample_bylevel': 0.94,
#               'gamma': 0.0002,
#               'reg_alpha': 0.61,
#               'reg_lambda': 4.92}


# print("\n" + "="*60)
# print("Simple Average (90-10)")
# print("="*60)

# cat_model =  CatBoostRegressor(**param_cat,
#                                loss_function='RMSE',
#                                random_seed=42,
#                                verbose=False,
#                                thread_count=-1,)

# lgb_model = lgb.LGBMRegressor(**param_lgb ,
#                                objective='regression',
#                                metric='rmse',
#                                boosting_type='gbdt',
#                                random_state=42,
#                                n_jobs=-1,
#                                verbose=-1    
#                             )  

# xgb_model = xgb.XGBRegressor(**param_xgb,
#                               random_state = 42,
#                               objective = 'reg:squarederror')


# #Lets see CV Score
# kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# cv_scores = []

# for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
#     X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
#     y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    
#     # Train cat
#     cat_model.fit(X_tr, y_tr)
#     cat_pred = cat_model.predict(X_val)
    
#     # Train LightGBM
#     lgb_model.fit(X_tr, y_tr)
#     lgb_pred = lgb_model.predict(X_val)

#     # Train XGBoost
#     xgb_model.fit(X_tr, y_tr)
#     xgb_pred = xgb_model.predict(X_val)
#     # Simple average
#     ensemble_pred = 0.3  * cat_pred + 0.3 * lgb_pred + 0.4 * xgb_pred
#     # Simple average
       
#     rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
#     cv_scores.append(rmse)
#     print(f"Fold {fold}: {rmse:.5f}")

# simple_avg_score = np.mean(cv_scores)
# print(f"\nSimple Average CV Score: {simple_avg_score:.5f} (+/- {np.std(cv_scores):.5f})")



# xgb_model.fit(X_train, y_train)
# xgb_pred = xgb_model.predict(X_test)

# lgb_model.fit(X_train, y_train)
# lgb_pred = lgb_model.predict(X_test)

# cat_model.fit(X_train,y_train,)
# cat_pred = cat_model.predict(X_test)

# ensemble_pred = 0.3  * cat_pred + 0.3 * lgb_pred + 0.4 * xgb_pred


# feature_importances = xgb_model.feature_importances_

# importance_df = pd.DataFrame({
#     'feature': X.columns, 
#     'importance': feature_importances
# })

# importance_df = importance_df.sort_values('importance', ascending=False)

# plt.style.use('fivethirtyeight')
# plt.figure(figsize=(10, 8))
# sns.barplot(x='importance', 
#             y='feature', 
#             data=importance_df.head(10)) 
# plt.title('Feature Importance (XGB)')
# plt.xlabel('Importance Score')
# plt.ylabel('Features')
# plt.tight_layout()
# plt.show()


# def create_meta_features(models, X_train, X_test, y_train, n_splits=5):
    
#     #Create out-of-fold predictions for training meta-model
    
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
#     # Arrays to store predictions
#     meta_train = np.zeros((len(X_train), len(models)))
#     meta_test = np.zeros((len(X_test), len(models)))
    
#     for i, model in enumerate(models):
#         print(f"Processing model {i+1}...")
#         test_preds = np.zeros(len(X_test))
        
#         for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
#             X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
#             y_tr = y_train.iloc[train_idx]
            
#             # Train model on fold
#             model.fit(X_tr, y_tr)
            
#             # Get out-of-fold predictions
#             meta_train[val_idx, i] = model.predict(X_val)
            
#             # Get test predictions for this fold
#             test_preds += model.predict(X_test) / n_splits
        
#         meta_test[:, i] = test_preds
    
#     return meta_train, meta_test

# # Tuned  models
# models = [cat_model,  lgb_model, xgb_model]

# # Create meta features
# meta_train, meta_test = create_meta_features(models, X_train, X_test, y_train)


# # Train meta-model
# from sklearn.linear_model import Ridge, LinearRegression
# meta_model = Ridge(alpha=0.1)  # Start with Ridge regression

# # meta_model = LinearRegression()  # Or try simple linear regression
# # meta_model = lgb.LGBMRegressor()  # Or simple LGBM

# meta_model.fit(meta_train, y_train)
# rmse = np.sqrt(mean_squared_error(y_train, meta_model.predict(meta_train)))
# print(f"rmse for {meta_model} : {rmse}")
# final_predictions = meta_model.predict(meta_test)


predictor_main.leaderboard()


y_pred = predictor_main.predict(X_test)


# Create a submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'accident_risk': y_pred
})

# Save the predictions to a CSV file
submission.to_csv('submission.csv', index=False)
# submission.to_csv('submissionV1.csv', index=False)

# Display the first few rows of the predictions
print(submission.head(10))

