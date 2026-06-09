import pandas as pd
import numpy as np
train=pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


CAT=['road_type', 'lighting', 'weather', 'time_of_day']
NUM=['id', 'num_lanes', 'curvature', 'speed_limit', 'road_signs_present', 'public_road', 'holiday', 'school_season', 'num_reported_accidents', 'accident_risk']
Bool=['road_signs_present', 'public_road', 'holiday', 'school_season']

def preprocessing(df_org):
    df=df_org.copy()
    
    df['base_risk'] = (
        0.3 * df["curvature"] + 
        0.2 * (df["lighting"] == "night").astype(int) + 
        0.1 * (df["weather"] != "clear").astype(int) + 
        0.2 * (df["speed_limit"] >= 60).astype(int) + 
        0.1 * (np.array(df["num_reported_accidents"]) > 2).astype(int)
    )
    df['is_night'] = (df["lighting"] == "night").astype(int)
    df['is_bad_weather'] = (df["weather"] != "clear").astype(int)
    df['high_speed_limit'] = (df["speed_limit"] >= 60).astype(int)
    df['many_previous_accidents'] = (df["num_reported_accidents"] > 2).astype(int)
    df['night_bad_weather'] = df['is_night'] * df['is_bad_weather']
    df['high_speed_curvy'] = df['high_speed_limit'] * df['curvature']
    df['curvy_accident_risk'] = df['curvature'] * (df['num_reported_accidents'] + 1)
    df['high_risk'] = ((df['num_reported_accidents'] > 2) & (df['curvature'] > 0.05) & (df['speed_limit'] >= 60)).astype(int)
    for i in CAT:
        df=pd.concat([df,pd.get_dummies(df[i],prefix=i).astype(int)],axis=1)
    df=df.drop(columns=CAT)
    df[Bool]=df[Bool].astype(int)
    return df

def adv_feature_extraction(df):
    df['combined_risk'] = (
    0.5 * df['lighting_night'] +           # Night is high risk
    0.2 * (df['speed_limit'] / df['speed_limit'].max()) +  # Normalize speed
    0.1 * df['curvature'] +               # Curvy roads
    0.05 * df['lighting_dim'] +           # Dim lighting
    0.05 * (1 - df['weather_clear']) +    # Bad weather adds risk
    0.05 * df['weather_foggy'] +
    0.05 * df['weather_rainy']
    )
    df['night_bad_weather'] = df['lighting_night'] * (df['weather_foggy'] + df['weather_rainy'])
    df['daylight_bad_weather'] = df['lighting_daylight'] * (df['weather_foggy'] + df['weather_rainy'])
    df['dim_bad_weather'] = df['lighting_dim'] * (df['weather_foggy'] + df['weather_rainy'])
    df['curvy_high_speed'] = df['curvature'] * df['speed_limit']
    df['night_high_speed'] = df['lighting_night'] * df['speed_limit']
    df['high_risk_night_curve'] = ((df['lighting_night'] == 1) & (df['curvature'] > 0.05)).astype(int)
    df['high_risk_speed_weather'] = ((df['speed_limit'] >= 60) & (df['weather_foggy'] | df['weather_rainy'])).astype(int)
    df['accident_severity_score'] = (df['many_previous_accidents'] * 0.5 + df['high_risk'] * 0.3 + df['num_reported_accidents'] * 0.2)
    df['bad_weather_curvy'] = ((df['is_bad_weather'] == 1) & (df['curvature'] > 0.5)).astype(int)
    df['speed_risk_score'] = df['high_speed_curvy'] * (df['high_risk'] + df['is_bad_weather'])
    df['curvature_risk'] = df['curvature'] * (df['high_risk'] + df['many_previous_accidents'])
    df['overall_risk_factor'] = (df['high_risk'] +df['many_previous_accidents'] +df['is_bad_weather'] +df['high_speed_curvy'])
    df['exp_risk'] = np.exp(df['curvature'] * df['high_speed_curvy']) * (1 + df['is_bad_weather'])
    return df


train_cl=adv_feature_extraction(preprocessing(train))
test_cl=adv_feature_extraction(preprocessing(test))

# def outlier_rev(df,col):
#     q_i=df[col].quantile(0.25)
#     q_h=df[col].quantile(0.75)
#     IQR=q_h-q_i
#     lb=q_i-IQR*0.5
#     ub=q_h+ IQR*0.5
#     return df[(df[col]>lb)&(df[col]<ub)]
# # for col in train_cl.columns:
# #     train_cl=outlier_rev(train_cl,col)
# train_cl=outlier_rev(train_cl,'accident_risk')


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
features_to_drop = [
    'weather_clear', 'lighting_night', 'high_risk_speed_weather', 'school_season', 'time_of_day_afternoon',
    'road_type_rural', 'road_type_urban', 'road_type_highway', 'dim_bad_weather', 'time_of_day_morning',
    'lighting_dim', 'road_signs_present', 'num_lanes', 'time_of_day_evening', 'combined_risk',
    'lighting_daylight', 'night_bad_weather', 'curvy_accident_risk', 'many_previous_accidents'
]

x=train_cl.drop(columns=['accident_risk','id'])
y=train_cl['accident_risk']
x=x.drop(columns=features_to_drop)
X_train, X_test, y_train, y_test = train_test_split(x,y,random_state=46, test_size=0.2)

# X_train = X_train.drop(columns=features_to_drop)
# X_test = X_test.drop(columns=features_to_drop)

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


from xgboost import XGBRegressor
import optuna
from sklearn.model_selection import cross_val_score

def fintuna(trial):
    param={
        'n_estimators':trial.suggest_int('n_estimators',200,1800),
        'learning_rate':trial.suggest_float('n_estimators',0.0001,0.9),
        'max_depth':trial.suggest_int('max_depth',1,19),
        'subsample':trial.suggest_float('subsample',0.1,1.0),
        'colsample_bytree':trial.suggest_float('colsample_bytree',0.1,1.0),
        
    }


from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

param={'bootstrap_type': 'Bayesian',
     'iterations': 1057,
     'learning_rate': 0.07403272632797935,
     'depth': 6,
     'l2_leaf_reg': 0.16785640341856986,
     'bagging_temperature': 0.14672200750237943,
      'verbose':False}
model=CatBoostRegressor(**param)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("RMSE:- ",np.sqrt(mean_squared_error(y_pred,y_test)))


import pandas as pd
from xgboost import plot_importance

# Get importance scores by gain
importance = model.get_booster().get_score(importance_type='gain')

# Convert to DataFrame for easier filtering
imp_df = pd.DataFrame({
    'feature': list(importance.keys()),
    'gain': list(importance.values())
}).sort_values(by='gain', ascending=False)
imp_df


from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
param_new={'n_estimators': 398, 'learning_rate': 0.06920596850598196, 'max_depth': 7, 'subsample': 0.6939388671428774, 'colsample_bytree': 0.9826669385263327, 'gamma': 0.005620245697525984, 'reg_lambda': 5.815804183386186, 'reg_alpha': 0.660126738370973, 'min_child_weight': 6}
# param={'n_estimators': 306, 'learning_rate': 0.07696492778557112, 'max_depth': 9, 'subsample': 0.6335514865449435, 'colsample_bytree': 0.8748970972775195, 'gamma': 0.0066247611993441235, 'reg_lambda': 6.2538830040679505, 'reg_alpha': 0.7394906082330828, 'min_child_weight': 7}
model=XGBRegressor(**param_new)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("RMSE:- ",np.sqrt(mean_squared_error(y_pred,y_test)))
import matplotlib.pyplot as plt
import xgboost as xgb

xgb.plot_importance(model)
plt.show()
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)
feature_importances

xgb.plot_importance(model, importance_type='weight')   # default is 'weight' (# of times a feature is used)
xgb.plot_importance(model, importance_type='gain')     # contribution to loss reduction
xgb.plot_importance(model, importance_type='cover')    # average coverage of splits





import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import optuna
import numpy as np

# === Scale the data ===
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
{'n_estimators': 1017,
 'learning_rate': 0.01677740339661301,
 'max_depth': 13,
 'reg_lambda': 0.3962215918345482,
 'num_leaves': 54,
 'feature_fraction': 0.6211200736936922, 'bagging_fraction': 0.8477460951414041,
 'bagging_freq': 6}
# === Define objective function ===
def finetuna(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 900, 1200),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.19, log=True),
        'max_depth': trial.suggest_int('max_depth', 9, 16),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.2, 0.5),
        'num_leaves': trial.suggest_int('num_leaves', 40, 70),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.7),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 5, 9),
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'random_state': 42,
        'verbosity': -1,
        'device': 'gpu',  # GPU enabled
        'gpu_platform_id': 0,
        'gpu_device_id': 0
    }

    try:
        model = lgb.LGBMRegressor(**params)
        kf = KFold(n_splits=3, shuffle=True, random_state=45)
        rmse = -cross_val_score(
            model,
            X_train_scaled,
            y_train_scaled,
            cv=kf,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        ).mean()
        return rmse
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')

# === Run Optuna study ===
study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=789)
)
study.optimize(finetuna, n_trials=50, show_progress_bar=True)

print("Best trial:")
print(study.best_trial.params)


from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# === Scale the data ===
scaler_X = StandardScaler()
scaler_y = StandardScaler()

param={'n_estimators': 1017, 'learning_rate': 0.01677740339661301, 'max_depth': 13, 'reg_lambda': 0.3962215918345482, 'num_leaves': 54, 'feature_fraction': 0.6211200736936922, 'bagging_fraction': 0.8477460951414041, 'bagging_freq': 6}
best_model= lgb.LGBMRegressor(**param)

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.fit_transform(y_test.values.reshape(-1, 1)).ravel()

best_model.fit(X_train_scaled,y_train_scaled)
X_test_scaled = scaler_X.fit_transform(X_test)
# Suppose y_pred_scaled are predictions from your best model
y_pred_scaled = best_model.predict(X_test_scaled)

# Convert predictions and true y back to original scale
y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_true_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))

rmse_original = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
print("RMSE (original scale):", rmse_original)


import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# === Scale the data ===
scaler_X = StandardScaler()
scaler_y = StandardScaler()

param={'n_estimators': 1017, 'learning_rate': 0.01677740339661301, 'max_depth': 13, 'reg_lambda': 0.3962215918345482, 'num_leaves': 54, 'feature_fraction': 0.6211200736936922, 'bagging_fraction': 0.8477460951414041, 'bagging_freq': 6}
best_model= lgb.LGBMRegressor(**param)

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
best_model.fit(scaler_X.transform(x),scaler_y.transform(y.values.reshape(-1, 1)).ravel())
y_pred_sub=best_model.predict(scaler_X.transform(test_cl.drop(columns=['id']).values))
y_pred_sub=pd.DataFrame(scaler_y.inverse_transform(y_pred_sub.reshape(-1, 1)),columns=['accident_risk'])



submission=pd.concat([pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')[['id']],y_pred_sub],axis=1)
submission.to_csv('submission.csv',index=False)


from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, cross_val_score
import optuna
from sklearn.metrics import make_scorer

def finetuna(trial):
    bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli'])
    {'bootstrap_type': 'Bayesian',
     'iterations': 949,
     'learning_rate': 0.08346754433541781,
     'depth': 5,
     'l2_leaf_reg': 0.24890733871385226,
     'bagging_temperature': 0.22177507454811832}
    params = {
        'iterations': trial.suggest_int('iterations', 800, 1200),  # Further reduced
        'learning_rate': trial.suggest_float('learning_rate', 0.075, 0.089, log=True),
        'depth': trial.suggest_int('depth', 3, 6),  # Shallower trees
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.17, 0.3),
        'bootstrap_type': 'Bayesian',
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        # 'task_type': 'GPU',
        'devices': '0',
        'random_state': 42,
        'verbose': False}

    if bootstrap_type == 'Bayesian':
        params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.1, 0.3)

    try:
        model = CatBoostRegressor(**params)
        kf = KFold(n_splits=2, shuffle=True, random_state=45)
        rmse = -cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=1).mean()
        return rmse
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')  # Return large value for failed trials

study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=512),pruner=optuna.pruners.MedianPruner())
study.optimize(finetuna,n_trials=250,show_progress_bar=True)


from xgboost import XGBRegressor
import optuna
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer
{'n_estimators': 324,
 'learning_rate': 0.07152613653996355,
 'max_depth': 8,
 'subsample': 0.5724085622892651,
 'colsample_bytree': 0.8391428510308703,
 'gamma': 0.0068855636141996685,
 'reg_lambda': 6.384858357170796,
 'reg_alpha': 0.7735828877784243,
 'min_child_weight': 7}
def fintuna(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 250, 400),
        'learning_rate': trial.suggest_float('learning_rate', 0.067, 0.087, log=True),
        'max_depth': trial.suggest_int('max_depth', 7, 10),
        'subsample': trial.suggest_float('subsample', 0.55, 0.75),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 0.99),
        'gamma': trial.suggest_float('gamma', 0.0055, 0.0075),
        'reg_lambda': trial.suggest_float('reg_lambda', 5.5, 7.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.66, 0.84, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 6, 8),
        'random_state': 42,
        'objective': 'reg:squarederror',
        'device':'cuda',
        'objective':'reg:pseudohubererror',
    }
    model=XGBRegressor(**param)
    kfold=KFold(n_splits=7,shuffle=True,random_state=46)
    rmse=-cross_val_score(model,X_train_scaled,y_train,cv=kfold,scoring='neg_root_mean_squared_error',n_jobs=-1).mean()
    return rmse
    
study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=789),pruner=optuna.pruners.MedianPruner())
study.optimize(fintuna,n_trials=250,show_progress_bar=True)


study.best_params


from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
param_new={'n_estimators': 398, 'learning_rate': 0.06920596850598196, 'max_depth': 7, 'subsample': 0.6939388671428774, 'colsample_bytree': 0.9826669385263327, 'gamma': 0.005620245697525984, 'reg_lambda': 5.815804183386186, 'reg_alpha': 0.660126738370973, 'min_child_weight': 6}
# param={'n_estimators': 306, 'learning_rate': 0.07696492778557112, 'max_depth': 9, 'subsample': 0.6335514865449435, 'colsample_bytree': 0.8748970972775195, 'gamma': 0.0066247611993441235, 'reg_lambda': 6.2538830040679505, 'reg_alpha': 0.7394906082330828, 'min_child_weight': 7}
param_latest={'n_estimators': 387,
 'learning_rate': 0.07380890285615214,
 'max_depth': 9,
 'subsample': 0.7497436303176783,
 'colsample_bytree': 0.8105739489400122,
 'gamma': 0.005583714194699566,
 'reg_lambda': 5.870200511945209,
 'reg_alpha': 0.6798788255117862,
 'min_child_weight': 6}
model=XGBRegressor(**param_latest)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("RMSE:- ",np.sqrt(mean_squared_error(y_pred,y_test)))
import matplotlib.pyplot as plt
import xgboost as xgb

xgb.plot_importance(model)
plt.show()
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)
feature_importances


xgb.plot_importance(model, importance_type='weight')   # default is 'weight' (# of times a feature is used)
xgb.plot_importance(model, importance_type='gain')     # contribution to loss reduction
xgb.plot_importance(model, importance_type='cover')    # average coverage of splits


# param={'n_estimators': 324,
#  'learning_rate': 0.07152613653996355,
#  'max_depth': 8,
#  'subsample': 0.5724085622892651,
#  'colsample_bytree': 0.8391428510308703,
#  'gamma': 0.0068855636141996685,
#  'reg_lambda': 6.384858357170796,
#  'reg_alpha': 0.7735828877784243,
#  'min_child_weight': 7}
# param={'n_estimators': 398, 'learning_rate': 0.06920596850598196, 'max_depth': 7, 'subsample': 0.6939388671428774, 'colsample_bytree': 0.9826669385263327, 'gamma': 0.005620245697525984, 'reg_lambda': 5.815804183386186, 'reg_alpha': 0.660126738370973, 'min_child_weight': 6}
param={'n_estimators': 387,
 'learning_rate': 0.07380890285615214,
 'max_depth': 9,
 'subsample': 0.7497436303176783,
 'colsample_bytree': 0.8105739489400122,
 'gamma': 0.005583714194699566,
 'reg_lambda': 5.870200511945209,
 'reg_alpha': 0.6798788255117862,
 'min_child_weight': 6}
model=model=XGBRegressor(**param)
model.fit(sc.transform(x),y)
y_pred_sub=model.predict(sc.transform(test_cl.drop(columns=['id']).drop(columns=features_to_drop).values))
y_pred_sub=pd.DataFrame(y_pred_sub,columns=['accident_risk'])
y_pred_sub


submission=pd.concat([pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')[['id']],y_pred_sub],axis=1)
submission.to_csv('submission.csv',index=False)


train_cl['accident_risk'].skew()


sns.histplot(train_cl['accident_risk'], kde=True, bins=30, color='teal')


train_cl.skew()


import seaborn as sns
import numpy as np
sns.boxplot(np.log1p(train_cl['accident_risk']))


study.best_params


!pip install optuna


import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import optuna
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Your existing parameter
param_first = {
    'n_estimators': 324,
    'learning_rate': 0.07152613653996355,
    'max_depth': 8,
    'subsample': 0.5724085622892651,
    'colsample_bytree': 0.8391428510308703,
    'gamma': 0.0068855636141996685,
    'reg_lambda': 6.384858357170796,
    'reg_alpha': 0.7735828877784243,
    'min_child_weight': 7
}

# Train first model
model1 = XGBRegressor(**param_first, random_state=42, device='cuda', objective='reg:pseudohubererror')
model1.fit(X_train_scaled, y_train)

# Calculate residuals
y_train_pred = model1.predict(X_train_scaled)
residuals = y_train - y_train_pred

{'n_estimators': 299, 'learning_rate': 0.07370371018835488, 'max_depth': 10, 'subsample': 0.7498421543488902, 'colsample_bytree': 0.811597945906766, 'gamma': 0.0074781365035608395, 'reg_lambda': 6.0822205275993975, 'reg_alpha': 0.8396097278475435, 'min_child_weight': 8}


# Objective function for residual model optimization
def objective_residual(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1400),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.001, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.001, 0.99),
        'gamma': trial.suggest_float('gamma', 0.0001, 0.1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 0.99, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 9),
        'random_state': 42,
        'device': 'cuda',
        'objective': 'reg:pseudohubererror',
    }
    
    model = XGBRegressor(**param)
    kfold = KFold(n_splits=7, shuffle=True, random_state=46)
    rmse = -cross_val_score(model, X_train_scaled, residuals, cv=kfold, 
                           scoring='neg_root_mean_squared_error', n_jobs=-1).mean()
    return rmse

# Optimize residual model
study_residual = optuna.create_study(direction='minimize', 
                                   sampler=optuna.samplers.TPESampler(seed=789),
                                   pruner=optuna.pruners.MedianPruner())
study_residual.optimize(objective_residual, n_trials=25, show_progress_bar=True)

print("Best parameters for residual model:")
print(study_residual.best_params)
print(f"Best RMSE for residual model: {study_residual.best_value:.4f}")

# Train final models with optimized parameters
best_params_residual = study_residual.best_params

# First model (using your existing parameters)
model1_final = XGBRegressor(**param_first, random_state=42, device='cuda', objective='reg:pseudohubererror')
model1_final.fit(X_train_scaled, y_train)

# Residual model (using optimized parameters)
model2_final = XGBRegressor(**best_params_residual)
model2_final.fit(X_train_scaled, residuals)

# Make predictions
y_pred1 = model1_final.predict(X_test_scaled)
y_pred2 = model2_final.predict(X_test_scaled)
y_pred_combined = y_pred1 + y_pred2

# Calculate final RMSE
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_combined))
print(f"\nFinal Combined RMSE: {final_rmse:.4f}")

# Compare with single model performance
single_model_rmse = np.sqrt(mean_squared_error(y_test, y_pred1))
print(f"Single Model RMSE: {single_model_rmse:.4f}")
print(f"Improvement: {single_model_rmse - final_rmse:.4f}")

# Feature importance for both models
feature_importances_1 = pd.DataFrame({
    'feature': X_train_scaled.columns if hasattr(X_train_scaled, 'columns') else [f'feature_{i}' for i in range(X_train_scaled.shape[1])],
    'importance_model1': model1_final.feature_importances_
}).sort_values(by='importance_model1', ascending=False)

feature_importances_2 = pd.DataFrame({
    'feature': X_train_scaled.columns if hasattr(X_train_scaled, 'columns') else [f'feature_{i}' for i in range(X_train_scaled.shape[1])],
    'importance_model2': model2_final.feature_importances_
}).sort_values(by='importance_model2', ascending=False)

print("\nTop 10 features for Model 1 (original):")
print(feature_importances_1.head(10))

print("\nTop 10 features for Model 2 (residuals):")
print(feature_importances_2.head(10))


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Initial model (trained on original targets)
param_latest = {
    'n_estimators': 387,
    'learning_rate': 0.07380890285615214,
    'max_depth': 9,
    'subsample': 0.7497436303176783,
    'colsample_bytree': 0.8105739489400122,
    'gamma': 0.005583714194699566,
    'reg_lambda': 5.870200511945209,
    'reg_alpha': 0.6798788255117862,
    'min_child_weight': 6
}

param_residual={'n_estimators': 721, 'learning_rate': 0.0010564619757733682, 'max_depth': 1, 'subsample': 0.9946251282399367, 'colsample_bytree': 0.6172515214703365, 'gamma': 0.029105568235227888, 'reg_lambda': 0.5008424253263292, 'reg_alpha': 0.001550864230283955, 'min_child_weight': 7}

# First model
model1 = XGBRegressor(**param_latest)
model1.fit(X_train, y_train)

# Calculate residuals on training data
y_train_pred = model1.predict(X_train)
residuals = y_train - y_train_pred

# Second model (trained on residuals)
model2 = XGBRegressor(**param_latest)  # Use same/different parameters
model2.fit(X_train, residuals)

# Predictions: First model's prediction + Second model's residual prediction
y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred_combined = y_pred1 + y_pred2

# Evaluate
rmse_combined = np.sqrt(mean_squared_error(y_test, y_pred_combined))
print(f"Combined RMSE: {rmse_combined}")

# Feature importance for the residual model
feature_importances_residuals = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model2.feature_importances_
}).sort_values(by='importance', ascending=False)

print("Residual Model Feature Importances:")
print(feature_importances_residuals)


import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Parameters for the first model (original target)
param_first = {
    'n_estimators': 324,
    'learning_rate': 0.07152613653996355,
    'max_depth': 8,
    'subsample': 0.5724085622892651,
    'colsample_bytree': 0.8391428510308703,
    'gamma': 0.0068855636141996685,
    'reg_lambda': 6.384858357170796,
    'reg_alpha': 0.7735828877784243,
    'min_child_weight': 7
}

# Parameters for the second model (residuals) - you can modify these
param_residual = {'n_estimators': 721, 'learning_rate': 0.0010564619757733682, 'max_depth': 1, 'subsample': 0.9946251282399367, 'colsample_bytree': 0.6172515214703365, 'gamma': 0.029105568235227888, 'reg_lambda': 0.5008424253263292, 'reg_alpha': 0.001550864230283955, 'min_child_weight': 7}

# Assuming you have your data prepared
# X_train, y_train for training
# test_data for final prediction

# Initialize and fit the scaler (if you're using one)
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)

# Train first model on original target
model1 = XGBRegressor(**param_first, random_state=42, objective='reg:pseudohubererror')
model1.fit(X_train_scaled, y_train)

# Calculate residuals from first model
y_train_pred = model1.predict(X_train_scaled)
residuals = y_train - y_train_pred

# Train second model on residuals
model2 = XGBRegressor(**param_residual, random_state=42, objective='reg:pseudohubererror')
model2.fit(X_train_scaled, residuals)

# Prepare test data
test_processed = test_cl.drop(columns=['id']).drop(columns=features_to_drop).values
test_scaled = sc.transform(test_processed)

# Make predictions
y_pred1 = model1.predict(test_scaled)
y_pred2 = model2.predict(test_scaled)
y_pred_combined = y_pred1 + y_pred2

# Create submission file
y_pred_sub = pd.DataFrame(y_pred_combined, columns=['accident_risk'])
submission = pd.concat([pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')[['id']], y_pred_sub], axis=1)
submission.to_csv('submission-over-residual-org.csv', index=False)

print("Submission file 'submission-over-residual-org.csv' created successfully!")
print(f"Predictions range: {y_pred_combined.min():.4f} to {y_pred_combined.max():.4f}")

