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


num_cols =  df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(exclude="number").columns.tolist()
num_cols.remove("accident_risk")

print(f"categorical columns : {cat_cols}")
print(f"numerical columns : {num_cols}")



df[num_cols].describe().T


# numerical features correlation
plt.figure(figsize=(8, 6))
correlation_matrix = df[num_cols + ['accident_risk']].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', 
            linewidths=1, color="blue")
plt.show()


#Categorical Columns Unique
for c in cat_cols :
   print(f" {c} (Uniques): {df[c].unique()}")


# Categorical features vs target

fig , axes = plt.subplots(2,4, figsize=(16,8))
axes = axes.flatten()
cmap = plt.get_cmap('magma')
colors = cmap([0.9,0.66,0.33])
target = 'accident_risk'
for i,col in enumerate(cat_cols) :
  
    grouped = df.groupby(col)[target].mean()
   
    axes[i].bar(grouped.index.astype(str), grouped.values , color=colors)  # .astype(str) to handle non-string indices
    
    axes[i].set_ylabel(f'Mean {target}')
    axes[i].set_title(f'{col} vs {target}')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(df['accident_risk'], bins=50, edgecolor='black', color='lightsalmon')
plt.title('Accident Risk Distribution')
plt.xlabel('Accident Risk')

plt.subplot(1, 2, 2)
df['accident_risk'].plot(kind='box', color='salmon')
plt.title('Accident Risk Box Plot')
plt.tight_layout()
plt.show()


# === 01 ===
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


# # === 02 ===
df['meta_night_weather'] = df['meta_night'] * df['meta_weather']
df['meta_night_curvature'] = df['meta_night'] * df['meta_curvature']
df['meta_speed_curvature'] = df['meta_speed'] * df['meta_curvature']
df['meta_weather_speed'] = df['meta_weather'] * df['meta_speed']
df['meta_accidents_speed'] = df['meta_accidents'] * df['meta_speed']

df_test['meta_night_weather'] = df_test['meta_night'] * df_test['meta_weather']
df_test['meta_night_curvature'] = df_test['meta_night'] * df_test['meta_curvature']
df_test['meta_speed_curvature'] = df_test['meta_speed'] * df_test['meta_curvature']
df_test['meta_weather_speed'] = df_test['meta_weather'] * df_test['meta_speed']
df_test['meta_accidents_speed'] = df_test['meta_accidents'] * df_test['meta_speed']



df['speed_squared'] = df['speed_limit'] ** 2
df_test['speed_squared'] = df_test['speed_limit'] ** 2

df['curvature_squared'] = df['curvature'] ** 2
df_test['curvature_squared'] = df_test['curvature'] ** 2

df['curvature_cubed'] = df['curvature'] ** 3
df_test['curvature_cubed'] = df_test['curvature'] ** 3

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


#convert Boolean columns

bool_cols = ["road_signs_present", "public_road","holiday", "school_season"]
for col in bool_cols :
    df[col]= df[col].astype(int)
    df_test[col]=df_test[col].astype(int)


#label encoding categorical feature
le = LabelEncoder()
cate_cols = df.select_dtypes(exclude="number").columns.tolist()
# cate_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
for col in cate_cols :
    df[col]= le.fit_transform(df[col])
    df_test[col]=le.transform(df_test[col])
    


df_test.head()


df.head()


print("Train Shape :" , df.shape)
print("Test Shape :" , df_test.shape)


X= df.drop('accident_risk', axis =1)
y= df['accident_risk']
X_test = df_test

X_train= X
y_train= y


param_lgb = {

    'n_estimators': 2700,
    'learning_rate': 0.01,
    'num_leaves': 99,
    'max_depth': 13,
    'min_child_samples': 10,
    'min_child_weight': 0.002,
    'subsample': 0.60,
    'subsample_freq': 1,
    'colsample_bytree': 0.83,
    'reg_alpha': 0.01,
    'reg_lambda':  0.70,
    'min_split_gain':  0.004,
    'feature_fraction': 0.9 , 

}
# param_cat = {'learning_rate': 0.10072351448192624, 'depth': 9, 'l2_leaf_reg': 5.267770868260042, 'random_strength': 0.8941225501211114, 'subsample': 0.7735149421688305}
param_cat = {
     'bagging_temperature' : 0.20,
     'border_count'        : 178,
     'depth'               : 8,
     'iterations'          : 1600,
     'l2_leaf_reg'         : 4,
     'learning_rate'       : 0.04,
     'random_strength'    : 0.32,
}

param_xgb = {
              'n_estimators': 1251,
              'learning_rate': 0.007450884004079273,
              'max_depth': 9, 'min_child_weight': 3,
              'subsample': 0.7168482209791528,
              'colsample_bytree': 0.7447751858960576,
              'colsample_bylevel': 0.9425051040083456,
              'gamma': 0.00022113931706211482,
              'reg_alpha': 0.6157756027561605,
              'reg_lambda': 4.922686388710151}



cat_model =  CatBoostRegressor(**param_cat,
                               loss_function='RMSE',
                               random_seed=42,
                               verbose=False,
                               thread_count=-1,)

lgb_model = lgb.LGBMRegressor(**param_lgb ,
                               objective='regression',
                               metric='rmse',
                               boosting_type='gbdt',
                               random_state=42,
                               n_jobs=-1,
                               verbose=-1 ,
                               tweedie_variance_power=1.5,   
                            ) 

xgb_model = xgb.XGBRegressor(**param_xgb,
                              random_state = 42,
                              objective = 'reg:squarederror')





print("\n" + "="*60)
print("METHOD 1: Simple Average ")
print("="*60)


kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    
    # Train cat
    cat_model.fit(X_tr, y_tr, verbose=False)
    cat_pred = cat_model.predict(X_val)
    
    # Train LightGBM
    lgb_model.fit(X_tr, y_tr)
    lgb_pred = lgb_model.predict(X_val)
    
        # Train XGBoost
    xgb_model.fit(X_tr, y_tr)
    xgb_pred = xgb_model.predict(X_val)
    # Simple average
    # ensemble_pred = 0.3  * cat_pred + 0.3 * lgb_pred + 0.4 * xgb_pred
    ensemble_pred = (cat_pred + lgb_pred +xgb_pred)/3
    
    rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
    cv_scores.append(rmse)
    print(f"Fold {fold}: {rmse:.5f}")

simple_avg_score = np.mean(cv_scores)
print(f"\nSimple Average CV Score: {simple_avg_score:.6f} (+/- {np.std(cv_scores):.6f})")



xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)

cat_model.fit(X_train,y_train,)
cat_pred = cat_model.predict(X_test)

ensemble_pred = (cat_pred + lgb_pred +  xgb_pred)/3


feature_importances = xgb_model.feature_importances_

importance_df = pd.DataFrame({
    'feature': X.columns, 
    'importance': feature_importances
})

importance_df = importance_df.sort_values('importance', ascending=False)

plt.style.use('fivethirtyeight')
plt.figure(figsize=(10, 8))
sns.barplot(x='importance', 
            y='feature', 
            data=importance_df.head(10)) 
plt.title('Feature Importance (XGB)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()


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


df_sub['accident_risk'] = ensemble_pred

df_sub.to_csv('submission.csv', index=False)

df_sub.head()

