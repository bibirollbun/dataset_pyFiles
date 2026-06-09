# Data handling
import pandas as pd           # fast tabular data manipulation
import numpy as np            # numeric operations, arrays, random seeds


# Visualization
import seaborn as sns            # high-level statistical plotting
import matplotlib.pyplot as plt  # plotting control (figures, axes)


# Scikit-learn utilities
from sklearn.model_selection import train_test_split             # simple train/test splits
from sklearn.preprocessing import OneHotEncoder                  # encode categorical features
from sklearn.preprocessing import StandardScaler, LabelEncoder   # scaling numeric features / encoding labels
from sklearn.model_selection import RandomizedSearchCV, KFold    # hyperparameter search & cross-validation
from sklearn.metrics import make_scorer, mean_squared_error      # custom scorer wrapper & RMSE metric


# Gradient-boosted models & helpers
import lightgbm as lgb                    # LightGBM gradient boosting framework
from scipy.stats import uniform, randint  # distributions for randomized hyperparameter search
from catboost import CatBoostRegressor    # CatBoost (handles categorical features natively)
import xgboost as xgb                     # XGBoost gradient boosting framework


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
df_sub = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


print(f'Train Dataframe Shape {df. shape}')
print(f'Test Dataframe shape {df_test.shape}')

##
df.columns

# Dropping "id" column in df Train
df=df.drop("id", axis=1)
df_test=df_test.drop("id", axis=1)

#
df.head()


#Check Train Null Value
df.isna().sum().sum()


#Check Duplicate Rows
df.duplicated().sum()
#Drop Duplicate Rows
df = df.drop_duplicates()

# recheck Duplicate Rows
df.duplicated().sum()
#Check Test Null Value 
df_test.isna().sum().sum()


num_cols =  df.select_dtypes(include='number').columns.tolist()
cat_cols = df.select_dtypes(exclude='number').columns.tolist()
num_cols.remove('accident_risk')

##
print(f'categorical columns : {cat_cols}')
print(f'numerical columns : {num_cols}')
#Numerical Features Analyse¶
df[num_cols].describe()


# numerical features correlation
plt.figure(figsize=(8, 6))
correlation_matrix = df[num_cols + ['accident_risk']].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', 
            linewidths=1, cmap='coolwarm')
plt.show()


#Categorical Columns Unique
for c in cat_cols :
   print(f" {c} (Uniques): {df[c].unique()}")


fig , axes = plt.subplots(2,4, figsize=(16,8))
axes = axes.flatten()
cmap = plt.get_cmap('coolwarm')
colors = cmap([0.9,0.66,0.33])
target = 'accident_risk'
for i, col in enumerate(cat_cols) :
  
    grouped = df.groupby(col)[target].mean()
   
    axes[i].bar(grouped.index.astype(str), grouped.values , color=colors)  # use: .astype(str), to handle non-string
    
    axes[i].set_ylabel(f'Mean {target}')
    axes[i].set_title(f'{col} vs {target}')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(df['accident_risk'], bins=50, edgecolor='black', color='red')
plt.title('Accident Risk Distribution')
plt.xlabel('Accident Risk')

plt.subplot(1, 2, 2)
df['accident_risk'].plot(kind='box')
plt.title('Accident Risk Box Plot')
plt.tight_layout()
plt.show()


# Combining two categorical
df['weather_lighting'] = df['weather'].astype(str) + '_' + df['lighting'].astype(str)
df_test['weather_lighting'] = df_test['weather'].astype(str) + '_' + df_test['lighting'].astype(str)

# Polynomial feature
df['speed_squared'] = df['speed_limit'] ** 2
df_test['speed_squared'] = df_test['speed_limit'] ** 2

df['curvature_squared'] = df['curvature'] ** 2
df_test['curvature_squared'] = df_test['curvature'] ** 2

# weighted meta-features
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

#Dataframe Shape - Final
df_test.head()

df.head()


def f(X):
    return \
    0.3 * X["curvature"] + \
    0.2 * (X["lighting"] == "night").astype(int) + \
    0.1 * (X["weather"] != "clear").astype(int) + \
    0.2 * (X["speed_limit"] >= 60).astype(int) + \
    0.1 * (X["num_reported_accidents"] > 2).astype(int)

df['meta'] = f(df)
df_test['meta'] = f(df_test)


X_train= df.drop('accident_risk', axis =1)
y_train= df['accident_risk']
X_test = df_test


# Lighgbm best param from random search cv
param_lgb = {
    
    'n_estimators'     : 2700,
    'learning_rate'    : 0.01,
    'num_leaves'       : 99,
    'max_depth'        : 13,
    'min_child_samples': 10,
    'min_child_weight' : 0.002,
    'subsample'        : 0.60,
    'subsample_freq'   : 1,
    'colsample_bytree' : 0.83,
    'reg_alpha'        : 0.01,
    'reg_lambda'       :  0.70,
    'min_split_gain'   :  0.004,
    'feature_fraction' : 0.9, 

 
}

# catboost best param from random search cv
param_cat = {
     'bagging_temperature': 0.20,
     'border_count'       : 178,
     'depth'              : 8,
     'iterations'         : 1600,
     'l2_leaf_reg'        : 4,
     'learning_rate'      : 0.04,
     'random_strength'    : 0.32,
     
}

# xgboost best param from random search cv
param_xgb = {
              'n_estimators'     : 1251,
              'learning_rate'    : 0.0074,
              'max_depth'        : 9,
              'min_child_weight' : 3,
              'subsample'        : 0.72,
              'colsample_bytree' : 0.74,
              'colsample_bylevel': 0.94,
              'gamma'            : 0.0002,
              'reg_alpha'        : 0.61,
              'reg_lambda'       : 4.92}


print("\n" + "="*60)
print("Simple Average (90-10)")
print("="*60)

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
                               verbose=-1    
                            )  

xgb_model = xgb.XGBRegressor(**param_xgb,
                              random_state = 42,
                              objective = 'reg:squarederror')


# Let's see CV Score
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []    # Store RMSE scores from each fold

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]      # Split features
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]      # Split targets
    
    
    # Train cat
    cat_model.fit(X_tr, y_tr)
    cat_pred = cat_model.predict(X_val)                               # Predict on validation set
    
    # Train LightGBM
    lgb_model.fit(X_tr, y_tr)
    lgb_pred = lgb_model.predict(X_val)

    # Train XGBoost
    xgb_model.fit(X_tr, y_tr)
    xgb_pred = xgb_model.predict(X_val)
    
    # Simple average - Weighted average ensemble (30% Cat, 30% LGB, 40% XGB)
    ensemble_pred = 0.3  * cat_pred + 0.3 * lgb_pred + 0.4 * xgb_pred

       
    rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))   # Calculate error
    cv_scores.append(rmse)                                     # Store score
    print(f"Fold {fold}: {rmse:.5f}")                          # Print fold result
 
simple_avg_score = np.mean(cv_scores)                          # Average across all folds
print(f"\nSimple Average CV Score: {simple_avg_score:.5f} (+/- {np.std(cv_scores):.5f})")


# Train XGBoost on ENTIRE training set
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)      # Predict on test set

# Train LightGBM on ENTIRE training set
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)

# Train CatBoost on ENTIRE training set
cat_model.fit(X_train, y_train)
cat_pred = cat_model.predict(X_test)

# Final ensemble prediction for submission
ensemble_pred = 0.3 * cat_pred + 0.3 * lgb_pred + 0.4 * xgb_pred


X = X_train                 # Redefine X

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


df_sub['accident_risk'] = ensemble_pred

df_sub.to_csv('submission.csv', index=False)

df_sub.head(10)

