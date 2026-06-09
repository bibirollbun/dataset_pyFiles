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


train_df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import make_scorer, mean_squared_error
import catboost as CatBoostRegressor
from scipy.stats import uniform, randint



train_df.head(5)


test_df.head(5)


train_df_null = train_df.isna().sum()
test_df_null = test_df.isna().sum()
print(f"Null values in train_df: {train_df_null}")
print(f"\nNull values in test_df:  {test_df_null}")


train_df.duplicated().sum()


test_df.duplicated().sum()


new_train_df = train_df.drop('id', axis = 1)
new_test_df = test_df.drop('id', axis = 1)


cat_col = ['road_type','lighting','weather','road_signs_present','public_road','time_of_day','holiday','school_season']
num_col = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents', 'accident_risk']



correlation_plot = train_df[num_col].corr()
plt.figure(figsize = (10,6))
sns.heatmap(
    correlation_plot,
    annot = True,
    cmap = 'coolwarm',
    fmt = ".2f"
)
plt.title("Correlation Heatmap")
plt.show()


train_df[cat_col].dtypes


train_df[num_col].describe().T


train_df[cat_col].describe().T


for c in cat_col:
    print(f"Unique Value for {c} : {train_df[c].unique()}")


fig, axes = plt.subplots(2,4, figsize = (12,6))
axes = axes.flatten()
cmap = plt.get_cmap('magma')
colors = cmap([0.9,0.66,0.33])

for i, col in enumerate(cat_col):
    grouped = train_df.groupby(col)['accident_risk'].mean()
    axes[i].bar(grouped.index.astype(str),grouped.values, color = colors)
    axes[i].set_ylabel(f"Mean (Accident Risk)")
    axes[i].set_title(f"{col} vs Target")
    #axes[i].tick_params(axis = 'x', rotation = 45)

plt.tight_layout()
plt.show()



plt.figure(figsize = (12,6))
plt.hist(train_df['accident_risk'], bins = 50, edgecolor = "black")
plt.xlabel("Accident Risk")
plt.title("Accident Risk Distribution Plot")
plt.show()


sns.displot(train_df, x= "accident_risk", kind= "kde", hue = "lighting")


train_df['weather-lighting'] = train_df['weather'].astype(str) + "_" + train_df['lighting'].astype(str)
test_df['weather-lighting'] = test_df['weather'].astype(str) + "_" + test_df['lighting'].astype(str)


train_df['squared_speed_limit'] = train_df['speed_limit'] ** 2
test_df['squared_speed_limit'] = test_df['speed_limit'] ** 2
train_df['squared_curvature'] = train_df['curvature'] ** 2
test_df['squared_curvature'] = test_df['curvature'] ** 2


#train_df['speed_limit'].hist(bins= 20)
#train_df['curvature'].hist(bins = 20)
#train_df['num_lanes'].hist(bins = 20)
train_df['num_reported_accidents'].hist(bins=20)


train_df['log_speed'] = np.log1p(train_df['speed_limit'])
test_df['log_speed'] = np.log1p(test_df['speed_limit'])
train_df['log_curvature'] = np.log1p(train_df['curvature'])
test_df['log_curvature'] = np.log1p(test_df['curvature'])
train_df['log_lanes'] = np.log1p(train_df['num_lanes'])
test_df['log_lanes'] = np.log1p(test_df['num_lanes'])
train_df['log_accidents'] = np.log1p(train_df['num_reported_accidents'])
test_df['log_accidents'] = np.log1p(test_df['num_reported_accidents'])



train_df = train_df.drop('id', axis = 1)
test_df = test_df.drop('id', axis = 1)


train_df.head(5)


bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
for cols in bool_cols:
    train_df[cols] = train_df[cols].astype(int)
    test_df[cols] = test_df[cols].astype(int)


train_df.head(5)


le = LabelEncoder()
cat_val = ['road_type', 'lighting', 'weather', 'time_of_day', 'weather-lighting']
for cols in cat_val:
    train_df[cols] = le.fit_transform(train_df[cols])
    test_df[cols] = le.fit_transform(test_df[cols])


train_df.shape


train_df.head(5)


print(f"Train_df shape: {train_df.shape} ")
print(f"Test_df shape: {test_df.shape} ")


X = train_df.drop("accident_risk", axis = 1)
y = train_df["accident_risk"]
X_test = test_df
X_train = X
y_train = y


import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
lgb_m = lgb.LGBMRegressor(
    random_state = 42,
    verbose = -1,
    objective = 'regression',
    metric = 'rmse',
    boosting_type = 'gbdt')
#verbose hides unnecessary training logs
#rmse -> Root Mean Squared Error
#gbdt -> Gradient Boosting Decision Tree, alteratives - rf, dart, goss
kfold = KFold(n_splits = 5, shuffle = True, random_state = 42)

cv_scores = []
for fold, (train_idx, val_idx) in enumerate (kfold.split(X_train), 1):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    lgb_m.fit(X_tr, y_tr)
    pred = lgb_m.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, pred))
    cv_scores.append(rmse)
    print(f"Fold{fold}: {rmse :.5f}")

avg_cv_score = np.mean(cv_scores)
print(f"Average Score: {avg_cv_score:.5f}")


from catboost import CatBoostRegressor
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

# catboost best param from random search cv
param_cat = {
     'bagging_temperature' : 0.20,
     'border_count'        : 178,
     'depth'               : 8,
     'iterations'          : 1600,
     'l2_leaf_reg'         : 4,
     'learning_rate'       : 0.04,
     'random_strength'    : 0.32,
     
}


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

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    
    # Train cat
    cat_model.fit(X_tr, y_tr)
    cat_pred = cat_model.predict(X_val)
    
    # Train LightGBM
    lgb_model.fit(X_tr, y_tr)
    lgb_pred = lgb_model.predict(X_val)
    
    # Simple average
    ensemble_pred = 0.1  * cat_pred + 0.9 * lgb_pred
    
    rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
    cv_scores.append(rmse)
    print(f"Fold {fold}: {rmse:.5f}")

simple_avg_score = np.mean(cv_scores)
print(f"\nSimple Average CV Score: {simple_avg_score:.5f} (+/- {np.std(cv_scores):.5f})")


def create_meta_features(models, X_train, X_test, y_train, n_splits=5):
    
    #Create out-of-fold predictions for training meta-model
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Arrays to store predictions
    meta_train = np.zeros((len(X_train), len(models)))
    meta_test = np.zeros((len(X_test), len(models)))
    
    for i, model in enumerate(models):
        print(f"Processing model {i+1}...")
        test_preds = np.zeros(len(X_test))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr = y_train.iloc[train_idx]
            
            # Train model on fold
            model.fit(X_tr, y_tr)
            
            # Get out-of-fold predictions
            meta_train[val_idx, i] = model.predict(X_val)
            
            # Get test predictions for this fold
            test_preds += model.predict(X_test) / n_splits
        
        meta_test[:, i] = test_preds
    
    return meta_train, meta_test

# Tuned  models
models = [cat_model,  lgb_model]

# Create meta features
meta_train, meta_test = create_meta_features(models, X_train, X_test, y_train)


from sklearn.linear_model import Ridge, LinearRegression
meta_model = Ridge(alpha=0.1)  # Start with Ridge regression

# meta_model = LinearRegression()  # Or try simple linear regression
# meta_model = lgb.LGBMRegressor()  # Or simple LGBM

meta_model.fit(meta_train, y_train)
rmse = np.sqrt(mean_squared_error(y_train, meta_model.predict(meta_train)))
print(f"rmse for {meta_model} : {rmse}")
final_predictions = meta_model.predict(meta_test)


sample_sub['accident_risk'] = final_predictions


sample_sub.to_csv('submission.csv', index=False)


sample_sub.head()




