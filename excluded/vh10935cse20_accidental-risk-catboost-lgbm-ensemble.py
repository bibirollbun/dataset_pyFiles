import pandas as pd
import numpy as np


train=pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


train.head(3)


train.shape


train.dtypes


train.info()


train.describe().T


train.isna().sum()


test.head(2)


test.shape


test.describe().T


test.info()


test.isna().sum()


train=train.drop('id',axis=1)


train.duplicated().sum()


train=train.drop_duplicates()


num_cols =  train.select_dtypes(include="number").columns.tolist()
cat_cols = train.select_dtypes(exclude="number").columns.tolist()
num_cols.remove("accident_risk")

print(f"categorical columns : {cat_cols}")
print(f"numerical columns : {num_cols}")


from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from scipy.stats import uniform, randint


bool_cols = ["road_signs_present", "public_road","holiday", "school_season"]
for col in bool_cols :
    train[col]= train[col].astype(int)
    test[col]=test[col].astype(int)


le = LabelEncoder()
cate_cols = train.select_dtypes(exclude="number").columns.tolist()
for col in cate_cols :
    train[col]= le.fit_transform(train[col])
    test[col]=le.transform(test[col])


X= train.drop('accident_risk', axis =1)
y= train['accident_risk']
X_test = test.drop('id', axis=1)

X_train= X
y_train= y


import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import make_scorer, mean_squared_error
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


param_cat = {
     'bagging_temperature' : 0.20,
     'border_count'        : 178,
     'depth'               : 8,
     'iterations'          : 1600,
     'l2_leaf_reg'         : 4,
     'learning_rate'       : 0.04,
     'random_strength'    : 0.32,
     
}


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


print("\n" + "="*60)
print("Simple Average (90-10)")
print("="*60)

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


cat_model.fit(X_train, y_train)
cat_test_pred = cat_model.predict(X_test)

lgb_model.fit(X_train, y_train)
lgb_test_pred = lgb_model.predict(X_test)

ensemble_pred = 0.5 * cat_test_pred + 0.5 * lgb_test_pred


df_sub = pd.DataFrame({'id': test['id'], 'accident_risk': ensemble_pred})


df_sub.to_csv('submission.csv', index=False)

