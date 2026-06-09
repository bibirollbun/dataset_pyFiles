import pandas as pd
import numpy as np


train=pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


train.head(3)


train.shape


train.info()


train.describe().T


train.isna().sum()


test.head(2)


test.shape


test.info()


test.describe().T


test.isna().sum()


train=train.drop('id',axis=1)


train.duplicated().sum()


train=train.drop_duplicates()


bool_cols=["road_signs_present","public_road","holiday","school_season"]
for i in bool_cols:
    train[i]=train[i].astype(int)
    test[i]=test[i].astype(int)


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


le=LabelEncoder()


cat_cols=train.select_dtypes(exclude="number").columns.tolist()
for i in cat_cols:
    train[i]=le.fit_transform(train[i])
    test[i]=le.transform(test[i])


X=train.drop('accident_risk',axis=1)
y=train['accident_risk']
X_test=test.drop('id',axis=1)
X_train=X
y_train=y


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
    'n_estimators': 2700,
    'learning_rate': 0.01,
    'max_depth': 13,
    'min_child_weight': 0.002,
    'subsample': 0.60,
    'colsample_bytree': 0.83,
    'reg_alpha': 0.01,
    'reg_lambda':  0.70,
    'gamma': 0.004, # Equivalent to min_split_gain in LightGBM
    'tree_method': 'hist', # Optimized for speed
}


cat_model =  CatBoostRegressor(**param_cat,
                               loss_function='RMSE',
                               random_seed=42,
                               verbose=False,
                               thread_count=-1,)


xgb_model = XGBRegressor(**param_xgb,
                             objective='reg:squarederror', # For regression with RMSE-like loss
                             eval_metric='rmse',
                             random_state=42,
                             n_jobs=-1,
                             verbosity=0
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
    xgb_model.fit(X_tr, y_tr)
    xgb_pred = xgb_model.predict(X_val)
    
    # Simple average
    ensemble_pred = 0.1  * cat_pred + 0.9 * xgb_pred
    
    rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
    cv_scores.append(rmse)
    print(f"Fold {fold}: {rmse:.5f}")

simple_avg_score = np.mean(cv_scores)
print(f"\nSimple Average CV Score: {simple_avg_score:.5f} (+/- {np.std(cv_scores):.5f})")


cat_model.fit(X_train, y_train)
cat_test_pred = cat_model.predict(X_test)

xgb_model.fit(X_train, y_train)
xgb_test_pred = xgb_model.predict(X_test)

ensemble_pred = 0.7 * cat_test_pred + 0.1 * xgb_test_pred


df_sub = pd.DataFrame({'id': test['id'], 'accident_risk': ensemble_pred})


df_sub.to_csv('submission.csv',index=False)

