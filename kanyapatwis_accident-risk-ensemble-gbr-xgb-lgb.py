import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
train


X_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
X_test


Y_train = train['accident_risk']
X_train = train.drop('accident_risk', axis = 1)


print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)


from sklearn.model_selection import train_test_split

X_train_split, X_train_val, Y_train_split, Y_train_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42 )


categorical_features = ['road_type', 'lighting','weather','road_signs_present', 
                        'public_road','time_of_day', 'holiday','school_season']

numeric_features = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

X_train_split_cat = X_train_split[categorical_features]
X_train_split_num = X_train_split[numeric_features]

X_train_val_cat = X_train_val[categorical_features]
X_train_val_num = X_train_val[numeric_features]

X_test_cat = X_test[categorical_features]
X_test_num = X_test[numeric_features]




from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # sparse=False to get a DataFrame/array
encoder.fit(X_train_split_cat)

X_train_split_cat_encoded = encoder.transform(X_train_split_cat)
X_train_val_cat_encoded = encoder.transform(X_train_val_cat)
X_test_encoded = encoder.transform(X_test_cat)


# Combine numeric + encoded categorical
X_train_split_final = np.hstack([X_train_split_num.values, X_train_split_cat_encoded])
X_train_val_final = np.hstack([X_train_val_num.values, X_train_val_cat_encoded])
X_test_final = np.hstack([X_test_num.values, X_test_encoded])


########## Gradient Boosting Regression ########## 


from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=4,            
    min_samples_leaf=3, random_state=42)

gbr.fit(X_train_split_final, Y_train_split)


y_pred_train_val_gbr = gbr.predict(X_train_val_final)

rmse_gbr = np.sqrt(mean_squared_error(Y_train_val, y_pred_train_val_gbr))
r2_gbr = r2_score(Y_train_val, y_pred_train_val_gbr)

print("Root Mean Squared Error:", rmse_gbr)
print("R² Score:", r2_gbr)


########## XGBoost ##########


pip install xgboost



import xgboost as xgb

xgb = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

xgb.fit(X_train_split_final,Y_train_split)


y_pred_xgb = xgb.predict(X_train_val_final)

rmse_xgb = np.sqrt(mean_squared_error(Y_train_val, y_pred_xgb))
r2_xgb = r2_score(Y_train_val, y_pred_xgb)

print("Root Mean Squared Error:", rmse_xgb)
print("R² Score:", r2_xgb)


test_preds_xgb = np.zeros(len(X_test_final))

fold_rmse = []

kf = KFold(n_splits = 3, shuffle = True, random_state = 42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
    print(f"\n===== Fold {fold+1} =====")

    X_train_xgbkf, X_val_xgbkf = X_train_full[train_idx], X_train_full[val_idx]
    y_train_xgbkf, y_val_xgbkf = Y_train[train_idx], Y_train[val_idx]

    xgb.fit(X_train_xgbkf,y_train_xgbkf)

    # predict on validation set
    y_pred_xbgkf = xgb.predict(X_val_xgbkf)

    rmse_xgbkf = np.sqrt(mean_squared_error(y_val_xgbkf, y_pred_xbgkf))
    fold_rmse.append(rmse_xgbkf)
    print(f"Fold {fold+1} RMSE: {rmse_xgbkf:.8f}")

    test_preds_xgb += xgb.predict(X_test_final) / kf.n_splits

print(f"\nAverage RMSE across folds: {np.mean(fold_rmse):.8f}")


# Predict
y_pred_test_xgb = test_preds_xgb
y_pred_test_xgb


########## Lightgbm ##########


pip install lightgbm


import lightgbm as lgb

lgb = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=-1, subsample=0.8,
        colsample_bytree=0.8, random_state=42)

lgb.fit(X_train_split_final,Y_train_split)


y_pred_lgb = lgb.predict(X_train_val_final)

rmse_lgb = np.sqrt(mean_squared_error(Y_train_val, y_pred_lgb))
r2_lgb = r2_score(Y_train_val, y_pred_lgb)

print("Root Mean Squared Error:", rmse_lgb)
print("R² Score:", r2_lgb)


y_pred_test_lgb = lgb.predict(X_test_final)
y_pred_test_lgb


# Combined all models using ensemble techniques

kf = KFold(n_splits=5, shuffle=True, random_state=42)

xgb_oof = np.zeros(len(X_train_full))
lgb_oof = np.zeros(len(X_train_full))
gbr_oof = np.zeros(len(X_train_full))

xgb_test_preds = np.zeros((len(X_test_final), 5))
lgb_test_preds = np.zeros((len(X_test_final), 5))
gbr_test_preds = np.zeros((len(X_test_final), 5))

for i, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
    X_tr, X_val = X_train_full[train_idx], X_train_full[val_idx]
    y_tr, y_val = Y_train[train_idx], Y_train[val_idx]
    
    # Train models on each fold
    xgb.fit(X_tr, y_tr)
    lgb.fit(X_tr, y_tr)
    gbr.fit(X_tr, y_tr)
    
    # Out-of-fold predictions (on unseen data)
    xgb_oof[val_idx] = xgb.predict(X_val)
    lgb_oof[val_idx] = lgb.predict(X_val)
    gbr_oof[val_idx] = gbr.predict(X_val)
    
    # Predict on test set
    xgb_test_preds[:, i] = xgb.predict(X_test_final)
    lgb_test_preds[:, i] = lgb.predict(X_test_final)
    gbr_test_preds[:, i] = gbr.predict(X_test_final)

# Average test predictions across folds
xgb_test_final = xgb_test_preds.mean(axis=1)
lgb_test_final = lgb_test_preds.mean(axis=1)
gbr_test_final = gbr_test_preds.mean(axis=1)



from sklearn.linear_model import Ridge
meta_X = np.column_stack([xgb_oof, lgb_oof, gbr_oof])
meta_model = Ridge()
meta_model.fit(meta_X, Y_train)

meta_X_pred = meta_model.predict(meta_X[val_idx])
rmse_stacked = np.sqrt(mean_squared_error(meta_X_pred, y_val))

meta_test_X = np.column_stack([xgb_test_final, lgb_test_final, gbr_test_final])
final_pred = meta_model.predict(meta_test_X)
final_pred


submission = pd.DataFrame({
    'id': X_test['id'],
    'accident_risk': final_pred
})

submission.to_csv("/kaggle/working/submission.csv", index=False)



submission

