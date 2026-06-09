# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

# Read data
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
training_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

# Combine train and extra data
train = pd.concat([train, training_extra], axis=0).reset_index(drop=True)

# Sample a smaller training dataset to speed up
train = train.sample(frac=0.3, random_state=42).reset_index(drop=True)

# Prepare features and target
X = train.drop(['id', 'Price'], axis=1)
y = train['Price']
X_test = test.drop(['id'], axis=1)

# Encode categorical features
cat_features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
for col in cat_features:
    lbl = LabelEncoder()
    X[col] = lbl.fit_transform(X[col].astype(str))
    X_test[col] = lbl.transform(X_test[col].astype(str))

# Initialize base models with fewer trees
lgb_model = LGBMRegressor(n_estimators=30, random_state=42)
xgb_model = XGBRegressor(n_estimators=30, random_state=42)

# Prepare arrays to store out-of-fold predictions
oof_preds_lgb = np.zeros(len(X))
oof_preds_xgb = np.zeros(len(X))

test_preds_lgb = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))

# Set up 3-fold cross-validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Train base models
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    lgb_model.fit(X_train, y_train)
    oof_preds_lgb[val_idx] = lgb_model.predict(X_val)
    test_preds_lgb += lgb_model.predict(X_test) / kf.n_splits
    
    xgb_model.fit(X_train, y_train)
    oof_preds_xgb[val_idx] = xgb_model.predict(X_val)
    test_preds_xgb += xgb_model.predict(X_test) / kf.n_splits

# Create meta features for stacking
X_meta = pd.DataFrame({
    'lgb': oof_preds_lgb,
    'xgb': oof_preds_xgb
})
X_test_meta = pd.DataFrame({
    'lgb': test_preds_lgb,
    'xgb': test_preds_xgb
})

# Train Level-2 meta model (Ridge)
ridge_model = Ridge()
ridge_model.fit(X_meta, y)
meta_preds = ridge_model.predict(X_meta)
meta_test_preds = ridge_model.predict(X_test_meta)

# Combine meta features with original features for Level-3 model
X_meta_level3 = pd.concat([X, X_meta], axis=1)
X_test_meta_level3 = pd.concat([X_test, X_test_meta], axis=1)

# Fill missing values
X_meta_level3 = X_meta_level3.fillna(X_meta_level3.mean())
X_test_meta_level3 = X_test_meta_level3.fillna(X_test_meta_level3.mean())

# Train Level-3 model (MLP Regressor)
mlp_model = MLPRegressor(hidden_layer_sizes=(50,), random_state=42, max_iter=500)
mlp_model.fit(X_meta_level3, y)
final_preds = mlp_model.predict(X_meta_level3)
final_test_preds = mlp_model.predict(X_test_meta_level3)

# Evaluate model performance
rmse_level2 = mean_squared_error(y, meta_preds, squared=False)
rmse_level3 = mean_squared_error(y, final_preds, squared=False)
print(f'Level-2 Ridge RMSE: {rmse_level2:.5f}')
print(f'Level-3 MLP RMSE: {rmse_level3:.5f}')

# Prepare submission
submission = pd.DataFrame({
    'id': test['id'],
    'Price': final_test_preds
})
submission.to_csv('submission.csv', index=False)


