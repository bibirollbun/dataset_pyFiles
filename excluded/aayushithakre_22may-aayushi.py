import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns



train = pd.read_csv('../input/traindataset/train.csv')
test = pd.read_csv('../input/testdataset/test.csv')



def preprocess(df):
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])  # Convert Male/Female to 0/1
    return df

train = preprocess(train)
test = preprocess(test)



X = train.drop(columns=['Calories', 'id'])
y = train['Calories']
X_test = test.drop(columns=['id'])



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))
#it was mentioned in the comp info


#linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_val)
lr_preds = np.maximum(lr_preds, 0)  # Clip negative predictions to zero
print("LR RMSLE:", rmsle(y_val, lr_preds))



#random forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_val)
rf_preds = np.maximum(rf_preds, 0)
print("RF RMSLE:", rmsle(y_val, rf_preds))


xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_val)
xgb_preds = np.maximum(xgb_preds, 0)
print("XGBoost RMSLE:", rmsle(y_val, xgb_preds))



lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
lgb_model.fit(X_train, y_train)
lgb_preds = lgb_model.predict(X_val)
print("LightGBM RMSLE:", rmsle(y_val, lgb_preds))



lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
lgb_model.fit(X_train, y_train)
lgb_preds1 = lgb_model.predict(X_val)
lgb_preds1 = np.maximum(lgb_preds1, 0)
print("LightGBM RMSLE:", rmsle(y_val, lgb_preds1))



print("LR RMSLE:", rmsle(y_val, lr_preds))
#print("RF RMSLE:", rmsle(y_val, rf_preds))
print("XGB RMSLE:", rmsle(y_val, xgb_preds))
print("LGB RMSLE:", rmsle(y_val, lgb_preds))
print("LGB RMSLE:", rmsle(y_val, lgb_preds1))


final_preds = xgb_model.predict(X_test)
final_preds = np.maximum(final_preds, 0)



submission = pd.DataFrame({
    'id': test['id'],
    'Calories': final_preds
})
submission.to_csv('submission1.csv', index=False)




from sklearn.metrics import r2_score

# Refit all models on full train data (not just X_train)
lr = LinearRegression().fit(X, y)
#rf = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1).fit(X, y)
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(X, y)
lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(X, y)

# Predict on training data
lr_train_preds = lr.predict(X)
#rf_train_preds = rf.predict(X)
xgb_train_preds = xgb_model.predict(X)
lgb_train_preds = lgb_model.predict(X)

# R2 scores
print("Model R² Accuracy on Training Data:")
print(f"Linear Regression: {r2_score(y, lr_train_preds):.4f}")
#print(f"Random Forest:     {r2_score(y, rf_train_preds):.4f}")
print(f"XGBoost:           {r2_score(y, xgb_train_preds):.4f}")
print(f"LightGBM:          {r2_score(y, lgb_train_preds):.4f}")

# R² = 1 → perfect prediction
# R² = 0 → model predicts no better than the mean
# R² < 0 → model performs worse than mean prediction

