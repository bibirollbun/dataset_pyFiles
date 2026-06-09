import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


train.head()


train.isna().any()


test.isna().any()


categorical_cols = test.select_dtypes(include=['object']).columns
numerical_cols = test.select_dtypes(include=['int64', 'float64']).columns

print(f"The categorical value columns are: {categorical_cols.values}")
print(f"The numerical value columns are: {numerical_cols.values}")


sns.set_style('whitegrid')
sns.histplot(data=train, x='accident_risk', palette='Set2', bins=50)
plt.title('Count of accident_risk')
plt.xlabel('y')
plt.ylabel('Count')
plt.show()


encoder = OrdinalEncoder()
train[categorical_cols] = encoder.fit_transform(train[categorical_cols])
test[categorical_cols] = encoder.transform(test[categorical_cols])


X = train.drop('accident_risk', axis=1)
y = train['accident_risk']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


xgb = XGBRegressor(
    verbosity=0, 
    use_label_encoder=False, 
    n_estimators=1000,
    eval_metric='rmse', 
    objective= 'reg:squarederror',
    random_state=42, 
    learning_rate=0.01,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=1.0,
)

lgbm = LGBMRegressor(
    learning_rate=0.05,
    n_estimators=850,
    allow_writing_files=False, 
    random_state=42,
    n_jobs=-1,
    subsample=0.6,
    num_leaves=200,
    max_depth=5,
    verbose=-1
)

catb = CatBoostRegressor(
    learning_rate=0.1,
    depth=8,
    verbose=0, 
    random_state=42,
    loss_function="RMSE",
    l2_leaf_reg=9,
    iterations=850
)


xgb.fit(X_train, y_train)


lgbm.fit(X_train, y_train)


catb.fit(X_train, y_train)


pred_xgb = xgb.predict(X_test)
pred_lgbm = lgbm.predict(X_test)
pred_catb = catb.predict(X_test)


rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb))
rmse_lgbm = np.sqrt(mean_squared_error(y_test, pred_lgbm))
rmse_catb = np.sqrt(mean_squared_error(y_test, pred_catb))

print(f"RMSE - XGBoost  : {rmse_xgb:.4f}")
print(f"RMSE - LightGBM : {rmse_lgbm:.4f}")
print(f"RMSE - CatBoost : {rmse_catb:.4f}")


final_pred_xgb = xgb.predict(test)
final_pred_lgbm = lgbm.predict(test)
final_pred_catb = catb.predict(test)

w_xgb = 0.33
w_lgbm = 0.34
w_catb = 0.33

avg_pred = (
    w_xgb * final_pred_xgb + 
    w_lgbm * final_pred_lgbm + 
    w_catb * final_pred_catb
)


sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
submission = pd.DataFrame({
    'id': sub['id'],
    'accident_risk': avg_pred
})

submission.to_csv('submission.csv', index=False)

