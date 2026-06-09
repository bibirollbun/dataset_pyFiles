import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_log_error,make_scorer

import os



data = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")


# Tranforming categorical data in numerical and using "id" as display
data['Sex'] = data['Sex'].map({'male':0,'female':1})
data = data.set_index('id')

display(data)



X = data.drop(columns=['Calories'])
Y = data['Calories']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

model = lgb.LGBMRegressor(
    n_estimators=5000,
    learning_rate=0.01,
    num_leaves=128,
    max_depth=10,             
    subsample=0.8,
    colsample_bytree=0.7,
    max_bin=255,              
    objective='regression',
    random_state=42
)

model.fit(
        X_train, y_train_log,
        eval_set=[(X_test, y_test_log)],
        eval_metric='rmsle',
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
)


importancias = model.feature_importances_
nomes_features = X.columns

df_importancias = pd.DataFrame({
    'Feature': nomes_features,
    'Importance': importancias
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=df_importancias, x='Importance', y='Feature', palette='viridis')
plt.title('Importância das Features')
plt.xlabel('Importância')
plt.ylabel('Features')
plt.tight_layout()
plt.show()


y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)

rmse = np.sqrt(mean_squared_log_error(y_test, y_pred))

print(f" XGBoost RMSLE: {rmse:.4f}")



pd_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

ids = pd_test['id']
pd_test['Sex'] = pd_test['Sex'].map({'male':0,'female':1})
pd_test = pd_test.drop(columns=['id'])

y_pred_log = model.predict(pd_test)
results = np.expm1(y_pred_log)

submission = pd.DataFrame({
    'id':ids,
    'Calories':results
})



submission = submission.set_index('id')
submission.to_csv('submission_calories.csv')

