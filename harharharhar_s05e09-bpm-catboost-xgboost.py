import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from catboost import cv, Pool
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


train.head()


train.columns


train_clean = train.drop(['id'],axis=1)
test_clean = test.drop(['id'],axis=1)


train_clean.info()


train_clean.describe()


data_corr = train_clean


corr = data_corr.corr()
corr


plt.figure(figsize=(16,10))
mask = np.triu(np.ones_like(corr))
sns.heatmap(corr, cmap='viridis', linewidths=0.9, annot=True,mask=mask)


#focus on bpm
plt.figure(figsize=(16,10))
mask = np.triu(np.ones_like(corr))
sns.heatmap(corr, cmap='viridis', linewidths=0.9, annot=True, vmax=0.005,vmin=-0.005, mask=mask)


sns.histplot(data=train_clean, x='BeatsPerMinute', bins=20, kde=True)


X = train_clean.drop(['BeatsPerMinute'], axis=1)
y = train_clean['BeatsPerMinute']
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
test_sca = scaler.transform(test_clean)


# CatBoost Hyperparameters
catboost_param_grid = {
    'iterations': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [3, 5, 10],
    'bootstrap_type': ['Bernoulli'],  
    'subsample': [0.6, 0.8, 1.0]  
}


catboost_model = CatBoostRegressor(loss_function='RMSE', verbose=0,)
catboost_random_search = RandomizedSearchCV(
    estimator=catboost_model,
    param_distributions=catboost_param_grid,
    n_iter=10, cv=3, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1
)
catboost_random_search.fit(X_train, y_train)
print(f"Best CatBoost params: {catboost_random_search.best_params_}")


# XGBoost Hyperparameters
xgboost_param_grid = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}


xgboost_model = XGBRegressor(eval_metric='rmse', n_jobs=-1)
xgboost_random_search = RandomizedSearchCV(
    estimator=xgboost_model, 
    param_distributions=xgboost_param_grid, 
    n_iter=10, cv=3, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1
)
xgboost_random_search.fit(X_train, y_train)
print(f"Best XGBoost params: {xgboost_random_search.best_params_}")


catboost_model = CatBoostRegressor(**catboost_random_search.best_params_)
xgboost_model = XGBRegressor(**xgboost_random_search.best_params_)
catboost_model.fit(X_train, y_train)
xgboost_model.fit(X_train, y_train)



feature_importances = catboost_model.get_feature_importance()
feature_names = test_clean.columns 
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)


plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


catboost_preds = catboost_model.predict(X_test)
xgboost_preds = xgboost_model.predict(X_test)


ensemble_preds = (catboost_preds + xgboost_preds) / 2


ensemble_rmse = mean_squared_error(y_test, ensemble_preds, squared=False)
print(f"Ensemble RMSE (averaged): {ensemble_rmse}")


catboost_preds_test = catboost_model.predict(test_sca)
xgboost_preds_test = xgboost_model.predict(test_sca)
stacked_preds_test = np.column_stack((catboost_preds_test, xgboost_preds_test))
meta_model = LinearRegression()
meta_model.fit(stacked_preds, y_test)
final_preds_test = meta_model.predict(stacked_preds_test)

submission = pd.DataFrame({
    'id': test['id'],
    'BeatsPerMinute': final_preds_test
})

submission.to_csv('/kaggle/working/submission_bpm_mean.csv', index=False)






