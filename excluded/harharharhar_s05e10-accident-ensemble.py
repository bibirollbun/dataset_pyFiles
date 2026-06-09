import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from catboost import cv, Pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


train.head()


train.info()


train.describe()


train_clean = train.drop(['id'],axis=1)
test_clean = test.drop(['id'],axis=1)


columns = ['road_type','lighting','weather','time_of_day','road_signs_present','public_road','holiday','school_season']
ohe = OneHotEncoder(sparse_output = False).set_output(transform='pandas')
feature_array = ohe.fit_transform(train_clean[columns])


train_encoded = pd.concat([train_clean, feature_array], axis=1).drop(columns=columns)


train_encoded.head()


feature_array_test = ohe.fit_transform(test_clean[columns])
test_encoded = pd.concat([test_clean, feature_array_test], axis=1).drop(columns=columns)


test_encoded.head()


corr = train_encoded.corr()


corr


plt.figure(figsize=(16,10))
mask = np.triu(np.ones_like(corr))
sns.heatmap(corr, cmap='Blues', vmin=0, linewidths=0.9,mask=mask)


sns.histplot(data=train_clean, x='accident_risk', bins=20, kde=True)


train_encoded['accident_risk_binned'] = pd.cut(
    train_clean['accident_risk'],
    bins=4,  
    labels=['Low', 'Moderate', 'High', 'Severe']
)
sns.violinplot(data=train_encoded, x='accident_risk_binned', y='curvature', palette='Pastel1')


plt.figure(figsize=(12, 6))
sns.histplot(data=train_encoded, x='curvature', hue='accident_risk_binned', bins=10, kde=True)


sns.violinplot(data=train_encoded, x='accident_risk_binned', y='speed_limit', palette='Pastel1')


plt.figure(figsize=(12, 6))
sns.histplot(data=train_encoded, x='speed_limit', hue='accident_risk_binned', bins=5, kde=True)


sns.violinplot(data=train_encoded, x='accident_risk_binned', y='lighting_night', palette='Pastel1')


plt.figure(figsize=(12, 6))
sns.histplot(data=train_encoded, x='lighting_night', hue='accident_risk_binned', bins=2, kde=True)


X = train_encoded.drop(['accident_risk', 'accident_risk_binned'], axis=1)
y = train_encoded['accident_risk']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=101)
test_sca = scaler.transform(test_encoded)


cat_params = {
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 10,
    'subsample': 0.8,
    'loss_function': 'RMSE',
    'verbose': 50,
}


cv_data = cv(
    pool=Pool(X_train, y_train),
    params={**cat_params, 'iterations': 500},  
    fold_count=5,
    shuffle=True,
    partition_random_seed=0,
    plot=True
)


best_iteration = cv_data['test-RMSE-mean'].idxmin()

cat_model = CatBoostRegressor(
    **cat_params,
    iterations=best_iteration
)

cat_model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    verbose=50,
    early_stopping_rounds=50
)


xgboost_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    eval_metric="rmse"
)

xgboost_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=50,
    verbose=50
)


feature_importances = cat_model.get_feature_importance()
feature_names = test_encoded.columns 
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)


plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='Blues')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


catboost_preds = cat_model.predict(X_test)
xgboost_preds = xgboost_model.predict(X_test)


ensemble_preds = (catboost_preds + xgboost_preds) / 2


catboost_preds_test = cat_model.predict(test_sca)
xgboost_preds_test = xgboost_model.predict(test_sca)
ensemble_preds = (catboost_preds_test + xgboost_preds_test) / 2

submission = pd.DataFrame({
    'id': test['id'],
    'accident_risk': ensemble_preds
})

submission.to_csv('/kaggle/working/submission_accident.csv', index=False)




