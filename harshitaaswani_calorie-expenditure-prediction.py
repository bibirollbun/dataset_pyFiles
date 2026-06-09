import numpy as np
import pandas as pd


train=pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")


test=pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


train.head()


train.tail()


train.shape


train.size


train.info()


train.columns


test.head()


test.tail()


test.shape


test.size


test.info()


test.columns


import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(train[['Age', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']], diag_kind='kde')
plt.show()


sns.countplot(data=train, x='Sex')
plt.title("Gender Distribution")
plt.show()


sns.histplot(train['Calories'], kde=True, bins=30, color='green')
plt.title("Distribution of Calories Burned")
plt.xlabel("Calories")
plt.ylabel("Frequency")
plt.show()


features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']

for col in features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=train[col])
    plt.title(f"Boxplot of {col}")
    plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(train.drop(columns=['id', 'Sex']).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


sns.boxplot(x='Sex', y='Calories', data=train)
plt.title("Calories Burned by Sex")
plt.show()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])


from sklearn.preprocessing import StandardScaler

features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
scaler = StandardScaler()
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])


from sklearn.model_selection import train_test_split

X = train.drop(['id', 'Calories'], axis=1)
y = train['Calories']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

# LightGBM
lgb = LGBMRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
lgb.fit(X_train, y_train)
lgb_pred = lgb.predict(X_val)
print("LightGBM RMSE:", mean_squared_error(y_val, lgb_pred, squared=False))


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# XGBoost
xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_val)
print("XGB RMSE:", mean_squared_error(y_val, xgb_pred, squared=False))


from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

# CatBoost
cat = CatBoostRegressor(iterations=200, learning_rate=0.1, random_state=42, verbose=0)
cat.fit(X_train, y_train)
cat_pred = cat.predict(X_val)
print("CatBoost RMSE:", mean_squared_error(y_val, cat_pred, squared=False))


from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

estimators = [
    ('xgb', XGBRegressor(n_estimators=200, learning_rate=0.1)),
    ('lgb', LGBMRegressor(n_estimators=200, learning_rate=0.1)),
    ('cat', CatBoostRegressor(iterations=200, learning_rate=0.1, verbose=0))
]

stack = StackingRegressor(estimators=estimators, final_estimator=Ridge())
stack.fit(X_train, y_train)
pred = np.expm1(stack.predict(X_val))
print("Stacked RMSE:", mean_squared_error(y_val, pred, squared=False))


from sklearn.ensemble import HistGradientBoostingRegressor

hgb = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=42)
hgb.fit(X_train, y_train)
hgb_pred = np.expm1(hgb.predict(X_val))
print("HGB RMSE:", mean_squared_error(y_val, hgb_pred, squared=False))


from sklearn.ensemble import VotingRegressor

ensemble = VotingRegressor(estimators=[('hgb', hgb), ('xgb', xgb), ('lgb', lgb), ('cat', cat)])
ensemble.fit(X_train, y_train)
ensemble_pred = ensemble.predict(X_val)
print("Ensemble RMSE:", mean_squared_error(y_val, ensemble_pred, squared=False))


final_preds = ensemble.predict(test.drop(['id'], axis=1))
submission = pd.DataFrame({'id': test['id'], 'Calories': final_preds})
submission.to_csv('submission.csv', index=False)




