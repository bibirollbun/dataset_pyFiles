# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")
test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")
train.head()


test.head()


train.info()
train.describe()


test.info()


train['datetime'] = pd.to_datetime(train['datetime'])
train.set_index('datetime', inplace=True)

plt.figure(figsize=(10, 6))
plt.plot(train['count'])
plt.title('Hourly Bike Rentals')
plt.xlabel('Date')
plt.ylabel('Count')
plt.show()


plt.hist(train['count'], bins=50, range=(0, 1000), color='blue', edgecolor='black')
plt.title('Distribution of Bike Rentals')
plt.xlabel('Number of Rentals')
plt.ylabel('Frequency')
plt.show()

plt.hist(train['casual'], bins=50, range=(0, 1000), color='green', edgecolor='black')
plt.title('Distribution of Bike Rentals(casual)')
plt.xlabel('Number of Rentals')
plt.ylabel('Frequency')
plt.show()

plt.hist(train['registered'], bins=50, range=(0, 1000), color='yellow', edgecolor='black')
plt.title('Distribution of Bike Rentals(registered)')
plt.xlabel('Number of Rentals')
plt.ylabel('Frequency')
plt.show()


plt.hist(train['temp'], bins=40, color='red', edgecolor='black')
plt.title('Distribution of temperatures')
plt.xlabel('Number of Rentals')
plt.ylabel('Frequency')
plt.show()

plt.hist(train['humidity'], bins=50, color='orange', edgecolor='black')
plt.title('Distribution of humidities')
plt.xlabel('Number of Rentals')
plt.ylabel('Frequency')
plt.show()

sns.countplot(x='weather', data = train)


plt.figure(figsize=(10, 6))
sns.boxplot(x='season', y='count', data=train)
plt.title('Distribution of Bike Rentals by Season')
plt.xlabel('Season')
plt.ylabel('Number of Rentals')
plt.show()

sns.countplot(x='season', data = train)


sns.countplot(x='holiday', data = train)


sns.countplot(x='workingday', data = train)



train=train.reset_index()

train['datetime']=pd.to_datetime(train['datetime'])
train['year']=train['datetime'].dt.year
train['month']=train['datetime'].dt.month
train['day']=train['datetime'].dt.day
train['hour']=train['datetime'].dt.hour
train['weekday']=train['datetime'].dt.weekday 

print(train.head())


train=train.drop(columns='humidity')
train.head()
test=test.drop(columns='humidity')


test.head()


test['datetime']=pd.to_datetime(test['datetime'])
test['year']=test['datetime'].dt.year
test['month']=test['datetime'].dt.month
test['day']=test['datetime'].dt.day
test['hour']=test['datetime'].dt.hour
test['weekday']=test['datetime'].dt.weekday 

print(test.head())


sns.countplot(x='weekday', data = train)


#season=pd.get_dummies(train['season'], prefix='season')
#train=pd.concat([train,season], axis=1)
#weather=pd.get_dummies(train['weather'], prefix='weather')
#train=pd.concat([train,weather], axis=1)
#train.head()


#season=pd.get_dummies(test['season'], prefix='season')
#test=pd.concat([test,season], axis=1)
#weather=pd.get_dummies(test['weather'], prefix='weather')
#test=pd.concat([test,weather], axis=1)
#test.head()


cor_mat= train[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat, mask=mask, square=True, annot=True, fmt='.2f', cbar=True)


# casualの特徴量とターゲット変数を定義
X_casual = train.drop(['count', 'casual', 'registered', 'datetime'], axis=1)
y_casual = train['casual']

# registeredの特徴量とターゲット変数を定義
X_registered = train.drop(['count', 'casual', 'registered', 'datetime'], axis=1)
y_registered = train['registered']


# 訓練データとテストデータに分割
X_casual_train, X_casual_test, y_casual_train, y_casual_test = train_test_split(X_casual, y_casual, test_size=0.2, random_state=42)
X_registered_train, X_registered_test, y_registered_train, y_registered_test = train_test_split(X_registered, y_registered, test_size=0.2, random_state=42)

# ランダムフォレストモデルの訓練
model_casual = RandomForestRegressor(n_estimators=100, random_state=42)
model_casual.fit(X_casual_train, y_casual_train)

model_registered = RandomForestRegressor(n_estimators=100, random_state=42)
model_registered.fit(X_registered_train, y_registered_train)

# 予測と評価
y_casual_pred = model_casual.predict(X_casual_test)
mse_casual = mean_squared_error(y_casual_test, y_casual_pred)
print(f"Mean Squared Error for Casual: {mse_casual}")

y_registered_pred = model_registered.predict(X_registered_test)
mse_registered = mean_squared_error(y_registered_test, y_registered_pred)
print(f"Mean Squared Error for Registered: {mse_registered}")

# テストデータに対する予測
test_X = test.drop(['datetime'], axis=1)
test_predictions_casual = model_casual.predict(test_X)
test_predictions_registered = model_registered.predict(test_X)
print("Test Predictions for Casual:")
print(test_predictions_casual)
print("Test Predictions for Registered:")
print(test_predictions_registered)


# 特徴量の重要度を取得
feature_importances_casual = model_casual.feature_importances_
feature_importances_registered = model_registered.feature_importances_

features = X_casual.columns
importance_df_casual = pd.DataFrame({'Feature': features, 'Importance': feature_importances_casual})
importance_df_registered = pd.DataFrame({'Feature': features, 'Importance': feature_importances_registered})

print(importance_df_casual)
print(importance_df_registered)

importance_df_casual = importance_df_casual.sort_values(by='Importance', ascending=False)
importance_df_registered = importance_df_registered.sort_values(by='Importance', ascending=False)

# 特徴量の重要度をプロット
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.barh(importance_df_casual['Feature'], importance_df_casual['Importance'])
plt.title('Feature Importances for Casual')
plt.xlabel('Importance')
plt.ylabel('Feature')

plt.subplot(1, 2, 2)
plt.barh(importance_df_registered['Feature'], importance_df_registered['Importance'])
plt.title('Feature Importances for Registered')
plt.xlabel('Importance')
plt.ylabel('Feature')

plt.tight_layout()
plt.show()



# casualの特徴量とターゲット変数を定義
X_casual = train.drop(['count', 'casual', 'registered', 'datetime'], axis=1)
y_casual = train['casual']

# registeredの特徴量とターゲット変数を定義
X_registered = train.drop(['count', 'casual', 'registered', 'datetime'], axis=1)
y_registered = train['registered']

# 訓練データとテストデータに分割
X_casual_train, X_casual_test, y_casual_train, y_casual_test = train_test_split(X_casual, y_casual, test_size=0.2, random_state=42)
X_registered_train, X_registered_test, y_registered_train, y_registered_test = train_test_split(X_registered, y_registered, test_size=0.2, random_state=42)

# ランダムフォレストモデルの訓練
model_casual = RandomForestRegressor(n_estimators=100, random_state=42)
model_casual.fit(X_casual_train, y_casual_train)

model_registered = RandomForestRegressor(n_estimators=100, random_state=42)
model_registered.fit(X_registered_train, y_registered_train)

# 予測
y_casual_pred = model_casual.predict(X_casual_test)
y_registered_pred = model_registered.predict(X_registered_test)

# RMSLEの計算
log_casual_pred = np.log1p(y_casual_pred)
log_casual_true = np.log1p(y_casual_test)
rmsle_casual = np.sqrt(np.mean(np.square(log_casual_pred - log_casual_true)))
print(f"RMSLE for Casual: {rmsle_casual}")

log_registered_pred = np.log1p(y_registered_pred)
log_registered_true = np.log1p(y_registered_test)
rmsle_registered = np.sqrt(np.mean(np.square(log_registered_pred - log_registered_true)))
print(f"RMSLE for Registered: {rmsle_registered}")

# テストデータに対する予測
test_X = test.drop(['datetime'], axis=1)
test_predictions_casual = model_casual.predict(test_X)
test_predictions_registered = model_registered.predict(test_X)
print("Test Predictions for Casual:")
print(test_predictions_casual)
print("Test Predictions for Registered:")
print(test_predictions_registered)


#線形回帰
# モデルの訓練
model_linear_casual = LinearRegression()
model_linear_casual.fit(X_casual_train, y_casual_train)

model_linear_registered = LinearRegression()
model_linear_registered.fit(X_registered_train, y_registered_train)

# 予測
y_casual_pred_linear = model_linear_casual.predict(X_casual_test)
y_registered_pred_linear = model_linear_registered.predict(X_registered_test)

# RMSLEの計算
log_casual_pred_linear = np.log1p(y_casual_pred_linear)
log_casual_true = np.log1p(y_casual_test)
rmsle_casual_linear = np.sqrt(np.mean(np.square(log_casual_pred_linear - log_casual_true)))
print(f"RMSLE for Casual (Linear Regression): {rmsle_casual_linear}")

log_registered_pred_linear = np.log1p(y_registered_pred_linear)
log_registered_true = np.log1p(y_registered_test)
rmsle_registered_linear = np.sqrt(np.mean(np.square(log_registered_pred_linear - log_registered_true)))
print(f"RMSLE for Registered (Linear Regression): {rmsle_registered_linear}")


#勾配ブースティング
# モデルの訓練
model_gb_casual = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_gb_casual.fit(X_casual_train, y_casual_train)

model_gb_registered = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_gb_registered.fit(X_registered_train, y_registered_train)

# 予測
y_casual_pred_gb = model_gb_casual.predict(X_casual_test)
y_registered_pred_gb = model_gb_registered.predict(X_registered_test)

# RMSLEの計算
log_casual_pred_gb = np.log1p(y_casual_pred_gb)
log_casual_true = np.log1p(y_casual_test)
rmsle_casual_gb = np.sqrt(np.mean(np.square(log_casual_pred_gb - log_casual_true)))
print(f"RMSLE for Casual (Gradient Boosting): {rmsle_casual_gb}")

log_registered_pred_gb = np.log1p(y_registered_pred_gb)
log_registered_true = np.log1p(y_registered_test)
rmsle_registered_gb = np.sqrt(np.mean(np.square(log_registered_pred_gb - log_registered_true)))
print(f"RMSLE for Registered (Gradient Boosting): {rmsle_registered_gb}")



#サポートベクターマシン(SVM)
# モデルの訓練
model_svm_casual = SVR()
model_svm_casual.fit(X_casual_train, y_casual_train)

model_svm_registered = SVR()
model_svm_registered.fit(X_registered_train, y_registered_train)

# 予測
y_casual_pred_svm = model_svm_casual.predict(X_casual_test)
y_registered_pred_svm = model_svm_registered.predict(X_registered_test)

# RMSLEの計算
log_casual_pred_svm = np.log1p(y_casual_pred_svm)
log_casual_true = np.log1p(y_casual_test)
rmsle_casual_svm = np.sqrt(np.mean(np.square(log_casual_pred_svm - log_casual_true)))
print(f"RMSLE for Casual (SVM): {rmsle_casual_svm}")

log_registered_pred_svm = np.log1p(y_registered_pred_svm)
log_registered_true = np.log1p(y_registered_test)
rmsle_registered_svm = np.sqrt(np.mean(np.square(log_registered_pred_svm - log_registered_true)))
print(f"RMSLE for Registered (SVM): {rmsle_registered_svm}")



#k近傍法
# モデルの訓練
model_knn_casual = KNeighborsRegressor(n_neighbors=5)
model_knn_casual.fit(X_casual_train, y_casual_train)

model_knn_registered = KNeighborsRegressor(n_neighbors=5)
model_knn_registered.fit(X_registered_train, y_registered_train)

# 予測
y_casual_pred_knn = model_knn_casual.predict(X_casual_test)
y_registered_pred_knn = model_knn_registered.predict(X_registered_test)

# RMSLEの計算
log_casual_pred_knn = np.log1p(y_casual_pred_knn)
log_casual_true = np.log1p(y_casual_test)
rmsle_casual_knn = np.sqrt(np.mean(np.square(log_casual_pred_knn - log_casual_true)))
print(f"RMSLE for Casual (k-Nearest Neighbors): {rmsle_casual_knn}")

log_registered_pred_knn = np.log1p(y_registered_pred_knn)
log_registered_true = np.log1p(y_registered_test)
rmsle_registered_knn = np.sqrt(np.mean(np.square(log_registered_pred_knn - log_registered_true)))
print(f"RMSLE for Registered (k-Nearest Neighbors): {rmsle_registered_knn}")



# Kaggle提出用のデータフレーム作成
submission = pd.DataFrame({'datetime': test['datetime'], 'count': test_predictions_casual + test_predictions_registered})

# CSVファイルとして保存
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")

