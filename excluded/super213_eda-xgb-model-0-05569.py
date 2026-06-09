import pandas as pd

df_train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df_train.head()


df_train.info()


df_train.isnull().sum()


from scipy import stats
import numpy as np

for col in df_train.columns:
    if df_train[col].dtype != 'object':
        z_scores = np.abs(stats.zscore(df_train[col].dropna()))
        outliers = df_train[z_scores > 3]
        print(f"列 {col} 中的异常值数量: {len(outliers)}")


import matplotlib.pyplot as plt
plt.boxplot(df_train['num_reported_accidents'])
plt.title('num_reported_accidents')
plt.show()


from scipy import stats
import seaborn as sns

for col in df_train.columns:
    if df_train[col].dtype != 'object':
        plt.figure(figsize=(12, 4))
        
        # 直方图 + KDE
        plt.subplot(1, 2, 1)
        sns.histplot(df_train[col].dropna(), kde=True, bins=30)
        plt.title(f'{col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        
        # QQ图
        plt.subplot(1, 2, 2)
        stats.probplot(df_train[col].dropna(), dist="norm", plot=plt)
        plt.title(f'{col} QQ plot')
        
        plt.tight_layout()
        plt.show()


df_num = df_train.select_dtypes(include=['int64', 'float64', 'bool'])
sns.heatmap(df_num.corr(), annot=True, cmap='YlGnBu')
plt.show()


sns.boxplot(
    x='speed_limit',
    y='accident_risk',
    data=df_train
)
plt.title('speed_limit vs accident_risk')
plt.show()


plt.hexbin(df_train['curvature'], df_train['accident_risk'], gridsize=30, cmap='viridis')
plt.xlabel('Curvature')
plt.ylabel('Accident Risk')
plt.title('Curvature vs Accident Risk')
plt.colorbar(label='Density')
plt.show()


df_train['curvature_bin'] = pd.qcut(df_train['curvature'], q=5)
sns.boxplot(x='curvature_bin', y='accident_risk', data=df_train)
plt.xticks(rotation=45)
plt.title('Accident Risk Distribution by Curvature Bins')
plt.show()


sns.boxplot(x='num_reported_accidents', y='accident_risk', data=df_train)
plt.xticks(rotation=45)
plt.title('num_reported_accidents VS accident_risk')
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(10,8))
sns.boxplot(x='road_type', y='accident_risk', data=df_train, ax=axes[0,0])
sns.boxplot(x='lighting', y='accident_risk', data=df_train, ax=axes[0,1])
sns.boxplot(x='weather', y='accident_risk', data=df_train, ax=axes[1,0])
sns.boxplot(x='time_of_day', y='accident_risk', data=df_train, ax=axes[1,1])
plt.tight_layout()
plt.show()


df_obj = df_train.select_dtypes(include=['object'])
df_obj.columns


for col in df_obj.columns:
    print(f"列名：{col}")
    print(df_obj[col].unique())
    print("\n")


df_train = pd.get_dummies(df_train, columns=['road_type'], drop_first=False)


lighting_map = {'daylight': 0, 'dim': 1, 'night': 2}
weather_map = {'clear': 0, 'foggy': 1, 'rainy': 2}

df_train['lighting'] = df_train['lighting'].map(lighting_map)
df_train['weather'] = df_train['weather'].map(weather_map)


time_map = {'morning': 0, 'afternoon': 1, 'evening': 2}
df_train['time_code'] = df_train['time_of_day'].map(time_map)

import numpy as np
df_train['time_sin'] = np.sin(2 * np.pi * df_train['time_code'] / 3)


df_train.head()


df_train = df_train.drop(columns=['id', 'time_of_day', 'curvature_bin', 'time_code'])


from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

models = {
    'XGBoost': XGBRegressor(random_state=42),
    'LightGBM': LGBMRegressor(random_state=42)
}


X = df_train.drop(columns=['accident_risk'])
y = df_train['accident_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} 模型得分: {r2_score(y_test, y_pred)}")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{name} 模型 RMSE: {rmse}")



df_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


# one-hot
df_test = pd.get_dummies(df_test, columns=['road_type'], drop_first=False)

# label
lighting_map = {'daylight': 0, 'dim': 1, 'night': 2}
weather_map = {'clear': 0, 'foggy': 1, 'rainy': 2}

df_test['lighting'] = df_test['lighting'].map(lighting_map)
df_test['weather'] = df_test['weather'].map(weather_map)

# sin
time_map = {'morning': 0, 'afternoon': 1, 'evening': 2}
df_test['time_code'] = df_test['time_of_day'].map(time_map)

import numpy as np
df_test['time_sin'] = np.sin(2 * np.pi * df_test['time_code'] / 3)


ids = df_test['id']
df_test = df_test.drop(columns=['id', 'time_of_day', 'time_code'])


XGBoost = models['XGBoost']
y_pred_test = XGBoost.predict(df_test)

submission = pd.DataFrame({
    'id': ids,
    'accident_risk': y_pred_test
})

submission.to_csv('submission.csv', index=False)




