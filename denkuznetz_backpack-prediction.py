import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import norm
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")


df_test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


df


df.duplicated().sum()


df.isnull().sum()


df.info()


df.describe()


df['Brand'].value_counts()


df = df.drop('id', axis = 1)
df_test = df_test.drop('id', axis = 1)


df


fig, axes = plt.subplots(2, 2, figsize=(15, 12))

sns.boxplot(x=df['Price'], ax=axes[0, 0])
axes[0, 0].set_title('Price')

sns.boxplot(x=df['Weight Capacity (kg)'], ax=axes[0, 1])
axes[0, 1].set_title('Weight Capacity (kg)')

sns.boxplot(x=df['Compartments'], ax=axes[1, 0])
axes[1, 0].set_title('Compartments')

plt.tight_layout()
plt.show()


df.isnull().sum()


categorical_features = ['Brand', 'Material', 'Style', 'Color', 'Laptop Compartment', 'Waterproof', 'Size']
df[categorical_features] = df[categorical_features].fillna('Unknown')


df.fillna({'Weight Capacity (kg)': df['Weight Capacity (kg)'].median()}, inplace=True)


df.isnull().sum()


df = pd.get_dummies(df, columns=['Brand', 'Material', 'Size', 'Laptop Compartment', 
                                  'Waterproof', 'Style', 'Color'], drop_first=True, dtype=int)

feature_columns = df.columns  


df_test = pd.get_dummies(df_test, columns=['Brand', 'Material', 'Size', 'Laptop Compartment', 
                                           'Waterproof', 'Style', 'Color'], drop_first=True, dtype=int)

missing_cols = set(feature_columns) - set(df_test.columns)
for col in missing_cols:
    df_test[col] = 0  

df_test = df_test[feature_columns]


df.head(8)


df_test.head(8)


df_test = df_test.drop('Price', axis = 1)


X__train = df.drop(['Price'], axis = 1)
y__train = df['Price']


feature_names = X__train.columns


X_train, X_test, y_train, y_test = train_test_split(X__train, y__train, test_size = 0.35, random_state=200) 


X__test = df_test


xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror')

# Запишем необходимые нам параметры для дальнейшего перебора.
param_xgb = {
    'n_estimators': [50, 100, 200],  
    'learning_rate': [0.1, 0.2],  
    'max_depth': [3, 5, 10],
    'subsample': [0.8, 1.0],
    'min_child_weight': [1, 3, 5]    
}

# Сделаем перебор заданных выше параметров, при этом разделив выборку данных на 5 частей.
grid_search__xgb = GridSearchCV(xgb_regressor, param_xgb, cv=5)

# Обучим модель на тренировочных данных
grid_search__xgb.fit(X_train, y_train)


grid_search__xgb.best_params_


best_gs_xgb_two = grid_search__xgb.best_estimator_


print('Score on train data = ', round(best_gs_xgb_two.score(X_train, y_train), 4))
print('Score on test data = ', round(best_gs_xgb_two.score(X_test, y_test), 4))










