# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train_data = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
train_data.head()


print(train_data.shape)
print(test_data.shape)


import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

target = train_data.accident_risk

plt.figure(figsize=(6,4))
sns.histplot(target, kde=True, bins=30, color='skyblue')
plt.xlabel("accident_risk")
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(6,4))
sns.boxplot(target, color='skyblue')
plt.ylabel("accident_risk")
plt.show()


train_data.select_dtypes(include=["number"]).drop("accident_risk", axis=1).columns


train_data.select_dtypes(include=["number"]).drop("accident_risk", axis=1).describe().round(decimals=2)


num_attributes = train_data.select_dtypes(include=["number"]).drop("accident_risk", axis=1).copy()

for col in num_attributes.columns:
    plt.figure(figsize=(6,4))
    sns.histplot(num_attributes[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


plt.figure(figsize=(6,4))
sns.boxplot(num_attributes["curvature"], color='skyblue')
plt.ylabel("curvature")
plt.show()


sns.boxplot(x='num_lanes', y='accident_risk', data=train_data)


sns.boxplot(x='speed_limit', y='accident_risk', data=train_data)


sns.boxplot(x='num_reported_accidents', y='accident_risk', data=train_data)


num_attributes = train_data.select_dtypes(include=["number"]).copy()
corr_matrix = num_attributes.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Numerical Variables')
plt.show()


train_data.select_dtypes(exclude=["number"]).columns


cat_attributes = train_data.select_dtypes(exclude=["number"]).copy()

for col in cat_attributes.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=col, data=train_data)
    plt.title(f'Count of categories in {col}')
    plt.xticks(rotation=45)  # para etiquetas largas
    plt.show()


for col in cat_attributes.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=col, y='accident_risk', data=train_data)
    plt.title(f'Boxplot of accident_risk by {col}')
    plt.xticks(rotation=45)  # rotar etiquetas si es necesario
    plt.show()


from scipy.stats import spearmanr

target_name = 'accident_risk'

results = []

for col in cat_attributes.columns:
    # 1️⃣ Promedio del target por categoría
    means = train_data.groupby(col)[target_name].mean()
    max_mean_diff = means.max() - means.min()  # cuanto más grande, más relación
    
    # 2️⃣ Spearman correlation tras codificar categorías
    codes = train_data[col].astype('category').cat.codes
    corr, _ = spearmanr(codes, train_data[target_name])
    
    results.append({
        'feature': col,
        'max_mean_diff': max_mean_diff,
        'spearman_corr': corr
    })

correlation_df = pd.DataFrame(results)
correlation_df = correlation_df.sort_values(by='max_mean_diff', ascending=False)

print(correlation_df)


for col in cat_attributes.columns:
    plt.figure(figsize=(8, 5))
    sns.pointplot(x=col, y=target_name, data=train_data, ci=95)
    plt.title(f'Mean accident_risk by {col} (with 95% CI)')
    plt.xticks(rotation=45)  
    plt.show()


irrelevant_cols = ["id", 'num_lanes', 'public_road', 'road_type', "time_of_day", "school_season", "road_signs_present"]
train_data_copy = train_data.drop(columns=irrelevant_cols)

train_data_copy.isna().sum().sort_values(ascending=False).head(10)


train_data_copy = train_data_copy[
    ~((train_data_copy['speed_limit'] <= 45) & (train_data_copy['accident_risk'] > 0.9)) &
    ~((train_data_copy['speed_limit'] > 60) & (train_data_copy['accident_risk'] < 0.05))
]
train_data_copy = train_data_copy[
    ~((train_data_copy['num_reported_accidents'] < 3) & (train_data_copy['accident_risk'] > 0.95))
]
train_data_copy = train_data_copy[
    ~(((train_data_copy['lighting'] == 'daylight') | (train_data_copy['lighting'] == 'dim')) & 
      (train_data_copy['accident_risk'] > 0.9)) &
    ~((train_data_copy['lighting'] == 'night') & (train_data_copy['accident_risk'] < 0.05))
]
print("Filtered size:", train_data_copy.shape)


from sklearn.model_selection import train_test_split

X = train_data_copy.drop(["accident_risk"], axis=1)
X = pd.get_dummies(X)

y = train_data_copy.accident_risk

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=243)


train_data_copy.head()


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Tomamos una muestra para entrenar rápido
train_sample = train_data_copy.sample(n=50000, random_state=1)
X_sample = pd.get_dummies(train_sample.drop("accident_risk", axis=1))
y_sample = train_sample["accident_risk"]

# Definir modelos optimizados
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'DecisionTree': DecisionTreeRegressor(max_depth=10),
    'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=1),
    'AdaBoost': AdaBoostRegressor(n_estimators=50, random_state=1),
    'XGBoost': XGBRegressor(n_estimators=50, max_depth=5, n_jobs=-1, objective='reg:squarederror', random_state=1)
}

results = {}

for name, model in models.items():
    model.fit(X_sample, y_sample)
    preds = model.predict(X_sample)
    mse = mean_squared_error(y_sample, preds)
    results[name] = mse

# Mostrar resultados ordenados por mejor MSE
results_sorted = dict(sorted(results.items(), key=lambda x: x[1]))
for model_name, mse in results_sorted.items():
    print(f"{model_name}: MSE = {mse:.6f}")


from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

xgb = XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=1)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 7],
    'learning_rate': [0.05, 0.1]
}

grid_search = GridSearchCV(xgb, param_grid, scoring='neg_mean_squared_error', cv=3, verbose=0)
grid_search.fit(train_X, train_y)

print("Best params:", grid_search.best_params_)
print("Best MSE:", -grid_search.best_score_)


best_xgb = grid_search.best_estimator_


test_data_copy = test_data.copy()

irrelevant_cols = ["id", 'num_lanes', 'public_road', 'road_type', "time_of_day", "school_season", "road_signs_present"]
test_data_copy = test_data_copy.drop(columns=irrelevant_cols)


test_data_copy.isna().sum().sort_values(ascending=False).head(10)


test_X = pd.get_dummies(test_data_copy)

test_preds = best_xgb.predict(test_X)

submission = pd.DataFrame({
    'id': test_data['id'],
    'accident_risk': test_preds
})

submission.to_csv('submission.csv', index=False)

print(submission.head())

