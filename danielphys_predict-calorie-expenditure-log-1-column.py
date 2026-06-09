# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import entropy

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#import xgboost as xgb
#import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


train_df.info()
print('--------------------')
test_df.info()


train_df.hist(bins=70, figsize=(30,30))


train_df['IMC'] = train_df['Weight']/((train_df['Height']/100)**2)
train_df['DHR'] = train_df['Duration']*train_df['Heart_Rate']
train_df.head()


for col in train_df.columns:
    if col != "Sex":
        skewness = skew(train_df[col].dropna()) # Manejo de NaN
        print(f"Skewness of {col}: {skewness:.4f}")
        plt.figure(figsize=(8, 6))  # Crea una nueva figura para cada histograma
        train_df[col].hist(alpha=0.5, label=col)
        plt.xlabel(col)
        plt.ylabel("Frecuencia")
        plt.title(f"Histograma de {col}")
        plt.legend()
        plt.show()


for col in train_df.columns:
    if col != "Sex":
        kurt = kurtosis(train_df[col].dropna(), fisher=True) # Manejo de NaN
        print(f"Kurtosis of {col}: {kurt:.4f}")

# Visualizar solo las columnas deseadas
columnas_a_visualizar = [col for col in train_df.columns if col != "Sex"]
train_df[columnas_a_visualizar].plot(kind='kde', subplots=True, figsize=(10, 5 * len(columnas_a_visualizar)), sharex=False)
plt.suptitle("Kernel Density Estimates")
plt.tight_layout()
plt.show()


def calculate_entropy_continuous(data_series, num_bins=20):
    """Estimates entropy of a continuous variable by discretizing it."""
    hist, bin_edges = np.histogram(data_series, bins=num_bins, density=True)
    pk = hist / np.sum(hist)  # Normalize histogram to get probabilities
    return entropy(pk)

for col in train_df.columns:
    if col != "Sex":
        try:
            entropy_value = calculate_entropy_continuous(train_df[col].dropna()) # Manejo de NaN
            print(f"Entropy of {col}: {entropy_value:.4f}")
        except TypeError as e:
            print(f"Skipping column '{col}': {e}")


transform_columns = ['Age', 'Height','Weight','Duration','Heart_Rate','Body_Temp','Calories','DHR','IMC']
train_df[transform_columns] = train_df[transform_columns].apply(lambda x: np.log1p(x))


train_df.hist(bins=70, figsize=(30,30))


for col in train_df.columns:
    if col != "Sex":
        skewness = skew(train_df[col].dropna()) # Manejo de NaN
        print(f"Skewness of {col}: {skewness:.4f}")
        plt.figure(figsize=(8, 6))  # Crea una nueva figura para cada histograma
        train_df[col].hist(alpha=0.5, label=col)
        plt.xlabel(col)
        plt.ylabel("Frecuencia")
        plt.title(f"Histograma de {col}")
        plt.legend()
        plt.show()


for col in train_df.columns:
    if col != "Sex":
        kurt = kurtosis(train_df[col].dropna(), fisher=True) # Manejo de NaN
        print(f"Kurtosis of {col}: {kurt:.4f}")

# Visualizar solo las columnas deseadas
columnas_a_visualizar = [col for col in train_df.columns if col != "Sex"]
train_df[columnas_a_visualizar].plot(kind='kde', subplots=True, figsize=(10, 5 * len(columnas_a_visualizar)), sharex=False)
plt.suptitle("Kernel Density Estimates")
plt.tight_layout()
plt.show()


def calculate_entropy_continuous(data_series, num_bins=20):
    """Estimates entropy of a continuous variable by discretizing it."""
    hist, bin_edges = np.histogram(data_series, bins=num_bins, density=True)
    pk = hist / np.sum(hist)  # Normalize histogram to get probabilities
    return entropy(pk)

for col in train_df.columns:
    if col != "Sex":
        try:
            entropy_value = calculate_entropy_continuous(train_df[col].dropna()) # Manejo de NaN
            print(f"Entropy of {col}: {entropy_value:.4f}")
        except TypeError as e:
            print(f"Skipping column '{col}': {e}")


train_df_encoded = pd.get_dummies(train_df, columns=['Sex'])
train_df_encoded.head()


# Calcular la matriz de correlación
correlation_matrix = train_df_encoded.corr(numeric_only=True)

# Visualizar la matriz de correlación con un mapa de calor
plt.figure(figsize=(10, 8))  # Ajusta el tamaño de la figura según necesites
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Mapa de Calor de la Matriz de Correlación entre Columnas')
plt.show()


columns = ['id']
train_df_useful = train_df_encoded.drop(columns, axis=1)


Y_train = train_df_useful["Calories"]
X_train = train_df_useful.drop(['Calories'], axis=1)


x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=42)


#=========================================================================
# Catboost regression: 
# Parameters: 
# iterations  "Number of gradient boosted trees. Equivalent to number 
#                of boosting rounds."
# learning_rate "Boosting learning rate (also known as “eta”)"
# depth     "Maximum depth of a tree. Increasing this value will make 
#                the model more complex and more likely to overfit." 
#=========================================================================
regressor=CatBoostRegressor(loss_function='RMSE',verbose=0, task_type='GPU')

#=========================================================================
# exhaustively search for the optimal hyperparameters
#=========================================================================
from sklearn.model_selection import GridSearchCV
# set up our search grid
param_grid = {
    "depth":    [12,13,14],
    "iterations": [400,500,600],
    "learning_rate": [0.04, 0.05,0.06],
    "l2_leaf_reg": [1,2,3]}

# try out every combination of the above values
search = GridSearchCV(regressor, param_grid, cv=5).fit(x_train, y_train)

print("The best hyperparameters are ",search.best_params_)


Cat_Boost_Regressor = CatBoostRegressor(
    l2_leaf_reg = 2,
    learning_rate = 0.05,
    iterations  = 500,
    depth     = 13,
    verbose=0,
    task_type='GPU')


Cat_Boost_Regressor = CatBoostRegressor(
    l2_leaf_reg = search.best_params_["l2_leaf_reg"],
    learning_rate = search.best_params_["learning_rate"],
    iterations  = search.best_params_["iterations"],
    depth     = search.best_params_["depth"],
    verbose=0,
    task_type='GPU')


Cat_Boost_Regressor = CatBoostRegressor(
    l2_leaf_reg = 1,
    learning_rate = 0.04,
    iterations  = 600,
    depth     = 12,
    verbose=0,
    task_type='GPU')


Cat_Boost_Regressor.fit(x_train, y_train)


Predict_Calories_Logp1 = Cat_Boost_Regressor.predict(x_test)


Cat_Boost_Regressor_RMSE = mean_squared_error(y_test, Predict_Calories_Logp1)
print('Cat Boost Regressor RMSE: ', Cat_Boost_Regressor_RMSE)


plt.scatter(y_test, Predict_Calories_Logp1)
plt.xlabel("Real Values")
plt.ylabel("Predictions")
plt.show()


#=========================================================================
# XGBoost regression: 
# Parameters: 
# n_estimators  "Number of gradient boosted trees. Equivalent to number 
#                of boosting rounds."
# learning_rate "Boosting learning rate (also known as “eta”)"
# max_depth     "Maximum depth of a tree. Increasing this value will make 
#                the model more complex and more likely to overfit." 
#=========================================================================
regressor=xgb.XGBRegressor(eval_metric='rmse', device='cuda')

#=========================================================================
# exhaustively search for the optimal hyperparameters
#=========================================================================
from sklearn.model_selection import GridSearchCV
# set up our search grid
param_grid = {"max_depth":    [3,4, 5],
              "n_estimators": [650,700, 750],
              "learning_rate": [0.049, 0.05,0.051]}

# try out every combination of the above values
search = GridSearchCV(regressor, param_grid, cv=5).fit(x_train, y_train)

print("The best hyperparameters are ",search.best_params_)


XGB_Regressor = xgb.XGBRegressor(learning_rate = search.best_params_["learning_rate"],
                           n_estimators  = search.best_params_["n_estimators"],
                           max_depth     = search.best_params_["max_depth"],
                           eval_metric='rmse',
                            device='cuda')


XGB_Regressor = xgb.XGBRegressor(learning_rate = 0.04,
                           n_estimators  = 600,
                           max_depth     = 4,
                           eval_metric='rmse',
                            device='cuda')


XGB_Regressor.fit(x_train, y_train)


Predict_Calories_Logp1 = XGB_Regressor.predict(x_test)


XGB_Regressor_RMSE = mean_squared_error(y_test, Predict_Calories_Logp1)
print("XGBoost Regressor RMSE:", XGB_Regressor_RMSE)


plt.scatter(y_test, Predict_Calories_Logp1)
plt.xlabel("Real Values")
plt.ylabel("Predictions")
plt.show()


test_df['IMC'] = test_df['Weight']/((test_df['Height']/100)**2)
test_df['DHR'] = test_df['Duration']*test_df['Heart_Rate']
test_df.head()


test_df.hist(bins=70, figsize=(30,30))


test_df = pd.get_dummies(test_df, columns=['Sex'])
test_df.head()


columnas_a_transformar = ['Age', 'Height','Weight','Duration','Heart_Rate','Body_Temp','DHR','IMC']
test_df[columnas_a_transformar] = test_df[columnas_a_transformar].apply(lambda x: np.log1p(x))


columns = ['id']
test_df_useful = test_df.drop(columns, axis=1)


prediction_Y=np.expm1(Cat_Boost_Regressor.predict(test_df_useful))


prediction_df = test_df.assign(Calories=prediction_Y)
prediction_df.head()


Delete_columns = ['Sex_female','Sex_male','Age','Height', 'Weight', 'Duration','Heart_Rate','Body_Temp','DHR','IMC']
Submission = prediction_df.drop(Delete_columns, axis=1)
Submission.head()


negativos_df = Submission[Submission['Calories'] < 0]
print(negativos_df)


Submission['Calories'] = Submission['Calories'].abs()


Submission.to_csv('Prediction_Catboost_Regressor_Log1p_10.csv', index=False)


prediction_Y=np.expm1(XGB_Regressor.predict(test_df_useful))


prediction_df = test_df.assign(Calories=prediction_Y)
prediction_df.head()


Delete_columns = ['Sex_female','Sex_male','Age','Height', 'Weight', 'Duration','Heart_Rate','Body_Temp']
Submission = prediction_df.drop(Delete_columns, axis=1)
Submission.head()


negativos_df = Submission[Submission['Calories'] < 0]
print(negativos_df)


Submission['Calories'] = Submission['Calories'].abs()


Submission.to_csv('Prediction_XGB_Regressor_Log1p_4.csv', index=False)




