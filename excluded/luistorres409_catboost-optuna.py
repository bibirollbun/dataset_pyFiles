# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


X_train = np.load('/kaggle/input/dt-intro/X_train.npy')
y_train = np.load('/kaggle/input/dt-intro/y_train.npy')
X_test = np.load('/kaggle/input/dt-intro/X_test.npy')


# Creating a dataframe with train values.

train_df = pd.DataFrame(X_train, columns=[f'feature_{i+1}' for i in range(25)])
test_df = pd.DataFrame(X_test, columns=[f'feature_{i+1}' for i in range(25)])
train_df


## Plotting the correlation matix

train_df['target'] = y_train

corr_matrix = train_df.corr()

plt.figure(figsize=(12, 8))  # Ajusta o tamanho do gráfico
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Matriz de Correlação das Features")
plt.show()


# Dropping some features because they are almost the same as other one

train_df = train_df.drop(['feature_11','feature_12','feature_13','feature_14','feature_15'],axis=1)
corr_matrix = train_df.corr()

plt.figure(figsize=(12, 8))  # Ajusta o tamanho do gráfico
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Matriz de Correlação das Features")
plt.show()


## Plotting the AutoCorrelation Function

from statsmodels.graphics.tsaplots import plot_acf


for feature in train_df.columns:

    plt.figure(figsize=(8, 5))
    plot_acf(train_df[feature], lags=500)  # Lags até 20 períodos
    plt.title(f'Função de Autocorrelação (ACF) {feature}')
    plt.show()


## Plotting the values distribuition

for feature in train_df.columns:

    
       # Criar um histograma com Seaborn
    sns.histplot(train_df[feature], bins=30, kde=True)  # kde=True adiciona a curva de densidade
    plt.title(f'Distribuição dos Valores {feature}')
    plt.xlabel('Valores')
    plt.ylabel('Frequência')
    plt.show()




features_cols = train_df.drop('target',axis=1).columns
features_cols


import xgboost as xgb

model = xgb.XGBRegressor(
    objective="reg:squarederror",  # Para regressão
    n_estimators=100,  # Número de árvores
    learning_rate=0.1,  # Taxa de aprendizado
    max_depth=10,  # Profundidade máxima da árvore
    random_state=42
)

model.fit(train_df.drop('target',axis=1), y_train)


predictions = model.predict(test_df[features_cols])


y_test_hat_pd = pd.DataFrame({
    'Id': list(range(len(X_test))),
    'Predicted': predictions.reshape(-1),
})
# Below is a small check that your output has the right type and shape
assert isinstance(y_test_hat_pd, pd.DataFrame)
assert all(y_test_hat_pd.columns == ['Id', 'Predicted'])
assert len(y_test_hat_pd) == 200

# If you pass the checks, the file is saved.
y_test_hat_pd.to_csv('y_test_hat.csv', index=False)


fig = plt.figure(figsize=(15,5))
plt.title('XGBoost Prediction')
plt.plot(predictions,label='Predictions')
plt.legend()
plt.show()


from catboost import CatBoostRegressor

model = CatBoostRegressor(iterations=100, 
                          depth=6, 
                          learning_rate=0.1, 
                          verbose=0, 
                          random_seed=42)

model.fit(train_df.drop('target',axis=1), y_train)


predictions = model.predict(test_df[features_cols])

y_test_hat_pd = pd.DataFrame({
    'Id': list(range(len(X_test))),
    'Predicted': predictions.reshape(-1),
})
# Below is a small check that your output has the right type and shape
assert isinstance(y_test_hat_pd, pd.DataFrame)
assert all(y_test_hat_pd.columns == ['Id', 'Predicted'])
assert len(y_test_hat_pd) == 200

# If you pass the checks, the file is saved.
y_test_hat_pd.to_csv('catboost_pred.csv', index=False)


fig = plt.figure(figsize=(15,5))
plt.title('Catboost Prediction')
plt.plot(predictions,label='Predictions')
plt.legend()
plt.show()


from sklearn.preprocessing import StandardScaler,MinMaxScaler

X_train = np.load('/kaggle/input/dt-intro/X_train.npy')
y_train = np.load('/kaggle/input/dt-intro/y_train.npy')
X_test = np.load('/kaggle/input/dt-intro/X_test.npy')

train_df = pd.DataFrame(X_train, columns=[f'feature_{i+1}' for i in range(25)])
test_df = pd.DataFrame(X_test, columns=[f'feature_{i+1}' for i in range(25)])

# Agora a linha abaixo não dará erro
train_df['target'] = y_train
train_df = train_df.drop(['feature_11','feature_12','feature_13','feature_14','feature_15'],axis=1)


train_df.corr()['target'].sort_values(ascending=True)


best_features = train_df.corr()['target'][train_df.corr()['target'] > 0.2].drop('target').index.tolist()
best_features


from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer


def feature_engineer(df,best_features):

    for feature in best_features:
    
        #df[f'{feature}_log'] = np.log(df[feature])
    
        #pt = PowerTransformer(method='yeo-johnson') 
        
        #df[f'{feature}_power_transform'] = pt.fit_transform(df[[feature]])
    
        qt = QuantileTransformer(output_distribution='normal')
        df[f'{feature}_quantile'] = qt.fit_transform(df[[feature]])
        

    return df


train_df = feature_engineer(train_df,best_features)
test_df = feature_engineer(test_df,best_features)


corr_matrix = train_df.corr()

plt.figure(figsize=(12, 8))  # Ajusta o tamanho do gráfico
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Matriz de Correlação das Features")
plt.show()


for feature in train_df.columns:

    
       # Criar um histograma com Seaborn
    sns.histplot(train_df[feature], bins=30, kde=True)  # kde=True adiciona a curva de densidade
    plt.title(f'Distribuição dos Valores {feature}')
    plt.xlabel('Valores')
    plt.ylabel('Frequência')
    plt.show()


# Seleciona apenas a correlação do target com as features
target_corr = train_df.corr()['target'].drop('target')  # Remove a autocorrelação (1.0)

# Plota um heatmap pequeno
plt.figure(figsize=(30, 5))
sns.heatmap(target_corr.to_frame().T, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlação das Features com o Target")
plt.show()


features_cols = train_df.drop('target',axis=1).columns
features_cols


from catboost import CatBoostRegressor

model = CatBoostRegressor(iterations=100, 
                          depth=6, 
                          learning_rate=0.1, 
                          verbose=0, 
                          random_seed=42)

model.fit(train_df.drop('target',axis=1), y_train)


pred = model.predict(test_df[features_cols])


predictions = model.predict(test_df[features_cols])

y_test_hat_pd = pd.DataFrame({
    'Id': list(range(len(X_test))),
    'Predicted': predictions.reshape(-1),
})
# Below is a small check that your output has the right type and shape
assert isinstance(y_test_hat_pd, pd.DataFrame)
assert all(y_test_hat_pd.columns == ['Id', 'Predicted'])
assert len(y_test_hat_pd) == 200

# If you pass the checks, the file is saved.
y_test_hat_pd.to_csv('catboost_pred_scaled.csv', index=False)


fig = plt.figure(figsize=(15,5))
plt.title('Catboost Prediction FE')
plt.plot(predictions,label='Predictions')
plt.legend()
plt.show()


from sklearn.feature_selection import mutual_info_regression

mi = mutual_info_regression(train_df, y_train)
mi_df = pd.DataFrame({'Feature': train_df.columns, 'MI': mi})
mi_df = mi_df.sort_values(by="MI", ascending=False)
mi_df


from sklearn.preprocessing import StandardScaler,MinMaxScaler
import catboost
import xgboost

X_train = np.load('/kaggle/input/dt-intro/X_train.npy')
y_train = np.load('/kaggle/input/dt-intro/y_train.npy')
X_test = np.load('/kaggle/input/dt-intro/X_test.npy')

train_df = pd.DataFrame(X_train, columns=[f'feature_{i+1}' for i in range(25)])
test_df = pd.DataFrame(X_test, columns=[f'feature_{i+1}' for i in range(25)])

# Agora a linha abaixo não dará erro
train_df['target'] = y_train
train_df = train_df.drop(['feature_11','feature_12','feature_13','feature_14','feature_15'],axis=1)


X = train_df.drop('target',axis=1).values
y = y_train


X


from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)



model = catboost.CatBoostRegressor(iterations=100, 
                          depth=6, 
                          learning_rate=0.1, 
                          verbose=0, 
                          random_seed=42)

model.fit(X_train, y_train)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pred = model.predict(X_test)

score =  mean_squared_error(y_test,pred)
score


def objective(trial):
    global model
   
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000, step=50),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_seed': 42
    }

    # Criar modelo CatBoost
    model = catboost.CatBoostRegressor(**params, verbose=0)

    model.fit(X_train,y_train)

    pred =  model.predict(X_test)

    score = mean_squared_error(y_test,pred)

    return score

def callback(study,trial):
    global best_model
    if study.best_trial == trial:
        best_model = model
    


import optuna

study = optuna.create_study(direction = "minimize")

## Otimização do Study Case
study.optimize(objective, n_trials = 1000, callbacks = [callback])


features_cols = train_df.drop('target',axis=1).columns


predictions = best_model.predict(test_df[features_cols])

y_test_hat_pd = pd.DataFrame({
    'Id': list(range(len(test_df))),
    'Predicted': predictions.reshape(-1),
})
# Below is a small check that your output has the right type and shape
assert isinstance(y_test_hat_pd, pd.DataFrame)
assert all(y_test_hat_pd.columns == ['Id', 'Predicted'])
assert len(y_test_hat_pd) == 200

# If you pass the checks, the file is saved.
y_test_hat_pd.to_csv('catboost_pred_best_optuna.csv', index=False)

