# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
"""
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#pip install holidays


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor



import holidays


train.head()


train.describe()


#train1.groupby(['num_sold']).value_counts()


train.isna().sum()


train['date']


ca_holidays = holidays.country_holidays('CA') # Canada
fi_holidays = holidays.country_holidays('FI') # Finland
it_holidays = holidays.country_holidays('IT') # Italy
ke_holidays = holidays.country_holidays('KE') # Kenya
no_holidays = holidays.country_holidays('NO') # Norway
sg_holidays = holidays.country_holidays('SG') # Singapore


def set_holiday(row):
    VAL_HOLIDAY = 999
    if row["country"] == "Canada" and row["date"] in ca_holidays:
        row["holiday"] = VAL_HOLIDAY

    elif row["country"] == "Finland" and row["date"] in fi_holidays:
        row["holiday"] = VAL_HOLIDAY

    elif row["country"] == "Italy" and row["date"] in it_holidays:
        row["holiday"] = VAL_HOLIDAY

    elif row["country"] == "Kenya" and row["date"] in ke_holidays:
        row["holiday"] = VAL_HOLIDAY


    elif row["country"] == "Norway" and row["date"] in no_holidays:
        row["holiday"] = VAL_HOLIDAY

    elif row["country"] == "Singapore" and row["date"] in sg_holidays:
        row["holiday"] = VAL_HOLIDAY
    else:
        row['holiday'] = 0

    return row





train = train.apply(set_holiday, axis=1)
test = test.apply(set_holiday, axis=1)


train.isna().sum()


train


# Import date
from datetime import date

def extract_day_month_year(dataframe, column):
    dataframe[column] = pd.to_datetime(dataframe[column])
    dataframe[column+'_day'] = dataframe[column].dt.day
    dataframe[column+'_month'] = dataframe[column].dt.month
    dataframe[column+'_year'] = dataframe[column].dt.year
    dataframe.drop(columns=[column], axis=1, inplace=True)
    


extract_day_month_year(train, 'date')


extract_day_month_year(test, 'date')


train.columns


test


train.dtypes


train['country'].value_counts()


train['product'].value_counts()


train.count()


# # Identificar colunas categóricas automaticamente (opcional)
# categorical_columns = df.select_dtypes(include=['object', 'category']).columns

# # Aplicar get_dummies nas colunas categóricas
# df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=False)  # drop_first=True para evitar colinearidade



# id              int64
# country        object
# store          object
# product        object
# num_sold      float64
# date_day        int32
# date_month      int32
# date_year       int32
# dtype: object

def convert_columns_category_onehotencoder(dataframe):
    # Identificar colunas categóricas automaticamente (opcional)
    categorical_columns = dataframe.select_dtypes(include=['object','category']).columns
    
    # Aplicar get_dummies nas colunas categóricas
    dataframe_encoded = pd.get_dummies(dataframe, columns=categorical_columns, drop_first=False)
    return dataframe_encoded
    


train_encoded = convert_columns_category_onehotencoder(train)


test_encoded = convert_columns_category_onehotencoder(test)


test_encoded


#experiment=pd.get_dummies(train_encoded, columns=['date_month','date_year'], drop_first=False)


test_encoded['country_Canada']


test_encoded.isna().sum()


train[train['num_sold'].isna() ]


# Aplicar o filtro corrigido
train_filtered = train_encoded.loc[
    train_encoded['num_sold'].isna() != True
]

print("Resultado:")
print(train_filtered)


train_filtered


sns.heatmap(train_filtered.corr())


train_filtered.dtypes


#num_columns = train_filtered.select_dtypes(include=['int64', 'float64']).columns
num_columns = train_filtered.columns


"""
#sns.boxplot(train_filtered['date_day'])

# Criar o grid de plots
fig, axes = plt.subplots(nrows=len(num_columns), ncols=1, figsize=(8, len(num_columns) * 4))

# Plotar cada boxplot
for i, col in enumerate(num_columns):
    sns.boxplot(data=train_filtered, x=col, ax=axes[i])
    axes[i].set_title(f'Boxplot for {col}')

plt.tight_layout()
plt.show()
"""



#pd.set_option('display.max_rows', None)


#train_filtered[['date_year','date_month','num_sold']].groupby(['date_year','date_month']).sum()


# Agrupar os dados
grouped_data = train_filtered[['date_year', 'date_month', 'num_sold']].groupby(['date_year', 'date_month']).sum().reset_index()

# Criar o gráfico
plt.figure(figsize=(12, 6))
sns.barplot(
    data=grouped_data, 
    x='date_month', 
    y='num_sold', 
    hue='date_year',  # Diferenciar por ano
    palette='tab10'
)

# Personalização do gráfico
plt.title("Número de Vendas por Mês e Ano", fontsize=16)
plt.xlabel("Mês", fontsize=12)
plt.ylabel("Número de Vendas (num_sold)", fontsize=12)
plt.legend(title="Ano")
plt.xticks(ticks=range(len(grouped_data['date_month'].unique())), labels=range(1, 13))

plt.tight_layout()
plt.show()


train_filtered['date_month']


train_filtered[ train_filtered['date_year'] == 2010 ]['num_sold']


train_filtered[ train_filtered['date_year'] == 2010  ]['num_sold']


train_filtered.columns


train_filtered.isna().sum()


train_filtered['holiday']


colunms = [
'date_year',
    'date_month',
    'date_day',
'country_Norway',
'product_Kaggle',
'product_Holographic Goose',
'store_Premium Sticker Mart',
'product_Kaggle Tiers',    
    'num_sold'
]

colunms


#train_filtered.groupby(['date_year','country_Norway'])['num_sold'].count()


#train_filtered = train_filtered[colunms]


#continue...


def train_and_evaluate(models, X, y, cv=5):
    """
    Treina e avalia modelos de machine learning usando a métrica mean_absolute_percentage_error (MAPE).
    
    Args:
        models (dict): Dicionário de modelos com nomes como chave e instâncias como valores.
        X (pd.DataFrame): Conjunto de features.
        y (pd.Series or np.ndarray): Conjunto de targets.
        cv (int): Número de folds para validação cruzada (default: 5).
    
    Returns:
        pd.DataFrame: DataFrame com os resultados de validação cruzada para cada modelo.
    """
    # Criar scorer personalizado para o MAPE
    mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
    
    results = []
    for name, model in models.items():
        print(f"Treinando e avaliando o modelo: {name}")
        # Realizar validação cruzada
        scores = cross_val_score(model, X, y, cv=cv, scoring=mape_scorer)
        
        # Salvar resultados
        results.append({
            'Model': name,
            'MAPE': scores,
            'Mean MAPE': -scores.mean(),  # Negativo porque o scorer retorna valores negativos
            'Std Dev MAPE': scores.std()
        })
    
    return model, pd.DataFrame(results)



from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import SGDRegressor
from sklearn import linear_model


from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor 
from sklearn.neural_network import MLPRegressor


# # Dicionário de modelos
# # models = {
# #     "Linear Regression": LinearRegression(),
# #     "Ridge Regression": Ridge(alpha=1.0),
# #     "Lasso Regression": Lasso(alpha=0.1),
# #     "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
# #     "Decision Tree": DecisionTreeRegressor(random_state=42)
# # }




xgb_params = {
    'n_estimators': 1078, 
    'learning_rate': 0.016084079332671603, 
    'max_depth': 10, 
    'min_child_weight': 8, 
    'subsample': 0.8732132237392727, 
    'colsample_bytree': 0.9756972730817159, 
    'reg_alpha': 3.386299962300141, 
    'reg_lambda': 8.964009483088061,
    'enable_categorical': 'True',
    'device': 'cuda'
}

# models = {
#     "SGDRegressor":SGDRegressor(max_iter=10000, tol=1e-3, alpha=0.1, shuffle=True, loss='squared_error'),
#     "Linear Regression": LinearRegression(),
#     "Ridge":Ridge(alpha=1.0),
#     "Lasso":Lasso(alpha=0.1),
#     "Random Forest Regressor":RandomForestRegressor(n_estimators=100, n_jobs=7, random_state=42),
#     "Decision Tree Regressor":DecisionTreeRegressor( random_state=42),
#     "Cat  Gradient Boosting Regressor" : CatBoostRegressor(loss_function='RMSE'),
#     "Gradient Boosting Regressor":GradientBoostingRegressor(n_estimators=100, random_state=42),
#     "MLP Regressor":MLPRegressor(random_state=1, max_iter=500),
#     "Xtreme Gradient Boosting Regressor": XGBRegressor(**xgb_params),
#     "MultiTask ElasticNet":linear_model.ElasticNet(alpha = 0.5),
#     "SVR":SVR(C=1.0, epsilon=0.2)
# }
    






#xgboost = XGBRegressor(**xgb_params)


#CatBoostRegressor_mod = CatBoostRegressor(loss_function='RMSE')


#SGDRegressor = SGDRegressor(max_iter=10000, tol=1e-3, alpha=0.1, shuffle=True, loss='squared_error')


train_filtered.columns


train_filtered = train_filtered[['id', 'num_sold', 'date_day', 'date_month', 'date_year',
       'country_Canada', 'country_Finland', 'country_Italy', 'country_Kenya',
       'country_Norway', 'country_Singapore', 'store_Discount Stickers',
       'store_Premium Sticker Mart', 'store_Stickers for Less',
       'product_Holographic Goose', 'product_Kaggle', 'product_Kaggle Tiers',
       'product_Kerneler', 'product_Kerneler Dark Mode']]





y=train_filtered['num_sold']
X=train_filtered.loc[:, (train_filtered.columns != 'num_sold')]


# model, results = train_and_evaluate(models, X, y)
# print("\nResultados da Validação Cruzada:")
# print(results)






xgb_params = {
    'n_estimators': 1078, 
    'learning_rate': 0.016084079332671603, 
    'max_depth': 10, 
    'min_child_weight': 8, 
    'subsample': 0.8732132237392727, 
    'colsample_bytree': 0.9756972730817159, 
    'reg_alpha': 3.386299962300141, 
    'reg_lambda': 8.964009483088061,
    'enable_categorical': 'True',
    'device': 'cpu'
}

# estimators = [
#     ("SGDRegressor", SGDRegressor(max_iter=10000, tol=1e-3, alpha=0.1, shuffle=True, loss='squared_error')),
#     ("Linear Regression", LinearRegression()),
#     ("Ridge Regression", Ridge(alpha=1.0)),
#     ("Lasso Regression", Lasso(alpha=0.1)),
#     ('Random Forest', RandomForestRegressor(n_estimators=100, n_jobs=7, random_state=42)),
#     ('Decision Tree', DecisionTreeRegressor( random_state=42)),
#     ('Cat Boost Regressor', CatBoostRegressor(loss_function='RMSE')),
#     #('Gradient Boosting Regressor', GradientBoostingRegressor(random_state=0)),
#     ('MLP Regressor', MLPRegressor(random_state=1, max_iter=500)),
#     ('Xtreme Gradient Boosting Regressor', XGBRegressor(**xgb_params)),
#     #('ElasticNet', linear_model.ElasticNet(alpha = 0.5)),
#     #('SVR',SVR(C=1.0, epsilon=0.2))
# ]

estimators = [
    ("SGDRegressor", SGDRegressor(max_iter=10000, tol=1e-3, alpha=0.1, shuffle=True, loss='squared_error')),
    ('Random Forest', RandomForestRegressor(n_estimators=100, n_jobs=7, random_state=42)),
    #("Cat  Gradient Boosting Regressor", GradientBoostingRegressor(random_state=0)),
    #("Xtreme Gradient Boosting Regressor", XGBRegressor(**xgb_params))
]

reg = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression()
)

reg



#print(model)


# #separando dados do alvo
# X = train_filtered.drop('num_sold', axis=1)
# y = train_filtered['num_sold']


from sklearn.model_selection import train_test_split


# Divisão do conjunto de dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


reg.fit(X_train, y_train)


# CatBoostRegressor.fit(X_train, y_train)


#xgboost.fit(X_train, y_train)


#SGDRegressor.fit(X_train, y_train)


#y_SGDRegressor_pred = SGDRegressor.predict(X_test)


#y_CatBoostRegressor_pred = CatBoostRegressor.predict(X_test)


#xgboost.predict(X_test)


#model.fit(X_train, y_train)


y_reg_pred = reg.predict(X_test)


#y_pred = reg.predict(X_test)


#mean_absolute_percentage_error(y_test, y_CatBoostRegressor_pred)


#mean_absolute_percentage_error(y_test, y_CatBoostRegressor_pred)


#mean_absolute_percentage_error(y_test, y_xgboost_pred)


#mean_absolute_percentage_error(y_test, y_pred)


mean_absolute_percentage_error(y_test, y_reg_pred)


#y_pred = reg.predict(X_test)


# RandomForest = RandomForestRegressor(n_estimators=100, random_state=42)
# RandomForest.fit()
# test_encoded

submission = reg.predict(X_test)



X_test.shape


submission.shape


submission


test_encoded


id = X_test['id']
id


## Submit notebooks to the challenge. Final


submission_final = pd.DataFrame({

        "id":id,

        "num_sold":submission

    })

submission_final.to_csv('ForecastingStickerSalesStackingRegressorV2_07012025.csv', index=False)


print(" Arquivo submission ForecastingStickerSales.csv pronto ")


submission_final













