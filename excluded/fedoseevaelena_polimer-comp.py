import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df_original = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv', low_memory=False)
target = df_original['Tc']


train_tc = df_original.dropna(subset=['Tc'])


train_tc=train_tc[['id','SMILES','Tc']]


train_tc.info()


dataset1 = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv', low_memory=False)


dataset1.info()
dataset1.columns=['SMILES', 'Tc']


dataset1.info()


train_tc_combined = pd.concat([train_tc[['SMILES', 'Tc']], dataset1], ignore_index=True)


train_tc_combined.shape[0]


print(f'В наборе данных имеется {train_tc_combined.duplicated().sum()} дублей.')
train_tc_combined.loc[train_tc_combined.duplicated()]
train_tc_combined=train_tc_combined.drop_duplicates()


from sklearn.feature_extraction.text import CountVectorizer

# Инициализация векторайзера
vectorizer = CountVectorizer()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

X=train_tc_combined['SMILES']
y=train_tc_combined['Tc']
# Предположим, у нас есть данные X (тексты) и y (метки категорий)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем пайплайн
text_lr = Pipeline([
('vect', CountVectorizer()),
('lr', LinearRegression()),
])

# Обучаем модель
text_lr.fit(X_train, y_train)

# Предсказываем на тестовой выборке
y_pred = text_lr.predict(X_test)

# Оцениваем точность
print(f"mse: {mean_squared_error(y_test, y_pred):.2f}")
print(f"mae: {mean_absolute_error(y_test, y_pred):.2f}")


train_rg = df_original.dropna(subset=['Rg'])
train_rg=train_rg[['id','SMILES','Rg']]
train_rg.describe()


# используем vectorize и линейную регрессию
from sklearn.linear_model import Lasso, LinearRegression, ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

X=train_rg['SMILES']
y=train_rg['Rg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем пайплайн
text_rg = Pipeline([
('vect', CountVectorizer()),
('ri', LinearRegression())
])

# Обучаем модель
text_rg.fit(X_train, y_train)

# Предсказываем на тестовой выборке
y_pred = text_rg.predict(X_test)

# Оцениваем точность
print(f"mse: {mean_squared_error(y_test, y_pred):.2f}")
print(f"mae: {mean_absolute_error(y_test, y_pred):.2f}")

rez=pd.DataFrame(data=X_test, columns=['SMILES'])
rez['Rg']=y_test
rez['predict_Rg']=y_pred
rez



train_tg = df_original.dropna(subset=['Tg'])


train_tg=train_tg[['id','SMILES','Tg']]


train_tg.info()


dataset3 = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv', low_memory=False)


dataset3.info()
dataset3.columns=['SMILES', 'Tg']


train_tg_combined = pd.concat([train_tg[['SMILES', 'Tg']], dataset3], ignore_index=True)


train_tg_combined


from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

X=train_tg_combined['SMILES']
y=train_tg_combined['Tg']

# y=y/y.abs().max()


from sklearn.linear_model import Lasso, LinearRegression, ElasticNet, Ridge
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем пайплайн
text_tg = Pipeline([
('vect', CountVectorizer()),
('lr2', Lasso()),
])

# Обучаем модель
text_tg.fit(X_train, y_train)

# Предсказываем на тестовой выборке
y_pred = text_tg.predict(X_test)

# Оцениваем точность
print(f"mse: {mean_squared_error(y_test, y_pred):.2f}")
print(f"mae: {mean_absolute_error(y_test, y_pred):.2f}")


train_ffv = df_original.dropna(subset=['FFV'])


train_ffv=train_ffv[['id','SMILES','FFV']]


train_ffv.info()


dataset4 = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv', low_memory=False)


dataset4.info()
# dataset4.columns=['SMILES', 'Tc']


dataset4.info()


train_ffv_combined = pd.concat([train_ffv[['SMILES', 'FFV']], dataset4], ignore_index=True)


print(f'В наборе данных имеется {train_ffv_combined.duplicated().sum()} дублей.')
train_ffv_combined.loc[train_ffv_combined.duplicated()]
train_ffv_combined=train_ffv_combined.drop_duplicates()


# Инициализация векторайзера
vectorizer = CountVectorizer()

X=train_ffv_combined['SMILES']
y=train_ffv_combined['FFV']
# Предположим, у нас есть данные X (тексты) и y (метки категорий)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем пайплайн
text_ffv = Pipeline([
('vect', CountVectorizer()),
('lr', LinearRegression()),
])

# Обучаем модель
text_ffv.fit(X_train, y_train)

# Предсказываем на тестовой выборке
y_pred = text_ffv.predict(X_test)

# Оцениваем точность
print(f"mse: {mean_squared_error(y_test, y_pred):.4f}")
print(f"mae: {mean_absolute_error(y_test, y_pred):.4f}")


rez=pd.DataFrame(data=X_test, columns=['SMILES'])
rez['FFV']=y_test
rez['predict_Rg']=y_pred
rez


train_density = df_original.dropna(subset=['Density'])


train_density=train_density[['id','SMILES','Density']]


# Инициализация векторайзера
vectorizer = CountVectorizer()

X=train_density['SMILES']
y=train_density['Density']
# Предположим, у нас есть данные X (тексты) и y (метки категорий)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем пайплайн
text_den = Pipeline([
('vect', CountVectorizer()),
('lr', LinearRegression()),
])

# Обучаем модель
text_den.fit(X_train, y_train)

# Предсказываем на тестовой выборке
y_pred = text_den.predict(X_test)

# Оцениваем точность
print(f"mse: {mean_squared_error(y_test, y_pred):.4f}")
print(f"mae: {mean_absolute_error(y_test, y_pred):.4f}")



# id	Tg	FFV	Tc	Density	Rg
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv', low_memory=False)
X_test=test['SMILES']


# Делаем предсказания
test_predict_Tg = text_tg.predict(X_test)
test_predict_FFV = text_ffv.predict(X_test)
test_predict_Tc = text_lr.predict(X_test)
test_predict_Density = text_den.predict(X_test)
test_predict_Rg = text_tg.predict(X_test)



# Сохранение submission.csv
submission = pd.DataFrame({
'id': test['id'],
'Tg': test_predict_Tg,
'FFV': test_predict_FFV,
'Tc': test_predict_Tc,
'Density': test_predict_Density,
'Rg': test_predict_Rg    
})

submission.to_csv('submission.csv', index=False)

