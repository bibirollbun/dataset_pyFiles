import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split , cross_validate , GridSearchCV
from category_encoders import BinaryEncoder
from sklearn.pipeline import Pipeline , make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score , mean_squared_error , accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')
df2 = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')


cat_col = df.select_dtypes('O').columns.tolist()

for col in cat_col:
    if df[col].isna().any():
        print(f"{col}: {df[col].isna().mean()}")


df.drop(['id','subdivision' , 'submarket'] , axis=1 , inplace=True)


cat_col2 = df2.select_dtypes('O').columns.tolist()

for col in cat_col2:
    if df2[col].isna().any():
        print(f"{col}: {df2[col].isna().mean()}")


df2.drop(['id','subdivision' , 'submarket'] , axis=1 , inplace=True)


numirec_col = df.select_dtypes(include='number').columns

for col in numirec_col:
    if df[col].isna().any():
        print(f"{col}: {df[col].isna().mean()}")


df.drop(['sale_nbr'] , axis=1 , inplace=True)


numirec_col2 = df2.select_dtypes(include='number').columns

for col in numirec_col2:
    if df2[col].isna().any():
        print(f"{col}: {df2[col].isna().mean()}")


df2.drop(['sale_nbr'] , axis=1 , inplace=True)


df['sale_date'] = pd.to_datetime(df['sale_date'])
df['year'] = df['sale_date'].dt.year
df['month'] = df['sale_date'].dt.month
df['day'] = df['sale_date'].dt.day
df.drop('sale_date' , axis=1 , inplace=True)


df2['sale_date'] = pd.to_datetime(df2['sale_date'])
df2['year'] = df2['sale_date'].dt.year
df2['month'] = df2['sale_date'].dt.month
df2['day'] = df2['sale_date'].dt.day
df2.drop('sale_date' , axis=1 , inplace=True)


x_train = df.drop('sale_price' , axis=1)
y_train = df['sale_price']
x_test = df2.copy()


cat_col = df.select_dtypes('O').columns.tolist()
transform = ColumnTransformer([
    ('bin' , BinaryEncoder() , cat_col)
] , remainder='passthrough')


pl = make_pipeline(transform , StandardScaler() ,GradientBoostingRegressor())


cv = cross_validate(estimator=pl , X = x_train , y = y_train , cv =5 ,scoring='r2' ,return_train_score=True)


cv['train_score'].mean()


cv['test_score'].mean()


pl.fit(x_train, y_train)


predictions = pl.predict(x_test)


ids = range(200000, 400000)
lower = predictions * 0.95
upper = predictions * 1.05


submission = pd.DataFrame({
    'id': id,
    'pi_lower': lower,
    'pi_upper': upper
})


submission.to_csv('submission.csv', index=False)




