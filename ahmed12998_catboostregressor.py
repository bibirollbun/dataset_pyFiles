import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer ,make_column_selector as selector
from sklearn.preprocessing import StandardScaler 
from category_encoders import BinaryEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split , cross_validate
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from catboost import CatBoostRegressor, Pool


df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


df_train.drop('id' , axis=1 , inplace=True)
df_test.drop('id' , axis=1 , inplace=True)


df_train.info()


col_num = df_train.select_dtypes(include='number').columns.tolist()
for col in col_num:
    if df_train[col].isna().any():
        print(f'{col}: {df_train[col].isna().mean()}')


df_train.dropna(subset=['Episode_Length_minutes' ,'Guest_Popularity_percentage', 'Number_of_Ads'] , axis=0 , inplace=True)


col_num2 = df_test.select_dtypes(include='number').columns.tolist()
for col in col_num2:
    if df_test[col].isna().any():
        print(f'{col}: {df_test[col].isna().mean()}')


df_test['Episode_Length_minutes'] = df_test['Episode_Length_minutes'].apply(
    lambda x: df_test['Episode_Length_minutes'].mean() if pd.isna(x) else x)
df_test['Guest_Popularity_percentage'] = df_test['Guest_Popularity_percentage'].apply(
    lambda x: df_test['Guest_Popularity_percentage'].mean() if pd.isna(x) else x)


sns.pairplot(df_train)


col_cat = df_train.select_dtypes(include='O').columns.tolist()
transf = ColumnTransformer([
    ('bin', BinaryEncoder() , col_cat)
],remainder='passthrough')


pl = make_pipeline(
    transf,
    CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        early_stopping_rounds=50,
        verbose=False
    )
)


x = df_train.drop('Listening_Time_minutes' , axis=1)
y = df_train['Listening_Time_minutes']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


pl.fit(x , y)


pl.score(x , y)


pred = pl.predict(df_test)


submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


submission['Listening_Time_minutes'] = pred


#submission.to_csv('/kaggle/input/playground-series-s5e4/submis.csv' , index=False)


pd.read_csv('/kaggle/input/submission33/submis.csv')

