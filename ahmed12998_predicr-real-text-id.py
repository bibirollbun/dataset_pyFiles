import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


df_train = pd.read_csv('/kaggle/input/submission55/train1.csv')
df_test = pd.read_csv('/kaggle/input/submission55/test1.csv')


df_train['text1'] = df_train['text1'].apply(lambda x : df_train['text1'].mode()[0] if pd.isna(x) else x)
df_train['text2'] = df_train['text2'].apply(lambda x : df_train['text2'].mode()[0] if pd.isna(x) else x)


df_test['text2'] = df_test['text2'].apply(lambda x : df_test['text2'].mode()[0] if pd.isna(x) else x)


df_train.drop(['Unnamed: 0','id'] , axis=1 , inplace=True)
df_test.drop('Unnamed: 0' , axis=1 , inplace=True)


x = df_train['text1'].fillna('') + ' ' + df_train['text2'].fillna('')
y = df_train['real_text_id']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)


vec = TfidfVectorizer(max_features=6000)
x_train = vec.fit_transform(x_train)
x_test = vec.transform(x_test)


lin = LogisticRegression(C = 1)


lin.fit(x_train , y_train)


lin.score(x_train, y_train)


df_test['id'] = df_test['id'].str.replace('article_', '').astype(int)


df_test_text = df_test['text1'].fillna('') + ' ' + df_test['text2'].fillna('')


X_test_final = vec.transform(df_test_text)


pred = lin.predict(X_test_final)


subm = pd.DataFrame({
    'id': df_test['id'],
    'real_text_id': pred
})


#subm.to_csv('submis.csv' , index=False)


pd.read_csv('/kaggle/input/submission55/submis.csv')

