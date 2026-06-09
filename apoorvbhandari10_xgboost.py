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


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import seaborn as sns
from scipy import sparse
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_test  = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


df_train.isnull().sum()


##Imputing 


impute_cols = ["Episode_Length_minutes","Guest_Popularity_percentage","Number_of_Ads"]
def impute_(df):
    for col in df.columns:
        for col in impute_cols:
            df[col] = df[col].fillna(df.groupby(["Podcast_Name","Genre","Publication_Day","Publication_Time"])[col].transform("median"))
            df[col] = df[col].fillna(df.groupby(["Podcast_Name","Genre","Publication_Day"])[col].transform("median"))
            df[col] = df[col].fillna(df.groupby(["Podcast_Name","Genre"])[col].transform("median"))
            df[col] = df[col].fillna(df.groupby("Podcast_Name")[col].transform("median"))
            df[col] = df[col].fillna(df[col].median())
        return df
    


df = impute_(df_train)


y = df["Listening_Time_minutes"]


df.drop(["id","Guest_Popularity_percentage","Listening_Time_minutes"],axis=1,inplace=True)


cat_cols = df.select_dtypes(include="object").columns.tolist()
num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()


def encode_(df):
    ohe = OneHotEncoder(sparse_output=True)
    x_sparse = ohe.fit_transform(df[cat_cols])
    return x_sparse


x_sparse = encode_(df)
from scipy import sparse
x_full = sparse.hstack([x_sparse,df[num_cols]])


x_train,x_test,y_train,y_test = train_test_split(x_full,y,test_size=0.2,random_state=42)


from xgboost import XGBRegressor


model = XGBRegressor()


model.fit(x_train,y_train)


y_preds = model.predict(x_test)


rmse = np.sqrt(mean_squared_error(y_test,y_preds))


rmse


df_test  = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')



df_test.head()


df_test = impute_(df_test)


df_temp = df_test.copy()


df_test.drop(["id","Guest_Popularity_percentage"],axis=1,inplace=True)


df_test.head()


df_test = encode_(df_test)


test_sparse = sparse.hstack([df_test,df_temp[num_cols]])


final_perdiction = model.predict(test_sparse)


df_test_ids = df_temp["id"].copy()


submission = pd.DataFrame()
submission["id"] = df_test_ids
submission["Listening_Time_minutes"] = final_perdiction


submission.to_csv('submission.csv',index=False)




