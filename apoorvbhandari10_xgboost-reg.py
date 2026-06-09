# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import ast
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/train.csv")
test_df = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/test.csv")
# suppliment_df = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/supplemental_english.py")


train_df.info()


train_df.head()


train_df.sort_values(by='price',ascending=False).head()


train_df['log_price'] = np.log1p(train_df['price'])  

# Plot the transformed data
train_df['log_price'].hist(bins=20, color='blue', edgecolor='black')
plt.title('Distribution of Log(Price)')
plt.xlabel('Log(Price)')
plt.ylabel('Frequency')
plt.show()


sns.kdeplot(train_df['price'], shade=True, color='green')
plt.title('KDE Plot of Price')
plt.xlabel('Price')
plt.ylabel('Density')
plt.show()


print(train_df['price'].describe())


with open('/kaggle/input/russian-car-plates-prices-prediction/supplemental_english.py', 'r', encoding='utf-8') as f:
    supplement_english = f.read()
          


region_codes = ast.literal_eval(supplement_english.split("REGION_CODES = ")[1].split("\n\n")[0])



region_name = {}
for regions , codes in region_codes.items():
    for code in codes:
        region_name[code]=regions


def extracting_plates(df):
    for i,plate in enumerate(df["plate"]):
        First_letter,remain_letter= (re.findall(r'[A-Z]+',plate)) #First and remaining letters of the plate
        numbers = re.search(r'\d+$',plate)   # Last numbers of the plate
        df.loc[i:i,"First_letter"]=First_letter
        df.loc[i:i,"Letter_series"]=remain_letter
        Last_char=remain_letter.replace(First_letter,"")
        df.loc[i:i,"Region_code"] = str(numbers.group())
    
        # extracting the middle numbers (Numeric codes)
        df.loc[i:i,"Number_series"] = plate.replace(numbers.group(),"").replace(First_letter,"").replace(remain_letter,"").replace(Last_char,"")
    
    df["Region_name"] = df["Region_code"].map(region_name).fillna("Unknown")
    return df


def date_features(df):
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    return df


train_df = extracting_plates(train_df)
train_df = date_features(train_df)


df_temp = train_df.copy()


train_df.drop(["date","price","plate","Region_code","id"],axis=1,inplace=True)


def encoding(df):
    features = ["First_letter","Letter_series","Region_name","Number_series"]
    for col in features:
        df.loc[:,col] = df[col].astype("string")
    for col in features:
        lbl=LabelEncoder()
        df.loc[:,col] = lbl.fit_transform(df[col])
    return df

    
    
    
            


train_df = encoding(train_df)


y=train_df["log_price"]


x_train,x_test,y_train,y_test = train_test_split(train_df.drop(["log_price"],axis=1),y,test_size=0.2,random_state=42)


model = XGBRegressor()
model.fit(x_train.values,y_train)


y_preds_train = model.predict(x_train.values)


y_preds_train = np.exp(y_preds_train)
print(np.sqrt(mean_squared_error(y_train,y_preds_train)))


y_preds_test = model.predict(x_test.values)
y_preds_test=np.exp(y_preds_test)
print(np.sqrt(mean_squared_error(y_test,y_preds_test)))


test_df.head()


test_df.info()


test_df['log_price'] = np.exp(test_df['price'])


test_df = extracting_plates(test_df)
test_df=date_features(test_df)



df_test_ids = test_df["id"].copy()


test_data = test_df["log_price"]


test_df.drop(["id","plate","date","price","Region_code","log_price"],axis=1,inplace=True)


test_df = encoding(test_df)


final_perdiction = model.predict(test_df.values)
final_perdiction=np.exp(final_perdiction)
submission = pd.DataFrame()
submission["id"] = df_test_ids
submission["price"] = final_perdiction


submission.to_csv('submission.csv',index=False)





