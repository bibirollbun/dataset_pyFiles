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


import re


df=pd.read_csv("/kaggle/input/playground-series-s4e9/train.csv")
df.head(10)
df = df.dropna(axis=0)


def extract_hp(text):
    match = re.search(r"(\d+\.?\d*)HP", text)
    return float(match.group(1)) if match else None

def extract_liters(text):
    match = re.search(r"(\d+\.?\d*)L", text)
    return float(match.group(1)) if match else None


def extract_cylinders(text):
    match = re.search(r"(\d+)\sCylinder", text)
    return int(match.group(1)) if match else None
df["HP"] = df["engine"].apply(extract_hp)
df["Liters"] = df["engine"].apply(extract_liters)
df["Cylinders"] = df["engine"].apply(extract_cylinders)

df.head(3)


df= pd.get_dummies(df,columns=["fuel_type","transmission","ext_col","int_col","clean_title","brand"],drop_first=True)


df.head(3)


df["accident"] = df["accident"].apply(lambda x: 0 if x == "None reported" else 1)





df.drop(columns=["engine", "model"], inplace=True )


df.head(3)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


td=pd.read_csv("/kaggle/input/playground-series-s4e9/test.csv")
td["HP"] = td["engine"].apply(extract_hp)
td["Liters"] = td["engine"].apply(extract_liters)
td["Cylinders"] = td["engine"].apply(extract_cylinders)
td= pd.get_dummies(td,columns=["fuel_type","transmission","ext_col","int_col","clean_title","brand"],drop_first=True)
td["accident"] = td["accident"].apply(lambda x: 0 if x == "None reported" else 1)
td.drop(columns=["engine", "model"], inplace=True )



train_x = df.drop(["price"], axis=1)  
train_y = df["price"] 

test_x = td
train_x = train_x.fillna(train_x.mean())  # Ortalamayla doldurur
test_x = test_x.reindex(columns=train_x.columns, fill_value=0) 
test_x = test_x.fillna(test_x.mean())  # Ortalamayla doldurur



model = LinearRegression()
model.fit(train_x , train_y)



predictions = model.predict(test_x)
result_df = pd.DataFrame({"id": test_x["id"], "price": predictions})
print(result_df.head())


result_df.head(-1)


result_df.to_csv('submission.csv', index=False)


