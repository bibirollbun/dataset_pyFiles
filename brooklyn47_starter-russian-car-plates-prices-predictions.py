
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# data visualization 
import matplotlib.pyplot as plt
import seaborn as sns 

#ignoring warnings
import warnings
warnings.filterwarnings('ignore')

#file directory 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/train.csv")
test = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/test.csv")
submission = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/sample_submission.csv")


data  = {
    "training data": train,
    "testing data": test,
    "submission data": submission
}


for name , df in data.items():
    print(f"__{name}__")
    print(f"shape : {df.shape} ")
    print(f"features : {df.columns} ")
    display(df)
    print("\n" + "__"*42)


# unique values: 
display(train.nunique())
print("\n" + "__"*42)
display(test.nunique())


test.drop(columns = ["price"] , inplace = True)


# info about the columns 
display(train.info())
print("\n" + "__"*42)
display(test.info())



train.drop(columns = ["id"] , inplace = True)
test.drop(columns = ["id"] , inplace = True)


# missing values 
print("--total missing values train--")
display(train.isna().sum())
print("\n") 
print("--total missing values in test--")
display(test.isna().sum())


#basic data about numerical features
print("train")
display(train.describe())

print("\n")
print("test")
display(test.describe())


# date to datetime format
train["date"] = pd.to_datetime(train["date"], errors="coerce")
        
# Extracting date feature
train["year"] = train["date"].dt.year
train["month"] = train["date"].dt.month
train["day"] = train["date"].dt.day
train["weekday"] = train["date"].dt.weekday


train.info()




