import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score,KFold
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import StandardScaler
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


train.head()


print("train shape :",train.shape)


print("test shape :",test.shape)


train.isna().sum()


test.isna().sum()


train.drop("id",axis=1,inplace=True)
train.dropna(subset=["Number_of_Ads"], inplace=True)
test_ids=test["id"]
test.drop("id",axis=1,inplace=True)


train.info()


train.describe()


categorical = train.select_dtypes(include=['object']).columns
for col in categorical:
    print(train[col].value_counts(),"\n")


train["Episode_Title"] = train["Episode_Title"].str.replace("Episode ", "").astype(int)
test["Episode_Title"] = test["Episode_Title"].str.replace("Episode ", "").astype(int)


train["Episode_Title"].describe()


train["Episode_Length_minutes"].fillna(train["Episode_Length_minutes"].mean(),inplace=True)
train["Guest_Popularity_percentage"].fillna(train["Guest_Popularity_percentage"].mean(),inplace=True)
test["Episode_Length_minutes"].fillna(test["Episode_Length_minutes"].mean(),inplace=True)
test["Guest_Popularity_percentage"].fillna(test["Guest_Popularity_percentage"].mean(),inplace=True)


for col in train.select_dtypes(include=['object']):
    train[col] = pd.factorize(train[col])[0]
for col in test.select_dtypes(include=['object']):
    test[col] = pd.factorize(test[col])[0]


train.head()


test.head()


corr_matrix = train.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(
    corr_matrix,
    annot=True,     
    fmt=".2f",       
    cmap="coolwarm", 
    vmin=-1,         
    vmax=1,          
    linewidths=0.5,
    square=True    
)

plt.title("Correlation Matrix", fontsize=16)
plt.xticks(rotation=45, ha="right") 
plt.tight_layout()
plt.show()


X_train = train.drop("Listening_Time_minutes",axis=1)
y_train = train["Listening_Time_minutes"]


model = LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(test)

submit=pd.DataFrame()
submit["id"]=test_ids
submit["Listening_Time_minutes"]=y_pred

submit.to_csv("submission.csv",index=False)

