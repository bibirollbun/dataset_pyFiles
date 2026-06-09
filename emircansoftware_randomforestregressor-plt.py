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


df=pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")



df.head()


df.info()


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.boxplot(x=df["Episode_Length_minutes"])
plt.title("Outlier Detection via Boxplot")
plt.show()


print(df[df["Episode_Length_minutes"] > 300])



df=df.drop(101637,axis=0)



df.describe()



df["Episode_Length_minutes"].describe()


df["Episode_Length_minutes"]=df["Episode_Length_minutes"].fillna(df["Episode_Length_minutes"].mean())



df["Guest_Popularity_percentage"]=df["Guest_Popularity_percentage"].fillna(df["Guest_Popularity_percentage"].mean())


df["Number_of_Ads"].value_counts()


counts = df["Number_of_Ads"].value_counts()

remove = counts[(counts == 1) | (counts == 2)].index

df = df[~df["Number_of_Ads"].isin(remove)]


df["Number_of_Ads"].value_counts()


df["Number_of_Ads"]=df["Number_of_Ads"].fillna(df["Number_of_Ads"].median())



df.head()


from sklearn.preprocessing import LabelEncoder
def preprocess_data(df):
    categories = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
    for col in categories:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


df=preprocess_data(df)



df.head()


X = df.drop(columns=['Listening_Time_minutes', 'id', 'Episode_Title',"Podcast_Name"])
y = df['Listening_Time_minutes']


import time
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from math import sqrt


def compare_Models(X, y):
    models = {
        'Linear Regression': LinearRegression(),
        'XGBoost': XGBRegressor(objective="reg:squarederror"),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
    }
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    results = []

    for name, model in models.items():
        start_time = time.time()  
        model.fit(X_train, y_train)  
        end_time = time.time()  

        y_pred = model.predict(X_test)  
        r2 = r2_score(y_test, y_pred)
        mse=mean_squared_error(y_test, y_pred)
        rmse=sqrt(mse)
        
       
        results.append({
            'Model': name,
            'Training Time (seconds)': end_time - start_time,
            'R2 Score': r2,
            "RMSE":rmse
        })

   
    return pd.DataFrame(results).sort_values(by='Training Time (seconds)', ascending=True)


#compare_Models(X,y)


test=pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")



test.head()


test.info()



test.describe()


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.boxplot(x=test["Episode_Length_minutes"])
plt.title("Outlier Detection via Boxplot")
plt.show()


print(test[test["Episode_Length_minutes"] > 300])



mean_value = test[test["Episode_Length_minutes"] <= 300]["Episode_Length_minutes"].mean()
test.loc[test["Episode_Length_minutes"] > 300, "Episode_Length_minutes"] = mean_value

test["Episode_Length_minutes"]=test["Episode_Length_minutes"].fillna(test["Episode_Length_minutes"].mean())


plt.figure(figsize=(10, 6))
sns.boxplot(x=test["Episode_Length_minutes"])
plt.title("Outlier Detection via Boxplot")
plt.show()


test["Guest_Popularity_percentage"]=test["Guest_Popularity_percentage"].fillna(test["Guest_Popularity_percentage"].mean())


test=test.drop(columns=["Episode_Title","Podcast_Name"])
test=test.set_index("id")


test["Number_of_Ads"].value_counts()



test.loc[test["Number_of_Ads"] == 2063.00, "Number_of_Ads"] = 1.0
test.loc[test["Number_of_Ads"] == 89.12, "Number_of_Ads"] = 0.0


test=preprocess_data(test)


test.head()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
model = RandomForestRegressor( n_jobs=-1)


model.fit(X_train,y_train)
y_pred=model.predict(X_test)
r2_Score=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=sqrt(mse)
print(f"R2 Score: {r2_Score} and RMSE: {rmse}")


from sklearn.model_selection import KFold, cross_val_score
def kfold_regression_models(X, y, model=RandomForestRegressor(), k=5):
   
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')  
    
    cv_scores = -cv_scores
    
    print(f"{model.__class__.__name__} - {k}-Fold Cross Validation Results:")
    print(f"Mean Squared Error for each fold: {cv_scores}")
    print(f"Average Mean Squared Error: {np.mean(cv_scores)}")
    print(f"Standard Deviation of MSE: {np.std(cv_scores)}")


#kfold_regression_models(X, y, k=5)


predict=model.predict(test)



print(predict)



predict=pd.DataFrame(predict,index=test.index)


predict.reset_index(inplace=True)


predict.columns=["id","Listening_Time_minutes"]
predict.head()


predict.to_csv("submission.csv", index=False)




