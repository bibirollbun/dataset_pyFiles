# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
%matplotlib inline
sns.set_style('white')
plt.style.use("dark_background")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import warnings
# Specifically ignore Deprecation Warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Ignore future warnings
warnings.filterwarnings('ignore', category=FutureWarning)

warnings.filterwarnings('ignore', category=RuntimeWarning)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


data = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
data.head()


data.dropna(subset=['num_sold'],inplace=True)
data.drop('id',axis=1,inplace=True)

X_test=test_df.copy()
X_test.drop('id',axis=1,inplace=True)


def extract_date_features(df, column_name):
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    
    df['Year'] = df[column_name].dt.year
    df['Month'] = df[column_name].dt.month_name()
    df['Day'] = df[column_name].dt.day_name()
    df['Weekday'] = df[column_name].dt.weekday
    df['Week'] = df[column_name].dt.isocalendar().week
    df['Month_Start'] = df[column_name].dt.is_month_start
    df['Month_End'] = df[column_name].dt.is_month_end
    df['Leap_Year'] = df[column_name].dt.is_leap_year
    
    df.drop(column_name, inplace=True, axis = 1)
    return df.head()

extract_date_features(data,column_name='date')
extract_date_features(X_test,column_name='date')


group1 = data.groupby(['Year'])['num_sold'].sum().reset_index().sort_values(by='num_sold',ascending=False)
plt.figure(figsize=(20,8))
sns.barplot(data=group1, x='Year',y='num_sold',palette='plasma')
plt.title('Total Stickers sold in each Year')
plt.ylabel('Total Sale')
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.show()


month_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
data['Month'] = pd.Categorical(data['Month'],categories=month_order,ordered=True)

day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
data['Day'] = pd.Categorical(data['Day'],categories=day_order,ordered=True)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (18,6))
sns.barplot(x=data['Month'],y=data['num_sold'],ax=axes[1],color='lightcoral')
sns.barplot(x=data['Day'],y=data['num_sold'],ax=axes[0],color='lightblue')
plt.xticks(rotation=45)

plt.show()
plt.tight_layout()


group2 = data.groupby(['Year','country'])['num_sold'].sum().reset_index().sort_values(by='num_sold',ascending=False)
plt.figure(figsize=(20,8))
sns.lineplot(data=group2, x='Year',y='num_sold',hue='country',linewidth=5,palette='husl')
plt.title('Total Stickers sold in each Country per Year')
plt.ylabel('Total Sale')
plt.xlabel('Year')
plt.grid(axis='x')
plt.xlim(2010,2016)
plt.legend(title='country', loc='upper right')
plt.xticks(rotation=45)
plt.show()


top_products = data.groupby(['Year','product'])['num_sold'].sum().reset_index().sort_values(by='num_sold',ascending=False)
plt.figure(figsize=(16, 8))
sns.lineplot(data=top_products, x='Year', y='num_sold', hue='product', linewidth=5,marker='o')
plt.title('Sales Trend of Products over the years')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Total num_sold', fontsize=14)
plt.legend(title='Product')
plt.show()


plt.figure(figsize=(20,8))
sns.barplot(data=group2, x='country',y='num_sold',hue='Year',palette='plasma')
plt.title('Sales in each Country per Year')
plt.ylabel('Total Sale')
plt.xlabel('store')
plt.legend(title='Year', loc='upper right')
plt.show()


group3 = data.groupby(['country','store'])['num_sold'].sum().reset_index().sort_values(by='num_sold',ascending=False)
plt.figure(figsize=(20,8))
sns.barplot(data=group3, x='country',y='num_sold',hue='store',palette='plasma')
plt.title('Total Stickers sold in each Country per store')
plt.ylabel('Total Sale')
plt.xlabel('country')
plt.grid(axis='x')
plt.legend(title='store', loc='upper right')
plt.show()


ct = pd.crosstab(data['store'], data['product'])
ct.plot(kind='bar', stacked=True,figsize=(20,8))
plt.title('Share of Products sold in each Store')
plt.xticks(rotation=45, ha='right')
plt.show()


ct1 = pd.crosstab(data['country'], data['store'])
ct1.plot(kind='bar', stacked=True,figsize=(20,8))
plt.title('Share of Products sold in each Store')
plt.xticks(rotation=45, ha='right')
plt.show()


sns.histplot(data['num_sold'],bins=50)
plt.title('Target varaible before transformation')
plt.show()


data['num_sold'] = np.log1p(data['num_sold'])
sns.histplot(data['num_sold'],bins=50)
plt.title('Target varaible after transformation')
plt.show()


#Nominal
def dummies(df,cols):
    for col in cols:
        dummies = pd.get_dummies(df[col],dtype=int,prefix=col)
        df = pd.concat([df,dummies],axis=1)
        df = df.drop(labels=col, axis=1)
    return df

nominal = ['country', 'store', 'product']
data = dummies(data,nominal)
X_test = dummies(X_test,nominal)


data['Month_Start']=data['Month_Start'].replace({True:1,False:0})
data['Month_End']=data['Month_End'].replace({True:1,False:0})
data['Leap_Year']=data['Leap_Year'].replace({True:1,False:0})
X_test['Month_Start']=X_test['Month_Start'].replace({True:1,False:0})
X_test['Month_End']=X_test['Month_End'].replace({True:1,False:0})
X_test['Leap_Year']=X_test['Leap_Year'].replace({True:1,False:0})

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()

data["Month"] = enc.fit_transform(data["Month"])
data["Month"]=data["Month"]+1 #to start the labels from 1 instead of 0
data["Day"] = enc.fit_transform(data["Day"])
data["Day"] = data["Day"]+1
X_test["Month"] = enc.fit_transform(X_test["Month"])
X_test["Month"]=X_test["Month"]+1 #to start the labels from 1 instead of 0
X_test["Day"] = enc.fit_transform(X_test["Day"])
X_test["Day"] = X_test["Day"]+1


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


X = data.drop(columns='num_sold')
Y = data['num_sold']
X_train,X_val,y_train,y_val = train_test_split(X,Y,test_size=0.2,random_state=40) 
print(X_train.shape,X_val.shape)


data_copy = pd.concat([X_train,y_train],axis=1)
corr = data_copy.corr()

fig, ax = plt.subplots(figsize=(30, 20))
sns.heatmap(corr, cmap='coolwarm', annot=True, ax=ax)
ax.set_title('Correlation Matrix')
plt.show()


def evaluate_models(X_train,X_val,y_train,y_val):
    models={
        "Linear Regression":LinearRegression(),
        "Random Forest Regression":RandomForestRegressor(),
        "Decision Tree Regressor":DecisionTreeRegressor(),
        "K-Nearest Neighbours":KNeighborsRegressor(),
        "XGBoost":XGBRegressor()
    }
    result={}
    for name ,model in models.items():
        model.fit(X_train,y_train)
        y_pred=model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        r2=r2_score(y_val,y_pred)
        result[name]={"R2_Score":r2*100,"MSE":mse}
    

    result_frame=pd.DataFrame(result).T

    return result_frame

result_frame=evaluate_models(X_train,X_val,y_train,y_val)
result_frame


xgb=XGBRegressor()
model_xgb=xgb.fit(X_train,y_train)
y_pred_val=model_xgb.predict(X_val)
y_pred_train=model_xgb.predict(X_train)
r2=r2_score(y_val,y_pred_val)


val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
print(f"Model r2: {r2}")
print(f"test_rmse : {val_rmse}")
print(f"train_rmse : {train_rmse}")

if val_rmse > train_rmse * 1.5:  # Arbitrary threshold
            print("The model might be overfitting.")

# Residuals
residuals_train = y_train - y_pred_train
residuals_val = y_val - y_pred_val


# Plot
sns.histplot(residuals_train, kde=True, label='Train Residuals', color='blue')
sns.histplot(residuals_val, kde=True, label='Val Residuals', color='orange')
plt.legend()
plt.show()


from sklearn.model_selection import cross_val_score, KFold
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Evaluate with Negative MSE (Scikit-learn uses negative scores for consistency)
mse_scores = cross_val_score(xgb, X, Y, scoring='neg_mean_squared_error', cv=kf)
rmse_scores = np.sqrt(-mse_scores)  # Convert negative MSE to RMSE

# Evaluate with R^2
r2_scores = cross_val_score(model_xgb, X, Y, scoring='r2', cv=kf)

# Results
print("RMSE Scores for each fold:", rmse_scores)
print("Mean RMSE:", rmse_scores.mean())
print("R^2 Scores for each fold:", r2_scores)
print("Mean R^2:", r2_scores.mean())


submission_ids = test_df['id']
predictions = model_xgb.predict(X_test)

predictions = np.expm1(predictions)


submission = pd.DataFrame({
    'id': submission_ids,
    'num_sold': predictions 
})
submission.head()


submission.to_csv('submission.csv', index=False)
print("File Saved!")
print(submission.head())

