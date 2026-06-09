import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler , LabelEncoder , OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error ,mean_absolute_error , r2_score


df = pd.read_csv("/kaggle/input/playground-series-s4e12/train.csv")
df


cat = ['Gender' ,'Marital Status' , 'Education Level', 'Occupation', 'Location', 'Policy Type', 'Customer Feedback' , 'Smoking Status', 'Exercise Frequency','Property Type']



df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'])



df = df.drop('id' , axis=1)



#df = df.drop('Policy Start Date' , axis=1)


for col in cat:
    df[col] = df[col].astype('category')


df.isnull().sum()


for column in df.columns:
    print(df[column].unique())
    print("____________________________________")


df.shape[0]


for i in df.columns:
    col_non = df[i].isnull().sum()
    na_per  = (col_non / df.shape[0]) * 100
    print(f"missing in {i} is : {na_per.round(2)} % ")



numerical_columns = df.select_dtypes(['float64' , 'int64']).columns
numerical_columns


def detect_outlier(column):
    Q1 = np.quantile( column , 0.25)
    Q3 = np.quantile( column , 0.75)
    IQR = Q3 - Q1
    Lower = Q1 - 1.5*IQR
    Upper = Q3 + 1.5*IQR
    Outlier = column[ (  ( column < Lower) |  (column > Upper ) ) ]
    return Outlier


out = {}
for col in numerical_columns:
    out[col] = detect_outlier(df[col])

out



out = pd.DataFrame(out)
out


cor = df.corr(numeric_only=True)
cor


px.imshow(cor)


df = df.drop('Policy Start Date' , axis=1)



LE = LabelEncoder()
for i in cat:
   df[i]= LE.fit_transform(df[i])

print(df.info())
df


X = df.drop('Premium Amount' , axis=1)
y = df['Premium Amount']


X_train , X_test , y_train, y_test = train_test_split(X ,y , test_size = 0.3 , random_state = 42 )



scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled


LR = LinearRegression()
LR


LR.fit(X_train_scaled , y_train)



X_test_scaled = scaler.fit_transform(X_test)
X_test_scaled


y_pred = LR.predict(X_test_scaled)
y_pred


LR_r2 = r2_score(y_test , y_pred)
LR_r2


LR_MSE = mean_squared_error(y_test , y_pred)
LR_MSE


import math
math.sqrt(LR_MSE)


LR_MAE = mean_absolute_error(y_test , y_pred)
LR_MAE

