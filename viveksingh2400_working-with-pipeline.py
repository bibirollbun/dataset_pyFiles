import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import pickle 
from sklearn.impute import SimpleImputer,KNNImputer


df = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')


test=pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')


df.info()


df.drop(['id','Policy Start Date'],axis='columns',inplace=True)


df.sample(5)


X= df.iloc[:10000,:-1]
y=df.iloc[:10000,-1]


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


trf1 = ColumnTransformer(transformers=[('mean1',SimpleImputer(strategy='mean'),[0,2,7,11,12,13])],
                         remainder='passthrough')


trf2 = ColumnTransformer(transformers=[('medium1',SimpleImputer(strategy='most_frequent'),[3,4,6,10,15])],remainder='passthrough')#10


X_train.info()


trf3 = ColumnTransformer(transformers=[
    ('edu',OrdinalEncoder(categories=[["Female","Male"],
                                      ["High School","Bachelor's","Master's","PhD"],
                                      ["Basic","Comprehensive","Premium"],
                                      ["Poor","Average","Good"],
                                      ["Rarely","Daily","Weekly","Monthly"],
                                      ],handle_unknown="use_encoded_value", unknown_value=-1),[1,5,9,14,16])
],remainder='passthrough')


trf4 = ColumnTransformer(
    ('ohe',OneHotEncoder(sparse_output=False,handle_unknown='ignore',drop='first'),[3,6,8,15,17]),remainder='passthrough')


X_train = trf1.fit_transform(X_train)
X_train = trf2.fit_transform(X_train)
X_train = trf3.fit_transform(X_train)
# X_train = trf4.fit_transform(X_train)


df_train = pd.DataFrame(X_train)
df_train 



# dtr = DecisionTreeRegressor()
# svm = SVR()
# rf = RandomForestRegressor()
# pipe1 = make_pipeline(trf1,trf2,trf3,trf4,dtr)
# pipe2 = make_pipeline(trf1,trf2,trf3,trf4,svm)
# pipe3 = make_pipeline(trf1,trf2,trf3,trf4,rf)


# pipe1.fit(X_train,y_train)




