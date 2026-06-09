# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
#import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score
scaler=StandardScaler()
file="/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT.csv"
df=pd.read_csv(file)
#df.dropna()
df



df=pd.get_dummies(df)
corr=df.corr()
plt.figure(figsize=(40,40))
sns.heatmap(corr,annot=True,cmap="coolwarm",fmt=".2f",linewidth =1)



def correlation (dataset, threshold):
     col_corr=set() 
     corr_matrix=dataset.corr()
     for i in range(len(corr_matrix.columns)):
          for j in range(i):
              if abs(corr_matrix.iloc[i, j]) > threshold: 
                  colname=corr_matrix.columns[i] 
                  col_corr.add(colname)
     return col_corr


#print(df.head(15))
df.dropna(subset=["CORRUCYSTIC_DENSITY"],inplace=True)
#df
#df=df.dropna()
x=df.drop(columns=["LOCAL_IDENTIFIER","CORRUCYSTIC_DENSITY"])
scaled=scaler.fit_transform(x)
x.columns=x.columns.astype(str)
newcols = []
for col in x.columns:
    #new_col = ''.join(ch if ch.isalnum() or ch == '' else '' for ch in col)
    new=""
    for i in col:
        if i.isalnum() or i==" " or i =="_":
            new=new+i
    newcols.append(new)
#print(newcols)
x.columns=newcols
y=np.log1p(df["CORRUCYSTIC_DENSITY"])
x=pd.get_dummies(x,drop_first=True)
x.columns=x.columns.astype(str)
#x=x.fillna(method="ffill")
x=x.fillna(x.mean())
y=y.fillna(y.mean())





#df.duplicated().sum()


x_tr,x_ta,y_tr,y_ta=train_test_split(x,y,test_size=0.2)
model=HistGradientBoostingRegressor()
#print(list(x_tr.columns)==list(x_ta.columns))
x_tr, x_ta = x_tr.align(x_ta, join='left', axis=1, fill_value=0)
x_tr=pd.get_dummies(x_tr)
'''corr=df.corr()
#plt.figure(figsize=(200,200))
sns.heatmap(corr,annot=True,cmap="coolwarm")'''
corr_c=correlation(x_tr,0.9)
x_tr.drop(columns=corr_c,inplace=True)
x_ta.drop(columns=corr_c,inplace=True)



model.fit(x_tr,y_tr)
prediction=model.predict(x_ta)
rmse=np.sqrt(mean_squared_error(y_ta,prediction))
r2=r2_score(y_ta,prediction)
print(rmse)
print(r2)




df_test = pd.read_csv("/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT_UNK.csv")
X_test = df_test.drop(columns=["LOCAL_IDENTIFIER"])
X_test = pd.get_dummies(X_test, drop_first=True)
X_test=X_test.fillna(X_test.mean())
X_test = X_test.reindex(columns=x_tr.columns, fill_value=0)
prediction_test = model.predict(X_test)
submission = pd.DataFrame({
    "LOCAL_IDENTIFIER": df_test["LOCAL_IDENTIFIER"].astype(int),
    "CORRUCYSTIC_DENSITY": prediction_test.astype(float)})
submission.to_csv("submission.csv", index=False)

print(submission.head())

