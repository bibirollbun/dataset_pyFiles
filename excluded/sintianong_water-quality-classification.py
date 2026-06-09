# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


data_train = pd.read_csv("/kaggle/input/water-quality-classification/data.csv")
data_test = pd.read_csv("/kaggle/input/water-quality-classification/test.csv")


data_train.shape


data_test.shape


data_train.info()


data_train.describe()


data_train.isnull().sum()


rows = 3
cols = 3
a=0
fig,ax=plt.subplots(rows,cols,figsize=(15,20))
for i in range(rows):
    for j in range (cols):
        column = data_train.columns[a]
        # Converting to wide dataframe
        data_wide = data_train.pivot(columns = 'Potability',values = column)
        # plotting multiple density plot
        data_wide.plot.kde(linewidth = 2, ax = ax[i,j],title=column)
        a+=1
plt.suptitle("Distribution Curve by Portability")


rows = 3
cols = 3
a=0
fig,ax=plt.subplots(rows,cols,figsize=(15,20))
for i in range(rows):
    for j in range (cols):
        column = data_train.columns[a]
        data_train.boxplot(column=column,by=['Potability'],ax=ax[i,j])
        a+=1
plt.suptitle("Boxplot by Portability")


# Compute correlation matrix
co_mtx = data_train.corr(numeric_only=True)

# Plot correlation heatmap
sns.heatmap(co_mtx, cmap="YlGnBu", annot=True, fmt='.2f')

plt.show()


data_train1 = data_train.copy()


data_train1 = data_train1.fillna(0)


data_train1.isnull().sum()


train,test = train_test_split(data_train1,test_size=0.2, random_state=25)

x_train = train.drop(columns=["Potability"])
y_train = train["Potability"]
x_test = test.drop(columns=["Potability"])
y_test = test["Potability"]


# Standardization
sc = StandardScaler() 
scaled = sc.fit_transform(x_train)

x_train = pd.DataFrame(scaled, columns=x_train.columns)
print(x_train)


model =[]
model.append(tree.DecisionTreeClassifier()) 
model.append(RandomForestClassifier())
model.append(SVC(kernel = "linear"))
model.append(KNeighborsClassifier(n_neighbors=3))
model.append(LogisticRegression())


model_score=[]


def modelscore(model,x,y):
    for i in range(len(model)):
        score=cross_val_score(model[i],x,y,cv=10)
        score = score.mean()
        model_score.append(score)
        print (f"Model {i} {model[i]}: {score}" )
        
modelscore(model,x_train,y_train)


model_name = [type(m).__name__ for m in model]

plt.figure(figsize=(12, 6))
plt.bar(model_name,model_score)
plt.title('Model Score')
plt.xlabel('Model')
plt.ylabel('Score')

plt.show()


model = RandomForestClassifier()


# Standardization
sc = StandardScaler() 
scaled = sc.fit_transform(x_test)
x_test = pd.DataFrame(scaled, columns=x_test.columns)

#prediction using highest score model
model = model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred))


data_test.info()


data_test_x = data_test.drop(columns=["id"])


data_test_x.isnull().sum()


data_test_x = data_test_x.fillna(0)


# Standardization
sc = StandardScaler() 
scaled = sc.fit_transform(data_test_x)

data_test_x = pd.DataFrame(scaled, columns=data_test_x.columns)
print(data_test_x)


potability = model.predict(data_test_x)


my_submission = pd.DataFrame({'Id': data_test["id"], 'Potability': potability})
my_submission.to_csv('submission-potability.csv', index=False)

print("Submission file created.")

