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


train=pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


train.head()


test.head()


#Checking for null values in train data

train.isnull().sum()


#Checking for null values in test data

test.isnull().sum()


train.shape


test.shape


#Dropping duplicates

train=train.dropna()
test=test.dropna()


train.shape #no duplicates


test.shape #no duplicates


#Identifying different types of fertilizers

train['Fertilizer Name'].value_counts()


train['Soil Type'].value_counts()


train['Crop Type'].value_counts()


#Understanding about distribution types

train['Fertilizer Name'].value_counts().plot(kind='bar',title="Fertilizers Types")


train['Soil Type'].value_counts().plot(kind='bar',title="Soil Types")


train['Crop Type'].value_counts().plot(kind='bar',title="Crop Types")


train.head()


train.describe()


train.info()


#Encoding since all are nominal data performing one-hot encoding

train=pd.get_dummies(train, columns=['Soil Type','Crop Type'],drop_first=True)


train.head()


#Feature Scaling

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_cols=['Temparature','Humidity','Moisture','Nitrogen','Potassium','Phosphorous']
train[scaled_cols]=scaler.fit_transform(train[scaled_cols])


y=train['Fertilizer Name']
x=train.drop(columns=['Fertilizer Name'])
x.head()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train = le.fit_transform(y)
y_train


# Training XGBoost

import xgboost as xgb

model=xgb.XGBClassifier(
    objective="multi:softprob",  
    num_class=len(le.classes_),  
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42
)

model.fit(x, y_train)


test=pd.get_dummies(test, columns=['Soil Type','Crop Type'],drop_first=True)


test[scaled_cols]=scaler.fit_transform(test[scaled_cols])


probs = model.predict_proba(test) 

# Getting top 3 predictions
top3_indices = np.argsort(probs, axis=1)[:, -3:][:, ::-1]  
top3_labels = le.inverse_transform(top3_indices.flatten()).reshape(top3_indices.shape)


submission = pd.DataFrame({
    "id": test["id"], 
    "Fertilizer Name": [' '.join(row) for row in top3_labels]
})

submission.to_csv("submission.csv", index=False)


df=pd.read_csv("submission.csv")
df.head()




