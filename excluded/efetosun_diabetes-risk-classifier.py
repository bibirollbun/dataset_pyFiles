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


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score



train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")


train.head(3)


train = train.drop(columns=["id"])


train.isnull().sum()


print(train.shape)


train.columns


train.head(5)


diabetes = train[train["diagnosed_diabetes"] == 1]
no_diabetes = train[train["diagnosed_diabetes"] == 0]

plt.figure(figsize=(10,6))
plt.hist(diabetes["age"],bins = 20 , alpha = 0.6 , label = "diabetes")
plt.hist(no_diabetes["age"],bins = 20 , alpha = 0.6 , label = "no_diabetes")
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Distribution by Diabetes Status")
plt.legend()
plt.show()


plt.figure(figsize=(8,6))
sns.boxplot(x = "diagnosed_diabetes" , y = "bmi" , data = train)
plt.xlabel("Diabetes (0=No, 1=Yes)")
plt.ylabel("BMI")
plt.title("BMI by Diabetes Status")
plt.show()


plt.figure(figsize=(6,4))
train.groupby("family_history_diabetes")["diagnosed_diabetes"].mean().plot(kind="bar")
plt.ylabel("Proportion of Diabetes")
plt.xlabel("Family History of Diabetes (0=No, 1=Yes)")
plt.title("Diabetes Proportion by Family History")
plt.show()



train.head(3)


train.info()


categorical_cols = [
    'gender', 'ethnicity', 'education_level', 
    'income_level', 'smoking_status', 'employment_status'
]


train.head(3)


train[categorical_cols] = train[categorical_cols].astype(str)
test[categorical_cols] = test[categorical_cols].astype(str)



encoder = OneHotEncoder(drop="first", sparse_output=False)

encoded_train = pd.DataFrame(
    encoder.fit_transform(train[categorical_cols]),
    columns=encoder.get_feature_names_out(categorical_cols),
    index=train.index
)

encoded_test = pd.DataFrame(
    encoder.transform(test[categorical_cols]),
    columns=encoder.get_feature_names_out(categorical_cols),
    index=test.index
)




numeric_cols = [col for col in train.columns if col not in categorical_cols + ["diagnosed_diabetes"]]

# Final train ve test
final_train = pd.concat([train[numeric_cols], encoded_train], axis=1)
final_test  = pd.concat([test[numeric_cols], encoded_test], axis=1)



train


train.info()


numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns

plt.figure(figsize=(20,15))
sns.heatmap(train[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap (Numeric Only)")
plt.show()



numeric_cols = [col for col in train.columns if col not in categorical_cols + ["diagnosed_diabetes"]]

final_train = pd.concat([train[numeric_cols], encoded_train], axis=1)
final_test  = pd.concat([test[numeric_cols],  encoded_test], axis=1)



y = train["diagnosed_diabetes"]
x = final_train


model = RandomForestClassifier(
    n_estimators=100,   
    max_depth=10,       
    n_jobs=-1,          
    random_state=42
)
model.fit(final_train, train["diagnosed_diabetes"])





x_train, x_test, y_train, y_test = train_test_split(
    final_train, 
    train["diagnosed_diabetes"], 
    train_size=0.70, 
    random_state=42
)



model.score(x_test , y_test)


train.head(3)


preds = model.predict(final_test)



preds


submission = sample_submission.copy()  
submission["diagnosed_diabetes"] = preds  
submission.to_csv("submission.csv", index=False)  









