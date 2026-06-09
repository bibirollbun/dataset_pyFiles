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


df_train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df_train


df_train.isna().sum()


df_train.describe()


df_train.duplicated().sum()


df_train.info()


df_train.shape


import matplotlib.pyplot as plt
import seaborn as sns


# numeric features (histograms, boxplots)
num_features = df_train.select_dtypes(include=['int64']).columns
num_features = num_features.drop('id')
plt.figure(figsize=(11, 6))

for col in num_features:
    plt.figure(figsize=(6,4))
    sns.histplot(df_train[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()



plt.figure(figsize=(6,4))
sns.countplot(x=df_train['y'])
plt.title("Target Distribution")
plt.xlabel("Target")
plt.ylabel("Count")
plt.show()


y = df_train['y']
y


df_test =pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


x = df_train.drop('y', axis=1)
y = df_train['y']


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y,test_size=0.2, random_state=42)


# Encoding
from sklearn.preprocessing import LabelEncoder
label_cols = ['marital', 'education', 'default', 'housing', 'loan', 'month']
le = LabelEncoder()
for col in label_cols:
    x_train[col] = le.fit_transform(x_train[col])
    x_val[col]   = le.transform(x_val[col])
    df_test[col]  = le.transform(df_test[col])

freq_cols = ['job', 'contact', 'poutcome']
for col in freq_cols:
    freq_map = x_train[col].value_counts().to_dict()
    
    x_train[col] = x_train[col].map(freq_map)
    x_val[col]   = x_val[col].map(freq_map)
    df_test[col] = df_test[col].map(freq_map)



# scaling
from sklearn.preprocessing import RobustScaler

num_cols = x_train.select_dtypes(include=['int64', 'float64']).columns
num_cols = num_cols.drop('id')
scaler = RobustScaler()
x_train[num_cols] = scaler.fit_transform(x_train[num_cols])
x_val[num_cols]   = scaler.transform(x_val[num_cols])
df_test[num_cols] = scaler.transform(df_test[num_cols])



train_with_target = x_train.copy()
train_with_target["target"] = y_train
train_with_target = train_with_target.drop('id', axis=1)

corr = train_with_target.corr()

# heatmap
plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap between Features and Target")
plt.show()


print(train_with_target.columns)



from sklearn.linear_model import LogisticRegression

lg = LogisticRegression()
lg.fit(x_train,y_train)


pred = lg.predict_proba(df_test)


pred_val = lg.predict(x_val)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Accuracy:", accuracy_score(y_val, pred_val))
print("\nClassification Report:\n")
print(classification_report(y_val, pred_val))
cm = confusion_matrix(y_val, pred_val)
print("\nConfusion Matrix:\n", cm)


from xgboost import XGBClassifier
xgb_model = XGBClassifier(
    n_estimators=200,     
    learning_rate=0.1,   
    max_depth=6,         
    subsample=0.8,       
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss" 
)

xgb_model.fit(x_train, y_train)
pred_test = xgb_model.predict_proba(df_test)
pred_xgb = xgb_model.predict(x_val)


print("Accuracy:", accuracy_score(y_val, pred_xgb))
print("\nClassification Report:\n")
print(classification_report(y_val, pred_xgb))
cm = confusion_matrix(y_val, pred_xgb)
print("\nConfusion Matrix:\n", cm)


sub = pd.DataFrame({"id": df_test['id'], "y": pred_test[:,1]})
sub


sub.to_csv("submission.csv", index=False)




