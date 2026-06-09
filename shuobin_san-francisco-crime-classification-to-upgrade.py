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


crime_df =  pd.read_csv("/kaggle/input/sf-crime/train.csv.zip")


print("犯罪类型数：",len(crime_df["Category"].unique()))
print("犯罪类型：", crime_df["Category"].unique())
from sklearn.preprocessing import LabelEncoder 

import matplotlib.pyplot as plt
import seaborn as sns


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(crime_df['Category'])
# Dates：在这天的几点（白天or晚上），月份，年份可能与Category有关。
# 可以将Dates字段拆分为年、月、日、小时等特征，分析不同时间段、月份、年份的犯罪类型分布。
crime_df['Dates'] = pd.to_datetime(crime_df['Dates'])
crime_df['Year'] = crime_df['Dates'].dt.year
crime_df['Month'] = crime_df['Dates'].dt.month
crime_df['Hour'] = crime_df['Dates'].dt.hour


from sklearn.preprocessing import LabelEncoder
crime_df["PdDistrict"] = LabelEncoder().fit_transform(crime_df["PdDistrict"])


# 将Address分为两个新特征，如果Address是路口，其值就作为新特征intersection的值，如果Address是Block，就提取对应的街道名或者大道名作为新特征street的值。然后对intersection和street进行类别编码处理。
from sklearn.preprocessing import LabelEncoder

print(len(crime_df["Address"].unique()))
# 提取 intersection 和 street 特征
crime_df['intersection'] = crime_df['Address'].apply(lambda x: x if '/' in x else None)
crime_df['street'] = crime_df['Address'].apply(
    lambda x: x.split('Block of ')[1] if 'Block of ' in x else (None if '/' in x else x)
)

# 类别编码
crime_df['intersection_encoded'] = LabelEncoder().fit_transform(
    crime_df['intersection'].fillna('NA')
)
crime_df['street_encoded'] = LabelEncoder().fit_transform(
    crime_df['street'].fillna('NA')
)



X = crime_df.loc[:, ['Year', 'Month', 'Hour','PdDistrict', 'street_encoded', 'intersection_encoded', 'X', 'Y']]


from sklearn.feature_selection import SelectKBest
print(X.info())
selector = SelectKBest(k="all").fit(X, y)
selected_features = selector.get_feature_names_out()
print(selected_features)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 

X_train, X_test, y_train, y_test = train_test_split(X.loc[:, selected_features], y, test_size=0.2, random_state=0)

model = RandomForestClassifier(n_estimators=100, class_weight="balanced")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# You must submit a csv file with the incident id, all candidate class names, and a probability for each class.



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, columns=model.classes_)
print(cm_df)


test_crime_df =  pd.read_csv("/kaggle/input/sf-crime/test.csv.zip")


# 1. 对 test_crime_df 做与训练集相同的特征工程
test_crime_df['Dates'] = pd.to_datetime(test_crime_df['Dates'])
test_crime_df['Year'] = test_crime_df['Dates'].dt.year
test_crime_df['Month'] = test_crime_df['Dates'].dt.month
test_crime_df['Hour'] = test_crime_df['Dates'].dt.hour

test_crime_df['intersection'] = test_crime_df['Address'].apply(lambda x: x if '/' in x else None)
test_crime_df['street'] = test_crime_df['Address'].apply(
    lambda x: x.split('Block of ')[1] if 'Block of ' in x else (None if '/' in x else x)
)

# 用训练集的编码器进行transform
test_crime_df['PdDistrict'] = LabelEncoder().fit_transform(test_crime_df["PdDistrict"])

test_crime_df['intersection'] = test_crime_df['Address'].apply(lambda x: x if '/' in x else None)
test_crime_df['street'] = test_crime_df['Address'].apply(
    lambda x: x.split('Block of ')[1] if 'Block of ' in x else (None if '/' in x else x)
)

# 类别编码
test_crime_df['intersection_encoded'] = LabelEncoder().fit_transform(
    test_crime_df['intersection'].fillna('NA')
)
test_crime_df['street_encoded'] = LabelEncoder().fit_transform(
    test_crime_df['street'].fillna('NA')
)

# 2. 选取与训练集一致的特征
X_submit = test_crime_df.loc[:, selected_features]

# 3. 预测概率
y_submit_proba = model.predict_proba(X_submit)

# 4. 构建提交文件
submit_df = pd.DataFrame(y_submit_proba, columns=label_encoder.classes_)
submit_df.insert(0, 'Id', test_crime_df['Id'])
submit_df.to_csv('submission.csv', index=False)

