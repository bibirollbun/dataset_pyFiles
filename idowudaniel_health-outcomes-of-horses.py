# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv('/kaggle/input/playground-series-s3e22/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s3e22/test.csv')


df_train.head()


df_test.head()


df_train.info()


df_train.isnull().sum()


df_train['age'] = df_train['age'].map({'young': 0, 'adult': 1})
df_train['surgery'] = df_train['surgery'].map({'no': 0, 'yes': 1})
df_train['surgical_lesion'] = df_train['surgical_lesion'].map({'no': 0, 'yes': 1})
df_train['cp_data'] = df_train['cp_data'].map({'no': 0, 'yes': 1})

df_test['age'] = df_test['age'].map({'young': 0, 'adult': 1})
df_test['surgery'] = df_test['surgery'].map({'no': 0, 'yes': 1})
df_test['surgical_lesion'] = df_test['surgical_lesion'].map({'no': 0, 'yes': 1})
df_test['cp_data'] = df_test['cp_data'].map({'no': 0, 'yes': 1})


numeric_cols = df_train.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df_train.select_dtypes(include=['object']).columns

test_numeric_cols = df_test.select_dtypes(include=['float64', 'int64']).columns
test_categorical_cols = df_test.select_dtypes(include=['object']).columns


for col in numeric_cols:
    if df_train[col].isnull().sum() > 0:
        median_val = df_train[col].median()
        df_train[col].fillna(median_val, inplace=True)
        
for col in test_numeric_cols:
    if df_test[col].isnull().sum() > 0:
        median_val = df_test[col].median()
        df_test[col].fillna(median_val, inplace=True)


for col in categorical_cols:
    if df_train[col].isnull().sum() > 0:
        mode_val = df_train[col].mode()[0]
        df_train[col].fillna(mode_val, inplace=True)
        
for col in test_categorical_cols:
    if df_test[col].isnull().sum() > 0:
        mode_val = df_test[col].mode()[0]
        df_test[col].fillna(mode_val, inplace=True)


print(df_train.isnull().sum().sort_values(ascending=False).head())


df_train = df_train.drop(['id', 'hospital_number'], axis=1)
df_test = df_test.drop(['id', 'hospital_number'], axis=1)


for col in ['abdomo_appearance', 'abdomen', 'temp_of_extremities', 'peripheral_pulse', 'mucous_membrane', 'capillary_refill_time', 'pain', 'peristalsis', 'abdominal_distention', 'nasogastric_tube', 'nasogastric_reflux', 'rectal_exam_feces']:
    df_train[col].fillna('unknown', inplace=True)

df_train = pd.get_dummies(df_train, columns=['abdomo_appearance', 'abdomen', 'temp_of_extremities', 
                                             'peripheral_pulse', 'mucous_membrane', 
                                             'capillary_refill_time', 'pain', 'peristalsis', 'abdominal_distention', 'nasogastric_tube', 'nasogastric_reflux', 'rectal_exam_feces'], drop_first=True)

for col in ['abdomo_appearance', 'abdomen', 'temp_of_extremities', 'peripheral_pulse', 'mucous_membrane', 'capillary_refill_time', 'pain', 'peristalsis', 'abdominal_distention', 'nasogastric_tube', 'nasogastric_reflux', 'rectal_exam_feces']:
    df_test[col].fillna('unknown', inplace=True)

df_test = df_test.reindex(columns=df_train.columns.drop('outcome'), fill_value=0)


num_cols = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
test_num_cols = df_test.select_dtypes(include=['int64', 'float64']).columns.tolist()
scaler = StandardScaler()
scaler.fit(df_train[num_cols])


df_train[num_cols] = scaler.transform(df_train[num_cols])
df_test[num_cols] = scaler.transform(df_test[num_cols])


X = df_train.drop(['outcome'], axis=1)
y = df_train['outcome']


le = LabelEncoder()
y = le.fit_transform(df_train['outcome'])  


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.info())


log_model = LogisticRegression(class_weight='balanced', random_state=42)
log_model.fit(X_train, y_train)


y_pred_log = log_model.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred_log))
print(classification_report(y_val, y_pred_log, target_names=le.classes_))


rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)


y_pred_rf = rf_model.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred_rf))
print(classification_report(y_val, y_pred_rf, target_names=le.classes_))


df_submission = pd.read_csv('/kaggle/input/playground-series-s3e22/sample_submission.csv')
df_submission.head()


df_submission['outcome'] = le.inverse_transform(rf_model.predict(df_test))
df_submission.to_csv('submission.csv', index=False)

