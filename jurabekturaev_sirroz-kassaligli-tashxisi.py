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


from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, log_loss
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


df_train = pd.read_csv("/kaggle/input/multiclassificationtask/train.csv", index_col=0)
df_test = pd.read_csv("/kaggle/input/multiclassificationtask/test.csv", index_col=0)


df_train.head()


df_train['Status'].value_counts()


# Status ustunidagi Y qiymatdan boshqa qiymatlarni olish
df_train = df_train[df_train['Status'] != 'Y']


df_train.isnull().sum()


# tushib qolgan qiymatlar(NaN) vizual ko'rinishda
import missingno
missingno.matrix(df_train)


# Kategorik ustunlardagi NaN qiymatlarni ustundagi eng ko'p uchragan qiymatlar bilan to'ldirish
cat_cols = ['Drug','Sex','Ascites','Hepatomegaly','Spiders','Edema']
df_train_cat = df_train[cat_cols]
simple_imputer = SimpleImputer(strategy='most_frequent')
df_train_cat_encoded = simple_imputer.fit_transform(df_train_cat)
df_train[cat_cols] = df_train_cat_encoded


# Numeric Pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])


# Full Pipeline
num_cols = df_train.drop(['Drug','Sex','Ascites','Hepatomegaly','Spiders','Edema','Status'], axis=1)

full_pipeline = ColumnTransformer([
    ('cat', OrdinalEncoder(), cat_cols),
    ('num', num_pipeline, list(num_cols))
])


# X, Y taqsimoti
X = df_train.drop('Status', axis=1)
Y = df_train['Status']


# LabelEncoder yordamida target qiymatlar sonli qiymatlarga aylantirildi
le = LabelEncoder()
Y_encoded = le.fit_transform(Y)


X_prepared = full_pipeline.fit_transform(X)


# train/test split
x_train, x_test, y_train, y_test = train_test_split(X_prepared, Y_encoded, test_size=0.2, random_state=42, stratify=Y_encoded)


# Random Forest modeli
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(x_train, y_train)


# bashorat axboroti
y_pred = rf_model.predict(x_test)
print(f"Classification report:\n{classification_report(y_test, y_pred, target_names=le.classes_)}")


# predict_proba
y_proba = np.clip(rf_model.predict_proba(x_test), 1e-15, 1 - 1e-15)


# Logarthmic Loss hisoblandi
loss = log_loss(y_test, y_proba)
loss


df_test.head()


df_test.isnull().sum()


# Kategorik ustunlardagi NaN qiymatlarni ustundagi eng ko'p uchragan qiymatlar bilan to'ldirish
cat_cols = ['Drug','Sex','Ascites','Hepatomegaly','Spiders','Edema']
df_test_cat = df_test[cat_cols]
df_test_cat_encoded = simple_imputer.transform(df_test_cat)
df_test[cat_cols] = df_test_cat_encoded


df_test_prepared = full_pipeline.transform(df_test)


# predict_proba olish 
proba = np.clip(rf_model.predict_proba(df_test_prepared), 1e-15, 1 - 1e-15)


submission = pd.DataFrame({'id': df_test.index, "Status_C": proba[:, 0], "Status_CL": proba[:, 1], "Status_D": proba[:, 2]})
submission


submission.to_csv("Submission.csv", index=False)




