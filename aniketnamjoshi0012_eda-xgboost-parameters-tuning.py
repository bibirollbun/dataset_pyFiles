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


train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


train_df


train_df.isna().sum(), train_df.shape


train_df.columns

for col in train_df.columns:
    if col != "id":
        print(train_df[col].value_counts())


import matplotlib.pyplot as plt
import seaborn as sns
col = train_df.columns
plt.plot(size=(16, 12))
sns.boxplot(x=train_df["Personality"], y=train_df[col[1]])


sns.boxplot(x=train_df["Personality"], y=train_df[col[3]])


sns.boxplot(x=train_df["Personality"], y=train_df[col[4]])


sns.boxplot(x=train_df["Personality"], y=train_df[col[6]])


sns.boxplot(x=train_df["Personality"], y=train_df[col[7]])


import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# Loading data
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


X_train = train.drop(['id', 'Personality'], axis=1)
y_train = train['Personality']
X_test = test.drop(['id'], axis=1)


num_cols = X_train.select_dtypes(include='number').columns
cat_cols = X_train.select_dtypes(include='object').columns


# Numeric Imputation
num_imputer = SimpleImputer(strategy='median')
X_train_num = pd.DataFrame(num_imputer.fit_transform(X_train[num_cols]), columns=num_cols)
X_test_num = pd.DataFrame(num_imputer.transform(X_test[num_cols]), columns=num_cols)


# Categorical imputation
cat_imputer = SimpleImputer(strategy='most_frequent')
X_train_cat = pd.DataFrame(cat_imputer.fit_transform(X_train[cat_cols]), columns=cat_cols)
X_test_cat = pd.DataFrame(cat_imputer.transform(X_test[cat_cols]), columns=cat_cols)

# Categorical encoding
for col in cat_cols:
    le = LabelEncoder()
    X_train_cat[col] = le.fit_transform(X_train_cat[col])
    X_test_cat[col] = le.transform(X_test_cat[col])



# Target encoding
le_target = LabelEncoder()
y_train_enc = le_target.fit_transform(y_train)  # Extrovert/Introvert → 0/1


X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)
X_test_final = pd.concat([X_test_num, X_test_cat], axis=1)


# Model
model = XGBClassifier(
    n_estimators=500, learning_rate=0.03, max_depth=7,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    use_label_encoder=False, eval_metric='logloss'
)
model.fit(X_train_final, y_train_enc)


# Prediction
y_pred = model.predict(X_test_final)
y_pred_label = le_target.inverse_transform(y_pred)


submission = pd.DataFrame({'id': test['id'], 'Personality': y_pred_label})
submission.to_csv('submission.csv', index=False)

