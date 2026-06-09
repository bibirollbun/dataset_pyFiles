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


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df


df.isnull().sum()


le = LabelEncoder()
df['Stage_fear'] = le.fit_transform(df['Stage_fear'])
df['Drained_after_socializing'] = le.fit_transform(df['Drained_after_socializing'])
df['Personality'] = le.fit_transform(df['Personality'])
df


plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()


df.corr()



df['Stage_fear'] = df['Stage_fear'].replace(2, np.nan)
df['Drained_after_socializing'] = df['Drained_after_socializing'].replace(2, np.nan)
df


# plt.figure(figsize=(10, 8))
# sns.heatmap(df_imputed.corr(), annot=True, cmap='coolwarm')
# plt.show()


X = df.drop('Personality', axis=1)
y = df['Personality']
display(X)
display(y)


model = LogisticRegression(class_weight='balanced')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


imputer = IterativeImputer(estimator=RandomForestRegressor(), random_state=42)

# Fit and transform
X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns
)
X_train = X_train.round()

X_test = X_test = pd.DataFrame(
    imputer.fit_transform(X_test),
    columns=X_test.columns
)
X_test = X_test.round()


X_train


X_test


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred)
accuracy_train = accuracy_score(y_train, model.predict(X_train))
r2_test = r2_score(y_test, y_pred)
r2_train = r2_score(y_train, model.predict(X_train))
print(accuracy_test, accuracy_train, r2_test, r2_train)


from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print("CV Accuracy:", scores.mean())



from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)



import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix

model = lgb.LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
test_df


test_df['Stage_fear'] = le.fit_transform(test_df['Stage_fear'])
test_df['Drained_after_socializing'] = le.fit_transform(test_df['Drained_after_socializing'])
test_df


test_df['Stage_fear'] = test_df['Stage_fear'].replace(2, np.nan)
test_df['Drained_after_socializing'] = test_df['Drained_after_socializing'].replace(2, np.nan)
test_df


imputed_test_df = pd.DataFrame(
    imputer.fit_transform(test_df),
    columns=test_df.columns
)
imputed_test_df = imputed_test_df.round() 


imputed_test_df


df1 = pd.DataFrame()
df1['id'] = imputed_test_df['id']
df1['Personality'] = model.predict(imputed_test_df)
df1['Personality'] = df1['Personality'].astype(int)
df1['Personality'] = df1['Personality'].map({0: 'Extrovert', 1: 'Introvert'})


df1.to_csv('submission.csv', index=False)




