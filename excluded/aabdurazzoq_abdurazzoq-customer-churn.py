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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics 
from catboost import CatBoostClassifier, Pool
from catboost.utils import eval_metric


url="/kaggle/input/binaryclassificationwithabankchurndataset/train.csv"
df = pd.read_csv(url)
df.sample(10)


df.shape


df.info()


df.describe()


df.columns


null = df.isnull().sum().sum()
print(f'Null Count in Train: {null}')


duplicates = df.duplicated().sum()
print(f'Duplicates: {duplicates}')


df['Exited'].value_counts()


corr_matrix = df.select_dtypes(include='number').corr().abs()
corr_matrix.style.background_gradient(cmap='coolwarm')


df.hist(bins=50, figsize=(20,15))
plt.show()


# Encode categorical variables
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

le_geo = LabelEncoder()
df['Geography'] = le_geo.fit_transform(df['Geography'])

# Create new features
df['AgeBalanceRatio'] = df['Balance'] / (df['Age'] + 1)
df['IsSenior'] = (df['Age'] > 50).astype(int)
df['HasHighBalance'] = (df['Balance'] > 100000).astype(int)
df['ProductsPerYear'] = df['NumOfProducts'] / (df['Tenure'] + 1)
df['CreditScorePerProduct'] = df['CreditScore'] / (df['NumOfProducts'] + 1)
df['Age_Products_Ratio'] = df['Age'] / (df['NumOfProducts'] + 1)
df['Products_Per_Age'] = df['NumOfProducts'] * 100 / df['Age']
df['Age_x_Products'] = df['Age'] * df['NumOfProducts']
df['Active_by_CreditCard'] = df['HasCrCard'] * df['IsActiveMember']
df['Products_Per_Tenure'] =  df['Tenure'] / df['NumOfProducts']


# Faqat raqamli ustunlarni tanlab olamiz
numeric_df = df.select_dtypes(include='number')

# Faqat Exited bilan bog'liq korrelyatsiyalarni olamiz
exited_corr = numeric_df.corr()[['Exited']].abs().sort_values(by='Exited', ascending=False)

# Korrelyatsiya jadvalini vizualizatsiya qilamiz
exited_corr.style.background_gradient(cmap='coolwarm')


# Drop unnecessary columns
df = df.drop(['id', 'CustomerId', 'Surname'], axis=1)


# Split data
X = df.drop('Exited', axis=1)
y = df['Exited']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# contain all estimators in the function
def evaluate_model(y_test, y_pred, name):
    # Model estimation 
    print(f"\n--- {name} ---")
    print(metrics.classification_report(y_test, y_pred))
    print(f"Model accuracy: {metrics.accuracy_score(y_test,y_pred)*100:.1f}%")

    # confusion matrix
    conf_mat = metrics.confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_mat, annot=True,fmt="g")
    plt.show()

    # roc curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ROC curve')
    display.plot()
    plt.show()


# Logistic regression
LR_model = LogisticRegression(max_iter=1000)
LR_model.fit(X_train, y_train)

y_pred = LR_model.predict(X_test)
evaluate_model(y_test, y_pred, 'Logistic regression')


# Random Forest
RF_model = RandomForestClassifier(n_estimators=200, random_state=42)
RF_model.fit(X_train, y_train)

y_pred = RF_model.predict(X_test)
evaluate_model(y_test, y_pred, 'Random Forest')


# XGBClassifier
XG_model = XGBClassifier()
XG_model.fit(X_train, y_train)

y_pred = RF_model.predict(X_test)
evaluate_model(y_test, y_pred, 'XGBClassifier')


# LGBMClassifier
lgbParams = {'n_estimators': 1000,
             'max_depth': 25, 
             'learning_rate': 0.025,
             'min_child_weight': 3.43,
             'min_child_samples': 216, 
             'subsample': 0.782,
             'subsample_freq': 4, 
             'colsample_bytree': 0.29, 
             'num_leaves': 21}
LG_model = LGBMClassifier(**lgbParams)
LG_model.fit(X_train, y_train)

y_pred = LG_model.predict(X_test)
evaluate_model(y_test, y_pred, 'LGBMClassifier')


CAT_model = CatBoostClassifier(eval_metric='AUC',learning_rate=0.022,iterations=1000)
CAT_model.fit(X_train, y_train)

y_pred = CAT_model.predict(X_test)
evaluate_model(y_test, y_pred, 'CAT_model')


# KNeighborsClassifier
KNN_model = KNeighborsClassifier(n_neighbors=10)
KNN_model.fit(X_train, y_train)

y_pred = KNN_model.predict(X_test)
evaluate_model(y_test, y_pred, 'KNeighborsClassifier')


# SVC
svcModel = SVC(probability=True, random_state=42)
svcModel.fit(X_train, y_train)

y_pred = svcModel.predict(X_test)
evaluate_model(y_test, y_pred, 'SVC')


url="/kaggle/input/binaryclassificationwithabankchurndataset/test.csv"
test_df = pd.read_csv(url)
test_df.sample(10)


ids = test_df['id']


X_submission = test_df


X_submission['AgeBalanceRatio'] = X_submission['Balance'] / (X_submission['Age'] + 1)
X_submission['IsSenior'] = (X_submission['Age'] > 50).astype(int)
X_submission['HasHighBalance'] = (X_submission['Balance'] > 100000).astype(int)
X_submission['ProductsPerYear'] = X_submission['NumOfProducts'] / (X_submission['Tenure'] + 1)
X_submission['CreditScorePerProduct'] = X_submission['CreditScore'] / (X_submission['NumOfProducts'] + 1)
X_submission['Age_Products_Ratio'] = X_submission['Age'] / (X_submission['NumOfProducts'] + 1)
X_submission['Products_Per_Age'] = X_submission['NumOfProducts'] * 100 / X_submission['Age']
X_submission['Age_x_Products'] = X_submission['Age'] * X_submission['NumOfProducts']
X_submission['Active_by_CreditCard'] = X_submission['HasCrCard'] * X_submission['IsActiveMember']
X_submission['Products_Per_Tenure'] =  X_submission['Tenure'] / X_submission['NumOfProducts']



X_submission = X_submission.drop(['id', 'CustomerId', 'Surname'], axis=1)


X_submission


X_submission['Geography'] = le_geo.transform(X_submission['Geography'])
X_submission['Gender'] = le_gender.transform(X_submission['Gender'])


X_submission


X = scaler.transform(X_submission)
y_probs = CAT_model.predict_proba(X)[:, 1]
y_probs


submission = pd.DataFrame({
    'id': ids,
    'Exited': y_probs
})

submission.to_csv('CAT_model.csv', index=False)

