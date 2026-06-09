!pip install catboost

import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn import metrics
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns


df = pd.read_csv("/kaggle/input/binaryclassificationwithabankchurndataset/train.csv")
df.sample(10)


df.shape


df.info()


df.describe()


df.isnull().sum()


df.duplicated().sum()


df.Exited.value_counts()


corr_matrix = df.select_dtypes(include='number').corr().abs()
corr_matrix.style.background_gradient(cmap='Blues')


df.hist(bins=50, figsize=(15,12))
plt.show()


# function to add new coluumns and encode categorial columns 
def transformer (x):
  le_gender = LabelEncoder()
  x['Gender']= le_gender.fit_transform(x['Gender'])
  le_geo = LabelEncoder()
  x['Geography']= le_geo.fit_transform(x['Geography'])
  x['Product_per_Age']=x['NumOfProducts']*100/x['Age']
  x['Credit_per_Age']=x['CreditScore']*100/x['Age']
  x['Age_by_Product']=x['Age']/x['NumOfProducts']*100
  x.drop(['id', 'CustomerId', 'Surname'], axis=1, inplace=True)
  return x


def evaluate_model(y_test, y_pred, name):
  """ model evaluation """
  # model estimation
  print(f"\n--- {name} ---")
  print(classification_report(y_test, y_pred))
  print(f"Model accuracy: {accuracy_score(y_test,y_pred)*100:.1f}%")

  # confusion matrix
  conf_matrix = confusion_matrix(y_test, y_pred)
  sns.heatmap(conf_matrix, annot=True,fmt="g")
  plt.show()

  # roc_curve
  fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
  roc_auc = metrics.auc(fpr, tpr)
  display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ROC curve')
  display.plot()
  plt.show()


# check new added columns are well-correlated to label column 
df.select_dtypes(include='number').corr()[['Exited']].abs().sort_values(by='Exited', ascending=False)


transformer(df)


# scale dataset and split into train and test sets 
x = df.drop('Exited', axis=1)
y = df['Exited']
scaler = StandardScaler()
x = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(x_train, y_train)

y_pred = lr_model.predict(x_test)
evaluate_model(y_test, y_pred, 'Logistic Regression')


# Desicion Tree Classifier
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train, y_train)

y_pred = dt_model.predict(x_test)
evaluate_model(y_test, y_pred, 'Desicion Tree')


# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

y_pred = rf_model.predict(x_test)
evaluate_model(y_test, y_pred, 'Random Forest')


cat_model = CatBoostClassifier(
    iterations=600,          
    learning_rate=0.05,         
    depth=6,             
    l2_leaf_reg=3,
    auto_class_weights='Balanced',
    verbose=100,  )
cat_model.fit(x_train, y_train)

y_pred = cat_model.predict(x_test)
evaluate_model(y_test, y_pred, 'CatBoost')


# KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

knn_model=KNeighborsClassifier()
param_grid = {"n_neighbors" : np.arange(1,8)}
knn_gscv = GridSearchCV(knn_model, param_grid, cv=5)
knn_gscv.fit(x_train, y_train)
print (knn_gscv.best_params_)
best_n_neighbors = knn_gscv.best_params_['n_neighbors']

knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors)
knn_model.fit(x_train, y_train)

y_pred = knn_model.predict(x_test)
evaluate_model(y_test, y_pred, 'KNeighborsClassifier')


# SVC
svc_model = SVC(probability=True)
svc_model.fit(x_train, y_train)

y_pred = svc_model.predict(x_test)
evaluate_model(y_test, y_pred, 'SVC')


# xgb
xgb_model = XGBClassifier()
xgb_model.fit(x_train, y_train)

y_pred = xgb_model.predict(x_test)
evaluate_model(y_test, y_pred, 'XGB')


test_df = pd.read_csv("/kaggle/input/binaryclassificationwithabankchurndataset/test.csv")
test_df.sample(10)


# take id of dataframe for the next use
id_df = test_df['id']


# use functions to transform data 
X_submission = transformer(test_df)
X_submission = scaler.transform(X_submission)


X_submission


# predict for test dataset
X = X_submission
y_probs = cat_model.predict_proba(X)[:,1]


# create output file
submission = pd.DataFrame({
    'id': id_df,
    'Exited': y_probs
})

submission.to_csv('submission.csv', index=False)

