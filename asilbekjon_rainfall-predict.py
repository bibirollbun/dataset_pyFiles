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
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import GradientBoostingClassifier



train=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')


train.info()


train.describe()


train.rainfall.value_counts()


train.drop('id',axis=1,inplace=True)


train


train.corrwith(train['rainfall']).abs().sort_values(ascending=False)


maj_class = train[train['rainfall'] == 1]
min_class = train[train['rainfall'] == 0]
# Oversample the minority class
min_oversampled = resample(min_class,
                                replace=True,
                                n_samples=len(maj_class),
                                random_state=42)
# Combine the majority class with the oversampled minority class
data_balanced = pd.concat([maj_class, min_oversampled])


data_balanced.rainfall.value_counts()


X=data_balanced.drop('rainfall',axis=1)
y=data_balanced['rainfall']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=42)


scaler=StandardScaler()
X_train_prepared=scaler.fit_transform(X_train)
X_test_prepared=scaler.transform(X_test)


DT = DecisionTreeClassifier()
DT.fit(X_train_prepared, y_train)
Y_pred = DT.predict(X_test_prepared)
y_proba = DT.predict_proba(X_test_prepared)[:, 1]
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print(f"Model aniqliligi: {accuracy_score(y_test, Y_pred)}")
print(classification_report(y_test, Y_pred))
## confusion matrix
conf_mat = confusion_matrix(y_test, Y_pred)
sns.heatmap(conf_mat, annot=True,fmt="g")
plt.show()

## ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, Y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ROC curve')
display.plot()
plt.show()


RF = RandomForestClassifier(n_estimators=200,
    class_weight={0: 1, 1: 2},
    random_state=42)
RF.fit(X_train_prepared, y_train)

# prediction
Y_pred2 = RF.predict(X_test_prepared)
y_proba2= RF.predict_proba(X_test_prepared)[:, 1]
print("ROC AUC:", roc_auc_score(y_test, y_proba2))
# Model Accuracy
print(f"Model aniqliligi: {accuracy_score(y_test, Y_pred2)}")
print(classification_report(y_test, Y_pred2))

## confusion matrix
conf_mat = confusion_matrix(y_test, Y_pred2)
sns.heatmap(conf_mat, annot=True,fmt="g")
plt.show()

## ROC curve
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_proba2)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ROC curve')
display.plot()
plt.show()


from sklearn.metrics import precision_recall_curve

# Probabilistik chiqishlar
y_proba = RF.predict_proba(X_test_prepared)[:, 1]

# Thresholdni aniqlash va tasvirlash
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

plt.plot(thresholds, recalls[:-1], "b-", label="Recall")
plt.plot(thresholds, precisions[:-1], "g-", label="Precision")
plt.xlabel("Threshold")
plt.legend()
plt.title("Threshold vs Precision & Recall")
plt.grid(True)
plt.show()


# Misol: threshold = 0.3 (siz tanlagan)
from sklearn.metrics import recall_score
threshold = 0.3
y_pred_custom = (y_proba >= threshold).astype(int)

print("Recall (custom threshold):", recall_score(y_test, y_pred_custom))
print(classification_report(y_test, y_pred_custom))



RFC = RandomForestClassifier(
    n_estimators=200,
    class_weight={0: 1, 1: 2},
    random_state=42
)
RFC.fit(X_train_prepared, y_train)

# Ehtimollik va threshold asosida prediktsiya
y_proba_weighted = RFC.predict_proba(X_test_prepared)[:, 1]
y_pred_weighted = (y_proba_weighted >= 0.3).astype(int)

print(classification_report(y_test, y_pred_weighted))



xgb = XGBClassifier()
xgb.fit(X_train_prepared, y_train)

# Prediction 
Y_pred3 = xgb.predict(X_test_prepared)
y_proba3= RF.predict_proba(X_test_prepared)[:, 1]
print("ROC AUC:", roc_auc_score(y_test, y_proba3))
print(f"Model aniqliligi: {accuracy_score(y_test, Y_pred3)}")

print(classification_report(y_test, Y_pred3))

## confusion matrix
conf_mat = confusion_matrix(y_test, Y_pred3)
sns.heatmap(conf_mat, annot=True,fmt="g")
plt.show()

## ROC curve
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, Y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ROC curve')
display.plot()
plt.show()


test=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


test_id=test.id


test.drop('id',axis=1,inplace=True)


test.isnull().sum()


test['winddirection'] = test['winddirection'].fillna(test['winddirection'].mean())


test_prepared=scaler.transform(test)
test_proba=RFC.predict_proba(test_prepared)
test_positive_proba = test_proba[:, 1]  
submission = pd.DataFrame({'id': test_id, 'rainfall': test_positive_proba})
submission.to_csv('submission_rainfall1.csv', index=False)


submission




