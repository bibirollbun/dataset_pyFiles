# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
from sklearn.feature_selection import RFECV
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


main_dir = '/kaggle/input/playground-series-s5e3/'
train = pd.read_csv(main_dir + 'train.csv')
test = pd.read_csv(main_dir + 'test.csv')
train_df, test_df = train.copy(), test.copy()


train_df.info()


test_df.info()


train_df.describe().T


for col in train_df.columns[:-1]:
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=train_df, y=col, x='rainfall',  palette={1: 'blue', 0: 'cornflowerblue'})
    plt.subplot(1, 2, 2)
    sns.histplot(data=train_df, x=col, hue='rainfall', kde=True, palette={1: 'blue', 0: 'cornflowerblue'}, multiple='dodge')
    plt.show()


plt.figure(figsize=(12, 10))
sns.heatmap(data=train_df.corr(), annot=True, linewidths=0.2, cmap='Blues', vmax=1, vmin=-1);


def suppress_outliers(dataset, numeric_cols):
    data = dataset.copy()
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
    
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        data.loc[data[col] < lower_bound, col] = lower_bound
        data.loc[data[col] > upper_bound, col] = upper_bound

    return data


def temparature(dataset):
    data = dataset.copy()
    data["average_temparature"] = round((data["maxtemp"] + data["mintemp"] + data["temparature"])/3, 1)
    return data.drop(columns=["maxtemp", "mintemp", "temparature"], axis=1)


def month(dataset):
    data = dataset.copy()
    data["month"] = data["day"].apply(lambda x: datetime.strptime(str(x), "%j").month)
    return data

def season(dataset):
    data = dataset.copy()
    data =  month(data)
    winter = [12, 1, 2]
    spring = [3, 4, 5]
    summer = [6, 7, 8]
    data["season"] = data["month"].apply(lambda x: 1 if x in winter \
                                                    else (2 if x in spring\
                                                    else(3 if x in summer else 4)))
    return data


def preprocess(dataset, cols):
    data = dataset.copy()
    data = suppress_outliers(data, cols)
    data = temparature(data)
    data = season(data)
    return data


numerical_cols = ['pressure', 'maxtemp', 'temparature', 'mintemp',
       'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
       'windspeed']

train_df = preprocess(train_df, numerical_cols)


train_df.head(5)


train_df.tail(5)


train_df.describe().T


x = train_df.drop(columns=["id", "day", "rainfall"], axis=1)
y = train_df[["rainfall"]]


sc = MinMaxScaler()
cols = x.drop(columns=["month", "season"]).columns
x[cols] = sc.fit_transform(x[cols])


x = x.drop(columns=['winddirection', 'month'])


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)


lr = LogisticRegression(random_state=42)
lr.fit(x_train, y_train)


pred = lr.predict_proba(x_test)
roc_auc_score(y_test, pred[:, 1])


y_pred = lr.predict(x_test)


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()


logit_roc_auc = roc_auc_score(y_test, lr.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.legend(loc="lower right")
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()


param_grid = [
    {'penalty':['l1','l2','elasticnet','none'],
    'C' : np.logspace(-4,4,20),
    'solver': ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter'  : [100,1000,2500,5000]
}
]


clf = GridSearchCV(lr,param_grid = param_grid, cv = 3, verbose=True,n_jobs=-1)
clf


model_search = clf.fit(x, y)


best_model = model_search.best_estimator_


model_search.score(x, y)


y_pred_ = best_model.predict(x_test)


logit_roc_auc = roc_auc_score(y_test, best_model.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.legend(loc="lower right")
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()


cm = confusion_matrix(y_test, y_pred_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()


pred_ = best_model.predict_proba(x_test)
roc_auc_score(y_test, pred_[:, 1])


test_df.head()


test_df = preprocess(test_df, numerical_cols)


test_df


x = test_df.drop(columns=["id", "day"], axis=1)
cols = x.drop(columns=["month", "season"]).columns
x[cols] = sc.fit_transform(x[cols])


x = x.drop(columns=['winddirection', 'month'])


y_pred_ = best_model.predict(x)


submission = pd.DataFrame()
submission['id'] = test_df['id']
submission['rainfall'] = y_pred_
submission


submission.to_csv('submission.csv', index=False)




