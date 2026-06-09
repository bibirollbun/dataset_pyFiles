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


# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore Warnings
from warnings import filterwarnings
filterwarnings("ignore")

from scipy import stats

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score




train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train.head()


print(f"Train : {train.shape[0]} rows and {train.shape[1]} cols")
print(f"Test : {test.shape[0]} rows and {test.shape[1]} cols")


print(f"*************************** Train *********************************")
print(train.info())
print(f"*************************** Test *********************************")
print(test.info())


test.isnull().sum()


train.describe()


# No duplicate Values
train.duplicated().sum()


train = train.drop(columns = ["id"])


test["winddirection"].fillna(train["winddirection"].median(), inplace = True)


test.isnull().sum().sum()


# In day Column there are some wrongly noted values there should be 6 times each day
print(train.shape[0] / train["day"].nunique())
plt.barh(train["day"].unique(), train["day"].value_counts() )


np.tile(np.arange(1, 366), 6)


train["day"] = np.tile(np.arange(1, 366), 6)
plt.barh(train["day"].unique(), train["day"].value_counts())


X = train.drop(columns = ["rainfall"])
y = train["rainfall"]

ids = test["id"]
test = test.drop(columns = ["id"])

X.shape, y.shape, test.shape, ids.shape


minmax_scaler = MinMaxScaler()
yeo_trans = PowerTransformer(method = "yeo-johnson")
boxcox_trans = PowerTransformer(method = "box-cox")



trans = ColumnTransformer(transformers = [
                                    ("trans1", minmax_scaler, [0]),
                                    ("trans2", boxcox_trans, [2, 3, 4, 6, 9, 10]),
                                    ("trans3", yeo_trans, [1, 5, 7, 8])
                            ],
                          remainder = "passthrough")
trans


X = pd.DataFrame(trans.fit_transform(X), columns = X.columns)
test = pd.DataFrame(trans.transform(test), columns = test.columns)


sns.heatmap(pd.concat((X, y), axis = 1).corr(), annot = True, fmt = ".2f", linewidths = 0.5, cmap = "summer")


X = X.drop(columns = ["pressure", "maxtemp", "temparature", "cloud"])
test = test.drop(columns = ["pressure", "maxtemp", "temparature", "cloud"])





split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 55)
for train_index, test_index in split.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

X_train.shape, X_test.shape, y_train.shape, y_test.shape


algos = {
    "Logistic Regression" : LogisticRegression(),
    "Decision Tree" : DecisionTreeClassifier(),
    "Random Forest" : RandomForestClassifier(),
    "Gradient Boosting" : GradientBoostingClassifier(),
    "Ada Boost" : AdaBoostClassifier(),
    "K nearest neighbours" : KNeighborsClassifier()
}
for name, algo in algos.items():
    cv_score = cross_val_score(estimator=algo, X=X, y=y, cv=10, scoring = "roc_auc").mean()
    print(name, cv_score)


knc = KNeighborsClassifier(n_neighbors = 10)
cross_val_score(knc, X, y, cv = 5, scoring = "roc_auc").mean()


knc = KNeighborsClassifier(n_neighbors = 10)
knc.fit(X_train, y_train)
y_pred = knc.predict(X_test)
roc_auc_score(y_test, y_pred)


lr = LogisticRegression(max_iter = 100)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
roc_auc_score(y_test, y_pred)


rf = RandomForestClassifier(max_depth = 7, n_estimators = 150, min_samples_split = 4, min_samples_leaf = 4)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
roc_auc_score(y_test, y_pred)


rf = RandomForestClassifier(max_depth = 7, n_estimators = 150, min_samples_split = 4, min_samples_leaf = 4)
cross_val_score(rf, X, y, cv = 5, scoring = "roc_auc").mean()


rf = RandomForestClassifier(max_depth = 7, n_estimators = 150, min_samples_split = 4, min_samples_leaf = 4)
rf.fit(X, y)
y_pred = rf.predict(test)
y_pred.shape


submission = pd.DataFrame(list(zip(ids, y_pred)), columns = ["id", "rainfall"])
submission


submission.to_csv("submission.csv", index = False)




