# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics 

import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_ur = '/kaggle/input/binaryclassificationwithabankchurndataset/train.csv'
train_df = pd.read_csv(train_ur, index_col=0)


# train data
print('train data\'s shape', train_df.shape)
train_df.head()


# now working with train data 
train_df.info()


# correlation
corr_data = train_df.drop(['CustomerId','Surname', 'Geography', 'Gender'], axis=1).corr()
corr_data.style.background_gradient('winter_r')


train_df.corrwith(train_df['Exited'], numeric_only=True).abs().sort_values(ascending=False)


return_rate = train_df['Exited'].value_counts()/len(train_df)*100
print(return_rate)


plt.figure(figsize=(5,5))
plt.pie(return_rate, autopct="%1.1f%%", labels=['Returned','Defaulted'])
plt.show()


fig, axes = plt.subplots(1,2, figsize=(15,5))

sns.countplot(data=train_df, x='Geography', ax=axes[0], hue='Exited')
axes[0].set_title('Loan Status By Country', size=15)

sns.countplot(data=train_df, x='Gender', ax=axes[1], hue='Exited')
axes[1].set_title('Loan Status By Gender')

fig.suptitle("Loan Status Analysis", size=30)
plt.show()


train_df.info()


train_df['Gender'] = train_df['Gender'].map({'Male':1, 'Female':0})
train_df['Geography'] = train_df['Geography'].map({'France':0,'Spain':1,'Germany':2})


X = train_df.drop(['Exited', 'Surname', 'CustomerId'], axis=1)
y = train_df['Exited']


scaler = StandardScaler()
X = scaler.fit_transform(X) 


# split data set to train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# contain all estimators in the function
def estimate_model(y_test, y_pred):
    # Model estimation 
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
LR_model = LogisticRegression()
LR_model.fit(X_train, y_train)

y_pred = LR_model.predict(X_test)
estimate_model(y_test, y_pred)


# KNN
# finding the best key
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
param_grid = {'n_neighbors': np.arange(1, 25)}
cv_grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
cv_grid.fit(X_train, y_train)


cv_grid.best_params_['n_neighbors']


# knn
knn = KNeighborsClassifier(n_neighbors=cv_grid.best_params_['n_neighbors'])
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
estimate_model(y_test, y_pred)


# support vector machine (SVM)
svm = SVC()
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
estimate_model(y_test, y_pred)


# tree model 
# finding the best hyperparametrs
param_grid_tree_mod = {
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': np.arange(2,10),
    'min_samples_leaf': np.arange(1,10)
}

cv_grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid_tree_mod, cv=5)
cv_grid_tree.fit(X_train, y_train)


cv_grid_tree.best_params_


tree_model = DecisionTreeClassifier(
    max_depth=cv_grid_tree.best_params_['max_depth'],
    min_samples_leaf=cv_grid_tree.best_params_['min_samples_leaf'],
    min_samples_split=cv_grid_tree.best_params_['min_samples_split'],
)
tree_model.fit(X_train, y_train)

y_pred = tree_model.predict(X_test)
estimate_model(y_test, y_pred)


# Random forest 
# finding best number of estimators 
param_grid_ran_for = {'n_estimators': np.arange(1, 25)}
cv_grid_ran_for = GridSearchCV(RandomForestClassifier(), param_grid=param_grid_ran_for, cv=5)
cv_grid_ran_for.fit(X_train, y_train)


# best n number of est
cv_grid_ran_for.best_params_['n_estimators']


RV_model = RandomForestClassifier(n_estimators=cv_grid_ran_for.best_params_['n_estimators'])
RV_model.fit(X_train, y_train)
y_pred = RV_model.predict(X_test)
estimate_model(y_test, y_pred)


xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
estimate_model(y_test, y_pred)


test_url = '/kaggle/input/binaryclassificationwithabankchurndataset/test.csv'
test_df = pd.read_csv(test_url, index_col=0)
test_df.head()


test_df['Gender'] = test_df['Gender'].map({'Male':1, 'Female':0})
test_df['Geography'] = test_df['Geography'].map({'France':0,'Spain':1,'Germany':2})
test_df.sample()


X_submit = test_df.drop(['Surname', 'CustomerId'], axis=1)
X_submit = scaler.fit_transform(X_submit)


test_pred = svm.predict(X_submit)

# Save predictions to submission.csv
submission = pd.DataFrame({"prediction": test_pred})
submission.to_csv("submission.csv", index=False)

print("Predictions saved to submission.csv")




