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


train = pd.read_csv("/kaggle/input/multiclassificationtask/train.csv")
test = pd.read_csv("/kaggle/input/multiclassificationtask/test.csv")
submit = pd.read_csv("/kaggle/input/multiclassificationtask/sample_submission.csv")
train.head()


test.head()


submit.head()


print("train.shape: ", train.shape)
print("test.shape: ", test.shape)
print("submit.shape: ", submit.shape)


train.info()


test.info()


submit.info()


train.isnull().sum()/len(train)


test.isnull().sum()/len(test)


train[train.Drug.isna()]


for i in train:
    if train[i].dtype == 'object':
        print(f"\n\nUnique values in ''''{i}'''' column are {train[i].unique()}")
        print(train[i].value_counts())


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
y = label.fit_transform(train[['Status']])
y


X = train.drop('Status', axis=1)
X


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

cat_attributes = X.select_dtypes(include=['object']).columns.to_list()
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', OneHotEncoder(handle_unknown='ignore')),
])

num_cols = X.select_dtypes(include=['float64']).columns.to_list()
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('categorical', cat_pipeline, cat_attributes),
    ('numerical', num_pipeline, num_cols)
])


X = preprocessor.fit_transform(X)


pd.DataFrame(X).isnull().sum()


pd.DataFrame(X)


for i in test:
    if test[i].dtype == 'object':
        print(f"\n\nUnique values in ''''{i}'''' column are {test[i].unique()}")
        print(test[i].value_counts())


test = preprocessor.fit_transform(test)


pd.DataFrame(test).isnull().sum()


pd.DataFrame(X).describe().T


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state=12)


# robust = StandardScaler()
# X_train = robust.fit_transform(X_train)
# X_test = robust.transform(X_test)
# test = robust.transform(test)


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss


def estimate_model(y_test, y_pred, y_proba, model_name):
    print(f"Model: {model_name}")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print("Log Loss:", log_loss(y_test, y_proba, labels=[0, 1, 2]))
    print(f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
    print("       ##############################            \n\n")


model = RandomForestClassifier(n_estimators=128, max_features='sqrt',
                                                            random_state=101)
model.fit(X_train,y_train)
preds = model.predict(X_test)
y_proba = model.predict_proba(X_test)

estimate_model(y_test, preds, y_proba, "RandomForest")


classifier= XGBClassifier(n_estimators=64, max_depth=5, objective='multi:softmax', num_class=len(np.unique(y)))
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_proba = model.predict_proba(X_test)
estimate_model(y_test, y_pred, y_proba, "XGBClassifier")


svc = SVC(probability=True)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
y_proba = model.predict_proba(X_test)
estimate_model(y_test, y_pred, y_proba, "SVC")


clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = model.predict_proba(X_test)
estimate_model(y_test, y_pred, y_proba, "GaussianNB")


from sklearn.ensemble import GradientBoostingClassifier
base_models = [
    ('rf', RandomForestClassifier(n_estimators=32, random_state=42)),
    ('svm', SVC(probability=True)),
    ('xgb', XGBClassifier(n_estimators=64))
]

meta_model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                                              max_depth=4, random_state=0)

stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)
stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_test)
y_proba = stacking_model.predict_proba(X_test)
estimate_model(y_test, y_pred, y_proba, "StackingClassifier")


base_models = [
    ('rf', RandomForestClassifier(n_estimators=32, random_state=42)),
    ('svm', SVC(probability=True)),
    ('xgb', XGBClassifier(n_estimators=64))
]

meta_model = LogisticRegression(max_iter=500)

stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)
stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_test)
y_proba = stacking_model.predict_proba(X_test)
estimate_model(y_test, y_pred, y_proba, "StackingClassifier")


test.shape


y_proba = stacking_model.predict_proba(test)


submit.head()


submission = pd.DataFrame(y_proba, columns=['Status_C', 'Status_CL', 'Status_D'])


submission.head()


for i in submission:
    submit[i] = submission[i]

submit.head()


submit.to_csv('submission.csv', index=False)




