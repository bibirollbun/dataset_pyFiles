# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier


from sklearn.pipeline import Pipeline
from sklearn import metrics


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/multiclassificationtask/train.csv', index_col=0)
train_df.head()


train_df.shape


train_df.describe()


train_df.info()


# checking NaN values 
train_df.isna().sum()


# split dataset X, y
X = train_df.drop('Status', axis=1)
y = train_df[['Status']]


train_df['Status'].value_counts()


encoder = LabelEncoder()
y = encoder.fit_transform(y)


X.info()


# automate processes with pipiline 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# extract categorical columns
# cat_attributes = ['Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Drug']
cat_attributes = X.select_dtypes(include=['object']).columns.to_list()
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', OneHotEncoder(handle_unknown='ignore')),
])

# numeric columns
num_cols = X.select_dtypes(include=['float64']).columns.to_list()
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# combine
preprocessor = ColumnTransformer([
    ('categorical', cat_pipeline, cat_attributes),
    ('numerical', num_pipeline, num_cols)
])


X_prepared = preprocessor.fit_transform(X)


# train, test split 
X_train, X_test, y_train, y_test = train_test_split(X_prepared, y, test_size=0.2, random_state=42)


def estimate_model(y_test, y_pred, y_proba, model_name):
    print(f"Model: {model_name}")
    print(f"Accuracy Score: {metrics.accuracy_score(y_test, y_pred):.4f}")
    print("Log Loss:", metrics.log_loss(y_test, y_proba, labels=[0, 1, 2]))
    print(f"Classification Report:\n{metrics.classification_report(y_test, y_pred, zero_division=0)}")
    print('='*50)



# using multiple models for preduction
models = {
    "SVM": SVC(kernel='linear', probability=True, decision_function_shape='ovo'),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y))),
    "LogisticRegression": LogisticRegression(multi_class='ovr', max_iter=500),
    "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500),
    "DecisionTree": DecisionTreeClassifier(),
}


for name, model in models.items():
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    estimate_model(y_test, y_pred, y_proba, name)


from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier

estimators = [
    ("rf", RandomForestClassifier()),
    ("xgb", XGBClassifier()),
    ("svm", SVC(probability=True)),
    ("mlp", MLPClassifier(max_iter=1000)),
]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    passthrough=True
)

stack.fit(X_train, y_train)

y_pred = stack.predict(X_test)
y_proba = stack.predict_proba(X_test)

# Evaluate
acc = metrics.accuracy_score(y_test, y_pred)
log_loss = metrics.log_loss(y_test, y_proba, labels=[0, 1, 2])
print(f"Stacked Model Accuracy: {acc:.4f}")
print(f'log-loss score: {log_loss}')


# as i got highest score in xgb_model, i'm gonna predict with that 
test_df = pd.read_csv('/kaggle/input/multiclassificationtask/test.csv')
test_df_prepared = preprocessor.transform(test_df)


model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)
print("Log Loss:", metrics.log_loss(y_test, y_proba, labels=[0, 1, 2]))


submission = pd.DataFrame(y_proba, columns=['Status_C', 'Status_CL', 'Status_D'])
submission['id'] = test_df['id']
submission = submission[['id', 'Status_C', 'Status_CL', 'Status_D']]
submission.to_csv('submission.csv', index=False)




