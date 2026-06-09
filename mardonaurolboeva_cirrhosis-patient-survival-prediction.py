# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,log_loss
from sklearn.pipeline import Pipeline
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df=pd.read_csv("/kaggle/input/multiclassificationtask/train.csv",index_col=0)
df.head()


df.info()


df.describe()


df.shape


df.isna().sum()


# X va y ajratish

X=df.drop(["Status"], axis=1)
y=df["Status"]


# target ustunni obj dan numeric ustunga o'tkazish(LabelEncoder)

y=LabelEncoder().fit_transform(y)


X.info()


X.columns


#nan qiymatlarni avtomatik to'ldirish, Categoric ustunlar(One_Hot encoding)

#cat_features=['Drug','Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']

cat_cols=X.select_dtypes(include=['object']).columns.to_list()
cat_pipeline=Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', OneHotEncoder(handle_unknown='ignore')),
])


num_cols=X.select_dtypes(include=['float64']).columns.to_list()
num_pipeline=Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])


preprocessor=ColumnTransformer([
    ('categorical', cat_pipeline, cat_cols),
    ('numerical', num_pipeline, num_cols)
])


X_prepared=preprocessor.fit_transform(X)


# Train_test_splite

X_train, X_test, y_train, y_test=train_test_split(X_prepared, y, test_size=0.2, random_state=42)


def estimate_model(y_test, y_pred, y_proba, model_name):
    print(f"Model: {model_name}")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Log Loss:", log_loss(y_test, y_proba,labels=[0,1,2]))
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
    print('='*50)
    


# Models

models={
    "SVM":SVC(kernel='linear', probability=True, decision_function_shape='ovo'),
    "RandomForest":RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y))),
    "LogisticRegression":LogisticRegression(multi_class='ovr', max_iter=500),
    "NeuralNetwork":MLPClassifier(hidden_layer_sizes=(100), max_iter=500),
    "DecisionTree": DecisionTreeClassifier(),

}


for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
    y_proba=model.predict_proba(X_test)
    estimate_model(y_test, y_pred, y_proba, name)


from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier

estimators=[
    ("rf", RandomForestClassifier()),
    ("xgb", XGBClassifier()),
    ("svm", SVC(probability=True)),
    ("mlp", MLPClassifier(max_iter=1000)),
]

stack=StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    passthrough=True
)

stack.fit(X_train, y_train)

y_pred=stack.predict(X_test)
y_proba=stack.predict_proba(X_test)

acc=accuracy_score(y_test, y_pred)
loss=log_loss(y_test, y_proba, labels=[0, 1, 2])
print(f"Stacked Model Accuracy:{acc:.4f}")
print(f"log-loss score:{log_loss}")


test_df=pd.read_csv("/kaggle/input/multiclassificationtask/test.csv")
test_df_prepared=preprocessor.transform(test_df)


model=XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

model.fit(X_train, y_train)
y_proba=model.predict_proba(X_test)
y_pred=model.predict(X_test)
print(f"Log Loss:", log_loss(y_test, y_proba,labels=[0,1,2]))


submission=pd.DataFrame(y_proba, columns=['Status_C','Status_CL', 'Status_D'])
submission['id']=test_df['id']
submission=submission[['id', 'Status_C','Status_CL' ,'Status_D',]]
submission.to_csv('submission.csv', index=False)




