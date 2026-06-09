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
import numpy as np


df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


df.head()


df.dtypes


df.info()


df.shape


df.isna().sum()


df.columns


df.education.unique()


df.corr(numeric_only=True)


df.dtypes


test = test.drop(columns=['id'])


X = df.drop(columns=['y','id'])
y = df['y']


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values='unknown',strategy='most_frequent').set_output(transform='pandas')
X = imputer.fit_transform(X)
test = imputer.transform(test)


from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler , OrdinalEncoder , LabelEncoder

ct = ColumnTransformer([
    ('onehot',OneHotEncoder(),['job','marital','default','housing','loan','contact','month','poutcome']),
    ('ordianl',OrdinalEncoder(categories=[['primary','secondary', 'tertiary']]),['education'])
],remainder='passthrough')

le = LabelEncoder()
y = le.fit_transform(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


X_train.head()


X_train.poutcome.unique()


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

models = {
    # "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    # "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    # "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
}

results = {}

for name, model in models.items():
    pipe = Pipeline(steps=[
        ('ct',ct),
        ('scale',StandardScaler()),
        ('model',model)
    ])

    pipe.fit(X_train, y_train)   
    preds_proba = pipe.predict_proba(X_test)[:, 1]  
    
    auc = roc_auc_score(y_test, preds_proba)  
    results[name] = auc
    print(f"{name} ROC-AUC Score: {auc:.4f}")

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
print("\nModel Performance (Best to Worst):")
for name, auc in sorted_results:
    print(f"{name}: {auc:.4f}")


model = sorted_results[0]


# model =  XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)

# pipe = Pipeline(steps=[
#         ('ct',ct),
#         ('scale',StandardScaler()),
#         ('model',model)
#     ])


pipe.fit(X,y)


test_pred = pipe.predict_proba(test)[:, 1]


sub = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
submission = pd.DataFrame({
    "id": sub['id'],
    "y": test_pred
})

submission.to_csv("submission.csv", index=False)

