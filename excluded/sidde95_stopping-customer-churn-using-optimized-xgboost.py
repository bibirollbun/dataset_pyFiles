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


train_df = pd.read_csv('/kaggle/input/playground-series-s4e1/train.csv')
train_df.sample(5)


test_df = pd.read_csv("/kaggle/input/playground-series-s4e1/test.csv")
test_df.sample(5)


sample_df = pd.read_csv("/kaggle/input/playground-series-s4e1/sample_submission.csv")
sample_df.sample(3)


sample_df.Exited.unique()


train_df.sample(3)


train_df.info()


train_df.isnull().sum()


# Removing unnecessary columns from the dataset, which has no impact on the dataset 
train_df = train_df.drop(columns = ['id', 'CustomerId', 'Surname'], axis = 1)
train_df.sample()


categorical_columns = train_df.select_dtypes(include = 'O').columns
numerical_columns = train_df.select_dtypes(exclude = 'O').columns

print(f"There are {len(categorical_columns)} categorical columns and {len(numerical_columns)} numerical columns")


import matplotlib.pyplot as plt
import seaborn as sns


train_df.Geography



# Categorical Plot
plt.figure(figsize= (15, 3))

def countplot():
    for i in range(len(categorical_columns)):
        plt.subplot(1, 2, i+1)
        sns.countplot(data = train_df, y = categorical_columns[i])
        plt.title(f"{categorical_columns[i]} plot")

    plt.tight_layout()    
    plt.show()

countplot()


# Numerical Plot

plt.figure(figsize= (15, 10))

def boxplot():
    for i in range(len(numerical_columns)):
        plt.subplot(3, 3, i+1)
        sns.boxplot(data = train_df, x = numerical_columns[i])
        plt.title(f"{numerical_columns[i]} plot")

    plt.tight_layout()    
    plt.show()

boxplot()


# Removing outliers

train_df = train_df[train_df.Age <= 85]


plt.subplot(121)
plt.pie(train_df.HasCrCard.value_counts(), labels = train_df.HasCrCard.value_counts().index, autopct = "%1.1f%%")
plt.title('HasCrCard plot')

plt.subplot(122)
plt.pie(train_df.Exited.value_counts(), labels = train_df.Exited.value_counts().index, autopct = "%1.1f%%")
plt.title('Exited')

plt.show()


train_df.sample()


train_df.Age = train_df.Age.astype(int)
train_df.HasCrCard = train_df.HasCrCard.astype(int)
train_df.IsActiveMember = train_df.IsActiveMember.astype(int)

train_df.Gender = train_df.Gender.replace({'Male': 1, 'Female': 0}) # Male = 1 and Female = 0

from sklearn.model_selection import train_test_split

X = train_df.drop('Exited', axis = 1)
y = train_df.Exited

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 9)

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

ohe = OneHotEncoder(drop = "first")
scaler = StandardScaler()

preprocessing = ColumnTransformer(
    [
        ('OneHotEncoder', ohe, X_train.select_dtypes(include = "O").columns),
        ('StandardScaler', scaler, X_train.select_dtypes(exclude = 'O').columns)
    ]
)

X_train_scaled = preprocessing.fit_transform(X_train)
X_test_scaled = preprocessing.transform(X_test)



X_train_scaled


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

models = {
          "Logistic Regression": LogisticRegression(),
          "Decision Tree": DecisionTreeClassifier(),
          "Random Forest": RandomForestClassifier(),
          "AdaBoost": AdaBoostClassifier(),
          "Gradient Boost": GradientBoostingClassifier(),
          "XGBoost": XGBClassifier(),
          "KNN": KNeighborsClassifier()
         }

result_list = []
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    ar_score = accuracy_score(y_test, y_pred)
    pre_score = precision_score(y_test, y_pred)
    rec_score = recall_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred)
    roc_score = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:,1])

    metrics_dict = {
    'model name': model_name,
    'accuracy score': ar_score,
    'precision score': pre_score,
    'recall score': rec_score,
    'f1-score': f_score,
    'roc_auc_score': roc_score 
    }

    result_list.append(metrics_dict)

metric_df = pd.DataFrame(result_list)
metric_df


metric_df.sort_values(by = ['accuracy score', 'roc_auc_score'])


from sklearn.model_selection import GridSearchCV, StratifiedKFold

gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0]
}

xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv = StratifiedKFold(5)

gridcv_models = [
    ("Gradient Boost Classifier", GradientBoostingClassifier(), gb_params),
    ("XGB Classifier", XGBClassifier(), xgb_params)
]


for model_name, model, params in gridcv_models:
    gridcv = GridSearchCV(estimator= model, cv = cv, scoring = 'roc_auc', verbose= 1, n_jobs = -1, param_grid= params)
    gridcv.fit(X_train_scaled, y_train)
    
    print(f"---------------{model_name}---------------")
    print(gridcv.best_params_)
    print(gridcv.best_estimator_)


hyper_param_tuned_models = {
    "Gradient Boost Classifier": GradientBoostingClassifier(learning_rate = 0.05, max_depth = 5, n_estimators = 200, subsample = 0.8),
    "XG Boost Classifier": XGBClassifier(colsample_bytree = 1, learning_rate = 0.05, max_depth = 5, n_estimators = 200, subsample = 0.7)
}

result_list = []
for model_name, model in hyper_param_tuned_models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    ar_score = accuracy_score(y_test, y_pred)
    pre_score = precision_score(y_test, y_pred)
    rec_score = recall_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred)
    roc_score = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:,1])

    hyper_param_metrics_dict = {
    'model name': model_name,
    'accuracy score': ar_score,
    'precision score': pre_score,
    'recall score': rec_score,
    'f1-score': f_score,
    'roc_auc_score': roc_score 
    }

    result_list.append(hyper_param_metrics_dict)

hyper_param_metrics_dict = pd.DataFrame(result_list)
hyper_param_metrics_dict


test_df.sample(5)


test_df.isnull().sum()


# Dropping unnecessary columns
test_df_copy = test_df.drop(['id','Surname', 'CustomerId'], axis = 1 )

# Data Preprocessing
test_df_copy.Gender = test_df_copy.Gender.replace({"Male": 1, "Female": 0}) 
test_df_copy.Age = train_df.Age.astype(int)
test_df_copy.HasCrCard = train_df.HasCrCard.astype(int)
test_df_copy.IsActiveMember = train_df.IsActiveMember.astype(int)
test_df_scaled = preprocessing.transform(test_df_copy)

# Model Prediction
model = XGBClassifier(colsample_bytree = 1, learning_rate = 0.05, max_depth = 5, n_estimators = 200, subsample = 0.7)
model.fit(X_train_scaled, y_train)
predictions = model.predict_proba(test_df_scaled)
predictions



submission = pd.DataFrame({"id": test_df.id,"Exited": predictions[:, 1]})
submission.head()


submission.to_csv("submission.csv", index = False)
print("Submission saved!")







