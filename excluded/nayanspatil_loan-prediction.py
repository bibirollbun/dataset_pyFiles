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


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


train.head()


test.head()


train.columns


test.columns


train.head()


train.info()


train.describe()


train.isnull().sum()


categorical = train.select_dtypes(include='object').columns
numerical = train.select_dtypes(exclude='object').columns


print(f'Catergorical columns names are {categorical} \n Numerical columns name are {numerical}')


for i in categorical:
    print(train[i].value_counts())
    print('======================')


import seaborn as sns
import matplotlib.pyplot as plt


for i in categorical:
    sns.countplot(x=i, data=train)
    plt.xticks(rotation=45, ha='right')
    plt.show()


for i in numerical:
    sns.histplot(x=i, data=train,kde=True, bins=20)
    plt.xticks(rotation=45, ha='right')
    plt.show()


for i in numerical:
    sns.boxplot(x=i, data=train)
    plt.show()


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


encoder = LabelEncoder()
for i in categorical:
    train[i] = encoder.fit_transform(train[i])
    test[i] = encoder.transform(test[i])


train.head()


test.head()


train.info()


col = [ 'annual_income', 'debt_to_income_ratio', 'credit_score','loan_amount', 'interest_rate', 'gender', 
       'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']


plt.figure(figsize=(11,8))
sns.heatmap(train.corr(), annot = True)
plt.show()


X = train.drop(['id', 'loan_paid_back'], axis=1)
y = train['loan_paid_back']


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    
    "AdaBoost": AdaBoostClassifier(random_state=42),
    
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}


results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"\nðŸ”¹ {name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))


param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 4, 5, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'min_child_weight': [1, 3, 5, 7],
    'scale_pos_weight': [1, 2, 3, 4]
}


from sklearn.model_selection import RandomizedSearchCV
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    scoring='accuracy',
    n_iter=30,             # Number of random combinations to try
    cv=3,                  # 3-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
print("âœ… Best Parameters:", random_search.best_params_)
print("âœ… Best CV Accuracy:", random_search.best_score_)


best_xgb = random_search.best_estimator_

y_pred = best_xgb.predict(X_test)

print("\nðŸŽ¯ Tuned XGBoost Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


test.head()


X_test = test.drop('id',axis=1)


loan_status = best_xgb.predict_proba(X_test)[:, 1]
loan_status


submission = pd.DataFrame({
    "id": test.id,
    "loan_status": np.round(loan_status,1)
})
submission


submission.to_csv("submission.csv", index=False)




