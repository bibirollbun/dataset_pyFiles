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


train = pd.read_csv("/kaggle/input/binaryclassificationwithabankchurndataset/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/binaryclassificationwithabankchurndataset/test.csv", index_col="id")
submit = pd.read_csv("/kaggle/input/binaryclassificationwithabankchurndataset/sample_submission.csv")

train.head()


train.duplicated().sum() #duplikatlar tekshirildi


train.Geography.value_counts(), train.Gender.value_counts(), train.Exited.value_counts() #kategorikal ustunlar tekshirildi


test.drop(['CustomerId', 'Surname'], axis=1, inplace=True)
train.drop(['CustomerId', 'Surname'], axis=1, inplace=True) #keraksiz ustunlar o'chirildi


train.info()


train.describe()


train = train.astype({
    'Exited': pd.Int8Dtype(),
    'IsActiveMember': pd.Int8Dtype(),
    'HasCrCard': pd.Int8Dtype()
})


test = test.astype({
    'IsActiveMember': pd.Int8Dtype(),
    'HasCrCard': pd.Int8Dtype()
})


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


one_hot = OneHotEncoder(dtype=np.int8, sparse_output=False ) #drop = 'first' ma'lumot kamligi uchun shart emas
stan_scal = StandardScaler()
col_trans = ColumnTransformer([
    ('stan_scal', stan_scal, ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']),
    ('one_hot', one_hot, ['Geography', 'Gender'])
], remainder='passthrough').set_output(transform='pandas')


trans_df = col_trans.fit_transform(train)
test_df = col_trans.fit_transform(test)
trans_df


trans_df.describe()


from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score


model_RF = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model_GB = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model_KNN = KNeighborsClassifier(n_neighbors=5)
model_SVC = SVC(probability=True, kernel='rbf', C=1.0)
model_LR = LogisticRegression()

models = [model_LR, model_SVC, model_KNN, model_GB, model_RF]


X_train = trans_df.drop(['remainder__Exited'], axis=1).copy()
y_train = trans_df['remainder__Exited'].copy()
X_test = test_df


for model in models:
  model.fit(X_train, y_train)
  y_pred = model.predict(X_train)

  print(f"Model: {model} \n")
  print(f"Accuracy: {accuracy_score(y_train, y_pred)} \n")
  print(f"Classification Report: {classification_report(y_train, y_pred)}")
  print("  "*25)
  print("*"*25)
  print("  "*25)


base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),  # Cheklangan chuqurlik
    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),  # Katta emas
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('svc', SVC(probability=True, kernel='rbf', C=1.0))
]

meta_model = LogisticRegression()

stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)


stacking_model.fit(X_train, y_train)

#train data uchun test
y_pred = stacking_model.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)
print(f"Stacking model accuracy: {accuracy:.4f}")


#Train uchun ROC AUC Score
y_proba_train = stacking_model.predict_proba(X_train)[:, 1]
roc_auc = roc_auc_score(y_train, y_proba_train)
print(f"ROC AUC Score: {roc_auc:.4f}")


#test data uchun test
y_stack_pred = stacking_model.predict_proba(X_test)[:, 1]


#kaggle uchun csv ga aylantirildi
ans = pd.DataFrame(y_stack_pred)
# ans.to_csv('answers.csv')
ans.columns = ['Exited']
ans['id'] = np.arange(15000, 25000 )
ans = ans[['id', 'Exited']]

#kaggle ushun csv. Mana shu kod bilan Kaggle ~93% aniqlikga erishildi
ans.to_csv('predict_4.csv', index=False)
ans

