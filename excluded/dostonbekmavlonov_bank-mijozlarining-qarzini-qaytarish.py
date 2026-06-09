import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



import pandas as pd
import os

# 1. Kaggle notebookdagi ma'lumotlar jildining asosiy yo'li
# Musobaqa nomi "binaryclassificationwithabankchurndataset"
base_path = '/kaggle/input/binaryclassificationwithabankchurndataset/'

# 2. Har bir faylning to'liq yo'lini aniqlaymiz
train_file_path = os.path.join(base_path, 'train.csv')
test_file_path = os.path.join(base_path, 'test.csv')
submission_file_path = os.path.join(base_path, 'sample_submission.csv')

# 3. Fayllarni o'qib, siz so'ragan o'zgaruvchilarga saqlaymiz
df_train = pd.read_csv(train_file_path)
df_test = pd.read_csv(test_file_path)
df_sub = pd.read_csv(submission_file_path)

# 4. Hammasi to'g'ri yuklanganini tekshirish uchun
print("--- df_train (O'qitish ma'lumotlari) ---")
print(df_train.head())
print("\n")

print("--- df_test (Test ma'lumotlari) ---")
print(df_test.head())
print("\n")

print("--- df_sub (Namuna submission fayli) ---")
print(df_sub.head())


# ma'lumotlarni tekshirish
df_train.head()


df_test.head()


# tozaligini ko'rish
df_train.isnull().sum()
df_test.isnull().sum()
df_sub.isnull().sum()


# drop customer id and surname
df_train.drop(['CustomerId', 'Surname'], axis=1, inplace=True)
df_test.drop(['CustomerId', 'Surname'], axis=1, inplace=True)


# labeling geography and gender
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_train['Geography'] = le.fit_transform(df_train['Geography'])
df_train['Gender'] = le.fit_transform(df_train['Gender'])
df_test['Geography'] = le.fit_transform(df_test['Geography'])
df_test['Gender'] = le.fit_transform(df_test['Gender'])


# correlation Exited
df_train.corr()['Exited'].abs().sort_values(ascending=False)


# creating good train set
cor = ['Age', 'NumOfProducts', 'IsActiveMember', 'Gender','Balance','Exited']
train = df_train[cor]
# train_set test_set
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(train, test_size=0.1, random_state=42)
x_train = train_set.drop('Exited', axis=1)
y_train = train_set['Exited']
x_test = test_set.drop('Exited', axis=1)
y_test = test_set['Exited']
cor1 = ['Age', 'NumOfProducts', 'IsActiveMember', 'Gender','Balance']
TEST = df_test[cor1] # for submission file
TEST_ID = df_test['id'] # for submission file



# standart scaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
TEST = sc.transform(TEST) # for submission file



# Lostic Reggerssion
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=0.1)
lr.fit(x_train, y_train)

# predict probablities
y_pred = lr.predict_proba(x_test)[:,1]


# best hyperparametr
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_search = GridSearchCV(lr, param_grid, cv=5)
grid_search.fit(x_train, y_train)
print("Best parameters:", grid_search.best_params_)


# model evaluation ROC
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)
print("ROC score:", roc_auc_score(y_test, y_pred))
# grapgh of ROC and y=x curve
from sklearn.metrics import roc_curve
fpr, tpr, threshold = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# Support vector machine
from sklearn.svm import SVC
# Modelni yaratishda 'probability=True' deb belgilaymiz
svm = SVC(kernel='rbf',probability=True, random_state=42)
svm.fit(x_train, y_train)

# predict probablities
y_pred = svm.predict_proba(x_test)[:,1]


# model evaluation ROC
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)
print("ROC score:", roc_auc_score(y_test, y_pred))
# grapgh of ROC and y=x curve
from sklearn.metrics import roc_curve
fpr, tpr, threshold = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=6,min_samples_split=8,random_state=42)
dt.fit(x_train, y_train)

# predict probablities
y_pred = dt.predict_proba(x_test)[:,1]


# finding best hyperparametr
from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10], 'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
grid_search = GridSearchCV(dt, param_grid, cv=5)
grid_search.fit(x_train, y_train)
print("Best parameters:", grid_search.best_params_)


# model evaluation ROC
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)
print("ROC score:", roc_auc_score(y_test, y_pred))
# grapgh of ROC and y=x curve
from sklearn.metrics import roc_curve
fpr, tpr, threshold = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50, max_depth=7, random_state=42)
rf.fit(x_train, y_train)

# predict probablities
y_pred = rf.predict_proba(x_test)[:,1]


# best hyperparametr
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [50, 100, 150, 200], 'max_depth': list(range(1, 11))}
grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(x_train, y_train)
print("Best parameters:", grid_search.best_params_)


# model evaluation ROC
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)
print("ROC score:", roc_auc_score(y_test, y_pred))
# grapgh of ROC and y=x curve
from sklearn.metrics import roc_curve
fpr, tpr, threshold = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


#XGB
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=150, max_depth=2,random_state=42)
xgb.fit(x_train, y_train)

# predict probablities
y_pred = xgb.predict_proba(x_test)[:,1]


# best hyperparametr
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [50, 100, 150,200], 'max_depth': list(range(1, 20))}
grid_search = GridSearchCV(xgb, param_grid, cv=5)
grid_search.fit(x_train, y_train)
print("Best parameters:", grid_search.best_params_)



# model evaluation ROC
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)
print("ROC score:", roc_auc_score(y_test, y_pred))
# grapgh of ROC and y=x curve
from sklearn.metrics import roc_curve
fpr, tpr, threshold = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


#XGB
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=150, max_depth=2,random_state=42)
xgb.fit(x_train, y_train)

# predict probablities
y_pred = xgb.predict_proba(TEST)[:,1]


# saving y_pred in csv file
y_pred_df = pd.DataFrame(y_pred, columns=['Exited'])

# columns id and Exited
y_pred_df['id'] = TEST_ID
y_pred_df = y_pred_df[['id', 'Exited']]

# saving csv file
y_pred_df.to_csv('sample_submission.csv', index=False)



y_pred_df.head()

