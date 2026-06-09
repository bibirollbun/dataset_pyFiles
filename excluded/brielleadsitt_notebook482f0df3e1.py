# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df_train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv',nrows=2000)
df_test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

df_train.dropna(inplace=True)
df_test.dropna(inplace=True) # will not get used in train/test split for test evaluation of classifiers

df_train['date'] = pd.to_datetime(df_train['date'], format='%Y-%m-%d')  # Convert to datetime
df_train['date'] = df_train['date'].apply(lambda x: x.toordinal())  # Converts to the number of days since 0001-01-01

unique_country_train = df_train['country'].unique()
country_dict_train = {key:val for val,key in enumerate(unique_country_train)}
df_train['country'] = df_train['country'].apply(lambda x: country_dict_train[x])

unique_store_train = df_train['store'].unique()
store_dict_train = {key:val for val,key in enumerate(unique_store_train)}
df_train['store'] = df_train['store'].apply(lambda x: store_dict_train[x])

unique_product_train = df_train['product'].unique()
product_dict_train = {key:val for val,key in enumerate(unique_product_train)}
df_train['product'] = df_train['product'].apply(lambda x: product_dict_train[x])

df_test['date'] = pd.to_datetime(df_test['date'], format='%Y-%m-%d')  # Convert to datetime
df_test['date'] = df_test['date'].apply(lambda x: x.toordinal())  # Converts to the number of days since 0001-01-01

unique_country_test = df_test['country'].unique()
country_dict_test = {key:val for val,key in enumerate(unique_country_test)}
df_test['country'] = df_test['country'].apply(lambda x: country_dict_test[x])

unique_store_test = df_test['store'].unique()
store_dict_test = {key:val for val,key in enumerate(unique_store_test)}
df_test['store'] = df_test['store'].apply(lambda x: store_dict_test[x])

unique_product_test = df_test['product'].unique()
product_dict_test = {key:val for val,key in enumerate(unique_product_test)}
df_test['product'] = df_test['product'].apply(lambda x: product_dict_test[x])

# Select features (X) and target (y)
X = df_train.drop(columns=["num_sold"])  # All columns except 'num_sold'
y = df_train["num_sold"]                 # Only the 'num_sold' column

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf_rfc = RandomForestClassifier(n_estimators=2000, max_depth=500)
clf_rfc.fit(X_train.drop(columns=['date']), y_train)
y_pred_rfc = clf_rfc.predict(X_test.drop(columns=['date']))
print("Accuracy of Random Forest Classifier without date:", accuracy_score(y_test, y_pred_rfc))
print("MAPE score of RFC without date: ", mean_absolute_percentage_error(y_test, y_pred_rfc))

# submission
y_pred_submit = clf_rfc.predict(df_test.drop(columns=['date']))

df_submit = df_test[['id']].copy()
df_submit.loc[:, 'num_sold'] = y_pred_submit

if os.path.exists('/kaggle/working/submission.csv'):
    os.remove('/kaggle/working/submission.csv')

df_submit.to_csv('/kaggle/working/submission.csv', index=False)

for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Create and train the model
# clf_gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
# clf_gbc.fit(X_train, y_train)
# y_pred_gbc = clf_gbc.predict(X_test)
# print("Accuracy of Gradient Boosting Classifier:", accuracy_score(y_test, y_pred_gbc))

# for i in ['store','product']:
# clf_lr = LogisticRegression(penalty='l2', max_iter=1000, solver='saga')
# clf_lr.fit(X_train, y_train)
# y_pred_lr = clf_lr.predict(X_test)
# print("Accuracy of Logistic Regressor:", accuracy_score(y_test, y_pred_lr))
    
#     clf_rfc = RandomForestClassifier(n_estimators=20, max_depth=5)
#     clf_rfc.fit(X_train[i].values.reshape(-1, 1), y_train)
#     y_pred_rfc = clf_rfc.predict(X_test[i].values.reshape(-1, 1))
#     print("Accuracy of Random Forest Classifier for column:", i, accuracy_score(y_test, y_pred_rfc))
    
# clf_knn = KNeighborsClassifier(n_neighbors=100)
# clf_knn.fit(X_train, y_train)
# y_pred_knn = clf_knn.predict(X_test)
# print("Accuracy of knn:", accuracy_score(y_test,y_pred_knn))

# clf_nb = GaussianNB()
# clf_nb.fit(X_train[i].values.reshape(-1, 1), y_train)
# y_pred_nb = clf_nb.predict(X_test[i].values.reshape(-1, 1))
# print("Accuracy of nb:", accuracy_score(y_test,y_pred_nb))
#     clf_lr = LogisticRegression(penalty='l2', max_iter=10, solver='saga')
#     clf_lr.fit(X_train[i].values.reshape(-1, 1), y_train)
#     y_pred_lr = clf_lr.predict(X_test[i].values.reshape(-1, 1))
#     print("Accuracy of Logistic Regressor with penalty 12 and max_iter 10 for column ", i, ':', accuracy_score(y_test, y_pred_lr))

#     clf_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
#     clf_dt.fit(X_train[i].values.reshape(-1, 1), y_train)
#     y_pred_dt = clf_dt.predict(X_test[i].values.reshape(-1, 1))
#     print("Accuracy of Decision Tree Classifier with max_depth 5 for column ", i, ':', accuracy_score(y_test, y_pred_dt))

# clf_dt = DecisionTreeClassifier(max_depth=50, random_state=42)
# clf_dt.fit(X_train.drop(columns=['date','id']), y_train)
# y_pred_dt = clf_dt.predict(X_test.drop(columns=['date','id']))
# print("Accuracy of Decision Tree Classifier with max_depth 5 for column ", accuracy_score(y_test, y_pred_dt))

# clf_nb = GaussianNB()
# clf_nb.fit(X_train, y_train)
# y_pred_nb = clf_nb.predict(X_test)
# print("Accuracy of nb for column ", accuracy_score(y_test,y_pred_nb))
# col_list = ['id','date','country','store','product']
# for i in col_list:
#     clf_rc = RidgeClassifier()
#     clf_rc.fit(X_train[i].values.reshape(-1, 1), y_train)
#     y_pred_rc = clf_rc.predict(X_test[i].values.reshape(-1, 1))
#     print("Accuracy of rc for column ", i, ':', accuracy_score(y_test,y_pred_rc))
    
#     clf_nb = GaussianNB()
#     clf_nb.fit(X_train[i].values.reshape(-1, 1), y_train)
#     y_pred_nb = clf_nb.predict(X_test[i].values.reshape(-1, 1))
#     print("Accuracy of nb for column ", i, ':', accuracy_score(y_test,y_pred_nb))

# col_list = ['id','date','country','store','product']
# for i in col_list:
#     not_list = [item for item in col_list if item != i]
#     for j in not_list:
#         plt.scatter(X_train[i],X_train[j])
#         plt.xlabel(i)
#         plt.ylabel(j)
#         plt.title(i)
#         plt.show()

# col_list = ['id','date','country','store','product']
# for i in col_list:
#     not_list = [item for item in col_list if item != i]
#     for j in not_list:
#         scatter = plt.scatter(X_train[i],y_train,c=X_train[j],cmap='viridis')
#         cbar = plt.colorbar(scatter)
#         cbar.set_label('Color-coded variable:'+j)
#         plt.xlabel(i)
#         plt.ylabel('num_sold')
#         plt.title(i)
#         plt.show()
# clf_rfc = RandomForestClassifier(n_estimators=2000, max_depth=500)
# clf_rfc.fit(X_train.drop(columns=['date']), y_train)
# y_pred_rfc = clf_rfc.predict(X_test.drop(columns=['date']))
# print("Accuracy of Random Forest Classifier withOUT date:", accuracy_score(y_test, y_pred_rfc))




