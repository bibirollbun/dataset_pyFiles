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


# Load libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error


train_data = pd.read_csv('/kaggle/input/playground-series-s3e16/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s3e16/test.csv')
subs_data = pd.read_csv('/kaggle/input/playground-series-s3e16/sample_submission.csv')


train_data.head()


test_data.head()


subs_data.head()


train_data.info()


train_data_missing = train_data.isna().mean() * 100
test_data_missing = test_data.isna().mean() * 100

print("Percentage missing value in Train Data")
print(train_data_missing)

print("\n Percentage missing value in Test Data")
print(test_data_missing)


train_data_noid = train_data.drop(['id'], axis=1)


train_data_noid.describe()


train_data_noid


cor_matrix = train_data_noid.corr(numeric_only=True)
sns.heatmap(cor_matrix, cmap="YlGnBu", annot=True)
plt.show()


print(cor_matrix["Age"].sort_values(ascending=True))


kolom = train_data_noid.columns
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12,12))

for i, ax in enumerate(axs.flatten()):
    ax.hist(train_data_noid[kolom[i]])
    ax.set_xlabel(f'{kolom[i]}')
    ax.set_ylabel('Frequency')
    
plt.tight_layout()
plt.show()


le = LabelEncoder()
train_data_noid['Sex'] = le.fit_transform(train_data_noid['Sex'])
test_data['Sex'] = le.fit_transform(test_data['Sex'])


train_data_noid.head()


test_data.head()


X_train = train_data_noid.drop(['Age'], axis=1)
y_train = train_data_noid.Age


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0)


log_reg = LogisticRegression(max_iter=1000, random_state=0)
log_reg.fit(X_train, y_train)


y_pred_log = log_reg.predict(X_test)


mae_log = mean_absolute_error(y_test, y_pred_log)

print(f'Mean Absolute Error: {mae_log}')


randomforest = RandomForestRegressor(max_depth=2, random_state=0)
randomforest.fit(X_train, y_train)


y_pred_rf = randomforest.predict(X_test)


mae_rf = mean_absolute_error(y_test, y_pred_rf)

print(f'Mean Absolute Error: {mae_rf}')


knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train, y_train)


y_pred_knn = knn_regressor.predict(X_test)


mae_knn = mean_absolute_error(y_test, y_pred_knn)

print(f'Mean Absolute Error: {mae_knn}')


id_pred = test_data['id']


prediction = log_reg.predict(test_data.drop(['id'], axis=1))


submission = pd.DataFrame({"id" : id_pred, "Crab Age" : prediction})


submission


submission.to_csv("submission.csv",index=False)

