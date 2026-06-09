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


import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")

for df in [train, test , submission]:
    print("=======================")
    print(df.head())
    


train


train.describe()


train.columns


train['default'].value_counts().plot(kind='bar')


train['job'].value_counts().plot(kind='pie',autopct="%1.1f%%")


train['marital'].value_counts().plot(kind='pie',autopct="%1.1f%%")


train['education'].value_counts().plot(kind='pie',autopct="%1.1f%%")


datas = ['job', 'marital', 'education', 'default','housing', 'loan', 'contact','campaign',
       'pdays', 'previous', 'poutcome', 'y']


for data in datas:
    plt.figure(figsize=(5,5))
    train[f"{data}"].value_counts().plot(kind='pie', autopct="%1.1f%%")
    plt.title(f"graph of {data}")
    plt.ylabel("")
    plt.show()


train_set = train.drop(['id','y'], axis=1)
test_set = train['y']

print(train_set)
print(test_set)


plt.scatter(train_set['age'], test_set, alpha=0.5)
plt.xlabel("Age")
plt.ylabel("Duration")
plt.title("Scatter Plot: Age vs Duration")
plt.show()


plt.scatter(train_set['age'], test_set, alpha=0.5)
plt.xlabel("Age")
plt.ylabel("Balance")
plt.title("Scatter Plot: Age vs Balance")
plt.show()


ages = train_set['age']


sns.histplot(ages, bins=20, kde=False, stat="density", color="skyblue")

mu, std = norm.fit(ages)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x,mu, std)
plt.plot(x,p,'r',linewidth=2)


plt.title("Gaussian Distribution of Age (μ={mu:.2f}, σ={std:.2f})")
plt.xlabel("Age")
plt.ylabel("Density")
plt.show()


train_set.dtypes


categorical_cols = ['job','marital','education','housing','loan','contact','month','poutcome']    #'default'
numeric_cols = ['age','balance','day','duration','campaign','pdays','previous']


preprocessor = ColumnTransformer(
    transformers = [
        ('num', StandardScaler(), numeric_cols),
        ('cat',OneHotEncoder(drop='first'), categorical_cols)
    ]
)


train_set_encoded = preprocessor.fit_transform(train_set)

encoded_cols = (
    numeric_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))
)

train_set_encoded = pd.DataFrame(train_set_encoded.toarray() if hasattr(train_set_encoded, "toarray") else train_set_encoded,
                                columns=encoded_cols)


train_set_encoded


test_set


X,x_test, Y, y_test = train_test_split(train_set_encoded, test_set, test_size=.1,)
print("Size of X : ", X.shape)
print("Size of Y : ", Y.shape)
print("Size of x_test : ", x_test.shape)
print("Size of y_test : ", y_test.shape)


# model = SVC(kernel='linear')
# model.fit(X, Y)

# print("Accuracy : ", model.score(x_test, y_test))


from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics  import accuracy_score


rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1,random_state=42)
rf_model.fit(X,Y)

y_pred = rf_model.predict(x_test)
print("Random Forest Accuracy : ", accuracy_score(y_test,y_pred))


dtrain = xgb.DMatrix(X, label=Y)
dtest  = xgb.DMatrix(x_test, y_test)

params = {
    'objective':'binary:logistic',
    'eval_metric': 'error',
    'max_depth': 6,
    'eta':0.3
}


xgb_model = xgb.train(params, dtrain, num_boost_round=100)


xgby_pred = (xgb_model.predict(dtest)>0.5).astype(int)
print("XGBoost Accuracy : ", accuracy_score(y_test,xgby_pred))




