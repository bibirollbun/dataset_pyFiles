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


train_d = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test_d = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")


import warnings
warnings.simplefilter(action = 'ignore',category = FutureWarning)


import warnings
warnings.filterwarnings('ignore')


# importing all libraries
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler,LabelEncoder,PowerTransformer,OneHotEncoder 
from sklearn.model_selection import train_test_split, StratifiedKFold,RandomizedSearchCV

from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error,roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import lightgbm as lgb
from lightgbm import LGBMClassifier, early_stopping, log_evaluation


train_d.head()


train_d.shape


train_d.describe().style.set_properties(**{
    'background-color': '#f2f2f2',   # light grey
    'color': 'black'
})


train_d.info()



train_d.nunique()


print("printing the unique values of the columns")


for col in train_d:
    unq = train_d[col].unique()
    print("-"*60)
    if len(unq) < 15:
        print(f"{col} : {unq}")
    else:
        print(f"{unq[:5]} \n There are So much Unique Vlaues")


sns.set_style("darkgrid")
num_cols = train_d.select_dtypes(include = ['int64','float64']).columns
plt.figure(figsize = (14,len(num_cols)*3))

for idx,col in enumerate(num_cols,1):
    plt.subplot(len(num_cols),2,2*idx-1)
    sns.histplot(train_d[col],kde = True,bins = 20)
    plt.title(f"Histplot of {col}")

    plt.subplot(len(num_cols),2,2*idx)
    sns.boxplot(x = train_d[col],color="#fc8d62")
    plt.title(f"Boxplot {col}")

plt.tight_layout()
plt.show()


# Cliping The Outliers
numerical = ['annual_income', 'debt_to_income_ratio', 'credit_score', 
                  'loan_amount', 'interest_rate']
for nums_col in numerical:
    q1 = train_d[nums_col].quantile(0.25)
    q2 = train_d[nums_col].quantile(0.75)
    IQR = q2 -q1
    lower_bound = q1 - 1.5 * IQR
    upper_bound = q2 + 1.5 * IQR
    train_d[nums_col] = train_d[nums_col].clip(lower = lower_bound, upper = upper_bound)
    test_d[nums_col] = test_d[nums_col].clip(lower = lower_bound, upper = upper_bound)


# for categorical Columns
object_cols = train_d.select_dtypes("object").columns
plt.figure(figsize = (14,len(object_cols)*3))

for idx,col in enumerate(object_cols,1):
    plt.subplot(len(object_cols),2,idx)
    sns.countplot(x = col,data = train_d)
    plt.title(f"Countplot of {col}")
    plt.xticks(rotation = 90)

plt.tight_layout()
plt.show()


# barplot of target varivable 
counts = train_d['loan_paid_back'].value_counts()
plt.figure(figsize = (8,6))

plt.bar(counts.index.astype(str),counts,color = ['#FF6F61','#955251'])
plt.title('Barplot of Target Columns')
plt.xlabel('Loan_Paid_Back  1.0 = YES and 0 = NO')
plt.ylabel('counts')
plt.show()


# splitting grade_subgrade columns
train_d['grade'] = train_d['grade_subgrade'].str[0]
train_d['subgrade'] = train_d['grade_subgrade'].str[1:].astype(int)

test_d['grade'] = test_d['grade_subgrade'].str[0]
test_d['subgrade'] = test_d['grade_subgrade'].str[1:].astype(int)

order_of_grade  = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6}
train_d['grade'] =  train_d['grade'].map(order_of_grade)
test_d['grade'] =  test_d['grade'].map(order_of_grade)

# drop grade_subgrade column
train_d = train_d.drop('grade_subgrade',axis = 1)
test_d = test_d.drop('grade_subgrade',axis = 1)


train_d = train_d.drop('id',axis = 1)
test_d = test_d.drop('id',axis = 1)


objects = ['gender', 'marital_status', 'education_level', 'employment_status',
       'loan_purpose']
ln = LabelEncoder()
for cols in objects:
    train_d[cols] = ln.fit_transform(train_d[cols])
    test_d[cols] = ln.transform(test_d[cols])


x = train_d.drop("loan_paid_back",axis = 1)
y = train_d['loan_paid_back']


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42,stratify = y)


lgb_train = lgb.Dataset(x,label = y,free_raw_data = True)


lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",  
    "learning_rate": 0.03,
    "num_leaves": 80,
    "max_depth": 6,
    "min_child_samples": 20,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 2,
    
    "feature_fraction": 0.85,
    "bagging_fraction": 0.9,
    "reg_alpha": 0.2,
    "reg_lambda": 0.4,
    "min_split_gain": 0.01,
    "min_data_in_leaf": 40,
    "n_jobs": -1,
    "device": "gpu",
    "verbose": -1,
    "random_state": 42
}


cv_result = lgb.cv(
    params = lgb_params,
    train_set = lgb_train,
    num_boost_round = 20000,
    nfold = 10,
    stratified = True,
    callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period = 150)],
    seed = 42
)


best_iter = len(cv_result['valid auc-mean'])


model = LGBMClassifier(**lgb_params,n_estimators = best_iter)

model.fit(x_train,y_train)

preds = model.predict_proba(test_d)[:,1]


submission = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")
submission['loan_paid_back'] = preds


submission.to_csv("submission.csv",index = False)
print("Submission File Is Submitted Successfully")


submission.head()




