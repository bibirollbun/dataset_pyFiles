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


import pandas
from skopt import gp_minimize
from skopt.learning.gaussian_process import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel,RBF
from skopt.space import Real,Integer
from sklearn.metrics import roc_auc_score

import lightgbm

# data process
train_data=pandas.read_csv("/kaggle/input/playground-series-s5e8/train.csv",delimiter=',',header=0,encoding="utf-8",index_col=0)
test_data=pandas.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv",delimiter=';',header=0,encoding="utf-8")
new_test_data=pandas.read_csv("/kaggle/input/playground-series-s5e8/test.csv",delimiter=',',header=0,encoding="utf-8",index_col=0)
# check For Missing Values
# train_isna=train_data.isna().sum(axis=1)
# print(train_isna.max())
# test_isna=train_data.isna().sum(axis=1)
# print(test_isna.max())
# See if there are duplicates
# train_duplicated=train_data.duplicated().sum()
# test_duplicated=test_data.duplicated().sum()
# print(train_duplicated,test_duplicated)

# Transform non-numeric data into numerical data
binary_list=["default","housing","loan"]
multi_list=["job","marital","education","contact","month","poutcome"]
binary_dict={"no":0,"yes":1}
multi_dict=[{
    "management":0,
    "blue-collar":1,
    "technician":2,
    "admin.":3,
    "services":4,
    "retired":5,
    "self-employed":6,
    "entrepreneur":7,
    "unemployed":8,
    "housemaid":9,
    "student":10,
    "unknown":11
},
{
    "married":0,
    "single":1,
    "divorced":2
},
{
    "primary":0,
    "secondary":1,
    "tertiary":2,
    "unknown":3
},
{
    "telephone":0,
    "cellular":1,
    "unknown":2
},
{
    "jan":0,
    "feb":1,
    "mar":2,
    "apr":3,
    "may":4,
    "jun":5,
    "jul":6,
    "aug":7,
    "sep":8,
    "oct":9,
    "nov":10,
    "dec":11
},
{
    "success":0,
    "failure":1,
    "other":2,
    "unknown":3
}]
for i in binary_list:
    train_data[i]=train_data[i].map(binary_dict)
    test_data[i] = test_data[i].map(binary_dict)
    new_test_data[i] = new_test_data[i].map(binary_dict)
    # View the category distribution
    #print(train_data[i].value_counts())
    #print(test_data[i].value_counts())
for i in range(len(multi_list)):
    train_data[multi_list[i]]=train_data[multi_list[i]].map(multi_dict[i])
    test_data[multi_list[i]] = test_data[multi_list[i]].map(multi_dict[i])
    new_test_data[multi_list[i]] = new_test_data[multi_list[i]].map(multi_dict[i])
    # View the category distribution
    # print(train_data[multi_list[i]].value_counts())
    # print(test_data[multi_list[i]].value_counts())
test_data["y"]=test_data["y"].map(binary_dict)
# print(train_data["y"].value_counts())
# print(test_data["y"].value_counts())# Categories are unevenly distributed
train_feature=train_data.iloc[:,0:-1]
train_label=train_data.iloc[:,-1]
test_feature=test_data.iloc[:,0:-1]
test_label=test_data.iloc[:,-1]

# Training the model requires adjusting the sample weights due to the uneven distribution 
# of sample labels. Since the model evaluation value is AUC value, it is necessary to use 
# cat_smooth the sklearn-compatible interface to optimize the hyperparameters:
"""
learning_rate
max_depth
max_bin
min_data_in_bin
weight
reg_lambda
min_split_gain
min_data_in_leaf
cat_smooth
n_estimators
"""

space=[
    Real(0.001,1,name="learning_rate"),Integer(4,128,name="num_leaves"),Integer(2,7,name="max_depth"),Integer(100,20000,name="max_bin"),
    Integer(4,50,name="min_data_in_bin"),Real(7,8,name="weight"),Real(0.001,200,name="reg_lambda"),Real(0.001,100,name="min_split_gain"),
    Integer(4,50,name="min_data_in_leaf"),Real(10,50,name="cat_smooth"),Integer(100,1000,name="n_estimators")]
def objective(params):
    learning_rate,num_leaves,max_depth,max_bin,min_data_in_bin,weight,reg_lambda,min_split_gain,\
    min_data_in_leaf,cat_smooth,n_estimators=params
    model=lightgbm.LGBMClassifier(
        objective="binary",
        max_depth=max_depth,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_split_gain=min_split_gain,
        reg_lambda=reg_lambda,
        min_data_in_leaf=min_data_in_leaf,
        cat_smooth=cat_smooth,
        scale_pos_weight=weight,
        max_bin=max_bin,
        min_data_in_bin=min_data_in_bin,
        verbose=-1
    )
    model.fit(train_feature,train_label,categorical_feature=binary_list+multi_list)
    y_pred_proba = model.predict_proba(test_feature)[:,1]
    auc_score = roc_auc_score(test_label,y_pred_proba)
    return -auc_score
kernel_function=ConstantKernel(constant_value=1)*RBF(length_scale=1)
gp_classifier = GaussianProcessRegressor(kernel=kernel_function,n_restarts_optimizer=10,alpha=1e-5)
res=gp_minimize(
    func=objective,
    base_estimator=gp_classifier,
    dimensions=space,
    n_calls=50,
    n_initial_points=20,
    acq_func="EI",
    acq_optimizer="lbfgs",
    xi=0.01,
    n_jobs=-1,
    verbose=True,
)
# Predictions are made using optimized parameters
learning_rate,num_leaves,max_depth,max_bin,min_data_in_bin,weight,reg_lambda,min_split_gain,\
    min_data_in_leaf,cat_smooth,n_estimators=res.x
model=lightgbm.LGBMClassifier(
    objective="binary",
    num_leaves=num_leaves,
    max_depth=max_depth,
    learning_rate=learning_rate,
    n_estimators=n_estimators,
    min_split_gain=min_split_gain,
    reg_lambda=reg_lambda,
    min_data_in_leaf=min_data_in_leaf,
    cat_smooth=cat_smooth,
    scale_pos_weight=weight,
    max_bin=max_bin,
    min_data_in_bin=min_data_in_bin,
    verbose=-1
)
model.fit(train_feature,train_label,categorical_feature=binary_list+multi_list)
y_pred_proba = model.predict_proba(new_test_data)[:,1]
submission = pandas.DataFrame({"id": new_test_data.index, "y":y_pred_proba})
submission.to_csv("/kaggle/working/submission.csv", index=False)





y_pred_proba = model.predict_proba(train_feature)[:,1]
train_auc_score = roc_auc_score(train_label,y_pred_proba)


train_auc_score


res.x

