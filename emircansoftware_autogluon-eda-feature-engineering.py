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


!pip install -U pip
!pip install -U "scikit-learn<1.4.0" "numpy<1.27" "pandas<2.3" "autogluon.tabular==1.1.1"


train=pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sub=pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


train.head()


train.info()


train.drop("id",axis=1,inplace=True)


num_cols=train.select_dtypes(include=np.number).columns.tolist()
cat_cols=train.select_dtypes(include=["object","bool"]).columns.tolist()


test.drop("id",axis=1,inplace=True)


for i in cat_cols:
    print(train[i].value_counts())


train["loan_paid_back"]=train["loan_paid_back"].replace({1.0:"Yes",0.0:"No"})
num_cols.remove("loan_paid_back")


train.head()


import seaborn as sns
import matplotlib.pyplot as plt

corr = train.corr(method='pearson',numeric_only=True)  
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation")
plt.show()


import pandas as pd
from scipy.stats import chi2_contingency

for i in cat_cols:
    table = pd.crosstab(train[i], train['loan_paid_back'])
    chi2, p, dof, expected = chi2_contingency(table)
    print(f"{i}: Chi2 = {chi2:.3f}, p-value = {p:.4f}")


for i in num_cols:
    plt.figure()
    sns.boxplot(x=train[i])
    plt.show()


train.head()


train["loan_paid_back"].value_counts()


from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()

for i in num_cols:
    train[i]=scaler.fit_transform(train[[i]])
    test[i]=scaler.transform(test[[i]])


train["loan_paid_back"]=train["loan_paid_back"].replace({"Yes":1.0,"No":0.0})

from sklearn.preprocessing import OrdinalEncoder
encode=OrdinalEncoder()

for i in cat_cols:
    train[i]=encode.fit_transform(train[[i]])
    test[i]=encode.transform(test[[i]])


train.head()


from autogluon.tabular import TabularPredictor


label="loan_paid_back"


predictor = TabularPredictor(label = label,
                             problem_type = 'binary',
                             eval_metric = 'roc_auc')


predictor.fit(
    train,
    num_bag_folds=10,
    num_bag_sets=2,
    feature_prune_kwargs={'prune_threshold': 0.001},
    presets="best_quality",  
    auto_stack=True,
    refit_full=True,
    save_space=False,
    time_limit=6*3600,  
    hyperparameters={
        'GBM': [
            {},  
            {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}}, 
            {'boosting_type': 'dart', 'ag_args': {'name_suffix': 'DART'}},  
        ],
        'CAT': {'iterations': 10000, 'early_stopping_rounds': 300},
        'XGB': {'n_estimators': 10000, 'early_stopping_rounds': 300},
        'NN_TORCH': {'num_layers': 4, 'dropout_prob': 0.2},
        'RF': [{'criterion': 'gini'}, {'criterion': 'entropy'}],
    },
)


results = predictor.fit_summary()


preds_proba = predictor.predict_proba(test)


if hasattr(preds_proba, 'columns'):
    positive_class = preds_proba.columns[-1]
    sub["loan_paid_back"] = preds_proba[positive_class]
else:
    sub["loan_paid_back"] = preds_proba[:, 1]


sub.to_csv("submission.csv", index=False)

