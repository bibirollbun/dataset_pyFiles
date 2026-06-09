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


%%capture
!pip install autogluon.tabular scikit-learn==1.5.2
import scipy
import numpy as np
import pandas as pd

from autogluon.tabular import TabularPredictor

import warnings
warnings.filterwarnings('ignore')


%%capture
!pip install feature-engine


import matplotlib.pyplot as plt
from feature_engine.encoding import WoEEncoder


train=pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sub=pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")
train=train.drop(columns=['id'],axis=1)
test=test.drop(columns=['id'],axis=1)


train.info()


train['loan_paid_back'].value_counts()


COLS = list(test)
NUM_COLS = test.select_dtypes(include=['int64','float64']).columns.tolist()
CAT_COLS = test.select_dtypes(include=[ 'object']).columns.tolist()


test['loan_paid_back']=0
train.shape, test.shape


train.shape[0]


#find distribution of annual income and loan amount

test['annual_income'].plot(kind='kde')
plt.title('annual_income distribution (KDE)')
plt.xlabel('annual_income')
plt.show()

test['loan_amount'].plot(kind='kde')
plt.title('loan_amount distribution (KDE)')
plt.xlabel('loan_amount')
plt.show()


train['annual_income']=list(np.log(1+train['annual_income']))
train['loan_amount']=list(np.log(1+train['loan_amount']))

test['annual_income']=list(np.log(1+test['annual_income']))
test['loan_amount']=list(np.log(1+test['loan_amount']))


df=pd.concat([train,test],axis=0)
for c in NUM_COLS:
    df[c] = (df[c] - df[c].mean()) / df[c].std()

train=df[ :train.shape[0]]
test=df[train.shape[0]: ]
train.shape, test.shape


test['annual_income'].min(),test['annual_income'].max()


for c in CAT_COLS:
    print(c,train[c].nunique())


woe_encoder = WoEEncoder(variables=CAT_COLS) 
woe_encoder.fit(train, train['loan_paid_back'])
train = woe_encoder.transform(train)
test = woe_encoder.transform(test)

test=test.drop(columns=['loan_paid_back'],axis=1)


def calculate_iv(df, feature, target):
    """
    Calculates the Information Value (IV) for a given feature.
    """
    temp_df = df[[feature, target]].copy()
    grouped = temp_df.groupby(feature)[target].agg(['count', lambda x: (x == 1).sum(), lambda x: (x == 0).sum()])
    grouped.columns = ['total', 'events', 'non_events']
    grouped['events'] = grouped['events'].apply(lambda x: max(x, 0.5))  # Add a small value
    grouped['non_events'] = grouped['non_events'].apply(lambda x: max(x, 0.5))
    total_events = grouped['events'].sum()
    total_non_events = grouped['non_events'].sum()
    grouped['pct_events'] = grouped['events'] / total_events
    grouped['pct_non_events'] = grouped['non_events'] / total_non_events
    grouped['woe'] = np.log(grouped['pct_non_events'] / grouped['pct_events'])
    grouped['iv_component'] = (grouped['pct_non_events'] - grouped['pct_events']) * grouped['woe']
    iv = grouped['iv_component'].sum()
    return iv

#IV for all cat features
for c in CAT_COLS:
    print(c, calculate_iv(train,c,'loan_paid_back'))


#select only those categories whose IV is in range 0.02-0.5
#train=train.drop(columns=['gender', 'marital_status', 'education_level','employment_status','loan_purpose'],axis=1)
#test=test.drop(columns=['gender', 'marital_status', 'education_level','employment_status','loan_purpose'],axis=1)


TimeLimit= 7.5


%%capture

predictor = TabularPredictor(label = 'loan_paid_back',
                             problem_type = 'binary',
                             eval_metric = 'roc_auc')

predictor.fit(train,
              presets = 'best_quality',
              time_limit = 3600 * TimeLimit,
              auto_stack = True,
              refit_full=True,
              verbosity = 1,
              num_bag_folds=9,
              ag_args_fit={'num_gpus': 1}
             )


predictor.leaderboard()


best_model = predictor.model_best
print(f'Best Model : {best_model}')


preds = predictor.predict_proba(test)


preds.tail(3)


sub['loan_paid_back']= list(preds[1])
sub.to_csv("submission_autogluon.csv",index=False)


sub.tail(3)


oof_predictions = predictor.predict_proba_oof()

oof_predictions.to_csv("oof_autogluon.csv",index=False)


predictor.save() 


sub_ensemble=pd.read_csv("/kaggle/input/ps-s5e11-blend/submission.csv")
sub['loan_paid_back']= 0.015 * sub['loan_paid_back'] + 0.985 * sub_ensemble['loan_paid_back']
sub.to_csv("submission_ensemble.csv",index=False)

