%%capture
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
!pip uninstall -y scikit-learn
!pip install scikit-learn==1.5.2


import pandas as pd, numpy as np
import gc


%%capture

!pip install autogluon
from autogluon.tabular import TabularPredictor

import warnings
warnings.simplefilter('ignore')


#train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
train = pd.read_csv("/kaggle/input/eda-introvert/train_filtered.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


train


train["Personality"].value_counts()


train=train.drop(columns=['Friends_circle_size','Drained_after_socializing'],axis=1)
test=test.drop(columns=['Friends_circle_size','Drained_after_socializing'],axis=1)


train.isnull().sum().sum()


test.isnull().sum().sum()


train.info()


test.info()


train.shape,test.shape


train=train.drop(columns=['id'],axis=1)
test=test.drop(columns=['id'],axis=1)
train = train.drop_duplicates()
train.shape


target='Personality'


predictor = TabularPredictor(
    label=target,
    problem_type='binary',
    eval_metric='accuracy',  
    path='ag_models',
    #sample_weight="SampleWeight"
)


%%capture
# Sample training data for testing only
#train = train.sample(n=2000, random_state=93)

TIME_LIMIT= 3 * 3600

#TIME_LIMIT= 0.5 * 3600

# Fit the model
predictor.fit(
    train,
    presets='best_quality',
    auto_stack=True,
    #refit_full=True,
    keep_only_best=True,
    save_space=True,
    time_limit=TIME_LIMIT,
    num_bag_folds=9,  #5,
    ag_args_fit={'num_gpus': 1},
    )


test_preds = predictor.predict(test)


sub = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
sub['Personality'] =  list(test_preds)
sub.to_csv("submission.csv", index=False)
sub.head()

