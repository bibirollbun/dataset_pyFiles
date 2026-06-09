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


!pip install ipywidgets
!pip install autogluon scikit-learn==1.5.2


import pandas as pd
train=pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
train.head()
test=pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
test.head()


train.shape


train.nunique()


train.info()


train.drop("id", axis=1, inplace=True)
test.drop("id", axis=1, inplace=True)





df=pd.read_csv("/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv")


df.head(5)


df=df[train.columns.tolist()]
df.head(5)


df.dtypes





df=pd.concat([df, train], ignore_index=True)
df=df.sample(frac=1)
df=df.reset_index(drop=True)


df.head(5)


df.describe()


df.shape


df.isnull().sum()


for col in df.columns:
    print(df[col].value_counts())
    print()


for i in ['family_history_diabetes','hypertension_history','cardiovascular_history']:
    df[i]=df[i].replace({0:'No',1:'Yes'})
    test[i]=test[i].replace({0:'No',1:'Yes'})


df.info()


df.dtypes


df[test.select_dtypes(include=['object','bool']).columns.tolist()].head()


label = 'diagnosed_diabetes'
df[label].value_counts()





from autogluon.tabular import TabularDataset, TabularPredictor


predictor = TabularPredictor(
    label=label,
    eval_metric ='roc_auc',
    problem_type="binary"
)
predictor.fit(df,presets='medium_quality', time_limit=3600*9,verbosity=3, ag_args_fit={'num_gpus': 1})
results = predictor.fit_summary()


predictor.leaderboard()


sub=pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")


sub[label]=predictor.predict_proba(test)[1]
sub.head(5)


sub.to_csv("./submission.csv", index=False)







