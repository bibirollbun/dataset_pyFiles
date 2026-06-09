!pip install autogluon==1.2
!pip install -U ipywidgets


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


df=pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv").drop(columns=['id'])
dt=pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
ids = dt['id']
dt=dt.drop(columns=["id"])


df.head()


dt.head()


label = 'y'
df[label].describe()


from autogluon.tabular import TabularDataset, TabularPredictor


from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label=label,
                             eval_metric='roc_auc',  # Use AUC for binary classification
                             problem_type='binary'   # Set to binary classification
                            ).fit(
    train_data=df,
    presets='high_quality',
    time_limit= 3600*5,
    verbosity=3,
    ag_args_fit={'num_gpus': 1}
)

results = predictor.fit_summary()



predictor.leaderboard()


probs = predictor.predict_proba(dt)



sub = pd.DataFrame({
    'id': ids,
    'y': probs[1]  # this accesses the column for class 1
})



sub.head()


sub.to_csv("submission.csv", index=False)


