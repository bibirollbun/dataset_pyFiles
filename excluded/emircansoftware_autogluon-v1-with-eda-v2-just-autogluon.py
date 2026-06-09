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


import warnings
warnings.filterwarnings("ignore")


!pip install autogluon



df=pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")



df.head()



df.info()


df["Sex"]=pd.get_dummies(df["Sex"],dtype=int,drop_first=True)
df["Sex"]=df["Sex"].astype("category")


df.head()


from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


target="Calories"


train_sample = df.copy()


predictor = TabularPredictor(
    label=target,  
    problem_type='regression',  
    eval_metric='mean_absolute_error',  
)



predictor.fit(
    train_sample,  
    presets='best_quality',  
    auto_stack=True, 
    refit_full=True, 
    keep_only_best=True,  
    save_space=True,  
    time_limit=1200,
    verbosity=3
)


test=pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv").reset_index(drop=True)


result=predictor.predict(test)


result


submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


submission["Calories"] = result



submission.head()


submission.to_csv("submission.csv", index=False)





