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


df=pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")


original_data=pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")


df = pd.concat([df, original_data], ignore_index=True)


df.head()


!pip install autogluon


label="Fertilizer Name"


train_data=df.drop("id",axis=1)


from autogluon.tabular import TabularPredictor


predictor=TabularPredictor(label=label,
                            problem_type="multiclass",
                            eval_metric="accuracy")


predictor.fit(
    train_data,
    presets="best_quality",
    auto_stack=True,
    refit_full=True,
    keep_only_best=True,
    save_space=True,
    time_limit=7200
)


test=pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


test.drop("id",axis=1,inplace=True)


probs = predictor.predict_proba(test)


top3_idx = np.argsort(-probs.values, axis=1)[:, :3]  

top3_labels = np.array(probs.columns)[top3_idx]

top3_joined = [" ".join(row) for row in top3_labels]


sub=pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


sub["Fertilizer Name"]=top3_joined


sub.head()


sub.to_csv("submission.csv",index=False)




