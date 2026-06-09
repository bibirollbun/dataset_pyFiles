!pip install autogluon==1.2
!pip install -U ipywidgets


import pandas as pd
import numpy as np 
import os
df=pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv").drop(columns=['id'])
dt=pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
ids = dt['id']
dt=dt.drop(columns=["id"])
df.head()



label = 'y'
df[label].describe()



from autogluon.tabular import TabularDataset, TabularPredictor

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





