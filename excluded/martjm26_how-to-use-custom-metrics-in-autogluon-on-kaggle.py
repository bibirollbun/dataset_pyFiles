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


train=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
train.head()


test=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
test.head()


display(train.info(), train.describe().T)


train=train.drop(['id'],axis=1)
test=test.drop(['id'],axis=1)
display(train.shape, test.shape)


%%writefile custom_metrics.py
import numpy as np
from autogluon.core.metrics import make_scorer

def mapk(y_true, y_pred_proba, k=3):
    map_scores = []
    for true, pred in zip(y_true, y_pred_proba):
        top_k_indices = np.argsort(pred)[::-1][:k]
        if true in top_k_indices:
            rank = list(top_k_indices).index(true) + 1
            map_scores.append(1.0 / rank)
        else:
            map_scores.append(0.0)
    return np.mean(map_scores)

mapk_scorer = make_scorer(
    name='mapk',
    score_func=mapk,
    optimum=1.0,
    greater_is_better=True,
    needs_proba=True,
    needs_threshold=False
)


# Import file (had to do this for it to work with Pickle in AutoGluon)
from custom_metrics import mapk_scorer
from autogluon.tabular import TabularDataset, TabularPredictor

label = 'Fertilizer Name'

predictor = TabularPredictor(label = label, eval_metric = mapk_scorer,
                            problem_type = "multiclass").fit(train, presets='medium_quality',
                            time_limit=3600,verbosity=3, ag_args_fit={'num_gpus': 1})
results = predictor.fit_summary()


predictor.leaderboard()


predictor.save('/kaggle/working/my_model')


#test_predictor = TabularPredictor.load('/kaggle/working/AutogluonModels/ag-20250602_163441')
#test_predictor = TabularPredictor.load('/kaggle/input/autogluon-ensemble/other/default/1/')


df = predictor.predict_proba(test)
df.head()


#df = test_predictor.predict_proba(test)
#df.head()


df[label] = df.apply(lambda x: ' '.join(x.sort_values(ascending=False).index[:3]), axis=1)
df.head()


sub = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
sub[label] = df[label]
sub.to_csv('submission.csv',index=False)
sub.head()




