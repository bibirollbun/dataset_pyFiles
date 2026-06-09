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


train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
train_df['badur'] = train_df['balance'] / train_df['duration']
train_df.head()


train_df['job'].unique()


x = train_df.drop(columns = ["y", 'id'])
y = train_df["y"]
x.head()





catfeatures = train_df.dtypes[train_df.dtypes == "object"].index.tolist() + ['campaign']


x.shape,y.shape


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5)

models = []
m = []

for i, (train_idx, val_idx) in enumerate(skf.split(x, y)):
    print('$$$$$$$$', i)
    x_train, y_train = x.iloc[train_idx], y[train_idx]
    x_val, y_val = x.iloc[val_idx], y[val_idx]

    from catboost import CatBoostClassifier 
    
    clf = CatBoostClassifier(cat_features = catfeatures, eval_metric = 'AUC', iterations = 5000, bootstrap_type = 'Poisson', leaf_estimation_method = 'Newton',
                            grow_policy='Depthwise',
                            task_type = 'GPU',
                            )
    clf.fit(x_train, y_train, eval_set=(x_val, y_val), verbose=250)
    models.append(clf)

    preds = clf.predict_proba(x_val)[:,1]

    m.append(roc_auc_score(y_val, preds))

print(np.mean(m))





from sklearn.metrics import roc_auc_score


preds = []
for model in models:
    preds.append(model.predict_proba(test_df.drop(columns='id'))[:,1])

preds = np.array(preds)
preds = np.mean(preds, axis=0)
preds.shape


sub = pd.DataFrame({"id": test_df["id"], "y": preds})
sub


sub.to_csv("submission.csv", index = False)




