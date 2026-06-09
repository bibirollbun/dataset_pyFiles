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


train=pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sub=pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")


train.head()


train.info()


!pip install -U pip
!pip install -U "scikit-learn<1.4.0" "numpy<1.27" "pandas<2.3" "autogluon.tabular==1.1.1"


from autogluon.tabular import TabularPredictor
label="diagnosed_diabetes"


predictor = TabularPredictor(label = label,
                             problem_type = 'binary',
                             eval_metric = 'roc_auc')


predictor.fit(
    train,
    num_bag_folds=10,
    num_bag_sets=2,
    presets="best_quality",  
    auto_stack=True,
    refit_full=True,
    save_space=False,
    time_limit=3*3600)


results = predictor.fit_summary()



preds_proba = predictor.predict_proba(test)



if hasattr(preds_proba, 'columns'):
    positive_class = preds_proba.columns[-1]
    sub["diagnosed_diabetes"] = preds_proba[positive_class]
else:
    sub["diagnosed_diabetes"] = preds_proba[:, 1]


sub.to_csv("submission.csv", index=False)

