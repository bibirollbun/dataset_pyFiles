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


oof_lgb = np.load("/kaggle/input/fork-of-s5e12-feature-en-lightgbm-968e65/lgb_oof.npy")
oof_xgb = np.load("/kaggle/input/s5e12-xgboost-diabetes-prediction-0-6962/xgb_oof.npy")
oof_nn = np.load("/kaggle/input/s5e12-ann-cv-deep-learning/nn_oof.npy")


lgb_preds = np.load("/kaggle/input/fork-of-s5e12-feature-en-lightgbm-968e65/lgb_test.npy")
xgb_preds = np.load("/kaggle/input/s5e12-xgboost-diabetes-prediction-0-6962/xgb_preds.npy")
nn_preds = np.load("/kaggle/input/s5e12-ann-cv-deep-learning/nn_preds.npy")


train_d = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
y = train_d['diagnosed_diabetes']


from sklearn.linear_model import LogisticRegression
stack_train = np.vstack([oof_lgb,oof_xgb,oof_nn]).T
stack_test = np.vstack([lgb_preds,xgb_preds,nn_preds]).T
log_model = LogisticRegression(max_iter = 2000)

log_model.fit(stack_train,y)

preds = log_model.predict_proba(stack_test)[:,1]


# Submission 
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


submission['diagnosed_diabetes'] = preds


submission.to_csv("submission.csv", index=False)


submission.head()




