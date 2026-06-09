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


!pip install autogluon==1.1.1 --quiet





import pandas as pd
from autogluon.tabular import TabularPredictor

# =======================
# Load data
# =======================
train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sub   = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")

# =======================
# Setup
# =======================
target = "BeatsPerMinute"
id_col = "id"

# =======================
# Train AutoGluon
# =======================
predictor = TabularPredictor(
    label=target,
    problem_type="regression",
    eval_metric="rmse",
    path="autogluon_models"
).fit(
    train.drop(columns=[id_col]),
    presets="best_quality",   # or "medium_quality" if you want faster runs
    num_bag_folds=5,          # 5-Fold CV
    num_stack_levels=1,       # stacking depth
    time_limit=3600           # training budget in seconds
)

# =======================
# Predictions
# =======================
preds = predictor.predict(test.drop(columns=[id_col]))

# =======================
# Submission
# =======================
sub[target] = preds
sub.to_csv("submission.csv", index=False)
print("✅ Submission file saved as submission.csv")

# (Optional) leaderboard and feature importance
print(predictor.leaderboard(silent=True))


