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


# Load predictions

xgb_preds = pd.read_csv("/kaggle/input/aya-best-xgboost-og-dataset/smoking_prediction_submission_xgb_optuna_with_og_AN_v3.csv") 
lgb_preds = pd.read_csv("/kaggle/input/lightgbm-press/smoking_prediction_submission_lightgbm_cpu.csv")

# Sanity check: ensure IDs match

assert (xgb_preds['id'] == lgb_preds['id']).all(), "IDs do not match!"

# Weighted average of the predicted probabilities

blended_preds = (0.7 * xgb_preds['smoking'] + 0.3 * lgb_preds['smoking'])

# Save final blended submission

submission = pd.DataFrame({ 'id': xgb_preds['id'], 'smoking': blended_preds })

submission.to_csv("smoking_prediction_blended_submission.csv", index=False)

print("Blended submission saved successfully!")

