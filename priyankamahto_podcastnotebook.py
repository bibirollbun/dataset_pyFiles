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


import os
import numpy as np
import pandas as pd


targetName = 'Listening_Time_minutes'
competitionDir = '/kaggle/input/podcast-listening-time-prediction'
submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


preds = []
fileName = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if (dirname != competitionDir) & ('submission.csv' in filename):
            df = pd.read_csv(os.path.join(dirname, filename))
            fileName.append(dirname)
            if len(df) == len(submission):
                try:
                    preds.append(df[targetName])
                except Exception:
                    pass


print(len(fileName))
for i in fileName:
    print(i)


# weights = [0.10,0.03,0.12,0.40,0.20,0.20,0.001]  
# weights = [0.25,0.25,0.22,0.28]  
# weights = [0.075,0.075,0.05,0.05,0.05,0.02,0.03,0.05,0.20,0.20,0.20]  (some notebooks became private)
weights = [0.05,0.33,0.02,0.05,0.03,0.33,0.10,0.05,0.04]  


# Convert `preds` to a NumPy array
preds_array = np.array(preds)

# Compute weighted sum
weighted_predictions = np.average(preds_array, axis=0, weights=weights)

# Update submission DataFrame
submission[targetName] = weighted_predictions
submission.to_csv("submission.csv", index=False)


print(preds_array.shape)


submission[targetName] = np.array(preds).mean(axis=0).transpose()
# submission.to_csv("submission.csv", index=False)


sub1 = pd.read_csv("/kaggle/input/podcast-nn-ydf-cat-xgb-lgbm-hgb/submission.csv")
sub2= pd.read_csv("/kaggle/input/ps-s5-e4-ensemble-of-solutions/submission.csv")
sub3_82 = pd.read_csv("/kaggle/input/predict-podcast-listening-time-ensemble-t-rk-e/submission.csv")
sub_pplte = pd.read_csv("/kaggle/input/predicting-podcast-listening-time-ensemble/submission.csv")
sub_tdmp = pd.read_csv("/kaggle/input/ensemble-of-only-three-diverse-models-podcast/submission.csv")

sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")

# sample_submission['Listening_Time_minutes']= (0.51 * sub1['Listening_Time_minutes']) + (0.48 * sub2['Listening_Time_minutes'])+ (0.01*submission[targetName])
# sample_submission['Listening_Time_minutes']= (0.25 * sub1['Listening_Time_minutes']) + (0.20 * sub2['Listening_Time_minutes'])+(0.54 * sub3_82['Listening_Time_minutes'])+ (0.01*submission[targetName])
# sample_submission['Listening_Time_minutes']= (0.40 * sub1['Listening_Time_minutes']) + (0.20 * sub2['Listening_Time_minutes'])+ (0.10 * sub_pplte['Listening_Time_minutes'])+(0.20 * sub3_82['Listening_Time_minutes'])+ (0.09 * sub_tdmp['Listening_Time_minutes'])+ (0.01*submission[targetName])
sample_submission['Listening_Time_minutes']= (0.10 * sub1['Listening_Time_minutes']) + (0.40 * sub2['Listening_Time_minutes']) + (0.15 * sub_pplte['Listening_Time_minutes'])+(0.04 * sub3_82['Listening_Time_minutes'])+ (0.3 * sub_tdmp['Listening_Time_minutes'])+ (0.01*submission[targetName])


sample_submission.to_csv('submission.csv', index=False)
sample_submission.head()




