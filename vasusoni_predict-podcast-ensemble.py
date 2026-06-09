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
weights = [0.04,0.33,0.02,0.05,0.03,0.16,0.10,0.01,0.05,0.17,0.04]  


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


df_podcast_nn_ydf_cat_xgb_lgbm_hgb = pd.read_csv("/kaggle/input/podcast-nn-ydf-cat-xgb-lgbm-hgb/submission.csv")
df_ps_s5_e4_ensemble_of_solutions = pd.read_csv("/kaggle/input/ps-s5-e4-ensemble-of-solutions/submission.csv")
df_11_83504_ensemble_prediction_of_listening_time = pd.read_csv("/kaggle/input/11-83504-ensemble-prediction-of-listening-time/submission.csv")
df_predicting_podcast_listening_time_ensemble = pd.read_csv("/kaggle/input/predicting-podcast-listening-time-ensemble/submission.csv")
df_ensemble_of_only_three_diverse_models_podcast = pd.read_csv("/kaggle/input/ensemble-of-only-three-diverse-models-podcast/submission.csv")
sub_datten = pd.read_csv("/kaggle/input/ps-s5-e4-division-attention/submission.csv")
sub_singlexgb = pd.read_csv("/kaggle/input/xgboost-single-model/submission.csv")

sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


sample_submission['Listening_Time_minutes'] = (
    0.30 * sub_singlexgb['Listening_Time_minutes'] +
    0.10 * df_podcast_nn_ydf_cat_xgb_lgbm_hgb['Listening_Time_minutes'] +
    0.20 * df_ps_s5_e4_ensemble_of_solutions['Listening_Time_minutes'] +
    0.06 * df_predicting_podcast_listening_time_ensemble['Listening_Time_minutes'] +
    0.04 * df_11_83504_ensemble_prediction_of_listening_time['Listening_Time_minutes'] +
    0.20 * df_ensemble_of_only_three_diverse_models_podcast['Listening_Time_minutes'] +
    0.09 * sub_datten['Listening_Time_minutes']+
    0.01 * submission[targetName]  

    
)

sample_submission.to_csv('submission.csv', index=False)
sample_submission.head()





