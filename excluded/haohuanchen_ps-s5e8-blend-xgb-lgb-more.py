import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np
import pandas as pd

test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sub1 = pd.read_csv("/kaggle/input/ps-s5e8-lightgb-model-add-original-dataset/submission.csv")  # 0.97541
sub2 = pd.read_csv("/kaggle/input/train-more-xgb-nn-lb-0-9774/submission_ensemble_train_more.csv")  # 0.97742
sub3 = pd.read_csv("/kaggle/input/ps-s5e8-submission/S5E8-0.97768.csv")  # 0.97768


sub = 0.25 * sub1['y'] + 0.75 * sub2['y']  # 0.97762
sub = 0.40 * sub       + 0.60 * sub3['y']


submission = pd.DataFrame({"id": test_df["id"], "y": sub})
submission.to_csv("submission.csv", index=False)
submission.head()

