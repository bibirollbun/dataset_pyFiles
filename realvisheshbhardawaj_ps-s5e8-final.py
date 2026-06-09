import os,random

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np
import pandas as pd

p = '/kaggle/input/'

test_df = pd.read_csv(p + "playground-series-s5e8/test.csv")
sub1    = pd.read_csv(p + "ps-s5e8-lightgb-model-add-original-dataset/submission.csv")      
sub2    = pd.read_csv(p + "train-more-xgb-nn-lb-0-9774/submission_ensemble_train_more.csv") 
sub3    = pd.read_csv(p+ "21-august-2025-ps-s5e8/submission 0.97756.csv")
sub4    = pd.read_csv(p + "21-august-2025-ps-s5e8/submission 0.977621.csv")



r1 = sub1['y']
r2 = sub2['y']

r = 0.33
r5 = r * r1 + (1 - r) * r2

r3 = sub3['y']
r4= sub4['y']

r6=r*r3+(1-r)*r4

sub123 = 0.77 * r6 + r5 * 0.23  


sub = sub123


submission = pd.DataFrame({"id": test_df["id"], "y": sub})
submission.to_csv("submission.csv", index=False)
submission.head()




