import numpy as np
import pandas as pd

dtrain = pd.read_csv('../input/playground-series-s5e9/train.csv', index_col='id')
sub_sample = pd.read_csv('../input/playground-series-s5e9/sample_submission.csv')
sub_import = pd.read_csv('../input/playgrounds5e9-lbrace-v1/submission.csv')

sub_sample['BeatsPerMinute'] = np.mean(dtrain['BeatsPerMinute'])
sub_sample.to_csv('submission_mean.csv', index=False)

per = sub_import['BeatsPerMinute'].values
min_per, max_per = np.min(per), np.max(per)

for i in range(len(per)):
    if per[i] < (min_per + 7):
        per[i] -= 0.6
    if per[i] > (max_per - 9):
        per[i] += 0.5

sub_sample['BeatsPerMinute'] = per
sub_sample.to_csv('submission_adjusted.csv', index=False)

per = sub_import['BeatsPerMinute'].copy()
guide = np.mean(per)

for i in range(len(per)):
    if per[i] < guide:
        per[i] = per[i] * 1.1 - guide * 0.1
    else:
        per[i] = per[i] * 1.2 - guide * 0.2

sub_sample['BeatsPerMinute'] = per
sub_sample.to_csv('submission.csv', index=False)




