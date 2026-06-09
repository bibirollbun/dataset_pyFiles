import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


sub_sample = pd.read_csv('../input/playground-series-s5e9/sample_submission.csv') 
# Thanks to: @princevegeta515 - Best Public Score: 26.38135
sub_import = pd.read_csv('/kaggle/input/38096lb/submission81.csv') 
per = sub_import['BeatsPerMinute'].values

# ..................................................................................................
sns.set()
plt.figure(figsize=(7, 4))
plt.hist(per, bins=80)

plt.gca().set_facecolor('lightgreen')
plt.suptitle('Before | BeatsPerMinute', y=0.96, fontsize=16, c='navy')

# ..................................................................................................
min_per = np.min(per)
print('Min:', round(min_per,3))

max_per = np.max(per)
print('Max:', round(max_per,3))

mean_per = np.mean(per)
print('Mean:', round(mean_per,3))


R = -0.0
guide = mean_per - R

# ....................................
per1 = [f for f in per if f < guide]
per2 = [f for f in per if f > guide]

len(per1), len(per2)


N = 5

for _ in range(N):
    for i in range(len(per)):
        per_guide = (per[i] + guide) / 2
        
        if per[i] <= guide:
            per[i] = (per[i] * 1.1) - (per_guide * 0.1)
        else:
            per[i] = (per[i] * 1.00) - (per_guide * 0.00)

# .......................................................................
sns.set()
plt.figure(figsize=(7, 4))
plt.hist(per, bins=80)

plt.gca().set_facecolor('pink')
plt.suptitle('After | BeatsPerMinute', y=0.96, fontsize=16, c='navy')

# .......................................................................
min_per = np.min(per)
print('Min:', round(min_per,3))

max_per = np.max(per)
print('Max:', round(max_per,3))

mean_per = np.mean(per)
print('Mean:', round(mean_per,3))


sub_sample['BeatsPerMinute'] = per
sub_sample.to_csv('submission.csv', index=False) 
sub_sample 

