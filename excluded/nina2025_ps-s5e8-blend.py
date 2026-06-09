import numpy as np
import pandas as pd

p = '/kaggle/input/'

test_df = pd.read_csv(p+"playground-series-s5e8/test.csv")
sub1    = pd.read_csv(p+"30-august-2025-ps-s5e8/submission 0.97772.csv")
sub2    = pd.read_csv(p+"train-more-xgb-nn-lb-0-9774/submission_ensemble_train_more.csv")
sub3    = pd.read_csv(p+"30-august-2025-ps-s5e8/submission 0.97768.csv")
sub4    = pd.read_csv(p+"30-august-2025-ps-s5e8/submission 0.97771.csv")


sub = 0.26 * sub1['y'] + 0.74 * sub2['y']
sub = 0.45 * sub       + 0.55 * sub3['y']
sub = 0.99 * sub       + 0.01 * sub4['y']


submission = pd.DataFrame({"id": test_df["id"], "y": sub})
submission.to_csv("submission.csv", index=False)
submission.head()


# https://www.kaggle.com/code/hzning/top-1-solution-0-97763-esay-is-all-you-need

df1 = pd.DataFrame({"id": test_df["id"], "y": sub2['y']})
df2 = pd.DataFrame({"id": test_df["id"], "y": sub4['y']})

df = pd.merge(df1, df2, on='id', suffixes=('_1', '_2'))

import seaborn as sns, matplotlib.pyplot as plt, warnings; warnings.filterwarnings('ignore')

fig, ax = plt.subplots(1,2, figsize=(12,4))
sns.kdeplot(df['y_1'], label='', fill=True, ax=ax[0])
sns.kdeplot(df['y_2'], label='', fill=True, ax=ax[0])
ax[0].set_title('Raw Probability Densities')

sns.scatterplot(x='y_1', y='y_2', data=df.sample(5000), ax=ax[1], alpha=0.3)
ax[1].plot([0,1],[0,1],'r--')
ax[1].set_title('Pairwise Probability Scatter')
plt.show()

