import os
import glob
import random
import numpy as np
import pandas as pd


Lucky_datasets = glob.glob(os.path.join('/kaggle/input/lucky-dataset', '*.csv'))
Lucky_datasets


random_counts = random.randint(3, len(Lucky_datasets))
random_counts


datasets = random.sample(Lucky_datasets, random_counts)
datasets


submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
submission.head()


target = np.zeros(len(submission))
random_weights = np.random.random(len(datasets))
random_weights /= random_weights.sum()
print(f"Weights: {random_weights}")

for i, name_dir in enumerate(datasets):
    df = pd.read_csv(name_dir)
    target += df['y'].values * random_weights[i]


submission.y = target / len(datasets)
submission.to_csv('submission.csv', index=False)
submission.head()




