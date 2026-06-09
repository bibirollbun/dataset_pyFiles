# Each Model Running and Kaggle Link


# Each Model Running and Kaggle Link


import pandas as pd
# let's sort it as per given submission sample
sub = pd.read_csv('/kaggle/input/cp-3501-retinamnist-v-2024/sample_submission.csv')
sub


# Your final best submission
sub.to_csv('submission.csv', index=False)
!head submission.csv




