print('データサイエンス研究会へようこそ！')


import pandas as pd

df_sample_submission = pd.read_csv('/kaggle/input/2025-stellar-temperature-challenge/sample_submission.csv')
df_sample_submission.to_csv('submission.csv', index=False)


df_sample_submission

