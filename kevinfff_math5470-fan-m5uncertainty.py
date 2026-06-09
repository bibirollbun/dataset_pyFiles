import pandas as pd
pd.read_csv('/kaggle/input/uncertainty-submission/uncertainty_submission.csv').to_csv('submission.csv',index=False)

