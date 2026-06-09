import pandas as pd

submission1 = pd.read_csv('/kaggle/input/drw-20250702/submission_12947.csv')
submission2 = pd.read_csv('/kaggle/input/drw-20250702/submission_12787.csv')

submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
submission['prediction'] = submission1['prediction'] + submission2['prediction'] 
submission.to_csv('submission.csv', index = False)




