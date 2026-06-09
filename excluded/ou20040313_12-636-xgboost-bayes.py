import pandas as pd

datapub = pd.read_csv('/kaggle/input/ps-s5e4-listening-time-division-attention/submission.csv')['Listening_Time_minutes'].values
dataxgb = pd.read_csv('/kaggle/input/s5e4-version3/621.csv')['Listening_Time_minutes'].values
Listening_Time_minutes = dataxgb * -0.08 + datapub * 1.08

ensemble = pd.DataFrame({
    'id': pd.read_csv('/kaggle/input/s5e4-version3/621.csv')['id'],
    'Listening_Time_minutes': Listening_Time_minutes
})
ensemble.to_csv('ensemble.csv', index=False)
ensemble

