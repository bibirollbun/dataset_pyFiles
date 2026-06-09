import numpy as np 
import pandas as pd


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
train.head(2)


y_true = train['Listening_Time_minutes']
y_true


def rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


rmse_value = rmse(y_true, y_true)
print(f'RMSE: {rmse_value}')


y_pred = [np.mean(y_true)] * len(y_true)

rmse_value = rmse(y_true, y_pred)
print(f'RMSE: {rmse_value}')


y_pred = [np.median(y_true)] * len(y_true)

rmse_value = rmse(y_true, y_pred)
print(f'RMSE: {rmse_value}')


# sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
# sub['Listening_Time_minutes'] = y_true
# sub.to_csv('submission.csv', index = False)


# sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
# sub['Listening_Time_minutes'] = np.mean(y_true)
# sub.to_csv('submission.csv', index = False)


# sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
# sub['Listening_Time_minutes'] = np.random.choice(y_true, size=len(sub))  
# sub.to_csv('submission.csv', index=False)


# sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
# sub['Listening_Time_minutes'] = np.min(y_true)
# sub.to_csv('submission.csv', index = False)


sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
sub['Listening_Time_minutes'] = np.max(y_true)
sub.to_csv('submission.csv', index = False)

