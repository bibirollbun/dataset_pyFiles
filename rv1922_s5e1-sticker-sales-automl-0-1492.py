import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
import optuna
from sklearn.metrics import mean_absolute_percentage_error
import h2o
from h2o.automl import H2OAutoML


h2o.init()


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


train.head()


train.info()


def trans_df(df):
    df['date'] = pd.to_datetime(df['date'])
    
    df['weekday_sv'] = df['date'].dt.strftime('%a').astype('category')
    df['weekday_num'] = df['date'].dt.strftime('%w').astype('category')
    df['day_of_month'] = df['date'].dt.strftime('%d').astype('category')
    df['month_name_sv'] = df['date'].dt.strftime('%b').astype('category')
    df['month_num'] = df['date'].dt.strftime('%m').astype('int')
    df['year_fv'] = df['date'].dt.strftime('%Y').astype('int')
    df['day_number_year'] = df['date'].dt.strftime('%j').astype('int')
    df['week_number_year'] = df['date'].dt.strftime('%W').astype('category')
    
    df['year_sin'] = np.sin(2 * np.pi * df['year_fv'])
    df['year_cos'] = np.cos(2 * np.pi * df['year_fv'])
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
    
    df['country'] = df['country'].astype('category')
    df['store'] = df['store'].astype('category')
    df['product'] = df['product'].astype('category')
    
    return df


train = trans_df(train)
test = trans_df(test)


train.drop('date', axis=1, inplace=True)
test.drop('date', axis=1, inplace=True)


train = train.dropna()
test = test.dropna()


target = "num_sold"  
features = [col for col in train.columns if col != target]


train_h2o = h2o.H2OFrame(train)
test_h2o = h2o.H2OFrame(test)


automl = H2OAutoML(max_runtime_secs=1500, seed=42)
automl.train(x=features, y=target, training_frame=train_h2o)


leaderboard = automl.leaderboard
print(leaderboard)


test.head()


predictions = automl.leader.predict(test_h2o)
predictions_df = predictions.as_data_frame()


submission = pd.DataFrame({
    'id': test['id'],  
    'num_sold': predictions_df['predict']
})


print(submission)


submission.to_csv('submission.csv', index=False)
print("File saved...")

