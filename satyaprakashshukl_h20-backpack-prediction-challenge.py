import pandas as pd
import h2o
from h2o.automl import H2OAutoML
from itertools import combinations
from scipy.stats import gmean, hmean
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
h2o.init()


data_folder = "/kaggle/input/playground-series-s5e2"



df_train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
df_train_ex = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
df_test  = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


df_train = pd.concat([df_train_ex, df_train], axis=0).reset_index(drop=True)
df_train.shape


df_train.shape


df_train.drop(columns=['id'], inplace=True)
df_test.drop(columns=['id'], inplace=True)


df_train.columns


df_train.head()


train_data = h2o.H2OFrame(df_train)


from h2o.frame import H2OFrame
with h2o.utils.threading.local_context(polars_enabled=True, datatable_enabled=True):
    pandas_df = train_data.as_data_frame()


train_data


test_data = h2o.H2OFrame(df_test)


aml = H2OAutoML(max_models = 20,
                  max_runtime_secs = 400,
                  max_runtime_secs_per_model = 4,seed=42)
aml.train(y='Price', training_frame=train_data)


leaderboard = aml.leaderboard
print(leaderboard)
best_model = aml.leader
print(best_model)


best_model = aml.leader



best_model


df_test = h2o.H2OFrame(df_test)


predictions = best_model.predict(df_test)
predictions_df = predictions.as_data_frame()


y_pred_original = ((predictions_df['predict'].values))  


y_pred_original


df_sub['Price'] =y_pred_original


df_sub.head()


df_sub.to_csv('submission.csv', index=False)




