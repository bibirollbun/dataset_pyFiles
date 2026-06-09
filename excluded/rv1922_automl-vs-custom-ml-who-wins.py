pip install LightAutoML


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
from sklearn.metrics import roc_auc_score
from lightautoml.report.report_deco import ReportDeco, ReportDecoUtilized
from lightautoml.addons.tabular_interpretation import SSWARM
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv').drop('id', axis=1)
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
original = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


original.columns = original.columns.str.strip()


original['rainfall'] = original['rainfall'].map({'yes': 1, 'no': 0})

train = pd.concat([train, original], axis=0, ignore_index=True)


test_ids = test['id']


train['hour'] = train.index % 24
train['day_of_year'] = train.index // 24 % 365
train['month'] = (train.index // 24 % 365) // 30
train['season'] = ((train.index // 24 % 365) // 91) % 4

train['temp_humidity_interaction'] = train['temparature'] * train['humidity']
train['dew_depression'] = train['temparature'] - train['dewpoint']
train['pressure_wind_interaction'] = train['pressure'] * train['windspeed']
train['temp_range'] = train['maxtemp'] - train['mintemp']
train['temp_gradient_6h'] = train['temparature'] - train['temparature'].shift(6)
train['high_humidity'] = (train['humidity'] > 80).astype(int)
train['humidity_increasing'] = (train['humidity'].diff() > 0).astype(int)
train['low_pressure_system'] = (train['pressure'] < 1013).astype(int)
train['falling_pressure'] = (train['pressure'].diff() < -1).astype(int)

wind_dir_rad = train['winddirection'] * (np.pi / 180)

train['extreme_humidity'] = ((train['humidity'] > train['humidity'].quantile(0.9)) | 
                            (train['humidity'] < train['humidity'].quantile(0.1))).astype(int)

train['extreme_pressure'] = ((train['pressure'] > train['pressure'].quantile(0.9)) | 
                            (train['pressure'] < train['pressure'].quantile(0.1))).astype(int)

train['wind_dir_sin'] = np.sin(wind_dir_rad)
train['wind_dir_cos'] = np.cos(wind_dir_rad)

train['hour_sin'] = np.sin(train['hour'] * (2 * np.pi / 24))
train['hour_cos'] = np.cos(train['hour'] * (2 * np.pi / 24))

train['temp_rising_humidity_high'] = ((train['temparature'].diff() > 0) & 
                                     (train['humidity'] > 70)).astype(int)

train['pressure_falling_wind_rising'] = ((train['pressure'].diff() < 0) & 
                                        (train['windspeed'].diff() > 0)).astype(int)


test['hour'] = test.index % 24
test['day_of_year'] = test.index // 24 % 365
test['month'] = (test.index // 24 % 365) // 30
test['season'] = ((test.index // 24 % 365) // 91) % 4

test['temp_humidity_interaction'] = test['temparature'] * test['humidity']
test['dew_depression'] = test['temparature'] - test['dewpoint']
test['pressure_wind_interaction'] = test['pressure'] * test['windspeed']
test['temp_range'] = test['maxtemp'] - test['mintemp']
test['temp_gradient_6h'] = test['temparature'] - test['temparature'].shift(6)
test['high_humidity'] = (test['humidity'] > 80).astype(int)
test['humidity_increasing'] = (test['humidity'].diff() > 0).astype(int)
test['low_pressure_system'] = (test['pressure'] < 1013).astype(int)
test['falling_pressure'] = (test['pressure'].diff() < -1).astype(int)

wind_dir_rad = test['winddirection'] * (np.pi / 180)

test['extreme_humidity'] = ((test['humidity'] > test['humidity'].quantile(0.9)) | 
                            (test['humidity'] < test['humidity'].quantile(0.1))).astype(int)

test['extreme_pressure'] = ((test['pressure'] > test['pressure'].quantile(0.9)) | 
                            (test['pressure'] < test['pressure'].quantile(0.1))).astype(int)

test['wind_dir_sin'] = np.sin(wind_dir_rad)
test['wind_dir_cos'] = np.cos(wind_dir_rad)

test['hour_sin'] = np.sin(test['hour'] * (2 * np.pi / 24))
test['hour_cos'] = np.cos(test['hour'] * (2 * np.pi / 24))

test['temp_rising_humidity_high'] = ((test['temparature'].diff() > 0) & 
                                     (test['humidity'] > 70)).astype(int)

test['pressure_falling_wind_rising'] = ((test['pressure'].diff() < 0) & 
                                        (test['windspeed'].diff() > 0)).astype(int)


train.head()


train.info()


train.describe().T


train['rainfall'].value_counts()


X = train.drop(columns=['rainfall'])  
y = train['rainfall']


task = Task('binary')
roles = {
    'target': 'rainfall'
}
automl = TabularAutoML(
    task = task,
    timeout = 3000,
    cpu_limit = 6,
    reader_params = {'n_jobs': 6, 'cv': 5, 'random_state': 42}
)


out_of_fold_predictions = automl.fit_predict(train, roles = roles, verbose = 2)
                                             
X_test = train.drop('rainfall',axis=1)
y_test = train['rainfall']


test_predictions = automl.predict(X_test).data[:, 0]  

roc_auc = roc_auc_score(y_test, test_predictions)

print(f"ROC-AUC: {roc_auc:.4f}")


test.head()


predictions = automl.predict(test).data[:, 0]

submission_ids = test['id']


submission = pd.DataFrame({
    'id': submission_ids,
    'loan_status': predictions  
})


submission


submission.to_csv('submission.csv', index=False)

print("File Saved!!")

