!pip install autogluon


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score

from autogluon.tabular import TabularPredictor
import phik


colors = ['#E8EDE7', "#81BECE", "#378BA4", "#036280", "#012E4A"]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col=0)
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col=0)


train.head()


train.shape


train.info()


test.info()


test['winddirection'] = test['winddirection'].fillna(test['winddirection'].median())


train = train.drop(columns=['day'])
test = test.drop(columns=['day'])


train['temp_range'] = train['maxtemp'] - train['mintemp']
train['max_min_temp_ratio'] = train['maxtemp'] / train['mintemp']
train['cloud_coverage'] = train['cloud'] / 100  
train['weather_severity'] = (train['cloud'] * train['humidity']) / (train['pressure'] * (train['sunshine'] + 1))
train['temp_humidity_index'] = (train['temparature'] * train['humidity']) / 100
train['pressure_temp_humidity'] = (train['pressure'] * train['temparature']) / train['humidity']

test['temp_range'] = test['maxtemp'] - test['mintemp']
test['max_min_temp_ratio'] = test['maxtemp'] / test['mintemp']
test['cloud_coverage'] = test['cloud'] / 100  
test['weather_severity'] = (test['cloud'] * test['humidity']) / (test['pressure'] * (test['sunshine'] + 1))
test['temp_humidity_index'] = (test['temparature'] * test['humidity']) / 100
test['pressure_temp_humidity'] = (test['pressure'] * test['temparature']) / test['humidity']


train.hist(figsize=(12, 10), bins=30, color='#81BECE')
plt.tight_layout()
plt.show()


corr_matrix = train.phik_matrix()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt='.2f')
plt.title('Correlation Matrix (Phi-K)')
plt.show()


X = train.drop(columns=['rainfall'])
y = train['rainfall']


predictor = TabularPredictor(label='rainfall', eval_metric="roc_auc").fit(
    train, 
    time_limit=3600,  
    presets="best_quality" 
)


y_final = predictor.predict_proba(test)[1]


submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv', index_col=0)
submission['rainfall'] = y_final
submission.head()


submission.to_csv('submission.csv')

