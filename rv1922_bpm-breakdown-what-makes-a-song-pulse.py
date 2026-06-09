pip install LightAutoML


import pandas as pd
import numpy as np
import os 
import time 
import seaborn as sns
from sklearn.model_selection import train_test_split
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightautoml.report.report_deco import ReportDeco, ReportDecoUtilized
from lightautoml.addons.tabular_interpretation import SSWARM
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
original = pd.read_csv('/kaggle/input/bpm-prediction-challenge/Train.csv')


train = pd.concat([train, original], axis=0, ignore_index=True)
tran = train.drop_duplicates().reset_index(drop=True)


train.head()


train.info()


train.describe().T


train['TrackDurationMin'] = train['TrackDurationMs'] / 60000

epsilon = 1e-6
train['Energy_Acoustic_Ratio'] = train['Energy'] / (train['AcousticQuality'] + epsilon)
train['Vocal_Instrument_Balance'] = train['VocalContent'] / (train['InstrumentalScore'] + epsilon)

train['MoodRhythm'] = train['MoodScore'] * train['RhythmScore']
train['RhythmEnergy'] = train['RhythmScore'] * train['Energy']
train['MoodAcoustic'] = train['MoodScore'] * train['AcousticQuality']
train['Vocal_Energy_Interaction'] = train['VocalContent'] * train['Energy']
train['PerformanceIntensity'] = train['LivePerformanceLikelihood'] * train['AudioLoudness']

train['Electronic_Proxy'] = (1 - train['AcousticQuality']) * train['Energy'] * train['RhythmScore']
train['Ambient_Proxy'] = train['AcousticQuality'] * (1 - train['Energy']) * (1 - train['RhythmScore'])

train['Ballad_Proxy'] = train['VocalContent'] * train['AcousticQuality'] * (1 - train['Energy']) * (1 - train['RhythmScore'])
train['Instrumental_Intensity_Proxy'] = train['InstrumentalScore'] * train['Energy'] * train['AudioLoudness']
train['Mood_Rhythm_Energy'] = train['MoodScore'] * train['RhythmScore'] * train['Energy']

train['Mood_Energy_Dissonance'] = train['MoodScore'] - train['Energy']
train['Rhythm_Loudness_Ratio'] = train['RhythmScore'] / (train['AudioLoudness'] + epsilon)


test['TrackDurationMin'] = test['TrackDurationMs'] / 60000

epsilon = 1e-6
test['Energy_Acoustic_Ratio'] = test['Energy'] / (test['AcousticQuality'] + epsilon)
test['Vocal_Instrument_Balance'] = test['VocalContent'] / (test['InstrumentalScore'] + epsilon)

test['MoodRhythm'] = test['MoodScore'] * test['RhythmScore']
test['RhythmEnergy'] = test['RhythmScore'] * test['Energy']
test['MoodAcoustic'] = test['MoodScore'] * test['AcousticQuality']
test['Vocal_Energy_Interaction'] = test['VocalContent'] * test['Energy']
test['PerformanceIntensity'] = test['LivePerformanceLikelihood'] * test['AudioLoudness']

test['Electronic_Proxy'] = (1 - test['AcousticQuality']) * test['Energy'] * test['RhythmScore']
test['Ambient_Proxy'] = test['AcousticQuality'] * (1 - test['Energy']) * (1 - test['RhythmScore'])

test['Ballad_Proxy'] = test['VocalContent'] * test['AcousticQuality'] * (1 - test['Energy']) * (1 - test['RhythmScore'])
test['Instrumental_Intensity_Proxy'] = test['InstrumentalScore'] * test['Energy'] * test['AudioLoudness']
test['Mood_Rhythm_Energy'] = test['MoodScore'] * test['RhythmScore'] * test['Energy']

test['Mood_Energy_Dissonance'] = test['MoodScore'] - test['Energy']
test['Rhythm_Loudness_Ratio'] = test['RhythmScore'] / (test['AudioLoudness'] + epsilon)


X = train.drop(columns=['BeatsPerMinute'])
y = train['BeatsPerMinute']


target_col = 'BeatsPerMinute'  
task = Task('reg', metric='mse')

roles = {
    'target': target_col
}

automl = TabularAutoML(
    task=task,
    timeout=5000,          
    cpu_limit=6,
    reader_params={'n_jobs': 6, 'cv': 5, 'random_state': 42}
)


oof_pred = automl.fit_predict(train, roles=roles, verbose=1)


test_pred = automl.predict(test)
submission[target_col] = test_pred.data[:, 0]
submission.to_csv('submission.csv', index=False)


print(submission.head())
print("✅ Submission file saved as submission.csv")


y_true = train[target_col].values
y_pred = oof_pred.data[:, 0]


rmse = mean_squared_error(y_true, y_pred, squared=False)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"OOF RMSE: {rmse:.4f}")
print(f"OOF MAE: {mae:.4f}")
print(f"OOF R²: {r2:.4f}")


residuals = y_true - y_pred

plt.figure(figsize=(7,7))
plt.scatter(y_true, y_pred, alpha=0.3)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("OOF: True vs Predicted")
plt.show()


plt.figure(figsize=(7,5))
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("OOF: Residual Plot")
plt.show()


plt.figure(figsize=(7,5))
plt.hist(residuals, bins=50, alpha=0.7)
plt.axvline(0, color='red', linestyle='--')
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.title("OOF: Error Distribution")
plt.show()

