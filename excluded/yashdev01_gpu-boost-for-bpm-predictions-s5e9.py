import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor, DeviceQuantileDMatrix
import xgboost as xgb
from xgboost.dask import DaskDMatrix
from dask.distributed import Client, LocalCluster
from dask_cuda import LocalCUDACluster
from sklearn.model_selection import train_test_split
import cupy as cp  # GPU arrays

import warnings


warnings.filterwarnings("ignore", category=UserWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


print(f"Dataset Shape: {train.shape}")
print(f'Features: {train.columns.tolist()}')


print(f"Dataset Shape: {test.shape}")
print(f'Features: {test.columns.tolist()}')


train.head(10)


num_feat = [col for col in train.select_dtypes(include="number") if col != "id"]


num_feat


def winsorize(series):
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return series.clip(lower, upper)


features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality']
train[features] = train[features].apply(winsorize, axis=0)


train.head(10)


train = train.assign(
    TrackDurationMin=lambda x: x['TrackDurationMs'] / 60000,
    Energy_Acoustic_Ratio=lambda x: x['Energy'] / (x['AcousticQuality'] + 1e-5),
    Vocal_Instrument_Balance=lambda x: x['VocalContent'] / (x['InstrumentalScore'] + 1e-5),
    MoodRhythm=lambda x: x['MoodScore'] * x['RhythmScore'],
    PerformanceIntensity=lambda x: x['LivePerformanceLikelihood'] * x['AudioLoudness'],
    RhythmEnergy=lambda x: x['RhythmScore'] * x['Energy'],
    MoodAcoustic=lambda x: x['MoodScore'] * x['AcousticQuality']
)


train.head()


target = "BeatsPerMinute"
X = train.drop(columns=[target, 'id'], errors='ignore')
y = train[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


param_dist = {
    "max_depth": [3, 7, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [200, 500],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.7, 1.0],
}


xgb_regressor = XGBRegressor(
    random_state=42,
    n_jobs=-1,
    tree_method="hist",  # efficient GPU histogram method
    device="cuda",       # ensure GPU usage
)


model = RandomizedSearchCV(
    estimator=xgb_regressor,
    param_distributions=param_dist,
    n_iter=20,
    scoring="neg_mean_squared_error",
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42,
)


model.fit(X_train, y_train)
print("Best parameters:", model.best_params_)


best_model = model.best_estimator_
y_pred = best_model.predict(X_test)

print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("RÂ² Score:", r2_score(y_test, y_pred))


# Preload columns into NumPy arrays for faster operations
td_ms = test['TrackDurationMs'].values
energy = test['Energy'].values
acoustic = test['AcousticQuality'].values
vocal = test['VocalContent'].values
instrumental = test['InstrumentalScore'].values
mood = test['MoodScore'].values
rhythm = test['RhythmScore'].values
live = test['LivePerformanceLikelihood'].values
loudness = test['AudioLoudness'].values


# Feature engineering (vectorized)
test = test.assign(
    TrackDurationMin=td_ms / 60000,
    Energy_Acoustic_Ratio=np.divide(energy, acoustic + 1e-5),
    Vocal_Instrument_Balance=np.divide(vocal, instrumental + 1e-5),
    MoodRhythm=mood * rhythm,
    PerformanceIntensity=live * loudness,
    RhythmEnergy=rhythm * energy,
    MoodAcoustic=mood * acoustic,
)


print(test.head())
print(f"Final shape: {test.shape}")


train_feat = best_model.get_booster().feature_names

X_test_final = test[train_feat]


y_pred_test = best_model.predict(X_test_final)


output = pd.DataFrame({
    "id": test["id"],
    target: y_pred_test
})
output.to_csv("submission.csv", index=False)

print("Predictions saved to test_predictions.csv")
print(output.head())




