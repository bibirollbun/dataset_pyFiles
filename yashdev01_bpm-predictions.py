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


def create_features(train):
    train = train.copy()
    eps = 1e-5

    # Durations
    train['TrackDurationMin'] = train['TrackDurationMs'] / 60000
    train['TrackDurationSec'] = train['TrackDurationMs'] / 1000

    # Transforms
    train['AudioLoudness_Abs'] = np.abs(train['AudioLoudness'])
    train['Energy_Log'] = np.log1p(train['Energy'])
    train['Energy_Sq'] = train['Energy'] ** 2
    train['Rhythm_Sq'] = train['RhythmScore'] ** 2
    train['Mood_Sq'] = train['MoodScore'] ** 2

    # Ratios / Products
    train['Energy_Rhythm_Ratio'] = train['Energy'] / (train['RhythmScore'] + eps)
    train['Vocal_Instrumental_Ratio'] = train['VocalContent'] / (train['InstrumentalScore'] + eps)
    train['Mood_Rhythm_Ratio'] = train['MoodScore'] / (train['RhythmScore'] + eps)
    train['Performance_Loudness_Ratio'] = train['LivePerformanceLikelihood'] / (train['AudioLoudness_Abs'] + eps)

    train['Energy_Rhythm_Product'] = train['Energy'] * train['RhythmScore']
    train['Mood_Energy_Product'] = train['MoodScore'] * train['Energy']
    train['Duration_Rhythm'] = train['TrackDurationMin'] * train['RhythmScore']
    train['Duration_Energy'] = train['TrackDurationMin'] * train['Energy']
    train['Duration_Mood'] = train['TrackDurationMin'] * train['MoodScore']

    # Dominance
    train['Vocal_Dominant'] = (train['VocalContent'] > train['InstrumentalScore']).astype(int)
    train['Content_Balance'] = np.abs(train['VocalContent'] - train['InstrumentalScore'])

    # Combos
    train['Danceability_Score'] = train['Energy'] * train['RhythmScore'] * (1 + train['MoodScore'])
    train['Performance_Intensity'] = train['LivePerformanceLikelihood'] * train['Energy'] * train['AudioLoudness_Abs']
    train['Rhythm_Mood_Energy'] = train['RhythmScore'] * train['MoodScore'] * train['Energy']

    # Grouped stats by duration bins
    bins = pd.cut(train['TrackDurationMin'], bins=5, labels=False)
    for col in ['Energy','RhythmScore','MoodScore','AudioLoudness']:
        train[f'{col}_dur_mean'] = train.groupby(bins)[col].transform('mean')
        train[f'{col}_dur_std'] = train.groupby(bins)[col].transform('std').fillna(0)

    return train

# Apply to train dataframe
train = create_features(train)


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


# Apply to test dataframe
test = create_features(test)


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

