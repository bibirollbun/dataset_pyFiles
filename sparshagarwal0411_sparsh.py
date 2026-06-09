# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
final = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
import warnings
warnings.filterwarnings("ignore")


train.info()


test.info()


final.head()


def feature_engineering(df):
    df = df.copy()
    
    # 1. Log transforms
    df["LogTrackDuration"] = np.log1p(df["TrackDurationMs"])  # log(1+x) to avoid 0
    df["LogAudioLoudness"] = np.log1p(df["AudioLoudness"].clip(lower=0))  # only positive
    
    # 2. Ratios / interactions
    df["RhythmPerTime"] = df["RhythmScore"] / (df["TrackDurationMs"] + 1)
    df["EnergyAcoustic"] = df["Energy"] * df["AcousticQuality"]
    df["InstrToVocal"] = df["InstrumentalScore"] / (df["VocalContent"] + 1)
    df["EnergyRhythm"] = df["Energy"] * df["RhythmScore"]
    df["LoudnessEnergy"] = df["AudioLoudness"] * df["Energy"]
    
    # 3. Squares (non-linear)
    df["EnergySq"] = df["Energy"] ** 2
    df["MoodSq"] = df["MoodScore"] ** 2
    
    # 4. Domain-specific
    df["LiveAcoustic"] = df["LivePerformanceLikelihood"] * df["AcousticQuality"]
    df["DanceabilityProxy"] = df["RhythmScore"] + df["MoodScore"] + df["Energy"]
    
    # 5. Binning
    df["TrackLengthBin"] = pd.cut(df["TrackDurationMs"], 
                                  bins=[0, 120000, 240000, 1000000], 
                                  labels=["Short", "Medium", "Long"])
    df["EnergyBin"] = pd.cut(df["Energy"], bins=[0, 0.33, 0.66, 1.0], 
                             labels=["Low", "Medium", "High"])
    
    # Convert bins to numeric codes
    df["TrackLengthBin"] = df["TrackLengthBin"].cat.codes
    df["EnergyBin"] = df["EnergyBin"].cat.codes

    return df
    # numeric_cols = df.drop(columns=["id", "BeatsPerMinute"]).select_dtypes(include=[np.number]).columns
    # poly = PolynomialFeatures(degree=2, include_bias=False)
    # poly_features = poly.fit_transform(df[numeric_cols])
    
    # poly_feature_names = poly.get_feature_names_out(numeric_cols)
    # df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
    
    # # Merge polynomial features
    # df_final = pd.concat([df, df_poly], axis=1)
    
    # # Scale (optional, for ML models like regression/NN)
    # scaler = StandardScaler()
    # df_final[numeric_cols] = scaler.fit_transform(df_final[numeric_cols])
    
    # return df_final


# Usage
df_engineered = feature_engineering(train)
train = df_engineered.copy()
print(train.shape)
train.head()


def feature_engineering(df):
    df = df.copy()
    
    # 1. Log transforms
    df["LogTrackDuration"] = np.log1p(df["TrackDurationMs"])  # log(1+x) to avoid 0
    df["LogAudioLoudness"] = np.log1p(df["AudioLoudness"].clip(lower=0))  # only positive
    
    # 2. Ratios / interactions
    df["RhythmPerTime"] = df["RhythmScore"] / (df["TrackDurationMs"] + 1)
    df["EnergyAcoustic"] = df["Energy"] * df["AcousticQuality"]
    df["InstrToVocal"] = df["InstrumentalScore"] / (df["VocalContent"] + 1)
    df["EnergyRhythm"] = df["Energy"] * df["RhythmScore"]
    df["LoudnessEnergy"] = df["AudioLoudness"] * df["Energy"]
    
    # 3. Squares (non-linear)
    df["EnergySq"] = df["Energy"] ** 2
    df["MoodSq"] = df["MoodScore"] ** 2
    
    # 4. Domain-specific
    df["LiveAcoustic"] = df["LivePerformanceLikelihood"] * df["AcousticQuality"]
    df["DanceabilityProxy"] = df["RhythmScore"] + df["MoodScore"] + df["Energy"]
    
    # 5. Binning
    df["TrackLengthBin"] = pd.cut(df["TrackDurationMs"], 
                                  bins=[0, 120000, 240000, 1000000], 
                                  labels=["Short", "Medium", "Long"])
    df["EnergyBin"] = pd.cut(df["Energy"], bins=[0, 0.33, 0.66, 1.0], 
                             labels=["Low", "Medium", "High"])
    
    # Convert bins to numeric codes
    df["TrackLengthBin"] = df["TrackLengthBin"].cat.codes
    df["EnergyBin"] = df["EnergyBin"].cat.codes

    return df
    
    # # 6. Polynomial features (2nd degree for all numeric predictors)
    # numeric_cols = df.drop(columns=["id"]).select_dtypes(include=[np.number]).columns
    # poly = PolynomialFeatures(degree=2, include_bias=False)
    # poly_features = poly.fit_transform(df[numeric_cols])
    
    # poly_feature_names = poly.get_feature_names_out(numeric_cols)
    # df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
    
    # # Merge polynomial features
    # df_final = pd.concat([df, df_poly], axis=1)
    
    # # Scale (optional, for ML models like regression/NN)
    # scaler = StandardScaler()
    # df_final[numeric_cols] = scaler.fit_transform(df_final[numeric_cols])
    
    # return df_final


# Usage
df_engineered = feature_engineering(test)
test = df_engineered.copy()
print(test.shape)
test.head()


X = train.drop(columns=["BeatsPerMinute", "id"])
y = train["BeatsPerMinute"]
X_test = test.copy()
X_test.drop('id', axis = 1, inplace = True)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# model = RandomForest()
# model.fit(X_train, y_train)
# pred = model.predict(X_val)
# mse = mean_squared_error(y_val, pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_val, pred)

# print("Mean Squared Error:", mse)
# print("RMSE:", rmse)
# print("R² Score:", r2)


def objective_hgb(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "max_iter": trial.suggest_int("max_iter", 200, 2000),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
        "random_state": trial.suggest_int("random_state", 0, 50),
    }

    kf = KFold(n_splits = 2, shuffle=True)
    rmses = []

    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = HistGradientBoostingRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        rmse = mean_squared_error(y_val, preds, squared=False)
        rmses.append(rmse)

    return np.mean(rmses)

study_hgb = optuna.create_study(direction="minimize")
study_hgb.optimize(objective_hgb, n_trials=6)

print("Best HGB RMSE:", study_hgb.best_value)


best_params = study_hgb.best_params
model = HistGradientBoostingRegressor(**best_params)

X_full = pd.concat([X_train, X_val])
y_full = pd.concat([y_train, y_val])

model.fit(X_full, y_full)


test_preds = model.predict(X_test)

submission = pd.DataFrame({
    "id": final["id"].values,   
    "BeatsPerMinute": test_preds
})

submission.to_csv("submission.csv", index=False)
submission.head()

