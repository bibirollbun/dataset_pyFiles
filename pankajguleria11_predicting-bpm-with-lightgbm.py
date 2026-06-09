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


TRAIN_PATH = '/kaggle/input/playground-series-s5e9/train.csv'
TEST_PATH  = '/kaggle/input/playground-series-s5e9/test.csv'
RANDOM_STATE = 42
N_JOBS = -1
TARGET = 'BeatsPerMinute'
ID_COL = 'id'

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

# Ensuring ID column
if ID_COL not in train.columns:
    train.insert(0, ID_COL, range(1, len(train) + 1))
if ID_COL not in test.columns:
    test.insert(0, ID_COL, range(1, len(test) + 1))


train.describe()


test.describe()


def fe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Rhythm_Energy'] = df['RhythmScore'] * df['Energy']
    df['Rhythm_Loudness'] = df['RhythmScore'] * df['AudioLoudness']
    df['Duration_Minutes'] = df['TrackDurationMs'] / 60000
    df['Duration_Energy_Ratio'] = df['TrackDurationMs'] / (df['Energy'] * 10000 + 1)
    df['RhythmScore_Squared'] = df['RhythmScore'] ** 2
    df['Energy_Squared'] = df['Energy'] ** 2
    df['Log_Duration'] = np.log1p(df['TrackDurationMs'])
    df['Acoustic_Instrumental_Ratio'] = df['AcousticQuality'] / (df['InstrumentalScore'] + 0.01)
    df['Vocal_Energy'] = df['VocalContent'] * df['Energy']
    df['Live_Energy'] = df['LivePerformanceLikelihood'] * df['Energy']
    df['Mood_Rhythm'] = df['MoodScore'] * df['RhythmScore']
    df['Audio_Intensity'] = (df['Energy'] * np.abs(df['AudioLoudness'])) / 10
    df['Performance_Character'] = (df['LivePerformanceLikelihood'] + df['MoodScore']) / 2
    df['Energy_Loudness_Ratio'] = df['Energy'] / (np.abs(df['AudioLoudness']) + 0.01)
    df['Rhythm_Duration_Density'] = df['RhythmScore'] / df['Duration_Minutes']
    return df

train_fe = fe(train)
test_fe = fe(test)


train_fe.describe()


test_fe.describe()


feature_cols = [c for c in train_fe.columns if c not in [TARGET, ID_COL]]
X_full = train_fe[feature_cols]
y_full = train_fe[TARGET]
X_test = test_fe[feature_cols]


from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)
X_test_scaled = scaler.transform(X_test)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
val_preds = np.zeros(len(X_full))
lgb_models = []

for train_idx, val_idx in kf.split(X_full):
    X_tr, X_va = X_full_scaled[train_idx], X_full_scaled[val_idx]
    y_tr, y_va = y_full.iloc[train_idx], y_full.iloc[val_idx]

    lgb_model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.3,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbosity=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)]
    )
    val_preds[val_idx] = lgb_model.predict(X_va)
    lgb_models.append(lgb_model)

cv_rmse = rmse(y_full, val_preds)
print(f"LightGBM 5-Fold CV RMSE: {cv_rmse:.4f}")


lgb_final = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.3,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    verbosity=-1
)
lgb_final.fit(X_full_scaled, y_full)


test_preds = lgb_final.predict(X_test_scaled)
submission = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: test_preds})
submission.head()


submission = submission.rename(columns={'Id': 'ID'})
submission.to_csv('submission.csv', index=False)




