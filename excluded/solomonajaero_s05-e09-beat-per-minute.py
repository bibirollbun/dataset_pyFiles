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


import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


from sklearn.preprocessing import PowerTransformer, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor


df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
df.head(5)


def data_overview(df):
    summary = pd.DataFrame({
        "DataType": df.dtypes,
        "Missing Values": df.isnull().sum(),
        "%Missing Value": (df.isnull().sum() / len(df)) * 100
    })
    
    return summary.reset_index().rename(columns={"index": "Features"})
data_overview(df)


numeric_cols = df.drop('id', axis = 1).select_dtypes(include=['int64', 'float64']).columns
n_cols = 4  
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
bins = 30  

colors = cm.tab20(np.linspace(0, 1, len(numeric_cols)))

plt.figure(figsize=(n_cols * 4, n_rows * 3))

for i, (col, color) in enumerate(zip(numeric_cols, colors), 1):
    plt.subplot(n_rows, n_cols, i)
    plt.hist(df[col], bins=bins, edgecolor='black', alpha=0.7, color=color)
    plt.title(col)
    plt.tight_layout()

plt.show()



def create_music_features(df):
    df["Rhythm_Energy"] = df["RhythmScore"] * df["Energy"]
    df["Rhythm_per_Duration"] = df["RhythmScore"] / (df["TrackDurationMs"] + 1)
    df["Energy_Loudness"] = df["Energy"] * df["AudioLoudness"]

    df["Vocal_to_Instrumental"] = df["VocalContent"] / (df["InstrumentalScore"] + 1e-6)
    df["Acoustic_Vocal"] = df["AcousticQuality"] * df["VocalContent"]

    df["Mood_Rhythm"] = df["MoodScore"] * df["RhythmScore"]
    df["Mood_Energy"] = df["MoodScore"] * df["Energy"]
    df["Live_Energy"] = df["LivePerformanceLikelihood"] * df["Energy"]

   
    df["Duration_per_Rhythm"] = df["TrackDurationMs"] / (df["RhythmScore"] + 1e-6)
    df["Duration_Energy"] = df["TrackDurationMs"] * df["Energy"]

    
    df["TempoIntensityIndex"] = (df["RhythmScore"] + df["Energy"] + df["MoodScore"]) / 3
    df["TextureIndex"] = (df["VocalContent"] + df["InstrumentalScore"]) / 2

    return df



df_eng = create_music_features(df)

test_eng = create_music_features(test_df) 
df_eng.head()


x = df_eng.drop(["id", "BeatsPerMinute"], axis=1)
test = test_eng.drop("id", axis=1)
y = df["BeatsPerMinute"]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# To store OOF predictions
oof_preds = {"lgb": np.zeros(len(x))}

# test predictions,
test_preds = {"lgb": np.zeros((len(test), kf.get_n_splits()))}

#scores
scores = {"lgb": []}


# ===================== LightGBM =====================
print("===== LightGBM =====")
for fold, (train_idx, valid_idx) in enumerate(kf.split(x, y), 1):
    X_train, X_valid = x.iloc[train_idx], x.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = lgb.LGBMRegressor(
        objective = 'regression',
        n_estimators=3333,
        max_depth= 13,
        num_leaves = 41,
        min_child_samples = 87,
        learning_rate= 0.015176526640436238,
        subsample= 0.8821314648081602,
        colsample_bytree= 0.94841717776534,
        reg_lambda = 4.503884561991619,
        reg_alpha = 7.311316805991778,
        metric = 'rmse',
        verbose = 0,
        random_state=42
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    # OOF preds
    preds_valid = model.predict(X_valid)
    oof_preds["lgb"][valid_idx] = preds_valid
    rmse = mean_squared_error(y_valid, preds_valid, squared=False)
    scores["lgb"].append(rmse)
    print(f"Fold {fold}: {rmse:.4f}")
    
    # Test preds
    preds_test = model.predict(test)
    test_preds["lgb"][:, fold-1] = preds_test

print(f"CV RMSE: {np.mean(scores['lgb']):.4f} ± {np.std(scores['lgb']):.4f}\n")


# ===================== Final averaged test predictions =====================
final_preds = {
    "lgb": test_preds["lgb"].mean(axis=1)
}


submission = pd.DataFrame({
    "id": test_df["id"],
    "BeatsPerMinute": final_preds["lgb"] 
})
submission.to_csv("submission.csv", index=False)




