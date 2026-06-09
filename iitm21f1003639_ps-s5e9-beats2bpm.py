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


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, PowerTransformer, StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import optuna
import shap
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.base import clone


import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")  
df.head()


df.info()


#checking for null values
df.isna().sum()


df.shape


# numeric_cols = [
#     "RhythmScore", "AudioLoudness", "VocalContent", "AcousticQuality",
#     "InstrumentalScore", "LivePerformanceLikelihood", "MoodScore",
#     "TrackDurationMs", "Energy"
# ]

# # Plot histograms
# plt.figure(figsize=(15, 12))
# for i, col in enumerate(numeric_cols, 1):
#     plt.subplot(4, 3, i)
#     plt.hist(df[col], bins=50, edgecolor='k', alpha=0.7)
#     plt.title(col)
#     plt.xlabel(col)
#     plt.ylabel("Frequency")

# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt


numeric_cols = [
    "RhythmScore", "AudioLoudness", "VocalContent", "AcousticQuality",
    "InstrumentalScore", "LivePerformanceLikelihood", "MoodScore",
    "TrackDurationMs", "Energy","BeatsPerMinute"
]

plt.figure(figsize=(15, 12))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(4, 4, i)  # 4x4 grid
    plt.boxplot(df[col], vert=True, patch_artist=True)
    plt.title(col)
    plt.ylabel("Value")

plt.tight_layout()
plt.show()



corr_matrix = df[numeric_cols].corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Correlation Heatmap", fontsize=14)
plt.show()


corr_with_target = df[numeric_cols].corr()["BeatsPerMinute"].sort_values(ascending=False)

# Print correlations
print("Correlation of features with BeatsPerMinute:\n")
print(corr_with_target)


def add_features(df):
    df = df.copy()
    
    
    df["TrackDurationMin"] = df["TrackDurationMs"] / 60000.0
    
   
    df["RhythmDensity"] = df["RhythmScore"] / (df["TrackDurationMin"] + 1e-6)
    df["VocalDensity"] = df["VocalContent"] / (df["TrackDurationMin"] + 1e-6)
    

    df["EnergyRhythm"] = df["Energy"] * df["RhythmScore"]
    df["LoudEnergy"] = df["AudioLoudness"] * df["Energy"]
    df["VocalVsInstrumental"] = df["VocalContent"] - df["InstrumentalScore"]
    df["LiveQuality"] = df["LivePerformanceLikelihood"] * df["AcousticQuality"]
    
    return df


df_fe = add_features(df)
df_fe.head()


df_fe = add_features(df)


numeric_cols = df_fe.select_dtypes(include=["float64", "int64"]).columns.tolist()


corr_with_target = df_fe[numeric_cols].corr()["BeatsPerMinute"].sort_values(ascending=False)

print("Correlation of all features (original + engineered) with BeatsPerMinute:\n")
print(corr_with_target)



import matplotlib.pyplot as plt
import seaborn as sns

numeric_cols = [
    "BeatsPerMinute", "MoodScore", "TrackDurationMs", "TrackDurationMin",
    "RhythmScore", "VocalContent", "LivePerformanceLikelihood",
    "InstrumentalScore", "VocalDensity", "LiveQuality", "LoudEnergy",
    "VocalVsInstrumental", "AcousticQuality", "EnergyRhythm",
    "RhythmDensity", "AudioLoudness", "Energy"
]

plt.figure(figsize=(18, 14))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(5, 4, i)  # grid layout
    sns.histplot(df_fe[col], bins=50, kde=True, color="steelblue")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()



log_transform = ["TrackDurationMs", "TrackDurationMin","VocalContent", "InstrumentalScore",
"LivePerformanceLikelihood", "VocalDensity","LiveQuality", "RhythmDensity"]
standard_scale = ["AudioLoudness", "VocalVsInstrumental", "LoudEnergy", "EnergyRhythm"]
minmax_scale = ["RhythmScore", "MoodScore", "Energy", "AcousticQuality"]

log_pipeline = Pipeline([
    ("log", FunctionTransformer(np.log1p, validate=False)),
    ("minmax", MinMaxScaler())
])

standard_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

minmax_pipeline = Pipeline([
    ("scaler", MinMaxScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("log_minmax", log_pipeline, log_transform),
        ("standard", standard_pipeline, standard_scale),
        ("minmax", minmax_pipeline, minmax_scale)
    ],
    remainder="passthrough"  
)



df_fe.head()


y=df_fe["BeatsPerMinute"]
df_ = df_fe.drop(["BeatsPerMinute", "id"], axis=1)


X=preprocessor.fit_transform(df_)


X.shape


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.6,
    colsample_bytree=0.6,
    random_state=42,
    n_jobs=-1
)


xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"RÂ²: {r2:.4f}")



df_test=pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")  
df_te=add_features(df_test)
x_test=df_te.drop(["id"],axis=1)
test_transform= preprocessor.transform(x_test)


y_pred=xgb_model.predict(test_transform)


sample_df = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")
submission_df = pd.DataFrame({
    'id': sample_df['id'],
    'BeatsPerMinute': y_pred
})

submission_df.to_csv('submission.csv', index=False)
submission_df.head()

