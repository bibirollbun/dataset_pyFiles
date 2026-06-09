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


!pip install tensorflow
!pip install xgboost


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.model_selection import train_test_split, cross_val_score,KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import tensorflow as tf
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras import layers



train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


train_df.head(20)


train_df.info()


train_df.describe()


train_df = train_df.drop(columns=['id'])
def corr(df,method='pearson'):
  corr = df.corr(method=method)

  plt.figure(figsize=(20,16))
  sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar=True)
  plt.title(f"{method.capitalize()} Correlation Matrix")
  plt.tight_layout()
  plt.show()
  return corr
print(corr(train_df))


def _add_features(df):
    df = df.copy()
    eps = 1e-6  
    clip_val = 1e6  

    # --- Base duration
    df["dur_min"] = df["TrackDurationMs"] / 60000.0
    dur = df["dur_min"].replace(0, eps)  

    instr1 = 1 + df["InstrumentalScore"]
    vocal1 = 1 + df["VocalContent"]
    acoustic1 = 1 + df["AcousticQuality"]
    energy1 = 1 + df["Energy"]
    live1 = 1 + df["LivePerformanceLikelihood"]
    track1 = 1 + df["TrackDurationMs"]

    # --- Per-minute features
    per_min_features = {
        "EnergyPerMin": df["Energy"] / (dur + eps),
        "RhythmPerMin": df["RhythmScore"] / (dur + eps),
        "VocalPerMin": df["VocalContent"] / (dur + eps),
        "InstrPerMin": df["InstrumentalScore"] / (dur + eps),
        "MoodPerMin": df["MoodScore"] / (dur + eps),
        "LoudPerMin": df["AudioLoudness"] / (dur + eps),
    }
    df = df.assign(**per_min_features)

    # --- First-order interactions
    df["DanceProxy"] = df["RhythmScore"] * df["Energy"]
    df["PulseProxy"] = df["DanceProxy"] / (live1 + eps)
    df["DriveProxy"] = df["EnergyPerMin"] * df["RhythmScore"]
    df["GrooveDensity"] = df["RhythmPerMin"] * df["Energy"]

    df["Vocality"] = df["VocalContent"] / (instr1 + eps)
    df["InstrDominance"] = df["InstrumentalScore"] / (vocal1 + eps)
    df["AcousticVsInstr"] = df["AcousticQuality"] / (instr1 + eps)
    df["ElectroBias"] = df["Energy"] / (acoustic1 + eps)

    # --- Loudness interactions
    df["LoudxEnergy"] = df["AudioLoudness"] * df["Energy"]
    df["LoudxRhythm"] = df["AudioLoudness"] * df["RhythmScore"]
    df["LoudxLive"] = df["AudioLoudness"] * df["LivePerformanceLikelihood"]
    df["LoudDensity"] = df["LoudPerMin"] * df["Energy"]

    # --- Live-performance effects
    df["LiveEnergy"] = df["LivePerformanceLikelihood"] * df["Energy"]
    df["LiveRhythm"] = df["LivePerformanceLikelihood"] * df["RhythmScore"]
    df["LiveTension"] = df["LivePerformanceLikelihood"] * (df["MoodScore"] + df["Energy"])

    # --- Mood–tempo couplings
    df["ArousalProxy"] = df["MoodScore"] * df["Energy"]
    df["MoodRhythm"] = df["MoodScore"] * df["RhythmScore"]
    df["MoodPerEnergy"] = df["MoodScore"] / (energy1 + eps)

    # --- Duration-based packing
    df["EnergyPacking"] = df["Energy"] / (track1 + eps)
    df["RhythmPacking"] = df["RhythmScore"] / (track1 + eps)
    df["Hookiness"] = (df["VocalContent"] + df["RhythmScore"]) / (track1 + eps)

    # --- Second-order
    for col in ["Energy", "RhythmScore", "MoodScore", "LivePerformanceLikelihood"]:
        df[f"{col}2"] = (df[col] ** 2)

    # --- Composite intent scores
    df["ClubIntent"] = df["DanceProxy"] / (acoustic1 + eps)
    df["AcousticGroove"] = df["RhythmScore"] * df["AcousticQuality"]
    df["BeatFocus"] = df["RhythmScore"] / (vocal1 + eps)

    # --- Clean-up: replace inf/nan and clip extremes
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df = df.clip(-clip_val, clip_val)

    return df

# Apply safely
train_added_features = _add_features(train_df)
test_added_features = _add_features(test_df)



train_added_features


corr(train_added_features)


train_added_features.info()


X_columns = train_added_features.columns.to_list()
X_columns.remove('BeatsPerMinute')
X = train_added_features[X_columns]
y = train_added_features['BeatsPerMinute']


def check_outliers(X):
    results = []
    for col in X:
        col_data = X[col].dropna()
        total_count = len(col_data)
        outlier_count = 0
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / total_count) * 100
        results.append({
            'Feature': col,
            'Total_Count': total_count,
            'Outlier_Count': outlier_count,
            'Outlier_Percentage': round(outlier_percentage, 2)
        })

    return pd.DataFrame(results)
print(check_outliers(X))


def replace_outliers_simple(df, columns=None, inplace=False):
    if inplace:
        data = df
    else:
        data = df.copy()
    if columns is None:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [col for col in columns if col in data.columns]

    for col in numeric_cols:
        if data[col].isna().all():
            continue
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data.loc[data[col] < lower_bound, col] = lower_bound
        data.loc[data[col] > upper_bound, col] = upper_bound

    return data
X = replace_outliers_simple(X)


print(check_outliers(X))


cat_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    verbose=0,
    random_seed=42,
    task_type="GPU",
    devices="0"
    
)
cat_model.fit(X, y)
feature_importances = cat_model.get_feature_importance()
feat_importance = pd.Series(feature_importances, index=X.columns)
top_feats = feat_importance.sort_values(ascending=False).head(20).index

X_selected = X[top_feats]
X_t = test_added_features[top_feats]





rf_params = dict(
    n_estimators=200,
    max_depth=6,       
    min_samples_split=10,    
    min_samples_leaf=5,      
    max_features='sqrt',    
    random_state=42,
    n_jobs=-1
)
n_splits = 5
kf = KFold(n_splits=n_splits,shuffle=True,random_state=42)
oof_preds = np.zeros(len(X_selected),dtype=np.float64)
test_preds_folds = np.zeros((len(X_t),n_splits),dtype=np.float64)
for fold_idx,(train_idx,valid_idx) in enumerate(kf.split(X_selected)):
    X_tr,X_val = X_selected.iloc[train_idx],X_selected.iloc[valid_idx]
    y_tr, y_val = (y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx],
                   y.iloc[valid_idx] if isinstance(y, pd.Series) else y[valid_idx])

    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_tr, y_tr)

    oof_preds[valid_idx] = rf.predict(X_val)
    test_preds_folds[:, fold_idx] = rf.predict(X_t)
rf_test_pred = test_preds_folds.mean(axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)
X_t_scaled = scaler.transform(X_t)
rf_mean = oof_preds.mean()
rf_std = oof_preds.std() + 1e-12
rf_oof_scaled = ((oof_preds.reshape(-1, 1) - rf_mean) / rf_std).astype(np.float32)
rf_test_pred_scaled = ((rf_test_pred.reshape(-1, 1) - rf_mean) / rf_std).astype(np.float32)
X_meta = np.hstack([X_scaled.astype(np.float32), rf_oof_scaled])
X_t_meta = np.hstack([X_t_scaled.astype(np.float32), rf_test_pred_scaled])
X_train_meta, X_val_meta, y_train, y_val = train_test_split(
    X_meta, y, test_size=0.2, random_state=42
)
input_dim = X_train_meta.shape[1]
def make_nn_model(input_dim, l2_reg=1e-4, dropout_rate=0.3, gaussian_noise_std=0.01):
    inp = keras.Input(shape=(input_dim,), name="meta_input")
    x = layers.GaussianNoise(gaussian_noise_std)(inp)
    x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(1, name="nn_pred")(x)
    model = keras.Model(inputs=inp, outputs=out)
    return model

nn = make_nn_model(input_dim=input_dim, l2_reg=1e-4, dropout_rate=0.3, gaussian_noise_std=0.01)
nn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.Huber(delta=1.0),
    metrics=['mae']
)

early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

print("Training NN on meta features (features + RF oof pred)...")
history = nn.fit(
    X_train_meta, y_train,
    validation_data=(X_val_meta, y_val),
    epochs=200,
    batch_size=256,
    callbacks=[early],
    verbose=2
)
nn_val_pred = nn.predict(X_val_meta).ravel()
print("NN val RMSE (only NN):", np.sqrt(mean_squared_error(y_val, nn_val_pred)))
nn_pred_train = nn.predict(X_train_meta).ravel()
rf_pred_train = X_train_meta[:, -1].ravel()   # last col is RF OOF scaled

nn_pred_val = nn_val_pred
rf_pred_val = X_val_meta[:, -1].ravel()

stack_X_train = np.vstack([nn_pred_train, rf_pred_train]).T
stack_X_val = np.vstack([nn_pred_val, rf_pred_val]).T

meta_model = Ridge(alpha=1.0)
meta_model.fit(stack_X_train, y_train)
stack_val_pred = meta_model.predict(stack_X_val)
print("Stacked val RMSE:", np.sqrt(mean_squared_error(y_val, stack_val_pred)))




nn_test_pred = nn.predict(X_t_meta).ravel()
rf_test_pred_scaled_arr = X_t_meta[:, -1].ravel()
stack_test = np.vstack([nn_test_pred, rf_test_pred_scaled_arr]).T
y_pred_test = meta_model.predict(stack_test)
submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
submission['BeatsPerMinute'] = y_pred_test
submission.to_csv('submission_cat_rf_nn_stacked.csv', index=False)
print("Saved submission_cat_rf_nn_stacked.csv")


submission.head()

