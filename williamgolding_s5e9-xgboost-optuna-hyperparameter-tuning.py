#These imports aren't in main notebook!!
import optuna

# Core data manipulation libraries
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical functions
from scipy.stats import skew
from scipy.stats import ttest_rel
from scipy.signal import find_peaks

# Machine learning preprocessing and modeling
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from cuml.preprocessing import TargetEncoder

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Read the data
train_df = pd.read_csv('../input/playground-series-s5e9/train.csv', index_col='id')
test_df = pd.read_csv('../input/playground-series-s5e9/test.csv', index_col='id')

# Verify shapes
print("Train Data Shape:", train_df.shape)
print("\nTest Data Shape:", test_df.shape)

# Remove rows with missing target, separate target from predictors
train_df.dropna(axis=0, subset=['BeatsPerMinute'], inplace=True) #inplace = True means that we remove it from X


#X_train = train_df.drop("BeatsPerMinute", axis=1)
#y_train = train_df["BeatsPerMinute"]


def fe1(df: pd.DataFrame) -> pd.DataFrame:
    df["Energy_AudioLoudness"] = df["Energy"] * df["AudioLoudness"]
    df["Mood_Acoustic"] = df["MoodScore"] * df["AcousticQuality"]
    #df["is_high_energy"] = (df["Energy"] > 0.7).astype(int) -> When I added these features my model actually did worse!!! I think XGBoost models do well at creating thresholds anyways...
    #df["is_acoustic"] = (df["AcousticQuality"] > 0.5).astype(int)
    #df["is_live"] = (df["LivePerformanceLikelihood"] > 0.5).astype(int)
    # The track duration expressed in minutes (converted from milliseconds).
    df["TrackDurationMin"] = df["TrackDurationMs"] / 60000
    # The ratio of overall energy to acoustic quality.
    df["Energy_Acoustic_Ratio"] = df["Energy"] / (df["AcousticQuality"] + 1e-5)
    # Measures the balance between vocal and instrumental elements.
    df["Vocal_Instrument_Balance"] = df["VocalContent"] / (df["InstrumentalScore"] + 1e-5)
    df["MoodRhythm"] = df["MoodScore"] * df["RhythmScore"]
    df["PerformanceIntensity"] = df["LivePerformanceLikelihood"] * df["AudioLoudness"]
    df["RhythmEnergy"] = df["RhythmScore"] * df["Energy"]

    return df

train_df = fe1(train_df)
test_df = fe1(test_df)


def fe2(df: pd.DataFrame) -> pd.DataFrame:
    """For each column, excluding the target "BeatsPerMinute", rounds to 8 decimal places"""
    FEATURES = [col for col in df.columns if col != "BeatsPerMinute"]
    for c in FEATURES:
        n = f"{c}_r{8}"
        df[n] = df[c].round(8)
            
    return df

train_df = fe2(train_df)
test_df = fe2(test_df)
train_df.head()


from xgboost import XGBRegressor

FOLDS = 5
SEED = 42

def objective(trial):
    #Suggest hyperparameter search space
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10.0),
        "random_state": SEED,
        "tree_method": "gpu_hist", #use GPU if available
    }

    oof_preds = np.zeros(len(train_df))
    fold_scores = []  # <-- collect per-fold RMSEs here
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        FEATURES = list(train_df.columns)
        FEATURES.remove("BeatsPerMinute")

        X_train = train_df.iloc[train_idx][FEATURES].copy()
        y_train = train_df.iloc[train_idx]["BeatsPerMinute"]
        X_valid = train_df.iloc[val_idx][FEATURES].copy()
        y_valid = train_df.iloc[val_idx]["BeatsPerMinute"]

        # Target encoding
        for c in FEATURES:
            n = f"TE_{c}"
            TE0 = TargetEncoder(n_folds=10, smooth=4, split_method='random', stat='mean')
            X_train[n] = TE0.fit_transform(X_train[c], y_train).astype("float32")
            X_valid[n] = TE0.transform(X_valid[c]).astype("float32")

        model = XGBRegressor(n_estimators=10_000, **params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=200,
            verbose=False
        )

        preds = model.predict(X_valid, iteration_range=(0, model.best_iteration + 1))
        oof_preds[val_idx] = preds

        fold_rmse = mean_squared_error(y_valid, preds, squared=False)
        fold_scores.append(fold_rmse)

    # now you have the fold-wise scores
    mean_rmse = np.mean(fold_scores)
    std_rmse = np.std(fold_scores)

    # log variance info to Optuna
    trial.set_user_attr("fold_scores", fold_scores)
    trial.set_user_attr("rmse_std", std_rmse)

    return mean_rmse

# Run optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

print("Best params:", study.best_params)
print("Best CV RMSE:", study.best_value)


for t in study.trials:
    print(f"Trial {t.number}: mean={t.value:.4f}, std={t.user_attrs['rmse_std']:.4f}, folds={t.user_attrs['fold_scores']}")


