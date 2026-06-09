import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, ElasticNet

import xgboost as xgb
import lightgbm as lgb

import warnings
from pathlib import Path


# globals
DATA_DIR = Path("/kaggle/input/playground-series-s5e9")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"
SEED = 128
TARGET_COLUMN = "BeatsPerMinute"

# configurations
warnings.filterwarnings("ignore")


print(f"[INFO] Initialization complete")


train_df = pd.read_csv(TRAIN_PATH, index_col="id")

print("Train data loaded successfully")
print(f"Total number of missing values: {train_df.isna().sum().sum()}")


test_df = pd.read_csv(TEST_PATH, index_col="id")

print("Test data loaded successfully")
print(f"Total number of missing values: {test_df.isna().sum().sum()}")


feature_columns = [column for column in test_df.columns]

print(f"Number of features in the dataset: {len(feature_columns)}\n")
print("Feature columns:")
for column in feature_columns:
    print(f"- {column}")


correlation_matrix = train_df.corr()
target_correlations = correlation_matrix[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values()

target_correlations.plot(kind="barh")
plt.title("Feature Correlations wrt Target Feature")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Features");


correlation_matrix = train_df.corr()
target_correlations = correlation_matrix[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values()

target_correlations.plot(kind="barh")
plt.title("Feature Correlations wrt Target Feature")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Features")
plt.xticks(np.linspace(-1.0, 1.0, 8));


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # copy to avoid directly altering the passed dataframe
    df = df.copy()

    # Rhythm/Energy
    df["RhythmEnergyProduct"] = df["RhythmScore"] * df["Energy"]
    df["RhythmEnergyRatio"] = df["RhythmScore"] / (df["Energy"] + 1e-9)

    # Loudness/Energy
    df["LoudnessEnergyProduct"] = df["AudioLoudness"] * df["Energy"]
    df["LoudnessEnergyRatio"] = df["AudioLoudness"] / (df["Energy"] + 1e-9)

    # Vocal/Instrumental
    df["VocalInstrumentalProduct"] = df["VocalContent"] * df["InstrumentalScore"]
    df["VocalInstrumentalRatio"] = df["VocalContent"] / (df["InstrumentalScore"] + 1e-9)

    # track duration in minutes
    df["TrackDurationMin"] = df["TrackDurationMs"] / 60000

    # TrackDurationMin/MoodScore
    df["TrackDurationMinMoodProduct"] = df["TrackDurationMin"] * df["MoodScore"]
    df["TrackDurationMinMoodRatio"] = df["TrackDurationMin"] / (df["MoodScore"] + 1e-9)

    # Rhythm/TrackDurationMin
    df["RhythmTrackDurationMinProduct"] = df["RhythmScore"] * df["TrackDurationMin"]
    df["RhythmTrackDurationMinRatio"] = df["RhythmScore"] / (df["TrackDurationMin"] + 1e-9)

    # Quality/LivePerformanceLikelihood
    df["QualityLivePerformanceLikelihoodProduct"] = df["AcousticQuality"] * df["LivePerformanceLikelihood"]
    df["QualityLivePerformanceLikelihoodRatio"] = df["AcousticQuality"] / (df["LivePerformanceLikelihood"] + 1e-9)

    # categorical bins
    bin_labels = ["VeryLow", "Low", "Medium", "High", "VeryHigh"]
    
    df["EnergyBin"] = pd.cut(
        df["Energy"],
        bins=len(bin_labels),
        labels=bin_labels
    )
    df["RhythmScoreBin"] = pd.cut(
        df["RhythmScore"],
        bins=len(bin_labels),
        labels=bin_labels
    )

    # set binned features to categorical dtype
    df["EnergyBin"] = df["EnergyBin"].astype("category")
    df["RhythmScoreBin"] = df["RhythmScoreBin"].astype("category")
    
    # target correlations
    correlation_table = train_df.corr(numeric_only=True)
    target_correlations = correlation_table[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(ascending=False)

    # polynomial features for most correlated features
    for feature in target_correlations.head(3).index:
        df[f"{feature}_squared"] = df[feature] ** 2
        df[f"{feature}_sqrt"] = np.sqrt(np.abs(df[feature]))
    
    return df


# engineer new features for the train dataset
train_engineered = create_features(train_df)
engineered_feature_columns = [column for column in train_engineered.columns
                              if column not in train_df.columns]

print(f"Engineered {len(engineered_feature_columns)} feature columns:")
for column in engineered_feature_columns:
    print(f"- {column}")


all_feature_columns = feature_columns + engineered_feature_columns
X = train_engineered[all_feature_columns]
y = train_engineered[TARGET_COLUMN]


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)


numeric_feature_columns = X_train.select_dtypes("number").columns
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train[numeric_feature_columns])
X_val_scaled = scaler.transform(X_val[numeric_feature_columns])


def rmse_score(predictions, targets):
    return np.sqrt(mean_squared_error(targets, predictions))


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str
) -> dict:

    model.fit(X_train, y_train)

    # predictions
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)

    # metrics
    train_rmse = rmse_score(train_predictions, y_train)
    val_rmse = rmse_score(val_predictions, y_val)
    val_r2 = r2_score(y_val, val_predictions)

    # cross-validation scores
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="neg_root_mean_squared_error" # they just HAD to only make the negative version of RMSE 
    )
    cv_rmse = -cv_scores.mean()
    cv_std = cv_scores.std()

    return {
        "model_name": model_name,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "val_r2": val_r2,
        "cv_rmse": cv_rmse,
        "cv_std": cv_std,
        "model": model
    }


models = {
    "XGBoost": xgb.XGBRegressor(random_state=SEED, n_jobs=-1, enable_categorical=True),
    "LightGBM": lgb.LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1, enable_categorical=True),
    "Ridge": Ridge(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=1.0, random_state=SEED)
}


results = []

for model_name, model in models.items():
    print(f"Training {model_name}")

    # Ridge and ElasticNet can only be trained on scaled numeric data
    if model_name in ["Ridge", "ElasticNet"]:
        result = evaluate_model(model, X_train_scaled, y_train, X_val_scaled, y_val, model_name)

    # XGBoost and LightGBM can process categorical features,
    # therefore they'll be trained on data with them included
    else:
        result = evaluate_model(model, X_train, y_train, X_val, y_val, model_name)

    results.append(result)


results_df = pd.DataFrame(results).sort_values("val_rmse")
results_df


best_model_name = results_df.iloc[0]["model_name"]

gb_params = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "enable_categorical": True,
    "n_jobs": -1,
    "verbose": -1
}

if best_model_name == "XGBoost":
    best_model = xgb.XGBRegressor(**gb_params)

elif best_model_name == "LightGBM":
    best_model = lgb.LGBMRegressor(**gb_params)

else:
    best_model = results_df.iloc[0]["model"]

print(f"Best model name: {best_model_name}")


print(f"Training & evaluating the best model ({best_model_name})")

if best_model_name in ["Ridge", "ElasticNet"]:
    best_model.fit(X_train_scaled, y_train)
    val_predictions = best_model.predict(X_val_scaled)
else:
    best_model.fit(X_train, y_train)
    val_predictions = best_model.predict(X_val)

final_rmse = rmse_score(val_predictions, y_val)
final_r2 = r2_score(y_val, val_predictions)

print(f"Final RMSE: {final_rmse:.6f}")
print(f"Final R^2: {final_r2:.6f}\n")


top_2_results = results_df.head(2)
weights = [0.51, 0.49]
ensemble_predictions = np.zeros(len(X_val))

top_2_results


print("Creating Ensembled Prediction with the Next Models:")

for i, (_, entry) in enumerate(top_2_results.iterrows()):
    model_name = entry["model_name"]
    model = entry["model"]
    weight = weights[i]

    if model_name in ["Ridge", "ElasticNet"]:
        predictions = model.predict(X_val_scaled)
    else:
        predictions = model.predict(X_val)

    ensemble_predictions += predictions * weight

    print(f"{model_name:<15} {weight*100:.0f}%")


ensemble_rmse = rmse_score(ensemble_predictions, y_val)
ensemble_r2 = r2_score(y_val, ensemble_predictions)

print(f"Ensemble RMSE: {ensemble_rmse:.6f}")
print(f"Ensemble R^2: {ensemble_r2:.6f}")


# preparing testing data
test_engineered = create_features(test_df)
X_test = test_engineered[all_feature_columns]
X_test_scaled = scaler.transform(X_test[numeric_feature_columns])

# array to store test predictions in
test_predictions = np.zeros(len(X_test))

print(f"Data prepared successfully")


for i, (_, entry) in enumerate(top_2_results.iterrows()):
    model_name = entry["model_name"]
    model = entry["model"]
    weight = weights[i]

    if model_name in ["Ridge", "ElasticNet"]:
        predictions = model.predict(X_test_scaled)
    else:
        predictions = model.predict(X_test)

    test_predictions += predictions * weight


submission_df = pd.read_csv(SAMPLE_PATH, index_col="id")
submission_df[TARGET_COLUMN] = test_predictions

submission_df.sample(5)


submission_df.to_csv("submission.csv")

