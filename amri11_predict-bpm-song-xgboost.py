import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, make_scorer
import lightgbm as lgb
import optuna
import xgboost as xgb
from datetime import timedelta
from sklearn.cluster import KMeans
import joblib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


%%time
train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')


train.head()


test.head()


train.info()


test.info()


train.describe()


test.describe()


train_numeric = train.select_dtypes(include=['number'])

# Calculate correlation
corrtrain_matrix = train_numeric.corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corrtrain_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Heatmap Correlation")
plt.show()


# --- Target Distribution: BPM ---
plt.figure(figsize=(8,5))
sns.histplot(train["BeatsPerMinute"], bins=40, kde=True, color="skyblue")
plt.title("Distribution of BPM (Target)", fontsize=14)
plt.xlabel("Beats per Minute")
plt.ylabel("Count")
plt.show()


# --- Numerical Features Distribution ---
exclude_cols = ["BeatsPerMinute", "id"]

num_cols = train.select_dtypes(include=["int64", "float64"]).columns
num_cols = [c for c in num_cols if c not in exclude_cols]

for col in num_cols:
    plt.figure(figsize=(8,5))
    sns.histplot(train[col], bins=40, kde=True, color="orange")
    plt.title(f"Distribution of {col}", fontsize=14)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()


# --- Numerical Features Distribution ---
exclude_cols = ["id"]

num_cols = test.select_dtypes(include=["int64", "float64"]).columns
num_cols = [c for c in num_cols if c not in exclude_cols]

for col in num_cols:
    plt.figure(figsize=(8,5))
    sns.histplot(test[col], bins=40, kde=True, color="orange")
    plt.title(f"Distribution of {col}", fontsize=14)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()


# feature engineering train
train["Energy_Rhythm"] = train["Energy"] * train["RhythmScore"]
train["Vocal_Instrument_Ratio"] = train["VocalContent"] / (train["InstrumentalScore"] + 1e-6)
train["Mood_Acoustic"] = train["MoodScore"] * train["AcousticQuality"]
train["Energy_per_min"] = train["Energy"] / (train["TrackDurationMs"] / 60000)
train["TrackDurationMin"] = train["TrackDurationMs"] / 60000
train["TrackLengthCat"] = pd.cut(
    train["TrackDurationMin"], 
    bins=[0, 2, 5, 10, 60], 
    labels=["short", "medium", "long", "very_long"])
train["Energy_sq"] = train["Energy"]**2
train["Rhythm_sqrt"] = np.sqrt(train["RhythmScore"])

features_for_cluster = ["RhythmScore", "Energy", "MoodScore", "VocalContent", "AcousticQuality"]
kmeans = KMeans(n_clusters=5, random_state=42)
train["Cluster"] = kmeans.fit_predict(train[features_for_cluster])

# feature engineering test
test["Energy_Rhythm"] = test["Energy"] * test["RhythmScore"]
test["Vocal_Instrument_Ratio"] = test["VocalContent"] / (test["InstrumentalScore"] + 1e-6)
test["Mood_Acoustic"] = test["MoodScore"] * test["AcousticQuality"]
test["Energy_per_min"] = test["Energy"] / (test["TrackDurationMs"] / 60000)
test["TrackDurationMin"] = test["TrackDurationMs"] / 60000
test["TrackLengthCat"] = pd.cut(
    test["TrackDurationMin"], 
    bins=[0, 2, 5, 10, 60], 
    labels=["short", "medium", "long", "very_long"])
test["Energy_sq"] = test["Energy"]**2
test["Rhythm_sqrt"] = np.sqrt(test["RhythmScore"])

features_for_cluster = ["RhythmScore", "Energy", "MoodScore", "VocalContent", "AcousticQuality"]
kmeans = KMeans(n_clusters=5, random_state=42)
test["Cluster"] = kmeans.fit_predict(test[features_for_cluster])


train.head()


train.info()


exclude_cols_train = ["id", "BeatsPerMinute"]
exclude_cols_test = "id"

# Take only numeric
numeric_cols_train = train.select_dtypes(include=['number']).columns.tolist()
numeric_cols_test = test.select_dtypes(include=['number']).columns.tolist()

# Drop except column
cols_to_check_train = [col for col in numeric_cols_train if col not in exclude_cols_train]
cols_to_check_test = [col for col in numeric_cols_test if col not in exclude_cols_test]

# IQR capping (bukan drop)
def cap_outliers_iqr(data, cols):
    df_cap = data.copy()
    for col in cols:
        Q1 = df_cap[col].quantile(0.25)
        Q3 = df_cap[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        # replace outliers dengan batas bawah/atas
        df_cap[col] = np.where(df_cap[col] < lower, lower,
                               np.where(df_cap[col] > upper, upper, df_cap[col]))
    return df_cap

# Apply ke train/test
train_ro = cap_outliers_iqr(train, cols_to_check_train)
test_ro = cap_outliers_iqr(test, cols_to_check_test)

# Encode kategorikal
le = LabelEncoder()
train_ro['TrackLengthCat'] = le.fit_transform(train_ro['TrackLengthCat'])
test_ro['TrackLengthCat'] = le.fit_transform(test_ro['TrackLengthCat'])

# Drop id
train_clean = train_ro.drop(['id'], axis=1)
test_clean = test_ro.drop(['id'], axis=1)


joblib.dump(le, "track_length_encoder.joblib")


train_clean.info()
test_clean.info()


x = train_clean.drop('BeatsPerMinute', axis=1)
y = train_clean['BeatsPerMinute']


# RMSE function
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# make score for GridSearchCV
rmse_scorer = make_scorer(rmse, greater_is_better=False)  # RMSE smaller is better

# KFold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kf.split(x):
    x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    param_grid = {
        'n_estimators': [200, 300, 500],
        'learning_rate': [0.001, 0.01, 0.1],
        'max_depth': [3, 5, 8],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }

    model_xgb = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    grid_search = GridSearchCV(
        model_xgb, param_grid, scoring=rmse_scorer, cv=5, n_jobs=-1
    )
    
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_val)

    # Evaluation RMSE
    val_rmse = rmse(y_val, y_pred)
    print(f"RMSE: {val_rmse:.4f}")


joblib.dump(best_model, "best_model.pkl")


# Getting importance as a DataFrame
importance_df = pd.DataFrame({
    'Feature': x_train.columns,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(importance_df)

# Visualization of the importance of features
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()


df_sub = sample.drop('BeatsPerMinute', axis=1)
df_sub.head


df_sub['BeatsPerMinute'] = best_model.predict(test_clean)
df_sub.to_csv('submission.csv', index=False)
df_sub.value_counts()
df_sub.head()

