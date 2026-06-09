import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV



base_path = "/kaggle/input/march-machine-learning-mania-2025/"

# Loading files
df_season = pd.read_csv(os.path.join(base_path, "MRegularSeasonDetailedResults.csv"))
df_submission = pd.read_csv(os.path.join(base_path, "SampleSubmissionStage2.csv"))

# List of statistics used
stat_cols = ["WScore", "LScore", "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WAst", "WTO", "WOR", "WDR"]


team_stats = df_season.groupby("WTeamID").agg({col: "mean" for col in stat_cols}).reset_index()
team_stats.rename(columns={"WTeamID": "TeamID"}, inplace=True)


team_stats.head()


team_stats.info()


team_stats.describe().transpose()


team_stats.describe().transpose().sum()


team_stats.isnull().sum()


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure (figsize=(15, 5))
sns.scatterplot(data = team_stats , x = 'TeamID', y = 'WScore',)

plt.title('Average winning scores per team')
plt.show


def create_features(df):
    df["Team1"] = df[["WTeamID", "LTeamID"]].min(axis=1)
    df["Team2"] = df[["WTeamID", "LTeamID"]].max(axis=1)
    df["Label"] = (df["WTeamID"] == df["Team1"]).astype(int)

    df = df.drop(columns=stat_cols, errors="ignore")

    df = df.merge(team_stats, left_on="Team1", right_on="TeamID", how="left")
    rename_dict_T1 = {col: col + "_T1" for col in stat_cols}
    df.rename(columns=rename_dict_T1, inplace=True)
    df.drop("TeamID", axis=1, inplace=True)

    df = df.merge(team_stats, left_on="Team2", right_on="TeamID", how="left", suffixes=("", "_T2"))
    rename_dict_T2 = {col: col + "_T2" for col in stat_cols}
    df.rename(columns=rename_dict_T2, inplace=True)
    df.drop("TeamID", axis=1, inplace=True)

    for col in stat_cols:
        df["Diff_" + col] = df[col + "_T1"] - df[col + "_T2"]

    return df


df_train = create_features(df_season.copy())

# ðŸ“Œ SÃ©lection des features et de la target
feature_cols = ["Diff_" + col for col in stat_cols]
X = df_train[feature_cols]
y = df_train["Label"]



pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("model", lgb.LGBMClassifier(random_state=42))
])

# Hyperparameter optimization with RandomizedSearchCV
param_grid = {
    'model__n_estimators': [500, 1000, 1500],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_depth': [5, 7, 10],
    'model__num_leaves': [31, 50, 70],
    'model__subsample': [0.7, 0.8, 0.9],
    'model__colsample_bytree': [0.7, 0.8, 0.9]
}

search = RandomizedSearchCV(pipeline, param_grid, n_iter=10, cv=3, scoring='neg_brier_score', random_state=42, n_jobs=-1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

search.fit(X_train, y_train)
print("Meilleurs hyperparamÃ¨tres :", search.best_params_)


y_val_proba = search.best_estimator_.predict_proba(X_val)[:, 1]
brier = brier_score_loss(y_val, y_val_proba)
print("Brier score loss sur validation:", brier)


def create_features_submission(df):
    df[["Season", "Team1", "Team2"]] = df["ID"].str.split("_", expand=True)
    df["Team1"] = df["Team1"].astype(int)
    df["Team2"] = df["Team2"].astype(int)

    df = df.merge(team_stats, left_on="Team1", right_on="TeamID", how="left")
    rename_dict_T1 = {col: col + "_T1" for col in stat_cols}
    df.rename(columns=rename_dict_T1, inplace=True)
    df.drop("TeamID", axis=1, inplace=True)

    df = df.merge(team_stats, left_on="Team2", right_on="TeamID", how="left", suffixes=("", "_T2"))
    rename_dict_T2 = {col: col + "_T2" for col in stat_cols}
    df.rename(columns=rename_dict_T2, inplace=True)
    df.drop("TeamID", axis=1, inplace=True)

    for col in stat_cols:
        df["Diff_" + col] = df[col + "_T1"] - df[col + "_T2"]

    return df

df_sub = create_features_submission(df_submission.copy())


# Selection of the same features as for training
X_sub = df_sub[feature_cols]

# Prediction for submission
df_submission["Pred"] = search.best_estimator_.predict_proba(X_sub)[:, 1]


df_submission[["ID", "Pred"]].to_csv("submission.csv", index=False)
print("Fichier de soumission gÃ©nÃ©rÃ© avec succÃ¨s !")

