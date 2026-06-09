import polars as pl 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.metrics import mean_squared_error, brier_score_loss

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.base import BaseEstimator, RegressorMixin


# DATA_ROOT_PATH = "./march-machine-learning-mania-2025"
DATA_ROOT_PATH = "/kaggle/input/march-machine-learning-mania-2025"



m_team = pl.read_csv(f"{DATA_ROOT_PATH}/MTeams.csv")
w_team = pl.read_csv(f"{DATA_ROOT_PATH}/WTeams.csv")

m_season = pl.read_csv(f"{DATA_ROOT_PATH}/MSeasons.csv")
w_season = pl.read_csv(f"{DATA_ROOT_PATH}/WSeasons.csv")

m_tournament_seeds = pl.read_csv(f"{DATA_ROOT_PATH}/MNCAATourneySeeds.csv")
w_tournament_seeds = pl.read_csv(f"{DATA_ROOT_PATH}/WNCAATourneySeeds.csv")

m_regular_season = pl.read_csv(f"{DATA_ROOT_PATH}/MRegularSeasonCompactResults.csv")
w_regular_season = pl.read_csv(f"{DATA_ROOT_PATH}/WRegularSeasonCompactResults.csv")


m_team.head()


w_team.head()


w_team = w_team.with_columns([
    pl.lit(None).alias("FirstD1Season"),
    pl.lit(None).alias("LastD1Season")
])

w_team = w_team.select(["TeamID", "TeamName", "FirstD1Season", "LastD1Season"])

team_df = pl.concat([m_team, w_team])


team_df


m_season.head()


w_season.head()


m_tournament_seeds.head()


w_tournament_seeds.head()


tournament_seeds = pl.concat([m_tournament_seeds, w_tournament_seeds])


display(m_regular_season.head())
display(m_regular_season.tail())
display(m_regular_season.shape)


display(w_regular_season.head())
display(w_regular_season.tail())
display(w_regular_season.shape)


season_results = pl.concat([m_regular_season, w_regular_season])
season_results.head()


season_results = season_results.with_columns([
    pl.when(pl.col("WTeamID") < pl.col("LTeamID"))
      .then(pl.col("WTeamID"))
      .otherwise(pl.col("LTeamID"))
      .alias("LowerTeamID"),
    
    pl.when(pl.col("WTeamID") < pl.col("LTeamID"))
      .then(pl.col("LTeamID"))
      .otherwise(pl.col("WTeamID"))
      .alias("HigherTeamID"),
    
    pl.when(pl.col("WTeamID") < pl.col("LTeamID"))
      .then(pl.lit(1))
      .otherwise(pl.lit(0))
      .alias("Target")
])

season_results = season_results.with_columns(
    (pl.col("Season").cast(pl.Utf8) + "_" +
     pl.col("LowerTeamID").cast(pl.Utf8) + "_" +
     pl.col("HigherTeamID").cast(pl.Utf8)
    ).alias("ID")
)

df_train = season_results.filter(pl.col("Season") <= 2020)
df_test = season_results.filter(pl.col("Season") >= 2021)


df_train.head()


df_test.head()


# Lower 用の DataFrame を作成
lower_team_df = team_df.select([
    pl.col("TeamID").alias("LowerTeamID"),
    pl.col("TeamName").alias("LowerTeamName"),
    pl.col("FirstD1Season").alias("LowerFirstD1Season"),
    pl.col("LastD1Season").alias("LowerLastD1Season")
])

# Higher 用の DataFrame を作成
higher_team_df = team_df.select([
    pl.col("TeamID").alias("HigherTeamID"),
    pl.col("TeamName").alias("HigherTeamName"),
    pl.col("FirstD1Season").alias("HigherFirstD1Season"),
    pl.col("LastD1Season").alias("HigherLastD1Season")
])


df_train = df_train.join(
    lower_team_df,
    on="LowerTeamID",
    how="left"
)
df_train = df_train.join(
    higher_team_df,
    on="HigherTeamID",
    how="left"
)


lower_seed_df = tournament_seeds.select([
    pl.col("Season"),
    pl.col("TeamID").alias("LowerTeamID"),
    pl.col("Seed").alias("LowerSeed")
])

# Higher用のシード情報 DataFrame を作成（TeamID を HigherTeamID としてエイリアス）
higher_seed_df = tournament_seeds.select([
    pl.col("Season"),
    pl.col("TeamID").alias("HigherTeamID"),
    pl.col("Seed").alias("HigherSeed")
])


df_train = df_train.join(
    lower_seed_df,
    on=["Season", "LowerTeamID"],
    how="left"
)
df_train = df_train.join(
    higher_seed_df,
    on=["Season", "HigherTeamID"],
    how="left"
)


pattern_Region = r"^(?P<SeedRegion>[A-Z]).*"
pattern_Num = r"^[A-Z](?P<SeedNum>\d+).*"
pattern_subSeed = r"^[A-Z]\d+(?P<SubSeedRegion>[a-z])?$"

# 各パーツを抽出して新規カラムに展開
df_train = df_train.with_columns([
    pl.col("HigherSeed")
      .str.extract(pattern_Region)
      .alias("HigherSeedRegion"),
    pl.col("HigherSeed")
      .str.extract(pattern_Num)
      .alias("HigherSeedNum"),
    pl.col("HigherSeed")
      .str.extract(pattern_subSeed)
      .alias("HigherSubSeedRegion")
])
# 数字部分 SeedNum を整数型に変換
df_train = df_train.with_columns(
    pl.col("HigherSeedNum").cast(pl.Int64)
)


# 各パーツを抽出して新規カラムに展開
df_train = df_train.with_columns([
    pl.col("LowerSeed")
      .str.extract(pattern_Region)
      .alias("LowerSeedRegion"),
    pl.col("LowerSeed")
      .str.extract(pattern_Num)
      .alias("LowerSeedNum"),
    pl.col("LowerSeed")
      .str.extract(pattern_subSeed)
      .alias("LowerSubSeedRegion")
])
# 数字部分 SeedNum を整数型に変換
df_train = df_train.with_columns(
    pl.col("LowerSeedNum").cast(pl.Int64)
)


df_train = df_train.drop(["ID", "WTeamID", "LTeamID", "HigherTeamName", "LowerTeamName", "HigherSeed", "LowerSeed", "WScore", "LScore"])


df_test = pl.read_csv(f'{DATA_ROOT_PATH}/SampleSubmissionStage1.csv')
ID = df_test["ID"]
df_test = df_test.with_columns(
    pl.col("ID").str.split("_").list.to_struct()).unnest("ID")
df_test = df_test.with_columns(
    ID
)

df_test.columns = ["Season", "LowerTeamID", "HigherTeamID", "Pred", "ID"]

df_test = df_test.with_columns(
    pl.col("Season").cast(pl.Int64),
    pl.col("LowerTeamID").cast(pl.Int64),
    pl.col("HigherTeamID").cast(pl.Int64),
    pl.col("Pred").cast(pl.Float32)
)


df_test = df_test.join(
    lower_team_df,
    on="LowerTeamID",
    how="left"
)
df_test = df_test.join(
    higher_team_df,
    on="HigherTeamID",
    how="left"
)


df_test = df_test.join(
    lower_seed_df,
    on=["Season", "LowerTeamID"],
    how="left"
)
df_test = df_test.join(
    higher_seed_df,
    on=["Season", "HigherTeamID"],
    how="left"
)


df_test = df_test.with_columns([
    pl.col("HigherSeed")
      .str.extract(pattern_Region)
      .alias("HigherSeedRegion"),
    pl.col("HigherSeed")
      .str.extract(pattern_Num)
      .alias("HigherSeedNum"),
    pl.col("HigherSeed")
      .str.extract(pattern_subSeed)
      .alias("HigherSubSeedRegion")
])
# 数字部分 SeedNum を整数型に変換
df_test = df_test.with_columns(
    pl.col("HigherSeedNum").cast(pl.Int64)
)


df_test = df_test.with_columns([
    pl.col("LowerSeed")
      .str.extract(pattern_Region)
      .alias("LowerSeedRegion"),
    pl.col("LowerSeed")
      .str.extract(pattern_Num)
      .alias("LowerSeedNum"),
    pl.col("LowerSeed")
      .str.extract(pattern_subSeed)
      .alias("LowerSubSeedRegion")
])
# 数字部分 SeedNum を整数型に変換
df_test = df_test.with_columns(
    pl.col("LowerSeedNum").cast(pl.Int64)
)


df_train = df_train.drop(["DayNum", "NumOT", "WLoc"])
df_test = df_test.drop(["Pred", "ID", "HigherSeed", "LowerSeed", "HigherTeamName", "LowerTeamName"])

df_train = df_train.to_pandas()
df_test = df_test.to_pandas()

cat_cols = list(df_train.select_dtypes("object").columns)

df_train[cat_cols] = df_train[cat_cols].astype("category")
df_test[cat_cols] = df_test[cat_cols].astype("category")


Y = df_train["Target"]
X = df_train.drop(["Target"], axis=1)

X_test = df_test.copy()


params = {
    'objective': 'binary',          # 2値分類
    'boosting_type': 'gbdt',          # GBDT
    'metric': 'binary_logloss',       # ログロス（クロスエントロピー）
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbosity': -1,
    'seed': 42
}


n_splits = 5
folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
cv_scores = []
fitted_models = []


for fold, (train_idx, valid_idx) in enumerate(folds.split(X,Y)):
    print(f"Fold {fold+1}")
    
    X_train, y_train = X.iloc[train_idx], Y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], Y.iloc[valid_idx]
    
    # LightGBM の Dataset 作成（categorical_feature は列名を指定）
    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=['WLoc'])
    valid_set = lgb.Dataset(X_valid, label=y_valid, categorical_feature=['WLoc'], reference=train_set)
    
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set = [(X_valid, y_valid)],
        callbacks = [lgb.log_evaluation(200), lgb.early_stopping(100)],
        categorical_feature = cat_cols)
    fitted_models.append(model)
    y_pred_valid = model.predict_proba(X_valid)[:,1]
    brier_score = brier_score_loss(y_valid, y_pred_valid)
    cv_scores.append(brier_score)

print("Average CV log loss: {:.5f}".format(np.mean(cv_scores)))



class VotingModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators
        
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        y_preds = [estimator.predict(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)
    
    def predict_proba(self, X):
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

model = VotingModel(fitted_models)


lgb.plot_importance(fitted_models[2], importance_type="split", figsize=(5,5))

plt.tight_layout()
plt.show()


import pandas as pd

y_pred = pd.Series(model.predict_proba(df_test)[:, 1], index=df_test.index)


plt.hist(y_pred)


sub = pd.read_csv(f'{DATA_ROOT_PATH}/SampleSubmissionStage1.csv')
sub["Pred"] = y_pred


sub.to_csv("./submission.csv", index=False)




