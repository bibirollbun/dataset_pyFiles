import os, re, pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# RAPIDS cuDF import for GPU preprocessing
import cudf

# Import CatBoost (GPU‑enabled)
from catboost import CatBoostClassifier

# Set pandas options and data path
pd.set_option('display.max_columns', None)
DATA_PATH = '/kaggle/input/march-machine-learning-mania-2025/'

#############################################
# Helper: treat_seed function
#############################################
def treat_seed(seed):
    return int(re.sub("[^0-9]", "", seed))

#############################################
# 1. READ & PREPARE THE DATA (Pandas)
#############################################
print("Files in DATA_PATH:")
for filename in sorted(os.listdir(DATA_PATH)):
    print(filename)

# --- Tournament Seeds (Men's and Women's) ---
df_seeds = pd.concat([
    pd.read_csv(os.path.join(DATA_PATH, "MNCAATourneySeeds.csv")),
    pd.read_csv(os.path.join(DATA_PATH, "WNCAATourneySeeds.csv"))
], ignore_index=True)

# --- Regular Season Results (Men's and Women's) ---
df_season_results = pd.concat([
    pd.read_csv(os.path.join(DATA_PATH, "MRegularSeasonCompactResults.csv")),
    pd.read_csv(os.path.join(DATA_PATH, "WRegularSeasonCompactResults.csv"))
], ignore_index=True)
df_season_results.drop(['NumOT', 'WLoc'], axis=1, inplace=True)
df_season_results['ScoreGap'] = df_season_results['WScore'] - df_season_results['LScore']

# Compute season-level features: wins, losses, score gap.
num_win = (df_season_results.groupby(['Season', 'WTeamID'])
           .count().reset_index()[['Season', 'WTeamID', 'DayNum']]
           .rename(columns={"DayNum": "NumWins", "WTeamID": "TeamID"}))
num_loss = (df_season_results.groupby(['Season', 'LTeamID'])
            .count().reset_index()[['Season', 'LTeamID', 'DayNum']]
            .rename(columns={"DayNum": "NumLosses", "LTeamID": "TeamID"}))
gap_win = (df_season_results.groupby(['Season', 'WTeamID'])
           .mean().reset_index()[['Season', 'WTeamID', 'ScoreGap']]
           .rename(columns={"ScoreGap": "GapWins", "WTeamID": "TeamID"}))
gap_loss = (df_season_results.groupby(['Season', 'LTeamID'])
            .mean().reset_index()[['Season', 'LTeamID', 'ScoreGap']]
            .rename(columns={"ScoreGap": "GapLosses", "LTeamID": "TeamID"}))
df_features_season_w = (df_season_results.groupby(['Season', 'WTeamID'])
                        .count().reset_index()[['Season', 'WTeamID']]
                        .rename(columns={"WTeamID": "TeamID"}))
df_features_season_l = (df_season_results.groupby(['Season', 'LTeamID'])
                        .count().reset_index()[['Season', 'LTeamID']]
                        .rename(columns={"LTeamID": "TeamID"}))
df_features_season = pd.concat([df_features_season_w, df_features_season_l], axis=0)\
                       .drop_duplicates()\
                       .sort_values(['Season', 'TeamID'])\
                       .reset_index(drop=True)
df_features_season = df_features_season.merge(num_win, on=['Season', 'TeamID'], how='left')
df_features_season = df_features_season.merge(num_loss, on=['Season', 'TeamID'], how='left')
df_features_season = df_features_season.merge(gap_win, on=['Season', 'TeamID'], how='left')
df_features_season = df_features_season.merge(gap_loss, on=['Season', 'TeamID'], how='left')
df_features_season.fillna(0, inplace=True)
df_features_season['WinRatio'] = df_features_season['NumWins'] / (df_features_season['NumWins'] + df_features_season['NumLosses'])
df_features_season['GapAvg'] = ((df_features_season['NumWins'] * df_features_season['GapWins'] - 
                                 df_features_season['NumLosses'] * df_features_season['GapLosses']) /
                                (df_features_season['NumWins'] + df_features_season['NumLosses']))
df_features_season.drop(['NumWins', 'NumLosses', 'GapWins', 'GapLosses'], axis=1, inplace=True)

# --- Tournament Results ---
df_tourney_results = pd.concat([
    pd.read_csv(os.path.join(DATA_PATH, "WNCAATourneyCompactResults.csv")),
    pd.read_csv(os.path.join(DATA_PATH, "MNCAATourneyCompactResults.csv"))
], ignore_index=True)
df_tourney_results.drop(['NumOT', 'WLoc'], axis=1, inplace=True)
df = df_tourney_results.copy()
df = df[df['Season'] >= 2016].reset_index(drop=True)

# --- Merge Seeds ---
df = pd.merge(
    df, df_seeds,
    how='left',
    left_on=['Season', 'WTeamID'],
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedW'})
df = pd.merge(
    df, df_seeds,
    how='left',
    left_on=['Season', 'LTeamID'],
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedL'})

df['SeedW'] = df['SeedW'].apply(treat_seed)
df['SeedL'] = df['SeedL'].apply(treat_seed)

# --- Merge Season-Level Features ---
df = pd.merge(
    df, df_features_season,
    how='left',
    left_on=['Season', 'WTeamID'],
    right_on=['Season', 'TeamID']
).rename(columns={'WinRatio': 'WinRatioW', 'GapAvg': 'GapAvgW'}).drop('TeamID', axis=1)
df = pd.merge(
    df, df_features_season,
    how='left',
    left_on=['Season', 'LTeamID'],
    right_on=['Season', 'TeamID']
).rename(columns={'WinRatio': 'WinRatioL', 'GapAvg': 'GapAvgL'}).drop('TeamID', axis=1)

# --- Create Mirrored Matches ---
def add_loosing_matches(df):
    win_rename = {
        "WTeamID": "TeamIdA", 
        "WScore": "ScoreA", 
        "LTeamID": "TeamIdB",
        "LScore": "ScoreB",
    }
    win_rename.update({c: c[:-1] + "A" for c in df.columns if c.endswith('W')})
    win_rename.update({c: c[:-1] + "B" for c in df.columns if c.endswith('L')})
    lose_rename = {
        "WTeamID": "TeamIdB", 
        "WScore": "ScoreB", 
        "LTeamID": "TeamIdA",
        "LScore": "ScoreA",
    }
    lose_rename.update({c: c[:-1] + "B" for c in df.columns if c.endswith('W')})
    lose_rename.update({c: c[:-1] + "A" for c in df.columns if c.endswith('L')})
    win_df = df.copy().rename(columns=win_rename)
    lose_df = df.copy().rename(columns=lose_rename)
    return pd.concat([win_df, lose_df], axis=0, sort=False)

df = add_loosing_matches(df)

# --- Compute Difference Features ---
cols_to_diff = ['Seed', 'WinRatio', 'GapAvg']
for col in cols_to_diff:
    df[col + 'Diff'] = df[col + 'A'] - df[col + 'B']
df['ScoreDiff'] = df['ScoreA'] - df['ScoreB']
df['WinA'] = (df['ScoreDiff'] > 0).astype(int)

#############################################
# 2. Prepare the Test Data
#############################################
df_test = pd.read_csv(os.path.join(DATA_PATH, "SampleSubmissionStage1.csv"))
df_test['Season'] = df_test['ID'].apply(lambda x: int(x.split('_')[0]))
df_test['TeamIdA'] = df_test['ID'].apply(lambda x: int(x.split('_')[1]))
df_test['TeamIdB'] = df_test['ID'].apply(lambda x: int(x.split('_')[2]))

df_test = pd.merge(
    df_test, df_seeds,
    how='left',
    left_on=['Season', 'TeamIdA'],
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedA'}).fillna('W01')

df_test = pd.merge(
    df_test, df_seeds,
    how='left',
    left_on=['Season', 'TeamIdB'],
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedB'}).fillna('W01')

df_test['SeedA'] = df_test['SeedA'].apply(treat_seed)
df_test['SeedB'] = df_test['SeedB'].apply(treat_seed)

df_test = pd.merge(
    df_test, df_features_season,
    how='left',
    left_on=['Season', 'TeamIdA'],
    right_on=['Season', 'TeamID']
).rename(columns={'WinRatio': 'WinRatioA', 'GapAvg': 'GapAvgA'}).drop('TeamID', axis=1)

df_test = pd.merge(
    df_test, df_features_season,
    how='left',
    left_on=['Season', 'TeamIdB'],
    right_on=['Season', 'TeamID']
).rename(columns={'WinRatio': 'WinRatioB', 'GapAvg': 'GapAvgB'}).drop('TeamID', axis=1)

df_test["SeedDiff"] = df_test["SeedA"] - df_test["SeedB"]
df_test["WinRatioDiff"] = df_test["WinRatioA"] - df_test["WinRatioB"]
df_test["GapAvgDiff"] = df_test["GapAvgA"] - df_test["GapAvgB"]

#############################################
# 3. Define Features & Convert to GPU (cuDF)
#############################################
# Numeric features:
features_numeric = ['SeedA', 'SeedB', 'WinRatioA', 'GapAvgA', 'WinRatioB', 'GapAvgB', 'SeedDiff', 'WinRatioDiff', 'GapAvgDiff']
# Categorical columns:
cat_cols = ['TeamIdA', 'TeamIdB']
features_gpu = features_numeric + [col + '_target' for col in cat_cols]

#############################################
# GPU Target Encoding using cuDF
#############################################
def gpu_target_encode_train(df, cat_cols, target):
    for col in cat_cols:
        df[col] = df[col].astype(str)
    global_mean = df[target].mean()
    mappings = {}
    for col in cat_cols:
        mapping = df.groupby(col)[target].mean().reset_index()
        mapping = mapping.rename(columns={target: col + '_target'})
        mappings[col] = mapping
        df = df.merge(mapping, on=col, how='left')
        df[col + '_target'] = df[col + '_target'].fillna(global_mean)
    return df, mappings

def gpu_target_encode_test(df, mappings, cat_cols):
    for col in cat_cols:
        df[col] = df[col].astype(str)
        mapping = mappings[col]
        df = df.merge(mapping, on=col, how='left')
        global_mean = mapping[col + '_target'].mean()
        df[col + '_target'] = df[col + '_target'].fillna(global_mean)
    return df

#############################################
# Convert Pandas DataFrames to cuDF DataFrames
#############################################
gdf = cudf.DataFrame.from_pandas(df)
gdf_test = cudf.DataFrame.from_pandas(df_test)

gdf, mappings = gpu_target_encode_train(gdf, cat_cols, 'WinA')
gdf_test = gpu_target_encode_test(gdf_test, mappings, cat_cols)

gdf = gdf.reset_index(drop=True)
gdf_test = gdf_test.reset_index(drop=True)

#############################################
# GPU Rescaling: median imputation + min–max scaling using cuDF
#############################################
def gpu_rescale(df, features):
    for col in features:
        df[col] = df[col].fillna(0)
    mins = df[features].min()
    maxs = df[features].max()
    for col in features:
        diff = maxs[col] - mins[col]
        if diff == 0:
            df[col] = 0.0
        else:
            df[col] = (df[col] - mins[col]) / diff
    return df

gdf = gpu_rescale(gdf, features_gpu)
gdf_test = gpu_rescale(gdf_test, features_gpu)

#############################################
# 4. Train a GPU-Enabled CatBoost Model on 50% of Training Data
#############################################
from catboost import CatBoostClassifier
# Convert GPU preprocessed training data back to Pandas.
X_full = gdf[features_gpu].to_pandas()
y_full = gdf['WinA'].to_pandas()

# Use 50% of the training data for model fitting.
X_train_half, X_val_half, y_train_half, y_val_half = train_test_split(
    X_full, y_full, train_size=0.5, random_state=42, stratify=y_full
)

cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    task_type="GPU",
    devices='0',
    verbose=100
)

cat_model.fit(X_train_half, y_train_half)

#############################################
# 5. Save, Load, and Iterate the Model 7 Times (Evaluate on 50% Training Data)
#############################################
cv_mses = []
for i in range(7):
    model_filename = f"catboost_model_iter_{i}.cbm"
    # Save the model using CatBoost's built-in method.
    cat_model.save_model(model_filename)
    # Load the model into a new instance.
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_filename)
    preds = loaded_model.predict_proba(X_train_half)[:, 1]
    mse = ((y_train_half - preds)**2).mean()
    cv_mses.append(mse)
    print(f"Iteration {i}: Training MSE = {mse:.4f}")

best_iter = int(np.argmin(cv_mses))
print(f"Best iteration: {best_iter} with Training MSE = {cv_mses[best_iter]:.4f}")

#############################################
# 6. Generate Test Set Predictions with the Best Model
#############################################
best_model_filename = f"catboost_model_iter_{best_iter}.cbm"
final_model = CatBoostClassifier()
final_model.load_model(best_model_filename)

X_test = gdf_test[features_gpu].to_pandas()
preds_test = final_model.predict_proba(X_test)[:, 1]
gdf_test['pred'] = preds_test

final_sub = gdf_test[['ID', 'pred']].to_pandas()
final_sub.to_csv('submission.csv', index=False)
print(final_sub.head())

sns.displot(final_sub['pred'], kde=True)
plt.title("Distribution of CatBoost Predictions (50% Training Data)")
plt.show()





