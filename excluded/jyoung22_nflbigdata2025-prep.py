#########################################################
# 1. Imports & Setup
#########################################################
import os
import gc
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)
import xgboost as xgb

warnings.filterwarnings('ignore')

# Kaggle environment (adjust if needed)
project_dir = '/kaggle/input/nfl-big-data-bowl-2025'

#########################################################
# 2. Define Columns & Helper Functions
#########################################################
plays_usecols = [
    'gameId','playId','penaltyYards','yardsGained',
    'preSnapHomeTeamWinProbability','quarter','down','yardsToGo',
    'absoluteYardlineNumber','offenseFormation','receiverAlignment',
    'gameClock','playDescription'
]
players_usecols = [
    'nflId','displayName','birthDate','collegeName',
    'position','height','weight'
]
games_usecols = [
    'gameId','homeTeamAbbr','visitorTeamAbbr','week','gameDate'
]
tracking_usecols = [
    'time','jerseyNumber','club','playDirection',
    'x','y','s','a','dis',
    'o','dir','event','frameId','frameType',
    'gameId','playId','nflId'
]
pff_usecols = [
    'gameId','playId','nflId','teamAbbr','OFF','PASS','RECV','RUN','RBLK'
]

def optimize_dtypes(dataframe):
    """Downcast numeric types & convert object columns to categories (where feasible)."""
    for col in dataframe.select_dtypes(include='float'):
        dataframe[col] = pd.to_numeric(dataframe[col], downcast='float')
    for col in dataframe.select_dtypes(include='int'):
        dataframe[col] = pd.to_numeric(dataframe[col], downcast='integer')
    for col in dataframe.select_dtypes(include='object'):
        num_unique_values = dataframe[col].nunique()
        num_total_values = len(dataframe[col])
        if float(num_unique_values) / num_total_values < 0.5:
            dataframe[col] = dataframe[col].astype('category')
    return dataframe

def determine_play_type(description):
    """Classify pass vs run (simple approach)."""
    desc_lower = str(description).lower()
    if 'pass' in desc_lower:
        return 'pass'
    elif ('left end' in desc_lower) or ('right end' in desc_lower) or ('up the middle' in desc_lower):
        return 'run'
    else:
        return 'unknown'

def feature_engineering_pipeline(data_subset, max_frames=350):
    """
    1) Filter out ball_snapped frames
    2) Identify QB
    3) Compute differences vs. QB
    4) Expand to time-series columns
    5) Forward fill missing values
    """
    columns_of_interest = [
        'gameId','playId','nflId','frameId','frameType','o','y',
        'position','down','yardsToGo'
    ]
    df_reduced = data_subset[columns_of_interest].copy()

    # Filter frames leading up to ball_snapped
    df_reduced = df_reduced[df_reduced['frameType'] != 'ball_snapped']

    # Identify QB in each frame
    df_qb = df_reduced[df_reduced['position'] == 'QB'][['gameId','playId','frameId','o','y']]
    df_qb = df_qb.rename(columns={'o':'o_qb','y':'y_qb'})

    # Merge QB data
    df_merged = pd.merge(df_reduced, df_qb, on=['gameId','playId','frameId'], how='left')

    # Compute difference from QB
    df_merged['y_diff_qb'] = df_merged['y'] - df_merged['y_qb']
    df_merged['o_diff_qb'] = df_merged['o'] - df_merged['o_qb']

    # Time-series columns
    variables = ['o','y']
    time_series_cols = [f"{var}_diff_qb_{i}" for var in variables for i in range(1, max_frames+1)]

    df_merged = df_merged.reset_index(drop=True)
    df_result = pd.DataFrame(
        np.nan,
        index=df_merged.index,
        columns=['nflId','playId','gameId','frameId','down','yardsToGo'] + time_series_cols
    )
    df_result[['nflId','playId','gameId','frameId','down','yardsToGo']] = \
        df_merged[['nflId','playId','gameId','frameId','down','yardsToGo']]

    # Fill time-series columns frame-by-frame
    unique_frames = sorted(df_merged['frameId'].unique())
    max_frame = min(len(unique_frames), max_frames)
    for frame in unique_frames[:max_frame]:
        frame_suffix = f"_{frame}"
        frame_data = df_merged[df_merged['frameId'] == frame]
        df_result.loc[frame_data.index, f'o_diff_qb{frame_suffix}'] = frame_data['o_diff_qb'].values
        df_result.loc[frame_data.index, f'y_diff_qb{frame_suffix}'] = frame_data['y_diff_qb'].values

    # Forward fill across frames per (nflId, playId, gameId)
    df_result[time_series_cols] = df_result.groupby(
        ['nflId','playId','gameId']
    )[time_series_cols].ffill()
    df_result[time_series_cols] = df_result[time_series_cols].fillna(0)

    df_result.drop_duplicates(subset=['nflId','playId','gameId','frameId'], inplace=True)
    return df_result

def sample_rows(df, max_rows=300_000, random_state=42):
    """Sample the DataFrame if it exceeds `max_rows`."""
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_state)
    return df

def aggregate_columns_by_mean(dataset, prefix, group_size=4):
    """Aggregate time-series columns that start with `prefix` into mean features."""
    matching_columns = [col for col in dataset.columns if col.startswith(prefix)]
    grouped_columns = [matching_columns[i:i + group_size] for i in range(0, len(matching_columns), group_size)]
    for i, group in enumerate(grouped_columns):
        dataset[f'{prefix}_mean_{i+1}'] = dataset[group].mean(axis=1)
    dataset.drop(columns=matching_columns, inplace=True)
    return dataset


#########################################################
# 3. Data Preparation Function
#########################################################
def prepare_data(weeks=range(1,10), chunk_size=500_000):
    """
    Reads each tracking_week_X.csv in chunks, merges with plays, players, games, PFF,
    runs feature engineering, samples, merges chunk outputs, returns a final DataFrame.

    Returns: DataFrame (df_all) with all weeks combined and key columns (including time-series expansions).
    """
    # A) Load small tables
    plays = pd.read_csv(os.path.join(project_dir, "plays.csv"), usecols=plays_usecols)
    plays["play_label"] = plays["playDescription"].apply(determine_play_type)
    plays = optimize_dtypes(plays)

    players = pd.read_csv(os.path.join(project_dir, "players.csv"), usecols=players_usecols)
    players = optimize_dtypes(players)

    games = pd.read_csv(os.path.join(project_dir, "games.csv"), usecols=games_usecols)
    games = optimize_dtypes(games)

    # Load PFF data
    pff_path = "/kaggle/input/nfl-bigdata-pffscores-2022/merged_player_offensive_pff_stats_corrected.csv"
    pff_stats = pd.read_csv(pff_path, usecols=pff_usecols)
    pff_stats = optimize_dtypes(pff_stats)

    positions_to_keep = ["C","WR","G","T","QB","RB","FB","TE"]

    weekly_final_files = []

    for w in weeks:
        tracking_csv = os.path.join(project_dir, f"tracking_week_{w}.csv")
        print(f"\n--- Processing Week {w} => {tracking_csv} ---")
        
        chunk_parquets = []
        chunk_num = 0

        # Read CSV in row-based sub-chunks
        for chunk in pd.read_csv(tracking_csv, usecols=tracking_usecols, chunksize=chunk_size):
            chunk_num += 1
            print(f"  - Chunk {chunk_num}: shape {chunk.shape}")

            # Filter BEFORE_SNAP & valid nflId
            chunk = chunk[(chunk["frameType"] == "BEFORE_SNAP") & (chunk["nflId"].notna())]
            chunk = optimize_dtypes(chunk)

            # Merge with plays, players, games, pff
            chunk = chunk.merge(plays, on=["gameId","playId"], how="left")
            chunk = chunk.merge(players, on="nflId", how="left")
            chunk = chunk.merge(games, on="gameId", how="left")
            chunk = chunk.merge(pff_stats, on=["gameId","playId","nflId"], how="left")
            chunk = optimize_dtypes(chunk)

            # Filter positions
            chunk = chunk[chunk["position"].isin(positions_to_keep)]

            # Feature Engineering
            fe_chunk = feature_engineering_pipeline(chunk)

            # Also keep columns from chunk that we want in final data
            keep_cols = [
                "gameId","playId","nflId","offenseFormation","position","down","yardsToGo",
                "play_label","teamAbbr","OFF","PASS","RECV","RUN","RBLK"
            ]
            keep_cols = list(set(keep_cols).intersection(chunk.columns))
            extra_info = chunk[keep_cols].drop_duplicates(["gameId","playId","nflId"])
            
            fe_chunk = fe_chunk.merge(extra_info, on=["gameId","playId","nflId"], how="left")

            # Filter out unknown label
            fe_chunk = fe_chunk[fe_chunk["play_label"] != "unknown"]

            # Sample
            fe_chunk = sample_rows(fe_chunk, max_rows=2_000_000)

            # Save chunk to Parquet
            out_path = f"week_{w}_chunk_{chunk_num}.parquet"
            fe_chunk.to_parquet(out_path, index=False)
            chunk_parquets.append(out_path)

            del chunk, fe_chunk, extra_info
            gc.collect()
        
        # Combine chunk files for this week
        df_week_parts = []
        for cp in chunk_parquets:
            tmp = pd.read_parquet(cp)
            df_week_parts.append(tmp)
            del tmp
            gc.collect()

        df_week = pd.concat(df_week_parts, ignore_index=True)
        del df_week_parts
        gc.collect()

        # Optional: sample the entire week's data
        df_week = sample_rows(df_week, max_rows=1_000_000)

        # Save final weekly file
        week_file = f"week_{w}_final.parquet"
        df_week.to_parquet(week_file, index=False)
        weekly_final_files.append(week_file)

        print(f"Week {w} final shape {df_week.shape}")
        del df_week
        gc.collect()

    # Combine all weeks
    print("\nCombining weekly files:")
    final_dfs = []
    for wf in weekly_final_files:
        tmp = pd.read_parquet(wf)
        # optional sampling again
        if len(tmp) > 500_000:
            tmp = tmp.sample(n=100_000, random_state=42)
        final_dfs.append(tmp)
        del tmp
        gc.collect()

    df_all = pd.concat(final_dfs, ignore_index=True)
    print(f"\nFinal combined shape: {df_all.shape}")
    del final_dfs
    gc.collect()

    return df_all


#########################################################
# 4. Modeling Function
#########################################################
def run_model(df_all):
    """
    Takes the final DataFrame from prepare_data,
    does train/test split, optional time-series aggregation,
    label encoding, trains XGBoost, outputs model + predictions.
    """
    # Train-Test Split (by playId)
    unique_plays = df_all["playId"].unique()
    train_plays, test_plays = train_test_split(unique_plays, test_size=0.2, random_state=42)

    train_data = df_all[df_all["playId"].isin(train_plays)].copy()
    test_data  = df_all[df_all["playId"].isin(test_plays)].copy()

    del df_all
    gc.collect()

    print(f"Train shape (before aggregation): {train_data.shape}, Test shape: {test_data.shape}")

    # Aggregate time-series columns (optional)
    train_data = aggregate_columns_by_mean(train_data, prefix='o_diff', group_size=4)
    train_data = aggregate_columns_by_mean(train_data, prefix='y_diff', group_size=4)

    test_data = aggregate_columns_by_mean(test_data, prefix='o_diff', group_size=4)
    test_data = aggregate_columns_by_mean(test_data, prefix='y_diff', group_size=4)

    # Drop NAs, label encode
    train_data.dropna(subset=["play_label"], inplace=True)
    test_data.dropna(subset=["play_label"], inplace=True)

    lbl = LabelEncoder()
    train_data["play_label_encoded"] = lbl.fit_transform(train_data["play_label"])
    test_data["play_label_encoded"]  = lbl.transform(test_data["play_label"])

    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    # Check if test has valid classes
    if test_data.shape[0] < 1 or len(test_data["play_label"].unique()) < 2:
        print("Skipping modeling because test set is not valid for AUC.")
        return None, None, None

    print(f"Train final shape: {train_data.shape}, Test final shape: {test_data.shape}")

    # Build feature matrices
    non_feature_cols = [
        'nflId','gameId','playId','frameId','play_label','play_label_encoded'
    ]
    feature_cols = [c for c in train_data.columns if c not in non_feature_cols]

    X_train = train_data[feature_cols]
    y_train = train_data["play_label_encoded"]

    X_test = test_data[feature_cols]
    y_test = test_data["play_label_encoded"]

    del train_data, test_data
    gc.collect()

    # Train XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest  = xgb.DMatrix(X_test,  label=y_test)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "seed": 42,
        "max_depth": 3
    }
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=100,
        early_stopping_rounds=10,
        evals=[(dtrain, "train"), (dtest, "test")],
        verbose_eval=10
    )

    # Evaluate
    y_pred_proba = model.predict(dtest)
    auc_val = roc_auc_score(y_test, y_pred_proba)
    print(f"Test AUC: {auc_val:.4f}")

    # Build predictions DataFrame
    X_test_with_id = X_test.reset_index(drop=True)
    y_test_array = y_test.reset_index(drop=True)

    predictions_df = pd.DataFrame({
        'y_pred_proba': y_pred_proba,
        'y_test':       y_test_array
    })
    for col in ["frameId","gameId","nflId","playId"]:
        if col in X_test_with_id.columns:
            predictions_df[col] = X_test_with_id[col]

    predictions_df.to_csv('predictions_and_actuals.csv', index=False)
    print("Saved predictions_and_actuals.csv, shape:", predictions_df.shape)

    return model, predictions_df, feature_cols


#########################################################
# 5. Main
#########################################################
def main():
    # 5A) Prepare data (weeks 1..9, chunk size ~500k, etc.)
    df_all = prepare_data(weeks=range(1,10), chunk_size=2_000_000)

    # 5B) Run modeling
    model, preds, feat_cols = run_model(df_all)

    # Optionally return them so you can keep analyzing
    return model, preds, feat_cols

if __name__ == "__main__":
    model, predictions, features = main()
    if model is not None:
        print("\nAll done! Model training completed.")
        print("Predictions shape:", predictions.shape)
        print("Feature columns used:", features)


