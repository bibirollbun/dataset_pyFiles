import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import torch
from joblib import Parallel, delayed
import warnings
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
warnings.filterwarnings('ignore')


### Configuration class
class Config:
    ROOT_DIR = Path('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final')
    
    SEED = 42
    K_FOLD = 5
    N_EPOCHS = 150
    PATIENCE = 15
    LR = 1e-3 # If training from scratch that should be maybe higher
    BATCH_SIZE = 128
    
    Y_MIN, Y_MAX = 0.0, 53.3
    X_MIN, X_MAX = 0.0, 120.0
    
    N_JOBS = 8  
    IS_NOTEBOOK = True
    
    #Attention Model Config
    pass
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def set_seed(self):
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed_all(self.SEED)
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        os.environ['PYTHONHASHSEED'] = str(self.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


### Load Data function
def load_single_week_data(TRAIN_ROOT_DIR, week):
    
    try:
        input_path = f"input_2023_w{week:02d}.csv"
        output_path = f"output_2023_w{week:02d}.csv"
        
        if not os.path.exists(os.path.join(TRAIN_ROOT_DIR, input_path)):
            raise FileNotFoundError(f"The file {input_path} does not exist in {TRAIN_ROOT_DIR}.")
        if not os.path.exists(os.path.join(TRAIN_ROOT_DIR, output_path)):
            raise FileNotFoundError(f"The file {output_path} does not exist in {TRAIN_ROOT_DIR}.")
        
        input_df, output_df = pd.read_csv(os.path.join(TRAIN_ROOT_DIR, input_path)), pd.read_csv(os.path.join(TRAIN_ROOT_DIR, output_path))
        
        return input_df, output_df
        
    except Exception as e:
        print(f"Error loading data for week {week:02d}: {e}")
        return None, None

def load_data(ROOT_DIR, N_JOBS):
    if not os.path.exists(ROOT_DIR):
        raise FileNotFoundError(f"The DIR {ROOT_DIR} does not exist.")

    train_root_dir = os.path.join(ROOT_DIR, 'train')
    
    if not os.path.exists(train_root_dir):
        raise FileNotFoundError(f"The DIR {train_root_dir} does not exist.")
    
    weeks = list(range(1, 19))
    results = Parallel(n_jobs=N_JOBS, backend='threading')(delayed(load_single_week_data)(train_root_dir, week) for week in tqdm(weeks, desc="Loading data"))
    
    input_dfs, output_dfs = zip(*results)
    input_df = pd.concat([df for df in input_dfs if df is not None], ignore_index=True)
    output_df = pd.concat([df for df in output_dfs if df is not None], ignore_index=True)
    
    print("Data loading complete.")
    print(f"Input data shape: {input_df.shape}")
    print(f"Output data shape: {output_df.shape}")
    
    print("Columns in input data:\n", input_df.columns.tolist())
    print("-"*50)
    print("Columns in output data:\n", output_df.columns.tolist())
    
    return input_df, output_df


# Plot functions
def plot_distribution(data, column, title, xlabel, ylabel, notebook=False):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], bins=30, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if notebook:
        plt.show()
    plt.savefig(f"{column}_distribution.png")


### Future Engineering functions
def merge_output_and_input(input_df, output_df):
    merged_df = (
        input_df.merge(
                output_df,
                on=["game_id", "play_id", "nfl_id", "frame_id"],
                how="inner",
                suffixes=("_input", "_target")
                )
            )
    
    print(f"Merged data shape: {merged_df.shape}")
    print("Columns in merged data:\n", merged_df.columns.tolist())

    return merged_df


def add_additional_features(df, is_notebook=False):
    df = df.copy()

    height_ft = df["player_height"].str.split("-", expand=True)[0].astype(float)
    height_in = df["player_height"].str.split("-", expand=True)[1].astype(float)
    df["height_m"] = (height_ft * 12 + height_in) * 0.0254
    df["weight_kg"] = df["player_weight"].astype(float) * 0.453592

    # Calculate BMI
    df["BMI"] = df["weight_kg"] / (df["height_m"] ** 2)
    plot_distribution(df, "BMI", "BMI Distribution", "BMI", "Frequency", is_notebook)

    # Calculate age
    birth_year = pd.to_datetime(df["player_birth_date"]).dt.year
    df["Age"] = 2023 - birth_year
    plot_distribution(df, "Age", "Age Distribution", "Age", "Frequency", is_notebook)
    
    
    #One hot encode player side, role and position
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    side_and_role_position_encoded = ohe.fit_transform(df[["player_side", "player_role", "player_position"]])
    side_role_position_cols = ohe.get_feature_names_out(["player_side", "player_role", "player_position"])
    df_side_role_position = pd.DataFrame(side_and_role_position_encoded, columns=side_role_position_cols, index=df.index)
    
    df = pd.concat([df, df_side_role_position], axis=1)
    
    #Degree to radian conversions and trigonometric features
    df["dir_rad"] = np.deg2rad(df["dir"])
    df["o_rad"]   = np.deg2rad(df["o"])
    
    df["dir_sin"] = np.sin(df["dir_rad"])
    df["dir_cos"] = np.cos(df["dir_rad"])
    df["o_sin"]   = np.sin(df["o_rad"])
    df["o_cos"]   = np.cos(df["o_rad"])
    
    #Components of velocity and acceleration
    df["vx"] = df["s"] * df["dir_cos"]
    df["vy"] = df["s"] * df["dir_sin"]
    df["ax"] = df["a"] * df["dir_cos"]
    df["ay"] = df["a"] * df["dir_sin"]
    
    #Angle difference between orientation and direction
    df["angle_diff"] = np.abs(df["o"] - df["dir"])
    df["angle_diff"] = np.where(df["angle_diff"] > 180, 360 - df["angle_diff"], df["angle_diff"])
    #Angle difference in radians
    df["angle_diff_rad"] = np.deg2rad(df["angle_diff"])
    plot_distribution(df, "angle_diff", "Angle Difference Distribution", "Angle Difference (degrees)", "Frequency", is_notebook)
    
    #Distance to ball landing spot
    df["dist_to_ball"] = np.sqrt((df["x_input"] - df["ball_land_x"])**2 + (df["y_input"] - df["ball_land_y"])**2)
    plot_distribution(df, "dist_to_ball", "Distance to Ball Landing Spot Distribution", "Distance to Ball Landing Spot", "Frequency", is_notebook)
    
    #Play direction encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    play_direction_encoded = encoder.fit_transform(df[["play_direction"]])
    play_direction_cols = encoder.get_feature_names_out(["play_direction"])
    df_play_direction = pd.DataFrame(play_direction_encoded, columns=play_direction_cols, index=df.index)
    
    df = pd.concat([df, df_play_direction], axis=1)
    
    
    #Drop unused columns
    df.drop(columns=["player_side", "player_birth_date", "player_height", "player_weight"], inplace=True)
    
    return df

def aggregate_features(df, groupby_cols, agg_cols):
    agg_df = df.groupby(groupby_cols)[agg_cols].agg(['mean', 'std', 'min', 'max'])
    
    agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]
    
    agg_df = agg_df.reset_index()
    
    print(f"Aggregated data shape: {agg_df.shape}")
    return agg_df
    
if __name__ == "__main__":
    config = Config()
    config.set_seed()
    
    print(f"Using device: {config.DEVICE}")
    print(f"Root directory: {config.ROOT_DIR}")
    print(f"Training for {config.N_EPOCHS} epochs with batch size {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LR}")
    print(f"K-Fold Cross Validation with K={config.K_FOLD}")
    print(f"Early stopping patience: {config.PATIENCE} epochs")
    print(f"Y range: [{config.Y_MIN}, {config.Y_MAX}]")
    print(f"X range: [{config.X_MIN}, {config.X_MAX}]")
    print("Configuration setup complete.")
    
    # Load data
    input_df, output_df = load_data(config.ROOT_DIR, config.N_JOBS)
    
    merged_df = merge_output_and_input(input_df, output_df)
    
    feature_engineered_df = add_additional_features(merged_df, config.IS_NOTEBOOK)
    
    print("Feature engineering complete.")
    print(f"Final data shape: {feature_engineered_df.shape}")
    print("Columns in final data:\n", feature_engineered_df.columns.tolist())
    print(feature_engineered_df.head())
    
    groupby_cols = ['nfl_id', 'player_name', 'player_position', 'player_role', 'game_id']
    agg_cols = ['s', 'a', 'vx', 'vy', 'ax', 'ay', 'dist_to_ball']
    
    aggregated_df = aggregate_features(feature_engineered_df, groupby_cols=groupby_cols, agg_cols=agg_cols)
    
    merged_df = pd.merge(feature_engineered_df, aggregated_df, on=groupby_cols, how='left', suffixes=('', '_agg'))
    final_df = merged_df.drop(columns=agg_cols)
    print(f"Final data shape after aggregation: {final_df.shape}")
    print('Columns in final data after aggregation:\n', final_df.columns.tolist())
    
    print("Aggregation complete. Feature engineering pipeline finished.")
    print(final_df.head())

