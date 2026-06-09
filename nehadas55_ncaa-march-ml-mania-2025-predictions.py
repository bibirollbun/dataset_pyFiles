try:
    import cudf
    import cuml
except ImportError:
    import os
    print("cuML not found. Installing it now...")
    os.system("pip install cudf cuml --extra-index-url=https://pypi.nvidia.com")
    import cudf
    import cuml


import os
import time
import torch
import itertools
import warnings
import optuna
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import tensorflow as tf
from tqdm import tqdm
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from cuml.linear_model import LogisticRegression as cuLogisticRegression
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.svm import SVC as cuSVM
from cuml.neighbors import KNeighborsClassifier as cuKNN
from xgboost import XGBClassifier, XGBRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Ignore all warnings!
warnings.filterwarnings("ignore")
os.environ["XGBOOST_VERBOSITY"] = "0"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


!nvidia-smi


device = "cuda" if torch.cuda.is_available() else "cpu"
print("GPU Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ==============================
# Data Loading
# ==============================

# Set the path for Kaggle dataset (adjust based on your Kaggle Notebook setup)
data_path = "../input/march-machine-learning-mania-2025/"


def safe_read_csv(filepath, usecols=None, dtype=None):
    """Helper function to safely read CSVs, returning None if the file is missing."""
    return pd.read_csv(filepath, usecols=usecols, dtype=dtype) if os.path.exists(filepath) else None


def extract_seed(seed):
    """Extracts numeric seed value from tournament seed data."""
    if isinstance(seed, str) and len(seed) > 1:
        return int(seed[1:3])
    return np.nan


def data_visualization(prefix, tournament_data):
    """
    Generate data visualizations for:
    - Win percentages
    - Point differentials
    - Tournament seed performance
    """

    gender_label = "Men's" if prefix == "M" else "Women's"

    # Correct column names
    win_pct_col = "WinPct"
    point_diff_col = "AvgPointDiff"
    seed_col = "WTeamSeed"

    # Check if required columns exist
    if win_pct_col not in tournament_data.columns:
        print(f"Warning: {win_pct_col} not found in dataset.")
        return

    if point_diff_col not in tournament_data.columns:
        print(f"Warning: {point_diff_col} not found in dataset.")
        return

    if seed_col not in tournament_data.columns:
        print(f"Warning: {seed_col} not found in dataset.")
        return

    # Win Percentage Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(tournament_data[win_pct_col], bins=20, kde=True)
    plt.title(f"{gender_label} NCAA - Win Percentage Distribution")
    plt.xlabel("Win Percentage")
    plt.ylabel("Frequency")
    plt.show()

    # Point Differential Trends
    plt.figure(figsize=(10, 6))
    sns.histplot(tournament_data[point_diff_col], bins=20, kde=True)
    plt.title(f"{gender_label} NCAA - Average Point Differential")
    plt.xlabel("Point Differential")
    plt.ylabel("Frequency")
    plt.show()

    # Tournament Seed Performance vs Win Percentage
    filtered_data = tournament_data.dropna(subset=[seed_col, win_pct_col])
    if not filtered_data.empty:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=filtered_data[seed_col].astype(int), y=filtered_data[win_pct_col])
        plt.title(f"{gender_label} NCAA - Seed Strength vs. Win Percentage")
        plt.xlabel("Tournament Seed")
        plt.ylabel("Win Percentage")
        plt.show()
    else:
        print(f"Warning: No data available for {seed_col} and {win_pct_col}.")


def plot_correlation_heatmap(df, title):
    """
    Plots a correlation heatmap for the given dataset.

    Parameters:
    - df: Processed DataFrame with numerical features.
    - title: Title for the heatmap.
    """
    plt.figure(figsize=(12, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    sns.heatmap(
        correlation_matrix, 
        annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5
    )
    plt.title(title)
    plt.show()
    time.sleep(20) # Pause for heatmap to show.


def process_ncaa_data(prefix, data_path):
    """
    Processes NCAA data for both men and women, including:
    - Regular season results (compact + detailed)
    - Tournament results (compact + detailed)
    - Team information, game cities, conference affiliations
    - Tournament seeds and slots
    - Massey Ordinals (for men)
    - Secondary tournaments

    Parameters:
    - prefix: 'M' for men's data, 'W' for women's data
    - data_path: Path to dataset folder

    Returns:
    - final_data: Merged dataset with features
    """

    dtype_map = {
        "Season": "int16", "TeamID": "int16", "WTeamID": "int16", 
        "LTeamID": "int16", "OrdinalRank": "float32"
    }

    # Load core datasets
    teams = pd.read_csv(os.path.join(data_path, f"{prefix}Teams.csv"))
    reg_results = pd.read_csv(os.path.join(data_path, f"{prefix}RegularSeasonCompactResults.csv"))
    detailed_results = pd.read_csv(os.path.join(data_path, f"{prefix}RegularSeasonDetailedResults.csv"))
    tourney_results = pd.read_csv(os.path.join(data_path, f"{prefix}NCAATourneyCompactResults.csv"))
    tourney_detailed = pd.read_csv(os.path.join(data_path, f"{prefix}NCAATourneyDetailedResults.csv"))
    tourney_seeds = pd.read_csv(os.path.join(data_path, f"{prefix}NCAATourneySeeds.csv"))
    tourney_slots = pd.read_csv(os.path.join(data_path, f"{prefix}NCAATourneySlots.csv"))
    team_conferences = pd.read_csv(os.path.join(data_path, f"{prefix}TeamConferences.csv"))
    conf_tourney_results = pd.read_csv(os.path.join(data_path, f"{prefix}ConferenceTourneyGames.csv"))
    secondary_tourney_results = pd.read_csv(os.path.join(data_path, f"{prefix}SecondaryTourneyCompactResults.csv"))
    secondary_tourney_teams = pd.read_csv(os.path.join(data_path, f"{prefix}SecondaryTourneyTeams.csv"))
    game_cities = pd.read_csv(os.path.join(data_path, f"{prefix}GameCities.csv"))

    # Load common datasets
    conferences = pd.read_csv(os.path.join(data_path, "Conferences.csv"))
    cities = pd.read_csv(os.path.join(data_path, "Cities.csv"))

    # For Men's Data: Include Massey Ordinals
    if prefix == "M":
        team_rankings = pd.read_csv(os.path.join(data_path, f"{prefix}MasseyOrdinals.csv"), 
                                    usecols=["Season", "TeamID", "OrdinalRank"])
        latest_rankings = team_rankings.sort_values(["Season", "TeamID", "OrdinalRank"]).drop_duplicates(["Season", "TeamID"], keep="last")

    # ==============================
    # Debugging Column Names
    # ==============================
    print("\n===== Checking Columns Before Merge =====")
    print("Tourney Results Columns:", tourney_results.columns)
    print("Tourney Seeds Columns:", tourney_seeds.columns)
    print("Tourney Slots Columns:", tourney_slots.columns)

    # Ensure `TeamID` exists in `tourney_seeds`
    if "TeamID" not in tourney_seeds.columns:
        print("WARNING: 'TeamID' not found in tourney_seeds! Checking possible alternatives...")
        print("Available Columns in tourney_seeds:", tourney_seeds.columns)

    # ==============================
    # Data Merging
    # ==============================

    # Merge game cities with cities info
    game_cities = game_cities.merge(cities, on="CityID", how="left")

    # Merge teams with their conferences
    team_conferences = team_conferences.merge(conferences, on="ConfAbbrev", how="left")

    # Merge tournament seeds using `WTeamID` and `LTeamID`
    tourney_results = tourney_results.merge(
        tourney_seeds.rename(columns={"TeamID": "WTeamID"}), 
        on=["Season", "WTeamID"], 
        how="left"
    ).rename(columns={"Seed": "WTeamSeed"})

    tourney_results = tourney_results.merge(
        tourney_seeds.rename(columns={"TeamID": "LTeamID"}), 
        on=["Season", "LTeamID"], 
        how="left"
    ).rename(columns={"Seed": "LTeamSeed"})

    # Merge tournament slots
    tourney_results = tourney_results.merge(tourney_slots, on="Season", how="left")

    # ==============================
    # Fix the 'WScore' Issue by Renaming Before Merging
    # ==============================
    print("Columns in reg_results before merging detailed_results:", reg_results.columns)
    print("Columns in detailed_results:", detailed_results.columns)

    # Rename detailed_results columns to prevent overwriting
    detailed_results = detailed_results.rename(columns={
        "WScore": "DWScore",  # Detailed WScore
        "LScore": "DLScore"   # Detailed LScore
    })

    # Merge detailed stats into regular season results
    reg_results = reg_results.merge(
        detailed_results[["Season", "DayNum", "WTeamID", "LTeamID", "DWScore", "DLScore"]],
        on=["Season", "DayNum", "WTeamID", "LTeamID"],
        how="left"
    )

    # Ensure `WScore` and `LScore` exist after merging
    reg_results["WScore"] = reg_results["DWScore"].fillna(reg_results["WScore"])
    reg_results["LScore"] = reg_results["DLScore"].fillna(reg_results["LScore"])
    reg_results.drop(columns=["DWScore", "DLScore"], inplace=True)

    # Merge Massey Ordinals (Men's Only)
    if prefix == "M":
        reg_results = reg_results.merge(
            latest_rankings.rename(columns={"TeamID": "WTeamID", "OrdinalRank": "WTeamRank"}), 
            on=["Season", "WTeamID"], 
            how="left"
        )

        reg_results = reg_results.merge(
            latest_rankings.rename(columns={"TeamID": "LTeamID", "OrdinalRank": "LTeamRank"}), 
            on=["Season", "LTeamID"], 
            how="left"
        )

    # ==============================
    # Merge Secondary Tournament Data
    # ==============================
    print("\n===== Checking Columns Before Merge (Secondary Tournament) =====")
    print("Secondary Tourney Results Columns:", secondary_tourney_results.columns)
    print("Secondary Tourney Teams Columns:", secondary_tourney_teams.columns)

    secondary_tourney_results = secondary_tourney_results.merge(
        secondary_tourney_teams.rename(columns={"TeamID": "WTeamID"}), 
        on=["Season", "WTeamID"], 
        how="left"
    )

    secondary_tourney_results = secondary_tourney_results.merge(
        secondary_tourney_teams.rename(columns={"TeamID": "LTeamID"}), 
        on=["Season", "LTeamID"], 
        how="left"
    )

    # ==============================
    # Feature Engineering
    # ==============================

    # Convert seeds to numeric values
    tourney_results["WTeamSeed"] = tourney_results["WTeamSeed"].astype(str).apply(lambda x: extract_seed(x) if pd.notna(x) else None)
    tourney_results["LTeamSeed"] = tourney_results["LTeamSeed"].astype(str).apply(lambda x: extract_seed(x) if pd.notna(x) else None)

    # Calculate Win Percentage
    win_counts = reg_results.groupby(["Season", "WTeamID"]).size().reset_index(name="Wins")
    game_counts = pd.concat([
        reg_results[["Season", "WTeamID"]],
        reg_results[["Season", "LTeamID"]].rename(columns={"LTeamID": "WTeamID"})
    ])
    game_counts = game_counts.groupby(["Season", "WTeamID"]).size().reset_index(name="TotalGames")
    win_percentages = win_counts.merge(game_counts, on=["Season", "WTeamID"])
    win_percentages["WinPct"] = win_percentages["Wins"] / win_percentages["TotalGames"]

    # Point Differential per Team per Season
    reg_results["PointDiff"] = reg_results["WScore"] - reg_results["LScore"]
    point_diff = reg_results.groupby(["Season", "WTeamID"])["PointDiff"].mean().reset_index().rename(columns={"PointDiff": "AvgPointDiff"})

    # Merge Features into Tournament Data
    final_data = tourney_results.merge(win_percentages, on=["Season", "WTeamID"], how="left")
    final_data = final_data.merge(point_diff, on=["Season", "WTeamID"], how="left")

    return final_data


# Process Men's Data
men_data = process_ncaa_data('M', data_path)
men_data.to_csv("processed_men_data.csv", index=False)

# Process Women's Data
women_data = process_ncaa_data('W', data_path)
women_data.to_csv("processed_women_data.csv", index=False)


print("Processed Men's data")
print(men_data.head())


# Visualize Men's data
data_visualization('M', men_data)


print("Men's NCAA Data Correlation Heatmap")
plot_correlation_heatmap(men_data, "Men's NCAA Correlation Heatmap")


print("Processed Women's data")
print(women_data.head())


# Visualize women's data
data_visualization('W', women_data)

print("Women's NCAA Data Correlation Heatmap")
plot_correlation_heatmap(women_data, "Women's NCAA Correlation Heatmap")


def preprocess_ncaa_data(df, drop_cols=None):
    """
    Preprocess NCAA basketball data for machine learning.

    Parameters:
    - df: DataFrame with NCAA game results
    - drop_cols: Columns to remove before training
    
    Returns:
    - X_train, X_test, y_train, y_test: Train/Test feature & target datasets
    """

    if drop_cols is None:
        drop_cols = ["Season", "DayNum", "WTeamID", "LTeamID", "WLoc", "NumOT"]

    df = df.copy().drop(columns=drop_cols).fillna(0)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        encoded_features = pd.DataFrame(encoder.fit_transform(df[categorical_cols]))
        encoded_features.columns = encoder.get_feature_names_out(categorical_cols)
        df = df.drop(columns=categorical_cols).reset_index(drop=True)
        df = pd.concat([df, encoded_features], axis=1)

    # Winsorization:"Cap Extreme Values"
    for col in df.columns:
            df[col] = winsorize(df[col], limits=[0.01, 0.01])
    
    win_df = df.drop(columns=["LScore"]).rename(columns={"WScore": "Score"})
    win_df["Win"] = 1  # Label for wins

    lose_df = df.drop(columns=["WScore"]).rename(columns={"LScore": "Score"})
    lose_df["Win"] = 0  # Label for losses

    final_df = pd.concat([win_df, lose_df])

    X = final_df.drop(columns=["Score", "Win"])
    y = final_df["Win"].astype(int)

    X_new = SelectKBest(f_classif, k=15).fit_transform(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_new, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    selected_feature_names = X.columns[SelectKBest(f_classif, k=15).fit(X, y).get_support()]
    X_train_df = pd.DataFrame(X_train_scaled, columns=selected_feature_names)
    X_test_df = pd.DataFrame(X_test_scaled, columns=selected_feature_names)

    return X_train_df, X_test_df, y_train, y_test


# Preprocess men's data
X_train_men, X_test_men, y_train_men, y_test_men = preprocess_ncaa_data(men_data)
print(f"Y-Train Men Distribution:\n{y_train_men.value_counts()}")
print(f"X-Train Men Head:\n{X_train_men.head()}")

# Preprocess women's data
X_train_women, X_test_women, y_train_women, y_test_women = preprocess_ncaa_data(women_data)
print(f"Y-Train Women Distribution:\n{y_train_women.value_counts()}")
print(f"X-Train Women Head:\n{X_train_women.head()}")

print("Men's and Women's Data Preprocessed for Training!")


# Support to map tensor to cuda.
X_train_men = cudf.DataFrame(X_train_men)
X_test_men = cudf.DataFrame(X_test_men)
y_train_men = cudf.Series(y_train_men)
y_test_men = cudf.Series(y_test_men)

X_train_women = cudf.DataFrame(X_train_women)
X_test_women = cudf.DataFrame(X_test_women)
y_train_women = cudf.Series(y_train_women)
y_test_women = cudf.Series(y_test_women)


# MLP ANN
class MLPNet(nn.Module):
    def __init__(self, input_size):
        super(MLPNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )  

    def forward(self, x):
        return self.model(x)


# ==============================
# Hyperparameter Optimization
# ==============================
def optimize_model(model_class, param_space, X_train, y_train, X_test, y_test, n_trials=20):
    """Optimize hyperparameters using Optuna."""
    # Convert cudf DataFrame to NumPy
    X_train_np = X_train.to_numpy() if isinstance(X_train, cudf.DataFrame) else X_train
    y_train_np = y_train.to_numpy() if isinstance(y_train, cudf.Series) else y_train
    X_test_np = X_test.to_numpy() if isinstance(X_test, cudf.DataFrame) else X_test
    y_test_np = y_test.to_numpy() if isinstance(y_test, cudf.Series) else y_test
    def objective(trial):
        params = {k: v(trial) for k, v in param_space.items()}
        model = model_class(**params)
        
        # Apply cross-validation
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def tune_models(X_train, y_train, X_test, y_test):
    """
    Optimizes hyperparameters for all models using Optuna and GridSearchCV.
    """
    best_params = {}

    # Convert cudf DataFrame to NumPy for compatibility
    X_train_np = X_train.to_numpy() if isinstance(X_train, cudf.DataFrame) else X_train
    y_train_np = y_train.to_numpy() if isinstance(y_train, cudf.Series) else y_train
    X_test_np = X_test.to_numpy() if isinstance(X_test, cudf.DataFrame) else X_test
    y_test_np = y_test.to_numpy() if isinstance(y_test, cudf.Series) else y_test

    # Optimizing XGBoost
    xgb_space = {
        "n_estimators": lambda trial: trial.suggest_int("n_estimators", 500, 1500, step=100),
        "max_depth": lambda trial: trial.suggest_int("max_depth", 4, 15),
        "learning_rate": lambda trial: trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": lambda trial: trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": lambda trial: trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": lambda trial: trial.suggest_float("gamma", 1, 10),
        "min_child_weight": lambda trial: trial.suggest_int("min_child_weight", 3, 10),
        "reg_lambda": lambda trial: trial.suggest_float("reg_lambda", 1.0, 5.0),
        "reg_alpha": lambda trial: trial.suggest_float("reg_alpha", 0.0, 5.0),
        "scale_pos_weight": lambda trial: sum(y_train_np) / (len(y_train_np) - sum(y_train_np)),
        "tree_method": lambda trial: "gpu_hist",
        "device": lambda trial: "cuda"
    }
    best_params["XGBoost"] = optimize_model(XGBClassifier, xgb_space, X_train_np, y_train_np, X_test_np, y_test_np, n_trials=30)

    # Optimizing Random Forest
    rf_space = {
        "n_estimators": lambda trial: trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": lambda trial: trial.suggest_int("max_depth", 5, 20),
        "min_samples_split": lambda trial: trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": lambda trial: trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": lambda trial: trial.suggest_float("max_features", 0.5, 1.0),
    }
    #best_params["Random Forest"] = optimize_model(cuRF, rf_space, X_train_np, y_train_np, X_test_np, y_test_np, n_trials=20)

    # Optimizing SVM
    svm_space = {
        "C": lambda trial: trial.suggest_float("C", 0.1, 20, log=True),
        "kernel": lambda trial: trial.suggest_categorical("kernel", ["linear", "rbf"]),
    }
    #best_params["SVM"] = optimize_model(cuSVM, svm_space, X_train_np, y_train_np, X_test_np, y_test_np, n_trials=20)

    # Optimizing k-NN
    #knn_params = {"n_neighbors": [3, 5, 7, 9, 11, 15]}
    #knn_model = cuKNN()
    #grid_knn = GridSearchCV(knn_model, knn_params, cv=5, scoring="accuracy")
    #grid_knn.fit(X_train_np, y_train_np)
    #best_params["k-NN"] = grid_knn.best_params_['n_neighbors']

    return best_params


best_params = tune_models(X_train_men, y_train_men, X_test_men, y_test_men)
best_xgb_params = best_params["XGBoost"]


# ==============================
# Define and Train Models
# ==============================
def get_models(input_size, best_xgb_params):
    """Returns a dictionary of optimized models."""
    return {
        "Logistic Regression": cuLogisticRegression(),
        "XGBoost": XGBClassifier(**best_xgb_params),
        "Neural Network": nn.DataParallel(MLPNet(input_size).to(device))
    }


models = get_models(X_train_men.shape[1], best_xgb_params)


def plot_MLP_loss(losses, label: str):
    plt.plot(losses, label=f"MLP Losses {label}")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.title(f"Neural Network Loss for {label}")
    plt.legend()


def evaluate_models(train_dict, test_dict, models, label: str):
    results = []
    X_train, y_train = train_dict["X_train"], train_dict["y_train"]
    X_test, y_test = test_dict["X_test"], test_dict["y_test"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.to_numpy())
    X_test_scaled = scaler.transform(X_test.to_numpy())

    # Convert target labels to NumPy explicitly
    y_train_np = y_train.to_numpy() if isinstance(y_train, cudf.Series) else y_train
    y_test_np = y_test.to_numpy() if isinstance(y_test, cudf.Series) else y_test

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        if model_name == "Neural Network":
            X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
            y_train_torch = torch.tensor(pd.get_dummies(y_train_np).values, dtype=torch.float32).to(device)
            X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

            train_dataset = TensorDataset(X_train_torch, y_train_torch)
            train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

            input_size = X_train.shape[1]
            mlp_model = MLPNet(input_size).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(mlp_model.parameters(), lr=0.0005, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

            num_epochs = 200
            losses = []
            for epoch in range(num_epochs):
                mlp_model.train()
                total_loss = 0
            
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = mlp_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * batch_X.size(0)
            
                total_loss /= len(train_loader.dataset)
                losses.append(total_loss)
                scheduler.step(total_loss)
            
                if epoch % 10 == 0:
                    print(f"Epoch [{epoch}/{num_epochs}], Loss: {total_loss:.4f}")
            plot_MLP_loss(losses, label)

            with torch.no_grad():
                y_pred_prob = mlp_model(X_test_torch)
                y_pred = torch.argmax(y_pred_prob, dim=1).cpu().numpy()

        else:
            model.fit(X_train_scaled, y_train_np)
            y_pred = model.predict(X_test_scaled)
            if model_name == "XGBoost":
                model.save_model(f"xgboost_{'men' if label == 'Men' else 'women'}_model.json")
            print(f"{model_name} Unique Predictions: {np.unique(y_pred)}")
            print(f"{model_name} Class Counts: {np.bincount(y_pred)}")


        # Convert predictions explicitly to NumPy
        y_pred_np = y_pred.to_numpy() if isinstance(y_pred, cudf.Series) else np.array(y_pred)

        results.append([
            model_name,
            accuracy_score(y_test_np, y_pred_np),
            precision_score(y_test_np, y_pred_np, average="binary"),
            recall_score(y_test_np, y_pred_np, average="binary"),
            f1_score(y_test_np, y_pred_np, average="binary")
        ])

    return pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])


# Define train and test dictionaries
train_dict_men = {"X_train": X_train_men, "y_train": y_train_men}
test_dict_men = {"X_test": X_test_men, "y_test": y_test_men}

train_dict_women = {"X_train": X_train_women, "y_train": y_train_women}
test_dict_women = {"X_test": X_test_women, "y_test": y_test_women}


print("Women's Training Set Class Distribution:")
print(y_train_women.value_counts(normalize=True))
print("Women's Testing Set Class Distribution:")
print(y_test_women.value_counts(normalize=True))


# Get results for men's models
men_results_df = evaluate_models(train_dict_men, test_dict_men, models, "Men")
print(f"Men's Results: {men_results_df}")

# Get results for women's models
women_results_df = evaluate_models(train_dict_women, test_dict_women, models, "Women")
print(f"Women's Results: {women_results_df}")


def plot_model_performance(results_df, title):
    """
    Plots the performance of models based on Accuracy, Precision, Recall, and F1 Score.

    Parameters:
    - results_df: DataFrame containing model performance metrics.
    - title: Title of the plot.
    """
    num_models = len(results_df)
    fig_width = max(10, num_models * 2)
    fig_height = 6
    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.barplot(
        data=results_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
        x="Model", y="Score", hue="Metric"
    )
    plt.title(title, fontsize=14)
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    rotation_angle = 0 if num_models <= 4 else 45
    plt.xticks(rotation=rotation_angle, ha="right", fontsize=10)
    plt.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.show()


# Ensure results DataFrame exists before plotting
if 'men_results_df' in globals() and not men_results_df.empty:
    plot_model_performance(men_results_df, "Men's NCAA Model Performance")

if 'women_results_df' in globals() and not women_results_df.empty:
    plot_model_performance(women_results_df, "Women's NCAA Model Performance")


# Load trained models
best_models = {
    "XGBoost": XGBClassifier(**best_xgb_params)
}


def check_missing_teams(gender):
    team_stats, teams, season = load_team_data(gender)
    existing_teams = set(team_stats["WTeamID"].unique())
    matchups = generate_matchups(teams, season)

    matchup_teams = set(matchups["LowerTeam"]).union(set(matchups["HigherTeam"]))
    missing_teams = matchup_teams - existing_teams

    print(f"\nTotal Unique Teams in Matchups: {len(matchup_teams)}")
    print(f"Total Teams in Processed Dataset: {len(existing_teams)}")
    print(f"Missing Teams ({len(missing_teams)}): {sorted(missing_teams)}")


def load_team_data(gender):
    """Loads preprocessed team statistics for men ('M') or women ('W')."""
    file_path = f"processed_{'men' if gender == 'M' else 'women'}_data.csv"
    team_stats = pd.read_csv(file_path)
    team_stats["WTeamID"] = team_stats["WTeamID"].astype(int)
    teams = team_stats["WTeamID"].unique()
    season = team_stats["Season"].max() if "Season" in team_stats.columns else 2024
    categorical_cols = ["WLoc", "Slot", "StrongSeed", "WeakSeed"]
    team_stats.drop(columns=[col for col in categorical_cols if col in team_stats], inplace=True)
    return team_stats, teams, season


def generate_matchups(teams, season):
    """Generates all possible matchups for the tournament."""
    matchups = list(itertools.combinations(teams, 2))
    matchup_df = pd.DataFrame(matchups, columns=["Team_A", "Team_B"])

    # Ensure Team_A has the lower ID
    matchup_df["LowerTeam"] = matchup_df[["Team_A", "Team_B"]].min(axis=1)
    matchup_df["HigherTeam"] = matchup_df[["Team_A", "Team_B"]].max(axis=1)
    
    # Generate ID as required by Kaggle format
    matchup_df["ID"] = matchup_df.apply(lambda row: f"{season}_{row['LowerTeam']}_{row['HigherTeam']}", axis=1)
    
    return matchup_df[["ID", "LowerTeam", "HigherTeam"]]


def create_matchup_features(team_a, team_b, stats_df):
    """Computes feature differences for a given matchup ensuring proper feature alignment."""
    
    stats_df = stats_df.groupby("WTeamID").mean().reset_index()

    stats_a = stats_df[stats_df["WTeamID"] == team_a].drop(columns=["WTeamID"])
    stats_b = stats_df[stats_df["WTeamID"] == team_b].drop(columns=["WTeamID"])

    # **Ensure both teams exist in stats_df**
    if stats_a.empty or stats_b.empty:
        print(f" Skipping matchup ({team_a}, {team_b}) due to missing stats.")
        return None  # Skip invalid matchups

    # Convert to numeric and drop non-numeric columns
    stats_a = stats_a.apply(pd.to_numeric, errors="coerce")
    stats_b = stats_b.apply(pd.to_numeric, errors="coerce")

    # Ensure they have the same columns
    common_cols = stats_a.columns.intersection(stats_b.columns)
    stats_a, stats_b = stats_a[common_cols], stats_b[common_cols]

    # **Handle Missing Values: Fill NaN with Mean**
    stats_a.fillna(stats_df.mean(), inplace=True)
    stats_b.fillna(stats_df.mean(), inplace=True)

    # **Compute Feature Differences**
    feature_diff = (stats_a.values - stats_b.values).flatten()
    
    if len(feature_diff) != len(common_cols):
        print(f" Skipping matchup ({team_a}, {team_b}) due to mismatch in feature counts: {len(feature_diff)} vs {len(common_cols)}")
        return None  

    return feature_diff


def create_matchup_featuressss(team_a, team_b, stats_df):
    """Computes feature differences for a given matchup, ensuring numeric data only."""
    stats_a = stats_df[stats_df["WTeamID"] == team_a].drop(columns=["WTeamID"])
    stats_b = stats_df[stats_df["WTeamID"] == team_b].drop(columns=["WTeamID"])

    # Convert columns to numeric, coercing errors to NaN
    stats_a = stats_a.apply(pd.to_numeric, errors="coerce")
    stats_b = stats_b.apply(pd.to_numeric, errors="coerce")

    stats_a = stats_a.select_dtypes(include=[np.number])
    stats_b = stats_b.select_dtypes(include=[np.number])

    if stats_a.shape[0] == 0 or stats_b.shape[0] == 0:
        print(f"Skipping ({team_a}, {team_b}) due to missing data.")
        return None

    if stats_a.shape[0] > 1:
        stats_a = stats_a.iloc[0]

    if stats_b.shape[0] > 1:
        stats_b = stats_b.iloc[0]

    stats_a = stats_a.to_numpy()
    stats_b = stats_b.to_numpy()

    if stats_a.shape != stats_b.shape:
        print(f"Skipping matchup ({team_a}, {team_b}) due to shape mismatch: {stats_a.shape} vs {stats_b.shape}")
        return None

    return (stats_a - stats_b).flatten()



def prepare_features(matchups, team_stats):
    """Applies feature engineering and standardization."""
    matchup_features = []
    valid_matchups = []

    for team_a, team_b in tqdm(matchups.values, desc="Generating Features"):
        features = create_matchup_features(team_a, team_b, team_stats)
        if features is not None:
            matchup_features.append(features)
            valid_matchups.append((team_a, team_b))

    matchup_features_df = pd.DataFrame(matchup_features)
    valid_matchups_df = pd.DataFrame(valid_matchups, columns=["Team_A", "Team_B"])

    for col in expected_features:
        if col not in matchup_features_df.columns:
            print(f" Adding missing column: {col}")
            matchup_features_df[col] = 0
    
    matchup_features_df = matchup_features_df[expected_features]
    
    # Standardize Features
    scaler = StandardScaler()
    matchup_features_scaled = scaler.fit_transform(matchup_features_df)

    # Convert to PyTorch Tensor
    matchup_features_tensor = torch.tensor(matchup_features_scaled, dtype=torch.float32).cuda()

    return valid_matchups_df, matchup_features_scaled, matchup_features_tensor


def predict_outcomes(models, features, tensor_features):
    """Predicts outcomes using all trained models and ensures probability output."""
    predictions = {}

    for name, model in models.items():
        if name == "Neural Network":
            model.eval()
            with torch.no_grad():
                outputs = model(tensor_features)
                preds = outputs[:, 1].cpu().numpy()
        else:
            preds = model.predict_proba(features)[:, 1]

        predictions[name] = preds

    return predictions



def plot_predictions(predictions_df, gender):
    """
    Plots the distribution of predicted probabilities for the given predictions.

    Parameters:
    - predictions_df (DataFrame): DataFrame with "Pred" column containing prediction probabilities.
    - gender (str): "M" for Men's tournament, "W" for Women's tournament.
    """

    plt.figure(figsize=(8, 5))
    plt.hist(predictions_df["Pred"], bins=30, edgecolor='black', alpha=0.7)
    
    # Plot settings
    plt.xlabel("Predicted Probability of LowerTeam Winning")
    plt.ylabel("Count of Matchups")
    plt.title(f"Prediction Distribution for {'Men' if gender == 'M' else 'Women'} Tournament")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Show the plot
    plt.show()


def generate_predictions(gender):
    print(f"\n Processing {'Men' if gender == 'M' else 'Women'} tournament predictions...\n")

    # Load team statistics and matchups
    team_stats, teams, season = load_team_data(gender)
    
    matchups = generate_matchups(teams, season)

    # **Find Missing Teams**
    missing_teams = set(teams) - set(team_stats["WTeamID"])

    # Prepare features
    valid_matchups, matchup_features = [], []
    for _, id, team_a, team_b in tqdm(matchups.itertuples(), desc="Generating Features"):
        features = create_matchup_features(team_a, team_b, team_stats)
        if features is not None:
            matchup_features.append(features)
            valid_matchups.append((id, team_a, team_b))

    # Convert to DataFrame
    matchup_features_df = pd.DataFrame(matchup_features)
    valid_matchups_df = pd.DataFrame(valid_matchups, columns=["ID", "Team_A", "Team_B"])

    # **Print Final Feature Shape**
    print(f"\n Feature Matrix Shape: {matchup_features_df.shape}")
    
    # Standardize Features
    scaler = StandardScaler()
    matchup_features_scaled = scaler.fit_transform(matchup_features_df)

    # Convert to PyTorch Tensor
    matchup_features_tensor = torch.tensor(matchup_features_scaled, dtype=torch.float32).cuda()

    return valid_matchups_df, matchup_features_scaled, matchup_features_tensor



# Run this to check for missing teams before predictions
check_missing_teams("M")
check_missing_teams("W")


def predict_with_xgboost(gender):
    """
    Loads the trained XGBoost model and makes predictions on the generated matchups.
    
    Parameters:
    - gender (str): "M" for Men's tournament, "W" for Women's tournament.
    
    Returns:
    - DataFrame containing matchups and predicted probabilities.
    """
    model_filename = f"xgboost_{'men' if gender == 'M' else 'women'}_model.json"
    xgb_model = XGBClassifier()
    xgb_model.load_model(model_filename)
    print(f"\n Loaded Model: {model_filename}")

    # Generate tournament matchups and features
    valid_matchups_df, matchup_features_scaled, _ = generate_predictions(gender)

    # **Retrieve Training Feature Names from Model**
    training_features = xgb_model.get_booster().feature_names

    # **Ensure `matchup_features_scaled` matches `training_features`**
    matchup_features_df = pd.DataFrame(matchup_features_scaled, columns=training_features)

    # **Convert to NumPy Array for XGBoost**
    predictions = xgb_model.predict_proba(matchup_features_df)[:, 1]

    # **Format Output**
    valid_matchups_df["Pred"] = predictions
    return valid_matchups_df


# Generate and Plot Predictions
men_predictions = predict_with_xgboost("M")
plot_predictions(men_predictions, "M")
print(men_predictions.head())

women_predictions = predict_with_xgboost("W")
plot_predictions(women_predictions, "W")
print(women_predictions.head())

