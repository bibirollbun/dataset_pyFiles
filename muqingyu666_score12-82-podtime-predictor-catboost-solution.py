import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool
import optuna
from optuna.samplers import TPESampler
import category_encoders as ce
import warnings

warnings.filterwarnings("ignore")

# Set a consistent style for visualizations
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

# Set global font family for plots
plt.rcParams["font.family"] = "times new roman"

# Load the dataset
train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
sample_df = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


# Function for preprocessing
def preprocess_data(df, is_train=True):
    # Create a copy to avoid modifying the original
    data = df.copy()

    # Handle missing values
    # Fill Episode_Length_minutes with median (since it has a strong correlation with target)
    if "Episode_Length_minutes" in data.columns:
        median_length = data["Episode_Length_minutes"].median()
        data["Episode_Length_missing"] = (
            data["Episode_Length_minutes"].isna().astype(int)
        )
        data["Episode_Length_minutes"].fillna(
            median_length, inplace=True
        )

    # Fill Guest_Popularity_percentage with median
    if "Guest_Popularity_percentage" in data.columns:
        median_guest_pop = data["Guest_Popularity_percentage"].median()
        data["Guest_Popularity_missing"] = (
            data["Guest_Popularity_percentage"].isna().astype(int)
        )
        data["Guest_Popularity_percentage"].fillna(
            median_guest_pop, inplace=True
        )

    # Fill Number_of_Ads (just 1 missing value)
    if "Number_of_Ads" in data.columns:
        data["Number_of_Ads"].fillna(
            data["Number_of_Ads"].median(), inplace=True
        )

    # Feature engineering
    # Log transform Number_of_Ads (since it's heavily skewed with skewness 6.03)
    if "Number_of_Ads" in data.columns:
        data["Number_of_Ads_Log"] = np.log1p(data["Number_of_Ads"])

    # Create interaction features
    if (
        "Host_Popularity_percentage" in data.columns
        and "Guest_Popularity_percentage" in data.columns
    ):
        data["Host_Guest_Popularity_Avg"] = (
            data["Host_Popularity_percentage"]
            + data["Guest_Popularity_percentage"]
        ) / 2
        data["Host_Guest_Popularity_Diff"] = abs(
            data["Host_Popularity_percentage"]
            - data["Guest_Popularity_percentage"]
        )

    # Create episode length bins (since this has strong correlation with target)
    if "Episode_Length_minutes" in data.columns:
        data["Episode_Length_Bins"] = pd.qcut(
            data["Episode_Length_minutes"],
            q=10,
            labels=False,
            duplicates="drop",
        )

    # Create a coefficient of how much of the episode is listened to (for training only)
    if (
        is_train
        and "Episode_Length_minutes" in data.columns
        and "Listening_Time_minutes" in data.columns
    ):
        data["Listen_Ratio"] = (
            data["Listening_Time_minutes"]
            / data["Episode_Length_minutes"]
        )
        data["Listen_Ratio"].fillna(0, inplace=True)
        data["Listen_Ratio"] = data["Listen_Ratio"].clip(
            0, 1
        )  # Clip between 0 and 1

    # Encode categorical features - let CatBoost handle them natively
    cat_features = [
        "Podcast_Name",
        "Genre",
        "Publication_Day",
        "Publication_Time",
        "Episode_Sentiment",
    ]

    return data, cat_features


# Preprocess training and test data
processed_train, cat_features = preprocess_data(train_df, is_train=True)
processed_test, _ = preprocess_data(test_df, is_train=False)


# Define features to use - exclude target, id, and title which has too many unique values
features = [
    col
    for col in processed_train.columns
    if col
    not in [
        "id",
        "Episode_Title",
        "Listening_Time_minutes",
        "Listen_Ratio",
    ]
]

X = processed_train[features]
y = processed_train["Listening_Time_minutes"]

# Split data for training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Define the Optuna objective function for hyperparameter tuning
def objective(trial):
    # Define the hyperparameters to optimize
    params = {
        "iterations": trial.suggest_int("iterations", 1000, 5000),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.3
        ),
        "depth": trial.suggest_int("depth", 6, 12),
        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg", 1e-3, 10.0, log=True
        ),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "bagging_temperature": trial.suggest_float(
            "bagging_temperature", 0.0, 1.0
        ),
        "random_strength": trial.suggest_float(
            "random_strength", 1e-3, 10.0, log=True
        ),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
        ),
        "min_data_in_leaf": trial.suggest_int(
            "min_data_in_leaf", 1, 100
        ),
        "one_hot_max_size": trial.suggest_int(
            "one_hot_max_size", 2, 25
        ),
        "random_seed": 42,
        "task_type": "GPU",  
    }

    # Create a CatBoostRegressor with the sampled hyperparameters
    model = CatBoostRegressor(
        **params,
        loss_function="RMSE",
        cat_features=cat_features,
        eval_metric="RMSE",
        verbose=0,
    )

    # Create CatBoost Pool objects for efficient handling of categorical features
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    # Train the model
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)

    # Make predictions on the validation set
    preds = model.predict(val_pool)

    # Calculate and return the RMSE
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    return rmse


# Run Optuna hyperparameter optimization
def optimize_hyperparameters(n_trials=100):
    print("Starting hyperparameter optimization...")

    # Create a study object and optimize the objective function
    study = optuna.create_study(
        direction="minimize", sampler=TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials)

    print(f"Best RMSE: {study.best_value}")
    print(f"Best hyperparameters: {study.best_params}")

    # Create importance plot
    try:
        optuna.visualization.plot_param_importances(study)
        plt.title("Hyperparameter Importances")
        plt.tight_layout()
        plt.show()
    except:
        print("Could not create importance plot.")

    return study.best_params


# Train the final model with the best hyperparameters
def train_final_model(best_params):
    print("Training final model with optimized hyperparameters...")

    best_params["task_type"] = "GPU"
    
    # Create the final model with the best parameters
    final_model = CatBoostRegressor(
        **best_params,
        loss_function="RMSE",
        cat_features=cat_features,
        eval_metric="RMSE",
        verbose=200,
    )

    # Create Pool objects for the entire training data
    train_pool = Pool(X, y, cat_features=cat_features)

    # Train the model on all training data
    final_model.fit(train_pool)

    return final_model


# Make predictions on the test set
def make_predictions(model, test_data):
    print("Making predictions on test data...")

    # Create a Pool object for the test data
    test_pool = Pool(test_data[features], cat_features=cat_features)

    # Make predictions
    test_preds = model.predict(test_pool)

    # Create a submission dataframe
    submission = pd.DataFrame(
        {"id": test_data["id"], "Listening_Time_minutes": test_preds}
    )

    return submission


# Feature importance analysis
def plot_feature_importance(model):
    feature_importance = model.get_feature_importance()
    feature_names = model.feature_names_

    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importance}
    ).sort_values("Importance", ascending=False)

    plt.figure(figsize=(14, 10))
    sns.barplot(
        x="Importance", y="Feature", data=importance_df.head(20)
    )
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.show()

    return importance_df


# Optimize hyperparameters
best_params = optimize_hyperparameters(50)

# Train the final model
final_model = train_final_model(best_params)

# Analyze feature importance
importance_df = plot_feature_importance(final_model)
print("Top 10 most important features:")
print(importance_df.head(10))


# Make predictions on the test set
def make_predictions(model, test_data):
    print("Making predictions on test data...")

    # Create a Pool object for the test data
    test_pool = Pool(test_data[features], cat_features=cat_features)

    # Make predictions
    test_preds = model.predict(test_pool)

    return test_preds

test_preds = make_predictions(final_model, processed_test)
sample_df['Listening_Time_minutes'] = test_preds

sample_df.to_csv("submission.csv", index=False)




