# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from catboost import CatBoostRegressor
from scipy import stats
# Importing warnings library to ignore warning messages
import warnings
warnings.filterwarnings('ignore')


# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Set plot style for better visuals
plt.style.use('seaborn')


def load_data():
    """
    Load train and test datasets, drop 'id' column, and store test IDs.
    """
    try:
        train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
        test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
    except FileNotFoundError:
        print("Error: Data files not found. Please check the file paths.")
        return None, None, None
    
    test_ids = test['id']  # Save test IDs for submission
    train = train.drop(columns=['id'])
    test = test.drop(columns=['id'])
    
    print("Train data shape:", train.shape)
    print("Test data shape:", test.shape)
    
    # Check for missing values
    print("Missing values in train data:")
    print(train.isnull().sum())
    print("Missing values in test data:")
    print(test.isnull().sum())
    
    display(train.head())  # Display first few rows
    
    return train, test, test_ids


def handle_missing_values(df, is_train=True):
    """
    For training set: Drop rows with missing values.
    For test set: Fill missing values with median (numerical) or mode (categorical) using pandas.
    """
    initial_shape = df.shape
    if is_train:
        # Drop rows with missing values in training set
        df = df.dropna()
        print(f"Dropped rows with missing values in training set. Shape changed from {initial_shape} to {df.shape}")
    else:
        # For test set, fill missing values to preserve row count
        # Numerical columns: Fill with median
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Categorical columns: Fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        print(f"Filled missing values in test set. Shape remains: {df.shape}")
    
    print("Missing values after handling:")
    print(df.isnull().sum())
    
    return df


def preprocess(df):
    """
    Clip numerical columns and encode categorical columns.
    """
    # Clip numerical columns to reasonable ranges
    df['Episode_Length_minutes'] = df['Episode_Length_minutes'].clip(0, 120)
    df['Host_Popularity_percentage'] = df['Host_Popularity_percentage'].clip(20, 100)
    df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].clip(0, 100)
    df['Number_of_Ads'] = df['Number_of_Ads'].clip(0, 3)

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        unique_categories = df[col].astype(str).unique()
        category_mapping = {cat: idx for idx, cat in enumerate(unique_categories)}
        df[col] = df[col].astype(str).map(category_mapping)
    
    print(f"Categorical columns encoded: {list(categorical_cols)}")
    return df


def clean_data(df, is_train=True):
    """
    Remove outliers using IQR and normalize numerical features.
    """
    # Define numerical columns to clean
    numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']
    
    if is_train:
        # Remove outliers using IQR for numerical columns
        for col in numerical_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std != 0:
                df[col] = (df[col] - mean) / std
        print(f"Outliers removed. New shape: {df.shape}")
    
    # Normalize numerical features
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    print("Numerical features normalized:", numerical_cols)
    
    return df


def feature_engineering(df):
    """
    Add advanced features: interaction features, time-based features, and group-based statistics.
    """
    # Interaction features
    df['Length_Host_Interaction'] = df['Episode_Length_minutes'] * df['Host_Popularity_percentage']
    df['Ads_per_Minute'] = df['Number_of_Ads'] / df['Episode_Length_minutes'].replace(0, 1)  # Avoid division by zero
    
    # Time-based feature: Weekend vs Weekday
    df['Is_Weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
    
    # Group-based statistics: Mean Episode_Length_minutes per Genre
    df['Genre_Mean_Length'] = df.groupby('Genre')['Episode_Length_minutes'].transform('mean')
    
    # Group-based statistics: Mean Host_Popularity_percentage per Podcast_Name
    df['Podcast_Mean_Host_Popularity'] = df.groupby('Podcast_Name')['Host_Popularity_percentage'].transform('mean')
    
    print("Advanced feature engineering completed. Added features: Length_Host_Interaction, Ads_per_Minute, Is_Weekend, Genre_Mean_Length, Podcast_Mean_Host_Popularity")
    return df


def plot_target_distribution(y):
    """
    Plot and display the distribution of the target variable.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(y, bins=50, kde=True, color='blue')
    plt.title("Distribution of Listening Time")
    plt.xlabel("Listening Time (minutes)")
    plt.ylabel("Frequency")
    plt.savefig("target_distribution.png")
    plt.show()


def plot_feature_importance(model, X):
    """
    Plot and display feature importance from the XGBoost model.
    """
    importance = model.feature_importances_
    features = X.columns
    sorted_idx = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance[sorted_idx], y=features[sorted_idx], palette='viridis')
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.savefig("feature_importance.png")
    plt.show()


def plot_predicted_vs_actual(y_true, y_pred):
    """
    Plot and display predicted vs actual values.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color='green')
    plt.plot([0, 120], [0, 120], 'r--')  # Diagonal line
    plt.title("Predicted vs Actual Listening Time")
    plt.xlabel("Actual Listening Time (minutes)")
    plt.ylabel("Predicted Listening Time (minutes)")
    plt.savefig("predicted_vs_actual.png")
    plt.show()


def plot_correlation_heatmap(X):
    """
    Plot a correlation heatmap for numerical features.
    """
    plt.figure(figsize=(12, 8))
    corr = X.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    plt.title("Correlation Heatmap of Features")
    plt.savefig("correlation_heatmap.png")
    plt.show()


def plot_residuals(y_true, y_pred):
    """
    Plot residuals (prediction errors) as a histogram.
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=50, kde=True, color='purple')
    plt.title("Residuals Distribution")
    plt.xlabel("Prediction Error (minutes)")
    plt.ylabel("Frequency")
    plt.savefig("residuals_distribution.png")
    plt.show()


def tune_xgboost(X_train, y_train):
    """
    Tune XGBoost hyperparameters using GridSearchCV.
    """
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 200]
    }
    
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=SEED)
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    print("Tuning XGBoost hyperparameters...")
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    print("Best parameters:", best_params)
    print("Best RMSE:", np.sqrt(-grid_search.best_score_))
    
    return best_params


def train_model(X_train, y_train, X_valid, y_valid, X_test):
    """
    Train XGBoost and CatBoost models with fixed hyperparameters and return predictions.
    """
    models = {
        'XGBoost': XGBRegressor(
            objective='reg:squarederror',
            random_state=SEED,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100
        ),
        'CatBoost': CatBoostRegressor(
            loss_function='RMSE',
            random_seed=SEED,
            depth=6,
            learning_rate=0.1,
            iterations=100,
            verbose=False
        )
    }
    
    valid_preds = {}
    test_preds = {}
    rmses = {}
    
    print("Training multiple models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predict on validation set
        valid_preds[name] = np.clip(model.predict(X_valid), 0, 120)
        
        # Predict on test set
        test_preds[name] = np.clip(model.predict(X_test), 0, 120)
        
        # Calculate RMSE
        rmses[name] = np.sqrt(mean_squared_error(y_valid, valid_preds[name]))
        print(f"{name} RMSE: {rmses[name]:.4f}")
    
    # Select XGBoost model for feature importance
    selected_model = 'XGBoost'
    print(f"Using {selected_model} for feature importance.")
    
    return models[selected_model], valid_preds, test_preds, rmses


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Evaluate model performance using RMSE, MAE, and R² metrics.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nEvaluation for {model_name}:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    return {'RMSE': rmse, 'MAE': mae, 'R²': r2}


def ensemble_predictions(test_preds_dict, weights=None):
    """
    Combine predictions from multiple models using weighted average.
    """
    if weights is None:
        weights = {'XGBoost': 0.6, 'CatBoost': 0.4}
    
    ensemble_preds = np.zeros_like(test_preds_dict['XGBoost'])
    total_weight = sum(weights.values())
    
    print("Creating ensemble predictions...")
    for model_name, weight in weights.items():
        ensemble_preds += weight * test_preds_dict[model_name]
        print(f"Adding {model_name} with weight {weight}")
    
    ensemble_preds /= total_weight
    ensemble_preds = np.clip(ensemble_preds, 0, 120)
    
    print("Ensemble predictions completed.")
    return ensemble_preds


def save_submission(test_ids, preds, filename="submission.csv"):
    """
    Save predictions to a submission CSV file.
    """
    submission = pd.DataFrame({
        'id': test_ids,
        'Listening_Time_minutes': preds
    })
    submission.to_csv(filename, index=False)
    print(f"Submission file saved: {filename}")
    display(submission.head())  # Display first few rows


# Load data
train, test, test_ids = load_data()
if train is None:
    raise SystemExit("Exiting due to data loading error.")

# Handle missing values
train = handle_missing_values(train, is_train=True)
test = handle_missing_values(test, is_train=False)

# Preprocess data
train = preprocess(train)
test = preprocess(test)

# Clean data
train = clean_data(train, is_train=True)
test = clean_data(test, is_train=False)

# Feature engineering
train = feature_engineering(train)
test = feature_engineering(test)

# Separate features and target
X = train.drop(columns=['Listening_Time_minutes'])
y = train['Listening_Time_minutes']

# Plot target distribution
plot_target_distribution(y)

# Split data
# Using pandas for splitting instead of sklearn
train_idx = np.random.choice(X.index, size=int(0.8 * len(X)), replace=False)
valid_idx = X.index.difference(train_idx)
X_train, X_valid = X.loc[train_idx], X.loc[valid_idx]
y_train, y_valid = y.loc[train_idx], y.loc[valid_idx]

print("Train set shape:", X_train.shape)
print("Validation set shape:", X_valid.shape)

# Train models (without hyperparameter tuning)
model, valid_preds, test_preds, rmses = train_model(X_train, y_train, X_valid, y_valid, test)

# Evaluate models
print("\nModel Performance Comparison:")
for name, preds in valid_preds.items():
    evaluate_model(y_valid, preds, name)

# Plot visualizations
plot_correlation_heatmap(X_train)
plot_feature_importance(model, X_train)
plot_predicted_vs_actual(y_valid, valid_preds['XGBoost'])
plot_residuals(y_valid, valid_preds['XGBoost'])

# Create ensemble predictions
ensemble_test_preds = ensemble_predictions(test_preds)

# Save submission
save_submission(test_ids, ensemble_test_preds)

