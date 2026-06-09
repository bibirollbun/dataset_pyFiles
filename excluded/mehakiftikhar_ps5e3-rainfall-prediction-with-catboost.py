# Import Basis
import pandas as pd 
import optuna
import numpy as np 
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
import seaborn as sns 
import math
from io import StringIO
from colorama import Fore, Style, init;
# Import necessary libraries
from IPython.core.display import display, HTML
from scipy.stats import skew  
# Import Plotly.go
import plotly.graph_objects as go
# import Subplots
from plotly.subplots import make_subplots
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import LabelEncoder, MinMaxScaler , StandardScaler , QuantileTransformer
from sklearn.impute import SimpleImputer

# Paellete
palette = ['#3b2307', '#ab6a1f']
color_palette = sns.color_palette(palette)

# Set the option to display all columns
pd.set_option('display.max_columns', None)


# Load train and test datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
train.set_index('id', inplace=True)

test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
test.set_index('id', inplace=True)

test = test.fillna('None')

# Load sample submission
submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')

# Load external/original dataset 
original = pd.read_csv('/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv')

# Convert 'rainfall' column to binary (Yes/No -> 1/0)
original['rainfall'] = original['rainfall'].map({'Yes': 1, 'No': 0})

# Standardize column names (remove leading/trailing spaces and lowercase all)
original.columns = original.columns.str.strip().str.lower()
train.columns = train.columns.str.strip().str.lower()
test.columns = test.columns.str.strip().str.lower()

# Preview column names
print("Train columns:", train.columns.tolist())
print("Original columns (after cleanup):", original.columns.tolist())
print("Test columns:", test.columns.tolist())

# Ensure all train columns exist in original 
for col in train.columns:
    if col not in original.columns and col != 'rainfall':
        print(f"Adding missing column '{col}' to original dataset with NaNs.")
        original[col] = np.nan

# Reorder original columns to match train (except target column 'rainfall')
feature_cols = [col for col in train.columns if col != 'rainfall']
original = original[feature_cols + ['rainfall']]

# Combine train and original datasets (stacking both vertically)
combined_train = pd.concat([train, original], ignore_index=True)

# Ensure test has all columns (some may be missing after adding external data features)
for col in combined_train.columns:
    if col != 'rainfall' and col not in test.columns:
        print(f"Adding missing column '{col}' to test set with NaNs.")
        test[col] = np.nan

# Final confirmation
print(f"Final combined training set shape: {combined_train.shape}")
print(f"Final test set shape: {test.shape}")


# Styled Heading Function 
def styled_heading(text, background_color='#ffbd70', text_color='#3b2307'):
    return f"""
    <div style="
        text-align: center;
        background: {background_color};
        font-family: 'Freehand521 BT', sans-serif;
        color: {text_color};
        padding: 15px;
        font-size: 30px;
        font-weight: bold;
        line-height: 1;
        border-radius: 20px 20px 0 0;
        margin: 20px 0;
        box-shadow: 4px 4px 6px rgba(0, 0, 0, 0.2);
        border: 3px dashed {text_color};
    ">
        {text}
    </div>
    """

# D_O Function
def D_O(train_df, heading_bg='lightblue', heading_color='black', text_bg='white', text_color='black'):
    try:
        # Head, Tail & Summary
        sections = [
            ("The Head of Dataset is:", train_df.head(5)),
            ("The Tail of Dataset is:", train_df.tail(5)),
            ("Numerical Summary of Data:", train_df.describe())
        ]
        for heading, data in sections:
            display(HTML(styled_heading(heading, background_color=heading_bg, text_color=heading_color)))
            display(HTML(data.to_html(index=False).replace(
                '<table border="1" class="dataframe">',
                f'<table style="border: 8px solid black; margin-bottom: 20px; background-color: {text_bg}; color: {text_color};">'
            ).replace('<td>', f'<td style="color: {text_color}; background-color: {text_bg};">')))
            print("\n")

        # Shape Information
        display(HTML(styled_heading("Shape of Data:", background_color=heading_bg, text_color=heading_color)))
        print(f'Rows: {train_df.shape[0]}')
        print(f'Columns: {train_df.shape[1]}')
        print("\n<br>\n")

        # Dataset Info
        display(HTML(styled_heading("Dataset Information:", background_color=heading_bg, text_color=heading_color)))
        buffer = StringIO()
        train_df.info(buf=buffer)
        buffer.seek(0)
        info_str = buffer.read()
        display(HTML(f"<pre style='color: {text_color}; background-color: {text_bg}; margin-bottom: 20px; font-family: Courier, monospace; font-size: 14px; padding: 10px; border: 8px solid black;'>{info_str}</pre>"))
        print("\n<br>\n")

        # Categorical Columns
        cat_cols_train = [col for col in train_df.columns if train_df[col].dtype == 'O']
        display(HTML(styled_heading("Categorical Columns:", background_color=heading_bg, text_color=heading_color)))
        if cat_cols_train:
            print(f'Categorical Columns: {cat_cols_train}')
        else:
            print('No Categorical Columns Found')
        print("\n<br>\n")

        # Numerical Columns (float + int both)
        num_cols_train = [col for col in train_df.columns if train_df[col].dtype in ['float64', 'int64']]
        display(HTML(styled_heading("Numerical Columns:", background_color=heading_bg, text_color=heading_color)))
        print(f'Numerical Columns: {num_cols_train}')
        print("\n<br>\n")

        # Null Values
        display(HTML(styled_heading("Missing Values Summary:", background_color=heading_bg, text_color=heading_color)))
        null_values = train_df.isnull().sum().reset_index()
        null_values.columns = ['Column', 'MissingCount']
        null_values['MissingPercentage'] = (null_values['MissingCount'] / train_df.shape[0]) * 100
        display(HTML(null_values.to_html(index=False).replace(
            '<table border="1" class="dataframe">',
            f'<table style="border: 8px solid black; margin-bottom: 20px; background-color: {text_bg}; color: {text_color};">'
        ).replace('<td>', f'<td style="color: {text_color}; background-color: {text_bg};">')))
        print("\n<br>\n")

        # Duplicate Rows Check
        display(HTML(styled_heading("Duplicate Rows Check:", background_color=heading_bg, text_color=heading_color)))
        duplicates = train_df.duplicated().sum()
        if duplicates > 0:
            print(f'Duplicates Found: {duplicates} rows')
        else:
            print('No Duplicates Found')
        print("\n<br>\n")

    except Exception as e:
        display(HTML(f"<div style='color: red; font-weight: bold;'> Error: {str(e)}</div>"))



D_O(train, heading_bg='#ffbd70', heading_color='#3b2307', text_bg='#ffbd70', text_color='#3b2307')


D_O(test, heading_bg='#ffbd70', heading_color='#3b2307', text_bg='#ffbd70', text_color='#3b2307')


def detect_outliers_iqr(df, columns):
    outlier_summary = []

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)

        outlier_summary.append({
            'Column': col,
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound,
            'Outlier Count': outlier_count
        })

    outlier_df = pd.DataFrame(outlier_summary)
    return outlier_df


# List of numerical columns
numerical_columns = ['pressure', 'maxtemp', 'temparature', 'mintemp', 
                     'dewpoint', 'humidity', 'cloud', 'sunshine', 
                     'winddirection', 'windspeed', 'rainfall']

# Run outlier detection
outlier_summary_df = detect_outliers_iqr(train, numerical_columns)

# Display the summary
print("Outlier Summary (Using IQR Method):")
print(outlier_summary_df)


def calculate_iqr_bounds(df, column):
    """
    Calculate the lower and upper bounds for outliers using the IQR method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

def detect_outliers(df, column):
    """
    Identify rows with outliers for a given column.
    """
    lower_bound, upper_bound = calculate_iqr_bounds(df, column)
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def handle_outliers(df, column, method="cap"):
    """
    Handle outliers using different strategies:
    - 'remove': Drops rows with outliers.
    - 'cap': Caps values at the IQR lower and upper bounds (Winsorization).
    - 'none': Leaves outliers as-is.

    Parameters:
    df (pd.DataFrame): The DataFrame.
    column (str): The column to process.
    method (str): The handling method ('remove', 'cap', 'none').

    Returns:
    pd.DataFrame: Processed DataFrame.
    """
    lower_bound, upper_bound = calculate_iqr_bounds(df, column)

    if method == "remove":
        # Remove rows with outliers
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    elif method == "cap":
        # Cap (Winsorize) outliers to the lower and upper bounds
        df[column] = np.clip(df[column], lower_bound, upper_bound)

    elif method == "none":
        # Do nothing (keep outliers as they are)
        pass

    else:
        raise ValueError("Invalid method. Choose from 'remove', 'cap', or 'none'.")

    return df

def handle_all_outliers(df, method_dict):
    """
    Handle outliers for all columns using a dictionary to specify methods for each column.
    Example:
    method_dict = {
        'pressure': 'cap',
        'maxtemp': 'none',
        'humidity': 'cap',
        'rainfall': 'remove'
    }
    """
    for column, method in method_dict.items():
        df = handle_outliers(df, column, method)
    return df


outlier_handling_methods = {
    'pressure': 'cap',
    'maxtemp': 'none',
    'temparature': 'none',
    'mintemp': 'cap',
    'dewpoint': 'cap',
    'humidity': 'cap',
    'cloud': 'cap',
    'sunshine': 'none',
    'winddirection': 'none',
    'windspeed': 'cap',
    'rainfall': 'none'  
}

# Apply outlier handling
train_cleaned = handle_all_outliers(train.copy(), outlier_handling_methods)

# Check shape before and after
print(f"Original Rows: {train.shape[0]}")
print(f"Cleaned Rows: {train_cleaned.shape[0]}")


# inconsistent rows where mintemp > maxtemp
inconsistent_temp_rows = train[train['mintemp'] > train['maxtemp']]

# Display inconsistent rows 
if len(inconsistent_temp_rows) > 0:
    print(f"âš ï¸� Found {len(inconsistent_temp_rows)} inconsistent rows where mintemp > maxtemp:")
    display(inconsistent_temp_rows)
else:
    print("âœ… No inconsistent mintemp > maxtemp cases found.")


def handle_inconsistent_temps(df, action="swap"):
    """
    Handle rows where mintemp > maxtemp.
    
    action:
    - 'swap' (default) - Swap mintemp and maxtemp
    - 'drop' - Drop the inconsistent rows
    - 'ignore' - Leave them as is (just warn)
    """
    inconsistent_rows = df[df['mintemp'] > df['maxtemp']]

    if len(inconsistent_rows) > 0:
        print(f"âš ï¸� Found {len(inconsistent_rows)} inconsistent rows where mintemp > maxtemp.")
        
        if action == "swap":
            print("ğŸ”„ Swapping mintemp and maxtemp for these rows...")
            for idx in inconsistent_rows.index:
                mintemp = df.at[idx, 'mintemp']
                maxtemp = df.at[idx, 'maxtemp']
                df.at[idx, 'mintemp'] = maxtemp
                df.at[idx, 'maxtemp'] = mintemp

        elif action == "drop":
            print("ğŸ—‘ï¸� Dropping inconsistent rows...")
            df = df[df['mintemp'] <= df['maxtemp']]

        elif action == "ignore":
            print("âš ï¸� Warning: Leaving inconsistent rows unchanged.")

        else:
            raise ValueError("Invalid action. Choose from 'swap', 'drop', or 'ignore'.")
    else:
        print("âœ… No inconsistent mintemp > maxtemp cases found.")
    
    return df


train_cleaned = handle_inconsistent_temps(train.copy(), action="swap")


def plot_numerical_distributions(df, numerical_columns):
    """
    Plot distributions of numerical features to understand their spread.
    """
    plt.figure(figsize=(16, 16))
    for idx, col in enumerate(numerical_columns, 1):
        plt.subplot(4, 3, idx)
        sns.histplot(df[col], kde=True, color='#3b2307')
        plt.title(f'{col} Distribution', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_feature_vs_rainfall(df, features):
    """
    Plot feature-target relationships (features vs rainfall).
    """
    plt.figure(figsize=(16, 10))
    for idx, feature in enumerate(features, 1):
        plt.subplot(2, 3, idx)
        sns.boxplot(x='rainfall', y=feature, data=df, palette=palette)
        plt.title(f'{feature} vs Rainfall', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df):
    """
    Plot correlation heatmap for all numerical features.
    """
    plt.figure(figsize=(12, 8))
    corr_matrix = df.corr(numeric_only=True)
    sns.heatmap(corr_matrix, annot=True, cmap=palette, fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()


# ======= EDA Execution =======

numerical_columns = [
    'pressure', 'maxtemp', 'temparature', 'mintemp',
    'dewpoint', 'humidity', 'cloud', 'sunshine',
    'winddirection', 'windspeed'
]

# Numerical Feature Distributions
print("Plotting Numerical Feature Distributions...")
plot_numerical_distributions(train_cleaned, numerical_columns)

# Feature-Target Relationships (Rainfall vs Features)
print("Visualizing Feature vs Rainfall Relationships...")
features_vs_rainfall = ['maxtemp', 'humidity', 'cloud', 'sunshine', 'windspeed']
plot_feature_vs_rainfall(train_cleaned, features_vs_rainfall)

# Correlation Heatmap
print("Plotting Correlation Heatmap...")
plot_correlation_heatmap(train_cleaned)


def add_date_features(df):
    """
    Create month and season features from the day column (assuming day is sequential or actual date).
    If day is numeric (like 1 to 365), we assume a year-based day.
    """
    # Assuming 'day' represents day number in a year (1-365)
    df['month'] = ((df['day'] - 1) // 30 + 1).clip(1, 12)

    # Define seasons based on months 
    season_mapping = {
        1: 'Winter', 2: 'Winter', 3: 'Spring',
        4: 'Spring', 5: 'Spring', 6: 'Summer',
        7: 'Summer', 8: 'Summer', 9: 'Autumn',
        10: 'Autumn', 11: 'Autumn', 12: 'Winter'
    }
    df['season'] = df['month'].map(season_mapping)

    # Optional: Add boolean seasonal flags (use if needed)
    df['isWinter'] = df['season'].eq('Winter').astype(int)
    df['isSummer'] = df['season'].eq('Summer').astype(int)
    df['isSpring'] = df['season'].eq('Spring').astype(int)
    df['isAutumn'] = df['season'].eq('Autumn').astype(int)

    return df


def categorize_wind_direction(df):
    """
    Convert wind direction (0-360 degrees) into compass categories (N, NE, E, SE, etc.)
    """
    def direction_category(degree):
        if degree >= 337.5 or degree < 22.5:
            return 'N'
        elif degree < 67.5:
            return 'NE'
        elif degree < 112.5:
            return 'E'
        elif degree < 157.5:
            return 'SE'
        elif degree < 202.5:
            return 'S'
        elif degree < 247.5:
            return 'SW'
        elif degree < 292.5:
            return 'W'
        else:
            return 'NW'

    df['wind_direction_cat'] = df['winddirection'].apply(direction_category)
    return df


def perform_feature_engineering(df):
    """
    Perform all necessary feature engineering steps.
    """
    print("Adding date-based features (month, season, seasonal flags)...")
    df = add_date_features(df)

    print("Categorizing wind direction into compass bins...")
    df = categorize_wind_direction(df)

    print("Feature engineering complete.")
    return df


# ===== Apply Feature Engineering =====
train_cleaned = perform_feature_engineering(train_cleaned)

# Check new columns added
print("New Columns Added: ", [col for col in train_cleaned.columns if col not in train.columns])


def calculate_numerical_correlations(df, target_column):
    """
    Calculates correlation between numerical features and the target column.
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols.remove(target_column)  
    correlation = df[numerical_cols + [target_column]].corr()[target_column].drop(target_column).sort_values(ascending=False)
    
    # Plot correlations
    plt.figure(figsize=(10, 5))
    sns.barplot(x=correlation.values, y=correlation.index, palette=palette)
    plt.title(f'Correlation with {target_column}')
    plt.show()
    
    return correlation

def encode_categorical_columns(df, categorical_columns):
    """
    Encodes categorical columns using LabelEncoder (for feature importance calculation with tree models).
    """
    df_encoded = df.copy()
    le = LabelEncoder()

    for col in categorical_columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])

    return df_encoded

def calculate_feature_importance(df, target_column):
    """
    Calculates feature importances using RandomForestClassifier.
    Works with numerical & encoded categorical columns.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    model = CatBoostClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    importances = importances.sort_values(by='Importance', ascending=False)

    # Plot Feature Importances
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=importances, palette=palette)
    plt.title('Feature Importances (CatBoost)')
    plt.show()

    return importances

def feature_importance_analysis(df, target_column, categorical_columns):
    """
    Full pipeline to analyze feature importance.
    """
    print(f"Numerical Feature Correlation with '{target_column}'")
    correlation = calculate_numerical_correlations(df, target_column)

    print(f"\nEncoding categorical columns: {categorical_columns}")
    df_encoded = encode_categorical_columns(df, categorical_columns)

    print(f"\nCalculating Feature Importances using CatBoostClassifier...")
    feature_importances = calculate_feature_importance(df_encoded, target_column)

    return correlation, feature_importances


categorical_columns = ['season', 'wind_direction_cat']  

correlation, feature_importances = feature_importance_analysis(train_cleaned, 'rainfall', categorical_columns)

# Display top features
print("\nTop Features Based on Importance:")
print(feature_importances.head(10))


# Define features and target
features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
            'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed', 
            'month', 'season', 'isWinter', 'isSummer', 'isSpring', 'isAutumn', 'wind_direction_cat']

target = 'rainfall'

# Train-test split
X = train_cleaned[features]
y = train_cleaned[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"âœ… Data Split Done: Train Shape = {X_train.shape}, Test Shape = {X_test.shape}")


# 0.85974 
# =================================================
# Train Optimized CatBoost with Stratified KFold & AUC Evaluation
# =================================================
def train_optimized_catboost(X, y, n_splits=5, seed=42, model_save_path=None):
    """
    Trains an optimized CatBoostClassifier with Stratified K-Fold Cross Validation.
    Automatically handles categorical features and evaluates using AUC.

    Parameters:
        X (DataFrame): Features DataFrame
        y (Series): Target Series
        n_splits (int): Number of cross-validation folds
        seed (int): Random seed for reproducibility
        model_save_path (str, optional): Path prefix to save models (optional)

    Returns:
        dict: Trained models, AUC scores, and summary statistics
    """

    print("\nTraining Optimized CatBoost with Stratified KFold...")

    # Auto-detect categorical features 
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Detected Categorical Features: {cat_features}")

    # cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    train_auc_scores = []
    val_auc_scores = []
    models = []

    # CatBoost optimized params 
    catboost_params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': seed,
        'depth': 8,
        'learning_rate': 0.03,
        'iterations': 2000,
        'l2_leaf_reg': 7,
        'bootstrap_type': 'Bayesian',
        'bagging_temperature': 1.0,
        'border_count': 128,
        'verbose': 200,
        'early_stopping_rounds': 100,
        'use_best_model': True
    }

    # cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nTraining Fold {fold}/{n_splits}...")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Define CatBoost model
        model = CatBoostClassifier(**catboost_params)

        # Train with proper categorical handling via Pool
        train_pool = Pool(X_train, label=y_train, cat_features=cat_features)
        val_pool = Pool(X_val, label=y_val, cat_features=cat_features)

        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        # Predict probabilities for AUC
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_val_proba = model.predict_proba(X_val)[:, 1]

        train_auc = roc_auc_score(y_train, y_train_proba)
        val_auc = roc_auc_score(y_val, y_val_proba)

        train_auc_scores.append(train_auc)
        val_auc_scores.append(val_auc)
        models.append(model)

        print(f"Fold {fold} - Train AUC: {train_auc:.4f}, Validation AUC: {val_auc:.4f}")

        # Save each fold model
        if model_save_path:
            model.save_model(f"{model_save_path}_fold{fold}.cbm")

    # Summary Stats
    mean_train_auc = np.mean(train_auc_scores)
    mean_val_auc = np.mean(val_auc_scores)

    print("\nCross-Validation Summary")
    print(f"Mean Train AUC: {mean_train_auc:.4f}")
    print(f"Mean Validation AUC: {mean_val_auc:.4f}")

    # Plot AUC per fold
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, n_splits + 1), train_auc_scores, marker='o', label='Train AUC', color='#3b2307')
    plt.plot(range(1, n_splits + 1), val_auc_scores, marker='o', label='Validation AUC', color='#ab6a1f')
    plt.title('CatBoost Cross-Validation AUC Scores')
    plt.xlabel('Fold')
    plt.ylabel('AUC')
    plt.xticks(range(1, n_splits + 1))
    plt.legend()
    plt.grid(True)
    plt.show()

    # Feature Importance Plot 
    plt.figure(figsize=(10, 6))
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': models[-1].get_feature_importance()
    }).sort_values(by='Importance', ascending=False)

    plt.barh(importance_df['Feature'], importance_df['Importance'], color=palette)
    plt.title('Feature Importance - Final Model')
    plt.gca().invert_yaxis()
    plt.show()

    return {
        'models': models,
        'train_auc_scores': train_auc_scores,
        'val_auc_scores': val_auc_scores,
        'mean_train_auc': mean_train_auc,
        'mean_val_auc': mean_val_auc,
        'cat_features': cat_features  
    }


# ======================
# Train & Evaluate
# ======================
catboost_results = train_optimized_catboost(X, y, model_save_path='catboost_model')
print("\nFinal Summary")
print(f"Mean Train AUC: {catboost_results['mean_train_auc']:.4f}")
print(f"Mean Validation AUC: {catboost_results['mean_val_auc']:.4f}")


def prepare_and_predict_submission(test, models, expected_columns, sample_submission_path=submission, output_path='submission.csv'):
    """
    Prepare test set (ensure columns match training), predict using ensemble of models, and save submission file.

    Parameters:
        test (DataFrame): Raw test data
        models (list): List of trained models from cross-validation
        expected_columns (list): List of columns from training data
        sample_submission_path (str): Path to sample submission (for IDs)
        output_path (str): Final submission file path

    Returns:
        DataFrame: Final submission DataFrame
    """

    print("\nEnsuring Test Data Matches Training Columns...")

    # Add missing columns with 0 or default
    for col in expected_columns:
        if col not in test.columns:
            
            test[col] = 0  

    # Reorder columns to match exactly
    test = test[expected_columns]

    print(f"Test data columns aligned with training data. Shape: {test.shape}")

    # Predict using all folds 
    print("\n Predicting Using All Fold Models...")
    all_prob_preds = np.zeros((test.shape[0], len(models)))

    for fold, model in enumerate(models):
        print(f" Predicting with Fold {fold+1}/{len(models)}")
        all_prob_preds[:, fold] = model.predict_proba(test)[:, 1]

    # Average predictions
    final_prob_preds = all_prob_preds.mean(axis=1)
    print(f"Averaged predictions across {len(models)} folds.")

    # Load sample submission to get 'id'
    sample_submission = pd.read_csv(sample_submission_path)

    # Create final submission DataFrame
    submission = pd.DataFrame({
        'id': sample_submission['id'],     
        'rainfall': final_prob_preds        
    })

    # Save to CSV
    submission.to_csv(output_path, index=False)
    print(f" Submission file saved as '{output_path}' with shape {submission.shape}")

    return submission


submission = prepare_and_predict_submission(
    test=test,
    models=catboost_results['models'],
    expected_columns=X.columns.tolist(),
    sample_submission_path='/kaggle/input/playground-series-s5e3/sample_submission.csv',  # Correct: String path to file
    output_path='submission_3.csv'
)

