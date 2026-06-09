pip install Levenshtein pgmpy


# Standard library imports
import os
import time
import logging
import warnings
from datetime import datetime
from collections import defaultdict
from itertools import combinations

# Third-party imports for data manipulation and analysis
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import boxcox, skew, spearmanr, pearsonr
from scipy.interpolate import LSQUnivariateSpline
from scipy.io import arff
import Levenshtein

# Machine learning imports
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold, 
    GridSearchCV, cross_val_score
)
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, MinMaxScaler,
    RobustScaler,
    KBinsDiscretizer, PowerTransformer
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.metrics import (
    mean_squared_error, accuracy_score, roc_auc_score,
    roc_curve, mean_squared_log_error
)
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Model-specific imports
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import statsmodels.api as sm

# Bayesian network imports
import pgmpy.estimators as ests
from pgmpy.estimators import TreeSearch
from pgmpy.models import BayesianNetwork
from pgmpy.metrics import structure_score
from pgmpy.inference import BeliefPropagation, VariableElimination

# Deep learning imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Utility imports
import joblib
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")


def load_data():
    """
    Load training, testing, and submission data.
    
    Returns:
        tuple: (train_df, test_df, submission_df)
    """
    logger.info("Loading data...")
    train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
    test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
    submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
    print("Data Loading Done!")
    print("Training Data Snap:")
    print(train.head(5))
    print("Test Data Snap:")
    print(test.head(5))
    print("Submission Format Snap:")
    print(submission.head(5))
    return train, test, submission


# Define numerical features
numerical_features = [
    "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp",
    "Calories", "BMR", 'Metabolic_Efficiency', 'Cardio_Stress',
    'Thermic_Effect', 'Power_Output', 'BVI', 'Age_Adj_Intensity',
    'Gender_Metabolic', 'HR_Drift', 'BCI', 'Thermal_Work',
    'Temp_Binary', 'HeartRate_Binary', 'Sex'
]
    
# Load data
train_df, test_df, submission_df = load_data()


def intersection_of_lists(list1, list2):
    return list(set(list1) & set(list2))


def difference_of_lists(list1, list2):
    return [item for item in list1 if item not in list2]


def remove_single_unique_or_all_nans(df):
    removed_columns = []
    for column in df.columns:
        if df[column].nunique() <= 1 or df[column].isna().all():
            removed_columns.append(column)
            df = df.drop(columns=[column])
    print(f"Removed columns due to all NaN or only 1 unique value: {removed_columns}")
    return df


def columns_with_missing_values(df):
    missing_cols = [col for col in df.columns if df[col].isna().values.any()]
    print(f"Missing data columns: {missing_cols}")
    return missing_cols


def columns_with_more_than_X_percent_unique(df, colNames, perc):
    total_rows = len(df)
    threshold = total_rows * 0.01 * perc  
    cols_with_high_uniques = [col for col in colNames if df[col].nunique() > threshold]
    print(f"Columns with high uniques , >= {perc} %  of number of rows in the data: {cols_with_high_uniques}")
    return cols_with_high_uniques


def get_numeric_and_non_numeric_columns(df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        print(f"Numeric columns: {numeric_cols}")
        print(f"Non-numeric columns: {non_numeric_cols}")
        return numeric_cols, non_numeric_cols


Target_Col = ['Calories']
Identifier_Cols = ['id']
X_Cols = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", 'Sex']
data = train_df.copy()


Numeric_Cols, Non_Numeric_Cols = get_numeric_and_non_numeric_columns(data[X_Cols])
MissingData_Cols = columns_with_missing_values(data[X_Cols])
GreaterThanTENpercUniQ_Cols = columns_with_more_than_X_percent_unique(data, Numeric_Cols, 75)
GreaterThanEIGHTpercUniQ_Cols = columns_with_more_than_X_percent_unique(data, Numeric_Cols, 50)
GreaterThanFIVEpercUniQ_Cols = columns_with_more_than_X_percent_unique(data, Numeric_Cols, 25)
GreaterThanONEpercUniQ_Cols = columns_with_more_than_X_percent_unique(data, Numeric_Cols, 1)
LessUniqueNA_Cols = intersection_of_lists(MissingData_Cols, difference_of_lists(GreaterThanONEpercUniQ_Cols, GreaterThanFIVEpercUniQ_Cols))


def plot_numeric_features(df, numerical_features, apply_box_cox=False):
    """
    Function to plot density plots for features with absolute skewness > 10, histograms otherwise,
    and box plots for all numerical features. Applies Box-Cox transformation if specified.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - numerical_features (list): List of numeric column names.
    - apply_box_cox (bool): If True, applies Box-Cox transformation to features with high skewness.
    """
    for feature in numerical_features:
        # Drop rows with missing values for the current feature
        valid_data = df[feature].dropna()

        if valid_data.empty:
            print(f"No valid data available for feature: {feature}")
            continue
        
        # Calculate skewness
        skewness = valid_data.skew()

        plt.figure(figsize=(12, 6))

        # Conditional plotting based on skewness
        if abs(skewness) > 10:
            if apply_box_cox:
                # Apply Box-Cox transformation (only for positive values)
                valid_data = valid_data[valid_data > 0]  # Box-Cox requires positive values
                if valid_data.empty:
                    print(f"No valid positive data available for Box-Cox transformation for {feature}")
                    continue
                transformed_data, _ = boxcox(valid_data)
                plt.subplot(1, 2, 1)
                sns.kdeplot(transformed_data, fill=True)
                plt.title(f"Density Plot of {feature} (Box-Cox Transformed)")
                plt.xlabel(f"{feature} (Box-Cox Transformed)")
                plt.ylabel("Density")
            else:
                # Density plot for features with high skewness (without transformation)
                plt.subplot(1, 2, 1)
                sns.kdeplot(valid_data, fill=True)
                plt.title(f"Density Plot of {feature} (Skewness: {skewness:.2f})")
                plt.xlabel(feature)
                plt.ylabel("Density")
        else:
            # Histogram for features with lower skewness
            plt.subplot(1, 2, 1)
            sns.histplot(valid_data, kde=True, bins=30)
            plt.title(f"Histogram of {feature} (Skewness: {skewness:.2f})")
            plt.xlabel(feature)
            plt.ylabel("Frequency")

        # Box plot for all features
        plt.subplot(1, 2, 2)
        sns.boxplot(x=valid_data)
        plt.title(f"Box Plot of {feature}")
        
        plt.tight_layout()
        plt.show()

        # Print additional statistics
        print(f"\nStatistics for {feature}:")
        print(f"Skewness: {skewness:.2f}")
        print(f"Number of Missing Values: {df[feature].isnull().sum()}")


def plot_categorical_features(df, categorical_features):
    """
    Function to plot pie charts for categorical features with fewer than 10 unique values,
    or bar graphs otherwise.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - categorical_features (list): List of categorical column names.
    """
    for feature in categorical_features:
        # Drop rows with missing values for the current feature
        valid_data = df[feature].dropna()
        
        if valid_data.empty:
            print(f"No valid data available for feature: {feature}")
            continue
        
        # Calculate value counts
        value_counts = valid_data.value_counts()
        
        # Decide chart type based on the number of unique values
        if value_counts.size < 11:
            # Pie chart for features with fewer than 11 unique values
            percentages = (value_counts / value_counts.sum()) * 100
            plt.figure(figsize=(8, 8))
            plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
            plt.title(f"Distribution of {feature} (Pie Chart)")
            plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
        else:
            # Bar graph for features with 11 or more unique values
            plt.figure(figsize=(10, 6))
            plt.bar(value_counts.index, value_counts.values, color=plt.cm.Paired.colors[:len(value_counts)])
            plt.title(f"Distribution of {feature} (Bar Graph)")
            plt.xlabel(feature)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()

        # Print additional statistics
        print(f"Statistics for {feature}:")
        print(f"Number of Unique Values: {df[feature].nunique()}")
        print(f"Missing Values in {feature}: {df[feature].isnull().sum()}")


def plot_correlation_heatmap(df, numerical_features, corr_type="spearman"):
    """
    Function to plot a correlation heatmap for numerical features using specified correlation type.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - numerical_features (list): List of numeric column names.
    - corr_type (str): Type of correlation ('spearman', 'pearson', 'kendall', 'mic', 'pps').
    """
    # Filter valid numerical columns
    valid_data = df[numerical_features].dropna()
    
    if corr_type == "spearman":
        correlation_matrix = valid_data.corr(method="spearman")
    elif corr_type == "pearson":
        correlation_matrix = valid_data.corr(method="pearson")
    elif corr_type == "kendall":
        correlation_matrix = valid_data.corr(method="kendall")
    else:
        raise ValueError(f"Unsupported correlation type: {corr_type}. Use 'spearman', 'pearson', 'kendall'.")
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Correlation Matrix of Numerical Features ({corr_type.title()} Correlation)")
    plt.show()


def plot_categorical_boxplots(df, categorical_features, label_col):
    """
    Function to plot box plots for categorical features against a label column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - categorical_features (list): List of categorical column names.
    - label_col (str): Name of the label column for the y-axis.
    """
    for feature in categorical_features:
        # Skip high-cardinality features
        if feature not in ["Podcast_Name", "Episode_Title"]:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature], y=df[label_col])
            plt.title(f"{feature} vs. {label_col}")
            plt.xlabel(feature)
            plt.ylabel(label_col)
            plt.xticks(rotation=45)
            plt.tight_layout()  # Ensure plots fit within the figure area
            plt.show()


def plot_numeric_analysis_with_sampling(df, numeric_cols, label_col, sample=0.1):
    """
    Function to compute MIC and PPS scores and plot univariate relationships between numeric columns
    and a label column, using sampled data for faster processing.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - numeric_cols (list): List of numeric column names.
    - label_col (str): Name of the label column (y-variable).
    - sample (float): Fraction of valid data to sample (between 0 and 1).
    """
    logging.info("Starting numeric analysis function.")
    mine = MINE()

    for numeric_col in numeric_cols:
        logging.info(f"Processing column: {numeric_col}")
        
        # Filter rows with non-missing values for both the numeric column and the label column
        valid_data = df[[numeric_col, label_col]].dropna()

        if valid_data.empty:
            logging.warning(f"No valid data available for {numeric_col} vs. {label_col}. Skipping...")
            continue

        logging.info(f"Valid data fetched for {numeric_col} vs. {label_col}. Rows: {len(valid_data)}")
        
        # Sample the data if the fraction is specified
        if 0 < sample < 1:
            valid_data = valid_data.sample(frac=sample, random_state=42)
            logging.info(f"Data sampled. Using {len(valid_data)} rows for analysis.")

        # MIC calculation
        logging.info(f"Calculating MIC for {numeric_col} vs. {label_col}.")
        mine.compute_score(valid_data[numeric_col].values, valid_data[label_col].values)
        mic_score = mine.mic()
        logging.info(f"MIC calculated: {mic_score:.2f}")

        # PPS calculation
        logging.info(f"Calculating PPS for {numeric_col} vs. {label_col}.")
        pps_score = pps.score(valid_data, x=numeric_col, y=label_col).get("ppscore", 0)
        logging.info(f"PPS calculated: {pps_score:.2f}")

        # Plot univariate relationship
        logging.info(f"Generating scatter plot for {numeric_col} vs. {label_col}.")
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_data[numeric_col], valid_data[label_col], alpha=0.6, color="blue")
        plt.title(f"{numeric_col} vs. {label_col} (MIC: {mic_score:.2f}, PPS: {pps_score:.2f})")
        plt.xlabel(numeric_col)
        plt.ylabel(label_col)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        logging.info(f"Scatter plot displayed for {numeric_col} vs. {label_col}.")

        # Log the results
        logging.info(f"MIC: {mic_score:.2f}, PPS: {pps_score:.2f} for {numeric_col} vs. {label_col}.")

    logging.info("Numeric analysis function completed.")


def generate_pair_plot_for_numeric_columns(df, numeric_cols, hue=None, sample_size=1000):
    """
    Function to generate a pair plot for a given list of numeric columns, with optional hue and sampling.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - numeric_cols (list): List of numeric columns to include in the pair plot.
    - hue (str, optional): Name of the categorical column for hue. If None, no hue is used.
    - sample_size (int): The number of rows to sample for the pair plot (default=5000).
    """
    logging.info("Starting pair plot generation.")

    # Check if all required numeric columns exist in the DataFrame
    if all(col in df.columns for col in numeric_cols):
        logging.info("All numeric columns found in the DataFrame.")
        
        # Sample the data if the dataset exceeds the sample size
        if len(df) > sample_size:
            logging.info(f"Sampling {sample_size} data points for the pair plot.")
            df_sample = df.sample(n=sample_size, random_state=42)
        else:
            logging.info("Using the entire dataset for the pair plot.")
            df_sample = df

        # Generate the pair plot
        logging.info("Generating the pair plot (this may take some time).")
        sns.pairplot(
            df_sample[numeric_cols].dropna(),
            hue=hue if hue and hue in df.columns else None,  # Use hue only if it's provided and valid
            palette='dark' if hue else None,
            diag_kind='kde',
            plot_kws={'alpha': 0.6, 's': 10},
        )
        title = f"Pairwise Relationships" + (f" by {hue}" if hue else "")
        plt.suptitle(title, y=1.02)
        plt.show()
        logging.info("Pair plot generated successfully.")
    else:
        missing_cols = [col for col in numeric_cols if col not in df.columns]
        logging.warning(f"Missing numeric columns for pair plot: {missing_cols}")


def plot_numeric_vs_target_density(df, numeric_cols, target_col):
    """
    Function to generate density plots (hexbin) for numeric columns against a target column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - numeric_cols (list): List of numeric column names to compare against the target column.
    - target_col (str): The target column (y-axis variable) for the density plot.
    """
    logging.info("Starting density plot generation.")
    
    # Check if the target column exists
    if target_col not in df.columns:
        logging.error(f"Target column '{target_col}' not found in the DataFrame.")
        return
    
    for numeric_col in numeric_cols:
        # Check if the numeric column exists in the DataFrame
        if numeric_col not in df.columns:
            logging.warning(f"Numeric column '{numeric_col}' not found in the DataFrame. Skipping...")
            continue
        
        logging.info(f"Generating density plot for {numeric_col} vs. {target_col}.")
        
        # Create the jointplot
        sns.jointplot(
            data=df,
            x=numeric_col,
            y=target_col,
            kind='hex',
            cmap='viridis',
            gridsize=40
        )
        plt.suptitle(f'Density of {numeric_col} vs. {target_col}', y=1.02)
        plt.tight_layout()
        plt.show()
        logging.info(f"Density plot for {numeric_col} vs. {target_col} generated successfully.")
    
    logging.info("Density plot generation completed.")


def generate_categorical_numeric_plot(
    df, 
    cat_col1, 
    cat_col2, 
    numeric_col, 
    cat1_order=None, 
    cat2_order=None, 
    figsize=(12, 6), 
    errorbar_ci=99
):
    """
    Function to generate a plot for a numeric column aggregated by two categorical columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - cat_col1 (str): Name of the first categorical column (e.g., 'Day_of_Week').
    - cat_col2 (str): Name of the second categorical column (e.g., 'Time_of_Day').
    - numeric_col (str): Name of the numeric column (e.g., 'Listening_Time_minutes').
    - cat1_order (list): Desired order of the first categorical column (optional).
    - cat2_order (list): Desired order of the second categorical column (optional).
    - figsize (tuple): Size of the plot (default=(12, 6)).
    - errorbar_ci (int): Confidence interval for error bars (default=99).

    Returns:
    None
    """
    try:
        logging.info("Starting categorical-numeric plot generation...")
        
        # Process the first categorical column
        if cat1_order:
            if cat_col1 in df.columns:
                logging.info(f"Processing {cat_col1} with specified order...")
                df[cat_col1] = pd.Categorical(df[cat_col1], categories=cat1_order, ordered=True)
            else:
                raise ValueError(f"Column '{cat_col1}' not found in the DataFrame.")
        
        # Process the second categorical column
        if cat2_order:
            if cat_col2 in df.columns:
                logging.info(f"Processing {cat_col2} with specified order...")
                df[cat_col2] = pd.Categorical(df[cat_col2], categories=cat2_order, ordered=True)
            else:
                raise ValueError(f"Column '{cat_col2}' not found in the DataFrame.")
        
        # Check if numeric column exists
        if numeric_col not in df.columns:
            raise ValueError(f"Numeric column '{numeric_col}' not found in the DataFrame.")
        
        # Prepare the data for plotting
        logging.info("Filtering data for valid rows...")
        plot_data = df.dropna(subset=[cat_col1, cat_col2, numeric_col])

        if plot_data.empty:
            logging.warning("No valid data available for plotting after filtering NaNs.")
            return
        
        # Generate the plot
        logging.info("Generating the plot...")
        plt.figure(figsize=figsize)
        palette = sns.color_palette("tab10", n_colors=len(cat2_order) if cat2_order else 10)
        sns.lineplot(
            data=plot_data,
            x=cat_col1,
            y=numeric_col,
            hue=cat_col2,
            hue_order=cat2_order,
            palette=palette,
            marker='o',
            errorbar=('ci', errorbar_ci)
        )
        plt.title(f'Average {numeric_col} by {cat_col1} and {cat_col2}')
        plt.xlabel(cat_col1)
        plt.ylabel(f'Average {numeric_col}')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title=cat_col2, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()
        logging.info("Plot generated successfully.")

    except Exception as e:
        logging.error(f"Error in categorical-numeric plot generation: {e}")


plot_categorical_features(data, Non_Numeric_Cols)


plot_categorical_boxplots(data, Non_Numeric_Cols, "Calories")


plot_numeric_features(data, Numeric_Cols, apply_box_cox=True)


plot_correlation_heatmap(data, Numeric_Cols, corr_type="spearman")


plot_correlation_heatmap(data, Numeric_Cols, corr_type="pearson")


plot_correlation_heatmap(data, Numeric_Cols, corr_type="kendall")


Numeric_Cols2 = ["Age", "Height", "Duration", "Heart_Rate", "Body_Temp","Weight"]
plot_numeric_vs_target_density(data, Numeric_Cols2, "Calories")


def create_features(train_df, test_df):
    """
    Comprehensive feature engineering function that combines all feature creation steps
    and adds polynomial features.
    
    Args:
        train_df (pd.DataFrame): Training dataframe
        test_df (pd.DataFrame): Test dataframe
        
    Returns:
        tuple: (processed_train_df, processed_test_df)
    """
    logger.info("Starting comprehensive feature engineering...")
    print("\n=== Starting Feature Engineering ===")
    
    # Create copies to avoid modifying original dataframes
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    
    # 1. Basic Preprocessing
    logger.info("Performing basic preprocessing...")
    print("Performing basic preprocessing...")
    
    # Convert sex to binary
    train_processed['Sex'] = train_processed['Sex'].map({'male': 1, 'female': 0})
    test_processed['Sex'] = test_processed['Sex'].map({'male': 1, 'female': 0})
    
    # Remove duplicates and get minimum calories for same features
    train_processed = train_processed.drop_duplicates(subset=train_processed.columns).reset_index(drop=True)
    train_processed = train_processed.groupby(['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])['Calories'].min().reset_index()
    
    # 2. Basic Features
    logger.info("Creating basic features...")
    print("Creating basic features...")
    
    # BMI and Intensity
    train_processed['BMI'] = train_processed['Weight'] / (train_processed['Height'] / 100) ** 2
    test_processed['BMI'] = test_processed['Weight'] / (test_processed['Height'] / 100) ** 2
    
    train_processed['Intensity'] = train_processed['Heart_Rate'] / train_processed['Duration']
    test_processed['Intensity'] = test_processed['Heart_Rate'] / test_processed['Duration']
    
    # 3. Polynomial Features
    logger.info("Creating polynomial features...")
    print("Creating polynomial features...")
    
    # Define features for polynomial combinations
    poly_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI']
    
    # Create polynomial features
    for i in range(len(poly_features)):
        for j in range(i+1, len(poly_features)):
            feat1, feat2 = poly_features[i], poly_features[j]
            
            # Multiplication
            train_processed[f'{feat1}_x_{feat2}'] = train_processed[feat1] * train_processed[feat2]
            test_processed[f'{feat1}_x_{feat2}'] = test_processed[feat1] * test_processed[feat2]
            
            # Division (avoid division by zero)
            train_processed[f'{feat1}_div_{feat2}'] = train_processed[feat1] / (train_processed[feat2] + 1e-6)
            test_processed[f'{feat1}_div_{feat2}'] = test_processed[feat1] / (test_processed[feat2] + 1e-6)
            
            # Square of each feature
            train_processed[f'{feat1}_squared'] = train_processed[feat1] ** 2
            test_processed[f'{feat1}_squared'] = test_processed[feat1] ** 2
    
    # 4. Advanced Physiological Features
    logger.info("Creating advanced physiological features...")
    print("Creating advanced physiological features...")
    
    # Basal Metabolic Rate (BMR)
    train_processed['BMR'] = train_processed['Weight'] / ((train_processed['Height'] / 100) ** 2)
    test_processed['BMR'] = test_processed['Weight'] / ((test_processed['Height'] / 100) ** 2)
    
    # Metabolic Efficiency Index
    train_processed['Metabolic_Efficiency'] = train_processed['BMR'] * (train_processed['Heart_Rate'] / train_processed['BMR'].median())
    test_processed['Metabolic_Efficiency'] = test_processed['BMR'] * (test_processed['Heart_Rate'] / test_processed['BMR'].median())
    
    # Cardiovascular Stress
    train_processed['Cardio_Stress'] = (train_processed['Heart_Rate'] / (220 - train_processed['Age'])) * train_processed['Duration']
    test_processed['Cardio_Stress'] = (test_processed['Heart_Rate'] / (220 - test_processed['Age'])) * test_processed['Duration']
    
    # Thermic Effect Ratio
    train_processed['Thermic_Effect'] = (train_processed['Body_Temp'] * 100) / (train_processed['Weight'] ** 0.5)
    test_processed['Thermic_Effect'] = (test_processed['Body_Temp'] * 100) / (test_processed['Weight'] ** 0.5)
    
    # Power Output Estimate
    train_processed['Power_Output'] = train_processed['Weight'] * train_processed['Duration'] * (train_processed['Heart_Rate'] / 1000)
    test_processed['Power_Output'] = test_processed['Weight'] * test_processed['Duration'] * (test_processed['Heart_Rate'] / 1000)
    
    # 5. Interaction Features
    logger.info("Creating interaction features...")
    print("Creating interaction features...")
    
    # Duration-based features
    train_durations = sorted(train_processed['Duration'].unique())
    for dur in train_durations:
        train_processed[f'HR_Dur_{int(dur)}'] = np.where(train_processed['Duration'] == dur, train_processed['Heart_Rate'], 0)
        test_processed[f'HR_Dur_{int(dur)}'] = np.where(test_processed['Duration'] == dur, test_processed['Heart_Rate'], 0)
        
        train_processed[f'Temp_Dur_{int(dur)}'] = np.where(train_processed['Duration'] == dur, train_processed['Body_Temp'], 0)
        test_processed[f'Temp_Dur_{int(dur)}'] = np.where(test_processed['Duration'] == dur, test_processed['Body_Temp'], 0)
    
    # Age-based features
    train_ages = sorted(train_processed['Age'].unique())
    for age in train_ages:
        train_processed[f'HR_Age_{int(age)}'] = np.where(train_processed['Age'] == age, train_processed['Heart_Rate'], 0)
        test_processed[f'HR_Age_{int(age)}'] = np.where(test_processed['Age'] == age, test_processed['Heart_Rate'], 0)
        
        train_processed[f'Temp_Age_{int(age)}'] = np.where(train_processed['Age'] == age, train_processed['Body_Temp'], 0)
        test_processed[f'Temp_Age_{int(age)}'] = np.where(test_processed['Age'] == age, test_processed['Body_Temp'], 0)
    
    # 6. Statistical Features
    logger.info("Creating statistical features...")
    print("Creating statistical features...")
    
    for col in ['Height', 'Weight', 'Heart_Rate', 'Body_Temp']:
        for agg in ['min', 'max']:
            agg_val = train_processed.groupby('Sex')[col].agg(agg).rename(f'Sex_{col}_{agg}')
            train_processed = train_processed.merge(agg_val, on='Sex', how='left')
            test_processed = test_processed.merge(agg_val, on='Sex', how='left')
    
    # 7. Additional Derived Features
    logger.info("Creating additional derived features...")
    print("Creating additional derived features...")
    
    # Body Volume Index
    train_processed['BVI'] = train_processed['Weight'] / ((train_processed['Height']/100) ** 3)
    test_processed['BVI'] = test_processed['Weight'] / ((test_processed['Height']/100) ** 3)
    
    # Age-Adjusted Intensity
    bins = [18, 25, 35, 45, 55, 65, 100]
    train_processed['Age_Adj_Intensity'] = train_processed['Duration'] * pd.cut(train_processed['Age'], bins).cat.codes
    test_processed['Age_Adj_Intensity'] = test_processed['Duration'] * pd.cut(test_processed['Age'], bins).cat.codes
    
    # Gender-Specific Metabolic Rate
    gender_coeff = {'male': 1.67, 'female': 1.55}
    train_processed['Gender_Metabolic'] = train_processed['Sex'].map(gender_coeff) * train_processed['BMR']
    test_processed['Gender_Metabolic'] = test_processed['Sex'].map(gender_coeff) * test_processed['BMR']
    
    # Cardiovascular Drift
    train_processed['HR_Drift'] = train_processed.groupby('Age')['Heart_Rate'].diff() / train_processed['Duration']
    test_processed['HR_Drift'] = test_processed.groupby('Age')['Heart_Rate'].diff() / test_processed['Duration']
    
    # Body Composition Index
    train_processed['BCI'] = (train_processed['Weight'] * 1000) / (train_processed['Height'] ** 1.5) * (1 / (train_processed['Age'] ** 0.2))
    test_processed['BCI'] = (test_processed['Weight'] * 1000) / (test_processed['Height'] ** 1.5) * (1 / (test_processed['Age'] ** 0.2))
    
    # Thermal Work Capacity
    train_processed['Thermal_Work'] = (train_processed['Body_Temp'] ** 2) * np.log1p(train_processed['Duration'])
    test_processed['Thermal_Work'] = (test_processed['Body_Temp'] ** 2) * np.log1p(test_processed['Duration'])
    
    # Binary Features
    train_processed['Temp_Binary'] = np.where(train_processed['Body_Temp'] <= 39.5, 0, 1)
    test_processed['Temp_Binary'] = np.where(test_processed['Body_Temp'] <= 39.5, 0, 1)
    
    train_processed['HeartRate_Binary'] = np.where(train_processed['Heart_Rate'] <= 99.5, 0, 1)
    test_processed['HeartRate_Binary'] = np.where(test_processed['Heart_Rate'] <= 99.5, 0, 1)
    
    # Log feature creation summary
    logger.info(f"Created {len(train_processed.columns)} features")
    print(f"Created {len(train_processed.columns)} features")
    
    return train_processed, test_processed


# Feature engineering
train_processed, test_processed = create_features(train_df, test_df)


def transform_features(train_df, test_df, numerical_features):
    """
    Transform features to handle skewness and outliers.
    
    Args:
        train_df (pd.DataFrame): Training dataframe
        test_df (pd.DataFrame): Test dataframe
        numerical_features (list): List of numerical feature names
        
    Returns:
        tuple: (transformed_train, transformed_test)
    """
    logger.info("Transforming features...")
    numeric_cols = [col for col in numerical_features if col != "Calories"]
    
    # Calculate original skewness
    original_skewness = train_df[numeric_cols].skew().sort_values(ascending=False)
    
    # Initialize transformed DataFrames
    train_df_transformed = train_df.copy()
    test_df_transformed = test_df.copy()
    
    # Store transformers for each column
    transformers = {}
    
    # Apply skewness correction
    for col in numeric_cols:
        if train_df[col].nunique() <= 1:
            continue
        
        if original_skewness[col] > 0.5:  # Right skew
            if (train_df[col] > 0).all():
                # Log transform
                train_df_transformed[col] = np.log1p(train_df[col])
                test_df_transformed[col] = np.log1p(test_df[col])
            else:
                # Yeo-Johnson transform
                pt = PowerTransformer(method='yeo-johnson')
                train_df_transformed[col] = pt.fit_transform(train_df[[col]])
                test_df_transformed[col] = pt.transform(test_df[[col]])
                transformers[col] = pt
        elif original_skewness[col] < -0.5:  # Left skew
            pt = PowerTransformer(method='yeo-johnson')
            train_df_transformed[col] = pt.fit_transform(train_df[[col]])
            test_df_transformed[col] = pt.transform(test_df[[col]])
            transformers[col] = pt
    
    return train_df_transformed, test_df_transformed

def remove_outliers(df, numeric_cols):
    """
    Remove only the most extreme 1% of outliers using percentile-based method.
    
    Args:
        df (pd.DataFrame): Input dataframe
        numeric_cols (list): List of numerical column names
        
    Returns:
        pd.DataFrame: DataFrame with only the most extreme 1% of outliers removed
    """
    logger.info("Removing extreme outliers (top and bottom 0.5%)...")
    df_cleaned = df.copy()
    
    # Calculate total rows to remove (1% of data)
    total_rows = len(df_cleaned)
    rows_to_remove = int(total_rows * 0.01)
    
    # Calculate outlier scores for each row
    outlier_scores = np.zeros(len(df_cleaned))
    
    for col in numeric_cols:
        # Calculate z-scores for each column
        z_scores = np.abs((df_cleaned[col] - df_cleaned[col].mean()) / df_cleaned[col].std())
        # Add to total outlier score
        outlier_scores += z_scores
    
    # Get indices of rows with highest outlier scores
    outlier_indices = np.argsort(outlier_scores)[-rows_to_remove:]
    
    # Remove only the most extreme outliers
    df_cleaned = df_cleaned.drop(df_cleaned.index[outlier_indices])
    
    logger.info(f"Removed {rows_to_remove} rows ({rows_to_remove/total_rows*100:.2f}% of data)")
    return df_cleaned


# Transform features
train_df_transformed, test_df_transformed = transform_features(train_processed, test_processed, numerical_features)
    
# Remove outliers
cleaned_train_df = remove_outliers(train_df_transformed, numerical_features)
cleaned_test_df = test_df_transformed.copy()


# Prepare data for modeling
X = cleaned_train_df.drop(columns=['Calories'])
y = np.log1p(cleaned_train_df['Calories'])
X_test = cleaned_test_df.drop(columns=['id'])


def save_model_and_metrics(model, model_name, metrics, model_dir='/kaggle/working/saved_models/'):
    """
    Save model and its metrics to disk.
    
    Args:
        model: Trained model object
        model_name (str): Name of the model (e.g., 'catboost', 'xgboost', 'neural_net')
        metrics (dict): Dictionary containing model metrics
        model_dir (str): Directory to save models
    """
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate timestamp for unique model version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_name}_{timestamp}.joblib"
    model_path = os.path.join(model_dir, model_filename)
    
    # Save model
    joblib.dump(model, model_path)
    
    # Save metrics
    metrics_filename = f"{model_name}_{timestamp}_metrics.joblib"
    metrics_path = os.path.join(model_dir, metrics_filename)
    joblib.dump(metrics, metrics_path)
    
    logger.info(f"Saved model and metrics to {model_dir}")
    print(f"Saved model and metrics to {model_dir}")
    
    return model_path, metrics_path


def train_catboost(X, y, X_test):
    """
    Train CatBoost model with cross-validation.
    
    Args:
        X (pd.DataFrame): Training features
        y (pd.Series): Target variable
        X_test (pd.DataFrame): Test features
        
    Returns:
        tuple: (predictions, out-of-fold predictions, scores, model, total_time)
    """
    logger.info("Starting CatBoost training...")
    print("\n=== Starting CatBoost Training ===")
    start_time = time.time()
    
    # Prepare data
    logger.info("Preparing data for CatBoost...")
    print("Preparing data for CatBoost...")
    
    # Ensure feature alignment
    logger.info("Aligning features between train and test...")
    print("Aligning features between train and test...")
    common_features = list(set(X.columns) & set(X_test.columns))
    X_cat = X[common_features].copy()
    X_test_cat = X_test[common_features].copy()
    
    # Create duration bins for stratified splitting
    bins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    duration_bins = bins.fit_transform(X_cat[['Duration']]).astype(int).flatten()

    # CatBoost parameters
    cat_params = {
        'iterations': 2500,
        'learning_rate': 0.02,
        'depth': 10,
        'loss_function': 'RMSE',
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'eval_metric': 'RMSE',
        'early_stopping_rounds': 200,
        'cat_features': ['Sex'],
        'verbose': 100,  # Show progress every 100 iterations
        'task_type': 'GPU',
        'thread_count': -1,  # Use all available threads
        'gpu_ram_part': 0.8,  # Use 80% of GPU memory
        'bootstrap_type': 'Bayesian',
        'bagging_temperature': 0.8,
        'random_strength': 0.8,
        'min_data_in_leaf': 20,
        'max_leaves': 31,
        'feature_border_type': 'UniformAndQuantiles',
        'leaf_estimation_iterations': 10,
        'boosting_type': 'Plain',
        'grow_policy': 'Lossguide',
        'max_bin': 256
    }

    # Initialize prediction arrays
    cat_preds = np.zeros(len(X_test_cat))
    cat_oof = np.zeros(len(X_cat))
    cat_scores = []
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    total_folds = skf.n_splits
    
    logger.info(f"Starting {total_folds}-fold cross-validation...")
    print(f"\nStarting {total_folds}-fold cross-validation...")
    logger.info(f"Training with {len(common_features)} features: {', '.join(common_features)}")
    print(f"Training with {len(common_features)} features")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_cat, duration_bins), 1):
        fold_start_time = time.time()
        logger.info(f"\nFold {fold}/{total_folds} - Starting training...")
        print(f"\nFold {fold}/{total_folds} - Starting training...")
        
        # Create and train model
        model = CatBoostRegressor(**cat_params)
        
        # Train model with progress logging
        model.fit(
            X_cat.iloc[train_idx], 
            y.iloc[train_idx],
            eval_set=(X_cat.iloc[val_idx], y.iloc[val_idx]),
            use_best_model=True,
            verbose=100
        )
        
        # Make predictions
        logger.info(f"Fold {fold} - Making predictions...")
        print(f"Fold {fold} - Making predictions...")
        cat_oof[val_idx] = model.predict(X_cat.iloc[val_idx])
        cat_preds += model.predict(X_test_cat) / skf.n_splits
        
        # Calculate fold score
        fold_score = np.sqrt(mean_squared_log_error(
            np.expm1(y.iloc[val_idx]), 
            np.expm1(cat_oof[val_idx])
        ))
        
        # Calculate fold timing
        fold_time = time.time() - fold_start_time
        
        logger.info(f"Fold {fold} completed:")
        logger.info(f"  - RMSLE Score: {fold_score:.5f}")
        logger.info(f"  - Time taken: {fold_time:.2f} seconds")
        print(f"\nFold {fold} completed:")
        print(f"  - RMSLE Score: {fold_score:.5f}")
        print(f"  - Time taken: {fold_time:.2f} seconds")
        
        cat_scores.append(fold_score)
        
        # Estimate remaining time
        if fold < total_folds:
            avg_fold_time = (time.time() - start_time) / fold
            remaining_folds = total_folds - fold
            estimated_time = avg_fold_time * remaining_folds
            logger.info(f"Estimated time remaining: {estimated_time/60:.1f} minutes")
            print(f"Estimated time remaining: {estimated_time/60:.1f} minutes")
    
    # Calculate and log final metrics
    total_time = time.time() - start_time
    mean_score = np.mean(cat_scores)
    std_score = np.std(cat_scores)
    
    logger.info("\nCatBoost Training Summary:")
    logger.info(f"  - Mean RMSLE: {mean_score:.5f} ± {std_score:.5f}")
    logger.info(f"  - Total training time: {total_time/60:.1f} minutes")
    logger.info(f"  - Average fold time: {total_time/total_folds:.1f} seconds")
    print("\n=== CatBoost Training Summary ===")
    print(f"  - Mean RMSLE: {mean_score:.5f} ± {std_score:.5f}")
    print(f"  - Total training time: {total_time/60:.1f} minutes")
    print(f"  - Average fold time: {total_time/total_folds:.1f} seconds")
    
    return cat_preds, cat_oof, cat_scores, model, total_time


# Train models and save them
cat_preds, cat_oof, cat_scores, cat_model, cat_total_time = train_catboost(X, y, X_test)


cat_params = {
        'iterations': 2500,
        'learning_rate': 0.02,
        'depth': 10,
        'loss_function': 'RMSE',
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'eval_metric': 'RMSE',
        'early_stopping_rounds': 200,
        'cat_features': ['Sex'],
        'verbose': 100,  # Show progress every 100 iterations
        'task_type': 'GPU',
        'thread_count': -1,  # Use all available threads
        'gpu_ram_part': 0.8,  # Use 80% of GPU memory
        'bootstrap_type': 'Bayesian',
        'bagging_temperature': 0.8,
        'random_strength': 0.8,
        'min_data_in_leaf': 20,
        'max_leaves': 31,
        'feature_border_type': 'UniformAndQuantiles',
        'leaf_estimation_iterations': 10,
        'boosting_type': 'Plain',
        'grow_policy': 'Lossguide',
        'max_bin': 256
    }
cat_metrics = {
        'train_rmsle': np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(cat_oof))),
        'val_rmsle': np.mean(cat_scores),
        'test_rmsle': None,  # We don't have true test values
        'improvement': 0,  # Will be updated in future runs
        'training_time': cat_total_time,
        'model_specs': cat_params
    }


# save_model_and_metrics(cat_model, 'catboost', cat_metrics)


# Create submission with CAT-Boost Baseline
final_preds = np.expm1(cat_preds) 
submission_df['Calories'] = np.clip(final_preds, 1, 314)
submission_df.to_csv('submission_CATBoost_Baseline.csv', index=False)


def train_xgboost(X, y, X_test):
    """
    Train XGBoost model with cross-validation.
    
    Args:
        X (pd.DataFrame): Training features
        y (pd.Series): Target variable
        X_test (pd.DataFrame): Test features
        
    Returns:
        tuple: (predictions, out-of-fold predictions, scores, model, total_time)
    """
    logger.info("Starting XGBoost training...")
    print("\n=== Starting XGBoost Training ===")
    start_time = time.time()
    
    # Prepare data
    logger.info("Preparing data for XGBoost...")
    print("Preparing data for XGBoost...")
    
    # Ensure feature alignment
    logger.info("Aligning features between train and test...")
    print("Aligning features between train and test...")
    common_features = list(set(X.columns) & set(X_test.columns))
    X_xgb = X[common_features].copy()
    X_test_xgb = X_test[common_features].copy()
    
    # Convert categorical features
    X_xgb['Sex'] = X_xgb['Sex'].astype(int)
    X_test_xgb['Sex'] = X_test_xgb['Sex'].astype(int)

    # Initialize prediction arrays
    xgb_oof = np.zeros(len(X))
    xgb_preds = np.zeros(len(X_test))
    xgb_scores = []  # Initialize scores list

    # XGBoost parameters
    xgb_params = {
        'max_depth': 10,
        'colsample_bytree': 0.75,
        'subsample': 0.9,
        'n_estimators': 2000,
        'learning_rate': 0.02,
        'gamma': 0.01,
        'max_delta_step': 2,
        'eval_metric': 'rmse',
        'enable_categorical': False,
        'random_state': 42,
        'early_stopping_rounds': 100,
        'tree_method': 'gpu_hist',
        'n_jobs': -1,  # Use all available CPU cores
        'n_gpus': -1,  # Use all available GPUs
        'predictor': 'gpu_predictor',
        'sampling_method': 'gradient_based',
        'max_bin': 256,
        'grow_policy': 'lossguide'
    }

    # Cross-validation
    kf = KFold(n_splits=20, shuffle=True, random_state=42)
    total_folds = kf.n_splits
    
    logger.info(f"Starting {total_folds}-fold cross-validation...")
    print(f"\nStarting {total_folds}-fold cross-validation...")
    logger.info(f"Training with {len(common_features)} features: {', '.join(common_features)}")
    print(f"Training with {len(common_features)} features")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_xgb), 1):
        fold_start_time = time.time()
        logger.info(f"\nFold {fold}/{total_folds} - Starting training...")
        print(f"\nFold {fold}/{total_folds} - Starting training...")
        
        # Create and train model
        model = XGBRegressor(**xgb_params)
        
        # Train model with progress logging
        model.fit(
            X_xgb.iloc[train_idx], 
            y.iloc[train_idx],
            eval_set=[(X_xgb.iloc[val_idx], y.iloc[val_idx])],
            verbose=100  # Show progress every 100 iterations
        )
        
        # Make predictions
        logger.info(f"Fold {fold} - Making predictions...")
        print(f"Fold {fold} - Making predictions...")
        xgb_oof[val_idx] = model.predict(X_xgb.iloc[val_idx])
        xgb_preds += model.predict(X_test_xgb) / kf.n_splits
        
        # Calculate fold score using RMSLE
        fold_score = np.sqrt(mean_squared_log_error(
            np.expm1(y.iloc[val_idx]), 
            np.expm1(xgb_oof[val_idx])
        ))
        
        # Store fold score
        xgb_scores.append(fold_score)
        
        # Calculate fold timing
        fold_time = time.time() - fold_start_time
        
        logger.info(f"Fold {fold} completed:")
        logger.info(f"  - RMSLE Score: {fold_score:.5f}")
        logger.info(f"  - Time taken: {fold_time:.2f} seconds")
        print(f"\nFold {fold} completed:")
        print(f"  - RMSLE Score: {fold_score:.5f}")
        print(f"  - Time taken: {fold_time:.2f} seconds")
        
        # Estimate remaining time
        if fold < total_folds:
            avg_fold_time = (time.time() - start_time) / fold
            remaining_folds = total_folds - fold
            estimated_time = avg_fold_time * remaining_folds
            logger.info(f"Estimated time remaining: {estimated_time/60:.1f} minutes")
            print(f"Estimated time remaining: {estimated_time/60:.1f} minutes")
    
    # Calculate and log final metrics
    total_time = time.time() - start_time
    mean_score = np.mean(xgb_scores)
    std_score = np.std(xgb_scores)
    
    logger.info("\nXGBoost Training Summary:")
    logger.info(f"  - Mean RMSLE: {mean_score:.5f} ± {std_score:.5f}")
    logger.info(f"  - Total training time: {total_time/60:.1f} minutes")
    logger.info(f"  - Average fold time: {total_time/total_folds:.1f} seconds")
    print("\n=== XGBoost Training Summary ===")
    print(f"  - Mean RMSLE: {mean_score:.5f} ± {std_score:.5f}")
    print(f"  - Total training time: {total_time/60:.1f} minutes")
    print(f"  - Average fold time: {total_time/total_folds:.1f} seconds")
    
    return xgb_preds, xgb_oof, xgb_scores, model, total_time


xgb_preds, xgb_oof, xgb_scores, xgb_model, xgb_total_time = train_xgboost(X, y, X_test)


xgb_params = {
        'max_depth': 9,
        'colsample_bytree': 0.7,
        'subsample': 0.9,
        'n_estimators': 3000,
        'learning_rate': 0.01,
        'gamma': 0.01,
        'max_delta_step': 2,
        'eval_metric': 'rmse',
        'enable_categorical': False,
        'random_state': 42,
        'early_stopping_rounds': 100,
        'tree_method': 'gpu_hist',
        'n_jobs': -1,  # Use all available CPU cores
        'gpu_id': 0,   # Use first GPU
        'predictor': 'gpu_predictor',
        'sampling_method': 'gradient_based',
        'max_bin': 256,
        'grow_policy': 'lossguide'
    }
xgb_metrics = {
        'train_rmsle': np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(xgb_oof))),
        'val_rmsle': np.mean(xgb_scores),
        'test_rmsle': None,  # We don't have true test values
        'improvement': 0,  # Will be updated in future runs
        'training_time': xgb_total_time,
        'model_specs': xgb_params
    }


# save_model_and_metrics(xgb_model, 'xgboost', xgb_metrics)


# Create submission with CAT-Boost Baseline
final_preds = np.expm1(xgb_preds) 
submission_df['Calories'] = np.clip(final_preds, 1, 314)
submission_df.to_csv('submission_XGBoost_Baseline2.csv', index=False)





def train_xgboost_parallel(X, y, X_test):
    """
    Train XGBoost model with parallel cross-validation across multiple GPUs.
    
    Args:
        X (pd.DataFrame): Training features
        y (pd.Series): Target variable
        X_test (pd.DataFrame): Test features
        
    Returns:
        tuple: (predictions, out-of-fold predictions, scores, model, total_time)
    """
    logger.info("Starting parallel XGBoost training...")
    print("\n=== Starting Parallel XGBoost Training ===")
    start_time = time.time()
    
    # Prepare data
    logger.info("Preparing data for XGBoost...")
    print("Preparing data for XGBoost...")
    
    # Ensure feature alignment
    logger.info("Aligning features between train and test...")
    print("Aligning features between train and test...")
    common_features = list(set(X.columns) & set(X_test.columns))
    X_xgb = X[common_features].copy()
    X_test_xgb = X_test[common_features].copy()
    
    # Convert categorical features
    X_xgb['Sex'] = X_xgb['Sex'].astype(int)
    X_test_xgb['Sex'] = X_test_xgb['Sex'].astype(int)

    # Initialize prediction arrays
    xgb_oof = np.zeros(len(X))
    xgb_preds = np.zeros(len(X_test))
    xgb_scores = []

    # Base XGBoost parameters
    base_params = {
        'max_depth': 9,
        'colsample_bytree': 0.7,
        'subsample': 0.9,
        'n_estimators': 3000,
        'learning_rate': 0.01,
        'gamma': 0.01,
        'max_delta_step': 2,
        'eval_metric': 'rmse',
        'enable_categorical': False,
        'random_state': 42,
        'early_stopping_rounds': 100,
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'sampling_method': 'gradient_based',
        'max_bin': 256,
        'grow_policy': 'lossguide'
    }

    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    total_folds = kf.n_splits
    
    logger.info(f"Starting {total_folds}-fold cross-validation in parallel...")
    print(f"\nStarting {total_folds}-fold cross-validation in parallel...")
    logger.info(f"Training with {len(common_features)} features: {', '.join(common_features)}")
    print(f"Training with {len(common_features)} features")

    def train_fold(fold_idx, train_idx, val_idx, gpu_id):
        """Train a single fold on specified GPU"""
        fold_start_time = time.time()
        logger.info(f"\nFold {fold_idx}/{total_folds} - Starting training on GPU {gpu_id}...")
        print(f"\nFold {fold_idx}/{total_folds} - Starting training on GPU {gpu_id}...")
        
        # Create fold-specific parameters
        fold_params = base_params.copy()
        fold_params['gpu_id'] = gpu_id
        
        # Create and train model
        model = XGBRegressor(**fold_params)
        
        # Train model with progress logging
        model.fit(
            X_xgb.iloc[train_idx], 
            y.iloc[train_idx],
            eval_set=[(X_xgb.iloc[val_idx], y.iloc[val_idx])],
            verbose=100
        )
        
        # Make predictions
        logger.info(f"Fold {fold_idx} - Making predictions...")
        print(f"Fold {fold_idx} - Making predictions...")
        fold_oof = model.predict(X_xgb.iloc[val_idx])
        fold_test_preds = model.predict(X_test_xgb)
        
        # Calculate fold score
        fold_score = np.sqrt(mean_squared_log_error(
            np.expm1(y.iloc[val_idx]), 
            np.expm1(fold_oof)
        ))
        
        # Calculate fold timing
        fold_time = time.time() - fold_start_time
        
        logger.info(f"Fold {fold_idx} completed on GPU {gpu_id}:")
        logger.info(f"  - RMSLE Score: {fold_score:.5f}")
        logger.info(f"  - Time taken: {fold_time:.2f} seconds")
        print(f"\nFold {fold_idx} completed on GPU {gpu_id}:")
        print(f"  - RMSLE Score: {fold_score:.5f}")
        print(f"  - Time taken: {fold_time:.2f} seconds")
        
        return fold_idx, fold_oof, fold_test_preds, fold_score, model, fold_time

    # Get number of available GPUs
    n_gpus = torch.cuda.device_count()
    logger.info(f"Found {n_gpus} GPUs")
    print(f"Found {n_gpus} GPUs")
    
    # Create process pool for parallel execution
    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        # Submit all folds for parallel execution
        future_to_fold = {
            executor.submit(
                train_fold, 
                fold_idx + 1, 
                train_idx, 
                val_idx, 
                fold_idx % n_gpus
            ): fold_idx 
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_xgb))
        }
        
        # Collect results as they complete
        fold_times = []
        for future in as_completed(future_to_fold):
            fold_idx, fold_oof, fold_test_preds, fold_score, model, fold_time = future.result()
            xgb_oof[kf.split(X_xgb)[fold_idx][1]] = fold_oof
            xgb_preds += fold_test_preds / total_folds
            xgb_scores.append(fold_score)
            fold_times.append(fold_time)
            
            # Estimate remaining time
            if len(fold_times) < total_folds:
                avg_fold_time = sum(fold_times) / len(fold_times)
                remaining_folds = total_folds - len(fold_times)
                estimated_time = avg_fold_time * remaining_folds
                logger.info(f"Estimated time remaining: {estimated_time/60:.1f} minutes")
                print(f"Estimated time remaining: {estimated_time/60:.1f} minutes")
    
    # Calculate and log final metrics
    total_time = time.time() - start_time
    mean_score = np.mean(xgb_scores)
    std_score = np.std(xgb_scores)
    
    logger.info("\nParallel XGBoost Training Summary:")
    logger.info(f"  - Mean RMSLE: {mean_score:.5f} ± {std_score:.5f}")
    logger.info(f"  - Total training time: {total_time/60:.1f} minutes")
    logger.info(f"  - Average fold time: {total_time/total_folds:.1f} seconds")
    print("\n=== Parallel XGBoost Training Summary ===")
    print(f"  - Mean RMSLE: {mean_score:.5f} ± {std_score:.5f}")
    print(f"  - Total training time: {total_time/60:.1f} minutes")
    print(f"  - Average fold time: {total_time/total_folds:.1f} seconds")
    
    return xgb_preds, xgb_oof, xgb_scores, model, total_time

def main():
    """Main execution function."""
    # ... existing code ...
    
    # Train models and save them
    cat_preds, cat_oof, cat_scores, cat_model, cat_total_time, cat_params = train_catboost(X, y, X_test)
    cat_metrics = {
        'train_rmsle': np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(cat_oof))),
        'val_rmsle': np.mean(cat_scores),
        'test_rmsle': None,  # We don't have true test values
        'improvement': 0,  # Will be updated in future runs
        'training_time': cat_total_time,
        'model_specs': cat_params
    }
    save_model_and_metrics(cat_model, 'catboost', cat_metrics)
    
    # Use parallel XGBoost training instead of sequential
    xgb_preds, xgb_oof, xgb_scores, xgb_model, xgb_total_time = train_xgboost_parallel(X, y, X_test)
    xgb_metrics = {
        'train_rmsle': np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(xgb_oof))),
        'val_rmsle': np.mean(xgb_scores),
        'test_rmsle': None,  # We don't have true test values
        'improvement': 0,  # Will be updated in future runs
        'training_time': xgb_total_time,
        'model_specs': base_params
    }
    save_model_and_metrics(xgb_model, 'xgboost', xgb_metrics)





def load_latest_model(model_name, model_dir='saved_models'):
    """
    Load the latest saved model and its metrics.
    
    Args:
        model_name (str): Name of the model to load
        model_dir (str): Directory containing saved models
        
    Returns:
        tuple: (model, metrics, model_path, metrics_path)
    """
    # Get all files for the model
    model_files = [f for f in os.listdir(model_dir) if f.startswith(model_name) and f.endswith('.joblib')]
    if not model_files:
        return None, None, None, None
    
    # Get latest model file
    latest_model_file = sorted(model_files)[-1]
    model_path = os.path.join(model_dir, latest_model_file)
    
    # Get corresponding metrics file
    metrics_file = latest_model_file.replace('.joblib', '_metrics.joblib')
    metrics_path = os.path.join(model_dir, metrics_file)
    
    # Load model and metrics
    model = joblib.load(model_path)
    metrics = joblib.load(metrics_path)
    
    logger.info(f"Loaded model from {model_path}")
    print(f"Loaded model from {model_path}")
    
    return model, metrics, model_path, metrics_path

def create_model_comparison_df(model_dir='saved_models'):
    """
    Create a DataFrame comparing all saved models.
    
    Args:
        model_dir (str): Directory containing saved models
        
    Returns:
        pd.DataFrame: Comparison DataFrame
    """
    comparison_data = []
    
    # Get all metrics files
    metrics_files = [f for f in os.listdir(model_dir) if f.endswith('_metrics.joblib')]
    
    for metrics_file in metrics_files:
        metrics_path = os.path.join(model_dir, metrics_file)
        metrics = joblib.load(metrics_path)
        
        # Extract model name and timestamp
        model_name = metrics_file.split('_')[0]
        timestamp = '_'.join(metrics_file.split('_')[1:-1])
        
        # Add to comparison data
        comparison_data.append({
            'model_name': model_name,
            'timestamp': timestamp,
            **metrics
        })
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df['timestamp'] = pd.to_datetime(comparison_df['timestamp'], format='%Y%m%d_%H%M%S')
    comparison_df = comparison_df.sort_values('timestamp', ascending=False)
    
    return comparison_df

def warm_start_training(X, y, X_test, model_name, model_dir='saved_models'):
    """
    Warm start training using the latest saved model.
    
    Args:
        X (pd.DataFrame): Training features
        y (pd.Series): Target variable
        X_test (pd.DataFrame): Test features
        model_name (str): Name of the model to warm start
        model_dir (str): Directory containing saved models
        
    Returns:
        tuple: (predictions, out-of-fold predictions, metrics)
    """
    # Load latest model
    model, old_metrics, model_path, metrics_path = load_latest_model(model_name, model_dir)
    
    if model is None:
        logger.info(f"No saved model found for {model_name}, starting fresh training")
        print(f"No saved model found for {model_name}, starting fresh training")
        return None, None, None
    
    logger.info(f"Warm starting {model_name} from {model_path}")
    print(f"Warm starting {model_name} from {model_path}")
    
    # Train model based on type
    if model_name == 'catboost':
        preds, oof = train_catboost(X, y, X_test, warm_start_model=model)
    elif model_name == 'xgboost':
        preds, oof = train_xgboost(X, y, X_test, warm_start_model=model)
    elif model_name == 'neural_net':
        preds, oof = train_neural_network(X, y, X_test, warm_start_model=model)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Calculate metrics
    metrics = {
        'train_rmsle': old_metrics.get('train_rmsle', 0),
        'val_rmsle': old_metrics.get('val_rmsle', 0),
        'test_rmsle': old_metrics.get('test_rmsle', 0),
        'improvement': old_metrics.get('improvement', 0),
        'training_time': old_metrics.get('training_time', 0),
        'model_specs': old_metrics.get('model_specs', {})
    }
    
    return preds, oof, metrics

def update_model_comparison(model_name, new_metrics, model_dir='saved_models'):
    """
    Update the model comparison DataFrame with new metrics.
    
    Args:
        model_name (str): Name of the model
        new_metrics (dict): New metrics to add
        model_dir (str): Directory containing saved models
        
    Returns:
        pd.DataFrame: Updated comparison DataFrame
    """
    # Load existing comparison
    comparison_df = create_model_comparison_df(model_dir)
    
    # Add new metrics
    new_row = {
        'model_name': model_name,
        'timestamp': datetime.now(),
        **new_metrics
    }
    
    comparison_df = pd.concat([pd.DataFrame([new_row]), comparison_df], ignore_index=True)
    
    # Save updated comparison
    comparison_path = os.path.join(model_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    
    return comparison_df


# # Create and display model comparison
# comparison_df = create_model_comparison_df()
# logger.info("\nModel Comparison:")
# logger.info(comparison_df)
# print("\nModel Comparison:")
# comparison_df


# Create submission with all three models
final_preds = 0.5 * np.expm1(xgb_preds) + 0.5 * np.expm1(cat_preds)
submission_df['Calories'] = np.clip(final_preds, 1, 314)
submission_df.to_csv('submission.csv', index=False)


def plot_predictions(cat_oof, xgb_oof):
    """
    Plot prediction distributions.
    
    Args:
        cat_oof (np.array): CatBoost out-of-fold predictions
        xgb_oof (np.array): XGBoost out-of-fold predictions
    """
    plt.figure(figsize=(15, 10))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Distribution of predictions
    ax1.hist(np.expm1(cat_oof), bins=50, alpha=0.4, label='CatBoost OOF', density=True)
    ax1.hist(np.expm1(xgb_oof), bins=50, alpha=0.4, label='XGBoost OOF', density=True)
    ax1.set_title("OOF Prediction Distribution")
    ax1.set_xlabel("Calories")
    ax1.set_ylabel("Density")
    ax1.legend()
    
    # Plot 2: Scatter plot of predictions
    ax2.scatter(np.expm1(cat_oof), np.expm1(xgb_oof), alpha=0.5, label='CatBoost vs XGBoost')
    
    # Add diagonal line
    min_val = min(np.min(np.expm1(cat_oof)), np.min(np.expm1(xgb_oof)))
    max_val = max(np.max(np.expm1(cat_oof)), np.max(np.expm1(xgb_oof)))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Correlation')
    
    ax2.set_title("Model Predictions Comparison")
    ax2.set_xlabel("Predictions from Model 1")
    ax2.set_ylabel("Predictions from Model 2")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


# Plot predictions
plot_predictions(cat_oof, xgb_oof)
    
# Print final statistics
logger.info("\nFinal Submission Preview:")
logger.info(submission_df.describe())
logger.info("\nFirst few predictions:")
logger.info(submission_df.head())

