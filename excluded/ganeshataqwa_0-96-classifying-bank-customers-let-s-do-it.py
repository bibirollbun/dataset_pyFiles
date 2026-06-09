# Basic Libraries
import numpy as np
import pandas as pd

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff

# Utility
import math
import warnings
warnings.filterwarnings('ignore')

# Preprocessing & Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

# Model Evaluation
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Models
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Optimization
import optuna


# Import Data
train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

SEED = 42
LABEL = 'y'


train.dtypes


train.head()


def check_categorical_values(df):
    """
    This function prints out all unique values for each categorical feature (object type) in the dataframe.
    It helps to check if there are any unexpected or invalid categories present.
    """
    # Select categorical columns (dtype 'object')
    categorical_columns = df.select_dtypes(include='object').columns

    # Iterate and print unique values for each categorical column
    for col in categorical_columns:
        print(f"Feature: {col}")
        print(f"Unique Categories ({df[col].nunique()}): {df[col].unique()}")
        print("-" * 50)


check_categorical_values(train)
print()
check_categorical_values(test)


def clean_categorical_values(df):
    """
    Clean specific known inconsistencies in categorical columns, such as removing periods from 'admin.'.
    This can be extended with more replacements as needed.
    """
    replacements = {
        'job': {
            'admin.': 'admin'
        }
        # Add more columns and mappings here if needed
    }

    for col, mapping in replacements.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)

    return df


# Apply the cleaning function
train = clean_categorical_values(train)
test = clean_categorical_values(test)

# Optional: Check again the unique values
check_categorical_values(train)
check_categorical_values(test)


def check_missing_values(df):
    """
    Check and display the number and percentage of missing values in each column of the dataframe.
    """
    total_missing = df.isnull().sum()
    percent_missing = (total_missing / len(df)) * 100

    missing_df = pd.DataFrame({
        'Missing Values': total_missing,
        'Percentage (%)': percent_missing
    })

    # Only show columns with at least 1 missing value
    missing_df = missing_df[missing_df['Missing Values'] > 0]

    if missing_df.empty:
        print("No missing values found in the dataframe.")
    else:
        print("Missing values per column:")
        print(missing_df.sort_values(by='Percentage (%)', ascending=False))


check_missing_values(train)
check_missing_values(test)


def add_is_first_contact(df):
    """
    Add 'is_first_contact' feature.

    Parameters:
    df (pd.DataFrame): The input dataframe containing the 'pdays' column.

    Returns:
    pd.DataFrame: DataFrame with the new 'is_first_contact' feature.
    """
    df['is_first_contact'] = np.where(df['pdays'] == -1, 'yes', 'no')
    return df


train = add_is_first_contact(train)
test = add_is_first_contact(test)


def add_contact_ratio(df):
    """
    Add 'contact_ratio' feature.

    Parameters:
    df (pd.DataFrame): The input dataframe containing 'campaign' and 'previous' columns.

    Returns:
    pd.DataFrame: DataFrame with the new 'contact_ratio' feature.
    """
    df['contact_ratio'] = df['campaign'] / (df['previous'] + 1)
    return df


train = add_contact_ratio(train)
test = add_contact_ratio(test)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(train['contact_ratio'], bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of Contact Ratio (Train)')
plt.xlabel('contact_ratio')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(test['contact_ratio'], bins=30, color='salmon', edgecolor='black')
plt.title('Histogram of Contact Ratio (Test)')
plt.xlabel('contact_ratio')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


def add_log_contact_ratio(df):
    """
    Add a 'log_contact_ratio' feature.

    Requirements:
    - 'contact_ratio' must already exist in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame with 'contact_ratio'.

    Returns:
    pd.DataFrame: DataFrame with a new column 'log_contact_ratio'.
    """
    df['log_contact_ratio'] = np.log1p(df['contact_ratio'])
    return df


train = add_log_contact_ratio(train)
test = add_log_contact_ratio(test)


# Drop 'contact_ratio' from both train and test
train.drop(columns=['contact_ratio'], inplace=True)
test.drop(columns=['contact_ratio'], inplace=True)


def add_economic_stability(df):
    """
    Add a new feature 'economic_stability' calculated as balance / age.

    Parameters:
    df (pd.DataFrame): Input DataFrame with 'balance' and 'age' columns.

    Returns:
    pd.DataFrame: DataFrame with the new 'economic_stability' column.
    """
    df['economic_stability'] = df['balance'] / df['age']
    return df


train = add_economic_stability(train)
test = add_economic_stability(test)


numeric = [
    'age', 'balance', 'day', 'duration',
    'campaign', 'pdays', 'previous',
    'log_contact_ratio', 'economic_stability'
]

categorical = [
    'job', 'marital', 'education', 'default', 'housing', 'loan',
    'contact', 'month', 'poutcome', 'is_first_contact'
]


def plot_category_distribution(df, target_col, categorical_cols):
    """
    Create bar plots showing the percentage distribution of a target variable across categorical features.

    Parameters:
    - df: pandas DataFrame containing the data
    - target_col: string, the name of the target column
    - categorical_cols: list of strings, names of the categorical columns to plot
    """
    for col in categorical_cols:
        plt.figure(figsize=(14, 5))
        
        # Calculate the percentage of each target class within each category of the feature
        percentage_df = (
            df.groupby(col)[target_col]
            .value_counts(normalize=True)  # Normalize to get proportions instead of raw counts
            .rename("percentage")
            .reset_index()
        )
        
        # Create the bar plot
        sns.barplot(
            data=percentage_df,
            x=col,
            y="percentage",
            hue=target_col,
            palette="coolwarm"
        )
        plt.title(f"Percentage Distribution of {target_col} across {col}")
        plt.xticks(rotation=90)  # Rotate x-axis labels for readability
        plt.ylabel("Percentage")
        plt.tight_layout()  # Adjust layout to prevent clipping
        plt.show()


plot_category_distribution(train, target_col=LABEL, categorical_cols=categorical)


def plot_category_pie_distribution(df, target_col, categorical_cols):
    """
    Create pie charts showing the percentage distribution of the target variable within each category
    of the given categorical features.

    Parameters:
    - df: pandas DataFrame
    - target_col: string, name of the target variable
    - categorical_cols: list of strings, names of categorical columns to analyze
    """
    for col in categorical_cols:
        unique_categories = df[col].dropna().unique()

        # Set up the plot grid
        n = len(unique_categories)
        cols = 3
        rows = (n + cols - 1) // cols
        plt.figure(figsize=(5 * cols, 5 * rows))
        
        for i, category in enumerate(sorted(unique_categories), 1):
            plt.subplot(rows, cols, i)

            # Filter data for that category
            data_slice = df[df[col] == category][target_col]

            # Count value distribution
            value_counts = data_slice.value_counts(normalize=True)
            
            # Plot pie chart
            plt.pie(
                value_counts,
                labels=value_counts.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=plt.cm.coolwarm([0.2, 0.8])[:len(value_counts)]
            )
            plt.title(f"{col}: {category}")
            plt.axis('equal')  # Equal aspect ratio ensures a circular pie chart
        
        plt.suptitle(f"Target Distribution by {col}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


plot_category_pie_distribution(train, target_col=LABEL, categorical_cols=categorical)


def add_is_high_conversion_month(df):
    """
    Add 'is_high_conversion_month' feature as 'yes' or 'no'.
    
    Parameters:
    df (pd.DataFrame): The input dataframe containing the 'month' column.
    
    Returns:
    pd.DataFrame: DataFrame with the new 'is_high_conversion_month' feature.
    """
    high_months = ['mar', 'oct', 'sep', 'dec']
    df['is_high_conversion_month'] = np.where(df['month'].isin(high_months), 'yes', 'no')
    return df


train = add_is_high_conversion_month(train)
test = add_is_high_conversion_month(test)


def plot_numeric_distribution(df, target_col, numeric_cols, plot_type='hist'):
    """
    Visualize the distribution of numeric features split by the target variable.

    Parameters:
    - df: pandas DataFrame
    - target_col: string, name of the target column (categorical/binary)
    - numeric_cols: list of strings, names of numeric columns
    - plot_type: 'hist' for histograms or 'box' for boxplots
    """
    for col in numeric_cols:
        plt.figure(figsize=(10, 5))
        
        if plot_type == 'hist':
            # Plot histogram by hue (target class)
            sns.histplot(
                data=df,
                x=col,
                hue=target_col,
                kde=True,
                element='step',
                stat='density',
                common_norm=False,
                palette='coolwarm'
            )
            plt.title(f"Histogram of {col} by {target_col}")
            plt.xlabel(col)
            plt.ylabel("Density")
        
        elif plot_type == 'box':
            # Plot boxplot split by target class
            sns.boxplot(
                data=df,
                x=target_col,
                y=col,
                palette='coolwarm'
            )
            plt.title(f"Boxplot of {col} by {target_col}")
            plt.xlabel(target_col)
            plt.ylabel(col)
        
        else:
            raise ValueError("plot_type must be either 'hist' or 'box'")
        
        plt.tight_layout()
        plt.show()


plot_numeric_distribution(train, target_col='y', numeric_cols=numeric, plot_type='hist')


plot_numeric_distribution(train, target_col='y', numeric_cols=numeric, plot_type='box')


def add_is_short_call(df, threshold=150):
    """
    Add 'is_short_call' feature.
    
    Parameters:
    df (pd.DataFrame): The input dataframe containing the 'duration' column.
    threshold (int): Duration threshold in seconds to define a short call.
    
    Returns:
    pd.DataFrame: DataFrame with the new 'is_short_call' feature.
    """
    df['is_short_call'] = np.where(df['duration'] <= threshold, 'yes', 'no')
    return df


train = add_is_short_call(train, threshold=150)
test = add_is_short_call(test, threshold=150)


train.head()


train.dtypes


check_missing_values(train)
check_missing_values(test)


kategori_nominal = ['job', 'marital', 'contact', 'poutcome', 'education']
kategori_ordinal = ['month']
kategori_boolean = ['default', 'housing', 'loan', 'is_high_conversion_month', 'is_first_contact', 'is_short_call']
numerik = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous', 'economic_stability', 'log_contact_ratio']

nominal_transformer = Pipeline(steps=[
    ('encoding', OneHotEncoder(handle_unknown='ignore'))
])

month_mapping = [['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                     'jul', 'aug', 'sep', 'oct', 'nov', 'dec']]

ordinal_transformer = Pipeline(steps=[
    ('encoding', OrdinalEncoder(categories=month_mapping))
])

def yes_no_to_binary(X):
    return np.where(X == 'yes', 1, 0)

boolean_transformer = Pipeline(steps=[
    ('binary', FunctionTransformer(yes_no_to_binary))
])

numerik_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
])

preprocessor = ColumnTransformer(
    transformers=[
        ('numerik', numerik_transformer, numerik),
        ('nominal', nominal_transformer, kategori_nominal),
        ('ordinal', ordinal_transformer, kategori_ordinal),
        ('boolean', boolean_transformer, kategori_boolean)
    ]
)


X_train = train.drop(columns=['id', LABEL])
y_train = train[LABEL]
X_test = test.drop(columns=['id'])


# Use the best parameters from LightGBM tuning (Trial 14)
best_model_name = 'LightGBM'
best_params = {
    'n_estimators': 328,
    'learning_rate': 0.1056388664124682,
    'max_depth': 9,
    'num_leaves': 135,
    'subsample': 0.8990360120946332,
    'colsample_bytree': 0.7494223825464681,
    'random_state': 42
}

print(f"\nğŸ�† Best model: {best_model_name} with ROC AUC = {0.9678881876084325:.4f}")

# Initialize the final model with the best parameters
final_model = LGBMClassifier(**best_params)

# Create the final pipeline including preprocessing and model
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', final_model)
])

# Fit the model to the training data and predict probabilities on the test set
final_pipeline.fit(X_train, y_train)
y_test_proba = final_pipeline.predict_proba(X_test)


submission = pd.DataFrame({
    'id': test['id'],
    LABEL: y_test_proba[:, 1]
})

submission.to_csv('submission.csv', index=False)
print("âœ… The file submission.csv has been saved successfully.")

