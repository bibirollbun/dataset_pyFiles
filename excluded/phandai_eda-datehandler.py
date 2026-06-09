#install scikit learn 1.5.2 as this version supports root_mean_squared_log_error
!pip uninstall scikit-learn -y
!pip install -q scikit-learn==1.5.2


import sklearn
sklearn.__version__


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_log_error,mean_squared_error, mean_absolute_error, r2_score

import optuna
import lightgbm as lgb

import torch
from sklearn.pipeline import Pipeline


train_df=pd.read_csv('/kaggle/input/big-oai-final-course-1/train.csv')
test_df=pd.read_csv('/kaggle/input/big-oai-final-course-1/test.csv')


# Check dataset shape and first rows
print(f"Dataset contains {train_df.shape[0]} rows and {train_df.shape[1]} columns.")
train_df.head()


train_df.info()


train_df.describe().round(2)


# Save 'id' column for submission
test_ids = test_df['id']

# Define the target column
target_column = 'Premium Amount'

# Select categorical and numerical columns (initial)
categorical_columns = train_df.select_dtypes(include=['object']).columns
numerical_columns = train_df.select_dtypes(exclude=['object']).columns

# Print out column information
print("Target Column:", target_column)
print("\nCategorical Columns:", categorical_columns.tolist())
print("\nNumerical Columns:", numerical_columns.tolist())


for column in categorical_columns:
    num_unique = train_df[column].nunique()
    print(f"'{column}' has {num_unique} unique categories.")


plt.figure(figsize=(15,9))
plt.title("Visualizing Missing Values")
sns.heatmap(train_df.isnull(), cbar=False, cmap=sns.color_palette('magma'), yticklabels=False);
plt.show()


# Function to calculate missing values, percentages, and data types
def missing_values_table(df):
    missing_count = df.isnull().sum()
    missing_percentage = 100 * missing_count / len(df)
    data_types = df.dtypes
    return pd.DataFrame({
        'Missing Values': missing_count,
        'Percentage (%)': missing_percentage,
        'Data Type': data_types
    })

# Create tables for train and test datasets
train_missing_table = missing_values_table(train_df)
test_missing_table = missing_values_table(test_df)

# Display the tables
print("Missing Values Table - Training Dataset:\n")
display(train_missing_table[train_missing_table['Missing Values'] > 0])  # Display only features with missing values
print("\n")

print("Missing Values Table - Test Dataset:\n")
display(test_missing_table[test_missing_table['Missing Values'] > 0])


# Create a color palette for the columns
palette = sns.color_palette('tab10', len(numerical_columns))
color_dict = dict(zip(numerical_columns, palette))

# Create a grid of subplots for histograms and boxplots only
fig = plt.figure(figsize=(15, 10 * len(numerical_columns)))  # Adjusted width since only one column of plots
gs = gridspec.GridSpec(2 * len(numerical_columns), 1, figure=fig)  # Single column grid

for i, column in enumerate(numerical_columns):
    if(column=="id"): continue
    if train_df[column].nunique() > 50:
        discrete = False
    else:
        discrete = True
    
    # Plot histogram with a unique color
    ax_hist = fig.add_subplot(gs[2 * i, 0])
    sns.histplot(
        data=train_df, x=column, fill=True, common_norm=False, alpha=0.6,
        linewidth=0.8, color=color_dict[column], ax=ax_hist, discrete=discrete
    )
    
    # Plot boxplot with the same unique color
    ax_box = fig.add_subplot(gs[2 * i + 1, 0])
    sns.boxplot(data=train_df, x=column, ax=ax_box, color=color_dict[column])
    ax_box.set_title(f'{column} (Boxplot)', fontsize=14)
    sns.despine(ax=ax_box)

plt.tight_layout()  # Adjust subplots to fit into the figure area
plt.show()


# Filtrer les colonnes catégorielles et exclure 'Policy Start Date'
filtered_columns = [col for col in categorical_columns if col != 'Policy Start Date']

# Créer des sous-graphiques pour barplots et boxplots
fig, axes = plt.subplots(len(filtered_columns), 2, figsize=(15, 5 * len(filtered_columns)))

for i, column in enumerate(filtered_columns):
    # Barplot à gauche
    sns.countplot(data=train_df, x=column, ax=axes[i, 0], palette='tab10')
    axes[i, 0].set_title(f'Distribution of {column}', fontsize=14)
    axes[i, 0].set_xlabel(column, fontsize=12)
    axes[i, 0].set_ylabel('Count', fontsize=12)
    sns.despine(ax=axes[i, 0])

    # Boxplot à droite
    sns.boxplot(data=train_df, x=column, y=target_column, ax=axes[i, 1], palette='tab10')
    axes[i, 1].set_title(f'{column} vs {target_column}', fontsize=14)
    axes[i, 1].set_xlabel(column, fontsize=12)
    axes[i, 1].set_ylabel(target_column, fontsize=12)
    sns.despine(ax=axes[i, 1])


plt.tight_layout()  # Ajustement global des sous-graphiques
plt.show()


def date(df):
    df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'])
    df['Year'] = df['Policy Start Date'].dt.year
    df['Day'] = df['Policy Start Date'].dt.day
    df['Month'] = df['Policy Start Date'].dt.month
    df['Month_name'] = df['Policy Start Date'].dt.month_name()
    df['Day_of_week'] = df['Policy Start Date'].dt.day_name()
    df['Week'] = df['Policy Start Date'].dt.isocalendar().week
    df['Year_sin'] = np.sin(2 * np.pi * df['Year'])
    df['Year_cos'] = np.cos(2 * np.pi * df['Year'])
    min_year = df['Year'].min()
    max_year = df['Year'].max()
    df['Year_sin'] = np.sin(2 * np.pi * (df['Year'] - min_year) / (max_year - min_year))
    df['Year_cos'] = np.cos(2 * np.pi * (df['Year'] - min_year) / (max_year - min_year))
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12) 
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)  
    df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)
    
    df.drop('Policy Start Date', axis=1, inplace=True)

    return df

# Apply the date function to both datasets
train_df = date(train_df)
test_df = date(test_df)


# Define features and target
numerical_features = [
    'Age', 'Annual Income', 'Number of Dependents', 'Health Score', 
    'Previous Claims', 'Vehicle Age', 'Credit Score', 'Insurance Duration', 
    'Year_sin', 'Year_cos', 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos'
]
categorical_features = [
    'Gender', 'Marital Status', 'Education Level', 'Occupation', 'Location',
    'Policy Type', 'Customer Feedback', 'Smoking Status', 'Exercise Frequency', 
    'Property Type', 'Month_name', 'Day_of_week'
]
target_column = 'Premium Amount'


# Split train data into features and target
X = train_df.drop(columns=[target_column, 'id', 'Year', 'Month', 'Day', 'Week'])
y = train_df[target_column]


# Preprocessing pipeline for numerical features
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
])

# Preprocessing pipeline for categorical features
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),  # Handle missing values
    ('onehot', OneHotEncoder(handle_unknown='ignore'))                      # Encode categorical features
])

# Combine pipelines into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numerical_features),
        ('cat', cat_pipeline, categorical_features)
    ]
)

# Preprocess train and test data
X_processed = preprocessor.fit_transform(X)
test_processed = preprocessor.transform(test_df.drop(columns=['id', 'Year', 'Month', 'Day', 'Week']))


# Split the data
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Model Imports ---
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb


# --- 1. Linear Regression ---
print("--- Training Linear Regression ---")
# Instantiate the model
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)
print("Linear Regression training complete.")

# Predict on the validation set
y_pred_lr = lr_model.predict(X_val)
print("-" * 30)



import numpy as np
from sklearn.metrics import mean_squared_log_error

# Assumes y_val, y_pred_lr, y_pred_ridge, y_pred_lasso are already defined.

def calculate_rmsle(y_true, y_pred):
    """Calculates RMSLE, clipping negative predictions to 0."""
    # Ensure no negative true values (RMSLE undefined)
    if np.any(y_true < 0):
        print(f"Warning: Negative true values found. RMSLE calculation may fail or be invalid.")
        # Optionally return NaN or raise error, here we proceed but warn.

    # Clip negative predictions to 0
    y_pred_clipped = np.maximum(y_pred, 0)

    msle = mean_squared_log_error(y_true, y_pred_clipped)
    rmsle = np.sqrt(msle)
    return rmsle

print("--- Evaluating Models using RMSLE ---")

# Evaluate Linear Regression
rmsle_lr = calculate_rmsle(y_val, y_pred_lr)
print(f"Linear Regression RMSLE: {rmsle_lr:.4f}")


sample_submission = pd.read_csv("/kaggle/input/big-oai-final-course-1/sample_submission.csv")
sample_submission.head()


# Predict
y_pred = lr_model.predict(test_processed)


submission_df = pd.DataFrame()

if 'id' in test_df.columns:
    submission_df['id'] = test_df['id'].values
else:
    submission_df['id'] = sample_submission['id'].values

target_column_name = sample_submission.columns[1] # Assumes target is the second column
submission_df[target_column_name] = y_pred

# Save
submission_filename = 'submission.csv'
submission_df.to_csv(submission_filename, index=False)

print(f"Submission file '{submission_filename}' created.")
print(submission_df.head())

