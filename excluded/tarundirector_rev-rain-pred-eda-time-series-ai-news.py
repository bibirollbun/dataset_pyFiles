# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
from IPython.display import display, HTML
from io import BytesIO
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


url = "https://img.freepik.com/free-photo/beautiful-city-view_23-2151002674.jpg"

html_code = f'''
<img src="{url}" style="width:100%; height:300px; object-fit: cover; border-radius: 20px;">
'''
display(HTML(html_code))



#ğŸ”� Ah-ha! You found the secret sauce! ğŸ�”


# Importing Libraries

import warnings
warnings.filterwarnings("ignore")

import optuna
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, median_absolute_error
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import catboost as cb
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit


# Reading .csv data file
train_data = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
original_data = pd.read_csv('/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv')


train_data.sample(5)


test_data.sample(5)


original_data.sample(5)


original_data.columns


# Removing spaces at the start and end of column names
original_data.columns = original_data.columns.str.strip()


# Converting 'rainfall' column to binary format
original_data['rainfall'] = original_data['rainfall'].map({'yes': 1, 'no': 0})


# Checking the number of rows and columns

num_train_rows, num_train_columns = train_data.shape

num_test_rows, num_test_columns = test_data.shape

num_original_rows, num_original_columns = original_data.shape

print("Training Data:")
print(f"Number of Rows: {num_train_rows}")
print(f"Number of Columns: {num_train_columns}\n")

print("Test Data:")
print(f"Number of Rows: {num_test_rows}")
print(f"Number of Columns: {num_test_columns}\n")

print("Original Data:")
print(f"Number of Rows: {num_original_rows}")
print(f"Number of Columns: {num_original_columns}")


# Creating a table for missing values, unique values and data types of the features

missing_values_train = pd.DataFrame({'Feature': train_data.columns,
                              '[TRAIN] No. of Missing Values': train_data.isnull().sum().values,
                              '[TRAIN] % of Missing Values': ((train_data.isnull().sum().values)/len(train_data)*100)})

missing_values_test = pd.DataFrame({'Feature': test_data.columns,
                             '[TEST] No.of Missing Values': test_data.isnull().sum().values,
                             '[TEST] % of Missing Values': ((test_data.isnull().sum().values)/len(test_data)*100)})

missing_values_original = pd.DataFrame({'Feature': original_data.columns,
                             '[ORIGINAL] No.of Missing Values': original_data.isnull().sum().values,
                             '[ORIGINAL] % of Missing Values': ((original_data.isnull().sum().values)/len(original_data)*100)})

unique_values = pd.DataFrame({'Feature': train_data.columns,
                              'No. of Unique Values[FROM TRAIN]': train_data.nunique().values})

feature_types = pd.DataFrame({'Feature': train_data.columns,
                              'DataType': train_data.dtypes})

merged_df = pd.merge(missing_values_train, missing_values_test, on='Feature', how='left')
merged_df = pd.merge(merged_df, missing_values_original, on='Feature', how='left')
merged_df = pd.merge(merged_df, unique_values, on='Feature', how='left')
merged_df = pd.merge(merged_df, feature_types, on='Feature', how='left')

merged_df.style.background_gradient(cmap='viridis')


# Count duplicate rows in train_data
train_duplicates = train_data.duplicated().sum()

# Count duplicate rows in test_data
test_duplicates = test_data.duplicated().sum()

# Count duplicate rows in original_data
original_duplicates = original_data.duplicated().sum()

# Print the results
print(f"Number of duplicate rows in train_data: {train_duplicates}")
print(f"Number of duplicate rows in test_data: {test_duplicates}")
print(f"Number of duplicate rows in original_data: {original_duplicates}")


# Having a look at the description of all the numerical columns present in the dataset
print('Description of all the numerical columns present in the train dataset')
train_data.describe().T.style.background_gradient(cmap='viridis')


# Having a look at the description of all the numerical columns present in the dataset
print('Description of all the numerical columns present in the test dataset')
test_data.describe().T.style.background_gradient(cmap='viridis')


# Having a look at the description of all the numerical columns present in the dataset
print('Description of all the numerical columns present in the original dataset')
original_data.describe().T.style.background_gradient(cmap='viridis')


original_data['day'] = range(1, len(original_data) + 1)
original_data['day'].describe()


numerical_variables = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']
target_variable = 'rainfall' 
categorical_variables = ['winddirection']


# Analysis of all NUMERICAL features

# Define a custom color palette
custom_palette = ['#3498db', '#e74c3c','#2ecc71']

# Add 'Dataset' column to distinguish between train and test data
train_data['Dataset'] = 'Train'
test_data['Dataset'] = 'Test'
original_data['Dataset'] = 'Original'

# Function to create and display a row of plots for a single variable
def create_variable_plots(variable):
    sns.set_style('whitegrid')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Box plot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=pd.concat([train_data, test_data,original_data.dropna()]), x=variable, y="Dataset", palette=custom_palette)
    plt.xlabel(variable)
    plt.title(f"Box Plot for {variable}")

    # Separate Histograms
    plt.subplot(1, 2, 2)
    sns.histplot(data=train_data, x=variable, color=custom_palette[0], kde=True, bins=30, label="Train")
    sns.histplot(data=test_data, x=variable, color=custom_palette[1], kde=True, bins=30, label="Test")
    sns.histplot(data=original_data.dropna(), x=variable, color=custom_palette[2], kde=True, bins=30, label="Original")
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title(f"Histogram for {variable} [TRAIN, TEST & ORIGINAL]")
    plt.legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()

# Perform univariate analysis for each variable
for variable in numerical_variables:
    create_variable_plots(variable)

# Drop the 'Dataset' column after analysis
train_data.drop('Dataset', axis=1, inplace=True)
test_data.drop('Dataset', axis=1, inplace=True)
original_data.drop('Dataset', axis=1, inplace=True)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define colors for Train, Test, and Original data
custom_palette = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green

# Function to create and display a grouped count plot for a single categorical variable
def create_categorical_barplot(variable):
    sns.set_style('whitegrid')

    # Combine the datasets and create a new column indicating the source
    train_data_copy = train_data.copy()
    test_data_copy = test_data.copy()
    original_data_copy = original_data.dropna().copy()

    train_data_copy['Dataset'] = 'Train'
    test_data_copy['Dataset'] = 'Test'
    original_data_copy['Dataset'] = 'Original'

    combined_data = pd.concat([train_data_copy, test_data_copy, original_data_copy])

    # Get sorted order of categories based on Train data count (small to big)
    train_counts = train_data[variable].value_counts().sort_values(ascending=True).index.tolist()

    # Plot grouped countplot (Horizontal bars)
    plt.figure(figsize=(14, 7))
    sns.countplot(
        data=combined_data, 
        x=variable,  # Swapped axes
        hue="Dataset", 
        palette=custom_palette, 
        dodge=True,  # Ensures grouped bars
        width=0.85,  # Further increased bar width
        order=train_counts  # Sorting categories by Train data count (small to big)
    )

    plt.ylabel("Count")
    plt.xlabel(variable)
    plt.title(f"Grouped Count Plot for {variable} [TRAIN, TEST & ORIGINAL]")
    plt.legend(title="Dataset")

    # Rotate x labels for better visibility
    plt.xticks(rotation=45, ha="right")

    # Show the plot
    plt.show()

# Perform univariate analysis for each categorical variable
for variable in categorical_variables:
    create_categorical_barplot(variable)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define custom color palette for Train, Test, and Original datasets
custom_palette = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green

# Function to create Wind Rose plot in a subplot
def create_wind_rose(ax, data, dataset_name, color):
    # Convert wind direction to radians
    wind_direction_radians = np.radians(data['winddirection'].dropna())

    # Create histogram bins (every 10Â°)
    bins = np.linspace(0, 2*np.pi, 37)  # 36 bins (every 10Â°)
    counts, bin_edges = np.histogram(wind_direction_radians, bins=bins)

    # Plot on the polar axis with improved style
    bars = ax.bar(bin_edges[:-1], counts, width=np.radians(10), color=color, edgecolor='black', alpha=0.8)

    # Formatting for professional appearance
    ax.set_theta_zero_location("N")  # North is at 0Â°
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_xticks(np.radians(np.arange(0, 360, 45)))  # Tick labels every 45Â°
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=10, fontweight='bold')

    # Add grid and labels for better readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_yticklabels([])  # Remove radial labels to avoid clutter
    ax.set_title(f"Wind Direction ({dataset_name})", fontsize=12, fontweight='bold', pad=10)

# Create a single row with three wind rose plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': 'polar'})

# Generate wind rose plots for Train, Test, and Original datasets
create_wind_rose(axes[0], train_data, "Train Data", custom_palette[0])  # Blue
create_wind_rose(axes[1], test_data, "Test Data", custom_palette[1])    # Red
create_wind_rose(axes[2], original_data.dropna(), "Original Data", custom_palette[2])  # Green

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


pie_chart_palette = ['#33638d', '#28ae80', '#d3eb0c', '#ff9a0b', '#7e03a8', '#35b779', '#fde725', '#440154', '#90d743', '#482173', '#22a884', '#f8961e']

countplot_color = '#5C67A3'

# Function to create and display a row of plots for a single target variable
def create_target_plots(variable):
    sns.set_style('whitegrid')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Pie Chart
    plt.subplot(1, 2, 1)
    train_data[variable].value_counts().plot.pie(
        autopct='%1.1f%%', colors=pie_chart_palette, wedgeprops=dict(width=0.3), startangle=140
    )
    plt.title(f"Pie Chart for {variable}")

    # Bar Graph
    plt.subplot(1, 2, 2)
    sns.countplot(
        data=pd.concat([train_data, original_data.dropna()]), 
        x=variable, 
        color=countplot_color,  # Using a single color for the countplot
        alpha=0.8  # Setting 80% opacity
    )
    plt.xlabel(variable)
    plt.ylabel("Count")
    plt.title(f"Bar Graph for {variable} [TRAIN & ORIGINAL Combined]")

    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Show the plots
    plt.show()

# Perform univariate analysis for target variable
create_target_plots(target_variable)


variables = [col for col in train_data.columns if col in numerical_variables]+['day']

# Adding variables to the existing list
test_variables = variables
train_variables = variables+ ['rainfall']

# Calculate correlation matrices for train_data and test_data
corr_train = train_data[train_variables].corr()
corr_test = test_data[test_variables].corr()

# Create masks for the upper triangle
mask_train = np.triu(np.ones_like(corr_train, dtype=bool))
mask_test = np.triu(np.ones_like(corr_test, dtype=bool))

# Set the text size and rotation
annot_kws = {"size": 8, "rotation": 45}

# Generate heatmaps for train_data
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
ax_train = sns.heatmap(corr_train, mask=mask_train, cmap='viridis', annot=True,
                      square=True, linewidths=.5, xticklabels=1, yticklabels=1, annot_kws=annot_kws)
plt.title('Correlation Heatmap - Train Data')

# Generate heatmaps for test_data
plt.subplot(1, 2, 2)
ax_test = sns.heatmap(corr_test, mask=mask_test, cmap='viridis', annot=True,
                     square=True, linewidths=.5, xticklabels=1, yticklabels=1, annot_kws=annot_kws)
plt.title('Correlation Heatmap - Test Data')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


# Selecting numerical features + target variable
variables = [col for col in train_data.columns if col in numerical_variables]
train_variables = variables + ['rainfall']

# Compute correlation with 'rainfall' and transpose for horizontal display
corr_train = train_data[train_variables].corr()[['rainfall']].T  # Transpose for horizontal orientation

# Set the text size and rotation
annot_kws = {"size": 10}  # Increased size for better visibility

# Generate horizontal heatmap without color bar
plt.figure(figsize=(10, 2))  # Adjusted for a horizontal layout
ax_train = sns.heatmap(corr_train, cmap='viridis', annot=True, 
                      square=False, linewidths=0.5, annot_kws=annot_kws, 
                      cbar=False)  # **Removed color bar**

# Formatting
plt.xticks(rotation=45, ha="right")  # Rotate labels for readability
plt.title('Correlation Heatmap - Train Data (ONLY TARGET)')
plt.yticks(rotation=0)  # Keep y-labels horizontal

# Show plot
plt.show()


# Define colors for Train and Test data
train_color = '#3498db'  # Blue
test_color = '#e74c3c'   # Red

# Create the plot
plt.figure(figsize=(12, 5))

# Plot Train Data
plt.plot(train_data['id'], train_data['day'], linestyle='-', color=train_color, label='Train Data', alpha=0.7)

# Plot Test Data
plt.plot(test_data['id'], test_data['day'], linestyle='-', color=test_color, label='Test Data', alpha=0.7)

# Formatting
plt.xlabel('ID')
plt.ylabel('Day')
plt.title('Trend Plot: Day vs ID')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Show plot
plt.show()


# Generate the expected repeating pattern (1-365 for 6 years)
expected_pattern = np.tile(np.arange(1, 366), 6)  # Repeats 1-365 exactly 6 times

# Check for incorrect labels
train_data['expected_day'] = expected_pattern[:len(train_data)]  # Assign expected pattern
train_data['day_mismatch'] = train_data['day'] != train_data['expected_day']  # Flag mismatches


flag_color = '#8B0000'   # Dark Red (for mismatched days)

# Generate expected repeating pattern (1-365 for 6 years)
expected_pattern = np.tile(np.arange(1, 366), 6)  # Repeats 1-365 exactly 6 times

# Assign expected pattern and flag mismatches
train_data['expected_day'] = expected_pattern[:len(train_data)]
train_data['day_mismatch'] = train_data['day'] != train_data['expected_day']  # Boolean flag

# Create the plot
plt.figure(figsize=(12, 5))

# Plot Train Data
plt.plot(train_data['id'], train_data['day'], linestyle='-', color=train_color, label='Train Data', alpha=0.7)

# Plot Test Data
plt.plot(test_data['id'], test_data['day'], linestyle='-', color=test_color, label='Test Data', alpha=0.7)

# Flag mismatched days using red markers
plt.scatter(
    train_data.loc[train_data['day_mismatch'], 'id'],  # X-axis: IDs of mismatched days
    train_data.loc[train_data['day_mismatch'], 'day'], # Y-axis: Corresponding incorrect days
    color=flag_color, marker='X', s=80, label='Mismatched Days', alpha=0.9
)

# Formatting
plt.xlabel('ID')
plt.ylabel('Day')
plt.title('Trend Plot: Day vs ID (Flagging Mismatches)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Show plot
plt.show()


train_data['day'] = train_data['expected_day']

# Get the last day value from train data
last_train_day = train_data['day'].iloc[-1]

# Generate sequential day numbers for the test dataset
test_data['day'] = np.arange(last_train_day + 1, last_train_day + 1 + len(test_data))

train_data.drop(columns=['expected_day', 'day_mismatch'], errors='ignore', inplace=True)  # Drop 'expected_day' if it exists


# Define colors
train_color = '#3498db'  # Blue
test_color = '#e74c3c'   # Red
rainfall_colors = {0: '#f1c40f', 1: '#2980b9'}  # Dark Yellow (no rainfall), Blue (rainfall)

# Numerical columns to plot
numerical_columns = test_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
for col in ['id', 'day', 'rainfall']:
    if col in numerical_columns:
        numerical_columns.remove(col)

# Plotting loop for each numerical variable
for column in numerical_columns:
    # Create figure with specific layout
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    # ---- Trend Plot (ID vs Variable) ----
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(train_data['id'], train_data[column], linestyle='-', color=train_color, label='Train Data', alpha=0.7)
    ax0.plot(test_data['id'], test_data[column], linestyle='-', color=test_color, label='Test Data', alpha=0.7)

    ax0.set_xlabel('ID', fontsize=14)
    ax0.set_ylabel(column, fontsize=14)
    ax0.set_title(f'Trend Plot: {column} vs ID', fontsize=16, fontweight='bold')  # âœ… Fix applied
    ax0.legend(fontsize=12)
    ax0.grid(True, linestyle='--', alpha=0.5)

    # ---- Scatter Plot (Day vs Variable) ----
    ax1 = fig.add_subplot(gs[1, 0])
    scatter = ax1.scatter(
        train_data['day'], train_data[column],
        c=train_data['rainfall'].map(rainfall_colors), alpha=0.7
    )
    ax1.set_xlabel('Day', fontsize=14)
    ax1.set_ylabel(column, fontsize=14)
    ax1.set_title(f'Scatter Plot: {column} vs Day (by Rainfall)', fontsize=16, fontweight='bold')  # âœ… Fix applied

    # Custom legend for rainfall
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='No Rainfall',
               markersize=10, markerfacecolor=rainfall_colors[0]),
        Line2D([0], [0], marker='o', color='w', label='Rainfall',
               markersize=10, markerfacecolor=rainfall_colors[1])
    ]
    ax1.legend(handles=legend_elements, title="Rainfall", fontsize=12, title_fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ---- KDE Plot (Variable distribution by Rainfall) ----
    ax2 = fig.add_subplot(gs[1, 1])
    sns.kdeplot(data=train_data, x=column, hue='rainfall', palette=rainfall_colors, ax=ax2, fill=True, common_norm=False, alpha=0.6)

    ax2.set_xlabel(column, fontsize=14)
    ax2.set_ylabel('Density', fontsize=14)
    ax2.set_title(f'Distribution (KDE) of {column} by Rainfall', fontsize=16, fontweight='bold')  
    ax2.legend(title='Rainfall', fontsize=12, title_fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Adjust layout spacing
    plt.tight_layout(pad=3.0)
    plt.show()

    # ---- Add clear separation after each variable ----
    plt.figure(figsize=(16, 0.3))  # Adjust spacing
    plt.axhline(y=0, color='gray', linewidth=5, linestyle='-') 
    plt.axis('off')
    plt.show()


# Impute the missing value with the median
test_data['winddirection'].fillna(test_data['winddirection'].median(), inplace=True)


import numpy as np
import pandas as pd
import scipy.stats as stats  # Importing for Box-Cox and Yeo-Johnson transformations

# Define function to categorize wind direction into sectors 
def wind_sector(direction):
    if pd.isna(direction):
        return np.nan  # Preserve missing values for later handling
    direction = float(direction)
    if direction >= 315 or direction < 45:
        return 'North'
    elif direction >= 45 and direction < 135:
        return 'East'
    elif direction >= 135 and direction < 225:
        return 'South'
    else:
        return 'West'

def perform_feature_engineering(df):
    """
    Applies feature engineering to the dataframe, creating new features for weather prediction.
    """
    
    # 1. Seasonal Features using 'day' (cyclical representation of the year)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)

    # 2. Lagged Features (previous day's values for key predictors)
    #    Shift by 1, then fill any remaining NaNs with 0 (or a median if desired)
    df['cloud_lag1'] = df['cloud'].shift(1).fillna(0)
    df['sunshine_lag1'] = df['sunshine'].shift(1).fillna(0)
    df['humidity_lag1'] = df['humidity'].shift(1).fillna(0)

    # 3. Rolling Statistics (3-day trends for key predictors)
    #    Use rolling(window=3, min_periods=1) so the first 1-2 rows won't be NaN. Backfill if needed.
    df['cloud_roll3_mean'] = df['cloud'].rolling(window=3, min_periods=1).mean().fillna(method='bfill')
    df['sunshine_roll3_mean'] = df['sunshine'].rolling(window=3, min_periods=1).mean().fillna(method='bfill')
    df['humidity_roll3_mean'] = df['humidity'].rolling(window=3, min_periods=1).mean().fillna(method='bfill')

    # 4. Interaction Features (combinations of highly correlated features)
    df['cloud_humidity'] = (df['cloud'] * df['humidity']).fillna(0)  # Replace missing with 0
    df['sunshine_cloud_ratio'] = (df['sunshine'] / (df['cloud'] + 1e-5)).fillna(0)

    # 5. Meteorological Features
    #    Compute temperature range and pressure difference
    df['temp_range'] = (df['maxtemp'] - df['mintemp']).fillna(df['maxtemp'].median())
    df['pressure_diff'] = df['pressure'].diff().fillna(0)

    # 6. Additional Time-Based Interactions with 'day'
    df['cloud_day_sin'] = (df['cloud'] * df['day_sin']).fillna(0)
    df['sunshine_day_cos'] = (df['sunshine'] * df['day_cos']).fillna(0)
    df['humidity_roll3_day_sin'] = (df['humidity_roll3_mean'] * df['day_sin']).fillna(0)

    # 7. Categorical Feature: Wind Direction
    #    Map wind direction to bins and replace missing with 'Unknown'
    df['wind_sector'] = df['winddirection'].apply(wind_sector).fillna('Unknown')
    
    # 7.1. Wind and Cloud Interaction Features (NEW)
    #    Captures how changes in wind and cloud metrics interact.
    df['change_in_direction'] = abs(df['winddirection'] - df['winddirection'].shift(1)).fillna(0)
    df['cloud_wind_interaction'] = df['cloud'] * np.log1p(df['windspeed'])
    df['wind_cloud_interaction'] = np.log1p(df['cloud']) * df['windspeed']

    # 8. Logarithmic and Transform Features for 'cloud' variable (NEW)
    df['cloud_log'] = np.log1p(df['cloud'])  # Log transformation to handle skewness
    df['cloud_sqrt'] = np.sqrt(df['cloud'])    # Square root transformation
    # Box-Cox transformation (requires strictly positive values; add 1 to avoid zero)
    df['cloud_boxcox'], lambda_bc = stats.boxcox(df['cloud'] + 1)
    # Yeo-Johnson transformation (handles negative values as well)
    df['cloud_yeojohnson'], lambda_yj = stats.yeojohnson(df['cloud'])

    # 9. Additional Meteorological Features (NEW)
    #    Combining logarithmic transformations for pressure and dewpoint, and cloud & sunshine
    df['log_pressure_dewpoint'] = np.log1p(df['pressure']) + np.log1p(df['dewpoint'])
    df['log_cloud_sunshine'] = np.log1p(df['cloud']) + np.log1p(df['sunshine'])
    df['cloudtest'] = (df['cloud'] == 88).astype(int)  # Binary flag if cloud equals 88
    df['sin_day2'] = np.sin(2 * np.pi * df['day'] / (365 * 2))  # Alternative cyclical feature (half frequency)
    df['cos_day2'] = np.cos(2 * np.pi * df['day'] / (365 * 2))
    df['wet_bulb'] = (2/3 * df['temparature'] + 1/3 * df['dewpoint'])  # Weighted average for wet bulb temperature
    
    return df

# ----------------------
# Apply Feature Engineering to Combined Train & Test Data
# ----------------------
id_test = test_data['id']

# Concatenate train & test, apply transformations, then split back
full_data = pd.concat([train_data, test_data], axis=0).sort_values('id')
full_data = perform_feature_engineering(full_data)

# Split back into train & test
train_data = full_data[full_data['rainfall'].notna()]
test_data = full_data[full_data['rainfall'].isna()]

# ----------------------
# List of Newly Created Features
# ----------------------
newly_created_vars = [
    # 1. Cyclical Seasonal Features
    'day_sin', 'day_cos',
    
    # 2. Lagged Features
    'cloud_lag1', 'sunshine_lag1', 'humidity_lag1',
    
    # 3. Rolling Statistics
    'cloud_roll3_mean', 'sunshine_roll3_mean', 'humidity_roll3_mean',
    
    # 4. Interaction Features
    'cloud_humidity', 'sunshine_cloud_ratio',
    
    # 5. Meteorological Features
    'temp_range', 'pressure_diff',
    
    # 6. Time-Based Interactions
    'cloud_day_sin', 'sunshine_day_cos', 'humidity_roll3_day_sin',
    
    # 7.1. Wind and Cloud Interaction Features (NEW)
    'change_in_direction', 'cloud_wind_interaction', 'wind_cloud_interaction',
    
    # 8. Logarithmic and Transform Features for 'cloud'
    'cloud_log', 'cloud_sqrt', 'cloud_boxcox', 'cloud_yeojohnson',
    
    # 9. Additional Meteorological Features
    'log_pressure_dewpoint', 'log_cloud_sunshine', 'cloudtest', 
    'sin_day2', 'cos_day2', 'wet_bulb'
]

# Categorical Features
categorical_new_feats = ['wind_sector']


# Compute correlation matrix only for newly created features
corr_train = train_data[newly_created_vars + ['rainfall']].corr()[['rainfall']]

# Heatmap visualization without color bar, displaying values vertically
plt.figure(figsize=(10, 2))
ax = sns.heatmap(
    corr_train.T,  # Transposing so features are on x-axis
    annot=True, 
    cmap='viridis', 
    linewidths=0.5, 
    cbar=False, 
    fmt=".2f", 
    annot_kws={"rotation": 90}  # Rotate annotations to be vertical
)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.title('Correlation Heatmap - New Engineered Features vs Rainfall')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

# Columns to encode
columns_to_encode = ['wind_sector']

# Perform one-hot encoding with prefix
encoded_data = pd.get_dummies(train_data[columns_to_encode], prefix=columns_to_encode)

# Ensure there are no duplicate column names before joining
train_data = train_data.drop(columns=columns_to_encode, errors="ignore")  # Drop original before merging
train_data = train_data.join(encoded_data)

# Prepare feature matrix X and target variable y
X = train_data.select_dtypes(include=['float64', 'int64']).drop(columns=['rainfall', 'id'], errors='ignore').copy()
y = train_data["rainfall"].copy()

# Train a Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf.fit(X, y)

# Get feature importances 
feature_importances = rf.feature_importances_
important_features = np.argsort(feature_importances)[::-1][:15]

# Get selected feature names and importance scores
selected_features = X.columns[important_features]
selected_importance = feature_importances[important_features]

print(f"Top {len(selected_features)} important features from Random Forest:")
print(selected_features)

# -------------------------------
# Visualization of Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x=selected_importance, y=selected_features, palette="viridis")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Top 15 Feature Importances from Random Forest")
plt.gca().invert_yaxis()  # Flip so the most important is at the top
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.show()


# Compute correlation matrix
corr_matrix = X[selected_features].corr().abs()

# Create a mask to filter highly correlated features 
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_correlation = [column for column in upper.columns if any(upper[column] > 0.80)]

# Remove highly correlated features
final_features = [f for f in selected_features if f not in high_correlation]

# Display final selected features
print(f"Final Selected Features After Correlation Filtering: {final_features}")

# Visualization: Correlation Heatmap Before & After Filtering
plt.figure(figsize=(12, 6))

# Before Filtering
plt.subplot(1, 2, 1)
sns.heatmap(corr_matrix, annot=False, cmap="viridis", linewidths=0.5)
plt.title("Feature Correlation Before Filtering")

# After Filtering (Subset of Final Features)
filtered_corr_matrix = X[final_features].corr().abs()
plt.subplot(1, 2, 2)
sns.heatmap(filtered_corr_matrix, annot=False, cmap="viridis", linewidths=0.5)
plt.title("Feature Correlation After Filtering")

plt.tight_layout()
plt.show()


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression  # Simple, fast model for RFE

# Step 1: Prepare Data
X = train_data.drop(columns=['rainfall'], errors='ignore')  # Feature set
y = train_data['rainfall']  # Target variable

# Step 2: Initialize RFE with Linear Regression as the estimator
n_features_to_select = 10  # Choose the number of top features to retain
estimator = LinearRegression()  # You can use other models like RandomForestRegressor

rfe = RFE(estimator, n_features_to_select=n_features_to_select)

# Step 3: Fit RFE to select the best features
rfe.fit(X, y)

# Step 4: Extract selected feature names
selected_rfe_features = X.columns[rfe.support_].tolist()

print("Selected Features using RFE:")
print(selected_rfe_features)


# Drop columns from both train and test data
#selected_features = final_features 
selected_features = ['dewpoint', 'cloud', 'sunshine', 'cloud_log', 'cloud_sqrt', 'cloud_boxcox', 'cloud_yeojohnson', 'log_pressure_dewpoint', 'log_cloud_sunshine', 'sin_day2']

train_data = train_data[selected_features + ['rainfall']]
test_data = test_data[ selected_features ]


import matplotlib.pyplot as plt
import seaborn as sns

# Identify numerical variables
columns_to_check = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Remove unwanted variables
columns_to_check = [col for col in columns_to_check if col not in ['rainfall', 'id']]

# Function to remove outliers using IQR and visualize only affected features
def remove_outliers_iqr_with_plot(data, column):
    Q1 = data[column].quantile(0.05)
    Q3 = data[column].quantile(0.95)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the data
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    # Calculate the number of rows deleted
    rows_deleted = len(data) - len(filtered_data)
    
    # Only proceed if outliers were detected (i.e., rows were deleted)
    if rows_deleted > 0:
        # Create a 1x2 plot for before & after visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Original Data Boxplot
        sns.boxplot(x=data[column], color='lightblue', ax=axes[0], 
                    flierprops={'marker': 'o', 'markersize': 5, 'markerfacecolor': 'red'})
        axes[0].set_title(f'Before Outlier Removal: {column}')

        # Highlight Q1, Q3, and Bounds in the first plot
        axes[0].axvline(Q1, color='green', linestyle='--', label='Q1 (5th Percentile)')
        axes[0].axvline(Q3, color='blue', linestyle='--', label='Q3 (95th Percentile)')
        axes[0].axvline(lower_bound, color='red', linestyle='-', label='Lower Bound')
        axes[0].axvline(upper_bound, color='red', linestyle='-', label='Upper Bound')
        axes[0].legend()

        # Boxplot after outlier removal
        sns.boxplot(x=filtered_data[column], color='lightgreen', ax=axes[1], 
                    flierprops={'marker': 'o', 'markersize': 5, 'markerfacecolor': 'red'})
        axes[1].set_title(f'After Outlier Removal: {column}')

        plt.suptitle(f'Outlier Detection & Removal for {column}')
        plt.tight_layout()
        plt.show()

        print(f"âœ… Outliers detected and removed for {column} â†’ {rows_deleted} rows deleted")

    return filtered_data, rows_deleted

# Apply function to each numerical column and visualize only affected features
rows_deleted_total = 0
features_with_outliers = []

for column in columns_to_check:
    train_data_filtered, rows_deleted = remove_outliers_iqr_with_plot(train_data, column)
    
    # Only update train_data if outliers were removed
    if rows_deleted > 0:
        train_data = train_data_filtered
        rows_deleted_total += rows_deleted
        features_with_outliers.append(column)

# Summary
print("\nğŸ“Š **Summary of Outlier Removal:**")
if features_with_outliers:
    print(f"Total rows deleted: {rows_deleted_total}")
    print(f"Features with outliers removed: {features_with_outliers}")
else:
    print("No significant outliers detected. No rows removed.")


y = train_data['rainfall']


from sklearn.preprocessing import MinMaxScaler

# Dropping selected columns for scaling
if all(col in train_data.columns for col in columns_to_encode):
    train_data_to_scale = train_data.drop(columns_to_encode, axis=1)
    test_data_to_scale = test_data.drop(columns_to_encode, axis=1)

else:
    train_data_to_scale = train_data
    test_data_to_scale = test_data

# Initialize MinMaxScaler
minmax_scaler = MinMaxScaler()

# Fit the scaler on the training data
minmax_scaler.fit(train_data_to_scale.drop(['rainfall'], axis=1))

# Scale the training data
scaled_data_train = minmax_scaler.transform(train_data_to_scale.drop(['rainfall'], axis=1))
scaled_train_df = pd.DataFrame(scaled_data_train, columns=train_data_to_scale.drop(['rainfall'], axis=1).columns)

# Scale the test data using the parameters from the training data
scaled_data_test = minmax_scaler.transform(test_data_to_scale)
scaled_test_df = pd.DataFrame(scaled_data_test, columns=test_data_to_scale.columns)


scaled_train_df.sample(3)


scaled_test_df.sample(3)


train_data_combined = scaled_train_df

test_data_combined = scaled_test_df


from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import numpy as np
import matplotlib.pyplot as plt

# Define Stratified K-Fold Cross-Validation
def stratified_cross_validation(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X, y):
        yield X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]


# Define Logistic Regression Model with Cross-Validation for Hyperparameter Tuning
log_reg_model = LogisticRegressionCV(
    cv=5,  # Cross-validation folds within training data
    penalty='l2',  # Ridge Regularization
    solver='liblinear',  # Good for small to medium datasets
    class_weight='balanced',  # Handles class imbalance
    random_state=42
)


# Evaluate model with Stratified K-Fold Cross-Validation
print(f"Training Logistic Regression with Stratified K-Fold Cross-Validation...")

X = train_data_combined.copy()

auc_scores = []
fold = 1
for X_train, X_val, y_train, y_val in stratified_cross_validation(X, y, n_splits=5):
    log_reg_model.fit(X_train, y_train)
    y_val_proba = log_reg_model.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_proba >= 0.5).astype(int)
    
    # Compute ROC-AUC
    fold_auc = roc_auc_score(y_val, y_val_proba)
    
    # Print metrics
    print(f"Fold {fold} - ROC-AUC: {fold_auc:.4f}")
    print(classification_report(y_val, y_val_pred))
    print("-" * 40)
    
    auc_scores.append(fold_auc)
    fold += 1

# Compute and print the average AUC score
avg_auc = np.mean(auc_scores)
print(f"Average ROC-AUC for Logistic Regression: {avg_auc:.4f}\n")


# Final training on the full dataset
print("Training Logistic Regression on the full dataset...")
log_reg_model.fit(X, y)

# Generate probability predictions on the test set
test_proba = log_reg_model.predict_proba(test_data_combined)[:, 1]


# Find the best fold based on the highest ROC-AUC score
best_fold_index = np.argmax(auc_scores)  # Get index of best ROC-AUC score
best_fold_data = list(stratified_cross_validation(X, y, n_splits=5))[best_fold_index]

X_train_best, X_val_best, y_train_best, y_val_best = best_fold_data
log_reg_model.fit(X_train_best, y_train_best)
y_val_proba_best = log_reg_model.predict_proba(X_val_best)[:, 1]

# Compute ROC Curve
fpr, tpr, _ = roc_curve(y_val_best, y_val_proba_best)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="#3498db", label=f"ROC Curve (AUC = {auc_scores[best_fold_index]:.4f})")
plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - Logistic Regression (Best Fold: {best_fold_index + 1})")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# Create submission file
submission_df = pd.DataFrame({
    'id': id_test,
    'rainfall': test_proba  # Predicted probabilities for rainfall
})

# Save to CSV
submission_df.to_csv("submission.csv", index=False)

# Display first 5 rows
submission_df.head(5)

