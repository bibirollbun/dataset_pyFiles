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


url = "https://cdn.shopify.com/s/files/1/0070/7032/files/kit-formerly-convertkit-waxDxYM2XI4-unsplash.jpg?v=1731360341"

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
from sklearn.impute import SimpleImputer


# Reading .csv data file
train_data = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
original_data = pd.read_csv('/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv')


train_data.sample(5)


test_data.sample(5)


original_data.sample(5)


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


numerical_variables = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage',]
target_variable = 'Listening_Time_minutes'
categorical_variables = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment', 'Number_of_Ads']


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


# Define a custom color palette again
custom_palette = ['#3498db', '#e74c3c','#2ecc71']

# Create a temporary copy and remove outliers (e.g., > 99th percentile)
def remove_outliers(series, threshold=0.99):
    return series < series.quantile(threshold)

# Create a combined dataframe (with dataset label) for plotting
train_temp = train_data.copy()
test_temp = test_data.copy()
original_temp = original_data.copy()

train_temp['Dataset'] = 'Train'
test_temp['Dataset'] = 'Test'
original_temp['Dataset'] = 'Original'

# Remove outliers from each version of the data
train_filtered = train_temp[remove_outliers(train_temp['Episode_Length_minutes'])]
test_filtered = test_temp[remove_outliers(test_temp['Episode_Length_minutes'])]
original_filtered = original_temp[remove_outliers(original_temp['Episode_Length_minutes'])]

# Plot only for Episode_Length_minutes
plt.figure(figsize=(12, 4))

# Box Plot
plt.subplot(1, 2, 1)
sns.boxplot(
    data=pd.concat([train_filtered, test_filtered, original_filtered.dropna()]),
    x='Episode_Length_minutes', y="Dataset", palette=custom_palette
)
plt.title("Box Plot for Episode_Length_minutes (No Outliers)")
plt.xlabel("Episode Length (minutes)")

# Histogram
plt.subplot(1, 2, 2)
sns.histplot(data=train_filtered, x='Episode_Length_minutes', color=custom_palette[0], kde=True, bins=30, label='Train')
sns.histplot(data=test_filtered, x='Episode_Length_minutes', color=custom_palette[1], kde=True, bins=30, label='Test')
sns.histplot(data=original_filtered.dropna(), x='Episode_Length_minutes', color=custom_palette[2], kde=True, bins=30, label='Original')
plt.xlabel("Episode Length (minutes)")
plt.ylabel("Frequency")
plt.title("Histogram for Episode_Length_minutes (No Outliers)")
plt.legend()

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import textwrap

# Define color palettes
pie_chart_palette = ['#33638d', '#28ae80', '#d3eb0c', '#ff9a0b', '#7e03a8', '#35b779', '#fde725', '#440154', '#90d743', '#482173', '#22a884', '#f8961e']
custom_palette = ['#3498db', '#e74c3c', '#2ecc71']  # Only first two used for train and test respectively
countplot_color = '#5C67A3'

# Add a 'dataset' column to differentiate train and test data
train_data = train_data.copy()
test_data = test_data.copy()
train_data['dataset'] = 'train'
test_data['dataset'] = 'test'

# Function to create and display a row of plots for a single categorical variable
def create_categorical_plots(variable):
    sns.set_style('whitegrid')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pie Chart - Handling many categories
    plt.subplot(1, 2, 1)

    combined = pd.concat([train_data, test_data])
    value_counts = combined[variable].value_counts()

    # Combine small categories into "Other" if they contribute less than 5%
    threshold = 0.05 * value_counts.sum()
    filtered_values = value_counts[value_counts >= threshold]
    filtered_values['Other'] = value_counts[value_counts < threshold].sum()

    wedges, texts, autotexts = plt.pie(
        filtered_values,
        autopct=lambda p: f'{p:.1f}%' if p > 5 else '',  # Hide labels < 5%
        colors=pie_chart_palette[:len(filtered_values)],
        startangle=140,
        wedgeprops=dict(width=0.3),
        explode=[0.05 if p > 5 else 0 for p in filtered_values],  # Slightly separate larger slices
        textprops={'fontsize': 10}  # Adjust font size
    )

    plt.title("\n".join(textwrap.wrap(f"Pie Chart for {variable} [TRAIN & TEST Combined]", width=50)))
    plt.legend(filtered_values.index, loc="upper left", bbox_to_anchor=(1, 1))

    # Bar Graph: Use hue for dataset (train and test)
    plt.subplot(1, 2, 2)
    sns.countplot(
        data=combined,
        x=variable,
        hue='dataset',
        palette=custom_palette[:2],
        alpha=0.8
    )
    plt.xlabel(variable)
    plt.ylabel("Count")
    plt.title("\n".join(textwrap.wrap(f"Bar Graph for {variable}  [TRAIN & TEST Combined]", width=50)))
    plt.xticks(rotation=30)  # Rotate labels for readability

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()

# Perform univariate analysis for each categorical variable
for variable in categorical_variables:
    create_categorical_plots(variable)

# Drop the 'Dataset' column after analysis
train_data.drop('dataset', axis=1, inplace=True)
test_data.drop('dataset', axis=1, inplace=True)


# Custom palette for datasets
custom_palette = ['#3498db', '#e74c3c', '#2ecc71']

# Tag datasets for comparison
train_data['Dataset'] = 'Train'
test_data['Dataset'] = 'Test'
original_data['Dataset'] = 'Original'

# Function to visualize a continuous target variable
def plot_continuous_target(variable):
    sns.set_style('whitegrid')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Boxplot across datasets
    plt.subplot(1, 2, 1)
    sns.boxplot(
        data=pd.concat([train_data, original_data.dropna()]),
        x=variable,
        y='Dataset',
        palette=custom_palette
    )
    plt.title(f"Box Plot for {variable}")
    plt.xlabel(variable)

    # --- Histogram with KDE overlay
    plt.subplot(1, 2, 2)
    sns.histplot(train_data[variable], color=custom_palette[0], kde=True, bins=30, label='Train')
    sns.histplot(original_data.dropna()[variable], color=custom_palette[2], kde=True, bins=30, label='Original')
    plt.title(f"Distribution of {variable} [Train/Original]")
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()

# ğŸ“Œ Call for your continuous target variable (e.g., 'Listening_Time_minutes')
plot_continuous_target('Listening_Time_minutes')

# Drop 'Dataset' column after use
train_data.drop('Dataset', axis=1, inplace=True)
test_data.drop('Dataset', axis=1, inplace=True)
original_data.drop('Dataset', axis=1, inplace=True)


variables = [col for col in train_data.columns if col in numerical_variables] + ['Number_of_Ads']

# Adding variables to the existing list
test_variables = variables
train_variables = variables+ ['Listening_Time_minutes']

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
variables = [col for col in train_data.columns if col in numerical_variables]+['Number_of_Ads']
train_variables = variables + ['Listening_Time_minutes']

# Compute correlation with 'rainfall' and transpose for horizontal display
corr_train = train_data[train_variables].corr()[['Listening_Time_minutes']].T  # Transpose for horizontal orientation

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


# Define an upper threshold to exclude extreme episode lengths (e.g., 99th percentile)
threshold = train_data['Episode_Length_minutes'].quantile(0.99)
# Filtered DataFrame (only for plotting, original data remains untouched)
filtered_df = train_data[train_data['Episode_Length_minutes'] <= threshold]
# Scatter plot with filtered data
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=filtered_df,
    x='Episode_Length_minutes',
    y='Listening_Time_minutes',
    hue='Genre',
    palette='viridis',
    alpha=0.7
)
# Plot styling
plt.title("Episode Length vs Listening Time (Excluding Top 1% Outliers)", fontsize=14, fontweight='bold')
plt.xlabel("Episode Length (minutes)")
plt.ylabel("Listening Time (minutes)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style and figure size
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

# Create a bar plot for average listening time, segmented by Episode_Sentiment
sns.barplot(
    data=train_data,
    x='Genre',
    y='Listening_Time_minutes',
    hue='Episode_Sentiment',
    palette='viridis'
)

# Titles and labels
plt.title('Average Listening Time by Genre and Episode Sentiment', fontsize=16, fontweight='bold')
plt.xlabel('Genre', fontsize=12)
plt.ylabel('Average Listening Time (minutes)', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Episode Sentiment')

# Layout adjustment
plt.tight_layout()
plt.show()


# List of categorical features to analyze
categorical_features = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

# Set style and color palette
sns.set_style("whitegrid")
palette = "viridis"

# Plot 2 features per row
for i in range(0, len(categorical_features), 2):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for j in range(2):
        if i + j < len(categorical_features):
            feature = categorical_features[i + j]
            ax = axes[j]
            sns.boxplot(
                y=feature,
                x='Listening_Time_minutes',
                data=train_data,
                palette=palette,
                ax=ax
            )
            ax.set_title(f"Listening Time by {feature}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Listening_Time_minutes")
            ax.set_ylabel(feature)

    plt.tight_layout()
    plt.show()


# Work on a copy to keep train_data intact
temp_df = train_data[['Episode_Title', 'Listening_Time_minutes']].copy()

# Extract numeric episode number from titles like "Episode 98"
temp_df['Episode_Number'] = temp_df['Episode_Title'].str.extract(r'(\d+)').astype(int)

# Group by episode number and calculate average listening time
avg_listen_by_ep = temp_df.groupby('Episode_Number')['Listening_Time_minutes'].mean().reset_index()

# Plotting
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Line plot with markers
sns.lineplot(
    data=avg_listen_by_ep,
    x='Episode_Number',
    y='Listening_Time_minutes',
    marker='o',
    label='Average Listening Time',
    color='mediumseagreen'
)

# Trend line (linear regression)
sns.regplot(
    data=avg_listen_by_ep,
    x='Episode_Number',
    y='Listening_Time_minutes',
    scatter=False,
    color='darkslategray',
    label='Trend Line'
)

# Formatting
plt.title("Average Listening Time vs Episode Number", fontsize=16, fontweight='bold')
plt.xlabel("Episode Number", fontsize=13)
plt.ylabel("Average Listening Time (minutes)", fontsize=13)
plt.legend()
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Group by Podcast_Name and calculate average Listening_Time_minutes
avg_listen_by_podcast = train_data.groupby('Podcast_Name')['Listening_Time_minutes'].mean().reset_index()

# Sort descending and keep only top 10
top_10_podcasts = avg_listen_by_podcast.sort_values('Listening_Time_minutes', ascending=False).head(10)

# Plot
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

sns.barplot(
    data=top_10_podcasts,
    y='Podcast_Name',
    x='Listening_Time_minutes',
    palette='viridis'
)

# Titles and labels
plt.title("Top 10 Podcasts by Average Listening Time", fontsize=16, fontweight='bold')
plt.xlabel("Average Listening Time (minutes)", fontsize=13)
plt.ylabel("Podcast Name", fontsize=13)

plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()


# List of numeric features that require imputation (based on your table)
numeric_features = ['Episode_Length_minutes', 'Guest_Popularity_percentage']

# Impute missing values in the training data using median
for col in numeric_features:
    median_val = train_data[col].median()
    train_data[col] = train_data[col].fillna(median_val)

# Impute missing values in the test data using the median from the training set
for col in numeric_features:
    median_val = train_data[col].median()
    test_data[col] = test_data[col].fillna(median_val)


import numpy as np
import pandas as pd

def create_features(df):
    """
    Given a dataframe with podcast data, create new features for modeling.
    This function excludes the target 'Listening_Time_minutes' when crafting features.

    Features engineered:
    - Ads_Per_Length: Number_of_Ads divided by Episode_Length_minutes.
    - Popularity_Ratio: Host_Popularity_percentage divided by (Guest_Popularity_percentage + 1e-6).
    - Episode_Number: Numeric episode number extracted from Episode_Title.
    - Log_Episode_Length: Natural logarithm of Episode_Length_minutes (to address skewness).
    - Popularity_Diff: Difference between Host_Popularity_percentage and Guest_Popularity_percentage.

    Parameters:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: DataFrame with new features added.
    """
    df = df.copy()

    # Feature: Ads_Per_Length (avoiding division by zero)
    df['Ads_Per_Length'] = df['Number_of_Ads'] / df['Episode_Length_minutes'].replace(0, np.nan)

    # Feature: Popularity_Ratio
    df['Popularity_Ratio'] = df['Host_Popularity_percentage'] / (df['Guest_Popularity_percentage'] + 1e-6)

    # Feature: Extract Episode_Number from Episode_Title (e.g., "Episode 98" -> 98)
    df['Episode_Number'] = df['Episode_Title'].str.extract(r'(\d+)').astype(int)

    # Feature: Log_Episode_Length to address skewness
    df['Log_Episode_Length'] = np.log(df['Episode_Length_minutes'].replace(0, np.nan))

    return df

# Apply the feature engineering function to both train and test data
train_data = create_features(train_data)
test_data = create_features(test_data)


# Drop columns from both train and test datasets
columns_to_drop = ['Podcast_Name', 'Episode_Title', 'Episode_Number', 'Episode_Length_minutes', 'Number_of_Ads']

train_data.drop(columns=columns_to_drop, inplace=True)
test_data.drop(columns=columns_to_drop, inplace=True)


import matplotlib.pyplot as plt
import seaborn as sns

# Identify numerical variables
columns_to_check = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Remove unwanted variables
columns_to_check = [col for col in columns_to_check if col not in ['Listening_Time_minutes', 'id','Ads_Per_Length', 'Popularity_Ratio']]

# Function to remove outliers using IQR and visualize only affected features
def remove_outliers_iqr_with_plot(data, column):
    Q1 = data[column].quantile(0.10)
    Q3 = data[column].quantile(0.90)
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
        axes[0].axvline(Q1, color='green', linestyle='--', label='Q1 (10th Percentile)')
        axes[0].axvline(Q3, color='blue', linestyle='--', label='Q3 (90th Percentile)')
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
print("\nğŸ“Š Summary of Outlier Removal:")
if features_with_outliers:
    print(f"Total rows deleted: {rows_deleted_total}")
    print(f"Features with outliers removed: {features_with_outliers}")
else:
    print("No significant outliers detected. No rows removed.")


y = train_data['Listening_Time_minutes']
id_test = test_data['id']
target = ['Listening_Time_minutes']
train_data.drop(columns=['id'], inplace=True)
test_data.drop(columns=['id'], inplace=True)


train_data.columns


# Selecting specific columns for encoding
columns_to_encode = ['Genre', 'Publication_Day', 'Episode_Sentiment', 'Publication_Time']
train_data_to_encode = train_data[columns_to_encode]
test_data_to_encode = test_data[columns_to_encode]

# Dropping selected columns for scaling
train_data_to_scale = train_data.drop(columns_to_encode, axis=1)
test_data_to_scale = test_data.drop(columns_to_encode, axis=1)

train_data_encoded = pd.get_dummies(train_data_to_encode, columns=columns_to_encode, drop_first=True)
test_data_encoded = pd.get_dummies(test_data_to_encode, columns=columns_to_encode, drop_first=True)


train_data_encoded.head()


test_data_encoded.head()


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
minmax_scaler.fit(train_data_to_scale.drop(target, axis=1))

# Scale the training data
scaled_data_train = minmax_scaler.transform(train_data_to_scale.drop(target, axis=1))
scaled_train_df = pd.DataFrame(scaled_data_train, columns=train_data_to_scale.drop(target, axis=1).columns)

# Scale the test data using the parameters from the training data
scaled_data_test = minmax_scaler.transform(test_data_to_scale)
scaled_test_df = pd.DataFrame(scaled_data_test, columns=test_data_to_scale.columns)


scaled_train_df.sample(3)


scaled_test_df.sample(3)


# Concatenate train datasets
train_data_combined = pd.concat([train_data_encoded.reset_index(drop=True), scaled_train_df.reset_index(drop=True)], axis=1)

# Concatenate test datasets
test_data_combined = pd.concat([test_data_encoded.reset_index(drop=True), scaled_test_df.reset_index(drop=True)], axis=1)


# Define XGBoost parameters (adjust as needed)
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'verbosity': 0  # to silence the output
}

# Assuming X, y, test_data_combined, and test_data are pre-defined DataFrames
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
rmse_scores = []
models = []  # store the model for each fold
best_iterations = []  # store best iteration per fold

X = train_data_combined.copy()


# Cross-validation using XGBoost
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Create DMatrix objects for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Set evaluation list for early stopping
    evals = [(dtrain, 'train'), (dval, 'valid')]

    # Train the model with early stopping
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=False
    )

    models.append(model)
    best_iterations.append(model.best_iteration)

    # âœ… Fixed line below using iteration_range instead of ntree_limit
    y_val_pred = model.predict(dval, iteration_range=(0, model.best_iteration))

    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    rmse_scores.append(rmse)
    print(f"Fold {fold} - RMSE: {rmse:.4f}")
    fold += 1

print(f"Average RMSE across folds: {np.mean(rmse_scores):.4f}")


import matplotlib.pyplot as plt
import seaborn as sns

# Compute correlation matrix
corr_matrix = X.corr()

# Get feature importances from the last trained model
importance_dict = model.get_score(importance_type='weight')
importance_df = pd.DataFrame.from_dict(importance_dict, orient='index', columns=['Importance'])
importance_df.index.name = 'Feature'
importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_df.reset_index(inplace=True)

# Create side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(22, 10))  # Adjust width as needed

# Plot 1: Correlation Heatmap
sns.heatmap(corr_matrix, cmap='viridis', center=0, square=True, 
            cbar_kws={'shrink': 0.5}, ax=axes[0])
axes[0].set_title('Feature Correlation Matrix', fontsize=14)

# Plot 2: Feature Importance
sns.barplot(
    data=importance_df,
    x='Importance',
    y='Feature',
    palette='viridis',
    ax=axes[1]
)
axes[1].set_title('XGBoost Feature Importance (by Weight)', fontsize=14)
axes[1].set_xlabel('Importance Score')
axes[1].set_ylabel('Features')

plt.tight_layout()
plt.show()


# Pick the best fold based on lowest RMSE
best_fold_index = np.argmin(rmse_scores)
print(f"\nBest fold is fold {best_fold_index+1} with RMSE={rmse_scores[best_fold_index]:.4f}")

# Recreate the data for that best fold
fold = 1
for train_idx, val_idx in kf.split(X):
    if fold - 1 == best_fold_index:
        X_val_best = X.iloc[val_idx]
        y_val_best = y.iloc[val_idx]
        model_best = models[best_fold_index]
        break
    fold += 1

# Predict again on the best fold
dval_best = xgb.DMatrix(X_val_best)
# âœ… Updated here
y_val_pred_best = model_best.predict(dval_best, iteration_range=(0, model_best.best_iteration))

# Pick a small subset (e.g. 15 points) for illustration
subset_size = 15
X_val_subset = X_val_best.iloc[:subset_size].copy()
y_val_subset = y_val_best.iloc[:subset_size].copy()
y_pred_subset = y_val_pred_best[:subset_size]

# Choose a numeric feature from X_val_subset to serve as the x-axis
if 'Episode_Length_minutes' in X_val_subset.columns:
    x_axis = X_val_subset['Episode_Length_minutes'].values
else:
    numeric_cols = X_val_subset.select_dtypes(include=np.number).columns
    x_axis = X_val_subset[numeric_cols[0]].values

# Sort by x_axis for a neat line plot
order = np.argsort(x_axis)
x_axis_sorted = x_axis[order]
y_actual_sorted = y_val_subset.values[order]
y_pred_sorted = y_pred_subset[order]

# Create the conceptual plot (Actual vs. Predicted for a sample of 15 points)
plt.figure(figsize=(10, 6))
plt.plot(x_axis_sorted, y_actual_sorted, 'o-', color='red', label='Actual', linewidth=2, markersize=6)
plt.plot(x_axis_sorted, y_pred_sorted, 's--', color='blue', label='Predicted', linewidth=2, markersize=6)

# Draw vertical lines to illustrate errors
for i in range(len(x_axis_sorted)):
    plt.vlines(
        x_axis_sorted[i],
        min(y_actual_sorted[i], y_pred_sorted[i]),
        max(y_actual_sorted[i], y_pred_sorted[i]),
        color='gray',
        linestyle=':',
        alpha=0.7
    )

plt.title("Conceptual RMSE Plot (Sample of 15 Points)", fontsize=16, fontweight='bold')
plt.xlabel("Episode Length (or first numeric feature)", fontsize=13)
plt.ylabel("Listening Time (Actual vs Predicted)", fontsize=13)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()


# Final training on the full dataset using the average best iteration from CV
dtrain_full = xgb.DMatrix(X, label=y)
avg_best_iteration = int(np.mean(best_iterations))
final_model = xgb.train(
    params,
    dtrain_full,
    num_boost_round=avg_best_iteration
)

# Generate predictions on the test set
dtest = xgb.DMatrix(test_data_combined)
# âœ… Updated line using iteration_range
test_preds = final_model.predict(dtest, iteration_range=(0, avg_best_iteration))

# Create submission file using the provided test IDs
submission_df = pd.DataFrame({
    'id': id_test,
    'Listening_Time_minutes': test_preds
})

# Save submission file
submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")

# Display first 5 rows of the submission file
submission_df.head(5)

