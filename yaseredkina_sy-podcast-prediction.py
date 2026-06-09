# Importing Libraries
import warnings
warnings.filterwarnings("ignore")

# Basic data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Preprocessing
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    OneHotEncoder,
    LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler

# Model selection and evaluation
from sklearn.model_selection import (
    train_test_split,
    KFold,
    GroupKFold,
    TimeSeriesSplit, 
    cross_val_score
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
    make_scorer
)

# Models
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor
)
from sklearn.svm import SVR, LinearSVC

# Advanced models
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Hyperparameter optimization
import optuna
from scipy.optimize import minimize

# Model saving
import joblib

# Regular expressions
import re


# Read .csv data file
train_data = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
original_data = pd.read_csv('/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv')



train_data.sample(5)


test_data.sample(5)


original_data.sample(5)


# Check number of rows and columns

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



# Check missing values
missing_train = train_data.isnull().sum().reset_index(name='Missing Train')
missing_test = test_data.isnull().sum().reset_index(name='Missing Test')
missing_original = original_data.isnull().sum().reset_index(name='Missing Original')

#Merge tables
merged_df = missing_train.merge(missing_test, on='index', how='left') \
                        .merge(missing_original, on='index', how='left') \
                        .merge(train_data.dtypes.reset_index(name='DataType'), on='index', how='left')

# Rename columns
merged_df.columns = ['Feature', 'Missing Train', 'Missing Test', 'Missing Original', 'DataType']

merged_df.style.background_gradient(cmap='Reds')


# Count duplicate rows in train_data
train_duplicates = train_data.duplicated().sum()
print(f"Number of duplicate rows in train_data: {train_duplicates}")

# Count duplicate rows in test_data
test_duplicates = test_data.duplicated().sum()
print(f"Number of duplicate rows in test_data: {test_duplicates}")

# Count duplicate rows in original_data
original_duplicates = original_data.duplicated().sum()
print(f"Number of duplicate rows in original_data: {original_duplicates}")


# Check unique values per column in train data
unique_values_train = train_data.nunique()
print ('Unique values TrainData')
print (unique_values_train)


# Check unique values per column in test data
unique_values_test = test_data.nunique()
print ('Unique values TestData')
print (unique_values_test)


# Check unique values per column in original data
unique_values_original = original_data.nunique()
print ('Unique values OriginalData')
print (unique_values_original)


# Numerical columns in train data
train_data.describe().T.style.background_gradient(cmap='Reds')


# Numerical columns in dataset
test_data.describe().T.style.background_gradient(cmap='Reds')


# Description of all the numerical columns in dataset
original_data.describe().T.style.background_gradient(cmap='Reds')


# Select numerical features
numerical_variables = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage']

# Define the target
target_variable = 'Listening_Time_minutes'

# Select categorical features
categorical_variables = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment', 'Number_of_Ads']


# Define a custom color palette
custom_red_palette = ['#fde0dc', '#f44336', '#b71c1c']


# Analysis of all NUMERICAL features

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
    sns.boxplot(data=pd.concat([train_data, test_data,original_data.dropna()]), x=variable, y="Dataset", palette=custom_red_palette)
    plt.xlabel(variable)
    plt.title(f"Box Plot for {variable}")

    # Separate Histograms
    plt.subplot(1, 2, 2)
    sns.histplot(data=train_data, x=variable, color=custom_red_palette[0], kde=True, bins=30, label="Train")
    sns.histplot(data=test_data, x=variable, color=custom_red_palette[1], kde=True, bins=30, label="Test")
    sns.histplot(data=original_data.dropna(), x=variable, color=custom_red_palette[2], kde=True, bins=30, label="Original")
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
    x='Episode_Length_minutes', y="Dataset", palette=custom_red_palette
)
plt.title("Box Plot for Episode_Length_minutes (No Outliers)")
plt.xlabel("Episode Length (minutes)")

# Histogram
plt.subplot(1, 2, 2)
sns.histplot(data=train_filtered, x='Episode_Length_minutes', color=custom_red_palette[0], kde=True, bins=30, label='Train')
sns.histplot(data=test_filtered, x='Episode_Length_minutes', color=custom_red_palette[1], kde=True, bins=30, label='Test')
sns.histplot(data=original_filtered.dropna(), x='Episode_Length_minutes', color=custom_red_palette[2], kde=True, bins=30, label='Original')
plt.xlabel("Episode Length (minutes)")
plt.ylabel("Frequency")
plt.title("Histogram for Episode_Length_minutes (No Outliers)")
plt.legend()

plt.tight_layout()
plt.show()


# Copy data and add label
train_data = train_data.copy()
test_data = test_data.copy()
train_data['dataset'] = 'train'
test_data['dataset'] = 'test'

# Combine into one DataFrame
combined = pd.concat([train_data, test_data])

# Plot for one categorical variable
def plot_categorical(variable):
    plt.figure(figsize=(14, 5))

    # Pie chart
    plt.subplot(1, 2, 1)
    values = combined[variable].value_counts()
    values_to_plot = values.copy()

    # Group rare categories into 'Other'
    threshold = 0.05 * values.sum()
    if (values < threshold).any():
        values_to_plot = values[values >= threshold]
        values_to_plot['Other'] = values[values < threshold].sum()

    plt.pie(
        values_to_plot,
        labels=values_to_plot.index,
        autopct='%1.1f%%',
        colors=custom_red_palette * (len(values_to_plot) // len(custom_red_palette) + 1)
    )
    plt.title(f"{variable} â€” Pie Chart")

    # Countplot split by dataset
    plt.subplot(1, 2, 2)
    sns.countplot(data=combined, x=variable, hue='dataset', palette=custom_red_palette[:2])
    plt.title(f"{variable} â€” Countplot")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

# Plot for all categorical variables
for var in categorical_variables:
    plot_categorical(var)

# Drop helper column
train_data.drop('dataset', axis=1, inplace=True)
test_data.drop('dataset', axis=1, inplace=True)


# Add labels to the datasets
train_data['dataset'] = 'Train'
original_data['dataset'] = 'Original'

# Combine both datasets
combined = pd.concat([train_data, original_data.dropna()])

# Quick plot for one numeric variable
def simple_numeric_plot(variable):
    plt.figure(figsize=(12, 4))

    # Boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=combined, x=variable, y='dataset', palette=custom_red_palette)
    plt.title(f"{variable} â€” Boxplot")
    plt.xlabel(variable)

    # Histogram
    plt.subplot(1, 2, 2)
    sns.histplot(data=train_data, x=variable, color=custom_red_palette[0], kde=True, label='Train')
    sns.histplot(data=original_data.dropna(), x=variable, color=custom_red_palette[1], kde=True, label='Original')
    plt.title(f"{variable} â€” Histogram")
    plt.xlabel(variable)
    plt.ylabel("Count")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Run the plot function for the target variable
simple_numeric_plot('Listening_Time_minutes')

# Drop helper column
train_data.drop('dataset', axis=1, inplace=True)
original_data.drop('dataset', axis=1, inplace=True)


# Select relevant numerical variables and include the 'Number_of_Ads' variable
variables = [col for col in train_data.columns if col in numerical_variables] + ['Number_of_Ads']
train_variables = variables + ['Listening_Time_minutes']

# Calculate the correlation matrices for train and test data
corr_train = train_data[train_variables].corr()
corr_test = test_data[variables].corr()

# Create arrays to mask the upper triangle of the correlation matrices
mask_train = np.triu(np.ones_like(corr_train, dtype=bool))
mask_test = np.triu(np.ones_like(corr_test, dtype=bool))

# Set annotation parameters
annot_kws = {"size": 8, "rotation": 45}

# Create the plots
plt.figure(figsize=(15, 5))

# Plot the correlation heatmap for the train data
plt.subplot(1, 2, 1)
sns.heatmap(corr_train, mask=mask_train, cmap=sns.color_palette(custom_red_palette, as_cmap=True), annot=True, annot_kws=annot_kws)
plt.title('Correlation Heatmap - Train Data')

# Plot the correlation heatmap for the test data
plt.subplot(1, 2, 2)
sns.heatmap(corr_test, mask=mask_test, cmap=sns.color_palette(custom_red_palette, as_cmap=True), annot=True, annot_kws=annot_kws)
plt.title('Correlation Heatmap - Test Data')

# Display the plots
plt.tight_layout()
plt.show()


# Select relevant numerical variables and include the 'Number_of_Ads' variable
variables = [col for col in train_data.columns if col in numerical_variables] + ['Number_of_Ads']
train_variables = variables + ['Listening_Time_minutes']

# Calculate the correlation for the train data and transpose it for horizontal display
corr_train = train_data[train_variables].corr()[['Listening_Time_minutes']].T

# Set annotation parameters
annot_kws = {"size": 10}

# Create the plot
plt.figure(figsize=(10, 2))
sns.heatmap(corr_train, cmap=sns.color_palette(custom_red_palette, as_cmap=True), annot=True,
            square=False, linewidths=0.5, annot_kws=annot_kws, cbar=False)

# Formatting the plot
plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
plt.title('Correlation Heatmap Train Data')
plt.yticks(rotation=0)  # Make y-axis labels horizontal

# Show the plot
plt.show()


# Define the upper threshold to exclude extreme values (e.g., 99th percentile)
threshold = train_data['Episode_Length_minutes'].quantile(0.99)

# Filter the data (original data remains unchanged)
filtered_df = train_data[train_data['Episode_Length_minutes'] <= threshold]

# Create a scatter plot with the filtered data
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=filtered_df,
    x='Episode_Length_minutes',
    y='Listening_Time_minutes',
    hue='Genre',  # Use 'Genre' to differentiate colors
    palette=custom_red_palette,  # Use custom red color palette
)

# Customize the plot
plt.title("Episode Length vs Listening Time (Excluding Top 1% Outliers)", fontsize=14, fontweight='bold')
plt.xlabel("Episode Length (minutes)")
plt.ylabel("Listening Time (minutes)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend next to the plot
plt.tight_layout()  # Automatically adjust layout for better fitting
plt.grid(True, linestyle='--', alpha=0.5)  # Add grid for easier reading
plt.show()


# Work on a copy to keep train_data intact
temp_df = train_data[['Episode_Title', 'Listening_Time_minutes']].copy()

# Extract numeric episode number from titles like "Episode 98"
temp_df['Episode_Number'] = temp_df['Episode_Title'].str.extract(r'(\d+)').astype(int)

# Create a new column 'Episode_Group' for grouping episodes by 5 (1-5, 6-10, etc.)
temp_df['Episode_Group'] = (temp_df['Episode_Number'] - 1) // 5 + 1

# Calculate the 1st and 99th percentiles of Listening_Time_minutes for each group of episodes
lower_percentile = temp_df['Listening_Time_minutes'].quantile(0.01)
upper_percentile = temp_df['Listening_Time_minutes'].quantile(0.99)

# Apply percentile clipping to remove extreme values
temp_df['Listening_Time_minutes'] = temp_df['Listening_Time_minutes'].clip(lower=lower_percentile, upper=upper_percentile)

# Group by 'Episode_Group' and calculate average listening time
avg_listen_by_group = temp_df.groupby('Episode_Group')['Listening_Time_minutes'].mean().reset_index()

# Create labels for the x-axis with episode ranges
episode_labels = [f"{(i-1)*5+1}-{i*5}" for i in avg_listen_by_group['Episode_Group']]

# Plotting
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Box plot to show distribution (with whiskers) for each episode group
sns.boxplot(
    data=temp_df,
    x='Episode_Group',
    y='Listening_Time_minutes',
    color='lightgray',  # Light color for the boxplot to highlight the lineplot
    width=0.6
)

# Line plot for average listening time using 'Reds' color palette
sns.lineplot(
    data=avg_listen_by_group,
    x='Episode_Group',
    y='Listening_Time_minutes',
    marker='o',  # Add markers to the line for clarity
    color='darkred'  # Set the line color to dark red
)

# Formatting
plt.title("Average Listening Time by Episode Group (5 episodes per group) with Percentile Clipping", fontsize=16, fontweight='bold')
plt.xlabel("Episode Range (by 5)", fontsize=13)
plt.ylabel("Listening Time (minutes)", fontsize=13)

# Set the x-ticks labels to the episode ranges
plt.xticks(ticks=range(len(avg_listen_by_group)), labels=episode_labels)

plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()


# Work on a copy to keep the original 'train_data' intact
temp_df = train_data[['Episode_Title', 'Listening_Time_minutes']].copy()

# Extract numeric episode number from titles like "Episode 98"
temp_df['Episode_Number'] = temp_df['Episode_Title'].str.extract(r'(\d+)').astype(int)

# Create a new column 'High_Listen_Episodes' to flag episodes numbered between 20 and 80
temp_df['High_Listen_Episodes'] = temp_df['Episode_Number'].apply(lambda x: 1 if 20 <= x <= 80 else 0)

# Add 'Episode_Number' and 'High_Listen_Episodes' back to the main DataFrame
train_data['Episode_Number'] = temp_df['Episode_Number']
train_data['High_Listen_Episodes'] = temp_df['High_Listen_Episodes']

# Check the result by displaying the first few rows of the relevant columns
train_data[['Episode_Title', 'Episode_Number', 'High_Listen_Episodes']].head()


# Set the style and figure size
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

# Bar plot for average listening time grouped by genre and episode sentiment
sns.barplot(
    data=train_data,
    x='Genre',  # Genre of the episode
    y='Listening_Time_minutes',  # Listening time in minutes
    hue='Episode_Sentiment',  # Sentiment of the episode
    palette=custom_red_palette  # Custom color palette
)

# Formatting the plot
plt.title('Average Listening Time by Genre and Episode Sentiment', fontsize=16, fontweight='bold')
plt.xlabel('Genre', fontsize=12)
plt.ylabel('Average Listening Time (minutes)', fontsize=12)
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.legend(title='Episode Sentiment')

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()


# Grouping: Calculate the average listening time grouped by sentiment and genre
grouped = (
    train_data
    .groupby(['Episode_Sentiment', 'Genre'])['Listening_Time_minutes']
    .mean()
    .reset_index()
)

# Define a custom red color palette (can be replaced or expanded)
custom_red_palette = ['#fde0dc', '#f44336', '#b71c1c']  # Light, vibrant, dark red

# Assign colors to genres (if there are more than 3 genres, the palette repeats)
import itertools
genres = grouped['Genre'].unique()
genre_palette = dict(zip(genres, itertools.cycle(custom_red_palette)))

# Plotting the graph
plt.figure(figsize=(10, 5))
sns.barplot(
    data=grouped,
    x='Episode_Sentiment',
    y='Listening_Time_minutes',
    hue='Genre',
    palette=genre_palette
)

# Formatting the plot
plt.title('Average Listening Time by Sentiment and Genre', fontsize=14, fontweight='bold')
plt.xlabel('Episode Sentiment')
plt.ylabel('Average Listening Time (minutes)')
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()


# List of numeric features that require imputation (based on the previous analysis)
numeric_features = ['Episode_Length_minutes', 'Guest_Popularity_percentage','Number_of_Ads']

# Impute missing values in the training data using the median of each feature
for col in numeric_features:
    median_val = train_data[col].median()  # Calculate the median of the feature
    train_data[col] = train_data[col].fillna(median_val)  # Replace missing values with the median

# Impute missing values in the test data using the median from the training set
for col in numeric_features:
    median_val = train_data[col].median()  # Use the median from the training data for consistency
    test_data[col] = test_data[col].fillna(median_val)  # Replace missing values in the test set


# Handle extreme values in 'Episode_Length_minutes' by capping at the 99th percentile

# 1. Calculate the 99th percentile for the training data
cap = train_data['Episode_Length_minutes'].quantile(0.99)  # Find the 99th percentile

# 2. Cap the values in both train and test datasets: replace values above the cap with the cap value
train_data['Episode_Length_minutes'] = train_data['Episode_Length_minutes'].clip(upper=cap)  # Apply cap to train data
test_data['Episode_Length_minutes'] = test_data['Episode_Length_minutes'].clip(upper=cap)  # Apply cap to test data


# Create a feature for episode completion ratio

# 1. Add to the training data
train_data['completion_ratio'] = (
    train_data['Listening_Time_minutes'] / train_data['Episode_Length_minutes']
).clip(upper=1.0)  # Cap the maximum at 1.0 (100%)

# 2. Add to the test data (if 'Listening_Time_minutes' exists, for example, for validation)
if 'Listening_Time_minutes' in test_data.columns:
    test_data['completion_ratio'] = (
        test_data['Listening_Time_minutes'] / test_data['Episode_Length_minutes']
    ).clip(upper=1.0)
else:
    test_data['completion_ratio'] = np.nan  # Leave it empty (NaN) if not available


# Creating the binary feature: is_weekend

# List of weekend days
weekend_days = ['Saturday', 'Sunday']

# 1. For train_data
# Create 'is_weekend' feature by checking if 'Publication_Day' is a weekend (Saturday or Sunday)
train_data['is_weekend'] = train_data['Publication_Day'].isin(weekend_days).astype(int)

# 2. For test_data
# Create 'is_weekend' feature for the test set in the same way
test_data['is_weekend'] = test_data['Publication_Day'].isin(weekend_days).astype(int)


# Grouping by Genre and Sentiment Combination

# Group the data by both 'Genre' and 'Episode_Sentiment' to calculate the average listening time for each combination
avg_listening_by_combo = (
    train_data
    .groupby(['Genre', 'Episode_Sentiment'])['Listening_Time_minutes']
    .mean()
    .reset_index()
)

# Create a new column for combined labels
# Combine 'Genre' and 'Episode_Sentiment' into a single column 'Genre_Sentiment' for easier display
avg_listening_by_combo['Genre_Sentiment'] = avg_listening_by_combo['Genre'] + ' / ' + avg_listening_by_combo['Episode_Sentiment']

# Custom Color Palette
# Define a custom color palette using the "Reds" color scheme for visualization
custom_palette = sns.color_palette("Reds", n_colors=len(avg_listening_by_combo))

# Create the horizontal bar plot
plt.figure(figsize=(10, 6))  # Set figure size
sns.barplot(
    data=avg_listening_by_combo,
    y='Genre_Sentiment',  # Y-axis: Combined genre and sentiment
    x='Listening_Time_minutes',  # X-axis: Average listening time
    palette=custom_palette  # Apply the custom color palette
)

# Formatting the plot
plt.title('Average Listening Time by Genre Ã— Sentiment', fontsize=14, fontweight='bold')  # Title
plt.xlabel('Average Listening Time (minutes)')  # X-axis label
plt.ylabel('Genre Ã— Sentiment')  # Y-axis label
plt.grid(True, linestyle='--', alpha=0.4)  # Add grid for better readability
plt.tight_layout()  # Adjust layout for better fitting
plt.show()  # Display the plot


# Define specific combinations of Genre and Episode_Sentiment that we want to track
combinations = [
    ('True Crime', 'Positive'),
    ('Music', 'Positive'),
    ('Health', 'Positive'),
    ('Education', 'Positive'),
    ('Business', 'Positive')
]

# Create a binary column for each Genre Ã— Sentiment combination
for genre, sentiment in combinations:
    # Construct the new column name (e.g., 'True Crime_Positive')
    column_name = f'{genre}_{sentiment}'
    
    # For training data: 1 if both genre and sentiment match, otherwise 0
    train_data[column_name] = (
        (train_data['Genre'] == genre) & 
        (train_data['Episode_Sentiment'] == sentiment)
    ).astype(int)
    
    # Repeat the same for test data
    test_data[column_name] = (
        (test_data['Genre'] == genre) & 
        (test_data['Episode_Sentiment'] == sentiment)
    ).astype(int)

# Display the new columns to verify
train_data[['True Crime_Positive', 'Music_Positive', 'Health_Positive', 'Education_Positive', 'Business_Positive']]


# Create new features capturing the difference and average popularity between host and guest

# For training data
train_data['popularity_difference'] = train_data['Host_Popularity_percentage'] - train_data['Guest_Popularity_percentage']
train_data['popularity_average'] = (train_data['Host_Popularity_percentage'] + train_data['Guest_Popularity_percentage']) / 2

# For test data
test_data['popularity_difference'] = test_data['Host_Popularity_percentage'] - test_data['Guest_Popularity_percentage']
test_data['popularity_average'] = (test_data['Host_Popularity_percentage'] + test_data['Guest_Popularity_percentage']) / 2

# Preview the new columns to ensure correctness
train_data[['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'popularity_difference', 'popularity_average']].head()


# Create a new feature: number of ads per minute of episode length

# For training data
train_data['ads_per_minute'] = train_data['Number_of_Ads'] / train_data['Episode_Length_minutes']

# For test data
test_data['ads_per_minute'] = test_data['Number_of_Ads'] / test_data['Episode_Length_minutes']

# Preview the new column to ensure it's calculated correctly
train_data[['Number_of_Ads', 'Episode_Length_minutes', 'ads_per_minute']].head()


# Counting NaN Values in Each Column
nan_counts = train_data.isna().sum()
print("Counting NaN Values in Each Column:\n", nan_counts)


# Extracting episode number from the title - using a more efficient method
train_data['Episode_Number'] = pd.to_numeric(train_data['Episode_Title'].str.extract(r'(\d+)')[0], errors='coerce')

# Calculating average listening time per episode - using transform for vectorization
train_data['Listening_Time_minutes_Avg'] = train_data.groupby('Episode_Number')['Listening_Time_minutes'].transform('mean')

# No need for merging data - it's already done using transform

# Grouping episodes in sets of five
train_data['Episode_Group'] = ((train_data['Episode_Number'] - 1) // 5) + 1

# Percentile transformation - using rank instead of percentile for better performance
train_data['Listening_Time_Percentile'] = train_data['Listening_Time_minutes'].rank(pct=True) * 100

# Generating features with genre and sentiment - using transform for efficiency
train_data['Listening_Time_minutes_Genre_Sentiment_Avg'] = train_data.groupby(
    ['Genre', 'Episode_Sentiment'])['Listening_Time_minutes'].transform('mean')

# Creating popularity features
train_data['popularity_difference'] = train_data['Host_Popularity_percentage'] - train_data['Guest_Popularity_percentage']
train_data['popularity_average'] = (train_data['Host_Popularity_percentage'] + train_data['Guest_Popularity_percentage']) / 2

# Generating the ads density feature
train_data['ads_per_minute'] = train_data['Number_of_Ads'] / train_data['Episode_Length_minutes']

# Final DataFrame with prepared features - adding the necessary columns
features = ['Episode_Number', 'Listening_Time_minutes_Avg', 'Episode_Group', 'Listening_Time_Percentile',
            'Listening_Time_minutes_Genre_Sentiment_Avg', 'popularity_difference', 'popularity_average', 
            'ads_per_minute', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Episode_Length_minutes',
            'Number_of_Ads']  # Adding the necessary columns
prepared_data = train_data[features]

# Checking and handling missing values
missing_values = prepared_data.isnull().sum()
if missing_values.sum() > 0:
    print(f"Missing values detected: {missing_values}")
    # Filling missing values with the mean
    prepared_data = prepared_data.fillna(prepared_data.mean())


print("Starting enhanced data preprocessing and model building...")

# Defining the target variable
target = train_data['Listening_Time_minutes']

# Splitting the data for model evaluation
X_train, X_val, y_train, y_val = train_test_split(
    prepared_data, target, test_size=0.15, random_state=42
)

# Creating additional features for model improvement
# 1. Adding polynomial features for numerical columns
for col in ['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Episode_Length_minutes']:
    if col in X_train.columns:
        X_train[f'{col}_squared'] = X_train[col] ** 2
        X_val[f'{col}_squared'] = X_val[col] ** 2

# 2. Adding feature interactions
X_train['pop_x_length'] = X_train['popularity_average'] * X_train['Episode_Length_minutes']
X_val['pop_x_length'] = X_val['popularity_average'] * X_val['Episode_Length_minutes']

X_train['ads_x_length'] = X_train['Number_of_Ads'] * X_train['Episode_Length_minutes']
X_val['ads_x_length'] = X_val['Number_of_Ads'] * X_val['Episode_Length_minutes']

# Checking for missing values after feature creation
if X_train.isnull().sum().sum() > 0:
    print("Missing values detected in X_train after feature creation")
    X_train = X_train.fillna(X_train.mean())

if X_val.isnull().sum().sum() > 0:
    print("Missing values detected in X_val after feature creation")
    X_val = X_val.fillna(X_val.mean())

# 3. Creating a stacking model
base_models = [
    ('lightgbm', lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42
    )),
    ('xgboost', xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42
    )),
    ('random_forest', RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ))
]

# Meta-model for stacking
meta_model = Ridge(alpha=1.0)

# Creating the stacking model
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1
)

print("Training stacking model...")
stacking_model.fit(X_train, y_train)

# Evaluating the model on validation set
val_preds = stacking_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"Validation RMSE: {rmse:.4f}")

# Training the final model on the full dataset
print("Training final model on full data...")
# Adding polynomial features to the full dataset
for col in ['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Episode_Length_minutes']:
    if col in train_data.columns:
        prepared_data[f'{col}_squared'] = prepared_data[col] ** 2

# Adding feature interactions
prepared_data['pop_x_length'] = prepared_data['popularity_average'] * prepared_data['Episode_Length_minutes']
prepared_data['ads_x_length'] = prepared_data['Number_of_Ads'] * prepared_data['Episode_Length_minutes']

# Training the final model
final_model = stacking_model.fit(prepared_data, target)

# Saving the model
joblib.dump(final_model, 'best_model_stacking.pkl')
print("Enhanced model saved to 'best_model_stacking.pkl'")

# Loading test dataset
print("Loading test data...")
test_data = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

# Preprocessing the test data as we did with the training data
print("Preprocessing test data...")

# Standard transformations
test_data['Episode_Number'] = pd.to_numeric(test_data['Episode_Title'].str.extract(r'(\d+)')[0], errors='coerce')
avg_listen_by_ep = train_data.groupby('Episode_Number')['Listening_Time_minutes'].mean()
test_data['Listening_Time_minutes_Avg'] = test_data['Episode_Number'].map(avg_listen_by_ep)
test_data['Listening_Time_minutes_Avg'].fillna(train_data['Listening_Time_minutes'].mean(), inplace=True)
test_data['Episode_Group'] = ((test_data['Episode_Number'] - 1) // 5) + 1
test_data['Listening_Time_Percentile'] = 50

# Genre and sentiment processing
genre_sentiment_avg = train_data.groupby(['Genre', 'Episode_Sentiment'])['Listening_Time_minutes'].mean()
test_data['genre_sentiment_key'] = test_data['Genre'] + '_' + test_data['Episode_Sentiment']
genre_sentiment_dict = {f"{genre}_{sentiment}": value for (genre, sentiment), value in genre_sentiment_avg.items()}
test_data['Listening_Time_minutes_Genre_Sentiment_Avg'] = test_data['genre_sentiment_key'].map(genre_sentiment_dict)
test_data['Listening_Time_minutes_Genre_Sentiment_Avg'].fillna(train_data['Listening_Time_minutes'].mean(), inplace=True)
test_data.drop('genre_sentiment_key', axis=1, inplace=True)

# Popularity features
test_data['popularity_difference'] = test_data['Host_Popularity_percentage'] - test_data['Guest_Popularity_percentage']
test_data['popularity_average'] = (test_data['Host_Popularity_percentage'] + test_data['Guest_Popularity_percentage']) / 2
test_data['ads_per_minute'] = test_data['Number_of_Ads'] / test_data['Episode_Length_minutes']

# Adding polynomial features to the test data
for col in ['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Episode_Length_minutes']:
    if col in test_data.columns:
        test_data[f'{col}_squared'] = test_data[col] ** 2

# Adding feature interactions
test_data['pop_x_length'] = test_data['popularity_average'] * test_data['Episode_Length_minutes']
test_data['ads_x_length'] = test_data['Number_of_Ads'] * test_data['Episode_Length_minutes']

# Ensure all features used for training are present in the test data
missing_features = set(prepared_data.columns) - set(test_data.columns)
if missing_features:
    print(f"The following features are missing in the test data: {missing_features}")
    for feature in missing_features:
        test_data[feature] = 0  # Or another default value

# Align the column order with the training data
test_features = test_data[prepared_data.columns]

# Checking for missing values
missing_values = test_features.isnull().sum()
if missing_values.sum() > 0:
    print(f"Missing values detected: {missing_values}")
    test_features = test_features.fillna(test_features.mean())

# Making predictions with the stacking model
print("Making predictions...")
predictions = final_model.predict(test_features)

# Creating submission file
submission = pd.DataFrame({
    'id': test_data['id'],
    'Listening_Time_minutes': predictions
})

# Saving the file
submission.to_csv('submission_improved.csv', index=False)

print("The 'submission_improved.csv' file has been successfully created.")
print(f"Number of predictions: {len(submission)}")
print(f"Prediction range: from {submission['Listening_Time_minutes'].min():.3f} to {submission['Listening_Time_minutes'].max():.3f}")
print(f"Average prediction: {submission['Listening_Time_minutes'].mean():.3f}")

