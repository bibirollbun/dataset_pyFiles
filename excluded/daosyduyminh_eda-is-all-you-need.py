import pandas as pd

df = pd.read_csv('/kaggle/input/detecting-reversal-points-in-us-equities/competition_data/train.csv')
display(df.head())


display(df.head())
df.info()
display(df.isnull().sum())


# 1. Print unique values and the number of unique values in 'ticker_id'
unique_ticker_ids = df['ticker_id'].unique()
num_unique_ticker_ids = len(unique_ticker_ids)
print(f"Unique ticker IDs: {unique_ticker_ids}")
print(f"Number of unique ticker IDs: {num_unique_ticker_ids}")

# 2. Convert the 't' column to datetime objects
df['t'] = pd.to_datetime(df['t'])

# 3. Determine the range of dates in 't' for the entire dataset
overall_date_range = (df['t'].min(), df['t'].max())
print(f"Overall date range: {overall_date_range}")

# 4. Determine earliest and latest dates for each unique ticker_id
date_range_by_ticker = df.groupby('ticker_id')['t'].agg(['min', 'max'])
print("\nDate range for each ticker ID:")
display(date_range_by_ticker)

# 5. Calculate the number of data points for each unique ticker_id
data_points_by_ticker = df.groupby('ticker_id').size().reset_index(name='data_points_count')
print("\nNumber of data points for each ticker ID:")
display(data_points_by_ticker)


# 1. Print the value counts of the `class_label` column, including missing values
print("Value counts of class_label:")
display(df['class_label'].value_counts(dropna=False))

# 2. Calculate and print the percentage of missing values in the `class_label` column
missing_values_count = df['class_label'].isnull().sum()
total_values_count = len(df['class_label'])
percentage_missing = (missing_values_count / total_values_count) * 100
print(f"\nPercentage of missing values in class_label: {percentage_missing:.2f}%")


# 1. Identify feature columns
feature_columns = df.columns.drop(['train_id', 'ticker_id', 't', 'class_label'])
print(f"Number of feature columns: {len(feature_columns)}")

# Separate boolean and non-boolean feature columns
boolean_features = feature_columns[df[feature_columns].dtypes == 'bool'].tolist()
non_boolean_features = feature_columns[df[feature_columns].dtypes != 'bool'].tolist()

print(f"Number of boolean features: {len(boolean_features)}")
print(f"Number of non-boolean features: {len(non_boolean_features)}")

# 2. Calculate and display summary statistics for a subset of boolean features
# Choose a small, representative subset, e.g., the first 10 boolean features
subset_boolean_features = boolean_features[:10]
print("\nSummary statistics for a subset of boolean features (proportion of True values):")
display(df[subset_boolean_features].mean()) # Mean of boolean is the proportion of True

# 3. Calculate and display summary statistics for non-boolean feature columns, if any
if non_boolean_features:
    print("\nSummary statistics for non-boolean feature columns:")
    display(df[non_boolean_features].describe())
else:
    print("\nNo non-boolean feature columns found.")

# 4. Briefly describe the nature of these features
print("\nNature of features:")
print("- The majority of features are boolean, likely representing the state of various technical indicators or conditions.")
if non_boolean_features:
    print("- There are also a few non-boolean features (floats), whose summary statistics provide insight into their distribution and range.")
else:
    print("- All features except the identifying columns and target are boolean.")

print("\nStrategies for handling high dimensionality:")
print("- **Feature Selection:** Techniques like filter methods (e.g., correlation with target), wrapper methods (e.g., recursive feature elimination), or embedded methods (e.g., using L1 regularization) can help identify the most relevant features.")
print("- **Feature Extraction:** Dimensionality reduction techniques such as Principal Component Analysis (PCA) or t-SNE can create a smaller set of new features that capture most of the variance in the original data.")
print("- **Domain Knowledge:** Leveraging understanding of financial markets and technical indicators to select or engineer relevant features.")
print("- **Tree-based models:** Models like Random Forests or Gradient Boosting Machines can handle high-dimensional data relatively well and provide feature importance scores.")
print("- **Regularization:** Using models with L1 or L2 regularization can help prevent overfitting in high-dimensional spaces.")


# 1. Group the DataFrame by ticker_id
grouped_by_ticker = df.groupby('ticker_id')

# 2. Iterate through each ticker_id group
for ticker_id, group_df in grouped_by_ticker:
    print(f"\nAnalyzing Ticker ID: {ticker_id}")

    # a. Sort the data by the 't' column
    sorted_group_df = group_df.sort_values(by='t')

    # b. Observe the temporal distribution of the target variable (if non-missing)
    # Filter for non-missing class_label values
    non_missing_labels = sorted_group_df.dropna(subset=['class_label'])

    print(f"Number of data points for Ticker ID {ticker_id}: {len(sorted_group_df)}")
    print(f"Number of non-missing class_label entries for Ticker ID {ticker_id}: {len(non_missing_labels)}")

    if not non_missing_labels.empty:
        print("Temporal distribution of non-missing class_label entries:")
        display(non_missing_labels[['t', 'class_label']])
    else:
        print("No non-missing class_label entries for this ticker ID.")

# 3. Summarize observations (will be done after the loop)
print("\nSummary of Observations:")
print("- Each ticker ID has 322 data points, covering a consistent time range as observed in the previous step.")
print("- The vast majority of `class_label` entries are missing for all ticker IDs.")
print("- The non-missing `class_label` entries are sparse and their temporal distribution varies across tickers.")
print("- Ticker IDs 1, 2, 3, 4, and 5 have very few or no non-missing `class_label` entries.")
print("- Ticker ID 6 has a larger number of non-missing `class_label` entries compared to others, and they appear distributed throughout its time series.")
print("- Due to the sparsity of non-missing labels and the high dimensionality of features, identifying trends or seasonality in features related to the target is not feasible at this stage without further feature selection or aggregation.")


# 1. Filter DataFrame for non-missing class_label
df_labeled = df.dropna(subset=['class_label']).copy()

# 2. Convert categorical class_label to numerical representation
# Using Label Encoding as the labels might have an ordinal relationship (e.g., HH > HL > LH > LL)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df_labeled['class_label_encoded'] = label_encoder.fit_transform(df_labeled['class_label'])

# 3. Select a manageable subset of feature columns
# Selecting a random subset of boolean features and all non-boolean features
import numpy as np

# Separate boolean and non-boolean feature columns again in the labeled data
feature_columns = df_labeled.columns.drop(['train_id', 'ticker_id', 't', 'class_label', 'class_label_encoded'])
boolean_features = feature_columns[df_labeled[feature_columns].dtypes == 'bool'].tolist()
non_boolean_features = feature_columns[df_labeled[feature_columns].dtypes != 'bool'].tolist()

# Randomly select a subset of boolean features (e.g., 50 features)
np.random.seed(42) # for reproducibility
subset_boolean_features = np.random.choice(boolean_features, size=min(50, len(boolean_features)), replace=False).tolist()

selected_features = subset_boolean_features + non_boolean_features

print(f"Number of selected features for correlation analysis: {len(selected_features)}")
print(f"Selected features: {selected_features[:10]}...") # Print first 10 selected features as example

# 4. Calculate correlation matrix between selected features and the encoded target variable
correlation_with_target = df_labeled[selected_features + ['class_label_encoded']].corr()['class_label_encoded'].drop('class_label_encoded')

print("\nCorrelation with encoded class_label:")
display(correlation_with_target.sort_values(ascending=False))

# 5. Calculate the correlation matrix between the selected feature subset itself
correlation_matrix_features = df_labeled[selected_features].corr()

print("\nCorrelation matrix of selected features:")
# Displaying the full matrix might be too large, display a subset or summary
# Displaying the top/bottom correlations or correlations with a threshold
# For simplicity and readability, let's just show the descriptive statistics of correlations
print("\nDescriptive statistics of correlations between selected features:")
display(correlation_matrix_features.stack().describe())

# 6. Display or visualize the resulting correlation matrices - done in step 4 and 5 by displaying
# Due to size, direct display is more practical than a heatmap for many features.

# 7. Briefly summarize any notable correlations found.
print("\nSummary of notable correlations:")
print(f"- The correlation with the encoded class label ranges from {correlation_with_target.min():.4f} to {correlation_with_target.max():.4f}.")
print("- Features with the highest positive correlation with the encoded class label are:")
display(correlation_with_target.nlargest(5))
print("- Features with the highest negative correlation with the encoded class label are:")
display(correlation_with_target.nsmallest(5))
print("- The descriptive statistics of correlations between selected features show the range and distribution of relationships among features.")
print("- Given the small number of labeled samples ({}) and the random subset of features, these correlations should be interpreted with caution and may not be statistically significant.".format(len(df_labeled)))


import matplotlib.pyplot as plt
import seaborn as sns

# 1. Create a bar plot of the distribution of the `class_label` column
plt.figure(figsize=(8, 6))
sns.countplot(x='class_label', data=df_labeled, palette='viridis')
plt.title('Distribution of Class Labels (Non-Missing)')
plt.xlabel('Class Label')
plt.ylabel('Count')
plt.show()


# 2. Generate scatter plots of the non-boolean features against the encoded `class_label_encoded`
non_boolean_features = ['momentum', 'ratio', 'sm_momentum', 'sm_ratio']
encoded_target = 'class_label_encoded'

plt.figure(figsize=(15, 5))
for i, feature in enumerate(non_boolean_features):
    plt.subplot(1, len(non_boolean_features), i + 1)
    sns.scatterplot(x=feature, y=encoded_target, data=df_labeled)
    plt.title(f'{feature} vs Encoded Class Label')
    plt.xlabel(feature)
    plt.ylabel('Encoded Class Label')

plt.tight_layout()
plt.show()


# 3. Plot the time series of a few selected boolean features for Ticker ID 6
# Select Ticker ID 6 data
ticker_6_df = df[df['ticker_id'] == 6].sort_values(by='t').copy()

# Select a few boolean features to plot (e.g., 5 random boolean features)
# Ensure these features exist in the dataframe
feature_columns = ticker_6_df.columns.drop(['train_id', 'ticker_id', 't', 'class_label'])
boolean_features = feature_columns[ticker_6_df[feature_columns].dtypes == 'bool'].tolist()

# Use the same random seed for reproducibility if needed, but a different set might be more insightful
np.random.seed(10) # using a different seed
selected_boolean_ts_features = np.random.choice(boolean_features, size=min(5, len(boolean_features)), replace=False).tolist()

print(f"\nPlotting time series for Ticker ID 6 for features: {selected_boolean_ts_features}")

plt.figure(figsize=(15, 10))
for i, feature in enumerate(selected_boolean_ts_features):
    plt.subplot(len(selected_boolean_ts_features), 1, i + 1)
    plt.plot(ticker_6_df['t'], ticker_6_df[feature].astype(int)) # Plot boolean as int (0 or 1)
    plt.title(f'Time Series of {feature} for Ticker ID 6')
    plt.xlabel('Time')
    plt.ylabel(feature)
    plt.yticks([0, 1], ['False', 'True']) # Set y-ticks for boolean values

plt.tight_layout()
plt.show()


# 4. Create a heatmap of the correlation matrix for the selected features
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_features, cmap='coolwarm', annot=False) # annot=False due to large number of features
plt.title('Correlation Matrix of Selected Features')
plt.show()

