# Import required libraries and set configurations
import warnings
warnings.filterwarnings('ignore')

from prettytable import PrettyTable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

plt.rc('figure', autolayout=True, figsize=(12, 6), titlesize=18, titleweight='bold')
plt.rc('axes', labelweight='bold', labelsize='large', titlesize=16, titleweight='bold', titlepad=10)


# Load datasets from Kaggle input folder
df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


df_train.head()


df_test.head()


df_train.shape, df_test.shape


df_train.info()


# Define Our Target
target = 'Listening_Time_minutes'

# Define a helper function to calculate missing value percentage
def missing_percentage(df, col):
    """Calculate the percentage of missing values for a DataFrame column."""
    return np.round(100 - df[col].count() / len(df) * 100, 1)

# Build a summary table for each feature
table = PrettyTable()
table.field_names = ['Feature', 'Data Type', 'Train Missing %', 'Test Missing %', 'Discrete Ratio (Train)']

rows = []
for column in df_train.columns:
    data_type = str(df_train[column].dtype)
    train_missing = missing_percentage(df_train, column)
    test_missing = missing_percentage(df_test, column) if column != target else "NA"
    discrete_ratio = np.round(df_train[column].nunique() / len(df_train), 4)
    
    rows.append([column, data_type, train_missing, test_missing, discrete_ratio])

table.add_rows(rows)
print(table)


# Drop the 'id' column as it's not needed for analysis
df_train.drop(columns=['id'], inplace=True)
df_test.drop(columns=['id'], inplace=True)


# Function to compute counts and percentages for a given column
def get_values(df, column_name, top_n=None, sort_by_column_name=False):
    vc_df = (
        df.groupby(column_name)
        .size()
        .reset_index(name='Count')
        .assign(Percentage=lambda x: (x['Count'] / x['Count'].sum()) * 100)
    )
    if sort_by_column_name:
        vc_df = vc_df.sort_values(by=column_name)

    if top_n is not None:
        vc_df = vc_df.sort_values(by='Count', ascending=False).head(top_n)

    return vc_df

# Function to plot a pie chart for the top categories
def plot_value_pie(df, column_name, top_n=9, sort_by_column_name=False):
    vc_df = get_values(df,column_name, top_n, sort_by_column_name)
    
    colors = plt.cm.Greys(np.linspace(0.9, 0.3, len(vc_df)))

    vc_df.set_index(column_name).plot.pie(
        y='Count', figsize=(5, 5), legend=False, ylabel='', colors=colors
    )
    plt.show()

# Function to plot a bar chart with count and percentage annotations
def plot_value_counts_bar(df, column_name, top_n=9, sort_by_column_name=False):
    vc_df = get_values(df,column_name, top_n, sort_by_column_name)
    
    vc_df = vc_df.sort_values('Count', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=vc_df,
        x=column_name,
        y='Count',
        palette='Greys'
    )
    # Annotate bars with count and percentage
    for patch, (_, row) in zip(ax.patches, vc_df.iterrows()):
        x_center = patch.get_x() + patch.get_width() / 2
        height = patch.get_height()
        label = f"{row['Count']:.0f} ({row['Percentage']:.2f}%)"
        ax.text(x_center, height + max(vc_df['Count']) * 0.02, label,
                ha='center', va='bottom')
    plt.show()


# Visualize the distribution of the 'Genre' feature using a bar chart
plot_value_counts_bar(df_train, 'Genre')


# Visualize categorical distributions
plot_value_counts_bar(df_train, 'Podcast_Name')


plot_value_counts_bar(df_train, 'Publication_Day')


plot_value_pie(df_train, 'Publication_Time')


# Plot distribution of the target variable: Listening Time (in minutes)
sns.histplot(df_train['Listening_Time_minutes'], bins=30, kde=True, color='black')
plt.title("Distribution of Listening Time (mins)")
plt.xlabel("Listening Time (mins)")
plt.ylabel("Frequency")
plt.show()


# Select only numeric columns for correlation analysis
numeric_df = df_train.select_dtypes(include=['number'])

# Calculate correlation on numeric columns
corr = numeric_df.corr()

# Plot the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues')
plt.show()


# Identify common columns between train and test
com_cols = df_train.columns.intersection(df_test.columns)
num_cols = df_train[com_cols].select_dtypes(include=['int64', 'float64']).columns
cat_cols = df_train[com_cols].select_dtypes(include=['object']).columns

# Impute missing values: numerical features with the median and categorical with the mode
df_train[num_cols] = df_train[num_cols].fillna(df_train[num_cols].median())
df_test[num_cols] = df_test[num_cols].fillna(df_test[num_cols].median())

df_train[cat_cols] = df_train[cat_cols].fillna(df_train[cat_cols].mode().iloc[0])
df_test[cat_cols] = df_test[cat_cols].fillna(df_test[cat_cols].mode().iloc[0])


# Encode key categorical features using LabelEncoder
le = LabelEncoder()

label_cols = ['Podcast_Name', 'Episode_Title']

for col in label_cols:
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])


# --- Feature Engineering ---

# Map Publication_Day to a numerical value for cyclical encoding
days_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                'Friday': 4, 'Saturday': 5, 'Sunday': 6}
df_train['day_num'] = df_train['Publication_Day'].map(days_mapping)
df_test['day_num'] = df_test['Publication_Day'].map(days_mapping)

# Cyclical features for day
df_train['day_sin'] = np.sin(2 * np.pi * df_train['day_num'] / 7)
df_train['day_cos'] = np.cos(2 * np.pi * df_train['day_num'] / 7)
df_test['day_sin'] = np.sin(2 * np.pi * df_test['day_num'] / 7)
df_test['day_cos'] = np.cos(2 * np.pi * df_test['day_num'] / 7)

# Weekend indicator from Publication_Day
df_train['is_weekend'] = df_train['Publication_Day'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)
df_test['is_weekend'] = df_test['Publication_Day'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)

# Popularity difference between host and guest
df_train['popularity_diff'] = df_train['Host_Popularity_percentage'] - df_train['Guest_Popularity_percentage']
df_test['popularity_diff'] = df_test['Host_Popularity_percentage'] - df_test['Guest_Popularity_percentage']

# Sentiment mapping (e.g., Negative: -1, Neutral: 0, Positive: 1)
sentiment_mapping = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
df_train['sentiment_score'] = df_train['Episode_Sentiment'].map(sentiment_mapping)
df_test['sentiment_score'] = df_test['Episode_Sentiment'].map(sentiment_mapping)

# Text-based feature: count the number of words in the episode title
df_train['title_word_count'] = df_train['Episode_Title'].apply(lambda x: len(str(x).split()))
df_test['title_word_count'] = df_test['Episode_Title'].apply(lambda x: len(str(x).split()))


display(df_train['Episode_Length_minutes'].head())


fig, axes = plt.subplots(2, 2, figsize=(20, 20))

# 1. Scatter Plot: Episode_Length_minutes vs. Listening_time_minutes
sns.scatterplot(
    data=df_train, 
    x="Episode_Length_minutes", 
    y="Listening_Time_minutes", 
    hue="Episode_Sentiment",  
    ax=axes[0, 0], 
    s=30
)
axes[0, 0].set_title("Episode Length vs. Listening Time")

# 2. Distribution Plot: Histogram with KDE of Episode_Length_minutes
sns.histplot(
    data=df_train, 
    x="Episode_Length_minutes", 
    bins=30, 
    kde=True, 
    ax=axes[0, 1]
)
axes[0, 1].set_title("Distribution of Episode Length")

# 3. Boxplot: Outlier Detection in Episode_Length_minutes
sns.boxplot(
    data=df_train, 
    x="Episode_Length_minutes", 
    ax=axes[1, 0]
)
axes[1, 0].set_title("Boxplot of Episode Length")

# 4. Scatter Plot: Log-Transformed Episode_Length_minutes vs. Listening_time_minutes
# Create the log-transformed feature (using log1p to handle potential zeros)
df_train['Episode_Length_log'] = np.log1p(df_train["Episode_Length_minutes"])
sns.scatterplot(
    data=df_train, 
    x="Episode_Length_log", 
    y="Listening_Time_minutes", 
    hue="Episode_Sentiment", 
    ax=axes[1, 1], 
    s=30
)
axes[1, 1].set_title("Log-Transformed Episode Length vs. Listening Time")

plt.tight_layout()
plt.show()


# Define bins and labels for Episode_Length_minutes
bins = [0, 30, 60, 90, np.inf]
labels = ['short', 'medium', 'long', 'very_long']

# Create binned feature for both train and test sets
df_train['Episode_Length_bin'] = pd.cut(df_train['Episode_Length_minutes'], bins=bins, labels=labels)
df_test['Episode_Length_bin'] = pd.cut(df_test['Episode_Length_minutes'], bins=bins, labels=labels)

# One-hot encode the binned episode length feature
df_train = pd.get_dummies(df_train, columns=['Episode_Length_bin'], drop_first=True)
df_test = pd.get_dummies(df_test, columns=['Episode_Length_bin'], drop_first=True)


# Identify any remaining categorical columns to one-hot encode
remaining_cat_cols = df_train.select_dtypes(include=['object']).columns.tolist()

if remaining_cat_cols:
    df_train = pd.get_dummies(df_train, columns=remaining_cat_cols, drop_first=True)
    df_test = pd.get_dummies(df_test, columns=remaining_cat_cols, drop_first=True)

# Align train and test sets to ensure they have the same features
df_train, df_test = df_train.align(df_test, join='left', axis=1, fill_value=0)


X = df_train.drop(columns=['Listening_Time_minutes'])
y = df_train['Listening_Time_minutes']

# Split the training data for hyperparameter tuning
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up a parameter grid for XGBoost
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Initialize and run GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb.XGBRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("Best parameters found:", grid_search.best_params_)

# Evaluate on the validation set
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val)
val_mse = mean_squared_error(y_val, y_val_pred)
print("Validation Mean Squared Error:", val_mse)

# Retrain on the full training dataset with the best parameters
final_model = grid_search.best_estimator_
final_model.fit(X, y)

# Generate predictions on the test dataset
if 'Listening_Time_minutes' in df_test.columns:
    df_test = df_test.drop(columns=['Listening_Time_minutes'])

test_predictions = final_model.predict(df_test)


df_sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


df_sub['Listening_Time_minutes'] = test_predictions
df_sub.to_csv('submission.csv', index=False)
print('Mission Complete \nShape:', df_sub.shape)

