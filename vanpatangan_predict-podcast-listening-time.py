import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')


# Read in data
train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


def check(df):
    """
    Generates a concise summary of DataFrame columns.
    """
    # Use list comprehension to iterate over each column
    summary = [
        [col, df[col].dtype, df[col].count(), df[col].nunique(), df[col].isnull().sum(), df.duplicated().sum()]
        for col in df.columns
    ]

    # Create a DataFrame from the list of lists
    df_check = pd.DataFrame(summary, columns=["column", "dtype", "instances", "unique", "sum_null", "duplicates"])

    return df_check


print("Training Data Summary")
display(check(train))
display(train.head())

print("Test Data Summary")
display(check(test))
display(test.head())


# Select only numeric columns
numeric_cols = train.select_dtypes(include=['float64', 'int64']).columns

# Set the number of plots per row/column
n_cols = 3
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

# Set figure size
plt.figure(figsize=(5 * n_cols, 4 * n_rows))

# Plot each numeric column
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.histplot(train[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')

plt.tight_layout()
plt.show()


# Listening_Time_minutes across different Genre categories
plt.figure(figsize=(12, 6))
genre_order = train.groupby('Genre')['Listening_Time_minutes'].median().sort_values(ascending=False).index
sns.boxplot(data=train, x='Genre', y='Listening_Time_minutes', order=genre_order)
plt.show()


# Episode Length vs. Listening Time
plt.figure(figsize=(10, 6))
sns.scatterplot(data=train, x='Episode_Length_minutes', y='Listening_Time_minutes', alpha=0.5)
plt.title('Episode Length vs. Listening Time')
plt.xlabel('Episode Length (minutes)')
plt.ylabel('Listening Time (minutes)')
plt.show()


# Correlation Matrix of Numerical Features
plt.figure(figsize=(8, 6))
sns.heatmap(train[['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Listening_Time_minutes']].corr(), annot=True, cmap='RdBu', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()


# Listening Time by Episode Sentiment
plt.figure(figsize=(12, 6))
sns.boxplot(data=train, x='Episode_Sentiment', y='Listening_Time_minutes')
plt.title('Listening Time by Episode Sentiment')
plt.show()


# Compute the average listening time per day
avg_listening = train.groupby('Publication_Day')['Listening_Time_minutes'].mean().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)

# Create the lollipop chart
plt.figure(figsize=(12, 6))
plt.stem(avg_listening.index, avg_listening.values, basefmt=" ", use_line_collection=True)
plt.scatter(avg_listening.index, avg_listening.values, color='purple', s=100)

# Style
plt.title('Average Listening Time by Publication Day', fontsize=14)
plt.xlabel('Publication Day')
plt.ylabel('Average Listening Time (minutes)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



from sklearn.impute import SimpleImputer

# Define numerical columns with missing values
num_cols = ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']

# Imputer (fit on train, apply to both)
imputer = SimpleImputer(strategy='median')
train[num_cols] = imputer.fit_transform(train[num_cols])
test[num_cols] = imputer.transform(test[num_cols]) # Use train's median


# Convert Publication_Time to numerical bins 
time_mapping = {
    'Night': 0,
    'Morning': 6,
    'Afternoon': 12,
    'Evening': 18
}

train['Publication_Hour'] = train['Publication_Time'].map(time_mapping)
test['Publication_Hour'] = test['Publication_Time'].map(time_mapping)

# Drop original column
train.drop('Publication_Time', axis=1, inplace=True)
test.drop('Publication_Time', axis=1, inplace=True)


# Episode Length / Popularity
train['Length_Host_Popularity'] = train['Episode_Length_minutes'] * train['Host_Popularity_percentage']
test['Length_Host_Popularity'] = test['Episode_Length_minutes'] * test['Host_Popularity_percentage']
train['Length_Guest_Popularity'] = train['Episode_Length_minutes'] * train['Guest_Popularity_percentage']
test['Length_Guest_Popularity'] = test['Episode_Length_minutes'] * test['Guest_Popularity_percentage']

# Ads / Episode Length 
train['Ads_Length_Ratio'] = train['Number_of_Ads'] / (train['Episode_Length_minutes'] + 1e-6)  
test['Ads_Length_Ratio'] = test['Number_of_Ads'] / (test['Episode_Length_minutes'] + 1e-6)


# Encode categorical variables 

# Target encoding for high cardinality columns (Too many unique values for one hot encoding)
for col in ['Podcast_Name', 'Episode_Title']:
    mean_target = train.groupby(col)['Listening_Time_minutes'].mean()
    train[col + '_encoded'] = train[col].map(mean_target)
    test[col + '_encoded'] = test[col].map(mean_target).fillna(mean_target.mean())  # Fill unseen categories

# One hot encoding for low cardinality columns
cat_cols = ['Genre', 'Publication_Day', 'Episode_Sentiment']
train = pd.get_dummies(train, columns=cat_cols, drop_first=True)
test = pd.get_dummies(test, columns=cat_cols, drop_first=True)

# Align train and test columns (in case of mismatched categories)
train, test = train.align(test, join='left', axis=1, fill_value=0)

# Drop unnecessary columns
test_id = test['id'].copy() # For submission later
columns_to_drop = ['Podcast_Name','Episode_Title', 'id']

train.drop(columns=columns_to_drop, inplace=True)
test.drop(columns=columns_to_drop, inplace=True)


# Popularity Ratio / Sentiment
train['Host_Popularity_Sentiment_Positive'] = train['Host_Popularity_percentage'] * train.get('Episode_Sentiment_Positive', 0)
test['Host_Popularity_Sentiment_Positive'] = test['Host_Popularity_percentage'] * test.get('Episode_Sentiment_Positive', 0)


from sklearn.model_selection import train_test_split

X = train.drop(columns=['Listening_Time_minutes'])  # Features
y = train['Listening_Time_minutes']  # Target

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Initialize KFold for crossvalidation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    # Split data for this fold
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Define parameters
params = {
    'objective': 'regression',          # For predicting target var
    'metric': 'rmse',                   # Root Mean Squared Error
    'boosting_type': 'gbdt',            # Gradient Boosted Decision Trees
    'num_leaves': 74,                   # Reduced to prevent overfitting
    'learning_rate': 0.09,              # Smaller = slower, more robust
    'feature_fraction': 0.9,            # Feature subsampling
    'bagging_freq': 7,
    'min_child_samples': 93,
    'max_depth': 8,
    'verbose': -1                       # Suppress warnings
}

# Train with early stopping using callback
model = lgb.train(
    params,
    train_data,
    num_boost_round=2000,               # Max iterations
    valid_sets=[val_data],              # Validation set for monitoring
    valid_names=['validation'],         # Name for validation set in output
    callbacks=[
        lgb.early_stopping(stopping_rounds=100), # Stop if no improvement for 100 rounds
        lgb.log_evaluation(period=10)            # Print progress every 10 rounds
    ]
)

 # Predict and calculate RMSE
y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
rmse_scores.append(rmse)
    
print(f"Fold {fold} RMSE: {rmse:.4f}")
print(f"Fold {fold} Best Iteration: {model.best_iteration}\n")

 # Cross validated performance
print(f"Average Validation RMSE: {np.mean(rmse_scores):.4f} Â± {np.std(rmse_scores):.4f}")



lgb.plot_importance(model, max_num_features=10)


# Align columns
X_val = X_val[X_train.columns]
test = test[X_train.columns]

# Predict on test set
test_pred = model.predict(test, num_iteration=model.best_iteration)


# Create submission
submission = pd.DataFrame({'id': test_id, 'Listening_Time_minutes': test_pred})
submission.to_csv('submission.csv', index=False)
submission.head()

