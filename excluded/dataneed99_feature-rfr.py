# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

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


train_path = "/kaggle/input/playground-series-s5e4/train.csv"  
test_path = "/kaggle/input/playground-series-s5e4/test.csv" 


train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


train_df.head()
test_df.head()


target_column = "Listening_Time_minutes"
y_train = train_df[target_column]

X_train = train_df.drop(columns=[target_column])
X_test = test_df


numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features =X_test.select_dtypes(exclude=['number']).columns.tolist()
print("Numerical Features:", numerical_features)  
print("Categorical Features:", categorical_features)


# Check for missing values in train dataset
print("Missing values in train dataset:")
print(train_df.isnull().sum())

# Check for missing values in test dataset
print("\nMissing values in test dataset:")
print(test_df.isnull().sum())


train_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].median(), inplace=True)
train_df['Guest_Popularity_percentage'].fillna(train_df['Guest_Popularity_percentage'].median(), inplace=True)

test_df['Episode_Length_minutes'].fillna(test_df['Episode_Length_minutes'].median(), inplace=True)
test_df['Guest_Popularity_percentage'].fillna(test_df['Guest_Popularity_percentage'].median(), inplace=True)


train_df['Number_of_Ads'].fillna(train_df['Number_of_Ads'].mode()[0], inplace=True)


print(train_df.isnull().sum())
print(test_df.isnull().sum())


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Define a custom color palette
custom_palette = ['#3498db', '#e74c3c', '#2ecc71']

# Ensure `X_train` and `X_test` exist before modification
X_train = X_train.copy()
X_test = X_test.copy()

# Add Source column
X_train['Source'] = 'Train'
X_test['Source'] = 'Test'

# Add target to X_train
X_train['Listening_Time_minutes'] = y_train

# Combine train and test for visualization
combined_df = pd.concat([X_train, X_test], ignore_index=True)

def generate_numerical_feature_visualizations(feature_name):
    """Generate box plot and histogram for numerical features."""
    sns.set(style='whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot
    sns.boxplot(data=combined_df, x=feature_name, y="Source", palette=custom_palette, ax=axes[0])
    axes[0].set_xlabel(feature_name)
    axes[0].set_title(f"Box Plot for {feature_name} Across Datasets")

    # Histogram
    sns.histplot(data=X_train, x=feature_name, color=custom_palette[0], kde=True, bins=30, label="Train", alpha=0.6, ax=axes[1])
    sns.histplot(data=X_test, x=feature_name, color=custom_palette[1], kde=True, bins=30, label="Test", alpha=0.6, ax=axes[1])
    axes[1].set_xlabel(feature_name)
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Histogram for {feature_name} (Train vs Test)")
    axes[1].legend(title="Dataset")

    plt.tight_layout()
    plt.show()

def generate_categorical_feature_visualizations(feature_name):
    """Generate box plot for categorical features vs Listening Time."""
    sns.set(style='whitegrid')
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=combined_df, x=feature_name, y="Listening_Time_minutes", palette=custom_palette)
    plt.xlabel(feature_name)
    plt.ylabel("Listening Time (Minutes)")
    plt.title(f"Box Plot for {feature_name} vs Target")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Ensure `numerical_features` and `categorical_features` are defined
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# Remove 'Listening_Time_minutes' from numerical features (it is the target variable)
if 'Listening_Time_minutes' in numerical_features:
    numerical_features.remove('Listening_Time_minutes')

# Generate visualizations
for feature in numerical_features:
    generate_numerical_feature_visualizations(feature)

for feature in categorical_features:
    generate_categorical_feature_visualizations(feature)

# Drop 'Source' column after visualization
X_train.drop(columns=['Source', 'Listening_Time_minutes'], inplace=True)
X_test.drop(columns=['Source'], inplace=True)


numerical_train = X_train[numerical_features]  
numerical_test = X_test[numerical_features]   

train_with_target = numerical_train.copy()
train_with_target['Listening_Time_minutes'] = y_train

test_with_target = numerical_test.copy()

corr_train = train_with_target.corr()
corr_test = test_with_target.corr()

mask_train = np.triu(np.ones_like(corr_train, dtype=bool))
mask_test = np.triu(np.ones_like(corr_test, dtype=bool))

annot_kws = {"size": 16, "rotation": 45}  

plt.figure(figsize=(24, 24))  

plt.subplot(2, 1, 1) 
sns.heatmap(corr_train, mask=mask_train, cmap='viridis', annot=True,
            square=True, linewidths=.5, xticklabels=1, yticklabels=1, annot_kws=annot_kws)
plt.title('Correlation Heatmap - Train Data', fontsize=24)

plt.subplot(2, 1, 2)  
sns.heatmap(corr_test, mask=mask_test, cmap='viridis', annot=True,
            square=True, linewidths=.5, xticklabels=1, yticklabels=1, annot_kws=annot_kws)
plt.title('Correlation Heatmap - Test Data', fontsize=24)

plt.tight_layout()

plt.show()


import numpy as np


# Select numerical features from train & test sets
numerical_train = X_train[numerical_features]
numerical_test = X_test[numerical_features]

# Create copies to avoid modifying original DataFrames
train_with_target = numerical_train.copy()
train_with_target['Listening_Time_minutes'] = y_train  # Add target column

# Compute correlation matrices
corr_train = train_with_target.corr()
corr_test = numerical_test.corr()  # Test set has no target column

# Create masks for upper triangle of the heatmaps
mask_train = np.triu(np.ones_like(corr_train, dtype=bool))
mask_test = np.triu(np.ones_like(corr_test, dtype=bool))

# Heatmap styling
annot_kws = {"size": 12}  

plt.figure(figsize=(18, 16))  # Adjusted size

# Train correlation heatmap
plt.subplot(2, 1, 1)
sns.heatmap(corr_train, mask=mask_train, cmap='viridis', annot=True,
            square=True, linewidths=0.5, xticklabels=True, yticklabels=True, annot_kws=annot_kws)
plt.title('Correlation Heatmap - Train Data', fontsize=18)

# Test correlation heatmap
plt.subplot(2, 1, 2)
sns.heatmap(corr_test, mask=mask_test, cmap='viridis', annot=True,
            square=True, linewidths=0.5, xticklabels=True, yticklabels=True, annot_kws=annot_kws)
plt.title('Correlation Heatmap - Test Data', fontsize=18)

plt.tight_layout()
plt.show()

# -------------------------------------------
# ðŸ”¹ Correlation of Features with Target Only
# -------------------------------------------

# Compute correlation with target variable
corr_train_target = train_with_target.corr()[['Listening_Time_minutes']].T  

plt.figure(figsize=(12, 3))  
sns.heatmap(corr_train_target, cmap='viridis', annot=True,
            square=False, linewidths=0.5, annot_kws=annot_kws,
            cbar=False)

plt.xticks(rotation=45, ha="right")  
plt.title('Feature Correlation with Target (Train Data)')
plt.yticks(rotation=0) 
plt.show()


categorical_pie_features = ["Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]

colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6', '#1abc9c', '#ff5733']

plt.figure(figsize=(16, 12))

for i, feature in enumerate(categorical_pie_features, 1):
    plt.subplot(2, 2, i)  
    counts = X_train[feature].value_counts()
    
    wedges, texts, autotexts = plt.pie(
        counts, labels=counts.index, autopct='%1.1f%%', colors=colors[:len(counts)], 
        startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)  
    plt.title(f"Distribution of {feature}", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()


def feature_eng(df):
    podc_dict = {'Mystery Matters': 0, 'Joke Junction': 1, 'Study Sessions': 2, 'Digital Digest': 3, 'Mind & Body': 4, 'Fitness First': 5, 'Criminal Minds': 6, 'News Roundup': 7, 'Daily Digest': 8, 'Music Matters': 9, 'Sports Central': 10, 'Melody Mix': 11, 'Game Day': 12, 'Gadget Geek': 13, 'Global News': 14, 'Tech Talks': 15, 'Sport Spot': 16, 'Funny Folks': 17, 'Sports Weekly': 18, 'Business Briefs': 19, 'Tech Trends': 20, 'Innovators': 21, 'Health Hour': 22, 'Comedy Corner': 23, 'Sound Waves': 24, 'Brain Boost': 25, "Athlete's Arena": 26, 'Wellness Wave': 27, 'Style Guide': 28, 'World Watch': 29, 'Humor Hub': 30, 'Money Matters': 31, 'Healthy Living': 32, 'Home & Living': 33, 'Educational Nuggets': 34, 'Market Masters': 35, 'Learning Lab': 36, 'Lifestyle Lounge': 37, 'Crime Chronicles': 38, 'Detective Diaries': 39, 'Life Lessons': 40, 'Current Affairs': 41, 'Finance Focus': 42, 'Laugh Line': 43, 'True Crime Stories': 44, 'Business Insights': 45, 'Fashion Forward': 46, 'Tune Time': 47}
    genr_dict = {'True Crime': 0, 'Comedy': 1, 'Education': 2, 'Technology': 3, 'Health': 4, 'News': 5, 'Music': 6, 'Sports': 7, 'Business': 8, 'Lifestyle': 9}
    week_dict = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    time_dict = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
    sent_dict = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    
    # Extract Episode Number before dropping the column
    if 'Episode_Title' in df.columns:
        df['Episode_Num'] = df['Episode_Title'].str[8:].astype('category')
        df = df.drop(columns=['Episode_Title'])  # Drop after extracting
    
    
    # Replace categorical features with numeric values
    df['Genre'] = df['Genre'].replace(genr_dict)
    df['Podcast_Name'] = df['Podcast_Name'].replace(podc_dict)
    df['Publication_Day'] = df['Publication_Day'].replace(week_dict)
    df['Publication_Time'] = df['Publication_Time'].replace(time_dict)
    df['Episode_Sentiment'] = df['Episode_Sentiment'].replace(sent_dict)
    
    # Convert categorical columns to category data type
    df['Genre'] = df['Genre'].astype('category')
    df['Podcast_Name'] = df['Podcast_Name'].astype('category')
    df['Publication_Day'] = df['Publication_Day'].astype('category')
    df['Publication_Time'] = df['Publication_Time'].astype('category')
    df['Episode_Sentiment'] = df['Episode_Sentiment'].astype('category')
    
    # Drop the 'Episode_Title' column after extracting 'Episode_Num'
    df = df.drop(columns=['Episode_Title'], errors = 'ignore')
    return df



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Apply feature engineering function
train_df = feature_eng(train_df)
test_df = feature_eng(test_df)

# Split the data into features (X) and target (y)
X = train_df.drop(columns=['Listening_Time_minutes'])
y = train_df['Listening_Time_minutes']

# Split into training and test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
rf = RandomForestRegressor(random_state=42)

# Hyperparameter tuning using RandomizedSearchCV
param_dist = {
    'n_estimators': np.arange(50, 501, 50),
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': np.arange(10, 101, 10),
    'min_samples_split': np.arange(2, 11),
    'min_samples_leaf': np.arange(1, 11),
    'bootstrap': [True, False]
}

# Setup RandomizedSearchCV with 3-fold cross-validation
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, 
                                   n_iter=3, cv=2, verbose=2, n_jobs=-1, random_state=42)

# Fit the model to training data
random_search.fit(X_train, y_train)

# Get the best estimator from the random search
best_rf_model = random_search.best_estimator_

# Make predictions on the validation set
y_pred = best_rf_model.predict(X_val)

# Evaluate the model
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)

print(f"Best Hyperparameters: {random_search.best_params_}")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")



# --- Apply the SAME preprocessing steps to test_df ---
podc_dict = {'Mystery Matters': 0, 'Joke Junction': 1, 'Study Sessions': 2, 'Digital Digest': 3, 'Mind & Body': 4, 'Fitness First': 5, 'Criminal Minds': 6, 'News Roundup': 7, 'Daily Digest': 8, 'Music Matters': 9, 'Sports Central': 10, 'Melody Mix': 11, 'Game Day': 12, 'Gadget Geek': 13, 'Global News': 14, 'Tech Talks': 15, 'Sport Spot': 16, 'Funny Folks': 17, 'Sports Weekly': 18, 'Business Briefs': 19, 'Tech Trends': 20, 'Innovators': 21, 'Health Hour': 22, 'Comedy Corner': 23, 'Sound Waves': 24, 'Brain Boost': 25, "Athlete's Arena": 26, 'Wellness Wave': 27, 'Style Guide': 28, 'World Watch': 29, 'Humor Hub': 30, 'Money Matters': 31, 'Healthy Living': 32, 'Home & Living': 33, 'Educational Nuggets': 34, 'Market Masters': 35, 'Learning Lab': 36, 'Lifestyle Lounge': 37, 'Crime Chronicles': 38, 'Detective Diaries': 39, 'Life Lessons': 40, 'Current Affairs': 41, 'Finance Focus': 42, 'Laugh Line': 43, 'True Crime Stories': 44, 'Business Insights': 45, 'Fashion Forward': 46, 'Tune Time': 47}
genr_dict = {'True Crime': 0, 'Comedy': 1, 'Education': 2, 'Technology': 3, 'Health': 4, 'News': 5, 'Music': 6, 'Sports': 7, 'Business': 8, 'Lifestyle': 9}
week_dict = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
time_dict = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
sent_dict = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    
# Only extract Episode_Num if Episode_Title exists
if 'Episode_Title' in test_df.columns:
    test_df['Episode_Num'] = test_df['Episode_Title'].str.extract(r'(\d+)').astype(float)
    test_df.drop('Episode_Title', axis=1, inplace=True)
else:
    print("No 'Episode_Title' column found. Skipping extraction.")

# Replace categorical features using the same dictionaries you used for train
test_df['Genre'] = test_df['Genre'].replace(genr_dict)
test_df['Podcast_Name'] = test_df['Podcast_Name'].replace(podc_dict)
test_df['Publication_Day'] = test_df['Publication_Day'].replace(week_dict)
test_df['Publication_Time'] = test_df['Publication_Time'].replace(time_dict)
test_df['Episode_Sentiment'] = test_df['Episode_Sentiment'].replace(sent_dict)

# Make sure the columns order matches
X_test = test_df[X_train.columns]  # Important: Match train feature columns exactly



# Predict on test data
test_preds = random_search.predict(X_test)

# Prepare submission
submission = pd.DataFrame({
    "id": test_df["id"],  # or whatever your ID column is called
    "Listening_Time_minutes": test_preds
})

# Save to CSV
submission.to_csv("submission.csv", index=False)
print("âœ… Submission file 'submission.csv' created successfully!")





