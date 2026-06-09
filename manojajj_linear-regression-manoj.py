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


# Visualization (optional but helpful)
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn for modeling and evaluation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Warnings
import warnings
warnings.filterwarnings("ignore")


# Load the training data
train_data = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')

# Display the first few rows to inspect the data
train_data.head()


train_data.info()


train_data.shape


# Function to replace NaN values with median for float columns
def replace_nan_with_median(df):
    # Iterate through all columns
    for column in df.columns:
        # Check if the column's data type is float (either float64 or float32)
        if df[column].dtype in ['float64', 'float32']:
            # Replace NaN values in that column with its median
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)
    
    return df

# Apply the function to replace NaN values
train_data = replace_nan_with_median(train_data)

 # Display the first few rows to confirm
train_data.head()


train_data['Genre'].value_counts()


# Mapping dictionary for 'Genre' to numeric values
genre_mapping = {
    'Sports': 1,
    'Technology': 2,
    'True Crime': 3,
    'Lifestyle': 4,
    'Comedy': 5,
    'Business': 6,
    'Health': 7,
    'News': 8,
    'Music': 9,
    'Education': 10
}

# Apply the mapping to the 'Genre' column
train_data['Genre'] = train_data['Genre'].map(genre_mapping)

train_data.head()


train_data['Publication_Day'].value_counts()


# Mapping dictionary for 'Publication_Day' to numeric values
publication_day_mapping = {
    'Sunday': 1,
    'Monday': 2,
    'Tuesday': 3,
    'Wednesday': 4,
    'Thursday': 5,     
    'Friday': 6,
    'Saturday': 7
}

# Apply the mapping to the 'Publication_Day' column
train_data['Publication_Day'] = train_data['Publication_Day'].map(publication_day_mapping)

train_data.head()


train_data['Publication_Time'].value_counts()


# Mapping dictionary for 'Publication_Time' to numeric values
publication_time_mapping = {
    'Night': 1,
    'Evening': 2,
    'Afternoon': 3,
    'Morning': 4
}

# Apply the mapping to the 'Publication_Time' column
train_data['Publication_Time'] = train_data['Publication_Time'].map(publication_time_mapping)

train_data.head()


train_data['Episode_Sentiment'].value_counts()


# Mapping dictionary for 'Episode_Sentiment' to numeric values
episode_sentiment_mapping = {
    'Neutral': 1,
    'Negative': 2,
    'Positive': 3
}

# Apply the mapping to the 'Episode_Sentiment' column
train_data['Episode_Sentiment'] = train_data['Episode_Sentiment'].map(episode_sentiment_mapping)

train_data.head()


new_train_data = train_data.drop(columns=['id', 'Podcast_Name', 'Episode_Title'])


# Calculate the correlation matrix
correlation_matrix = new_train_data.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# Draw the heatmap with annotations
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Set the title
plt.title('Correlation Matrix for train_data')

# Show the plot
plt.show()


X = new_train_data.drop(columns=['Listening_Time_minutes', 'Genre', 'Publication_Day'])
y = new_train_data['Listening_Time_minutes']


# Split data into training and validation sets (80% train, 20% test)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2)


# Initialize scaler
scaler = StandardScaler()

# Fit only on training data and transform both training and validation
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# Train the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


# Predict on validation data
y_pred = lr_model.predict(X_val)


# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Linear Regression RMSE: {rmse:.4f}")


# Plot Actual vs Predicted
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_val, y=y_pred, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', linestyle='--')
plt.xlabel('Actual Listening Time')
plt.ylabel('Predicted Listening Time')
plt.title('Actual vs Predicted Listening Time')
plt.grid(True)
plt.show()


# Plot Residuals
residuals = y_val - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=50, kde=True)
plt.xlabel('Residuals')
plt.title('Residuals Distribution')
plt.grid(True)
plt.show()


# 1. Load the test data
test_data = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

test_data.head()


# Apply mappings (make sure these mappings are already defined)
test_data['Publication_Time'] = test_data['Publication_Time'].map(publication_time_mapping)
test_data['Episode_Sentiment'] = test_data['Episode_Sentiment'].map(episode_sentiment_mapping)

# Optionally fill any NaNs after mapping
test_data.fillna(0, inplace=True)  # or use median for floats if needed

# Preview
test_data.head()


# Drop unnecessary columns
new_test_data = test_data.drop(columns=['id', 'Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day'])


# Now use the same scaler to transform the test data
new_test_data_scaled = scaler.transform(new_test_data)


test_predictions = lr_model.predict(new_test_data_scaled)

test_predictions = np.clip(test_predictions, 0, None)

# Create submission
submission = pd.DataFrame({
    'id': test_data['id'],
    'Listening_Time_minutes': test_predictions
})
submission.to_csv('submission.csv', index=False)

