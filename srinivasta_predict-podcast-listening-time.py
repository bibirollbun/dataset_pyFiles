import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")

# Store the original 'id' values from the test DataFrame
test_ids = test['id']  # Store test 'id' values


# Handle missing values (imputation)
for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']:
    train[col] = train[col].fillna(train[col].mean())
    test[col] = test[col].fillna(train[col].mean())  # Use train data statistics for test imputation

# Feature engineering (Publication_Hour and Episode Number)
def extract_hour(time_str):
    if time_str == 'Morning':
        return 8
    elif time_str == 'Afternoon':
        return 14
    elif time_str == 'Evening':
        return 19
    elif time_str == 'Night':
        return 22
    else:
        return None

def extract_episode_number(title):
    try:
        return int(title.split(' ')[1])
    except (IndexError, ValueError):
        return 0

train['Publication_Hour'] = train['Publication_Time'].apply(extract_hour)
test['Publication_Hour'] = test['Publication_Time'].apply(extract_hour)

train['Episode_Number'] = train['Episode_Title'].apply(extract_episode_number)
test['Episode_Number'] = test['Episode_Title'].apply(extract_episode_number)

# One-hot encode categorical features
categorical_features = ['Genre', 'Episode_Sentiment', 'Podcast_Name', 'Publication_Day', 'Publication_Time']
train = pd.get_dummies(train, columns=categorical_features, dummy_na=False, drop_first=False)
test = pd.get_dummies(test, columns=categorical_features, dummy_na=False, drop_first=False)

# Align columns in train and test to ensure consistency
train, test = train.align(test, join='outer', axis=1, fill_value=0)

# Extract target variable and features
y = train['Listening_Time_minutes']
X = train.drop(columns=['Listening_Time_minutes', 'Episode_Title', 'id'])

# Remove 'Listening_Time_minutes' from the test set (if present)
if 'Listening_Time_minutes' in test.columns:
    test = test.drop(columns=['Listening_Time_minutes'])

# Drop 'Episode_Title' and 'id' from the test set
test = test.drop(columns=['Episode_Title', 'id'])  # Dropping 'id' here

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
test_predictions = model.predict(test)

# Create submission file using the original test_ids
submission_df = pd.DataFrame({'id': test_ids, 'Listening_Time_minutes': test_predictions})
submission_df.to_csv('submission.csv', index=False)



display(submission_df)

