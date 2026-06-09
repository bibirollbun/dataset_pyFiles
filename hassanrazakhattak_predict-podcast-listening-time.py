# Importing Libraries
# First, we import all necessary Python libraries for data processing, visualization, modeling, and evaluation

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')

train.head()


test.head()


train.info()


test.info()


train.isnull().sum()


test.isnull().sum()


# For train
train['Episode_Length_minutes'] = train['Episode_Length_minutes'].fillna(train['Episode_Length_minutes'].mean())
train['Guest_Popularity_percentage'] = train['Guest_Popularity_percentage'].fillna(train['Guest_Popularity_percentage'].mean())
train['Number_of_Ads'] = train['Number_of_Ads'].fillna(train['Number_of_Ads'].mean())

#remove column with name id from train
train.drop(columns=['id'], inplace = True ,errors = 'ignore')

# For test
test['Episode_Length_minutes'] = test['Episode_Length_minutes'].fillna(test['Episode_Length_minutes'].mean())
test['Guest_Popularity_percentage'] = test['Guest_Popularity_percentage'].fillna(test['Guest_Popularity_percentage'].mean())

#remove column with name id from test
test.drop(columns=['id'], inplace = True ,errors = 'ignore')


# Automatically detect columns
number_columns = train.select_dtypes(include=(['int64', 'float64'])).columns.tolist()
catogorical_columns = train.select_dtypes(include=(['object'])).columns.tolist()



# Check missing values in numerical columns
train[number_columns].isnull().mean().sort_values(ascending=False)

# Basic stats
train[number_columns].describe()


# Visualize distributions

for col in number_columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(train[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()



# Check missing values in categorical columns
train[catogorical_columns].isnull().mean().sort_values(ascending=False)

# Unique values and top categories
for col in catogorical_columns:
    print(f"\nColumn: {col}")
    print(f"Unique values: {train[col].nunique()}")
    print(train[col].value_counts().head(5))



for col in number_columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(train[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()


# Encode them
le = LabelEncoder()
for col in catogorical_columns:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])



# Prepare train and test 
X = train.drop(['Listening_Time_minutes'], axis=1)
y = train['Listening_Time_minutes']
X_test = test.copy()  # prepare X_test here!



# Label Encode categorical columns
label_encoders = {}

for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])  # Important: transform test data too
    label_encoders[col] = le



# Split train data into training and validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
final_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Train model
final_model.fit(X_train, y_train)
print("âœ… Model training complete!")

# Predict on validation set
y_val_pred = final_model.predict(X_val)
print("ğŸ”® Predicting on validation set...")

# Evaluate on validation set
mse = mean_squared_error(y_val, y_val_pred)
rmse = mse ** 0.5
print(f"ğŸ“ˆ Validation RMSE: {rmse:.4f}")




# Predict on test data
test_predictions = final_model.predict(X_test)
print("ğŸ”® Test set predictions ready!")


# Create a range of IDs starting from 750000
starting_id = 750000
submission = pd.DataFrame({
    'id': range(starting_id, starting_id + len(test_predictions)),
    'Listening_Time_minutes': test_predictions
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

print("ğŸ“„ Submission file saved as 'submission.csv' âœ…")


