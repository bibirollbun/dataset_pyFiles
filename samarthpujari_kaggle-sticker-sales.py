import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder


# Load data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


# Explore the data
print("Train Data Head:")
train_df.head()


print("Test Data Head:")
test_df.head()


# Check for missing values
print("Missing values in train data:")
train_df.isnull().sum()


print("Missing values in test data:")
test_df.isnull().sum()


# Drop rows with missing target values
train_df = train_df.dropna(subset=['num_sold'])


# Convert 'date' to datetime and extract features
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day

test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day


# Drop the original 'date' column
train_df = train_df.drop(columns=['date'])
test_df = test_df.drop(columns=['date'])


# Encode categorical features
le = LabelEncoder()
if 'country' in train_df.columns:
    train_df['country'] = le.fit_transform(train_df['country'])
    if 'country' in test_df.columns:
        test_df['country'] = le.transform(test_df['country'])

if 'store' in train_df.columns:
    train_df['store'] = le.fit_transform(train_df['store'])
    test_df['store'] = le.transform(test_df['store'])

if 'product' in train_df.columns:
    train_df['product'] = le.fit_transform(train_df['product'])
    test_df['product'] = le.transform(test_df['product'])


# Ensure test columns align with train columns
missing_cols = set(train_df.columns) - set(test_df.columns) - {'num_sold'}
for col in missing_cols:
    test_df[col] = 0  # Add missing columns with default value

# Prepare training data
X = train_df.drop(columns=['num_sold', 'id'])
y = train_df['num_sold']


# Split data for validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)


# Validate the model
y_pred = model.predict(X_valid)
validation_mape = mean_absolute_percentage_error(y_valid, y_pred)
print(f"Validation MAPE: {validation_mape}")


# Prepare test data
X_test = test_df.drop(columns=['id'], errors='ignore')


# Predict on test data
test_predictions = model.predict(X_test)


# Create submission file
sample_submission['num_sold'] = test_predictions
sample_submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")

