import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder


# Load the datasets
train_path = '/kaggle/input/playground-series-s5e1/train.csv'
test_path = '/kaggle/input/playground-series-s5e1/test.csv'
submission_path = '/kaggle/input/playground-series-s5e1/sample_submission.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
submission_df = pd.read_csv(submission_path)


# 1. Dataset Analysis
print("Training Data Info:")
print(train_df.info())
print("\nTest Data Info:")
print(test_df.info())

print("\nSample Training Data:")
print(train_df.head())

print("\nSample Test Data:")
print(test_df.head())


# Ensure variables are defined
train_df = train_df.copy()
test_df = test_df.copy()


# 2. Preprocessing
def preprocess_data(df, is_train=True):
    # Parse date
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday

    # Encode categorical variables
    encoder = LabelEncoder()
    for column in ['country', 'store', 'product']:
        df[column] = encoder.fit_transform(df[column])

    # Drop original date
    df.drop('date', axis=1, inplace=True)

    if is_train:
        # Fill missing values for training
        if 'num_sold' in df.columns:
            df['num_sold'] = df['num_sold'].fillna(df['num_sold'].median())

    return df

train_df = preprocess_data(train_df, is_train=True)
test_df = preprocess_data(test_df, is_train=False)



# Splitting features and target
X = train_df.drop(columns=['num_sold', 'id'])
y = train_df['num_sold']


# Train-test split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# 3. Model Training
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)


# 4. Evaluation
val_preds = model.predict(X_val)
mape = mean_absolute_percentage_error(y_val, val_preds)
print(f"Validation MAPE: {mape:.4f}")


# 5. Visualization
sns.barplot(x=X.columns, y=model.feature_importances_)
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.show()


# 6. Prediction for Test Set
test_features = test_df.drop(columns=['id'])
test_preds = model.predict(test_features)
submission_df['num_sold'] = test_preds


# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("Submission file saved as submission.csv")

