import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Step 1: Extract valid records from raw string
def parse_raw_data(raw_text):
    # Split the raw text into lines, assuming each line is a record
    records = raw_text.strip().split('\n')

    # Clean and filter non-empty valid records
    cleaned_records = []
    for r in records:
        # Use regex to find the fields within each record
        # This regex is more flexible and accounts for potential variations
        match = re.match(r'(\d+),(\d+\.?\d*),([YN]o?),(\d+\.?\d*),(\d+\.?\d*),([YN]o?),(\d+\.?\d*),(\d+\.?\d*),([YN]o?),?([a-zA-Z]*)', r.strip())
        if match:
            cleaned_records.append(list(match.groups()))

    # Define columns - ensure it matches the number of groups in the regex
    columns = [
        'id', 'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
        'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
        'Post_frequency', 'Personality'
    ]

    # Create DataFrame
    df = pd.DataFrame(cleaned_records, columns=columns)

    # Convert numeric columns
    numeric_cols = [
        'Time_spent_Alone', 'Social_event_attendance',
        'Going_outside', 'Drained_after_socializing',
        'Friends_circle_size', 'Post_frequency'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Map Yes/No to 1/0
    yes_no_cols = ['Stage_fear', 'Going_outside', 'Drained_after_socializing']
    for col in yes_no_cols:
        # Handle potential None values from regex not matching 'Yes' or 'No'
        df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(-1) # Use -1 or another indicator for missing Yes/No

    return df

# Step 2: Load and prepare data
def load_data(train_path, test_path):
    try:
        with open(train_path, 'r') as f:
            train_raw = f.read()
        train_df = parse_raw_data(train_raw)

        with open(test_path, 'r') as f:
            test_raw = f.read()
        test_df = parse_raw_data(test_raw)

        return train_df, test_df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure 'train.csv' and 'test.csv' are in the correct directory.")
        return pd.DataFrame(), pd.DataFrame() # Return empty dataframes on error


# Step 3: Train and predict
def train_and_predict(train_df, test_df):
    if train_df.empty or test_df.empty:
        print("Cannot train or predict with empty dataframes.")
        return pd.Series(dtype=int), [] # Return empty results

    X = train_df.drop(columns=['id', 'Personality'])
    y = train_df['Personality'].map({'Introvert': 0, 'Extrovert': 1})

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    X_test = test_df.drop(columns=['id', 'Personality']) # Also drop Personality from test data if it exists
    X_test = imputer.transform(X_test)
    y_pred = model.predict(X_test)

    y_pred_labels = ['Introvert' if p == 0 else 'Extrovert' for p in y_pred]
    return test_df['id'].astype(int), y_pred_labels

# Main
if __name__ == "__main__":
    train_path = '/kaggle/input/playground-series-s5e7/train.csv'
    test_path = '/kaggle/input/playground-series-s5e7/test.csv'

    print("Parsing train and test data...")
    train_df, test_df = load_data(train_path, test_path)

    print("Train data shape:", train_df.shape)
    print("Test data shape:", test_df.shape)

    if not train_df.empty and not test_df.empty:
        print("Training model and predicting...")
        ids, predictions = train_and_predict(train_df, test_df)

        # Save submission
        submission = pd.DataFrame({'id': ids, 'Personality': predictions})
        submission.to_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv', index=False)
        print("Submission saved to submission.csv")
    else:
        print("Skipping training and prediction due to empty data.")




