import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train_data = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


train_data.head()


## Summary of train data
train_data.info()


## Description of train data
train_data.describe()


## Finding count of missing value
train_data.isna().sum()


## For EDA we usually drop the data having nan value

d1 = train_data.copy()
d1 = d1.dropna()
d1.isna().sum()


## Converting Introvert --> 0 and Extrovert --> 1

map_personality = {"Introvert": 0, "Extrovert": 1}
d1["Personality"] = d1["Personality"].map(map_personality)


# Group by Time_spent_Alone and Personality
grouped = d1.groupby(["Time_spent_Alone", "Personality"]).size().unstack(fill_value=0)

# Normalize row-wise (per time)
normalized = grouped.div(grouped.sum(axis=1), axis=0)

# Plot
normalized.plot(kind="bar", stacked=True, title="Normalized Personality Distribution per Time Spent Alone")



## Converting Stagefear to 0 and 1 

map_stagefear = {"Yes" : 1, "No" : 0}
d1["Stage_fear"] = d1["Stage_fear"].map(map_stagefear)
d1["Stage_fear"]


# Group by Stage_fear and Personality
grouped = d1.groupby(["Stage_fear", "Personality"]).size().unstack(fill_value=0)

# Normalize row-wise (per time)
normalized = grouped.div(grouped.sum(axis=1), axis=0)

# Plot
normalized.plot(kind="bar", stacked=True, title="Normalized Personality Distribution per Time Spent Alone")



# Group by Social_event_attendence and Personality
grouped = d1.groupby(["Social_event_attendance", "Personality"]).size().unstack(fill_value=0)

# Normalize row-wise (per time)
normalized = grouped.div(grouped.sum(axis=1), axis=0)

# Plot
normalized.plot(kind="bar", stacked=True, title="Normalized Personality Distribution per Time Spent Alone")



# Group by Stage_fear and Personality
grouped = d1.groupby(["Going_outside", "Personality"]).size().unstack(fill_value=0)

# Normalize row-wise (per time)
normalized = grouped.div(grouped.sum(axis=1), axis=0)

# Plot
normalized.plot(kind="bar", stacked=True, title="Normalized Personality Distribution per Time Spent Alone")



train_data.head()


train_data["Personality"].value_counts()


train_data.isna().sum()


## Fill nan values

def fill_nan(data):
    data['Time_spent_Alone'] = data['Time_spent_Alone'].fillna(data['Time_spent_Alone'].mean())
    data["Stage_fear"] = data["Stage_fear"].fillna(data["Stage_fear"].mean())
    data["Social_event_attendance"] = data["Social_event_attendance"].fillna(data["Social_event_attendance"].mean())
    data["Going_outside"] = data["Going_outside"].fillna(data["Going_outside"].mean())
    data["Drained_after_socializing"] = data["Drained_after_socializing"].fillna(data["Drained_after_socializing"].mean())
    data["Friends_circle_size"] = data["Friends_circle_size"].fillna(data["Friends_circle_size"].mean())
    data["Post_frequency"] = data["Post_frequency"].fillna(data["Post_frequency"].mean())

    return data


## Mapping function 

yes_no_map = {"No" : 0, "Yes" : 1}
map_personality = {"Introvert": 0, "Extrovert": 1}


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import numpy as np
import os

def preprocess_data(df, is_train=True, impute_map=None, train_cols=None):
    """
    Preprocesses the data by handling missing values and performing one-hot encoding.

    Args:
        df (pd.DataFrame): The input DataFrame to preprocess.
        is_train (bool): Flag to indicate if this is the training set.
                         If True, it calculates imputation values.
                         If False, it uses the provided imputation values.
        impute_map (dict): A dictionary containing values for imputation (used when is_train=False).
        train_cols (list): The list of columns from the training set after encoding (used when is_train=False).

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
        dict: The imputation map (if is_train=True).
        list: The final column list (if is_train=True).
    """
    if is_train:
        impute_map = {}
        print("--- Calculating Imputation Values from Training Data ---")
    
    # --- 1. Handle Missing Values ---
    numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    for col in numerical_cols:
        if is_train:
            impute_map[col] = df[col].median()
        df[col] = df[col].fillna(impute_map[col])

    categorical_cols = ['Stage_fear', 'Drained_after_socializing']
    for col in categorical_cols:
        if is_train:
            impute_map[col] = df[col].mode()[0]
        df[col] = df[col].fillna(impute_map[col])

    print(f"\n--- Missing Values in {'Training' if is_train else 'Test'} Data After Imputation: {'Success' if df.isnull().sum().sum() == 0 else 'Failed'} ---")

    # --- 2. Feature Engineering: One-Hot Encoding ---
    df = pd.get_dummies(df, columns=['Stage_fear', 'Drained_after_socializing'], drop_first=True)
    
    if is_train:
        # Return the processed dataframe, the map of values used for imputation, and the final column list
        return df, impute_map, df.columns.tolist()
    else:
        # Align columns of test data with training data
        processed_train_cols = [col for col in train_cols if col != 'Personality']
        missing_cols = set(processed_train_cols) - set(df.columns)
        for c in missing_cols:
            df[c] = 0
        # Ensure the order and presence of columns match the training data
        df = df[processed_train_cols]
        return df

def build_personality_model(X_train, y_train_encoded):
    """
    Builds and trains an XGBoost model.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train_encoded (np.array): The numerically encoded training target variable.

    Returns:
        XGBClassifier: The trained model.
    """
    # Initialize XGBClassifier
    # use_label_encoder=False is recommended to avoid deprecation warnings.
    # eval_metric='mlogloss' is suitable for multi-class classification.
    xgb_classifier = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    
    print("\n--- Training the XGBoost model... ---")
    xgb_classifier.fit(X_train, y_train_encoded)
    print("Model Training Complete.")
    return xgb_classifier

def predict_on_test_data(model, impute_map, train_cols, test_file_path, submission_file_path, label_encoder):
    """
    Loads test data, preprocesses it, makes predictions, and saves the submission file.

    Args:
        model (XGBClassifier): The trained model.
        impute_map (dict): The imputation map from the training phase.
        train_cols (list): The list of columns from the processed training data.
        test_file_path (str): Path to the test data CSV.
        submission_file_path (str): Path to the sample submission CSV.
        label_encoder (LabelEncoder): The fitted label encoder for decoding predictions.
    """
    print("\n--- Starting Prediction on Test Data ---")
    # Load test data and the sample submission file
    test_df = pd.read_csv(test_file_path)
    submission_df = pd.read_csv(submission_file_path)
    
    # Preprocess the test data using the training impute_map and columns
    processed_test_df = preprocess_data(test_df, is_train=False, impute_map=impute_map, train_cols=train_cols)

    # Make predictions (will be numerical)
    predictions_encoded = model.predict(processed_test_df)
    
    # Decode predictions back to original string labels
    predictions_decoded = label_encoder.inverse_transform(predictions_encoded)
    
    # Update the submission file with decoded predictions
    submission_df['Personality'] = predictions_decoded
    
    # Save the final submission file
    output_path = 'submission.csv'
    submission_df.to_csv(output_path, index=False)
    print(f"\n--- Submission file created successfully at: {output_path} ---")
    print("Submission file head:")
    print(submission_df.head())


# --- Main Execution Block ---
# This block assumes you have 'train.csv', 'test.csv', and 'sample_submission.csv'
# in the same directory as the script.

if __name__ == "__main__":
    # Define file paths
    TRAIN_CSV = '/kaggle/input/playground-series-s5e7/train.csv'
    TEST_CSV = '/kaggle/input/playground-series-s5e7/test.csv'
    SAMPLE_SUBMISSION_CSV = '/kaggle/input/playground-series-s5e7/sample_submission.csv'

    # Check if required files exist and create dummies if not (for demonstration)
    required_files = [TRAIN_CSV, TEST_CSV, SAMPLE_SUBMISSION_CSV]
    for f in required_files:
        if not os.path.exists(f):
            print(f"Error: Required file not found at '{f}'. Creating dummy file.")
            if f == 'train.csv':
                pd.DataFrame({
                    'Time_spent_Alone': [2, 5, 1, 8], 'Stage_fear': ['low', 'high', 'medium', 'low'],
                    'Social_event_attendance': [1, 0, 1, 1], 'Going_outside': [3, 1, 4, 5],
                    'Drained_after_socializing': ['yes', 'no', 'yes', 'no'], 'Friends_circle_size': [10, 2, 15, 5],
                    'Post_frequency': [4, 0, 5, 1], 'Personality': ['Introvert', 'Extrovert', 'Introvert', 'Extrovert']
                }).to_csv(f, index=False)
            elif f == 'test.csv':
                 pd.DataFrame({
                    'Time_spent_Alone': [3, 6], 'Stage_fear': ['low', 'medium'],
                    'Social_event_attendance': [0, 1], 'Going_outside': [2, 4],
                    'Drained_after_socializing': ['yes', 'no'], 'Friends_circle_size': [5, 20],
                    'Post_frequency': [2, 6]
                }).to_csv(f, index=False)
            elif f == 'sample_submission.csv':
                pd.DataFrame({'id': [0, 1], 'Personality': ['', '']}).to_csv(f, index=False)

    # 1. Load and preprocess the training data
    train_df = pd.read_csv(TRAIN_CSV)
    processed_train_df, impute_map, train_cols = preprocess_data(train_df.copy(), is_train=True)

    # 2. Separate features (X) and target (y)
    X = processed_train_df.drop('Personality', axis=1)
    y = processed_train_df['Personality']

    # 3. Encode the target variable (y)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 4. Build and train the XGBoost model
    trained_model = build_personality_model(X, y_encoded)

    # 5. Predict on the test set and generate the submission file
    predict_on_test_data(trained_model, impute_map, train_cols, TEST_CSV, SAMPLE_SUBMISSION_CSV, label_encoder)





