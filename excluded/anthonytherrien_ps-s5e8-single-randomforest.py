# ---------------------------
#  Import libraries
# ---------------------------
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ---------------------------
#  Load datasets
# ---------------------------
def load_data():
    # Load training and test data
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
    return train_df, test_df

# ---------------------------
#  Encode categorical features
# ---------------------------
def encode_data(train_df, test_df):
    # Copy DataFrames
    train = train_df.copy()
    test = test_df.copy()

    # Identify categorical columns
    cat_cols = train.select_dtypes(include='object').columns

    # Encode each categorical column
    for col in cat_cols:
        encoder = LabelEncoder()
        train[col] = encoder.fit_transform(train[col].astype(str))
        test[col] = encoder.transform(test[col].astype(str))

    return train, test

# ---------------------------
#  Train and evaluate model
# ---------------------------
def train_model(X_train, y_train):
    # Define RandomForest model with fixed parameters
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    # Fit model on training data
    model.fit(X_train, y_train)

    return model

# ---------------------------
#  Generate predictions
# ---------------------------
def generate_submission(model, X_test, test_ids):
    # Predict probabilities for class 1
    preds = model.predict_proba(X_test)[:, 1]

    # Create submission DataFrame
    submission = pd.DataFrame({
        "id": test_ids,
        "y": preds
    })

    # Save submission file
    submission.to_csv("/kaggle/working/submission.csv", index=False)

# ---------------------------
#  Main function
# ---------------------------
def main():
    # Load raw data
    train_df, test_df = load_data()

    # Encode categorical features
    train_encoded, test_encoded = encode_data(train_df, test_df)

    # Prepare training and test matrices
    X_train = train_encoded.drop(columns=["id", "y"])
    y_train = train_encoded["y"]
    X_test = test_encoded.drop(columns=["id"])
    test_ids = test_df["id"]

    # Train model
    model = train_model(X_train, y_train)

    # Generate and save submission
    generate_submission(model, X_test, test_ids)

# ---------------------------
#  Entry point
# ---------------------------
if __name__ == "__main__":
    main()

