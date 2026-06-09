# ---------------------------
#  Import libraries
# ---------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier

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
#  Train LightGBM model
# ---------------------------
def train_model(X_train, y_train):
    # Initialize model with fixed parameters
    model = LGBMClassifier(
        n_estimators=986,
        max_depth=12,
        learning_rate=0.17391215481581662,
        subsample=0.7789450128416529,
        colsample_bytree=0.744191975780123,
        reg_alpha=3.1351493773665604,
        reg_lambda=8.719413909953529,
        random_state=42,
        boosting_type="gbdt",
        objective="binary",
        metric="auc",
        verbose=-1
    )

    # Fit the model
    model.fit(X_train, y_train)

    return model

# ---------------------------
#  Generate predictions
# ---------------------------
def generate_submission(model, X_test, test_ids):
    # Predict probabilities for class 1
    preds = model.predict_proba(X_test)[:, 1]

    # Prepare submission DataFrame
    submission = pd.DataFrame({
        "id": test_ids,
        "y": preds
    })

    # Save to CSV
    submission.to_csv("/kaggle/working/submission.csv", index=False)

# ---------------------------
#  Main function
# ---------------------------
def main():
    # Load raw data
    train_df, test_df = load_data()

    # Encode categorical features
    train_encoded, test_encoded = encode_data(train_df, test_df)

    # Prepare training and test sets
    X_train = train_encoded.drop(columns=["id", "y"])
    y_train = train_encoded["y"]
    X_test = test_encoded.drop(columns=["id"])
    test_ids = test_df["id"]

    # Train the model
    model = train_model(X_train, y_train)

    # Generate predictions and save submission
    generate_submission(model, X_test, test_ids)

# ---------------------------
#  Entry point
# ---------------------------
if __name__ == "__main__":
    main()

