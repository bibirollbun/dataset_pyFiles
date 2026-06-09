# ---------------------------
#  Import libraries
# ---------------------------
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

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
#  Train model on full data
# ---------------------------
def train_best_model(X_train, y_train):
    # Define best parameters from Optuna trial 251
    best_params = {
        "n_estimators": 606,
        "max_depth": 11,
        "learning_rate": 0.07710321517350463,
        "subsample": 0.9838046621394767,
        "colsample_bytree": 0.6033929370929036,
        "gamma": 0.10652235074557941,
        "reg_alpha": 4.194789360118641,
        "reg_lambda": 9.846023792396192,
        "use_label_encoder": False,
        "eval_metric": "auc"
    }

    # Initialize model with best parameters
    model = XGBClassifier(**best_params)

    # Fit model on entire training data
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

    # Extract features and labels
    X_train = train_encoded.drop(columns=["id", "y"])
    y_train = train_encoded["y"]
    X_test = test_encoded.drop(columns=["id"])
    test_ids = test_df["id"]

    # Train final model
    model = train_best_model(X_train, y_train)

    # Generate predictions and submission
    generate_submission(model, X_test, test_ids)

# ---------------------------
#  Entry point
# ---------------------------
if __name__ == "__main__":
    main()


