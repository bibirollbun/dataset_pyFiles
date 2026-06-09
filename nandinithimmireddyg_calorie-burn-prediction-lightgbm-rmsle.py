!pip install lightgbm --quiet



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb



train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")



def add_features(df):
    df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
    df["Intensity"] = df["Heart_Rate"] / df["Duration"]
    df["HRxDuration"] = df["Heart_Rate"] * df["Duration"]
    df["WeightxDuration"] = df["Weight"] * df["Duration"]
    df["AgexSex"] = df["Age"] * df["Sex"].map({"male": 1, "female": 0})
    return df

train = add_features(train)
test = add_features(test)



#Encode Sex Feature

le = LabelEncoder()
train["Sex"] = le.fit_transform(train["Sex"])  # male:1, female:0
test["Sex"] = le.transform(test["Sex"])



features = [
    'Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp',
    'BMI', 'Intensity', 'HRxDuration', 'WeightxDuration', 'AgexSex'
]
target = "Calories"



X = train[features]
y = np.log1p(train[target])  # log1p for RMSLE optimization
X_test = test[features]



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



# Convert to LightGBM Dataset
train_set = lgb.Dataset(X_train, label=y_train)
val_set = lgb.Dataset(X_val, label=y_val)



# Define parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "verbosity": -1,
    "random_state": 42
}



# Train model using callbacks for early stopping
from lightgbm import early_stopping, log_evaluation

model = lgb.train(
    params,
    train_set,
    num_boost_round=1000,
    valid_sets=[val_set],
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=100)
    ]
)



val_preds = model.predict(X_val)
val_rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(val_preds)))
print(f"Validation RMSLE: {val_rmsle:.5f}")


# Predict on test set and save submission
# Reload sample submission template
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")

# Predict on test set
test_preds = model.predict(X_test)

# Fill predictions using inverse of log1p (since we trained on log-transformed target)
submission["Calories"] = np.expm1(test_preds)

# Save final CSV for Kaggle submission
submission.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")

