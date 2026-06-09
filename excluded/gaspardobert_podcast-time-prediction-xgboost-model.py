import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold


data = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
orig_data = pd.read_csv("/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv")
orig_data.dropna(inplace=True)
data = pd.concat([data, orig_data], axis=0)
test_data = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
data.head()


#Drop unused features

data = data.drop(columns=["id", "Episode_Title"], axis=1)
test_ids = test_data["id"]
test_data.drop(["id", "Episode_Title"], axis=1, inplace=True)
data.head()


print(len(data))
data.isna().sum(axis=0)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np

# Combine test and train data to ensure they are encoded the same way
data["is_train"] = 1
test_data["is_train"] = 0
combined_data = pd.concat([data, test_data], axis=0)

# Fill missing values
combined_data["Episode_Length_minutes"] = combined_data["Episode_Length_minutes"].fillna(combined_data["Episode_Length_minutes"].mean())
combined_data["Guest_Popularity_percentage"] = combined_data["Guest_Popularity_percentage"].fillna(combined_data["Guest_Popularity_percentage"].mean())
combined_data["Number_of_Ads"] = combined_data["Number_of_Ads"].fillna(0)

# Some new features (facultative)
combined_data["Ads_Per_Minute"] = combined_data["Number_of_Ads"] / combined_data["Episode_Length_minutes"]
combined_data["Host_Guest_Popularity_Ratio"] = combined_data["Host_Popularity_percentage"] / (combined_data["Guest_Popularity_percentage"] + 1e-6)

# Encoding of object type values
obj_cols = [col for col in combined_data.columns if combined_data[col].dtype == "object" and col != "Podcast_Name"]
label_encoders = {}
for col in obj_cols:
    le = LabelEncoder()
    combined_data[col] = le.fit_transform(combined_data[col])
    label_encoders[col] = le

# Remove Podcast_Name feature (not used)
combined_data.drop(["Podcast_Name"], axis=1, inplace=True)


# Split back test and train data
data = combined_data[combined_data["is_train"] == 1].drop(columns=["is_train"])
test_data = combined_data[combined_data["is_train"] == 0].drop(columns=["is_train", "Listening_Time_minutes"])

# Set features and target
X = data.drop(columns=["Listening_Time_minutes"])
y = data["Listening_Time_minutes"]


X.head()


test_data.head()


#Convertir data into numpy arrays for better processing
X = np.array(X)
y = np.array(y)


# Defining KFolds
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Train data predictions (to calculate the preformance of the model)
oof_xgb = np.zeros(len(X))
# Test data predictions (for the submission)
pred_xgb = np.zeros(len(test_data))

for i, (train_index, test_index) in enumerate(kf.split(X)):

    print("#" * 25)
    print(f"### Fold {i + 1}")
    print("#" * 25)

    x_train_fold = X[train_index]
    y_train_fold = y[train_index]
    x_valid_fold = X[test_index]
    y_valid_fold = y[test_index]

    # Define the model (feel free to adjust hyperparameters)
    model = XGBRegressor(
        n_estimators=5000,
        device = "cuda",
        max_depth=15,
        learning_rate=0.01,
        colsample_bytree=0.9,
        subsample=0.9,
        random_state=42,
        eval_metric="rmse",
        alpha=1,
        early_stopping_rounds=100
    )

    # Fitting the model using eval set to prevent over fitting with early_stopping_rounds
    model.fit(
        x_train_fold,
        y_train_fold,
        eval_set=[(x_valid_fold, y_valid_fold)],
        verbose=100
    )

    # Predict on validation data
    oof_xgb[test_index] = model.predict(x_valid_fold)
    # Predict on test data for sumbission
    pred_xgb += model.predict(test_data)

# Compute average test predictions
pred_xgb /= FOLDS


# Evaluate the model using RMSE (Root Mean Squared Error)

xgb_rmse = np.sqrt(mean_squared_error(y, oof_xgb))
print(f"XGBoost RMSE: {xgb_rmse:.4f}")


# Load sumbissions
sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
# Write the predictions
sub["Listening_Time_minutes"] = pred_xgb
sub.head()


# Save submission file
sub.to_csv("submission.csv", index=False)
print("Sucessfully saved predictions to submission.csv")

