import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from itertools import combinations
from tqdm import tqdm
import lightgbm as lgb
from lightgbm import LGBMRegressor


data = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
orig_data = pd.read_csv("/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv")
orig_data.dropna(inplace=True)
data = pd.concat([data, orig_data], axis=0)
test_data = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
data.head()


# Drop ID
data = data.drop(columns=["id"], axis=1)
test_ids = test_data["id"]
test_data.drop(["id"], axis=1, inplace=True)
data.head()


print(len(data))
data.isna().sum()


def feature_engineering(df, is_train=True):
    podc_dict = {'Mystery Matters': 0, 'Joke Junction': 1, 'Study Sessions': 2, 'Digital Digest': 3, 'Mind & Body': 4, 'Fitness First': 5, 'Criminal Minds': 6, 'News Roundup': 7, 'Daily Digest': 8, 'Music Matters': 9, 'Sports Central': 10, 'Melody Mix': 11, 'Game Day': 12, 'Gadget Geek': 13, 'Global News': 14, 'Tech Talks': 15, 'Sport Spot': 16, 'Funny Folks': 17, 'Sports Weekly': 18, 'Business Briefs': 19, 'Tech Trends': 20, 'Innovators': 21, 'Health Hour': 22, 'Comedy Corner': 23, 'Sound Waves': 24, 'Brain Boost': 25, "Athlete's Arena": 26, 'Wellness Wave': 27, 'Style Guide': 28, 'World Watch': 29, 'Humor Hub': 30, 'Money Matters': 31, 'Healthy Living': 32, 'Home & Living': 33, 'Educational Nuggets': 34, 'Market Masters': 35, 'Learning Lab': 36, 'Lifestyle Lounge': 37, 'Crime Chronicles': 38, 'Detective Diaries': 39, 'Life Lessons': 40, 'Current Affairs': 41, 'Finance Focus': 42, 'Laugh Line': 43, 'True Crime Stories': 44, 'Business Insights': 45, 'Fashion Forward': 46, 'Tune Time': 47}
    genr_dict = {'True Crime': 0, 'Comedy': 1, 'Education': 2, 'Technology': 3, 'Health': 4, 'News': 5, 'Music': 6, 'Sports': 7, 'Business': 8, 'Lifestyle': 9}
    week_dict = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    time_dict = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
    sent_dict = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    
    # Ensure Episode_Title is a string and handle missing values
    df['Episode_Num'] = df['Episode_Title'].str[8:].astype('category')
    
    # Convert Genre to object type before replacing
    df['Genre'] = df['Genre'].astype('object').replace(genr_dict)
    
    df['Podcast_Name'] = df['Podcast_Name'].astype('object').replace(podc_dict)
    df['Publication_Day'] = df['Publication_Day'].astype('object').replace(week_dict)
    df['Publication_Time'] = df['Publication_Time'].astype('object').replace(time_dict)
    df['Episode_Sentiment'] = df['Episode_Sentiment'].astype('object').replace(sent_dict)
    
    df['Genre'] = df['Genre'].astype('category')
    df['Podcast_Name'] = df['Podcast_Name'].astype('category')
    df['Publication_Day'] = df['Publication_Day'].astype('category')
    df['Publication_Time'] = df['Publication_Time'].astype('category')
    df['Episode_Sentiment'] = df['Episode_Sentiment'].astype('category')

    df = df.drop(columns=['Episode_Title'])

    # Fill missing values and create new features
    df["Number_of_Ads"] = df["Number_of_Ads"].fillna(0).clip(0, 3).astype(np.uint8)
    df["Episode_Length_minutes"] = df["Episode_Length_minutes"].fillna(60)
    df["SinEpLen"] = np.sin(2 * np.pi * df["Episode_Length_minutes"] / 60)
    df["CosEpLen"] = np.cos(2 * np.pi * df["Episode_Length_minutes"] / 60)
    df["ELen_Int"] = np.floor(df["Episode_Length_minutes"])
    df["ELen_Dec"] = df["Episode_Length_minutes"] - df["ELen_Int"]

    # Convert categorical columns to appropriate types and fill missing values
    cat_cols = ["Podcast_Name", "Genre", "Episode_Sentiment"]
    df[cat_cols] = df[cat_cols].astype("string").fillna("missing")

    # For numerical columns, fill missing values with appropriate defaults
    num_cols = [col for col in df.select_dtypes(include=["number"]).columns if col not in ["Listening_Time_minutes", "Episode_Title"]]
    df["Number_of_Ads"] = df["Number_of_Ads"].fillna(0).clip(0, 3).astype(np.uint8)
    df["ELen_Int"] = df["ELen_Int"].fillna(0).astype(np.int64)

    # Fill missing values for Guest_Popularity_percentage with the mean value
    df["Guest_Popularity_percentage"] = df["Guest_Popularity_percentage"].fillna(
        df["Guest_Popularity_percentage"].mean()
    )

    df["Host_Guest_Popularity"] = df["Host_Popularity_percentage"] * df["Guest_Popularity_percentage"]
    df["Ads_Length_Interaction"] = df["Number_of_Ads"] * df["Episode_Length_minutes"]

    df["Host_Popularity_Squared"] = df["Host_Popularity_percentage"] ** 2
    df["Guest_Popularity_Squared"] = df["Guest_Popularity_percentage"] ** 2
    df["Length_Squared"] = df["Episode_Length_minutes"] ** 2

    df["Log_Host_Popularity"] = np.log1p(df["Host_Popularity_percentage"])
    df["Log_Guest_Popularity"] = np.log1p(df["Guest_Popularity_percentage"])
    df["Log_Length"] = np.log1p(df["Episode_Length_minutes"])

    return df

# Combine train, original, and test data for consistent processing
combined_data = pd.concat([data, orig_data, test_data], axis=0, ignore_index=True)

# Apply feature engineering
combined_data = feature_engineering(combined_data)

print(f"Shape -> {combined_data.shape}")
print("Feature Engineering Complete!")


encode_columns = ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 
                  'Number_of_Ads', 'Episode_Sentiment', 'Publication_Day', 'Publication_Time']
pair_size = [2, 3, 4]

new_columns = {}

for r in pair_size:
    for cols in tqdm(list(combinations(encode_columns, r)), desc=f"Generating combinations of size {r}"):
        new_col_name = '_'.join(cols)

        # Combine columns into a single string
        combined_col = combined_data[list(cols)].astype(str).agg('_'.join, axis=1)
        combined_col = pd.Categorical(combined_col).codes  # Encode as numeric

        new_columns[new_col_name] = combined_col

combined_data = pd.concat([combined_data, pd.DataFrame(new_columns)], axis=1)

# Encode all categorical columns to numeric
for col in combined_data.select_dtypes(include=['category', 'string']).columns:
    combined_data[col] = combined_data[col].astype('category').cat.codes

# Split train and test data after adding new features
data = combined_data[combined_data["Listening_Time_minutes"].notna()]
test_data = combined_data[combined_data["Listening_Time_minutes"].isna()].drop(columns=["Listening_Time_minutes"])

# Set features and target
X = data.drop(columns=["Listening_Time_minutes"])
y = data["Listening_Time_minutes"]


X.head()


test_data.head()


# Defining folds
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Train data predictions (to calculate the performance of the model)
oof_xgb = np.zeros(len(X))
# Test data predictions (for the submission)
pred_xgb = np.zeros(len(test_data))

xgb_model = XGBRegressor(
        n_estimators=5000,
        max_depth=12,
        learning_rate=0.05,
        colsample_bytree=0.9,
        subsample=0.9,
        random_state=42,
        eval_metric="rmse",
        early_stopping_rounds=100,
        tree_method="hist",
    )

for i, (train_index, test_index) in enumerate(kf.split(X)):

    print("#" * 25)
    print(f"### Fold {i + 1}")
    print("#" * 25)

    x_train_fold = X.iloc[train_index]
    y_train_fold = y.iloc[train_index]
    x_valid_fold = X.iloc[test_index]
    y_valid_fold = y.iloc[test_index]
    X_test = test_data[X.columns].copy()

    # Train the model on the current fold
    xgb_model.fit(
        x_train_fold,
        y_train_fold,
        eval_set=[(x_valid_fold, y_valid_fold)],
        verbose=100
    )

    # Predict on validation data
    oof_xgb[test_index] = xgb_model.predict(x_valid_fold)
    # Predict on test data for submission
    pred_xgb += xgb_model.predict(X_test)

# Compute average test predictions
pred_xgb /= FOLDS


# Evaluate the model using RMSE (Root Mean Squared Error)

xgb_rmse = np.sqrt(mean_squared_error(y, oof_xgb))
print(f"XGBoost RMSE: {xgb_rmse:.4f}")


# Initialize LGBM Model
lgbm_model = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=73,
    objective="l2",
    early_stopping_rounds=50,
    metric="rmse",
    num_leaves=2048
)

# Define folds
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Train data predictions (to calculate the performance of the model)
oof_lgbm = np.zeros(len(X))
# Test data predictions (for the submission)
pred_lgbm = np.zeros(len(test_data))

for i, (train_index, test_index) in enumerate(kf.split(X)):
    print("#" * 25)
    print(f"### Fold {i + 1}")
    print("#" * 25)

    x_train_fold = X.iloc[train_index]
    y_train_fold = y.iloc[train_index]
    x_valid_fold = X.iloc[test_index]
    y_valid_fold = y.iloc[test_index]

    # Train the model on the current fold
    lgbm_model.fit(
        x_train_fold,
        y_train_fold,
        eval_set=[(x_valid_fold, y_valid_fold)],
        eval_metric="rmse",
        callbacks=[lgb.log_evaluation(100)]
    )

    # Predict on validation data
    oof_lgbm[test_index] = lgbm_model.predict(x_valid_fold)
    # Predict on test data for submission
    pred_lgbm += lgbm_model.predict(test_data)

# Compute average test predictions
pred_lgbm /= FOLDS


# Evaluate the model using RMSE (Root Mean Squared Error)

lgbm_rmse = np.sqrt(mean_squared_error(y, oof_lgbm))
print(f"LGBMRegressor RMSE: {lgbm_rmse:.4f}")


oof_combined = 0.55 * oof_xgb + 0.45 * oof_lgbm 

combined_rmse = np.sqrt(mean_squared_error(y, oof_combined))
print(f"Combined models RMSE: {combined_rmse:.4f}")


com_preds = 0.55 * pred_xgb + 0.45 * pred_lgbm

# Load sumbissions
sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
# Write the predictions
sub["Listening_Time_minutes"] = com_preds
sub.head()


# Save submission file
sub.to_csv("submission.csv", index=False)
print("Sucessfully saved predictions to submission.csv")

