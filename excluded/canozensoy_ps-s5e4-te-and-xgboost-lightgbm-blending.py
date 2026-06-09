# === Load Libraries ===
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import lightgbm as lgb


# === Load Data ===
train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


# === Map Categorical Variables ===
genre_map = {"True Crime": 0, "Comedy": 1, "Education": 2, "Technology": 3, "Health": 4, "News": 5,
             "Music": 6, "Sports": 7, "Business": 8, "Lifestyle": 9}
day_map = {"Thursday": 0, "Saturday": 1, "Tuesday": 2, "Monday": 3, "Sunday": 4, "Wednesday": 5, "Friday": 6}
time_map = {"Night": 0, "Afternoon": 1, "Evening": 2, "Morning": 3}
sentiment_map = {"Positive": 0, "Negative": 1, "Neutral": 2}

for df in [train_df, test_df]:
    df["Genre"] = df["Genre"].map(genre_map)
    df["Publication_Day"] = df["Publication_Day"].map(day_map)
    df["Publication_Time"] = df["Publication_Time"].map(time_map)
    df["Episode_Sentiment"] = df["Episode_Sentiment"].map(sentiment_map)


# === Label Encoding for Text Columns ===
for col in ["Podcast_Name", "Episode_Title"]:
    le = LabelEncoder()
    le.fit(pd.concat([train_df[col], test_df[col]], axis=0).astype(str))
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))


# === Handle Missing Values ===
for col in ["Episode_Length_minutes", "Guest_Popularity_percentage"]:
    mean_val = train_df[col].mean()
    train_df[col] = train_df[col].fillna(mean_val)
    test_df[col] = test_df[col].fillna(mean_val)


# === Feature Engineering ===
for df in [train_df, test_df]:
    df['Ads_Per_Minute'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1e-3)
    df['Length_Time_Interaction'] = df['Episode_Length_minutes'] * df['Publication_Time']
    df['Log_Episode_Length'] = np.log1p(df['Episode_Length_minutes'])
    df['Squared_Host_Popularity'] = df['Host_Popularity_percentage']**2
    df['Squared_Guest_Popularity'] = df['Guest_Popularity_percentage']**2


# === Target Encoding Function ===
def target_encode(train_df, test_df, col, target):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_df[f"{col}_TE"] = np.nan

    for tr_idx, val_idx in kf.split(train_df):
        X_tr, X_val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
        means = X_tr.groupby(col)[target].mean()
        train_df.loc[train_df.index[val_idx], f"{col}_TE"] = train_df.loc[train_df.index[val_idx], col].map(means)

    # For test set
    global_means = train_df.groupby(col)[target].mean()
    test_df[f"{col}_TE"] = test_df[col].map(global_means)

    return train_df, test_df


# === Apply Target Encoding ===
for col in ["Podcast_Name", "Genre"]:
    train_df, test_df = target_encode(train_df, test_df, col, "Listening_Time_minutes")


# === Outlier Removal on Target ===
Q1 = train_df["Listening_Time_minutes"].quantile(0.01)
Q3 = train_df["Listening_Time_minutes"].quantile(0.99)
train_df = train_df[(train_df["Listening_Time_minutes"] >= Q1) & (train_df["Listening_Time_minutes"] <= Q3)]


# === Define Training and Target ===
target = "Listening_Time_minutes"
drop_cols = ["id", target]
features = [col for col in train_df.columns if col not in drop_cols]
X = train_df[features]
y = train_df[target]
X_test = test_df.drop("id", axis=1)


# === Split Train/Validation for Evaluation ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# === Train XGBoost Model ===
xgb_model = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=12,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.4,
    reg_lambda=2,
    random_state=42,
    tree_method="hist",
    verbosity=1
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=100
)

xgb_val_pred = xgb_model.predict(X_val)
xgb_test_pred = xgb_model.predict(X_test)


# === Train LightGBM Model ===
lgb_model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=12,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=5,
    reg_alpha=0.4,
    reg_lambda=2,
    random_state=42
)

from lightgbm import early_stopping, log_evaluation

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[early_stopping(stopping_rounds=50), log_evaluation(100)]
)

lgb_val_pred = lgb_model.predict(X_val)
lgb_test_pred = lgb_model.predict(X_test)


# === Blend Predictions ===
val_pred = 0.5 * xgb_val_pred + 0.5 * lgb_val_pred
test_pred = 0.5 * xgb_test_pred + 0.5 * lgb_test_pred


# === Evaluate ===
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Blended Validation RMSE: {rmse:.5f}")


# === Retrain Full Models ===
xgb_model.fit(X, y)
lgb_model.fit(X, y)

xgb_test_pred_full = xgb_model.predict(X_test)
lgb_test_pred_full = lgb_model.predict(X_test)


# === Blend Final Test Predictions ===
final_test_pred = 0.5 * xgb_test_pred_full + 0.5 * lgb_test_pred_full


# === Predict and Save Submission ===
submission = pd.DataFrame({"id": test_df["id"], "Listening_Time_minutes": final_test_pred})
submission.to_csv("submission.csv", index=False)
print("ðŸš€ Submission Saved!")

