import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
target = train["Listening_Time_minutes"]

# =============== ğŸ”§ Feature Engineering ===============
def preprocess(df):
    df = df.copy()
    
    # Episode Number
    df["Episode_Number"] = df["Episode_Title"].str.extract(r'(\d+)').astype(float)

    # Time encoding
    time_map = {"Morning": 8, "Afternoon": 14, "Evening": 18, "Night": 22}
    df["Publication_Hour"] = df["Publication_Time"].map(time_map)

    # Day of Week encoding
    day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2,
               "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    df["Publication_Day_Num"] = df["Publication_Day"].map(day_map)

    # Sentiment Encoding
    df["Episode_Sentiment"] = df["Episode_Sentiment"].map({
        "Negative": -1, "Neutral": 0, "Positive": 1
    })

    # Fill missing
    df["Episode_Length_minutes"] = df["Episode_Length_minutes"].fillna(df["Episode_Length_minutes"].median())
    df["Guest_Popularity_percentage"] = df["Guest_Popularity_percentage"].fillna(df["Guest_Popularity_percentage"].median())

    # Custom interaction features
    df["Guest_x_Length"] = df["Guest_Popularity_percentage"] * df["Episode_Length_minutes"]
    df["Number_x_Sentiment"] = df["Episode_Number"] * df["Episode_Sentiment"]
    
    df.fillna(0, inplace=True)
    return df

train = preprocess(train)
test = preprocess(test)

# Label Encoding
le = LabelEncoder()
all_genres = pd.concat([train["Genre"], test["Genre"]])
le.fit(all_genres)
train["Genre"] = le.transform(train["Genre"])
test["Genre"] = le.transform(test["Genre"])

# Drop unused columns
drop_cols = ["id", "Podcast_Name", "Episode_Title", "Publication_Time", "Publication_Day", "Listening_Time_minutes"]
features = [col for col in train.columns if col not in drop_cols]

# =============== ğŸš€ LightGBM Model ===============
kf = KFold(n_splits=5, shuffle=True, random_state=42)
preds = np.zeros(len(test))
scores = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(train)):
    X_train, X_val = train.iloc[tr_idx][features], train.iloc[val_idx][features]
    y_train, y_val = target.iloc[tr_idx], target.iloc[val_idx]

    model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        boosting_type='gbdt',
        learning_rate=0.01,
        num_leaves=64,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        n_estimators=2000,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )

    val_preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, val_preds, squared=False)
    print(f"ğŸ“‰ Fold {fold+1} RMSE: {rmse:.5f}")
    scores.append(rmse)

    preds += model.predict(test[features]) / kf.n_splits

print(f"\nâœ… CV RMSE: {np.mean(scores):.5f}")

# =============== ğŸ“� Submission ===============
submission["Listening_Time_minutes"] = preds
submission.to_csv("submission.csv", index=False)


