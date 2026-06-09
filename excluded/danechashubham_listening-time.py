# Import Libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load Data
train = pd.read_csv("/kaggle/input/listening-time-update/train.csv")
test = pd.read_csv("/kaggle/input/listening-time-update/test.csv")
submission = pd.read_csv("/kaggle/input/listening-time-update/sample_submission.csv")

# ===============================
# Feature Engineering
# ===============================
def extract_episode_number(ep_title):
    try:
        return int(ep_title.split()[-1])
    except:
        return np.nan

# Apply transformation
train['Episode_Number'] = train['Episode_Title'].apply(extract_episode_number)
test['Episode_Number'] = test['Episode_Title'].apply(extract_episode_number)

# Fill missing numerical values
num_cols = ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']
for col in num_cols:
    train[col].fillna(train[col].median())
    test[col].fillna(train[col].median())

train['Episode_Number'].fillna(train['Episode_Number'].median(), inplace=True)
test['Episode_Number'].fillna(train['Episode_Number'].median(), inplace=True)

# Combine train & test for consistent encoding
train['is_train'] = 1
test['is_train'] = 0
test['Listening_Time_minutes'] = 0  # used for set the placeholder
data = pd.concat([train, test])

# Encode categorical features
cat_cols = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
for col in cat_cols:
    data[col] = data[col].astype('category').cat.codes

# Split data again
train = data[data['is_train'] == 1].drop(['is_train'], axis=1)
test = data[data['is_train'] == 0].drop(['is_train', 'Listening_Time_minutes'], axis=1)

# ===============================
# Modeling with XGBoost
# ===============================
features = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time',
            'Episode_Length_minutes', 'Host_Popularity_percentage',
            'Guest_Popularity_percentage', 'Number_of_Ads',
            'Episode_Sentiment', 'Episode_Number']
target = 'Listening_Time_minutes'

X = train[features]
y = train[target]
X_test = test[features]

# XGBoost parameters
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.01,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42
}

# Cross-validation training
kf = KFold(n_splits=5, shuffle=True, random_state=42)
predictions = np.zeros(X_test.shape[0])
val_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Training Fold {fold + 1}...")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dval, "validation")],
        early_stopping_rounds=100,
        verbose_eval=100
    )
    
    val_pred = model.predict(dval)
    rmse = mean_squared_error(y_val, val_pred, squared=False)
    val_scores.append(rmse)
    predictions += model.predict(dtest) / kf.n_splits

print(f"\n Cross-validated RMSE: {np.mean(val_scores):.4f}")

# ===============================
# Generate Submission
# ===============================
submission['Listening_Time_minutes'] = predictions
submission.to_csv("submission.csv", index=False)
print("submission.csv generated!")


