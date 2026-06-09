import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import zipfile


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

# Verify test set has 250,000 rows as required by the competition
assert test.shape[0] == 250000, f"Expected 250000 test rows, found {test.shape[0]}"



le = LabelEncoder()
train["Sex_le"] = le.fit_transform(train["Sex"])
test["Sex_le"]  = le.transform(test["Sex"])


train["Height_m"] = train["Height"] / 100
test["Height_m"]  = test["Height"]  / 100

train["BMI"] = train["Weight"] / (train["Height_m"] ** 2)
test["BMI"]  = test["Weight"] / (test["Height_m"]  ** 2)


train["Calories_per_min"] = train["Calories"] / train["Duration"]


train["Dur_HR"]  = train["Duration"] * train["Heart_Rate"]
test["Dur_HR"]   = test["Duration"]  * test["Heart_Rate"]

train["Dur_BMI"] = train["Duration"] * train["BMI"]
test["Dur_BMI"]  = test["Duration"]  * test["BMI"]


features = ["Sex_le", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Dur_HR", "Dur_BMI"]
X_train = train[features]
y_train = train["Calories_per_min"]
X_test  = test[features]


lgb_train = lgb.Dataset(X_train, label=y_train)


params = {
    "objective": "regression_l1",
    "metric": "l1",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 42
}
callbacks = [
    lgb.early_stopping(stopping_rounds=50),
    lgb.log_evaluation(period=100)
]


cv_results = lgb.cv(
    params=params,
    train_set=lgb_train,
    num_boost_round=2000,
    nfold=5,
    stratified=False,
    callbacks=callbacks
)
mean_key = next(k for k in cv_results.keys() if k.endswith("-mean"))
best_rounds = len(cv_results[mean_key])
print("Optimal Boosting Rounds from CV:", best_rounds)


model = lgb.train(
    params=params,
    train_set=lgb_train,
    num_boost_round=best_rounds,
    callbacks=[lgb.log_evaluation(period=100)]
)


preds_norm = model.predict(X_test)
preds_raw  = preds_norm * test["Duration"].values
preds_raw  = np.clip(preds_raw, a_min=0, a_max=None)

# Sanity check
assert preds_raw.shape[0] == 250000, f"Expected 250k predictions, got {preds_raw.shape[0]}"
print("Sample raw calorie predictions:", preds_raw[:5])


submission = pd.DataFrame({
    "id":       test["id"],
    "Calories": preds_raw
})
assert submission.shape == (250000, 2), f"Submission must be (250k,2), got {submission.shape}"

# Save files
submission.to_csv("submission.csv", index=False)
submission.to_csv("submission.csv.gz", index=False, compression="gzip")

# Optionally zip the CSV
with zipfile.ZipFile("/kaggle/working/submission.zip", mode="w", compression=zipfile.ZIP_DEFLATED) as z:
    z.write("submission.csv", arcname="submission.csv")

print("Created: submission.csv, submission.csv.gz, submission.zip")

