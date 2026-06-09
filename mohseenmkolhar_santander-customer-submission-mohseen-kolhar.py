import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


path = "/kaggle/input/santander-customer-transaction-prediction-dataset/"
df = pd.read_csv(path + "train.csv")


print(df["target"].value_counts(normalize=True))
#To see how many customers transacted (target = 1). It's around 10 percent.


features = [col for col in df.columns if col.startswith("var_")]


df["row_mean"] = df[features].mean(axis=1)
df["row_std"] = df[features].std(axis=1)
df["row_min"] = df[features].min(axis=1)
df["row_max"] = df[features].max(axis=1)
df["row_range"] = df["row_max"] - df["row_min"]
df["count_pos"] = (df[features] > 1.0).sum(axis=1)
df["count_high"] = (df[features] > 2.0).sum(axis=1)
df["count_low"] = (df[features] < -1.0).sum(axis=1)
df["count_zero"] = (df[features] == 0).sum(axis=1)

#calculate average, spread, and count of strong signals in each row


transacted = df[df["target"] == 1]
non_transacted = df[df["target"] == 0]

summary = pd.DataFrame({
    "Metric": ["Mean", "Std Dev"],
    "row_mean (T)": [transacted["row_mean"].mean(), transacted["row_mean"].std()],
    "row_mean (N)": [non_transacted["row_mean"].mean(), non_transacted["row_mean"].std()],
    "row_std (T)": [transacted["row_std"].mean(), transacted["row_std"].std()],
    "row_std (N)": [non_transacted["row_std"].mean(), non_transacted["row_std"].std()],
    "count_pos (T)": [transacted["count_pos"].mean(), transacted["count_pos"].std()],
    "count_pos (N)": [non_transacted["count_pos"].mean(), non_transacted["count_pos"].std()],
})

print(summary.round(4))
#compare average behavior of both groups


bins = [0, 130, 140, 150, 160, 170, 180, 200]
df["countpos_band"] = pd.cut(df["count_pos"], bins=bins)



grouped = df.groupby("countpos_band")["target"].agg(["count", "sum", "mean"])
print(grouped)



df["logic_score"] = 0
df.loc[df["count_pos"] > 160, "logic_score"] += 1
df.loc[df["row_mean"] > 6.8, "logic_score"] += 1
df.loc[df["row_std"] < 9.5, "logic_score"] += 1
df["predicted"] = df["logic_score"] >= 2


match = df[df["predicted"]]
precision = match["target"].mean()
print("Precision of logic:", round(precision * 100, 2), "%")


df["behavior_score"] = (
    (df["count_pos"] - 150) * 0.5 +
    (df["row_mean"] - 6.5) * 0.3 +
    (df["row_std"] - 9.0) * 0.2
)


df_sorted = df.sort_values(by="behavior_score", ascending=False)
top_customers = df_sorted.head(5000)
print("Precision:", round(top_customers["target"].mean(), 4))


df["band"] = "default"
df.loc[df["count_pos"] > 160, "band"] = "very_high_pos"
df.loc[(df["count_pos"] > 150) & (df["row_std"] > 10), "band"] = "high_pos_high_std"
df.loc[df["count_high"] > 30, "band"] = "high_spikes"


df["feature_count"] = len(features)
df["percent_high"] = df["count_high"] / df["feature_count"]
df["percent_pos"] = df["count_pos"] / df["feature_count"]
df["percent_low"] = df["count_low"] / df["feature_count"]

cols_to_export = [
    "ID_code", "target", "row_mean", "row_std", "row_min", "row_max",
    "row_range", "count_pos", "count_high", "count_low", "count_zero",
    "feature_count", "percent_high", "percent_pos", "percent_low"
]

df[cols_to_export].to_csv("train_behavioral_features.csv", index=False)
print("Exported: train_behavioral_features.csv")


from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report


df = pd.read_csv(path + "train.csv")


X = df.drop(columns=["ID_code", "target"])
y = df["target"]


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


model = LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    n_estimators=1000,
    random_state=42,
    verbose=-1
)


model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=100)
    ]
)


y_pred_proba = model.predict_proba(X_val)[:, 1]
y_pred_class = model.predict(X_val)


auc = roc_auc_score(y_val, y_pred_proba)
acc = accuracy_score(y_val, y_pred_class)

print(f"ROC AUC Score: {auc:.4f}")
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred_class))


output_df = df.loc[X_val.index, ["ID_code", "target"]].copy()
output_df["prediction"] = y_pred_proba
output_df["predicted_class"] = y_pred_class
output_df.to_csv("sample_submission.csv", index=False)
print("Saved to sample_submission.csv")


df["z_above_2"] = (np.abs((df[features] - df[features].mean(axis=1).values[:, None]) / df[features].std(axis=1).values[:, None]) > 2).sum(axis=1)
df["z_above_2"] > 10



df["pos_neg_ratio"] = (df[features] > 1.0).sum(axis=1) / ((df[features] < -1.0).sum(axis=1) + 1)



df["row_skew"] = df[features].skew(axis=1)




rule_flags = {
    "z_above_2_flag": (df["z_above_2"] > 10),
    "pos_neg_ratio_flag": (df["pos_neg_ratio"] > 5),
    "row_mean_percentile_flag": (df["row_mean_percentile"] > 0.90),
    "row_entropy_flag": (df["row_entropy"] > 2.5),
    "row_skew_flag": (df["row_skew"] > 1.0),
    "count_medium_flag": (df["count_medium"] > 30)
}

summary_data = []

for name, condition in rule_flags.items():
    flagged = df[condition]
    precision = flagged["target"].mean()
    count = len(flagged)
    summary_data.append({
        "rule_name": name,
        "rule_flag": 1,
        "precision": precision,
        "count_customers": count
    })



print(summary_df)




