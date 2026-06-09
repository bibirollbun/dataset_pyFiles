import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


original_df = pd.read_csv("/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv")
train_df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sample_submission_df = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")


corr = original_df.select_dtypes(include="number").corr()

plt.figure(figsize=(30,20))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=True)
plt.title("Correlation Heatmap")
plt.show()



original_df.shape


original_df.info()


cat_cols = original_df.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = original_df.select_dtypes(include=["int64", "float64"]).columns.tolist()


original_df[num_cols].corr()["diagnosed_diabetes"].sort_values(ascending=False)


original_df["glucose_ratio"] = original_df["glucose_postprandial"] / (original_df["glucose_fasting"] + 1)
original_df["glucose_mean"] = (original_df["glucose_postprandial"] + original_df["glucose_fasting"]) / 2
original_df["hba1c_glucose"] = original_df["hba1c"] * original_df["glucose_fasting"]
original_df["glucose_diff"] = original_df["glucose_postprandial"] - original_df["glucose_fasting"]



original_df["bmi_age"] = original_df["bmi"] / (original_df["age"] + 1)
original_df["waist_bmi"] = original_df["waist_to_hip_ratio"] * original_df["bmi"]
original_df["obesity_score"] = original_df["bmi"] * original_df["waist_to_hip_ratio"]



original_df["bp_gap"] = original_df["systolic_bp"] - original_df["diastolic_bp"]
original_df["bp_mean"] = (original_df["systolic_bp"] + original_df["diastolic_bp"]) / 2



original_df["chol_ratio"] = original_df["ldl_cholesterol"] / (original_df["hdl_cholesterol"] + 1)
original_df["lipid_risk"] = original_df["cholesterol_total"] / (original_df["hdl_cholesterol"] + 1)
original_df["trig_hdl_ratio"] = original_df["triglycerides"] / (original_df["hdl_cholesterol"] + 1)



original_df["activity_bmi"] = original_df["physical_activity_minutes_per_week"] / (original_df["bmi"] + 1)
original_df["sleep_activity"] = original_df["sleep_hours_per_day"] * original_df["physical_activity_minutes_per_week"]
original_df["diet_activity_score"] = original_df["diet_score"] * original_df["physical_activity_minutes_per_week"]



original_df["metabolic_score"] = (
    original_df["bmi"] +
    original_df["waist_to_hip_ratio"] * 10 +
    original_df["triglycerides"] / 100 +
    original_df["glucose_fasting"] / 50
)



original_df["glucose_fasting_log"] = np.log1p(original_df["glucose_fasting"])
original_df["insulin_log"] = np.log1p(original_df["insulin_level"])
original_df["triglycerides_log"] = np.log1p(original_df["triglycerides"])



original_df["hba1c_ratio"] = original_df["hba1c"] / (original_df["glucose_fasting"] + 1)


original_df


sns.countplot(x=original_df["diagnosed_diabetes"])
plt.title("Target Distribution")
plt.show()


cols = [
    "hba1c",
    "glucose_fasting",
    "glucose_postprandial",
    "bmi"
]

original_df[cols].hist(bins=30, figsize=(12,6))
plt.show()



sns.boxplot(x=original_df["diagnosed_diabetes"], y=original_df["hba1c"])
plt.title("HbA1c vs Target")
plt.show()



corr = original_df.select_dtypes(include="number").corr()

plt.figure(figsize=(40,20))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=True)
plt.title("Correlation Heatmap")
plt.show()



sns.scatterplot(
    x=original_df["glucose_fasting"],
    y=original_df["glucose_ratio"],
    hue=original_df["diagnosed_diabetes"],
    alpha=0.3
)
plt.legend()
plt.show()



sns.boxplot(y=original_df["triglycerides"])
plt.show()



full_df = pd.concat([train_df, test_df], ignore_index=True)


full_df.columns


full_df.shape


cat_cols1 = full_df.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols1 = full_df.select_dtypes(include=["int64", "float64"]).columns.tolist()


full_df[num_cols1].corr()["diagnosed_diabetes"].sort_values(ascending=False)


full_df["age_squared"] = full_df["age"] ** 2
full_df["age_log"] = np.log1p(full_df["age"])
full_df["bmi_age"] = full_df["bmi"] * full_df["age"]

full_df["obesity_score"] = (
    (full_df["bmi"] >= 30).astype(int) +
    (full_df["waist_to_hip_ratio"] > 0.9).astype(int)
)

full_df["activity_bmi"] = full_df["physical_activity_minutes_per_week"] / (full_df["bmi"] + 1)

full_df["bp_mean"] = (full_df["systolic_bp"] + full_df["diastolic_bp"]) / 2
full_df["bp_gap"] = full_df["systolic_bp"] - full_df["diastolic_bp"]

full_df["chol_ratio"] = full_df["cholesterol_total"] / (full_df["hdl_cholesterol"] + 1)
full_df["trig_hdl_ratio"] = full_df["triglycerides"] / (full_df["hdl_cholesterol"] + 1)
full_df["triglycerides_log"] = np.log1p(full_df["triglycerides"])

full_df["lipid_risk"] = (
    (full_df["ldl_cholesterol"] > 130).astype(int) +
    (full_df["triglycerides"] > 150).astype(int)
)

full_df["metabolic_score"] = (
    full_df["bmi"] +
    full_df["bp_mean"] +
    full_df["chol_ratio"] +
    full_df["trig_hdl_ratio"]
)

full_df["sleep_activity"] = (
    full_df["sleep_hours_per_day"] *
    full_df["physical_activity_minutes_per_week"]
)

full_df["screen_sleep_ratio"] = (
    full_df["screen_time_hours_per_day"] /
    (full_df["sleep_hours_per_day"] + 1)
)

full_df["diet_activity_score"] = (
    full_df["diet_score"] *
    full_df["physical_activity_minutes_per_week"]
)

full_df["alcohol_bmi"] = (
    full_df["alcohol_consumption_per_week"] *
    full_df["bmi"]
)

full_df["family_history_diabetes"] = full_df["family_history_diabetes"].astype(int)
full_df["hypertension_history"] = full_df["hypertension_history"].astype(int)
full_df["cardiovascular_history"] = full_df["cardiovascular_history"].astype(int)







full_df.shape


full_df.drop(columns=["id", "alcohol_consumption_per_week", "sleep_hours_per_day", "screen_time_hours_per_day"], axis=1, inplace = True)


full_df.shape


full_df.info()


!pip install catboost
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


train_df = full_df[full_df["diagnosed_diabetes"].notna()].copy()
test_df  = full_df[full_df["diagnosed_diabetes"].isna()].copy()

X = train_df.drop(columns=["diagnosed_diabetes"])
y = train_df["diagnosed_diabetes"]

X_test = test_df.drop(columns=["diagnosed_diabetes"])



cat_cols = X.select_dtypes(include="object").columns.tolist()

print("Categorical Columns:")
print(cat_cols)



X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



model = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.05,
    depth=8,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=200,
    early_stopping_rounds=100
)



model.fit(
    X_train,
    y_train,
    cat_features=cat_cols,
    eval_set=(X_val, y_val),
    use_best_model=True
)



val_preds = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_preds)

print(f"Validation AUC: {auc:.5f}")



final_model = CatBoostClassifier(
    iterations=model.best_iteration_,
    learning_rate=0.05,
    depth=8,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=200
)

final_model.fit(
    X,
    y,
    cat_features=cat_cols
)



test_preds = final_model.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({
    "id": test_df.index,
    "diagnosed_diabetes": test_preds
})

submission.to_csv("submission.csv", index=False)
print("submission.csv saved")





