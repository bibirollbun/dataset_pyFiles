# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings(action="ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


pd.set_option("display.max_columns", None)


diabetes = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
diabetes[:5]


diabetes.shape


diabetes.columns


diabetes.duplicated().sum()


diabetes.isna().sum()


diabetes.describe().round(2)


sns.set_theme(style="darkgrid", palette="colorblind")


fig, axes = plt.subplots(1,2,figsize=(18,6.5))

sns.countplot(data=diabetes, x="diagnosed_diabetes", ax=axes[0])

axes[0].set_title("Class Balance: Diabetic vs Non-Diabetic Cases", fontsize=16, fontweight="bold", pad=10)
axes[0].set_xlabel("Diagnosed Diabetes", fontsize=12, labelpad=10)
axes[0].set_ylabel("Frequency", fontsize=12, labelpad=10)
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
axes[0].set_xticklabels(["Non-Diabetic", "Diabetic"])

for container in axes[0].containers:
    axes[0].bar_label(container=container, fmt=lambda x: f"{x/1000:.0f}K")

diabetes_status = diabetes["diagnosed_diabetes"].value_counts().sort_values()
axes[1].pie(diabetes_status.values, labels=["Non-Diabetic", "Diabetic"], autopct="%1.1f%%", 
            startangle=90)
axes[1].set_title("Diabetes Distribution", fontsize=16, fontweight="bold")

plt.tight_layout()
plt.show()


diabetes["age_category"] = pd.cut(
    diabetes["age"],
    bins=[18, 35, 55, 70, float("inf")],
    labels=["Young Adult", "Middle-Aged", "Senior", "Elderly"],
    right=True
)


diabetes["bmi_category"] = pd.cut(
    diabetes["bmi"],
    bins=[0, 18.5, 24.9, 29.9, float("inf")],
    labels=["Underweight", "Normal", "Overweight", "Obese"],
    right=False
)


cat_columns = ["age_category", "bmi_category", "gender", "ethnicity", "education_level", 
            "income_level", "smoking_status", "employment_status"]

fig, axes = plt.subplots(4,2,figsize=(18,22))
axes = axes.flatten()

title_map = {
    "age_category": "Diabetes Cases by Age Group",
    "bmi_category": "Diabetes Cases by BMI Category",
    "gender": "Diabetes Cases by Gender",
    "ethnicity": "Diabetes Cases by Ethnicity",
    "education_level": "Diabetes Cases by Education Level",
    "income_level": "Diabetes Cases by Income Level",
    "smoking_status": "Diabetes Cases by Smoking Status",
    "employment_status": "Diabetes Cases by Employment Status"
}

xlabel_map = {
    "age_category": "Age Group",
    "bmi_category": "BMI Category",
    "gender": "Gender",
    "ethnicity": "Ethnicity",
    "education_level": "Education Level",
    "income_level": "Income Level",
    "smoking_status": "Smoking Status",
    "employment_status": "Employment Status"
}

for i, col in enumerate(cat_columns):
    sns.countplot(
        data=diabetes,
        x=col,
        hue="diagnosed_diabetes",
        ax=axes[i]
    )

    axes[i].set_title(title_map[col], fontsize=18, fontweight="bold", pad=10)
    axes[i].set_ylabel("Frequency", fontsize=14, labelpad=10)
    axes[i].set_xlabel(xlabel_map[col], fontsize=14, labelpad=10)
    axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x/1000:.0f}K"))
    axes[i].legend(title="Diabetes Status", labels=["Non-Diabetic", "Diabetic"], loc="best")
    
for j in range(len(cat_columns), len(axes)):
    plt.delaxes(axes[j])

plt.tight_layout()
plt.show()


from scipy.stats import pearsonr

health_metrics = ["bmi", "waist_to_hip_ratio", "systolic_bp", "diastolic_bp", "cholesterol_total", 
                  "hdl_cholesterol", "ldl_cholesterol", "triglycerides"]

corr = {
    feature: pearsonr(diabetes[feature], diabetes["diagnosed_diabetes"])[0]
    for feature in health_metrics
}

corr_df = pd.DataFrame(data=list(corr.items()), 
                       columns=["Feature", "Pearson Correlation"]).sort_values(by="Pearson Correlation", ascending=False)
corr_df


plt.figure(figsize=(7,9))
sns.heatmap(
    data=corr_df[["Pearson Correlation"]].set_index(keys=corr_df["Feature"]),
    annot=True
)

plt.title("Numeric Features Correlation with Diagnosed Diabetes (Target Variable)", fontdict={"fontsize": 18, "fontweight": "bold"}, pad=12)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.violinplot(data=diabetes, x="diagnosed_diabetes", y="bmi", hue="diagnosed_diabetes", ax=axes[0])
axes[0].set_title("BMI Distribution", fontsize=15, fontweight="bold")
axes[0].set_xticklabels(["Non-Diabetic", "Diabetic"])
axes[0].set_xlabel("Diabetes Status", fontsize=12, labelpad=10)
axes[0].set_ylabel("BMI", fontsize=12, labelpad=10)
axes[0].legend_.remove()

sns.violinplot(data=diabetes, x="diagnosed_diabetes", y="waist_to_hip_ratio", hue="diagnosed_diabetes", ax=axes[1])
axes[1].set_title("Waist-to-Hip Ratio Distribution", fontsize=15, fontweight="bold")
axes[1].set_xticklabels(["Non-Diabetic", "Diabetic"])
axes[1].set_xlabel("Diabetes Status", fontsize=12, labelpad=10)
axes[1].set_ylabel("Waist-to-Hip Ratio", fontsize=12, labelpad=10) 
axes[1].legend_.remove()

plt.tight_layout()
plt.show()


lifestyle_metrics = ["alcohol_consumption_per_week", "physical_activity_minutes_per_week", "diet_score", 
                    "sleep_hours_per_day", "screen_time_hours_per_day"]

corr = {
    feature: pearsonr(diabetes[feature], diabetes["diagnosed_diabetes"])[0] 
    for feature in lifestyle_metrics
}

corr_df = pd.DataFrame(data=list(corr.items()), 
                       columns=["Feature", "Pearson Correlation"]).sort_values(by="Pearson Correlation", ascending=False)
corr_df


plt.figure(figsize=(8,7))
sns.heatmap(
    data=corr_df[["Pearson Correlation"]].set_index(keys=corr_df["Feature"]),
    annot=True,
    cbar=True
)

plt.title("Lifestyle Correlation with Diabetes", fontdict={"fontsize": 18, "fontweight": "bold"}, pad=12)
plt.tight_layout()
plt.show()


risk_factors = ["family_history_diabetes", "hypertension_history", "cardiovascular_history"]

corr = {
    feature: pearsonr(diabetes[feature], diabetes["diagnosed_diabetes"])[0] 
    for feature in risk_factors
}

corr_df = pd.DataFrame(data=list(corr.items()), 
                       columns=["Feature", "Pearson Correlation"]).sort_values(by="Pearson Correlation", ascending=False)
corr_df


fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, col in enumerate(risk_factors):
    sns.countplot(data=diabetes, x=col, hue="diagnosed_diabetes", ax=axes[i])
    axes[i].set_title(col.replace("_", " ").title(), fontsize=14, fontweight="bold")
    axes[i].set_xticklabels(["No", "Yes"])
    axes[i].set_xlabel("")
    axes[i].legend(title="Diabetes", labels=["Non-Diabetic", "Diabetic"])
    axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

plt.tight_layout()
plt.show()


print("BLOOD PRESSURE PROFILE:")
print(f"\nSYSTOLIC BP")
print(diabetes["systolic_bp"].describe().round(2))

print(f"\nDIASTOLIC BP")
print(diabetes["diastolic_bp"].describe().round(2))


fig, axes = plt.subplots(1,2,figsize=(18,6))
axes = axes.flatten()

bp_cols = ["systolic_bp", "diastolic_bp"]

title_mapping = {
    "systolic_bp": "Systolic Blood Pressure Distribution",
    "diastolic_bp": "Diastolic Blood Pressure Distribution"
}

xlabel_mapping = {
    "systolic_bp": "Systolic BP",
    "diastolic_bp": "Diastolic BP"
}

for i, col in enumerate(bp_cols):
    sns.histplot(data=diabetes, x=col, kde=True, ax=axes[i], bins=30)

    axes[i].set_title(title_mapping[col], fontsize=15, fontweight="bold", pad=10)
    axes[i].set_xlabel(xlabel_mapping[col], fontsize=12, labelpad=10)
    axes[i].set_ylabel("Frequency", fontsize=12, labelpad=10)
    axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

    col_mean = diabetes[col].mean()
    col_std = diabetes[col].std()

    axes[i].axvline(x=col_mean, color="#FF0000", linestyle="--", linewidth=2, label=f"Mean = {col_mean:.2f}")
    axes[i].axvline(x=col_mean + col_std, color="#4CAF50", linestyle='-', alpha=0.7, label=f'+1 SD = {col_mean + col_std:.2f}')
    axes[i].axvline(x=col_mean - col_std, color="#4CAF50", linestyle='-', alpha=0.7, label=f'-1 SD = {col_mean - col_std:.2f}')
    axes[i].axvline(x=col_mean + (2*col_std), color="#FF9800", linestyle='-', alpha=0.7, label=f'+2 SD = {col_mean + (2*col_std):.2f}')
    axes[i].axvline(x=col_mean - (2*col_std), color="#FF9800", linestyle='-', alpha=0.7, label=f'-2 SD = {col_mean - (2*col_std):.2f}')
    axes[i].axvline(x=col_mean + (3*col_std), color="#F44336", linestyle='-', alpha=0.7, label=f'+3 SD = {col_mean + (3*col_std):.2f}')
    axes[i].axvline(x=col_mean - (3*col_std), color="#F44336", linestyle='-', alpha=0.7, label=f'-3 SD = {col_mean - (3*col_std):.2f}')
    axes[i].legend(frameon=True, shadow=True, loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(15, 6))

bp_cols = ["systolic_bp", "diastolic_bp"]

title_mapping = {
    "systolic_bp": "Systolic Blood Pressure Distribution",
    "diastolic_bp": "Diastolic Blood Pressure Distribution"
}

xlabel_mapping = {
    "systolic_bp": "Systolic BP",
    "diastolic_bp": "Diastolic BP"
}

for i, col in enumerate(bp_cols):
    sns.boxplot(
        data=diabetes, 
        x="diagnosed_diabetes",
        y=col, 
        ax=axes[i], 
        hue="diagnosed_diabetes",
        width=0.9
    )

    axes[i].set_title(title_mapping[col], fontsize=15, fontweight="bold", pad=10)
    axes[i].set_ylabel(xlabel_mapping[col], fontsize=12, labelpad=10)
    axes[i].set_xlabel("Diabetes Status", fontsize=12, labelpad=10)
    axes[i].set_xticklabels(["Non-Diabetic", "Diabetic"])
    axes[i].legend_.remove()

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(18, 6))

diabetes_rate = diabetes.groupby("smoking_status")["diagnosed_diabetes"].mean() * 100
diabetes_rate = diabetes_rate.sort_values(ascending=False)
sns.barplot(x=diabetes_rate.index, y=diabetes_rate.values, ax=axes[0])
axes[0].set_title("Diabetes Rate by Smoking Status", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Smoking Status", fontsize=12)
axes[0].set_ylabel("Diabetes Rate (%)", fontsize=12)
for container in axes[0].containers:
    axes[0].bar_label(container=container, fmt=lambda x: f"{x:.2f}%")

sns.boxplot(data=diabetes, x="smoking_status", y="bmi", hue="diagnosed_diabetes", ax=axes[1])
axes[1].set_title("BMI by Smoking Status & Diabetes", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Smoking Status", fontsize=12)
axes[1].set_ylabel("BMI", fontsize=12)
axes[1].legend(title="Diabetes", labels=["Non-Diabetic", "Diabetic"])

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

smoking_bmi = diabetes.groupby(["smoking_status", "bmi_category"])["diagnosed_diabetes"].mean() * 100
smoking_bmi = smoking_bmi.unstack()
smoking_bmi.plot(kind="bar", ax=axes[0], width=0.8)
axes[0].set_title("Diabetes Rate: Smoking Status × BMI Category", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Smoking Status", fontsize=12)
axes[0].set_ylabel("Diabetes Rate (%)", fontsize=12)
axes[0].legend(title="BMI Category")
axes[0].tick_params(axis="x", rotation=0)

sns.boxplot(data=diabetes, x="smoking_status", y="physical_activity_minutes_per_week", 
            hue="diagnosed_diabetes", ax=axes[1])
axes[1].set_title("Physical Activity by Smoking Status & Diabetes", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Smoking Status", fontsize=12)
axes[1].set_ylabel("Physical Activity (min/week)", fontsize=12)
axes[1].legend(title="Diabetes", labels=["Non-Diabetic", "Diabetic"])

plt.tight_layout()
plt.show()


pivot_table = diabetes.pivot_table(
    values="diagnosed_diabetes", 
    index="smoking_status", 
    columns="bmi_category", 
    aggfunc="mean"
) * 100

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={"label": "Diabetes Rate (%)"})
plt.title("Diabetes Rate (%): Smoking Status × BMI Category", fontsize=14, fontweight="bold")
plt.xlabel("BMI Category", fontsize=12)
plt.ylabel("Smoking Status", fontsize=12)
plt.tight_layout()
plt.show()


cat_cols = diabetes.select_dtypes(include="object")
    
for col in cat_cols:
    unique_categories = diabetes[col].unique()
    num_unique_categories = diabetes[col].nunique()

    print(f"Total Unique {col.title()}: {num_unique_categories}")
    print(f"Unique {col.title()}: {unique_categories}")
    print("="*100)
    print("\n")


diabetes["metabolic_risk"] = (
    (diabetes["bmi"] >= 30).astype(int) +
    (diabetes["waist_to_hip_ratio"] > 0.9).astype(int) +
    (diabetes["systolic_bp"] >= 130).astype(int) +
    (diabetes["triglycerides"] >= 150).astype(int)
)

diabetes["cholesterol_ratio"] = diabetes["cholesterol_total"] / (diabetes["hdl_cholesterol"] + 1)
diabetes["age_bmi_risk"] = diabetes["age"] * diabetes["bmi"] / 100
diabetes["genetic_lifestyle_risk"] = diabetes["family_history_diabetes"] * diabetes["bmi"]

diabetes["cv_risk_score"] = (
    diabetes["hypertension_history"] +
    diabetes["cardiovascular_history"] +
    (diabetes["systolic_bp"] > 140).astype(int)
)


new_features_association = ["metabolic_risk", "cholesterol_ratio", "age_bmi_risk", "genetic_lifestyle_risk",
                            "cv_risk_score"]

corr = {
    feature: pearsonr(diabetes[feature], diabetes["diagnosed_diabetes"])[0]
    for feature in new_features_association
}

corr_df = pd.DataFrame(data=list(corr.items()), 
                       columns=["Feature", "Pearson Correlation"]).sort_values(by="Pearson Correlation", ascending=False)


from scipy.stats import chi2_contingency

selected_cols = ["family_history_diabetes", "hypertension_history", "cardiovascular_history", 
                 "alcohol_consumption_per_week", "gender", "ethnicity", "education_level", 
                 "income_level", "smoking_status", "employment_status", "cv_risk_score"]

chi2_results = {}

for col in selected_cols:
    tbl = pd.crosstab(diabetes[col], diabetes["diagnosed_diabetes"])

    chi2_stat, p_val, _, _ = chi2_contingency(tbl)

    decision = ["Reject Null => Strong evidence of association -> keep the feature" if p_val < 0.05
                else "Accept Null => No strong evidence of association -> can drop the feature"]
    
    chi2_results[col] = {
        "chi2_stat": chi2_stat,
        "p_value": p_val,
        "decision": decision
    }

chi2_df = pd.DataFrame(data=chi2_results).T.sort_values(by="p_value")
pd.set_option("display.max_colwidth", None)
chi2_df


from sklearn.model_selection import train_test_split

np.random.seed(42)

X = diabetes.drop(columns=["id", "diagnosed_diabetes", "age_category", "bmi_category"])
y = diabetes["diagnosed_diabetes"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)


from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

onehot_cat_features = ["gender", "ethnicity", "smoking_status", "employment_status"]
education_order = ['No formal', 'Highschool', 'Graduate', 'Postgraduate']
incomelevel_order = ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High']
ordinal_cat_features = ["education_level", "income_level"]

ordinal_encoder = OrdinalEncoder(
    categories=[education_order, incomelevel_order],
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

one_hot = OneHotEncoder(
    handle_unknown="ignore",
    drop="first"
)

transformer = ColumnTransformer(transformers=[
    ("ordinal_encode", ordinal_encoder, ordinal_cat_features),
    ("one_hot", one_hot, onehot_cat_features)
], remainder="passthrough")

X_train_encoded = transformer.fit_transform(X_train)
X_val_encoded = transformer.transform(X_val)

feature_names = transformer.get_feature_names_out()
feature_names = [col.replace("remainder__", "").replace("one_hot__", "").replace("ordinal_encode__", "") for col in feature_names]

X_train = pd.DataFrame(data=X_train_encoded, columns=feature_names, index=X_train.index)
X_val = pd.DataFrame(data=X_val_encoded, columns=feature_names, index=X_val.index)


X_train[:5]


X_val[:5]


from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
    )

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# Stratified sampling for faster training
X_train_sample, _, y_train_sample, _ = train_test_split(
    X_train, y_train, 
    train_size=100000,
    stratify=y_train,
    random_state=42
)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', verbosity=0, random_state=42),
    "LightGBM": LGBMClassifier(verbose=-1, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', verbosity=0, random_state=42),
    "Naive Bayes": GaussianNB()
}

model_scores = []

for name, model in models.items():
    model.fit(X_train_sample, y_train_sample)

    train_score = model.score(X_train_sample, y_train_sample)
    test_score = model.score(X_val, y_val)

    y_probs = model.predict_proba(X_val)[:,1]
    roc_score = roc_auc_score(y_val, y_probs)

    model_scores.append(
        {   
            "model": name,
            "training accuracy": train_score,
            "test accuracy": test_score,
            "roc score": roc_score
        }
    )

df_results = pd.DataFrame(data=model_scores).sort_values("roc score", ascending=False)
df_results.set_index(keys="model", inplace=True)
df_results


ax = df_results.plot(kind="bar", figsize=(14,6))
ax.tick_params(rotation=0)
ax.set_xlabel("Model", fontsize=14, labelpad=10)
ax.set_ylabel("Score (%)", fontsize=14, labelpad=10)
ax.set_title("Mode Scores", fontsize=20, fontweight="bold")

plt.tight_layout()
plt.show()


from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=2500,
    learning_rate=0.15,
    num_leaves=120,
    max_depth=2,
    colsample_bytree=0.5,
    subsample=0.85,
    reg_alpha=5.0,
    reg_lambda=20.0,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
    metric="auc",
    objective="binary",
    boosting_type="gbdt",
    verbosity=-1
)

model.fit(X_train, y_train)

y_preds = model.predict(X_val)


results = pd.DataFrame(
    {
        "actual": y_val,
        "predicted": y_preds
    }
)

print(f"FIRST 10 PREDICTIONS:")
results[:10]


y_probs = model.predict_proba(X_val)[:,1]

roc_score = roc_auc_score(y_val, y_probs)
print(f"ROC Score: {roc_score}")
print("Different ROC scores beacuse of the training data size!")


from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_val, y_probs)

plt.plot(fpr, tpr, color="olive", label="ROC")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve", fontsize=14, fontweight="bold")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()


test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
test_df_2 = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
test_df[:5]


test_df.shape


test_df["metabolic_risk"] = (
    (test_df["bmi"] >= 30).astype(int) +
    (test_df["waist_to_hip_ratio"] > 0.9).astype(int) +
    (test_df["systolic_bp"] >= 130).astype(int) +
    (test_df["triglycerides"] >= 150).astype(int)
)

test_df["cholesterol_ratio"] = test_df["cholesterol_total"] / (test_df["hdl_cholesterol"] + 1)
test_df["age_bmi_risk"] = test_df["age"] * test_df["bmi"] / 100
test_df["genetic_lifestyle_risk"] = test_df["family_history_diabetes"] * test_df["bmi"]

test_df["cv_risk_score"] = (
    test_df["hypertension_history"] +
    test_df["cardiovascular_history"] +
    (test_df["systolic_bp"] > 140).astype(int)
)


test_encoded = transformer.transform(test_df)

test_df = pd.DataFrame(data=test_encoded, columns=feature_names, index=test_df.index)
test_df[:5]


kaggle_predictions = model.predict_proba(test_df)[:,1]

print("First 10 predictions:")
print(kaggle_predictions[:10])


submission = pd.DataFrame(
    {
        "id": test_df_2["id"],
        "diagnosed_diabetes": kaggle_predictions
    }
)
submission.to_csv("submission.csv", index=False)
print(f"\nSubmission saved! Total predictions: {len(kaggle_predictions)}")

