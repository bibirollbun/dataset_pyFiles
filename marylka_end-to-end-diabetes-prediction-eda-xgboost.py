import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
from scipy.interpolate import make_interp_spline

from sklearn.preprocessing import LabelEncoder
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score

import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    "font.size": 12,
    "axes.edgecolor": "#333333",
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "figure.figsize": (10, 6),
    "grid.color": "#AAAAAA",
    "grid.alpha": 0.3,
    "axes.grid": True,
})


colors = ["#08090a", "#d3e7ed" ,"#3b63ad", "#415A77", "#6b8bb5"]
custom_cmap = LinearSegmentedColormap.from_list("custom_blues", colors)


#help viz functions
def barplot(ax, labels, values, title, cmap=custom_cmap):

    
    for i, (lab, val) in enumerate(zip(labels, values)):
        ax.bar(lab, val, color=cmap(val / max(values)))
        
    ax.set_title(title)



df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


test.head()


df.head()


len(df.columns)


df.info()


df.describe()


Age_groups = df.groupby('age').size().reset_index(name="number")


Age_groups


labels = Age_groups['age']
values = Age_groups['number']

fig, ax = plt.subplots()
barplot(ax, labels, values, 'Age distribution')
plt.show()


fig, ax = plt.subplots()
ax.boxplot(df['age'], vert=False)
plt.show()


age_grouped = (
    df
    .groupby(["age", "diagnosed_diabetes"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={0: "no diabetic", 1: "diabetic"})
)

age_grouped.head()


fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("#F5F5F5")
ax.set_facecolor("white")


norm = plt.Normalize(
    vmin=min(age_grouped.min()),
    vmax=max(age_grouped.max())
)




ax.plot(
    age_grouped.index,
    age_grouped["diabetic"],
    lw=3,
    color=custom_cmap(norm(age_grouped["diabetic"].mean())),
    label="diabetic"
)


ax.plot(
    age_grouped.index,
    age_grouped["no diabetic"],
    lw=3,
    color=custom_cmap(norm(age_grouped["no diabetic"].mean())),
    label="no diabetic"
)

ax.set_title(
    "Number of people diabetic and no-diabetic vs. Age",
    fontsize=18,
    fontweight="bold",
    pad=15
)

ax.set_xlabel("Age", fontsize=13)
ax.set_ylabel("Number of people", fontsize=13)

ax.legend(frameon=False)
ax.grid(alpha=0.3)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.show()


df.groupby('alcohol_consumption_per_week').size()


alcohol_grouped = (
    df
    .groupby(["alcohol_consumption_per_week", "diagnosed_diabetes"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={0: "no diabetic", 1: "diabetic"})
)

alcohol_grouped.head()


fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.patch.set_facecolor("#F5F5F5")

max_val = alcohol_grouped.values.max()

plots = [
    ("no diabetic", "no diabetic and alcohol usage"),
    ("diabetic", "diabetic and alcohol usage")
]

for ax, (col, title) in zip(axes, plots):
    ax.set_facecolor("white")

    bars = ax.bar(
        alcohol_grouped.index,
        alcohol_grouped[col]
    )

    for bar, val in zip(bars, alcohol_grouped[col]):
        bar.set_color(custom_cmap(val / max_val))
        bar.set_edgecolor("#333333")
        bar.set_linewidth(0.4)

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Alcohol per week")
    ax.grid(axis="y", alpha=0.3)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

axes[0].set_ylabel("Number of people")

plt.tight_layout()
plt.show()


df['physical_activity_minutes_per_week'].max()


pd.cut(df["physical_activity_minutes_per_week"],bins=10)


df['physical_activity_minutes_group'] = pd.cut(df["physical_activity_minutes_per_week"],[0, 15, 30, 60, 120, 400, 500, 600], precision=0, labels=['0', '15', '30', '60', '120', '400', '500'])


phys_grouped = (
    df
    .groupby(["physical_activity_minutes_group", "diagnosed_diabetes"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={0: "no diabetic", 1: "diabetic"})
)

phys_grouped.head()


fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.patch.set_facecolor("#F5F5F5")

max_val = phys_grouped.values.max()

plots = [
    ("no diabetic", "no diabetic and phisical activity"),
    ("diabetic", "diabetic and phisical activity")
]

for ax, (col, title) in zip(axes, plots):
    ax.set_facecolor("white")

    bars = ax.bar(
        phys_grouped.index,
        phys_grouped[col]
    )

    for bar, val in zip(bars, phys_grouped[col]):
        bar.set_color(custom_cmap(val / max_val))
        bar.set_edgecolor("#333333")
        bar.set_linewidth(0.4)

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Physical activity in min. per week")
    ax.grid(axis="y", alpha=0.3)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

axes[0].set_ylabel("Number of people")

plt.tight_layout()
plt.show()


df["diet_score"]


diet_sore_mean_value_in_both_groups = df.groupby('diagnosed_diabetes')['diet_score'].mean()

diet_sore_mean_value_in_both_groups.head()


df['sleep_hours_per_day'].min()


df['sleep_hours_per_day'].max()


df['sleep_hours_per_day_group'] = pd.cut(df["sleep_hours_per_day"],[0, 4, 5, 6, 7, 8, 9, 10], precision=0, labels=['3', '4', '5', '6', '7', '8', '9'])


sleep_hours_group = (df
    .groupby(['sleep_hours_per_day_group','diagnosed_diabetes'])
    .size()
    .unstack(fill_value=0)
    .rename(columns={0: "no diabetic", 1: "diabetic"})
)


sleep_hours_group.head()


fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.patch.set_facecolor("#F5F5F5")

max_val = sleep_hours_group.values.max()

plots = [
    ("no diabetic", "no diabetic and sleeping houers"),
    ("diabetic", "diabetic and sleeping houers")
]

for ax, (col, title) in zip(axes, plots):
    ax.set_facecolor("white")

    bars = ax.bar(
        sleep_hours_group.index,
        sleep_hours_group[col]
    )

    for bar, val in zip(bars, sleep_hours_group[col]):
        bar.set_color(custom_cmap(val / max_val))
        bar.set_edgecolor("#333333")
        bar.set_linewidth(0.4)

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Daily sleeping houers")
    ax.grid(axis="y", alpha=0.3)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

axes[0].set_ylabel("Number of people")

plt.tight_layout()
plt.show()


df['screen_time_hours_per_day']


print(df['screen_time_hours_per_day'].min(), df['screen_time_hours_per_day'].max())


df['screen_time_hours_per_day_group'] = pd.cut(df["screen_time_hours_per_day"],[0, 5, 10, 15], precision=0, labels=['(0,5]', '(5,10]', '(10,16]'])


screen_time_group = (df
    .groupby(['screen_time_hours_per_day_group','diagnosed_diabetes'])
    .size()
    .unstack(fill_value=0)
    .rename(columns={0: "no diabetic", 1: "diabetic"})
)
screen_time_group.head()


fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.patch.set_facecolor("#F5F5F5")

max_val = screen_time_group.values.max()

plots = [
    ("no diabetic", "no diabetic and screen houers"),
    ("diabetic", "diabetic and screen houers")
]

for ax, (col, title) in zip(axes, plots):
    ax.set_facecolor("white")

    bars = ax.bar(
        screen_time_group.index,
        screen_time_group[col]
    )

    for bar, val in zip(bars, screen_time_group[col]):
        bar.set_color(custom_cmap(val / max_val))
        bar.set_edgecolor("#333333")
        bar.set_linewidth(0.4)

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Daily screen time")
    ax.grid(axis="y", alpha=0.3)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

axes[0].set_ylabel("Number of people")

plt.tight_layout()
plt.show()


df['bmi'].min(), df['bmi'].max()


df['bmi']


df['bmi_types'] = pd.cut(df["bmi"],[0, 24.9, 29.9, 30], precision=0, labels=['normal', 'slightly obese', 'obese'])


bmi_grouped = (
    df
    .groupby(["bmi_types", "diagnosed_diabetes"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={0: "no diabetic", 1: "diabetic"})
)

bmi_grouped.head()


fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.patch.set_facecolor("#F5F5F5")

max_val = bmi_grouped.values.max()

plots = [
    ("no diabetic", "no diabetic and bmi"),
    ("diabetic", "diabetic and bmi")
]

for ax, (col, title) in zip(axes, plots):
    ax.set_facecolor("white")

    bars = ax.bar(
        bmi_grouped.index,
        bmi_grouped[col]
    )

    for bar, val in zip(bars, bmi_grouped[col]):
        bar.set_color(custom_cmap(val / max_val))
        bar.set_edgecolor("#333333")
        bar.set_linewidth(0.4)

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("BMI")
    ax.grid(axis="y", alpha=0.3)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

axes[0].set_ylabel("Number of people")

plt.tight_layout()
plt.show()


df['waist_to_hip_ratio'].min(), df['waist_to_hip_ratio'].max()


waist_to_hip_ratio_mean_value_in_both_groups = df.groupby('diagnosed_diabetes')['waist_to_hip_ratio'].mean()

waist_to_hip_ratio_mean_value_in_both_groups.head()


df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
df['MAP'] = df['diastolic_bp'] + ( df['systolic_bp'] - df['diastolic_bp'] )/3


pulse_pressure_grouped = (
    df
    .groupby(["pulse_pressure", "diagnosed_diabetes"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={0: "no diabetic", 1: "diabetic"})
)

pulse_pressure_grouped.head()


map_grouped = (
    df
    .groupby(["MAP", "diagnosed_diabetes"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={0: "no diabetic", 1: "diabetic"})
)

map_grouped.head()


fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.patch.set_facecolor("#F5F5F5")

max_val = pulse_pressure_grouped.values.max()

plots = [
    ("no diabetic", "no diabetic and pulse pressure"),
    ("diabetic", "diabetic and pulse pressure")
]

for ax, (col, title) in zip(axes, plots):
    ax.set_facecolor("white")

    bars = ax.bar(
        pulse_pressure_grouped.index,
        pulse_pressure_grouped[col]
    )

    for bar, val in zip(bars, pulse_pressure_grouped[col]):
        bar.set_color(custom_cmap(val / max_val))
        bar.set_edgecolor("#333333")
        bar.set_linewidth(0.4)

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Pulse pressure")
    ax.grid(axis="y", alpha=0.3)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

axes[0].set_ylabel("Number of people")

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.patch.set_facecolor("#F5F5F5")

max_val = map_grouped.values.max()

plots = [
    ("no diabetic", "no diabetic and MAP"),
    ("diabetic", "diabetic and MAP")
]

for ax, (col, title) in zip(axes, plots):
    ax.set_facecolor("white")

    bars = ax.bar(
        map_grouped.index,
        map_grouped[col]
    )

    for bar, val in zip(bars, map_grouped[col]):
        bar.set_color(custom_cmap(val / max_val))
        bar.set_edgecolor("#333333")
        bar.set_linewidth(0.4)

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("MAP")
    ax.grid(axis="y", alpha=0.3)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

axes[0].set_ylabel("Number of people")

plt.tight_layout()
plt.show()


df['heart_rate'].min(), df['heart_rate'].max()


heart_rate_grouped = (
    df
    .groupby(["heart_rate", "diagnosed_diabetes"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={0: "no diabetic", 1: "diabetic"})
)

heart_rate_grouped.head()


fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.patch.set_facecolor("#F5F5F5")

max_val = heart_rate_grouped.values.max()

plots = [
    ("no diabetic", "no diabetic and heart rate"),
    ("diabetic", "diabetic and heart rate")
]

for ax, (col, title) in zip(axes, plots):
    ax.set_facecolor("white")

    bars = ax.bar(
        heart_rate_grouped.index,
        heart_rate_grouped[col]
    )

    for bar, val in zip(bars, heart_rate_grouped[col]):
        bar.set_color(custom_cmap(val / max_val))
        bar.set_edgecolor("#333333")
        bar.set_linewidth(0.4)

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("heart rate")
    ax.grid(axis="y", alpha=0.3)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

axes[0].set_ylabel("Number of people")

plt.tight_layout()
plt.show()


df['cholesterol_total'].min(), df['cholesterol_total'].max()


df['hdl_cholesterol'].min(), df['hdl_cholesterol'].max()


df['ldl_cholesterol'].min(), df['ldl_cholesterol'].max()


df['triglycerides'].min(), df['triglycerides'].max()


df.groupby('gender').size()


df['gender'] = df['gender'].map({'Male': 1, 'Other':1, 'Female': 0})
test['gender'] = test['gender'].map({'Male': 1, 'Other':1, 'Female': 0})


df.groupby('gender').size()


df.groupby('ethnicity').size()


le = LabelEncoder()
df['ethnicity'] = le.fit_transform(df['ethnicity'])
test['ethnicity'] = le.fit_transform(test['ethnicity'])


df.groupby('ethnicity').size()


df.groupby('education_level').size()


le = LabelEncoder()
df['education_level'] = le.fit_transform(df['education_level'])
test['education_level'] = le.fit_transform(test['education_level'])


df.groupby('education_level').size()


df.groupby('income_level').size()


le = LabelEncoder()
df['income_level'] = le.fit_transform(df['income_level'])
test['income_level'] = le.fit_transform(test['income_level'])


df.groupby('income_level').size()


df.groupby('smoking_status').size()


le = LabelEncoder()
df['smoking_status'] = le.fit_transform(df['smoking_status'])
test['smoking_status'] = le.fit_transform(test['smoking_status'])


df.groupby('smoking_status').size()


df.groupby('employment_status').size()


le = LabelEncoder()
df['employment_status'] = le.fit_transform(df['employment_status'])
test['employment_status'] = le.fit_transform(test['employment_status'])


df.groupby('family_history_diabetes').size()


df.groupby('hypertension_history').size()


df.groupby('cardiovascular_history').size()


df.groupby('diagnosed_diabetes').size()


columns = ['age', 'alcohol_consumption_per_week',
       'physical_activity_minutes_per_week', 'diet_score',
       'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
       'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
       'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
       'triglycerides', 'gender', 'ethnicity', 'education_level',
       'income_level', 'smoking_status', 'employment_status',
       'family_history_diabetes', 'hypertension_history',
       'cardiovascular_history']


X = df[columns]
y = df["diagnosed_diabetes"]

imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)

X_train, X_val, y_train, y_val = train_test_split(
    X_imputed, y, test_size=0.2, stratify=y, random_state=42
)



model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=42
)

model.fit(X_train, y_train)


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# Mean absolute SHAP value per feature
shap_importance = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({
    "feature": X_train.columns,
    "shap_importance": shap_importance
}).sort_values(by="shap_importance", ascending=False)

shap_df


threshold = shap_df["shap_importance"].mean()

selected_features = shap_df[
    shap_df["shap_importance"] >= threshold
]["feature"].tolist()

selected_features


shap.summary_plot(shap_values, X_train, plot_type="bar")


y_proba = model.predict_proba(X_val)[:, 1]

thresholds = np.linspace(0.1, 0.9, 81)

results = []

for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    precision = precision_score(y_val, y_pred_t, zero_division=0)
    recall = recall_score(y_val, y_pred_t)

    results.append((t, precision, recall))

results = np.array(results)

# warunek medyczny
MIN_PRECISION = 0.60

valid = results[results[:, 1] >= MIN_PRECISION]
best_threshold = valid[np.argmax(valid[:, 2])][0]

best_threshold


y_pred_opt = (y_proba >= best_threshold).astype(int)

print(f"Optimal threshold: {best_threshold:.2f}")
print("Recall:", recall_score(y_val, y_pred_opt))
print("Precision:", precision_score(y_val, y_pred_opt))


y_pred_opt = (y_proba >= 0.5).astype(int)

print(f"Optimal threshold: {0.5:.2f}")
print("Recall:", recall_score(y_val, y_pred_opt))
print("Precision:", precision_score(y_val, y_pred_opt))


X_test = test.drop("id", axis=1)

imputer = SimpleImputer(strategy="median")
X_test_imputed = pd.DataFrame(
    imputer.fit_transform(X_test),
    columns=X_test.columns
)


y_test_proba = model.predict_proba(X_test_imputed)[:, 1]


submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": y_test_proba
})


submission.to_csv("submission.csv", index=False)

print("Submission file created: submission.csv")
submission.head()

