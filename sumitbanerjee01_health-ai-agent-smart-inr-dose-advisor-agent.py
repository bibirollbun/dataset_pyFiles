# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



# Load dataset (adjust path if needed)
import pandas as pd
path = '/kaggle/input/inr-warfarin-timeline-datasets/INR_Warfarin_100k_Timeline_Dataset_Reconstructed_Age_Gender.csv'
df = pd.read_csv(path, parse_dates=['Visit_Date','Dose_Start_Date','Dose_End_Date','Next_INR_Test_Date'])
print('Rows, Columns:', df.shape)
df.head()

# Basic statistics and missing values
print('Columns:', df.columns.tolist())
print('\nMissing values per column:')
print(df.isna().sum())
print('\nDescriptive statistics (numeric):')
display(df.describe(include='number').round(3))

# Basic visualizations
import matplotlib.pyplot as plt

# Distribution of Current_INR
plt.figure(figsize=(8,4))
plt.hist(df['Current_INR'], bins=40)
plt.title('Distribution of Current_INR')
plt.xlabel('INR')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# Age vs Current_INR (sampled 3000 points)
plt.figure(figsize=(8,4))
plt.scatter(df['Age'].sample(3000, random_state=1), df['Current_INR'].sample(3000, random_state=1), s=5)
plt.title('Age vs Current_INR (sampled 3000 points)')
plt.xlabel('Age')
plt.ylabel('Current_INR')
plt.grid(True)
plt.show()

# Daily dose frequency
dose_counts = df['Daily_Dose_mg'].value_counts().sort_index()
plt.figure(figsize=(8,4))
plt.bar(dose_counts.index.astype(str), dose_counts.values)
plt.title('Daily Dose (mg) Frequency')
plt.xlabel('Dose (mg)')
plt.ylabel('Count')
plt.show()

# Diagnosis distribution
diag_counts = df['Diagnosis'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(diag_counts.values, labels=diag_counts.index, autopct='%1.1f%%')
plt.title('Diagnosis distribution')
plt.show()



# Supervised models for AdviceLabel (Increase/Maintain/Decrease) and Next_INR

import numpy as np

# Copy dataframe for modeling
df2 = df.copy()

# Map textual advice to numeric labels
label_map = {'Decrease Dose':0, 'Maintain Dose':1, 'Increase Dose':2}
df2['AdviceLabel'] = df2['Advice'].map(label_map)

# Create Next_INR if not present (synthetic small noise shift around Current_INR)
if 'Next_INR' not in df2.columns:
    df2['Next_INR'] = (df2['Current_INR'] + np.random.normal(0, 0.25, size=len(df2))).clip(1.0,5.0).round(2)

print('Label distribution:')
print(df2['AdviceLabel'].value_counts())

from sklearn.model_selection import train_test_split

# Feature columns for advice model
cat_cols = ['Gender','Diagnosis','Surgery_Type','Symptoms']
num_cols = ['Age','Previous_INR','Current_INR','Daily_Dose_mg','Dose_Duration_Days']

X = df2[cat_cols + num_cols]
y = df2['AdviceLabel']

# Train/test split for advice classifier
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train.shape, X_test.shape



# Classification report:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error

# Preprocessor for numeric + categorical features
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Advice classifier pipeline
clf_pipeline = Pipeline(steps=[
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

# Next INR regressor pipeline
reg_pipeline = Pipeline(steps=[
    ('pre', preprocessor),
    ('reg', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

# Fit classifier for advice
clf_pipeline.fit(X_train, y_train)
preds = clf_pipeline.predict(X_test)
print('Classification report:')
print(classification_report(y_test, preds, target_names=['Decrease','Maintain','Increase']))

# Train regressor to predict Next_INR
y_reg = df2['Next_INR']
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
reg_pipeline.fit(Xr_train, yr_train)
reg_preds = reg_pipeline.predict(Xr_test)
rmse = mean_squared_error(yr_test, reg_preds, squared=False)
print('\nRegressor RMSE:', round(rmse, 3))



# INR Status models (Low/Target/High) + saving models for the agent

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

print("Loading dataset...")
df = pd.read_csv(
    "/kaggle/input/inr-warfarin-timeline-datasets/INR_Warfarin_100k_Timeline_Dataset_Reconstructed_Age_Gender.csv",
    parse_dates=["Visit_Date","Dose_Start_Date","Dose_End_Date","Next_INR_Test_Date"]
)

# ---------- TARGET ----------
# INR status from Current_INR (Low / Target / High)
df["INR_Status"] = df["Current_INR"].apply(
    lambda x: "Low" if x < 2 else ("High" if x > 3 else "Target")
)
status_map = {"Low": 0, "Target": 1, "High": 2}
df["INR_Status_Label"] = df["INR_Status"].map(status_map)

# ---------- FEATURES ----------
cat_cols = ["Gender", "Diagnosis", "Surgery_Type", "Symptoms"]
num_cols = ["Age", "Previous_INR", "Current_INR", "Daily_Dose_mg", "Dose_Duration_Days"]

# Make categorical features safe for modeling
df[cat_cols] = df[cat_cols].fillna("Unknown").astype(str)

X = df[cat_cols + num_cols]
y_reg = df["Current_INR"]         # Regressor learns to approximate Current_INR
y_cls = df["INR_Status_Label"]    # Classifier learns INR status (Low/Target/High)

# Split into train/test
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)
_, _, y_cls_train, y_cls_test = train_test_split(
    X, y_cls, test_size=0.2, random_state=42
)

# ColumnTransformer preprocessing pipeline
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# ---------- REGRESSION MODEL ----------
reg_model = Pipeline([
    ("pre", preprocess),
    ("rf", RandomForestRegressor(n_estimators=200, random_state=42))
])
print("Training regression model...")
reg_model.fit(X_train, y_reg_train)

# ---------- CLASSIFICATION MODEL ----------
cls_model = Pipeline([
    ("pre", preprocess),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
])
print("Training classification model...")
cls_model.fit(X_train, y_cls_train)

# ---------- SAVE MODELS ----------
os.makedirs("models", exist_ok=True)
joblib.dump(reg_model, "models/next_inr_regressor.joblib")
joblib.dump(cls_model, "models/next_inr_status_clf.joblib")
joblib.dump(
    {"status_map": status_map, "cat_cols": cat_cols, "num_cols": num_cols},
    "models/meta_inr_models.joblib"
)

print("âœ… Models saved in 'models/' folder.")



# Agent Definitions: INRInterpreterAgent + DoseAdvisorAgent

import pandas as pd
import joblib

# Load previously saved models
reg = joblib.load("models/next_inr_regressor.joblib")
cls = joblib.load("models/next_inr_status_clf.joblib")
meta = joblib.load("models/meta_inr_models.joblib")

status_map = meta["status_map"]
inv_status_map = {v: k for k, v in status_map.items()}
cat_cols = meta["cat_cols"]
num_cols = meta["num_cols"]

def INRInterpreterAgent(current_inr):
    """
    Simple rule-based INR interpretation.
    Returns: "Low", "Target", or "High" based on current INR.
    """
    if current_inr < 2.0:
        return "Low"
    elif current_inr > 3.0:
        return "High"
    else:
        return "Target"


def DoseAdvisorAgent(row):
    """
    Takes a single visit (row as a dict) and returns dose & follow-up advice.

    Expected keys in `row`:
      - Visit_Date
      - Age, Gender
      - Diagnosis, Surgery_Type, Symptoms
      - Previous_INR, Current_INR
      - Daily_Dose_mg
      - Dose_Duration_Days
      - (optional) Dose_Pattern: e.g. "9/9/8"

    The agent:
      - Uses models to predict the next INR and INR status.
      - Adjusts the dose pattern up or down by 1 mg if needed.
      - Chooses follow-up time (2 days / 5 days / 1 week / 2 weeks / 1 month)
        based on INR levels and trends.
    """

    # Convert dict into a DataFrame row for the model
    row_df = pd.DataFrame([row])

    # Make categorical columns safe: replace NaN with "Unknown" and ensure string type
    row_df[cat_cols] = row_df[cat_cols].fillna("Unknown").astype(str)

    # Select only the columns the model was trained on
    Xrow = row_df[cat_cols + num_cols]

    # Model predictions
    pred_next_inr = reg.predict(Xrow)[0]
    cls_label = cls.predict(Xrow)[0]
    next_status = inv_status_map[int(cls_label)]

    current_inr = float(row["Current_INR"])
    previous_inr = float(row.get("Previous_INR", current_inr))

    # ----- Dose pattern handling -----
    # Use Dose_Pattern if provided, otherwise fall back to a single daily dose
    pattern_str_raw = str(row.get("Dose_Pattern", row["Daily_Dose_mg"]))
    try:
        pattern_list = [
            int(p.strip())
            for p in pattern_str_raw.split("/")
            if p.strip() != ""
        ]
    except Exception:
        # Fallback: single dose pattern
        pattern_list = [int(row["Daily_Dose_mg"])]
        pattern_str_raw = str(row["Daily_Dose_mg"])

    # Keep doses in a safe synthetic range [1, 12] mg
    pattern_list = [min(max(d, 1), 12) for d in pattern_list]

    # Base advice from predicted status
    if next_status == "Low":
        base_advice = "Increase Dose"
        new_pattern_list = [min(d + 1, 12) for d in pattern_list]
    elif next_status == "High":
        base_advice = "Decrease Dose"
        new_pattern_list = [max(d - 1, 1) for d in pattern_list]
    else:
        base_advice = "Maintain Dose"
        new_pattern_list = pattern_list[:]

    # ----- Follow-up duration logic (in days + human text) -----
    # We combine current INR, predicted INR and trend into simple heuristic rules.
    very_high_threshold = 4.5
    high_threshold = 3.5

    notes = []

    # Default values
    followup_days = 14       # 2 weeks
    followup_text = "2 weeks"

    # Very high INR scenario -> urgent re-check (2 days) and note about holding doses
    if current_inr >= very_high_threshold or pred_next_inr >= very_high_threshold:
        followup_days = 2
        followup_text = "2 days"
        # IMPORTANT: this is synthetic logic for a project, not medical advice.
        notes.append(
            "INR is in a very high synthetic range. In real clinical practice, "
            "doctors often consider holding 1â€“2 warfarin doses and repeating INR urgently."
        )

    # High INR: above target but not extreme -> 5 days follow-up
    elif current_inr > 3.0 or pred_next_inr > 3.0:
        followup_days = 5
        followup_text = "5 days"

    # Stable in range for at least two consecutive visits -> 1 month
    elif 2.0 <= current_inr <= 3.0 and 2.0 <= previous_inr <= 3.0:
        followup_days = 30
        followup_text = "1 month"

    # Slightly off (mild low or mild high) -> 1 week
    elif 1.8 <= current_inr < 2.0 or 3.0 < current_inr <= high_threshold:
        followup_days = 7
        followup_text = "1 week"
    else:
        # For any other situation, keep the default 2 weeks.
        followup_days = 14
        followup_text = "2 weeks"

    # Rebuild dose pattern string
    new_pattern_str = "/".join(str(d) for d in new_pattern_list)
    suggested_dose_day1 = new_pattern_list[0]

    visit_date = pd.to_datetime(row["Visit_Date"])
    next_test_date = (visit_date + pd.Timedelta(days=followup_days)).date()

    return {
        "Predicted_Next_INR": round(float(pred_next_inr), 2),
        "Predicted_Status": next_status,
        "Advice": base_advice,
        "Suggested_Dose_Pattern": new_pattern_str,
        "Suggested_Dose_mg": int(suggested_dose_day1),   # day-1 dose
        "Followup_Duration_Days": int(followup_days),
        "Followup_Duration_Text": followup_text,
        "Next_Test_Date": str(next_test_date),
        "Notes": " ".join(notes) if notes else "",
    }



# Interactive User Agent Demo (console-style input/output)

from datetime import datetime

def ask_date(prompt, default=None):
    """Ask user for a date with basic validation."""
    while True:
        s = input(prompt).strip()
        if s == "":
            if default is not None:
                return default
            return pd.Timestamp.today().normalize()
        try:
            return pd.to_datetime(s)
        except Exception:
            print("â�Œ Invalid date format. Example: 2025-06-18")

def ask_float(prompt, min_val=None, max_val=None):
    """Ask user for a float value with optional bounds."""
    while True:
        s = input(prompt).strip()
        try:
            v = float(s)
            if min_val is not None and v < min_val:
                print(f"â�Œ Value must be at least {min_val}.")
                continue
            if max_val is not None and v > max_val:
                print(f"â�Œ Value must be at most {max_val}.")
                continue
            return v
        except ValueError:
            print("â�Œ Please enter a numeric value (e.g. 2.5).")

def ask_int(prompt, min_val=None, max_val=None):
    """Ask user for an integer value with optional bounds."""
    while True:
        s = input(prompt).strip()
        try:
            v = int(s)
            if min_val is not None and v < min_val:
                print(f"â�Œ Value must be at least {min_val}.")
                continue
            if max_val is not None and v > max_val:
                print(f"â�Œ Value must be at most {max_val}.")
                continue
            return v
        except ValueError:
            print("â�Œ Please enter an integer value (e.g. 7, 8, 9).")

def ask_pattern(prompt):
    """
    Ask user for a dose pattern like '7', '7/8', '9/9/8'.
    Returns: (pattern_string, [int doses]).
    """
    while True:
        s = input(prompt).strip()
        if s == "":
            print("â�Œ Dose pattern cannot be empty.")
            continue
        try:
            parts = [p.strip() for p in s.split("/") if p.strip() != ""]
            vals = [int(p) for p in parts]
            if any(v < 1 or v > 12 for v in vals):
                print("â�Œ Each dose must be between 1 and 12 mg.")
                continue
            return s, vals
        except Exception:
            print("â�Œ Invalid pattern. Examples: '7', '7/8', '9/9/8'.")

print("ğŸ©º INR Dose Advisor â€“ USER INPUT MODE\n")

# 1) Visit date (usually the current INR test date)
visit_date = ask_date("Visit Date (YYYY-MM-DD) [blank = today]: ")

# 2) Basic demographics
age = ask_int("Age (years): ", min_val=1, max_val=120)
gender = input("Gender (Male/Female) [blank = Unknown]: ").strip() or "Unknown"

print("\nCommon Diagnosis: RHD, MS, MR, AS, AR")
diagnosis = input("Diagnosis [default RHD]: ").strip() or "RHD"

print("\nExamples Surgery_Type: MVR, AVR, DVR, None")
surgery_type = input("Surgery_Type [default DVR]: ").strip() or "None"

print("\nExamples Symptoms: None, Leg Pain, Head Cold Sensation, Bleeding, Bruising, Dizziness")
symptoms = input("Symptoms [default None]: ").strip() or "None"

# 3) Previous INR + date
prev_inr_date = ask_date("\nPrevious INR test date (YYYY-MM-DD): ")
prev_inr = ask_float("Previous INR value: ", min_val=0.5, max_val=8.0)

# 4) Current INR + date
curr_inr_date = ask_date("Current INR test date (YYYY-MM-DD) [blank = Visit Date]: ", default=visit_date)
curr_inr = ask_float("Current INR value: ", min_val=0.5, max_val=8.0)

# 5) Target INR max (for info only)
target_inr_max = ask_float("Target INR Max (e.g. 3.0) [blank = 3.0]: ", min_val=1.0, max_val=5.0)

# 6) Dose pattern
dose_pattern_str, pattern_list = ask_pattern("\nCurrent warfarin dose pattern (e.g. 7, 7/8, 9/9/8): ")

# Estimate how long this dose pattern has been used.
dose_days = (curr_inr_date - prev_inr_date).days
if dose_days <= 0:
    dose_days = len(pattern_list)  # fallback to pattern length

daily_dose_for_model = pattern_list[0]

# Build the row for the agent
user_row = {
    "Patient_ID": 99999,
    "Visit_Date": curr_inr_date,
    "Age": age,
    "Gender": gender,
    "Diagnosis": diagnosis,
    "Surgery_Type": surgery_type,
    "Symptoms": symptoms,
    "Previous_INR": prev_inr,
    "Current_INR": curr_inr,
    "Daily_Dose_mg": daily_dose_for_model,
    "Dose_Duration_Days": dose_days,
    "Target_INR_Max": target_inr_max,
    "Dose_Pattern": dose_pattern_str,
    "Previous_INR_Date": prev_inr_date,
    "Current_INR_Date": curr_inr_date,
}

print("\n=============== INPUT SUMMARY ===============")
print({
    "Age": user_row["Age"],
    "Gender": user_row["Gender"],
    "Diagnosis": user_row["Diagnosis"],
    "Surgery_Type": user_row["Surgery_Type"],
    "Previous_INR": user_row["Previous_INR"],
    "Previous_INR_Date": str(user_row["Previous_INR_Date"].date()),
    "Current_INR": user_row["Current_INR"],
    "Current_INR_Date": str(user_row["Current_INR_Date"].date()),
    "Dose_Pattern": user_row["Dose_Pattern"],
    "Dose_Duration_Days": user_row["Dose_Duration_Days"],
    "Symptoms": user_row["Symptoms"],
    "Visit_Date": str(user_row["Visit_Date"].date()),
})
print("=============================================")

print("\nINR Status (rule-based):", INRInterpreterAgent(user_row["Current_INR"]))

result = DoseAdvisorAgent(user_row)

print("\nğŸ’Š Dose Advisor Output:")
for k, v in result.items():
    print(f"  {k}: {v}")


