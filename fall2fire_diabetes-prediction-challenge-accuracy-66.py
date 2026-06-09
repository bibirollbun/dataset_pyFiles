import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df.head()


df.isnull().sum()


df.shape


df.describe()


plt.figure(figsize=(10,7))
sns.boxplot(df)
plt.show()


df=df.drop(columns=['id'])


for i in df:
    x=df[i].value_counts()
    print(x)
    sns.barplot(x=x.index,y=x.values)
    plt.show()


# Ordinal columns 
education_mapping = {
    "No formal": 0,
    "Highschool": 1,
    "Graduate": 2,
    "Postgraduate": 3
}
df['education_level'] = df['education_level'].map(education_mapping)
income_mapping = {
    "Low": 0,
    "Lower-Middle": 1,
    "Middle": 2,
    "Upper-Middle": 3,
    "High": 4
}
df['income_level'] = df['income_level'].map(income_mapping)

df = pd.get_dummies(df,columns=['gender', 'ethnicity', 'smoking_status', 'employment_status'],
                    drop_first=True,dtype='int')


df.head(10)


corr = df.corr()
plt.figure(figsize=(30,20))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# Core feature engineering
df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1)
df['lipid_risk'] = (df['cholesterol_total'] + df['ldl_cholesterol'] +
                    df['triglycerides']) - df['hdl_cholesterol']
df['obesity_risk'] = df['bmi'] * df['waist_to_hip_ratio']
df['lifestyle_risk'] = (
    df['alcohol_consumption_per_week'] +
    df['screen_time_hours_per_day'] -
    (df['physical_activity_minutes_per_week'] / 60)
)

# ----- Extra engineered features (safe version: add only if available) -----
if 'resting_heart_rate' in df.columns and 'sleep_hours_per_day' in df.columns:
    df["bp_stress"] = df["systolic_bp"] + (df["resting_heart_rate"] / (df["sleep_hours_per_day"] + 1))

if 'waist_to_hip_ratio' in df.columns and 'bmi' in df.columns:
    df["waist_bmi_ratio"] = df["waist_to_hip_ratio"] * df["bmi"]

if 'ldl_cholesterol' in df.columns and 'hdl_cholesterol' in df.columns:
    df["cholesterol_ratio"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + 1)

if 'physical_activity_minutes_per_week' in df.columns and 'alcohol_consumption_per_week' in df.columns:
    df["activity_effectiveness"] = df["physical_activity_minutes_per_week"] / (df["alcohol_consumption_per_week"] + 1)

if 'age' in df.columns and 'cholesterol_total' in df.columns and 'bmi' in df.columns:
    df["age_metabolic_risk"] = df["age"] * (df["cholesterol_total"] + df["bmi"])



df.head()


corr = df.corr()
plt.figure(figsize=(30,20))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Target and features
X = df.drop('diagnosed_diabetes', axis=1)
y = df['diagnosed_diabetes']

# Identify categorical columns
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
weights = {0: class_weights[0], 1: class_weights[1]}
# Model
model = CatBoostClassifier(
    iterations=3000,
    depth=10,
    learning_rate=0.02,
    subsample=0.9,
    colsample_bylevel=0.8,
    l2_leaf_reg=7,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    bagging_temperature=0.5,
    od_wait=80,
    class_weights=weights,
    verbose=200
)

# Fit
model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=cat_cols, use_best_model=True)

# Validation evaluation using probability
y_valid_prob = model.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(y_valid, y_valid_prob)
print("Validation ROC-AUC:", auc)


from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

fpr, tpr, thresholds = roc_curve(y_valid, y_valid_prob)
best_threshold = thresholds[np.argmax(tpr - fpr)]
y_pred_thr = (y_valid_prob >= best_threshold).astype(int)

print(classification_report(y_valid, y_pred_thr))


print("Best Threshold:", best_threshold)


# ------------------------------
# Load test data
# ------------------------------
test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

# ------------------------------
# Preprocessing (same as training)
# ------------------------------

# Ordinal Encoding
education_mapping = {
    "No formal": 0,
    "Highschool": 1,
    "Graduate": 2,
    "Postgraduate": 3
}
income_mapping = {
    "Low": 0,
    "Lower-Middle": 1,
    "Middle": 2,
    "Upper-Middle": 3,
    "High": 4
}

test_df['education_level'] = test_df['education_level'].map(education_mapping)
test_df['income_level'] = test_df['income_level'].map(income_mapping)

# One-hot encoding
test_df = pd.get_dummies(
    test_df,
    columns=['gender', 'ethnicity', 'smoking_status', 'employment_status'],
    drop_first=True,
    dtype='int'
)

# Core feature engineering
# Core feature engineering
test_df['bp_ratio'] = test_df['systolic_bp'] / (test_df['diastolic_bp'] + 1)
test_df['lipid_risk'] = (test_df['cholesterol_total'] + test_df['ldl_cholesterol'] +
                         test_df['triglycerides']) - test_df['hdl_cholesterol']
test_df['obesity_risk'] = test_df['bmi'] * test_df['waist_to_hip_ratio']
test_df['lifestyle_risk'] = (
    test_df['alcohol_consumption_per_week'] +
    test_df['screen_time_hours_per_day'] -
    (test_df['physical_activity_minutes_per_week'] / 60)
)

# ----- Extra engineered features (safe version: add only if available) -----
if 'resting_heart_rate' in test_df.columns and 'sleep_hours_per_day' in test_df.columns:
    test_df["bp_stress"] = test_df["systolic_bp"] + (test_df["resting_heart_rate"] / (test_df["sleep_hours_per_day"] + 1))

if 'waist_to_hip_ratio' in test_df.columns and 'bmi' in test_df.columns:
    test_df["waist_bmi_ratio"] = test_df["waist_to_hip_ratio"] * test_df["bmi"]

if 'ldl_cholesterol' in test_df.columns and 'hdl_cholesterol' in test_df.columns:
    test_df["cholesterol_ratio"] = test_df["ldl_cholesterol"] / (test_df["hdl_cholesterol"] + 1)

if 'physical_activity_minutes_per_week' in test_df.columns and 'alcohol_consumption_per_week' in test_df.columns:
    test_df["activity_effectiveness"] = test_df["physical_activity_minutes_per_week"] / (test_df["alcohol_consumption_per_week"] + 1)

if 'age' in test_df.columns and 'cholesterol_total' in test_df.columns and 'bmi' in test_df.columns:
    test_df["age_metabolic_risk"] = test_df["age"] * (test_df["cholesterol_total"] + test_df["bmi"])
# ------------------------------
# Align columns (Important)
# ------------------------------
X_test = test_df.reindex(columns=X_train.columns, fill_value=0)

# ------------------------------
# Model Prediction (probabilities for leaderboard score)
# ------------------------------
y_pred = model.predict_proba(X_test)[:, 1]

# ------------------------------
# Create submission file
# ------------------------------
submission = pd.DataFrame({
    "id": test_df["id"],
    "diagnosed_diabetes": y_pred
})

submission.to_csv("submission.csv", index=False)
print("submission.csv saved successfully!")




