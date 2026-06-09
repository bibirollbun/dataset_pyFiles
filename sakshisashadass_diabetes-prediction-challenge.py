import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv") 
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv") 
num_cols = train.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = train.select_dtypes(include=['object']).columns.tolist()
train.head()


train.info()
train.describe()
train['diagnosed_diabetes'].value_counts(normalize=True)
num_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = train.select_dtypes(include=['object']).columns.tolist()

# Remove the id and target variable 
num_cols = [col for col in num_cols if col not in ['id', 'diagnosed_diabetes']]
train[num_cols].skew()
train[num_cols].kurt()
train[num_cols].hist(figsize=(15, 12), bins=30)
plt.tight_layout()
plt.show()


def percent_diabetic(df, col):
    """Returns % of diabetics in each category/bin of a column."""
    return (df.groupby(col)['diagnosed_diabetes'].mean() * 100).round(2)

binned_data = train.copy()

for col in num_cols:
    # Create 5 equal-width bins for each numeric column
    binned_data[col + "_bin"] = pd.qcut(train[col], q=5, duplicates='drop')

results = {}

for col in cat_cols:
    results[col] = percent_diabetic(train, col)

for col in num_cols:
    results[col] = percent_diabetic(binned_data, col + "_bin")

for col, breakdown in results.items():
    print(f"\n==============================")
    print(f"   {col} — % Diabetic Breakdown")
    print(f"==============================")
    print(breakdown)


import matplotlib.pyplot as plt
import seaborn as sns

# Select only numerical columns
num_cols = train.select_dtypes(include=['int64','float64']).columns

plt.figure(figsize=(14,10))
sns.heatmap(train[num_cols].corr(), 
            annot=True, 
            cmap='coolwarm', 
            fmt=".2f",
            square=True)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()



# -----------------------------
# 1. Separate features & target
# -----------------------------
X = train.drop(columns=['diagnosed_diabetes'])
y = train['diagnosed_diabetes']

# -----------------------------
# 2. Identify column types
# -----------------------------
num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# -----------------------------
# 3. Preprocessing pipeline
# -----------------------------
preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first'), cat_cols)
    ]
)

# -----------------------------
# 4. Logistic Regression model
# -----------------------------
log_reg = LogisticRegression(max_iter=500)

pipeline = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', log_reg)
])

# -----------------------------
# 5. Fit model
# -----------------------------
pipeline.fit(X, y)

# -----------------------------
# 6. Extract coefficients
# -----------------------------
# Get feature names after encoding
encoded_features = (
    num_cols +
    list(pipeline.named_steps['preprocess']
         .named_transformers_['cat']
         .get_feature_names_out(cat_cols))
)

coefficients = pipeline.named_steps['model'].coef_[0]

# Create a dataframe of coefficients
coef_df = pd.DataFrame({
    'feature': encoded_features,
    'coefficient': coefficients,
    'abs_coeff': np.abs(coefficients)
}).sort_values(by='abs_coeff', ascending=False)

coef_df



cat_cols = [
    'gender',
    'ethnicity',
    'education_level',
    'income_level',
    'smoking_status',
    'employment_status'
]
num_cols = [
    'age',
    'alcohol_consumption_per_week',
    'physical_activity_minutes_per_week',
    'diet_score',
    'sleep_hours_per_day',
    'screen_time_hours_per_day',
    'bmi',
    'waist_to_hip_ratio',
    'systolic_bp',
    'diastolic_bp',
    'heart_rate',
    'cholesterol_total',
    'hdl_cholesterol',
    'ldl_cholesterol',
    'triglycerides',
    'family_history_diabetes',
    'hypertension_history',
    'cardiovascular_history'
]



from sklearn.preprocessing import LabelEncoder

train_fe = train.copy()
test_fe = test.copy()

encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    train_fe[col] = le.fit_transform(train_fe[col].astype(str))
    test_fe[col] = le.transform(test_fe[col].astype(str))
    encoders[col] = le


def add_engineered_features(df):
    df = df.copy()

    # Pulse pressure
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']

    # Cholesterol ratios
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1)
    df['tg_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1)
    df['total_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1)

    # Obesity severity
    df['obesity_severity'] = df['bmi'] * df['waist_to_hip_ratio']

    # Sedentary index
    df['sedentary_index'] = df['screen_time_hours_per_day'] / (df['physical_activity_minutes_per_week'] + 1)

    # Diet × activity
    df['diet_activity_interaction'] = df['diet_score'] * df['physical_activity_minutes_per_week']

    # Metabolic score
    df['metabolic_score'] = (
        df['tg_hdl_ratio'] +
        df['ldl_hdl_ratio'] +
        df['obesity_severity']
    )

    # Smoking risk
    df['smoking_risk'] = df['smoking_status']

    # Employment risk
    df['employment_risk'] = (df['employment_status'] == df['employment_status'].max()).astype(int)

    # Socioeconomic score
    df['socioeconomic_score'] = df['education_level'] + df['income_level']

    # Age × family history
    df['age_family_interaction'] = df['age'] * df['family_history_diabetes']

    # BMI × gender
    df['bmi_gender_interaction'] = df['bmi'] * df['gender']

    # Activity × employment
    df['activity_employment_interaction'] = (
        df['physical_activity_minutes_per_week'] * df['employment_risk']
    )

    # Income × diet
    df['income_diet_interaction'] = df['income_level'] * df['diet_score']

    return df

train_fe = add_engineered_features(train_fe)
test_fe = add_engineered_features(test_fe)



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = train_fe.drop(columns=['diagnosed_diabetes', 'id'])
y = train_fe['diagnosed_diabetes']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

preds = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, preds))



from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='auc'
)

xgb_model.fit(X_train, y_train)

# Predictions
xgb_pred = xgb_model.predict(X_val)
xgb_pred_proba = xgb_model.predict_proba(X_val)[:, 1]

# Metrics
xgb_acc = accuracy_score(y_val, xgb_pred)
xgb_auc = roc_auc_score(y_val, xgb_pred_proba)

print("XGBoost Accuracy:", xgb_acc)
print("XGBoost AUC:", xgb_auc)



import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary'
)

lgb_model.fit(X_train, y_train)

# Predictions
lgb_pred = lgb_model.predict(X_val)
lgb_pred_proba = lgb_model.predict_proba(X_val)[:, 1]

# Metrics
lgb_acc = accuracy_score(y_val, lgb_pred)
lgb_auc = roc_auc_score(y_val, lgb_pred_proba)

print("LightGBM Accuracy:", lgb_acc)
print("LightGBM AUC:", lgb_auc)



results = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM'],
    'Accuracy': [xgb_acc, lgb_acc],
    'AUC': [xgb_auc, lgb_auc]
})

results



import lightgbm as lgb
import pandas as pd

# Train LightGBM on the FULL training data
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary'
)

lgb_model.fit(X, y)

# Predict probabilities for the test set
test_probs = lgb_model.predict_proba(test_fe.drop(columns=['id']))[:, 1]

# Build submission dataframe
submission = pd.DataFrame({
    'id': test_fe['id'],
    'diagnosed_diabetes': test_probs
})

# Save to CSV
submission.to_csv('/kaggle/working/submission.csv', index=False)

submission.head()


