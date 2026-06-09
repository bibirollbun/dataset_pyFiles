import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


train_df.head(5)


test_df.head(5)


train_df.isnull().sum()


train_df.info()


train_df.describe()


numerical_cols_train = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'id' in numerical_cols_train:
    numerical_cols_train.remove('id')
if 'diagnosed_diabetes' in numerical_cols_train:
    numerical_cols_train.remove('diagnosed_diabetes')

plt.figure(figsize=(20, 25))
for i, col in enumerate(numerical_cols_train, 1):
    plt.subplot(5, 4, i) # Adjust subplot grid as needed
    sns.histplot(train_df[col], kde=True)
    plt.title(f'Distribution of {col} in train_df')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


numerical_cols_train = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'id' in numerical_cols_train:
    numerical_cols_train.remove('id')
if 'diagnosed_diabetes' in numerical_cols_train:
    numerical_cols_train.remove('diagnosed_diabetes')

plt.figure(figsize=(20, 25))
for i, col in enumerate(numerical_cols_train, 1):
    plt.subplot(5, 4, i) # Adjust subplot grid as needed
    sns.boxplot(y=train_df[col])
    plt.title(f'Box Plot of {col} in train_df')
    plt.ylabel(col)
plt.tight_layout()
plt.show()


numerical_cols_with_target = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
correlation_matrix = train_df[numerical_cols_with_target].corr(method='pearson')

plt.figure(figsize=(20, 18))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix of Numerical Features in train_df', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


def create_bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi <= 24.9:
        return 'Normal weight'
    elif 25.0 <= bmi <= 29.9:
        return 'Overweight'
    else:
        return 'Obesity'

train_df['bmi_category'] = train_df['bmi'].apply(create_bmi_category)
test_df['bmi_category'] = test_df['bmi'].apply(create_bmi_category)


def create_blood_pressure_category(systolic_bp, diastolic_bp):
    if systolic_bp < 120 and diastolic_bp < 80:
        return 'Normal'
    elif 120 <= systolic_bp <= 129 and diastolic_bp < 80:
        return 'Elevated'
    elif (130 <= systolic_bp <= 139) or (80 <= diastolic_bp <= 89):
        return 'Hypertension Stage 1'
    elif (systolic_bp >= 140) or (diastolic_bp >= 90):
        return 'Hypertension Stage 2'
    elif (systolic_bp >= 180) or (diastolic_bp >= 120):
        return 'Hypertensive Crisis'
    return 'Undefined'

train_df['blood_pressure_category'] = train_df.apply(lambda row: create_blood_pressure_category(row['systolic_bp'], row['diastolic_bp']), axis=1)
test_df['blood_pressure_category'] = test_df.apply(lambda row: create_blood_pressure_category(row['systolic_bp'], row['diastolic_bp']), axis=1)


train_df['cholesterol_ratio'] = train_df['hdl_cholesterol'] / train_df['cholesterol_total']
test_df['cholesterol_ratio'] = test_df['hdl_cholesterol'] / test_df['cholesterol_total']

train_df['cholesterol_ratio'] = train_df['cholesterol_ratio'].replace([float('inf'), -float('inf')], 0)
test_df['cholesterol_ratio'] = test_df['cholesterol_ratio'].replace([float('inf'), -float('inf')], 0)


from sklearn.preprocessing import StandardScaler

def add_pseudo_hba1c(train_df, test_df):
  cols = [
      'bmi',
      'waist_to_hip_ratio',
      'triglycerides',
      'hdl_cholesterol',
      'physical_activity_minutes_per_week',
      'age'
  ]

  scaler = StandardScaler()
  train_z = scaler.fit_transform(train_df[cols])
  test_z = scaler.transform(test_df[cols])

  for df, z in zip([train_df, test_df], [train_z, test_z]):
    df["pseudo_hba1c_v1"] = (
        z[:, 0]  # bmi
        + z[:, 1]  # waist_to_hip_ratio
        + z[:, 2]  # triglycerides
        - z[:, 3]  # hdl_cholesterol
        - z[:, 4]  # physical_activity_minutes_per_week
        + z[:, 5]  # age
        )
  return train_df, test_df

train_df, test_df = add_pseudo_hba1c(train_df, test_df)


print('gender colum', train_df['gender'].unique())
print('ethnicity colum', train_df['ethnicity'].unique())
print('education_level colum', train_df['education_level'].unique())
print('income_level colum', train_df['income_level'].unique())
print('smoking_status colum', train_df['smoking_status'].unique())
print('employment_status colum', train_df['employment_status'].unique())


train_df = pd.get_dummies(train_df, columns=['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status', 'bmi_category', 'blood_pressure_category'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status', 'bmi_category', 'blood_pressure_category'], drop_first=True)


train_df


from sklearn.model_selection import train_test_split

X_full_train = train_df.drop(['id', 'diagnosed_diabetes'], axis=1)
y_full_train = train_df['diagnosed_diabetes']
X_test = test_df.drop('id', axis=1)

common_cols = list(set(X_full_train.columns) | set(X_test.columns))

X_full_train = X_full_train.reindex(columns=common_cols, fill_value=0)
X_test = X_test.reindex(columns=common_cols, fill_value=0)

X_train, X_val, y_train, y_val = train_test_split(X_full_train, y_full_train, test_size=0.2, random_state=42, stratify=y_full_train)


from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,

    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=1000,
    early_stopping_rounds=50,
)

xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)


from sklearn.metrics import roc_auc_score

val_probabilities = xgb_model.predict_proba(X_val)[:, 1]

val_roc_auc = roc_auc_score(y_val, val_probabilities)

print(f"ROC AUC: {val_roc_auc:.4f}")


probabilities = xgb_model.predict_proba(X_test)[:, 1]
submission_df = pd.DataFrame({'id': test_df['id'], 'diagnosed_diabetes': probabilities})
submission_df.to_csv('submission.csv', index=False)


submission_df.head(5)

