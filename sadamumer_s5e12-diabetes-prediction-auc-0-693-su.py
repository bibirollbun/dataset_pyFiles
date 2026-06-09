import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


print(train.shape)
train.head()


print("Target distribution:")
print(train['diagnosed_diabetes'].value_counts())


print("\nPercentage:")
print(train['diagnosed_diabetes'].value_counts(normalize=True) * 100)


plt.figure(figsize=(6, 4))
sns.countplot(x='diagnosed_diabetes', data=train, palette='coolwarm')
plt.title('Distribution of Diagnosed Diabetes')
plt.xlabel('Diagnosed Diabetes (1 = Yes)')
plt.ylabel('Count')
plt.show()


numerical_cols = train.select_dtypes(include=['float64', 'int64']).columns.drop(['id', 'diagnosed_diabetes'])

print("Numerical features summary:")
train[numerical_cols].describe()


key_nums = ['age', 'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'triglycerides', 
            'physical_activity_minutes_per_week', 'sleep_hours_per_day']

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.ravel()

for i, col in enumerate(key_nums):
    sns.histplot(train[col], kde=True, ax=axes[i], color='skyblue')
    axes[i].set_title(f'Dist. of {col}',fontsize=20)
    axes[i].tick_params(axis='x', labelsize=18)

# Hide extra subplots
for j in range(i+1, 9):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 3, figsize=(18, 15))
axes = axes.ravel()

for i, col in enumerate(['age', 'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'triglycerides', 'physical_activity_minutes_per_week']):
    sns.boxplot(x='diagnosed_diabetes', y=col, data=train, ax=axes[i], palette='coolwarm')
    axes[i].set_title(f'{col}',fontsize=20)
    axes[i].tick_params(axis='x', labelsize=18)
    axes[i].tick_params(axis='y',labelsize=12)

plt.tight_layout()
plt.show()


categorical_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 
                    'smoking_status', 'employment_status',
                    'family_history_diabetes', 'hypertension_history', 'cardiovascular_history']

fig, axes = plt.subplots(3, 3, figsize=(18, 20))
axes = axes.ravel()

for i, col in enumerate(categorical_cols):
    order = train[col].value_counts().index
    sns.countplot(x=col, hue='diagnosed_diabetes', data=train, ax=axes[i], palette='coolwarm', order=order)
    axes[i].set_title(f'{col} vs Diabetes',fontsize=20)
    axes[i].tick_params(axis='x', rotation=45, labelsize=18)
    axes[i].tick_params(axis='y',labelsize=12)

plt.tight_layout()
plt.show()


corr = train[numerical_cols.tolist() + ['diagnosed_diabetes']].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Correlation Heatmap (Numerical Features + Target)',fontsize=20)
plt.show()


print("Top correlations with diagnosed_diabetes:")
(corr['diagnosed_diabetes'].abs().sort_values(ascending=False).head(10))


# Identify categorical columns
categorical_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 
                    'smoking_status', 'employment_status']

# Also treat binary history as categorical if needed, but they are already 0/1
binary_cols = ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']

# Combine features (exclude id and target)
features = [col for col in train.columns if col not in ['id', 'diagnosed_diabetes']]

# Label encode categoricals
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    le_dict[col] = le


X = train[features]
y = train['diagnosed_diabetes']

X_test = test[features]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


model = xgb.XGBClassifier(
    n_estimators=800,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42,
    tree_method='hist',
    early_stopping_rounds=50,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)


val_preds = model.predict_proba(X_val)[:, 1]
auc_score = roc_auc_score(y_val, val_preds)
print(f'Validation AUC: {auc_score:.5f}')


import matplotlib.pyplot as plt

xgb.plot_importance(model, max_num_features=20)
plt.title('Top 20 Feature Importance',fontsize=20)
plt.show()


test_preds = model.predict_proba(X_test)[:, 1]

submission['diagnosed_diabetes'] = test_preds
submission.to_csv('submission.csv', index=False)
submission.head()

