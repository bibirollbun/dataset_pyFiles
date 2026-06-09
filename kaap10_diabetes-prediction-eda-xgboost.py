import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Sklearn for modeling
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# Settings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
pd.set_option('display.max_columns', None)

print("Libraries Imported Successfully!")


# Load datasets
# Note: Check the path on the right side panel if this fails
train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

print(f"Train Shape: {train_df.shape}")
print(f"Test Shape: {test_df.shape}")

# Let's look at the first 5 rows
train_df.head()


# Check for missing values and data types
print("Missing Values in Train:")
print(train_df.isnull().sum().sum())

print("\nMissing Values in Test:")
print(test_df.isnull().sum().sum())

# Basic stats
train_df.describe().T.style.background_gradient(cmap='Blues')


plt.figure(figsize=(8, 6))
sns.countplot(x='diagnosed_diabetes', data=train_df, palette='viridis')
plt.title('Distribution of Target Variable (Diabetes)', fontsize=15)
plt.xlabel('Diagnosed Diabetes (0=No, 1=Yes)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

# Calculate percentage
percentage = train_df['diagnosed_diabetes'].value_counts(normalize=True) * 100
print(f"Percentage of Diabetes Positive: {percentage[1]:.2f}%")
print(f"Percentage of Diabetes Negative: {percentage[0]:.2f}%")


from sklearn.preprocessing import LabelEncoder

# 1. Identify columns that are text (object)
object_cols = train_df.select_dtypes(include=['object']).columns
print(f"Text Columns found: {list(object_cols)}")

# 2. Convert Text to Numbers (Label Encoding)
le = LabelEncoder()

for col in object_cols:
    # Train data fit & transform
    train_df[col] = le.fit_transform(train_df[col])
    # Test data transform only (to keep consistency)
    test_df[col] = le.transform(test_df[col])

print("Encoding Completed! All columns are now numbers.")

# 3. Now run the Correlation Matrix
# Drop ID for correlation
corr_matrix = train_df.drop(['id'], axis=1).corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=15)
plt.show()


def feature_engineering(df):
    # 1. BMI Category (bmi column available hai)
    # Underweight: <18.5, Normal: 18.5-24.9, Overweight: 25-29.9, Obese: >30
    df['BMI_Cat'] = pd.cut(df['bmi'], 
                           bins=[0, 18.5, 24.9, 29.9, 100], 
                           labels=[0, 1, 2, 3])
    
    # 2. Blood Pressure Risk (systolic_bp & diastolic_bp available hain)
    # Agar BP > 130/80 hai toh Risk hai
    df['Hypertension_Risk'] = ((df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 80)).astype(int)
    
    # 3. Cholesterol Ratio (Total / HDL) - Yeh heart health ka bada indicator hai
    # Avoid division by zero
    df['Cholesterol_Ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-5)
    
    # 4. Age Groups (age available hai)
    df['Age_Group'] = pd.cut(df['age'], 
                             bins=[0, 30, 50, 100], 
                             labels=[0, 1, 2])
    
    # 5. Waist to Hip Risk (Central Obesity)
    # Ratio > 0.9 is considered high risk
    df['Central_Obesity'] = (df['waist_to_hip_ratio'] > 0.9).astype(int)
    
    return df

# Apply to both Train and Test
print("Starting Feature Engineering...")
train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

# Convert new categories to numbers just in case
train_df['BMI_Cat'] = train_df['BMI_Cat'].astype(int)
train_df['Age_Group'] = train_df['Age_Group'].astype(int)
test_df['BMI_Cat'] = test_df['BMI_Cat'].astype(int)
test_df['Age_Group'] = test_df['Age_Group'].astype(int)

print("Feature Engineering Completed! ✅ Columns added.")
train_df.head()


# Define features (X) and target (y)
X = train_df.drop(['id', 'diagnosed_diabetes'], axis=1)
y = train_df['diagnosed_diabetes']
X_test = test_df.drop(['id'], axis=1)

# Split data for validation (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Data Split Successful!")


# Re-define X and y (because we added new columns)
X = train_df.drop(['id', 'diagnosed_diabetes'], axis=1)
y = train_df['diagnosed_diabetes']
X_test = test_df.drop(['id'], axis=1)

# Split again
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Updated XGBoost Model
model = XGBClassifier(
    n_estimators=1500,        # Increased from 1000
    learning_rate=0.03,       # Lower learning rate for better accuracy
    max_depth=5,              # Slightly less complex trees to prevent overfitting
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=100,
    eval_metric='auc'
)

# Train
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)


# Predict probabilities (not just classes)
y_pred_prob = model.predict_proba(X_val)[:, 1]

# Calculate ROC AUC Score
score = roc_auc_score(y_val, y_pred_prob)
print(f"Validation ROC AUC Score: {score:.5f}")


feature_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x=feature_imp, y=feature_imp.index, palette='magma')
plt.title('Feature Importance (XGBoost)', fontsize=15)
plt.show()


# Predict on Test Data
test_preds = model.predict_proba(X_test)[:, 1]

# Create submission DataFrame
submission['diagnosed_diabetes'] = test_preds

# Save to CSV
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully! Ready to submit")
submission.head()

