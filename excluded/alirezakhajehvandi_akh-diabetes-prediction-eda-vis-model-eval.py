import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


train_data_path = "./data/train.csv"
test_data_path = "./data/test.csv"


df_train = pd.read_csv(train_data_path)
df_test = pd.read_csv(test_data_path)


df_train


df_train.info()


df_train.head(3)


df_train.describe()


df_train["diagnosed_diabetes"].value_counts(normalize=True) * 100


print(df_train['gender'].value_counts())
print("------------------------------------")
print(df_train['ethnicity'].value_counts())
print("------------------------------------")
print(df_train['education_level'].value_counts())
print("------------------------------------")
print(df_train['income_level'].value_counts())
print("------------------------------------")
print(df_train['smoking_status'].value_counts())
print("------------------------------------")
print(df_train['employment_status'].value_counts())


# --------------------------------------------------------------
# 1. IMPORT ALL THE LIBRARIES WE WILL NEED
# --------------------------------------------------------------

import pandas as pd   # pandas helps us work with tables (dataframes)
import numpy as np    # numpy helps with numbers and arrays
import seaborn as sns # seaborn helps us make beautiful plots
import matplotlib.pyplot as plt # matplotlib is used by seaborn internally

# This line makes seaborn's plots look nicer automatically
sns.set(style="whitegrid")

# --------------------------------------------------------------
# 2. LOAD YOUR DATA (you already did it, so this is optional)
# --------------------------------------------------------------

# df_train = pd.read_csv("train.csv")
# df_test  = pd.read_csv("test.csv")

# --------------------------------------------------------------
# 3. OPTIONAL: CREATE A SMALL SAMPLE FOR FAST TESTING
# --------------------------------------------------------------
# We use 'sample' when we want to run things quickly.
# random_state=42 means you always get the same sample each time (good for reproducibility)

df_sample = df_train.sample(5000, random_state=42)  # take 5,000 rows for fast EDA

# --------------------------------------------------------------
# 4. CHOOSE WHICH DATA TO USE
# --------------------------------------------------------------
# If you want to use full dataset â†’ set df = df_train
# If you want to use sample only â†’ set df = df_sample

USE_SAMPLE = False  # ğŸ”¹ change to True if you want faster EDA

df = df_sample if USE_SAMPLE else df_train

# Print what we are using
print("Using SAMPLE Data" if USE_SAMPLE else "Using FULL Data")
print("Number of rows:", len(df))



# Check for missing values
# Check if any column has missing values
df_train.isnull().sum()


plt.subplots(figsize=(2, 3))
# Using sample for faster plotting
sns.countplot(x='diagnosed_diabetes', data=df_train)
plt.title("Distribution of Diabetes (Sample 5000)")
plt.show()


# Only numeric columns
numeric_cols = df_train.select_dtypes(include=np.number).columns.tolist()
df_train[numeric_cols].describe()


numeric_cols


# Select numeric columns except 'id' and target
numeric_cols = ['age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 
                'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 
                'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 
                'heart_rate', 'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides']

# Plot distributions using sample for speed
plt.figure(figsize=(15, 20))  # bigger figure for multiple plots

for i, col in enumerate(numeric_cols):
    plt.subplot(5, 3, i+1)  # create a grid of plots (5 rows x 3 columns)
    sns.histplot(df_sample[col], kde=True, bins=30, color='skyblue')  # histogram + smooth curve
    plt.title(f'Distribution of {col}')
    
plt.tight_layout()
plt.show()


# # Same code but replace df_sample with df_train
# plt.figure(figsize=(15, 20))

# for i, col in enumerate(numeric_cols):
#     plt.subplot(5, 3, i+1)
#     sns.histplot(df_train[col], kde=True, bins=50, color='salmon')  # more bins for full data
#     plt.title(f'Distribution of {col}')
    
# plt.tight_layout()
# plt.show()


# First, select numeric features including the target
numeric_cols_with_target = numeric_cols + ['diagnosed_diabetes']

# Calculate correlation matrix (Pearson correlation by default)
corr_matrix = df_sample[numeric_cols_with_target].corr()  # using sample for speed

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title("Correlation Matrix (Sample 5000)")
plt.show()



# Full dataset for precise correlation
corr_matrix_full = df_train[numeric_cols_with_target].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix_full, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title("Correlation Matrix (Full Dataset)")
plt.show()



# List of categorical columns
categorical_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']

# Plot diabetes distribution for each categorical feature using sample
plt.figure(figsize=(18, 15))  # bigger figure

for i, col in enumerate(categorical_cols):
    plt.subplot(3, 2, i+1)  # 3 rows x 2 columns of plots
    sns.countplot(x=col, hue='diagnosed_diabetes', data=df_sample, palette='Set2')
    plt.title(f'Diabetes by {col}')
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.legend(title='Diabetes', labels=['No', 'Yes'])

plt.tight_layout()
plt.show()



# Copy the dataset to avoid modifying original
df = df_train.copy()  # use df_train.copy() for full dataset

# Categorical columns
categorical_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']

# Use pandas get_dummies for One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)  # drop_first=True avoids dummy variable trap

# Show first 5 rows
df_encoded.head(3)



for i in range(len(df_encoded.columns)):
    print (df_encoded.columns[i])


df_train.tail(3)


# Example: check first 5 rows after encoding income_level
df_sample_encoded = pd.get_dummies(df_sample, columns=['income_level'], drop_first=True)
df_sample_encoded[['income_level_Low', 'income_level_Lower-Middle', 'income_level_Middle', 'income_level_Upper-Middle']].head()



# Target column
y = df_encoded['diagnosed_diabetes']  # 0 or 1

# Features (all columns except id and target)
X = df_encoded.drop(['id', 'diagnosed_diabetes'], axis=1)

# Show shapes
print("Features shape:", X.shape)
print("Target shape:", y.shape)



from sklearn.preprocessing import StandardScaler

# Identify numeric columns in X
numeric_cols_in_X = X.select_dtypes(include=np.number).columns.tolist()

# Initialize scaler
scaler = StandardScaler()

# Fit and transform numeric columns
X[numeric_cols_in_X] = scaler.fit_transform(X[numeric_cols_in_X])

# Show first 5 rows
X.head()



from sklearn.model_selection import train_test_split

# Split data: 80% train, 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify keeps target proportion same
)

# Show shapes
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("y_train:", y_train.shape)
print("y_val:", y_val.shape)






# Logistic Regression model
from sklearn.linear_model import LogisticRegression

# Metrics to evaluate the model
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, RocCurveDisplay



# Initialize the logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)  # max_iter=1000 ensures convergence

# Train the model on training data
model.fit(X_train, y_train)



# Predict on validation set
y_pred = model.predict(X_val)  # predicted labels (0 or 1)
y_prob = model.predict_proba(X_val)[:, 1]  # predicted probabilities for class 1



# Accuracy
acc = accuracy_score(y_val, y_pred)

# F1 score
f1 = f1_score(y_val, y_pred)

# ROC-AUC
roc_auc = roc_auc_score(y_val, y_prob)

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)

print("Accuracy:", round(acc, 4))
print("F1-score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))
print("Confusion Matrix:\n", cm)



# [[TN FP]
#  [FN TP]]



'''Python
[[TN FP]
 [FN TP]]
'''


RocCurveDisplay.from_predictions(y_val, y_prob)
plt.title("ROC Curve - Logistic Regression")
plt.show()



from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,  # number of trees
    max_depth=10,      # maximum depth of each tree
    random_state=42,
    n_jobs=-1          # use all CPU cores
)



rf_model.fit(X_train, y_train)


# Predictions
y_pred_rf = rf_model.predict(X_val)
y_prob_rf = rf_model.predict_proba(X_val)[:, 1]

# Evaluation
acc_rf = accuracy_score(y_val, y_pred_rf)
f1_rf = f1_score(y_val, y_pred_rf)
roc_auc_rf = roc_auc_score(y_val, y_prob_rf)
cm_rf = confusion_matrix(y_val, y_pred_rf)

print("Random Forest Results:")
print("Accuracy:", round(acc_rf, 4))
print("F1-score:", round(f1_rf, 4))
print("ROC-AUC:", round(roc_auc_rf, 4))
print("Confusion Matrix:\n", cm_rf)



importances = rf_model.feature_importances_
features = X_train.columns

# Create a DataFrame for plotting
feat_imp = pd.DataFrame({'feature': features, 'importance': importances}).sort_values(by='importance', ascending=False)

# Plot top 15 features
plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=feat_imp.head(15), palette='viridis', hue=y)
plt.title("Random Forest - Top 15 Feature Importance")
plt.show()



# If not installed: !pip install xgboost
from xgboost import XGBClassifier

# Initialize model
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric='logloss'
)



xgb_model.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_val)
y_prob_xgb = xgb_model.predict_proba(X_val)[:, 1]

# Evaluation
acc_xgb = accuracy_score(y_val, y_pred_xgb)
f1_xgb = f1_score(y_val, y_pred_xgb)
roc_auc_xgb = roc_auc_score(y_val, y_prob_xgb)
cm_xgb = confusion_matrix(y_val, y_pred_xgb)

print("XGBoost Results:")
print("Accuracy:", round(acc_xgb, 4))
print("F1-score:", round(f1_xgb, 4))
print("ROC-AUC:", round(roc_auc_xgb, 4))
print("Confusion Matrix:\n", cm_xgb)



xgb_importances = xgb_model.feature_importances_
feat_imp_xgb = pd.DataFrame({'feature': features, 'importance': xgb_importances}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=feat_imp_xgb.head(15), palette='magma', hue=y)
plt.title("XGBoost - Top 15 Feature Importance")
plt.show()






# If not installed: !pip install lightgbm
from lightgbm import LGBMClassifier

# Initialize model
lgb_model = LGBMClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)


lgb_model.fit(X_train, y_train)

# Predictions
y_pred_lgb = lgb_model.predict(X_val)
y_prob_lgb = lgb_model.predict_proba(X_val)[:, 1]

# Evaluation
acc_lgb = accuracy_score(y_val, y_pred_lgb)
f1_lgb = f1_score(y_val, y_pred_lgb)
roc_auc_lgb = roc_auc_score(y_val, y_prob_lgb)
cm_lgb = confusion_matrix(y_val, y_pred_lgb)

print("LightGBM Results:")
print("Accuracy:", round(acc_lgb, 4))
print("F1-score:", round(f1_lgb, 4))
print("ROC-AUC:", round(roc_auc_lgb, 4))
print("Confusion Matrix:\n", cm_lgb)



lgb_importances = lgb_model.feature_importances_
feat_imp_lgb = pd.DataFrame({'feature': features, 'importance': lgb_importances}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=feat_imp_lgb.head(15), palette='coolwarm', hue=y)
plt.title("LightGBM - Top 15 Feature Importance")
plt.show()



# Create a dictionary of results
results = {
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM'],
    'Accuracy': [acc, acc_rf, acc_xgb, acc_lgb],
    'F1-score': [f1, f1_rf, f1_xgb, f1_lgb],
    'ROC-AUC': [roc_auc, roc_auc_rf, roc_auc_xgb, roc_auc_lgb]
}

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Sort by ROC-AUC (best metric for imbalanced data)
results_df = results_df.sort_values(by='ROC-AUC', ascending=False)

# Show table
results_df



plt.figure(figsize=(10,5))
sns.barplot(x='Model', y='ROC-AUC', data=results_df, palette='Set2')
plt.title("Model Comparison - ROC-AUC")
plt.ylim(0.5, 1)  # better scale
plt.show()



lgb_model.predict_proba(X_train)[:, 1]


lgb_model.predict(X_train)


df_test


# Make a copy of test data
df_test_proc = df_test.copy()

# One-hot encode categorical columns
categorical_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']
df_test_proc = pd.get_dummies(df_test_proc, columns=categorical_cols, drop_first=True)

# Align test data to train data columns
df_test_proc = df_test_proc.reindex(columns=X_train.columns, fill_value=0)  # missing columns filled with 0

# Scale numeric columns
numeric_cols_in_X = X_train.select_dtypes(include=np.number).columns.tolist()
df_test_proc[numeric_cols_in_X] = scaler.transform(df_test_proc[numeric_cols_in_X])



df_test_proc


# Predict probabilities using the best model (example: LightGBM)
y_test_prob = lgb_model.predict_proba(df_test_proc)[:, 1]

# Create submission DataFrame
submission = pd.DataFrame({
    'id': df_test['id'],
    'diagnosed_diabetes': y_test_prob
})

# Show first 5 rows
submission.head()



submission.to_csv('submission.csv', index=False)




