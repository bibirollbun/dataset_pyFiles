import pandas as pd

# Load the training data
df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')

# Basic exploration
print(df.shape)
print(df.head())
print(df.info())
print(df.describe())


# Basic info
print("Dataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nTarget distribution:")
print(df['diagnosed_diabetes'].value_counts())
print(df['diagnosed_diabetes'].value_counts(normalize=True))

# Missing values
print("\nMissing values:")
print(df.isnull().sum())



# Look at just numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
print("Numeric features:", numeric_cols)

# Correlations with target (numeric only for now)
corr_with_diabetes = df[numeric_cols].corr()['diagnosed_diabetes'].sort_values(ascending=False)
print(corr_with_diabetes)


# Encode categorical variables
categorical_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Now see all correlations
corr_matrix = df_encoded.corr()
corr_with_diabetes_full = corr_matrix['diagnosed_diabetes'].sort_values(ascending=False)
print(corr_with_diabetes_full)



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Train/test split
X = df_encoded.drop('diagnosed_diabetes', axis=1)
y = df_encoded['diagnosed_diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model 1: Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)  # Binary predictions
lr_proba = lr.predict_proba(X_test)[:, 1]  # Probabilities
print("Logistic Regression:")
print(classification_report(y_test, lr_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, lr_proba):.4f}\n")

# Model 2: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]
print("Random Forest:")
print(classification_report(y_test, rf_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, rf_proba):.4f}\n")

# Model 3: XGBoost
xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_proba = xgb.predict_proba(X_test)[:, 1]
print("XGBoost:")
print(classification_report(y_test, xgb_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, xgb_proba):.4f}")



from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Calculate class weights
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = {classes[i]: weights[i] for i in range(len(classes))}

# Train XGBoost with balanced weights
xgb_balanced = XGBClassifier(
    n_estimators=100,
    scale_pos_weight=weights[0]/weights[1],  # XGBoost's way to handle imbalance
    random_state=42,
    eval_metric='logloss'
)
xgb_balanced.fit(X_train, y_train)
xgb_pred_balanced = xgb_balanced.predict(X_test)

print("XGBoost with Class Weights:")
print(classification_report(y_test, xgb_pred_balanced))



from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'max_depth': [5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200],
    'min_child_weight': [1, 3, 5]
}

# Grid search (warning: this will take 10-15 minutes)
grid_search = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss'),
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best ROC-AUC:", grid_search.best_score_)

# Train with best parameters
best_xgb = grid_search.best_estimator_
best_pred = best_xgb.predict(X_test)
print("\nTuned XGBoost Results:")
print(classification_report(y_test, best_pred))



import matplotlib.pyplot as plt

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print(feature_importance.head(10))

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
plt.xlabel('Importance')
plt.title('Top 10 Features for Diabetes Prediction')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()



# Load test data
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
test_ids = test['id']

# Encode test data (same as training)
test_encoded = pd.get_dummies(test.drop('id', axis=1), columns=['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status'], drop_first=True)

# ALIGN columns with training data
# Get missing columns
missing_cols = set(X_train.columns) - set(test_encoded.columns)
for col in missing_cols:
    test_encoded[col] = 0

# Remove extra columns
extra_cols = set(test_encoded.columns) - set(X_train.columns)
test_encoded = test_encoded.drop(columns=extra_cols)

# Reorder columns to match training
test_encoded = test_encoded[X_train.columns]

# Now predict
predictions = xgb.predict_proba(test_encoded)[:, 1]

# Create submission
submission = pd.DataFrame({'id': test_ids, 'diagnosed_diabetes': predictions})
submission.to_csv('submission.csv', index=False)





