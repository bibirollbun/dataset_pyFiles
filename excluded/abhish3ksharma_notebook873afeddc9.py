# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# The following code snippet is for demonstrating the available files; it's commented out
# to prevent redundant output in the final single-cell script.
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Suppress all warnings for a clean output
warnings.filterwarnings("ignore")

# ==========================================
# 1. Data Loading and Initial Preprocessing
# ==========================================
data = pd.read_csv('/kaggle/input/predicting-euphoria-in-the-streets/train.csv')
df = pd.DataFrame(data)

# Separate target variable 'Y'
y = df['Y'].astype(int)   # Convert bool to int (0/1)

# Drop 'id' and 'Y' from features
exclude = ['id','Y']
df1 = df.drop(columns=exclude)

# Impute NaN values using the mean of each column
# This is a two-step process to handle inf/-inf values which can interfere with mean calculation
df1_filled = df1.fillna(df1.mean(numeric_only=True))

# Replace infinite values with NaN, then impute these new NaNs with the mean again
df1_filled = df1_filled.replace([np.inf, -np.inf], np.nan)
df1_filled = df1_filled.fillna(df1_filled.mean(numeric_only=True))

X = df1_filled

# ==========================================
# 2. Binary Classification Model Comparison (Re-run for context)
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(
        eval_metric='logloss', use_label_encoder=False, random_state=42),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
}

results = []
for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predict probabilities
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Compute ROC-AUC
    auc = roc_auc_score(y_test, y_pred_prob)
    
    results.append({
        'Model': name,
        'ROC-AUC': auc
    })

results_df = pd.DataFrame(results).sort_values(by='ROC-AUC', ascending=False)
print("\nâœ… Model Comparison (sorted by ROC-AUC):")
print(results_df.to_string(index=False))


# ==========================================
# 3. Final Random Forest Binary Classifier Training
# (Using the parameters from the initial comparison step, n_estimators=200)
# ==========================================

# Using the full feature set (X) for training the initial RF model for feature importance
rf_model = RandomForestClassifier(
    n_estimators=200,       # number of trees
    max_depth=6,            # controls tree depth (a reasonable starting point, original was default/none)
    random_state=42,
    n_jobs=-1               # Use all processors
)

rf_model.fit(X_train, y_train)

# -------------------------
# 4. Predict and evaluate (Full Features)
# -------------------------
y_pred = rf_model.predict(X_test)
y_pred_prob = rf_model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_prob)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nâœ… Random Forest Performance (Full Features):")
print(f"ROC-AUC: {auc:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------
# 5. Feature Importance and Selection
# -------------------------
# Get feature importances
feature_importances = rf_model.feature_importances_
features = X.columns
feat_df = pd.DataFrame({'feature': features, 'importance': feature_importances})
feat_df = feat_df.sort_values(by='importance', ascending=False).reset_index(drop=True)

print("\nTop 10 Feature Importances from Random Forest Model (Full Features):")
print(feat_df.head(10).to_string(index=False))

# Re-using the features kept in the original notebook for consistency
final_features = ['x_1', 'x_2', 'x_4', 'x_6', 'x_8', 'x_10', 'x_11', 'x_12', 'x_13', 'x_15', 'x_16', 'x_17', 'x_18', 'x_19', 'x_20', 'x_21']
X_reduced = X[final_features]

print(f"\nFeatures kept for reduced model ({len(final_features)}): {final_features}")

# ==========================================
# 6. Final Random Forest Model Training (Reduced Features)
# ==========================================

# Split data with reduced features
X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42, stratify=y
)

rf_model_reduced = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)

rf_model_reduced.fit(X_train_reduced, y_train_reduced)

# -------------------------
# 7. Predict and evaluate (Reduced Features)
# -------------------------
y_pred_reduced = rf_model_reduced.predict(X_test_reduced)
y_pred_prob_reduced = rf_model_reduced.predict_proba(X_test_reduced)[:, 1]

auc_reduced = roc_auc_score(y_test_reduced, y_pred_prob_reduced)
acc_reduced = accuracy_score(y_test_reduced, y_pred_reduced)
f1_reduced = f1_score(y_test_reduced, y_pred_reduced)

print(f"\nâœ… Random Forest Performance (Reduced Features):")
print(f"ROC-AUC: {auc_reduced:.4f}")
print(f"Accuracy: {acc_reduced:.4f}")
print(f"F1-score: {f1_reduced:.4f}")
print("\nClassification Report (Reduced Features):\n", classification_report(y_test_reduced, y_pred_reduced))
print("\nConfusion Matrix (Reduced Features):\n", confusion_matrix(y_test_reduced, y_pred_reduced))

# -------------------------
# 8. Plot ROC curve
# -------------------------
fpr, tpr, _ = roc_curve(y_test_reduced, y_pred_prob_reduced)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'RF Reduced (AUC = {auc_reduced:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest Reduced Features')
plt.legend()
plt.grid(True)
plt.show()


# ==========================================
# 9. Prediction on Test Data and Submission
# ==========================================

# Load test data
test_path = "/kaggle/input/predicting-euphoria-in-the-streets/test.csv"
test_df = pd.read_csv(test_path)

# Preprocess test data (matching the training data preprocessing)
test_df_filled = test_df.replace([np.inf, -np.inf], np.nan).fillna(df1_filled.mean(numeric_only=True))

# Select only the reduced set of features
X_test_final = test_df_filled[final_features]
ids = test_df_filled['id']

# Predict probabilities and final binary class using the reduced model
y_pred_prob_final = rf_model_reduced.predict_proba(X_test_final)[:, 1]
y_pred_final = (y_pred_prob_final >= 0.5).astype(int)  # binary prediction (threshold = 0.5)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': ids,
    'Y': y_pred_final
})

# Save to CSV
output_path = "/kaggle/working/submission.csv"
submission.to_csv(output_path, index=False)

print(f"\nâœ… Prediction file saved to: {output_path}")
print("\nSubmission Head:")
print(submission.head())

# ==========================================
# 10. Feature Importance Plot (Final Reduced Model)
# ==========================================

# Get feature importances from the final reduced model
importances_reduced = rf_model_reduced.feature_importances_
features_reduced = X_reduced.columns

feat_imp_reduced = pd.DataFrame({'Feature': features_reduced, 'Importance': importances_reduced})
feat_imp_reduced = feat_imp_reduced.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10,6))
plt.barh(feat_imp_reduced['Feature'], feat_imp_reduced['Importance'], color='coral')
plt.gca().invert_yaxis()
plt.title('Random Forest Feature Importances (Reduced Model)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# ===============================================================
# Kaggle Notebook: Predicting Euphoria in the Streets (Random Forest)
# ===============================================================

import numpy as np
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ==========================================
# 1. Data Loading and Initial Preprocessing
# ==========================================
data = pd.read_csv('/kaggle/input/predicting-euphoria-in-the-streets/train.csv')
df = pd.DataFrame(data)

# Separate target variable 'Y'
y = df['Y'].astype(int)

# Drop 'id' and 'Y' from features
exclude = ['id', 'Y']
df1 = df.drop(columns=exclude)

# Handle NaN and infinite values
df1_filled = df1.fillna(df1.mean(numeric_only=True))
df1_filled = df1_filled.replace([np.inf, -np.inf], np.nan)
df1_filled = df1_filled.fillna(df1_filled.mean(numeric_only=True))

X = df1_filled

# ==========================================
# 2. Train-Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 3. Initial Random Forest Model & Feature Importance
# ==========================================
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Feature importances
feature_importances = rf_model.feature_importances_
features = X.columns
feat_df = pd.DataFrame({'feature': features, 'importance': feature_importances})
feat_df = feat_df.sort_values(by='importance', ascending=False).reset_index(drop=True)

print("Top 10 Feature Importances from Initial Random Forest Model:")
print(feat_df.head(10).to_string(index=False))

# Use same final feature set as the original XGBoost version for consistency
final_features = ['x_1', 'x_2', 'x_4', 'x_6', 'x_8', 'x_10', 'x_11', 'x_12', 'x_13',
                  'x_15', 'x_16', 'x_17', 'x_18', 'x_19', 'x_20', 'x_21']

X_reduced = X[final_features]
print(f"\nFeatures kept after selection ({len(final_features)}): {final_features}")

# ==========================================
# 4. Final Random Forest Model (Reduced Features)
# ==========================================
X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42, stratify=y
)

rf_model_reduced = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_model_reduced.fit(X_train_reduced, y_train_reduced)

# Predictions
y_pred_prob = rf_model_reduced.predict_proba(X_test_reduced)[:, 1]
y_pred = rf_model_reduced.predict(X_test_reduced)

auc = roc_auc_score(y_test_reduced, y_pred_prob)
acc = accuracy_score(y_test_reduced, y_pred)
f1 = f1_score(y_test_reduced, y_pred)

print(f"\nâœ… Random Forest Performance (Reduced Features):")
print(f"ROC-AUC: {auc:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test_reduced, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_reduced, y_pred))

# ==========================================
# 5. Prediction on Test Data and Submission
# ==========================================
test_path = "/kaggle/input/predicting-euphoria-in-the-streets/test.csv"
test_df = pd.read_csv(test_path)

# Preprocess test data
test_df_filled = test_df.replace([np.inf, -np.inf], np.nan).fillna(df1_filled.mean(numeric_only=True))

# Use reduced feature set
X_test_final = test_df_filled[final_features]
ids = test_df_filled['id']

# Predict
y_pred_prob_final = rf_model_reduced.predict_proba(X_test_final)[:, 1]
y_pred_final = (y_pred_prob_final >= 0.5).astype(int)

# Submission
submission = pd.DataFrame({
    'id': ids,
    'Y': y_pred_final
})

output_path = "/kaggle/working/classification1_rf.csv"
submission.to_csv(output_path, index=False)

print(f"\nâœ… Prediction file saved to: {output_path}")
print("\nSubmission Head:")
print(submission.head())

# ==========================================
# 6. Feature Importance Plot (Final Model)
# ==========================================
importances_reduced = rf_model_reduced.feature_importances_
features_reduced = X_reduced.columns

feat_imp_reduced = pd.DataFrame({'Feature': features_reduced, 'Importance': importances_reduced})
feat_imp_reduced = feat_imp_reduced.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
plt.barh(feat_imp_reduced['Feature'], feat_imp_reduced['Importance'], color='lightgreen')
plt.gca().invert_yaxis()
plt.title('Random Forest Feature Importances (Reduced Model)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# The following code snippet is for demonstrating the available files; it's commented out
# to prevent redundant output in the final single-cell script.
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.ensemble import RandomForestClassifier

# Suppress all warnings for a clean output
warnings.filterwarnings("ignore")

# ==========================================
# 1. Data Loading and Initial Preprocessing
# ==========================================
data = pd.read_csv('/kaggle/input/predicting-euphoria-in-the-streets/train.csv')
df = pd.DataFrame(data)

# Separate target variable 'Y'
y = df['Y'].astype(int)   # Convert bool to int (0/1)

# Drop 'id' and 'Y' from features
exclude = ['id','Y']
df1 = df.drop(columns=exclude)

# Impute NaN values using the mean of each column
# This is a two-step process to handle inf/-inf values which can interfere with mean calculation
df1_filled = df1.fillna(df1.mean(numeric_only=True))

# Replace infinite values with NaN, then impute these new NaNs with the mean again
df1_filled = df1_filled.replace([np.inf, -np.inf], np.nan)
# Recompute mean based on finite values for final imputation
df1_filled = df1_filled.fillna(df1_filled.mean(numeric_only=True)) 

# Use the full, cleaned feature set
X = df1_filled
# Define the full feature set for later use on the test data
full_features = X.columns.tolist()

# ==========================================
# 2. Data Splitting
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 3. Final Optimized Random Forest Model Training (Full Features)
#    - Increased n_estimators (more trees)
#    - Removed max_depth constraint (allowing deeper trees for higher accuracy)
# ==========================================

# Model definition with parameters aimed at maximizing performance
rf_model_optimized = RandomForestClassifier(
    n_estimators=500,       # Increased number of trees
    max_depth=None,         # Allow trees to grow deep (default for max accuracy)
    min_samples_split=2,    # Default value
    min_samples_leaf=1,     # Default value
    random_state=42,
    class_weight='balanced', # Helps with class imbalance
    n_jobs=-1               # Use all processors
)

print("Training optimized Random Forest model on the full feature set...")
rf_model_optimized.fit(X_train, y_train)
print("Training complete.")

# -------------------------
# 4. Predict and evaluate
# -------------------------
y_pred = rf_model_optimized.predict(X_test)
y_pred_prob = rf_model_optimized.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_prob)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nâœ… Optimized Random Forest Performance (Full Features):")
print(f"ROC-AUC: {auc:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------
# 5. Plot ROC curve
# -------------------------
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'RF Optimized (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Optimized Random Forest')
plt.legend()
plt.grid(True)
plt.show()


# ==========================================
# 6. Prediction on Test Data and Submission
# ==========================================

# Load test data
test_path = "/kaggle/input/predicting-euphoria-in-the-streets/test.csv"
test_df = pd.read_csv(test_path)

# Preprocess test data (matching the training data preprocessing)
test_df_filled = test_df.replace([np.inf, -np.inf], np.nan).fillna(df1_filled.mean(numeric_only=True))

# Select the same full set of features used for final training
X_test_final = test_df_filled[full_features]
ids = test_df_filled['id']

# Predict probabilities and final binary class using the optimized model
y_pred_prob_final = rf_model_optimized.predict_proba(X_test_final)[:, 1]
y_pred_final = (y_pred_prob_final >= 0.5).astype(int)  # binary prediction (threshold = 0.5)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': ids,
    'Y': y_pred_final
})

# Save to CSV
output_path = "/kaggle/working/submission.csv"
submission.to_csv(output_path, index=False)

print(f"\nâœ… Prediction file saved to: {output_path}")
print("\nSubmission Head:")
print(submission.head())

# ==========================================
# 7. Feature Importance Plot (Optimized Model)
# ==========================================

# Get feature importances from the final model
importances = rf_model_optimized.feature_importances_
features = X.columns

feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_imp = feat_imp.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10,6))
plt.barh(feat_imp['Feature'], feat_imp['Importance'], color='green')
plt.gca().invert_yaxis()
plt.title('Random Forest Feature Importances (Optimized Model)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Top contributing features
print("\nTop 10 Feature Importances:")
print(feat_imp.head(10).to_string(index=False))


# ===============================================================
# Kaggle Notebook: Predicting Euphoria in the Streets (Optimized Random Forest)
# ===============================================================

import numpy as np
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ==========================================
# 1. Data Loading and Initial Preprocessing
# ==========================================
data = pd.read_csv('/kaggle/input/predicting-euphoria-in-the-streets/train.csv')
df = pd.DataFrame(data)

# Separate target variable 'Y'
y = df['Y'].astype(int)

# Drop 'id' and 'Y' from features
exclude = ['id', 'Y']
df1 = df.drop(columns=exclude)

# Handle NaN and infinite values
df1 = df1.replace([np.inf, -np.inf], np.nan)
df1 = df1.fillna(df1.mean(numeric_only=True))

# Optional: Cap extreme outliers using percentile clipping
for col in df1.columns:
    lower, upper = df1[col].quantile(0.01), df1[col].quantile(0.99)
    df1[col] = np.clip(df1[col], lower, upper)

X = df1.copy()

# ==========================================
# 2. Train-Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 3. Hyperparameter Optimization (RandomizedSearchCV)
# ==========================================
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

param_dist = {
    'n_estimators': [300, 400, 500, 600],
    'max_depth': [8, 10, 12, 14, 16, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.8]
}

cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

rf_search = RandomizedSearchCV(
    rf_base,
    param_distributions=param_dist,
    n_iter=25,
    scoring='roc_auc',
    cv=cv_strategy,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

print("ğŸ”� Running RandomizedSearchCV for Random Forest...")
rf_search.fit(X_train, y_train)

print(f"\nBest AUC from CV: {rf_search.best_score_:.4f}")
print(f"Best Parameters:\n{rf_search.best_params_}")

# ==========================================
# 4. Train Final Model with Best Parameters
# ==========================================
best_params = rf_search.best_params_
rf_model = RandomForestClassifier(
    **best_params,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# ==========================================
# 5. Feature Importance and Selection
# ==========================================
feature_importances = rf_model.feature_importances_
features = X.columns
feat_df = pd.DataFrame({'feature': features, 'importance': feature_importances})
feat_df = feat_df.sort_values(by='importance', ascending=False).reset_index(drop=True)

print("\nTop 15 Feature Importances:")
print(feat_df.head(15).to_string(index=False))

# Keep only top 20 features for final model
top_features = feat_df.head(20)['feature'].tolist()
X_reduced = X[top_features]
print(f"\nFeatures kept after selection ({len(top_features)}): {top_features}")

# ==========================================
# 6. Retrain on Reduced Features
# ==========================================
X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42, stratify=y
)

rf_model_reduced = RandomForestClassifier(
    **best_params,
    random_state=42,
    n_jobs=-1
)
rf_model_reduced.fit(X_train_reduced, y_train_reduced)

# Predictions
y_pred_prob = rf_model_reduced.predict_proba(X_test_reduced)[:, 1]
y_pred = rf_model_reduced.predict(X_test_reduced)

auc = roc_auc_score(y_test_reduced, y_pred_prob)
acc = accuracy_score(y_test_reduced, y_pred)
f1 = f1_score(y_test_reduced, y_pred)

print(f"\nâœ… Optimized Random Forest Performance (Reduced Features):")
print(f"ROC-AUC: {auc:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test_reduced, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_reduced, y_pred))

# ==========================================
# 7. Prediction on Test Data and Submission
# ==========================================
test_path = "/kaggle/input/predicting-euphoria-in-the-streets/test.csv"
test_df = pd.read_csv(test_path)

# Preprocess test data same way as train
test_df = test_df.replace([np.inf, -np.inf], np.nan)
test_df = test_df.fillna(df1.mean(numeric_only=True))
for col in top_features:
    lower, upper = df1[col].quantile(0.01), df1[col].quantile(0.99)
    test_df[col] = np.clip(test_df[col], lower, upper)

X_test_final = test_df[top_features]
ids = test_df['id']

# Predict
y_pred_prob_final = rf_model_reduced.predict_proba(X_test_final)[:, 1]
y_pred_final = (y_pred_prob_final >= 0.5).astype(int)

# Submission
submission = pd.DataFrame({
    'id': ids,
    'Y': y_pred_final
})

output_path = "/kaggle/working/submission.csv"
submission.to_csv(output_path, index=False)

print(f"\nâœ… Prediction file saved to: {output_path}")
print("\nSubmission Head:")
print(submission.head())

# ==========================================
# 8. Feature Importance Plot (Final Model)
# ==========================================
importances_reduced = rf_model_reduced.feature_importances_
features_reduced = X_reduced.columns

feat_imp_reduced = pd.DataFrame({'Feature': features_reduced, 'Importance': importances_reduced})
feat_imp_reduced = feat_imp_reduced.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
plt.barh(feat_imp_reduced['Feature'], feat_imp_reduced['Importance'], color='gold')
plt.gca().invert_yaxis()
plt.title('Optimized Random Forest Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


