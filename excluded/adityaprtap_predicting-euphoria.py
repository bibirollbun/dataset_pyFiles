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


import pandas as pd

data = pd.read_csv('/kaggle/input/predicting-euphoria-in-the-streets/train.csv')
df = pd.DataFrame(data)
print(df.info())
print(df.head())


y=df['Y']
exclude = ['id','Y']
df1 = df.drop(columns=exclude)
print(df1.info())


df1_filled = df1.fillna(df1.mean(numeric_only=True))
print(df1_filled.info())


import numpy as np

np.isinf(df1_filled).sum()
df1_filled = df1_filled.replace([np.inf, -np.inf], np.nan)
df1_filled = df1_filled.fillna(df1_filled.mean(numeric_only=True))
print(np.isinf(df1_filled.values).any())   # should be False
print(df1_filled.isna().sum().sum())       # should be 0



# ==========================================
# Binary Classification Model Comparison
# ==========================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# 1. Split data
# -------------------------
X = df1_filled
y = df['Y'].astype(int)   # Convert bool to int (0/1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# 2. Scale features
# -------------------------
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

# -------------------------
# 3. Define models
# -------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(
        eval_metric='logloss', use_label_encoder=False, random_state=42),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
}

# -------------------------
# 4. Train, predict, and evaluate
# -------------------------
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

# -------------------------
# 5. Display results
# -------------------------
results_df = pd.DataFrame(results).sort_values(by='ROC-AUC', ascending=False)
print("\nâœ… Model Comparison (sorted by ROC-AUC):")
print(results_df.to_string(index=False))



# ==========================================
# Final XGBoost Binary Classifier Training
# ==========================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# 1. Split data
# -------------------------
X = df1_filled
y = df['Y'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# 2. Train final XGBoost model
# -------------------------
xgb_model = XGBClassifier(
    n_estimators=300,       # number of boosting rounds
    learning_rate=0.05,     # smaller = slower but more stable learning
    max_depth=6,            # controls model complexity
    subsample=0.8,          # row sampling
    colsample_bytree=0.8,   # feature sampling
    eval_metric='logloss',
    random_state=42,
    use_label_encoder=False
)

xgb_model.fit(X_train, y_train)

# -------------------------
# 3. Predict and evaluate
# -------------------------
y_pred = xgb_model.predict(X_test)
y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_prob)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"âœ… XGBoost Performance:")
print(f"ROC-AUC: {auc:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------
# 4. Plot ROC curve
# -------------------------
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost')
plt.legend()
plt.grid(True)
plt.show()



import pandas as pd
import numpy as np

# -------------------------
# 1. Load test data
# -------------------------
test_path = "/kaggle/input/predicting-euphoria-in-the-streets/test.csv"
test_df = pd.read_csv(test_path)

# -------------------------
# 2. Preprocess test data
# -------------------------
# Match the same cleaning you applied to train data
test_df_filled = test_df.replace([np.inf, -np.inf], np.nan).fillna(df1_filled.mean(numeric_only=True))

# Drop ID column for prediction, keep it separately
X_test_final = test_df_filled.drop('id', axis=1)
ids = test_df_filled['id']

# -------------------------
# 3. Predict using trained XGBoost model
# -------------------------
y_pred_prob = xgb_model.predict_proba(X_test_final)[:, 1]
y_pred = (y_pred_prob >= 0.5).astype(int)  # binary prediction (threshold = 0.5)

# -------------------------
# 4. Create submission DataFrame
# -------------------------
submission = pd.DataFrame({
    'id': ids,
    'Y': y_pred
})

# -------------------------
# 5. Save to CSV in Kaggle working directory
# -------------------------
output_path = "/kaggle/working/classification.csv"
submission.to_csv(output_path, index=False)

print(f"âœ… Prediction file saved to: {output_path}")
print(submission.head())



import matplotlib.pyplot as plt

# Number of features
n_cols = X.shape[1]
cols = X.columns

# Set up the grid (e.g., 7 rows Ã— 3 columns)
n_rows = (n_cols + 2) // 3
plt.figure(figsize=(15, 5 * n_rows))

for i, col in enumerate(cols, 1):
    plt.subplot(n_rows, 3, i)
    plt.scatter(X[y == 0][col], y[y == 0], color='green', label='Y = 0', alpha=0.6)
    plt.scatter(X[y == 1][col], y[y == 1], color='red', label='Y = 1', alpha=0.6)
    plt.title(f"{col} vs Y")
    plt.xlabel(col)
    plt.ylabel("Y")
    plt.legend()

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import pandas as pd

# Get feature importances
importances = xgb_model.feature_importances_  # if your model is xgboost.XGBClassifier
features = X.columns

feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_imp = feat_imp.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10,6))
plt.barh(feat_imp['Feature'], feat_imp['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.title('XGBoost Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Top contributing features
print(feat_imp.head(10))



# ==========================================
# Retrain XGBoost after dropping low-importance features
# ==========================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# 1. Identify features to drop
# -------------------------
# Feature importances from your previous model
feature_importances = xgb_model.feature_importances_
features = X.columns

# Create a DataFrame of features and their importances
feat_df = pd.DataFrame({'feature': features, 'importance': feature_importances})

# Get importance of x_12
threshold = feat_df.loc[feat_df['feature'] == 'x_9', 'importance'].values[0]

# Drop x_12 and all features with importance less than or equal to it
features_to_keep = feat_df.loc[feat_df['importance'] > threshold, 'feature'].tolist()
X_reduced = X[features_to_keep]

print(f"Features kept ({len(features_to_keep)}): {features_to_keep}")

# -------------------------
# 2. Split data
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# 3. Train XGBoost model
# -------------------------
xgb_model_reduced = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42,
    use_label_encoder=False
)

xgb_model_reduced.fit(X_train, y_train)

# -------------------------
# 4. Predict and evaluate
# -------------------------
y_pred = xgb_model_reduced.predict(X_test)
y_pred_prob = xgb_model_reduced.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_prob)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"âœ… XGBoost Performance after feature reduction:")
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
plt.plot(fpr, tpr, label=f'XGBoost Reduced (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost Reduced Features')
plt.legend()
plt.grid(True)
plt.show()



import pandas as pd
import numpy as np

# -------------------------
# 1. Load test data
# -------------------------
test_path = "/kaggle/input/predicting-euphoria-in-the-streets/test.csv"
test_df = pd.read_csv(test_path)

# -------------------------
# 2. Preprocess test data
# -------------------------
# Match the same cleaning you applied to train data
test_df_filled = test_df.replace([np.inf, -np.inf], np.nan).fillna(df1_filled.mean(numeric_only=True))

# Drop ID column for prediction, keep it separately
X_test_final = test_df_filled[['x_1', 'x_2', 'x_4', 'x_6', 'x_8', 
                                'x_10', 'x_11', 'x_12', 'x_13', 'x_15', 
                                'x_16', 'x_17', 'x_18', 'x_19', 'x_20', 'x_21']]
ids = test_df_filled['id']

# -------------------------
# 3. Predict using trained XGBoost model
# -------------------------
y_pred_prob = xgb_model_reduced.predict_proba(X_test_final)[:, 1]
y_pred = (y_pred_prob >= 0.5).astype(int)  # binary prediction (threshold = 0.5)

# -------------------------
# 4. Create submission DataFrame
# -------------------------
submission = pd.DataFrame({
    'id': ids,
    'Y': y_pred
})

# -------------------------
# 5. Save to CSV in Kaggle working directory
# -------------------------
output_path = "/kaggle/working/classification1.csv"
submission.to_csv(output_path, index=False)

print(f"âœ… Prediction file saved to: {output_path}")
print(submission.head())



# ==========================================
# Final Random Forest Binary Classifier Training
# ==========================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# 1. Split data
# -------------------------
X = df1_filled
y = df['Y'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# 2. Train final Random Forest model
# -------------------------
model_random_forest = RandomForestClassifier(
    n_estimators=300,        # number of trees
    max_depth=10,            # limit depth to prevent overfitting
    min_samples_split=2,     # minimum samples required to split a node
    min_samples_leaf=1,      # minimum samples at a leaf node
    max_features='sqrt',     # number of features considered for split
    bootstrap=True,          # use bootstrap samples
    random_state=42,
    n_jobs=-1                # use all available cores
)

model_random_forest.fit(X_train, y_train)

# -------------------------
# 3. Predict and evaluate
# -------------------------
y_pred = model_random_forest.predict(X_test)
y_pred_prob = model_random_forest.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_prob)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"âœ… Random Forest Performance:")
print(f"ROC-AUC: {auc:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------
# 4. Plot ROC curve
# -------------------------
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.grid(True)
plt.show()



import pandas as pd
import numpy as np

# -------------------------
# 1. Load test data
# -------------------------
test_path = "/kaggle/input/predicting-euphoria-in-the-streets/test.csv"
test_df = pd.read_csv(test_path)

# -------------------------
# 2. Preprocess test data
# -------------------------
# Match the same cleaning you applied to train data
test_df_filled = test_df.replace([np.inf, -np.inf], np.nan).fillna(df1_filled.mean(numeric_only=True))

# Drop ID column for prediction, keep it separately
X_test_final = test_df_filled.drop('id', axis=1)
ids = test_df_filled['id']

# -------------------------
# 3. Predict using trained Random Forest model
# -------------------------
y_pred_prob = model_random_forest.predict_proba(X_test_final)[:, 1]
y_pred = (y_pred_prob >= 0.5).astype(int)  # binary prediction (threshold = 0.5)

# -------------------------
# 4. Create submission DataFrame
# -------------------------
submission = pd.DataFrame({
    'id': ids,
    'Y': y_pred
})

# -------------------------
# 5. Save to CSV in Kaggle working directory
# -------------------------
output_path = "/kaggle/working/classification_random_forest.csv"
submission.to_csv(output_path, index=False)

print(f"âœ… Prediction file saved to: {output_path}")
print(submission.head())



import pandas as pd

# Get feature importances
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model_random_forest.feature_importances_
})

# Sort by importance (descending)
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Display top 15
print(feature_importances.head(15))



import matplotlib.pyplot as plt

top_n = 20  # number of features to show
plt.figure(figsize=(8, 6))
plt.barh(
    feature_importances['Feature'].head(top_n)[::-1],
    feature_importances['Importance'].head(top_n)[::-1]
)
plt.xlabel("Feature Importance")
plt.title(f"Top {top_n} Most Important Features - Random Forest")
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.show()



import pandas as pd

# Assume df1_filled is your numeric DataFrame
outlier_summary = pd.DataFrame(columns=['Q1', 'Q3', 'IQR', 'Lower Bound', 'Upper Bound', 'Outlier Count'])

for col in df1_filled.select_dtypes(include=[np.number]).columns:
    Q1 = df1_filled[col].quantile(0.25)
    Q3 = df1_filled[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_count = ((df1_filled[col] < lower) | (df1_filled[col] > upper)).sum()
    outlier_summary.loc[col] = [Q1, Q3, IQR, lower, upper, outlier_count]

print("ðŸ“Š Outlier summary (IQR method):")
display(outlier_summary.sort_values('Outlier Count', ascending=False))



import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

# --- Step 1. Identify numeric columns ---
numeric_cols = df1_filled.select_dtypes(include=[np.number]).columns

# --- Step 2. Identify which have outliers ---
outlier_summary = pd.DataFrame(columns=['Q1', 'Q3', 'IQR', 'Lower Bound', 'Upper Bound', 'Outlier Count'])
for col in numeric_cols:
    Q1 = df1_filled[col].quantile(0.25)
    Q3 = df1_filled[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_count = ((df1_filled[col] < lower) | (df1_filled[col] > upper)).sum()
    outlier_summary.loc[col] = [Q1, Q3, IQR, lower, upper, outlier_count]

# --- Step 3. Select columns that actually have outliers ---
cols_with_outliers = outlier_summary[outlier_summary['Outlier Count'] > 0].index.tolist()
print("Columns with outliers:", cols_with_outliers)

# --- Step 4. Handle zero or near-zero variance columns (skip scaling for them) ---
low_var_cols = outlier_summary[outlier_summary['IQR'] == 0].index.tolist()
cols_to_scale = [col for col in cols_with_outliers if col not in low_var_cols]

print("Columns skipped (zero variance):", low_var_cols)
print("Columns scaled:", cols_to_scale)

# --- Step 5. Apply RobustScaler to outlier columns ---
scaler = RobustScaler()
df1_scaled = df1_filled.copy()

df1_scaled[cols_to_scale] = scaler.fit_transform(df1_scaled[cols_to_scale])

print("\nâœ… Scaling complete! Hereâ€™s a quick summary:")
print(df1_scaled[cols_to_scale].describe().T)

# Optional: Verify fewer outliers remain
new_outlier_summary = pd.DataFrame(columns=['Outlier Count'])
for col in cols_to_scale:
    Q1 = df1_scaled[col].quantile(0.25)
    Q3 = df1_scaled[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_count = ((df1_scaled[col] < lower) | (df1_scaled[col] > upper)).sum()
    new_outlier_summary.loc[col] = [outlier_count]

print("\nðŸ“‰ Outlier counts after scaling:")
display(new_outlier_summary.sort_values('Outlier Count', ascending=False))



import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

# --- Your known outlier columns ---
cols_with_outliers = [
    'x_14','x_10','x_12','x_11','x_9','x_13','x_19','x_20','x_18',
    'x_17','x_21','x_3','x_15','x_6','x_2','x_7','x_8','x_1','x_4'
]

# --- Step 1. Compute IQR bounds for those columns ---
outlier_summary = pd.DataFrame(columns=['Q1', 'Q3', 'IQR', 'Lower Bound', 'Upper Bound'])
for col in cols_with_outliers:
    Q1 = df1_filled[col].quantile(0.25)
    Q3 = df1_filled[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_summary.loc[col] = [Q1, Q3, IQR, lower, upper]

# --- Step 2. Clip values to reduce impact of extreme outliers ---
df1_clipped = df1_filled.copy()
for col in cols_with_outliers:
    lower = outlier_summary.loc[col, 'Lower Bound']
    upper = outlier_summary.loc[col, 'Upper Bound']
    df1_clipped[col] = df1_clipped[col].clip(lower, upper)

# --- Step 3. Scale clipped features using RobustScaler ---
scaler = RobustScaler()
df1_scaled = df1_clipped.copy()
df1_scaled[cols_with_outliers] = scaler.fit_transform(df1_clipped[cols_with_outliers])

# --- Step 4. (Optional) Inspect results ---
print("âœ… Scaling complete.")
print("Scaled columns:", cols_with_outliers)
print("\nScaled data preview:")
display(df1_scaled[cols_with_outliers].describe().T)

# Optional: quick before/after comparison for one column
col_example = 'x_10'
import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(df1_filled[col_example], bins=50)
plt.title(f"Before scaling: {col_example}")
plt.subplot(1,2,2)
plt.hist(df1_scaled[col_example], bins=50)
plt.title(f"After scaling: {col_example}")
plt.show()



# ==========================================
# Final XGBoost Binary Classifier Training
# ==========================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# 1. Split data
# -------------------------
X = df1_scaled
y = df['Y'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# 2. Train final XGBoost model
# -------------------------
xgb_model = XGBClassifier(
    n_estimators=300,       # number of boosting rounds
    learning_rate=0.05,     # smaller = slower but more stable learning
    max_depth=6,            # controls model complexity
    subsample=0.8,          # row sampling
    colsample_bytree=0.8,   # feature sampling
    eval_metric='logloss',
    random_state=42,
    use_label_encoder=False
)

xgb_model.fit(X_train, y_train)

# -------------------------
# 3. Predict and evaluate
# -------------------------
y_pred = xgb_model.predict(X_test)
y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_prob)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"âœ… XGBoost Performance:")
print(f"ROC-AUC: {auc:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------
# 4. Plot ROC curve
# -------------------------
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost')
plt.legend()
plt.grid(True)
plt.show()



# ==========================================
# Final Random Forest Binary Classifier Training
# ==========================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# 1. Split data
# -------------------------
X = df1_scaled
y = df['Y'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# 2. Train final Random Forest model
# -------------------------
model_random_forest = RandomForestClassifier(
    n_estimators=300,        # number of trees
    max_depth=10,            # limit depth to prevent overfitting
    min_samples_split=2,     # minimum samples required to split a node
    min_samples_leaf=1,      # minimum samples at a leaf node
    max_features='sqrt',     # number of features considered for split
    bootstrap=True,          # use bootstrap samples
    random_state=42,
    n_jobs=-1                # use all available cores
)

model_random_forest.fit(X_train, y_train)

# -------------------------
# 3. Predict and evaluate
# -------------------------
y_pred = model_random_forest.predict(X_test)
y_pred_prob = model_random_forest.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_prob)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"âœ… Random Forest Performance:")
print(f"ROC-AUC: {auc:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------
# 4. Plot ROC curve
# -------------------------
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.grid(True)
plt.show()



import pandas as pd

# Assume df1_filled is your numeric DataFrame
outlier_summary = pd.DataFrame(columns=['Q1', 'Q3', 'IQR', 'Lower Bound', 'Upper Bound', 'Outlier Count'])

for col in df1_scaled.select_dtypes(include=[np.number]).columns:
    Q1 = df1_scaled[col].quantile(0.25)
    Q3 = df1_scaled[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_count = ((df1_filled[col] < lower) | (df1_filled[col] > upper)).sum()
    outlier_summary.loc[col] = [Q1, Q3, IQR, lower, upper, outlier_count]

print("ðŸ“Š Outlier summary (IQR method):")
display(outlier_summary.sort_values('Outlier Count', ascending=False))


