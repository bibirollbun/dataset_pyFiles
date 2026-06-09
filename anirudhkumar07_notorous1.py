# ===================== FINAL COMPLETE CLASSIFICATION PIPELINE =====================
# Covers:
# 1. Data Cleaning & Preprocessing
# 2. Data Visualization & Outlier Analysis
# 3. Encoding, Scaling & Correlation Analysis
# 4. Model Training, Evaluation & Hyperparameter Tuning
# 5. Test Prediction & Kaggle Submission CSV
# ================================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ------------------------------------------------------------------------------
# ğŸ”´ CHANGE ONLY THESE FOR A NEW DATASET
# ------------------------------------------------------------------------------
TRAIN_FILE = "/kaggle/input/mse-2-ai-201-b-ai-a/train.csv"                  # Kaggle path if needed
TEST_FILE  = "/kaggle/input/mse-2-ai-201-b-ai-a/test.csv"
TARGET_COL = "Class"       # Target column
ID_COL     = "id"                         # Set None if not present
# ------------------------------------------------------------------------------

# ===================== 1ï¸�âƒ£ LOAD DATA =====================
train_df = pd.read_csv(TRAIN_FILE)
test_df  = pd.read_csv(TEST_FILE)

# ===================== 2ï¸�âƒ£ DATA CLEANING & PREPROCESSING =====================

# Remove rows with missing target (never impute target)
train_df = train_df.dropna(subset=[TARGET_COL])

# Remove duplicate rows
train_df.drop_duplicates(inplace=True)

# Separate features and target
y = train_df[TARGET_COL].copy()
X = train_df.drop(columns=[TARGET_COL]).copy()

# Identify numerical and categorical columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns

# Handle missing numerical values using median
for col in num_cols:
    X[col].fillna(X[col].median(), inplace=True)
    if col in test_df.columns:
        test_df[col].fillna(test_df[col].median(), inplace=True)

# Handle missing categorical values using constant
for col in cat_cols:
    X[col].fillna("missing", inplace=True)
    if col in test_df.columns:
        test_df[col].fillna("missing", inplace=True)

# ===================== 3ï¸�âƒ£ DATA VISUALIZATION & OUTLIER ANALYSIS =====================

# Target distribution
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Target Variable Distribution")
plt.show()

# Outlier detection using boxplots
for col in num_cols:
    plt.figure(figsize=(5,3))
    sns.boxplot(x=X[col])
    plt.title(f"Outlier Analysis: {col}")
    plt.show()

# ===================== 4ï¸�âƒ£ CORRELATION ANALYSIS =====================

numeric_corr = train_df.select_dtypes(include=["int64", "float64"])
if numeric_corr.shape[1] > 1:
    plt.figure(figsize=(10,6))
    sns.heatmap(numeric_corr.corr(), cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.show()

# ===================== 5ï¸�âƒ£ ENCODING CATEGORICAL VARIABLES =====================

# One-hot encoding (safe for unseen categories)
X = pd.get_dummies(X, drop_first=True)
test_df = pd.get_dummies(test_df, drop_first=True)

# Align train and test columns
X, test_df = X.align(test_df, join="left", axis=1, fill_value=0)

# ===================== 6ï¸�âƒ£ NUMERIC SAFETY FIXES =====================

# Remove constant (zero-variance) columns
constant_cols = [c for c in X.columns if X[c].nunique() <= 1]
X.drop(columns=constant_cols, inplace=True)
test_df.drop(columns=constant_cols, inplace=True, errors="ignore")

# Replace infinite values
X.replace([np.inf, -np.inf], 0, inplace=True)
test_df.replace([np.inf, -np.inf], 0, inplace=True)

# Final NaN cleanup
X.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

# ===================== 7ï¸�âƒ£ FEATURE SCALING =====================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_df)

# Sanity check
assert not np.isnan(X_scaled).any(), "NaN detected in features"
assert not np.isinf(X_scaled).any(), "Inf detected in features"

# ===================== 8ï¸�âƒ£ TRAINâ€“VALIDATION SPLIT (SAFE) =====================

min_class_count = y.value_counts().min()

if min_class_count < 2:
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print("âš ï¸� Stratification skipped due to rare class")
else:
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

# ===================== 9ï¸�âƒ£ MODEL TRAINING =====================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ===================== ğŸ”Ÿ MODEL EVALUATION =====================

val_preds = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_preds))
print("Confusion Matrix:\n", confusion_matrix(y_val, val_preds))

# ===================== 1ï¸�âƒ£1ï¸�âƒ£ HYPERPARAMETER TUNING =====================

param_grid = {
    "max_depth": [None, 10],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("Best Parameters:", grid.best_params_)

# ===================== 1ï¸�âƒ£2ï¸�âƒ£ FINAL MODEL TRAINING =====================

best_model.fit(X_scaled, y)

# ===================== 1ï¸�âƒ£3ï¸�âƒ£ TEST PREDICTION =====================

test_predictions = best_model.predict(test_scaled)

# ===================== 1ï¸�âƒ£4ï¸�âƒ£ CREATE SUBMISSION CSV =====================

if ID_COL and ID_COL in test_df.columns:
    submission = pd.DataFrame({
        ID_COL: test_df[ID_COL],
        TARGET_COL: test_predictions
    })
else:
    submission = pd.DataFrame({
        "id": range(1, len(test_predictions) + 1),
        TARGET_COL: test_predictions
    })

submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv created successfully")

# ===================== END OF FINAL PIPELINE =====================

# b) Deep learning can automatically learn complex patterns from large data
# c) Overfitting due to high model complexity
# b) They reduce vanishing gradient problems
# c) Time or energy saved after reaching the destination efficiently
# c) To discover better actions that may give higher long-term rewards
# d) Insufficient exploration (agent stuck exploiting)
# c) Convolutional Neural Network (CNN)
# b) They detect local patterns like edges and textures
# c) Overfitting to training conditions (poor generalization)
# c) They explain sudden demand spikes
# c) Rare events not well represented in training data
# Predictions rely on broader neighborhood averaging
# c) No historical sales data (cold start problem)
# d) Large under-forecast causing lost sales
# c) They may be unnecessarily complex for simple patterns
# Overfitting due to increased variance
# Concept drift over time
# d) CNN â†’ Package damage detection, RL â†’ warehouse optimization, DL â†’ demand patterns
# c) Predictions violate supply chain constraints (storage, delivery limits)
# b) Batch normalization statistics differ between training and inference due to non-stationary data



1. Amazon uses a deep learning model for product recommendations. Performance improves as more data is added. Why is deep learning suitable here?  
**Answer:** b) Deep learning can automatically learn complex patterns from large data[1][2]

2. A deep neural network trains well but performs poorly on unseen data. What is the MOST likely issue?  
**Answer:** c) Overfitting due to high model complexity[3]

3. Why are ReLU activations commonly used in deep networks?  
**Answer:** b) They reduce vanishing gradient problems[4][5]

4. Amazon uses reinforcement learning to optimise warehouse robot paths. What represents the reward?  
**Answer:** c) Time or energy saved after reaching the destination efficiently[6]

5. In reinforcement learning, why is exploration important?  
**Answer:** c) To discover better actions that may give higher long-term rewards[7][8]

6. An RL agent always chooses the same action even when itâ€™s not optimal. What is the MOST likely cause?  
**Answer:** d) Insufficient exploration (agent stuck exploiting)[8][7]

7. Amazon uses computer vision to detect damaged packages from images. Which deep learning model is MOST suitable?  
**Answer:** c) Convolutional Neural Network (CNN)[9]

8. Why are convolution layers effective for image processing?  
**Answer:** b) They detect local patterns like edges and textures[9]

9. A CV model performs well on training images but fails on real warehouse images with different lighting. What is the MAIN issue?  
**Answer:** c) Overfitting to training conditions (poor generalization)[3]

10. Amazon forecasts daily product demand using historical sales data. Why are holidays and promotions important features?  
**Answer:** c) They explain sudden demand spikes[10]

11. Demand forecasts work well normally but fail during major sales events like Prime Day. Why?  
**Answer:** c) Rare events not well represented in training data[10][3]

12. Why does increasing K generally reduce variance but increase bias?  
**Answer:** Predictions rely on broader neighborhood averaging[3]

13. Why is forecasting demand for new products difficult?  
**Answer:** c) No historical sales data (cold start problem)[10]

14. Amazon wants to reduce stockouts. Which forecasting error is more dangerous?  
**Answer:** d) Large under-forecast causing lost sales[10]

15. Why are deep learning models sometimes avoided for simple forecasting tasks?  
**Answer:** c) They may be unnecessarily complex for simple patterns.[3]

16. Adding a new feature reduces training error but increases test error. What does this indicate?  
**Answer:** Overfitting due to increased variance.[3]

17. Which situation can break a supervised model in production even if accuracy was high during testing?  
**Answer:** Concept drift over time[3]

18. Which combination BEST matches the task?  
**Answer:** d) CNN â†’ Package damage detection, RL â†’ warehouse optimization, DL â†’ demand patterns[6][9]

19. A demand model shows excellent accuracy but is rejected by operations. Why?  
**Answer:** c) Predictions violate supply chain constraints (storage, delivery limits)[10]

20. What is the MOST likely deep-learningâ€“specific root cause of the unstable behavior after adding more data?  
**Answer:** b) Batch normalization statistics differ between training and inference due to non-stationary data[3]


