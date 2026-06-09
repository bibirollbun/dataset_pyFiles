import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# ==========================================
# 1. Configuration & Data Loading
# ==========================================
# Paths for Kaggle environment
TRAIN_PATH = '/kaggle/input/playground-series-s5e11/train.csv'
TEST_PATH = '/kaggle/input/playground-series-s5e11/test.csv'
SUBMISSION_PATH = 'submission.csv'

print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# ==========================================
# 2. Preprocessing
# ==========================================
target = 'loan_paid_back'

# Separate features and target
X = train_df.drop(columns=[target, 'id'])
y = train_df[target]
X_test = test_df.drop(columns=['id'])

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns found: {categorical_cols}")

# Encode categorical variables
# OrdinalEncoder is preferred over OneHotEncoder for tree-based models on high-cardinality data
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Fit on training data and transform both train and test
X[categorical_cols] = encoder.fit_transform(X[categorical_cols])
X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])

# ==========================================
# 3. Model Training
# ==========================================
# Split data for internal validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training HistGradientBoostingClassifier...")
# This model is similar to LightGBM/XGBoost but built into Scikit-Learn
model = HistGradientBoostingClassifier(
    random_state=42,
    max_iter=300,           # Number of boosting iterations
    learning_rate=0.05,     # Slower learning rate for better generalization
    max_leaf_nodes=31,      # Maximum leaves per tree
    l2_regularization=1.0,  # Regularization to prevent overfitting
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20
)

model.fit(X_train, y_train)

# ==========================================
# 4. Evaluation
# ==========================================
y_pred_val = model.predict_proba(X_val)[:, 1]
val_score = roc_auc_score(y_val, y_pred_val)
print(f"Validation ROC AUC Score: {val_score:.5f}")

# ==========================================
# 5. Submission Generation
# ==========================================
print("Generating predictions for test set...")
y_pred_test = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    'id': test_df['id'],
    'loan_paid_back': y_pred_test
})

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")
print(submission.head())

