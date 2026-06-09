# ==============================
# ğŸ“�  Import Libraries
# ==============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# === Preprocessing ===
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# === Models ===
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# === Metrics ===
from sklearn.metrics import roc_auc_score

# ==============================
# ğŸ“‚ Load Datasets
# ==============================
df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")

# Initial Exploration
display(df.head())

# ==============================
# 1. Define Features (X) and Target (y)
# ==============================
# All columns except 'id' and the target 'diagnosed_diabetes'
X = df.drop(columns=['id', 'diagnosed_diabetes']) 
y = df['diagnosed_diabetes']

# We'll use this for the final submission
X_test = test.drop(columns=['id'])

# ==============================
# 2. Create Preprocessing Pipeline
# ==============================

# Identify numeric and categorical columns automatically
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Create transformers
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Create the preprocessor with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ],
    remainder='passthrough'
)

# ==============================
# 3. Define Baseline Models (Classifiers)
# ==============================

models = {
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=9,
        random_state=42,
        learning_rate=0.01,
        colsample_bytree=0.6,
        n_jobs=-1
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=200, 
        learning_rate=0.1, 
        max_depth=-1, 
        random_state=42, 
        n_jobs=-1,
        verbose=-1
    ),
    "CatBoost": CatBoostClassifier(
        n_estimators=200, 
        learning_rate=0.1, 
        depth=6, 
        random_state=42, 
        verbose=0
    )
}

# ==============================
# 4. K-Fold Cross-Validation & Base Submissions
# ==============================
# StratifiedKFold for classification
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    # Create a pipeline: preprocessing + model
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    # Store AUC for each fold
    fold_auc = []

    # 5-Fold CV
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        pipeline.fit(X_train, y_train)
        
        # Predict probabilities for the positive class (1)
        preds_proba = pipeline.predict_proba(X_val)[:, 1]
        
        auc = roc_auc_score(y_val, preds_proba)
        fold_auc.append(auc)
    
    print(f"{name} mean ROC AUC: {np.mean(fold_auc):.4f}")
    
    # ========== FULL FIT + SUBMISSION ==========
    pipeline.fit(X, y)
    
    # Predict probabilities for the test set
    test_preds_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Create submission dataframe with the correct target column name
    submission = pd.DataFrame({
        "id": test["id"],
        "diagnosed_diabetes": test_preds_proba
    })
    
    submission.to_csv(f"submission_{name}.csv", index=False)
    print(f"Saved submission_{name}.csv")

