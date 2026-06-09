import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

# Explore data structure
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("\nTrain columns:", train_df.columns.tolist())
print("\nMissing values in train:", train_df.isnull().sum().sum())
print("Missing values in test:", test_df.isnull().sum().sum())

# Separate features and target
X = train_df.drop(['id', 'y'], axis=1)
y = train_df['y']
test_ids = test_df['id']
X_test = test_df.drop('id', axis=1)

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Preprocessing function
def preprocess_data(X, X_test=None, categorical_cols=None, numerical_cols=None):
    """
    Preprocess the data: handle missing values, encode categorical variables, scale numerical features
    """
    X_processed = X.copy()
    
    if X_test is not None:
        X_test_processed = X_test.copy()
    
    # Handle missing values
    if categorical_cols:
        # For categorical columns, fill with mode
        for col in categorical_cols:
            if col in X_processed.columns:
                mode_val = X_processed[col].mode()[0] if not X_processed[col].mode().empty else 'Unknown'
                X_processed[col].fillna(mode_val, inplace=True)
                if X_test is not None:
                    X_test_processed[col].fillna(mode_val, inplace=True)
    
    if numerical_cols:
        # For numerical columns, fill with median
        for col in numerical_cols:
            if col in X_processed.columns:
                median_val = X_processed[col].median()
                X_processed[col].fillna(median_val, inplace=True)
                if X_test is not None:
                    X_test_processed[col].fillna(median_val, inplace=True)
    
    # Label encode categorical variables
    label_encoders = {}
    if categorical_cols:
        for col in categorical_cols:
            if col in X_processed.columns:
                le = LabelEncoder()
                # Fit on combined train + test data to handle unseen categories
                combined = pd.concat([X_processed[col], X_test_processed[col] if X_test is not None else pd.Series()], axis=0)
                le.fit(combined)
                X_processed[col] = le.transform(X_processed[col])
                if X_test is not None:
                    X_test_processed[col] = le.transform(X_test_processed[col])
                label_encoders[col] = le
    
    # Scale numerical features
    if numerical_cols:
        scaler = StandardScaler()
        X_processed[numerical_cols] = scaler.fit_transform(X_processed[numerical_cols])
        if X_test is not None:
            X_test_processed[numerical_cols] = scaler.transform(X_test_processed[numerical_cols])
    
    if X_test is not None:
        return X_processed, X_test_processed, label_encoders
    else:
        return X_processed, label_encoders

# Preprocess the data
X_processed, X_test_processed, label_encoders = preprocess_data(
    X, X_test, categorical_cols, numerical_cols
)

# Check class distribution
print(f"Class distribution: {y.value_counts()}")
print(f"Positive class ratio: {y.mean():.4f}")

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# Model 1: XGBoost
print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='auc',
    use_label_encoder=False
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False
)

# Model 2: LightGBM - FIXED: Remove verbose parameter
print("Training LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    metric='auc',
    verbose=-1  # Set verbosity in the constructor instead
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50)]
    # verbose parameter removed from fit() method
)

# Model 3: CatBoost
print("Training CatBoost...")
cat_model = cb.CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_state=42,
    verbose=False,
    eval_metric='AUC'
)

cat_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50,
    verbose=False
)

# Evaluate models
models = {
    'XGBoost': xgb_model,
    'LightGBM': lgb_model,
    'CatBoost': cat_model
}

print("\nModel Evaluation:")
for name, model in models.items():
    if name == 'XGBoost':
        y_pred = model.predict_proba(X_val, iteration_range=(0, model.best_iteration))[:, 1]
    elif name == 'LightGBM':
        y_pred = model.predict_proba(X_val)[:, 1]  # LightGBM automatically uses best iteration
    else:
        y_pred = model.predict_proba(X_val)[:, 1]
    
    auc = roc_auc_score(y_val, y_pred)
    print(f"{name} Validation AUC: {auc:.4f}")

# Cross-validation with the best model
print("\nCross-validation...")
best_model = lgb_model  # Based on validation performance
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_processed, y, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"Cross-validation AUC scores: {cv_scores}")
print(f"Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Retrain best model on full training data
print("Retraining on full data...")
best_model.fit(X_processed, y)

# Make predictions on test set
print("Making predictions...")
test_preds = best_model.predict_proba(X_test_processed)[:, 1]

# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'y': test_preds
})

# Save submission
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file saved as 'submission.csv'")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': X_processed.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10))




