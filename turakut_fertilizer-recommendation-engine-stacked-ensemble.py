!kaggle competitions download -c playground-series-s5e6


# Import libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import top_k_accuracy_score
from sklearn.base import clone
import os

# Custom MAP@3 metric
def mapk(y_true, y_pred, k=3):
    ap = []
    for true, pred in zip(y_true, y_pred):
        score = 0.0
        num_hits = 0.0
        
        for i, p in enumerate(pred[:k]):
            if p == true:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        ap.append(score / min(len(pred), k))
    return np.mean(ap)

# Load data
print("Loading data...")
train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)


# Identify features
target_column = 'Fertilizer Name'
categorical_features = ['Soil Type', 'Crop Type']
numeric_features = [col for col in train_df.columns 
                   if col not in categorical_features + [target_column, 'id'] 
                   and pd.api.types.is_numeric_dtype(train_df[col])]

print("\nIdentified features:")
print("Categorical:", categorical_features)
print("Numeric:", numeric_features)

# Prepare data
X_train = train_df[categorical_features + numeric_features]
y_train = train_df[target_column]
X_test = test_df[categorical_features + numeric_features]

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y_train)

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numeric_features)
])


# Random Forest Pipeline
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        n_jobs=-1))
])

# HistGradientBoosting Pipeline
histgbt_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(
        max_iter=150,
        random_state=42))
])

# Cross-validation setup
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Storage for predictions
rf_models = []
rf_val_preds = np.zeros((len(X_train), len(le.classes_)))
histgbt_models = []
histgbt_val_preds = np.zeros((len(X_train), len(le.classes_)))

print("\nTraining Level 1 models...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_encoded), 1):
    print(f"\nFold {fold}")
    
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_encoded[train_idx], y_encoded[val_idx]
    
    # Train Random Forest
    print("Training Random Forest...")
    rf_model_fold = clone(rf_model)
    rf_model_fold.fit(X_train_fold, y_train_fold)
    rf_models.append(rf_model_fold)
    rf_val_preds[val_idx] = rf_model_fold.predict_proba(X_val_fold)
    
    # Train HistGradientBoosting
    print("Training HistGradientBoosting...")
    histgbt_model_fold = clone(histgbt_model)
    histgbt_model_fold.fit(X_train_fold, y_train_fold)
    histgbt_models.append(histgbt_model_fold)
    histgbt_val_preds[val_idx] = histgbt_model_fold.predict_proba(X_val_fold)


print("\nTraining Meta-Model (Logistic Regression)...")
meta_val_preds = np.zeros((len(X_train), len(le.classes_)))
meta_models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_encoded), 1):
    print(f"Meta-Fold {fold}")
    
    X_meta_train = np.hstack((rf_val_preds[train_idx], histgbt_val_preds[train_idx]))
    X_meta_val = np.hstack((rf_val_preds[val_idx], histgbt_val_preds[val_idx]))
    y_meta_train = y_encoded[train_idx]
    
    meta_model = LogisticRegression(
        max_iter=500,
        C=0.1,
        solver='lbfgs',
        multi_class='multinomial',
        random_state=42
    )
    meta_model.fit(X_meta_train, y_meta_train)
    meta_models.append(meta_model)
    meta_val_preds[val_idx] = meta_model.predict_proba(X_meta_val)

# Evaluate
top3_preds = np.argsort(meta_val_preds, axis=1)[:, -3:][:, ::-1]
top3_acc = top_k_accuracy_score(y_encoded, meta_val_preds, k=3)
map3_score = mapk(y_encoded.tolist(), top3_preds.tolist(), k=3)

print("\nValidation Results:")
print(f"Top-3 Accuracy: {top3_acc:.4f}")
print(f"MAP@3: {map3_score:.4f}")


print("\nGenerating Test Predictions...")
meta_test_pred_probs = np.zeros((len(X_test), len(le.classes_)))

for i in range(len(rf_models)):
    rf_probs = rf_models[i].predict_proba(X_test)
    hist_probs = histgbt_models[i].predict_proba(X_test)
    
    stacked_test = np.hstack((rf_probs, hist_probs))
    meta_probs = meta_models[i].predict_proba(stacked_test)
    meta_test_pred_probs += meta_probs / len(meta_models)

# Get top 3 predictions
top3_preds = np.argsort(meta_test_pred_probs, axis=1)[:, -3:][:, ::-1]
top3_labels = le.inverse_transform(top3_preds.ravel()).reshape(top3_preds.shape)
fertilizer_preds = [' '.join(row) for row in top3_labels]

# Save submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': fertilizer_preds
})

submission.to_csv('/kaggle/working/submission.csv', index=False)
print("\nSubmission saved to /kaggle/working/submission.csv")
print("First 5 predictions:")
print(submission.head())

