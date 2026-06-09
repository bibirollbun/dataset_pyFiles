import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")


# Configuration
class CFG:
    train_path = '/kaggle/input/playground-series-s5e7/train.csv'
    test_path = '/kaggle/input/playground-series-s5e7/test.csv'
    sample_sub_path = '/kaggle/input/playground-series-s5e7/sample_submission.csv'
    target = 'Personality'
    n_folds = 5
    seed = 42

# Load data
train = pd.read_csv(CFG.train_path, index_col='id')
test = pd.read_csv(CFG.test_path, index_col='id')

# Preprocess categorical features
for col in ["Stage_fear", "Drained_after_socializing"]:
    train[col] = train[col].map({"No": 0, "Yes": 1})
    test[col] = test[col].map({"No": 0, "Yes": 1})


# Encode target
train[CFG.target] = train[CFG.target].map({"Extrovert": 0, "Introvert": 1})

# Features and target
X = train.drop(CFG.target, axis=1)
y = train[CFG.target].astype(int)  # Ensure target is integer
X_test = test.copy()


# Base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=CFG.seed)),
    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=CFG.seed)),
    ('lgb', LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=CFG.seed))
]

# Meta model
meta_model = LogisticRegression()

# Stacking pipeline
stacked_model = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("classifier", StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=CFG.n_folds,
        passthrough=True,
        n_jobs=-1
    ))
])


# Cross-validation
cv = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    try:
        stacked_model.fit(X_train, y_train)
        preds = stacked_model.predict(X_val)
        score = accuracy_score(y_val, preds)
        scores.append(score)
        print(f"âœ… Fold {fold + 1} Accuracy: {score:.4f}")
    except Exception as e:
        print(f"â�Œ Fold {fold + 1} raised an error: {e}")

print(f"ğŸ“Š Mean CV Accuracy: {np.mean(scores):.4f}")


# Train on full data and generate predictions
stacked_model.fit(X, y)
final_preds = stacked_model.predict(X_test)


# Create correct submission DataFrame
submission = pd.DataFrame({
    "id": X_test.index,
    "Personality": ["Extrovert" if p == 0 else "Introvert" for p in final_preds]
})

# Save
submission.to_csv("submission.csv", index=False)
print("âœ… Final submission file created: submission.csv")


