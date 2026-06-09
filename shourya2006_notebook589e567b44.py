#Optimized Binary Classification Pipeline
# With Automatic Model Selection + Stacking + Threshold Tuning

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/predicting-euphoria-in-the-streets/train.csv")
test = pd.read_csv("/kaggle/input/predicting-euphoria-in-the-streets/test.csv")

id_col = "id"
target_col = "Y"

X = train.drop(columns=[id_col, target_col])
y = train[target_col].astype(int)
X_test = test.drop(columns=[id_col])






# Preprocessing


# Replace infinities and NaNs
X = X.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Impute with median
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Standardize
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)



models = {
    "RandomForest": RandomForestClassifier(n_estimators=400, max_depth=10, random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=400, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RidgeClassifier": RidgeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42, iterations=700, depth=6, learning_rate=0.05),
    "LightGBM": LGBMClassifier(n_estimators=700, learning_rate=0.05, random_state=42)
}



#CV Training Loop + F1 Optimization

NFOLDS = 5
best_f1 = 0
best_model_name = None
best_test_pred = None
best_threshold = 0.5

for name, model in models.items():
    print(f"Training {name} ...")
    
    oof = np.zeros(len(X))
    preds_test = np.zeros(len(X_test))
    
    skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X_scaled.iloc[tr_idx], y.iloc[tr_idx]
        X_val, y_val = X_scaled.iloc[val_idx], y.iloc[val_idx]
        
        model.fit(X_tr, y_tr)
        
        #Robust probability handling
        try:
            val_pred = model.predict_proba(X_val)[:, 1]
            test_pred = model.predict_proba(X_test_scaled)[:, 1]
        except AttributeError:
            if hasattr(model, "decision_function"):
                val_pred = model.decision_function(X_val)
                val_pred = (val_pred - val_pred.min()) / (val_pred.max() - val_pred.min())
                test_pred = model.decision_function(X_test_scaled)
                test_pred = (test_pred - test_pred.min()) / (test_pred.max() - test_pred.min())
            else:
                val_pred = model.predict(X_val)
                test_pred = model.predict(X_test_scaled)
        
        oof[val_idx] = val_pred
        preds_test += test_pred / NFOLDS

    # Threshold tuning
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = [f1_score(y, (oof > t).astype(int)) for t in thresholds]
    best_idx = np.argmax(f1_scores)
    f1 = f1_scores[best_idx]
    threshold = thresholds[best_idx]
    
    print(f"{name}: Best threshold={threshold:.2f}, CV F1={f1:.5f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_test_pred = preds_test
        best_threshold = threshold

print(f"\nBest Base Model: {best_model_name} | F1={best_f1:.5f} | Threshold={best_threshold:.2f}")



# Select top-performing models for stacking
stack_models = [
    ('rf', RandomForestClassifier(n_estimators=400, max_depth=10, random_state=42)),
    ('cat', CatBoostClassifier(verbose=0, random_state=42, iterations=700, depth=6, learning_rate=0.05)),
    ('lgbm', LGBMClassifier(n_estimators=700, learning_rate=0.05, random_state=42))
]

stack_final = LogisticRegression(max_iter=1000, random_state=42)

stack_clf = StackingClassifier(
    estimators=stack_models,
    final_estimator=stack_final,
    n_jobs=-1,
    passthrough=True
)


oof_stack = np.zeros(len(X))
test_stack = np.zeros(len(X_test))
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X_scaled.iloc[tr_idx], y.iloc[tr_idx]
    X_val, y_val = X_scaled.iloc[val_idx], y.iloc[val_idx]
    
    stack_clf.fit(X_tr, y_tr)
    val_pred = stack_clf.predict_proba(X_val)[:, 1]
    test_pred = stack_clf.predict_proba(X_test_scaled)[:, 1]
    
    oof_stack[val_idx] = val_pred
    test_stack += test_pred / NFOLDS



# Tune threshold for stacking
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = [f1_score(y, (oof_stack > t).astype(int)) for t in thresholds]
best_idx = np.argmax(f1_scores)
stack_f1 = f1_scores[best_idx]
stack_threshold = thresholds[best_idx]

print(f"\nFinal Stacking F1: {stack_f1:.5f} | Threshold={stack_threshold:.2f}")


submission = pd.DataFrame({
    id_col: test[id_col],
    target_col: (test_stack > stack_threshold).astype(int)
})
submission.to_csv("submission.csv", index=False)

print("\nSubmission file saved")
print(f"Best F1 Achieved: {stack_f1:.5f}")


