# ---------------------- IMPORT LIBRARIES ----------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve

# Boosting libraries
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")


# ---------------------- LOAD DATA ----------------------
# Load datasets
df_train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df_test  = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

# Remove duplicates & target NaN
df_train.dropna(subset=['y'], inplace=True)
df_train.drop_duplicates(inplace=True)


# ---------------------- FEATURE AND TARGET ----------------------
# Target mapping
y = df_train['y'].astype(int)
X = df_train.drop(columns=['y'])


# Binary encoding
binary_cols = ['default','housing','loan']
for col in binary_cols:
    if col in X.columns:
        X[col] = X[col].map({'yes':1,'no':0})
    if col in df_test.columns:
        df_test[col] = df_test[col].map({'yes':1,'no':0})

# One-hot encoding
categorical_cols = ['job','marital','education','contact','month','poutcome']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
df_test = pd.get_dummies(df_test, columns=[c for c in categorical_cols if c in df_test.columns], drop_first=True)

# Align test columns
df_test = df_test.reindex(columns=X.columns, fill_value=0)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ---------------------- XGBOOST RANDOMIZED SEARCH ----------------------
xgb = XGBClassifier(
    use_label_encoder=False,  
    eval_metric='logloss',    
    random_state=42
)

xgb_params = {
    "n_estimators": [500, 1000],
    "subsample": [0.7, 0.8],
    "max_depth": [5, 7],
    "learning_rate": [0.05, 0.01],
    "colsample_bytree": [0.8, 1.0]
}

xgb_cv = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=xgb_params,
    n_iter=10,
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42,
    scoring="roc_auc"
)

# Fit on training set
xgb_cv.fit(X_train, y_train)


# Best parameters
print("Best XGBoost Params:", xgb_cv.best_params_)


# Train XGBoost with best parameters
xgb_tuned = XGBClassifier(max_depth=7,subsample=0.8, n_estimators=1000, learning_rate=0.05, colsample_bytree=1.0).fit(X_train, y_train)


# ---------------------- Validation Metrics ----------------------

y_val_pred = xgb_tuned.predict(X_val)
y_val_proba = xgb_tuned.predict_proba(X_val)[:,1]

acc_xgb = accuracy_score(y_val, y_val_pred)
roc_xgb = roc_auc_score(y_val, y_val_proba)
f1_xgb = f1_score(y_val, y_val_pred)
recall_xgb = recall_score(y_val, y_val_pred)
prec_xgb = precision_score(y_val, y_val_pred)

print("XGBoost Validation Metrics:")
print(f"Accuracy: {acc_xgb}")
print(f"ROC-AUC: {roc_xgb}")
print(f"F1 Score: {f1_xgb}")
print(f"Recall: {recall_xgb}")
print(f"Precision: {prec_xgb}")



# ---------------------- XGBOOST FEATURE IMPORTANCE ----------------------
top_n = 15
xgb_importance = pd.Series(xgb_tuned.feature_importances_, index=X_train.columns).sort_values(ascending=False)
xgb_top_features = xgb_importance.head(top_n)
xgb_top_percent = 100 * xgb_top_features / xgb_top_features.sum()

plt.figure(figsize=(10,6))
sns.barplot(x=xgb_top_percent.values, y=xgb_top_percent.index, palette="rocket")
plt.title(f"Top {top_n} XGBoost Feature Importance")
plt.xlabel("Importance (%)")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()



# ---------------------- LIGHTGBM RANDOMIZED SEARCH ----------------------
lgbm = LGBMClassifier(verbose=-1, random_state=42)
lgbm_params = {
    "learning_rate": [0.05, 0.01],
    "max_depth": [5, 7, -1],
    "n_estimators": [500, 1000],
    "num_leaves": [31, 63]
}

lgbm_cv = RandomizedSearchCV(
    estimator=lgbm,
    param_distributions=lgbm_params,
    n_iter=8,
    cv=3,
    n_jobs=-1,
    verbose=1,
    random_state=42,
    scoring="roc_auc"
)

# Fit on training set
lgbm_cv.fit(X_train, y_train)


print("Best LightGBM Params:", lgbm_cv.best_params_)


# Train with best params
lgbm_tuned = LGBMClassifier(
     num_leaves=63, 
    n_estimators=1000,
    learning_rate=0.05, 
    max_depth= 7,
    verbose=-1,
    random_state=42
).fit(X_train, y_train)


# ---------------------- Validation Metrics ----------------------
y_val_pred = lgbm_tuned.predict(X_val)
y_val_proba = lgbm_tuned.predict_proba(X_val)[:,1]

acc_lgbm = accuracy_score(y_val, y_val_pred)
roc_lgbm = roc_auc_score(y_val, y_val_proba)
f1_lgbm = f1_score(y_val, y_val_pred)
recall_lgbm = recall_score(y_val, y_val_pred)
prec_lgbm = precision_score(y_val, y_val_pred)

print("LightGBM Validation Metrics:")
print("Accuracy:", acc_lgbm)
print("ROC-AUC:", roc_lgbm)
print("F1:", f1_lgbm)
print("Recall:", recall_lgbm)
print("Precision:", prec_lgbm)


# LIGHTGBM FEATURE IMPORTANCE
lgbm_importance = pd.Series(lgbm_tuned.feature_importances_, index=X_train.columns).sort_values(ascending=False)
lgbm_top_features = lgbm_importance.head(top_n)
lgbm_top_percent = 100 * lgbm_top_features / lgbm_top_features.sum()

plt.figure(figsize=(10,6))
sns.barplot(x=lgbm_top_percent.values, y=lgbm_top_percent.index, palette="coolwarm")
plt.title(f"Top {top_n} LightGBM Feature Importance")
plt.xlabel("Importance (%)")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# ---------------------- CATBOOST RANDOMIZED SEARCH ----------------------
catb = CatBoostClassifier(verbose=0, random_state=42)
catb_params = {
    "iterations": [200, 500],
    "learning_rate": [ 0.05, 0.1],
    "depth": [4, 5, 6],
    "l2_leaf_reg": [1, 3]
}

catb_cv = RandomizedSearchCV(
    estimator=catb,
    param_distributions=catb_params,
    n_iter=5,
    cv=3,
    n_jobs=-1,
    verbose=1,
    random_state=42,
    scoring="roc_auc"
)
catb_cv.fit(X_train, y_train)



print("Best CatBoost Params:", catb_cv.best_params_)


catb_tuned = CatBoostClassifier(depth=5, iterations=500, learning_rate=0.1,l2_leaf_reg=3 , verbose=0).fit(X_train, y_train)


# ---------------------- Validation Metrics ----------------------
y_val_pred = catb_tuned.predict(X_val)
y_val_proba = catb_tuned.predict_proba(X_val)[:,1]

acc_catb = accuracy_score(y_val, y_val_pred)
roc_catb = roc_auc_score(y_val, y_val_proba)
f1_catb = f1_score(y_val, y_val_pred)
recall_catb = recall_score(y_val, y_val_pred)
prec_catb = precision_score(y_val, y_val_pred)

print("CatBoost Validation Metrics:")
print("Accuracy:", acc_catb)
print("ROC-AUC:", roc_catb)
print("F1:", f1_catb)
print("Recall:", recall_catb)
print("Precision:", prec_catb)


# CATBOOST FEATURE IMPORTANCE
catb_importance = pd.Series(catb_tuned.feature_importances_, index=X_train.columns).sort_values(ascending=False)
catb_top_features = catb_importance.head(top_n)
catb_top_percent = 100 * catb_top_features / catb_top_features.sum()

plt.figure(figsize=(10,6))
sns.barplot(x=catb_top_percent.values, y=catb_top_percent.index, palette="viridis")
plt.title(f"Top {top_n} CatBoost Feature Importance")
plt.xlabel("Importance (%)")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# ---------------------- ENSEMBLE PREDICTIONS ----------------------
xgb_val_proba = xgb_tuned.predict_proba(X_val)[:,1]
lgbm_val_proba = lgbm_tuned.predict_proba(X_val)[:,1]
catb_val_proba = catb_tuned.predict_proba(X_val)[:,1]


# Weighted average ensemble
y_val_ensemble = 0.34*xgb_val_proba + 0.33*lgbm_val_proba + 0.33*catb_val_proba

# Default threshold
threshold = 0.5
y_pred_ensemble = (y_val_ensemble >= threshold).astype(int)

# ---------------------- THRESHOLD TUNING FOR F1 ----------------------
# Threshold tuning using validation set
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_ensemble)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

# Tuned predictions
y_val_pred_tuned = (y_val_ensemble >= best_threshold).astype(int)

print("Best Threshold (validation):", best_threshold)
print("Best F1 Score (validation):", f1_scores[best_idx])



# ---------------------- ENSEMBLE PERFORMANCE ----------------------
print("Ensemble Metrics (Validation Set, Tuned Threshold):")
print("Accuracy:", accuracy_score(y_val, y_val_pred_tuned))
print("ROC-AUC:", roc_auc_score(y_val, y_val_ensemble))
print("F1:", f1_score(y_val, y_val_pred_tuned))
print("Recall:", recall_score(y_val, y_val_pred_tuned))
print("Precision:", precision_score(y_val, y_val_pred_tuned))
print("\nClassification Report:\n", classification_report(y_val, y_val_pred_tuned))


# ---------------------- CONFUSION MATRIX ----------------------
cm = confusion_matrix(y_val, y_val_pred_tuned)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
disp.plot(cmap='Blues')
plt.title("Ensemble Confusion Matrix")
plt.show()


# ---------------------- Test Set Predictions ----------------------
# Predict probabilities for the test set using each model
xgb_test_proba = xgb_tuned.predict_proba(df_test)[:,1]
lgbm_test_proba = lgbm_tuned.predict_proba(df_test)[:,1]
catb_test_proba = catb_tuned.predict_proba(df_test)[:,1]

# Combine predictions using weighted ensemble
y_test_ensemble = 0.34*xgb_test_proba + 0.33*lgbm_test_proba + 0.33*catb_test_proba

# Apply the best threshold found on the validation set
y_test_pred = (y_test_ensemble >= best_threshold).astype(int)

# Create submission CSV
submission = pd.DataFrame({
    "ID": df_test["ID"] if "ID" in df_test.columns else range(len(df_test)),
    "y": y_test_pred
})
submission.to_csv("submission.csv", index=False)
print("submission.csv has been created ✅")


