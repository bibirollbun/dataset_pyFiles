# 1. Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
#from sklearn.impute import KNNImputer
#from sklearn.impute import KNNImputer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Regressors
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Classifiers
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Evaluation
from sklearn.metrics import accuracy_score, f1_score, recall_score, mean_squared_log_error, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



# Load
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# Encode Target
le_target = LabelEncoder()
train['Personality'] = le_target.fit_transform(train['Personality'])  # Extrovert=1, Introvert=0

# Split features/target
X = train.drop(['id', 'Personality'], axis=1)
y = train['Personality']
X_test = test.drop(['id'], axis=1)

# Label Encode categorical
cat_cols = X.select_dtypes(include='object').columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# Impute missing
# imputer = SimpleImputer(strategy='mean')
#imputer = KNNImputer(n_neighbors=4)
rf = RandomForestRegressor(n_estimators=100)
imputer = IterativeImputer(estimator=rf, verbose=2, max_iter=5, tol=1e-10, imputation_order='roman')

X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Train/Val split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)



regressors = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

print("===== Regressor Results =====")
for name, model in regressors.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    preds_clipped = np.clip(np.round(preds), 0, 1).astype(int)

    acc = accuracy_score(y_val, preds_clipped)
    f1 = f1_score(y_val, preds_clipped)
    recall = recall_score(y_val, preds_clipped)
    rmsle = np.sqrt(mean_squared_log_error(y_val, np.clip(preds, 0, 1)))

    print(f"--- {name} ---")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f} | RMSLE: {rmsle:.4f}")



classifiers = {
    "Logistic": LogisticRegression(max_iter=1000),
    "RidgeClassifier": RidgeClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}

print("\n===== Classifier Results =====")
for name, model in classifiers.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    recall = recall_score(y_val, preds)

    print(f"--- {name} ---")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f}")



# Train best models
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
lgb = LGBMClassifier(random_state=42)

xgb.fit(X_train, y_train)
lgb.fit(X_train, y_train)

# Ensemble prediction using soft voting
proba_xgb = xgb.predict_proba(X_val)[:, 1]
proba_lgb = lgb.predict_proba(X_val)[:, 1]
ensemble_proba = (proba_xgb + proba_lgb) / 2
ensemble_pred = (ensemble_proba >= 0.5).astype(int)

# Evaluate ensemble
acc = accuracy_score(y_val, ensemble_pred)
f1 = f1_score(y_val, ensemble_pred)
recall = recall_score(y_val, ensemble_pred)

print("\n=== Ensemble (XGB + LGBM) ===")
print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f}")



# Predict on test
test_proba_xgb = xgb.predict_proba(X_test_scaled)[:, 1]
test_proba_lgb = lgb.predict_proba(X_test_scaled)[:, 1]
final_proba = (test_proba_xgb + test_proba_lgb) / 2
final_preds = (final_proba >= 0.5).astype(int)
final_labels = le_target.inverse_transform(final_preds)

submission = pd.DataFrame({'id': test['id'], 'Personality': final_labels})
submission.to_csv('submission.csv', index=False)
submission.head()





