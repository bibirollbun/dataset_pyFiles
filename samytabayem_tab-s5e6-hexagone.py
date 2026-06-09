import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb


df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


FEATURES = [col for col in df_test.columns if col != "id"]
CAT_COLS = ["Soil Type", "Crop Type"]
NUM_COLS = [col for col in FEATURES if col not in CAT_COLS]
TARGET = "Fertilizer Name"



enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df_train[CAT_COLS] = enc.fit_transform(df_train[CAT_COLS])
df_test[CAT_COLS] = enc.transform(df_test[CAT_COLS])


imp = SimpleImputer(strategy="median")
df_train[NUM_COLS] = imp.fit_transform(df_train[NUM_COLS])
df_test[NUM_COLS] = imp.transform(df_test[NUM_COLS])

scaler = RobustScaler()
df_train[NUM_COLS] = scaler.fit_transform(df_train[NUM_COLS])
df_test[NUM_COLS] = scaler.transform(df_test[NUM_COLS])


X = df_train[FEATURES]
y = df_train[TARGET]
X_test = df_test[FEATURES]



cb = CatBoostClassifier(verbose=0, random_state=42)
lgbm = LGBMClassifier(random_state=42)
xgbm = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

estimators = [
    ('catboost', cb),
    ('lgbm', lgbm),
    ('xgb', xgbm)
]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1
)


cb = CatBoostClassifier(verbose=0, random_state=42)
lgbm = LGBMClassifier(random_state=42)
xgbm = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)



estimators = [
    ('catboost', cb),
    ('lgbm', lgbm),
    ('xgb', xgbm)
]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1
)


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_scores = []
f1_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    stack.fit(X_train, y_train)
    val_preds = stack.predict(X_val)
    
    acc = accuracy_score(y_val, val_preds)
    f1 = f1_score(y_val, val_preds, average="weighted")
    
    acc_scores.append(acc)
    f1_scores.append(f1)
    
    print(f"Fold {fold+1} Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")

print(f"\nğŸ“Š Moyenne des scores :")
print(f"âœ”ï¸� Accuracy moyenne: {np.mean(acc_scores):.4f}")
print(f"âœ”ï¸� F1-score moyenne (weighted): {np.mean(f1_scores):.4f}")



stack.fit(X, y)
final_preds = stack.predict(X_test)

# 9. GÃ©nÃ©ration du fichier de soumission
sub[TARGET] = final_preds
sub.to_csv("submission.csv", index=False)
print("\nâœ… Fichier 'submission.csv' gÃ©nÃ©rÃ© avec succÃ¨s.")




