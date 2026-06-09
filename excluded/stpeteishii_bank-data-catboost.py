


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from catboost import CatBoostClassifier, Pool


# --- 1. Load your dataset ---
df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")  

# --- 2. Separate target and features ---
target_col = "y"  
y = df[target_col]
X = df.drop(columns=[target_col])

# --- 3. Identify categorical columns (CatBoost can handle raw strings directly) ---
cat_cols = X.select_dtypes(include="object").columns.tolist()


# --- 4. Split into training and validation sets ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 5. Create CatBoost Pools with categorical feature info ---
train_pool = Pool(data=X_train, label=y_train, cat_features=cat_cols)
val_pool = Pool(data=X_val, label=y_val, cat_features=cat_cols)

# --- 6. Define and train the CatBoost model ---
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    cat_features=cat_cols,
    verbose=100,
    early_stopping_rounds=50
)

model.fit(train_pool, eval_set=val_pool)


# --- 7. Evaluation ---
y_pred_proba = model.predict_proba(X_val)[:, 1]
y_pred = model.predict(X_val)

print("ROC AUC:", roc_auc_score(y_val, y_pred_proba))
print(classification_report(y_val, y_pred))


# --- 8. Optional: Predict on test data ---
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
test_pred_proba = model.predict_proba(test_df)[:, 1]


submit=pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
submit['y']=test_pred_proba
submit.to_csv('submission.csv',index=False)
display(submit)




