import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42



train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")



TARGET = "diagnosed_diabetes"
ID_COL = "id"

X = train.drop(columns=[TARGET, ID_COL])
y = train[TARGET]

X_test = test.drop(columns=[ID_COL])



cat_cols = X.select_dtypes(include="object").columns

for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]], axis=0)
    le.fit(combined)

    X[col] = le.transform(X[col])
    X_test[col] = le.transform(X_test[col])



drop_cols = [
    "heart_rate",
    "screen_time_hours_per_day",
    "alcohol_consumption_per_week",
    "diet_score"
]

X = X.drop(columns=drop_cols)
X_test = X_test.drop(columns=drop_cols)



num_cols = X.columns

for col in num_cols:
    X[col] = X[col].rank(pct=True)
    X_test[col] = X_test[col].rank(pct=True)



import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42

kf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Fold {fold + 1}")

    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = xgb.XGBClassifier(
        n_estimators=800,        # fixed trees
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.3,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_tr, y_tr)

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1]

test_preds /= kf.n_splits



oof_auc = roc_auc_score(y, oof_preds)
print(f"OOF ROC AUC: {oof_auc:.5f}")



for col in [
    "triglycerides",
    "cholesterol_total",
    "ldl_cholesterol",
    "hdl_cholesterol"
]:
    X[col] = np.log1p(X[col])
    X_test[col] = np.log1p(X_test[col])



from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=2000)
lr.fit(X_scaled, y)

lr_test_preds = lr.predict_proba(X_test_scaled)[:, 1]



final_preds = 0.7 * test_preds + 0.3 * lr_test_preds



import pandas as pd

test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")



test_df.head()



test_ids = test_df["id"]



submission = pd.DataFrame({
    "id": test_ids,                 # must match Kaggle exactly
    "diagnosed_diabetes": final_preds  # PROBABILITIES, not 0/1
})

submission.to_csv("submission.csv", index=False)



print(submission.shape)
print(submission.head())



xgb_model = xgb.XGBClassifier(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=20,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=3.0,
    gamma=0.5,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X, y)

test_preds = xgb_model.predict_proba(X_test)[:, 1]



submission = pd.DataFrame({
    "id": test_ids,
    "diagnosed_diabetes": test_preds
})

submission.to_csv("submission.csv", index=False)



X = df.drop(columns=["diagnosed_diabetes", "id"])
y = df["diagnosed_diabetes"].astype(int)

X_test = test_df.drop(columns=["id"])



import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score

import xgboost as xgb



import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score

import xgboost as xgb



train_df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_df  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

print(train_df.shape, test_df.shape)



X = train_df.drop(columns=["diagnosed_diabetes", "id"])
y = train_df["diagnosed_diabetes"].astype(int)

X_test = test_df.drop(columns=["id"])
test_ids = test_df["id"]



cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

X[cat_cols] = encoder.fit_transform(X[cat_cols])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])



log_cols = [
    "cholesterol_total",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides"
]

for col in log_cols:
    X[col] = np.log1p(X[col])
    X_test[col] = np.log1p(X_test[col])



model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=2.0,
    reg_lambda=5.0,
    gamma=0.5,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)



test_preds = model.predict_proba(X_test)[:, 1]



submission = pd.DataFrame({
    "id": test_ids,
    "diagnosed_diabetes": test_preds
})

submission.to_csv("submission1.csv", index=False)



print(submission.shape)
print(submission.head())
print(submission["diagnosed_diabetes"].min(),
      submission["diagnosed_diabetes"].max())



print(submission.shape)
print(submission.head())
print(submission["diagnosed_diabetes"].min(),
      submission["diagnosed_diabetes"].max())



import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
import lightgbm as lgb



train_df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_df  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

X = train_df.drop(columns=["diagnosed_diabetes", "id"])
y = train_df["diagnosed_diabetes"].astype(int)

X_test = test_df.drop(columns=["id"])
test_ids = test_df["id"]



cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X[cat_cols] = encoder.fit_transform(X[cat_cols])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])



log_cols = ["cholesterol_total", "hdl_cholesterol", "ldl_cholesterol", "triglycerides"]

for col in log_cols:
    X[col] = np.log1p(X[col])
    X_test[col] = np.log1p(X_test[col])



xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=2.0,
    reg_lambda=5.0,
    gamma=0.5,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X, y)
xgb_preds = xgb_model.predict_proba(X_test)[:, 1]




lgb_model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    num_leaves=15,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=2.0,
    reg_lambda=5.0,
    random_state=42,
    n_jobs=-1
)

lgb_model.fit(X, y)
lgb_preds = lgb_model.predict_proba(X_test)[:, 1]



final_preds = 0.5 * xgb_preds + 0.5 * lgb_preds



from sklearn.metrics import roc_auc_score, accuracy_score

# If you have a validation set, use it:
# val_preds = 0.5 * xgb_val_preds + 0.5 * lgb_val_preds
# y_val = your validation labels

# If not, we can evaluate on the training set (less realistic)
xgb_preds_train = xgb_model.predict_proba(X)[:, 1]
lgb_preds_train = lgb_model.predict_proba(X)[:, 1]

# Blend predictions
final_preds_train = 0.5 * xgb_preds_train + 0.5 * lgb_preds_train

# Convert probabilities to 0/1 using 0.5 threshold
final_class_train = (final_preds_train >= 0.5).astype(int)

# Metrics
roc_auc = roc_auc_score(y, final_preds_train)
accuracy = accuracy_score(y, final_class_train)

print(f"Training Accuracy: {accuracy:.5f}")
print(f"Training ROC AUC: {roc_auc:.5f}")



submission = pd.DataFrame({
    "id": test_ids,
    "diagnosed_diabetes": final_preds
})

submission.to_csv("submission5.csv", index=False)





