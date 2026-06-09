import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)


X = train.drop(columns=["id", "loan_paid_back"])
y = train["loan_paid_back"]
X_test = test.drop(columns=["id"])


X = X.fillna(-999)
X_test = X_test.fillna(-999)



def grade_to_num(g):
    if pd.isna(g): return -1
    letter = g[0]
    number = int(g[1])
    base = (ord(letter) - ord('A')) * 5
    return base + number


X["grade_subgrade"] = X["grade_subgrade"].apply(grade_to_num)
X_test["grade_subgrade"] = X_test["grade_subgrade"].apply(grade_to_num)

# Label encode categorical columns
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))


X["income_to_loan_ratio"] = X["annual_income"] / (X["loan_amount"] + 1)
X_test["income_to_loan_ratio"] = X_test["annual_income"] / (X_test["loan_amount"] + 1)

X["loan_to_income_ratio"] = X["loan_amount"] / (X["annual_income"] + 1)
X_test["loan_to_income_ratio"] = X_test["loan_amount"] / (X_test["annual_income"] + 1)

X["credit_to_interest_ratio"] = X["credit_score"] / (X["interest_rate"] + 1)
X_test["credit_to_interest_ratio"] = X_test["credit_score"] / (X_test["interest_rate"] + 1)


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

test_pred = np.zeros(len(X_test))
val_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f" Fold {fold + 1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=5000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(period=200)
        ]
    )

    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_val, val_pred)
    val_scores.append(auc)
    print(f"Fold {fold + 1} AUC: {auc:.4f}")

    #accumulate test predictions
    test_pred += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits

print("\n Average AUC across folds:", np.mean(val_scores))


print("Average AUC across folds:", np.mean(val_scores))


submission = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": test_pred
})

submission.to_csv("submission.csv", index=False)
print("submission.csv saved!")
submission.head()




