import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
target = train.columns[-1]

def create_frequency_features(train_df, test_df, cols, num, cat):
    train, test = train_df.copy(), test_df.copy()

    for col in cols:
        freq = train[col].value_counts(normalize=True)
        train[f"{col}_freq"] = train[col].map(freq)
        test[f"{col}_freq"] = test[col].map(freq).fillna(train[f"{col}_freq"].mean())

        if col in num:
            for q in [5, 10, 15]:
                try:
                    train[f"{col}_bin{q}"], bins = pd.qcut(train[col], q=q, labels=False, retbins=True, duplicates="drop")
                    test[f"{col}_bin{q}"] = pd.cut(test[col], bins=bins, labels=False, include_lowest=True)
                except Exception:
                    train[f"{col}_bin{q}"] = test[f"{col}_bin{q}"] = 0

    new_num = train.drop(columns=cat + [target]).columns.tolist()
    return train, test

cols = train.drop(columns=target).columns.tolist()
cat = [col for col in cols if train[col].dtype in ["object", "category"] and col != target]
num = [col for col in cols if train[col].dtype not in ["object", "category", "bool"] and col not in ["id", target]]

train, test = create_frequency_features(train, test.copy(), cols, num, cat)
train[cat], test[cat] = train[cat].astype("category"), test[cat].astype("category")
train.drop(columns="id", inplace=True)
train.drop_duplicates(inplace=True)

dtrain = xgb.DMatrix(
    train.drop(columns=target),
    label=train[target],
    enable_categorical=True
)

params = {
    'tree_method': 'hist',
    'device': 'cuda',
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'random_state': 42,
    'max_depth': 4,
    'scale_pos_weight': 1
}

cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    nfold=5,
    num_boost_round=2000,
    metrics='auc',
    verbose_eval=False,
    early_stopping_rounds=50
)

best_round = cv_results['test-auc-mean'].idxmax()
best_auc = cv_results['test-auc-mean'][best_round]

model = XGBClassifier(**params, enable_categorical=True, n_estimators=best_round)
model.fit(train.drop(columns=target), train[target])

pred = model.predict_proba(test.drop(columns="id"))[:, 1]

sub = pd.DataFrame({
    "id": test["id"],
    target: pred
})
sub.to_csv("submission.csv", index=False)


