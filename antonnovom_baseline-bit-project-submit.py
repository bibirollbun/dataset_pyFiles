import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from catboost import CatBoostClassifier, Pool

DATA_PATH = "/kaggle/input/playground-series-s4e8/"
train = pd.read_csv(DATA_PATH + "train.csv")
test  = pd.read_csv(DATA_PATH + "test.csv")

y = (train["class"] == "p").astype(int)   # p=1, e=0
X = train.drop(columns=["class"])
X_test = test.copy()

def basic_clean(df):
    df = df.copy()
    # часто встречается '?': считаем пропуском
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].replace("?", np.nan).fillna("Missing")
    # числовые пропуски -> медиана
    num_cols = df.select_dtypes(exclude=["object"]).columns
    for c in num_cols:
        med = df[c].median()
        df[c] = df[c].fillna(med)
    return df

X = basic_clean(X)
X_test = basic_clean(X_test)

def add_freq_enc(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()
    cat_cols = train_df.select_dtypes(include=["object"]).columns
    for c in cat_cols:
        vc = train_df[c].value_counts(dropna=False)
        train_df[c+"_freq"] = train_df[c].map(vc).fillna(0).astype(np.float32)
        test_df[c+"_freq"]  = test_df[c].map(vc).fillna(0).astype(np.float32)
    return train_df, test_df

X, X_test = add_freq_enc(X, X_test)

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

cat_cols = X_tr.select_dtypes(include=["object"]).columns.tolist()
cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols]

train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
val_pool   = Pool(X_val, y_val, cat_features=cat_idx)

model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.06,
    depth=8,
    loss_function="Logloss",
    eval_metric="Logloss",
    random_seed=42,
    l2_leaf_reg=6.0,
    od_type="Iter",
    od_wait=80,
    task_type="GPU"  
)

model.fit(train_pool, eval_set=val_pool, verbose=200)

val_proba = model.predict_proba(X_val)[:, 1]
ths = np.linspace(0.02, 0.98, 97)
mccs = [matthews_corrcoef(y_val, (val_proba >= t).astype(int)) for t in ths]
best_t = float(ths[int(np.argmax(mccs))])
best_mcc = float(np.max(mccs))
print("Best threshold:", best_t, "Val MCC:", best_mcc)

test_proba = model.predict_proba(X_test)[:, 1]
test_pred = np.where(test_proba >= best_t, "p", "e")
sub = pd.DataFrame({"id": test["id"], "class": test_pred})
sub.to_csv("submission_baseline.csv", index=False)
print(sub.head())


