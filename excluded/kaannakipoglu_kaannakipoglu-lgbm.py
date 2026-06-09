# Cell 1 ───────────────────────────────────────────────
import os, warnings, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

DATA_DIR = "/kaggle/input/playground-series-s5e6"
RNG_SEED = 42



# Cell 2 ───────────────────────────────────────────────
train_df   = pd.read_csv(f"{DATA_DIR}/train.csv")
test_df    = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print("Train:", train_df.shape, "  Test:", test_df.shape)
train_df.head()



# Cell 3 ───────────────────────────────────────────────
def tidy(df):
    df.columns = (df.columns.str.strip()
                            .str.replace(" ", "_")
                            .str.replace("-", "_"))
    return df

train_df = tidy(train_df)
test_df  = tidy(test_df)

TARGET = "Fertilizer_Name"
ID_COL  = "id"



# Cell 4 ───────────────────────────────────────────────
cat_cols = [c for c in train_df.columns
            if train_df[c].dtype == "object" and c not in (TARGET, ID_COL)]

for col in cat_cols:
    train_df[col] = train_df[col].astype("category")
    test_df[col]  = test_df[col].astype("category")

label_map = {lbl: i for i, lbl in enumerate(sorted(train_df[TARGET].unique()))}
inv_map   = {i: lbl for lbl, i in label_map.items()}
train_df["label_int"] = train_df[TARGET].map(label_map)

features = [c for c in train_df.columns if c not in (TARGET, "label_int", ID_COL)]

X_train, X_val, y_train_int, y_val_int = train_test_split(
    train_df[features], train_df["label_int"],
    test_size=0.20, stratify=train_df["label_int"], random_state=RNG_SEED)

print("Categorical columns:", cat_cols)
print("Classes:", label_map)



# Cell 5 ───────────────────────────────────────────────
train_set = lgb.Dataset(X_train, y_train_int, categorical_feature=cat_cols)
val_set   = lgb.Dataset(X_val,   y_val_int,   categorical_feature=cat_cols,
                        reference=train_set)

params = dict(
    objective        = "multiclass",
    num_class        = len(label_map),
    metric           = "multi_logloss",
    learning_rate    = 0.05,
    num_leaves       = 128,
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq     = 1,
    seed             = RNG_SEED,
)

model = lgb.train(
    params, train_set,
    num_boost_round       = 4000,
    valid_sets            = [val_set],
    callbacks             = [lgb.early_stopping(200), lgb.log_evaluation(250)]
)

def mapk(actual, prob, k=3):
    topk = np.argsort(prob, axis=1)[:, -k:][:, ::-1]
    score = 0.0
    for a, preds in zip(actual, topk):
        try:
            rank = np.where(preds == a)[0][0] + 1
            score += 1.0 / rank
        except IndexError:
            pass
    return score / len(actual)

val_prob = model.predict(X_val, num_iteration=model.best_iteration)
map3     = mapk(y_val_int.values, val_prob, k=3)
top1_acc = accuracy_score(y_val_int, val_prob.argmax(axis=1))

print(f"\n Validation MAP@3: {map3:.4f}   (Top-1 Accuracy: {top1_acc:.4f})\n")
print(classification_report(y_val_int.map(inv_map),
                            pd.Series(val_prob.argmax(axis=1)).map(inv_map)))



# Cell 6 ───────────────────────────────────────────────
lgb.plot_importance(model, max_num_features=20, figsize=(8,5))
plt.tight_layout(); plt.show()



# Cell 7 ───────────────────────────────────────────────
SUB_COL = "Fertilizer Name"          

full_set = lgb.Dataset(train_df[features], train_df["label_int"],
                       categorical_feature=cat_cols)

final_model = lgb.train(
    params, full_set,
    num_boost_round = model.best_iteration + 50
)

test_prob = final_model.predict(test_df[features])
top3_idx  = np.argsort(test_prob, axis=1)[:, -3:][:, ::-1]

submission_labels = (
    pd.DataFrame(top3_idx)
      .replace(inv_map)
      .agg(" ".join, axis=1)
)

sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")   
sub[SUB_COL] = submission_labels

sub.to_csv("/kaggle/working/submission.csv", index=False)


