# If you're running this outside Kaggle, set DATA_PATH accordingly.
# On Kaggle, the competition data lives here:
DATA_PATH = "/kaggle/input/spaceship-titanic"

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Reproducibility
SEED = 42
np.random.seed(SEED)

pd.set_option("display.max_columns", 200)

# Optional imports for visuals
import matplotlib.pyplot as plt



train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))

print(train.shape, test.shape)
train.head()


# Target balance
ax = train["Transported"].value_counts(normalize=True).sort_index().plot(kind="bar")
ax.set_title("Target balance: Transported")
ax.set_xlabel("Transported")
ax.set_ylabel("Proportion")
plt.show()

# Missingness
missing = train.isna().mean().sort_values(ascending=False)
display(missing[missing > 0].to_frame("missing_rate").style.format("{:.1%}"))


SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # PassengerId â†’ group features
    grp = df["PassengerId"].str.split("_", expand=True)
    df["Group"] = pd.to_numeric(grp[0], errors="coerce")
    df["PassengerNumber"] = pd.to_numeric(grp[1], errors="coerce")

    # Cabin split
    cabin = df["Cabin"].astype(str).str.split("/", expand=True)
    df["Deck"] = cabin[0].where(df["Cabin"].notna(), "Unknown")
    df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    df["Side"] = cabin[2].where(df["Cabin"].notna(), "Unknown")

    # Name features
    df["Name"] = df["Name"].fillna("Unknown")
    df["Surname"] = df["Name"].str.split().str[-1]

    # Spending
    for c in SPEND_COLS:
        df[c] = df[c].fillna(0)

    df["TotalSpend"] = df[SPEND_COLS].sum(axis=1)
    df["NumServicesUsed"] = (df[SPEND_COLS] > 0).sum(axis=1)
    df["HasSpend"] = (df["TotalSpend"] > 0).astype(int)

    for c in SPEND_COLS + ["TotalSpend"]:
        df[f"log1p_{c}"] = np.log1p(df[c])

    # Age
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["IsChild"] = (df["Age"] < 13).astype(int)
    df["AgeBin"] = pd.cut(
        df["Age"],
        bins=[-1, 12, 18, 30, 45, 60, 200],
        labels=["child", "teen", "young", "adult", "mid", "senior"],
    ).astype(str)

    # Simple, safe categorical fills
    for c in ["HomePlanet", "Destination", "VIP", "CryoSleep", "Deck", "Side", "Surname", "AgeBin"]:
        df[c] = df[c].fillna("Unknown").astype(str)

    # CryoSleep consistency: sleeping passengers shouldn't spend
    sleep_mask = df["CryoSleep"].isin(["True", "1", "true", "TRUE"])
    df.loc[sleep_mask, SPEND_COLS + ["TotalSpend", "NumServicesUsed", "HasSpend"]] = 0
    for c in SPEND_COLS + ["TotalSpend"]:
        df.loc[sleep_mask, f"log1p_{c}"] = 0

    return df

# Build combined frame so group-size is consistent across train+test
all_df = pd.concat([train.drop(columns=["Transported"]), test], ignore_index=True)
all_df = add_features(all_df)

# Group size & "alone" feature (computed on combined data to avoid train/test mismatch)
all_df["GroupSize"] = all_df.groupby("Group")["PassengerId"].transform("count")
all_df["IsAlone"] = (all_df["GroupSize"] == 1).astype(int)

train_fe = all_df.iloc[: len(train)].copy()
test_fe  = all_df.iloc[len(train):].copy()

y = train["Transported"].astype(int).values

# Drop raw identifiers (keep engineered parts)
DROP_COLS = ["PassengerId", "Name", "Cabin"]
X = train_fe.drop(columns=DROP_COLS)
X_test = test_fe.drop(columns=DROP_COLS)

cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if X[c].dtype != "object"]

print("X shape:", X.shape)
print("Categorical cols:", len(cat_cols))
print("Numeric cols:", len(num_cols))


# Optional: install CatBoost if not present (Kaggle usually has it already)
try:
    from catboost import CatBoostClassifier, Pool
except ImportError:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost"])
    from catboost import CatBoostClassifier, Pool

N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof = np.zeros(len(X), dtype=float)
test_pred = np.zeros(len(X_test), dtype=float)

params = dict(
    loss_function="Logloss",
    eval_metric="Accuracy",
    iterations=5000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3,
    random_seed=SEED,
    od_type="Iter",
    od_wait=200,
    verbose=200,
)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_va, y_va = X.iloc[va_idx], y[va_idx]

    tr_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
    va_pool = Pool(X_va, y_va, cat_features=cat_cols)
    te_pool = Pool(X_test, cat_features=cat_cols)

    model = CatBoostClassifier(**params)
    model.fit(tr_pool, eval_set=va_pool, use_best_model=True)

    oof[va_idx] = model.predict_proba(va_pool)[:, 1]
    test_pred += model.predict_proba(te_pool)[:, 1] / N_SPLITS

# CV accuracy at default threshold
acc_05 = accuracy_score(y, (oof >= 0.5).astype(int))
print("OOF accuracy @0.50:", acc_05)


thresholds = np.linspace(0.25, 0.75, 101)
accs = [accuracy_score(y, (oof >= t).astype(int)) for t in thresholds]
best_i = int(np.argmax(accs))
best_t = thresholds[best_i]
best_acc = accs[best_i]

print("Best threshold:", best_t)
print("Best OOF accuracy:", best_acc)

plt.plot(thresholds, accs)
plt.axvline(best_t, linestyle="--")
plt.title("Threshold vs OOF accuracy")
plt.xlabel("Threshold")
plt.ylabel("Accuracy")
plt.show()


submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Transported": (test_pred >= best_t)
})
submission.to_csv("submission.csv", index=False)
submission.head()

