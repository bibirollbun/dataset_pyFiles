import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("default")
sns.set_theme()

TRAIN_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
TEST_PATH  = "/kaggle/input/playground-series-s5e12/test.csv"

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

print("Train shape:", train.shape)
print("Test shape :", test.shape)

train.head()


# Column names
print(train.columns.tolist())

# Info: dtypes, nulls, etc.
train.info()

# Basic statistics for numeric columns
train.describe().T


# Columns not present in test -> likely ['id', target]
extra_cols = sorted(set(train.columns) - set(test.columns))
print("Columns only in train:", extra_cols)


ID_COL = "id"          # change if your id column has another name
TARGET = [c for c in extra_cols if c != ID_COL][0]  # pick the non-id col

print("ID_COL:", ID_COL)
print("TARGET:", TARGET)


print(train[TARGET].value_counts())
print("\nProportion:")
print(train[TARGET].value_counts(normalize=True))

plt.figure(figsize=(4,3))
sns.countplot(data=train, x=TARGET)
plt.title("Target distribution")
plt.show()


feature_cols = [c for c in train.columns if c not in [ID_COL, TARGET]]

num_cols = train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in feature_cols if c not in num_cols]

print("Numeric features    :", num_cols)
print("Categorical features:", cat_cols)


if num_cols:
    grouped_means = train.groupby(TARGET)[num_cols].mean().T
    display(grouped_means.sort_index())

    # Optional: show features where difference between classes is largest
    # (works best if TARGET is binary 0/1; adapt if it has 0/1/2 etc.)
    if train[TARGET].nunique() == 2:
        diff = grouped_means.iloc[:, 1] - grouped_means.iloc[:, 0]
        diff.name = "mean_diff"
        display(diff.sort_values(ascending=False))


# Pick up to 4 interesting numeric columns manually
cols_to_plot = num_cols[:4]  # replace with specific names once you see columns

for col in cols_to_plot:
    plt.figure(figsize=(5,3))
    sns.kdeplot(data=train, x=col, hue=TARGET, common_norm=False)
    plt.title(f"{col} distribution by {TARGET}")
    plt.tight_layout()
    plt.show()


def plot_categorical_relation(df, col, target):
    ct = pd.crosstab(df[col], df[target], normalize="index") * 100
    print(f"\n=== {col} vs {target} (%) ===")
    display(ct.round(1))

    ct.plot(kind="bar", stacked=True)
    plt.ylabel("Row %")
    plt.title(f"{col} vs {target}")
    plt.legend(title=target)
    plt.tight_layout()
    plt.show()

# Treat small-cardinality columns as categorical
small_card_cols = [c for c in feature_cols if train[c].nunique() <= 10]

for col in small_card_cols[:8]:   # first few; can expand
    plot_categorical_relation(train, col, TARGET)


corr = train[num_cols + [TARGET]].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
plt.title("Correlation heatmap")
plt.tight_layout()
plt.show()

# Correlation of each feature with target
target_corr = corr[TARGET].drop(TARGET).sort_values(ascending=False)
print("Correlation with target:")
display(target_corr)





from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

ID_COL = "id"
TARGET = "diagnosed_diabetes"

feature_cols = [c for c in train.columns if c not in [ID_COL, TARGET]]

# 1. Split numeric / categorical
num_cols = train[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = [c for c in feature_cols if c not in num_cols]

print("Numeric:", num_cols)
print("Categorical:", cat_cols)

# 2. One-hot encode categoricals
X = train[feature_cols].copy()
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)  # drop_first to avoid dummy trap
y = train[TARGET].astype(int)

print("X shape after encoding:", X.shape)

# 3. Train/valid split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. RandomForest baseline
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 5. Evaluate (binary case)
proba_valid = model.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(y_valid, proba_valid)
print("Validation AUC:", auc)






import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

TRAIN_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
TEST_PATH  = "/kaggle/input/playground-series-s5e12/test.csv"
SUB_PATH   = "/kaggle/input/playground-series-s5e12/sample_submission.csv"

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

ID_COL = "id"
TARGET = "diagnosed_diabetes"

feature_cols = [c for c in train.columns if c not in [ID_COL, TARGET]]

# Split numeric / categorical
num_cols = train[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = [c for c in feature_cols if c not in num_cols]

print("Numeric features    :", num_cols)
print("Categorical features:", cat_cols)

# --- Make a shared category mapping train+test, then convert to int codes ---
for col in cat_cols:
    # combine then create categories
    all_vals = pd.concat([train[col], test[col]], axis=0).astype("category")
    cats = all_vals.cat.categories

    train[col] = pd.Categorical(train[col], categories=cats).codes
    test[col]  = pd.Categorical(test[col],  categories=cats).codes

# X, y
X = train[feature_cols]
y = train[TARGET].astype(int)
X_test = test[feature_cols]



# train/valid split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

lgbm = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=64,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

lgbm.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc",
    categorical_feature=cat_cols,  # these are the int-coded categorials
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(100),
    ],
)

proba_valid = lgbm.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(y_valid, proba_valid)
print(f"\nValidation AUC: {auc:.5f}")



booster = lgbm.booster_
importance_gain = booster.feature_importance(importance_type="gain")
feature_names = booster.feature_name()

fi_df = pd.DataFrame({
    "feature": feature_names,
    "importance_gain": importance_gain,
}).sort_values("importance_gain", ascending=False)

print("\nTop 20 features by gain:")
display(fi_df.head(20))

# Plot top 30
top_n = 30
plt.figure(figsize=(8, 10))
sns.barplot(data=fi_df.head(top_n), x="importance_gain", y="feature")
plt.title(f"LightGBM Feature Importance (gain, top {top_n})")
plt.tight_layout()
plt.show()



# use best_iteration_ from validation model if available
best_iter = lgbm.best_iteration_ if lgbm.best_iteration_ is not None else lgbm.n_estimators
print("Using n_estimators =", best_iter)

lgbm_full = LGBMClassifier(
    n_estimators=best_iter,
    learning_rate=0.03,
    num_leaves=64,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

lgbm_full.fit(
    X,
    y,
    categorical_feature=cat_cols,
)

test_pred = lgbm_full.predict_proba(X_test)[:, 1]

sub = pd.read_csv(SUB_PATH)
sub["diagnosed_diabetes"] = test_pred
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv")
sub.head()








