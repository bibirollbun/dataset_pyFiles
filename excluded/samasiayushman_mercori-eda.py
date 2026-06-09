import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/mercor-cheating-detection/train.csv")
train.head(3)


train.columns


train.dtypes


for col in train.columns:
    print(col)
    print(train[col].value_counts(dropna=False))
    print("")
    print("")    
    print("")


pd.crosstab(train['high_conf_clean'], train['is_cheating'], dropna=False)


train.describe()


for c in [col for col in train.columns if col.startswith("feature_")]:
    print(c, train[c].nunique())




labeled = train[train.is_cheating.notna()]

labeled[[c for c in train.columns if c.startswith("feature_")] + ["is_cheating"]].corr()["is_cheating"].sort_values()



missing = train.isna().mean().sort_values(ascending=False)
missing[missing > 0]
missing = missing*100


missing.to_frame("missing_pct").style.background_gradient(cmap="Reds")


import matplotlib.pyplot as plt

numeric_features = [c for c in train.columns if c.startswith("feature_")]

train[numeric_features].hist(
    bins=30,
    figsize=(18, 14),
    layout=(6, 3)
)
plt.tight_layout()
plt.show()


train[
    ((train.is_cheating == 0) | (train.is_cheating == 1)) &
    (train.high_conf_clean.notna())
].shape


train[train.is_cheating.notna() & train.high_conf_clean.notna()].shape


train[train.is_cheating.isna() & train.high_conf_clean.isna()].shape


train[train.high_conf_clean.notna()].is_cheating.isna().all()


test = pd.read_csv("/kaggle/input/mercor-cheating-detection/test.csv")

features = [c for c in train.columns if c.startswith("feature_")]

train_stats = train[features].describe().loc[["mean", "std"]]
test_stats  = test[features].describe().loc[["mean", "std"]]

(train_stats - test_stats).abs().mean(axis=1)


!pip install lightgbm --quiet 


import lightgbm as lgb

X = labeled[features]
y = labeled["is_cheating"]

model = lgb.LGBMClassifier(n_estimators=200)
model.fit(X, y)

pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)








