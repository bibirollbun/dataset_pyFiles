# !pip -q install catboost

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest

from catboost import CatBoostClassifier, Pool

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 120)



train = pd.read_csv("/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/train.csv")
test  = pd.read_csv("/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/test.csv")

TO_DROP = ['id', 'Exited', 'CustomerId', 'Surname']

y = train["Exited"]
X0 = train.drop(columns=TO_DROP)


train.isna().sum() / len(train)


train['Exited'].value_counts()


print(y.value_counts(normalize=True))


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = train
target = "Exited"

df = df.drop(columns=["CustomerId", "Surname"])
categoricals = ["Geography", "Gender"]
df_enc = pd.get_dummies(df, columns=categoricals, drop_first=True)

corr = df_enc.corr(method="pearson")

plt.figure(figsize=(14, 12))
sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": .75}
)
plt.title("Correlation heatmap (Pearson)")
plt.tight_layout()
plt.show()

corr_target = (
    corr[target]
    .drop(target)
    .sort_values(ascending=False)
)

plt.figure(figsize=(6, 8))
sns.barplot(
    x=corr_target.values,
    y=corr_target.index,
    orient="h"
)
plt.axvline(0, color="k", linewidth=0.8)
plt.title("Feature ↔ Exited correlation")
plt.tight_layout()
plt.show()



print(corr_target)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def churn_by_bins(
        df: pd.DataFrame,
        feature: str,
        bins='quantile',
        q=4,
        custom_bins=None,
        target='Exited',
        plot=True
    ) -> pd.DataFrame:
    """
    Возвращает DataFrame со статистикой churn-rate по бинам
    и рисует barplot.
    """
    if bins == 'quantile':
        labels = [f'Q{i+1}' for i in range(q)]
        binned = pd.qcut(df[feature], q=q, labels=labels, duplicates='drop')
    elif bins == 'uniform':
        labels = [f'bin{i+1}' for i in range(q)]
        binned = pd.cut(df[feature], bins=q, labels=labels)
    elif bins == 'custom':
        if custom_bins is None:
            raise ValueError("custom_bins must be provided when bins='custom'")
        binned = pd.cut(df[feature], bins=custom_bins, right=False)
    else:
        raise ValueError("bins must be 'quantile', 'uniform' or 'custom'")

    base_rate = df[target].mean()
    stats = (
        df
        .assign(_bin=binned)
        .groupby('_bin')[target]
        .agg(['count', 'mean'])
        .rename(columns={'mean': 'churn_rate'})
    )
    stats['lift'] = stats['churn_rate'] / base_rate

    if plot:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(
            x=stats.index.astype(str),
            y='churn_rate',
            data=stats.reset_index(),
            ax=ax
        )
        ax.set_title(f'{feature}: churn-rate by bin')
        ax.set_ylabel('Churn rate')
        ax.set_xlabel(feature + ' bins')
        ax.axhline(base_rate, ls='--', lw=1, c='k', label='base rate')
        ax.legend()
        plt.tight_layout()
        plt.show()

    return stats

def target_lift(
        df: pd.DataFrame,
        features,
        target='Exited',
        top_n=15
    ) -> pd.DataFrame:
    """
    Считает lift target по уникальным значениям (или комбинациям) features.
    
    lift = (segment churn rate) / (global churn rate)
    """
    if isinstance(features, str):
        features = [features]

    base_rate = df[target].mean()

    grp = (
        df
        .groupby(features)[target]
        .agg(['count', 'mean'])
        .rename(columns={'mean': 'churn_rate'})
        .reset_index()
    )
    grp['lift'] = grp['churn_rate'] / base_rate
    grp['abs_distance'] = (grp['lift'] - 1).abs()

    return (
        grp
        .sort_values('abs_distance', ascending=False)
        .drop(columns='abs_distance')
        .head(top_n)
    )

age_stats = churn_by_bins(df, 'Age', bins='quantile', q=4)
bal_stats = churn_by_bins(df, 'Balance',
                            bins='custom',
                            custom_bins=[0, 1e4, 5e4, 1e5, 2e5, np.inf])

print(target_lift(df, 'Geography').head())
print(target_lift(df, ['Geography', 'Age']).head())
print(target_lift(df, ['Gender', 'Age']).head())
print(target_lift(df, ['Geography', 'Gender', 'Age']).head())





from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

def add_lift_based_flags(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # базовые
    d["AgeBucket"] = pd.cut(
        d["Age"], [0, 40, 50, 60, 70, np.inf],
        labels=["<40", "40_50", "50_60", "60_70", "70+"]
    )

    # 1) Германия 40-60
    d["GER_40_60"] = (
        (d["Geography"] == "Germany") &
        (d["AgeBucket"].isin(["40_50", "50_60"]))
    ).astype(int)

    # 2) «Европа (не DE) 50-60»
    d["EU_50_60_highrisk"] = (
        (d["Geography"].isin(["France", "Spain"])) &
        (d["AgeBucket"] == "50_60")
    ).astype(int)

    # 3) Пол-возрастные
    d["AGE50_60_Female"] = (
        (d["Gender"] == "Female") & (d["AgeBucket"] == "50_60")
    ).astype(int)
    d["AGE50_60_Male"] = (
        (d["Gender"] == "Male") & (d["AgeBucket"] == "50_60")
    ).astype(int)

    # 4) «Хорошие» <40 мужчины
    d["YoungMaleSafe"] = (
        (d["Gender"] == "Male") & (d["AgeBucket"] == "<40")
    ).astype(int)

    return d


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # очистка ID
    d.drop(columns=[c for c in ("CustomerId", "Surname", "id") if c in d],
           errors="ignore", inplace=True)

    # существующие числовые / бинарные
    d["Balance_to_Salary"] = d["Balance"] / d["EstimatedSalary"].clip(lower=1)
    d["bit_prod"] = (d["NumOfProducts"] < 3.0).astype(int)
    d["NumOfProducts"] = d["NumOfProducts"].map(str)
    # Age buckets по новым наблюдениям
    age_bins  = [0, 40, 50, 60, 70, np.inf]
    age_lbls  = ["<40", "40-50", "50-60", "60-70", "70+"]

    # бинарный флаг 50+
    # d["Age50plus"] = (d["Age"] >= 50).astype(int)

    d = add_lift_based_flags(d)

    # комбинированные категории
    d["GeoSex"]    = d["Geography"] + "_" + d["Gender"]
    d["AgeGeo"]    = d["AgeBucket"].astype(str) + "_" + d["Geography"]
    d["AgeGender"] = d["AgeBucket"].astype(str) + "_" + d["Gender"]
    return d


lift_cat_features = [
    'GER_40_60',
    'EU_50_60_highrisk',
    'AGE50_60_Female',
    'AGE50_60_Male',
    'YoungMaleSafe',
    "bit_prod"
]
cat_cols = [
    "Geography", "Gender",
    "GeoSex",
    "AgeBucket", "AgeGeo", "AgeGender",
    "NumOfProducts"
]

cat_idx = [add_features(train.iloc[[0]]).columns.get_loc(c) for c in cat_cols]


features = add_features(train)
lift = target_lift(features, ['Geography','AgeBucket']).sort_values('lift', ascending=False).head(10)
lift


lift = target_lift(features, ['Geography','AgeBucket', 'Gender']).sort_values('lift', ascending=False).head(10)
lift


lift = target_lift(features, ['AgeBucket','Gender']).sort_values('lift', ascending=False).head(10)
lift


lift = target_lift(features, ['Geography', 'NumOfProducts']).sort_values('lift', ascending=False).head(10)
lift


lift = target_lift(features, ['Gender', 'NumOfProducts']).sort_values('lift', ascending=False).head(10)
lift





features.head()


X_full = add_features(X0)

num_cols = X_full.select_dtypes(include=["number"]).columns

iso = IsolationForest(contamination=0.02, random_state=42)
mask_inlier = iso.fit_predict(X_full[num_cols]) == 1
print(f"Outliers removed: {(~mask_inlier).sum()} "
      f"({(~mask_inlier).mean():.1%} of train)")

X = X0.loc[mask_inlier].reset_index(drop=True)
y = y.loc[mask_inlier].reset_index(drop=True)

tmp = add_features(X.iloc[[0]])


from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

feat_eng = FunctionTransformer(add_features, validate=False)

params = dict(iterations=1000, learning_rate=0.02, depth=5,
         l2_leaf_reg=7, border_count=128, bagging_temperature=7,
         bootstrap_type="Bayesian", loss_function="Logloss", eval_metric="AUC",
         random_seed=42, verbose=False, early_stopping_rounds=200, cat_features=cat_cols)

pipe = Pipeline(steps=[
    ("fe",  feat_eng),
    ("cat", CatBoostClassifier(**params))
])

param_dist = {
    "cat__iterations":      [1000],
    "cat__learning_rate":   [0.02],
    "cat__depth":           [7],
    "cat__l2_leaf_reg":     [7],
    "cat__bagging_temperature": [7],   
    "cat__border_count":    [128]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=1,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    refit=True,
    verbose=2,
    random_state=42
)

search.fit(X, y)
print("\nBest CV ROC-AUC:", search.best_score_)
print("Best params:", search.best_params_)


best_model = search.best_estimator_ 
best_model.fit(X, y)

test_fe = test.drop(columns=[c for c in ("id", "CustomerId", "Surname") if c in test], errors="ignore")
pred_test = best_model.predict_proba(test_fe)[:, 1]

submission = pd.DataFrame({
    "id": test["id"],
    "Exited": pred_test
})
submission.to_csv("submission.csv", index=False)
submission.head()





# from sklearn.isotonic import IsotonicRegression

# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# oof_pred = np.zeros(len(X))

# for tr_idx, val_idx in skf.split(X, y):
#     best_model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
#     oof_pred[val_idx] = best_model.predict_proba(X.iloc[val_idx])[:, 1]

# print("OOF AUC (до калибровки):",
#       roc_auc_score(y, oof_pred).round(4))

# iso = IsotonicRegression(out_of_bounds="clip")
# iso.fit(oof_pred, y)

# best_model.fit(X, y)

# raw_test = best_model.predict_proba(
#     test.drop(['id', 'CustomerId', 'Surname'], axis=1)
# )[:, 1]

# cal_test = iso.transform(raw_test)

# submission = pd.DataFrame({
#     'id': test['id'],
#     'Exited': cal_test
# })
# submission['id'] = submission['id'].astype(int)
# submission.to_csv('submission_calibrated.csv', index=False)

