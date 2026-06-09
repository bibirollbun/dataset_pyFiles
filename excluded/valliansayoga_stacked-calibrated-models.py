!pip install feature-engine -q


import numpy as np
import pandas as pd
import os
import warnings


warnings.filterwarnings("ignore")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


def basic_preprocess(data, is_train=True):
    df = data.copy()
    
    month_encode = pd.date_range("2020-01-01", "2020-12-31", freq="M").strftime("%b").str.lower().tolist()
    month_encode = {v: k  for k, v in enumerate(month_encode)}

    if is_train:
        if df.y.dtype != int:
            df.y = df.y.eq("yes").astype(int)
    df.month = df.month.map(month_encode)

    binaries = ["default", "housing", "loan"]
    for col in binaries:
        df[col] = df[col].eq("no").astype(int)
        
    return df


sub = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sub


sub = basic_preprocess(sub, is_train=False)


df_ori = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", sep=";")
df_ori = basic_preprocess(df_ori, is_train=True)
df_ori.head()


df_ori.shape


df_comp = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv").drop("id", axis=1).sample(100000-45211, random_state=42)
df_comp = basic_preprocess(df_comp, is_train=True)
df_comp


df = pd.concat([df_ori, df_comp], ignore_index=True)
# df = df_ori
df.shape


df.isna().sum()


df.y.unique()


import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 5)
sns.countplot(x=pd.qcut(df.age, 9), hue=df.y)


sns.countplot(x=pd.qcut(df.balance, 6), hue=df.y)


sns.countplot(x=pd.qcut(df.campaign, 2), hue=df.y)


sns.countplot(x=pd.qcut(df.duration, 6), hue=df.y)


sns.boxplot(df, orient="h", width=0.5, dodge=False)
plt.show()

sns.boxplot(np.log1p(df.select_dtypes("number")), orient="h", width=0.5, dodge=False)


plt.rcParams["figure.figsize"] = (6, 3)
categorical = df.select_dtypes(object).columns

for col in categorical:
    sns.countplot(y=df[col], hue=df.y)
    plt.show()


categories = df.select_dtypes(object).columns

for col in categories:
    df[col] = pd.Categorical(df[col])
    sub[col] = pd.Categorical(sub[col])


from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from feature_engine.encoding import WoEEncoder, OrdinalEncoder, MeanEncoder, RareLabelEncoder
from feature_engine.creation import CyclicalFeatures
from sklearn.neighbors import KNeighborsClassifier

cv = StratifiedKFold(4, shuffle=True, random_state=42)
shared_ensemble_params = {
    "class_weight": "balanced",
    "random_state": 42,
    "max_features": 0.8,
    "max_depth": 8
}
bases = [
    RandomForestClassifier(150, n_jobs=-1, **shared_ensemble_params),
    HistGradientBoostingClassifier(
        learning_rate=0.01,
        max_iter=1000,
        scoring="roc_auc", 
        l2_regularization=0.25,
        **shared_ensemble_params
    ),
    ExtraTreesClassifier(150, n_jobs=-1, **shared_ensemble_params),
    LogisticRegression(n_jobs=-1, C=0.1, random_state=42, class_weight="balanced"),
    KNeighborsClassifier(n_jobs=-1),
    KNeighborsClassifier(n_jobs=-1, weights="distance"),
]

for i, base in enumerate(bases):
    name = str(base)
    base = make_pipeline(
        WoEEncoder(),
        CyclicalFeatures(["day", "month"], {"day": 31, "month": 11}), 
        StandardScaler(),
        base
    )
    bases[i] = (name, CalibratedClassifierCV(base, method="isotonic", n_jobs=-1, cv=cv))

model = StackingClassifier(
    bases,
    LogisticRegression(n_jobs=-1, C=0.1, random_state=42, class_weight="balanced"),
    n_jobs=-1,
    verbose=2
)



X = df.drop("y", axis=1)
y = df.y


# cv_results = cross_validate(model, X, y, scoring="roc_auc", return_estimator=True, cv=cv, n_jobs=-1)



from sklearn.metrics import roc_auc_score
# print(cv_results["test_score"])
# print(np.nanmean(cv_results["test_score"]))

model.fit(X, y)
proba_all = model.predict_proba(X)[:, 1]
roc_auc = roc_auc_score(y, proba_all)
roc_auc


# 0.9526478520364879
# 0.9528598086829199


from sklearn.metrics import roc_auc_score

model.fit(X, y)
proba_all = model.predict_proba(X)[:, 1]
roc_auc = roc_auc_score(y, proba_all)
roc_auc


sub.shape


prob_train = model.predict_proba(X)[:, 1]
pred_train = np.where(prob_train > 0.5, 1, 0)

prob_sub = model.predict_proba(sub[X.columns])[:, 1]
prob_sub[:5]


from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay

cr = classification_report(y, pred_train)
roc_auc = roc_auc_score(y, prob_train)
ConfusionMatrixDisplay.from_predictions(y, pred_train)
plt.grid(False)
plt.show()
print(cr)


print("roc_auc:", roc_auc)
fig = plt.figure(figsize=(4, 4))
ax = fig.gca()
RocCurveDisplay.from_predictions(y, prob_train, plot_chance_level=True, despine=True, ax=ax)
plt.grid(False)



sub["y"] = prob_sub
sub.head()


sub[["id", "y"]].to_csv("submission.csv", index=False)


!head submission.csv

