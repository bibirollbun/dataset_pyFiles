import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import warnings


warnings.filterwarnings("ignore")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_ori = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", sep=";")
df_ori.y = df_ori.y.eq("yes").astype(int)
df_ori.head()


df_ori.y.unique()


df_comp = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv").drop("id", axis=1)
df_comp.head()


df_train = pd.concat([df_ori, df_comp], ignore_index=True)


import seaborn as sns
import matplotlib.pyplot as plt


sns.set_theme()

fig, ax = plt.subplots(1, 3, sharey=True, figsize=(9, 3))
sns.countplot(x=df_ori.y, ax=ax[0])
sns.countplot(x=df_comp.y, ax=ax[1])
sns.countplot(x=df_train.y, ax=ax[2])

for a, name in zip(ax, ["original", "competition", "combined"]):
    a.set_title(name)


categorics = df_train.select_dtypes("object").columns
categorics = df_train.columns.isin(categorics)
categorics


from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline


cv = StratifiedKFold(3)

est = HistGradientBoostingClassifier(
    max_iter=500,
    # categorical_features=categorics,
    l2_regularization=0.2,
    random_state=42,
    class_weight="balanced"
)
model = CalibratedClassifierCV(est, method="isotonic", cv=cv, n_jobs=-1)

transformer = make_column_transformer(
    (OrdinalEncoder(), make_column_selector(dtype_include=object)),
    remainder="passthrough"
)

model = make_pipeline(
    transformer,
    model
)


X = df_train.drop("y", axis=1)
y = df_train.y
model.fit(X, y)


pred = model.predict(X)
prob = model.predict_proba(X)[:, 1]


from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay

cr = classification_report(y, pred)
roc_auc = roc_auc_score(y, prob)
ConfusionMatrixDisplay.from_predictions(y, pred)
plt.grid(False)

RocCurveDisplay.from_predictions(y, prob)
plt.grid(False)
print(cr)
print("roc_auc:", roc_auc)


sub = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sub


sub["y"] = model.predict_proba(sub.drop("id", axis=1))[:, 1]
sub.head()


sub[["id", "y"]].to_csv("submission.csv", index=False)


!head submission.csv

