DIR = "/kaggle/input/playground-series-s5e3/"
COL_ID = "id"
COL_TARGET = "rainfall"
FNAME_SUBMIT = "submission.csv"


import numpy as np
import pandas as pd
import warnings
warnings.simplefilter("ignore")


X = pd.read_csv(DIR + "train.csv", index_col=COL_ID)
X_test = pd.read_csv(DIR + "test.csv", index_col=COL_ID)
y = X.pop(COL_TARGET)
X.head()


mins = pd.concat([
    X.min(),
    X_test.min(),
], axis=1).min(axis=1)
X -= mins
X_test -= mins
X.head()


means = X_test.mean()
X = X.fillna(means)
X_test = X_test.fillna(means)
X.head()


def cyclic_encoding(data, col):
    MAX_RADS = 2 * np.pi
    s = data[col]
    s_norm = s / s.max() * MAX_RADS
    cyc = pd.concat([
        np.cos(s_norm).rename(f"{col}_cos"),
        np.sin(s_norm).rename(f"{col}_sin"),
    ], axis=1)
    data = pd.concat([
        data.drop(columns=col),
        cyc - cyc.min(),
    ], axis=1)
    return data

X = cyclic_encoding(X, "day")
X_test = cyclic_encoding(X_test, "day")
X.head()


X = np.log1p(X)
X_test = np.log1p(X_test)
X.head()


from sklearn.svm import SVC
from sklearn.metrics.pairwise import additive_chi2_kernel

def build_model():
    return SVC(
        C=np.sqrt(11),
        kernel=additive_chi2_kernel,
        probability=True,
        class_weight="balanced",
        tol=0.15,
        break_ties=True,
    )


from sklearn.model_selection import cross_validate
cv = cross_validate(
    build_model(), X, y,
    scoring="roc_auc",
    cv=7, n_jobs=7,
    return_train_score=True,
    return_estimator=True,
)
cv = pd.DataFrame(cv)
cv.drop(columns="estimator").mean().round(5)


submission = np.array([
    model.predict_proba(X_test)[:, 1].tolist() 
    for model in cv.estimator.tolist()
]).mean(axis=0)
submission = pd.Series(submission, name=COL_TARGET, index=X_test.index)
submission.to_csv(FNAME_SUBMIT)
submission.head()

