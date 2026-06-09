# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ==============================
# FULL PIPELINE - BANK MARKETING
# ==============================

import pandas as pd, numpy as np, os
import cudf
from itertools import combinations
from sklearn.metrics import roc_auc_score

# -------------------------------
# STEP 1: LOAD DATA
# -------------------------------
def load_data(PATH):
    train = cudf.read_csv(f"{PATH}train.csv").set_index('id')
    test = cudf.read_csv(f"{PATH}test.csv").set_index('id')
    test['y'] = np.random.randint(0, 2, len(test))

    orig = cudf.read_csv(
        "/kaggle/input/bank-marketing-dataset-full/bank-full.csv",
        delimiter=";"
    )
    orig['y'] = orig.y.map({'yes':1,'no':0})
    orig['id'] = (np.arange(len(orig))+1e6).astype('int')
    orig = orig.set_index('id')

    return train, test, orig


# -------------------------------
# STEP 2: EDA + SPLIT NUM/CAT
# -------------------------------
def analyze_columns(combine):
    CATS, NUMS = [], []
    for c in combine.columns[:-1]:
        if combine[c].dtype=='object':
            CATS.append(c)
        else:
            NUMS.append(c)
    return CATS, NUMS


# -------------------------------
# STEP 3: LABEL ENCODE
# -------------------------------
def label_encode(combine, NUMS, CATS):
    CATS1, SIZES = [], {}
    for c in NUMS + CATS:
        n = c
        if c in NUMS: 
            n = f"{c}2"
            CATS1.append(n)
        combine[n],_ = combine[c].factorize()
        SIZES[n] = combine[n].max()+1
        combine[c] = combine[c].astype('int32')
        combine[n] = combine[n].astype('int32')
    return combine, CATS1, SIZES


# -------------------------------
# STEP 4: COMBINE COLUMNS (PAIRWISE)
# -------------------------------
def combine_pairs(combine, CATS, CATS1, SIZES):
    pairs = combinations(CATS + CATS1, 2)
    new_cols, CATS2 = {}, []
    for c1, c2 in pairs:
        name = "_".join(sorted((c1, c2)))
        new_cols[name] = combine[c1] * SIZES[c2] + combine[c2]
        CATS2.append(name)
    if new_cols:
        new_df = cudf.DataFrame(new_cols)
        combine = cudf.concat([combine, new_df], axis=1)
    return combine, CATS2


# -------------------------------
# STEP 5: COUNT ENCODING
# -------------------------------
def count_encode(combine, CATS, CATS1, CATS2):
    CE, CC = [], CATS+CATS1+CATS2
    combine['i'] = np.arange(len(combine))
    for c in CC:
        tmp = combine.groupby(c).y.count().astype('int32')
        tmp.name = f"CE_{c}"
        CE.append(f"CE_{c}")
        combine = combine.merge(tmp, on=c, how='left')
    combine = combine.sort_values('i')
    return combine, CE


# -------------------------------
# STEP 6: TARGET ENCODING W/ ORIG
# -------------------------------
def target_encode(train, test, orig, CATS, CATS1, CATS2):
    TE, CC = [], CATS+CATS1+CATS2
    for c in CC:
        tmp = orig.groupby(c).y.mean().astype('float32')
        NAME = f"TE_ORIG_{c}"
        tmp.name = NAME
        TE.append(NAME)
        train = train.merge(tmp, on=c, how='left')
        test  = test.merge(tmp, on=c, how='left')
    train = train.sort_values('i')
    test  = test.sort_values('i')
    return train, test, TE


# -------------------------------
# STEP 7: NORMALIZE
# -------------------------------
def normalize(train, test, CE):
    LOG = ['balance','duration','campaign','pdays','previous']
    for c in LOG+CE:
        if c in LOG: 
            mn = min((train[c].min(), test[c].min()))
            train[c] = train[c] - mn
            test[c]  = test[c] - mn
        train[c] = np.log1p(train[c])
        test[c]  = np.log1p(test[c])
    return train, test


# -------------------------------
# STEP 8: DEFINE FEATURES
# -------------------------------
def define_features(CATS, NUMS, CATS1, CE, TE):
    FEATURES = CATS+NUMS+CATS1+CE+TE
    TARGET_COL = 'y'
    return FEATURES, TARGET_COL


# -------------------------------
# STEP 9: TRAIN MODELS (XGB + NN)
# -------------------------------
def train_models(train, test, FEATURES, TARGET_COL):
    # NOTE: Place your XGB + NN training code here
    # including KFold, oof, preds, etc.
    # Keeping placeholders to preserve your pipeline.
    oof_preds = np.zeros(len(train))
    oof_preds2 = np.zeros(len(train))
    preds = np.zeros(len(test))
    test_preds = np.zeros(len(test))
    test_preds2 = np.zeros(len(test))
    return oof_preds, oof_preds2, preds, test_preds, test_preds2


# -------------------------------
# STEP 10: ENSEMBLE + SUBMISSION
# -------------------------------
def make_submissions(PATH, train, test, preds, oof, 
                     oof_preds, oof_preds2, test_preds, test_preds2):

    # XGB Sub
    preds_xgb = (test_preds+test_preds2)/2.
    sub = pd.read_csv(f"{PATH}sample_submission.csv")
    sub['y'] = preds_xgb
    sub.to_csv("submission_xgb_train_more.csv",index=False)

    # NN Sub
    sub = pd.read_csv(f"{PATH}sample_submission.csv")
    sub['y'] = preds
    sub.to_csv("submission_nn_train_more.csv",index=False)

    # Ensemble Sub
    oof_xgb = (oof_preds+oof_preds2)/2.
    best_m, best_w = 0, 0
    for w in np.arange(0,1.01,0.01):
        oof_ensemble = (1-w)*oof_xgb + w*oof
        m = roc_auc_score(train.y.to_numpy(), oof_ensemble)
        if m>best_m:
            best_m, best_w = m, w
    sub = pd.read_csv(f"{PATH}sample_submission.csv")
    sub['y'] = (1-best_w)*preds_xgb + best_w*preds
    sub.to_csv("submission_ensemble_train_more.csv",index=False)


# -------------------------------
# MAIN PIPELINE
# -------------------------------
def main(PATH):
    # Load
    train, test, orig = load_data(PATH)
    combine = cudf.concat([train,test,orig],axis=0)

    # Process
    CATS, NUMS = analyze_columns(combine)
    combine, CATS1, SIZES = label_encode(combine, NUMS, CATS)
    combine, CATS2 = combine_pairs(combine, CATS, CATS1, SIZES)
    combine, CE = count_encode(combine, CATS, CATS1, CATS2)

    # Split
    train = combine.iloc[:len(train)]
    test  = combine.iloc[len(train):len(train)+len(test)]
    orig  = combine.iloc[-len(orig):]
    del combine

    # TE
    train, test, TE = target_encode(train, test, orig, CATS, CATS1, CATS2)

    # Normalize
    train, test = normalize(train, test, CE)

    # Features
    FEATURES, TARGET_COL = define_features(CATS, NUMS, CATS1, CE, TE)

    # Train Models
    oof_preds, oof_preds2, preds, test_preds, test_preds2 = train_models(
        train, test, FEATURES, TARGET_COL
    )

    # Submissions
    make_submissions(PATH, train, test, preds, 
                     oof_preds, oof_preds2, oof_preds, test_preds, test_preds2)


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    PATH = "/kaggle/input/playground-series-s5e8/"
    main(PATH)





