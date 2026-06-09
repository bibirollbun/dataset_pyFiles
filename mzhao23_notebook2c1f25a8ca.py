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


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df_train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")
df_test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")





def create_X(df):
    dt = pd.to_datetime(df["datetime"])
    hour = pd.get_dummies(dt.dt.hour, prefix="hour", dtype=int)
    dow  = pd.get_dummies(dt.dt.dayofweek, prefix="day", dtype=int)
    mon  = pd.get_dummies(dt.dt.month, prefix="mon", dtype=int)   # helpful
    yr   = pd.get_dummies(dt.dt.year,  prefix="yr",  dtype=int)   # helpful

    season  = pd.get_dummies(df["season"],  prefix="season",  dtype=int)
    weather = pd.get_dummies(df["weather"], prefix="weather", dtype=int)

    X = pd.concat([hour, dow, mon, yr, season, weather], axis=1)
    X["holiday"]   = df["holiday"].astype(int)
    X["workingday"]= df["workingday"].astype(int)
    X["temp"]      = df["temp"]
    X["atemp"]     = df["atemp"]
    X["humidity"]  = df["humidity"]
    X["windspeed"] = df["windspeed"]    # add this back
    return X

def create_Y(df):
    y_train = pd.DataFrame()
    y_train["log1count"] = np.log1p(df["count"].values)
    return y_train
df_X_tr = create_X(df_train)
df_y_tr = create_Y(df_train)


N, d = df_X_tr.shape
X_tr = df_X_tr.to_numpy()
y_tr = df_y_tr["log1count"].to_numpy()





def lin_reg(XTX, XTy, yTy, ind_subset = None):
    if (ind_subset != None):
        ind_subset = [0] + [i + 1 for i in ind_subset]
        XTX = XTX[np.ix_(ind_subset, ind_subset)]
        XTy = XTy[ind_subset]
        yTy = yTy
    betas = np.linalg.pinv(XTX) @ XTy
    return betas
def evaluate_r2(betas, XTX, XTy, yTy, ind_subset = None):
    if (ind_subset != None):
        ind_subset = [0] + [i + 1 for i in ind_subset]
        XTX = XTX[np.ix_(ind_subset, ind_subset)]
        XTy = XTy[ind_subset]
        yTy = yTy
    r2 = (betas.T @ (2 * XTy - XTX @ betas))/(yTy)
    return r2


def split(N, n_splits = 5):
    indices = list(range(N))
    np.random.shuffle(indices)
    size = N//n_splits
    ind_splits = [indices[size * i: (size * (i + 1) if i < n_splits - 1 else N)] for i in range(n_splits)]
    tr_vl_pairs = [([k for j in range(n_splits) for k in ind_splits[j] if j != i], ind_splits[i]) for i in range(n_splits)]
    return tr_vl_pairs
    
def create_data_splits(X, y, n_splits = 5):
    N, d = X.shape
    k_split = split(N, n_splits = n_splits)
    out = []
    for tr, vl in k_split:
        # build Gram *with* intercept here so the rest of your code stays the same
        Xt_tr = np.c_[np.ones(len(tr)), X[tr]]
        Xt_vl = np.c_[np.ones(len(vl)), X[vl]]
        out.append((
            (Xt_tr.T @ Xt_tr, Xt_tr.T @ y[tr], float(y[tr].T @ y[tr])),
            (Xt_vl.T @ Xt_vl, Xt_vl.T @ y[vl], float(y[vl].T @ y[vl]))
        ))
    return out


def out_of_sample_eval(data_splits, ind_subset):
    sum_r2 = 0
    for (XTX_tr, XTy_tr, yTy_tr), (XTX_vl, XTy_vl, yTy_vl) in data_splits:
        betas = lin_reg(XTX_tr, XTy_tr, yTy_tr, ind_subset = ind_subset)
        sum_r2 += evaluate_r2(betas, XTX_vl, XTy_vl, yTy_vl, ind_subset = ind_subset)
    return sum_r2/len(data_splits)



def sequential_feature_selection(X, y, d, max_feats = 15, n_splits = 5):
    data_splits = create_data_splits(X, y, n_splits=n_splits)

    chosen = []
    best_layer_r2 = -np.inf
    while len(chosen) < max_feats:
        best_this_round_r2 = -np.inf
        best_feat = None
        for i in range(d):
            if i in chosen: 
                continue
            cv_r2 = out_of_sample_eval(data_splits, chosen + [i])
            if cv_r2 > best_this_round_r2:
                best_this_round_r2, best_feat = cv_r2, i
        if best_this_round_r2 <= best_layer_r2 or best_feat is None:
            break
        chosen.append(best_feat)
        best_layer_r2 = best_this_round_r2

    # Final fit on FULL DATA (add ONE bias here)
    n = X.shape[0]
    X_with_bias = np.c_[np.ones(n), X]
    XTX_full = X_with_bias.T @ X_with_bias
    XTy_full = X_with_bias.T @ y
    yTy_full = float(y.T @ y)

    betas_sub = lin_reg(XTX_full, XTy_full, yTy_full, ind_subset=chosen)

    betas_full = np.zeros(d + 1)      # [intercept, d features]
    betas_full[0] = betas_sub[0]
    for k, feat in enumerate(chosen):
        betas_full[feat + 1] = betas_sub[k + 1]
    return betas_full, chosen, best_layer_r2



betas_full, chosen, best_layer_r2 = sequential_feature_selection(X_tr, y_tr, d)
print(betas_full[:8], chosen, best_layer_r2)

# Test matrix must match: add ONE bias
df_X_ts = create_X(df_test)
X_ts = np.c_[np.ones(df_X_ts.shape[0]), df_X_ts.to_numpy()]
y_hat_ts = X_ts @ betas_full
pred = pd.DataFrame({
    "datetime": df_test["datetime"],
    "count": np.clip(np.expm1(y_hat_ts), 0, None)
})
pred.to_csv('submission.csv', index=False)




