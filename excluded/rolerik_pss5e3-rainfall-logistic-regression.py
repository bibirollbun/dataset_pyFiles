import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn

print("Numpy version: ", np.__version__)
print("Pandas version: ", pd.__version__)
print("Matplotlib version: ", mpl.__version__)
print("Sklearn version: ", sklearn.__version__)

datadir = "/kaggle/input/playground-series-s5e3/"



def set_name(self, name):
    self.name = name
    return self

def short_info(self):
    print(f"DataFrame {self.name}: shape {self.shape}, nulls {self.isna().sum().sum()}")

pd.DataFrame.set_name = set_name
pd.DataFrame.si = short_info



train = pd.read_csv(datadir + "train.csv", index_col="id").set_name("train")
test = pd.read_csv(datadir + "test.csv", index_col="id").set_name("test")

train.si()
test.si()

train["day"] = train.index % 365 + 1
test["day"] = test.index % 365 + 1
train.head()



# Fix test by imputing missing value and prepending some training data for lag features
test["winddirection"] = test["winddirection"].ffill()
test = pd.concat([train.iloc[-1:], test], join="inner").set_name("test")



from sklearn.linear_model import LinearRegression

def add_lags(X, feats, lag):
    for f in feats:
        for l in range(1, lag+1):
            lag_feat_name = f"{f}_lag_{l}"
            X[lag_feat_name] = X[f].shift(l)
    return X.dropna().set_name(X.name)

def season(t, n):
    s = {}
    s["const"] = np.ones(t.shape[0])
    for f in range(1, n+1):
        s[f"cos_{f}"] = np.cos(2*np.pi*f*t / 365)
        s[f"sin_{f}"] = np.sin(2*np.pi*f*t / 365)
    return pd.DataFrame(s, index=t.index)

def deseason(X, y):
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, y)
    y_pred = reg.predict(X)
    return y - y_pred



train["difftemp"] = train["maxtemp"] - train["mintemp"]

cols_deseason = ["pressure", "maxtemp", "temparature", "mintemp", "difftemp", "dewpoint", "humidity", "cloud", "sunshine", "winddirection", "windspeed"]
S = season(train["day"], 3)
train = pd.concat([S, train], axis=1, join="inner").set_name("train")

for c in cols_deseason:
    train[c] = deseason(S, train[c])

cols_lag = ["pressure", "temparature", "difftemp", "dewpoint", "humidity", "cloud", "sunshine", "winddirection", "windspeed"]
train = add_lags(train, cols_lag, 1)

cols_feat = [c for c in train.columns if c not in ["day", "rainfall"]]
col_target = "rainfall"
print("Number of features: ", len(cols_feat))
train.si()
train.head()



test["difftemp"] = test["maxtemp"] - test["mintemp"]

S = season(test["day"], 3)
test = pd.concat([S, test], axis=1, join="inner").set_name("test")

for c in cols_deseason:
    test[c] = deseason(S, test[c])

cols_lag = ["pressure", "temparature", "difftemp", "dewpoint", "humidity", "cloud", "sunshine", "winddirection", "windspeed"]
test = add_lags(test, cols_lag, 1)

test.si()
test.head()



from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

aucscorer = make_scorer(roc_auc_score, needs_proba=True)

logit_params = {
    "solver": "liblinear",
    "penalty": "l2",
    "fit_intercept": False,
    "class_weight": "balanced",
    "C": 0.01,
    "tol": 1e-10,
    "max_iter": 50
}



%%time

X = train[cols_feat]
y = train[col_target]
nfeat_max = len(cols_feat)
nfeat_min = 3
n_iter = 5

all_scores = []
good_score_threshhold = 0.896
feat_counts = {}

for i in range(n_iter):
    print("Iteration", i)
    cols = cols_feat.copy()
    scores = []
    
    while len(cols) >= nfeat_min:
        X = train[cols]
        X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=False, test_size=365)
    
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        clf = LogisticRegression(**logit_params)
        clf.fit(X_train_scaled, y_train)
        
        X_val_scaled = scaler.transform(X_val)
        y_pred = clf.predict_proba(X_val_scaled)[:,1]
        score = roc_auc_score(y_val, y_pred)
        scores.append(score)
        if (score > good_score_threshhold):
            for c in cols:
                if c in feat_counts:
                    feat_counts[c] += 1
                else:
                    feat_counts[c] = 1
        #print(f"#features = {len(cols)}, score = {score}")
    
        r = permutation_importance(clf, X_val_scaled, y_val, scoring=aucscorer)
        imp = pd.Series(r["importances_mean"], index=cols).sort_values(ascending=False)
        exclude_col = imp.index[-1]
        #print("Excluding feature ", exclude_col)
        cols.remove(exclude_col)

    all_scores.append(scores)



fig,ax = plt.subplots(1, 1, figsize=(9, 6))
ax.grid(True)
for s in all_scores:
    ax.plot(range(nfeat_max, nfeat_min-1, -1), s)



feat_ser = pd.Series(feat_counts.values(), index=feat_counts.keys()).sort_values(ascending=False)
n_good_features = feat_ser.shape[0]
print("Good features: ", n_good_features)
n_selected_features = min(n_good_features, 17)
selected_features = list(feat_ser.index[:n_selected_features])
feat_ser



class FixedSplitter:
    def __init__(self, split_index):
        self._split_index = split_index

    def get_n_splits(self, X, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        return [(np.arange(0, self._split_index, dtype=np.int32), np.arange(self._split_index, X.shape[0], dtype=np.int32))]


class FixedSplitter2:
    def __init__(self):
        pass

    def get_n_splits(self, X, y=None, groups=None):
        return 2

    def split(self, X, y=None, groups=None):
        return [(np.arange(0, 1460, dtype=np.int32), np.arange(1460, 1606, dtype=np.int32))
            ,(np.arange(0, 1825, dtype=np.int32), np.arange(1825, 1971, dtype=np.int32))]


class FixedSplitter3:
    def __init__(self):
        pass

    def get_n_splits(self, X, y=None, groups=None):
        return 2

    def split(self, X, y=None, groups=None):
        return [(np.arange(0, 1460, dtype=np.int32), np.arange(1606, 2187, dtype=np.int32))
            ,(np.arange(0, 1825, dtype=np.int32), np.arange(1971, 2187, dtype=np.int32))]



class WindowSplitter:
    def __init__(self, n_splits=5, test_ratio=0.2):
        self.n_splits = n_splits
        self.test_ratio = test_ratio

    def get_n_splits(self, X, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        idx = 0
        test_size = n_samples * self.test_ratio // (1 + self.test_ratio * (self.n_splits - 1))
        train_size = n_samples - self.n_splits * test_size
        end_idx = idx + train_size + test_size
        while (end_idx <= n_samples):
            yield (np.arange(idx, idx + train_size, dtype=np.int32), np.arange(idx + train_size, end_idx, dtype=np.int32))
            idx += test_size
            end_idx = idx + train_size + test_size



%%time

print(selected_features)
X = train[selected_features]
y = train[col_target]

pipe = Pipeline([("scaler", StandardScaler()), ("logitreg", LogisticRegression(**logit_params))])

lst_scores = []

splitters = {
    "Validate last 2 years": FixedSplitter(X.shape[0] * 4 / 6),
    "Validate 146 days of last 2 years": FixedSplitter2(),
    "Validate last 2 years except 146 days": FixedSplitter3(),
    "Windowed 5-fold split": WindowSplitter(n_splits=5, test_ratio=0.25)
}

for n, s in splitters.items():
    scores = cross_val_score(pipe, X, y, scoring=aucscorer, cv=s)
    result = float(np.mean(scores))
    lst_scores.append(result)
    print(f"{n}: {result:.6f}")

lst_weights = [1, 1, 1, 1]
print("Scores: ", lst_scores)
print("Weights: ", lst_weights)

totalresult = np.average(lst_scores, weights=lst_weights)
print("Total weighted score:", totalresult)



X_train = train[selected_features]
y_train = train[col_target]
X_test = test[selected_features]

scaler = StandardScaler()
X_train_prep = scaler.fit_transform(X_train)
X_test_prep = scaler.transform(X_test)

clf = LogisticRegression(**logit_params)
clf.fit(X_train_prep, y_train)

pred = clf.predict_proba(X_test_prep)
print(pred.shape)
pred[0:10]



sub = pd.DataFrame({"id": test.index, "rainfall": pred[:, 1]})
display(sub.head())
subname = f"submission_logit_deseason_17_1.csv"
sub.to_csv(subname, index=False)


