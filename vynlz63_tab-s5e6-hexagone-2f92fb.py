# ===============================================================
# 0. IMPORTS & SEED
# ===============================================================
import os, random, warnings, optuna, xgboost as xgb
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.impute  import SimpleImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import category_encoders as ce

SEED = 42
np.random.seed(SEED); random.seed(SEED)

# ===============================================================
# 1. DATA
# ===============================================================
PATH = "/kaggle/input/playground-series-s5e6/"
train = pd.read_csv(PATH + "train.csv")
test  = pd.read_csv(PATH + "test.csv")
sub   = pd.read_csv(PATH + "sample_submission.csv")

FEATS     = [c for c in test.columns if c != "id"]
CAT_COLS  = ["Soil Type", "Crop Type"]
NUM_COLS  = [c for c in FEATS if c not in CAT_COLS]
TARGET    = "Fertilizer Name"

train[CAT_COLS] = train[CAT_COLS].fillna("None")
test[CAT_COLS]  = test[CAT_COLS].fillna("None")

lbl_y = LabelEncoder()
y     = lbl_y.fit_transform(train[TARGET])
X     = train[FEATS]
Xt    = test[FEATS]
n_cls = y.max() + 1

# ===============================================================
# 2. ENCODING & PIPELINE
# ===============================================================
num_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc" , RobustScaler())
])

te  = ce.TargetEncoder(cols=CAT_COLS, smoothing=0.25)
loo = ce.LeaveOneOutEncoder(cols=CAT_COLS, sigma=0.1)
cnt = ce.CountEncoder(cols=CAT_COLS)
cbe = ce.CatBoostEncoder(cols=CAT_COLS, a=1.0)

def fit_enc(x, y_):
    te.fit(x, y_); loo.fit(x, y_); cnt.fit(x); cbe.fit(x, y_)
    num_pipe.fit(x[NUM_COLS])
    return (te, loo, cnt, cbe, num_pipe)

def tr_all(enc, df):
    te_, loo_, cnt_, cbe_, num_p = enc
    return np.hstack([
        num_p.transform(df[NUM_COLS]),
        te_ .transform(df).values,
        loo_.transform(df).values,
        cnt_.transform(df).values,
        cbe_.transform(df).values
    ])

# ===============================================================
# 3. OPTUNA OBJECTIVE XGB
# ===============================================================
def obj_xgb(t, Xtr, ytr, Xva, yva):
    print("ğŸ§ª Nouveau trial Optuna")
    try:
        p = {
            "objective": "multi:softprob",
            "num_class": n_cls,
            "seed": SEED,
            "booster": "dart",
            "learning_rate": t.suggest_float("lr", 0.05, 0.3, log=True),
            "max_depth": t.suggest_int("depth", 4, 10),
            "subsample": t.suggest_float("sub", 0.6, 1.0),
            "colsample_bytree": t.suggest_float("col", 0.6, 1.0),
            "rate_drop": t.suggest_float("rd", 0, 0.3),
            "skip_drop": t.suggest_float("sd", 0, 0.3),
            "lambda": t.suggest_float("l2", 0, 5),
            "n_estimators": 500  # rÃ©duit pour debug rapide
        }
        print("ğŸ“¦ ParamÃ¨tres :", p)
        m = xgb.XGBClassifier(**p, eval_metric="mlogloss", verbosity=1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], early_stopping_rounds=30, verbose=True)
        return log_loss(yva, m.predict_proba(Xva))
    except Exception as e:
        print("â�Œ ERREUR DANS LE TRIAL :", e)
        raise

# ===============================================================
# 4. CV & BAGGING
# ===============================================================
folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
oof_xgb = np.zeros((len(X), n_cls))
test_xgb = np.zeros((len(Xt), n_cls))

for f, (tr, va) in enumerate(folds.split(X, y), 1):
    print(f"\nğŸ”� Fold {f}")
    Xtr, ytr = X.iloc[tr], y[tr]
    Xva, yva = X.iloc[va], y[va]
    enc = fit_enc(Xtr, ytr)
    Xtr_enc, Xva_enc, Xt_enc = map(lambda d: tr_all(enc, d), [Xtr, Xva, Xt])

    for s in [13, 42, 2025]:
        print(f"ğŸŒ± Seed {s}")
        st = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=s))
        st.optimize(lambda t: obj_xgb(t, Xtr_enc, ytr, Xva_enc, yva),
                    n_trials=5, show_progress_bar=True)
        m = xgb.XGBClassifier(**st.best_params,
                              objective="multi:softprob",
                              num_class=n_cls,
                              seed=s,
                              eval_metric="mlogloss",
                              verbosity=0)
        m.fit(Xtr_enc, ytr, eval_set=[(Xva_enc, yva)],
              early_stopping_rounds=30, verbose=False)
        oof_xgb[va] += m.predict_proba(Xva_enc) / 3
        test_xgb += m.predict_proba(Xt_enc) / 15

# ===============================================================
# 5. META LEARNING (XGB encore)
# ===============================================================
rng = np.random.default_rng(SEED)
noise = lambda arr: arr * rng.uniform(0.95, 1.05, arr.shape)

X_meta  = oof_xgb * 0.8 + noise(oof_xgb) * 0.2
Xt_meta = test_xgb

def meta_obj(t):
    p = {
        "objective": "multi:softprob",
        "num_class": n_cls,
        "seed": SEED,
        "learning_rate": t.suggest_float("lr", 0.05, 0.2, log=True),
        "max_depth": t.suggest_int("depth", 3, 6),
        "lambda": t.suggest_float("l2", 0, 3),
        "n_estimators": 200
    }
    m = xgb.XGBClassifier(**p, eval_metric="mlogloss", verbosity=0)
    m.fit(X_meta, y, eval_set=[(X_meta, y)], early_stopping_rounds=30, verbose=False)
    return log_loss(y, m.predict_proba(X_meta))

print("\nğŸ“ˆ Optuna Meta Learning...")
meta_st = optuna.create_study(direction="minimize",
                              sampler=optuna.samplers.TPESampler(seed=SEED))
meta_st.optimize(meta_obj, n_trials=30, show_progress_bar=True)

meta = xgb.XGBClassifier(**meta_st.best_params,
                         objective="multi:softprob",
                         num_class=n_cls,
                         seed=SEED,
                         eval_metric="mlogloss",
                         verbosity=0)
meta.fit(X_meta, y)

def map3(pred):
    top3 = np.argsort(pred, 1)[:, -3:][:, ::-1]
    return np.mean([1 if y[i] in top3[i] else 0 for i in range(len(y))])

print(f"\nğŸ”¥ MAP@3 OOF = {map3(meta.predict_proba(X_meta)):.5f}")

# ===============================================================
# 6. SUBMISSION
# ===============================================================
pred = meta.predict_proba(Xt_meta)
top3 = np.argsort(pred, 1)[:, -3:][:, ::-1]
labels = lbl_y.inverse_transform(top3.ravel()).reshape(top3.shape)

pd.DataFrame({
    "id": sub.id,
    "Fertilizer Name": [" ".join(row) for row in labels]
}).to_csv("submission.csv", index=False)

print("âœ… submission.csv Ã©crit")


