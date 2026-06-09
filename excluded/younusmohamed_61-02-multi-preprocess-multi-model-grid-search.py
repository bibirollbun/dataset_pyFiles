import numpy as np, pandas as pd, time, itertools, warnings, gc, os, logging
from pathlib import Path
from tqdm.auto import tqdm

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, ParameterSampler  
from sklearn.metrics import log_loss, accuracy_score, make_scorer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier, Pool

import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
logging.getLogger("lightgbm").setLevel(logging.ERROR)
os.environ["NUMEXPR_MAX_THREADS"] = "8"
os.environ["XGB_DISABLE_GPU_WARN"] = "1"
# lgb.set_config(verbosity = -1)           # silence LGBM

SEED, N_SPLITS = 42, 5
RS = np.random.RandomState(SEED)
cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
save_dir = Path("./model_outputs"); save_dir.mkdir(exist_ok=True)


DATA_P = Path("/kaggle/input/playground-series-s5e7")
RAW_COLS = ["Time_spent_Alone","Stage_fear","Social_event_attendance",
            "Going_outside","Drained_after_socializing",
            "Friends_circle_size","Post_frequency"]

train = pd.read_csv(DATA_P/"train.csv").set_index("id")
test  = pd.read_csv(DATA_P/"test.csv").set_index("id")
y     = train["Personality"].map({"Extrovert":0, "Introvert":1})

print(f"train {train.shape}   test {test.shape}")


def build_preproc(cat_cols, num_cols):
    """Ordinal-encode categoricals, median-impute numerics, keep pandas output."""
    cat_pipe = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OrdinalEncoder(dtype=np.int16, handle_unknown="use_encoded_value",
                       unknown_value=-1),
        FunctionTransformer(lambda x: x + 1)
    )
    num_pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(lambda x: (x+1).astype(np.int16))
    )
    return ColumnTransformer(
        [("cat", cat_pipe, cat_cols),
         ("num", num_pipe, num_cols)],
        remainder="passthrough",
        verbose_feature_names_out=False
    ).set_output(transform="pandas")


deep_rf  = {"n_estimators": RS.randint(400,1201,20),
            "max_depth":   RS.randint(4,11,20),
            "min_samples_leaf": RS.randint(4,25,20)}

deep_lgb = {"n_estimators": RS.randint(800,1601,20),
            "learning_rate": 10**RS.uniform(-2.3,-1.3,20),
            "max_depth": RS.randint(4,9,20),
            "colsample_bytree": RS.uniform(0.4,0.95,20)}

deep_xgb = {"n_estimators": RS.randint(800,1601,20),
            "learning_rate": 10**RS.uniform(-2.3,-1.3,20),
            "max_depth": RS.randint(4,9,20),
            "subsample": RS.uniform(0.7,1.0,20),
            "colsample_bytree": RS.uniform(0.4,0.95,20)}

cat_space = dict(
    depth          = range(4, 9),                          # 4 … 8
    learning_rate  = 10 ** RS.uniform(-2.3, -1.3, 1000),   # 1e-2.3 … 1e-1.3
    iterations     = range(800, 1601),                     # 800 … 1600
    l2_leaf_reg    = range(1, 6)                           # 1 … 5
)

deep_cat = list(           # ≈ 20 random draws → list-of-dicts
    ParameterSampler(cat_space, n_iter=20, random_state=SEED)
)

shallow_grids = {
    "LogReg": {"C":10**RS.uniform(-3,1,12)},
    "HGB":    {"learning_rate":10**RS.uniform(-2, -0.8,10),
               "max_depth": RS.randint(3,7,10)},
    "MLP":    {"hidden_layer_sizes":[(128,64),(256,128),(64,)],
               "alpha":10**RS.uniform(-5,-2,10)}
}


neg_ll = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
results, preds_store = [], {}


# ---------- Set A · raw ----------------------------------------------------
dfA_tr, dfA_te = train.copy(), test.copy()

# ---------- Set B · EDA-driven flags --------------------------------------
dfB_tr, dfB_te = dfA_tr.copy(), dfA_te.copy()
for col in RAW_COLS:                              # missing flags
    dfB_tr[f"{col}_miss"] = dfB_tr[col].isna().astype(int)
    dfB_te[f"{col}_miss"] = dfB_te[col].isna().astype(int)

thr = train["Time_spent_Alone"].mean() + 2*train["Time_spent_Alone"].std()
dfB_tr["high_alone_flag"] = (dfB_tr["Time_spent_Alone"] > thr).astype(int)
dfB_te["high_alone_flag"] = (dfB_te["Time_spent_Alone"] > thr).astype(int)

dfB_tr["activity_ratio"] = dfB_tr["Social_event_attendance"]/(dfB_tr["Time_spent_Alone"]+1)
dfB_te["activity_ratio"] = dfB_te["Social_event_attendance"]/(dfB_te["Time_spent_Alone"]+1)
dfB_tr["outside_ratio"]  = dfB_tr["Going_outside"]/(dfB_tr["Time_spent_Alone"]+1)
dfB_te["outside_ratio"]  = dfB_te["Going_outside"]/(dfB_te["Time_spent_Alone"]+1)

# ---------- Set C · B + all 2-gram interactions ----------------------------
dfC_tr, dfC_te = dfB_tr.copy(), dfB_te.copy()
for c1, c2 in itertools.combinations(RAW_COLS, 2):
    tag = f"{c1}-{c2}"
    dfC_tr[tag] = dfC_tr[c1].astype(str) + "_" + dfC_tr[c2].astype(str)
    dfC_te[tag] = dfC_te[c1].astype(str) + "_" + dfC_te[c2].astype(str)

# ---------- Set D · C + all 3-gram interactions ----------------------------
dfD_tr, dfD_te = dfC_tr.copy(), dfC_te.copy()
for c1, c2, c3 in itertools.combinations(RAW_COLS, 3):
    tag = f"{c1}-{c2}-{c3}"
    dfD_tr[tag] = dfD_tr[c1].astype(str) + "_" + dfD_tr[c2].astype(str) + "_" + dfD_tr[c3].astype(str)
    dfD_te[tag] = dfD_te[c1].astype(str) + "_" + dfD_te[c2].astype(str) + "_" + dfD_te[c3].astype(str)

# ---------- master dictionary ---------------------------------------------
feature_sets = {
    "A_raw" : (dfA_tr, dfA_te),
    "B_eda" : (dfB_tr, dfB_te),
    "C_50f" : (dfC_tr, dfC_te),
    "D_100f": (dfD_tr, dfD_te)
}
print({k: v[0].shape for k,v in feature_sets.items()})


for fs_name, (df_tr, df_te) in feature_sets.items():
    feature_cols = [c for c in df_tr.columns if c != "Personality"]
    cat_cols = [c for c in feature_cols if df_tr[c].dtype=="object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    pre = build_preproc(cat_cols,num_cols).fit(df_tr[feature_cols])
    X_tr = pre.transform(df_tr[feature_cols])
    X_te = pre.transform(df_te[feature_cols])

    ### model catalogue per feature set
    model_defs = {
        "RF":  (RandomForestClassifier(random_state=SEED, n_jobs=-1),
                deep_rf),
        "LGB": (lgb.LGBMClassifier(objective="binary", metric="binary_logloss",
                                   device_type="gpu", random_state=SEED,verbosity=-1),
                deep_lgb),
        "XGB": (xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                  tree_method="hist", random_state=SEED, device="cuda"),
                deep_xgb),
        "CAT": (CatBoostClassifier(loss_function="Logloss",
                                   task_type="GPU", random_state=SEED, verbose=False),
                deep_cat),
        # simple models
        "LogReg": (LogisticRegression(max_iter=400, n_jobs=-1),
                   shallow_grids["LogReg"]),
        "HGB":    (HistGradientBoostingClassifier(random_state=SEED),
                   shallow_grids["HGB"]),
        "MLP":    (MLPClassifier(max_iter=200, random_state=SEED),
                   shallow_grids["MLP"])
    }

    for mdl_name,(base_est, grid) in model_defs.items():
        tag = f"{fs_name}_{mdl_name}"
        t0  = time.perf_counter()

        rs  = RandomizedSearchCV(base_est, grid, n_iter=len(grid),
                                 scoring=neg_ll, cv=cv, n_jobs=-1,
                                 random_state=SEED, verbose=0)
        # CatBoost needs Pool for CV; fallback to manual loop
        if mdl_name == "CAT":
            best_ll, best_params = np.inf, None
            for params in tqdm(deep_cat, desc=f"Tune {tag}", leave=False):
                est = base_est.copy()
                est.set_params(**params)
                fold_ll = []
                for tr, va in cv.split(X_tr, y):
                    est.fit(Pool(X_tr.iloc[tr], y.iloc[tr], cat_features=None))
                    fold_ll.append(
                        log_loss(y.iloc[va],
                                 est.predict_proba(X_tr.iloc[va])[:, 1])
                    )
                if np.mean(fold_ll) < best_ll:
                    best_ll, best_params = np.mean(fold_ll), params
            best_est = base_est.set_params(**best_params)
        else:
            rs.fit(X_tr, y)
            best_est = rs.best_estimator_

        # final 5-fold OOF with best params -------------------------------
        oof = np.zeros(len(train)); preds = np.zeros(len(test))
        for tr,va in cv.split(X_tr,y):
            if mdl_name=="CAT":
                best_est.fit(Pool(X_tr.iloc[tr], y.iloc[tr], cat_features=None))
            else:
                best_est.fit(X_tr.iloc[tr], y.iloc[tr])
            oof[va]  = best_est.predict_proba(X_tr.iloc[va])[:,1]
            preds   += best_est.predict_proba(X_te)[:,1]/N_SPLITS

        ll = log_loss(y,oof)
        acc= accuracy_score(y,(oof>=0.5).astype(int))
        secs=int(time.perf_counter()-t0)

        # save artefacts --------------------------------------------------
        np.save(save_dir/f"oof_{tag}.npy", oof)
        pd.DataFrame({"id":test.index,
                      "Personality":np.where(preds>=0.5,"Introvert","Extrovert")}
                    ).to_csv(save_dir/f"sub_{tag}.csv",index=False)

        results.append(dict(Tag=tag, FS=fs_name, Model=mdl_name,
                            LL=round(ll,5), ACC=round(acc,5), Sec=secs))
        preds_store[tag]=preds
        print(f"{tag:<18}  LL={ll:.5f}  ACC={acc:.5f}  ({secs} s)")
        gc.collect()


res_df = pd.DataFrame(results).sort_values("LL")
display(res_df.head(20))

plt.figure(figsize=(10,6))
sns.barplot(x="LL", y="Tag", data=res_df.head(20), palette="crest")
plt.title("Top-20 tuned models – lower log-loss is better")
plt.tight_layout(); plt.show()


best_tag = res_df.iloc[0]["Tag"]
pd.read_csv(save_dir/f"sub_{best_tag}.csv").to_csv("submission.csv", index=False)
print("Best CV model:", best_tag, "→ submission.csv written")

