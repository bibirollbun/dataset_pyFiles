# %%bash
# pip -q install --upgrade "scikit-learn==1.5.2" "scikeras==0.12.0" tensorflow -U
# # quiet TF C++ logs
# export TF_CPP_MIN_LOG_LEVEL=2


import gc
import itertools
import joblib
import json
import logging 
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
import warnings

from catboost  import CatBoostRegressor
from lightgbm import LGBMRegressor
from pathlib import Path
#from scikeras.wrappers import KerasRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_predict, KFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tqdm import tqdm
from xgboost   import XGBRegressor

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
RANDOM_STATE = 42

logging.getLogger("lightgbm").setLevel(logging.CRITICAL)
logging.getLogger("xgboost").setLevel(logging.CRITICAL)


DATA_PATH = Path('/kaggle/input/playground-series-s5e5')
train = pd.read_csv(DATA_PATH/'train.csv')
test  = pd.read_csv(DATA_PATH/'test.csv')

# train = pd.read_csv('train.csv')
# test  = pd.read_csv('test.csv')

y = train['Calories']
X = train.drop(columns = ['Calories'])

le = LabelEncoder().fit(X['Sex'])
X['Sex']    = le.transform(X['Sex'])
test['Sex'] = le.transform(test['Sex'])

cv = KFold(n_splits = 5, shuffle = True, random_state = RANDOM_STATE)

def rmsle(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.log1p(np.clip(y_pred, 0, None)) -
                                     np.log1p(y_true))))
rmsle_scorer = make_scorer(rmsle, greater_is_better=False)
num_cols = X.columns.tolist()

def build_mlp(n_hidden=2, n_units=128, dropout=0.15, lr=1e-3,
              input_shape=(8,)):
    keras.backend.clear_session()
    x = inp = keras.Input(shape=input_shape)
    for _ in range(n_hidden):
        x = layers.Dense(n_units, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    return model

X.shape




def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lots of engineered, domain‑aware features."""
    out = df.copy()

    # ── core physiological add‑ons ──────────────────────────────────────────
    out["BMI"]        = out["Weight"] / ((out["Height"]/100) ** 2)
    out["WtHt_ratio"] = out["Weight"] / out["Height"]

    out["HR_pct_max"] = out["Heart_Rate"] / (220 - out["Age"])
    out["BMR"] = (
        10 * out["Weight"]
        + 6.25 * out["Height"]
        - 5 * out["Age"]
        + np.where(out["Sex"] == 1, 5, -161)
    )
    out["MET"] = 0.014 * out["Heart_Rate"] + 0.438    # HR → metabolic equiv.

    # ── HR‑intensity buckets (categorical zones) ────────────────────────────
    out["HR_zone"] = pd.cut(
        out["HR_pct_max"],
        bins=[0, .5, .68, .85, 1],
        labels=["Rest", "FatBurn", "Cardio", "Peak"],
        include_lowest=True,
    )
    out = pd.concat([out, pd.get_dummies(out["HR_zone"], prefix="HRzone")], axis=1)

    # ── age & BMI buckets  (niche cohorts) ─────────────────────────────────
    out["Age_bin"] = pd.cut(
        out["Age"],
        bins=[0, 18, 25, 35, 45, 55, 65, 100],
        labels=["<18", "18‑25", "25‑35", "35‑45", "45‑55", "55‑65", "65+"],
    )
    out = pd.concat([out, pd.get_dummies(out["Age_bin"], prefix="AgeB")], axis=1)

    out["BMI_bin"] = pd.cut(
        out["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Under", "Normal", "Over", "Obese"],
    )
    out = pd.concat([out, pd.get_dummies(out["BMI_bin"], prefix="BMIB")], axis=1)

    # ── numeric transforms (square / cube / log) ───────────────────────────
    num_feats = [
        "Age", "Height", "Weight", "Duration",
        "Heart_Rate", "Body_Temp",  
        "BMI", "HR_pct_max", "BMR", "MET"
    ]
    for c in num_feats:
        out[f"{c}_sq"]   = out[c] ** 2
        out[f"{c}_cube"] = out[c] ** 3
        out[f"log_{c}"]  = np.log1p(out[c])

    # ── pairwise interactions (product + ratio) ────────────────────────────
    for c1, c2 in itertools.combinations(num_feats, 2):
        out[f"{c1}_x_{c2}"]   = out[c1] * out[c2]
        out[f"{c1}_div_{c2}"] = out[c1] / (out[c2] + 1e-3)

    # drop helper categoricals (dummies already kept)
    out.drop(columns=["HR_zone", "Age_bin", "BMI_bin"], inplace=True)

    return out

# ▸ apply to both sets
X   = enrich(X)
test = enrich(test)

print(f"✅ Feature‑engineering done – train: {X.shape},  test: {test.shape}")
print("Total added features count:", X.shape[1] - 8)


model_space = {
    # "Ridge": (
    #     Ridge(random_state=RANDOM_STATE),
    #     dict(alpha=loguniform(1e-2, 1e3))
    # ),

    # "Lasso": (
    #     Lasso(random_state=RANDOM_STATE, max_iter=5000),
    #     dict(alpha=loguniform(1e-3, 10))
    # ),

    # "ElasticNet": (
    #     ElasticNet(random_state=RANDOM_STATE, max_iter=5000),
    #     dict(alpha=loguniform(1e-3, 10), l1_ratio=uniform(0, 1))
    # ),

    # # "RandomForest": (
    # #     RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    # #     dict(n_estimators=randint(200, 600),
    # #          max_depth=randint(6, 20),
    # #          min_samples_leaf=randint(1, 5))
    # # ),

    # # "ExtraTrees": (
    # #     ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    # #     dict(n_estimators=randint(200, 600),
    # #          max_depth=randint(6, 20),
    # #          min_samples_leaf=randint(1, 5))
    # # ),

    # "HistGBR": (
    #     HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    #     dict(max_depth=randint(3, 10),
    #          learning_rate=loguniform(1e-3, 0.2),
    #          max_iter=randint(200, 800))
    #),

    # "GradientBoosting": (
    #     GradientBoostingRegressor(random_state=RANDOM_STATE),
    #     dict(learning_rate=loguniform(1e-3, 0.2),
    #          n_estimators=randint(200, 800),
    #          max_depth=randint(3, 6))
    # ),

    # "KNN": (
    #     KNeighborsRegressor(),
    #     dict(n_neighbors=randint(3, 25), weights=['uniform', 'distance'])
    # ),

    # GPU‑CAPABLE
    "XGBoost": (
        XGBRegressor(
            objective='reg:squarederror',
            tree_method='gpu_hist',
            predictor='gpu_predictor',
            random_state=RANDOM_STATE,
            n_jobs=2,
            verbosity=0 ),
        dict(n_estimators=randint(400, 1200),
             max_depth=randint(4, 10),
             learning_rate=loguniform(0.01, 0.2),
             subsample=uniform(0.6, 0.4),
             colsample_bytree=uniform(0.6, 0.4))
    ),

    "LightGBM": (
        LGBMRegressor(
            objective='rmse',
            device_type='gpu',
            gpu_use_dp=False,
            random_state=RANDOM_STATE,
            n_jobs=2,
            verbosity=-1 ),
        dict(n_estimators=randint(400, 1200),
             max_depth=randint(-1, 10),
             learning_rate=loguniform(0.01, 0.2),
             num_leaves=randint(31, 300),
             subsample=uniform(0.6, 0.4),
             colsample_bytree=uniform(0.6, 0.4))
    ),

    # "CatBoost": (
    #     CatBoostRegressor(
    #         loss_function='RMSE',
    #         task_type='GPU',
    #         random_state=RANDOM_STATE,
    #         verbose=0),
    #     dict(iterations=randint(400, 1200),
    #          depth=randint(4, 10),
    #          learning_rate=loguniform(0.01, 0.2),
    #          l2_leaf_reg=loguniform(1, 10))
    # ),

    # — lean base estimator —
    "CatBoost": (
    CatBoostRegressor(
        task_type="GPU",
        devices="0",
        loss_function="RMSE",
        random_state=RANDOM_STATE,
        allow_writing_files=False,
        verbose=False,          # keep notebook clean
        max_bin=64,             # OK on GPU
        gpu_ram_part=0.20,      # leave head‑room
        bootstrap_type="Bernoulli",
        grow_policy="Depthwise"
    ),
    dict(
        iterations    = randint(300, 800),
        depth         = randint(4, 8),
        learning_rate = loguniform(0.03, 0.15),
        l2_leaf_reg   = loguniform(1, 6),
        subsample     = uniform(0.65, 0.25)   # SAFE – row‑sampling
    ))
    
 #     "KerasMLP_GPU": (
 #     KerasRegressor(
 #         model        = build_mlp,
 #         verbose      = 0,
 #         random_state = RANDOM_STATE,

 #         # —— critical line ——
 #         metrics      = None          # prevents the 'loss' metric bug
 #     ),
 #     {
 #         "model__n_hidden" : randint(1, 4),
 #         "model__n_units"  : randint(64, 256),
 #         "model__dropout"  : uniform(0.05, 0.25),
 #         "model__lr"       : loguniform(1e-4, 3e-3),
 #         "fit__epochs"     : randint(25, 60),
 #         "fit__batch_size" : randint(1024, 4096)
 #     }
 # )
}


best_models, results, oof_preds = {}, [], {}

for name, (base_est, param_dist) in model_space.items():
    print(f"\n➡️ {name}")
    pipe = Pipeline([("scale", StandardScaler()), ("est", base_est)])

    # ▼  use n_jobs = 1  ONLY for the Keras search
    local_n_jobs = 1 if name == "KerasMLP_GPU" else -1
    search = RandomizedSearchCV(
        pipe,
        param_distributions={f"est__{k}": v for k, v in param_dist.items()},
        n_iter=25, cv=cv, scoring=rmsle_scorer,
        n_jobs=local_n_jobs, random_state=RANDOM_STATE, verbose=0
    )

    t0 = time.time()
    search.fit(X, y)
    runtime = (time.time() - t0) / 60
    cv_rmsle = -search.best_score_
    print(f"   best RMSLE = {cv_rmsle:.4f} | time = {runtime:.1f} min")

    # ▼ Safe serial OOF for Keras (no multiprocessing)
    if name == "KerasMLP_GPU":
        oof = np.zeros(len(y))
        for tr, vl in cv.split(X):
            est = search.best_estimator_
            est.fit(X.iloc[tr], y.iloc[tr])
            oof[vl] = est.predict(X.iloc[vl])
    else:
        oof = cross_val_predict(search.best_estimator_, X, y,
                                cv=cv, method="predict", n_jobs=-1)

    # save artefacts
    best_models[name], oof_preds[name] = search.best_estimator_, oof
    pd.DataFrame({"id": train.index, f"oof_{name}": oof}).to_csv(f"oof_{name}.csv", index=False)
    results.append({"Model": name, "RMSLE_CV": cv_rmsle, "Time_min": runtime})
    gc.collect()

res_df = pd.DataFrame(results).sort_values("RMSLE_CV")


res_df


fig, ax = plt.subplots(figsize=(12,6))

# bar = CV RMSLE
sns.barplot(
    data   = res_df,
    x      = "Model",
    y      = "RMSLE_CV",
    palette= "viridis",
    ax     = ax
)
ax.set_ylabel("RMSLE (CV)")
ax.set_xlabel("")
ax.set_title("Cross‑validated RMSLE – lower is better")

# line = tuning time (on the secondary y‑axis)
ax2 = ax.twinx()
sns.lineplot(
    data   = res_df,
    x      = "Model",
    y      = "Time_min",
    marker = "o",
    linewidth = 2,
    color  = "red",
    ax     = ax2
)
ax2.set_ylabel("Tuning time (minutes)", color="red")
ax2.tick_params(axis="y", labelcolor="red")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


oof_df = pd.DataFrame({"id": train.index})
for mdl, vec in oof_preds.items():
    oof_df[mdl] = vec

# ▸ gives a tidy file we can load for stacking / hill‑climbing
oof_df.to_csv("oof_predictions_pg_s5e5.csv", index=False)
print("Saved: oof_predictions_pg_s5e5.csv   (shape:", oof_df.shape, ")")


SUB_PATH = Path("./subs")
SUB_PATH.mkdir(exist_ok=True)

for mdl, est in best_models.items():
    preds = est.predict(test)
    preds = np.clip(preds, 0, None)
    sub = pd.DataFrame({
        "id":   test['id'],   # Kaggle PS S5E5 uses the row index as id
        "Calories": preds
    })

    fname = SUB_PATH / f"sub_{mdl}.csv"
    sub.to_csv(fname, index=False)
    print(f"Wrote {fname.name:<25}  →  {sub.shape[0]} rows")


feature_names = X.columns.tolist()          
id2name       = {f"f{i}": col for i, col in enumerate(feature_names)}

xgb_best  = best_models["XGBoost"].named_steps["est"]   # from Pipeline
booster   = xgb_best.get_booster()
gain_raw  = booster.get_score(importance_type="gain")   # {'f0': …}

gain_named = {id2name.get(fid, fid): val for fid, val in gain_raw.items()}

imp_df = (pd.Series(gain_named, name="Gain")
            .rename_axis("Feature")
            .reset_index()
            .sort_values("Gain", ascending=False)
            .head(15))

print("Top features by *gain* (XGBoost)")
display(imp_df.style.format({"Gain":"{:.1f}"}))


# optional bar plot
plt.figure(figsize=(8,5))
sns.barplot(data=imp_df, y="Feature", x="Gain", palette="viridis")
plt.title("XGBoost feature importance (gain)")
plt.xlabel("Average gain")
plt.ylabel("")
plt.tight_layout()
plt.show()

