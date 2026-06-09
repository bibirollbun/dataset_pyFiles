# pack_c_generator.py  (â‰ˆ 2 min CPU)
import pandas as pd, numpy as np, itertools, pathlib as pl
DATA_DIR = "/kaggle/input/playground-series-s5e5"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

def add_packC(df):
    # Sex Ã— 5-year AgeGroup stats
    df["AgeGroup"] = (df["Age"]//5)*5
    grp = df.groupby(["Sex","AgeGroup"])
    for col in ["Heart_Rate","Body_Temp","Duration","Weight"]:
        stat = grp[col].transform("mean")
        df[f"{col}_sg_mean"] = stat
        df[f"{col}_sg_diff"] = df[col] - stat
    # Poly deg 2 for main numeric cols
    base = ["Duration","Heart_Rate","Body_Temp","BMI"]
    for a,b in itertools.combinations_with_replacement(base,2):
        df[f"{a}_x_{b}"] = df[a]*df[b]
    return df

for d in (train,test):
    d["BMI"] = d.Weight/(d.Height/100.0)**2
train_c, test_c = add_packC(train.copy()), add_packC(test.copy())
train_c.to_parquet("packC_train.parquet"); test_c.to_parquet("packC_test.parquet")



import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from pathlib import Path

torch.backends.cudnn.benchmark = True

def rmsle_t(pred, target):
    return torch.sqrt(F.mse_loss(torch.log1p(pred), torch.log1p(target)))

# ------------------------------------------------------------------
# 0.  Hyper-params
# ------------------------------------------------------------------
PACK_DIR      = "/kaggle/input/playground-series-s5e5"      # path to Pack C
FOLDS         = 5
SEED          = 42
HIDDEN_DIM    = 512
N_LAYERS      = 4
LR            = 5e-4
BATCH         = 32_768
EPOCHS        = 120
PATIENCE      = 12            # early-stop
WEIGHT_DECAY  = 1e-5

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------
# 1.  Load + preprocess data
# ------------------------------------------------------------------
train = pd.read_csv(f"{PACK_DIR}/train.csv")
test  = pd.read_csv(f"{PACK_DIR}/test.csv")

# keep sex masks
train_sex = train['Sex'].map({'male':0,'female':1}).values.astype('int8')
test_sex  = test['Sex'].map({'male':0,'female':1}).values.astype('int8')

y_true = train['Calories'].values.astype('float32')

# drop id, target, sex from features
train_feats = train.drop(columns=["id","Calories","Sex"])
test_feats  = test.drop(columns=["id","Sex"])

# z-score normalization on train
mean, std = train_feats.values.mean(0), train_feats.values.std(0) + 1e-6
X_train_np = (train_feats.values - mean) / std
X_test_np  = (test_feats.values - mean) / std

# torch tensors
X_train = torch.from_numpy(X_train_np).float().to(device)
y_train = torch.from_numpy(y_true).float().to(device)
X_test  = torch.from_numpy(X_test_np).float().to(device)

# ------------------------------------------------------------------
# 2.  Model defs
# ------------------------------------------------------------------
class AttentionEnsemble(nn.Module):
    def __init__(self, dim, gamma=0.96875, use_groupnorm=True, num_groups=8):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key   = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.gn    = nn.GroupNorm(num_groups, dim) if use_groupnorm else nn.Identity()

    def forward(self, x):
        Q, K, V = self.query(x), self.key(x), self.value(x)
        logits  = self.gamma * (Q * K)
        return self.gn(torch.sigmoid(logits) * V)

class AttentionNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.blocks = nn.ModuleList([AttentionEnsemble(hidden_dim) for _ in range(n_layers)])
        self.head   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        x = self.encoder(x)
        for blk in self.blocks:
            x = x + blk(x)
        return F.relu(self.head(x).squeeze())

# ------------------------------------------------------------------
# 3.  Per-sex CV & training
# ------------------------------------------------------------------
kf = KFold(FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train), dtype="float32")
test_pred = np.zeros(len(test), dtype="float32")

for sex in [0, 1]:
    # indices for this sex
    tr_idx_all = np.where(train_sex == sex)[0]
    te_idx_all = np.where(test_sex  == sex)[0]

    # skip if no samples
    if len(tr_idx_all) == 0: continue

    # accumulate sex-specific test preds
    sex_test_pred = torch.zeros(len(te_idx_all), device=device)

    # CV loop within this sex
    for fold, (tr_idx, va_idx) in enumerate(kf.split(tr_idx_all), 1):
        torch.cuda.empty_cache()
        net = AttentionNN(X_train.shape[1]).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        scaler = torch.cuda.amp.GradScaler()

        best_loss, stalls = 1e9, 0
        # map to global indices
        train_idx = tr_idx_all[tr_idx]
        valid_idx = tr_idx_all[va_idx]

        for epoch in range(EPOCHS):
            net.train()
            perm = torch.randperm(len(train_idx), device=device)
            for i in range(0, len(train_idx), BATCH):
                batch_idx = train_idx[perm[i:i+BATCH].cpu()]
                with torch.cuda.amp.autocast():
                    pred = net(X_train[batch_idx])
                    loss = rmsle_t(pred, y_train[batch_idx])
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            sched.step()

            # validation
            net.eval()
            with torch.no_grad(), torch.cuda.amp.autocast():
                val_pred = net(X_train[valid_idx])
                val_loss = rmsle_t(val_pred, y_train[valid_idx]).item()
                print(epoch, val_loss)

            if val_loss < best_loss - 1e-4:
                best_loss = val_loss; stalls = 0
                best_w = {k:v.cpu().clone() for k,v in net.state_dict().items()}
            else:
                stalls += 1
            if stalls >= PATIENCE:
                break

        # load best and predict
        net.load_state_dict(best_w)
        # OOF
        with torch.no_grad(), torch.cuda.amp.autocast():
            oof_preds = net(X_train[valid_idx]).cpu().numpy().astype('float32')
        oof[valid_idx] = oof_preds
        # test for this fold
        with torch.no_grad(), torch.cuda.amp.autocast():
            sex_test_pred += net(X_test[te_idx_all]).float() / FOLDS
        torch.cuda.empty_cache()
        print(f"sex={sex} fold {fold} best RMSLE {best_loss:.4f}")

    # assign sex-specific test preds
    test_pred[te_idx_all] = sex_test_pred.cpu().numpy()

# ------------------------------------------------------------------
# 4.  Save & report
# ------------------------------------------------------------------
Path("oof").mkdir(exist_ok=True)
Path("testpred").mkdir(exist_ok=True)
np.save("oof/gated_attn_per_sex.npy", oof)
np.save("testpred/gated_attn_per_sex.npy", test_pred)

print("OOF RMSLE:", mean_squared_log_error(y_true, oof, squared=False))
print("âœ… saved oof/gated_attn_per_sex.npy & testpred/gated_attn_per_sex.npy")



sub = pd.DataFrame({"id":test.id,
                    "Calories": np.clip(np.load("testpred/gated_attn_per_sex.npy"), 1, 314)})
sub.to_csv("submission_dated_attn_per_sex.csv", index=False)


sub


pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb, json, csv, os, warnings
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from scipy.optimize import nnls
from pathlib import Path, PurePath
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
# 1) CONFIG
# -------------------------------------------------------------------
DATA_DIR   = "/kaggle/input/playground-series-s5e5"
N_SPLITS   = 5
SEED       = 42
CAT_COLS   = ["Sex"]
NUM_COLS   = ["Age","Height","Weight","Duration","Heart_Rate","Body_Temp"]

LGB_PAR = dict(
    objective='rmse', metric='rmse', learning_rate=0.03,
    n_estimators=2000, num_leaves=255, subsample=0.8,
    colsample_bytree=0.7, feature_fraction=0.7,
    reg_alpha=0.1, device='gpu', seed=SEED
)
LGB_PAR_RATIO = LGB_PAR.copy()
LGB_PAR_RATIO.update(num_leaves=511, min_child_samples=20)

CAT_PAR = dict(
    iterations=1500, learning_rate=0.05, depth=10,
    bagging_temperature=1.5, l2_leaf_reg=6,
    task_type='GPU', loss_function='RMSE', eval_metric='RMSE',
    random_seed=SEED, verbose=100
)

# -------------------------------------------------------------------
# 2) LOAD + FEATURE ENGINEERING
# -------------------------------------------------------------------
train = pd.read_csv(PurePath(DATA_DIR, "train.csv"))
test  = pd.read_csv(PurePath(DATA_DIR, "test.csv"))

def add_feats(df):
    df["BMI"]        = df.Weight/(df.Height/100)**2
    df["Workload"]   = df.Weight*df.Duration
    df["MET"]        = df.Heart_Rate/60*df.Duration
    df["TempDev"]    = df.Body_Temp-37
    df["HR_per_min"] = df.Heart_Rate/df.Duration.clip(lower=1e-3)
    df["Age_HR"]     = df.Age*df.Heart_Rate
    base = NUM_COLS
    df["row_mean"]   = df[base].mean(1)
    df["row_std"]    = df[base].std(1)
    df["row_max"]    = df[base].max(1)
    df["row_min"]    = df[base].min(1)
    return df

train = add_feats(train.copy())
test  = add_feats(test.copy())
for d in (train, test):
    d["Sex"] = d.Sex.astype("category")

FEATS_FULL  = [c for c in train.columns if c not in ["id","Calories"]]
FEATS_NODUR = [c for c in FEATS_FULL if c!="Duration"]

# prepare output arrays
oof_cat   = np.zeros(len(train))
oof_lgb   = np.zeros(len(train))
oof_nodur = np.zeros(len(train))
oof_ratio = np.zeros(len(train))
test_cat   = np.zeros(len(test))
test_lgb   = np.zeros(len(test))
test_nodur = np.zeros(len(test))
test_ratio = np.zeros(len(test))

# -------------------------------------------------------------------
# 3) SEX-WISE STACKING + BLEND
# -------------------------------------------------------------------
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
for sex in train.Sex.cat.categories:
    # indices
    idx_tr = train.index[train.Sex == sex]
    idx_te = test.index[test.Sex == sex]
    if len(idx_tr)==0: continue

    # group-specific predictions
    grp_oof_cat   = np.zeros(len(idx_tr))
    grp_oof_lgb   = np.zeros(len(idx_tr))
    grp_oof_nodur = np.zeros(len(idx_tr))
    grp_oof_ratio = np.zeros(len(idx_tr))
    grp_test_cat   = np.zeros(len(idx_te))
    grp_test_lgb   = np.zeros(len(idx_te))
    grp_test_nodur = np.zeros(len(idx_te))
    grp_test_ratio = np.zeros(len(idx_te))

    y_true_grp = train.Calories.values[idx_tr]
    logy = np.log1p(y_true_grp)

    # CV within sex
    for fold, (tr, va) in enumerate(kf.split(idx_tr),1):
        tr_idx = idx_tr[tr]
        va_idx = idx_tr[va]
        X_tr, X_va = train.loc[tr_idx], train.loc[va_idx]
        y_tr, y_va = np.log1p(X_tr.Calories), np.log1p(X_va.Calories)

        # CatBoost
        cb = CatBoostRegressor(**CAT_PAR)
        cb.fit(X_tr[FEATS_FULL], y_tr,
               eval_set=(X_va[FEATS_FULL], y_va),
               cat_features=[FEATS_FULL.index(c) for c in CAT_COLS])
        grp_oof_cat[va] = np.expm1(cb.predict(X_va[FEATS_FULL]))
        grp_test_cat  += np.expm1(cb.predict(test.loc[idx_te, FEATS_FULL]))/N_SPLITS

        # LGB full
        lg = lgb.LGBMRegressor(**LGB_PAR, early_stopping_rounds=100)
        lg.fit(X_tr[FEATS_FULL], y_tr,
               eval_set=[(X_va[FEATS_FULL], y_va)],
               categorical_feature=CAT_COLS)
        grp_oof_lgb[va] = np.expm1(lg.predict(X_va[FEATS_FULL]))
        grp_test_lgb  += np.expm1(lg.predict(test.loc[idx_te, FEATS_FULL]))/N_SPLITS

        # LGB no Duration
        lg_nd = lgb.LGBMRegressor(**LGB_PAR, early_stopping_rounds=100)
        lg_nd.fit(X_tr[FEATS_NODUR], y_tr,
                  eval_set=[(X_va[FEATS_NODUR], y_va)],
                  categorical_feature=CAT_COLS)
        grp_oof_nodur[va] = np.expm1(lg_nd.predict(X_va[FEATS_NODUR]))
        grp_test_nodur += np.expm1(lg_nd.predict(test.loc[idx_te, FEATS_NODUR]))/N_SPLITS

        # Ratio model
        ratio_tr = X_tr.Calories / X_tr.Duration.clip(lower=1e-3)
        ratio_va = X_va.Calories / X_va.Duration.clip(lower=1e-3)
        lg_rt = lgb.LGBMRegressor(**LGB_PAR_RATIO, early_stopping_rounds=100)
        lg_rt.fit(X_tr[FEATS_FULL], np.log1p(ratio_tr),
                  eval_set=[(X_va[FEATS_FULL], np.log1p(ratio_va))],
                  categorical_feature=CAT_COLS)
        grp_oof_ratio[va] = np.expm1(lg_rt.predict(X_va[FEATS_FULL])) * X_va.Duration.values
        grp_test_ratio   += np.expm1(lg_rt.predict(test.loc[idx_te, FEATS_FULL])) * test.loc[idx_te, 'Duration'].values / N_SPLITS

    # blend per sex
    log_preds = np.log1p(np.column_stack([grp_oof_cat, grp_oof_lgb, grp_oof_nodur, grp_oof_ratio]))
    weights = nnls(log_preds, np.log1p(y_true_grp))[0]
    weights /= weights.sum()
    grp_blend_oof = (np.column_stack([grp_oof_cat, grp_oof_lgb, grp_oof_nodur, grp_oof_ratio]) * weights).sum(1)
    grp_blend_test = (np.column_stack([grp_test_cat, grp_test_lgb, grp_test_nodur, grp_test_ratio]) * weights).sum(1)

    # assign to global
    oof_cat[idx_tr]   = grp_oof_cat
    oof_lgb[idx_tr]   = grp_oof_lgb
    oof_nodur[idx_tr] = grp_oof_nodur
    oof_ratio[idx_tr] = grp_oof_ratio
    test_cat[idx_te]   = grp_test_cat
    test_lgb[idx_te]   = grp_test_lgb
    test_nodur[idx_te] = grp_test_nodur
    test_ratio[idx_te] = grp_test_ratio

    # save per-sex weights
    with open(f"blend_weights_{sex}.json", 'w') as f:
        json.dump(dict(zip(['cat','lgb','nodur','ratio'], weights.tolist())), f)
    print(f"Sex={sex} blend weights", np.round(weights,3))
    print(f"Sex={sex} OOF RMSLE: {mean_squared_log_error(train.Calories.values[idx_tr], grp_blend_oof, squared=False):.6f}")

# -------------------------------------------------------------------
# 4) SAVE GLOBAL ARTIFACTS + COMBINED SUBMISSION
# -------------------------------------------------------------------
Path("oof").mkdir(exist_ok=True)
Path("testpred").mkdir(exist_ok=True)
for m in ['cat','lgb','nodur','ratio']:
    np.save(f"oof/{m}.npy", globals()[f"oof_{m}"])
    np.save(f"testpred/{m}.npy", globals()[f"test_{m}"])
# combined global blend
log_preds_all = np.log1p(np.column_stack([oof_cat, oof_lgb, oof_nodur, oof_ratio]))
global_weights = nnls(log_preds_all, np.log1p(train.Calories.values))[0]
global_weights /= global_weights.sum()
global_blend = (np.column_stack([oof_cat, oof_lgb, oof_nodur, oof_ratio]) * global_weights).sum(1)
np.save("oof/blend.npy", global_blend)
np.save("testpred/blend.npy", (np.column_stack([test_cat, test_lgb, test_nodur, test_ratio]) * global_weights).sum(1))
sub = pd.DataFrame({"id": test.id, "Calories": np.load("testpred/blend.npy")})
sub.to_csv("submission_per_sex.csv", index=False)
print("âœ… Saved per-sex & global artifacts.")



# ============================================================
#  sex_routed_stack.py
# ============================================================

import numpy as np, pandas as pd, lightgbm as lgb, json, warnings
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from scipy.optimize import nnls
from pathlib import Path, PurePath

warnings.filterwarnings("ignore")

# -------------------- 1) CONFIG ----------------------------
DATA_DIR    = "/kaggle/input/playground-series-s5e5"
N_SPLITS    = 5
SEED        = 42
CAT_COLS    = ["Sex"]
NUM_COLS    = ["Age","Height","Weight","Duration","Heart_Rate","Body_Temp"]

LGB_PAR = dict(
    objective="regression_l2", metric="rmse", learning_rate=0.03,
    n_estimators=2000, num_leaves=255, subsample=0.8,
    colsample_bytree=0.7, feature_fraction=0.7,
    reg_alpha=0.1, device="gpu", random_state=SEED
)
LGB_PAR_RATIO = LGB_PAR.copy()
LGB_PAR_RATIO.update(num_leaves=511, min_child_samples=20)

CAT_PAR = dict(
    iterations=1500, learning_rate=0.05, depth=10,
    bagging_temperature=1.5, l2_leaf_reg=6,
    task_type="GPU", loss_function="RMSE", eval_metric="RMSE",
    random_seed=SEED, verbose=100
)

# -------------------- 2) LOAD + FE -------------------------
def add_feats(df):
    df["BMI"]        = df.Weight/(df.Height/100)**2
    df["Workload"]   = df.Weight*df.Duration
    df["MET"]        = df.Heart_Rate/60*df.Duration
    df["TempDev"]    = df.Body_Temp-37
    df["HR_per_min"] = df.Heart_Rate/df.Duration.clip(1e-3)
    df["Age_HR"]     = df.Age*df.Heart_Rate
    base = NUM_COLS
    df["row_mean"] = df[base].mean(axis=1)
    df["row_std"]  = df[base].std(axis=1)
    df["row_max"]  = df[base].max(axis=1)
    df["row_min"]  = df[base].min(axis=1)
    return df

train = pd.read_csv(PurePath(DATA_DIR,"train.csv"))
test  = pd.read_csv(PurePath(DATA_DIR,"test.csv"))
train = add_feats(train)
test  = add_feats(test)

# ensure Sex is categorical
train["Sex"] = train.Sex.astype("category")
test ["Sex"] = test.Sex.astype("category")

FEATS_FULL  = [c for c in train.columns if c not in ["id","Calories"]]
FEATS_NODUR = [c for c in FEATS_FULL if c!="Duration"]

# placeholders for global OOF/test
final_oof  = np.zeros(len(train))
final_test = np.zeros(len(test))

# (optional) keep per-model arrays too
oof_cat, oof_lgb, oof_nodur, oof_ratio = \
    [np.zeros(len(train)) for _ in range(4)]
test_cat, test_lgb, test_nodur, test_ratio = \
    [np.zeros(len(test))  for _ in range(4)]

# -------------------- 3) SEX-WISE CV & BLEND -----------------
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for sex in train.Sex.cat.categories:
    # row indices by sex
    idx_tr = train.index[train.Sex==sex].to_list()
    idx_te = test.index[test.Sex==sex].to_list()
    if not idx_tr:
        continue

    # male: only ratio model
    if sex == "male":
        grp_oof_ratio = np.zeros(len(idx_tr))
        grp_test_ratio = np.zeros(len(idx_te))

        for tr,va in kf.split(idx_tr):
            tr_idx = [idx_tr[i] for i in tr]
            va_idx = [idx_tr[i] for i in va]

            X_tr = train.loc[tr_idx, FEATS_FULL]
            X_va = train.loc[va_idx, FEATS_FULL]
            # build ratio target
            ratio_tr = train.Calories.loc[tr_idx] / train.Duration.loc[tr_idx].clip(1e-3)
            ratio_va = train.Calories.loc[va_idx] / train.Duration.loc[va_idx].clip(1e-3)

            model = lgb.LGBMRegressor(**LGB_PAR_RATIO, early_stopping_rounds=100)
            model.fit(
                X_tr, np.log1p(ratio_tr),
                eval_set=[(X_va, np.log1p(ratio_va))],
                categorical_feature=CAT_COLS,
                
            )
            # OOF
            pred_va = np.expm1(model.predict(X_va)) * train.Duration.loc[va_idx].values
            grp_oof_ratio[[*range(len(va))]] = pred_va  # fill in order
            # test
            te_X = test.loc[idx_te, FEATS_FULL]
            grp_test_ratio += (np.expm1(model.predict(te_X)) * test.Duration.loc[idx_te].values) / N_SPLITS

        # scatter
        final_oof[idx_tr]  = grp_oof_ratio
        final_test[idx_te] = grp_test_ratio
        # keep per-model if needed
        oof_ratio[idx_tr]   = grp_oof_ratio
        test_ratio[idx_te]  = grp_test_ratio

        print(f"[male] ratio-model OOF RMSLE:",
              mean_squared_log_error(train.Calories.loc[idx_tr], grp_oof_ratio, squared=False))

    # female: blend of all 4
    else:
        grp_oof = {m:np.zeros(len(idx_tr)) for m in ["cat","lgb","nodur","ratio"]}
        grp_te  = {m:np.zeros(len(idx_te)) for m in ["cat","lgb","nodur","ratio"]}

        for tr,va in kf.split(idx_tr):
            tr_idx = [idx_tr[i] for i in tr]
            va_idx = [idx_tr[i] for i in va]

            X_tr_f = train.loc[tr_idx]
            X_va_f = train.loc[va_idx]

            y_tr = np.log1p(X_tr_f.Calories)
            y_va = np.log1p(X_va_f.Calories)

            # 1) CatBoost
            cb = CatBoostRegressor(**CAT_PAR)
            cb.fit(
                X_tr_f[FEATS_FULL], y_tr,
                eval_set=(X_va_f[FEATS_FULL], y_va),
                cat_features=[FEATS_FULL.index(c) for c in CAT_COLS],
                use_best_model=True
            )
            grp_oof["cat"][va] += np.expm1(cb.predict(X_va_f[FEATS_FULL]))
            grp_te ["cat"]   += np.expm1(cb.predict(test.loc[idx_te,FEATS_FULL]))/N_SPLITS

            # 2) LGBM full
            lg = lgb.LGBMRegressor(**LGB_PAR, early_stopping_rounds=100)
            lg.fit(
                X_tr_f[FEATS_FULL], y_tr,
                eval_set=[(X_va_f[FEATS_FULL], y_va)],
                categorical_feature=CAT_COLS,
                
                
            )
            grp_oof["lgb"][va] += np.expm1(lg.predict(X_va_f[FEATS_FULL]))
            grp_te ["lgb"]   += np.expm1(lg.predict(test.loc[idx_te,FEATS_FULL]))/N_SPLITS

            # 3) LGBM no-Duration
            lg2 = lgb.LGBMRegressor(**LGB_PAR, early_stopping_rounds=100)
            lg2.fit(
                X_tr_f[FEATS_NODUR], y_tr,
                eval_set=[(X_va_f[FEATS_NODUR], y_va)],
                categorical_feature=CAT_COLS,
              
            )
            grp_oof["nodur"][va] += np.expm1(lg2.predict(X_va_f[FEATS_NODUR]))
            grp_te ["nodur"] += np.expm1(lg2.predict(test.loc[idx_te,FEATS_NODUR]))/N_SPLITS

            # 4) LGBM ratio
            ratio_tr = X_tr_f.Calories / X_tr_f.Duration.clip(1e-3)
            ratio_va = X_va_f.Calories / X_va_f.Duration.clip(1e-3)
            lg_rt = lgb.LGBMRegressor(**LGB_PAR_RATIO, early_stopping_rounds=100)
            lg_rt.fit(
                X_tr_f[FEATS_FULL], np.log1p(ratio_tr),
                eval_set=[(X_va_f[FEATS_FULL], np.log1p(ratio_va))],
                categorical_feature=CAT_COLS,
                
            )
            pr_va = np.expm1(lg_rt.predict(X_va_f[FEATS_FULL])) * X_va_f.Duration.values
            grp_oof["ratio"][va] += pr_va
            te_rt = np.expm1(lg_rt.predict(test.loc[idx_te,FEATS_FULL])) * test.Duration.loc[idx_te].values
            grp_te["ratio"] += te_rt / N_SPLITS

        # stack OOF matrix & solve NNLS
        M_oof = np.column_stack([grp_oof[m] for m in ["cat","lgb","nodur","ratio"]])
        logy  = np.log1p(train.Calories.loc[idx_tr].values)
        w,_   = nnls(np.log1p(M_oof), logy)
        w    /= w.sum()

        # build blended preds
        blend_oof  = (M_oof * w).sum(1)
        M_te       = np.column_stack([grp_te[m] for m in ["cat","lgb","nodur","ratio"]])
        blend_test = (M_te * w).sum(1)

        # scatter
        final_oof[idx_tr]  = blend_oof
        final_test[idx_te] = blend_test

        # optionally keep per-model
        for m in ["cat","lgb","nodur","ratio"]:
            globals()[f"oof_{m}"][idx_tr]   = grp_oof[m]
            globals()[f"test_{m}"][idx_te]  = grp_te[m]

        # save female weights
        with open(f"blend_weights_female.json","w") as f:
            json.dump(dict(zip(["cat","lgb","nodur","ratio"], w.tolist())), f)

        print(f"[female] blend OOF RMSLE:",
              mean_squared_log_error(train.Calories.loc[idx_tr], blend_oof, squared=False))
        print("female weights:", np.round(w,3))

# -------------------- 4) SAVE & SUBMIT -------------------------
Path("oof").mkdir(exist_ok=True)
Path("testpred").mkdir(exist_ok=True)

# save per-model if you like:
for m in ["cat","lgb","nodur","ratio"]:
    np.save(f"oof/{m}.npy", globals()[f"oof_{m}"])
    np.save(f"testpred/{m}.npy", globals()[f"test_{m}"])

# full routed
np.save("oof/sex_route.npy",  final_oof)
np.save("testpred/sex_route.npy", final_test)

# write submission
pd.DataFrame({"id": test.id, "Calories": final_test})\
  .to_csv("submission_sex_route.csv", index=False)

print("âœ… Done!  Final sex-routed OOF RMSLE:",
      mean_squared_log_error(train.Calories.values, final_oof, squared=False))



import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from pathlib import Path

# 1) CONFIG
DATA_DIR  = "/kaggle/input/playground-series-s5e5"
N_SPLITS  = 5
SEED      = 42

LGB_PAR_RATIO = dict(
    objective="regression_l2", metric="rmse", learning_rate=0.03,
    n_estimators=2000, num_leaves=511, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.7, feature_fraction=0.7,
    reg_alpha=0.1, device="gpu", random_state=SEED
)

# 2) LOAD EXISTING ARRAYS
# these come from your previous full run
oof_full    = np.load("oof/sex_route.npy")
test_full   = np.load("testpred/sex_route.npy")

# 3) LOAD DATA + FE
def add_feats(df):
    df["BMI"]        = df.Weight/(df.Height/100)**2
    df["Workload"]   = df.Weight*df.Duration
    df["MET"]        = df.Heart_Rate/60*df.Duration
    df["TempDev"]    = df.Body_Temp-37
    df["HR_per_min"] = df.Heart_Rate/df.Duration.clip(1e-3)
    df["Age_HR"]     = df.Age*df.Heart_Rate
    base = ["Age","Height","Weight","Duration","Heart_Rate","Body_Temp"]
    df["row_mean"] = df[base].mean(axis=1)
    df["row_std"]  = df[base].std(axis=1)
    df["row_max"]  = df[base].max(axis=1)
    df["row_min"]  = df[base].min(axis=1)
    return df

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
train = add_feats(train)
test  = add_feats(test)
train["Sex"] = train.Sex.astype("category")
test["Sex"]  = test.Sex.astype("category")

FEATS_FULL = [c for c in train.columns if c not in ["id","Calories"]]

# 4) ISOLATE MALE INDICES
male_idx_tr = train.index[train.Sex=="male"].tolist()
male_idx_te = test.index[test.Sex=="male"].tolist()

# 5) RETRAIN MALE RATIOâ€�MODEL SLICE
grp_oof_m   = np.zeros(len(male_idx_tr))
grp_test_m  = np.zeros(len(male_idx_te))
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for tr,va in kf.split(male_idx_tr):
    tr_idx = [male_idx_tr[i] for i in tr]
    va_idx = [male_idx_tr[i] for i in va]

    X_tr = train.loc[tr_idx, FEATS_FULL]
    X_va = train.loc[va_idx, FEATS_FULL]

    # build and fit ratio model
    ratio_tr = train.Calories.loc[tr_idx] / train.Duration.loc[tr_idx].clip(1e-3)
    ratio_va = train.Calories.loc[va_idx] / train.Duration.loc[va_idx].clip(1e-3)

    m = lgb.LGBMRegressor(**LGB_PAR_RATIO, early_stopping_rounds=100)
    m.fit(
        X_tr, np.log1p(ratio_tr),
        eval_set=[(X_va, np.log1p(ratio_va))],
        categorical_feature=["Sex"]
    )

    # correct OOF assignment:
    pred_va = np.expm1(m.predict(X_va)) * train.Duration.loc[va_idx].values
    grp_oof_m[ va ] = pred_va

    # maleâ€�only test
    teX = test.loc[male_idx_te, FEATS_FULL]
    grp_test_m += (np.expm1(m.predict(teX)) * test.Duration.loc[male_idx_te].values) / N_SPLITS

# 6) MERGE BACK INTO GLOBAL ARRAYS
oof_full[ male_idx_tr ]  = grp_oof_m
test_full[ male_idx_te ] = grp_test_m

# 7) SAVE UPDATED ARRAYS & SUBMISSION
Path("oof").mkdir(exist_ok=True)
Path("testpred").mkdir(exist_ok=True)

np.save("oof/sex_route_updated.npy", oof_full)
np.save("testpred/sex_route_updated.npy", test_full)

pd.DataFrame({"id": test.id, "Calories": test_full}) \
  .to_csv("submission_updated.csv", index=False)

print("ğŸ�‰ Retrained male slice. New OOF RMSLE:",
      mean_squared_log_error(train.Calories, oof_full, squared=False))



# ============================================================
#  blend_grid_or_nnls.py
# ============================================================
import os, glob, itertools, numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import nnls
from sklearn.metrics import mean_squared_log_error
from tqdm import tqdm

BLEND_MODE = "nnls"   # "nnls"  or  "grid"
GRID_STEP  = 0.05     # weight increment if BLEND_MODE == "grid"
SEED       = 42       # just for reproducibility of product ordering

# ------------------------------------------------------------
# 1) discover files
# ------------------------------------------------------------
oof_dir      = Path("oof")
testpred_dir = Path("testpred")

oof_files   = sorted(oof_dir.glob("*.npy"))
oof_files = [f for f in oof_files if not f.name in ['cat.npy', 'lgb.npy', 'nodur.npy', 'ratio.npy', 'sex_route.npy']]
test_files  = {f.name: testpred_dir/f.name for f in testpred_dir.glob("*.npy")}
model_names = [f.stem for f in oof_files]

# ensure test counterpart exists
oof_arrays  = []
test_arrays = []
for f, name in zip(oof_files, model_names):
    test_f = test_files.get(f.name)
    if not test_f:
        print(f"âš ï¸�  No matching testpred for {name}, skipping.")
        continue
    oof_arrays.append(np.load(f))
    test_arrays.append(np.load(test_f))

X_oof   = np.column_stack(oof_arrays)
X_test  = np.column_stack(test_arrays)
print(f"Using {X_oof.shape[1]} models:", model_names)

# ------------------------------------------------------------
# 2) true target
# ------------------------------------------------------------
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")   # adjust path if needed
y_true = train["Calories"].values
log_y  = np.log1p(y_true)
log_X  = np.log1p(X_oof)

# ------------------------------------------------------------
# 3) blend search
# ------------------------------------------------------------
if BLEND_MODE == "nnls":
    w, _ = nnls(log_X, log_y)
    w /= w.sum()
    print("NNLS weights:", dict(zip(model_names, w.round(3))))
else:
    # brute grid
    step = GRID_STEP
    grid_vals = np.arange(0, 1+1e-9, step)
    best_rmsle = 1e9
    best_w = None
    for comb in tqdm(itertools.product(grid_vals, repeat=len(model_names))):
        if abs(sum(comb)-1) > 1e-6:
            continue
        pred = (X_oof * comb).sum(1)
        rmsle = mean_squared_log_error(y_true, pred, squared=False)
        if rmsle < best_rmsle:
            best_rmsle = rmsle
            best_w = comb
    w = np.array(best_w)
    print("Grid search weights:", dict(zip(model_names, w.round(3))))

# ------------------------------------------------------------
# 4) evaluate & save
# ------------------------------------------------------------
blend_oof  = (X_oof  * w).sum(1)
blend_test = (X_test * w).sum(1)
rmsle = mean_squared_log_error(y_true, blend_oof, squared=False)
print(f"Blended OOF RMSLE = {rmsle:.6f}")

Path("oof_1").mkdir(exist_ok=True,)
Path("testpred_1").mkdir(exist_ok=True)
np.save("oof_1/blend_grid_or_nnls.npy",  blend_oof)
np.save("testpred_1/blend_grid_or_nnls.npy", blend_test)

# ------------------------------------------------------------
# 5) optional submission file
# ------------------------------------------------------------
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")   # adjust path if needed
pd.DataFrame({"id": test_df.id, "Calories": blend_test}) \
  .to_csv("submission_blend_grid_or_nnls.csv", index=False)

print("âœ… Saved submission_blend_grid_or_nnls.csv")





