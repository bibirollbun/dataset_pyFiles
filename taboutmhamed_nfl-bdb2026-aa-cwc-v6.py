# =======================
# 0) CONFIG & UTILITAIRES
# =======================
import os, sys, json, gc, math, warnings, zipfile, itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss,
    precision_recall_curve, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

DATA_ROOT = Path("/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final")
OUT_DIR   = Path("./artifacts"); OUT_DIR.mkdir(exist_ok=True, parents=True)

SEED        = 42
MAX_PLAYS   = None     # None = full ; ex 800 pour itérer vite
USE_TTR     = True     # TTR calculé au lancer (contexte)
N_FOLDS     = 5        # GroupKFold par game_id

rng = np.random.default_rng(SEED)

def info(*msg): print("::", *msg)

def savefig(path, dpi=150):
    path = Path(path)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

# versions (repro)
import numpy, pandas, sklearn, matplotlib
print(sys.version)
print("numpy", numpy.__version__, "pandas", pandas.__version__, "sklearn", sklearn.__version__, "matplotlib", matplotlib.__version__)



# ==========================================
# 1) LECTURE RAPIDE: OUTPUT + INPUT + SUPP
# ==========================================
def read_csv_fast(path, usecols):
    try:
        return pd.read_csv(path, usecols=usecols, engine="pyarrow")
    except Exception:
        return pd.read_csv(path, usecols=usecols)

cols_out = ["game_id","play_id","nfl_id","frame_id","x","y"]
cols_in  = ["game_id","play_id","nfl_id","frame_id","player_role","x","y","s","a","dir",
            "ball_land_x","ball_land_y","num_frames_output","player_name","player_position","player_side"]
cols_supp= ["game_id","play_id","pass_result","pass_length",
            "team_coverage_man_zone","team_coverage_type",
            "route_of_targeted_receiver","defensive_team","possession_team"]

train_dir = DATA_ROOT/"train"
out_files = sorted(train_dir.glob("output_2023_w*.csv"))
in_files  = sorted(train_dir.glob("input_2023_w*.csv"))
supp_path = DATA_ROOT/"supplementary_data.csv"

supp = read_csv_fast(supp_path, [c for c in cols_supp if c!="possession_team"])
info("supp shape:", supp.shape)

outs = [read_csv_fast(f, cols_out) for f in out_files]
out  = pd.concat(outs, ignore_index=True)
info("out:", out.shape)

ins  = [read_csv_fast(f, cols_in) for f in in_files]
iin  = pd.concat(ins, ignore_index=True)
info("iin:", iin.shape)

# Downsample plays si besoin
if MAX_PLAYS is not None:
    keep = (supp.drop_duplicates(["game_id","play_id"])
                .sample(n=min(MAX_PLAYS, len(supp)), random_state=SEED))
    key = set(zip(keep.game_id, keep.play_id))
    out  = out[out.apply(lambda r: (r.game_id, r.play_id) in key, axis=1)].copy()
    iin  = iin[iin.apply(lambda r: (r.game_id, r.play_id) in key, axis=1)].copy()
    supp = supp[supp.apply(lambda r: (r.game_id, r.play_id) in key, axis=1)].copy()
    info("Downsampled:", len(key), "plays")

# Normalisations robustes
supp["pass_result"] = supp["pass_result"].astype("string").str.strip().str.upper()
supp["pass_length"] = pd.to_numeric(supp["pass_length"], errors="coerce")

def norm_mz(series):
    s = series.astype("string").str.strip().str.upper()
    s = s.replace({"MAN_COVERAGE":"MAN","ZONE_COVERAGE":"ZONE"})
    s = s.replace({"MAN":"Man","ZONE":"Zone"})
    return s.fillna("Unknown")

if "team_coverage_man_zone" not in supp.columns:
    supp["team_coverage_man_zone"] = "Unknown"
else:
    supp["team_coverage_man_zone"] = norm_mz(supp["team_coverage_man_zone"])

# Gardes pour colonnes optionnelles
for col in ["route_of_targeted_receiver","team_coverage_type","defensive_team"]:
    if col not in supp.columns:
        supp[col] = "Unknown"



# =======================================================
# 2) FRAME LANCER + WR ciblé + DB coverage + point L (x,y)
# =======================================================
at_throw = (iin.sort_values(["game_id","play_id","nfl_id","frame_id"])
              .groupby(["game_id","play_id","nfl_id"], observed=False)
              .tail(1)[["game_id","play_id","nfl_id","x","y","s","a","dir","player_role",
                        "ball_land_x","ball_land_y","player_name","player_position","player_side"]])

# WR ciblé (un par play)
wr_throw = at_throw[at_throw["player_role"]=="Targeted Receiver"].copy()
wr_throw = (wr_throw.sort_values(["game_id","play_id"])
                      .drop_duplicates(["game_id","play_id"], keep="first"))
wr_throw.rename(columns={"x":"x0_wr","y":"y0_wr"}, inplace=True)

# DB coverage (potentiellement plusieurs)
db_throw = at_throw[at_throw["player_role"]=="Defensive Coverage"].copy()
db_throw.rename(columns={"x":"x0_db","y":"y0_db"}, inplace=True)

# Table WR + point L
wrL = wr_throw[["game_id","play_id","nfl_id","player_name","player_position",
                "x0_wr","y0_wr","ball_land_x","ball_land_y"]].copy()
wrL.rename(columns={"nfl_id":"wr_id","player_name":"wr_name","player_position":"wr_pos"}, inplace=True)

# Suivi post-lancer du WR ciblé
out_wr = out.merge(wrL[["game_id","play_id","wr_id","ball_land_x","ball_land_y"]],
                   on=["game_id","play_id"], how="inner")
out_wr["d_wr_L"] = np.hypot(out_wr["x"]-out_wr["ball_land_x"], out_wr["y"]-out_wr["ball_land_y"])



# =======================================================
# 3) METRIQUES POST-LANCER : AA,CWC et features avancées
# =======================================================
# DB par play (set des nfl_id DB)
defenders_by_play = (db_throw.groupby(['game_id','play_id'], observed=False)['nfl_id']
                     .unique().apply(set).to_dict())
wr_keys = set(zip(wrL.game_id, wrL.play_id))

def angle_to_L(x, y, Lx, Ly):
    return math.atan2(Ly - y, Lx - x)

AA_rows = []
for (g,p), grp in out_wr.groupby(["game_id","play_id"], observed=False):
    if (g,p) not in wr_keys:
        continue
    wrp = wrL[(wrL["game_id"]==g) & (wrL["play_id"]==p)].iloc[0]
    Lx, Ly = wrp["ball_land_x"], wrp["ball_land_y"]
    if pd.isna(Lx) or pd.isna(Ly):  # si point L manquant, on saute (évite KeyError plus tard)
        continue

    # DB frames de ce play
    db_ids = defenders_by_play.get((g,p), set())
    out_db = out[(out["game_id"]==g) & (out["play_id"]==p) & (out["nfl_id"].isin(db_ids))].copy()

    tmp = grp.sort_values("frame_id")[["frame_id","x","y","d_wr_L"]].copy()

    # d_DEFmin->L par frame
    if len(out_db):
        dmin_by_frame = (
            out_db.assign(dx=lambda d: d["x"]-Lx,
                          dy=lambda d: d["y"]-Ly)
                  .assign(dist=lambda d: np.hypot(d["dx"], d["dy"]))
                  .groupby("frame_id", observed=False)["dist"].min()
        )
        tmp["d_defmin_L"] = tmp["frame_id"].map(dmin_by_frame)
    else:
        tmp["d_defmin_L"] = np.nan

    # AA(t)
    tmp["AA"] = tmp["d_defmin_L"] - tmp["d_wr_L"]

    # Angle vers L (stabilité directionnelle)
    tmp["theta_wr_to_L"] = [angle_to_L(x,y,Lx,Ly) for x,y in zip(tmp["x"], tmp["y"])]

    # CWC
    d0 = float(tmp["d_wr_L"].iloc[0]) if len(tmp) else np.nan
    dT = float(tmp["d_wr_L"].iloc[-1]) if len(tmp) else np.nan
    CWC_norm = (1 - dT/d0) if (pd.notna(d0) and d0>0 and pd.notna(dT)) else np.nan

    # Résumés
    if len(tmp):
        lead_rate     = float(np.mean(tmp["AA"] > 0))
        AA_arrival    = float(tmp["AA"].iloc[-1])
        AA_integrated = float(np.nansum(tmp["AA"].to_numpy()))

        # streak AA>0
        lead_streak_max = 0; cur = 0
        for v in (tmp["AA"] > 0).to_numpy():
            cur = cur + 1 if v else 0
            lead_streak_max = max(lead_streak_max, cur)
        closing_speed_arrival = (tmp["d_wr_L"].iloc[-3] - tmp["d_wr_L"].iloc[-1]) if len(tmp) >= 3 else np.nan
        ang_std = float(np.nanstd(tmp["theta_wr_to_L"].to_numpy()))
    else:
        lead_rate=AA_arrival=AA_integrated=closing_speed_arrival=ang_std=np.nan
        lead_streak_max=0

    AA_rows.append({
        "game_id": g, "play_id": p,
        "lead_rate": lead_rate,
        "AA_arrival": AA_arrival,
        "AA_integrated": AA_integrated,
        "d0": d0, "dT": dT,
        "CWC_norm_dist": CWC_norm,
        "lead_streak_max": lead_streak_max,
        "closing_speed_arrival": closing_speed_arrival,
        "ang_std_wr_to_L": ang_std
    })

feat_core = pd.DataFrame(AA_rows)
info("feat_core:", feat_core.shape)



# ======================================
# 4) TTR au lancer (optionnel) + FUSIONS
# ======================================
def time_to_reach(x0, y0, Lx, Ly, v0=0.0, vmax=8.0, accel=3.0):
    d = float(np.hypot(Lx - x0, Ly - y0))
    t_accel = max((vmax - v0)/accel, 0.0)
    d_accel = v0*t_accel + 0.5*accel*t_accel**2
    if d <= 2*d_accel:
        return 2*math.sqrt(max(d/accel, 1e-9))
    else:
        d_cruise = d - 2*d_accel
        return 2*t_accel + d_cruise/max(vmax,1e-6)

ttr_rows = []
if USE_TTR:
    for (g,p), wrp_grp in wrL.groupby(["game_id","play_id"], observed=False):
        wrp = wrp_grp.iloc[0]
        Lx, Ly = wrp["ball_land_x"], wrp["ball_land_y"]
        if pd.isna(Lx) or pd.isna(Ly):
            continue
        xw, yw = wrp["x0_wr"], wrp["y0_wr"]
        wr0 = at_throw[(at_throw["game_id"]==g)&(at_throw["play_id"]==p)&(at_throw["nfl_id"]==wrp["wr_id"])]
        v0w = float(wr0["s"].iloc[0]) if len(wr0) and pd.notna(wr0["s"].iloc[0]) else 0.0
        t_wr = time_to_reach(xw, yw, Lx, Ly, v0=v0w)

        dbs = db_throw[(db_throw["game_id"]==g)&(db_throw["play_id"]==p)]
        if len(dbs):
            t_dbs = []
            for xd, yd, sv in zip(dbs["x0_db"], dbs["y0_db"], dbs["s"].fillna(0.0)):
                t_dbs.append(time_to_reach(float(xd), float(yd), Lx, Ly, v0=float(sv)))
            t_defmin = float(np.min(t_dbs))
        else:
            t_defmin = np.nan

        ttr_rows.append({"game_id":g,"play_id":p,
                         "TTR_WR0":t_wr,"TTR_DEFmin0":t_defmin,
                         "TTR_advantage0": (t_defmin - t_wr) if pd.notna(t_defmin) else np.nan})

ttr_feat = (pd.DataFrame(ttr_rows) if ttr_rows
            else pd.DataFrame(columns=["game_id","play_id","TTR_WR0","TTR_DEFmin0","TTR_advantage0"]))
info("ttr_feat:", ttr_feat.shape)

# Fusion + buckets air-yards
feat_all = (feat_core
            .merge(supp[["game_id","play_id","pass_result","pass_length",
                         "team_coverage_man_zone","team_coverage_type",
                         "route_of_targeted_receiver","defensive_team"]],
                   on=["game_id","play_id"], how="left")
            .merge(wrL[["game_id","play_id","wr_id","wr_name","wr_pos","ball_land_x","ball_land_y"]],
                   on=["game_id","play_id"], how="left"))
if USE_TTR and len(ttr_feat):
    feat_all = feat_all.merge(ttr_feat, on=["game_id","play_id"], how="left")

def bucket_air(y):
    if pd.isna(y): return "Unknown"
    if y>=20: return "20+"
    if y>=10: return "10-19"
    return "0-9" if y>=0 else "<0"

feat_all["air_bucket"] = feat_all["pass_length"].apply(bucket_air)
info("feat_all:", feat_all.shape)
feat_all.head(5)



# ==================================================
# 5) EVAL OOF (GroupKFold par game_id) + 2 modèles
# ==================================================
def build_matrix(df, cols):
    cols = [c for c in cols if c in df.columns]
    if len(cols)==0:
        return np.empty((len(df),0)), cols
    X = np.column_stack([pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy() for c in cols])
    return X, cols

def oof_scores(df, feature_cols, model="lr", n_folds=5, seed=42):
    df = df.dropna(subset=["pass_result"]).copy()
    y = (df["pass_result"]=="C").astype(int).to_numpy()
    groups = df["game_id"].to_numpy()
    gkf = GroupKFold(n_splits=n_folds)

    oof = np.zeros(len(df))
    for tr, va in gkf.split(df, y, groups):
        X_tr, _ = build_matrix(df.iloc[tr], feature_cols)
        X_va, _ = build_matrix(df.iloc[va], feature_cols)
        if model == "lr":
            base = LogisticRegression(max_iter=1000, random_state=seed)
            clf  = CalibratedClassifierCV(base, method="isotonic", cv=5)
        elif model == "hgb":
            base = HistGradientBoostingClassifier(
                max_depth=3, max_iter=300, learning_rate=0.05,
                l2_regularization=0.0, random_state=seed
            )
            clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        else:
            raise ValueError("model must be 'lr' or 'hgb'")
        clf.fit(X_tr, y[tr])
        oof[va] = clf.predict_proba(X_va)[:,1]
    return df, y, oof

base_cols = ["pass_length"]
aa_cols   = ["pass_length","lead_rate","AA_arrival","AA_integrated","CWC_norm_dist",
             "closing_speed_arrival","ang_std_wr_to_L","lead_streak_max"]
ttr_cols  = aa_cols + [c for c in ["TTR_advantage0","TTR_WR0","TTR_DEFmin0"] if c in feat_all.columns]

df_lr, y_lr, oof_lr = oof_scores(feat_all, ttr_cols, model="lr", n_folds=N_FOLDS, seed=SEED)
df_hg, y_hg, oof_hg = oof_scores(feat_all, ttr_cols, model="hgb", n_folds=N_FOLDS, seed=SEED)

def scores(y, p):
    return {
        "AUC":   float(roc_auc_score(y,p)),
        "Brier": float(brier_score_loss(y,p)),
        "LogLoss":float(log_loss(y,p))
    }

scorecard = {
    "LogReg+isotonic": scores(y_lr, oof_lr),
    "HistGB+sigmoid":  scores(y_hg, oof_hg),
    "feature_set": ttr_cols
}
Path(OUT_DIR/"scorecard.json").write_text(json.dumps(scorecard, indent=2))
print(json.dumps(scorecard, indent=2))



# ==================================================
# 6) DIAGNOSTICS & FIGURES
# ==================================================
def plot_calibration(probs, y, bins=10, title="Calibration"):
    edges = np.linspace(0,1,bins+1)
    binid = np.digitize(probs, edges)-1
    obs, exp, n = [], [], []
    for b in range(bins):
        m = (binid==b)
        if m.sum()==0: continue
        obs.append(y[m].mean()); exp.append(probs[m].mean()); n.append(int(m.sum()))
    plt.figure(figsize=(6,5))
    plt.plot([0,1],[0,1],'--', lw=1)
    plt.plot(exp, obs, marker='o')
    for ex, ob, cnt in zip(exp, obs, n):
        plt.text(ex, ob, str(cnt), fontsize=8, ha='left', va='bottom')
    plt.xlabel("Predicted"); plt.ylabel("Observed"); plt.title(title)

# Choix du modèle pour les plots (HGB souvent meilleur)
y = y_hg; oof = oof_hg; dfm = df_hg

# Calibration globale
plot_calibration(oof, y, title="Calibration (OOF) — AA/CWC(+TTR) — HGB")
savefig(OUT_DIR/"calibration_oof.png")

# ROC
fpr, tpr, _ = roc_curve(y, oof)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y,oof):.3f}")
plt.plot([0,1],[0,1],'--',lw=1)
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (OOF) — HGB"); plt.legend()
savefig(OUT_DIR/"roc_oof.png")

# PR
prec, rec, _ = precision_recall_curve(y, oof)
plt.figure(figsize=(6,5))
plt.plot(rec, prec)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall (OOF) — HGB")
savefig(OUT_DIR/"pr_oof.png")

# Distributions air-yards
plt.figure(figsize=(6,5))
pd.to_numeric(dfm["pass_length"], errors="coerce").dropna().plot(kind="hist", bins=40)
plt.xlabel("Air-yards"); plt.title("Distribution des air-yards")
savefig(OUT_DIR/"dist_air_yards.png")

# Scatter AA_arrival vs air-yards
if "AA_arrival" in dfm.columns:
    plt.figure(figsize=(6,5))
    plt.scatter(dfm["pass_length"], dfm["AA_arrival"], s=10, alpha=0.6)
    plt.xlabel("Air-yards"); plt.ylabel("AA_arrival"); plt.title("AA_arrival vs air-yards")
    savefig(OUT_DIR/"scatter_AA_arrival_vs_air.png")

# Box AA_arrival par route (top 12)
if "route_of_targeted_receiver" in dfm.columns and dfm["route_of_targeted_receiver"].notna().any():
    top_routes = (dfm["route_of_targeted_receiver"].fillna("Unknown").value_counts().head(12).index.tolist())
    if len(top_routes):
        plt.figure(figsize=(9,5))
        dfm[dfm["route_of_targeted_receiver"].isin(top_routes)].boxplot(
            column="AA_arrival", by="route_of_targeted_receiver", rot=45)
        plt.suptitle(""); plt.title("AA_arrival par route (top 12)"); plt.ylabel("AA_arrival")
        savefig(OUT_DIR/"box_AA_arrival_by_route.png")

# Hexbin AA_arrival (si L présent en quantité)
if {"ball_land_x","ball_land_y","AA_arrival"}.issubset(dfm.columns):
    ok = dfm[["ball_land_x","ball_land_y","AA_arrival"]].notna().all(axis=1)
    if ok.sum()>200:
        plt.figure(figsize=(7,4))
        hb = plt.hexbin(dfm.loc[ok,"ball_land_x"], dfm.loc[ok,"ball_land_y"],
                        C=dfm.loc[ok,"AA_arrival"], gridsize=30, reduce_C_function=np.nanmean)
        plt.colorbar(hb, label="AA_arrival moyen")
        plt.xlim(0,120); plt.ylim(0,53.3); plt.title("Carte AA_arrival au point d'atterrissage")
        plt.xlabel("x (yd)"); plt.ylabel("y (yd)")
        savefig(OUT_DIR/"hexbin_AA_arrival_field.png")

# Calibration par buckets d’air-yards
for buck in ["0-9","10-19","20+"]:
    sub = dfm[dfm["air_bucket"]==buck]
    if len(sub) < 150: 
        continue
    yb = (sub["pass_result"]=="C").astype(int).to_numpy()
    # aligner indices: positions dans dfm
    pos = dfm.index.get_indexer(sub.index)
    pos = pos[pos>=0]
    pb  = oof[pos]
    plot_calibration(pb, yb, title=f"Calibration OOF — {buck} air-yards")
    fn = f"calibration_oof_{buck.replace('+','plus')}.png".replace('20plus','20+')
    savefig(OUT_DIR/fn.replace('+','plus'))



# ==================================================
# 7) IMPORTANCES (permutation) + PDP 1D
# ==================================================
feat_cols = ttr_cols
X = np.column_stack([pd.to_numeric(dfm[c], errors="coerce").fillna(0.0).to_numpy() for c in feat_cols])
hgb = HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.05,
                                     l2_regularization=0.0, random_state=SEED)
hgb.fit(X, y)

# permutation importance
pi = permutation_importance(hgb, X, y, n_repeats=10, random_state=SEED, n_jobs=-1)
imp_df = pd.DataFrame({"feature": feat_cols, "importance": pi.importances_mean})
imp_df = imp_df.sort_values("importance", ascending=False)
imp_df.to_csv(OUT_DIR/"perm_importance_hgb.csv", index=False)

plt.figure(figsize=(7,5))
plt.barh(imp_df["feature"], imp_df["importance"])
plt.gca().invert_yaxis()
plt.xlabel("Permutation importance"); plt.title("HGB — permutation importance")
savefig(OUT_DIR/"perm_importance_hgb.png")

# PDP 1D (4 features clés si présentes)
pdp_feats = [f for f in ["lead_rate","CWC_norm_dist","AA_arrival","TTR_advantage0"] if f in feat_cols]
for feat in pdp_feats[:4]:
    fig, ax = plt.subplots(figsize=(6,4))
    PartialDependenceDisplay.from_estimator(hgb, X, [feat_cols.index(feat)], ax=ax)
    ax.set_title(f"PDP — {feat}")
    savefig(OUT_DIR/f"pdp_{feat}.png")



# ==================================================
# 8) LEADERBOARDS + MATRICES + TIMELINES
# ==================================================
# WR leaderboard (min n=10)
wr_lead = (feat_all.groupby(["wr_id","wr_name","wr_pos"], observed=False)
           .agg(n=("lead_rate","size"),
                lead_rate=("lead_rate","mean"),
                AA_arrival=("AA_arrival","mean"),
                CWC_norm=("CWC_norm_dist","mean"))
           .reset_index())
wr_lead = wr_lead[wr_lead["n"]>=10].sort_values(["lead_rate","AA_arrival"], ascending=[False,False])
wr_lead.to_csv(OUT_DIR/"leaderboard_wr_named.csv", index=False)

# Défense leaderboard (min n=50)
def_lead = (feat_all.groupby("defensive_team", observed=False)
            .agg(n=("lead_rate","size"),
                 CWC_norm=("CWC_norm_dist","mean"),
                 AA_arrival=("AA_arrival","mean"))
            .reset_index())
def_lead = def_lead[def_lead["n"]>=50].sort_values(["CWC_norm"], ascending=False)
def_lead.to_csv(OUT_DIR/"leaderboard_defense.csv", index=False)

# Matrice route × Man/Zone
tab = (feat_all.assign(MZ=lambda d: d["team_coverage_man_zone"].fillna("Unknown"))
       .groupby(["route_of_targeted_receiver","MZ"], observed=False)
       .agg(n=("lead_rate","size"),
            lead_rate=("lead_rate","mean"),
            AA_arrival=("AA_arrival","mean"),
            CWC=("CWC_norm_dist","mean"))
       .reset_index())
tab.to_csv(OUT_DIR/"route_by_manzone_table.csv", index=False)

# Timelines AA(t) — deux exemples sûrs
ex_base = dfm.dropna(subset=["AA_integrated"])
examples = ex_base.sample(n=min(2, len(ex_base)), random_state=SEED) if len(ex_base) else pd.DataFrame()
for _, row in examples.iterrows():
    g, p = row["game_id"], row["play_id"]
    Lx, Ly = row.get("ball_land_x"), row.get("ball_land_y")
    if pd.isna(Lx) or pd.isna(Ly): 
        continue
    tmp_wr = out_wr[(out_wr["game_id"]==g)&(out_wr["play_id"]==p)].sort_values("frame_id").copy()
    dbs = db_throw[(db_throw["game_id"]==g)&(db_throw["play_id"]==p)]
    out_db = out[(out["game_id"]==g)&(out["play_id"]==p)&(out["nfl_id"].isin(dbs["nfl_id"].unique()))]
    dmin = out_db.groupby("frame_id", observed=False).apply(
        lambda df: np.min(np.hypot(df["x"]-Lx, df["y"]-Ly))
    )
    tmp_wr["d_defmin_L"] = tmp_wr["frame_id"].map(dmin)
    tmp_wr["AA"] = tmp_wr["d_defmin_L"] - tmp_wr["d_wr_L"]

    plt.figure(figsize=(7,4))
    plt.plot(tmp_wr["frame_id"], tmp_wr["AA"], label="AA(t)")
    plt.axhline(0, color="k", lw=1)
    ttl = f"AA timeline — game {g} play {p} — result={row['pass_result']}"
    plt.title(ttl); plt.xlabel("frame"); plt.ylabel("AA")
    savefig(OUT_DIR/f"aa_timeline_{g}_{p}.png")



# ==================================================
# 9) WRITEUP + MEDIAPACK + ZIP GLOBAL
# ==================================================
writeup = f"""
# Arrival Advantage & Catch Window Compression: lire la bataille en vol
**Track:** University (Notebook public attaché).  
**Scope:** frames **après le lancer** jusqu’à catch/incomplete/interception.

## Motivation
Quand la balle est en l’air, la bataille pour le **point d’atterrissage L** décide l’issue. Nous proposons deux métriques post-lancer, interprétables :
- **Arrival Advantage (AA)** : qui mène la course à L, frame par frame.
- **Catch Window Compression (CWC)** : vitesse de fermeture de la fenêtre de réception.

## Données & périmètre
Tracking 2023 : `output_2023_w*.csv` (post-lancer), `input_2023_w*.csv` (uniquement pour ancrer **TTR au lancer**), `supplementary_data.csv`. Analyses **strictement post-lancer** pour AA/CWC.

## Définitions
AA(t)=d_DEFmin(L)−d_WR(L). **lead_rate**=%frames AA(t)>0. **AA_arrival**=AA au dernier frame. **CWC_norm=1−dT/d0**. **(Optionnel)** **TTR_advantage0=(DEF−WR)** au **frame du lancer** (contexte).

## Méthode
GroupKFold par match, calibration (isotone/sigmoid), features avancées (closing_speed, ang_std, lead_streak). Modèles **LogReg** et **HistGB**.

## Résultats (OOF)
- LogReg+isotone — {json.dumps(Path(OUT_DIR/'scorecard.json').read_text())[:0]}

(voir `scorecard.json` pour les chiffres exacts)

## Applications
**WR room** : lead_rate & AA_arrival par concept.  
**DB room** : viser AA_arrival ≤ 0 et haute CWC sur explosifs.  
**Game-planning** : matrice route × couverture.

## Usage hebdo
Publier chaque lundi, par équipe et par concept (route × couverture), le **lead_rate**, l’**AA_arrival** et la **CWC**, avec alerte sur les concepts explosifs (air-yards ≥ 20) et seuil opérationnel p(catch) > 0,60.

## Limites & pistes
Sensibilité à L ; TTR simple ; extensions : densité multi-DB, temps de vol, incertitude de L.

## Citation
Lopez, M., Bliss, T., Blake, A., & Howard, A. (2025). *NFL Big Data Bowl 2026 – Analytics* [Competition dataset]. Kaggle. CC BY-NC 4.0.

## Licence
Données : CC BY-NC 4.0. Code : MIT (OSI).
""".strip()

Path(OUT_DIR/"writeup_draft.md").write_text(writeup)

# MediaPack (images/csv/json/md)
with zipfile.ZipFile(OUT_DIR/"MediaPack_COMPLIANT.zip","w",compression=zipfile.ZIP_DEFLATED) as z:
    for f in os.listdir(OUT_DIR):
        if f.endswith((".png",".csv",".json",".md")):
            z.write(OUT_DIR/f, arcname=f)

# Pack global (hors .zip)
with zipfile.ZipFile(OUT_DIR/"SubmissionPack.zip","w",compression=zipfile.ZIP_DEFLATED) as z:
    for f in os.listdir(OUT_DIR):
        if f.endswith(".zip"): 
            continue
    # ajoute tout sauf les .zip (boucle correcte)
    for f in os.listdir(OUT_DIR):
        if not f.endswith(".zip"):
            z.write(OUT_DIR/f, arcname=f)

print("Saved writeup & packs in", OUT_DIR)



# ==================================================
# 10) ABLATION + BOOTSTRAP (AUC & Brier — 95% CI)
# ==================================================
def oof_scores_model(df, feature_cols, model="lr", n_folds=5, seed=42):
    df = df.dropna(subset=["pass_result"]).copy()
    y = (df["pass_result"]=="C").astype(int).to_numpy()
    groups = df["game_id"].to_numpy()
    gkf = GroupKFold(n_splits=n_folds)

    oof = np.zeros(len(df))
    for tr, va in gkf.split(df, y, groups):
        X_tr, _ = build_matrix(df.iloc[tr], feature_cols)
        X_va, _ = build_matrix(df.iloc[va], feature_cols)
        if model == "lr":
            base = LogisticRegression(max_iter=1000, random_state=seed)
            clf  = CalibratedClassifierCV(base, method="isotonic", cv=5)
        elif model == "hgb":
            base = HistGradientBoostingClassifier(
                max_depth=3, max_iter=300, learning_rate=0.05,
                l2_regularization=0.0, random_state=seed
            )
            clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        else:
            raise ValueError("model must be 'lr' or 'hgb'")
        clf.fit(X_tr, y[tr])
        oof[va] = clf.predict_proba(X_va)[:,1]
    return df, y, oof

def bootstrap_ci(y, p, B=400, seed=42, agg="AUC"):
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if agg=="AUC":
            val = roc_auc_score(yb, pb)
        elif agg=="Brier":
            val = brier_score_loss(yb, pb)
        else:
            val = log_loss(yb, pb)
        stats.append(val)
    stats = np.array(stats)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(np.mean(stats)), float(lo), float(hi)

base_cols   = ["pass_length"]
aa_core     = ["pass_length","lead_rate","AA_arrival","AA_integrated","CWC_norm_dist"]
ttr_only    = [c for c in ["TTR_advantage0","TTR_WR0","TTR_DEFmin0"] if c in feat_all.columns]
extras_cols = ["closing_speed_arrival","ang_std_wr_to_L","lead_streak_max"]

sets = {
    "baseline": base_cols,
    "+AA/CWC": aa_core,
    "+TTR": aa_core + ttr_only,
    "+TTR+extras": aa_core + ttr_only + [c for c in extras_cols if c in feat_all.columns]
}

def ablation_table(model_name, out_prefix):
    rows_auc, rows_brier = [], []
    for name, cols in sets.items():
        df_o, y_o, p_o = oof_scores_model(feat_all, cols, model=("lr" if model_name=="lr" else "hgb"),
                                          n_folds=N_FOLDS, seed=SEED)
        AUC_mean, AUC_lo, AUC_hi = bootstrap_ci(y_o, p_o, B=400, seed=SEED, agg="AUC")
        Br_mean,  Br_lo,  Br_hi  = bootstrap_ci(y_o, p_o, B=400, seed=SEED, agg="Brier")
        rows_auc.append({"model":model_name, "feature_set":name, "AUC":AUC_mean, "AUC_lo":AUC_lo, "AUC_hi":AUC_hi, "k":len(cols)})
        rows_brier.append({"model":model_name, "feature_set":name, "Brier":Br_mean, "Brier_lo":Br_lo, "Brier_hi":Br_hi, "k":len(cols)})
    auc_df   = pd.DataFrame(rows_auc).sort_values("AUC", ascending=False)
    brier_df = pd.DataFrame(rows_brier).sort_values("Brier", ascending=True)
    auc_df.to_csv(OUT_DIR/f"{out_prefix}_auc.csv", index=False)
    brier_df.to_csv(OUT_DIR/f"{out_prefix}_brier.csv", index=False)
    return auc_df, brier_df

auc_lr,   brier_lr   = ablation_table("lr",  "ablation_lr")
auc_hgb,  brier_hgb  = ablation_table("hgb", "ablation_hgb")

def bar_with_ci(df, metric="AUC", title="", fname="plot.png", higher_is_better=True):
    order = df.sort_values(metric, ascending=not higher_is_better)
    xs = np.arange(len(order))
    m  = order[metric].to_numpy()
    lo = order[f"{metric}_lo"].to_numpy()
    hi = order[f"{metric}_hi"].to_numpy()
    err_lo = m - lo
    err_hi = hi - m

    plt.figure(figsize=(8,5))
    plt.bar(xs, m, yerr=[err_lo, err_hi], capsize=4)
    plt.xticks(xs, order["feature_set"].to_list(), rotation=0)
    plt.ylabel(metric); plt.title(title)
    savefig(OUT_DIR/fname)

bar_with_ci(auc_lr,   "AUC",   "Ablation (AUC 95% CI) — LogReg+isotonic", "ablation_auc_ci_lr.png",   True)
bar_with_ci(auc_hgb,  "AUC",   "Ablation (AUC 95% CI) — HGB+sigmoid",    "ablation_auc_ci_hgb.png",  True)
bar_with_ci(brier_lr, "Brier", "Ablation (Brier 95% CI) — LogReg+isotonic", "ablation_brier_ci_lr.png", False)
bar_with_ci(brier_hgb,"Brier", "Ablation (Brier 95% CI) — HGB+sigmoid",    "ablation_brier_ci_hgb.png",False)

print("Ablation + bootstrap CIs saved in ./artifacts")



!ls -Rlth /kaggle/working/artifacts


%%bash
cd /kaggle/working
zip -r ArtifactsBundle.zip artifacts -x "artifacts/*.zip"


