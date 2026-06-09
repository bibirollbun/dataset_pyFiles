# =========================================================================================
# MABe Challenge: Social Action Recognition (Final Fixed v22)
# =========================================================================================
import sys
import os
import gc
import re
import ast
import json
import joblib
import itertools
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from pathlib import Path
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter1d, binary_closing
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

# -----------------------------------------------------------------------------------------
# 1. CONFIGURATION & SETUP
# -----------------------------------------------------------------------------------------
if os.path.exists("/kaggle/input/mabe-package"):
    !pip install -q --no-index --find-links=/kaggle/input/mabe-package xgboost==3.1.1
else:
    print("⚠️ 'mabe-package' not found. Using pre-installed XGBoost.")

# --- GLOBAL CONSTANTS ---
INDEX_COLS = ["video_id", "agent_mouse_id", "target_mouse_id", "video_frame"]

class Config:
    seed = 42
    n_folds = 3
    negative_ratio = 10
    max_samples = 1_000_000 # High capacity
    
    # Paths
    input_dir = Path("/kaggle/input/MABe-mouse-behavior-detection")
    train_trk = input_dir / "train_tracking"
    train_ant = input_dir / "train_annotation"
    test_trk = input_dir / "test_tracking"
    work_dir = Path("/kaggle/working")
    
    # Behaviors
    body_parts = ["ear_left", "ear_right", "nose", "neck", "body_center",
                  "lateral_left", "lateral_right", "hip_left", "hip_right", "tail_base", "tail_tip"]
    
    self_behaviors = ["biteobject", "climb", "dig", "exploreobject", "freeze", 
                      "genitalgroom", "huddle", "rear", "rest", "run", "selfgroom"]
    
    pair_behaviors = ["allogroom", "approach", "attack", "attemptmount", "avoid", "chase",
                      "chaseattack", "defend", "disengage", "dominance", "dominancegroom",
                      "dominancemount", "ejaculate", "escape", "flinch", "follow", "intromit",
                      "mount", "reciprocalsniff", "shepherd", "sniff", "sniffbody",
                      "sniffface", "sniffgenital", "submit", "tussle"]

    # XGBoost Params
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "max_bin": 64,
        "seed": 42
    }

(Config.work_dir / "models").mkdir(parents=True, exist_ok=True)
(Config.work_dir / "features").mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------------------
# 2. FEATURE ENGINEERING ENGINE
# -----------------------------------------------------------------------------------------
class FeatureEngineer:
    @staticmethod
    def _cast(df):
        return df.with_columns(pl.col(c).cast(pl.Float32) for c in df.columns if c not in ["video_frame", "mouse_id", "bodypart"])

    @staticmethod
    def make_self(meta, trk):
        trk = FeatureEngineer._cast(trk)
        scale, fps = meta["pix_per_cm_approx"], meta["frames_per_second"]
        pivot = trk.pivot(on=["bodypart"], index=["video_frame", "mouse_id"], values=["x", "y"]).sort(["mouse_id", "video_frame"])
        n_mice = sum(1 for i in range(1,5) if f"mouse{i}_strain" in meta and meta[f"mouse{i}_strain"] is not None)
        track_map = {m: pivot.filter(pl.col("mouse_id")==m) for m in range(1, n_mice+1)}
        results = []
        vid_id = str(meta["video_id"])
        
        for agent in range(1, n_mice+1):
            if agent not in track_map: continue
            base = pl.DataFrame({"video_id": vid_id, "agent_mouse_id": agent, "target_mouse_id": -1, "video_frame": trk["video_frame"].unique().sort()}).with_columns([pl.col("video_id").cast(pl.Utf8), pl.col("agent_mouse_id").cast(pl.Int32), pl.col("target_mouse_id").cast(pl.Int32), pl.col("video_frame").cast(pl.Int32)])
            p = track_map[agent].select(pl.col("video_frame"), pl.exclude("video_frame").name.prefix("agent_"))
            
            # --- FIX 1: Fill missing parts for self ---
            for bp in Config.body_parts:
                if f"agent_x_{bp}" not in p.columns: 
                    p = p.with_columns([pl.lit(None, pl.Float32).alias(f"agent_x_{bp}"), pl.lit(None, pl.Float32).alias(f"agent_y_{bp}")])

            feats = p.with_columns([
                *[(((pl.col(f"agent_x_{b1}")-pl.col(f"agent_x_{b2}")).pow(2) + (pl.col(f"agent_y_{b1}")-pl.col(f"agent_y_{b2}")).pow(2)).sqrt()/scale).alias(f"aa_{b1}_{b2}") for b1,b2 in itertools.combinations(Config.body_parts, 2)],
                *[(((pl.col(f"agent_x_{bp}").diff()).pow(2) + (pl.col(f"agent_y_{bp}").diff()).pow(2)).sqrt()/scale*fps).rolling_mean(int(t*fps/1000), center=True).alias(f"spd_{bp}_{t}") for bp, t in itertools.product(["ear_left", "tail_base"], [500])]
            ]).select(pl.exclude([c for c in p.columns if "_x_" in c or "_y_" in c]))
            results.append(base.join(feats, on="video_frame", how="left"))
        return pl.concat(results, how="vertical") if results else pl.DataFrame()

    @staticmethod
    def make_pair(meta, trk):
        trk = FeatureEngineer._cast(trk)
        scale, fps = meta["pix_per_cm_approx"], meta["frames_per_second"]
        pivot = trk.pivot(on=["bodypart"], index=["video_frame", "mouse_id"], values=["x", "y"]).sort(["mouse_id", "video_frame"])
        n_mice = sum(1 for i in range(1,5) if f"mouse{i}_strain" in meta and meta[f"mouse{i}_strain"] is not None)
        track_map = {m: pivot.filter(pl.col("mouse_id")==m) for m in range(1, n_mice+1)}
        results = []
        vid_id = str(meta["video_id"])
        
        for ag, tg in itertools.permutations(range(1, n_mice+1), 2):
            if ag not in track_map or tg not in track_map: continue
            base = pl.DataFrame({"video_id": vid_id, "agent_mouse_id": ag, "target_mouse_id": tg, "video_frame": trk["video_frame"].unique().sort()}).with_columns([pl.col("video_id").cast(pl.Utf8), pl.col("agent_mouse_id").cast(pl.Int32), pl.col("target_mouse_id").cast(pl.Int32), pl.col("video_frame").cast(pl.Int32)])
            
            a = track_map[ag].select(pl.col("video_frame"), pl.exclude("video_frame").name.prefix("agent_"))
            t = track_map[tg].select(pl.col("video_frame"), pl.exclude("video_frame").name.prefix("target_"))
            
            # --- FIX 2: Fill missing parts for PAIR ---
            # This prevents the 'ColumnNotFoundError'
            for bp in Config.body_parts:
                if f"agent_x_{bp}" not in a.columns: 
                    a = a.with_columns([pl.lit(None, pl.Float32).alias(f"agent_x_{bp}"), pl.lit(None, pl.Float32).alias(f"agent_y_{bp}")])
                if f"target_x_{bp}" not in t.columns: 
                    t = t.with_columns([pl.lit(None, pl.Float32).alias(f"target_x_{bp}"), pl.lit(None, pl.Float32).alias(f"target_y_{bp}")])
            
            merged = a.join(t, on="video_frame", how="inner")
            
            def get_dist(b1, b2): return ((pl.col(f"agent_x_{b1}")-pl.col(f"target_x_{b2}")).pow(2) + (pl.col(f"agent_y_{b1}")-pl.col(f"target_y_{b2}")).pow(2)).sqrt()/scale
            base_dist = get_dist("body_center", "body_center")
            
            feats = merged.with_columns([
                *[get_dist(b1, b2).alias(f"at_{b1}_{b2}") for b1, b2 in itertools.product(["nose", "body_center", "tail_base"], repeat=2)],
                *[base_dist.rolling_mean(int(t*fps/1000), center=True).alias(f"d_mean_{t}") for t in [500, 1000]],
                *[base_dist.rolling_std(int(t*fps/1000), center=True).alias(f"d_std_{t}") for t in [500, 1000]],
                *[base_dist.rolling_min(int(t*fps/1000), center=True).alias(f"d_min_{t}") for t in [500, 1000]],
                *[(((pl.col(f"agent_x_{bp}").diff()).pow(2) + (pl.col(f"agent_y_{bp}").diff()).pow(2)).sqrt()/scale*fps).rolling_mean(int(t*fps/1000), center=True).alias(f"ag_spd_{bp}_{t}") for bp, t in itertools.product(["ear_left"], [500])],
                *[(((pl.col(f"target_x_{bp}").diff()).pow(2) + (pl.col(f"target_y_{bp}").diff()).pow(2)).sqrt()/scale*fps).rolling_mean(int(t*fps/1000), center=True).alias(f"tg_spd_{bp}_{t}") for bp, t in itertools.product(["ear_left"], [500])]
            ]).select(pl.exclude([c for c in merged.columns if "_x_" in c or "_y_" in c]))
            results.append(base.join(feats, on="video_frame", how="left"))
        return pl.concat(results, how="vertical") if results else pl.DataFrame()

# -----------------------------------------------------------------------------------------
# 3. TRAINER CLASS
# -----------------------------------------------------------------------------------------
class Trainer:
    def __init__(self):
        self.models = {}
        
    def tune_threshold(self, preds, y_true):
        best_th, best_f1 = 0.5, 0.0
        for th in np.arange(0.1, 0.9, 0.05):
            score = f1_score(y_true, (preds >= th), zero_division=0)
            if score > best_f1: best_f1, best_th = score, th
        return best_th

    def train_behavior(self, lab_id, behavior, indices, features, labels):
        out_dir = Config.work_dir / "models" / lab_id / behavior
        out_dir.mkdir(parents=True, exist_ok=True)
        
        if labels.select(pl.sum("label")).item() == 0:
            return 0.0
        
        kf = StratifiedGroupKFold(n_splits=Config.n_folds, shuffle=True, random_state=Config.seed)
        oof = np.zeros(len(labels), dtype=np.float32)
        
        X = features.to_pandas()
        y = labels.to_pandas()
        groups = indices.get_column("video_id").to_numpy()
        
        for fold, (t_idx, v_idx) in enumerate(kf.split(X, y, groups=groups)):
            X_tr, y_tr = X.iloc[t_idx], y.iloc[t_idx]
            X_val, y_val = X.iloc[v_idx], y.iloc[v_idx]
            
            pos_count = y_tr.sum().iloc[0] if isinstance(y_tr, pd.DataFrame) else y_tr.sum()
            ratio = (len(y_tr) - pos_count) / (pos_count + 1e-6)
            
            params = Config.xgb_params.copy()
            params["scale_pos_weight"] = min(ratio, 100.0)
            
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dvalid = xgb.DMatrix(X_val, label=y_val)
            
            model = xgb.train(params, dtrain, num_boost_round=350, evals=[(dtrain, "tr"), (dvalid, "val")], early_stopping_rounds=20, verbose_eval=False)
            
            p_val = model.predict(dvalid)
            th = self.tune_threshold(p_val, y_val)
            oof[v_idx] = p_val
            
            model.save_model(out_dir / f"fold_{fold}.json")
            with open(out_dir / f"fold_{fold}_th.txt", "w") as f: f.write(str(th))
            
        return f1_score(y, (oof >= 0.5).astype(int))

# -----------------------------------------------------------------------------------------
# 4. EXECUTION PIPELINE
# -----------------------------------------------------------------------------------------
def run_training():
    print(">>> 1. Loading Training Metadata...")
    train_df = pl.read_csv(Config.input_dir / "train.csv")
    
    train_beh = (
        train_df.filter(pl.col("behaviors_labeled").is_not_null())
        .with_columns(pl.col("behaviors_labeled").map_elements(eval, return_dtype=pl.List(pl.Utf8)).alias("b"))
        .explode("b")
        .with_columns([
            pl.col("video_id").cast(pl.Utf8),
            pl.col("b").str.split(",").list[0].str.replace_all("'","").alias("agent"),
            pl.col("b").str.split(",").list[1].str.replace_all("'","").alias("target"),
            pl.col("b").str.split(",").list[2].str.replace_all("'","").alias("action")
        ])
    )
    
    trainer = Trainer()
    
    # 1. Self Behaviors
    print(">>> [TRAIN] Processing SELF Behaviors...")
    for (lab, action), grp in tqdm(train_beh.filter(pl.col("action").is_in(Config.self_behaviors)).group_by("lab_id", "action")):
        idxs, feats, lbls = [], [], []
        total = 0
        for r in grp.rows(named=True):
            if total >= Config.max_samples: break
            vid, ag = str(r["video_id"]), r["agent"]
            path = Config.input_dir / "train_tracking" / f"{lab}/{vid}.parquet"
            if not path.exists(): continue
            
            ag_id = int(re.search(r"mouse(\d+)", ag).group(1))
            trk = pl.read_parquet(path)
            sf = FeatureEngineer.make_self(r, trk).filter(pl.col("agent_mouse_id")==ag_id)
            
            a_path = Config.input_dir / "train_annotation" / lab / f"{vid}.parquet"
            ant = pl.scan_parquet(a_path).filter((pl.col("action")==action) & (pl.col("agent_id")==ag_id)).collect() if a_path.exists() else pl.DataFrame()
            pos_frames = set()
            for ar in ant.rows(named=True): pos_frames.update(range(ar["start_frame"], ar["stop_frame"]))
            
            l = sf.select(pl.col("video_frame").is_in(pos_frames).cast(pl.Int8).alias("label"))
            if l.select(pl.sum("label")).item() == 0: continue
            
            pos_idx = np.where(l["label"]==1)[0]
            neg_idx = np.where(l["label"]==0)[0]
            if len(neg_idx) > len(pos_idx) * Config.negative_ratio:
                neg_idx = np.random.choice(neg_idx, len(pos_idx)*Config.negative_ratio, replace=False)
            keep = np.sort(np.concatenate([pos_idx, neg_idx]))
            
            idxs.append(sf[keep].select(["video_id"])); feats.append(sf[keep].select(pl.exclude(INDEX_COLS))); lbls.append(l[keep])
            total += len(keep)
            
        if idxs:
            trainer.train_behavior(lab, action, pl.concat(idxs), pl.concat(feats), pl.concat(lbls))
            gc.collect()

    # 2. Pair Behaviors
    print(">>> [TRAIN] Processing PAIR Behaviors...")
    for (lab, action), grp in tqdm(train_beh.filter(pl.col("action").is_in(Config.pair_behaviors)).group_by("lab_id", "action")):
        idxs, feats, lbls = [], [], []
        total = 0
        for r in grp.rows(named=True):
            if total >= Config.max_samples: break
            vid, ag, tg = str(r["video_id"]), r["agent"], r["target"]
            path = Config.input_dir / "train_tracking" / f"{lab}/{vid}.parquet"
            if not path.exists(): continue
            
            ag_id = int(re.search(r"mouse(\d+)", ag).group(1))
            tg_id = int(re.search(r"mouse(\d+)", tg).group(1))
            trk = pl.read_parquet(path)
            pf = FeatureEngineer.make_pair(r, trk).filter((pl.col("agent_mouse_id")==ag_id) & (pl.col("target_mouse_id")==tg_id))
            
            a_path = Config.input_dir / "train_annotation" / lab / f"{vid}.parquet"
            ant = pl.scan_parquet(a_path).filter((pl.col("action")==action) & (pl.col("agent_id")==ag_id) & (pl.col("target_id")==tg_id)).collect() if a_path.exists() else pl.DataFrame()
            pos_frames = set()
            for ar in ant.rows(named=True): pos_frames.update(range(ar["start_frame"], ar["stop_frame"]))
            
            l = pf.select(pl.col("video_frame").is_in(pos_frames).cast(pl.Int8).alias("label"))
            if l.select(pl.sum("label")).item() == 0: continue
            
            pos_idx = np.where(l["label"]==1)[0]
            neg_idx = np.where(l["label"]==0)[0]
            if len(neg_idx) > len(pos_idx) * Config.negative_ratio:
                neg_idx = np.random.choice(neg_idx, len(pos_idx)*Config.negative_ratio, replace=False)
            keep = np.sort(np.concatenate([pos_idx, neg_idx]))
            
            idxs.append(pf[keep].select(["video_id"])); feats.append(pf[keep].select(pl.exclude(INDEX_COLS))); lbls.append(l[keep])
            total += len(keep)
            
        if idxs:
            trainer.train_behavior(lab, action, pl.concat(idxs), pl.concat(feats), pl.concat(lbls))
            gc.collect()

def infer():
    print(">>> [INFER] Starting Prediction Phase...")
    test_df = pl.read_csv(Config.input_dir / "test.csv")
    preds = []
    
    models = {} 
    for lab_dir in (Config.work_dir / "models").glob("*"):
        lab = lab_dir.name
        models[lab] = {}
        for beh_dir in lab_dir.glob("*"):
            beh = beh_dir.name
            fold_models = []
            for fold_f in beh_dir.glob("*.json"):
                try:
                    bst = xgb.Booster()
                    bst.load_model(str(fold_f))
                    with open(str(fold_f).replace(".json", "_th.txt")) as f: th = float(f.read().strip())
                    fold_models.append((bst, bst.feature_names, th))
                except: pass
            if fold_models: models[lab][beh] = fold_models
            
    for row in tqdm(test_df.rows(named=True)):
        vid_id, lab = str(row["video_id"]), row["lab_id"]
        path = Config.input_dir / "test_tracking" / f"{lab}/{vid_id}.parquet"
        if not path.exists() or lab not in models: continue
        
        trk = pl.read_parquet(path)
        
        sf = FeatureEngineer.make_self(row, trk)
        if sf.height > 0:
            pdf = sf.to_pandas()
            for c in pdf.columns: 
                if pdf[c].dtype=='object': pdf[c]=pdf[c].astype('float32')
            
            for ag in pdf["agent_mouse_id"].unique():
                sub = pdf[pdf["agent_mouse_id"]==ag]
                frames = sub["video_frame"].values
                for beh in Config.self_behaviors:
                    key = f"{lab}_{beh}"
                    if beh not in models[lab]: continue
                    ensemble = models[lab][beh]
                    
                    req = ensemble[0][0].feature_names
                    missing = [c for c in req if c not in sub.columns]
                    if missing: 
                        for c in missing: sub[c] = np.nan
                            
                    dtest = xgb.DMatrix(sub[req])
                    avg_p, avg_t = 0, 0
                    for m, th in ensemble:
                        avg_p += m.predict(dtest)
                        avg_t += th
                    avg_p /= len(ensemble); avg_t /= len(ensemble)
                    
                    avg_p = gaussian_filter1d(avg_p, 2.0)
                    active = binary_closing(avg_p > (avg_t * 0.95), structure=[1]*10)
                    
                    for idx in np.where(active)[0]:
                        preds.append((vid_id, f"mouse{ag}", "self", beh, frames[idx]))

        pf = FeatureEngineer.make_pair(row, trk)
        if pf.height > 0:
            pdf = pf.to_pandas()
            for c in pdf.columns: 
                if pdf[c].dtype=='object': pdf[c]=pdf[c].astype('float32')
                
            for ag in pdf["agent_mouse_id"].unique():
                for tg in pdf[pdf["agent_mouse_id"]==ag]["target_mouse_id"].unique():
                    sub = pdf[(pdf["agent_mouse_id"]==ag) & (pdf["target_mouse_id"]==tg)]
                    frames = sub["video_frame"].values
                    for beh in Config.pair_behaviors:
                        key = f"{lab}_{beh}"
                        if beh not in models[lab]: continue
                        ensemble = models[lab][beh]
                        
                        req = ensemble[0][0].feature_names
                        missing = [c for c in req if c not in sub.columns]
                        if missing: 
                            for c in missing: sub[c] = np.nan
                                
                        dtest = xgb.DMatrix(sub[req])
                        avg_p, avg_t = 0, 0
                        for m, th in ensemble:
                            avg_p += m.predict(dtest)
                            avg_t += th
                        avg_p /= len(ensemble); avg_t /= len(ensemble)
                        
                        avg_p = gaussian_filter1d(avg_p, 2.0)
                        active = binary_closing(avg_p > (avg_t * 0.95), structure=[1]*10)
                        
                        for idx in np.where(active)[0]:
                            preds.append((vid_id, f"mouse{ag}", f"mouse{tg}", beh, frames[idx]))
        gc.collect()

    if not preds:
        pl.DataFrame({'video_id': [0], 'agent_id':['m1'], 'target_id':['m2'], 'action':['x'], 'start_frame':[0], 'stop_frame':[10]}).write_csv(Config.work_dir / "submission.csv")
    else:
        df = pl.DataFrame(preds, schema=["video_id", "agent_id", "target_id", "action", "start_frame"], orient="row")
        df = df.sort(["video_id", "agent_id", "target_id", "start_frame"])
        
        df = df.with_columns([
            pl.col("video_id").shift(1).alias("pv"), pl.col("agent_id").shift(1).alias("pa"), 
            pl.col("target_id").shift(1).alias("pt"), pl.col("action").shift(1).alias("pact"), 
            pl.col("start_frame").shift(1).alias("pf")
        ])
        df = df.with_columns(
            ((pl.col("video_id")!=pl.col("pv")) | (pl.col("agent_id")!=pl.col("pa")) | 
             (pl.col("target_id")!=pl.col("pt")) | (pl.col("action")
                                                    !=pl.col("pact")) | 
             ((pl.col("start_frame")-pl.col("pf"))>5)).fill_null(True).cum_sum().alias("seg")
        )
        final = df.group_by(["seg", "video_id", "agent_id", "target_id", "action"]).agg([
            pl.col("start_frame").min(), pl.col("start_frame").max().alias("stop_frame")
        ]).select(["video_id", "agent_id", "target_id", "action", "start_frame", "stop_frame"])
        
        final.filter(pl.col("stop_frame") > pl.col("start_frame")).with_row_index("row_id").write_csv(Config.work_dir / "submission.csv")
    print(">>> DONE.")

if __name__ == "__main__":
    run_training()
    infer()

