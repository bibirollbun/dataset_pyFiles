"""
AIRR-ML-25 Challenge - FINAL GPU PRODUCTION (with fixed submission)
==================================================================
- GPU-based feature selection (XGBoost) to avoid CPU mutual-info crashes.
- Correct submission generation:
  * Start from sample_submissions.csv
  * Update ONLY Task-1 rows for test_dataset_* with predicted probabilities
  * Leave Task-2 rows (train_dataset_*) exactly as-is
"""

import os
import gc
import warnings
import subprocess
from pathlib import Path
from collections import Counter
from typing import Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
import lightgbm as lgb
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")


# =====================================================================
# CONFIGURATION
# =====================================================================
class Config:
    DATA_ROOT = Path("/kaggle/input/adaptive-immune-profiling-challenge-2025/")
    TRAIN_DIR = DATA_ROOT / "train_datasets" / "train_datasets"
    TEST_DIR = DATA_ROOT / "test_datasets" / "test_datasets"
    SAMPLE_SUBMISSION = DATA_ROOT / "sample_submissions.csv"

    # Keep strictly to [3, 4] to prevent RAM explosion
    K_LIST = [3, 4]
    TOP_KMER = 400
    MAX_SEQUENCES_PER_FILE = 50000

    # Public clone settings
    PUB_MAX_FILES = 20
    PUB_MIN_FREQ = 0.18
    PUB_ENRICH = 6.0
    PUB_TOP_N = {7: 5000, 8: 3000, "default": 2000}

    N_SPLITS = 5
    RANDOM_STATE = 42
    EARLY_STOP = 100

    SCALE_POS_WEIGHT = {
        1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 5.04, 8: 2.05
    }


# =====================================================================
# AMINO ACID PROPERTIES
# =====================================================================
AA_PROPERTIES = {
    "A": {"hydro": 1.8, "vol": 88.6, "charge": 0, "polar": 0},
    "R": {"hydro": -4.5, "vol": 173.4, "charge": 1, "polar": 1},
    "N": {"hydro": -3.5, "vol": 114.1, "charge": 0, "polar": 1},
    "D": {"hydro": -3.5, "vol": 111.1, "charge": -1, "polar": 1},
    "C": {"hydro": 2.5, "vol": 108.5, "charge": 0, "polar": 0},
    "Q": {"hydro": -3.5, "vol": 143.8, "charge": 0, "polar": 1},
    "E": {"hydro": -3.5, "vol": 138.4, "charge": -1, "polar": 1},
    "G": {"hydro": -0.4, "vol": 60.1, "charge": 0, "polar": 0},
    "H": {"hydro": -3.2, "vol": 153.2, "charge": 0.5, "polar": 1},
    "I": {"hydro": 4.5, "vol": 166.7, "charge": 0, "polar": 0},
    "L": {"hydro": 3.8, "vol": 166.7, "charge": 0, "polar": 0},
    "K": {"hydro": -3.9, "vol": 168.6, "charge": 1, "polar": 1},
    "M": {"hydro": 1.9, "vol": 162.9, "charge": 0, "polar": 0},
    "F": {"hydro": 2.8, "vol": 189.9, "charge": 0, "polar": 0},
    "P": {"hydro": -1.6, "vol": 112.7, "charge": 0, "polar": 0},
    "S": {"hydro": -0.8, "vol": 89.0, "charge": 0, "polar": 1},
    "T": {"hydro": -0.7, "vol": 116.1, "charge": 0, "polar": 1},
    "W": {"hydro": -0.9, "vol": 227.8, "charge": 0, "polar": 0},
    "Y": {"hydro": -1.3, "vol": 193.6, "charge": 0, "polar": 1},
    "V": {"hydro": 4.2, "vol": 140.0, "charge": 0, "polar": 0},
}


# =====================================================================
# GPU UTILITIES
# =====================================================================
def check_gpu() -> bool:
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def get_gpu_memory() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            used, total = map(int, result.stdout.strip().split(","))
            return f"{used}/{total} MB"
    except Exception:
        pass
    return "N/A"


# =====================================================================
# DATA UTILITIES
# =====================================================================
def dataset_id_from_name(name: str) -> int:
    for part in name.replace("_", " ").split():
        if part.isdigit():
            return int(part)
    return 1


def read_repertoire(tsv_path: Path, max_seqs: Optional[int] = None) -> pd.DataFrame:
    cols = ["junction_aa", "v_call", "j_call", "templates"]
    try:
        header = pd.read_csv(tsv_path, sep="\t", nrows=0)
        usecols = [c for c in cols if c in header.columns]
        df = pd.read_csv(tsv_path, sep="\t", usecols=usecols)
    except Exception:
        return pd.DataFrame(columns=cols)

    if max_seqs and len(df) > max_seqs:
        if "templates" in df.columns:
            weights = pd.to_numeric(df["templates"], errors="coerce").fillna(1.0).values
            s = weights.sum()
            if s <= 0:
                df = df.sample(n=max_seqs, random_state=42).reset_index(drop=True)
            else:
                weights = weights / s
                idx = np.random.choice(len(df), max_seqs, replace=False, p=weights)
                df = df.iloc[idx].reset_index(drop=True)
        else:
            df = df.sample(n=max_seqs, random_state=42).reset_index(drop=True)

    for col in cols:
        if col not in df.columns:
            df[col] = "" if col != "templates" else 1.0

    df["junction_aa"] = df["junction_aa"].fillna("").astype(str)
    df["templates"] = pd.to_numeric(df["templates"], errors="coerce").fillna(1.0)
    return df


# =====================================================================
# FEATURE ENGINEERING
# =====================================================================
class FeatureExtractor:
    def __init__(self, k_list=None):
        self.k_list = k_list or [3, 4]

    def gene_family(self, gene_call: str) -> str:
        if not isinstance(gene_call, str) or not gene_call:
            return "UNK"
        return gene_call.split("*")[0].split("-")[0].upper() or "UNK"

    def extract_all(
        self,
        df: pd.DataFrame,
        pub_dict: Optional[Dict] = None,
        meta_row: Optional[pd.Series] = None,
        ds_id: int = 1
    ) -> Dict[str, float]:
        seqs = df["junction_aa"].dropna().astype(str).tolist()
        seqs = [s for s in seqs if len(s) > 0]

        features: Dict[str, float] = {}

        # 1) K-mers
        for k in self.k_list:
            c = Counter()
            total = 0
            for seq in seqs:
                if len(seq) < k:
                    continue
                for i in range(len(seq) - k + 1):
                    kmer = seq[i:i + k]
                    if all(ch in AA_PROPERTIES for ch in kmer):
                        c[kmer] += 1
                        total += 1
            if total > 0:
                features.update({f"kmer_{k}_{km}": v / total for km, v in c.items()})

        # 2) Positional k-mers (start/end)
        k_pos = 3
        start_c, end_c = Counter(), Counter()
        ns, ne = 0, 0
        for seq in seqs:
            if len(seq) < k_pos:
                continue
            sk, ek = seq[:k_pos], seq[-k_pos:]
            if all(ch in AA_PROPERTIES for ch in sk):
                start_c[sk] += 1
                ns += 1
            if all(ch in AA_PROPERTIES for ch in ek):
                end_c[ek] += 1
                ne += 1
        if ns > 0:
            features.update({f"pos_start_{km}": v / ns for km, v in start_c.most_common(20)})
        if ne > 0:
            features.update({f"pos_end_{km}": v / ne for km, v in end_c.most_common(20)})

        # 3) Physicochemical
        hydro, vol = [], []
        for seq in seqs:
            h, v = 0.0, 0.0
            cnt = 0
            for aa in seq:
                if aa in AA_PROPERTIES:
                    h += AA_PROPERTIES[aa]["hydro"]
                    v += AA_PROPERTIES[aa]["vol"]
                    cnt += 1
            if cnt > 0:
                hydro.append(h / cnt)
                vol.append(v / cnt)
        if hydro:
            features["phys_hydro_mean"] = float(np.mean(hydro))
            features["phys_vol_mean"] = float(np.mean(vol))

        # 4) V families
        if "v_call" in df.columns:
            v_fam = df["v_call"].apply(self.gene_family)
            for fam, freq in v_fam.value_counts(normalize=True).head(30).items():
                features[f"v_fam_{fam}"] = float(freq)

        # 5) Length stats
        lens = [len(s) for s in seqs]
        if lens:
            features["len_mean"] = float(np.mean(lens))
            features["len_std"] = float(np.std(lens))

        # 6) Metadata
        if meta_row is not None:
            if "sex" in meta_row.index:
                features["meta_sex_male"] = 1.0 if str(meta_row["sex"]).upper() in ["M", "MALE"] else 0.0
            if ds_id == 7 and "race" in meta_row.index:
                features["meta_race_white"] = 1.0 if "white" in str(meta_row["race"]).lower() else 0.0
            if ds_id == 7 and "sequencing_run_id" in meta_row.index:
                features["meta_run_hash"] = (hash(str(meta_row["sequencing_run_id"])) % 100) / 100.0
            if ds_id == 8:
                for hla in ["A", "B", "C", "DRB1"]:
                    if hla in meta_row.index:
                        features[f"meta_hla_{hla}"] = 1.0 if pd.notna(meta_row[hla]) else 0.0

        # 7) Public clones
        if pub_dict:
            seq_set = set(seqs)
            hits = [pub_dict[s]["score"] for s in seq_set if s in pub_dict]
            features["pub_score_sum"] = float(sum(hits))
            features["pub_hits"] = float(len(hits))

        return features


# =====================================================================
# PUBLIC CLONE MINING
# =====================================================================
def mine_public_clones(
    dataset_path: Path,
    max_files: int = 20,
    min_freq: float = 0.18,
    enrichment: float = 6.0,
    top_n: int = 2000
) -> Dict[str, Dict]:
    meta = pd.read_csv(dataset_path / "metadata.csv")
    pos_files = meta[meta["label_positive"] == True]["filename"].tolist()[:max_files]  # noqa: E712
    neg_files = meta[meta["label_positive"] == False]["filename"].tolist()[:max_files]  # noqa: E712

    if not pos_files:
        return {}

    def get_seqs(files):
        c = Counter()
        for f in files:
            try:
                df = pd.read_csv(dataset_path / f, sep="\t", usecols=["junction_aa"])
                c.update(df["junction_aa"].dropna().unique())
            except Exception:
                pass
        return c

    pos_c = get_seqs(pos_files)
    neg_c = get_seqs(neg_files)

    scored = []
    n_pos, n_neg = max(1, len(pos_files)), max(1, len(neg_files))

    for seq, count in pos_c.items():
        pf = count / n_pos
        nf = neg_c.get(seq, 0) / n_neg
        if pf >= min_freq and pf > nf * enrichment:
            score = float(np.log((pf + 1e-6) / (nf + 1e-6)))
            scored.append({"seq": seq, "score": score})

    scored.sort(key=lambda x: -x["score"])
    return {item["seq"]: item for item in scored[:top_n]}


# =====================================================================
# ENSEMBLE TRAINER (GPU FIXED)
# =====================================================================
class EnsembleTrainer:
    def __init__(self, use_gpu: bool = True, random_state: int = 42):
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.models = {}
        self.weights = {"xgb": 0.5, "lgb": 0.5}
        self.feature_cols = []

    def select_features_gpu(self, X_df: pd.DataFrame, y: np.ndarray, top_k: int = 400):
        print(f"   Selecting top {top_k} features using GPU XGBoost... ", end="")

        all_cols = X_df.columns.tolist()
        dtrain = xgb.DMatrix(X_df, label=y)

        params = {
            "tree_method": "hist",
            "device": "cuda",
            "max_depth": 4,
            "learning_rate": 0.1,
            "reg_lambda": 1.0,
            "verbosity": 0,
        }

        bst = xgb.train(params, dtrain, num_boost_round=20)
        scores = bst.get_score(importance_type="gain")  # {feature_name: gain}

        sorted_feats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [f[0] for f in sorted_feats[:top_k]]

        if len(selected) < top_k:
            remaining = [c for c in all_cols if c not in selected]
            selected.extend(remaining[:top_k - len(selected)])

        print(f"Done ({len(selected)}).")
        return selected

    def train(self, df: pd.DataFrame, ds_id: int):
        y = df["label_positive"].values.astype(np.float32)
        X_df = df.drop(columns=["ID", "dataset", "label_positive"], errors="ignore")

        self.feature_cols = self.select_features_gpu(X_df, y, top_k=Config.TOP_KMER)
        X = X_df[self.feature_cols].values.astype(np.float32)

        print(f"  Training ensemble on {len(X)} samples, {len(self.feature_cols)} cols")

        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 15,
            "seed": self.random_state,
            "scale_pos_weight": Config.SCALE_POS_WEIGHT.get(ds_id, 1.0),
            "tree_method": "hist",
            "device": "cuda",
            "verbosity": 0,
        }

        lgb_params = {
            "objective": "binary",
            "metric": "auc",
            "device": "gpu",
            "max_depth": 6,
            "learning_rate": 0.02,
            "num_leaves": 31,
            "min_child_samples": 20,
            "scale_pos_weight": Config.SCALE_POS_WEIGHT.get(ds_id, 1.0),
            "verbosity": -1,
        }

        # robust n_splits
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        min_class = max(2, min(pos, neg)) if (pos > 0 and neg > 0) else 2
        n_splits = min(Config.N_SPLITS, min_class)

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        oof_xgb = np.zeros(len(y), dtype=np.float32)
        oof_lgb = np.zeros(len(y), dtype=np.float32)

        cv_xgb, cv_lgb, best_iters = [], [], []

        for tr_idx, va_idx in kf.split(X, y):
            X_tr, X_val = X[tr_idx], X[va_idx]
            y_tr, y_val = y[tr_idx], y[va_idx]

            # XGB
            dtr = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_val)
            bst = xgb.train(
                xgb_params,
                dtr,
                num_boost_round=1000,
                evals=[(dval, "v")],
                early_stopping_rounds=Config.EARLY_STOP,
                verbose_eval=False,
            )
            oof_xgb[va_idx] = bst.predict(dval)
            cv_xgb.append(roc_auc_score(y_val, oof_xgb[va_idx]))
            best_iters.append(int(bst.best_iteration or 0))

            # LGB
            lgb_tr = lgb.Dataset(X_tr, label=y_tr)
            lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_tr)
            lgb_bst = lgb.train(
                lgb_params,
                lgb_tr,
                num_boost_round=1000,
                valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(Config.EARLY_STOP, verbose=False)],
            )
            oof_lgb[va_idx] = lgb_bst.predict(X_val)
            cv_lgb.append(roc_auc_score(y_val, oof_lgb[va_idx]))

        print(f"  CV: XGB={float(np.mean(cv_xgb)):.4f}, LGB={float(np.mean(cv_lgb)):.4f}")

        # Stacking weights (safe)
        meta = LogisticRegression(max_iter=2000)
        meta.fit(np.column_stack([oof_xgb, oof_lgb]), y)
        w = np.clip(meta.coef_[0], 0, None)
        if float(w.sum()) <= 0:
            self.weights = {"xgb": 0.5, "lgb": 0.5}
        else:
            self.weights = {"xgb": float(w[0] / w.sum()), "lgb": float(w[1] / w.sum())}

        rounds = int(np.mean(best_iters)) + 50
        rounds = max(rounds, 100)

        self.models["xgb"] = xgb.train(xgb_params, xgb.DMatrix(X, label=y), num_boost_round=rounds)
        self.models["lgb"] = lgb.train(lgb_params, lgb.Dataset(X, label=y), num_boost_round=800)

        return self, self.feature_cols, float(np.mean(cv_xgb))

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32)
        p1 = self.models["xgb"].predict(xgb.DMatrix(X))
        p2 = self.models["lgb"].predict(X)
        return p1 * self.weights["xgb"] + p2 * self.weights["lgb"]


# =====================================================================
# PARALLEL HELPERS
# =====================================================================
def process_file_parallel(row, path: Path, ds_id: int, pub_dict: Dict, extractor: FeatureExtractor):
    try:
        df = read_repertoire(path / row["filename"], Config.MAX_SEQUENCES_PER_FILE)
        feats = extractor.extract_all(df, pub_dict, row, ds_id)
        return {
            **feats,
            "ID": row.get("repertoire_id", Path(row["filename"]).stem),
            "label_positive": int(row["label_positive"]),
            "dataset": path.name,
        }
    except Exception:
        return None


def process_test_parallel(tsv: Path, path: Path, ds_id: int, pub_dict: Dict, extractor: FeatureExtractor):
    try:
        df = read_repertoire(tsv, Config.MAX_SEQUENCES_PER_FILE)
        feats = extractor.extract_all(df, pub_dict, None, ds_id)
        return {**feats, "ID": tsv.stem, "dataset": path.name}
    except Exception:
        return None


# =====================================================================
# SUBMISSION CREATION (FIXED)
# =====================================================================
def create_submission_from_final(
    final_pred_df: pd.DataFrame,
    sample_path: str | Path,
    output_path: str = "submission.csv",
) -> pd.DataFrame:
    """
    final_pred_df must contain: ['ID','dataset','label_positive_probability'] for test_dataset_* rows.
    This updates ONLY Task-1 rows (test_dataset_*) in the sample submission and leaves Task-2 intact.
    """
    sample = pd.read_csv(sample_path)

    pred_map = (
        final_pred_df.drop_duplicates(subset=["dataset", "ID"])
        .set_index(["dataset", "ID"])["label_positive_probability"]
    )

    test_mask = sample["dataset"].astype(str).str.startswith("test_dataset_")
    idx = pd.MultiIndex.from_frame(sample.loc[test_mask, ["dataset", "ID"]])

    new_vals = pred_map.reindex(idx).to_numpy()
    old_vals = sample.loc[test_mask, "label_positive_probability"].to_numpy()

    sample.loc[test_mask, "label_positive_probability"] = np.where(pd.isna(new_vals), old_vals, new_vals)
    sample.to_csv(output_path, index=False)
    return sample


# =====================================================================
# MAIN PIPELINE
# =====================================================================
def main():
    print("AIRR-ML-25: GPU OPTIMIZED PIPELINE (submission fixed)")
    gpu_ok = check_gpu()
    print(f"GPU detected: {gpu_ok} | GPU mem: {get_gpu_memory()}")

    train_sets = sorted([d.name for d in Config.TRAIN_DIR.glob("train_dataset_*")])
    test_sets = sorted([d.name for d in Config.TEST_DIR.glob("test_dataset_*")])

    extractor = FeatureExtractor(Config.K_LIST)
    bundles = {}

    for ds_name in train_sets:
        ds_id = dataset_id_from_name(ds_name)
        print(f"\nTraining on {ds_name} (id={ds_id})")

        pub_dict = mine_public_clones(
            Config.TRAIN_DIR / ds_name,
            max_files=Config.PUB_MAX_FILES,
            min_freq=Config.PUB_MIN_FREQ,
            enrichment=Config.PUB_ENRICH,
            top_n=Config.PUB_TOP_N.get(ds_id, Config.PUB_TOP_N["default"]),
        )

        meta = pd.read_csv(Config.TRAIN_DIR / ds_name / "metadata.csv")
        print("  Extracting features...")
        res = Parallel(n_jobs=12, backend="loky")(
            delayed(process_file_parallel)(row, Config.TRAIN_DIR / ds_name, ds_id, pub_dict, extractor)
            for _, row in tqdm(meta.iterrows(), total=len(meta), leave=False)
        )
        df = pd.DataFrame([r for r in res if r is not None])

        trainer = EnsembleTrainer(use_gpu=True, random_state=Config.RANDOM_STATE)
        trainer, fcols, score = trainer.train(df, ds_id)

        bundles[ds_name] = {"trainer": trainer, "cols": fcols, "pub": pub_dict, "score": score}
        print(f"  Stored bundle for {ds_name} | AUC proxy: {score:.4f}")

        del df, res
        gc.collect()

    print("\nPREDICTING...")
    preds = []
    default_train = train_sets[0] if train_sets else None

    for test_name in test_sets:
        ds_id = dataset_id_from_name(test_name)
        train_key = f"train_dataset_{ds_id}"
        if train_key not in bundles:
            train_key = default_train

        if train_key is None:
            raise RuntimeError("No training bundles available.")

        bundle = bundles[train_key]
        print(f"  {test_name} -> {train_key}")

        files = sorted((Config.TEST_DIR / test_name).glob("*.tsv"))
        res = Parallel(n_jobs=12, backend="loky")(
            delayed(process_test_parallel)(f, Config.TEST_DIR / test_name, ds_id, bundle["pub"], extractor)
            for f in tqdm(files, leave=False)
        )
        test_df = pd.DataFrame([r for r in res if r is not None])

        # Align columns to training features
        X = pd.DataFrame(0.0, index=np.arange(len(test_df)), columns=bundle["cols"])
        for c in bundle["cols"]:
            if c in test_df.columns:
                X[c] = test_df[c].astype(np.float32)

        p = bundle["trainer"].predict(X.values)

        sub_part = test_df[["ID", "dataset"]].copy()
        sub_part["label_positive_probability"] = p.astype(float)
        preds.append(sub_part)

        del test_df, X, res
        gc.collect()

    # ---- Submission (FIXED) ----
    final = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(
        columns=["ID", "dataset", "label_positive_probability"]
    )
    create_submission_from_final(final, Config.SAMPLE_SUBMISSION, "best_submission.csv")
    print("\nDone! Saved submission.csv")


if __name__ == "__main__":
    main()








