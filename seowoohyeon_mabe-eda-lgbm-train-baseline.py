# ===============================================
# MABe mouse: train ì�¸ë�±ì‹± + ë©”íƒ€ ë§¤ì¹­ + EDA (robust) + tqdm
# ===============================================
import os, glob, json, random, warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
tqdm.pandas()  # pandas progress_apply í™œì„±í™”

warnings.filterwarnings("ignore")

# ------------------------
# ê²½ë¡œ/ì„¤ì •
# ------------------------
ROOT = "/kaggle/input/MABe-mouse-behavior-detection"
SPLIT = "train"   # "test"ë¡œ ë°”ê¿”ë�„ ë�™ì�‘
TRACKING_DIR = f"{ROOT}/{SPLIT}_tracking"
ANNOT_DIR    = f"{ROOT}/{SPLIT}_annotation"
META_CSV     = f"{ROOT}/{SPLIT}.csv"

# ì§„í–‰ë°” ì„¤ì •: ë°”ê¹¥ ë£¨í”„/ì•ˆìª½ ë£¨í”„(leave ì¶©ë�Œ ë°©ì§€)
TQDM_OUTER = dict(mininterval=0.2, leave=True)
TQDM_INNER = dict(mininterval=0.2, leave=False)

# ------------------------
# ìœ í‹¸
# ------------------------
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """ì—´ ì�´ë¦„ì�„ ì†Œë¬¸ì�� + ê³µë°±â†’ì–¸ë�”ìŠ¤ì½”ì–´ë¡œ ë³€ê²½."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def as_str(x):
    """video_id ë“± í‚¤ë¥¼ ë¬¸ì��ì—´ë¡œ ê°•ì œ."""
    if pd.isna(x):
        return None
    return str(x)

def _is_file_ok(p: Path):
    try:
        return p.is_file()
    except Exception:
        return False

def _clean_path_list(x):
    """
    ë¦¬ìŠ¤íŠ¸/ìŠ¤ì¹¼ë�¼ ì–´ë–¤ í˜•íƒœë¡œ ì™€ë�„, ì‹¤ì œ ì¡´ì�¬í•˜ëŠ” íŒŒì�¼ ê²½ë¡œ(str)ë§Œ ê±¸ëŸ¬ì„œ ë¦¬ìŠ¤íŠ¸ë¡œ.
    ë””ë ‰í† ë¦¬ëŠ” ì œì™¸.
    """
    if isinstance(x, list):
        cand = x
    elif pd.isna(x):
        cand = []
    else:
        cand = [x]
    out = []
    for v in cand:
        if isinstance(v, str) and v.lower() != "nan" and len(v.strip()) > 0:
            p = Path(v)
            if _is_file_ok(p):
                out.append(v)
    return out

def _sum_lists_strict(series):
    """groupby.agg ë•Œ ì‚¬ìš©: ìœ íš¨ íŒŒì�¼ ê²½ë¡œë§Œ ë¬¶ì–´ì„œ ë¦¬ìŠ¤íŠ¸ ë°˜í™˜"""
    out = []
    for v in series:
        if isinstance(v, list):
            for s in v:
                if isinstance(s, str) and _is_file_ok(Path(s)):
                    out.append(s)
        elif isinstance(v, str) and _is_file_ok(Path(v)):
            out.append(v)
    return out

# ------------------------
# 1) ë©”íƒ€ ë¡œë“œ
# ------------------------
meta = pd.read_csv(META_CSV)
meta = normalize_cols(meta)

if "video_id" not in meta.columns:
    raise RuntimeError("train.csvì—� video_id ì»¬ëŸ¼ì�´ ì—†ìŠµë‹ˆë‹¤.")
meta["video_id"] = meta["video_id"].map(as_str)

print("[meta] columns:", list(meta.columns)[:12], "...")

# ------------------------
# 2) ë””ë ‰í† ë¦¬ ì�¸ë�±ì‹± (tqdm ì �ìš©)
#    êµ¬ì¡°: /train_tracking/<lab_id>/<video_id>/*.parquet|*.csv
#          /train_annotation/<lab_id>/<video_id>/*.parquet|*.csv|*.json|*.jsonl
# ------------------------
def index_side(root_dir: str, kind: str):
    """
    kind: "tracking" or "annotation"
    ê°� lab_id í�´ë�” í•˜ìœ„ì—�ì„œ video_idë³„ íŒŒì�¼ ê²½ë¡œë¥¼ ìˆ˜ì§‘
    """
    rows = []
    root = Path(root_dir)
    if not root.exists():
        return pd.DataFrame(columns=["lab_id","video_id",f"{kind}_files"])

    labs = sorted([d for d in root.iterdir() if d.is_dir()])
    for lab_dir in tqdm(labs, desc=f"Index labs ({kind})", **TQDM_OUTER):
        lab_id = lab_dir.name

        vids = sorted([d for d in lab_dir.iterdir() if d.is_dir()])
        for vid_dir in tqdm(vids, desc=f"videos in {lab_id}", **TQDM_INNER):
            video_id = vid_dir.name
            files = [str(p) for p in vid_dir.iterdir() if p.is_file()]
            rows.append({"lab_id": lab_id, "video_id": as_str(video_id), f"{kind}_files": files})

        # lab ë£¨íŠ¸ì—� ë°”ë¡œ ë†“ì�¸ íŒŒì�¼ë�„ ì²˜ë¦¬(ë“œë¬¼ì§€ë§Œ ëŒ€ë¹„)
        top_files = [str(p) for p in lab_dir.iterdir() if p.is_file()]
        for f in top_files:
            p = Path(f)
            if p.suffix.lower() in [".parquet", ".csv", ".json", ".jsonl"]:
                stem = p.stem
                video_guess = stem.split("_")[0]
                rows.append({"lab_id": lab_id, "video_id": as_str(video_guess), f"{kind}_files": [str(p)]})

    if not rows:
        return pd.DataFrame(columns=["lab_id","video_id",f"{kind}_files"])

    df = pd.DataFrame(rows)
    df = (df.groupby(["lab_id","video_id"], as_index=False)
            .agg({f"{kind}_files": _sum_lists_strict}))
    return df

trk_index = index_side(TRACKING_DIR, "tracking")
ann_index = index_side(ANNOT_DIR,    "annotation")

# ------------------------
# 3) íŠ¸ë�˜í‚¹/ì–´ë…¸í…Œì�´ì…˜ ì�¸ë�±ìŠ¤ ë³‘í•© + ê²½ë¡œ ì •ë¦¬
# ------------------------
index_df = trk_index.merge(ann_index, on=["lab_id","video_id"], how="outer")

for col in ["tracking_files","annotation_files"]:
    if col not in index_df.columns:
        index_df[col] = [[] for _ in range(len(index_df))]
    index_df[col] = index_df[col].apply(_clean_path_list)

print("\n[index] rows:", len(index_df))
print(index_df.head(3))

# ------------------------
# 4) ë©”íƒ€ ë³‘í•© (lab_id + video_id â†’ ì‹¤íŒ¨ ì‹œ video_idë§Œìœ¼ë¡œ ë³´ê°•)
# ------------------------
base = index_df.copy()

merged = base.merge(meta, on=["lab_id","video_id"], how="left")
if "frames_per_second" in merged.columns and merged["frames_per_second"].isna().all():
    # lab_idê°€ ë‹¤ë¥´ê±°ë‚˜ ë¹„ì–´ì�ˆëŠ” ê³µê°œ ë³µì œì…‹ ëŒ€ë¹„: video_idë§Œìœ¼ë¡œ fallback
    meta_by_vid = meta.drop_duplicates("video_id")
    # ì¤‘ë³µ ì»¬ëŸ¼ ì •ë¦¬ í›„ ì�¬ë¨¸ì§€
    drop_cols = [c for c in meta.columns if c in merged.columns and c not in ["lab_id","video_id"]]
    merged = merged.drop(columns=drop_cols, errors="ignore")
    merged = merged.merge(meta_by_vid, on="video_id", how="left", suffixes=("","_byvid"))

fps_col_exists = "frames_per_second" in merged.columns
fps_filled = int(merged["frames_per_second"].notna().sum()) if fps_col_exists else 0
print(f"\nì´� ë¹„ë””ì˜¤(ì�¸ë�±ìŠ¤): {len(merged)}")
print(f"FPSê°€ ì±„ì›Œì§„ ë¹„ë””ì˜¤ ìˆ˜: {fps_filled}/{len(merged)}  (FPS ì»¬ëŸ¼ ì¡´ì�¬={fps_col_exists})")

# ------------------------
# 5) ìš”ì•½ í†µê³„ ê³„ì‚° (ë¹ ë¥¸ ì¶”ì • + tqdm)
# ------------------------
def frames_from_parquet_metadata(files, max_files=3):
    """Parquet metadata.num_rows ë˜�ëŠ” CSV ì�¼ë¶€ ì�½ê¸°ë¡œ ëŒ€ë�µì �ì�¸ í”„ë ˆì�„ìˆ˜ ì¶”ì •"""
    import pyarrow.parquet as pq
    nmax = None
    if not isinstance(files, list) or len(files) == 0:
        return np.nan
    cnt = 0
    for f in files:
        if cnt >= max_files: break
        p = Path(f)
        if not p.is_file(): 
            continue
        suf = p.suffix.lower()
        try:
            if suf == ".parquet":
                pf = pq.ParquetFile(f)
                n = pf.metadata.num_rows
                nmax = n if nmax is None else max(nmax, n)
                cnt += 1
            elif suf == ".csv":
                # csvëŠ” video_frameë§Œ ìƒ˜í”Œ ì�¼ë¶€ ë¡œë“œ
                tmp = pd.read_csv(f, usecols=["video_frame"], nrows=200000)
                if "video_frame" in tmp.columns and len(tmp):
                    m = pd.to_numeric(tmp["video_frame"], errors="coerce").max()
                    if pd.notna(m):
                        nmax = int(max(nmax or 0, m))
                cnt += 1
        except:
            pass
    return np.nan if nmax is None else int(nmax)

def rough_counts_from_tracking_fast(files, nrows=10000, max_files=3):
    """mouse_id/bodypart ê³ ìœ  ê°œìˆ˜ ë¹ ë¥¸ ì¶”ì •"""
    mice, parts = set(), set()
    if not isinstance(files, list) or len(files) == 0:
        return pd.Series([np.nan, np.nan])
    cnt = 0
    for f in files:
        if cnt >= max_files: break
        p = Path(f)
        if not p.is_file(): continue
        try:
            if p.suffix.lower() == ".parquet":
                tmp = pd.read_parquet(f, columns=["mouse_id","bodypart"])
            elif p.suffix.lower() == ".csv":
                tmp = pd.read_csv(f, usecols=["mouse_id","bodypart"], nrows=nrows)
            else:
                tmp = None
            if tmp is not None:
                if "mouse_id" in tmp.columns:
                    mice |= set(tmp["mouse_id"].dropna().astype(str).unique())
                if "bodypart" in tmp.columns:
                    parts |= set(tmp["bodypart"].dropna().astype(str).unique())
                cnt += 1
        except:
            pass
    nm = len(mice) if mice else np.nan
    nb = len(parts) if parts else np.nan
    return pd.Series([nm, nb])

def count_segments_from_ann(files, max_files=60):
    """annotationì—�ì„œ ì„¸ê·¸ë¨¼íŠ¸(í–‰ë�™ íƒ€ì�„ìŠ¤íŒ¬) ê°œìˆ˜ ì¹´ìš´íŠ¸"""
    n = 0
    if not isinstance(files, list) or len(files) == 0:
        return 0
    cnt = 0
    for f in files:
        if cnt >= max_files:
            break
        try:
            p = Path(f)
            if not _is_file_ok(p) or p.suffix.lower() not in [".parquet",".csv",".json",".jsonl"]:
                continue
            if p.suffix.lower() == ".parquet":
                tmp = pd.read_parquet(p)
            elif p.suffix.lower() == ".csv":
                tmp = pd.read_csv(p)
            elif p.suffix.lower() in [".json",".jsonl"]:
                tmp = pd.read_json(f, lines=(p.suffix.lower()==".jsonl"))
            else:
                tmp = None
            if tmp is not None and len(tmp):
                n += len(tmp)
            cnt += 1
        except Exception:
            pass
    return int(n)

# summary ìƒ�ì„±
summary = merged[["lab_id","video_id","tracking_files","annotation_files"]].copy()
summary["has_annotation"] = summary["annotation_files"].apply(lambda x: bool(isinstance(x, list) and len(x)))
# tqdmë¡œ ì„¸ê·¸ë¨¼íŠ¸ ì¹´ìš´íŠ¸
summary["n_segments"] = summary["annotation_files"].progress_apply(count_segments_from_ann)
# tqdmë¡œ í”„ë ˆì�„ìˆ˜ ì¶”ì •
summary["frames_est"] = summary["tracking_files"].progress_apply(frames_from_parquet_metadata)
# tqdmë¡œ mouse/bodypart ì¶”ì •
summary[["n_mice_est","n_bodyparts_est"]] = summary["tracking_files"].progress_apply(
    rough_counts_from_tracking_fast
)

# ë©”íƒ€ í•„ë“œ keep
keep_cols = [
    "frames_per_second","video_duration_sec","pix_per_cm_approx",
    "video_width_pix","video_height_pix",
    "arena_width_cm","arena_height_cm","arena_shape","arena_type",
    "body_parts_tracked","behaviors_labeled","tracking_method"
]
for c in keep_cols:
    if c not in merged.columns:
        merged[c] = np.nan

per_video = summary.merge(merged[["lab_id","video_id"]+keep_cols],
                          on=["lab_id","video_id"], how="left")

print("\n[Per-Video Summary] shape:", per_video.shape)
disp_cols = ["lab_id","video_id","has_annotation","n_segments","frames_est",
             "n_mice_est","n_bodyparts_est","frames_per_second","video_duration_sec",
             "video_width_pix","video_height_pix"]
print(per_video[disp_cols].head(8).to_string(index=False))

# ------------------------
# 6) ìƒ�ìœ„ í–‰ë�™ ì¹´ìš´íŠ¸ (annotation â†’ action)
# ------------------------
def top_actions_global(df_index, max_files=400, nrows_each=20000, topk=20):
    files = []
    for _, r in df_index.iterrows():
        fs = r.get("annotation_files") or []
        files.extend([f for f in fs if isinstance(f, str) and Path(f).is_file()])

    random.seed(0)
    random.shuffle(files)
    files = files[:max_files]

    cnt = Counter()
    for f in tqdm(files, desc="Top actions scan", **TQDM_OUTER):
        try:
            p = Path(f)
            if p.suffix.lower() == ".parquet":
                tmp = pd.read_parquet(f, columns=["action"])
            elif p.suffix.lower() == ".csv":
                tmp = pd.read_csv(f, usecols=["action"], nrows=nrows_each)
            elif p.suffix.lower() in [".json",".jsonl"]:
                tmp = pd.read_json(f, lines=(p.suffix.lower()==".jsonl"))
                tmp = tmp[["action"]] if "action" in tmp.columns else None
            else:
                tmp = None
            if tmp is None or "action" not in tmp.columns:
                continue
            vc = tmp["action"].dropna().astype(str).value_counts()
            for k, v in vc.items():
                cnt[k] += int(v)
        except Exception:
            pass

    if not cnt:
        return pd.DataFrame(columns=["action","count"])
    return pd.DataFrame(cnt.most_common(topk), columns=["action","count"])

top20_actions = top_actions_global(merged, max_files=400)
print("\n[Top Actions] (up to 20)")
print(top20_actions.to_string(index=False))

# ------------------------
# 7) EDA plots
# ------------------------
plt.figure(figsize=(14,4))
ax = plt.subplot(1,3,1)
per_video["lab_id"].value_counts().head(15).plot(kind="bar", ax=ax)
ax.set_title("Top labs by #videos")
ax.set_ylabel("#videos"); ax.set_xlabel("lab_id"); ax.tick_params(axis='x', labelrotation=75)

ax = plt.subplot(1,3,2)
per_video["frames_per_second"].dropna().astype(float).plot(kind="hist", bins=30, alpha=0.8, ax=ax)
ax.set_title("FPS distribution"); ax.set_xlabel("FPS")

ax = plt.subplot(1,3,3)
per_video["video_duration_sec"].dropna().astype(float).plot(kind="hist", bins=30, alpha=0.8, ax=ax)
ax.set_title("Duration (sec) distribution"); ax.set_xlabel("seconds")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14,4))
ax = plt.subplot(1,3,1)
per_video["frames_est"].dropna().astype(int).plot(kind="hist", bins=40, alpha=0.8, ax=ax)
ax.set_title("Estimated #frames (from tracking)"); ax.set_xlabel("#frames")

ax = plt.subplot(1,3,2)
per_video["n_mice_est"].dropna().astype(int).plot(kind="hist", bins=10, alpha=0.8, ax=ax)
ax.set_title("Estimated #mice (from tracking)"); ax.set_xlabel("#mice")

ax = plt.subplot(1,3,3)
per_video["n_bodyparts_est"].dropna().astype(int).plot(kind="hist", bins=20, alpha=0.8, ax=ax)
ax.set_title("Estimated #bodyparts (from tracking)"); ax.set_xlabel("#bodyparts")
plt.tight_layout()
plt.show()

if len(top20_actions):
    plt.figure(figsize=(8,4))
    plt.barh(top20_actions["action"], top20_actions["count"])
    plt.gca().invert_yaxis()
    plt.title("Top actions (global)")
    plt.xlabel("count")
    plt.tight_layout()
    plt.show()

# ------------------------
# 8) ì €ì�¥
# ------------------------
per_video.to_csv("/kaggle/working/mabe_train_per_video_summary.csv", index=False)
print("\nSaved: /kaggle/working/mabe_train_per_video_summary.csv")



# ============================================================
# MABe: í–‰ë�™ ì¤‘ì‹¬ EDA (ë£¨íŠ¸/í•˜ìœ„ í˜¼í•© êµ¬ì¡° robust) + tqdm
# ============================================================
import os, random, warnings
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
warnings.filterwarnings("ignore")

try:
    import seaborn as sns
    HAS_SNS = True
    sns.set_context("notebook")
except:
    HAS_SNS = False

ROOT = "/kaggle/input/MABe-mouse-behavior-detection"
SPLIT = "train"
TRACK_DIR = f"{ROOT}/{SPLIT}_tracking"
ANN_DIR   = f"{ROOT}/{SPLIT}_annotation"
META_CSV  = f"{ROOT}/{SPLIT}.csv"

ALLOWED_TRK = {".parquet",".csv"}
ALLOWED_ANN = {".parquet",".csv",".json",".jsonl"}

SAMPLE_N  = 24     # í•„ìš”ì‹œ ì¦�ê°€
MAX_RETRY = 3

# ---------------- Meta ----------------
meta = pd.read_csv(META_CSV)
meta.columns = [c.strip().lower().replace(" ","_") for c in meta.columns]
if "video_id" not in meta.columns:
    raise RuntimeError("train.csvì—� video_id ì»¬ëŸ¼ì�´ ì—†ìŠµë‹ˆë‹¤.")
meta["video_id"] = meta["video_id"].astype(str)

# ---------------- ì�¸ë�±ì‹±: tracking ----------------
def fast_unique_video_ids(file_path, max_rows=200000):
    """ë£¨íŠ¸ íŒŒì�¼ì—�ì„œ video_id ê³ ìœ ê°’ë§Œ ë¹ ë¥´ê²Œ ì¶”ì¶œ(ì—†ìœ¼ë©´ None)."""
    p = Path(file_path)
    try:
        if p.suffix.lower()==".parquet":
            df = pd.read_parquet(p, columns=["video_id"])
        elif p.suffix.lower()==".csv":
            df = pd.read_csv(p, usecols=["video_id"], nrows=max_rows)
        else:
            return None
        if "video_id" in df.columns:
            vids = df["video_id"].dropna().astype(str).unique().tolist()
            return vids
    except Exception:
        pass
    return None

def index_tracking(root):
    """
    trk_by_vid[(lab,vid)] = [files...]   # í•˜ìœ„ ë¹„ë””ì˜¤ í�´ë�”
    trk_by_lab[lab]       = [files...]   # lab ë£¨íŠ¸ íŒŒì�¼
    trk_file_vids[file]   = set(video_ids) or None
    """
    trk_by_vid = defaultdict(list)
    trk_by_lab = defaultdict(list)
    trk_file_vids = dict()

    root = Path(root)
    if not root.exists(): 
        return trk_by_vid, trk_by_lab, trk_file_vids

    labs = sorted([d for d in root.iterdir() if d.is_dir()])
    for lab_dir in tqdm(labs, desc="Index tracking labs"):
        lab = lab_dir.name
        # í•˜ìœ„ ë¹„ë””ì˜¤ í�´ë�”
        vids = sorted([d for d in lab_dir.iterdir() if d.is_dir()])
        for vd in vids:
            files = [str(p) for p in vd.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_TRK]
            if files:
                trk_by_vid[(lab, vd.name)].extend(files)
        # lab ë£¨íŠ¸ íŒŒì�¼
        top = [str(p) for p in lab_dir.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_TRK]
        if top:
            trk_by_lab[lab].extend(top)
            # video_id ì¶”ì •(ê°€ë²¼ìš´ ìƒ˜í”Œ)
            for f in top:
                vids = fast_unique_video_ids(f, max_rows=100000)
                trk_file_vids[f] = set(vids) if vids else None
    return trk_by_vid, trk_by_lab, trk_file_vids

# ---------------- ì�¸ë�±ì‹±: annotation ----------------
def index_annotations(root):
    ann_by_vid = defaultdict(list)
    ann_by_lab = defaultdict(list)
    ann_file_vids = dict()

    root = Path(root)
    if not root.exists(): 
        return ann_by_vid, ann_by_lab, ann_file_vids

    labs = sorted([d for d in root.iterdir() if d.is_dir()])
    for lab_dir in tqdm(labs, desc="Index annotation labs"):
        lab = lab_dir.name
        # í•˜ìœ„ ë¹„ë””ì˜¤ í�´ë�”
        vids = sorted([d for d in lab_dir.iterdir() if d.is_dir()])
        for vd in vids:
            files = [str(p) for p in vd.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_ANN]
            if files:
                ann_by_vid[(lab, vd.name)].extend(files)
        # lab ë£¨íŠ¸ íŒŒì�¼
        top = [str(p) for p in lab_dir.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_ANN]
        if top:
            ann_by_lab[lab].extend(top)
            for f in top:
                # video_id ë¹ ë¥´ê²Œ ì¶”ì •
                try:
                    p = Path(f)
                    if p.suffix.lower()==".parquet":
                        df = pd.read_parquet(f, columns=["video_id"])
                    elif p.suffix.lower()==".csv":
                        df = pd.read_csv(f, usecols=["video_id"], nrows=150000)
                    elif p.suffix.lower() in [".json",".jsonl"]:
                        df = pd.read_json(f, lines=(p.suffix.lower()==".jsonl"))
                        df = df[["video_id"]] if "video_id" in df.columns else None
                    else:
                        df = None
                    vids = df["video_id"].dropna().astype(str).unique().tolist() if (df is not None and "video_id" in df.columns) else None
                except Exception:
                    vids = None
                ann_file_vids[f] = set(vids) if vids else None
    return ann_by_vid, ann_by_lab, ann_file_vids

trk_by_vid, trk_by_lab, trk_file_vids = index_tracking(TRACK_DIR)
ann_by_vid, ann_by_lab, ann_file_vids = index_annotations(ANN_DIR)

print(f"[index] tracking vid-folders: {len(trk_by_vid)} | labs with root-track files: {len(trk_by_lab)}")
print(f"[index] annotation vid-folders: {len(ann_by_vid)} | labs with root-ann files: {len(ann_by_lab)}")

# ---------------- í›„ë³´ (lab, video) ë§Œë“¤ê¸° ----------------
# ë©”íƒ€ì—�ì„œ labë³„ video_id ë¦¬ìŠ¤íŠ¸
meta_labs = meta[["lab_id","video_id"]].dropna()
meta_labs["lab_id"] = meta_labs["lab_id"].astype(str)
meta_labs["video_id"] = meta_labs["video_id"].astype(str)

candidates = []
for lab, g in meta_labs.groupby("lab_id"):
    vids = g["video_id"].astype(str).unique().tolist()
    has_ann = (lab in ann_by_lab) or any((lab, v) in ann_by_vid for v in vids)
    has_trk = (lab in trk_by_lab) or any((lab, v) in trk_by_vid for v in vids)
    if not (has_ann and has_trk):
        continue
    for v in vids:
        candidates.append((lab, v))

print(f"[candidates] from meta & availability: {len(candidates)}")

# ---------------- ë¡œë�” ----------------
def read_annotations_any(path):
    p = Path(path)
    try:
        if p.suffix.lower()==".parquet":
            df = pd.read_parquet(p)
        elif p.suffix.lower()==".csv":
            df = pd.read_csv(p)
        elif p.suffix.lower() in [".json",".jsonl"]:
            df = pd.read_json(p, lines=(p.suffix.lower()==".jsonl"))
        else:
            return None
    except Exception:
        return None
    # ì»¬ëŸ¼ í†µì�¼
    ren = {
        "agent_id":"agent_id","target_id":"target_id","action":"action",
        "start_frame":"start_frame","stop_frame":"stop_frame",
        "start":"start_frame","stop":"stop_frame","end_frame":"stop_frame",
        "video_id":"video_id"
    }
    keep = [c for c in df.columns if c in ren]
    if not keep: return None
    df = df.rename(columns={c:ren[c] for c in keep})
    need = ["action","start_frame","stop_frame"]
    if not all(c in df.columns for c in need): return None
    for c in ["start_frame","stop_frame"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["action","start_frame","stop_frame"])
    return df

def read_tracking_any(path, usecols=None, nrows=None):
    p = Path(path)
    try:
        if p.suffix.lower()==".parquet":
            return pd.read_parquet(p, columns=usecols)
        elif p.suffix.lower()==".csv":
            return pd.read_csv(p, usecols=usecols, nrows=nrows)
    except Exception:
        return None
    return None

def robust_xy_cols(df):
    cols = df.columns
    vf = "video_frame" if "video_frame" in cols else None
    mid= "mouse_id"    if "mouse_id"    in cols else None
    x  = "x" if "x" in cols else ("X" if "X" in cols else None)
    y  = "y" if "y" in cols else ("Y" if "Y" in cols else None)
    if None in [vf, mid, x, y]: 
        return None
    return dict(video_frame=vf, mouse_id=mid, x=x, y=y)

# ---------------- (lab, vid) â†’ íŒŒì�¼ ì„ íƒ� ----------------
def tracking_files_for(lab, vid):
    """ê°€ëŠ¥í•˜ë©´ vid í�´ë�”, ì—†ìœ¼ë©´ lab ë£¨íŠ¸ íŒŒì�¼ë¡œ ëŒ€ì²´."""
    files = []
    if (lab, vid) in trk_by_vid:
        files += trk_by_vid[(lab, vid)]
    files += trk_by_lab.get(lab, [])
    return files

def annotation_files_for(lab, vid):
    files = []
    if (lab, vid) in ann_by_vid:
        files += ann_by_vid[(lab, vid)]
    files += ann_by_lab.get(lab, [])
    return files

# ---------------- ì„¸ê·¸ë¨¼íŠ¸ ë¡œë“œ ----------------
def load_segments_for_video(lab, vid):
    files = annotation_files_for(lab, vid)
    if not files: 
        return None
    out = []
    for f in files:
        df = read_annotations_any(f)
        if df is None or df.empty:
            continue
        if "video_id" in df.columns:
            m = df["video_id"].astype(str)==str(vid)
            if not m.any():
                # ë£¨íŠ¸ íŒŒì�¼ì�´ì§€ë§Œ ì�´ ë¹„ë””ì˜¤ê°€ ì—†ì�„ ìˆ˜ ì�ˆì�Œ â†’ ìŠ¤í‚µ
                continue
            df = df[m]
        df["lab_id"]=lab; df["video_id"]=str(vid)
        if "agent_id" not in df.columns:  df["agent_id"]=np.nan
        if "target_id" not in df.columns: df["target_id"]=df["agent_id"]
        out.append(df[["lab_id","video_id","agent_id","target_id","action","start_frame","stop_frame"]])
    return pd.concat(out, ignore_index=True) if out else None

# ---------------- íŠ¸ë�˜í‚¹ â†’ ì„¼íŠ¸ë¡œì�´ë“œ/ìŒ�íŠ¹ì§• ----------------
def centroid_per_mouse_frame(df, col):
    g = df.groupby([col["video_frame"], col["mouse_id"]], as_index=False)[[col["x"], col["y"]]].mean()
    g = g.rename(columns={col["video_frame"]:"video_frame", col["mouse_id"]:"mouse_id", col["x"]:"cx", col["y"]:"cy"})
    g = g.sort_values(["mouse_id","video_frame"])
    g["vx"] = g.groupby("mouse_id")["cx"].diff()
    g["vy"] = g.groupby("mouse_id")["cy"].diff()
    g["speed_px_per_frame"] = np.sqrt(g["vx"]**2 + g["vy"]**2)
    return g

def pairwise_features(df_cent):
    out=[]
    for f, sub in df_cent.groupby("video_frame"):
        arr = sub[["mouse_id","cx","cy","speed_px_per_frame"]].values
        for i in range(len(arr)):
            for j in range(i+1,len(arr)):
                mi, xi, yi, si = arr[i]
                mj, xj, yj, sj = arr[j]
                dist = float(np.hypot(xi-xj, yi-yj))
                rels = float(abs(si-sj))
                out.append([int(f), str(int(mi)), str(int(mj)), dist, rels])
                out.append([int(f), str(int(mj)), str(int(mi)), dist, rels])
    return pd.DataFrame(out, columns=["video_frame","agent_id","target_id","pair_dist","pair_rel_speed"])

def build_pairwise_for_video(lab, vid, fps, file_cap=3, nrows_cap=250_000):
    files = tracking_files_for(lab, vid)
    if not files: 
        return None
    # ìš°ì„ ìˆœìœ„: (ë£¨íŠ¸ íŒŒì�¼ ì¤‘ video_id í�¬í•¨/ì�¼ì¹˜) + (vid í�´ë�” íŒŒì�¼)
    def score(f):
        vs = trk_file_vids.get(f)
        return 0 if (vs is not None and str(vid) in vs) else 1
    files = sorted(files, key=score)[:file_cap]

    parts=[]
    for f in files:
        df = read_tracking_any(f, usecols=["video_frame","mouse_id","bodypart","x","y"], nrows=nrows_cap)
        if df is None or df.empty:
            df = read_tracking_any(f, usecols=["video_frame","mouse_id","x","y"], nrows=nrows_cap)
        if df is None or df.empty: 
            continue
        # video_id ì»¬ëŸ¼ì�´ ì�ˆìœ¼ë©´ í•„í„°
        if "video_id" in df.columns:
            df = df[df["video_id"].astype(str)==str(vid)]
        col = robust_xy_cols(df)
        if col is None: 
            continue
        parts.append(df[[col["video_frame"], col["mouse_id"], col["x"], col["y"]]])
    if not parts:
        return None

    T = pd.concat(parts, ignore_index=True)
    col = robust_xy_cols(T)
    cent = centroid_per_mouse_frame(T, col)
    pairs = pairwise_features(cent)
    if pairs.empty: 
        return None
    pairs["lab_id"]=lab; pairs["video_id"]=str(vid)
    pairs["fps"]=fps
    pairs["pair_rel_speed_px_s"] = pairs["pair_rel_speed"] * fps if pd.notna(fps) and fps>0 else np.nan
    return pairs

# ---------------- ìƒ˜í”Œ & ë¡œë“œ ----------------
random.seed(0)

def sample_with_retry(cands, n, max_retry):
    for k in range(max_retry):
        take = cands if len(cands)<=n*(k+1) else random.sample(cands, n*(k+1))
        segs=[]
        for lab, vid in tqdm(take, desc=f"scan annotations (try {k+1})"):
            d = load_segments_for_video(lab, vid)
            if d is not None and not d.empty:
                segs.append(d)
        if segs:
            return take, pd.concat(segs, ignore_index=True)
    return take, pd.DataFrame(columns=["lab_id","video_id","agent_id","target_id","action","start_frame","stop_frame"])

sampled, segs = sample_with_retry(candidates, SAMPLE_N, MAX_RETRY)
print(f"[info] sampled pairs: {len(sampled)} | seg rows: {len(segs)}")

# ---------------- í–‰ë�™ EDA ----------------
if len(segs)==0:
    print("âš ï¸� ìœ íš¨í•œ annotation ì„¸ê·¸ë¨¼íŠ¸ë¥¼ ì°¾ì§€ ëª»í–ˆìŠµë‹ˆë‹¤. SAMPLE_Nì�„ ë�” í‚¤ìš°ê±°ë‚˜ ë‹¤ë¥¸ labì�„ ì„ íƒ�í•˜ì„¸ìš”.")
else:
    segs["start_frame"] = pd.to_numeric(segs["start_frame"], errors="coerce").astype("Int64")
    segs["stop_frame"]  = pd.to_numeric(segs["stop_frame"],  errors="coerce").astype("Int64")
    segs = segs.dropna(subset=["start_frame","stop_frame","action"]).reset_index(drop=True)

    fps_lookup = meta.set_index("video_id")["frames_per_second"].to_dict()
    segs["fps"] = segs["video_id"].map(fps_lookup)
    segs["dur_frames"] = (segs["stop_frame"] - segs["start_frame"] + 1).clip(lower=0)
    segs["dur_sec"]    = segs["dur_frames"] / segs["fps"]

    # 1) í–‰ë�™ ë¶„í�¬
    act_counts = segs["action"].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(9,4))
    if HAS_SNS: sns.barplot(x=act_counts.index[:20], y=act_counts.values[:20], color="#6aa6ff")
    else:       plt.bar(act_counts.index[:20], act_counts.values[:20])
    plt.xticks(rotation=60, ha="right"); plt.title("Action counts (sample)"); plt.tight_layout(); plt.show()

    # 2) í–‰ë�™ë³„ ì§€ì†�ì‹œê°„
    top_actions = act_counts.head(10).index.tolist()
    dur_plot = segs.dropna(subset=["dur_sec"])
    dur_plot = dur_plot[dur_plot["action"].isin(top_actions)]
    if len(dur_plot):
        plt.figure(figsize=(10,4))
        if HAS_SNS:
            sns.boxplot(data=dur_plot, x="action", y="dur_sec", showfliers=False)
            sns.stripplot(data=dur_plot.sample(n=min(2500, len(dur_plot)), random_state=0),
                          x="action", y="dur_sec", color="k", alpha=.2, jitter=.25, size=2)
        else:
            for i,a in enumerate(top_actions,1):
                v = dur_plot[dur_plot["action"]==a]["dur_sec"].dropna()
                plt.boxplot([v], positions=[i])
            plt.xticks(range(1,len(top_actions)+1), top_actions, rotation=60, ha="right")
        plt.ylabel("duration (sec)"); plt.title("Duration per action (top)"); plt.tight_layout(); plt.show()

    # 3) í–‰ë�™ ì „ì�´
    trans = Counter()
    for (lab, vid), g in segs.groupby(["lab_id","video_id"]):
        g2 = g.sort_values(["agent_id","start_frame"])
        for agent, gg in g2.groupby("agent_id"):
            prev = None
            for a in gg["action"]:
                if prev is not None and a!=prev:
                    trans[(prev, a)] += 1
                prev = a
    if trans:
        trans_df = pd.DataFrame([(a,b,c) for (a,b),c in trans.items()], columns=["from","to","#"])
        piv = trans_df.pivot_table(index="from", columns="to", values="#", fill_value=0).astype(int)
        plt.figure(figsize=(max(6,0.6*len(piv.columns)), max(4,0.6*len(piv.index))))
        if HAS_SNS: sns.heatmap(piv, cmap="Reds")
        else:
            plt.imshow(piv.values, cmap="Reds"); plt.colorbar()
            plt.xticks(range(len(piv.columns)), piv.columns, rotation=60, ha="right")
            plt.yticks(range(len(piv.index)),   piv.index)
        plt.title("Action transition (agent-wise)"); plt.tight_layout(); plt.show()

# ---------------- pairwise íŠ¸ë�˜í‚¹ íŠ¹ì§• â†” ë�¼ë²¨ ë¹„êµ� ----------------
beh_feats=[]
for (lab, vid) in tqdm(sampled, desc="build pairwise feats"):
    fps = meta.loc[meta["video_id"]==str(vid), "frames_per_second"].dropna()
    fps = float(fps.iloc[0]) if len(fps) else np.nan
    bf = build_pairwise_for_video(lab, vid, fps)
    if bf is not None and not bf.empty:
        beh_feats.append(bf)
pair_df = pd.concat(beh_feats, ignore_index=True) if beh_feats else pd.DataFrame()

if len(segs) and not pair_df.empty:
    # frame-level ë�¼ë²¨ í™•ì�¥
    def expand_labels(seg_df, actions_keep):
        lab=[]
        for _, r in seg_df.iterrows():
            if r["action"] not in actions_keep: 
                continue
            lab.append(pd.DataFrame({
                "video_frame": np.arange(int(r["start_frame"]), int(r["stop_frame"])+1, dtype=int),
                "agent_id":    str(r.get("agent_id", np.nan)),
                "target_id":   str(r.get("target_id", np.nan)),
                "action":      r["action"]
            }))
        return pd.concat(lab, ignore_index=True) if lab else pd.DataFrame(columns=["video_frame","agent_id","target_id","action"])

    top_actions = segs["action"].value_counts().head(6).index.tolist()
    frame_labels=[]
    for (lab, vid), g in segs[segs["action"].isin(top_actions)].groupby(["lab_id","video_id"]):
        fl = expand_labels(g, top_actions)
        if len(fl):
            fl["lab_id"]=lab; fl["video_id"]=str(vid)
            frame_labels.append(fl)
    frame_labels = pd.concat(frame_labels, ignore_index=True) if frame_labels else pd.DataFrame()

    if not frame_labels.empty:
        key = ["lab_id","video_id","video_frame","agent_id","target_id"]
        for df in (pair_df, frame_labels):
            df[key] = df[key].astype(str)
        joined = pair_df.merge(frame_labels[key+["action"]], on=key, how="left")
        for a in top_actions:
            sub = joined.copy()
            sub["pos"] = sub["action"].eq(a)
            pos = sub[sub["pos"]==True]
            neg = sub[sub["pos"]==False]
            if len(pos)==0 or len(neg)==0:
                continue
            neg = neg.sample(n=min(len(pos)*2, len(neg)), random_state=0)

            fig, axes = plt.subplots(1,2, figsize=(10,3.4))
            for ax, col, title in zip(
                axes,
                ["pair_dist","pair_rel_speed_px_s"],
                ["pairwise distance (px)", "relative speed (px/s)"],
            ):
                posv = np.log1p(pos[col].dropna()); negv = np.log1p(neg[col].dropna())
                if HAS_SNS:
                    sns.kdeplot(posv, ax=ax, label=f"{a}=ON", fill=True, alpha=.35)
                    sns.kdeplot(negv, ax=ax, label="OFF",  fill=True, alpha=.35)
                else:
                    ax.hist(posv, bins=40, alpha=.5, label=f"{a}=ON", density=True)
                    ax.hist(negv, bins=40, alpha=.5, label="OFF", density=True)
                ax.set_title(f"{a}: {title} (log1p)"); ax.legend()
            plt.tight_layout(); plt.show()
else:
    print("âš ï¸� segs ë˜�ëŠ” pair_dfê°€ ë¹„ì–´ on/off ë¹„êµ� ìŠ¤í‚µ.")



# ============================================================
# Fast & Clean pipeline (effective tweaks only)
#  - Train: scan only videos that have annotations
#  - Speed toggles: FRAME_STRIDE / NEIGHBOR_TOPK / NEG_RATIO / CV_SPLITS
#  - Robust segmentation: smoothing + hysteresis + min frames + gap merge
#  - Remove zero-length segments (start==stop)
#  - Cache train pairs/labels/actions
#  - LightGBM: GPU try -> CPU fallback, early stopping
# ============================================================

import os, gc, json, math, random, shutil, subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score

# ---------------- Paths ----------------
ROOT = "/kaggle/input/MABe-mouse-behavior-detection"
TRK_DIR = f"{ROOT}/train_tracking"
ANN_DIR = f"{ROOT}/train_annotation"
TEST_TRK_DIR = f"{ROOT}/test_tracking"

# ---------------- Config (speed/quality knobs) ----------------
TOP_ACTIONS_K       = None        # None=all actions in annotations
FRAME_STRIDE        = 16           # use every Nth frame (train & test)
EXPAND_STEP         = FRAME_STRIDE
SCALE_XY            = 0.5         # scale coords (I/O ì˜�í–¥ X, ì†�ë�„ ì•ˆì „)
NEG_RATIO           = 3           # negatives downsample per action
CV_SPLITS           = 3           # GroupKFold splits
NEIGHBOR_TOPK       = 2           # None=all targets, else top-K nearest only
MAX_TRACK_FILES     = 1           # <=1 file per video (ì†�ë�„â†‘)
CSV_NROWS_CAP       = None        # None=all rows
MAX_SECONDS_TRAIN   = None        # cut head seconds if fps known (None=all)
TRAIN_ONLY_ANN_PAIRS= True        # build only annotated (agent,target) pairs
USE_GPU_TRY         = False       # try LightGBM GPU; fallback to CPU

DEBUG_MODE          = False       # True => keep only a fraction of videos
DEBUG_FRACTION      = 0.10        # when DEBUG_MODE=True

RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# ---------------- LightGBM / Fallback ----------------
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

# ============================================================
# Helpers
# ============================================================
def _which(cmd):
    try:
        return shutil.which(cmd)
    except Exception:
        return None

# FPS lookup from meta if present
fps_lookup = {}
for meta in ["train.csv", "test.csv"]:
    mp = Path(ROOT) / meta
    if mp.exists():
        try:
            m = pd.read_csv(mp)
            if {"video_id","frames_per_second"}.issubset(m.columns):
                fps_lookup.update(
                    m.assign(video_id=m["video_id"].astype(str))
                     .set_index("video_id")["frames_per_second"].to_dict()
                )
        except Exception:
            pass

def _iter_lab_entries(root_dir):
    root = Path(root_dir)
    if not root.exists(): return []
    labs = sorted([d for d in root.iterdir() if d.is_dir()])
    for lab_dir in labs:
        for ent in sorted(lab_dir.iterdir()):
            if ent.is_file() and ent.suffix.lower() in (".parquet",".csv"):
                yield lab_dir.name, ent.stem, [ent]
            elif ent.is_dir():
                files = [p for p in ent.iterdir() if p.is_file() and p.suffix.lower() in (".parquet",".csv")]
                if files:
                    yield lab_dir.name, ent.name, files

def _select_track_files(files):
    # prefer parquet, limit count
    return sorted(files, key=lambda p: (p.suffix.lower() != ".parquet", str(p)))[:MAX_TRACK_FILES]

def _read_tracking_cols(fp: Path):
    usecols = ["video_frame","mouse_id","x","y","bodypart"]
    try:
        if fp.suffix.lower()==".parquet":
            df = pd.read_parquet(fp, columns=usecols)
        else:
            hdr = pd.read_csv(fp, nrows=0)
            cols = [c for c in usecols if c in hdr.columns]
            df = pd.read_csv(fp, usecols=cols, nrows=CSV_NROWS_CAP)

        if "bodypart" not in df.columns:
            df["bodypart"] = np.nan

        df["video_frame"] = pd.to_numeric(df["video_frame"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["video_frame","mouse_id","x","y"])
        # scale coords (no I/O effect, but consistent with train/test)
        df["x"] = (df["x"].astype(np.float32) * SCALE_XY).astype(np.float32)
        df["y"] = (df["y"].astype(np.float32) * SCALE_XY).astype(np.float32)
        df["mouse_id"] = df["mouse_id"].astype(str)
        return df
    except Exception:
        return pd.DataFrame(columns=["video_frame","mouse_id","x","y","bodypart"])

def _centroid_from_tracking(trk, fps):
    cent = (trk.groupby(["video_frame","mouse_id"], as_index=False)
               .agg(x=("x","mean"), y=("y","mean")))
    if FRAME_STRIDE and FRAME_STRIDE>1:
        cent = cent[cent["video_frame"] % FRAME_STRIDE == 0]
    cent = cent.sort_values(["mouse_id","video_frame"])
    cent["vx_pf"] = cent.groupby("mouse_id")["x"].diff().fillna(0).astype(np.float32)
    cent["vy_pf"] = cent.groupby("mouse_id")["y"].diff().fillna(0).astype(np.float32)
    if fps and np.isfinite(fps) and fps>0:
        cent["vx_ps"] = (cent["vx_pf"] * fps).astype(np.float32)
        cent["vy_ps"] = (cent["vy_pf"] * fps).astype(np.float32)
    else:
        cent["vx_ps"] = cent["vx_pf"].astype(np.float32)
        cent["vy_ps"] = cent["vy_pf"].astype(np.float32)
    if MAX_SECONDS_TRAIN and fps and fps>0:
        max_frame = int(fps * MAX_SECONDS_TRAIN)
        cent = cent[cent["video_frame"] <= max_frame]
    return cent

def _pair_rows_from_cent(cent, lab_id, video_id, fps, ann_pair_filter=None):
    rows = []
    for vf, g in cent.groupby("video_frame"):
        ms = g["mouse_id"].tolist()
        n = len(ms)
        if n < 2: continue
        g = g.set_index("mouse_id")
        xs, ys = g["x"].to_numpy(), g["y"].to_numpy()
        vxs, vys = g["vx_ps"].to_numpy(), g["vy_ps"].to_numpy()
        dx = xs[:,None] - xs[None,:]
        dy = ys[:,None] - ys[None,:]
        dist = np.hypot(dx, dy)
        rvx = vxs[:,None] - vxs[None,:]
        rvy = vys[:,None] - vys[None,:]
        rs  = np.hypot(rvx, rvy)
        ids = np.array(ms, dtype=object)

        for i in range(n):
            d = dist[i].copy(); d[i] = np.inf
            if NEIGHBOR_TOPK is not None and NEIGHBOR_TOPK < n:
                nbr_idx = np.argpartition(d, NEIGHBOR_TOPK)[:NEIGHBOR_TOPK]
            else:
                nbr_idx = [j for j in range(n) if j != i]
            a = str(ids[i])
            for j in nbr_idx:
                b = str(ids[j])
                if (ann_pair_filter is not None) and ((str(lab_id), str(video_id), a, b) not in ann_pair_filter):
                    continue
                rows.append([lab_id, str(video_id), int(vf), a, b,
                             float(dist[i,j]), float(rs[i,j]), fps])
    return rows

def build_pair_df_from_dir(TRK_DIR, ann_pair_filter=None, keep_only=None):
    rows = []
    it = list(_iter_lab_entries(TRK_DIR))
    if keep_only is not None:
        keep_only = set(keep_only)
        it = [t for t in it if (t[0], t[1]) in keep_only]

    for lab_id, video_id, files in tqdm(it, desc=f"scan {Path(TRK_DIR).name}"):
        files = _select_track_files(files)
        parts = []
        for fp in files:
            df = _read_tracking_cols(fp)
            if len(df): parts.append(df)
        if not parts: continue
        trk = pd.concat(parts, ignore_index=True)
        trk["video_frame"] = trk["video_frame"].astype("Int64")
        fps = float(fps_lookup.get(str(video_id), 0) or 0)
        cent = _centroid_from_tracking(trk, fps)
        if len(cent)==0: continue
        rows.extend(_pair_rows_from_cent(cent, lab_id, video_id, fps, ann_pair_filter=ann_pair_filter))
        del trk, cent; gc.collect()

    cols = ["lab_id","video_id","video_frame","agent_id","target_id",
            "pair_dist","pair_rel_speed_px_s","fps"]
    df = pd.DataFrame(rows, columns=cols)
    if len(df):
        df[["pair_dist","pair_rel_speed_px_s","fps"]] = df[["pair_dist","pair_rel_speed_px_s","fps"]].astype(np.float32)
        for c in ["lab_id","video_id","agent_id","target_id"]:
            df[c] = df[c].astype(str)
        df["video_frame"] = df["video_frame"].astype(np.int32)
    return df

# ---------------- Annotations ----------------
def load_train_annotations(ANN_DIR):
    rows = []
    it = list(_iter_lab_entries(ANN_DIR))
    for lab_id, video_id, files in tqdm(it, desc="scan train ann"):
        for f in files:
            try:
                if f.suffix.lower()==".parquet": df = pd.read_parquet(f)
                elif f.suffix.lower()==".csv":   df = pd.read_csv(f)
                else:                             df = pd.read_json(f, lines=(f.suffix.lower()==".jsonl"))
            except Exception:
                continue
            ren = {"start":"start_frame","stop":"stop_frame","end_frame":"stop_frame"}
            df = df.rename(columns=ren)
            need = {"action","start_frame","stop_frame"}
            if not need.issubset(df.columns): continue
            for c in ["start_frame","stop_frame"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["action","start_frame","stop_frame"])
            if "agent_id" not in df.columns:  df["agent_id"] = np.nan
            if "target_id" not in df.columns: df["target_id"] = df["agent_id"]
            df["lab_id"] = lab_id; df["video_id"] = str(video_id)
            rows.append(df[["lab_id","video_id","agent_id","target_id","action","start_frame","stop_frame"]])
    ann = (pd.concat(rows, ignore_index=True)
           if rows else pd.DataFrame(columns=["lab_id","video_id","agent_id","target_id","action","start_frame","stop_frame"]))
    if len(ann):
        for c in ["lab_id","video_id","agent_id","target_id","action"]:
            ann[c] = ann[c].astype(str)
        ann["start_frame"] = ann["start_frame"].astype(np.int32)
        ann["stop_frame"]  = ann["stop_frame"].astype(np.int32)
    return ann

def expand_segments_to_frames(seg_df, actions_keep=None, step=1):
    outs = []
    for _, r in seg_df.iterrows():
        a = r["action"]
        if actions_keep is not None and a not in actions_keep: continue
        s, e = int(r["start_frame"]), int(r["stop_frame"])
        if e < s: continue
        vf = np.arange(s, e+1, step=step, dtype=np.int32)
        outs.append(pd.DataFrame({
            "video_frame": vf,
            "agent_id":   str(r.get("agent_id", np.nan)),
            "target_id":  str(r.get("target_id", np.nan)),
            "action":     a,
            "lab_id":     str(r.get("lab_id","")),
            "video_id":   str(r.get("video_id","")),
        }))
    return (pd.concat(outs, ignore_index=True)
            if outs else pd.DataFrame(columns=["video_frame","agent_id","target_id","action","lab_id","video_id"]))

def build_features(df):
    df = df.copy()
    df["pair_dist"] = pd.to_numeric(df.get("pair_dist"), errors="coerce").astype(np.float32)
    df["pair_rel_speed_px_s"] = pd.to_numeric(df.get("pair_rel_speed_px_s"), errors="coerce").astype(np.float32)
    df["fps_f"] = pd.to_numeric(df.get("fps"), errors="coerce").astype(np.float32)
    df["pair_dist_log1p"] = np.log1p(df["pair_dist"].clip(lower=0)).astype(np.float32)
    df["rel_speed_log1p"] = np.log1p(df["pair_rel_speed_px_s"].clip(lower=0)).astype(np.float32)
    feats = ["pair_dist","pair_rel_speed_px_s","pair_dist_log1p","rel_speed_log1p","fps_f"]
    return df, feats

# ---------------- Segmentation (robust) ----------------
def moving_average(x, w):
    if w <= 1: return x
    return np.convolve(x, np.ones(w)/w, mode="same")

def probs_to_segments(df_prob, action, thr, fps_lookup,
                      smooth_frac=0.35, min_len_sec=0.6, max_gap_sec=0.6,
                      frame_stride=1, thr_low_frac=0.8):
    def _fps_eff(vid):
        raw = float(fps_lookup.get(str(vid), 0) or 0)
        return (raw / max(1, frame_stride)) if raw>0 else 4.0

    out = []
    for (lab, vid, ag, tg), g in df_prob.groupby(["lab_id","video_id","agent_id","target_id"]):
        g = g.sort_values("video_frame")
        vf = g["video_frame"].to_numpy(dtype=int)
        p  = g["prob"].to_numpy(dtype=float)
        fps_eff = _fps_eff(vid)
        win = max(1, int(round(smooth_frac * fps_eff)))
        sm = moving_average(p, win) if win>1 else p

        hi, lo = float(thr), float(thr)*float(thr_low_frac)
        on = False
        mask = np.zeros_like(sm, dtype=bool)
        for i, val in enumerate(sm):
            if on:
                if val >= lo: mask[i] = True
                else: on = False
            else:
                if val >= hi: on = True; mask[i] = True

        starts, ends = [], []
        for i in range(len(mask)):
            if mask[i] and (i==0 or not mask[i-1]): starts.append(vf[i])
            if mask[i] and (i==len(mask)-1 or not mask[i+1]): ends.append(vf[i])
        k = min(len(starts), len(ends)); starts, ends = starts[:k], ends[:k]
        segs = [[int(s), int(e)] for s, e in zip(starts, ends)]

        min_len_frames = max(3, int(round(min_len_sec*fps_eff)))   # >=3 frames
        max_gap_frames = max(1, int(round(max_gap_sec*fps_eff)))

        # length filter
        segs = [s for s in segs if (s[1]-s[0]+1) >= min_len_frames]

        # merge close segments
        merged = []
        for s,e in segs:
            if not merged: merged.append([s,e]); continue
            ps,pe = merged[-1]
            if s - pe - 1 <= max_gap_frames:
                merged[-1][1] = max(pe, e)
            else:
                merged.append([s,e])

        for s,e in merged:
            if e > s:  # remove start==stop
                out.append([lab, vid, ag, tg, action, int(s), int(e)])

    return pd.DataFrame(out, columns=["lab_id","video_id","agent_id","target_id","action","start_frame","stop_frame"])

# ---------------- LGB utils ----------------
def make_lgb_params(try_gpu=True):
    base = dict(
        objective="binary", metric="auc",
        learning_rate=0.05, num_leaves=31,
        feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
        min_data_in_leaf=50, verbose=-1, force_col_wise=True,
        num_threads=os.cpu_count() or 4
    )
    if try_gpu:
        base.update({"device":"gpu","device_type":"gpu"})
    else:
        base.update({"device":"cpu","device_type":"cpu"})
    return base

def make_lgb_callbacks(early_stopping_rounds, debug_mode):
    cb = [lgb.early_stopping(early_stopping_rounds)]
    cb.append(lgb.log_evaluation(period=0 if debug_mode else 50))
    return cb

def _neg_downsample(df, label_col="y", ratio=NEG_RATIO):
    if not ratio or ratio<=0: return df
    pos = df[df[label_col]==1]; neg = df[df[label_col]==0]
    if len(pos)==0 or len(neg)==0: return df
    need = min(len(neg), int(len(pos)*ratio))
    neg_s = neg.sample(n=need, random_state=RANDOM_STATE) if len(neg)>need else neg
    return pd.concat([pos, neg_s], ignore_index=True)

# ============================================================
# Cache
# ============================================================
DBG_SIG = f"_dbg{int(DEBUG_MODE)}_{str(DEBUG_FRACTION).replace('.','p')}"
CFG_SIG = f"s{FRAME_STRIDE}_exp{EXPAND_STEP}_sc{SCALE_XY}_k{NEIGHBOR_TOPK or 'all'}_ann{int(TRAIN_ONLY_ANN_PAIRS)}_sec{MAX_SECONDS_TRAIN or 0}_mf{MAX_TRACK_FILES}{DBG_SIG}"
pair_cache = f"/kaggle/working/pair_df_tr__{CFG_SIG}.parquet"
frame_labels_cache = f"/kaggle/working/frame_labels__{CFG_SIG}.parquet"
actions_cache = f"/kaggle/working/actions__{CFG_SIG}.json"

# ============================================================
# Load / Build Train
#  - **íš¨ìœ¨ í�¬ì�¸íŠ¸**: í›ˆë ¨ ìŠ¤ìº”ì�€ ì£¼ì„�ì�´ ì¡´ì�¬í•˜ëŠ” ë¹„ë””ì˜¤ë§Œ!
# ============================================================
if Path(pair_cache).exists() and Path(frame_labels_cache).exists() and Path(actions_cache).exists():
    pair_df_tr = pd.read_parquet(pair_cache)
    frame_labels = pd.read_parquet(frame_labels_cache)
    ACTIONS = json.loads(Path(actions_cache).read_text())
    print("[cache] loaded train pairs/labels/actions")
else:
    segs = load_train_annotations(ANN_DIR)
    if segs.empty:
        raise RuntimeError("No training annotations found.")
    vc = segs["action"].value_counts()
    ACTIONS = (vc.index.tolist() if TOP_ACTIONS_K is None else vc.head(TOP_ACTIONS_K).index.tolist())

    # whitelist of (lab_id, video_id) that actually have annotations
    keep_only = segs[["lab_id","video_id"]].drop_duplicates()
    if DEBUG_MODE:
        sampled = []
        for lab, sub in keep_only.groupby("lab_id"):
            k = max(1, int(math.ceil(len(sub)*DEBUG_FRACTION)))
            sampled.append(sub.sample(n=k, random_state=RANDOM_STATE))
        keep_only = pd.concat(sampled, ignore_index=True)
    keep_only = list(map(tuple, keep_only.values.tolist()))

    ann_pair_filter = None
    if TRAIN_ONLY_ANN_PAIRS:
        uniq_pairs = segs[["lab_id","video_id","agent_id","target_id"]].drop_duplicates()
        ann_pair_filter = set(map(tuple, uniq_pairs.values.tolist()))

    pair_df_tr = build_pair_df_from_dir(TRK_DIR, ann_pair_filter=ann_pair_filter, keep_only=keep_only)
    frame_labels = expand_segments_to_frames(segs, actions_keep=ACTIONS, step=EXPAND_STEP)

    for df_ in (pair_df_tr, frame_labels):
        for c in ["lab_id","video_id","video_frame","agent_id","target_id"]:
            if c in df_.columns:
                df_[c] = df_[c].astype(str)

    pair_df_tr.to_parquet(pair_cache, index=False)
    frame_labels.to_parquet(frame_labels_cache, index=False)
    Path(actions_cache).write_text(json.dumps(ACTIONS))
    print("[cache] saved train pairs/labels/actions")

print(f"[train] actions: {len(ACTIONS)} | train pairs: {pair_df_tr.shape}")

# ============================================================
# Train per action
# ============================================================
MODELS, VAL_SCORES = {}, {}
key_cols = ["lab_id","video_id","video_frame","agent_id","target_id"]
NUM_BOOST_ROUND = 600 if DEBUG_MODE else 1200
EARLY_STOP_ROUND = 20 if DEBUG_MODE else 50

for action in ACTIONS:
    print(f"\n=== Train [{action}] ===")
    lab_a = frame_labels[frame_labels["action"]==action].copy()
    lab_a["y"] = 1
    data = pair_df_tr.merge(lab_a[key_cols+["y"]], on=key_cols, how="left")
    data["y"] = data["y"].fillna(0).astype(np.int8)

    data, feat_cols = build_features(data)
    data = _neg_downsample(data, "y", NEG_RATIO)

    X = data[feat_cols]
    y = data["y"].values.astype(np.int8)
    groups = data["video_id"].astype(str).values

    gkf = GroupKFold(n_splits=max(2, min(CV_SPLITS, len(np.unique(groups)))))
    f1s, aucs, thr_list, iter_list = [], [], [], []
    if HAS_LGB: params = make_lgb_params(try_gpu=USE_GPU_TRY)
    oof = np.zeros(len(X), dtype=np.float32)

    for tr_idx, va_idx in gkf.split(X, y, groups):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        if HAS_LGB:
            dtr, dva = lgb.Dataset(Xtr, label=ytr), lgb.Dataset(Xva, label=yva, reference=None)
            tried_gpu = USE_GPU_TRY
            try:
                booster = lgb.train(
                    params, dtr,
                    num_boost_round=NUM_BOOST_ROUND,
                    valid_sets=[dva], valid_names=["val"],
                    callbacks=make_lgb_callbacks(EARLY_STOP_ROUND, DEBUG_MODE),
                )
            except Exception as e:
                if tried_gpu:
                    print(f"âš ï¸� GPU failed â†’ CPU: {e}")
                    booster = lgb.train(
                        make_lgb_params(try_gpu=False), dtr,
                        num_boost_round=NUM_BOOST_ROUND,
                        valid_sets=[dva], valid_names=["val"],
                        callbacks=make_lgb_callbacks(EARLY_STOP_ROUND, DEBUG_MODE),
                    )
                else:
                    raise
            p = booster.predict(Xva, num_iteration=booster.best_iteration)
            best_iter = booster.best_iteration
        else:
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            sc = StandardScaler()
            Xtr_s, Xva_s = sc.fit_transform(Xtr), sc.transform(Xva)
            clf = LogisticRegression(max_iter=200, class_weight="balanced")
            clf.fit(Xtr_s, ytr)
            p = clf.predict_proba(Xva_s)[:,1]
            best_iter = None

        oof[va_idx] = p
        thrs = np.linspace(0.1, 0.9, 17)
        f1_list = [f1_score(yva, (p>=t).astype(int)) for t in thrs]
        thr_list.append(float(thrs[int(np.argmax(f1_list))]))
        try: aucs.append(roc_auc_score(yva, p))
        except ValueError: aucs.append(np.nan)
        f1s.append(max(f1_list))
        if best_iter is not None: iter_list.append(best_iter)

    # full fit with median best_iter
    if HAS_LGB:
        best_iters = int(np.median(iter_list)) if len(iter_list) else (300 if DEBUG_MODE else 600)
        dall = lgb.Dataset(X, label=y)
        try:
            booster = lgb.train(make_lgb_params(try_gpu=USE_GPU_TRY), dall,
                                num_boost_round=best_iters,
                                valid_sets=[dall], valid_names=["train"])
        except Exception as e:
            if USE_GPU_TRY:
                print(f"âš ï¸� full-train GPUâ†’CPU: {e}")
                booster = lgb.train(make_lgb_params(try_gpu=False), dall,
                                    num_boost_round=best_iters,
                                    valid_sets=[dall], valid_names=["train"])
            else:
                raise
        entry = dict(model=booster, feat_cols=feat_cols,
                     thr=float(np.nanmedian(thr_list)),
                     best_iter=best_iters, kind="lgb")
    else:
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        sc_all = StandardScaler(); Xs = sc_all.fit_transform(X)
        clf_all = LogisticRegression(max_iter=200, class_weight="balanced")
        clf_all.fit(Xs, y)
        entry = dict(model=(sc_all, clf_all), feat_cols=feat_cols,
                     thr=float(np.nanmedian(thr_list)),
                     best_iter=None, kind="logreg")

    MODELS[action] = entry
    VAL_SCORES[action] = dict(
        auc=float(np.nanmean(aucs)), f1=float(np.nanmean(f1s)), thr=float(np.nanmedian(thr_list))
    )
    print(f"[{action}] CV AUC={VAL_SCORES[action]['auc']:.3f} | F1={VAL_SCORES[action]['f1']:.3f} | thr={VAL_SCORES[action]['thr']:.2f}")
    del data, X, y; gc.collect()

# ============================================================
# Build test pairs (same stride & scaling)
# ============================================================
def build_pair_df_test(TEST_TRK_DIR):
    # test ì „ë¶€ ìŠ¤ìº” (í›ˆë ¨ì²˜ëŸ¼ í•„í„° ë¶ˆê°€)
    return build_pair_df_from_dir(TEST_TRK_DIR, ann_pair_filter=None, keep_only=None)

pair_df_test = build_pair_df_test(TEST_TRK_DIR)
print("[test] pairs:", pair_df_test.shape)

import re

def _nat_key(s):
    s = str(s)
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]

def build_mouse_label_map(pair_df):
    if pair_df is None or len(pair_df) == 0:
        return {}
    m = {}
    for vid, g in pair_df.groupby("video_id"):
        ids = (
            pd.Index(g["agent_id"].astype(str))
              .append(pd.Index(g["target_id"].astype(str)))
              .unique()
        )
        ids_sorted = sorted(ids, key=_nat_key)
        for i, mid in enumerate(ids_sorted):
            m[(str(vid), str(mid))] = f"mouse{i+1}"
    return m

mouse_map = build_mouse_label_map(pair_df_test)

# ============================================================
# Predict & segment
# ============================================================
def predict_action_probs(test_df, action, entry):
    feat_cols = entry["feat_cols"]; X = test_df[feat_cols]
    if entry["kind"]=="lgb":
        booster = entry["model"]; return booster.predict(X, num_iteration=entry["best_iter"])
    sc, clf = entry["model"];    return clf.predict_proba(sc.transform(X))[:,1]

if len(pair_df_test):
    test_df, _ = build_features(pair_df_test)
    key_cols = ["lab_id","video_id","agent_id","target_id","video_frame"]
    seg_list = []
    for action in ACTIONS:
        entry = MODELS[action]
        for c in entry["feat_cols"]:
            if c not in test_df.columns: test_df[c] = np.nan
        probs = predict_action_probs(test_df, action, entry)
        df_prob = pd.concat([test_df[key_cols].reset_index(drop=True),
                             pd.Series(probs, name="prob")], axis=1)
        segs = probs_to_segments(
            df_prob, action, entry["thr"], fps_lookup,
            smooth_frac=0.8, min_len_sec=2, max_gap_sec=1,
            frame_stride=FRAME_STRIDE, thr_low_frac=0.8
        )
        seg_list.append(segs)

    final_pred = (pd.concat(seg_list, ignore_index=True)
                  if seg_list else pd.DataFrame(columns=["lab_id","video_id","agent_id","target_id","action","start_frame","stop_frame"]))
else:
    final_pred = pd.DataFrame(columns=["lab_id","video_id","agent_id","target_id","action","start_frame","stop_frame"])

# (double guard) remove zero-length once more if any
if len(final_pred):
    final_pred = final_pred[final_pred["stop_frame"] > final_pred["start_frame"]].reset_index(drop=True)

print(f"[pred] segments: {len(final_pred)}")


# ============================================================
# ğŸ”§ Clean & Coalesce before saving (drop-in patch)
#  - í—ˆìš© ë�¼ë²¨ë§Œ ìœ ì§€
#  - ì ‘í•©/ì˜¤íƒˆì�� ë�¼ë²¨ ì��ë�™ ë³´ì •
#  - ê°€ê¹Œìš´ êµ¬ê°„ ë³‘í•©(í”„ë ˆì�„ ê°„ê²© <= FRAME_STRIDE)
# ============================================================

# 1) í—ˆìš© ë�¼ë²¨ ì…‹ (í›ˆë ¨ ì£¼ì„�ì—�ì„œ ë½‘ì�€ ACTIONSì™€ êµ�ì§‘í•©)
KNOWN_ALLOWED = {"approach", "avoid", "chase", "attack"}
ALLOWED_ACTIONS = [a.lower() for a in ACTIONS if str(a).lower() in KNOWN_ALLOWED]
if not ALLOWED_ACTIONS:  # ìº�ì‹œ/ì£¼ì„� ì�´ìƒ� ì‹œ ì•ˆì „ ê¸°ë³¸ê°’
    ALLOWED_ACTIONS = sorted(list(KNOWN_ALLOWED))
ALLOWED_SET = set(ALLOWED_ACTIONS)

def _clean_action(a, allowed_set=ALLOWED_SET, priority=ALLOWED_ACTIONS):
    if a is None or (isinstance(a, float) and np.isnan(a)): 
        return np.nan
    s = str(a).strip().lower()
    # ì •í™• ë§¤ì¹˜
    if s in allowed_set:
        return s
    # ë¶™ê±°ë‚˜ ì„�ì�¸ ì¼€ì�´ìŠ¤("chaseattack", "submitchase" ë“±): í�¬í•¨ë�œ í•©ë²• í† í�°ë§Œ ì¶”ì¶œ
    hits = [lab for lab in allowed_set if lab in s]
    if len(hits) == 1:
        return hits[0]
    # ì—¬ëŸ¬ ê°œê°€ ë�™ì‹œì—� ë“¤ì–´ì�ˆë‹¤ë©´ ìš°ì„ ìˆœìœ„ë¡œ 1ê°œ ì„ íƒ�
    for lab in priority:
        if lab in hits:
            return lab
    return np.nan  # ë��ê¹Œì§€ ê²°ì • ëª» í•˜ë©´ ë²„ë¦¼

def _coalesce_segments(df, max_gap_frames=FRAME_STRIDE):
    """ê°™ì�€ (video, agent, target, action)ì—�ì„œ ì�¸ì ‘/ê²¹ì¹¨ êµ¬ê°„ ë³‘í•©"""
    keys = ["video_id","agent_id","target_id","action"]
    out = []
    df2 = df.copy()
    df2["start_frame"] = df2["start_frame"].astype(int)
    df2["stop_frame"]  = df2["stop_frame"].astype(int)
    for key, g in df2.sort_values(keys + ["start_frame","stop_frame"]).groupby(keys, dropna=False):
        cur_s = cur_e = None
        for s, e in g[["start_frame","stop_frame"]].itertuples(index=False, name=None):
            if cur_s is None:
                cur_s, cur_e = s, e
                continue
            # ì�¸ì ‘/ê²¹ì¹¨ì�´ë©´ ë³‘í•© (ê°„ê²© <= max_gap_frames)
            if s <= cur_e + max_gap_frames:
                cur_e = max(cur_e, e)
            else:
                out.append((*key, cur_s, cur_e))
                cur_s, cur_e = s, e
        if cur_s is not None:
            out.append((*key, cur_s, cur_e))
    return pd.DataFrame(out, columns=keys+["start_frame","stop_frame"])

# --- ì �ìš©: final_pred ì •ë¦¬ ---
if len(final_pred):
    # 1) ë�¼ë²¨ ì •ë¦¬
    final_pred["action"] = final_pred["action"].apply(_clean_action)
    final_pred = final_pred.dropna(subset=["action"]).reset_index(drop=True)

    # 2) (ì˜µì…˜) êµ¬ê°„ ë³‘í•© â€“ stride ê°„ê²© ì�´í•˜ë©´ í•˜ë‚˜ë¡œ í•©ì¹˜ê¸°
    final_pred = _coalesce_segments(final_pred, max_gap_frames=FRAME_STRIDE)

    # 3) ì�˜ëª»ë�œ êµ¬ê°„ ì œê±°
    final_pred = final_pred[final_pred["stop_frame"] > final_pred["start_frame"]].reset_index(drop=True)

    # ë””ë²„ê·¸: ë‚¨ì�€ ë�¼ë²¨ í™•ì�¸
    print("[post] actions kept:", sorted(final_pred["action"].unique()))



# ============================================================
# Save submission
# ============================================================
sub = final_pred[["video_id","agent_id","target_id","action","start_frame","stop_frame"]].copy()

if len(sub):
    sub["agent_id"]  = [mouse_map.get((str(v), str(a)),  "mouse1") for v, a in zip(sub["video_id"], sub["agent_id"])]
    sub["target_id"] = [mouse_map.get((str(v), str(t)),  "mouse1") for v, t in zip(sub["video_id"], sub["target_id"])]
sub = sub.reset_index(drop=True)
sub.insert(0, "row_id", np.arange(len(sub)))

save_path = "/kaggle/working/submission.csv"
sub.to_csv(save_path, index=False)
print(f"[save] {save_path} (rows={len(sub)})")
try:
    display(sub.head(20))
except Exception:
    pass





