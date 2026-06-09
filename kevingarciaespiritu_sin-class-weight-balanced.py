# ================== LIBRERÍAS ==================
import os
from pathlib import Path
import re
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# ================== CONFIGURACIÓN ==================
BASE = "/kaggle/input/fake-or-real-the-impostor-hunt/data"
TRAIN_DIR = Path(BASE) / "train"
TEST_DIR  = Path(BASE) / "test"
TRAIN_CSV = Path(BASE) / "train.csv"

# ================== FUNCIONES AUXILIARES ==================
POSSIBLE_TEXT_EXTS = {".txt", ".text", ".md", ".json", ".html", ".htm", ".rst"}

def read_text_file(p: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return p.read_text(encoding=enc, errors="ignore")
        except Exception:
            continue
    try:
        return p.read_bytes().decode("utf-8", errors="ignore")
    except Exception:
        return ""

def list_text_files(pair_dir: Path):
    files = [p for p in pair_dir.rglob("*") if p.is_file() and p.suffix.lower() in POSSIBLE_TEXT_EXTS]
    if not files:
        files = [p for p in pair_dir.rglob("*") if p.is_file()]
    return sorted(files)

def id_to_dirname(id_val):
    s = str(id_val)
    if s.startswith("article_"):
        return s
    try:
        n = int(s)
        return f"article_{n:04d}"
    except ValueError:
        return s

def dirname_to_numeric_id(dirname: str) -> int:
    s = str(dirname)
    if s.startswith("article_"):
        return int(s.split("_")[1])
    return int(s)

def id_article_str(id_num: int) -> str:
    return f"article_{id_num:04d}"

_suffix_re = re.compile(r"(?:^|_)([ab01])$")
def try_extract_suffix_from_stem(stem: str):
    s = stem.lower().strip()
    m = _suffix_re.search(s)
    if m:
        return m.group(1)
    for suf in ("_a","_b","_0","_1"):
        if s.endswith(suf):
            return suf[-1]
    return None

def normalize_tid(tid: str):
    s = str(tid).strip().lower()
    if "_" in s:
        suf = s.split("_")[-1]
        if suf in {"a","b","0","1"}:
            return ("suffix", suf)
    if s in {"0","1","a","b"}:
        return ("bin", s)
    return ("exact", s)

def pick_label_for_files(files, real_tid_raw):
    mode, key = normalize_tid(real_tid_raw)
    stems = [f.stem.lower() for f in files]
    mapping = {f:0 for f in files}
    if mode == "exact":
        if key in stems:
            mapping[files[stems.index(key)]] = 1
            return mapping
        for i, st in enumerate(stems):
            if st.endswith("_"+key) or st.startswith(key+"_"):
                mapping[files[i]] = 1
                return mapping
    if mode in {"suffix","bin"}:
        for i, st in enumerate(stems):
            suf = try_extract_suffix_from_stem(st)
            if suf == key:
                mapping[files[i]] = 1
                return mapping
        if len(files) >= 2:
            mapping[files[0 if key in {"a","0"} else 1]] = 1
            return mapping
    if files:
        mapping[files[0]] = 1
    return mapping

# ================== CARGA DE DATOS ==================
train_meta = pd.read_csv(TRAIN_CSV)
if not {"id","real_text_id"}.issubset(train_meta.columns):
    raise KeyError("Faltan columnas 'id' y 'real_text_id' en train.csv")

rows, skipped = [], []
for _, r in tqdm(train_meta.iterrows(), total=len(train_meta)):
    pair_dirname = id_to_dirname(r["id"])
    pair_dir = TRAIN_DIR / pair_dirname
    if not pair_dir.exists():
        skipped.append((pair_dirname, "no_dir"));  continue

    files = list_text_files(pair_dir)
    if len(files) < 2:
        skipped.append((pair_dirname, "menos_de_2_archivos"));  continue

    lab_map = pick_label_for_files(files, r["real_text_id"])
    if 1 not in lab_map.values():
        skipped.append((pair_dirname, "no_match_real_text_id"));  continue

    for f in files[:2]:
        rows.append({
            "pair_id": pair_dirname,
            "text_id": f.stem,
            "text": read_text_file(f),
            "label": lab_map[f]
        })

print(f"Pares saltados: {len(skipped)}")
train_df = pd.DataFrame(rows)
print("Train texts:", train_df.shape)
print(train_df.head(3))

# ================== MODELO SIN CLASS_WEIGHT ==================
vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
X = vectorizer.fit_transform(train_df["text"])
y = train_df["label"]

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = LogisticRegression(max_iter=2000)  # SIN class_weight
clf.fit(X_tr, y_tr)

y_pred = clf.predict(X_va)
y_proba = clf.predict_proba(X_va)[:, 1]
print("Accuracy (val):", accuracy_score(y_va, y_pred))
print("AUC (val):", roc_auc_score(y_va, y_proba))

# ================== PREDICCIÓN Y SUBMISSION ==================
pred_rows = []
test_dirs = sorted([d for d in TEST_DIR.iterdir() if d.is_dir()])
expected_ids = [dirname_to_numeric_id(d.name) for d in test_dirs]

for d in test_dirs:
    id_num = dirname_to_numeric_id(d.name)
    files = list_text_files(d)
    if not files:
        pred_rows.append({"id": id_num, "real_text_id": f"{id_article_str(id_num)}_a"})
        continue

    files = sorted(files)[:2]
    texts = [read_text_file(f) for f in files]
    Xc = vectorizer.transform(texts)
    proba = clf.predict_proba(Xc)[:, 1]
    best_idx = int(proba.argmax())

    tag = try_extract_suffix_from_stem(files[best_idx].stem)
    if tag not in {"a", "b"}:
        tag = "a" if best_idx == 0 else "b"

    pred_rows.append({
        "id": id_num,
        "real_text_id": f"{id_article_str(id_num)}_{tag}"
    })

submission = pd.DataFrame(pred_rows)
submission = submission.sort_values("id").drop_duplicates("id").reset_index(drop=True)
submission["id"] = submission["id"].astype(int)
submission["real_text_id"] = submission["real_text_id"].astype(str)

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print("✅ Submission guardado en:", out_path)
print(submission.head())


