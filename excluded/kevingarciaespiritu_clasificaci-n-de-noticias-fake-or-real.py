# ClasificaciÃ³n de Noticias: Fake or Real? (Competencia Kaggle)


import os
from pathlib import Path
import re
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# ===================== Config =====================
BASE = "/kaggle/input/fake-or-real-the-impostor-hunt/data"
TRAIN_DIR = Path(BASE) / "train"
TEST_DIR  = Path(BASE) / "test"
TRAIN_CSV = Path(BASE) / "train.csv"

# ===================== Utils =====================
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

def detect_tid_format(train_meta: pd.DataFrame) -> str:
    """
    Devuelve: 'ab' | '01' | 'full_ab' | 'full_01'
    segÃºn cÃ³mo venga real_text_id en train.csv.
    """
    vals = train_meta["real_text_id"].astype(str).str.lower().str.strip()
    if vals.isin({"a","b"}).all():
        return "ab"
    if vals.isin({"0","1"}).all():
        return "01"
    if vals.str.endswith(("_a","_b")).all():
        return "full_ab"
    if vals.str.endswith(("_0","_1")).all():
        return "full_01"
    # fallback por mayorÃ­a
    if (vals.str.endswith(("_a","_b")).mean() > 0.8):
        return "full_ab"
    if (vals.isin({"0","1"}).mean() > 0.8):
        return "01"
    return "ab"

_suffix_re = re.compile(r"(?:^|_)([ab01])$")

def try_extract_suffix_from_stem(stem: str):
    """ Intenta extraer 'a'/'b'/'0'/'1' del nombre del archivo (stem). """
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
    """
    Marca cuÃ¡l archivo es el real segÃºn real_text_id.
    Intenta match exacto por stem, luego por sufijo, y si falla usa posiciÃ³n.
    """
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

def format_tid(id_numeric: int, chosen_file_stem: str, choice_idx: int, fmt: str) -> str:
    """
    Genera el real_text_id en el formato detectado a partir del train.
    """
    suf = try_extract_suffix_from_stem(chosen_file_stem)
    if fmt == "ab":
        return (suf if suf in {"a","b"} else ("a" if choice_idx == 0 else "b"))
    if fmt == "01":
        return (suf if suf in {"0","1"} else ("0" if choice_idx == 0 else "1"))
    if fmt == "full_ab":
        tag = suf if suf in {"a","b"} else ("a" if choice_idx == 0 else "b")
        return f"article_{id_numeric:04d}_{tag}"
    if fmt == "full_01":
        tag = suf if suf in {"0","1"} else ("0" if choice_idx == 0 else "1")
        return f"article_{id_numeric:04d}_{tag}"
    return "a" if choice_idx == 0 else "b"

def allowed_values_for_id(id_numeric: int, fmt: str):
    """ Conjunto EXACTO de valores vÃ¡lidos por id. """
    if fmt == "ab":       return {"a","b"}
    if fmt == "01":       return {"0","1"}
    if fmt == "full_ab":  return {f"article_{id_numeric:04d}_a", f"article_{id_numeric:04d}_b"}
    if fmt == "full_01":  return {f"article_{id_numeric:04d}_0", f"article_{id_numeric:04d}_1"}
    return {"a","b"}

def ensure_full_match(ids_expected, df_ids):
    se = set(ids_expected)
    sd = set(df_ids)
    return (se == sd), sorted(list(se - sd))[:10], sorted(list(sd - se))[:10]

# ===================== Cargar train.csv =====================
train_meta = pd.read_csv(TRAIN_CSV)  # columnas: id, real_text_id
if not {"id","real_text_id"}.issubset(train_meta.columns):
    raise KeyError(f"train.csv debe tener columnas id y real_text_id. Tiene: {list(train_meta.columns)}")

tid_fmt = detect_tid_format(train_meta)
print("ğŸ‘‰ Formato detectado para real_text_id:", tid_fmt)

# ===================== Construir dataset =====================
rows, skipped = [], []
for _, r in tqdm(train_meta.iterrows(), total=len(train_meta)):
    pair_dirname = id_to_dirname(r["id"])
    pair_dir = TRAIN_DIR / pair_dirname
    if not pair_dir.exists():
        skipped.append((pair_dirname, "no_dir"))
        continue

    files = list_text_files(pair_dir)
    if len(files) < 2:
        skipped.append((pair_dirname, f"menos_de_2_archivos ({len(files)})"))
        continue

    lab_map = pick_label_for_files(files, r["real_text_id"])
    if 1 not in lab_map.values():
        skipped.append((pair_dirname, "no_match_real_text_id"))
        continue

    for f in files[:2]:
        rows.append({
            "pair_id": pair_dirname,
            "text_id": f.stem,
            "text": read_text_file(f),
            "label": lab_map[f]
        })

if skipped:
    print(f"âš ï¸� Pares saltados: {len(skipped)} (ejemplos):", skipped[:5])

train_df = pd.DataFrame(rows)
if train_df.empty:
    raise RuntimeError("No se pudo construir el dataset de entrenamiento (0 filas). Revisa estructura/nombres.")

print("Train texts:", train_df.shape)
print(train_df.head(4))

# ===================== Modelo =====================
vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
X = vectorizer.fit_transform(train_df["text"])
y = train_df["label"]

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(X_tr, y_tr)

y_pred = clf.predict(X_va)
y_proba = clf.predict_proba(X_va)[:, 1]
print("Accuracy (val):", accuracy_score(y_va, y_pred))
print("AUC (val):", roc_auc_score(y_va, y_proba))

# ===================== Test: calcular best_idx por carpeta =====================
test_dirs = sorted([d for d in TEST_DIR.iterdir() if d.is_dir()])
expected_num_ids = [dirname_to_numeric_id(d.name) for d in test_dirs]
expected_art_ids = [id_article_str(dirname_to_numeric_id(d.name)) for d in test_dirs]
print("Total carpetas en test:", len(test_dirs))

choices = []  # (id_num, id_art, best_idx, chosen_stem)
for d in test_dirs:
    id_num = dirname_to_numeric_id(d.name)
    id_art = id_article_str(id_num)
    files = list_text_files(d)
    if not files:
        best_idx = 0
        chosen_stem = "a"
    else:
        files = sorted(files)[:2]
        texts = [read_text_file(f) for f in files]
        Xc = vectorizer.transform(texts)
        proba = clf.predict_proba(Xc)[:, 1]
        best_idx = int(proba.argmax())
        chosen_stem = files[best_idx].stem
    choices.append((id_num, id_art, best_idx, chosen_stem))

print("Ejemplo de 3 elecciones:", choices[:3])

# ===================== Generar TODAS las variantes de submission =====================
submissions = {}

# A) id numÃ©rico + real_text_id = a/b
rows = [{"id": idn, "real_text_id": ("a" if bidx==0 else "b")} for idn, ida, bidx, stem in choices]
submissions["submission_numeric_ab.csv"] = pd.DataFrame(rows)

# B) id numÃ©rico + real_text_id = 0/1
rows = [{"id": idn, "real_text_id": ("0" if bidx==0 else "1")} for idn, ida, bidx, stem in choices]
submissions["submission_numeric_01.csv"] = pd.DataFrame(rows)

# C) id numÃ©rico + real_text_id = article_####_a/b
rows = [{"id": idn, "real_text_id": f"{id_article_str(idn)}_{'a' if bidx==0 else 'b'}"} for idn, ida, bidx, stem in choices]
submissions["submission_numeric_full_ab.csv"] = pd.DataFrame(rows)

# D) id numÃ©rico + real_text_id = article_####_0/1
rows = [{"id": idn, "real_text_id": f"{id_article_str(idn)}_{'0' if bidx==0 else '1'}"} for idn, ida, bidx, stem in choices]
submissions["submission_numeric_full_01.csv"] = pd.DataFrame(rows)

# E) id = article_#### + real_text_id = a/b
rows = [{"id": ida, "real_text_id": ("a" if bidx==0 else "b")} for idn, ida, bidx, stem in choices]
submissions["submission_article_ab.csv"] = pd.DataFrame(rows)

# F) id = article_#### + real_text_id = 0/1
rows = [{"id": ida, "real_text_id": ("0" if bidx==0 else "1")} for idn, ida, bidx, stem in choices]
submissions["submission_article_01.csv"] = pd.DataFrame(rows)

# ===================== Validar y guardar TODAS =====================
def save_and_report(name, df, expected_ids, numeric_ids: bool):
    if numeric_ids:
        df["id"] = df["id"].astype(int)
    else:
        df["id"] = df["id"].astype(str)

    df["real_text_id"] = df["real_text_id"].astype(str)
    df = df.sort_values("id").drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)

    ids_match, missing, extra = ensure_full_match(expected_ids, df["id"].tolist())

    print(f"\n=== {name} ===")
    print("Filas:", len(df))
    print("IDs coinciden con test?:", ids_match)
    if not ids_match:
        print("Missing (primeros 10):", missing)
        print("Extra (primeros 10):", extra)
    print(df.head(3))

    out_path = f"/kaggle/working/{name}"
    df.to_csv(out_path, index=False)
    return out_path

saved_paths = []
saved_paths.append(save_and_report("submission_numeric_ab.csv",      submissions["submission_numeric_ab.csv"].copy(),      expected_num_ids, True))
saved_paths.append(save_and_report("submission_numeric_01.csv",     submissions["submission_numeric_01.csv"].copy(),     expected_num_ids, True))
saved_paths.append(save_and_report("submission_numeric_full_ab.csv",submissions["submission_numeric_full_ab.csv"].copy(),expected_num_ids, True))
saved_paths.append(save_and_report("submission_numeric_full_01.csv",submissions["submission_numeric_full_01.csv"].copy(),expected_num_ids, True))
saved_paths.append(save_and_report("submission_article_ab.csv",     submissions["submission_article_ab.csv"].copy(),     expected_art_ids, False))
saved_paths.append(save_and_report("submission_article_01.csv",     submissions["submission_article_01.csv"].copy(),     expected_art_ids, False))

print("\nArchivos generados en /kaggle/working/:")
for p in saved_paths:
    print(" -", p)


