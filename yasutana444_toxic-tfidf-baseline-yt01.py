# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# -------------------------------
# 0) パッケージの準備
# -------------------------------
# ・iterative-stratification: 多ラベルのラベル比率を保つKFold（重要）
!pip -q install iterative-stratification==0.1.7

import os, re, gc, sys, random, json, math, glob, zipfile
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold



# -------------------------------
# 1) 実験の基本設定（必要に応じて変更）
# -------------------------------
SEED = 42
TARGETS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]  # 予測する6ラベル
TEXTCOL = "comment_text"   # テキスト列名（Jigsaw互換）
IDCOL   = "id"             # ID列名
N_SPLITS = 3               # 3-fold CV
USE_LIGHT_FEATURES = True  # 追加の軽量特徴（大文字比率・記号連打など）を使うか
max_features_word = 100_000
max_features_char = 100_000
MIN_DF = 5

# Transformer（RoBERTa）オプション
USE_TRANSFORMER   = False              # ← True にするとRoBERTaで追加学習＆ブレンド
TRANSFORMER_MODEL = "roberta-base"     # 例: "microsoft/deberta-v3-base"
TRANS_EPOCHS      = 2                  # 軽めにエポック2で感触確認
TRANS_MAX_LEN     = 192                # 先頭〜末尾を並行に使うなら短めから
TRANS_BATCH       = 16                 # VRAMに応じて調整（GPU推奨）

# 乱数固定（再現性）
def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed)
seed_everything()


# =========================================================
# 2) /kaggle/input を探索 → .zip を自動展開 → train/test/sample_submission を検出
# =========================================================

def list_input(top="/kaggle/input", depth=3, max_items=200):
    """学習前のデバッグ用：/kaggle/input 配下の様子を上限つきで表示"""
    print(">>> Tree of", top)
    cnt = 0
    for root, dirs, files in os.walk(top):
        d = root[len(top):].count(os.sep)
        if d > depth: 
            continue
        for f in files:
            print(os.path.join(root, f))
            cnt += 1
            if cnt >= max_items:
                print("... (truncated)")
                return

def unzip_all_zips_under(root, outdir="/kaggle/working/_unzipped"):
    """
    /kaggle/input 配下に *.zip しかないケースがあるため、
    一括で /kaggle/working/_unzipped に展開しておく。
    """
    os.makedirs(outdir, exist_ok=True)
    unzipped_dirs = set()
    for z in glob.glob(os.path.join(root, "**/*.zip"), recursive=True):
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(outdir)  # 中身を丸ごと展開（CSVが複数でもOK）
                unzipped_dirs.add(outdir)
        except zipfile.BadZipFile:
            print("Bad zip skipped:", z)
    return list(unzipped_dirs)

def find_trio_dirs(search_roots):
    """
    指定ルート配下から「train/test/sample_submission の3CSVが揃うディレクトリ」を列挙して返す。
    名前が多少違っていても（train_*.csv など）拾えるよう緩めの判定も実装。
    """
    cands = []
    for root in search_roots:
        # root配下の全CSVのあるディレクトリを集める
        for d in sorted({os.path.dirname(p) for p in glob.glob(os.path.join(root, "**/*.csv"), recursive=True)}):
            files = {os.path.basename(p).lower() for p in glob.glob(os.path.join(d, "*.csv"))}
            # 厳密一致
            need = {"train.csv","test.csv","sample_submission.csv"}
            if need.issubset(files):
                cands.append(d); continue
            # 緩い一致（train_*.csv / sample_*submission*.csv など）
            has_train = any(fn.startswith("train") and fn.endswith(".csv") for fn in files)
            has_test  = any(fn.startswith("test")  and fn.endswith(".csv") for fn in files)
            has_sub   = any(("sample" in fn and "submission" in fn and fn.endswith(".csv")) for fn in files)
            if has_train and has_test and has_sub:
                cands.append(d)
    return sorted(set(cands))

def pick_csv(dirpath, pref_name, fallback_starts=None, must_contain=None):
    """
    dirpath 内から望ましいCSVを1つ選ぶ:
      - pref_name に完全一致があればそれ
      - なければ fallback_starts（接頭辞）や must_contain（含む語）で代替を探す
    """
    cand = glob.glob(os.path.join(dirpath, pref_name))
    if cand: 
        return cand[0]
    falls = sorted(glob.glob(os.path.join(dirpath, "*.csv")))
    for p in falls:
        bn = os.path.basename(p).lower()
        if fallback_starts and any(bn.startswith(s) for s in fallback_starts):
            return p
        if must_contain and all(t in bn for t in must_contain):
            return p
    return None

# 現在の /kaggle/input の構造を出力（どんな状況でも原因が分かりやすい）
list_input("/kaggle/input", depth=3)

# .zip を検出していれば /kaggle/working/_unzipped に自動展開
unzipped_dirs = unzip_all_zips_under("/kaggle/input", outdir="/kaggle/working/_unzipped")

# 探索対象：/kaggle/input 全域 + 解凍先 + /kaggle/working（保険）
search_roots = ["/kaggle/input"]
if unzipped_dirs:
    search_roots += unzipped_dirs
search_roots += ["/kaggle/working"]

# 3点セットを含む候補ディレクトリを取得
trio_dirs = find_trio_dirs(search_roots)
if not trio_dirs:
    raise FileNotFoundError("train/test/sample_submission の3点セットが見つかりません。Add Data と ZIP展開を確認してください。")

# 通常は候補は1つ。最初の候補を採用
DATA_DIR = trio_dirs[0]
train_path = pick_csv(DATA_DIR, "train.csv", fallback_starts=["train"])
test_path  = pick_csv(DATA_DIR, "test.csv",  fallback_starts=["test"])
sub_path   = pick_csv(DATA_DIR, "sample_submission.csv", must_contain=["sample","submission"])

print("\n[Resolved paths]")
print("DATA_DIR :", DATA_DIR)
print("train_csv:", train_path)
print("test_csv :", test_path)
print("sample  :", sub_path)
assert all([train_path, test_path, sub_path]), "3ファイルのパス解決に失敗しました"



# =========================================================
# 3) データ読み込み & “壊さない”前処理（最低限の正規化）
# =========================================================
train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)
sub   = pd.read_csv(sub_path)

print("\ntrain shape:", train.shape, "| test shape:", test.shape)
print("positives  :", train[TARGETS].sum().to_dict())  # 各ラベルの陽性数（不均衡の把握）

def normalize_text(s: str) -> str:
    """
    重要：毒性検出は“綴りそのもの（伏字・記号・伸ばし）”が効く。
    過度なクリーニングは避け、URL/メンションをマスク→空白整理のみ。
    """
    if not isinstance(s, str): 
        return ""
    s = s.strip()
    s = re.sub(r"http\S+|www\.\S+", "<URL>", s)      # URLは消さずに <URL> に置換
    s = re.sub(r"@[A-Za-z0-9_]+", "<USER>", s)       # @user を <USER> に置換
    s = re.sub(r"\s+", " ", s)                       # 連続空白を1個に
    return s

train["text_norm"] = train[TEXTCOL].apply(normalize_text)
test["text_norm"]  = test[TEXTCOL].apply(normalize_text)


# =========================================================
# 5) （任意）軽量メタ特徴を追加（疎行列に横結合）
# =========================================================
# 大文字比率、!/? 連打、引用記号の多さ、<URL>/<USER>の有無など。
# 効けば AUC +0.002〜0.01 程度の上振れが期待できることが多い。
def light_features(texts: pd.Series) -> csr_matrix:
    feats = np.zeros((len(texts), 8), dtype=np.float32)
    for i, s in enumerate(texts.astype(str).values):
        L = max(len(s), 1)
        caps = sum(1 for ch in s if ch.isupper())
        ex = s.count("!")
        qm = s.count("?")
        quotes = s.count('"') + s.count("'")
        # 連続記号（!!??など）の最長長さ
        rep_punct = max([len(m) for m in re.findall(r'([!?.,;:\-_=])\1{1,}', s)] or [0])
        has_url  = 1.0 if "<URL>"  in s else 0.0
        has_user = 1.0 if "<USER>" in s else 0.0
        feats[i] = [caps/L, ex/L, qm/L, quotes/L, rep_punct, has_url, has_user, L]
    return csr_matrix(feats)




# =========================================================
# 4) TF-IDF（軽量＆高速化設定）
# =========================================================
WORD_NGRAM = (1,2)
CHAR_NGRAM = (3,4)                # 3–4 に絞ると計算が軽くて強いことが多い
max_features_word = 100_000       # ← 20万→10万
max_features_char = 100_000
MIN_DF = 5                        # ← 3→5（激レアn-gramを捨てる）

word_vec = TfidfVectorizer(
    analyzer="word", ngram_range=WORD_NGRAM,
    min_df=MIN_DF, max_features=max_features_word,
    lowercase=True, strip_accents="unicode", sublinear_tf=True,
    dtype=np.float32               # ← float32で軽量化
)
char_vec = TfidfVectorizer(
    analyzer="char", ngram_range=CHAR_NGRAM,
    min_df=MIN_DF, max_features=max_features_char,
    lowercase=True, strip_accents=None, sublinear_tf=True,
    dtype=np.float32
)

X_word = word_vec.fit_transform(train["text_norm"])
X_char = char_vec.fit_transform(train["text_norm"])
X = hstack([X_word, X_char]).tocsr()

X_test_word = word_vec.transform(test["text_norm"])
X_test_char = char_vec.transform(test["text_norm"])
X_test = hstack([X_test_word, X_test_char]).tocsr()

y = train[TARGETS].values.astype(np.int8)
print("X shape     :", X.shape)
print("X_test shape:", X_test.shape)

# （任意）軽量メタ特徴：有効ならそのまま使えますが、速度優先ならオフでもOK
if USE_LIGHT_FEATURES:
    X_lite      = light_features(train["text_norm"])
    X_test_lite = light_features(test["text_norm"])
    X      = hstack([X, X_lite]).tocsr()
    X_test = hstack([X_test, X_test_lite]).tocsr()
    print("X (with light feats):", X.shape)


# =========================================================
# 6) 学習・検証（高速版：liblinear + 進捗ログ）
# =========================================================
from time import time
from sklearn.linear_model import LogisticRegression
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# 開発中は3foldで十分（仕上げで5に戻す）
N_SPLITS_FAST = 3 if 'N_SPLITS_FAST' not in globals() else N_SPLITS_FAST
mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS_FAST, shuffle=True, random_state=SEED)

def train_one_label_fast(oof_pred, test_pred, label_idx, tr_idx, va_idx, label_name):
    # liblinear は二値の疎×高次元に強く、確率出力も可能
    clf = LogisticRegression(
        solver="liblinear",
        penalty="l2",
        C=1.0,
        max_iter=500,
        class_weight="balanced",
        random_state=SEED
    )
    t0 = time()
    clf.fit(X[tr_idx], y[tr_idx, label_idx])
    oof_pred[va_idx, label_idx] = clf.predict_proba(X[va_idx])[:, 1]
    test_pred[:, label_idx]     += clf.predict_proba(X_test)[:, 1]
    print(f"    - {label_name:<14} done in {time()-t0:.1f}s")
    return clf

oof = np.zeros((len(train), len(TARGETS)), dtype=float)
test_pred = np.zeros((len(test),  len(TARGETS)), dtype=float)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(mskf.split(X, y), 1):
    print(f"\n==== Fold {fold}/{mskf.n_splits} ====")
    for j, tgt in enumerate(TARGETS):
        _ = train_one_label_fast(oof, test_pred, j, tr_idx, va_idx, tgt)
    # fold結果
    aucs = [roc_auc_score(y[va_idx, j], oof[va_idx, j]) for j in range(len(TARGETS))]
    print("  AUC:", {t: round(a,4) for t,a in zip(TARGETS, aucs)}, "| mean:", round(np.mean(aucs), 4))
    fold_scores.append(aucs)

# テスト予測のfold平均化
test_pred /= mskf.n_splits

# OOFの最終AUC
label_auc = {tgt: roc_auc_score(y[:, i], oof[:, i]) for i, tgt in enumerate(TARGETS)}
macro_auc = float(np.mean(list(label_auc.values())))
print("\nOOF AUC (label-wise):", {k: round(v, 4) for k, v in label_auc.items()})
print("OOF AUC (macro mean):", round(macro_auc, 4))


# =========================================================
# 7) 提出CSVの作成 & 誤り解析（FP/FNサンプルの保存） with safety checks
# =========================================================
submit = sub.copy()
submit[TARGETS] = test_pred

# 1) 提出ファイルを /kaggle/working に保存
SUB_PATH = "/kaggle/working/submission_tfidf_lr.csv"
submit.to_csv(SUB_PATH, index=False)

# 2) 誤りサンプルも保存（任意）
ERR_PATH = "/kaggle/working/error_samples.csv"
err_rows = []
for i, tgt in enumerate(TARGETS):
    true = y[:, i]; prob = oof[:, i]
    df_tmp = pd.DataFrame({
        "id": train[IDCOL],
        "text": train["comment_text"],
        "norm": train["text_norm"],
        "true": true,
        "prob": prob
    })
    df_tmp["error_type"] = np.where((df_tmp["true"]==0) & (df_tmp["prob"]>=0.8), "FP(>=0.8)",
                            np.where((df_tmp["true"]==1) & (df_tmp["prob"]<=0.2), "FN(<=0.2)", "OK"))
    fp_top = df_tmp.query("error_type=='FP(>=0.8)'").sort_values("prob", ascending=False).head(30).copy()
    fn_top = df_tmp.query("error_type=='FN(<=0.2)'").sort_values("prob", ascending=True).head(30).copy()
    fp_top["label"] = tgt; fn_top["label"] = tgt
    err_rows.append(fp_top); err_rows.append(fn_top)
pd.concat(err_rows, axis=0, ignore_index=True).to_csv(ERR_PATH, index=False)

# 3) ちゃんと出来たかチェック（Versionの失敗箇所を可視化）
import os
def _check(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[SaveVersion] {name} が作成されていません: {path}\n"
                                f"- Add data の設定と、前のセルに例外が無いか確認してください。")
    size = os.path.getsize(path)
    if size == 0:
        raise RuntimeError(f"[SaveVersion] {name} が 0 byte です: {path}\n"
                           f"- 直前の処理が失敗している可能性。logsを確認してください。")
    print(f"{name} OK : {path} ({size} bytes)")

_check(SUB_PATH, "submission")
_check(ERR_PATH, "error_samples")

print("Saved submission:", SUB_PATH)
print("Saved error samples:", ERR_PATH)



import pandas as pd, numpy as np, os

SUB_PATH = "/kaggle/working/submission_tfidf_lr.csv"
df = pd.read_csv(SUB_PATH)
print("shape:", df.shape)
print("cols :", df.columns.tolist())

TARGETS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
print(df[TARGETS].describe().T[["min","max","mean","std"]])

for c in TARGETS:
    print(c, "unique:", df[c].nunique(), "min:", df[c].min(), "max:", df[c].max())





