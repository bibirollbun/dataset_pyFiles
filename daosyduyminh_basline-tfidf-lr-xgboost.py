# ============================================
# Mercor AI Text Detection – Strong Baseline (Kaggle-ready, no internet)
# Ensemble: [TFIDF-char LR] + [TFIDF-word LR] + [Style-Features XGBoost] -> Stacking LR
# Saves: /kaggle/working/submission.csv
# ============================================

import os, gc, re, string, math, time, warnings, random
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from scipy.sparse import hstack, csr_matrix

import xgboost as xgb

warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5
rng = np.random.RandomState(SEED)

DATA_DIR = "/kaggle/input/mercor-ai-detection"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH  = f"{DATA_DIR}/test.csv"

assert os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH), "Check dataset paths."

# --------------------------
# 1) Load data
# --------------------------
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

# Safety: enforce expected columns
expected_cols_train = {"id", "topic", "answer", "is_cheating"}
expected_cols_test  = {"id", "topic", "answer"}
assert expected_cols_train.issubset(train.columns), f"train.csv missing columns: {expected_cols_train - set(train.columns)}"
assert expected_cols_test.issubset(test.columns),   f"test.csv missing columns: {expected_cols_test - set(test.columns)}"

train["topic"]  = train["topic"].fillna("")
train["answer"] = train["answer"].fillna("")
test["topic"]   = test["topic"].fillna("")
test["answer"]  = test["answer"].fillna("")

y = train["is_cheating"].astype(int).values

# --------------------------
# 2) Simple cleaning
# (giữ khá nguyên bản để tránh làm mất tín hiệu)
# --------------------------
def basic_norm(s: str) -> str:
    # Lower + strip weird whitespace; không thay đổi nặng tay để giữ phong cách viết
    s = s.replace("\u00A0", " ").replace("\t"," ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

for col in ["topic", "answer"]:
    train[col] = train[col].astype(str).map(basic_norm)
    test[col]  = test[col].astype(str).map(basic_norm)

# --------------------------
# 3) Style / statistical features (no external models)
# --------------------------
STOPWORDS = {
    # Gọn nhẹ đủ dùng (không cần NLTK)
    "a","an","the","and","or","but","if","while","for","to","of","in","on","at","by","with","from","as",
    "is","am","are","was","were","be","been","being","it","this","that","these","those","i","you","he","she","we","they",
    "me","him","her","us","them","my","your","his","her","our","their",
    "will","would","can","could","should","may","might","do","does","did","done","doing",
    "not","no","so","than","then","there","here","when","where","why","how",
    "however","therefore","moreover","furthermore","additionally","overall","hence","thus"
}

def count_syllables_en(word: str) -> int:
    # Ước lượng syllable rất đơn giản, đủ cho readability proxy
    w = word.lower()
    w = re.sub(r'[^a-z]', '', w)
    if not w:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    for ch in w:
        is_vowel = ch in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    if w.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)

def shannon_entropy_char(s: str) -> float:
    if not s:
        return 0.0
    # entropy trên phân phối ký tự
    counts = np.array([s.count(c) for c in set(s)], dtype=float)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))

def repeat_ngram_ratio(words, n=3):
    if len(words) < n:
        return 0.0
    ngrams = []
    for i in range(len(words)-n+1):
        ngrams.append(tuple(words[i:i+n]))
    if not ngrams:
        return 0.0
    uniq = set(ngrams)
    return 1.0 - len(uniq)/len(ngrams)

def style_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = {}
    texts = df["answer"].tolist()
    topics = df["topic"].tolist()

    wc = []
    avg_wlen = []
    ttr = []  # type-token ratio
    stop_ratio = []
    punct_ratio = []
    upper_ratio = []
    digit_ratio = []
    sent_cnt = []
    comma_per_100w = []
    quote_ratio = []
    paren_ratio = []
    colon_semicolon_ratio = []
    entropy_char = []
    trigram_repeat = []
    flesch_kincaid_like = []

    rhetorical_markers = ["however","therefore","moreover","furthermore","additionally","thus","hence","in conclusion"]
    rhet_count = []

    for s in texts:
        # tokens basic
        tokens = re.findall(r"\b\w+\b", s.lower())
        words = [t for t in tokens if t.isalpha()]
        n_words = len(words)
        wc.append(n_words)

        if n_words > 0:
            avg_wlen.append(np.mean([len(w) for w in words]))
            uniq_words = len(set(words))
            ttr.append(uniq_words / n_words)
            stop_ratio.append(sum(w in STOPWORDS for w in words)/n_words)
            comma_per_100w.append(s.count(",")/(n_words/100.0 + 1e-6))
            quote_ratio.append((s.count('"') + s.count("'"))/(len(s)+1e-6))
            paren_ratio.append((s.count("(")+s.count(")"))/(len(s)+1e-6))
            colon_semicolon_ratio.append((s.count(":")+s.count(";"))/(len(s)+1e-6))
        else:
            avg_wlen.append(0.0); ttr.append(0.0); stop_ratio.append(0.0)
            comma_per_100w.append(0.0); quote_ratio.append(0.0); paren_ratio.append(0.0); colon_semicolon_ratio.append(0.0)

        punct = sum(ch in string.punctuation for ch in s)
        punct_ratio.append(punct/(len(s)+1e-6))
        upper = sum(1 for ch in s if ch.isupper())
        upper_ratio.append(upper/(len(s)+1e-6))
        digits = sum(1 for ch in s if ch.isdigit())
        digit_ratio.append(digits/(len(s)+1e-6))

        # sentence approximation
        sentences = re.split(r"[.!?]+", s)
        sentences = [x.strip() for x in sentences if x.strip()]
        sent_cnt.append(len(sentences))

        # char entropy
        entropy_char.append(shannon_entropy_char(s))

        # repetition ratio (trigram)
        trigram_repeat.append(repeat_ngram_ratio(words, n=3))

        # readability proxy (Flesch-Kincaid-like)
        if n_words > 0 and len(sentences) > 0:
            syllables = sum(count_syllables_en(w) for w in words)
            # FK Reading Ease (approx): 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
            fk = 206.835 - 1.015*(n_words/(len(sentences)+1e-6)) - 84.6*(syllables/(n_words+1e-6))
            flesch_kincaid_like.append(fk)
        else:
            flesch_kincaid_like.append(0.0)

        # rhetorical markers count per 100 words
        rc = sum(s.lower().count(m) for m in rhetorical_markers)
        rhet_count.append(rc/(n_words/100.0 + 1e-6))

    feats["ans_word_count"] = wc
    feats["ans_avg_word_len"] = avg_wlen
    feats["ans_type_token_ratio"] = ttr
    feats["ans_stopword_ratio"] = stop_ratio
    feats["ans_punct_ratio"] = punct_ratio
    feats["ans_upper_ratio"] = upper_ratio
    feats["ans_digit_ratio"] = digit_ratio
    feats["ans_sentence_count"] = sent_cnt
    feats["ans_comma_per_100w"] = comma_per_100w
    feats["ans_quote_ratio"] = quote_ratio
    feats["ans_paren_ratio"] = paren_ratio
    feats["ans_colon_semicolon_ratio"] = colon_semicolon_ratio
    feats["ans_char_entropy"] = entropy_char
    feats["ans_trigram_repeat_ratio"] = trigram_repeat
    feats["ans_fk_like"] = flesch_kincaid_like
    feats["ans_rhetorical_markers_per_100w"] = rhet_count

    # Topic simple stats
    tlen = [len(t) for t in topics]
    t_wc = [len(re.findall(r"\b\w+\b", t)) for t in topics]
    feats["topic_len"] = tlen
    feats["topic_word_count"] = t_wc

    return pd.DataFrame(feats)

train_feats = style_features(train)
test_feats  = style_features(test)

# Standardize numeric features for tree+linear blender fairness
scaler = StandardScaler()
train_feats_scaled = pd.DataFrame(scaler.fit_transform(train_feats), columns=train_feats.columns, index=train.index)
test_feats_scaled  = pd.DataFrame(scaler.transform(test_feats), columns=test_feats.columns, index=test.index)

# --------------------------
# 4) Topic Target Encoding (leak-free via CV)
# --------------------------
def target_encode(train_series, y, test_series, n_splits=5, seed=SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_te = np.zeros(train_series.shape[0], dtype=float)
    global_mean = y.mean()

    for tr_idx, val_idx in skf.split(np.zeros_like(y), y):
        tr_topics = train_series.iloc[tr_idx]
        tr_y = y[tr_idx]
        # mean by topic
        means = pd.DataFrame({"topic": tr_topics.values, "y": tr_y}).groupby("topic")["y"].mean()
        oof_te[val_idx] = train_series.iloc[val_idx].map(means).fillna(global_mean).values

    # test encoding using full train
    full_means = pd.DataFrame({"topic": train_series.values, "y": y}).groupby("topic")["y"].mean()
    test_te = test_series.map(full_means).fillna(global_mean).values

    return oof_te.reshape(-1,1), test_te.reshape(-1,1)

train_topic_te, test_topic_te = target_encode(train["topic"], y, test["topic"], n_splits=N_FOLDS, seed=SEED)

# --------------------------
# 5) TF-IDF features (char + word)
# (fit on FULL train text -> unsupervised, acceptable; giúp tiết kiệm thời gian)
# --------------------------
tfidf_char = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3,6),
    min_df=2,
    max_features=300_000,
    sublinear_tf=True,
    lowercase=True,
)
tfidf_word = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1,2),
    min_df=2,
    max_features=200_000,
    token_pattern=r"(?u)\b\w+\b",
    sublinear_tf=True,
    lowercase=True,
)

print("Fitting TF-IDF (char)...")
Xc_train = tfidf_char.fit_transform(train["answer"])
Xc_test  = tfidf_char.transform(test["answer"])
print("Char TF-IDF shapes:", Xc_train.shape, Xc_test.shape)

print("Fitting TF-IDF (word)...")
Xw_train = tfidf_word.fit_transform(train["answer"])
Xw_test  = tfidf_word.transform(test["answer"])
print("Word TF-IDF shapes:", Xw_train.shape, Xw_test.shape)

# Topic TF-IDF (very light)
tfidf_topic = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=1, max_features=50_000, sublinear_tf=True, lowercase=True)
Xt_train = tfidf_topic.fit_transform(train["topic"])
Xt_test  = tfidf_topic.transform(test["topic"])

# --------------------------
# 6) Models (base learners)
# --------------------------
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

seed_everything(SEED)

# Logistic Regression configs for sparse high-dim TF-IDF
LR_C_CHAR = 4.0
LR_C_WORD = 3.0

# XGBoost for numeric style features (+ topic TE)
xgb_params = dict(
    n_estimators=800,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    reg_alpha=0.0,
    min_child_weight=1.0,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",  # fast CPU on Kaggle
    random_state=SEED,
)

# --------------------------
# 7) K-Fold OOF training for base models + blender
# --------------------------
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_char = np.zeros(len(train), dtype=float)
oof_word = np.zeros(len(train), dtype=float)
oof_xgb  = np.zeros(len(train), dtype=float)

test_char = np.zeros(len(test), dtype=float)
test_word = np.zeros(len(test), dtype=float)
test_xgb  = np.zeros(len(test), dtype=float)

print("\n==== K-FOLD TRAINING BASE MODELS ====\n")
for fold, (tr_idx, va_idx) in enumerate(skf.split(Xc_train, y), 1):
    print(f"[Fold {fold}/{N_FOLDS}]")
    Xc_tr, Xc_va = Xc_train[tr_idx], Xc_train[va_idx]
    Xw_tr, Xw_va = Xw_train[tr_idx], Xw_train[va_idx]
    Xt_tr, Xt_va = Xt_train[tr_idx], Xt_train[va_idx]

    y_tr, y_va = y[tr_idx], y[va_idx]

    # ---- Base 1: LR on char TFIDF + topic TFIDF
    X_char_tr = hstack([Xc_tr, Xt_tr]).tocsr()
    X_char_va = hstack([Xc_va, Xt_va]).tocsr()

    lr_char = LogisticRegression(
        C=LR_C_CHAR, solver="saga", penalty="l2", max_iter=4000, random_state=SEED
    )
    lr_char.fit(X_char_tr, y_tr)
    oof_char[va_idx] = lr_char.predict_proba(X_char_va)[:,1]

    # ---- Base 2: LR on word TFIDF + topic TFIDF
    X_word_tr = hstack([Xw_tr, Xt_tr]).tocsr()
    X_word_va = hstack([Xw_va, Xt_va]).tocsr()

    lr_word = LogisticRegression(
        C=LR_C_WORD, solver="saga", penalty="l2", max_iter=4000, random_state=SEED
    )
    lr_word.fit(X_word_tr, y_tr)
    oof_word[va_idx] = lr_word.predict_proba(X_word_va)[:,1]

    # ---- Base 3: XGB on numeric style features + topic TE
    F_tr = np.hstack([train_feats_scaled.iloc[tr_idx].values, train_topic_te[tr_idx]])
    F_va = np.hstack([train_feats_scaled.iloc[va_idx].values, train_topic_te[va_idx]])

    dtr = xgb.DMatrix(F_tr, label=y_tr)
    dva = xgb.DMatrix(F_va, label=y_va)

    # early stopping for robustness
    xgb_model = xgb.train(
        params={**xgb_params},
        dtrain=dtr,
        num_boost_round=4000,
        evals=[(dtr,"train"), (dva,"valid")],
        early_stopping_rounds=100,
        verbose_eval=False
    )
    oof_xgb[va_idx] = xgb_model.predict(dva)

    # Predict test fold-wise (averaging later)
    X_char_test = hstack([Xc_test, Xt_test]).tocsr()
    X_word_test = hstack([Xw_test, Xt_test]).tocsr()
    Ft = np.hstack([test_feats_scaled.values, test_topic_te])

    test_char += lr_char.predict_proba(X_char_test)[:,1] / N_FOLDS
    test_word += lr_word.predict_proba(X_word_test)[:,1] / N_FOLDS

    dte = xgb.DMatrix(Ft)
    test_xgb += xgb_model.predict(dte) / N_FOLDS

    # fold AUCs
    auc_c = roc_auc_score(y_va, oof_char[va_idx])
    auc_w = roc_auc_score(y_va, oof_word[va_idx])
    auc_x = roc_auc_score(y_va, oof_xgb[va_idx])
    print(f"  AUC char={auc_c:.4f} | word={auc_w:.4f} | xgb={auc_x:.4f}")

# Overall OOF AUCs for base learners
auc_char = roc_auc_score(y, oof_char)
auc_word = roc_auc_score(y, oof_word)
auc_xgb  = roc_auc_score(y, oof_xgb)
print("\nBase OOF AUCs:")
print(f"  char-LR: {auc_char:.5f}")
print(f"  word-LR: {auc_word:.5f}")
print(f"  xgb-style: {auc_xgb:.5f}")

# --------------------------
# 8) Blender (stacking) using OOF preds + numeric feats
# --------------------------
# Meta-train features = [oof_char, oof_word, oof_xgb] + scaled numeric feats
meta_train = np.hstack([
    oof_char.reshape(-1,1),
    oof_word.reshape(-1,1),
    oof_xgb.reshape(-1,1),
    train_feats_scaled.values,
    train_topic_te
])
meta_test = np.hstack([
    test_char.reshape(-1,1),
    test_word.reshape(-1,1),
    test_xgb.reshape(-1,1),
    test_feats_scaled.values,
    test_topic_te
])

blender = LogisticRegression(C=2.0, solver="lbfgs", max_iter=2000, random_state=SEED)
blender.fit(meta_train, y)

oof_blend = blender.predict_proba(meta_train)[:,1]
auc_blend = roc_auc_score(y, oof_blend)
print(f"\n==== Stacking (Blender) OOF AUC: {auc_blend:.5f} ====\n")

test_pred = blender.predict_proba(meta_test)[:,1]
test_pred = np.clip(test_pred, 1e-6, 1-1e-6)

# --------------------------
# 9) Save submission
# --------------------------
sub = pd.DataFrame({"id": test["id"], "is_cheating": test_pred})
save_path = "/kaggle/working/submission.csv"
sub.to_csv(save_path, index=False)
print(f"Saved: {save_path}")
print(sub.head())


