# Imports 
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
print('torch:', torch.__version__)
print('pandas:', pd.__version__)
print('numpy:', np.__version__)



# Paths (auto-adapts to Kaggle / local)
if os.path.exists('/kaggle/input'):
    DATA_DIR = '/kaggle/input/jigsaw-agile-community-rules'
    TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
    TEST_CSV  = os.path.join(DATA_DIR, 'test.csv')
    # Set this path after mounting EmbeddingGemma as a Kaggle Dataset
    MODEL_DIR = '/kaggle/input/embeddinggemma/transformers/embeddinggemma-300m/1'
    WORKDIR = '/kaggle/working'
else:
    PROJ_ROOT = Path.cwd().resolve()
    TRAIN_CSV = str(PROJ_ROOT / 'data' / 'train.csv')
    TEST_CSV  = str(PROJ_ROOT / 'data' / 'test.csv')
    # Locally, can point directly to an extracted directory or an HF model name (requires internet)
    MODEL_DIR = str(PROJ_ROOT / 'models' / 'embeddinggemma_local')
    WORKDIR = str(PROJ_ROOT)

SUB_PATH = os.path.join(WORKDIR, 'submission.csv')
print('Train path:', TRAIN_CSV)
print('Test path:', TEST_CSV)
print('Model dir:', MODEL_DIR)
print('Submission will be saved to:', SUB_PATH)


# Random seed 
def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
print('Seed set to 42')


# Load data and text
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
print('Train shape:', train_df.shape)
print('Test shape:', test_df.shape)

TARGET_COL = 'rule_violation'
y = train_df[TARGET_COL].astype(int).values

def sanitize_text(s: str) -> str:
    s = (s or '').strip()
    return s if s else '_'

train_rules = [sanitize_text(x) for x in train_df['rule'].astype(str).tolist()]
train_bodies = [sanitize_text(x) for x in train_df['body'].astype(str).tolist()]
test_rules  = [sanitize_text(x) for x in test_df['rule'].astype(str).tolist()]
test_bodies = [sanitize_text(x) for x in test_df['body'].astype(str).tolist()]

POS_COLS = ['positive_example_1', 'positive_example_2']
NEG_COLS = ['negative_example_1', 'negative_example_2']

def collect_examples(df, cols):
    values = df[cols].fillna('').astype(str).values.tolist()
    return [[sanitize_text(v) for v in row] for row in values]

train_pos_examples = collect_examples(train_df, POS_COLS)
train_neg_examples = collect_examples(train_df, NEG_COLS)
test_pos_examples = collect_examples(test_df, POS_COLS)
test_neg_examples = collect_examples(test_df, NEG_COLS)



# Load model and encode using sentence_transformers
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

# Local/Offline: For a completely offline setup, set environment variables and ensure MODEL_DIR is a local directory
# os.environ['HF_HUB_OFFLINE'] = '1'
# os.environ['TRANSFORMERS_OFFLINE'] = '1'

st_model = SentenceTransformer(MODEL_DIR, device=device)

# Probe native dimension
probe = st_model.encode(train_rules[:2] if len(train_rules) >= 2 else ['rule','rule'],
                         batch_size=2, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
native_dim = probe.shape[1]
DIM = min(768, native_dim)
print('Native dim =', native_dim, '; Using DIM =', DIM)

def st_encode(texts: list[str], batch_size: int = 128, dim: int | None = 512, normalize: bool = True) -> np.ndarray:
    embs = st_model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=True,
    )
    if dim is not None and dim > 0 and dim < embs.shape[1]:
        embs = embs[:, :dim]
    # Fallback cleaning
    embs = np.nan_to_num(embs, nan=0.0, posinf=0.0, neginf=0.0)
    return embs

BATCH = 128
print('Encoding train RULE...')
e_rule_tr = st_encode(train_rules, batch_size=BATCH, dim=DIM, normalize=True)
print('Encoding train BODY...')
e_body_tr = st_encode(train_bodies, batch_size=BATCH, dim=DIM, normalize=True)
print('Encoding test RULE...')
e_rule_te = st_encode(test_rules, batch_size=BATCH, dim=DIM, normalize=True)
print('Encoding test BODY...')
e_body_te = st_encode(test_bodies, batch_size=BATCH, dim=DIM, normalize=True)
print('Embeddings shapes:', e_rule_tr.shape, e_body_tr.shape, e_rule_te.shape, e_body_te.shape)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cos = np.sum(a * b, axis=1, keepdims=True)
    return np.clip(cos, -1.0, 1.0)


def build_pair_features(rule_emb: np.ndarray, body_emb: np.ndarray) -> np.ndarray:
    cosine = cosine_sim(rule_emb, body_emb)
    hadamard = rule_emb * body_emb
    diff = rule_emb - body_emb
    abs_diff = np.abs(diff)
    l1 = np.sum(abs_diff, axis=1, keepdims=True)
    l2 = np.sqrt(np.sum(diff ** 2, axis=1, keepdims=True))
    angle = np.arccos(np.clip(cosine, -1.0 + 1e-7, 1.0 - 1e-7))

    def stats_block(mat: np.ndarray):
        return [
            np.min(mat, axis=1, keepdims=True),
            np.mean(mat, axis=1, keepdims=True),
            np.max(mat, axis=1, keepdims=True),
            np.std(mat, axis=1, keepdims=True),
        ]

    feature_blocks = [
        rule_emb,
        body_emb,
        hadamard,
        abs_diff,
        cosine,
        l1,
        l2,
        angle,
        *stats_block(abs_diff),
        *stats_block(hadamard),
    ]
    feats = np.hstack(feature_blocks)
    return np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)


SENT_SPLIT_RE = re.compile(r"[。.!?！？]+|\n+")
MAX_SENTENCES = 12
TOPK_SENTENCES = 3


def split_sentences(text: str, max_sentences: int = MAX_SENTENCES) -> list[str]:
    parts = [seg.strip() for seg in SENT_SPLIT_RE.split(text) if seg.strip()]
    if not parts:
        fallback = text.strip() or '_'
        return [fallback]
    return parts[:max_sentences]


def build_sentence_features(rule_emb: np.ndarray, bodies: list[str]) -> np.ndarray:
    sentence_lists: list[list[str]] = []
    for text in bodies:
        sentence_lists.append(split_sentences(text))

    all_sentences: list[str] = [sent for sents in sentence_lists for sent in sents]
    if all_sentences:
        sent_emb = st_encode(all_sentences, batch_size=BATCH, dim=DIM, normalize=True)
    else:
        sent_emb = np.zeros((0, rule_emb.shape[1]), dtype=np.float32)

    feats = np.zeros((len(bodies), 7), dtype=float)
    cursor = 0
    for idx, sentences in enumerate(sentence_lists):
        count = len(sentences)
        if count == 0:
            slice_emb = np.zeros((0, rule_emb.shape[1]))
        else:
            slice_emb = sent_emb[cursor:cursor + count]
        cursor += count

        if count == 0:
            feats[idx] = 0.0
            continue

        cos_vals = slice_emb @ rule_emb[idx]
        max_cos = float(np.max(cos_vals))
        mean_cos = float(np.mean(cos_vals))
        std_cos = float(np.std(cos_vals))
        topk = min(TOPK_SENTENCES, count)
        topk_mean = float(np.mean(np.sort(cos_vals)[-topk:]))
        max_pos = int(np.argmax(cos_vals))
        pos_ratio = max_pos / (count - 1) if count > 1 else 0.0
        sent_len = float(len(sentences[max_pos]))

        feats[idx] = [
            max_cos,
            mean_cos,
            std_cos,
            topk_mean,
            pos_ratio,
            sent_len,
            float(count),
        ]

    return np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)


def build_example_alignment_features(
        rule_emb: np.ndarray,
        pos_lists: list[list[str]],
        neg_lists: list[list[str]],
) -> np.ndarray:
    def encode_blocks(blocks: list[list[str]]) -> np.ndarray:
        if not blocks:
            return np.zeros((len(rule_emb), 0, rule_emb.shape[1]), dtype=float)
        slots = len(blocks[0]) if blocks[0] else 0
        if slots == 0:
            return np.zeros((len(rule_emb), 0, rule_emb.shape[1]), dtype=float)
        flat = [text for row in blocks for text in row]
        emb = st_encode(flat, batch_size=BATCH, dim=DIM, normalize=True)
        return emb.reshape(len(blocks), slots, -1)

    pos_emb = encode_blocks(pos_lists)
    neg_emb = encode_blocks(neg_lists)

    if pos_emb.shape[1] > 0:
        pos_cos = np.einsum('ij,ikj->ik', rule_emb, pos_emb)
        pos_max = np.max(pos_cos, axis=1, keepdims=True)
        pos_mean = np.mean(pos_cos, axis=1, keepdims=True)
    else:
        pos_max = np.zeros((len(rule_emb), 1))
        pos_mean = np.zeros((len(rule_emb), 1))

    if neg_emb.shape[1] > 0:
        neg_cos = np.einsum('ij,ikj->ik', rule_emb, neg_emb)
        neg_max = np.max(neg_cos, axis=1, keepdims=True)
        neg_mean = np.mean(neg_cos, axis=1, keepdims=True)
    else:
        neg_max = np.zeros((len(rule_emb), 1))
        neg_mean = np.zeros((len(rule_emb), 1))

    gap_max = pos_max - neg_max
    gap_mean = pos_mean - neg_mean

    feats = np.hstack([pos_max, pos_mean, neg_max, neg_mean, gap_max, gap_mean])
    return np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)


pair_tr = build_pair_features(e_rule_tr, e_body_tr)
pair_te = build_pair_features(e_rule_te, e_body_te)

sentence_tr = build_sentence_features(e_rule_tr, train_bodies)
sentence_te = build_sentence_features(e_rule_te, test_bodies)

example_tr = build_example_alignment_features(e_rule_tr, train_pos_examples, train_neg_examples)
example_te = build_example_alignment_features(e_rule_te, test_pos_examples, test_neg_examples)

X_tr = np.nan_to_num(np.hstack([pair_tr, sentence_tr, example_tr]), nan=0.0, posinf=1.0, neginf=-1.0)
X_te = np.nan_to_num(np.hstack([pair_te, sentence_te, example_te]), nan=0.0, posinf=1.0, neginf=-1.0)
print('Feature shapes:', X_tr.shape, X_te.shape)



# Diagnostic print: confirm no feature collapse 
print('y shape:', y.shape, 'pos_rate:', y.mean())
print('e_rule_tr norm mean/std:', np.linalg.norm(e_rule_tr, axis=1).mean(), np.linalg.norm(e_rule_tr, axis=1).std())
print('e_body_tr norm mean/std:', np.linalg.norm(e_body_tr, axis=1).mean(), np.linalg.norm(e_body_tr, axis=1).std())
cos_tr = np.sum(e_rule_tr * e_body_tr, axis=1)
print('cosine train min/mean/max:', float(cos_tr.min()), float(cos_tr.mean()), float(cos_tr.max()))
print('X_tr var mean:', float(np.var(X_tr, axis=0).mean()))
print('Has any all-zero column:', bool(np.all(X_tr == 0, axis=0).any()))
print('Has any NaN:', bool(np.isnan(X_tr).any()), 'Has any Inf:', bool(np.isinf(X_tr).any()))


rf_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_oof = np.zeros(len(train_df), dtype=float)

for fold, (tr_idx, va_idx) in enumerate(rf_skf.split(X_tr, y), start=1):
    X_tr_f, y_tr_f = X_tr[tr_idx], y[tr_idx]
    X_va_f, y_va_f = X_tr[va_idx], y[va_idx]

    rf_clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=42 + fold,
    )
    rf_clf.fit(X_tr_f, y_tr_f)
    pred = rf_clf.predict_proba(X_va_f)[:, 1]
    rf_oof[va_idx] = pred
    auc = roc_auc_score(y_va_f, pred)
    print(f'[RF] Fold {fold}/5 AUC = {auc:.6f}')

rf_oof_auc = roc_auc_score(y, rf_oof)
print(f'[RF] OOF AUC = {rf_oof_auc:.6f}')



rf_final = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    class_weight='balanced_subsample',
    n_jobs=-1,
    random_state=42,
)
rf_final.fit(X_tr, y)
preds_rf = rf_final.predict_proba(X_te)[:, 1]

if 'row_id' not in test_df.columns:
    test_df['row_id'] = np.arange(len(test_df))
submission_rf = test_df[['row_id']].copy()
submission_rf['rule_violation'] = preds_rf
submission_rf.to_csv(Path(WORKDIR) / 'submission.csv', index=False)
print('Random Forest submission saved to', Path(WORKDIR) / 'submission.csv')


