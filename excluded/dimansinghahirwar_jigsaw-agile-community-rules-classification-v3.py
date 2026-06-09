import os, gc, random
import numpy as np, pandas as pd
from pathlib import Path
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin, clone

RANDOM_SEED = 42
N_SPLITS = 5
VERBOSE = 1

def seed_everything(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
seed_everything(RANDOM_SEED)

INPUT_DIR = Path("../input/jigsaw-agile-community-rules")
TRAIN_PATH = INPUT_DIR / "train.csv"
TEST_PATH = INPUT_DIR / "test.csv"
SAMPLE_SUB_PATH = INPUT_DIR / "sample_submission.csv"
assert TRAIN_PATH.exists() and TEST_PATH.exists() and SAMPLE_SUB_PATH.exists(), "Dataset files missing"
print("âœ… Paths OK")



train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
print(train.shape, test.shape)
train.head()



import re
URL_RE = re.compile(r'https?://\S+|www\.\S+')

def basic_normalize(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(URL_RE, " URL ", s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def token_set(s: str):
    return set(basic_normalize(s).split())

def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter/union if union else 0.0

class RuleAwareBuilder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scalar_names_ = [
            'len_body','len_rule_ctx','url_ct_body','caps_ratio',
            'jaccard_rule','jaccard_pos1','jaccard_pos2','jaccard_neg1','jaccard_neg2'
        ]
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        bodies = X['body'].astype(str).fillna("")
        rule_ctxs = (
            "RULE: " + X['rule'].astype(str).fillna("") + " ||| " +
            "SUB: " + X['subreddit'].astype(str).fillna("") + " ||| " +
            "POS1: " + X['positive_example_1'].astype(str).fillna("") + " ||| " +
            "POS2: " + X['positive_example_2'].astype(str).fillna("") + " ||| " +
            "NEG1: " + X['negative_example_1'].astype(str).fillna("") + " ||| " +
            "NEG2: " + X['negative_example_2'].astype(str).fillna("")
        )
        def url_count(s): return len(re.findall(URL_RE, s))
        def caps_ratio(s):
            if not s: return 0.0
            caps = sum(1 for ch in s if ch.isupper())
            letters = sum(1 for ch in s if ch.isalpha())
            return caps / letters if letters else 0.0
        body_norm = bodies.apply(basic_normalize)
        rule_norm = pd.Series(rule_ctxs).apply(basic_normalize)
        body_tok = body_norm.apply(token_set)
        rule_tok = rule_norm.apply(token_set)
        pos1_tok = X['positive_example_1'].astype(str).fillna("").apply(basic_normalize).apply(token_set)
        pos2_tok = X['positive_example_2'].astype(str).fillna("").apply(basic_normalize).apply(token_set)
        neg1_tok = X['negative_example_1'].astype(str).fillna("").apply(basic_normalize).apply(token_set)
        neg2_tok = X['negative_example_2'].astype(str).fillna("").apply(basic_normalize).apply(token_set)
        j_rule = np.array([jaccard(a,b) for a,b in zip(body_tok, rule_tok)])
        j_p1   = np.array([jaccard(a,b) for a,b in zip(body_tok, pos1_tok)])
        j_p2   = np.array([jaccard(a,b) for a,b in zip(body_tok, pos2_tok)])
        j_n1   = np.array([jaccard(a,b) for a,b in zip(body_tok, neg1_tok)])
        j_n2   = np.array([jaccard(a,b) for a,b in zip(body_tok, neg2_tok)])
        df = pd.DataFrame({
            'body': bodies,
            'rule_context': rule_ctxs,
            'subreddit': X['subreddit'].astype(str).fillna("")
        })
        df['len_body'] = bodies.str.len().values
        df['len_rule_ctx'] = pd.Series(rule_ctxs).str.len().values
        df['url_ct_body'] = bodies.apply(url_count).values
        df['caps_ratio'] = bodies.apply(caps_ratio).values
        df['jaccard_rule'] = j_rule
        df['jaccard_pos1'] = j_p1
        df['jaccard_pos2'] = j_p2
        df['jaccard_neg1'] = j_n1
        df['jaccard_neg2'] = j_n2
        return df

class BodyOnlyBuilder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        def url_count(s): return len(re.findall(URL_RE, s))
        def caps_ratio(s):
            if not s: return 0.0
            caps = sum(1 for ch in s if ch.isupper())
            letters = sum(1 for ch in s if ch.isalpha())
            return caps / letters if letters else 0.0
        bodies = X['body'].astype(str).fillna("")
        df = pd.DataFrame({'body': bodies, 'subreddit': X['subreddit'].astype(str).fillna("")})
        df['len_body'] = bodies.str.len().values
        df['url_ct_body'] = bodies.apply(url_count).values
        df['caps_ratio'] = bodies.apply(caps_ratio).values
        return df



body_word_ra = TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=2,
                               max_features=250_000, strip_accents='unicode', sublinear_tf=True)
body_char_ra = TfidfVectorizer(analyzer='char', ngram_range=(2,6), min_df=3,
                               max_features=300_000, sublinear_tf=True)
rule_word_ra = TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=2,
                               max_features=200_000, strip_accents='unicode', sublinear_tf=True)
sub_ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)

text_features_ra = ColumnTransformer(
    transformers=[
        ('body_word', body_word_ra, 'body'),
        ('body_char', body_char_ra, 'body'),
        ('rule_word', rule_word_ra, 'rule_context'),
        ('subreddit', sub_ohe, ['subreddit']),
        ('scalars', 'passthrough', [
            'len_body','len_rule_ctx','url_ct_body','caps_ratio',
            'jaccard_rule','jaccard_pos1','jaccard_pos2','jaccard_neg1','jaccard_neg2'
        ]),
    ],
    remainder='drop',
    verbose_feature_names_out=False,
)

body_word_bo = TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=2,
                               max_features=250_000, strip_accents='unicode', sublinear_tf=True)
body_char_bo = TfidfVectorizer(analyzer='char', ngram_range=(2,6), min_df=3,
                               max_features=300_000, sublinear_tf=True)

text_features_bo = ColumnTransformer(
    transformers=[
        ('body_word', body_word_bo, 'body'),
        ('body_char', body_char_bo, 'body'),
        ('subreddit', sub_ohe, ['subreddit']),
        ('scalars', 'passthrough', ['len_body','url_ct_body','caps_ratio']),
    ],
    remainder='drop',
    verbose_feature_names_out=False,
)



@dataclass
class ModelSpec:
    name: str
    pipeline: Pipeline

def make_specs():
    specs = []
    # RA: LogReg
    logreg_ra = LogisticRegression(max_iter=12000, solver='saga', C=2.0, n_jobs=-1, class_weight='balanced')
    specs.append(ModelSpec('RA_logreg', Pipeline([
        ('builder', RuleAwareBuilder()),
        ('feats', text_features_ra),
        ('clf', logreg_ra)
    ])))
    # RA: Calibrated LinearSVC
    linsvc_ra = CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced', max_iter=25000), method='sigmoid', cv=3)
    specs.append(ModelSpec('RA_linsvc_cal', Pipeline([
        ('builder', RuleAwareBuilder()),
        ('feats', text_features_ra),
        ('clf', linsvc_ra)
    ])))
    # BO: LogReg
    logreg_bo = LogisticRegression(max_iter=12000, solver='saga', C=2.0, n_jobs=-1, class_weight='balanced')
    specs.append(ModelSpec('BO_logreg', Pipeline([
        ('builder', BodyOnlyBuilder()),
        ('feats', text_features_bo),
        ('clf', logreg_bo)
    ])))
    # BO: Calibrated LinearSVC
    linsvc_bo = CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced', max_iter=25000), method='sigmoid', cv=3)
    specs.append(ModelSpec('BO_linsvc_cal', Pipeline([
        ('builder', BodyOnlyBuilder()),
        ('feats', text_features_bo),
        ('clf', linsvc_bo)
    ])))
    return specs

SPECS = make_specs()
[s.name for s in SPECS]



def per_rule_auc(y_true, y_pred, rules):
    scores = []
    df = pd.DataFrame({'y': y_true, 'p': y_pred, 'rule': rules})
    for r, g in df.groupby('rule'):
        if g['y'].nunique() == 2:
            scores.append(roc_auc_score(g['y'], g['p']))
    return (np.mean(scores) if scores else np.nan)

def kfold_oof(specs, train_df, n_splits=5, seed=42):
    X = train_df.copy()
    y = train_df['rule_violation'].values
    rules = train_df['rule'].values
    stratify_key = train_df['rule'].astype('category').cat.codes * 2 + train_df['rule_violation']
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = {s.name: np.zeros(len(train_df)) for s in specs}
    folds = {s.name: [] for s in specs}
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, stratify_key)):
        print(f"\n===== Fold {fold+1}/{n_splits} =====")
        X_tr, X_va = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_va = y[trn_idx], y[val_idx]
        for s in specs:
            pipe = clone(s.pipeline)
            pipe.fit(X_tr, y_tr)
            if hasattr(pipe.named_steps['clf'], 'predict_proba'):
                p = pipe.predict_proba(X_va)[:,1]
            else:
                dfc = pipe.decision_function(X_va)
                p = 1/(1+np.exp(-dfc))
            oof[s.name][val_idx] = p
            folds[s.name].append(pipe)
            auc_global = roc_auc_score(y_va, p) if len(np.unique(y_va))==2 else np.nan
            auc_rules = per_rule_auc(y_va, p, X_va['rule'].values)
            print(f"{s.name}: AUC={auc_global:.4f} | per-rule AUC={auc_rules:.4f}")
    print("\n===== OOF Results =====")
    for s in specs:
        auc_g = roc_auc_score(y, oof[s.name])
        auc_r = per_rule_auc(y, oof[s.name], rules)
        print(f"{s.name:<15} AUC={auc_g:.5f} | per-rule AUC={auc_r:.5f}")
    return oof, folds

oof, fold_models = kfold_oof(SPECS, train, n_splits=N_SPLITS, seed=RANDOM_SEED)



from itertools import product

oof_names = list(oof.keys())
oof_mat = np.column_stack([oof[k] for k in oof_names])
y = train['rule_violation'].values
rules = train['rule'].values

def per_rule_and_global(ws):
    ws = np.array(ws, dtype=float)
    ws = ws / (ws.sum() + 1e-12)
    pred = (oof_mat * ws).sum(1)
    return per_rule_auc(y, pred, rules), roc_auc_score(y, pred)

grid = np.arange(0, 1.01, 0.1)
best_auc_r, best_auc_g, best_ws = -1, -1, None
for ws in product(grid, repeat=oof_mat.shape[1]):
    if sum(ws) == 0: continue
    auc_r, auc_g = per_rule_and_global(ws)
    if auc_r > best_auc_r or (abs(auc_r - best_auc_r) < 1e-6 and auc_g > best_auc_g):
        best_auc_r, best_auc_g, best_ws = auc_r, auc_g, ws

best_ws = np.array(best_ws, dtype=float)
best_ws = best_ws / (best_ws.sum() + 1e-12)
print("Best weights (opt. per-rule AUC):")
for name, w in zip(oof_names, best_ws):
    print(f"  {name:<15} -> {w:.3f}")
print(f"Per-rule AUC={best_auc_r:.5f} | Global AUC={best_auc_g:.5f}")



final_models = []
for s in SPECS:
    pipe = clone(s.pipeline)
    pipe.fit(train, train['rule_violation'].values)
    final_models.append((s.name, pipe))
    gc.collect()

test_preds = []
for name, model in final_models:
    if hasattr(model.named_steps['clf'], 'predict_proba'):
        p = model.predict_proba(test)[:,1]
    else:
        dfc = model.decision_function(test)
        p = 1/(1+np.exp(-dfc))
    test_preds.append(p)

test_mat = np.column_stack(test_preds)
final_pred = (test_mat * best_ws).sum(1)

sub = sample_sub.copy(); sub['rule_violation'] = final_pred
sub.to_csv('submission.csv', index=False)
print('Saved submission.csv')
sub.head()


