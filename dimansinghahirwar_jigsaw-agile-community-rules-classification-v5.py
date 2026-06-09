import os, gc, random, warnings
import numpy as np, pandas as pd
from pathlib import Path
from itertools import product

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import normalize as spnorm
from scipy import sparse

warnings.filterwarnings('ignore')
RANDOM_SEED = 42
N_SPLITS = 5
ALPHA_PRIOR = 2.0

def seed_everything(seed=RANDOM_SEED):
    random.seed(seed); np.random.seed(seed)
seed_everything()

INPUT_DIR = Path("../input/jigsaw-agile-community-rules")
TRAIN_PATH = INPUT_DIR / "train.csv"
TEST_PATH = INPUT_DIR / "test.csv"
SAMPLE_SUB_PATH = INPUT_DIR / "sample_submission.csv"
assert TRAIN_PATH.exists() and TEST_PATH.exists() and SAMPLE_SUB_PATH.exists(), 'Dataset files missing'
print('✅ Paths OK')



train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
print(train.shape, test.shape)



import re
URL_RE = re.compile(r'https?://\S+|www\.\S+')
def normalize(s: str) -> str:
    if not isinstance(s, str): return ''
    s = s.lower(); s = re.sub(URL_RE, ' URL ', s); s = re.sub(r'\d+',' 0 ', s)
    s = re.sub(r'\s+',' ', s).strip(); return s

def sparse_cos(a, b):
    a = spnorm(a, norm='l2', copy=False); b = spnorm(b, norm='l2', copy=False)
    return (a.multiply(b)).sum(axis=1).A.ravel()



def build_proto_vectors(df):
    C = df.copy()
    for col in ['body','rule','positive_example_1','positive_example_2','negative_example_1','negative_example_2']:
        C[col] = C[col].astype(str).fillna('').apply(normalize)
    word_vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=250_000, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer='char', ngram_range=(2,6), min_df=3, max_features=350_000, sublinear_tf=True)
    all_text = pd.concat([C['body'], C['rule'], C['positive_example_1'], C['positive_example_2'], C['negative_example_1'], C['negative_example_2']], axis=0)
    word_vec.fit(all_text); char_vec.fit(all_text)
    return word_vec, char_vec

def build_proto_feats_with_vecs(df, wv, cv, sub_prior_map=None, sub_prior_fallback=None):
    C = df.copy()
    for col in ['body','rule','positive_example_1','positive_example_2','negative_example_1','negative_example_2']:
        C[col] = C[col].astype(str).fillna('').apply(normalize)
    Bw = wv.transform(C['body']); Br = wv.transform(C['rule'])
    P1w = wv.transform(C['positive_example_1']); P2w = wv.transform(C['positive_example_2'])
    N1w = wv.transform(C['negative_example_1']); N2w = wv.transform(C['negative_example_2'])
    Bc = cv.transform(C['body']); Rc = cv.transform(C['rule'])
    P1c = cv.transform(C['positive_example_1']); P2c = cv.transform(C['positive_example_2'])
    N1c = cv.transform(C['negative_example_1']); N2c = cv.transform(C['negative_example_2'])
    Pavgw = (P1w + P2w) * 0.5; Navgw = (N1w + N2w) * 0.5
    Pavgc = (P1c + P2c) * 0.5; Navgc = (N1c + N2c) * 0.5
    feats = pd.DataFrame({
        'w_body_rule': sparse_cos(Bw, Br),
        'w_body_p1': sparse_cos(Bw, P1w), 'w_body_p2': sparse_cos(Bw, P2w), 'w_body_pavg': sparse_cos(Bw, Pavgw),
        'w_body_n1': sparse_cos(Bw, N1w), 'w_body_n2': sparse_cos(Bw, N2w), 'w_body_navg': sparse_cos(Bw, Navgw),
        'c_body_rule': sparse_cos(Bc, Rc),
        'c_body_p1': sparse_cos(Bc, P1c), 'c_body_p2': sparse_cos(Bc, P2c), 'c_body_pavg': sparse_cos(Bc, Pavgc),
        'c_body_n1': sparse_cos(Bc, N1c), 'c_body_n2': sparse_cos(Bc, N2c), 'c_body_navg': sparse_cos(Bc, Navgc),
    })
    feats['w_pos_minus_neg'] = feats['w_body_pavg'] - feats['w_body_navg']
    feats['c_pos_minus_neg'] = feats['c_body_pavg'] - feats['c_body_navg']
    # ALWAYS include a sub_prior column
    if sub_prior_map is None:
        prior_vals = np.full(len(df), float(sub_prior_fallback) if sub_prior_fallback is not None else 0.5)
    else:
        prior_vals = df['subreddit'].astype(str).map(sub_prior_map).fillna(float(sub_prior_fallback) if sub_prior_fallback is not None else 0.5).values
    feats['sub_prior'] = prior_vals
    # simple lex
    feats['len_body'] = df['body'].astype(str).str.len().values
    feats['url_ct'] = df['body'].astype(str).apply(lambda s: len(re.findall(URL_RE, s))).values
    return feats



sub_ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)
body_word = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=300_000, sublinear_tf=True, strip_accents='unicode')
body_char = TfidfVectorizer(analyzer='char', ngram_range=(2,6), min_df=3, max_features=400_000, sublinear_tf=True)
rule_word = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=220_000, sublinear_tf=True, strip_accents='unicode')

class RuleAwareBuilder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        df = pd.DataFrame()
        df['body'] = X['body'].astype(str).fillna('')
        df['rule_context'] = (
            'RULE: ' + X['rule'].astype(str).fillna('') + ' ||| ' +
            'POS1: ' + X['positive_example_1'].astype(str).fillna('') + ' ||| ' +
            'POS2: ' + X['positive_example_2'].astype(str).fillna('') + ' ||| ' +
            'NEG1: ' + X['negative_example_1'].astype(str).fillna('') + ' ||| ' +
            'NEG2: ' + X['negative_example_2'].astype(str).fillna('')
        )
        df['subreddit'] = X['subreddit'].astype(str).fillna('')
        return df

class BodyOnlyBuilder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return pd.DataFrame({'body': X['body'].astype(str).fillna(''), 'subreddit': X['subreddit'].astype(str).fillna('')})

text_ra = ColumnTransformer([
    ('bw', body_word, 'body'), ('bc', body_char, 'body'), ('rw', rule_word, 'rule_context'), ('sub', sub_ohe, ['subreddit'])
], remainder='drop', verbose_feature_names_out=False)
text_bo = ColumnTransformer([
    ('bw', body_word, 'body'), ('bc', body_char, 'body'), ('sub', sub_ohe, ['subreddit'])
], remainder='drop', verbose_feature_names_out=False)
text_char = ColumnTransformer([
    ('bc', body_char, 'body')
], remainder='drop', verbose_feature_names_out=False)

svc_ra = CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced', max_iter=30000), method='sigmoid', cv=3)
svc_bo = CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced', max_iter=30000), method='sigmoid', cv=3)
svc_ch = CalibratedClassifierCV(LinearSVC(C=0.5, class_weight='balanced', max_iter=30000), method='sigmoid', cv=3)

pipe_ra = Pipeline([('build', RuleAwareBuilder()), ('feats', text_ra), ('clf', svc_ra)])
pipe_bo = Pipeline([('build', BodyOnlyBuilder()), ('feats', text_bo), ('clf', svc_bo)])
pipe_ch = Pipeline([('build', BodyOnlyBuilder()), ('feats', text_char), ('clf', svc_ch)])



def per_rule_auc(y_true, y_pred, rules):
    scores = []
    df = pd.DataFrame({'y': y_true, 'p': y_pred, 'rule': rules})
    for _, g in df.groupby('rule'):
        if g['y'].nunique() == 2:
            scores.append(roc_auc_score(g['y'], g['p']))
    return np.mean(scores) if scores else np.nan

wv_full, cv_full = build_proto_vectors(train)

oof_ra = np.zeros(len(train)); oof_bo = np.zeros(len(train)); oof_ch = np.zeros(len(train)); oof_pr = np.zeros(len(train))

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
strat = train['rule'].astype('category').cat.codes * 2 + train['rule_violation']
for fold, (tr, va) in enumerate(skf.split(train, strat)):
    print(f"\n===== Fold {fold+1}/{N_SPLITS} =====")
    tr_df, va_df = train.iloc[tr], train.iloc[va]
    ytr, yva = tr_df['rule_violation'].values, va_df['rule_violation'].values
    # Laplace-smoothed subreddit prior from tr_df only
    grp = tr_df.groupby('subreddit')['rule_violation'].agg(['sum','count'])
    mu = ytr.mean()
    prior_map = (grp['sum'] + ALPHA_PRIOR*mu) / (grp['count'] + ALPHA_PRIOR)
    # Prototype with the same columns for tr/va
    proto_tr = build_proto_feats_with_vecs(tr_df, wv_full, cv_full, sub_prior_map=prior_map, sub_prior_fallback=mu)
    proto_va = build_proto_feats_with_vecs(va_df, wv_full, cv_full, sub_prior_map=prior_map, sub_prior_fallback=mu)
    prc = HistGradientBoostingClassifier(learning_rate=0.08, max_depth=None, max_iter=500, max_leaf_nodes=31, random_state=RANDOM_SEED)
    prc.fit(proto_tr, ytr)
    pp = prc.predict_proba(proto_va)[:,1]; oof_pr[va]=pp
    print('Proto:   AUC=%.4f | per-rule=%.4f' % (roc_auc_score(yva, pp), per_rule_auc(yva, pp, va_df['rule'].values)))

    # Linear heads
    ra = clone(pipe_ra); ra.fit(tr_df, ytr); pr = ra.predict_proba(va_df)[:,1]; oof_ra[va]=pr
    print('RA_svc:  AUC=%.4f | per-rule=%.4f' % (roc_auc_score(yva, pr), per_rule_auc(yva, pr, va_df['rule'].values)))
    bo = clone(pipe_bo); bo.fit(tr_df, ytr); pb = bo.predict_proba(va_df)[:,1]; oof_bo[va]=pb
    print('BO_svc:  AUC=%.4f | per-rule=%.4f' % (roc_auc_score(yva, pb), per_rule_auc(yva, pb, va_df['rule'].values)))
    ch = clone(pipe_ch); ch.fit(tr_df, ytr); pc = ch.predict_proba(va_df)[:,1]; oof_ch[va]=pc
    print('Char_svc:AUC=%.4f | per-rule=%.4f' % (roc_auc_score(yva, pc), per_rule_auc(yva, pc, va_df['rule'].values)))

print('\n===== OOF SUMMARY =====')
y = train['rule_violation'].values; rules = train['rule'].values
for name, o in [('RA',oof_ra), ('BO',oof_bo), ('CHAR',oof_ch), ('PROTO',oof_pr)]:
    print(f"{name:<6} AUC={roc_auc_score(y,o):.5f} | per-rule={per_rule_auc(y,o,rules):.5f}")



from itertools import product
oof_mat = np.column_stack([oof_pr, oof_ra, oof_bo, oof_ch]); names = ['proto','ra','bo','char']
def evaluate(ws):
    ws = np.array(ws, dtype=float); ws = ws / (ws.sum() + 1e-12)
    pred = (oof_mat * ws).sum(1)
    return per_rule_auc(y, pred, rules), roc_auc_score(y, pred)
best = (-1, -1, None)
rng = np.random.default_rng(RANDOM_SEED)
for _ in range(20000):
    ws = rng.dirichlet(np.ones(oof_mat.shape[1]))
    auc_r, auc_g = evaluate(ws)
    if auc_r > best[0] or (abs(auc_r-best[0])<1e-6 and auc_g>best[1]):
        best = (auc_r, auc_g, ws)
grid = np.arange(0, 1.01, 0.1)
for ws in product(grid, repeat=oof_mat.shape[1]):
    if sum(ws)==0: continue
    auc_r, auc_g = evaluate(ws)
    if auc_r > best[0] or (abs(auc_r-best[0])<1e-6 and auc_g>best[1]):
        best = (auc_r, auc_g, np.array(ws)/sum(ws))
best_auc_r, best_auc_g, best_ws = best
print('Best weights:')
for n,w in zip(names, best_ws):
    print(f'  {n:<6} -> {w:.3f}')
print(f'Per-rule AUC={best_auc_r:.5f} | Global AUC={best_auc_g:.5f}')



grp_full = train.groupby('subreddit')['rule_violation'].agg(['sum','count'])
mu_full = train['rule_violation'].mean()
prior_full = (grp_full['sum'] + ALPHA_PRIOR*mu_full) / (grp_full['count'] + ALPHA_PRIOR)

wv_all, cv_all = build_proto_vectors(train)
proto_train_feats = build_proto_feats_with_vecs(train, wv_all, cv_all, sub_prior_map=prior_full, sub_prior_fallback=mu_full)
proto_test_feats  = build_proto_feats_with_vecs(test,  wv_all, cv_all, sub_prior_map=prior_full, sub_prior_fallback=mu_full)

proto_full = HistGradientBoostingClassifier(learning_rate=0.08, max_depth=None, max_iter=500, max_leaf_nodes=31, random_state=RANDOM_SEED)
proto_full.fit(proto_train_feats, train['rule_violation'].values)
p_proto = proto_full.predict_proba(proto_test_feats)[:,1]

ra_full = clone(pipe_ra); ra_full.fit(train, train['rule_violation'].values); p_ra = ra_full.predict_proba(test)[:,1]
bo_full = clone(pipe_bo); bo_full.fit(train, train['rule_violation'].values); p_bo = bo_full.predict_proba(test)[:,1]
ch_full = clone(pipe_ch); ch_full.fit(train, train['rule_violation'].values); p_ch = ch_full.predict_proba(test)[:,1]

ws = best_ws
final_pred = ws[0]*p_proto + ws[1]*p_ra + ws[2]*p_bo + ws[3]*p_ch
sub = sample_sub.copy(); sub['rule_violation'] = final_pred
sub.to_csv('submission.csv', index=False)
print('Saved submission.csv')
sub.head()


