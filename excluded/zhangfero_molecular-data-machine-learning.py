!pip install -q rdkit-pypi
!pip install HROCH


# === STEP 1: åŸºç¡€åº“å¯¼å…¥ ===
import numpy as np, pandas as pd, sympy as sp
from HROCH import SymbolicRegressor, Xicor
import signal, time
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.ensemble import VotingRegressor
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.feature_extraction.text import CountVectorizer


# === STEP 2: å·¥å…·å‡½æ•° ===
class TimeOutException(Exception): pass

def alarm_handler(signum, frame):
    print(f"raising TimeOutException")
    raise TimeOutException

def simplify(expr: sp.Expr, **args):
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(3)
    try:
        expr2 = sp.simplify(expr, **args)
    except Exception as e:
        print('simplify ' + str(e))
        return expr
    signal.alarm(0)
    return expr2

def round_floats(expr: sp.Expr, precision=4):
    expr2 = expr
    for a in sp.preorder_traversal(expr):
        if isinstance(a, sp.Float):
            expr2 = expr2.subs(a, sp.Float(round(a, precision), precision))
    return expr2

def get_eq(columns, expr: str, precision=4):
    features = [c.replace('.', '_') for c in columns]
    eq = round_floats(sp.parse_expr(expr))
    model_str = str(simplify(eq, ratio=1))
    mapping1 = {'x'+str(i): '$$$'+str(i) for i, _ in enumerate(features)}
    mapping2 = {'$$$'+str(i): k for i, k in enumerate(features)}
    for k, v in reversed(mapping1.items()):
        model_str = model_str.replace(k, v)
    for k, v in reversed(mapping2.items()):
        model_str = model_str.replace(k, v)
    return round_floats(sp.parse_expr(model_str, local_dict={k: sp.Symbol(k) for k in features}), precision=precision)


# === STEP 3: è¯»å�–æ•°æ�® ===
train_df = pd.read_csv('/kaggle/input/molecular-machine-learning/train.csv')
test_df = pd.read_csv('/kaggle/input/molecular-machine-learning/test.csv')
sub_df = pd.read_csv('/kaggle/input/molecular-machine-learning/sample_submission.csv')


def smiles_to_morgan_bits(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return [f'frag_{bit}' for bit in fp.GetOnBits()]

train_smiles_fragments = train_df['Smiles'].apply(smiles_to_morgan_bits)
test_smiles_fragments = test_df['Smiles'].apply(smiles_to_morgan_bits)

vectorizer = CountVectorizer(tokenizer=lambda x: x, lowercase=False, binary=True)
train_frag_df = pd.DataFrame(
    vectorizer.fit_transform(train_smiles_fragments).toarray(),
    columns=vectorizer.get_feature_names_out(),
    index=train_df.index
)
test_frag_df = pd.DataFrame(
    vectorizer.transform(test_smiles_fragments).toarray(),
    columns=vectorizer.get_feature_names_out(),
    index=test_df.index
)

X_base = train_df.drop(columns=["Batch_ID", "T80", "Smiles"])
X_test_base = test_df.drop(columns=["Batch_ID", "T80", "Smiles"])

X_full = pd.concat([X_base, train_frag_df], axis=1)
X_test_full = pd.concat([X_test_base, test_frag_df], axis=1)
y = train_df["T80"]


# === STEP 4: ç‰¹å¾�åˆ†ç»„å¹¶ Xicor æ�’åº� ===
# === ç‰¹å¾�åˆ†ç»„ ===
frag_features = [f for f in X_full.columns if f.startswith("frag_")]
num_features = [f for f in X_full.columns if f not in frag_features]

# === Xicor åˆ†åˆ«æ�’åº� ===
corel_num = sorted([(c, Xicor(X_full[c].values, y.values)) for c in num_features], key=lambda x: x[1], reverse=True)
corel_frag = sorted([(c, Xicor(X_full[c].values, y.values)) for c in frag_features], key=lambda x: x[1], reverse=True)

# === å�ªä¿�ç•™ Top 30 ç‰¹å¾� ===
top_num_features = [c for c, _ in corel_num[:30]]
top_frag_features = [c for c, _ in corel_frag[:100]]
X_num = X_full[top_num_features]
X_frag = X_full[top_frag_features]
X_test_num = X_test_full[top_num_features]
X_test_frag = X_test_full[top_frag_features]


import matplotlib.pyplot as plt
import numpy as np

# å�‡è®¾ä½ å·²æœ‰ corel_frag
xicor_scores = np.array([score for _, score in corel_frag])
indices = np.arange(1, len(xicor_scores) + 1)

# å·®åˆ†å�˜åŒ–ï¼šç”¨æ�¥è§‚å¯Ÿæ‹�ç‚¹è¶‹åŠ¿
deltas = np.diff(xicor_scores)

# ç®€å�•æ–¹æ³•ï¼šæ‰¾åˆ°ç¬¬ä¸€ä¸ªâ€œå�˜åŒ–é��å¸¸ç¼“æ…¢â€�çš„ä½�ç½®
# ä½ å�¯ä»¥æ ¹æ�®ä¸‹é™�é‡�å°�äº�é˜ˆå€¼è®¾ç½®æ‹�ç‚¹ï¼Œæ¯”å¦‚ 0.01 æˆ– 0.005
threshold = 0.3
top_k = next((i for i, d in enumerate(deltas) if abs(d) < threshold), 30)

# å�¯è§†åŒ–
plt.figure(figsize=(10, 5))
plt.plot(indices, xicor_scores, label='Xicor Score (frag)', linewidth=2)
plt.axvline(top_k, color='red', linestyle='--', label=f"Suggested cutoff: Top {top_k}")
plt.title("Xicor Score vs Fragment Feature Rank")
plt.xlabel("Fragment Feature Rank")
plt.ylabel("Xicor Score")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(f"âœ… å»ºè®®é€‰æ‹©ç¢�ç‰‡ç‰¹å¾� Top-{top_k} ä¸ªï¼Œå��ç»­è¾¹é™…å�˜åŒ–è¾ƒå°�")


# === STEP 5: åˆ†åˆ«è®­ç»ƒä¸¤ä¸ª SymbolicRegressor ===
reg_num = SymbolicRegressor(time_limit=100, metric='MSLE', target_clip=[0, 1000], random_state=0)
reg_frag = SymbolicRegressor(time_limit=100, metric='MSLE', target_clip=[0, 1000], random_state=1)
reg_num.fit(X_num, y)
reg_frag.fit(X_frag, y)


# === STEP 6: æ�„å»º VotingRegressor è��å�ˆä¸¤ç±»æ¨¡å�‹ ===
from sklearn.base import BaseEstimator, RegressorMixin

class NumModel(BaseEstimator, RegressorMixin):
    def fit(self, X, y): return self
    def predict(self, X): return reg_num.predict(X[top_num_features])

class FragModel(BaseEstimator, RegressorMixin):
    def fit(self, X, y): return self
    def predict(self, X): return reg_frag.predict(X[top_frag_features])

voter = VotingRegressor(estimators=[
    ('num', NumModel()),
    ('frag', FragModel())
])
voter.fit(X_full, y)


# === STEP 7: è¾“å‡ºè¡¨è¾¾å¼�ä¸�è¯„ä¼° ===
eq_num = get_eq(top_num_features, reg_num.get_models()[0].equation)
eq_frag = get_eq(top_frag_features, reg_frag.get_models()[0].equation)
print("\nğŸ“˜ æ•°å€¼ç‰¹å¾�è¡¨è¾¾å¼�ä½¿ç”¨å�˜é‡�æ•°:", len(eq_num.free_symbols))
print(eq_num)
print("\nğŸ“˜ ç¢�ç‰‡ç‰¹å¾�è¡¨è¾¾å¼�ä½¿ç”¨å�˜é‡�æ•°:", len(eq_frag.free_symbols))
print(eq_frag)


# === STEP 8: é¢„æµ‹å¹¶æ��äº¤ ===
test_preds = np.clip(voter.predict(X_test_full), 0, 1000)
sub_df["T80"] = test_preds
sub_df.to_csv("submission.csv", index=False)
print("\nâœ… Submission saved to submission.csv")

