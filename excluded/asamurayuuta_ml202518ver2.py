# ============================================================
#  Random Acts of Pizza • TF-IDF + Logistic Regression Baseline
# ============================================================
import io, zipfile, pathlib, re
import pandas as pd, numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

BASE = '/kaggle/input/random-acts-of-pizza'   # ← Kaggle 自动挂载

# ---------- 1. 读取数据（兼容 ND-JSON 与 JSON 数组） ----------
def load(split: str) -> pd.DataFrame:
    zpath = pathlib.Path(BASE) / f'{split}.json.zip'
    if not zpath.exists():
        raise FileNotFoundError(f'{zpath} 不存在，检查 Input 面板是否已添加数据集')
    with zipfile.ZipFile(zpath) as z:
        inner = z.namelist()[0]
        raw   = z.read(inner)
    try:                         # 先尝试 ND-JSON
        return pd.read_json(io.BytesIO(raw), lines=True)
    except ValueError:           # 再 fallback 到 JSON 数组
        return pd.read_json(io.BytesIO(raw))

train = load('train')
test  = load('test')
print(f'✔ 数据读取成功: train{train.shape}, test{test.shape}')

# ---------- 2. 侦测正文列 ----------
CANDS = [
    'request_text_edit_aware',
    'request_text',
    'request_text_original',
    'request_text_edit'
]
def pick_text_col(df: pd.DataFrame) -> str:
    for c in CANDS:
        if c in df.columns:
            return c
    raise KeyError('正文列不存在！')

text_col_tr = pick_text_col(train)
text_col_te = pick_text_col(test)
print(f'train 用列: {text_col_tr} | test 用列: {text_col_te}')

# ---------- 3. 清洗并拼接标题 ----------
def clean(txt: str) -> str:
    txt = str(txt).lower()
    txt = re.sub(r'http\S+', ' link ', txt)      # 去 URL
    txt = re.sub(r'[^a-z\s]', ' ', txt)          # 只留字母
    return txt

train['request_title'] = train['request_title'].fillna('')
test ['request_title'] = test ['request_title'].fillna('')

train['clean'] = (train[text_col_tr] + ' ' + train['request_title']).apply(clean)
test ['clean'] = (test [text_col_te] + ' ' + test ['request_title']).apply(clean)

# ---------- 4. TF-IDF 特征 ----------
tfidf = TfidfVectorizer(
    ngram_range=(1, 3),
    min_df=1,
    max_df=0.95,
    strip_accents='unicode',
    sublinear_tf=True
)
X_tr_txt = tfidf.fit_transform(train['clean'])
X_te_txt = tfidf.transform(test['clean'])

# ---------- 5. 数值特征 ----------
for df, col in ((train, text_col_tr), (test, text_col_te)):
    txt = df[col].str.lower()
    df['char_len']   = txt.str.len()
    df['num_links']  = txt.str.count('http')
    df['num_thanks'] = txt.str.count('thank')
    df['num_please'] = txt.str.count('please')
    df['num_exclaim']= txt.str.count('!')

NUM_FEATS = ['char_len', 'num_links', 'num_thanks', 'num_please', 'num_exclaim']
X_tr_num = sp.csr_matrix(train[NUM_FEATS].values)
X_te_num = sp.csr_matrix(test [NUM_FEATS].values)

X_train = sp.hstack([X_tr_txt, X_tr_num])
X_test  = sp.hstack([X_te_txt, X_te_num])
y       = train['requester_received_pizza'].astype(int).values
print(f'特征维度: {X_train.shape[1]:,}')

# ---------- 6. 模型 + 交叉验证 ----------
model = LogisticRegression(
    C=2.0,
    class_weight='balanced',
    max_iter=1000,
    n_jobs=-1
)
cv  = StratifiedKFold(5, shuffle=True, random_state=42)
auc = cross_val_score(model, X_train, y, cv=cv, scoring='roc_auc').mean()
print(f'5-fold CV AUC: {auc:.3f}')

# ---------- 7. 训练 & 预测 ----------
model.fit(X_train, y)
test_prob = model.predict_proba(X_test)[:, 1]     # 概率输出

# ---------- 8. 生成提交文件 ----------
sub = pd.DataFrame({
    'request_id': test['request_id'],
    'requester_received_pizza': test_prob
})
sub.to_csv('/kaggle/working/submission.csv', index=False)


