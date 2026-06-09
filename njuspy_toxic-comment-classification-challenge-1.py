import pandas as pd, numpy as np
train = pd.read_csv('/kaggle/input/toxic-comment/train.csv')
test = pd.read_csv('/kaggle/input/toxic-comment/test.csv')

display(train.head()) 

lens = train.comment_text.str.len()
print(lens.mean(), lens.std(), lens.max())
lens.hist();


label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1 - train[label_cols].max(axis=1) # max为0 即取1时没有标签
display(train.describe())

# 填补空缺
COMMENT = 'comment_text'
train.fillna({COMMENT: "unknown"}, inplace=True)
test.fillna({COMMENT: "unknown"}, inplace=True)


import re, string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

n = train.shape[0]
vec = TfidfVectorizer(
    ngram_range=(1,2),           # 用 1-gram 和 2-gram（词+短词组）
    tokenizer=tokenize,          # 自定义分词：把标点单独成词
    token_pattern=None,          # 自定义 避免冲突告警
    min_df=3,                    # 词至少在 3 个文档中出现才保留（去掉极罕见噪声）
    max_df=0.9,                  # 出现于 >90% 文档的词丢弃（如非常常见的停用词）
    strip_accents='unicode',     # 统一去重音/变体（拉丁语系常用）
    use_idf=True,                   # 启用 IDF
    smooth_idf=True,                # IDF 平滑，避免分母为 0
    sublinear_tf=True               # TF 做对数缩放，弱化长文本/重复的影响
)
trn_term_doc = vec.fit_transform(train[COMMENT])
test_term_doc = vec.transform(test[COMMENT])

trn_term_doc, test_term_doc


from sklearn.linear_model import LogisticRegression
def pr(y_i, y):
    p = trn_term_doc[y==y_i].sum(0) # 每个词 在标签y那列 对应值为 y_i的TF-IDF总和
    return (p+1) / ((y==y_i).sum()+1) # 条件概率 或者说词对 y=y_i的平均贡献


def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y)) # 每个词对标签的权重向量
    m = LogisticRegression(C=4, solver='liblinear', dual=True)  # 正则化系数; 对偶训练
    x_nb = trn_term_doc.multiply(r) # 原训练集 乘上 每个词对标签的权重向量

    m.fit(x_nb, y)
    accuracy = m.score(x_nb, y)  # 计算训练集上的准确率
    print(f"Training accuracy for label: {accuracy:.4f}")  # 打印准确率

    return m, r

preds = np.zeros((len(test), len(label_cols))) # 初始化预测答案

for i, j in enumerate(label_cols):
    print('fit', j)
    m,r = get_mdl(train[j]) # 对 j 那个标签 计算模型
    preds[:,i] = m.predict_proba(test_term_doc.multiply(r))[:,1] # 二分类结果为1的概率


subm = pd.read_csv('/kaggle/input/toxic-comment2/sample_submission.csv')
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
submission.to_csv('submission.csv', index=False)

