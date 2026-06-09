import numpy as np
import pandas as pd

df_train = pd.read_csv('/kaggle/input/quora-question-pairs/train.csv.zip')
display(df_train.head())
df_test = pd.read_csv('/kaggle/input/quora-question-pairs/test.csv')
display(df_test.head())

for col in ['question1','question2']:
    df_train[col] = df_train[col].fillna('').astype(str)
    df_test[col]  = df_test[col].fillna('').astype(str)


import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
pal = sns.color_palette()

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

dist_train = train_qs.apply(len) # apply长度
dist_test = test_qs.apply(len)
plt.figure(figsize=(15, 10)) # 直方图
plt.hist(dist_train, bins=200, range=[0, 200], color=pal[2], density=True, label='train')
plt.hist(dist_test, bins=200, range=[0, 200], color=pal[1], density=True, alpha=0.5, label='test')
plt.legend()

print('mean-train {:.2f} mean-test {:.2f} max-train {:.2f} max-test {:.2f}'.format(dist_train.mean(), 
                          dist_test.mean(),dist_train.max(), dist_test.max())) # 均值 最大值


dist_train = train_qs.apply(lambda x: len(x.split(' ')))
dist_test = test_qs.apply(lambda x: len(x.split(' ')))

plt.figure(figsize=(15, 10))
plt.hist(dist_train, bins=50, range=[0, 50], color=pal[2], density=True, label='train')
plt.hist(dist_test, bins=50, range=[0, 50], color=pal[1], density=True, alpha=0.5, label='test')
plt.legend()

print('mean-train {:.2f} std-train {:.2f} mean-test {:.2f} std-test {:.2f} max-train {:.2f} max-test {:.2f}'.format(dist_train.mean(), 
                          dist_train.std(), dist_test.mean(), dist_test.std(), dist_train.max(), dist_test.max()))


from wordcloud import WordCloud
cloud = WordCloud(width=1440, height=1080).generate(' '.join(train_qs.astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')


qmarks = np.mean(train_qs.apply(lambda x: '?' in x))  # 包含问号的问题比例
math = np.mean(train_qs.apply(lambda x: '[math]' in x))  # 包含数学标签的问题比例
fullstop = np.mean(train_qs.apply(lambda x: '.' in x))  # 包含句号的问题比例
numbers = np.mean(train_qs.apply(lambda x: max([y.isdigit() for y in x],default=False)))  # 包含数字的比例

print('Questions with question marks: {:.2f}%'.format(qmarks * 100))
print('Questions with [math] tags: {:.2f}%'.format(math * 100))
print('Questions with full stops: {:.2f}%'.format(fullstop * 100))
print('Questions with numbers: {:.2f}%'.format(numbers * 100))


def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:  # 移除停用词
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:  # 移除停用词
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    # 计算共享单词比例
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

from nltk.corpus import stopwords
stops = set(stopwords.words("english"))
plt.figure(figsize=(15, 5))
df_train['word_match'] = df_train.apply(word_match_share, axis=1)
df_test['word_match'] = df_test.apply(word_match_share, axis=1)
plt.hist(df_train['word_match'][df_train['is_duplicate'] == 0], bins=20, density=True, label='Not Duplicate')
plt.hist(df_train['word_match'][df_train['is_duplicate'] == 1], bins=20, density=True, alpha=0.7, label='Duplicate')
plt.legend()


from sklearn.feature_extraction.text import TfidfVectorizer

all_questions = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist() +
                          df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

tfidf = TfidfVectorizer(ngram_range=(1,1), min_df=3, stop_words='english')  # 可改 min_df, ngram_range
tfidf.fit(all_questions)

# transform
q1_train_tfidf = tfidf.transform(df_train['question1'])
q2_train_tfidf = tfidf.transform(df_train['question2'])
q1_test_tfidf = tfidf.transform(df_test['question1'])
q2_test_tfidf = tfidf.transform(df_test['question2'])

# 逐行计算稀疏矩阵的余弦相似度（向量化、避免循环里的 dense 转换）
def cosine_sim_sparse(a, b):
    # a, b: sparse matrices with same shape (n_rows, n_features)
    num = a.multiply(b).sum(axis=1).A1               # numerator (dot product)
    denom = np.sqrt(a.multiply(a).sum(axis=1).A1) * np.sqrt(b.multiply(b).sum(axis=1).A1)
    # 防止分母为0
    denom = np.where(denom == 0, 1e-9, denom)
    return num / denom

df_train['tfidf_cosine'] = cosine_sim_sparse(q1_train_tfidf, q2_train_tfidf)
df_test['tfidf_cosine']  = cosine_sim_sparse(q1_test_tfidf,  q2_test_tfidf)



feature_cols = ['word_match', 'tfidf_cosine']
X = df_train[feature_cols].fillna(0)
X_test = df_test[feature_cols].fillna(0)
y = df_train['is_duplicate'].values


pos = X[y == 1]
neg = X[y == 0]

p = 0.165   # 目标正样本比例
scale = ((len(pos) / (len(pos) + len(neg))) / p) - 1
# 扩倍复制 neg
while scale > 1:
    neg = pd.concat([neg, neg])
    scale -= 1
if scale > 0:
    neg = pd.concat([neg, neg[:int(scale * len(neg))]])

X_bal = pd.concat([pos, neg])
y_bal = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_bal, y_bal, test_size=0.2, random_state=4242, shuffle=True)


from sklearn.metrics import log_loss, roc_auc_score
import xgboost as xgb

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.02,
    'max_depth': 4,
    'seed': 4242,
    'silent': 1
}

d_train = xgb.DMatrix(X_train, label=y_train)
d_valid = xgb.DMatrix(X_valid, label=y_valid)
d_test  = xgb.DMatrix(X_test)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
bst = xgb.train(params, d_train, num_boost_round=400, evals=watchlist,
                early_stopping_rounds=50, verbose_eval=10)

p_valid = bst.predict(d_valid)
print("Validation logloss:", log_loss(y_valid, p_valid))
print("Validation AUC:", roc_auc_score(y_valid, p_valid))


p_test = bst.predict(d_test)

sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = p_test
sub.to_csv('submission.csv', index=False)

