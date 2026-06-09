import numpy as np
import pandas as pd
from sklearn import *
import xgboost as xgb
from nltk.corpus import stopwords
es = set(stopwords.words('english'))
from tqdm import tqdm
tqdm.pandas(desc="Mapping Progress")

p = '/kaggle/input/jigsaw-agile-community-rules/'
train = pd.read_csv(p+'train.csv')
test = pd.read_csv(p+'test.csv')
sub = pd.read_csv(p+'sample_submission.csv')


train['neg'] = train['subreddit'] + ' ' + train['rule'] + ' ' + train['negative_example_1'] + ' ' + train['negative_example_2']
train['pos'] = train['positive_example_1'] + ' ' + train['positive_example_2']

test['neg'] = test['subreddit'] + ' ' + test['rule'] + ' ' + test['negative_example_1'] + ' ' + test['negative_example_2']
test['pos'] = test['positive_example_1'] + ' ' + test['positive_example_2']


exp = {}

def f_experience(c, s):
    global exp
    it = {'memory':10_000, 'inference':0.5, 'sentiment':1e-10,}
    for i in range(len(c)):
        words = set([str(w) for w in str(c[i]).lower().split(' ')])
        for w in words:
            try:
                exp[w]['inference'] += 1
                exp[w]['sentiment'] += s[i]
            except:
                m = [0. for m_ in range(it['memory'])]
                exp[w] = {}
                exp[w]['inference'] = 1
                exp[w]['sentiment'] = s[i]
    for w in exp:
        exp[w]['sentiment'] /= exp[w]['inference'] + it['sentiment']
    return exp


print(train.shape, test.shape)
dfall = pd.concat((train, test))
print(dfall.shape)

EXP = f_experience(dfall['neg'].values, [1 for i in range(len(dfall))])
EXP = f_experience(dfall['pos'].values, [0 for i in range(len(dfall))])


tox = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv')
cols = [c for c in tox.columns if c not in ['id','comment_text']]
tox['target'] = tox[cols].max(axis=1)
tox = tox[tox['target']>0].reset_index(drop=True)
print(tox.shape)

EXP = f_experience(tox['comment_text'].values, tox.target)


#tox = pd.read_csv('/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/all_data.csv')
#tox = tox[tox['rating']=='rejected'].reset_index(drop=True)
#tox['target'] = 1
#print(tox.shape)

#EXP = f_experience(tox['comment_text'].values, tox.target)


dfall['char_len'] = dfall['body'].map(len)
dfall['word_len'] = dfall['body'].map(lambda x: len(str(x).split(' ')))
gqid = dfall.groupby(['row_id']).agg(
    achar_len=('char_len', 'mean'), 
    mnchar_len=('char_len', 'min'), 
    mxchar_len=('char_len', 'max'), 
    aword_len=('word_len', 'mean'),
    mnword_len=('word_len', 'min'), 
    mxword_len=('word_len', 'max'), 
)


tfidf = feature_extraction.text.TfidfVectorizer(analyzer="word", ngram_range=(1,3), min_df = 2, max_df = 0.95,  stop_words="english", dtype=np.float32,  max_features = 1_000)
tfidf.fit(dfall['body'])
#fe = [k for k in tfidf.vocabulary_]
fe = [str(k) for k in range(len(tfidf.vocabulary_))]

def nlpit(df):
    global gqid, tfidf, fe
    df = df.reset_index(drop=True)
    df['inference_sum'] = df['body'].map(lambda x: np.sum([exp[w]['inference'] if w in exp else 0 for w in str(x).lower().split(' ')]))
    df['inference_mean'] = df['body'].map(lambda x: np.mean([exp[w]['inference'] if w in exp else 0 for w in str(x).lower().split(' ')]))
    df['sentiment_sum'] = df['body'].map(lambda x: np.sum([exp[w]['sentiment'] if w in exp else 0.5 for w in str(x).lower().split(' ')]))
    df['sentiment_mean'] = df['body'].map(lambda x: np.mean([exp[w]['sentiment'] if w in exp else 0.5 for w in str(x).lower().split(' ')]))
    df = df.merge(gqid, on=['row_id'], how='left')
    df['rchar_len'] = df['body'].map(len)
    df['-a_rchar_len'] = df['rchar_len'] - df['achar_len']
    df['-mn_rchar_len'] = df['rchar_len'] - df['mnchar_len']
    df['-mx_rchar_len'] = df['rchar_len'] - df['mxchar_len']
    df['rword_len'] = df['body'].map(lambda x: len(str(x).split(' ')))
    df['-a_rword_len'] = df['rword_len'] - df['aword_len']
    df['-mn_rword_len'] = df['rword_len'] - df['mnword_len']
    df['-mx_rword_len'] = df['rword_len'] - df['mxword_len']
    #df['Prompt'] = df.apply(lambda r: r['body'], axis=1)
    #df['Perplexity'] = df['Prompt'].progress_map(lambda x: scorer.get_perplexity(str(x))) #need to cache results
    #df['Perplexity'] = cplex
    #df['Perplexity_ScaleC'] = df['Perplexity'] / (df['rchar_len'] + 1)
    #df['Perplexity_ScaleW'] = df['Perplexity'] / (df['rword_len'] + 1)
    txt = pd.DataFrame(tfidf.transform(df['body']).toarray(), columns=fe)
    df = pd.concat([df, txt], axis=1)
    df = df.fillna(-1)
    df = df.reset_index(drop=True)
    return df


tcols = ['sentiment_mean', 'sentiment_sum', 'inference_sum', 'inference_mean', 'achar_len', 
         #'Perplexity', 'Perplexity_ScaleC', 'Perplexity_ScaleW',
         'mnchar_len', 'mxchar_len', 'aword_len', 'mnword_len', 'mxword_len', 'rchar_len', 
         '-a_rchar_len', '-mn_rchar_len', '-mx_rchar_len', 'rword_len', '-a_rword_len', 
         '-mn_rword_len', '-mx_rword_len'] + fe #

params = {
    'objective': 'multi:softprob',
    'num_class': 2,
    'eval_metric': 'auc',
    'max_depth': 11,
    'learning_rate': 0.02,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'tree_method': 'hist', 
    #'device': 'cuda',
    'random_state': 42
}

skf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(train), 2))
preds = np.zeros((len(test), 2))

for fold, (trn_idx, val_idx) in enumerate(skf.split(train, train.rule_violation)):
    print(f"\nFold {fold+1}")

    dfFold = pd.DataFrame(train.values[trn_idx], columns=train.columns)

    #To Avoid Overfit and Get Bettter Metric Results
    neg = dfFold[dfFold['rule_violation']==1]['body'].values
    pos = dfFold[dfFold['rule_violation']==0]['body'].values
    exp = EXP.copy()

    exp = f_experience(neg, [1 for i in range(len(neg))])
    exp = f_experience(pos, [0 for i in range(len(pos))])

    #remove stopwords
    for sw in es:
        if sw in exp:
            del exp[sw]
        
    trainx = nlpit(train)
    X = trainx[tcols]
    Y = trainx['rule_violation']
    T = nlpit(test)

    dtrain = xgb.DMatrix(X.values[trn_idx], label=Y[trn_idx])
    dvalid = xgb.DMatrix(X.values[val_idx], label=Y[val_idx])

    model = xgb.train(params, dtrain, num_boost_round=5000,
                      evals=[(dvalid, 'valid')],
                      early_stopping_rounds=50,
                      verbose_eval=100)
    oof_preds[val_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration))
    preds += model.predict(xgb.DMatrix(T[tcols]), iteration_range=(0, model.best_iteration)) / skf.n_splits
    
metrics.roc_auc_score(train['rule_violation'], oof_preds[:,1])


model.feature_names = tcols
xgb.plot_importance(model, max_num_features = 30)


test['rule_violation'] = preds[:,1]
test[['row_id','rule_violation']].to_csv('submission.csv', index=False)

