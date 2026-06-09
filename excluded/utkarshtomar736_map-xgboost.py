import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')


for df in [train, test]:
    df['StudentExplanation'] = df['StudentExplanation'].fillna('no_explanation')
    df['QuestionText'] = df['QuestionText'].fillna('')
    df['MC_Answer'] = df['MC_Answer'].fillna('')

train['Category'] = train['Category'].fillna('UnknownCategory')
train['Misconception'] = train['Misconception'].fillna('UnknownMisconception')
train['target'] = train['Category'] + ':' + train['Misconception']


# Group rare classes
label_counts = train['target'].value_counts()
rare_labels = label_counts[label_counts < 2].index
train['target'] = train['target'].apply(lambda x: 'Other' if x in rare_labels else x)


# from small-batch wins: length bins + keywords
for df in [train, test]:
    df['text'] = df['QuestionText'] + ' ' + df['MC_Answer'] + ' ' + df['StudentExplanation']
    
    # Length binning
    df['exp_length'] = df['StudentExplanation'].str.len()
    df['length_bin'] = pd.cut(df['exp_length'], bins=[0, 50, 150, np.inf], labels=['short', 'medium', 'long'])
    df['text'] += ' length_' + df['length_bin'].astype(str)
    
    # Keyword counts
    keywords = ['fraction', 'decimal', 'add', 'subtract', 'because']
    df['kw_count'] = df['text'].apply(lambda x: sum(x.lower().count(kw) for kw in keywords))
    df['text'] += ' kw_' + df['kw_count'].astype(str)


# Numeric for XGBoost
le = LabelEncoder()
train['target_enc'] = le.fit_transform(train['target'])

# Step 5: Vectorization (TF-IDF with bigrams, from baseline)
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train['text'])
y_train = train['target_enc']


use_gpu = True
num_class = len(le.classes_)
params = {
    'objective': 'multi:softprob',  # For probabilities
    'num_class': num_class,
    'eval_metric': 'mlogloss',     # Multi-class log loss
    'learning_rate': 0.1,
    'n_estimators': 200,           # Rounds from small-batch
    'seed': 42,
    'tree_method': 'hist'
}

if use_gpu:
    params['device'] = 'cuda'


dtrain = xgb.DMatrix(X_train, label=y_train)
model = xgb.train(params, dtrain, num_boost_round=params['n_estimators'])


X_test = vectorizer.transform(test['text'])
dtest = xgb.DMatrix(X_test)
test_proba = model.predict(dtest)


# Format top-3 predictions (rank by descending prob)
top_k = 3
predictions = []
for proba in test_proba:
    top_indices = np.argsort(proba)[-top_k:][::-1]
    top_labels = le.inverse_transform(top_indices)
    predictions.append(' '.join(top_labels))


submission = pd.DataFrame({'row_id': test['row_id'], 'Category:Misconception': predictions})
submission.to_csv('submission.csv', index=False)
print('Submission file created: submission.csv')




