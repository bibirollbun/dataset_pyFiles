#importing essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns
from scipy.sparse import hstack
import collections

#importing preprocessing libraries
from sklearn.feature_extraction.text import TfidfVectorizer

#importing algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

#importing pipelining libraries
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#importing nlp libraries
import spacy
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from wordcloud import WordCloud

#importing evaluation libraries
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')


!pip -q install textstat
!pip -q install textblob
!pip -q install pyspellchecker
import textstat
from textblob import TextBlob
from spellchecker import SpellChecker
import re


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

paths = []
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
print(paths)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv(paths[1])
test_df = pd.read_csv(paths[2])


train_df.head()


train_df.describe(include="object")


train_df.info()


sns.countplot(data=train_df,x='is_cheating')


X_train = train_df[['answer', 'topic']]
y_train = train_df['is_cheating']


nlp = spacy.load("en_core_web_sm", disable = ["ner"])

def extract_features(text):
    doc = nlp(text)
    words = [t for t in doc if not t.is_punct and not t.is_space]
    features = {}
    word_count = len(words)
    avg_word_len= sum(len(w) for w in words) / len(words) if words else 0
    sentence_count = len(list(doc.sents))
    
    spell = SpellChecker()
    text_words = text.lower().split()
    misspelled = spell.unknown(text_words)
    
    if sentence_count == 0:
        avg_sentence_length = 0
    else:
        avg_sentence_length = word_count / sentence_count

    unique_words = set(token.lemma_.lower() for token in words)
    ttr = len(unique_words) / word_count if word_count > 0 else 0

    contractions = re.findall(r"\b\w+['’]\w+\b", text)
    
    punctuation_counts = collections.Counter(token.text for token in doc if token.is_punct)
    
    pos_counts = collections.Counter(token.pos_ for token in doc)
    
    total_tokens = len(doc)
    # Calculate densities (percentage of all tokens)
    pos_density = {
        f"pos_density_{pos}": count / total_tokens
        for pos, count in pos_counts.items()
    }
    stopwords = [t for t in doc if t.is_stop]
    features['stopword_percentage'] = len(stopwords) / word_count
    features['word_count'] = word_count
    features['sentence_count'] = sentence_count
    features['avg_sentence_length'] = avg_sentence_length
    features['type_token_ratio']= ttr
    features['punct_count_comma']= punctuation_counts.get(',', 0)
    features['punct_count_period']= punctuation_counts.get('.', 0)
    features['punct_count_all']= sum(punctuation_counts.values())
    features['contraction_count'] = len(contractions)

    all_pos_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'NOUN', 'PRON', 'PROPN', 'VERB', 'PUNCT', 'SYM', 'NUM']
    for tag in all_pos_tags:
        key = f"pos_density_{tag}"
        if key not in pos_density:
            features[key] = 0.0
        else:
            features[key] = pos_density[key]

    #Readibility scores
    flesch_grade = textstat.flesch_kincaid_grade(text)
    gunning_fog = textstat.gunning_fog(text)
    features['flesch_kincaid_grade'] = flesch_grade
    features['gunning_fog'] = gunning_fog

    #Sentiment scores
    sentiment = TextBlob(text).sentiment
    features['subjectivity'] = sentiment.subjectivity
    features['polarity'] = sentiment.polarity
    
    return features


ling_features_series = train_df['answer'].apply(extract_features)
ling_features_df = pd.DataFrame.from_records(ling_features_series)
print(f"Linguistic features shape: {ling_features_df.shape}")


len(ling_features_df.columns.tolist())


fig,ax = plt.subplots(5,5,figsize=(15,15))
ax = ax.flatten()
for idx,feat in enumerate(ling_features_df.columns.tolist()):
    sns.histplot(
        data =ling_features_df,
        x=feat,
        hue=y_train,
        ax= ax[idx],
        kde=True
    )
plt.tight_layout()


ai_generated_text = '\n'.join(train_df[train_df.is_cheating==1].answer)
human_text = '\n'.join(train_df[train_df.is_cheating==0].answer)


ai_cloud = WordCloud(width=800, height=400, background_color='white').generate(ai_generated_text)
human_cloud = WordCloud(width=800, height=400, background_color='white').generate(human_text)


fig,ax = plt.subplots(1,2,figsize=(12,12))
ax[0].imshow(ai_cloud)
ax[0].axis('off')
ax[0].set_title('Ai Generated Text')

ax[1].imshow(human_cloud)
ax[1].axis('off')
ax[1].set_title('Human Written Text')


tfidf_vec = TfidfVectorizer(
    ngram_range=(1, 4),        
    analyzer='char_wb',        
    max_features=10000
)
tfidf_train_features = tfidf_vec.fit_transform(train_df['answer'])


X_train_features = hstack([
    tfidf_train_features,             
    ling_features_df.values     
])


X_train_features.shape


print("Preprocessing test data (TF-IDF)...")
tfidf_test_features = tfidf_vec.transform(test_df['answer'])

print("Preprocessing test data (Linguistic)...")
ling_test_features_series = test_df['answer'].apply(extract_features)
ling_test_features_df = pd.DataFrame.from_records(ling_test_features_series)

X_test_features = hstack([
    tfidf_test_features,
    ling_test_features_df.values
])


try:
    X_train_features = X_train_features.toarray()
    X_test_features = X_test_features.toarray()

except:
    print("Already converted to Dense")


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier

lr = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('lr', LogisticRegression(random_state=42))
])

sgd = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('sgd', SGDClassifier(random_state=42))
])

lin_svc = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('lin_svc', LinearSVC(random_state=42))
])

dtc = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('rfc', DecisionTreeClassifier(random_state=42))
])

knc = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('knc', KNeighborsClassifier())
])

rfc = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('rfc', RandomForestClassifier(random_state=42))
])

etc = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('etc', ExtraTreesClassifier(random_state=42))
])

abc = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('abc', AdaBoostClassifier(random_state=42))
])

hgbc = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('hgbc', HistGradientBoostingClassifier(random_state=42))
])

xgbc = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('xgbc', XGBClassifier(random_state=42))
])

lgbm = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('lgbm', LGBMClassifier(random_state=42))
])

cc = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('cc', CalibratedClassifierCV())
])


from sklearn.model_selection import cross_val_score

estimators = {'lr':lr, 'sgd':sgd, 'lin_svc':lin_svc, 'dtc':dtc, 'knc':knc,
              'rfc':rfc, 'etc':etc, 'abc':abc, 'hgbc':hgbc, 'xgbc':xgbc, 
              'cc':cc, 'lgbm':lgbm}
results = [(name, cross_val_score(estimator, X_train_features, y_train, cv=3, scoring='roc_auc').mean()) for name, estimator in estimators.items()]



pd.DataFrame(results).sort_values(by=[1], ascending=False)


# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import uniform, loguniform, randint

# param_dist_lr = {
#     'lr__C': loguniform(1e-3, 1e2),           # 0.001 to 100
#     'lr__penalty': ['l1', 'l2', 'elasticnet'],
#     'lr__solver': ['liblinear', 'saga'],
#     'lr__l1_ratio': uniform(0, 1)             # only used for elasticnet
# }

# search = RandomizedSearchCV(
#     estimator=lr,
#     param_distributions=param_dist_lr,
#     n_iter=20,
#     cv=3,
#     scoring='roc_auc',
#     n_jobs=-1,
#     verbose=2,
#     random_state=42
# )

# search.fit(X_train_features, y_train)


# print(search.best_params_)


# param_dist_lin_svc = {
#     'lin_svc__C': loguniform(1e-3, 1e2),
#     'lin_svc__loss': ['hinge', 'squared_hinge'],
#     'lin_svc__tol': loguniform(1e-5, 1e-2)
# }

# search = RandomizedSearchCV(
#     estimator=lin_svc,
#     param_distributions=param_dist_lin_svc,
#     n_iter=20,
#     cv=3,
#     scoring='roc_auc',
#     n_jobs=-1,
#     verbose=2,
#     random_state=42
# )

# search.fit(X_train_features, y_train)


# print(search.best_params_)


# param_dist_hgbc = {
#     'hgbc__learning_rate': loguniform(1e-3, 1e0),
#     'hgbc__max_depth': [None, 3, 5, 8, 12],
#     'hgbc__max_leaf_nodes': randint(15, 255),      # integer range
#     'hgbc__min_samples_leaf': randint(5, 50)
# }

# search = RandomizedSearchCV(
#     estimator=hgbc,
#     param_distributions=param_dist_hgbc,
#     n_iter=20,
#     cv=3,
#     scoring='roc_auc',
#     n_jobs=-1,
#     verbose=2,
#     random_state=42
# )

# search.fit(X_train_features, y_train)


# print(search.best_params_)


# param_dist_xgbc = {
#     'xgbc__n_estimators': randint(100, 800),
#     'xgbc__learning_rate': loguniform(1e-3, 1e0),
#     'xgbc__max_depth': randint(2, 12),
#     'xgbc__subsample': uniform(0.5, 0.5),          # 0.5 to 1.0
#     'xgbc__colsample_bytree': uniform(0.5, 0.5),   # 0.5 to 1.0
#     'xgbc__gamma': loguniform(1e-4, 1e1),
#     'xgbc__reg_lambda': loguniform(1e-3, 1e2),
#     'xgbc__reg_alpha': loguniform(1e-4, 1e1)
# }

# search = RandomizedSearchCV(
#     estimator=xgbc,
#     param_distributions=param_dist_xgbc,
#     n_iter=20,
#     cv=3,
#     scoring='roc_auc',
#     n_jobs=-1,
#     verbose=2,
#     random_state=42
# )

# search.fit(X_train_features, y_train)


# print(search.best_params_)


# param_dist_lgbm = {
#     'lgbm__n_estimators': randint(100, 800),
#     'lgbm__learning_rate': loguniform(1e-3, 1e0),
#     'lgbm__num_leaves': randint(20, 255),
#     'lgbm__max_depth': randint(-1, 12),
#     'lgbm__subsample': uniform(0.5, 0.5),           # 0.5 to 1.0
#     'lgbm__colsample_bytree': uniform(0.5, 0.5)
# }

# search = RandomizedSearchCV(
#     estimator=lgbm,
#     param_distributions=param_dist_lgbm,
#     n_iter=20,
#     cv=3,
#     scoring='roc_auc',
#     n_jobs=-1,
#     verbose=2,
#     random_state=42
# )

# search.fit(X_train_features, y_train)


# print(search.best_params_)


from sklearn.ensemble import VotingClassifier

lr = LogisticRegression(C=29.79454462591363, l1_ratio=0.5978999788110851, penalty='elasticnet', solver='saga', random_state=42)
lin_svc_base = LinearSVC(C=0.001267425589893723, loss='squared_hinge', tol=0.0014655354118727707, random_state=42)
hgbc = HistGradientBoostingClassifier(learning_rate=0.3800329214045198, max_depth=12, max_leaf_nodes=87, min_samples_leaf=43, random_state=42)
xgbc = XGBClassifier(colsample_bytree=0.5917022549267169, gamma=0.0033205591037519565, learning_rate=0.03752055855124281, max_depth=10, n_estimators=660,
                     reg_alpha=0.04206039057901997, reg_lambda=0.09984006580328653, subsample=0.5233328316068078, random_state=42)
lgbm = LGBMClassifier(colsample_bytree=0.6975751180009072, learning_rate=0.6025271171095381, max_depth=6, n_estimators=561,
                      num_leaves=234, subsample=0.7604171300129119, random_state=42)

lin_svc = CalibratedClassifierCV(lin_svc_base, cv=3)

final_pipeline = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('vc', VotingClassifier(
        estimators=[
            # ('lr', lr),
            # ('lin_svc', lin_svc),
            ('hgbc', hgbc),
            ('xgbc', xgbc),
            ('lgbm', lgbm)
        ],
        voting='soft'   # or 'soft' if all models support predict_proba
    ))
])


cv_scores = cross_val_score(
    final_pipeline,
    X_train_features,
    y_train,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    error_score='raise'
)

print("Cross-Validation AUC Scores:", cv_scores)
print(f"Mean CV AUC:   {np.mean(cv_scores):.4f}")
print(f"Std Dev CV AUC: {np.std(cv_scores):.4f}")


# Train final model
final_pipeline.fit(X_train_features, y_train)

print("Making predictions on the test set...")
preds = final_pipeline.predict(X_test_features)


print("Creating submission file...")
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'label': preds
})

submission_df.to_csv('submission.csv', index=False)

print(submission_df.head())

