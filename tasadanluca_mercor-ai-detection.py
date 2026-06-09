import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from scipy.sparse import hstack
import spacy
import numpy as np
import warnings
import collections
warnings.filterwarnings('ignore')


!pip -q install textstat
!pip -q install textblob
# !pip uninstall -y spellchecker
!pip -q install pyspellchecker
import textstat
from spellchecker import SpellChecker
import re
from textblob import TextBlob


train_path = r'/kaggle/input/mercor-ai-detection/train.csv'
test_path = r'/kaggle/input/mercor-ai-detection/test.csv'
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
print(len(train_data), len(test_data))
print(train_data.shape, test_data.shape)
print(train_data.keys(), test_data.keys())


X_train = train_data[['answer', 'topic']]
y_train = train_data['is_cheating']



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
    #features['spelling_error_count'] = len(misspelled)
    
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

    # #Readibility scores
    # flesch_grade = textstat.flesch_kincaid_grade(text)
    # gunning_fog = textstat.gunning_fog(text)
    # features['flesch_kincaid_grade'] = flesch_grade
    # features['gunning_fog'] = gunning_fog

    #Sentiment scores
    sentiment = TextBlob(text).sentiment
    features['subjectivity'] = sentiment.subjectivity
    features['polarity'] = sentiment.polarity
    
    return features


from tqdm import tqdm
tfidf_vec = TfidfVectorizer(
    ngram_range=(1, 4),        
    analyzer='char_wb',        
    max_features=10000
)
tfidf_train_features = tfidf_vec.fit_transform(train_data['answer'])

tqdm.pandas(desc="Extracting features")
ling_features_series = train_data['answer'].progress_apply(extract_features)
ling_features_df = pd.DataFrame.from_records(ling_features_series)
print(f"Linguistic features shape: {ling_features_df.shape}")

X_train_features = hstack([
    tfidf_train_features,             
    ling_features_df.values     
])

X_train_features = X_train_features.tocsr()


print("Preprocessing test data (TF-IDF)...")
tfidf_test_features = tfidf_vec.transform(test_data['answer'])

print("Preprocessing test data (Linguistic)...")
ling_test_features_series = test_data['answer'].progress_apply(extract_features)
ling_test_features_df = pd.DataFrame.from_records(ling_test_features_series)

X_test_features = hstack([
    tfidf_test_features,
    ling_test_features_df.values
]).tocsr()


from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier


sgd_pipeline = Pipeline([
    ('scaler', StandardScaler(with_mean = False)),
    ('sgd', SGDClassifier(
        loss='log_loss',
        random_state=42,
        n_jobs=-1
    ))
])
knn_pipeline = Pipeline([
    ('scaler', StandardScaler(with_mean = False)),
    ('knn', KNeighborsClassifier(
        n_neighbors=10, # This is a key parameter to tune
        n_jobs=-1
    ))
])

# cnb_pipeline = Pipeline([
#     ('scaler', StandardScaler(with_mean = False)),
#     ('cnb', ComplementNB())
# ])
estimators = [
    ('lgbm', LGBMClassifier(
        random_state=42,
        n_estimators=2000,
        learning_rate=0.05,
        n_jobs=-1,
        verbose = -1,
        device = 'gpu'
    )),
    ('sgd_pipeline', sgd_pipeline),
    ('knn_pipeline', knn_pipeline)
]


# trust_model = LogisticRegression(random_state=42)
# stacking_model = StackingClassifier(
#     estimators=estimators, 
#     final_estimator=trust_model, 
#     cv=5,
#     n_jobs=-1,
# )
lgbm_model = LGBMClassifier(
    random_state=42,
    n_estimators=2000,
    learning_rate=0.05,
    n_jobs=-1,
    verbose = -1,
    device = 'gpu'
)
xgb_model = XGBClassifier(
        random_state=42,
        n_estimators=2000,  
        n_jobs=-1,
    )


from sklearn.ensemble import VotingClassifier
voting_model = VotingClassifier(
    estimators=estimators,
    voting='soft',
    weights=[0.4, 0.3, 0.3] 
)
#voting_model.fit(X_train_features, y_train)


# for name, est in estimators:
#     print(f"Testing: {name}")
#     try:
#         est.fit(X_train_features[:100], y_train[:100])
#         print(f"{name} works fine")
#     except Exception as e:
#         print(f"{name} failed: {e}")





cv_scores = cross_val_score(
    voting_model,
    X_train_features, 
    y_train, 
    cv=5, 
    scoring='roc_auc',
    error_score='raise'

)

print(f"Cross-Validation AUC Scores: {cv_scores}")
print(f"Mean CV AUC:   {np.mean(cv_scores):.4f}")
print(f"Std Dev CV AUC: {np.std(cv_scores):.4f}")
voting_model.fit(X_train_features, y_train)

print("Making predictions on the test set...")
test_probabilities = voting_model.predict_proba(X_test_features)
probs_class_1 = test_probabilities[:, 1]





print("Creating submission file...")
submission_df = pd.DataFrame({
    'id': test_data['id'],
    'label': probs_class_1
})

submission_df.to_csv('submission.csv', index=False)

print(submission_df.head())

