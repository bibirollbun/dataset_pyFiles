# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from io import StringIO
from sklearn.model_selection import train_test_split
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import spacy
from textstat.textstat import textstat
from collections import Counter
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.sparse import hstack, csr_matrix


with open("/kaggle/input/mercor-ai-detection/train.csv","r") as file:
    data=file.read()
    train_df=pd.read_csv(StringIO(data))
    print(train_df.head())



X=train_df.drop('is_cheating', axis=1)
X=X.drop('id',axis=1)
print(X.head())


with open("/kaggle/input/mercor-ai-detection/test.csv","r") as file:
    data=file.read()
    test_df=pd.read_csv(StringIO(data))
    print(test_df.head())


from nltk.corpus import words
try:
    ENGLISH_WORDS = set(words.words())
except LookupError:
    # Fallback/error handling if nltk 'words' is not downloaded
    ENGLISH_WORDS = set() 
    print("Warning: NLTK 'words' corpus not found. Misspelling feature will be skipped.")

def load_spacy_model():
    
    global NLP_MODEL
    if NLP_MODEL is None:
        try:
            # Load the small English pipeline
            print("Loading SpaCy model 'en_core_web_sm'...")
            NLP_MODEL = spacy.load("en_core_web_sm")
            print("Model loaded successfully.")
        except OSError:
            # This handles cases where the model needs to be downloaded first
            print("SpaCy model 'en_core_web_sm' not found locally. Please run:")
            print("!python -m spacy download en_core_web_sm")
            print("... and ensure 'textstat' is installed: !pip install textstat")
            return None
    return NLP_MODEL

def count_mispelled_words(text):
    if not text:
        return 0, 0.0
    

    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if not tokens:
        return 0, 0.0
    
    total_words = len(tokens)
    mispelled_count = 0
    
    for word in tokens:
        if word not in ENGLISH_WORDS:
            mispelled_count += 1
            

    return mispelled_count, mispelled_count / total_words


def create_nlp_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs comprehensive NLP feature engineering on the hardcoded 'answer' column,
    including basic counts, readability, SpaCy metrics, and NEW misspelling features.

    Args:
        df (pd.DataFrame): The input DataFrame (e.g., train or test set).
        This DataFrame MUST contain a column named 'answer'.

    Returns:
        pd.DataFrame: The DataFrame with new NLP feature columns added.
    """

    nlp = load_spacy_model()
    if nlp is None:
        print("Feature engineering aborted due to missing SpaCy model.")
        return df


    text_column = 'answer'
    
    if text_column not in df.columns:
        print(f"Error: DataFrame must contain a column named '{text_column}'. Aborting.")
        return df

    print(f"Applying NLP feature engineering on column '{text_column}'...")
    



    text_data = df[text_column].astype(str)


    df['word_count'] = text_data.apply(lambda x: len(str(x).split()))
    df['char_count'] = text_data.apply(lambda x: len(str(x)))
    df['unique_word_count'] = text_data.apply(lambda x: len(set(str(x).split())))
    df['sentence_count'] = text_data.apply(lambda x: len(re.split(r'[.!?]+', str(x).strip())))
    

    if ENGLISH_WORDS: 
        print("Calculating misspelling features...")
        misspell_results = text_data.apply(count_mispelled_words).apply(pd.Series)
        misspell_results.columns = ['mispelled_word_count', 'mispelled_word_ratio']

        df['mispelled_word_count'] = misspell_results['mispelled_word_count']
        df['mispelled_word_ratio'] = misspell_results['mispelled_word_ratio']
    

    df['fk_grade'] = text_data.apply(textstat.flesch_kincaid_grade)

    df['dale_chall_score'] = text_data.apply(textstat.dale_chall_readability_score)

    df['gunning_fog'] = text_data.apply(textstat.gunning_fog)
    

    

    docs = list(nlp.pipe(text_data))
    
    pos_noun_counts = []
    pos_verb_counts = []
    pos_adj_counts = []
    named_entity_counts = []

    for doc in docs:
        token_counts = {'NOUN': 0, 'VERB': 0, 'ADJ': 0}
        

        for token in doc:
            if token.pos_ in token_counts:
                token_counts[token.pos_] += 1
        
        pos_noun_counts.append(token_counts['NOUN'])
        pos_verb_counts.append(token_counts['VERB'])
        pos_adj_counts.append(token_counts['ADJ'])
        

        named_entity_counts.append(len(doc.ents))

    df['pos_noun_count'] = pos_noun_counts
    df['pos_verb_count'] = pos_verb_counts
    df['pos_adj_count'] = pos_adj_counts
    df['named_entity_count'] = named_entity_counts

    print("Feature engineering complete.")
    return df


X = create_nlp_features(X.copy()) 
print(X.head())


from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix


xcopy = X.copy()

tfidf_vectorizer = TfidfVectorizer(
    stop_words='english', 
    ngram_range=(1, 2), 
    max_features=10000
)


print("Vectorizing 'topic' column...")
X_topic_features = tfidf_vectorizer.fit_transform(xcopy['topic'].astype(str))
topic_feature_names = [f'topic_{n}' for n in tfidf_vectorizer.get_feature_names_out()]

print(f"'topic' TF-IDF shape: {X_topic_features.shape}")


TEXT_COLUMN = 'answer' 


tfidf_vectorizer_text = TfidfVectorizer(
    stop_words='english', 
    ngram_range=(1, 2), 
    max_features=40000 
)

print(f"Vectorizing '{TEXT_COLUMN}' column...")
X_answer_features = tfidf_vectorizer_text.fit_transform(xcopy[TEXT_COLUMN].astype(str))

print(f"'{TEXT_COLUMN}' TF-IDF shape: {X_answer_features.shape}")


X_combined_tfidf = hstack([X_topic_features, X_answer_features])

print(f"Combined TF-IDF Feature Shape: {X_combined_tfidf.shape}")


xcopy = xcopy.drop(['topic', TEXT_COLUMN], axis=1)

print(f"\nFinal xcopy columns (numerical features only): {xcopy.columns.tolist()}")
print(f"Final xcopy shape: {xcopy.shape}")



print(xcopy.head())
print(xcopy.shape)


y=train_df['is_cheating']
print(y.head())


# train-val split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    xcopy, 
    y, 
    test_size=0.2,      
    random_state=90,    
    stratify=y          
)
print(f"X_train Samples: {X_train.shape[0]} ({y_train.shape[0]})")
print(f"X_val Samples: {X_val.shape[0]} ({y_val.shape[0]})")


from sklearn.model_selection import train_test_split

X_train_dense, X_val_dense, X_train_sparse, X_val_sparse, y_train, y_val = train_test_split(
    xcopy,              
    X_combined_tfidf,   
    y,                 
    test_size=0.2,       
    random_state=90,     
    stratify=y          
)


print(f"--- Dense Feature Split (NLP Features) ---")
print(f"X_train_dense Shape: {X_train_dense.shape}")
print(f"X_val_dense Shape: {X_val_dense.shape}")
print(f"y_train Shape: {y_train.shape}")
print(f"y_val Shape: {y_val.shape}")

print(f"\n--- Sparse Feature Split (TF-IDF Features) ---")

print(f"X_train_sparse Shape: {X_train_sparse.shape}")
print(f"X_val_sparse Shape: {X_val_sparse.shape}")



test_df = create_nlp_features(test_df.copy()) 


print(test_df.head())


X_test_xcopy = test_df.copy() 


TOPIC_COLUMN = 'topic'
TEXT_COLUMN = 'answer' 

print(f"Starting feature transformation for the Test Data...")

X_test_topic_features = tfidf_vectorizer.transform(X_test_xcopy[TOPIC_COLUMN].astype(str))
print(f"'topic' TF-IDF test shape: {X_test_topic_features.shape}")

X_test_answer_features = tfidf_vectorizer_text.transform(X_test_xcopy[TEXT_COLUMN].astype(str))
print(f"'{TEXT_COLUMN}' TF-IDF test shape: {X_test_answer_features.shape}")


X_test_combined_tfidf = hstack([X_test_topic_features, X_test_answer_features])

print(f"Combined Test TF-IDF Feature Shape: {X_test_combined_tfidf.shape}")


X_test_xcopy = X_test_xcopy.drop([TOPIC_COLUMN, TEXT_COLUMN], axis=1)

print(f"\nFinal X_test_xcopy columns (numerical features only): {X_test_xcopy.columns.tolist()}")
print(f"Final X_test_xcopy shape: {X_test_xcopy.shape}")


if X_combined_tfidf.shape[1] != X_test_combined_tfidf.shape[1]:
    print("\nERROR: Column count mismatch in sparse features! Check TF-IDF parameters.")


print(y_train.value_counts())


print(y_val.value_counts())


# sparse data model - 1
from sklearn.linear_model import LogisticRegression

print("--- Training Model 1: Sparse TF-IDF Features (Logistic Regression) ---")

model_sparse = LogisticRegression(
    C=0.5,             # Regularization strength (can be tuned)
    solver='saga',     # Good for sparse data
    max_iter=1000,     # Increase max_iter for convergence
    random_state=90,
    n_jobs=-1
)

model_sparse.fit(X_train_sparse, y_train)


val_pred_sparse = model_sparse.predict_proba(X_val_sparse)[:, 1]
auc_linreg = roc_auc_score(y_val, val_pred_sparse)
print(f"✨ Linear Regression ROC-AUC: {auc_linreg:.4f}")
print("Sparse data Model trained successfully.")


from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV # Needed to get probabilities for ensemble

print("\n--- Testing Alternative Sparse Model: LinearSVC ---")

# LinearSVC does not natively support predict_proba, so we wrap it
svc_model = LinearSVC(C=0.1, random_state=90, max_iter=1000)

# CalibratedClassifierCV ensures we get reliable probabilities for the ensemble
model_sparse_svc = CalibratedClassifierCV(svc_model, method='isotonic', cv=5)

# Fit and evaluate (use X_train_sparse or X_train_sparse_selected)
model_sparse_svc.fit(X_train_sparse, y_train)

val_pred_svc = model_sparse_svc.predict_proba(X_val_sparse)[:, 1]
auc_svc = roc_auc_score(y_val, val_pred_svc)

print(f"✨ LinearSVC ROC-AUC: {auc_svc:.4f}")
print("Sparse data Model trained successfully.")


from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV 
from sklearn.metrics import roc_auc_score

print("--- Alternative 1: Calibrated LinearSVC (Sparse Model) ---")

# 1. Base Model: LinearSVC (tunes regularization C and penalty='l2')
base_svc = LinearSVC(
    C=0.0009,                 # Start with C=0.1 (can be tuned)
    loss='squared_hinge',  # Standard loss for SVC
    random_state=90, 
    max_iter=1000
)

# 2. Wrapper: CalibratedClassifierCV to get probabilities
model_sparse_alt = CalibratedClassifierCV(
    base_svc, 
    method='isotonic', # Use 'isotonic' for best calibration when sample count is high
    cv=5               # 5-fold cross-validation inside the calibration
)

model_sparse_alt.fit(X_train_sparse, y_train)
val_pred_svc = model_sparse_alt.predict_proba(X_val_sparse)[:, 1]
auc_svc = roc_auc_score(y_val, val_pred_svc)

print(f"✅ LinearSVC ROC-AUC: {auc_svc:.4f}")


from sklearn.ensemble import RandomForestClassifier

print("\n--- Training Model 2: Dense NLP Features (Random Forest) ---")

model_dense = RandomForestClassifier(
    n_estimators=300,  # Number of trees
    max_depth=10,      # Limit depth to prevent overfitting
    random_state=90,
    n_jobs=-1
)

# Note: We fit on the dense DataFrame X_train_dense
model_dense.fit(X_train_dense, y_train)

# Generate probability predictions on the validation set
val_pred_dense = model_dense.predict_proba(X_val_dense)[:, 1]

print("Model 2 trained successfully.")


import lightgbm as lgb
from sklearn.metrics import roc_auc_score

print("\n--- Alternative 2: LightGBM (Dense Model) ---")


lgb_model = lgb.LGBMClassifier(
    objective='binary', 
    metric='auc',
    n_estimators=300, 
    learning_rate=0.05, # Slower learning rate often leads to better results
    num_leaves=31, 
    random_state=90, 
    n_jobs=-1
)

lgb_model.fit(
    X_train_dense, y_train,
    eval_set=[(X_val_dense, y_val)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
)

val_pred_lgbm = lgb_model.predict_proba(X_val_dense)[:, 1]
auc_lgbm = roc_auc_score(y_val, val_pred_lgbm)

print(f"✅ LightGBM ROC-AUC: {auc_lgbm:.4f}")


from sklearn.metrics import roc_auc_score
import numpy as np

# Single Model Evaluation (for comparison)
auc_sparse = roc_auc_score(y_val, val_pred_svc)
auc_dense = roc_auc_score(y_val, val_pred_dense)

print(f"\nModel 1 (Sparse) ROC-AUC: {auc_sparse:.4f}")
print(f"Model 2 (Dense) ROC-AUC: {auc_dense:.4f}")

# Ensemble Averaging
# We combine the predictions by simple averaging (50/50 weights)
# You can later tune these weights for better performance!
ENSEMBLE_WEIGHT_SPARSE = 0.4
ENSEMBLE_WEIGHT_DENSE = 0.6

val_pred_ensemble = (val_pred_svc * ENSEMBLE_WEIGHT_SPARSE) + \
                    (val_pred_dense * ENSEMBLE_WEIGHT_DENSE)

# Final Ensemble Evaluation
auc_ensemble = roc_auc_score(y_val, val_pred_ensemble)

print(f"\nFINAL ENSEMBLE ROC-AUC: {auc_ensemble:.4f}")


print(X_test_xcopy['id'])


import pandas as pd
import numpy as np

# ensemble weights
ENSEMBLE_WEIGHT_SPARSE = 0.4
ENSEMBLE_WEIGHT_DENSE = 0.6

print("Generating sparse model test predictions...")
test_pred_sparse = model_sparse.predict_proba(X_test_combined_tfidf)[:, 1]


TRAINING_COLUMNS = X_train_dense.columns.tolist()


X_test_xcopy_for_pred = X_test_xcopy.copy() 
if 'id' in X_test_xcopy_for_pred.columns:
    # Use the original test_df ID column for the final submission later
    test_ids = X_test_xcopy_for_pred['id'] 
    X_test_xcopy_for_pred = X_test_xcopy_for_pred.drop('id', axis=1)

X_test_xcopy_for_pred = X_test_xcopy_for_pred[TRAINING_COLUMNS]

print("Generating dense model test predictions...")
# Predict probabilities using the dense model 
test_pred_dense = model_dense.predict_proba(X_test_xcopy_for_pred)[:, 1]

final_test_predictions = (test_pred_sparse * ENSEMBLE_WEIGHT_SPARSE) + \
                         (test_pred_dense * ENSEMBLE_WEIGHT_DENSE)

binary_predictions = (final_test_predictions >= 0.5).astype(int)


submission_df = pd.DataFrame({
    'id': test_ids, # Use the saved ID column
    'is_cheating': binary_predictions 
})

submission_df.to_csv('submission.csv', index=False)

print(submission_df.tail())

