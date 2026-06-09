import sys
import gc
import re
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV


from sklearn.feature_extraction.text import TfidfVectorizer

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier


def math_text_preprocessor(text):
    # Math-specific preprocessing
    text = re.sub(r'\$(.*?)\$', r' MATH_EXPR \1 MATH_EXPR ', text)  # Preserve math expressions
    text = re.sub(r'\\\w+', ' ', text)  # Remove LaTeX commands but keep content
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation except math symbols
    text = re.sub(r'\d+', ' NUM ', text)  # Normalize numbers
    return text.lower().strip()



test = pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv')
sub = pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/sample_submission.csv')
train = pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv')



train["Question"] = train["Question"].apply(math_text_preprocessor)
test["Question"]  = test["Question"].apply(math_text_preprocessor)




train = train.drop_duplicates(subset=['Question'])

train.reset_index(drop=True, inplace=True)


test.Question.values


LOWERCASE = False
VOCAB_SIZE = 30522


# Creating Byte-Pair Encoding tokenizer
raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))


# Adding normalization and pre_tokenizer
raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

# Adding special tokens and creating trainer instance
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)



# Creating huggingface dataset object
tok_data = pd.concat([ train[["Question"]],  test[["Question"]] ]).reset_index(drop=True)

dataset = Dataset.from_pandas(tok_data)

def train_corp_iter():
    """
    A generator function for iterating over a dataset in chunks.
    """    
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["Question"]

# Training from iterator REMEMBER it's training on test set...
raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)

tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)


tokenized_texts_test = []

# Tokenize test set with new tokenizer
for text in tqdm(test['Question'].tolist()):
    tokenized_texts_test.append(tokenizer.tokenize(text))


# Tokenize train set
tokenized_texts_train = []

for text in tqdm(train['Question'].tolist()):
    tokenized_texts_train.append(tokenizer.tokenize(text))


tokenized_texts_test[1]


def dummy(text):
    """
    A dummy function to use as tokenizer for TfidfVectorizer. It returns the text as it is since we already tokenized it.
    """
    return text


# Fitting TfidfVectoizer on test set

vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, analyzer = 'word',
    tokenizer = dummy,
    preprocessor = dummy,
    token_pattern = None, strip_accents='unicode'
                            )

vectorizer.fit(tokenized_texts_test)

# Getting vocab
vocab = vectorizer.vocabulary_

#print(vocab)


# # Here we fit our vectorizer on train set but this time we use vocabulary from test fit.
# vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, vocabulary=vocab,
#                             analyzer = 'word',
#                             tokenizer = dummy,
#                             preprocessor = dummy,
#                             token_pattern = None, strip_accents='unicode'
#                             )

math_stop_words = {'find', 'prove', 'show', 'calculate', 'determine', 'let', 'given', 'solve'}

# Create TF-IDF pipeline with math-specific features
tfidf_pipe = make_pipeline(
    TfidfVectorizer(
        stop_words=list(math_stop_words),
        ngram_range=(1, 2),  # Add bigrams
        max_features=25000,
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,  # Use 1 + log(tf)
        analyzer='word',
        token_pattern=r'\b[^\d\W]+\b',  # Exclude pure numbers
        tokenizer = dummy,
        preprocessor = dummy,
    ),
    FunctionTransformer(lambda x: x.tocsc()) 
)


tf_train = tfidf_pipe.fit_transform(tokenized_texts_train)
tf_test = tfidf_pipe.transform(tokenized_texts_test)

del vectorizer
gc.collect()


y_train = train['label'].values


tf_train


tf_train.shape


tf_test.shape




CLASS_NAMES = [
    "Algebra",
    "Geometry and Trigonometry",
    "Calculus and Analysis",
    "Probability and Statistics",
    "Number Theory",
    "Combinatorics and Discrete Math",
    "Linear Algebra",
    "Abstract Algebra and Topology"
]


NUM_FOLDS = 5
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

# Initialize arrays
oof_preds = np.zeros(tf_train.shape[0], dtype=int)
test_preds = np.zeros((tf_test.shape[0], NUM_FOLDS), dtype=int)

optimal_logreg = LogisticRegression(
    class_weight='balanced',
    max_iter=2000,
    random_state=42
)

# Updated cross-validation loop
for fold, (trn_idx, val_idx) in enumerate(skf.split(tf_train, y_train)):
    print(f"\nFold {fold+1}/{NUM_FOLDS}")
    
    # Split data
    X_trn, X_val = tf_train[trn_idx], tf_train[val_idx]
    y_trn, y_val = y_train[trn_idx], y_train[val_idx]
    
    model = optimal_logreg #make_pipeline(optimal_logreg)
    
    # Fit model
    model.fit(X_trn, y_trn)
    
    # Validation predictions
    y_val_pred = model.predict(X_val)
    oof_preds[val_idx] = y_val_pred
    
    # Test predictions stacking
    test_preds[:, fold] = model.predict(tf_test)
    
    # Fold metrics
    print(classification_report(y_val, y_val_pred, target_names=CLASS_NAMES))
    fold_f1 = f1_score(y_val, y_val_pred, average="micro")
    print(f"Fold {fold+1} F1 (micro): {fold_f1:.4f}")



# Ensemble predictions (majority vote)
final_test_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=test_preds)

# OOF evaluation
oof_f1 = f1_score(y_train, oof_preds, average="micro")
print("\nOverall OOF Metrics:")
print(classification_report(y_train, oof_preds, target_names=CLASS_NAMES))
print(f"Overall OOF F1 (micro): {oof_f1:.4f}")

submission = pd.DataFrame({"id": test["id"], "label": final_test_preds})
submission.to_csv("submission.csv", index=False)
print("\nSubmission saved to submission.csv")



submission

