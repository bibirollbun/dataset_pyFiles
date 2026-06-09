import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import itertools

train_essays = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/train_essays.csv')
train_prompts = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/train_prompts.csv')
test_essays = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
sample_submit = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/sample_submission.csv')


train_essays.shape, train_prompts.shape, test_essays.shape


train_essays.head(5)


train_prompts


train_prompts['source_text'].iloc[0]


train_prompts['instructions'].iloc[0]


train_prompts['instructions'].iloc[1]


train_essays['generated'].value_counts()


train_essays['generated']


train_essays['essay_length'] = train_essays['text'].apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(train_essays[train_essays['generated'] == 0]['essay_length'], 
             color="skyblue", label='Student Essays', kde=True)
sns.histplot(train_essays[train_essays['generated'] == 1]['essay_length'], 
             color="red", label='LLM Generated Essays', kde=True)

plt.title('Distribution of Essay Lengths')
plt.xlabel('Essay Length (Number of Characters)')
plt.ylabel('Frequency')
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='generated', y='essay_length', data=train_essays)
plt.title('Comparison of Essay Lengths by Source')
plt.xlabel('Essay Source')
plt.ylabel('Essay Length')
plt.xticks([0, 1], ['Student-written', 'LLM-generated'])
plt.show()


def calculate_text_metrics_simple(text):
    words = text.split()
    sentences = text.split('.')
    word_count = len(words)
    unique_word_count = len(set(words))
    sentence_count = len(sentences)
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    return word_count, unique_word_count, sentence_count, avg_word_length

train_essays['metrics'] = train_essays['text'].apply(calculate_text_metrics_simple)

train_essays[
    ['word_count', 'unique_word_count', 'sentence_count', 'avg_word_length']
] = pd.DataFrame(train_essays['metrics'].tolist(), index=train_essays.index)

train_essays.drop('metrics', axis=1, inplace=True)

comparison_metrics = train_essays.groupby('generated')[
    ['word_count', 'unique_word_count', 'sentence_count', 'avg_word_length']
].mean()

comparison_metrics


def plot_most_common_words(text_series, num_words=30, title="Most Common Words"):
    all_text = ' '.join(text_series).lower()
    words = all_text.split()
    word_freq = Counter(words)
    common_words = word_freq.most_common(num_words)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=[word for word, freq in common_words], y=[freq for word, freq in common_words])
    plt.title(title)
    plt.xticks(rotation=45)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.show()

plot_most_common_words(train_essays[train_essays['generated'] == 0]['text'], title="Most Common Words in Student Essays")
plot_most_common_words(train_essays[train_essays['generated'] == 1]['text'], title="Most Common Words in LLM-generated Essays")


import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast
from datasets import Dataset
from tqdm.auto import tqdm

# Load the training set (larger-scale daigt-v2) and the test set
train = pd.read_csv("/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv", sep=',')
test  = test_essays.copy()
sub   = sample_submit.copy()

print("Available columns in train:", train.columns.tolist())

gc.collect()


# Dynamically select the label column & clean up unnecessary objects
candidates = ['generated','label','target','drcat']
found = [c for c in candidates if c in train.columns]

if not found:
    raise KeyError(f"No label column among {candidates}, please check train.columns")

target_col = found[0]
print(f"Using '{target_col}' as label column.")

y_train = train[target_col].values

gc.collect()



# Build and train the BPE Tokenizer
LOWERCASE = False
VOCAB_SIZE = 30522

raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
raw_tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFC()] + ([normalizers.Lowercase()] if LOWERCASE else [])
)
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)

hf_test = Dataset.from_pandas(test[['text']])
def test_corpus_iter():
    for i in range(0, len(hf_test), 1000):
        yield hf_test[i : i + 1000]["text"]

raw_tokenizer.train_from_iterator(test_corpus_iter(), trainer=trainer)

tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="[UNK]", pad_token="[PAD]",
    cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]"
)


# Tokenize & TF-IDF
def dummy(tokens):
    return tokens

tokenized_train = [tokenizer.tokenize(txt) for txt in tqdm(train['text'], desc="Tokenizing Train")]
tokenized_test  = [tokenizer.tokenize(txt) for txt in tqdm(test['text'],  desc="Tokenizing Test")]

vectorizer = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy,
    preprocessor=dummy,
    token_pattern=None,
    ngram_range=(3,5),
    lowercase=False,
    sublinear_tf=True,
    strip_accents='unicode'
)
vectorizer.fit(tokenized_test)
vocab = vectorizer.vocabulary_

vectorizer = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy,
    preprocessor=dummy,
    token_pattern=None,
    ngram_range=(3,5),
    lowercase=False,
    sublinear_tf=True,
    strip_accents='unicode',
    vocabulary=vocab
)
tf_train = vectorizer.fit_transform(tokenized_train)
tf_test  = vectorizer.transform(tokenized_test)

del vectorizer
gc.collect()


# 5-fold cross-validation evaluation & OOF prediction
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(tf_train, y_train), 1):
    X_tr, X_val = tf_train[tr_idx], tf_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    clf_sgd = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber", random_state=42)
    clf_nb  = MultinomialNB(alpha=0.02)
    ensemble = VotingClassifier(
        estimators=[('sgd', clf_sgd), ('nb', clf_nb)],
        voting='soft', weights=[0.7, 0.3], n_jobs=-1
    )
    ensemble.fit(X_tr, y_tr)

    oof_preds[val_idx] = ensemble.predict_proba(X_val)[:,1]
    test_preds       += ensemble.predict_proba(tf_test)[:,1] / n_splits

    print(f"Fold {fold} AUC: {roc_auc_score(y_val, oof_preds[val_idx]):.4f}")

print(f"Overall CV AUC: {roc_auc_score(y_train, oof_preds):.4f}")



# Full training & generation of submission files
final_ensemble = VotingClassifier(
    estimators=[
        ('sgd', SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber", random_state=42)),
        ('nb', MultinomialNB(alpha=0.02))
    ],
    voting='soft', weights=[0.7, 0.3], n_jobs=-1
)
final_ensemble.fit(tf_train, y_train)

sub['generated'] = final_ensemble.predict_proba(tf_test)[:,1]
sub.to_csv('submission.csv', index=False)




