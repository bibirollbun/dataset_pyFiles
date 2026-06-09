import os
import pandas as pd
# from langdetect import detect, DetectorFactory
# from langdetect.lang_detect_exception import LangDetectException
import unicodedata
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.metrics import accuracy_score
import numpy as np


def read_texts_from_dir(dir_path):
  """
  Reads the texts from a given directory and saves them in the pd.DataFrame with columns ['id', 'file_1', 'file_2'].

  Params:
    dir_path (str): path to the directory with data
  """
  # Count number of directories in the provided path
  dir_count = sum(os.path.isdir(os.path.join(root, d)) for root, dirs, _ in os.walk(dir_path) for d in dirs)
  data=[0 for _ in range(dir_count)]
  print(f"Number of directories: {dir_count}")

  # For each directory, read both file_1.txt and file_2.txt and save results to the list
  i=0
  for folder_name in sorted(os.listdir(dir_path)):
    folder_path = os.path.join(dir_path, folder_name)
    if os.path.isdir(folder_path):
      try:
        with open(os.path.join(folder_path, 'file_1.txt'), 'r', encoding='utf-8') as f1:
          text1 = f1.read().strip()
        with open(os.path.join(folder_path, 'file_2.txt'), 'r', encoding='utf-8') as f2:
          text2 = f2.read().strip()
        index = int(folder_name[-4:])
        data[i]=(index, text1, text2)
        i+=1
      except Exception as e:
        print(f"Error reading directory {folder_name}: {e}")

  # Change list with results into pandas DataFrame
  df = pd.DataFrame(data, columns=['id', 'file_1', 'file_2']).set_index('id')
  return df


# Use the above function to load both train and test data
train_path="/kaggle/input/fake-or-real-the-impostor-hunt/data/train"
df_train=read_texts_from_dir(train_path)
test_path="/kaggle/input/fake-or-real-the-impostor-hunt/data/test"
df_test=read_texts_from_dir(test_path)
df_train_gt=pd.read_csv("/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv")
df_train_gt.head(2)


df = df_train.merge(df_train_gt, on="id", how="left")
df['real_text_id'] = df['real_text_id'].astype(int)
print(df.head())


def get_real(row):
    return row['file_1'] if row['real_text_id'] == 1 else row['file_2']
def get_fake(row):
    return row['file_2'] if row['real_text_id'] == 1 else row['file_1']

df['real_text'] = df.apply(get_real, axis=1)
df['fake_text'] = df.apply(get_fake, axis=1)

df[['id', 'real_text', 'fake_text']].head()



# Calculate character lengths
df['real_len'] = df['real_text'].str.len()
df['fake_len'] = df['fake_text'].str.len()
length_diffs = df['real_len'] - df['fake_len']

# Side-by-side subplot for length distributions
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Real vs. Fake Text Lengths
axs[0].hist(df['real_len'], bins=30, color='#6A5ACD', alpha=0.65, label='Real Text', density=True, edgecolor='black')
axs[0].hist(df['fake_len'], bins=30, color='#FFCC66', alpha=0.65, label='Fake Text', density=True, edgecolor='black')
axs[0].set_title('Text Length Distribution', fontsize=13)
axs[0].set_xlabel('Character Count')
axs[0].set_ylabel('Density')
axs[0].legend(frameon=False)

# Plot 2: Length Difference (Real - Fake)
axs[1].hist(length_diffs, bins=25, color='#8B5CF6', alpha=0.75, edgecolor='black')
axs[1].set_title('Length Difference (Real - Fake)', fontsize=13)
axs[1].set_xlabel('Length Difference')
axs[1].set_ylabel('Count')

plt.tight_layout()
plt.show()




# Calculate token counts and average word length
df['real_token_count'] = df['real_text'].str.split().apply(len)
df['fake_token_count'] = df['fake_text'].str.split().apply(len)

df['real_avg_wordlen'] = df['real_text'].apply(lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0)
df['fake_avg_wordlen'] = df['fake_text'].apply(lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Token count distribution
axs[0].hist(df['real_token_count'], bins=30, color='#6A5ACD', alpha=0.65, label='Real Text', density=True, edgecolor='black')
axs[0].hist(df['fake_token_count'], bins=30, color='#FFCC66', alpha=0.65, label='Fake Text', density=True, edgecolor='black')
axs[0].set_title('Token Count Distribution', fontsize=13)
axs[0].set_xlabel('Token Count')
axs[0].set_ylabel('Density')
axs[0].legend(frameon=False)

# Average word length distribution
axs[1].hist(df['real_avg_wordlen'], bins=25, color='#6A5ACD', alpha=0.65, label='Real Text', density=True, edgecolor='black')
axs[1].hist(df['fake_avg_wordlen'], bins=25, color='#FFCC66', alpha=0.65, label='Fake Text', density=True, edgecolor='black')
axs[1].set_title('Average Word Length Distribution', fontsize=13)
axs[1].set_xlabel('Average Word Length')
axs[1].set_ylabel('Density')
axs[1].legend(frameon=False)

plt.tight_layout()
plt.show()



import re

def latin_char_ratio(text):
    if not text or len(text) == 0:
        return 0
    latin_count = len(re.findall(r'[a-zA-Z]', text))
    return latin_count / len(text)

df['real_latin_ratio'] = df['real_text'].apply(latin_char_ratio)
df['fake_latin_ratio'] = df['fake_text'].apply(latin_char_ratio)

fig, ax = plt.subplots(figsize=(7,4))
ax.hist(df['real_latin_ratio'], bins=30, alpha=0.65, color='#6A5ACD', label='Real Text', edgecolor='black', density=True)
ax.hist(df['fake_latin_ratio'], bins=30, alpha=0.65, color='#FFCC66', label='Fake Text', edgecolor='black', density=True)
ax.set_title('Distribution of Latin Character Ratio', fontsize=14)
ax.set_xlabel('Proportion of Latin Characters')
ax.set_ylabel('Density')
ax.legend(frameon=False)
plt.tight_layout()
plt.show()



import string

# Calculate punctuation density for real and fake texts
def punct_density(text):
    punct_count = sum(1 for c in text if c in string.punctuation)
    return punct_count / max(1, len(text))

df['real_punct_density'] = df['real_text'].apply(punct_density)
df['fake_punct_density'] = df['fake_text'].apply(punct_density)

# Plot
fig, ax = plt.subplots(figsize=(7,4))
ax.hist(df['real_punct_density'], bins=25, color='#6A5ACD', alpha=0.65, label='Real Text', density=True, edgecolor='black')
ax.hist(df['fake_punct_density'], bins=25, color='#FFCC66', alpha=0.65, label='Fake Text', density=True, edgecolor='black')
ax.set_title('Distribution of Punctuation Density', fontsize=16)
ax.set_xlabel('Punctuation Marks per Character')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.show()



from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def stopword_ratio(text):
    tokens = text.split()
    if not tokens:
        return 0
    return sum(1 for w in tokens if w.lower() in stop_words) / len(tokens)

df['real_stopword_prop'] = df['real_text'].apply(stopword_ratio)
df['fake_stopword_prop'] = df['fake_text'].apply(stopword_ratio)

def lexical_diversity(text):
    tokens = text.split()
    return len(set(tokens)) / (len(tokens) + 1e-9)  # avoid division by zero

df['real_lexical_div'] = df['real_text'].apply(lexical_diversity)
df['fake_lexical_div'] = df['fake_text'].apply(lexical_diversity)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
# Stopword Proportion
axes[0].hist(df['real_stopword_prop'], bins=20, color='#6A5ACD', alpha=0.6, label='Real Text', density=True, edgecolor='black')
axes[0].hist(df['fake_stopword_prop'], bins=20, color='#FFECB3', alpha=0.7, label='Fake Text', density=True, edgecolor='black')
axes[0].set_title('Distribution of Stopword Proportion', fontsize=16)
axes[0].set_xlabel('Stopwords per Token')
axes[0].set_ylabel('Density')
axes[0].legend()

# Lexical Diversity
axes[1].hist(df['real_lexical_div'], bins=20, color='#6A5ACD', alpha=0.6, label='Real Text', density=True, edgecolor='black')
axes[1].hist(df['fake_lexical_div'], bins=20, color='#FFECB3', alpha=0.7, label='Fake Text', density=True, edgecolor='black')
axes[1].set_title('Lexical Diversity (Unique Words per Token)', fontsize=16)
axes[1].set_xlabel('Lexical Diversity')
axes[1].set_ylabel('Density')
axes[1].legend()

plt.tight_layout()
plt.show()



# Helper functions
def get_digit_ratio(text):
    return sum(c.isdigit() for c in text) / max(len(text), 1)

def get_upper_ratio(text):
    return sum(c.isupper() for c in text) / max(len(text), 1)

def avg_word_length(text):
    words = re.findall(r'\b\w+\b', text)
    return np.mean([len(w) for w in words]) if words else 0

def vocab_size(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return len(set(words))

def punct_ratio(text):
    return sum(1 for c in text if c in '.,;:!?') / max(len(text), 1)

def rare_word_ratio(text, common_words):
    words = re.findall(r'\b\w+\b', text.lower())
    rare = [w for w in words if w not in common_words]
    return len(rare) / max(len(words), 1) if words else 0

def sent_len_std(text):
    sentences = re.split(r'[.!?]', text)
    lens = [len(s.split()) for s in sentences if s.strip()]
    return np.std(lens) if lens else 0

def repetition_score(text, n=3):
    words = text.lower().split()
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    counts = {}
    for ng in ngrams:
        counts[ng] = counts.get(ng, 0) + 1
    repeated = sum(1 for v in counts.values() if v > 1)
    return repeated / max(len(ngrams), 1) if ngrams else 0

def generic_phrase_count(text):
    generic_phrases = ['in conclusion', 'it is well known', 'in summary', 'to sum up']
    return sum(phrase in text.lower() for phrase in generic_phrases)

# A simple "common words" list for rare word calculation
common_words = set("""
the be to of and a in that have i it for not on with he as you do at this
but his by from they we say her she or an will my one all would there their
what so up out if about who get which go me when make can like time no just him
know take people into year your good some could them see other than then now look
only come its over think also back after use two how our work first well way even
new want because any these give day most us
""".split())

# Main feature extraction
def get_features(row):
    feats = {}
    t1, t2 = row['file_1'], row['file_2']

    # Length difference
    feats['len_diff'] = len(t1) - len(t2)
    # Sentence count difference
    feats['sent_diff'] = t1.count('.') - t2.count('.')
    # Digit ratio difference
    feats['digit_diff'] = get_digit_ratio(t1) - get_digit_ratio(t2)
    # Uppercase ratio difference
    feats['upper_diff'] = get_upper_ratio(t1) - get_upper_ratio(t2)
    # Average word length diff
    feats['avg_wordlen_diff'] = avg_word_length(t1) - avg_word_length(t2)
    # Vocabulary size diff
    feats['vocab_diff'] = vocab_size(t1) - vocab_size(t2)
    # Punctuation ratio diff
    feats['punct_diff'] = punct_ratio(t1) - punct_ratio(t2)
    # Rare word ratio diff
    feats['rare_word_ratio_diff'] = rare_word_ratio(t1, common_words) - rare_word_ratio(t2, common_words)
    # Sentence length std diff
    feats['sentlen_std_diff'] = sent_len_std(t1) - sent_len_std(t2)
    # Trigram repetition diff
    feats['trigram_repetition_diff'] = repetition_score(t1, n=3) - repetition_score(t2, n=3)
    # Generic phrase count diff
    feats['generic_phrase_diff'] = generic_phrase_count(t1) - generic_phrase_count(t2)

    return pd.Series(feats)

# Now extract features and target
X = df.apply(get_features, axis=1)
y = (df['real_text_id'] == 1).astype(int)  # 1 if file_1 is real, else 0



df


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Split into train and validation sets for a quick sanity check
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit interpretable logistic regression
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predictions and evaluation
y_pred = clf.predict(X_val)
print(classification_report(y_val, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

# Show feature importances
feature_importance = pd.Series(clf.coef_[0], index=X.columns)
feature_importance.sort_values(ascending=False).plot(kind='barh', figsize=(7,5), color='#6A5ACD')
plt.title("Feature Importance (Logistic Regression Weights)")
plt.xlabel("Weight")
plt.tight_layout()
plt.show()



# Extract features for the test set
X_test = df_test.apply(get_features, axis=1)
# Predict: 1 if file_1 is real, 0 if file_2 is real
test_preds = clf.predict(X_test)
# df_submission = pd.DataFrame({
#     'id': df_test.reset_index()['id'],
#     'real_text_id': [1 if pred == 1 else 2 for pred in test_preds]
# })
# df_submission.to_csv('submission.csv', index=False)


df_test.to_csv("test_set.csv")


df_test




