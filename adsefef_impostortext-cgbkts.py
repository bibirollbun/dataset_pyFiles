# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

max_files_to_show = 10  # Show only the first 10 file paths

file_counter = 0
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        file_counter += 1
        if file_counter >= max_files_to_show:
            print(f"...and {file_counter}+ more files hidden")
            break
    if file_counter >= max_files_to_show:
        break

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


def read_texts_from_dir(dir_path):
    data = []
    for folder_name in sorted(os.listdir(dir_path)):
        folder_path = os.path.join(dir_path, folder_name)
        if os.path.isdir(folder_path):
            try:
                with open(os.path.join(folder_path, 'file_1.txt'), 'r', encoding='utf-8') as f1:
                    text1 = f1.read().strip()
                with open(os.path.join(folder_path, 'file_2.txt'), 'r', encoding='utf-8') as f2:
                    text2 = f2.read().strip()
                # extract numeric ID from folder name like 'article_0090' -> 90
                index = int(folder_name.split('_')[-1])
                data.append((index, text1, text2))
            except Exception as e:
                print(f"Error reading {folder_name}: {e}")
    return pd.DataFrame(data, columns=['id', 'file_1', 'file_2']).set_index('id')



train_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train"
df_train = read_texts_from_dir(train_path)

label_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv"
df_train_gt = pd.read_csv(label_path)



df_train = df_train.reset_index()
df = df_train.merge(df_train_gt, on='id', how='left')

def split_real_fake(row):
    if row['real_text_id'] == 1:
        return pd.Series([row['file_1'], row['file_2']])
    else:
        return pd.Series([row['file_2'], row['file_1']])

df[['real_text', 'fake_text']] = df.apply(split_real_fake, axis=1)
df_real_fake = df[['id', 'real_text', 'fake_text']]



df_real_fake.to_csv("/kaggle/working/real_vs_fake_texts.csv", index=False)



# âœ… 1. Install langdetect (run this once at the top of the notebook)
!pip install langdetect --quiet



# âœ… 2. Import libraries
import pandas as pd
import unicodedata
import string
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 42
# âœ… 3. Define helper functions
def english_ratio(text, n=10):
    text = str(text).translate(str.maketrans('', '', string.punctuation + '\n'))
    chunks = [' '.join(text.split()[i:i+n]) for i in range(0, len(text.split()), n)]
    if not chunks:
        return 0
    english_count = 0
    for chunk in chunks:
        try:
            if detect(chunk) == 'en':
                english_count += 1
        except LangDetectException:
            continue
    return english_count / len(chunks)

def latin_ratio(text):
    text = str(text).translate(str.maketrans('', '', string.punctuation + '\n'))
    non_space_chars = [c for c in text if c != ' ']
    if not non_space_chars:
        return 0
    latin_chars = [c for c in non_space_chars if 'LATIN' in unicodedata.name(c, '')]
    return len(latin_chars) / len(non_space_chars)



# âœ… 4. Load your CSV
df = pd.read_csv("/kaggle/working/real_vs_fake_texts.csv")  # adjust path if needed
df = df.dropna(subset=['real_text', 'fake_text'])  # remove incomplete rows



# âœ… 5. Apply ratios to both real and fake text
df['real_english_ratio'] = df['real_text'].apply(english_ratio)
df['fake_english_ratio'] = df['fake_text'].apply(english_ratio)
df['real_latin_ratio'] = df['real_text'].apply(latin_ratio)
df['fake_latin_ratio'] = df['fake_text'].apply(latin_ratio)



# âœ… 6. Save to CSV for review
df_output = df[['id', 'real_text', 'fake_text', 'real_english_ratio', 'real_latin_ratio', 'fake_english_ratio', 'fake_latin_ratio']]
df_output.to_csv("/kaggle/working/real_vs_fake_ratios.csv", index=False)



df['english_ratio_diff'] = df['real_english_ratio'] - df['fake_english_ratio']
df['latin_ratio_diff'] = df['real_latin_ratio'] - df['fake_latin_ratio']



import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(df['real_english_ratio'], df['fake_english_ratio'], alpha=0.7)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("Real English Ratio")
plt.ylabel("Fake English Ratio")
plt.title("Real vs Fake English Ratios")
plt.grid(True)
plt.show()



plt.figure(figsize=(10, 4))
plt.hist(df['english_ratio_diff'], bins=20, alpha=0.7, label='English Ratio Diff (Real - Fake)')
plt.hist(df['latin_ratio_diff'], bins=20, alpha=0.7, label='Latin Ratio Diff (Real - Fake)')
plt.axvline(0, color='black', linestyle='--')
plt.legend()
plt.title("Difference Distributions: Real - Fake")
plt.xlabel("Difference Value")
plt.grid(True)
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

melted = pd.melt(df, 
                 id_vars='id', 
                 value_vars=['real_english_ratio', 'fake_english_ratio'], 
                 var_name='type', 
                 value_name='english_ratio')
melted['label'] = melted['type'].str.extract('(real|fake)')

plt.figure(figsize=(10, 5))
sns.histplot(data=melted, x='english_ratio', hue='label', bins=20, kde=True, element="step")
plt.axvline(0.7, color='red', linestyle='--')  # example threshold
plt.title('English Ratio Distribution: Real vs Fake')
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Melt into long format
latin_melted = pd.melt(
    df,
    id_vars='id',
    value_vars=['real_latin_ratio', 'fake_latin_ratio'],
    var_name='type',
    value_name='latin_ratio'
)
latin_melted['label'] = latin_melted['type'].str.extract(r'(real|fake)')

# Plot
plt.figure(figsize=(10, 5))
sns.histplot(data=latin_melted, x='latin_ratio', hue='label', bins=20, kde=True, element="step")
plt.axvline(0.95, color='red', linestyle='--')  # example threshold line
plt.title('Latin Ratio Distribution: Real vs Fake')
plt.xlabel('latin_ratio')
plt.grid(True)
plt.show()



# Summary for both real and fake texts
summary_stats = {
    "Real English Ratio": {
        "min": df['real_english_ratio'].min(),
        "mean": df['real_english_ratio'].mean(),
        "max": df['real_english_ratio'].max(),
        "10th pct": df['real_english_ratio'].quantile(0.1),
        "25th pct": df['real_english_ratio'].quantile(0.25),
    },
    "Fake English Ratio": {
        "min": df['fake_english_ratio'].min(),
        "mean": df['fake_english_ratio'].mean(),
        "max": df['fake_english_ratio'].max(),
        "90th pct": df['fake_english_ratio'].quantile(0.9),
        "75th pct": df['fake_english_ratio'].quantile(0.75),
    },
    "Real Latin Ratio": {
        "min": df['real_latin_ratio'].min(),
        "mean": df['real_latin_ratio'].mean(),
        "max": df['real_latin_ratio'].max(),
        "10th pct": df['real_latin_ratio'].quantile(0.1),
        "25th pct": df['real_latin_ratio'].quantile(0.25),
    },
    "Fake Latin Ratio": {
        "min": df['fake_latin_ratio'].min(),
        "mean": df['fake_latin_ratio'].mean(),
        "max": df['fake_latin_ratio'].max(),
        "90th pct": df['fake_latin_ratio'].quantile(0.9),
        "75th pct": df['fake_latin_ratio'].quantile(0.75),
    }
}

# Print clean summary
for key, stats in summary_stats.items():
    print(f"\n{key}")
    for stat, value in stats.items():
        print(f"  {stat}: {value:.4f}")



# Apply rule-based fake detection
def is_likely_fake(eng_ratio, lat_ratio, eng_thresh=0.90, lat_thresh=0.96):
    return (eng_ratio < eng_thresh) or (lat_ratio < lat_thresh)

# Predict which text is real based on the rule
predictions = []
for _, row in df.iterrows():
    real_is_fake = is_likely_fake(row['real_english_ratio'], row['real_latin_ratio'])
    fake_is_fake = is_likely_fake(row['fake_english_ratio'], row['fake_latin_ratio'])

    # If real is not fake and fake is fake => predict real_text = real
    if not real_is_fake and fake_is_fake:
        predictions.append(1)
    # If real is fake and fake is not => predict fake_text is real
    elif real_is_fake and not fake_is_fake:
        predictions.append(2)
    else:
        # Tie or unclear â€” fallback to default (e.g., assume 1)
        predictions.append(1)

# Reconstruct ground truth: 1 if real_text came from file_1, else 2
# We assume that for original data: file_1 is the one that became real_text if real_text_id == 1
gt = df['real_text'].index.to_series().map(lambda i: df_train_gt.loc[df_train_gt['id'] == i, 'real_text_id'].values[0])

# Evaluate accuracy
accuracy = (gt.values == predictions).mean()
print(f"Rule-based prediction accuracy: {accuracy:.4f}")



# Apply rule to each text
def is_likely_fake(eng_ratio, lat_ratio, eng_thresh=0.90, lat_thresh=0.96):
    return (eng_ratio < eng_thresh) or (lat_ratio < lat_thresh)

df['real_tagged_fake'] = df.apply(lambda row: is_likely_fake(row['real_english_ratio'], row['real_latin_ratio']), axis=1)
df['fake_tagged_fake'] = df.apply(lambda row: is_likely_fake(row['fake_english_ratio'], row['fake_latin_ratio']), axis=1)

# Combine into one long format
fake_check = []

for _, row in df.iterrows():
    if row['real_tagged_fake']:
        fake_check.append(False)  # Real text wrongly tagged as fake
    if row['fake_tagged_fake']:
        fake_check.append(True)   # Fake text correctly tagged as fake

# Compute precision of fake tag
if fake_check:
    fake_tag_precision = sum(fake_check) / len(fake_check) * 100
    print(f"ðŸŽ¯ Precision of fake-tagging rule: {fake_tag_precision:.2f}%")
    print(f"ðŸ”¢ Count of tagged-as-fake texts: {len(fake_check)}")
else:
    print("No texts were tagged as fake under this rule.")



import pandas as pd
import numpy as np
import re
import string
import unicodedata
from langdetect import detect, DetectorFactory, LangDetectException
from nltk.corpus import stopwords

DetectorFactory.seed = 42
stop_words = set(stopwords.words("english"))

# --- 1. Feature extraction function ---
def extract_features(text):
    if pd.isna(text) or not str(text).strip():
        return {
            'is_empty': 1,
            'char_count': 0,
            'word_count': 0,
            'avg_word_len': 0,
            'unique_token_ratio': 0,
            'repeat_count': 0,
            'english_ratio': 0,
            'latin_ratio': 0,
            'punct_ratio': 0,
            'space_ratio': 0,
            'stopword_ratio': 0,
            'sentence_count': 0,
            'avg_sentence_len': 0,
            'count_double_hash': 0,
            'count_double_star': 0
        }

    text = str(text)
    tokens = text.split()
    words = [w.strip(string.punctuation) for w in tokens if w.strip()]
    unique_tokens = set(words)

    try:
        chunked = [' '.join(words[i:i+10]) for i in range(0, len(words), 10)]
        eng_ratio = sum(detect(chunk) == 'en' for chunk in chunked) / len(chunked)
    except:
        eng_ratio = 0

    non_space_chars = [c for c in text if c != ' ']
    latin_chars = [c for c in non_space_chars if 'LATIN' in unicodedata.name(c, '')]

    sentence_split = re.split(r'[.!?]+', text)
    sentence_lengths = [len(s.split()) for s in sentence_split if s.strip()]

    stopword_count = sum(w.lower() in stop_words for w in words)

    return {
        'is_empty': 0,
        'char_count': len(text),
        'word_count': len(words),
        'avg_word_len': np.mean([len(w) for w in words]) if words else 0,
        'unique_token_ratio': len(unique_tokens) / len(words) if words else 0,
        'repeat_count': len(words) - len(unique_tokens),
        'english_ratio': eng_ratio,
        'latin_ratio': len(latin_chars) / len(non_space_chars) if non_space_chars else 0,
        'punct_ratio': sum(c in string.punctuation for c in text) / len(text),
        'space_ratio': text.count(' ') / len(text),
        'stopword_ratio': stopword_count / len(words) if words else 0,
        'sentence_count': len(sentence_lengths),
        'avg_sentence_len': np.mean(sentence_lengths) if sentence_lengths else 0,
        'count_double_hash': text.count('##'),
        'count_double_star': text.count('**')
    }

# --- 2. Apply to train data ---
features = []

for _, row in df.iterrows():
    real_feats = extract_features(row['real_text'])
    fake_feats = extract_features(row['fake_text'])

    combined = {f'real_{k}': v for k, v in real_feats.items()}
    combined.update({f'fake_{k}': v for k, v in fake_feats.items()})

    # Label: 1 if real_text is file_1, else 0
    real_text_id = df_train_gt[df_train_gt['id'] == row['id']]['real_text_id'].values[0]
    combined['label'] = 1 if real_text_id == 1 else 0

    features.append(combined)

# --- 3. Create train dataframe ---
df_features = pd.DataFrame(features)
X_train = df_features.drop('label', axis=1)
y_train = df_features['label']



from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# --- 1. Split data for quick validation ---
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# --- 2. Train XGBoost model ---
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train_split, y_train_split)

# --- 3. Predict & evaluate ---
y_pred = model.predict(X_val_split)
print("ðŸ§  Classification Report:")
print(classification_report(y_val_split, y_pred))
print(f"ðŸŽ¯ Accuracy: {accuracy_score(y_val_split, y_pred):.4f}")



# --- 4. Plot feature importance ---
plt.figure(figsize=(10, 8))
xgb_importance = model.feature_importances_
sorted_idx = np.argsort(xgb_importance)[::-1]
top_n = 15

plt.barh(range(top_n), xgb_importance[sorted_idx][:top_n][::-1])
plt.yticks(range(top_n), X_train.columns[sorted_idx][:top_n][::-1])
plt.xlabel("Feature Importance")
plt.title("Top 15 Features (XGBoost)")
plt.tight_layout()
plt.show()



import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import os
from langdetect import detect, DetectorFactory, LangDetectException
import unicodedata
import string
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Setup
DetectorFactory.seed = 42
stop_words = set(stopwords.words("english"))

# === 1. Feature extraction function ===
def extract_features(text):
    if pd.isna(text) or not str(text).strip():
        return {'is_empty': 1, 'char_count': 0, 'word_count': 0, 'avg_word_len': 0, 'unique_token_ratio': 0,
                'repeat_count': 0, 'english_ratio': 0, 'latin_ratio': 0, 'punct_ratio': 0, 'space_ratio': 0,
                'stopword_ratio': 0, 'sentence_count': 0, 'avg_sentence_len': 0, 'count_double_hash': 0,
                'count_double_star': 0}

    text = str(text)
    tokens = text.split()
    words = [w.strip(string.punctuation) for w in tokens if w.strip()]
    unique_tokens = set(words)

    try:
        chunked = [' '.join(words[i:i+10]) for i in range(0, len(words), 10)]
        eng_ratio = sum(detect(chunk) == 'en' for chunk in chunked) / len(chunked)
    except:
        eng_ratio = 0

    non_space_chars = [c for c in text if c != ' ']
    latin_chars = [c for c in non_space_chars if 'LATIN' in unicodedata.name(c, '')]
    sentence_split = re.split(r'[.!?]+', text)
    sentence_lengths = [len(s.split()) for s in sentence_split if s.strip()]
    stopword_count = sum(w.lower() in stop_words for w in words)

    return {
        'is_empty': 0,
        'char_count': len(text),
        'word_count': len(words),
        'avg_word_len': np.mean([len(w) for w in words]) if words else 0,
        'unique_token_ratio': len(unique_tokens) / len(words) if words else 0,
        'repeat_count': len(words) - len(unique_tokens),
        'english_ratio': eng_ratio,
        'latin_ratio': len(latin_chars) / len(non_space_chars) if non_space_chars else 0,
        'punct_ratio': sum(c in string.punctuation for c in text) / len(text),
        'space_ratio': text.count(' ') / len(text),
        'stopword_ratio': stopword_count / len(words) if words else 0,
        'sentence_count': len(sentence_lengths),
        'avg_sentence_len': np.mean(sentence_lengths) if sentence_lengths else 0,
        'count_double_hash': text.count('##'),
        'count_double_star': text.count('**')
    }

# === 2. Flatten train data ===
rows = []
for _, row in df.iterrows():
    real_feats = extract_features(row['real_text'])
    real_feats['label'] = 1
    fake_feats = extract_features(row['fake_text'])
    fake_feats['label'] = 0
    rows.extend([real_feats, fake_feats])

df_flat = pd.DataFrame(rows)
X = df_flat.drop(columns=['label'])
y = df_flat['label']

# === 3. Train pointwise model ===
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X, y)

print("âœ… Trained model")
print(classification_report(y, model.predict(X)))



import glob

# === 1. Get test article paths ===
test_dir = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"
test_articles = sorted(os.listdir(test_dir))

submission_rows = []

for article in test_articles:
    path_1 = os.path.join(test_dir, article, "file_1.txt")
    path_2 = os.path.join(test_dir, article, "file_2.txt")

    try:
        with open(path_1, 'r', encoding='utf-8') as f1:
            text1 = f1.read()
    except:
        text1 = ""

    try:
        with open(path_2, 'r', encoding='utf-8') as f2:
            text2 = f2.read()
    except:
        text2 = ""

    f1_feats = extract_features(text1)
    f2_feats = extract_features(text2)

    df_test_pair = pd.DataFrame([f1_feats, f2_feats])
    probs = model.predict_proba(df_test_pair)[:, 1]  # prob of being real

    predicted_real = 1 if probs[0] > probs[1] else 2
    submission_rows.append({
        'id': int(article.split('_')[1]),
        'real_text_id': predicted_real
    })

# === 2. Save submission ===
submission_df = pd.DataFrame(submission_rows).sort_values(by='id')
submission_df.to_csv("submission.csv", index=False)
print("âœ… submission.csv saved")


