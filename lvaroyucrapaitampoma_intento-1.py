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

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ==========================
# 1. Imports
# ==========================
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

# ==========================
# 2. Paths
# ==========================
BASE_DIR = "/kaggle/input/fake-or-real-the-impostor-hunt/data"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

print("Train CSV:", TRAIN_CSV)
print("Train DIR:", TRAIN_DIR)
print("Test DIR :", TEST_DIR)

# ==========================
# 3. Load CSV
# ==========================
train_df = pd.read_csv(TRAIN_CSV)
print("Train shape:", train_df.shape)
print(train_df.head())

# ==========================
# 4. Load training articles
# ==========================
def load_article_text(article_id, file_id, folder=TRAIN_DIR):
    file_path = os.path.join(folder, f"article_{article_id:04d}", f"file_{file_id}.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

pairs = []
for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
    art_id = row["id"]
    real_id = row["real_text_id"]

    text1 = load_article_text(art_id, 1)
    text2 = load_article_text(art_id, 2)

    pairs.append((art_id, text1, text2, real_id))

train_texts = pd.DataFrame(pairs, columns=["id", "text1", "text2", "real_text_id"])
print("Loaded training pairs:", train_texts.shape[0])

# ==========================
# 5. TF-IDF + Similarity Features
# ==========================
all_texts = list(train_texts["text1"]) + list(train_texts["text2"])

vectorizer = TfidfVectorizer(max_features=20000)
tfidf = vectorizer.fit_transform(all_texts)

X1 = tfidf[:len(train_texts)]
X2 = tfidf[len(train_texts):]

cos_sims = []
len_diffs = []

for i in range(len(train_texts)):
    cos_sim = cosine_similarity(X1[i], X2[i])[0][0]
    len_diff = abs(len(train_texts.loc[i, "text1"]) - len(train_texts.loc[i, "text2"]))
    cos_sims.append(cos_sim)
    len_diffs.append(len_diff)

features = pd.DataFrame({
    "cos_sim": cos_sims,
    "len_diff": len_diffs
})
print(features.head())

# ==========================
# 6. Labels
# ==========================
labels = (train_texts["real_text_id"] == 1).astype(int)

# ==========================
# 7. Train / Validation split
# ==========================
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

# ==========================
# 8. Train LightGBM
# ==========================
params = {
    "objective": "binary",
    "metric": "binary_error",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "seed": 42,
}

print("Training LightGBM...")
gbm = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_val],
    num_boost_round=500,
    callbacks=[early_stopping(50), log_evaluation(50)]
)

# ==========================
# 9. Predict Test Set
# ==========================
print("Preparing test set...")

test_articles = sorted(os.listdir(TEST_DIR))
predictions = []

for art in tqdm(test_articles):
    art_id = int(art.split("_")[1])

    t1 = load_article_text(art_id, 1, folder=TEST_DIR)
    t2 = load_article_text(art_id, 2, folder=TEST_DIR)

    tfidf_test = vectorizer.transform([t1, t2])

    cos_sim = cosine_similarity(tfidf_test[0], tfidf_test[1])[0][0]
    len_diff = abs(len(t1) - len(t2))

    feat = pd.DataFrame([[cos_sim, len_diff]], columns=["cos_sim", "len_diff"])
    prob = gbm.predict(feat)[0]

    pred = 1 if prob > 0.5 else 2
    predictions.append((art_id, pred))

submission = pd.DataFrame(predictions, columns=["id", "real_text_id"])
print(submission.head())

# ==========================
# 10. Save Submission
# ==========================
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv saved:", submission.shape)



# ==========================
# 10. Save Submission
# ==========================
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv saved:", submission.shape)

