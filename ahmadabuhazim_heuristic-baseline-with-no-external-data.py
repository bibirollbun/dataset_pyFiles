# ==============================
#  Fake or Real: Feature-Based Heuristic Model (No External Dataset)
# ==============================

#  1. Import Libraries
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

#  2. Load Test Data from Competition Directory
def load_test_pairs(test_dir):
    pairs = []
    ids = []
    
    for article_dir in tqdm(os.listdir(test_dir)):
        if not article_dir.startswith("article_"):
            continue

        id = int(article_dir.split("_")[1])
        dir_path = os.path.join(test_dir, article_dir)

        with open(os.path.join(dir_path, "file_1.txt"), "r", encoding="utf-8", errors="ignore") as f:
            text1 = f.read()
        with open(os.path.join(dir_path, "file_2.txt"), "r", encoding="utf-8", errors="ignore") as f:
            text2 = f.read()

        pairs.append((text1, text2))
        ids.append(id)

    return pairs, ids

# Load test text pairs and their IDs
test_dir = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"
test_pairs, test_ids = load_test_pairs(test_dir)

#  3. Feature-Based Heuristic Functions

def latin_ratio(text):
    latin_chars = sum(1 for c in text if re.match(r"[A-Za-z]", c))
    return latin_chars / len(text) if text else 0

def avg_sentence_length(text):
    sentences = re.split(r'[.!?]', text)
    lengths = [len(s.split()) for s in sentences if len(s.split()) > 0]
    return np.mean(lengths) if lengths else 0

def word_diversity(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return len(set(words)) / len(words) if words else 0

def cosine_sim(text1, text2, vectorizer):
    vecs = vectorizer.transform([text1, text2])
    return cosine_similarity(vecs[0], vecs[1])[0][0]

#  4. Predict Based on Feature Scores
print("Extracting features and making predictions...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
all_texts = [t for pair in test_pairs for t in pair]
vectorizer.fit(all_texts)

predictions = []
features_debug = []

for text1, text2 in tqdm(test_pairs):
    lat1 = latin_ratio(text1)
    lat2 = latin_ratio(text2)
    len1 = avg_sentence_length(text1)
    len2 = avg_sentence_length(text2)
    div1 = word_diversity(text1)
    div2 = word_diversity(text2)
    sim = cosine_sim(text1, text2, vectorizer)

    score1 = lat1 + len1 + div1
    score2 = lat2 + len2 + div2

    prediction = 1 if score1 > score2 else 2
    predictions.append(prediction)
    features_debug.append((lat1, lat2, len1, len2, div1, div2, sim))

#  5. Create Submission File
submission = pd.DataFrame({
    "id": test_ids,
    "real_text_id": predictions
}).sort_values("id")

submission.to_csv("submission.csv", index=False)
print("Submission file saved!")

#  6. Visualize Feature Differences
lat_diffs = [abs(a - b) for a, b, *_ in features_debug]
plt.hist(lat_diffs, bins=30, color='skyblue', edgecolor='black')
plt.title("Latin Character Ratio Difference")
plt.xlabel("Difference")
plt.ylabel("Number of Samples")
plt.show()




