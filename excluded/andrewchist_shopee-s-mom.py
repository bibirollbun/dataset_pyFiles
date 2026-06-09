# import torch
# from torchvision import models

# model = models.resnet50(pretrained=True)
# torch.save(model.state_dict(), "resnet50.pth")

# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer('all-MiniLM-L6-v2')

# model.save('all-MiniLM-L6-v2-local')


import os
import re
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
from torchvision import models, transforms
from catboost import CatBoostClassifier
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
TMP_DIR = "/tmp/shopee_embeddings"
os.makedirs(TMP_DIR, exist_ok=True)
MAX_MATCHES = 50

train = pd.read_csv("/kaggle/input/shopee-product-matching/train.csv")
test = pd.read_csv("/kaggle/input/shopee-product-matching/test.csv")

TRAIN_IMG_DIR = "/kaggle/input/shopee-product-matching/train_images"
TEST_IMG_DIR = "/kaggle/input/shopee-product-matching/test_images"

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train['title_clean'] = train['title'].apply(clean_text)
test['title_clean'] = test['title'].apply(clean_text)

text_model = SentenceTransformer('/kaggle/input/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2',
                                device=DEVICE)

train_title_emb = text_model.encode(
    train['title_clean'].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
).astype(np.float32)

test_title_emb = text_model.encode(
    test['title_clean'].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
).astype(np.float32)

# vectorizer = TfidfVectorizer(max_features=1500)
# vectorizer.fit(train['title_clean'])
# train_title_emb = vectorizer.transform(train['title_clean']).toarray().astype(np.float32)
# test_title_emb  = vectorizer.transform(test['title_clean']).toarray().astype(np.float32)

model = models.resnet50(pretrained=False)
model.load_state_dict(torch.load("/kaggle/input/resnet_nn/pytorch/default/1/resnet50.pth", map_location=DEVICE))
model.fc = nn.Identity()
model = model.to(DEVICE)
model.eval()

def compute_img_embeddings(df, folder, prefix):
    embeddings = []
    for i, fname in enumerate(tqdm(df["image"], desc=f"Embedding {prefix}")):
        tmp_path = os.path.join(TMP_DIR, f"{prefix}_{i}.npy")
        img_path = os.path.join(folder, fname)
        if os.path.exists(tmp_path):
            emb = np.load(tmp_path)
        else:
            with torch.no_grad():
                img = load_image(img_path).to(DEVICE)
                emb = model(img).cpu().numpy().flatten()
                np.save(tmp_path, emb)
        embeddings.append(emb)
    return normalize(np.stack(embeddings).astype(np.float32))

train_img_emb = compute_img_embeddings(train, TRAIN_IMG_DIR, "train")
test_img_emb  = compute_img_embeddings(test, TEST_IMG_DIR, "test")

def phash_to_bits(series):
    bits = []
    for h in series:
        h_int = int(h, 16)
        b = bin(h_int)[2:].zfill(64)
        bits.append([float(x) for x in b])
    return np.array(bits, dtype=np.float32)

train_phash_emb = phash_to_bits(train["image_phash"])
test_phash_emb  = phash_to_bits(test["image_phash"])

def pair_features(i, j, img_emb, text_emb, phash_emb, titles):
    # image
    img_cos = np.dot(img_emb[i], img_emb[j])
    img_l2  = np.linalg.norm(img_emb[i] - img_emb[j])
    img_dot = np.sum(img_emb[i] * img_emb[j])

    # text
    txt_cos = np.dot(text_emb[i], text_emb[j])
    txt_l2  = np.linalg.norm(text_emb[i] - text_emb[j])
    txt_dot = np.sum(text_emb[i] * text_emb[j])

    # phash
    phash_sim = np.mean(phash_emb[i] == phash_emb[j])

    # text meta
    len_diff = abs(len(titles[i]) - len(titles[j]))

    return [img_cos, img_l2, img_dot,
            txt_cos, txt_l2, txt_dot,
            phash_sim, len_diff]

def build_pairs(df, img_emb, text_emb, phash_emb, titles, n_pos=20000, n_neg=20000):
    pairs, labels = [], []

    for gid, idxs in df.groupby("label_group").groups.items():
        idxs = list(idxs)
        for a, b in combinations(idxs, 2):
            pairs.append(pair_features(a, b, img_emb, text_emb, phash_emb, titles))
            labels.append(1)
            if len(pairs) >= n_pos: break
        if len(pairs) >= n_pos: break

    rng = np.random.default_rng(42)
    while len(labels) < n_pos + n_neg:
        a, b = rng.integers(0, len(df), size=2)
        if df.iloc[a].label_group != df.iloc[b].label_group:
            pairs.append(pair_features(a, b, img_emb, text_emb, phash_emb, titles))
            labels.append(0)

    return np.array(pairs), np.array(labels)

print("Building training pairs...")
X, y = build_pairs(train, train_img_emb, train_title_emb, train_phash_emb, train["title_clean"].tolist())

from sklearn.neighbors import NearestNeighbors

def predict_matches_safe(test_df, train_df,
                         test_img_emb, train_img_emb,
                         test_text_emb, train_text_emb,
                         test_phash_emb, train_phash_emb,
                         test_titles, train_titles,
                         max_matches=50):
    """
    Predict matching posting_ids for the test set based on train embeddings.
    """

    all_matches = []

    # Fit nearest neighbors on train embeddings
    knn = NearestNeighbors(n_neighbors=max_matches, metric="cosine")
    knn.fit(train_img_emb)

    # For each test embedding, find similar items in train
    distances, indices = knn.kneighbors(test_img_emb)

    for i in range(len(test_df)):
        candidates = indices[i]  # indices of train items

        # Compute pair features for all candidates
        feats = [pair_features(
            i_test := i,  # test index
            j_train := j,
            np.vstack([test_img_emb, train_img_emb]),
            np.vstack([test_text_emb, train_text_emb]),
            np.vstack([test_phash_emb, train_phash_emb]),
            test_titles + train_titles
        ) for j in candidates]

        probs = clf.predict_proba(feats)[:,1]
        idxs = np.argsort(-probs)[:max_matches]

        # Map to train posting_ids
        match_ids = train_df.iloc[candidates[idxs]]["posting_id"].tolist()

        # Optionally include the test item itself
        if test_df.iloc[i]["posting_id"] not in match_ids:
            match_ids.insert(0, test_df.iloc[i]["posting_id"])

        all_matches.append(" ".join(match_ids[:max_matches]))

    return all_matches

# Usage
test["matches"] = predict_matches_safe(
    test_df=test,
    train_df=train,
    test_img_emb=test_img_emb,
    train_img_emb=train_img_emb,
    test_text_emb=test_title_emb,
    train_text_emb=train_title_emb,
    test_phash_emb=test_phash_emb,
    train_phash_emb=train_phash_emb,
    test_titles=test["title_clean"].tolist(),
    train_titles=train["title_clean"].tolist(),
    max_matches=MAX_MATCHES
)

# Save submission
sub = test[["posting_id", "matches"]]
sub.to_csv("submission.csv", index=False)
print("submission.csv saved with", len(sub), "rows")



# pd.read_csv('submission.csv').head()


# import os
# import pandas as pd
# import numpy as np
# from PIL import Image
# from tqdm import tqdm

# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from sklearn.metrics.pairwise import cosine_similarity

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# IMG_SIZE = 224
# TMP_DIR = "/tmp/shopee_embeddings"
# os.makedirs(TMP_DIR, exist_ok=True)
# SIM_THRESHOLD = 0.9

# train = pd.read_csv("/kaggle/input/shopee-product-matching/train.csv")
# test = pd.read_csv("/kaggle/input/shopee-product-matching/test.csv")

# train.head()


# TRAIN_IMG_DIR = "/kaggle/input/shopee-product-matching/train_images"
# TEST_IMG_DIR = "/kaggle/input/shopee-product-matching/test_images"

# transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# ])

# def load_image(path):
#     img = Image.open(path).convert("RGB")
#     return transform(img).unsqueeze(0)

# # text
# import re

# def clean_text(text):
#     text = str(text).lower()             
#     text = re.sub(r'[^a-z0-9\s]', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# train['title_clean'] = train['title'].apply(clean_text)
# test['title_clean'] = test['title'].apply(clean_text)

# # from sentence_transformers import SentenceTransformer

# # model = SentenceTransformer('all-MiniLM-L6-v2')
# # train_title_emb = model.encode(train['title_clean'], convert_to_numpy=True, normalize_embeddings=True)
# # test_title_emb  = model.encode(test['title_clean'], convert_to_numpy=True, normalize_embeddings=True)

# from sklearn.feature_extraction.text import TfidfVectorizer

# vectorizer = TfidfVectorizer(max_features=1000)
# vectorizer.fit(train['title_clean'])

# train_title_emb = vectorizer.transform(train['title_clean']).toarray().astype(np.float32)
# test_title_emb  = vectorizer.transform(test['title_clean']).toarray().astype(np.float32)

# # imgs
# model = models.resnet50(pretrained=False)

# model.load_state_dict(torch.load("/kaggle/input/resnet_nn/pytorch/default/1/resnet50.pth", map_location=DEVICE))
# model.fc = nn.Identity()
# model = model.to(DEVICE)
# model.eval()

# def compute_img_embeddings(df, folder, prefix):
#     embeddings = []
#     for i, fname in enumerate(tqdm(df["image"], desc=f"Embedding {prefix}")):
#         tmp_path = os.path.join(TMP_DIR, f"{prefix}_{i}.npy")
#         img_path = os.path.join(folder, fname)

#         if os.path.exists(tmp_path):
#             emb = np.load(tmp_path)
#         else:
#             with torch.no_grad():
#                 img = load_image(img_path).to(DEVICE)
#                 emb = model(img).cpu().numpy().flatten()
#                 np.save(tmp_path, emb)
#         embeddings.append(emb)
#     return np.stack(embeddings).astype(np.float32)

# train_img_emb = compute_img_embeddings(train, TRAIN_IMG_DIR, "train")
# test_img_emb  = compute_img_embeddings(test, TEST_IMG_DIR, "test")

# train["matches"] = train.groupby("label_group")["posting_id"].transform(lambda x: " ".join(x))
# print(type(train_img_emb), train_img_emb.dtype, train_img_emb.shape)
# print(type(train_title_emb), train_title_emb.dtype, train_title_emb.shape)


# # combined vectors of img and txt
# train_emb_combined = np.hstack([train_img_emb, train_title_emb])
# test_emb_combined  = np.hstack([test_img_emb, test_title_emb])

# train_emb_combined /= np.linalg.norm(train_emb_combined, axis=1, keepdims=True)
# test_emb_combined  /= np.linalg.norm(test_emb_combined, axis=1, keepdims=True)

# sims = cosine_similarity(test_emb_combined, test_emb_combined)

# test_matches = []
# for i in range(len(test)):
#     row_sims = sims[i].copy()
#     idxs = [i]
#     other_idxs = np.where(row_sims >= SIM_THRESHOLD)[0].tolist()
#     idxs = list(dict.fromkeys(idxs + other_idxs))[:50]
#     match_ids = test.iloc[idxs]["posting_id"].tolist()
#     test_matches.append(" ".join(match_ids))

# test["matches"] = test_matches

# sub = test[["posting_id", "matches"]]
# sub.to_csv("submission.csv", index=False)
# print("submission.csv saved")





# # Normalize embeddings (important for cosine)
# train_embeddings = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)

# # Cosine similarity among train set
# sims_train = cosine_similarity(train_embeddings, train_embeddings)

# train_matches = []
# for i in range(len(train)):
#     row_sims = sims_train[i].copy()

#     # Always include itself
#     idxs = [i]

#     # Add all others above threshold
#     other_idxs = np.where(row_sims >= SIM_THRESHOLD)[0].tolist()

#     # Merge, remove duplicates, cap at 50
#     idxs = list(dict.fromkeys(idxs + other_idxs))[:50]

#     # Collect posting_ids
#     match_ids = train.iloc[idxs]["posting_id"].tolist()
#     train_matches.append(" ".join(match_ids))

# train["matches_pred"] = train_matches

# # ====================
# # Save for inspection
# # ====================
# train_debug = train[["posting_id", "label_group", "matches_pred"]]
# train_debug.to_csv("train_matches_debug.csv", index=False)
# print("✅ train_matches_debug.csv saved for inspection")


# pd.read_csv('/kaggle/working/train_matches_debug.csv').head(10)


# pd.read_csv('/kaggle/input/shopee-product-matching/sample_submission.csv').head()




