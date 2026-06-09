import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_LOGGING_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from google.protobuf import message_factory as _message_factory
if not hasattr(_message_factory.MessageFactory, "GetPrototype"):
    def _GetPrototype(self, descriptor):
        return _message_factory.GetMessageClass(descriptor)
    _message_factory.MessageFactory.GetPrototype = _GetPrototype

import re
import numpy as np
import pandas as pd
import torch

from tqdm.auto import tqdm
from urllib.parse import urlparse

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

dataset_path = "/kaggle/input/jigsaw-agile-community-rules"
train_file = f"{dataset_path}/train.csv"
test_file  = f"{dataset_path}/test.csv"

model_root = "/kaggle/input/bge-finetuned-model/bge_finetuned_model"
print("Model path:", model_root)

train_df = pd.read_csv(train_file)
test_df  = pd.read_csv(test_file)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

def cleaner(text):
    if not text:
        return text

    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'

    def rep(m):
        url = m.group(0)
        try:
            p = urlparse(url)
            domain = p.netloc.lower().replace("www.","")
            parts = [x for x in p.path.split("/") if x]
            if parts:
                return f"<url>: ({domain}/{parts[0]})"
            return f"<url>: ({domain})"
        except:
            return "<url>: (unknown)"

    return re.sub(url_pattern, rep, str(text))

print("Loading tokenizer + model ...")
tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
base_model = AutoModel.from_pretrained(model_root, local_files_only=True).to(device)
base_model.eval()
print("Model loaded.")

@torch.no_grad()
def encode_texts(texts, batch_size=64, max_length=256, normalize=True):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        for k in enc:
            enc[k] = enc[k].to(device)

        outputs = base_model(**enc)
        last_hidden = outputs.last_hidden_state
        attn_mask   = enc["attention_mask"].unsqueeze(-1)

        masked = last_hidden * attn_mask
        summed = masked.sum(dim=1)
        counts = attn_mask.sum(dim=1).clamp(min=1)
        mean_pooled = summed / counts

        embs = mean_pooled.detach().cpu().numpy().astype(np.float32)

        if normalize:
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
            embs = embs / norms

        all_embs.append(embs)

    return np.vstack(all_embs).astype(np.float32)

# Thu thập toàn bộ text cần embed: body + ví dụ positive/negative
all_texts = []
for _, r in test_df.iterrows():
    all_texts.append(cleaner(str(r["body"])))

example_cols = ["positive_example_1","positive_example_2",
                "negative_example_1","negative_example_2"]
for col in example_cols:
    if col in test_df.columns:
        for x in test_df[col]:
            if pd.notna(x):
                all_texts.append(cleaner(str(x)))

all_texts = list(set(all_texts))
print("Số text để embed:", len(all_texts))

all_emb = encode_texts(all_texts, batch_size=64, max_length=256, normalize=True)
text2emb = {t: e for t, e in zip(all_texts, all_emb)}
print("Số text2emb:", len(text2emb))

def build_kmeans(vecs, n_clusters=20, random_state=42):
    vecs = np.asarray(vecs, dtype=np.float32)
    unique_vecs = np.unique(vecs, axis=0)

    if len(unique_vecs) < 2:
        km = KMeans(n_clusters=1, n_init=10, max_iter=300, random_state=random_state)
        km.fit(unique_vecs)
        return km

    k = min(n_clusters, len(unique_vecs))
    km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=random_state)
    km.fit(unique_vecs)
    return km

rule_clusters = {}
unique_rules = test_df["rule"].unique()
print("Số rule trong test:", len(unique_rules))

for rule in tqdm(unique_rules):
    df_r = test_df[test_df["rule"] == rule]

    # positive = VI PHẠM, negative = KHÔNG VI PHẠM
    pos_embs = []  # positive examples -> violation prototypes
    neg_embs = []  # negative examples -> non-violation prototypes

    for _, r in df_r.iterrows():
        # Lấy positive_example_* (VI PHẠM) vào pos_embs
        for col in ["positive_example_1","positive_example_2"]:
            if col in df_r.columns and pd.notna(r[col]):
                t = cleaner(str(r[col]))
                if t in text2emb:
                    pos_embs.append(text2emb[t])

        # Lấy negative_example_* (KHÔNG VI PHẠM) vào neg_embs
        for col in ["negative_example_1","negative_example_2"]:
            if col in df_r.columns and pd.notna(r[col]):
                t = cleaner(str(r[col]))
                if t in text2emb:
                    neg_embs.append(text2emb[t])

    # Nếu thiếu một phía thì bỏ qua rule này
    if len(pos_embs) == 0 or len(neg_embs) == 0:
        continue

    pos_embs = np.array(pos_embs, dtype=np.float32)
    neg_embs = np.array(neg_embs, dtype=np.float32)

    km_pos = build_kmeans(pos_embs, n_clusters=20)  # cluster cho VI PHẠM
    km_neg = build_kmeans(neg_embs, n_clusters=20)  # cluster cho KHÔNG VI PHẠM

    pos_centers = km_pos.cluster_centers_.astype(np.float32)
    neg_centers = km_neg.cluster_centers_.astype(np.float32)

    pos_centers /= (np.linalg.norm(pos_centers, axis=1, keepdims=True) + 1e-12)
    neg_centers /= (np.linalg.norm(neg_centers, axis=1, keepdims=True) + 1e-12)

    rule_clusters[rule] = {
        "pos_centers": pos_centers,  # prototype từ positive (vi phạm)
        "neg_centers": neg_centers,  # prototype từ negative (không vi phạm)
    }

print("Số rule có cluster:", len(rule_clusters))

def topk_mean_cosine(vec, centers, k=3):
    vec = vec.reshape(1, -1)
    sims = cosine_similarity(vec, centers)[0]
    k = min(k, len(sims))
    topk = np.partition(sims, -k)[-k:]
    return float(topk.mean())

row_ids = []
scores  = []

for row in tqdm(test_df.itertuples(index=False), total=len(test_df)):
    body = cleaner(str(row.body))
    rule = row.rule
    rid  = row.row_id

    if (body in text2emb) and (rule in rule_clusters):
        emb = text2emb[body]
        pos_centers = rule_clusters[rule]["pos_centers"]  # VI PHẠM
        neg_centers = rule_clusters[rule]["neg_centers"]  # KHÔNG VI PHẠM

        s_pos = topk_mean_cosine(emb, pos_centers, k=3)  # giống VI PHẠM
        s_neg = topk_mean_cosine(emb, neg_centers, k=3)  # giống KHÔNG VI PHẠM

        # positive = violation nên score = giống positive - giống negative
        score = s_pos - s_neg
    else:
        score = 0.0

    row_ids.append(rid)
    scores.append(score)

pred_df = pd.DataFrame({"row_id": row_ids, "rule_violation": scores})

print(pred_df.head())
print("Số dòng submission:", len(pred_df))

out_path = "/kaggle/working/submission.csv"
pred_df.to_csv(out_path, index=False)
print("Saved to:", out_path)

