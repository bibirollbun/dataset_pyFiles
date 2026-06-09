import os
os.environ["WANDB_MODE"] = "disabled"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import pandas as pd
import numpy as np
import torch
import random
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    models
)
from sentence_transformers.losses import TripletLoss
from sklearn.cluster import MiniBatchKMeans
# 亮点：引入UMAP+层次聚类替换原KMeans，簇中心更稳健
from umap import UMAP
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import roc_auc_score
import re
from urllib.parse import urlparse
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def cleaner(text):
    if not text: return text
    url_pattern = r'https?://[^\s<>\\\"{}|\\\\^`\[\]]+'
    def repl(match):
        try:
            domain = urlparse(match.group(0)).netloc.lower()
            if domain.startswith('www.'): domain = domain[4:]
            return f"<url>: ({domain})"
        except: return "<url>: (unknown)"
    return re.sub(url_pattern, repl, str(text))

def load_data(split='test'):
    df = pd.read_csv(f'/kaggle/input/jigsaw-agile-community-rules/{split}.csv')
    print(f"Loaded {len(df)} {split} examples")
    return df

def load_model(model_path):
    word_emb = models.Transformer(model_path, max_seq_length=128, do_lower_case=True)
    pool = models.Pooling(word_emb.get_word_embedding_dimension(), pooling_mode="mean")
    model = SentenceTransformer(modules=[word_emb, pool])
    model.half().eval()
    return model

def release_model(model):
    del model
    torch.cuda.empty_cache()

def create_triplet_dataset(df, augmentation_factor=2, random_seed=42, subsample_fraction=1.0):
    """Create triplet dataset from test data: anchor=rule, positive=compliant, negative=violating"""
    random.seed(random_seed)
    np.random.seed(random_seed)

    anchors = []
    positives = []
    negatives = []

    print("Creating rule-aligned triplets from test data...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing test rows"):
        rule = cleaner(str(row['rule']))

        pos_examples = []  # compliant → triplet positive
        neg_examples = []  # violating → triplet negative

        for neg_col in ['negative_example_1', 'negative_example_2']:
            if pd.notna(row[neg_col]):
                pos_examples.append(cleaner(str(row[neg_col])))

        for pos_col in ['positive_example_1', 'positive_example_2']:
            if pd.notna(row[pos_col]):
                neg_examples.append(cleaner(str(row[pos_col])))

        for pos_ex in pos_examples:
            for neg_ex in neg_examples:
                anchors.append(rule)
                positives.append(pos_ex)
                negatives.append(neg_ex)

    if augmentation_factor > 0:
        print(f"Adding {augmentation_factor}x augmentation...")

        rule_positives = {}
        rule_negatives = {}

        for rule in df['rule'].unique():
            rule_df = df[df['rule'] == rule]

            pos_pool = []
            neg_pool = []

            for _, row in rule_df.iterrows():
                for neg_col in ['negative_example_1', 'negative_example_2']:
                    if pd.notna(row[neg_col]):
                        pos_pool.append(cleaner(str(row[neg_col])))
                for pos_col in ['positive_example_1', 'positive_example_2']:
                    if pd.notna(row[pos_col]):
                        neg_pool.append(cleaner(str(row[pos_col])))

            rule_positives[rule] = list(set(pos_pool))
            rule_negatives[rule] = list(set(neg_pool))

        for rule in df['rule'].unique():
            clean_rule = cleaner(str(rule))
            pos_pool = rule_positives[rule]
            neg_pool = rule_negatives[rule]

            n_samples = min(augmentation_factor * len(pos_pool), len(pos_pool) * len(neg_pool))

            for _ in range(n_samples):
                if pos_pool and neg_pool:
                    anchors.append(clean_rule)
                    positives.append(random.choice(pos_pool))
                    negatives.append(random.choice(neg_pool))

    combined = list(zip(anchors, positives, negatives))
    random.shuffle(combined)

    original_count = len(combined)
    if subsample_fraction < 1.0:
        n_samples = int(len(combined) * subsample_fraction)
        combined = combined[:n_samples]
        print(f"Subsampled {original_count} -> {len(combined)} triplets ({subsample_fraction*100:.1f}%)")

    anchors, positives, negatives = zip(*combined) if combined else ([], [], [])

    print(f"Created {len(anchors)} triplets from test data")

    return Dataset.from_dict({
        'anchor': list(anchors),
        'positive': list(positives),
        'negative': list(negatives)
    })

# ---------- 训练（照抄 speed-run 框架） ----------
def finetune(model, dataset, epochs=1, lr=2e-5, batch=16, grad_accum=2, margin=0.25):
    """完全照抄 speed-run 的 Trainer 方式，但模型先转回 fp32"""
    # 关键：恢复 fp32，否则 AMP 会报错
    model = model.float().train()

    dataset.set_format('pandas', columns=['anchor', 'positive', 'negative'])
    loss = TripletLoss(model=model, triplet_margin=margin)
    args = SentenceTransformerTrainingArguments(
        output_dir="./tmp",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=0.1,
        learning_rate=lr,
        fp16=True,               # Trainer 自己会做 fp16
        gradient_checkpointing=True,
        save_strategy="no",
        logging_steps=max(1, len(dataset) // batch // 4),
        report_to="none"
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=loss
    )
    trainer.train()
    model.half().eval()  # 训完再压回 fp16
    return model

# ---------- 编码（memmap 防oom） ----------
def encode_texts(model, texts, batch=32):
    texts = list(set(texts))
    dim = model.get_sentence_embedding_dimension()
    n = len(texts)
    mm = np.memmap('/tmp/emb.npy', dtype='float32', mode='w+', shape=(n, dim))
    for i in tqdm(range(0, n, batch), desc="encode"):
        end = min(i+batch, n)
        embs = model.encode(texts[i:end], normalize_embeddings=True, convert_to_numpy=True)
        mm[i:end] = embs
        del embs
        torch.cuda.empty_cache()
    mm.flush(); del mm
    mm = np.memmap('/tmp/emb.npy', dtype='float32', mode='r', shape=(n, dim))
    d = {txt: mm[i] for i, txt in enumerate(texts)}
    os.remove('/tmp/emb.npy')
    return d

# ---------- KMeans 多簇中心（你的逻辑） ----------
# 亮点：UMAP+层次聚类替换原KMeans，簇中心更稳健
def cluster_or_mean(embs, k):
    """embs: list[np.ndarray] -> 聚类后返回 (k?, dim) 的归一化中心矩阵"""
    if not embs:                       # 空列表直接返回
        return np.zeros((0, 0))
    embs = np.stack(embs, axis=0)      # <-- 关键：list -> ndarray
    if len(embs) < k:                  # 样本太少直接均值
        return np.mean(embs, axis=0, keepdims=True)

    # 高维先降维再聚类
    red = UMAP(n_components=32, random_state=42).fit_transform(embs) if len(embs) > 32 else embs
    lbl = AgglomerativeClustering(n_clusters=k).fit_predict(red)

    # 现在 embs 是 ndarray，可以布尔索引
    cents = np.array([embs[lbl == i].mean(0) for i in np.unique(lbl)])
    cents = cents / np.linalg.norm(cents, axis=1, keepdims=True)
    return cents

def build_centroids(df, text2emb, k_pos=3, k_neg=3):
    rule_centroids = {}
    for rule, sub in tqdm(df.groupby('rule'), desc='KMeans'):
        pos_embs, neg_embs = [], []
        for _, row in sub.iterrows():
            for c in ['positive_example_1', 'positive_example_2']:
                if pd.notna(row[c]):
                    t = cleaner(str(row[c]))
                    if t in text2emb: pos_embs.append(text2emb[t])
            for c in ['negative_example_1', 'negative_example_2']:
                if pd.notna(row[c]):
                    t = cleaner(str(row[c]))
                    if t in text2emb: neg_embs.append(text2emb[t])

        if pos_embs and neg_embs:
            rule_centroids[rule] = {'pos': cluster_or_mean(pos_embs, k_pos),
                                    'neg': cluster_or_mean(neg_embs, k_neg)}
    return rule_centroids

# ---------- 预测（你的逻辑） ----------
def predict(df, text2emb, rule_centroids):
    rows, scores = [], []
    for rule, sub in df.groupby('rule'):
        if rule not in rule_centroids: continue
        pos_cents = rule_centroids[rule]['pos']  # (k_pos, 768)
        neg_cents = rule_centroids[rule]['neg']  # (k_neg, 768)
        for _, row in sub.iterrows():
            body = cleaner(str(row['body']))
            if body not in text2emb: continue
            q = text2emb[body].reshape(1, -1)
            pos_d = cosine_distances(q, pos_cents).min()
            neg_d = cosine_distances(q, neg_cents).min()
            rows.append(row['row_id'])
            scores.append(neg_d - pos_d)
    return np.array(rows), np.array(scores)

# ---------- 单模型全流程 ----------
def run_one(model_path, df_train, df_test, name):
    print(f"\n===== {name} : speed-run finetune + KMeans =====")
    # 1. 加载原始模型
    model = load_model(model_path)
    # 2. 构造 triplet dataset（负例=其他规则正例）
    dataset = create_triplet_dataset(df_test)
    # 3. 微调（照抄 speed-run 框架）
    model = finetune(model, dataset, epochs=1, lr=2e-5, batch=16, grad_accum=2)
    # 4. 重新编码
    train_texts = list(set(text for col in ['body', 'positive_example_1', 'positive_example_2', 'negative_example_1', 'negative_example_2', 'rule']
                           for t in df_train[col].dropna() for text in [cleaner(str(t))]))
    train_t2e = encode_texts(model, train_texts)
    train_centroids = build_centroids(df_train, train_t2e)
    train_rows, train_scores = predict(df_train, train_t2e, train_centroids)
    train_order = pd.Series(train_scores, index=train_rows).reindex(df_train['row_id']).values

    # 5. 测试集
    test_texts = list(set(text for col in ['body', 'positive_example_1', 'positive_example_2', 'negative_example_1', 'negative_example_2', 'rule']
                           for t in df_test[col].dropna() for text in [cleaner(str(t))]))
    test_t2e = encode_texts(model, test_texts)
    test_centroids = build_centroids(df_test, test_t2e)
    test_rows, test_scores = predict(df_test, test_t2e, test_centroids)

    release_model(model)
    return train_order, test_rows, test_scores

# ---------- 主入口 ----------
def main():
    set_seed(42)
    df_train = load_data('train')
    df_test = load_data('test')

    path1 = "/kaggle/input/bge-m3/transformers/default/1"
    path2 = "/kaggle/input/baai/transformers/bge-base-en-v1.5/1"

    # 串行训练 & 推理
    train1, test_rows1, test_scores1 = run_one(path1, df_train, df_test, "bge-m3")
    train2, test_rows2, test_scores2 = run_one(path2, df_train, df_test, "bge-base")

    # 融合
    w1, w2 = 0.5, 0.5
    train_ensemble = w1 * train1 + w2 * train2
    test_ensemble = w1 * np.array(test_scores1) + w2 * np.array(test_scores2)

    # 训练 AUC
    y_true = df_train['rule_violation'].values
    auc = roc_auc_score(y_true, train_ensemble)
    print(f"\n>>> Ensemble AUC on TRAIN : {auc:.6f}")

    # 提交
    sub = pd.DataFrame({'row_id': test_rows1, 'rule_violation': test_ensemble})
    sub.to_csv('submission.csv', index=False)
    print(f"Saved submission.csv with {len(sub)} rows. "
          f"Min={test_ensemble.min():.4f}  Max={test_ensemble.max():.4f}  Mean={test_ensemble.mean():.4f}")

if __name__ == "__main__":
    main()

