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


!pip -q install open-clip-torch==2.24.0 faiss-cpu==1.8.0.post1 torchmetrics==1.4.0.post0 umap-learn==0.5.6 networkx==3.2.1 rich==13.7.1 --no-input


!pip install faiss-cpu


!pip install open-clip-torch==2.24




import os, gc, math, time, random, json, pathlib, collections, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# Viz
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# kNN
import faiss

# CLIP
import open_clip

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder




SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

class CFG:
    DATA_DIR = "/kaggle/input/h-and-m-personalized-fashion-recommendations"
    IMG_DIR  = "/kaggle/input/h-and-m-personalized-fashion-recommendations/images"
    WORK_DIR = "/kaggle/working"

    # subset for speed (tune up later)
    MAX_USERS = 20000
    MAX_ITEMS = 80000
    MIN_USER_INTERACTIONS = 5
    MIN_ITEM_INTERACTIONS = 5

    USE_TIME_SPLIT = True  # 7-day holdout

    # CLIP
    CLIP_MODEL = "ViT-B-32"
    CLIP_PRETRAIN = "laion2b_s34b_b79k"
    CLIP_BATCH = 128
    IMG_SIZE = 224

    # II graph fusion
    TOPK_II = 20
    ALPHA = 0.5  # image
    BETA  = 0.4  # text
    GAMMA = 0.1  # co-occur

    # LightGCN
    EMBED_DIM = 64
    LAYERS = 3
    LR = 1e-3
    WEIGHT_DECAY = 0.0
    BATCH_SIZE = 4096
    EPOCHS = 5

    # losses
    LAMBDA_G = 0.1
    TAU = 0.4

    # eval
    K_EVAL = 20

cfg = CFG()
print(cfg.__dict__)



ARTICLES_CSV = os.path.join(cfg.DATA_DIR, "articles.csv")
CUSTOMERS_CSV = os.path.join(cfg.DATA_DIR, "customers.csv")
TRANS_CSV    = os.path.join(cfg.DATA_DIR, "transactions_train.csv")

articles = pd.read_csv(ARTICLES_CSV)
customers = pd.read_csv(CUSTOMERS_CSV)
transactions = pd.read_csv(TRANS_CSV, parse_dates=["t_dat"])

print(articles.shape, customers.shape, transactions.shape)
display(articles.head(2))
display(customers.head(2))
display(transactions.head(2))



daily = transactions.groupby(transactions["t_dat"].dt.date).size()
plt.figure(figsize=(10,3)); daily.plot(); plt.title("Transactions per day"); plt.tight_layout(); plt.show()

if "product_type_name" in articles.columns:
    top_types = articles["product_type_name"].value_counts().head(15)
    plt.figure(figsize=(8,4)); sns.barplot(x=top_types.values, y=top_types.index)
    plt.title("Top product types in catalog"); plt.tight_layout(); plt.show()



user_cnt = transactions["customer_id"].value_counts()
item_cnt = transactions["article_id"].value_counts()

keep_users = user_cnt[user_cnt >= cfg.MIN_USER_INTERACTIONS].index
keep_items = item_cnt[item_cnt >= cfg.MIN_ITEM_INTERACTIONS].index

df = transactions[transactions["customer_id"].isin(keep_users) & transactions["article_id"].isin(keep_items)].copy()

if cfg.MAX_USERS:
    sel_users = set(df["customer_id"].drop_duplicates().sample(min(cfg.MAX_USERS, df["customer_id"].nunique()), random_state=SEED))
    df = df[df["customer_id"].isin(sel_users)]
if cfg.MAX_ITEMS:
    sel_items = set(df["article_id"].drop_duplicates().sample(min(cfg.MAX_ITEMS, df["article_id"].nunique()), random_state=SEED))
    df = df[df["article_id"].isin(sel_items)]

user_le = LabelEncoder().fit(df["customer_id"])
item_le = LabelEncoder().fit(df["article_id"])
df["uid"] = user_le.transform(df["customer_id"])
df["iid"] = item_le.transform(df["article_id"])

n_users = df["uid"].nunique()
n_items = df["iid"].nunique()
print("Users:", n_users, "Items:", n_items, "Interactions:", len(df))

tmax = df["t_dat"].max()
if cfg.USE_TIME_SPLIT:
    cutoff = tmax - pd.Timedelta(days=7)
    train_df = df[df["t_dat"] <= cutoff]
    test_df  = df[df["t_dat"] > cutoff]
    cutoff_val = cutoff - pd.Timedelta(days=3)
    val_df = train_df[train_df["t_dat"] > cutoff_val]
    train_df = train_df[train_df["t_dat"] <= cutoff_val]
else:
    train_df, tail = train_test_split(df, test_size=0.2, random_state=SEED)
    val_df, test_df = train_test_split(tail, test_size=0.5, random_state=SEED)

print("Train/Val/Test:", len(train_df), len(val_df), len(test_df))



from collections import defaultdict

def build_user_pos(df_part):
    pos = defaultdict(set)
    for u, i in zip(df_part["uid"].values, df_part["iid"].values):
        pos[int(u)].add(int(i))
    return pos

user_pos_train = build_user_pos(train_df)
user_pos_val   = build_user_pos(val_df)
user_pos_test  = build_user_pos(test_df)

lens = [len(v) for v in user_pos_train.values()]
plt.figure(figsize=(6,3)); sns.histplot(lens, bins=50); plt.title("Train positives per user"); plt.tight_layout(); plt.show()



# Align articles to filtered items
art = articles[articles["article_id"].isin(item_le.inverse_transform(np.arange(n_items)))].copy()
art["iid"] = item_le.transform(art["article_id"])
art = art.set_index("iid").sort_index()

def build_text(row):
    parts = []
    if isinstance(row.get("detail_desc", None), str) and len(row["detail_desc"])>0:
        parts.append(row["detail_desc"][:200])
    if isinstance(row.get("product_type_name", None), str):
        parts.append(row["product_type_name"])
    if isinstance(row.get("index_name", None), str):
        parts.append(row["index_name"])
    return ". ".join(parts) if parts else "fashion item"

texts = [build_text(art.loc[i].to_dict()) for i in range(n_items)]

def article_id_to_path(article_id):
    s = str(article_id).zfill(10)
    return os.path.join(cfg.IMG_DIR, s[:3], s + ".jpg")

img_paths = [article_id_to_path(int(item_le.inverse_transform([i])[0])) for i in range(n_items)]
print(texts[0][:120], "\n", img_paths[0])



clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(cfg.CLIP_MODEL, pretrained=cfg.CLIP_PRETRAIN, device=DEVICE)
clip_tokenizer = open_clip.get_tokenizer(cfg.CLIP_MODEL)

def encode_texts(texts, batch=512):
    all_feats = []
    for i in tqdm(range(0, len(texts), batch), desc="CLIP Text"):
        tok = clip_tokenizer(texts[i:i+batch]).to(DEVICE)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(DEVICE=='cuda')):
            feats = clip_model.encode_text(tok)
        feats = F.normalize(feats, dim=1)
        all_feats.append(feats.detach().cpu())
    return torch.cat(all_feats, dim=0)

from PIL import Image

def load_image_safe(p):
    try:
        with Image.open(p) as im:
            im = im.convert("RGB")
        return clip_preprocess(im)
    except:
        return None

def encode_images(paths, batch=cfg.CLIP_BATCH):
    all_feats, batch_imgs = [], []
    for i, p in enumerate(tqdm(paths, desc="CLIP Image")):
        img = load_image_safe(p)
        if img is None:
            batch_imgs.append(torch.zeros(3, cfg.IMG_SIZE, cfg.IMG_SIZE))
        else:
            batch_imgs.append(img)
        if len(batch_imgs)==batch or i==len(paths)-1:
            imgs = torch.stack(batch_imgs).to(DEVICE)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(DEVICE=='cuda')):
                feats = clip_model.encode_image(imgs)
            feats = F.normalize(feats, dim=1)
            all_feats.append(feats.detach().cpu())
            batch_imgs = []
    return torch.cat(all_feats, dim=0)

Z_txt = encode_texts(texts).numpy().astype("float32")
Z_img = encode_images(img_paths).numpy().astype("float32")

def rowwise_cos(a, b): return (a*b).sum(-1)
g = rowwise_cos(Z_img, Z_txt)
g = np.clip(g, 0.0, 1.0).astype("float32")
print("Z_img:", Z_img.shape, "Z_txt:", Z_txt.shape, "g mean:", g.mean())

plt.figure(figsize=(6,3)); sns.histplot(g, bins=50); plt.title("Grounding score g"); plt.tight_layout(); plt.show()



from PIL import Image
def show_examples(idx_list, cols=5):
    rows = math.ceil(len(idx_list)/cols)
    plt.figure(figsize=(3*cols, 3*rows))
    for k, idx in enumerate(idx_list):
        p = img_paths[idx]
        try:
            with Image.open(p) as im: im = im.convert("RGB")
        except: 
            im = Image.new("RGB", (224,224), (200,200,200))
        plt.subplot(rows, cols, k+1)
        plt.imshow(im); plt.axis("off")
        title = (texts[idx][:35] + "...") if len(texts[idx])>35 else texts[idx]
        plt.title(f"iid={idx}  g={g[idx]:.2f}\n{title}", fontsize=9)
    plt.tight_layout(); plt.show()

show_examples(random.sample(range(n_items), k=min(10, n_items)))



def faiss_knn(x, k):
    index = faiss.IndexFlatIP(x.shape[1])
    faiss.normalize_L2(x)
    index.add(x)
    sims, ids = index.search(x, k+1)  # includes self
    return sims[:,1:], ids[:,1:]

print("kNN img...")
img_sims, img_ids = faiss_knn(Z_img.copy(), cfg.TOPK_II)
print("kNN txt...")
txt_sims, txt_ids = faiss_knn(Z_txt.copy(), cfg.TOPK_II)

# co-occur (Jaccard) from TRAIN
user_baskets = collections.defaultdict(list)
for u,i in zip(train_df["uid"].values, train_df["iid"].values):
    user_baskets[int(u)].append(int(i))

item_user = collections.defaultdict(set)
for u, items in user_baskets.items():
    for it in items:
        item_user[it].add(u)

co_ids = np.zeros_like(img_ids)
co_sims = np.zeros_like(img_sims)
from collections import Counter
for i in tqdm(range(n_items), desc="Co-occur kNN"):
    users_i = item_user.get(i, set())
    co_counter = Counter()
    for u in users_i:
        for j in user_baskets[u]:
            if j!=i: co_counter[j]+=1
    if len(co_counter)==0:
        co_ids[i]=img_ids[i]; co_sims[i]=0.0; continue
    top = co_counter.most_common(cfg.TOPK_II*2)
    jscores, jidx = [], []
    for j,cij in top:
        u_j = item_user.get(j, set())
        inter = cij
        union = len(users_i) + len(u_j) - inter
        jacc = inter / union if union>0 else 0.0
        jscores.append(jacc); jidx.append(j)
    order = np.argsort(-np.array(jscores))[:cfg.TOPK_II]
    sel_idx = np.array(jidx)[order]
    sel_sim = np.array(jscores)[order]
    if len(sel_idx)<cfg.TOPK_II:
        need = cfg.TOPK_II - len(sel_idx)
        sel_idx = np.concatenate([sel_idx, img_ids[i,:need]])
        sel_sim = np.concatenate([sel_sim, np.zeros(need)])
    co_ids[i]  = sel_idx
    co_sims[i] = sel_sim

def fuse_and_gate(img_ids, img_sims, txt_ids, txt_sims, co_ids, co_sims, g, alpha, beta, gamma):
    rows, cols, vals = [], [], []
    for i in range(n_items):
        nbrs = {}
        for ids, sims, w in [(img_ids, img_sims, alpha), (txt_ids, txt_sims, beta), (co_ids, co_sims, gamma)]:
            for j, s in zip(ids[i], sims[i]):
                nbrs[j] = nbrs.get(j, 0.0) + w * float(s)
        for j, s in nbrs.items():
            w = s * float(0.5*(g[i] + g[j]))
            if w>0:
                rows.append(i); cols.append(j); vals.append(w)
    return np.array(rows), np.array(cols), np.array(vals, dtype="float32")

rows, cols, vals = fuse_and_gate(img_ids, img_sims, txt_ids, txt_sims, co_ids, co_sims, g, cfg.ALPHA, cfg.BETA, cfg.GAMMA)
print("II edges:", len(vals))

def build_norm_adj(n, rows, cols, vals):
    all_r = np.concatenate([rows, cols])
    all_c = np.concatenate([cols, rows])
    all_v = np.concatenate([vals, vals])
    deg = np.bincount(all_r, weights=all_v, minlength=n) + 1e-8
    deg2 = np.sqrt(deg)
    norm_v = all_v / (deg2[all_r] * deg2[all_c])
    idx = np.vstack([all_r, all_c])
    A = torch.sparse_coo_tensor(indices=torch.tensor(idx, dtype=torch.long),
                                values=torch.tensor(norm_v, dtype=torch.float32),
                                size=(n, n))
    return A.coalesce()

A_II = build_norm_adj(n_items, rows, cols, vals)
A_II



n_total = n_users + n_items

def build_ui_norm_adj(train_df, n_users, n_items):
    uu = train_df["uid"].values.astype("int64")
    ii = train_df["iid"].values.astype("int64") + n_users
    v  = np.ones_like(uu, dtype="float32")
    rows = np.concatenate([uu, ii])
    cols = np.concatenate([ii, uu])
    vals = np.concatenate([v, v])

    deg = np.bincount(rows, weights=vals, minlength=n_users+n_items) + 1e-8
    deg2 = np.sqrt(deg)
    norm_v = vals / (deg2[rows] * deg2[cols])
    idx = np.vstack([rows, cols])
    A = torch.sparse_coo_tensor(indices=torch.tensor(idx, dtype=torch.long),
                                values=torch.tensor(norm_v, dtype=torch.float32),
                                size=(n_users+n_items, n_users+n_items))
    return A.coalesce()

A_UI = build_ui_norm_adj(train_df, n_users, n_items)

def lift_item_adj_to_total(A_II, n_users, n_items):
    idx = A_II.indices()
    val = A_II.values()
    idx_lift = torch.vstack([idx[0]+n_users, idx[1]+n_users])
    A = torch.sparse_coo_tensor(idx_lift, val, (n_users+n_items, n_users+n_items)).coalesce()
    return A

A_II_total = lift_item_adj_to_total(A_II, n_users, n_items)
A_UI, A_II_total = A_UI.to(DEVICE), A_II_total.to(DEVICE)




# import matplotlib.pyplot as plt
# import networkx as nx
# from PIL import Image
# import numpy as np

# def visualize_user_neighborhood_enhanced(u, max_items=10):
#     """
#     Visualize a user's multimodal neighborhood:
#     - Blue nodes = User
#     - Orange nodes = Purchased items
#     - Green nodes = Similar (recommended) items
#     Each item node displays the product name and optionally a thumbnail.
#     """
#     items = list(user_pos_train.get(u, []))[:max_items]
#     G = nx.Graph()
#     G.add_node(f"User_{u}", type='user', label=f"User {u}")

#     for it in items:
#         prod_name = art.loc[it, "product_type_name"]
#         G.add_node(f"Item_{it}", type='item', label=prod_name)
#         G.add_edge(f"User_{u}", f"Item_{it}", relation='purchased')
#         # add top 2 similar items
#         for j in img_ids[it][:2]:
#             if j not in art.index: continue
#             neigh_name = art.loc[j, "product_type_name"]
#             G.add_node(f"Sim_{j}", type='similar', label=neigh_name)
#             G.add_edge(f"Item_{it}", f"Sim_{j}", relation='similar')

#     pos = nx.spring_layout(G, k=0.7, seed=42)
#     node_colors = []
#     for n in G.nodes():
#         t = G.nodes[n]['type']
#         if t == 'user':
#             node_colors.append('tab:blue')
#         elif t == 'item':
#             node_colors.append('tab:orange')
#         else:
#             node_colors.append('tab:green')

#     edge_colors = ['blue' if G.edges[e]['relation']=='purchased' else 'orange' for e in G.edges()]
#     labels = {n:G.nodes[n]['label'] for n in G.nodes()}

#     plt.figure(figsize=(10,7))
#     nx.draw(G, pos, labels=labels, node_color=node_colors, edge_color=edge_colors,
#             node_size=1200, font_size=8, font_weight='bold')
#     plt.title(f"Multimodal Neighborhood for User {u}", fontsize=13)
#     plt.show()


# visualize_user_neighborhood_enhanced(random.randint(0, n_users-1))



from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def add_image_to_node(ax, pos, node, path, zoom=0.15):
    try:
        im = Image.open(path).convert("RGB")
        img = OffsetImage(im, zoom=zoom)
        ab = AnnotationBbox(img, pos[node], frameon=False)
        ax.add_artist(ab)
    except:
        pass

def visualize_user_neighborhood_with_images(u, max_items=6):
    items = list(user_pos_train.get(u, []))[:max_items]
    G = nx.Graph()
    G.add_node(f"User_{u}", type='user')
    for it in items:
        G.add_node(f"Item_{it}", type='item')
        G.add_edge(f"User_{u}", f"Item_{it}")
        for j in img_ids[it][:2]:
            G.add_node(f"Sim_{j}", type='similar')
            G.add_edge(f"Item_{it}", f"Sim_{j}")
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(12,8))
    nx.draw(G, pos, with_labels=False, node_color="white", edge_color="gray", node_size=100, ax=ax)
    add_image_to_node(ax, pos, f"User_{u}", random.choice(img_paths))
    for n in G.nodes():
        if n.startswith("Item_"):
            iid = int(n.split("_")[1])
            add_image_to_node(ax, pos, n, img_paths[iid])
    plt.title(f"Visual Neighborhood of User {u}", fontsize=14)
    plt.show()


visualize_user_neighborhood_enhanced(random.randint(0, n_users-1))




class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, d=64, layers=3):
        super().__init__()
        self.emb = nn.Embedding(n_users+n_items, d)
        nn.init.xavier_uniform_(self.emb.weight)
        self.layers = layers
        self.n_users = n_users
        self.n_items = n_items

    def propagate(self, A_ui, A_ii):
        E0 = self.emb.weight
        acc = [E0]
        E = E0
        for _ in range(self.layers):
            E = 0.5*torch.sparse.mm(A_ui, E) + 0.5*torch.sparse.mm(A_ii, E)
            acc.append(E)
        return torch.stack(acc, dim=0).mean(0)

    def forward(self, A_ui, A_ii):
        E = self.propagate(A_ui, A_ii)
        U = E[:self.n_users]
        I = E[self.n_users:]
        return U, I

def bpr_loss(u_e, i_pos_e, i_neg_e):
    pos = (u_e*i_pos_e).sum(-1)
    neg = (u_e*i_neg_e).sum(-1)
    return -F.logsigmoid(pos - neg).mean()

user_pos_array = {u: np.array(list(items)) for u, items in user_pos_train.items()}
all_items = np.arange(n_items)

def sample_batch(batch_size=4096):
    users = np.random.choice(list(user_pos_array.keys()), size=batch_size, replace=True)
    pos_items = np.array([np.random.choice(user_pos_array[u]) for u in users])
    neg_items = []
    for u in users:
        while True:
            j = np.random.randint(0, n_items)
            if j not in user_pos_array[u]:
                neg_items.append(j); break
    return users, pos_items, np.array(neg_items)

g_t = torch.tensor(g, dtype=torch.float32, device=DEVICE)



@torch.no_grad()
def compute_embeddings(model):
    U, I = model(A_UI, A_II_total)
    return U, I

def evaluate_split(model, user_pos_true, K=20):
    model.eval()
    U, I = compute_embeddings(model)
    recalls, ndcgs = [], []
    users_list = list(user_pos_true.keys())
    for start in range(0, len(users_list), 512):
        batch_users = users_list[start:start+512]
        u_emb = U[batch_users]
        scores = torch.matmul(u_emb, I.T)
        # mask train positives
        for bi, u in enumerate(batch_users):
            train_pos = list(user_pos_train.get(u, []))
            if train_pos:
                scores[bi, torch.tensor(train_pos, device=DEVICE)] = -1e9
        topk = torch.topk(scores, k=K, dim=1).indices.cpu().numpy()
        for bi, u in enumerate(batch_users):
            truth = set(user_pos_true.get(u, []))
            if not truth: continue
            preds = list(topk[bi])
            hit = len(set(preds) & truth)
            recalls.append(hit / min(K, len(truth)))
            dcg = 0.0
            for rank, it in enumerate(preds, start=1):
                if it in truth:
                    dcg += 1.0 / math.log2(rank+1)
            idcg = sum(1.0 / math.log2(r+1) for r in range(1, min(K, len(truth))+1))
            ndcgs.append(dcg / idcg if idcg>0 else 0.0)
    R = float(np.mean(recalls)) if recalls else 0.0
    N = float(np.mean(ndcgs)) if ndcgs else 0.0
    return R, N

def coverage_at_k(model, K=20, sample_users=2000):
    model.eval()
    U, I = compute_embeddings(model)
    users_list = list(user_pos_test.keys())[:sample_users]
    seen = set()
    for start in range(0, len(users_list), 512):
        batch_users = users_list[start:start+512]
        u_emb = U[batch_users]
        scores = torch.matmul(u_emb, I.T)
        for bi, u in enumerate(batch_users):
            train_pos = list(user_pos_train.get(u, []))
            if train_pos:
                scores[bi, torch.tensor(train_pos, device=DEVICE)] = -1e9
        topk = torch.topk(scores, k=K, dim=1).indices.cpu().numpy()
        for row in topk: seen.update(row.tolist())
    return len(seen) / n_items

def grounding_at_k(model, K=20, sample_users=2000):
    model.eval()
    U, I = compute_embeddings(model)
    users_list = list(user_pos_test.keys())[:sample_users]
    vals = []
    for start in range(0, len(users_list), 512):
        batch_users = users_list[start:start+512]
        u_emb = U[batch_users]
        scores = torch.matmul(u_emb, I.T)
        for bi, u in enumerate(batch_users):
            train_pos = list(user_pos_train.get(u, []))
            if train_pos:
                scores[bi, torch.tensor(train_pos, device=DEVICE)] = -1e9
        topk = torch.topk(scores, k=K, dim=1).indices
        vals.append(g_t[topk].mean().item())
    return float(np.mean(vals)) if vals else 0.0



model = LightGCN(n_users, n_items, d=cfg.EMBED_DIM, layers=cfg.LAYERS).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

def train(epochs=cfg.EPOCHS, log_every=200):
    for ep in range(1, epochs+1):
        model.train()
        losses = []
        steps = max(1, len(train_df)//cfg.BATCH_SIZE)
        for it in range(steps):
            users, pos_items, neg_items = sample_batch(cfg.BATCH_SIZE)
            users_t = torch.tensor(users, dtype=torch.long, device=DEVICE)
            pos_t   = torch.tensor(pos_items, dtype=torch.long, device=DEVICE)
            neg_t   = torch.tensor(neg_items, dtype=torch.long, device=DEVICE)

            Ue, Ie = model(A_UI, A_II_total)
            u_e = Ue[users_t]; i_pos_e = Ie[pos_t]; i_neg_e = Ie[neg_t]

            loss_bpr = bpr_loss(u_e, i_pos_e, i_neg_e)
            loss_gr  = cfg.LAMBDA_G * torch.relu(cfg.TAU - g_t[pos_t]).mean()
            loss = loss_bpr + loss_gr

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(loss.item())
            if (it+1)%log_every==0:
                print(f"ep{ep} step{it+1}/{steps} loss={np.mean(losses):.4f}")

        # epoch eval
        Rv, Nv = evaluate_split(model, user_pos_val, K=cfg.K_EVAL)
        Rt, Nt = evaluate_split(model, user_pos_test, K=cfg.K_EVAL)
        Cov = coverage_at_k(model, K=cfg.K_EVAL)
        Grd = grounding_at_k(model, K=cfg.K_EVAL)
        print(f"[E{ep}] Val R@{cfg.K_EVAL}={Rv:.4f} N@{cfg.K_EVAL}={Nv:.4f} | "
              f"Test R@{cfg.K_EVAL}={Rt:.4f} N@{cfg.K_EVAL}={Nt:.4f} | "
              f"Cov@{cfg.K_EVAL}={Cov:.4f} | Ground@{cfg.K_EVAL}={Grd:.3f}")

train()



from PIL import Image

@torch.no_grad()
def recommend_for_user(u, K=10, show=True):
    model.eval()
    U, I = compute_embeddings(model)
    scores = torch.matmul(U[u:u+1], I.T).squeeze(0)
    train_pos = list(user_pos_train.get(u, []))
    if train_pos:
        scores[torch.tensor(train_pos, device=DEVICE)] = -1e9
    topk = torch.topk(scores, k=K).indices.cpu().numpy().tolist()
    if show:
        cols = 5; rows = math.ceil(K/cols)
        plt.figure(figsize=(3*cols,3*rows))
        for idx, it in enumerate(topk):
            p = img_paths[it]
            try:
                with Image.open(p) as im: im = im.convert("RGB")
            except:
                im = Image.new("RGB", (224,224), (200,200,200))
            plt.subplot(rows, cols, idx+1)
            plt.imshow(im); plt.axis("off")
            title = (texts[it][:30]+"...") if len(texts[it])>30 else texts[it]
            plt.title(f"iid={it}  g={g[it]:.2f}\n{title}", fontsize=9)
        plt.tight_layout(); plt.show()
    return topk

u_example = random.randint(0, n_users-1)
print("User:", u_example)
_ = recommend_for_user(u_example, K=10, show=True)



import pandas as pd

# Store your evaluation results from training logs
performance_table = pd.DataFrame({
    "Epoch": [1, 2, 3, 4, 5],
    "Val_Recall@20": [0.0078, 0.0101, 0.0101, 0.0116, 0.0148],
    "Val_NDCG@20": [0.0061, 0.0065, 0.0064, 0.0070, 0.0078],
    "Test_Recall@20": [0.0072, 0.0076, 0.0077, 0.0095, 0.0114],
    "Test_NDCG@20": [0.0045, 0.0047, 0.0047, 0.0053, 0.0060],
    "Coverage@20": [0.0067, 0.0045, 0.0047, 0.0045, 0.0050],
    "Grounding@20": [0.281, 0.271, 0.269, 0.266, 0.264]
})

# Display nicely
print("\n=== Model Performance Metrics ===\n")
print(performance_table.to_string(index=False))



plt.figure(figsize=(7,4))
plt.plot(performance_table["Epoch"], performance_table["Test_Recall@20"], '-o', label='Recall@20')
plt.plot(performance_table["Epoch"], performance_table["Test_NDCG@20"], '-s', label='NDCG@20')
plt.plot(performance_table["Epoch"], performance_table["Grounding@20"], '-^', label='Grounding')
plt.xlabel("Epoch"); plt.ylabel("Score")
plt.title("Learning Trend of LightGCN (BPR + Grounding)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()






final_metrics = {
    "Recall@20": 0.0114,
    "NDCG@20": 0.0060,
    "Coverage@20": 0.0050,
    "Grounding@20": 0.264
}

sns.barplot(x=list(final_metrics.keys()), y=list(final_metrics.values()), palette="viridis")
plt.title("Final Model Evaluation Metrics")
plt.ylabel("Score")
plt.tight_layout()
plt.show()



plt.figure(figsize=(7,4))
sns.histplot(user_recalls, bins=30, kde=True, color='skyblue')
plt.title("Distribution of Recall@20 Across Users")
plt.xlabel("Recall@20"); plt.ylabel("User Count")
plt.tight_layout(); plt.show()



subset = np.random.choice(n_items, size=2000, replace=False)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
proj = tsne.fit_transform(Z_img[subset])

plt.figure(figsize=(8,6))
sns.scatterplot(x=proj[:,0], y=proj[:,1],
                hue=art.iloc[subset]["product_type_name"], 
                legend=False, s=40, alpha=0.8)
plt.title("t-SNE of Item Embeddings (CLIP-Based)")
plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
plt.tight_layout(); plt.show()



plt.figure(figsize=(7,5))
sns.scatterplot(x=grounding_vals, y=recall_vals, alpha=0.7)
plt.xlabel("Avg Grounding Score of Recommendations")
plt.ylabel("User Recall@20")
plt.title("Grounding vs Recommendation Accuracy")
plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()



types = art.iloc[rec_items]["product_type_name"].value_counts().head(10)
sns.barplot(y=types.index, x=types.values, palette="crest")
plt.title("Top Recommended Product Categories")
plt.xlabel("Recommendation Frequency")
plt.tight_layout(); plt.show()



# framework network visulization
visualize_user_neighborhood_enhanced(random.randint(0, n_users-1))





