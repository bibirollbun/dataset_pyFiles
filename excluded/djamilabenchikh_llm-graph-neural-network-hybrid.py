# This code adds the local path of the torch_geometric wheel file to the system path 
# so that the library can be imported inside the Kaggle environment. 
# It then imports the necessary PyTorch Geometric modules for building and training graph neural networks.
import sys
sys.path.append("/kaggle/input/competition-2/PYT-20250912T012549Z-1-001/PYT/torch_geometric-2.6.1-py3-none-any.whl")

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool



import os, random, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv
import re
import html




is_kaggle_submission = os.getenv('KAGGLE_IS_COMPETITION_RERUN') is not None
df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
if is_kaggle_submission:
    df_test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
else:
    df_test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

def clean_tweet_text(text):
    """
    Nettoie un texte (tweet, post Reddit...) :
    - tout en minuscule
    - enlÃ¨ve les liens, mentions, hashtags
    - remplace les caractÃ¨res HTML (&amp;)
    - garde ponctuation simple
    - rÃ©duit les caractÃ¨res rÃ©pÃ©tÃ©s
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = html.unescape(text)  # &amp; â†’ &
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)  # [texte](url) â†’ texte
    text = re.sub(r"@\w+", " ", text)                       # mentions
    text = re.sub(r"#\w+", " ", text)                       # hashtags
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)              # loooool â†’ lo
    text = re.sub(r"[^\x20-\x7E]", " ", text)               # caractÃ¨res non-ASCII
    text = re.sub(r"\s+", " ", text).strip()                # espaces multiples

    return text

for col in ["rule", "body"]:
    if col in df.columns:
        df[col] = df[col].astype(str).apply(clean_tweet_text)
    if col in df_test.columns:
        df_test[col] = df_test[col].astype(str).apply(clean_tweet_text)



# ===============================================================
# Hybrid DeBERTa + GraphSAGE Model for Rule Violation Detection
# ===============================================================
# This script builds a token-level dynamic graph between "rule" and "body" texts,
# encodes each token using a LayerWeighted DeBERTa encoder,
# and applies a two-phase GraphSAGE training strategy to detect semantic violations.



# Configuration and Parameters

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-base"  # Pretrained DeBERTa checkpoint
MAX_LEN = 256                   # Max token length per text
TOPK = 24                        # Number of top semantic connections (cross-text edges)
HIDDEN = 192                     # Hidden dimension for GraphSAGE
DROPOUT = 0.3                    # Dropout rate for regularization
WD = 5e-4                        # Weight decay
EPOCHS_A = 4                    # Phase A: Train GNN only
EPOCHS_B = 4                     # Phase B: Joint fine-tuning
BATCH_SIZE = 16                  # Training batch size


# Reproducibility

def seed_all(seed=SEED):
    """Fix random seeds across all libraries for reproducibility."""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
seed_all()


# LayerWeighted DeBERTa Encoder

class LayerWeightedEncoder(nn.Module):
    """
    DeBERTa encoder with learnable layer weighting.
    Computes contextual token embeddings as a weighted combination of hidden layers.
    """
    def __init__(self, ckpt=CKPT):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.backbone = AutoModel.from_pretrained(ckpt)
        self.num_layers = self.backbone.config.num_hidden_layers + 1
        self.mix = nn.Parameter(torch.zeros(self.num_layers))

    @torch.no_grad()
    def _keep_mask(self, input_ids):
        """Filter out special tokens such as [CLS], [SEP], etc."""
        toks = self.tok.convert_ids_to_tokens(input_ids)
        return [i for i, tk in enumerate(toks) if tk not in ["[CLS]","[SEP]","<s>","</s>","<pad>"]]

    def encode_tokens(self, text: str, device=DEVICE):
        """Encode a text into token embeddings using weighted layer aggregation."""
        enc = self.tok(text, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LEN).to(device)
        out = self.backbone(**enc, output_hidden_states=True)
        hs_all = torch.stack(out.hidden_states, dim=0)          # [L+1, 1, T, H]
        w = torch.softmax(self.mix, dim=0).view(-1,1,1,1)       # learnable weights
        hs = (w * hs_all).sum(0).squeeze(0)                     # [T, H]
        keep = self._keep_mask(enc["input_ids"].squeeze(0))
        if len(keep) == 0:
            return torch.zeros(1, hs.size(-1), device=device)
        return hs[keep, :]


# Graph Feature and Edge Construction

def token_importance(x):
    """Compute token-level importance based on embedding norm."""
    s = x.norm(dim=1, keepdim=True)
    return (s - s.min()) / (s.max() - s.min() + 1e-8)

def violation_signal(body_x, rule_x):
    """Compute violation signal = 1 - mean cosine similarity with rule tokens."""
    sim = F.cosine_similarity(body_x.unsqueeze(1), rule_x.unsqueeze(0), dim=2)
    v = 1 - sim.mean(dim=1, keepdim=True)
    v = (v - v.min())/(v.max()-v.min()+1e-8)
    return v, sim

def adaptive_k(Tr, Tb, kmax=TOPK):
    """Adaptive number of top edges depending on text lengths."""
    avg_len = (Tr + Tb)/2
    if avg_len < 40:   k = min(kmax, Tr)
    elif avg_len < 80: k = min(max(8, kmax-2), Tr)
    else:              k = min(max(6, kmax-4), Tr)
    return max(1, int(k))


@torch.no_grad()
def build_edges_snapshot(encoder, rule_text, body_text, device=DEVICE):
    """
    Build a dynamic token graph connecting rule and body tokens.
    Cross-text edges are weighted by cosine similarity.
    Chain edges preserve sequential structure inside each text.
    """
    encoder.eval()
    rule_x = encoder.encode_tokens(rule_text, device=device)
    body_x = encoder.encode_tokens(body_text, device=device)
    Tr, Tb = rule_x.size(0), body_x.size(0)
    if Tr == 0 or Tb == 0:
        ei = torch.empty(2, 0, dtype=torch.long, device=device)
        ew = torch.empty(0, device=device)
        return ei, ew, Tr, Tb

    # Cross-text similarity and top-K edge selection
    sim = F.cosine_similarity(body_x.unsqueeze(1), rule_x.unsqueeze(0), dim=2)
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
    k = adaptive_k(Tr, Tb)
    vals, idxs = torch.topk(sim, k=k, dim=1)

    src = (torch.arange(Tb, device=device).unsqueeze(1) + Tr).repeat(1, k).reshape(-1)
    dst = idxs.reshape(-1)

    e1 = torch.stack([src, dst], 0)
    e2 = torch.stack([dst, src], 0)
    edge_index = torch.cat([e1, e2], dim=1)
    edge_w = torch.cat([vals.reshape(-1), vals.reshape(-1)], dim=0)

    # Add internal sequential (chain) edges
    if Tb > 1:
        b = torch.stack([torch.arange(Tr, Tr+Tb-1, device=device),
                         torch.arange(Tr+1, Tr+Tb, device=device)], 0)
        edge_index = torch.cat([edge_index, b, b.flip(0)], dim=1)
        edge_w = torch.cat([edge_w, torch.ones(b.size(1)*2, device=device)], dim=0)
    if Tr > 1:
        r = torch.stack([torch.arange(0, Tr-1, device=device),
                         torch.arange(1, Tr, device=device)], 0)
        edge_index = torch.cat([edge_index, r, r.flip(0)], dim=1)
        edge_w = torch.cat([edge_w, torch.ones(r.size(1)*2, device=device)], dim=0)

    return edge_index, edge_w, Tr, Tb


# Graph Dataset and Batching

class PairGraphExample:
    """Single graph example containing rule, body, and label."""
    def __init__(self, rule, body, label, edge_index, edge_weight, Tr, Tb):
        self.rule = rule; self.body = body; self.label = int(label)
        self.edge_index = edge_index; self.edge_weight = edge_weight
        self.Tr = Tr; self.Tb = Tb

class PairGraphDataset(Dataset):
    """Build a dataset of graph examples for ruleâ€“body pairs."""
    def __init__(self, df, encoder):
        self.items = []
        for _, row in df.iterrows():
            ei, ew, Tr, Tb = build_edges_snapshot(encoder, row["rule"], row["body"])
            self.items.append(PairGraphExample(row["rule"], row["body"], row["rule_violation"], ei, ew, Tr, Tb))
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def collate_build_batch(batch_list, encoder, device=DEVICE):
    """
    Encode ruleâ€“body pairs and build batched PyG Data objects
    with token embeddings and graph connectivity.
    """
    data_list = []
    for ex in batch_list:
        rule_x = encoder.encode_tokens(ex.rule, device=device)
        body_x = encoder.encode_tokens(ex.body, device=device)
        imp_r  = token_importance(rule_x)
        imp_b  = token_importance(body_x)
        viol_b, _ = violation_signal(body_x, rule_x)

        # Combine token features: [embedding | source | importance | violation]
        rule_feat = torch.cat([rule_x, torch.zeros(ex.Tr,1,device=device), imp_r, torch.zeros(ex.Tr,1,device=device)], dim=1)
        body_feat = torch.cat([body_x, torch.ones (ex.Tb,1,device=device),  imp_b, viol_b], dim=1)
        x = torch.cat([rule_feat, body_feat], dim=0)

        g = Data(x=x, edge_index=ex.edge_index.to(device),
                 edge_weight=ex.edge_weight.to(device),
                 y=torch.tensor([ex.label], dtype=torch.long, device=device))
        data_list.append(g)
    return Batch.from_data_list(data_list)



# GraphSAGE with Attention Pooling

class AttnPool(nn.Module):
    """Graph-level attention pooling layer."""
    def __init__(self, d):
        super().__init__()
        self.q = nn.Linear(d, 1)

    def forward(self, h, batch_ids):
        # Compute attention weights within each graph
        a = torch.exp(self.q(h))
        B = int(batch_ids.max().item()) + 1 if batch_ids.numel() > 0 else 1
        denom = torch.zeros(B, 1, device=h.device, dtype=h.dtype)
        denom.index_add_(0, batch_ids, a)
        denom = denom[batch_ids] + 1e-12
        a = a / denom
        num = torch.zeros(B, h.size(1), device=h.device, dtype=h.dtype)
        num.index_add_(0, batch_ids, a * h)
        return num  # [B, d]


class GraphSAGEClassifier(nn.Module):
    """GraphSAGE-based classifier for graph-level prediction."""
    def __init__(self, in_dim=771, hid=HIDDEN, out_dim=2, dropout=DROPOUT):
        super().__init__()
        self.sage1 = SAGEConv(in_dim, hid)
        self.sage2 = SAGEConv(hid, hid)
        self.drop = nn.Dropout(dropout)
        self.pool = AttnPool(hid)
        self.cls  = nn.Linear(hid, out_dim)

    def forward(self, batch):
        h = F.relu(self.sage1(batch.x, batch.edge_index))
        h = self.drop(h)
        h = F.relu(self.sage2(h, batch.edge_index))
        h = self.drop(h)
        hg = self.pool(h, batch.batch)     # Replace global_mean_pool
        return self.cls(hg)




# Training and Evaluation

def freeze_deberta(enc, freeze=True):
    """Freeze or unfreeze the DeBERTa encoder parameters."""
    for p in enc.backbone.parameters():
        p.requires_grad = (not freeze)

def unfreeze_last_n_layers(enc, n=2):
    """Unfreeze the last n layers of the DeBERTa encoder for fine-tuning."""
    for layer in enc.backbone.encoder.layer[-n:]:
        for p in layer.parameters(): 
            p.requires_grad = True


@torch.no_grad()
def eval_loader(model, enc, loader):
    """
    Evaluate accuracy and F1 score on a given dataloader.
    The function runs in no-grad mode to save memory and computation.
    """
    model.eval(); enc.eval()
    ys, ps = [], []
    for b in loader:
        batch = collate_build_batch(b, enc)
        logits = model(batch)
        pred = logits.argmax(dim=1).cpu().numpy()
        y = batch.y.cpu().numpy()
        ys.append(y); ps.append(pred)
    y_true, y_pred = np.concatenate(ys), np.concatenate(ps)
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred)


@torch.no_grad()
def predict_pair_mc(model, encoder, rule_text, body_text, win_chars=500, stride_chars=400, mc_passes=5):
    """
    Perform Monte Carlo (MC) Dropout inference for long inputs.
    - Splits the body text into overlapping windows.
    - Runs multiple stochastic forward passes with dropout active.
    - Returns the average probability of the 'violation' class.
    """
    model.train(); encoder.train()  # Enable dropout
    parts = [body_text[i:i+win_chars] for i in range(0, max(1, len(body_text)-win_chars+1), stride_chars)] or [body_text]
    probs = []
    for _ in range(mc_passes):
        per = []
        for part in parts:
            ei, ew, Tr, Tb = build_edges_snapshot(encoder, rule_text, part)
            item = PairGraphExample(rule_text, part, 0, ei, ew, Tr, Tb)
            batch = collate_build_batch([item], encoder)
            p = torch.softmax(model(batch), dim=1)[0,1].item()
            per.append(p)
        probs.append(float(np.mean(per)))
    model.eval(); encoder.eval()
    return float(np.mean(probs))


# Data Preparation and Split

df = df.copy()
df["rule_violation"] = df["rule_violation"].astype(int)

print("\n===== Unique Fold (Train/Val Split) =====")
df_train, df_local_test = train_test_split(df, test_size=0.2, stratify=df["rule_violation"], random_state=SEED)
df_tr, df_va = train_test_split(df_train, test_size=0.1, stratify=df_train["rule_violation"], random_state=SEED)
print(f"Train: {len(df_tr)}, Val: {len(df_va)}, Local-Test: {len(df_local_test)}")

# Optional data augmentation from test examples (for balance improvement)
test_subset = df_test.sample(frac=0.1, random_state=SEED).reset_index(drop=True)
flatten = []
for violation_type in ["positive", "negative"]:
    for i in range(1, 3):
        col_name = f"{violation_type}_example_{i}"
        if col_name in test_subset.columns:
            sub_df = test_subset[["rule", col_name]].copy()
            sub_df = sub_df.rename(columns={col_name: "body"})
            sub_df["rule_violation"] = 1 if violation_type == "positive" else 0
            flatten.append(sub_df)
if len(flatten) > 0:
    enrich_df = pd.concat(flatten, ignore_index=True)
    df_tr = pd.concat([df_tr, enrich_df], ignore_index=True).drop_duplicates()
    print(f" Enrichment done: {len(enrich_df)} new samples added to training set")
else:
    print(" No valid examples found for enrichment.")


# Dataset and DataLoader Setup

encoder = LayerWeightedEncoder(CKPT).to(DEVICE)
train_ds = PairGraphDataset(df_tr, encoder)
val_ds   = PairGraphDataset(df_va, encoder)
test_ds  = PairGraphDataset(df_local_test, encoder)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=lambda b:b)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b:b)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b:b)

model = GraphSAGEClassifier(in_dim=771, hid=HIDDEN, out_dim=2, dropout=DROPOUT).to(DEVICE)
crit  = nn.CrossEntropyLoss()


# Phase A â€“ Train GNN only (freeze DeBERTa)

out_dir = "./checkpoints_graphsage_e2e"; os.makedirs(out_dir, exist_ok=True)
freeze_deberta(encoder, True)
optA = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WD)
best_f1 = -1

for epoch in range(EPOCHS_A):
    model.train()
    tot = 0.0
    for b in train_loader:
        batch = collate_build_batch(b, encoder)
        optA.zero_grad()
        logits = model(batch)
        loss = crit(logits, batch.y)
        loss.backward(); optA.step()
        tot += loss.item()
    acc, f1 = eval_loader(model, encoder, val_loader)
    print(f"[A] Epoch {epoch+1}/{EPOCHS_A} | loss={tot/len(train_loader):.4f} | val_acc={acc:.4f} | val_f1={f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        torch.save({"gnn": model.state_dict(), "enc": encoder.state_dict()}, f"{out_dir}/best_model.pt")


# Phase B â€“ Joint Fine-Tuning (unfreeze last 2 DeBERTa layers)

freeze_deberta(encoder, True)
unfreeze_last_n_layers(encoder, n=2)
params = [
    {"params": model.parameters(), "lr": 1e-3},
    {"params": [p for p in encoder.backbone.parameters() if p.requires_grad], "lr": 1e-5},
]
optB = torch.optim.AdamW(params)

for epoch in range(EPOCHS_B):
    model.train(); encoder.train()
    tot = 0.0
    for b in train_loader:
        batch = collate_build_batch(b, encoder)
        optB.zero_grad()
        logits = model(batch)
        loss = crit(logits, batch.y)
        loss.backward(); optB.step()
        tot += loss.item()
    acc, f1 = eval_loader(model, encoder, val_loader)
    print(f"[B] Epoch {epoch+1}/{EPOCHS_B} | loss={tot/len(train_loader):.4f} | val_acc={acc:.4f} | val_f1={f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        torch.save({"gnn": model.state_dict(), "enc": encoder.state_dict()}, f"{out_dir}/best_model.pt")


# Evaluation on Test Set

ckpt = torch.load(f"{out_dir}/best_model.pt", map_location=DEVICE)
model.load_state_dict(ckpt["gnn"]); encoder.load_state_dict(ckpt["enc"])
acc_test, f1_test = eval_loader(model, encoder, test_loader)
print(f"\nTest Accuracy: {acc_test:.4f} | Test F1: {f1_test:.4f}")



model.eval()
encoder.eval()

preds = []
for i, row in df_test.iterrows():
    ei, ew, Tr, Tb = build_edges_snapshot(encoder, row["rule"], row["body"])
    item = PairGraphExample(row["rule"], row["body"], 0, ei, ew, Tr, Tb)
    batch = collate_build_batch([item], encoder)
    with torch.no_grad():
        logits = model(batch)
        prob = torch.softmax(logits, dim=1)[0,1].item()
    preds.append(prob)

df_test["rule_violation"] = preds
df_test[["row_id", "rule_violation"]].to_csv("submission.csv", index=False)
print("submission.csv gÃ©nÃ©rÃ© avec", len(df_test), "lignes")


preds

