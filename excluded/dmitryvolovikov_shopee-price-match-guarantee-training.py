import pandas as pd
import numpy as np
import torch
import os
import random
    

seed=42

os.environ['PYTHONHASHSEED']=str(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)




device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from torch import nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import tqdm 
from tqdm import tqdm
from PIL import Image
import math
from transformers import AutoTokenizer, AutoModel
import joblib



TEST_MODE=True


train=pd.read_csv('/kaggle/input/shopee-product-matching/train.csv')
test=pd.read_csv('/kaggle/input/shopee-product-matching/test.csv')
sample=pd.read_csv('/kaggle/input/shopee-product-matching/sample_submission.csv')

train_img_dir='/kaggle/input/shopee-product-matching/train_images'
test_img_dir='/kaggle/input/shopee-product-matching/test_images'


train.nunique()


if TEST_MODE:
    #train=train.sample(n=100, random_state=seed).reset_index(drop=True)
    train=train[:32000]
else:
    train=train


label2id = {lg: i for i, lg in enumerate(sorted(train['label_group'].unique()))}
id2label = {i: lg for lg, i in label2id.items()}


train['class_id']=train['label_group'].map(label2id)





train_data, eval_data=train_test_split(train, test_size=0.2)
#stratify=train['class_id']


class ShopeeDataset(Dataset):
    def __init__(self, df, img_root, transform, train):
        self.df=df
        self.img_root=img_root
        self.transform=transform
        self.train=train
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row=self.df.iloc[idx]
        image_path=row['image']
        image_path_all=os.path.join(self.img_root, f'{image_path}')
        image=Image.open(image_path_all).convert('RGB')

        if self.transform is not None:
            image=self.transform(image)
        else:
            image=image
        
        if self.train:
            return({
                'image': image,
                'label': torch.tensor(row['class_id'], dtype=torch.long),
                'posting_id': row['posting_id']
            })
        else:
            return({
                'image': image,
                'posting_id': row['posting_id']
            })
        
        


transforms_train=v2.Compose([
    v2.Resize((224, 224)), 
    v2.RandomHorizontalFlip(),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transforms_eval=v2.Compose([
    v2.Resize((224,224)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transforms_test=v2.Compose([
    v2.Resize((224,224)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset=ShopeeDataset(train_data, train_img_dir, transform=transforms_train, train=True)
eval_dataset=ShopeeDataset(eval_data,train_img_dir, transform=transforms_eval, train=True)
test_dataset=ShopeeDataset(test, test_img_dir, transform=transforms_test, train=False)


train_dataloader=DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
eval_dataloader=DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=4)
test_dataloader=DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


EPOCHS=1


model=timm.create_model('eca_nfnet_l1', pretrained=True,  num_classes=0).to(device)


# === TEXT encoder (XLM-RoBERTa) ===
TEXT_MODEL_NAME = 'xlm-roberta-base'  # мультиязычный, оффлайн часто есть в кеше Kaggle
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, use_fast=True)
text_model = AutoModel.from_pretrained(TEXT_MODEL_NAME).to(device).eval()

# mean pooling по маске
def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()          # (B, L, 1)
    summed = (last_hidden_state * mask).sum(dim=1)       # (B, H)
    denom = mask.sum(dim=1).clamp(min=1e-6)              # (B, 1)
    return summed / denom




@torch.no_grad()
def build_text_embeddings_for_df(df, batch_size=256, max_len=64):
    """
    Возвращает L2-нормированные эмбеддинги заголовков (N, 768) в float16.
    Порядок совпадает с df.
    """
    titles = df['title'].fillna('').astype(str).tolist()
    embs = []
    for start in tqdm(range(0, len(titles), batch_size), desc="Embed/text"):
        batch = titles[start:start+batch_size]
        tok = text_tokenizer(batch, padding=True, truncation=True,
                             max_length=max_len, return_tensors='pt')
        tok = {k: v.to(device, non_blocking=True) for k, v in tok.items()}
        out = text_model(**tok)
        sent = mean_pooling(out.last_hidden_state, tok['attention_mask'])  # (B, 768)
        sent = nn.functional.normalize(sent, dim=1)
        embs.append(sent.cpu())
    embs = torch.cat(embs, dim=0).numpy().astype('float32')  # (N,768)
    # safety L2 + экономия RAM
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    return (embs / norms).astype('float16')

def topk_chunked_cos(embs_f16: np.ndarray, K: int, qbs: int = 128):
    """
    OOM-safe top-K по косинусу с помощью torch (без FAISS).
    embs_f16: (N, D) float16, L2-нормированные
    Возвращает sims, idxs: по (N, K) в float32 / int32
    """
    N, D = embs_f16.shape
    device_t = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    db = torch.from_numpy(embs_f16.astype('float32', copy=False)).to(device_t, non_blocking=True)

    K = min(K, N)
    idxs_list, sims_list = [], []
    for start in tqdm(range(0, N, qbs), desc="TopK (torch-chunk)"):
        q = db[start:start+qbs]                 # (qbs, D)
        S = torch.matmul(q, db.T)               # (qbs, N) — косинус
        vals, ids = torch.topk(S, k=K, dim=1, largest=True, sorted=True)
        idxs_list.append(ids.cpu().numpy().astype('int32'))
        sims_list.append(vals.cpu().numpy().astype('float32'))
        del S, vals, ids
        if device_t.type == 'cuda':
            torch.cuda.empty_cache()
    idxs = np.vstack(idxs_list)   # (N, K)
    sims = np.vstack(sims_list)   # (N, K)
    del db
    return sims, idxs



feat_dim = model.num_features 

embedding_head = nn.Sequential(
    nn.Linear(feat_dim, 512, bias=False),
    nn.BatchNorm1d(512)
).to(device)


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5, easy_margin=False):
        super().__init__()
        
        self.in_features=in_features
        self.out_features=out_features
        self.m=m
        self.s=s
        self.easy_margin=easy_margin

        self.weight=nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m=math.cos(self.m)
        self.sin_m=math.sin(self.m)

        self.th=math.cos(math.pi-m)
        self.mm=math.sin(math.pi-m)*m
        
        

    def forward(self, embeddings, labels):
        weights=nn.functional.normalize(self.weight)
        cosine=nn.functional.linear(embeddings, weights)
        sine=torch.sqrt(torch.clamp(1.0-cosine**2, min=1e-6))

        phi=cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1,1), 1.0)

        
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits = logits * self.s
        return logits


NUM_CLASSES = train['class_id'].nunique()


arcface_head=ArcFace(512, NUM_CLASSES, s=30.0, m=0.5).to(device)


criterion=torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    


optimizer=torch.optim.AdamW(list(model.parameters()) +
    list(embedding_head.parameters()) +
    list(arcface_head.parameters()), lr=3e-4, weight_decay=0.05)


scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


for epoch in range(1, EPOCHS+1):
    model.train()
    embedding_head.train()
    arcface_head.train()
    running_loss, num_correct, n=0.0,0,0
    pbar=tqdm(train_dataloader, desc='train', leave=False)
    for step, batch in enumerate(pbar):
        optimizer.zero_grad()
        X=batch['image'].to(device)
        y=batch['label'].to(device)
    
        feats=model(X)
        embeddings = embedding_head(feats)
        embeddings=nn.functional.normalize(embeddings, dim=1)
        logits=arcface_head(embeddings, y)
        loss=criterion(logits, y)
        loss.backward()
        
        
        optimizer.step()
        running_loss+=loss.item()*X.size(0)
        pbar.set_postfix(loss=running_loss/((step+1)*X.size(0)))

    scheduler.step()
# ===== Validation: build embeddings -> cosine-kNN -> global tau -> F1 =====
import numpy as np

model.eval(); embedding_head.eval(); arcface_head.eval()

@torch.no_grad()
def build_val_embeddings(loader):
    embs, ids, labels = [], [], []
    for b in tqdm(loader, desc="Embed/val"):
        x = b['image'].to(device, non_blocking=True)
        e = model(x)                         # (B, feat_dim)
        e = embedding_head(e)                # (B, 512)
        e = nn.functional.normalize(e, dim=1)
        embs.append(e.cpu())
        ids.extend(b['posting_id'])
        # важно: eval_dataset должен возвращать label (class_id)
        labels.extend(b['label'].cpu().numpy().tolist())
    embs = torch.cat(embs, dim=0).numpy()    # (N, 512)
    labels = np.array(labels)
    return embs, ids, labels

def build_preds(embs, ids, K=50, tau=0.50, mutual=True, cap=50):
    """
    embs: (N, D) L2-нормированные векторы
    ids:  список posting_id в том же порядке
    Возвращает dict: posting_id -> set(predicted_ids) (включая self), с ограничением cap.
    """
    # Косинусная матрица: для L2-нормированных это просто e @ e^T
    S = embs @ embs.T        # (N, N)
    N = S.shape[0]

    # Предвычислим top-K индексы для каждого i (быстро и экономно)
    topk_idx = np.argsort(-S, axis=1)[:, :K]  # индексы соседей по убыванию сходства

    preds = {}
    for i in range(N):
        cand = []
        for j in topk_idx[i]:
            if S[i, j] >= tau:
                if not mutual or i in topk_idx[j]:   # взаимность (mutual-kNN) уменьшает false-merge
                    cand.append(ids[j])
        # обязательно self-match
        if ids[i] not in cand:
            cand = [ids[i]] + cand
        # кап по условию соревна
        preds[ids[i]] = set(cand[:cap])
    return preds

def f1_matches(ids, labels, preds):
    """
    Средний F1 по каждому посту:
      - истина: все posting_id той же class_id
      - предсказание: preds[posting_id]
    """
    # сгруппируем истину: class_id -> set(posting_id)
    truth = {}
    for pid, g in zip(ids, labels):
        truth.setdefault(g, set()).add(pid)

    f1s = []
    for pid, g in zip(ids, labels):
        T = truth[g]
        P = preds[pid]
        inter = len(T & P)
        denom = len(T) + len(P)
        f1s.append( 2*inter/denom if denom > 0 else 0.0 )
    return float(np.mean(f1s))

# 1) эмбеддинги на валидации
val_embs, val_ids, val_labels = build_val_embeddings(eval_dataloader)
# === TEXT embeddings на валидации (в порядке eval_dataset.df) ===
val_text_embs = build_text_embeddings_for_df(eval_dataset.df, batch_size=256, max_len=64)  # (Nv,768) float16

# === top-K для image и text (шире cap, потом обрежем) ===
KQ = 100  # ширина кандидатов перед обрезкой до 50
sims_img, idxs_img = topk_chunked_cos(val_embs.astype('float16'), K=KQ, qbs=128)     # (Nv,KQ)
sims_txt, idxs_txt = topk_chunked_cos(val_text_embs,              K=KQ, qbs=256)     # (Nv,KQ)

def build_preds_fused_union(ids, idxs_img, sims_img, idxs_txt, sims_txt,
                            tau, alpha=0.7, K_cap=50, mutual=True):
    """
    Смешиваем косинусы: score = alpha*img + (1-alpha)*txt.
    (оба в [-1,1]) → приводим к [0,1] для стабильности.
    Кандидаты = объединение topK от image и text.
    """
    N = len(ids)
    preds = []
    for i in range(N):
        cand_idx = set(idxs_img[i]).union(set(idxs_txt[i]))
        # карты скорингов текущей строки
        map_img = {int(j): float(s) for j, s in zip(idxs_img[i], sims_img[i])}
        map_txt = {int(j): float(s) for j, s in zip(idxs_txt[i], sims_txt[i])}

        fused = []
        for j in cand_idx:
            si = (map_img.get(int(j), 0.0) + 1.0) / 2.0   # → [0,1]
            st = (map_txt.get(int(j), 0.0) + 1.0) / 2.0   # → [0,1]
            s  = alpha * si + (1.0 - alpha) * st
            fused.append((j, s))

        fused.sort(key=lambda x: -x[1])  # по убыванию
        # mutual — проверяем только по image (дёшево и достаточно)
        keep = []
        for j, s in fused:
            if s < tau: 
                continue
            if (not mutual) or (np.any(idxs_img[j] == i)):
                keep.append(ids[j])

        if ids[i] not in keep:
            keep = [ids[i]] + keep
        preds.append(set(keep[:K_cap]))
    # в формате, который ждёт f1_matches (dict)
    return {ids[i]: preds[i] for i in range(N)}

# Поиск (alpha, tau) по сетке
alphas = np.linspace(0.4, 0.9, 6)     # 0.40..0.90
taus   = np.linspace(0.20, 0.80, 31)  # 0.20..0.80
best_f1, best_tau, best_alpha = -1.0, None, None
for a in alphas:
    for t in taus:
        preds = build_preds_fused_union(val_ids, idxs_img, sims_img, idxs_txt, sims_txt,
                                        tau=float(t), alpha=float(a), K_cap=50, mutual=True)
        f1 = f1_matches(val_ids, val_labels, preds)
        if f1 > best_f1:
            best_f1, best_tau, best_alpha = f1, float(t), float(a)

print(f"[VAL FUSION] Best F1={best_f1:.4f} at tau={best_tau:.2f}, alpha={best_alpha:.2f}")

# (на всякий случай) L2-норм ещё раз, если вызывалась неявно
val_embs = val_embs / np.linalg.norm(val_embs, axis=1, keepdims=True)

# 2) поищем глобальный порог tau на сетке (стабильнее одного «подогнанного» фолда)
grid = np.linspace(0.20, 0.80, 31)  # шаг 0.02
best_f1, best_tau = -1.0, None
for tau in grid:
    preds = build_preds(val_embs, val_ids, K=50, tau=float(tau), mutual=True, cap=50)
    f1 = f1_matches(val_ids, val_labels, preds)
    if f1 > best_f1:
        best_f1, best_tau = f1, float(tau)

print(f"[VAL] Best F1={best_f1:.4f} at tau={best_tau:.2f}")

# (необязательно) Быстрый sanity-check: средний размер предсказанных групп
preds_best = build_preds(val_embs, val_ids, K=50, tau=best_tau, mutual=True, cap=50)
avg_group_size = np.mean([len(v) for v in preds_best.values()])
print(f"[VAL] Avg predicted group size: {avg_group_size:.2f}")
# === Save checkpoint(s) ===
SAVE_DIR = '/kaggle/working'
os.makedirs(SAVE_DIR, exist_ok=True)

ckpt = {
    'backbone_name': 'resnet50',
    'feat_dim': model.num_features,       # для resnet50 = 2048
    'emb_dim': 512,
    'num_classes': NUM_CLASSES,
    'arcface_cfg': {'s': 30.0, 'm': 0.5, 'easy_margin': False},
    'state_dict': {
        'backbone': model.state_dict(),
        'embedding_head': embedding_head.state_dict(),
        'arcface_head': arcface_head.state_dict(),
    },
    'label2id': label2id,                 # важно для повторного обучения/отчётов
    'best_tau': float(best_tau),          # пригодится для инференса
    'val_f1': float(best_f1),
    'epoch': EPOCHS,
}

torch.save(ckpt, os.path.join(SAVE_DIR, 'arcface_resnet50_shopee_ckpt.pth'))
print('[SAVE] Full checkpoint ->', os.path.join(SAVE_DIR, 'arcface_resnet50_shopee_ckpt.pth'))

# Лёгкий пакет для инференса (без arcface головы и без оптимизатора)
embed_pkg = {
    'backbone_name': 'resnet50',
    'feat_dim': model.num_features,
    'emb_dim': 512,
    'state_dict': {
        'backbone': model.state_dict(),
        'embedding_head': embedding_head.state_dict(),
    },
    'best_tau': float(best_tau),
}
torch.save(embed_pkg, os.path.join(SAVE_DIR, 'embedding_extractor.pth'))
print('[SAVE] Embedding extractor ->', os.path.join(SAVE_DIR, 'embedding_extractor.pth'))

        





#need inference notebook

