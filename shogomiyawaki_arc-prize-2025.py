# =========================================================
# ARC Prize 2025 — Transformer baseline (full working cell)
# Load → Train → Predict → submission.json
# =========================================================

import os, json, random, time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Detect data directory
# -----------------------
def exists(p): 
    try: return Path(p).exists()
    except: return False

KAGGLE_DIR = "/kaggle/input/arc-prize-2025"
LOCAL_DIR  = "/mnt/data"
BASE = KAGGLE_DIR if exists(KAGGLE_DIR) else LOCAL_DIR

PATH_TRAIN_CHAL = BASE + "/arc-agi_training_challenges.json"
PATH_EVAL_CHAL  = BASE + "/arc-agi_evaluation_challenges.json"
PATH_SAMPLE_SUB = BASE + "/sample_submission.json"

print("Base path:", BASE)

# -----------------------
# Load JSON
# -----------------------
def load_json(path):
    with open(path,"r") as f:
        return json.load(f)

train_challenges = load_json(PATH_TRAIN_CHAL)
eval_challenges  = load_json(PATH_EVAL_CHAL)
sample_sub       = load_json(PATH_SAMPLE_SUB)

print("Train tasks:", len(train_challenges))
print("Eval tasks :", len(eval_challenges))

# =========================================================
# Token definitions
# =========================================================
PAD = 0
COLOR_OFFSET = 1        # grid色は 0..9 → 1..10 にシフト（PAD=0と衝突しない）
HW_OFFSET = 20          # 高さ/幅は 20..70 に配置（SPECIAL群と離す）
SPECIAL_OFFSET = 100    # 特殊トークンの基点

SPECIAL = {
    "<PAD>": PAD,
    "<BOQ>": SPECIAL_OFFSET + 0,
    "<EOQ>": SPECIAL_OFFSET + 1,
    "<EX>":  SPECIAL_OFFSET + 2,
    "<IN>":  SPECIAL_OFFSET + 3,
    "<OUT>": SPECIAL_OFFSET + 4,
    "<TST>": SPECIAL_OFFSET + 5,
    "|":     SPECIAL_OFFSET + 6,
    ";":     SPECIAL_OFFSET + 7,
    "<EOG>": SPECIAL_OFFSET + 8,
}

VOCAB = 256  # 余裕を持たせる

print("Vocab max index:", max(SPECIAL.values()), "Vocab size:", VOCAB)


def encode_grid(grid):
    H, W = len(grid), len(grid[0])
    seq = [H + HW_OFFSET, W + HW_OFFSET, SPECIAL["|"]]
    for r in range(H):
        seq += [(v + COLOR_OFFSET) for v in grid[r]]
        if r < H-1: seq.append(SPECIAL[";"])
    return seq


def encode_context(task, test=None):
    toks = [SPECIAL["<BOQ>"]]
    for ex in task["train"]:
        toks += [SPECIAL["<EX>"], SPECIAL["<IN>"]] + encode_grid(ex["input"])
        toks += [SPECIAL["<OUT>"]] + encode_grid(ex["output"])
    if test is not None:
        toks += [SPECIAL["<TST>"], SPECIAL["<IN>"]] + encode_grid(test)
    toks += [SPECIAL["<EOQ>"]]
    return toks

def decode_grid(tokens, si):
    try:
        H = tokens[si] - HW_OFFSET
        W = tokens[si+1] - HW_OFFSET
        if not (1 <= H <= 50 and 1 <= W <= 50): return None
        i = si + 3
        grid, row = [], []
        while len(grid) < H and i < len(tokens):
            t = tokens[i]; i += 1
            if t == SPECIAL[";"]:
                if len(row) != W: return None
                grid.append(row); row = []
                continue
            if t == SPECIAL["<EOG>"]: break
            color = t - COLOR_OFFSET
            if not (0 <= color <= 9): return None
            row.append(color)
            if len(row) == W:
                if len(grid) < H-1 and i < len(tokens) and tokens[i] == SPECIAL[";"]:
                    i += 1
                grid.append(row); row = []
        return grid if len(grid) == H else None
    except:
        return None


# =========================================================
# Dataset
# =========================================================
class ARCDataset(Dataset):
    def __init__(self, challenges, max_tasks=None):
        self.items=[]
        ids=list(challenges.keys())
        if max_tasks: ids=ids[:max_tasks]
        for tid in ids:
            tk=challenges[tid]
            context = encode_context(tk)
            for ex in tk["train"]:
                out=ex["output"]
                dec_in=[SPECIAL["<OUT>"]]+encode_grid(out)
                dec_out=dec_in[1:]+[SPECIAL["<EOG>"]]
                self.items.append((context,dec_in,dec_out))
        random.shuffle(self.items)

    def __len__(self): return len(self.items)
    def __getitem__(self,idx): return self.items[idx]

def pad_batch(seqs,pad=0):
    L=max(len(s) for s in seqs)
    return [s+[pad]*(L-len(s)) for s in seqs]

def _clamp_to_vocab(seq, vocab=VOCAB):
    return [int(min(max(t, 0), vocab-1)) for t in seq]

def pad_batch(seqs, pad=PAD):
    L = max(len(s) for s in seqs)
    return [s + [pad]*(L-len(s)) for s in seqs]

def collate(batch):
    enc, din, dout = zip(*batch)
    enc  = [ _clamp_to_vocab(s) for s in enc  ]
    din  = [ _clamp_to_vocab(s) for s in din  ]
    dout = [ _clamp_to_vocab(s) for s in dout ]
    return (
        torch.tensor(pad_batch(enc),  dtype=torch.long),
        torch.tensor(pad_batch(din),  dtype=torch.long),
        torch.tensor(pad_batch(dout), dtype=torch.long),
    )


# =========================================================
# Transformer
# =========================================================
class ARCTransformer(nn.Module):
    def __init__(self, d=256, h=8, n=6, ff=1024, p=0.1, max_len=4096):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d, padding_idx=PAD)  # ← ここ重要
        self.pos = nn.Embedding(max_len, d)
        enc = nn.TransformerEncoderLayer(d, h, ff, dropout=p, batch_first=True)
        dec = nn.TransformerDecoderLayer(d, h, ff, dropout=p, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, n)
        self.dec = nn.TransformerDecoder(dec, n)
        self.lm  = nn.Linear(d, VOCAB)

    def forward(self,enc,dec_in):
        B,L=enc.size()
        pos_e=self.pos(torch.arange(L,device=enc.device))[None]
        pos_d=self.pos(torch.arange(dec_in.size(1),device=dec_in.device))[None]
        e=self.enc(self.emb(enc)+pos_e)
        m=nn.Transformer.generate_square_subsequent_mask(dec_in.size(1)).to(dec_in.device)
        d=self.dec(self.emb(dec_in)+pos_d, e, tgt_mask=m)
        return self.lm(d)

    @torch.no_grad()
    def beam(self,enc,max_len=400,beam=5,topk=8,ret=2):
        beams=[(torch.tensor([[SPECIAL["<OUT>"]]],device=enc.device),0.0)]
        for _ in range(max_len):
            nb=[]
            for s,sc in beams:
                lp=torch.log_softmax(self.forward(enc,s)[:,-1,:],dim=-1)[0]
                v,i=torch.topk(lp,k=topk)
                for val,idx in zip(v.tolist(),i.tolist()):
                    ns=torch.cat([s,torch.tensor([[idx]],device=enc.device)],1)
                    nb.append((ns,sc+val))
            nb.sort(key=lambda x:x[1],reverse=True)
            beams=nb[:beam]
            done=[b for b in beams if b[0][0,-1].item()==SPECIAL["<EOG>"]]
            if len(done)>=ret: break
        return [b[0][0].tolist() for b in beams[:ret]]

# =========================================================
# Train
# =========================================================
device="cuda" if torch.cuda.is_available() else "cpu"
print("device:",device)

MAX_TASKS=200
BATCH=8
EPOCHS=1

train_ds=ARCDataset(train_challenges,max_tasks=MAX_TASKS)
train_loader=DataLoader(train_ds,batch_size=BATCH,shuffle=True,collate_fn=collate)
print("Train samples:",len(train_ds))

model=ARCTransformer().to(device)
opt=torch.optim.AdamW(model.parameters(),lr=2e-4)
loss_fn=nn.CrossEntropyLoss(ignore_index=0,label_smoothing=0.1)

for ep in range(EPOCHS):
    model.train(); t0=time.time()
    for step,(enc,din,dout) in enumerate(train_loader,1):
        enc,din,dout=enc.to(device),din.to(device),dout.to(device)
        loss=loss_fn(model(enc,din).reshape(-1,VOCAB),dout.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step%100==0:
            print(f"[ep{ep+1}]step{step} loss={loss.item():.4f}")
    print("Epoch done",time.time()-t0)

torch.save(model.state_dict(),"arc_model.pt")
print("✅ Saved model")

# =========================================================
# Predict eval → submission.json
# =========================================================
def predict_task(t):
    inp=t["test"][0]["input"]
    enc=encode_context(t,inp)
    enc=torch.tensor(enc).unsqueeze(0).to(device)
    toks=model.beam(enc,ret=2)
    outs=[]
    for tk in toks:
        si=tk.index(SPECIAL["<OUT>"])+1 if SPECIAL["<OUT>"] in tk else 0
        g=decode_grid(tk,si)
        if g is not None: outs.append(g)
    while len(outs)<2: outs.append(inp)
    return outs[0],outs[1]

sub={}
for tid,t in eval_challenges.items():
    a1,a2=predict_task(t)
    sub[tid]=[{"attempt_1":a1,"attempt_2":a2}]

with open("submission.json","w") as f:
    json.dump(sub,f)

print("✅ submission.json written")


