import os
import math
import random
from typing import List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter



# ------------------------- Reproducibility -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)




# ------------------------- Utilities -------------------------
VOCAB = {"A":1, "C":2, "G":3, "U":4, "N":5}  # 0 reserved for PAD


def seq_to_ids(seq: str, max_len: int) -> List[int]:
    ids = [VOCAB.get(ch, VOCAB['N']) for ch in seq.upper()][:max_len]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return ids

# dot-bracket to contact map (kept for completeness)
def dotbracket_to_contacts(dot: str) -> np.ndarray:
    stack = []
    n = len(dot)
    mat = np.zeros((n, n), dtype=np.uint8)
    pairs = {')': '(', ']': '[', '}': '{'}
    openers = set(['(', '[', '{', '<'])
    closers = set([')', ']', '}', '>'])
    for i, ch in enumerate(dot):
        if ch in openers:
            stack.append((ch, i))
        elif ch in closers:
            if not stack:
                continue
            top_ch, top_i = stack.pop()
            expected = pairs.get(ch, '(')
            if top_ch != expected:
                pass
            mat[top_i, i] = 1
            mat[i, top_i] = 1
    return mat




# ------------------------- Dataset -------------------------
class RNADataset(Dataset):
    """Dataset built from a sequences dataframe. The contact maps are attached
    later via a custom collate function that uses precomputed contacts_by_id.
    Expects seq_df containing column 'target_id' and 'sequence'."""
    def __init__(self, seq_df: pd.DataFrame, max_len: int = 512):
        self.df = seq_df.reset_index(drop=True)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row['sequence']).strip()
        seq_ids = seq_to_ids(seq, self.max_len)
        seq_len = min(len(seq), self.max_len)
        return {
            'id': row['target_id'],
            'seq_ids': torch.LongTensor(seq_ids),
            'seq_len': seq_len
        }

# collate that pads and attaches contacts_by_id provided externally
def collate_with_contacts(batch, contacts_map, max_len):
    ids = [b['id'] for b in batch]
    seq_ids = torch.stack([b['seq_ids'] for b in batch])
    seq_lens = torch.tensor([b['seq_len'] for b in batch], dtype=torch.long)
    B, L = seq_ids.shape
    contacts = torch.zeros((B, L, L), dtype=torch.float32)
    for i, rid in enumerate(ids):
        key = str(rid)
        if key in contacts_map:
            mat = contacts_map[key]
            # ensure shape matches L
            if mat.shape[0] >= L:
                contacts[i] = torch.from_numpy(mat[:L, :L]).float()
            else:
                tmp = np.zeros((L, L), dtype=np.uint8)
                tmp[:mat.shape[0], :mat.shape[1]] = mat
                contacts[i] = torch.from_numpy(tmp).float()
        else:
            # fallback to zeros
            contacts[i] = torch.zeros((L, L), dtype=torch.float32)
    return {'id': ids, 'seq_ids': seq_ids, 'seq_lens': seq_lens, 'contact': contacts}




# ------------------------- Model -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)

class Encoder1D(nn.Module):
    def __init__(self, vocab_size=6, emb_dim=128, n_heads=8, ff_dim=256, n_layers=3, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(emb_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, seq_ids, src_key_padding_mask=None):
        x = self.embedding(seq_ids) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_enc(x)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.layer_norm(x)
        return x

class Pairwise2DUNet(nn.Module):
    def __init__(self, in_channels=256, base_channels=64):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, base_channels, 3, padding=1), nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(base_channels, base_channels*2, 3, padding=1), nn.BatchNorm2d(base_channels*2), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1), nn.BatchNorm2d(base_channels*4), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1), nn.BatchNorm2d(base_channels*2), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(base_channels*2, base_channels, 3, padding=1), nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True))
        self.out_conv = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = F.max_pool2d(c1, 2)
        c2 = self.conv2(p1)
        p2 = F.max_pool2d(c2, 2)
        c3 = self.conv3(p2)
        u2 = self.up2(c3)
        if u2.size() != c2.size():
            u2 = F.interpolate(u2, size=c2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, c2], dim=1))
        u1 = self.up1(d2)
        if u1.size() != c1.size():
            u1 = F.interpolate(u1, size=c1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, c1], dim=1))
        out = self.out_conv(d1)
        return out.squeeze(1)

class RNAContactPredictor(nn.Module):
    def __init__(self, enc: Encoder1D, pairwise_channels=256):
        super().__init__()
        self.enc = enc
        emb_dim = enc.embedding.embedding_dim
        self.proj = nn.Linear(emb_dim, pairwise_channels // 2)
        self.pair_net = Pairwise2DUNet(in_channels=pairwise_channels, base_channels=32)

    def forward(self, seq_ids, seq_lens=None):
        src_key_padding_mask = (seq_ids == 0)
        h = self.enc(seq_ids, src_key_padding_mask=src_key_padding_mask)
        p = self.proj(h)
        B, L, C = p.shape
        a = p.unsqueeze(2)
        b = p.unsqueeze(1)
        pair = torch.cat([a.expand(-1, -1, L, -1), b.expand(-1, L, -1, -1)], dim=-1)
        pair = pair.permute(0, 3, 1, 2).contiguous()
        logits = self.pair_net(pair)
        return logits




# ------------------------- Contact building from coordinates -------------------------
THRESH = 8.0  # angstroms
MAX_COORD_PER_ROW = 100  # safe cap for flattened rows


def parse_struct_and_pos_from_id(id_str):
    parts = str(id_str).rsplit('_', maxsplit=1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    return str(id_str), None


def build_contacts_from_labels(labels_df: pd.DataFrame, seq_df: Optional[pd.DataFrame] = None, max_len: int = 200, threshold: float = THRESH):
    contacts_by_id = {}
    coords_by_id = {}
    cols = labels_df.columns.tolist()

    # detect flattened-per-structure format (many x_k columns)
    x_columns = sorted([c for c in cols if c.lower().startswith('x_')], key=lambda s: int(s.split('_')[1]) if '_' in s and s.split('_')[1].isdigit() else 0)

    if len(x_columns) > 1 and labels_df.shape[0] > 0 and labels_df['ID'].nunique() == labels_df.shape[0]:
        # one row per structure, columns x_1..x_n
        for _, row in labels_df.iterrows():
            sid = row['ID']
            coords = []
            k = 1
            while True:
                xk = f'x_{k}'
                yk = f'y_{k}'
                zk = f'z_{k}'
                if xk in labels_df.columns and yk in labels_df.columns and zk in labels_df.columns:
                    x = row[xk]; y = row[yk]; z = row[zk]
                    if pd.isna(x) or pd.isna(y) or pd.isna(z):
                        break
                    coords.append((float(x), float(y), float(z)))
                    k += 1
                else:
                    break
            coords_by_id[str(sid)] = coords
    else:
        # one row per residue
        temp = defaultdict(dict)
        for _, row in labels_df.iterrows():
            sid_full = row['ID']
            struct_id, pos = parse_struct_and_pos_from_id(sid_full)
            if 'resid' in labels_df.columns:
                try:
                    pos = int(row['resid'])
                except:
                    pass
            # determine coordinate columns
            if 'x_1' in labels_df.columns:
                x = row['x_1']; y = row['y_1']; z = row['z_1']
            elif 'x' in labels_df.columns and 'y' in labels_df.columns and 'z' in labels_df.columns:
                x = row['x']; y = row['y']; z = row['z']
            else:
                # try to find first triplet x_k,y_k,z_k
                found = False
                for c in labels_df.columns:
                    if c.lower().startswith('x_') and '_' in c:
                        base = c.split('_',1)[1]
                        xc, yc, zc = f'x_{base}', f'y_{base}', f'z_{base}'
                        if xc in labels_df.columns and yc in labels_df.columns and zc in labels_df.columns:
                            x = row[xc]; y = row[yc]; z = row[zc]
                            found = True
                            break
                if not found:
                    continue
            if pd.isna(x) or pd.isna(y) or pd.isna(z):
                continue
            if pos is None:
                pos = len(temp[struct_id]) + 1
            temp[struct_id][int(pos)] = (float(x), float(y), float(z))

        for struct_id, d in temp.items():
            ordered = [d[k] for k in sorted(d.keys())]
            coords_by_id[str(struct_id)] = ordered

    # compute contact maps
    for sid, coords in coords_by_id.items():
        L = min(len(coords), max_len)
        if L == 0:
            contacts_by_id[sid] = np.zeros((max_len, max_len), dtype=np.uint8)
            continue
        arr = np.array(coords[:L])
        dists = np.sqrt(np.sum((arr[:, None, :] - arr[None, :, :])**2, axis=-1))
        contact = (dists <= threshold).astype(np.uint8)
        np.fill_diagonal(contact, 0)
        if contact.shape[0] < max_len:
            pad = np.zeros((max_len, max_len), dtype=np.uint8)
            pad[:L, :L] = contact
            contact = pad
        contacts_by_id[sid] = contact
    return contacts_by_id, coords_by_id




# ------------------------- Metrics & Loss -------------------------

def contact_metrics(pred_logits: torch.Tensor, target: torch.Tensor, seq_lens: torch.Tensor, threshold: float = 0.5):
    preds = (torch.sigmoid(pred_logits) > threshold).int()
    t = target.int()
    B, L, _ = preds.shape
    results = {'tp':0, 'fp':0, 'fn':0}
    for i in range(B):
        n = int(seq_lens[i].item())
        p = preds[i, :n, :n]
        g = t[i, :n, :n]
        diag = torch.eye(n, dtype=torch.bool, device=preds.device)
        p = p & (~diag)
        g = g & (~diag)
        # upper triangle mask
        mask = torch.triu(torch.ones((n,n), dtype=torch.bool, device=preds.device), diagonal=1)
        p_u = p[mask]
        g_u = g[mask]
        tp = int(((p_u==1) & (g_u==1)).sum().item())
        fp = int(((p_u==1) & (g_u==0)).sum().item())
        fn = int(((p_u==0) & (g_u==1)).sum().item())
        results['tp'] += tp
        results['fp'] += fp
        results['fn'] += fn
    tp = results['tp']; fp = results['fp']; fn = results['fn']
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp':tp, 'fp':fp, 'fn':fn}


def masked_bce_loss_logits(logits, targets, seq_lens, pos_weight_tensor=None):
    B, L, _ = logits.shape
    losses = []
    for i in range(B):
        n = int(seq_lens[i].item())
        log = logits[i, :n, :n]
        tar = targets[i, :n, :n]
        mask = torch.triu(torch.ones((n, n), dtype=torch.bool, device=logits.device), diagonal=1)
        log_u = log[mask]
        tar_u = tar[mask]
        if pos_weight_tensor is not None:
            loss = F.binary_cross_entropy_with_logits(log_u, tar_u, pos_weight=pos_weight_tensor)
        else:
            loss = F.binary_cross_entropy_with_logits(log_u, tar_u)
        losses.append(loss)
    return torch.stack(losses).mean()




# ------------------------- Training Loop -------------------------

def train_one_epoch_comp(model, dataloader, optimizer, scaler, device, epoch, writer=None, accumulation_steps=1, pos_weight=None):
    model.train()
    total_loss = 0.0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    optimizer.zero_grad()
    for step, batch in pbar:
        seq_ids = batch['seq_ids'].to(device)
        seq_lens = batch['seq_lens'].to(device)
        contacts = batch['contact'].to(device)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(seq_ids, seq_lens)
            loss = masked_bce_loss_logits(logits, contacts, seq_lens, pos_weight_tensor=pos_weight)
            loss = loss / accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        pbar.set_description(f"Epoch {epoch} loss: {total_loss/(step+1):.4f}")

    avg_loss = total_loss / len(dataloader)
    if writer:
        writer.add_scalar('train/loss', avg_loss, epoch)
    return avg_loss


def validate_comp(model, dataloader, device, epoch, writer=None, pos_weight=None):
    model.eval()
    total_loss = 0.0
    agg_tp = agg_fp = agg_fn = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Valid'):
            seq_ids = batch['seq_ids'].to(device)
            seq_lens = batch['seq_lens'].to(device)
            contacts = batch['contact'].to(device)
            logits = model(seq_ids, seq_lens)
            loss = masked_bce_loss_logits(logits, contacts, seq_lens, pos_weight_tensor=pos_weight)
            total_loss += loss.item()
            metrics = contact_metrics(logits, contacts, seq_lens)
            agg_tp += metrics['tp']; agg_fp += metrics['fp']; agg_fn += metrics['fn']

    precision = agg_tp / (agg_tp + agg_fp) if agg_tp + agg_fp > 0 else 0.0
    recall = agg_tp / (agg_tp + agg_fn) if agg_tp + agg_fn > 0 else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
    avg_loss = total_loss / len(dataloader)
    if writer:
        writer.add_scalar('valid/loss', avg_loss, epoch)
        writer.add_scalar('valid/precision', precision, epoch)
        writer.add_scalar('valid/recall', recall, epoch)
        writer.add_scalar('valid/f1', f1, epoch)
    return {'loss': avg_loss, 'precision': precision, 'recall': recall, 'f1': f1}




# ------------------------- Main Script -------------------------
if __name__ == '__main__':
    # ------------------------- Competition-aware Main Script -------------------------
    # Paths - adjust to your Kaggle dataset paths
    MAX_LEN = 200  # safe default for Stanford RNA dataset
    BATCH_SIZE = 8
    N_EPOCHS = 20
    LR = 3e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    ACCUM_STEPS = 1
    MIXED_PREC = True if DEVICE == 'cuda' else False

    # competition files (adjust if mounted differently)
    DATA_DIR = '/kaggle/input/stanford-rna-3d-folding'
    TRAIN_SEQ = os.path.join(DATA_DIR, 'train_sequences.csv')
    TRAIN_LABELS = os.path.join(DATA_DIR, 'train_labels.csv')
    VAL_SEQ = os.path.join(DATA_DIR, 'validation_sequences.csv')
    VAL_LABELS = os.path.join(DATA_DIR, 'validation_labels.csv')
    TEST_SEQ = os.path.join(DATA_DIR, 'test_sequences.csv')
    SAMPLE_SUB = os.path.join(DATA_DIR, 'sample_submission.csv')

    OUT_DIR = 'checkpoints'
    os.makedirs(OUT_DIR, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(OUT_DIR, 'runs'))

    # ------------------------- Load CSVs -------------------------
    train_seq_df = pd.read_csv(TRAIN_SEQ)
    train_lbl_df = pd.read_csv(TRAIN_LABELS)
    val_seq_df = pd.read_csv(VAL_SEQ)
    val_lbl_df = pd.read_csv(VAL_LABELS)

    # Build contacts_by_id from coordinate labels
    contacts_by_id, coords_by_id = build_contacts_from_labels(train_lbl_df, seq_df=train_seq_df, max_len=MAX_LEN, threshold=THRESH)
    val_contacts_by_id, val_coords_by_id = build_contacts_from_labels(val_lbl_df, seq_df=val_seq_df, max_len=MAX_LEN, threshold=THRESH)
    print('Built contacts for', len(contacts_by_id), 'train targets and', len(val_contacts_by_id), 'val targets')

    # create dataset objects
    train_ds = RNADataset(train_seq_df, max_len=MAX_LEN)
    val_ds = RNADataset(val_seq_df, max_len=MAX_LEN)

    # DataLoaders using our custom collate
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda batch: collate_with_contacts(batch, contacts_by_id, MAX_LEN),
                              num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=lambda batch: collate_with_contacts(batch, val_contacts_by_id, MAX_LEN),
                            num_workers=2, pin_memory=True)

    # ------------------------- Model (small for competition) -------------------------
    enc = Encoder1D(vocab_size=len(VOCAB)+1, emb_dim=64, n_heads=4, ff_dim=256, n_layers=2, max_len=MAX_LEN, dropout=0.1)
    model = RNAContactPredictor(enc, pairwise_channels=128)
    model = model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler() if MIXED_PREC else None

    # Precompute pos_weight from training contacts
    total_pos = 0
    total_pairs = 0
    for sid, mat in contacts_by_id.items():
        # estimate n from matrix (count nonzero rows)
        n = int(np.count_nonzero(np.sum(mat, axis=1)) + np.sum(np.any(mat,axis=0) | np.any(mat,axis=1)))
        n = min(MAX_LEN, mat.shape[0])
        tri = n * (n - 1) // 2
        total_pairs += tri
        total_pos += int(mat[:n, :n].sum() / 2)  # since symmetric
    pos_weight = torch.tensor(((total_pairs - total_pos) / (total_pos + 1e-6)), dtype=torch.float32, device=DEVICE)
    print(f"Computed pos_weight={pos_weight.item():.4f} (pos={total_pos}, pairs={total_pairs})")

    # ------------------------- Training loop -------------------------
    best_f1 = 0.0
    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_one_epoch_comp(model, train_loader, optimizer, scaler, DEVICE, epoch, writer, accumulation_steps=ACCUM_STEPS, pos_weight=pos_weight)
        val_stats = validate_comp(model, val_loader, DEVICE, epoch, writer, pos_weight=pos_weight)
        scheduler.step(val_stats['loss'])

        print(f"Epoch {epoch} -> train_loss: {train_loss:.4f}, val_loss: {val_stats['loss']:.4f}, f1: {val_stats['f1']:.4f}")

        # checkpoint best
        if val_stats['f1'] > best_f1:
            best_f1 = val_stats['f1']
            ckpt_path = os.path.join(OUT_DIR, f'best_model_epoch{epoch}_f1{best_f1:.4f}.pt')
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'f1': best_f1}, ckpt_path)
            print(f"Saved best model to {ckpt_path}")




    # ------------------------- Inference / submission -------------------------
    # load best model if exists
    ckpts = [p for p in os.listdir(OUT_DIR) if p.endswith('.pt')]
    if ckpts:
        latest = sorted(ckpts)[-1]
        print('Loading checkpoint', latest)
        ckpt = torch.load(os.path.join(OUT_DIR, latest), map_location=DEVICE)
        model.load_state_dict(ckpt['model_state'])

    # prepare test loader
    test_seq_df = pd.read_csv(TEST_SEQ)
    test_ds = RNADataset(test_seq_df, max_len=MAX_LEN)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=lambda b: collate_with_contacts(b, {}, MAX_LEN), num_workers=1)

    # produce submission rows incrementally to save memory
    submission_rows = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Test'):
            seq_ids = batch['seq_ids'].to(DEVICE)
            seq_lens = batch['seq_lens'].to(DEVICE)
            ids = batch['id']
            logits = model(seq_ids, seq_lens)[0]  # [L,L]
            prob = torch.sigmoid(logits)
            n = int(seq_lens[0].item())
            for i in range(n):
                for j in range(i+1, n):
                    submission_rows.append([ids[0], i, j, int((prob[i, j] > 0.5).item())])

    submission_df = pd.DataFrame(submission_rows, columns=['id', 'position_i', 'position_j', 'is_paired'])
    submission_df.to_csv('submission.csv', index=False)
    print('Wrote submission.csv')

    writer.close()

# End of file


