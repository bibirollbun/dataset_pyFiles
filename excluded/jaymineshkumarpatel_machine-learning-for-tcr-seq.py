import os
os.environ["TORCH_USE_CUDA_DSA"] = "1"



# AIRR-MLðŸ§¬25: Deep CNN + Attention MIL on GPU

import sys
import glob
from collections import defaultdict
from typing import List, Tuple, Iterator, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Basic utilities
# =========================

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
PAD_IDX = len(AMINO_ACIDS)  # padding token index
VOCAB_SIZE = len(AMINO_ACIDS) + 1  # +1 for PAD

def load_data_generator(
    data_dir: str,
    metadata_filename: str = "metadata.csv"
) -> Iterator[Union[Tuple[str, pd.DataFrame, bool], Tuple[str, pd.DataFrame]]]:
    metadata_path = os.path.join(data_dir, metadata_filename)
    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        for row in metadata_df.itertuples(index=False):
            file_path = os.path.join(data_dir, row.filename)
            try:
                df = pd.read_csv(file_path, sep="\t")
                yield row.repertoire_id, df, bool(row.label_positive)
            except FileNotFoundError:
                print(f"Warning: missing file '{row.filename}'")
    else:
        for file_path in sorted(glob.glob(os.path.join(data_dir, "*.tsv"))):
            try:
                df = pd.read_csv(file_path, sep="\t")
                filename = os.path.basename(file_path)
                yield filename, df
            except Exception as e:
                print(f"Warning: error reading '{file_path}': {e}")


def load_full_dataset(data_dir: str) -> pd.DataFrame:
    metadata_path = os.path.join(data_dir, "metadata.csv")
    data_loader = load_data_generator(data_dir=data_dir)
    dfs = []
    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        total = len(metadata_df)
        for rep_id, df, label in tqdm(data_loader, total=total, desc="Loading full dataset"):
            df["ID"] = rep_id
            df["label_positive"] = label
            dfs.append(df)
    else:
        tsv_files = glob.glob(os.path.join(data_dir, "*.tsv"))
        total = len(tsv_files)
        for fname, df in tqdm(data_loader, total=total, desc="Loading full dataset"):
            rep_id = os.path.basename(fname).replace(".tsv", "")
            df["ID"] = rep_id
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def get_repertoire_ids(data_dir: str) -> List[str]:
    metadata_path = os.path.join(data_dir, "metadata.csv")
    if os.path.exists(metadata_path):
        meta = pd.read_csv(metadata_path)
        return meta["repertoire_id"].tolist()
    tsv_files = glob.glob(os.path.join(data_dir, "*.tsv"))
    return [os.path.basename(f).replace(".tsv", "") for f in sorted(tsv_files)]


def validate_dirs_and_files(train_dir: str, test_dirs: List[str], out_dir: str) -> None:
    assert os.path.isdir(train_dir), f"Train dir {train_dir} missing"
    assert os.path.isfile(os.path.join(train_dir, "metadata.csv")), "metadata.csv missing in train"
    assert glob.glob(os.path.join(train_dir, "*.tsv")), "No .tsv in train dir"

    for td in test_dirs:
        assert os.path.isdir(td), f"Test dir {td} missing"
        assert glob.glob(os.path.join(td, "*.tsv")), f"No .tsv in test dir {td}"

    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "tmp.test")
    with open(tmp, "w") as f:
        f.write("ok")
    os.remove(tmp)


def get_dataset_pairs(train_root: str, test_root: str) -> List[Tuple[str, List[str]]]:
    test_groups = defaultdict(list)
    for tname in sorted(os.listdir(test_root)):
        if not tname.startswith("test_dataset_"):
            continue
        base = tname.replace("test_dataset_", "").split("_")[0]
        test_groups[base].append(os.path.join(test_root, tname))
    pairs = []
    for tname in sorted(os.listdir(train_root)):
        if not tname.startswith("train_dataset_"):
            continue
        base = tname.replace("train_dataset_", "")
        train_path = os.path.join(train_root, tname)
        pairs.append((train_path, test_groups.get(base, [])))
    return pairs


def save_tsv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def concatenate_output_files(out_dir: str) -> pd.DataFrame:
    preds_files = sorted(glob.glob(os.path.join(out_dir, "*_test_predictions.tsv")))
    seq_files = sorted(glob.glob(os.path.join(out_dir, "*_important_sequences.tsv")))
    dfs = []
    for f in preds_files + seq_files:
        try:
            dfs.append(pd.read_csv(f, sep="\t"))
        except Exception as e:
            print(f"Warning reading {f}: {e}")
    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
    else:
        all_df = pd.DataFrame(columns=["ID","dataset","label_positive_probability","junction_aa","v_call","j_call"])
    for col in ["label_positive_probability","junction_aa","v_call","j_call"]:
        if col in all_df.columns:
            all_df[col] = all_df[col].fillna(-999.0)
    out_path = os.path.join(out_dir, "submissions.csv")
    all_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with shape {all_df.shape}")
    return all_df

# =========================
# Sequence encoding
# =========================

def encode_sequence(seq: str, max_len: int) -> List[int]:
    if not isinstance(seq, str):
        seq = ""
    seq = seq.strip()
    ids = [AA_TO_IDX.get(ch, PAD_IDX) for ch in seq][:max_len]
    if len(ids) < max_len:
        ids += [PAD_IDX] * (max_len - len(ids))
    return ids


def build_repertoire_tensors(
    data_dir: str,
    max_seqs_per_rep: int = 512,
    max_len: int = 25,
    for_training: bool = True
):
    """
    For each repertoire:
      - sample up to max_seqs_per_rep sequences
      - encode junction_aa as integer tokens
    Returns:
      rep_ids: list of repertoire IDs
      X: tensor [N, max_seqs, max_len]
      y: tensor [N] or None
      per_rep_seq_lists: list of sequence DataFrames (for later importance scoring)
    """
    loader = load_data_generator(data_dir=data_dir)
    metadata_path = os.path.join(data_dir, "metadata.csv")
    has_meta = os.path.exists(metadata_path)

    rep_ids = []
    labels = []
    tensors = []
    rep_seq_dfs = []

    # determine quantile of lengths to set max_len adaptively if desired
    # (here we use provided max_len for simplicity)

    for item in tqdm(loader, desc=f"Building repertoires ({'train' if for_training else 'test'})"):
        if has_meta:
            rep_id, df, label = item
        else:
            rep_file, df = item
            rep_id = os.path.basename(rep_file).replace(".tsv","")
            label = None

        # drop missing junction_aa
        df = df.dropna(subset=["junction_aa"])
        if df.empty:
            continue

        # sample or truncate sequences
        if len(df) > max_seqs_per_rep:
            df = df.sample(max_seqs_per_rep, random_state=42)
        else:
            df = df.sample(len(df), random_state=42)  # shuffle

        # encode each sequence
        seq_tensor = []
        for s in df["junction_aa"].tolist():
            seq_tensor.append(encode_sequence(s, max_len=max_len))
        # pad with dummy sequences if needed
        if len(seq_tensor) < max_seqs_per_rep:
            pad_seq = [PAD_IDX] * max_len
            seq_tensor += [pad_seq] * (max_seqs_per_rep - len(seq_tensor))
        seq_tensor = torch.tensor(seq_tensor, dtype=torch.long)  # [max_seqs, max_len]

        rep_ids.append(rep_id)
        tensors.append(seq_tensor.unsqueeze(0))  # [1, max_seqs, max_len]
        rep_seq_dfs.append(df[["junction_aa","v_call","j_call"]].reset_index(drop=True))

        if has_meta:
            labels.append(int(label))

    if not tensors:
        return [], None, None, []

    X = torch.cat(tensors, dim=0)  # [N, max_seqs, max_len]
    if has_meta and for_training:
        y = torch.tensor(labels, dtype=torch.float32)
    else:
        y = None
    return rep_ids, X, y, rep_seq_dfs

# =========================
# Deep MIL model (CNN + attention)
# =========================

class CNNSeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, num_filters=64, kernel_sizes=(5,7)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        convs = []
        for k in kernel_sizes:
            convs.append(nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k//2))
        self.convs = nn.ModuleList(convs)
        self.activation = nn.ReLU()

        self.out_dim = num_filters * len(kernel_sizes)

    def forward(self, x):
        """
        x: [B, L] integer tokens
        return: [B, out_dim]
        """
        emb = self.embedding(x)           # [B, L, E]
        emb = emb.transpose(1, 2)         # [B, E, L]
        conv_outs = []
        for conv in self.convs:
            h = conv(emb)                 # [B, C, L]
            h = self.activation(h)
            h = torch.max(h, dim=2).values  # max over L -> [B, C]
            conv_outs.append(h)
        h_cat = torch.cat(conv_outs, dim=1)  # [B, out_dim]
        return h_cat


class AttentionMIL(nn.Module):
    def __init__(self, input_dim, att_dim=64):
        super().__init__()
        self.att_mlp = nn.Sequential(
            nn.Linear(input_dim, att_dim),
            nn.Tanh(),
            nn.Linear(att_dim, 1)  # scalar attention logit per sequence
        )

    def forward(self, seq_repr):
        """
        seq_repr: [B, S, D]
        returns: (rep_repr [B, D], att_weights [B, S])
        """
        B, S, D = seq_repr.shape
        logits = self.att_mlp(seq_repr)         # [B, S, 1]
        logits = logits.squeeze(-1)             # [B, S]
        weights = F.softmax(logits, dim=1)      # [B, S]
        rep_repr = torch.bmm(weights.unsqueeze(1), seq_repr).squeeze(1)  # [B, D]
        return rep_repr, weights


class DeepRepertoireNet(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=32, num_filters=64,
                 kernel_sizes=(5,7), att_dim=64, hidden_dim=64):
        super().__init__()
        self.seq_encoder = CNNSeqEncoder(vocab_size, embed_dim, num_filters, kernel_sizes)
        self.att_pool = AttentionMIL(self.seq_encoder.out_dim, att_dim=att_dim)
        self.fc = nn.Sequential(
            nn.Linear(self.seq_encoder.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        x: [B, S, L]
        Returns: logits [B], att_weights [B, S], seq_repr [B, S, D]
        """
        B, S, L = x.shape
        x_flat = x.view(B * S, L)
        seq_repr = self.seq_encoder(x_flat)      # [B*S, D]
        D = seq_repr.shape[1]
        seq_repr = seq_repr.view(B, S, D)        # [B, S, D]
        rep_repr, att_weights = self.att_pool(seq_repr)  # [B, D], [B, S]
        logits = self.fc(rep_repr).squeeze(-1)   # [B]
        return logits, att_weights, seq_repr


class RepertoireDataset(Dataset):
    def __init__(self, X_tensor, y_tensor=None):
        self.X = X_tensor
        self.y = y_tensor

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


def train_one_model(
    X, y,
    num_epochs=15,
    batch_size=4,
    lr=1e-3,
    weight_decay=1e-4,
    device="cpu",
    val_ratio=0.2,
    seed=42,
    plot_curves=True
):
    """
    Train DeepRepertoireNet on X, y (tensors), return model and (optionally) training curves.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # train/val split
    idx = np.arange(len(y))
    tr_idx, val_idx = train_test_split(idx, test_size=val_ratio, random_state=seed, stratify=y.numpy())
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_ds = RepertoireDataset(X_tr, y_tr)
    val_ds = RepertoireDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = DeepRepertoireNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_auc = -np.inf
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(1, num_epochs+1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits, _, _ = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(loss.item())

        # validation
        model.eval()
        val_losses = []
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits, _, _ = model(xb)
                loss = F.binary_cross_entropy_with_logits(logits, yb)
                val_losses.append(loss.item())
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(yb.cpu().numpy())

        val_auc = roc_auc_score(all_labels, all_probs)
        mean_tr = float(np.mean(train_losses))
        mean_val = float(np.mean(val_losses))
        history["train_loss"].append(mean_tr)
        history["val_loss"].append(mean_val)
        history["val_auc"].append(val_auc)

        print(f"Epoch {epoch:02d} | train_loss={mean_tr:.4f} | val_loss={mean_val:.4f} | val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    if plot_curves:
        fig, ax1 = plt.subplots(figsize=(6,4))
        ax1.plot(history["train_loss"], label="train loss")
        ax1.plot(history["val_loss"], label="val loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax2 = ax1.twinx()
        ax2.plot(history["val_auc"], color="green", label="val AUC")
        ax2.set_ylabel("AUC")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.title("Training curves")
        plt.show()

    return model, history

# =========================
# ImmuneStatePredictor wrapper
# =========================

class ImmuneStatePredictor:
    """
    Deep CNN+attention MIL model, template-compatible.
    """

    def __init__(self, n_jobs: int = 1, device: str = "cpu", **kwargs):
        self.n_jobs = n_jobs
        # if device == "cuda" and not torch.cuda.is_available():
        #     print("CUDA requested but not available; falling back to CPU.")
        #     device = "cpu"
        self.device = device
        self.model = None
        self.rep_seq_dfs_ = None  # per-rep sequence DataFrames (train only)
        self.train_rep_ids_ = None
        self.max_seqs_per_rep = kwargs.get("max_seqs_per_rep", 512)
        self.max_len = kwargs.get("max_len", 25)
        self.num_epochs = kwargs.get("num_epochs", 15)
        self.batch_size = kwargs.get("batch_size", 4)
        self.lr = kwargs.get("lr", 1e-3)
        self.weight_decay = kwargs.get("weight_decay", 1e-4)

    def fit(self, train_dir_path: str):
        print(f"Building tensors and training DeepRepertoireNet for {train_dir_path}...")
        rep_ids, X, y, rep_seq_dfs = build_repertoire_tensors(
            train_dir_path,
            max_seqs_per_rep=self.max_seqs_per_rep,
            max_len=self.max_len,
            for_training=True
        )
        if X is None or y is None:
            raise ValueError("No training data found.")
        self.train_rep_ids_ = rep_ids
        self.rep_seq_dfs_ = rep_seq_dfs

        device = self.device
        X = X.to(device)
        # y stays on CPU; moved in batches

        self.model, _ = train_one_model(
            X, y,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            device=self.device,
            val_ratio=0.2,
        )
        # after training, precompute sequence-level attention on full train to score sequences
        self.important_sequences_ = self._identify_associated_sequences_internal(train_dir_path)
        print("Training complete.")
        return self

    def predict_proba(self, test_dir_path: str) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not trained yet.")

        print(f"Preparing test repertoires from {test_dir_path}...")
        rep_ids, X, _, _ = build_repertoire_tensors(
            test_dir_path,
            max_seqs_per_rep=self.max_seqs_per_rep,
            max_len=self.max_len,
            for_training=False
        )
        if X is None or len(rep_ids) == 0:
            return pd.DataFrame()

        device = self.device
        X = X.to(device)

        ds = RepertoireDataset(X)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(device)
                logits, _, _ = self.model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)

        dataset_name = os.path.basename(test_dir_path)
        preds = pd.DataFrame({
            "ID": rep_ids,
            "dataset": [dataset_name] * len(rep_ids),
            "label_positive_probability": all_probs
        })
        preds["junction_aa"] = -999.0
        preds["v_call"] = -999.0
        preds["j_call"] = -999.0
        preds = preds[["ID","dataset","label_positive_probability","junction_aa","v_call","j_call"]]
        print(f"Predicted {len(preds)} repertoires in {test_dir_path}.")
        return preds

    def _identify_associated_sequences_internal(self, train_dir_path: str, top_k: int = 50000) -> pd.DataFrame:
        """
        Internal: use attention weights * per-sequence contribution to rank sequences.
        """
        print("Scoring sequences for label association...")
        device = self.device
        model = self.model
        model.eval()

        # Rebuild tensor with per-repertoire sequences in the same order as rep_seq_dfs_
        rep_ids, X, y, rep_seq_dfs = build_repertoire_tensors(
            train_dir_path,
            max_seqs_per_rep=self.max_seqs_per_rep,
            max_len=self.max_len,
            for_training=True
        )
        X = X.to(device)

        ds = RepertoireDataset(X, y)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        all_scores = []  # list of dicts: junction_aa, v_call, j_call, score

        with torch.no_grad():
            idx_offset = 0
            for xb, yb in loader:
                xb = xb.to(device)
                logits, att_weights, seq_repr = model(xb)  # B, [B,S], [B,S,D]
                probs = torch.sigmoid(logits)              # [B]

                B, S = att_weights.shape
                att_np = att_weights.cpu().numpy()
                probs_np = probs.cpu().numpy()

                # quick visualization: distribution of attention weights (optional)
                # sns.histplot(att_np.flatten(), bins=50); plt.show()

                for i in range(B):
                    global_idx = idx_offset + i
                    rep_df = rep_seq_dfs[global_idx]  # actual number of sequences may be < S
                    num_real = min(len(rep_df), S)
                    # sequence importance = attention_weight * sign(logit) * |logit|
                    # (approx contribution)
                    logit_i = float(logits[i].item())
                    for j in range(num_real):
                        score = att_np[i, j] * logit_i
                        row = rep_df.iloc[j]
                        all_scores.append({
                            "junction_aa": row["junction_aa"],
                            "v_call": row.get("v_call", np.nan),
                            "j_call": row.get("j_call", np.nan),
                            "score": score
                        })
                idx_offset += B

        seq_df = pd.DataFrame(all_scores)
        # aggregate by unique (junction_aa, v_call, j_call): mean score
        seq_df = seq_df.groupby(["junction_aa","v_call","j_call"], as_index=False)["score"].mean()
        seq_df = seq_df.sort_values("score", ascending=False).head(top_k)

        dataset_name = os.path.basename(train_dir_path)
        seq_df["dataset"] = dataset_name
        seq_df["ID"] = range(1, len(seq_df)+1)
        seq_df["ID"] = seq_df["dataset"] + "_seq_top_" + seq_df["ID"].astype(str)
        seq_df["label_positive_probability"] = -999.0
        seq_df = seq_df[["ID","dataset","label_positive_probability","junction_aa","v_call","j_call"]]

        # simple viz: top motif lengths
        plt.figure(figsize=(6,3))
        seq_df["junction_aa"].str.len().hist(bins=20)
        plt.title("Top sequence length distribution")
        plt.xlabel("length")
        plt.ylabel("count")
        plt.show()

        return seq_df

    def identify_associated_sequences(self, train_dir_path: str, top_k: int = 50000) -> pd.DataFrame:
        # Wrapper to comply with template; we already call internal version during fit.
        return self._identify_associated_sequences_internal(train_dir_path, top_k=top_k)

# =========================
# Pipeline helpers
# =========================

def _train_predictor(predictor: ImmuneStatePredictor, train_dir: str):
    print(f"Fitting model on {train_dir} ...")
    predictor.fit(train_dir)


def _generate_predictions(predictor: ImmuneStatePredictor, test_dirs: List[str]) -> pd.DataFrame:
    all_preds = []
    for td in test_dirs:
        print(f"Predicting on {td} ...")
        preds = predictor.predict_proba(td)
        if preds is not None and not preds.empty:
            all_preds.append(preds)
    if all_preds:
        return pd.concat(all_preds, ignore_index=True)
    return pd.DataFrame()


def _save_predictions(predictions: pd.DataFrame, out_dir: str, train_dir: str):
    if predictions.empty:
        raise ValueError("No predictions to save.")
    path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_test_predictions.tsv")
    save_tsv(predictions, path)
    print(f"Saved predictions to {path}")


def _save_important_sequences(predictor: ImmuneStatePredictor, out_dir: str, train_dir: str):
    seqs = predictor.important_sequences_
    if seqs is None or seqs.empty:
        raise ValueError("No important sequences found.")
    path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_important_sequences.tsv")
    save_tsv(seqs, path)
    print(f"Saved important sequences to {path}")


def main(train_dir: str, test_dirs: List[str], out_dir: str, n_jobs: int, device: str):
    validate_dirs_and_files(train_dir, test_dirs, out_dir)
    predictor = ImmuneStatePredictor(
        n_jobs=n_jobs,
        device=device,
        max_seqs_per_rep=512,
        max_len=25,
        num_epochs=15,
        batch_size=4,
        lr=1e-3,
        weight_decay=1e-4,
    )
    _train_predictor(predictor, train_dir)
    preds = _generate_predictions(predictor, test_dirs)
    _save_predictions(preds, out_dir, train_dir)
    _save_important_sequences(predictor, out_dir, train_dir)


def run():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--test_dirs", required=True, nargs="+")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_jobs", type=int, default=2)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()
    main(args.train_dir, args.test_dirs, args.out_dir, args.n_jobs, args.device)


# =========================
# Kaggle notebook entry
# =========================




import os


if __name__ == "__main__":
    PATH_DATASET = "/kaggle/input/adaptive-immune-profiling-challenge-2025"
    TRAIN_ROOT = os.path.join(PATH_DATASET, "train_datasets", "train_datasets")
    TEST_ROOT = os.path.join(PATH_DATASET, "test_datasets", "test_datasets")
    OUT_ROOT = "/kaggle/working/results_deep_mil"

    device = 'cpu'
    print(device)
    print("Using device:", device)

    pairs = get_dataset_pairs(TRAIN_ROOT, TEST_ROOT)
    print("Dataset pairs:", pairs)

    
    for train_path, test_paths in pairs:
        if not test_paths:
            print(f"No test sets for {train_path}, skipping.")
            continue
        main(train_path, test_paths, OUT_ROOT, n_jobs=2, device=device)

    concatenate_output_files(OUT_ROOT)



# AIRR-MLðŸ§¬25: Improved Deep CNN + Attention MIL

import os
import sys
import glob
from collections import defaultdict
from typing import List, Tuple, Iterator, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Basic utilities
# =========================

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
PAD_IDX = len(AMINO_ACIDS)
VOCAB_SIZE = len(AMINO_ACIDS) + 1

def load_data_generator(
    data_dir: str,
    metadata_filename: str = "metadata.csv"
) -> Iterator[Union[Tuple[str, pd.DataFrame, bool], Tuple[str, pd.DataFrame]]]:
    metadata_path = os.path.join(data_dir, metadata_filename)
    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        for row in metadata_df.itertuples(index=False):
            file_path = os.path.join(data_dir, row.filename)
            try:
                df = pd.read_csv(file_path, sep="\t")
                yield row.repertoire_id, df, bool(row.label_positive)
            except FileNotFoundError:
                print(f"Warning: missing file '{row.filename}'")
    else:
        for file_path in sorted(glob.glob(os.path.join(data_dir, "*.tsv"))):
            try:
                df = pd.read_csv(file_path, sep="\t")
                filename = os.path.basename(file_path)
                yield filename, df
            except Exception as e:
                print(f"Warning: error reading '{file_path}': {e}")


def load_full_dataset(data_dir: str) -> pd.DataFrame:
    metadata_path = os.path.join(data_dir, "metadata.csv")
    data_loader = load_data_generator(data_dir=data_dir)
    dfs = []
    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        total = len(metadata_df)
        for rep_id, df, label in tqdm(data_loader, total=total, desc="Loading full dataset"):
            df["ID"] = rep_id
            df["label_positive"] = label
            dfs.append(df)
    else:
        tsv_files = glob.glob(os.path.join(data_dir, "*.tsv"))
        total = len(tsv_files)
        for fname, df in tqdm(data_loader, total=total, desc="Loading full dataset"):
            rep_id = os.path.basename(fname).replace(".tsv","")
            df["ID"] = rep_id
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def get_repertoire_ids(data_dir: str) -> List[str]:
    metadata_path = os.path.join(data_dir, "metadata.csv")
    if os.path.exists(metadata_path):
        meta = pd.read_csv(metadata_path)
        return meta["repertoire_id"].tolist()
    tsv_files = glob.glob(os.path.join(data_dir, "*.tsv"))
    return [os.path.basename(f).replace(".tsv", "") for f in sorted(tsv_files)]


def validate_dirs_and_files(train_dir: str, test_dirs: List[str], out_dir: str) -> None:
    assert os.path.isdir(train_dir), f"Train dir {train_dir} missing"
    assert os.path.isfile(os.path.join(train_dir, "metadata.csv")), "metadata.csv missing in train"
    assert glob.glob(os.path.join(train_dir, "*.tsv")), "No .tsv in train dir"

    for td in test_dirs:
        assert os.path.isdir(td), f"Test dir {td} missing"
        assert glob.glob(os.path.join(td, "*.tsv")), f"No .tsv in test dir {td}"

    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "tmp.test")
    with open(tmp, "w") as f:
        f.write("ok")
    os.remove(tmp)


def get_dataset_pairs(train_root: str, test_root: str) -> List[Tuple[str, List[str]]]:
    test_groups = defaultdict(list)
    for tname in sorted(os.listdir(test_root)):
        if not tname.startswith("test_dataset_"):
            continue
        base = tname.replace("test_dataset_", "").split("_")[0]
        test_groups[base].append(os.path.join(test_root, tname))
    pairs = []
    for tname in sorted(os.listdir(train_root)):
        if not tname.startswith("train_dataset_"):
            continue
        base = tname.replace("train_dataset_", "")
        train_path = os.path.join(train_root, tname)
        pairs.append((train_path, test_groups.get(base, [])))
    return pairs


def save_tsv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def concatenate_output_files(out_dir: str) -> pd.DataFrame:
    preds_files = sorted(glob.glob(os.path.join(out_dir, "*_test_predictions.tsv")))
    seq_files = sorted(glob.glob(os.path.join(out_dir, "*_important_sequences.tsv")))
    dfs = []
    for f in preds_files + seq_files:
        try:
            dfs.append(pd.read_csv(f, sep="\t"))
        except Exception as e:
            print(f"Warning reading {f}: {e}")
    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
    else:
        all_df = pd.DataFrame(columns=["ID","dataset","label_positive_probability","junction_aa","v_call","j_call"])
    for col in ["label_positive_probability","junction_aa","v_call","j_call"]:
        if col in all_df.columns:
            all_df[col] = all_df[col].fillna(-999.0)
    out_path = os.path.join(out_dir, "submissions.csv")
    all_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with shape {all_df.shape}")
    return all_df

# =========================
# Sequence encoding
# =========================

def encode_sequence(seq: str, max_len: int) -> List[int]:
    if not isinstance(seq, str):
        seq = ""
    seq = seq.strip()
    ids = [AA_TO_IDX.get(ch, PAD_IDX) for ch in seq][:max_len]
    if len(ids) < max_len:
        ids += [PAD_IDX] * (max_len - len(ids))
    return ids


def build_repertoire_tensors(
    data_dir: str,
    max_seqs_per_rep: int = 1024,
    max_len: int = 35,
    for_training: bool = True
):
    loader = load_data_generator(data_dir=data_dir)
    metadata_path = os.path.join(data_dir, "metadata.csv")
    has_meta = os.path.exists(metadata_path)

    rep_ids = []
    labels = []
    tensors = []
    rep_seq_dfs = []

    for item in tqdm(loader, desc=f"Building repertoires ({'train' if for_training else 'test'})"):
        if has_meta:
            rep_id, df, label = item
        else:
            rep_file, df = item
            rep_id = os.path.basename(rep_file).replace(".tsv","")
            label = None

        df = df.dropna(subset=["junction_aa"])
        if df.empty:
            continue

        # shuffle and subsample up to max_seqs_per_rep
        if len(df) > max_seqs_per_rep:
            df = df.sample(max_seqs_per_rep, random_state=42)
        else:
            df = df.sample(len(df), random_state=42)

        seq_tensor = [encode_sequence(s, max_len=max_len) for s in df["junction_aa"].tolist()]
        if len(seq_tensor) < max_seqs_per_rep:
            pad_seq = [PAD_IDX] * max_len
            seq_tensor += [pad_seq] * (max_seqs_per_rep - len(seq_tensor))
        seq_tensor = torch.tensor(seq_tensor, dtype=torch.long)  # [S, L]

        rep_ids.append(rep_id)
        tensors.append(seq_tensor.unsqueeze(0))  # [1,S,L]
        rep_seq_dfs.append(df[["junction_aa","v_call","j_call"]].reset_index(drop=True))

        if has_meta:
            labels.append(int(label))

    if not tensors:
        return [], None, None, []

    X = torch.cat(tensors, dim=0)  # [N,S,L]
    if has_meta and for_training:
        y = torch.tensor(labels, dtype=torch.float32)
    else:
        y = None
    return rep_ids, X, y, rep_seq_dfs

# =========================
# Deep MIL model (larger CNN + attention)
# =========================

class CNNSeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_filters=128, kernel_sizes=(5,7,9), dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        convs = []
        for k in kernel_sizes:
            convs.append(nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k//2))
        self.convs = nn.ModuleList(convs)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_dim = num_filters * len(kernel_sizes)

    def forward(self, x):
        # x: [B,L]
        emb = self.embedding(x)       # [B,L,E]
        emb = emb.transpose(1, 2)     # [B,E,L]
        conv_outs = []
        for conv in self.convs:
            h = conv(emb)             # [B,C,L]
            h = self.activation(h)
            h = F.max_pool1d(h, kernel_size=h.size(2)).squeeze(2)  # [B,C]
            conv_outs.append(h)
        h_cat = torch.cat(conv_outs, dim=1)      # [B,out_dim]
        h_cat = self.dropout(h_cat)
        return h_cat


class AttentionMIL(nn.Module):
    def __init__(self, input_dim, att_dim=128, dropout=0.1):
        super().__init__()
        self.att_mlp = nn.Sequential(
            nn.Linear(input_dim, att_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(att_dim, 1)
        )

    def forward(self, seq_repr):
        # seq_repr: [B,S,D]
        logits = self.att_mlp(seq_repr).squeeze(-1)     # [B,S]
        weights = F.softmax(logits, dim=1)              # [B,S]
        rep_repr = torch.bmm(weights.unsqueeze(1), seq_repr).squeeze(1)  # [B,D]
        return rep_repr, weights


class DeepRepertoireNet(nn.Module):
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        embed_dim=64,
        num_filters=128,
        kernel_sizes=(5,7,9),
        att_dim=128,
        hidden_dim=128,
        dropout=0.3
    ):
        super().__init__()
        self.seq_encoder = CNNSeqEncoder(
            vocab_size, embed_dim=embed_dim,
            num_filters=num_filters, kernel_sizes=kernel_sizes,
            dropout=dropout*0.5
        )
        self.att_pool = AttentionMIL(self.seq_encoder.out_dim, att_dim=att_dim, dropout=dropout*0.5)
        self.fc = nn.Sequential(
            nn.Linear(self.seq_encoder.out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: [B,S,L]
        B,S,L = x.shape
        x_flat = x.view(B*S, L)
        seq_repr_flat = self.seq_encoder(x_flat)   # [B*S,D]
        D = seq_repr_flat.shape[1]
        seq_repr = seq_repr_flat.view(B,S,D)       # [B,S,D]
        rep_repr, att_weights = self.att_pool(seq_repr)  # [B,D],[B,S]
        logits = self.fc(rep_repr).squeeze(-1)     # [B]
        return logits, att_weights, seq_repr


class RepertoireDataset(Dataset):
    def __init__(self, X_tensor, y_tensor=None):
        self.X = X_tensor
        self.y = y_tensor

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


def train_one_model(
    X, y,
    num_epochs=30,
    batch_size=8,
    lr=1e-3,
    weight_decay=5e-5,
    device="cuda",
    val_ratio=0.2,
    label_smoothing=0.05,
    seed=42,
    plot_curves=True
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    idx = np.arange(len(y))
    tr_idx, val_idx = train_test_split(idx, test_size=val_ratio, random_state=seed, stratify=y.numpy())
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_ds = RepertoireDataset(X_tr, y_tr)
    val_ds = RepertoireDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = DeepRepertoireNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_auc = -np.inf
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(1, num_epochs+1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits, _, _ = model(xb)
            # label smoothing: y=(1-eps) for positives
            y_smooth = yb * (1.0 - label_smoothing) + 0.5 * label_smoothing
            loss = F.binary_cross_entropy_with_logits(logits, y_smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        model.eval()
        val_losses = []
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits, _, _ = model(xb)
                y_smooth = yb * (1.0 - label_smoothing) + 0.5 * label_smoothing
                loss = F.binary_cross_entropy_with_logits(logits, y_smooth)
                val_losses.append(loss.item())
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(yb.cpu().numpy())

        val_auc = roc_auc_score(all_labels, all_probs)
        mean_tr = float(np.mean(train_losses))
        mean_val = float(np.mean(val_losses))
        history["train_loss"].append(mean_tr)
        history["val_loss"].append(mean_val)
        history["val_auc"].append(val_auc)
        print(f"Epoch {epoch:02d} | train_loss={mean_tr:.4f} | val_loss={mean_val:.4f} | val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    if plot_curves:
        fig, ax1 = plt.subplots(figsize=(6,4))
        ax1.plot(history["train_loss"], label="train loss")
        ax1.plot(history["val_loss"], label="val loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax2 = ax1.twinx()
        ax2.plot(history["val_auc"], color="green", label="val AUC")
        ax2.set_ylabel("AUC")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.title("Training curves (improved model)")
        plt.show()

    return model, history

# =========================
# ImmuneStatePredictor wrapper
# =========================

class ImmuneStatePredictor:
    """
    Deep CNN+attention MIL model, improved and template-compatible.
    """

    def __init__(self, n_jobs: int = 1, device: str = "cpu", **kwargs):
        self.n_jobs = n_jobs
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available; falling back to CPU.")
            device = "cpu"
        self.device = device
        self.model = None
        self.rep_seq_dfs_ = None
        self.train_rep_ids_ = None

        self.max_seqs_per_rep = kwargs.get("max_seqs_per_rep", 1024)
        self.max_len = kwargs.get("max_len", 35)
        self.num_epochs = kwargs.get("num_epochs", 30)
        self.batch_size = kwargs.get("batch_size", 8)
        self.lr = kwargs.get("lr", 1e-3)
        self.weight_decay = kwargs.get("weight_decay", 5e-5)

        self.important_sequences_ = None

    def fit(self, train_dir_path: str):
        print(f"Building tensors and training DeepRepertoireNet for {train_dir_path}...")
        rep_ids, X, y, rep_seq_dfs = build_repertoire_tensors(
            train_dir_path,
            max_seqs_per_rep=self.max_seqs_per_rep,
            max_len=self.max_len,
            for_training=True
        )
        if X is None or y is None:
            raise ValueError("No training data found.")
        self.train_rep_ids_ = rep_ids
        self.rep_seq_dfs_ = rep_seq_dfs

        device = self.device
        X = X.to(device if device == "cuda" else "cpu")

        self.model, _ = train_one_model(
            X, y,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            device=self.device,
            val_ratio=0.2,
            label_smoothing=0.05,
        )

        self.important_sequences_ = self._identify_associated_sequences_internal(train_dir_path)
        print("Training complete.")
        return self

    def predict_proba(self, test_dir_path: str) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not trained yet.")

        print(f"Preparing test repertoires from {test_dir_path}...")
        rep_ids, X, _, _ = build_repertoire_tensors(
            test_dir_path,
            max_seqs_per_rep=self.max_seqs_per_rep,
            max_len=self.max_len,
            for_training=False
        )
        if X is None or len(rep_ids) == 0:
            return pd.DataFrame()

        device = self.device
        X = X.to(device if device == "cuda" else "cpu")

        ds = RepertoireDataset(X)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(device)
                logits, _, _ = self.model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)

        dataset_name = os.path.basename(test_dir_path)
        preds = pd.DataFrame({
            "ID": rep_ids,
            "dataset": [dataset_name] * len(rep_ids),
            "label_positive_probability": all_probs
        })
        preds["junction_aa"] = -999.0
        preds["v_call"] = -999.0
        preds["j_call"] = -999.0
        preds = preds[["ID","dataset","label_positive_probability","junction_aa","v_call","j_call"]]
        print(f"Predicted {len(preds)} repertoires in {test_dir_path}.")
        return preds

    def _identify_associated_sequences_internal(self, train_dir_path: str, top_k: int = 50000) -> pd.DataFrame:
        print("Scoring sequences for label association (improved model)...")
        device = self.device
        model = self.model
        model.eval()

        rep_ids, X, y, rep_seq_dfs = build_repertoire_tensors(
            train_dir_path,
            max_seqs_per_rep=self.max_seqs_per_rep,
            max_len=self.max_len,
            for_training=True
        )
        X = X.to(device if device == "cuda" else "cpu")

        ds = RepertoireDataset(X, y)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        all_scores = []

        with torch.no_grad():
            idx_offset = 0
            for xb, yb in loader:
                xb = xb.to(device)
                logits, att_weights, seq_repr = model(xb)
                B,S = att_weights.shape
                att_np = att_weights.cpu().numpy()
                logits_np = logits.cpu().numpy()

                for i in range(B):
                    global_idx = idx_offset + i
                    rep_df = rep_seq_dfs[global_idx]
                    num_real = min(len(rep_df), S)
                    logit_i = float(logits_np[i])
                    for j in range(num_real):
                        score = att_np[i, j] * logit_i
                        row = rep_df.iloc[j]
                        all_scores.append({
                            "junction_aa": row["junction_aa"],
                            "v_call": row.get("v_call", np.nan),
                            "j_call": row.get("j_call", np.nan),
                            "score": score
                        })
                idx_offset += B

        seq_df = pd.DataFrame(all_scores)
        seq_df = seq_df.groupby(["junction_aa","v_call","j_call"], as_index=False)["score"].mean()
        seq_df = seq_df.sort_values("score", ascending=False).head(top_k)

        dataset_name = os.path.basename(train_dir_path)
        seq_df["dataset"] = dataset_name
        seq_df["ID"] = range(1, len(seq_df)+1)
        seq_df["ID"] = seq_df["dataset"] + "_seq_top_" + seq_df["ID"].astype(str)
        seq_df["label_positive_probability"] = -999.0
        seq_df = seq_df[["ID","dataset","label_positive_probability","junction_aa","v_call","j_call"]]

        # Visualization: sequence length distribution of top hits
        plt.figure(figsize=(6,3))
        seq_df["junction_aa"].str.len().hist(bins=20)
        plt.title("Top sequence length distribution (improved model)")
        plt.xlabel("length")
        plt.ylabel("count")
        plt.show()

        return seq_df

    def identify_associated_sequences(self, train_dir_path: str, top_k: int = 50000) -> pd.DataFrame:
        return self._identify_associated_sequences_internal(train_dir_path, top_k=top_k)

# =========================
# Pipeline helpers
# =========================

def _train_predictor(predictor: ImmuneStatePredictor, train_dir: str):
    print(f"Fitting model on {train_dir} ...")
    predictor.fit(train_dir)


def _generate_predictions(predictor: ImmuneStatePredictor, test_dirs: List[str]) -> pd.DataFrame:
    all_preds = []
    for td in test_dirs:
        print(f"Predicting on {td} ...")
        preds = predictor.predict_proba(td)
        if preds is not None and not preds.empty:
            all_preds.append(preds)
    if all_preds:
        return pd.concat(all_preds, ignore_index=True)
    return pd.DataFrame()


def _save_predictions(predictions: pd.DataFrame, out_dir: str, train_dir: str):
    if predictions.empty:
        raise ValueError("No predictions to save.")
    path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_test_predictions.tsv")
    save_tsv(predictions, path)
    print(f"Saved predictions to {path}")


def _save_important_sequences(predictor: ImmuneStatePredictor, out_dir: str, train_dir: str):
    seqs = predictor.important_sequences_
    if seqs is None or seqs.empty:
        raise ValueError("No important sequences found.")
    path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_important_sequences.tsv")
    save_tsv(seqs, path)
    print(f"Saved important sequences to {path}")


def main(train_dir: str, test_dirs: List[str], out_dir: str, n_jobs: int, device: str):
    validate_dirs_and_files(train_dir, test_dirs, out_dir)
    predictor = ImmuneStatePredictor(
        n_jobs=n_jobs,
        device=device,
        max_seqs_per_rep=1024,
        max_len=35,
        num_epochs=30,
        batch_size=8,
        lr=1e-3,
        weight_decay=5e-5,
    )
    _train_predictor(predictor, train_dir)
    preds = _generate_predictions(predictor, test_dirs)
    _save_predictions(preds, out_dir, train_dir)
    _save_important_sequences(predictor, out_dir, train_dir)


def run():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--test_dirs", required=True, nargs="+")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    args = parser.parse_args()
    main(args.train_dir, args.test_dirs, args.out_dir, args.n_jobs, args.device)


if __name__ == "__main__":
    PATH_DATASET = "/kaggle/input/adaptive-immune-profiling-challenge-2025"
    TRAIN_ROOT = os.path.join(PATH_DATASET, "train_datasets", "train_datasets")
    TEST_ROOT = os.path.join(PATH_DATASET, "test_datasets", "test_datasets")
    OUT_ROOT = "/kaggle/working/results_deep_mil_improved"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    pairs = get_dataset_pairs(TRAIN_ROOT, TEST_ROOT)
    print("Dataset pairs:", pairs)

    for train_path, test_paths in pairs:
        if not test_paths:
            print(f"No test sets for {train_path}, skipping.")
            continue
        main(train_path, test_paths, OUT_ROOT, n_jobs=4, device=device)

    concatenate_output_files(OUT_ROOT)
# AIRR-MLðŸ§¬25: Improved Deep CNN + Attention MIL

import os
import sys
import glob
from collections import defaultdict
from typing import List, Tuple, Iterator, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Basic utilities
# =========================

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
PAD_IDX = len(AMINO_ACIDS)
VOCAB_SIZE = len(AMINO_ACIDS) + 1

def load_data_generator(
    data_dir: str,
    metadata_filename: str = "metadata.csv"
) -> Iterator[Union[Tuple[str, pd.DataFrame, bool], Tuple[str, pd.DataFrame]]]:
    metadata_path = os.path.join(data_dir, metadata_filename)
    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        for row in metadata_df.itertuples(index=False):
            file_path = os.path.join(data_dir, row.filename)
            try:
                df = pd.read_csv(file_path, sep="\t")
                yield row.repertoire_id, df, bool(row.label_positive)
            except FileNotFoundError:
                print(f"Warning: missing file '{row.filename}'")
    else:
        for file_path in sorted(glob.glob(os.path.join(data_dir, "*.tsv"))):
            try:
                df = pd.read_csv(file_path, sep="\t")
                filename = os.path.basename(file_path)
                yield filename, df
            except Exception as e:
                print(f"Warning: error reading '{file_path}': {e}")


def load_full_dataset(data_dir: str) -> pd.DataFrame:
    metadata_path = os.path.join(data_dir, "metadata.csv")
    data_loader = load_data_generator(data_dir=data_dir)
    dfs = []
    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        total = len(metadata_df)
        for rep_id, df, label in tqdm(data_loader, total=total, desc="Loading full dataset"):
            df["ID"] = rep_id
            df["label_positive"] = label
            dfs.append(df)
    else:
        tsv_files = glob.glob(os.path.join(data_dir, "*.tsv"))
        total = len(tsv_files)
        for fname, df in tqdm(data_loader, total=total, desc="Loading full dataset"):
            rep_id = os.path.basename(fname).replace(".tsv","")
            df["ID"] = rep_id
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def get_repertoire_ids(data_dir: str) -> List[str]:
    metadata_path = os.path.join(data_dir, "metadata.csv")
    if os.path.exists(metadata_path):
        meta = pd.read_csv(metadata_path)
        return meta["repertoire_id"].tolist()
    tsv_files = glob.glob(os.path.join(data_dir, "*.tsv"))
    return [os.path.basename(f).replace(".tsv", "") for f in sorted(tsv_files)]


def validate_dirs_and_files(train_dir: str, test_dirs: List[str], out_dir: str) -> None:
    assert os.path.isdir(train_dir), f"Train dir {train_dir} missing"
    assert os.path.isfile(os.path.join(train_dir, "metadata.csv")), "metadata.csv missing in train"
    assert glob.glob(os.path.join(train_dir, "*.tsv")), "No .tsv in train dir"

    for td in test_dirs:
        assert os.path.isdir(td), f"Test dir {td} missing"
        assert glob.glob(os.path.join(td, "*.tsv")), f"No .tsv in test dir {td}"

    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "tmp.test")
    with open(tmp, "w") as f:
        f.write("ok")
    os.remove(tmp)


def get_dataset_pairs(train_root: str, test_root: str) -> List[Tuple[str, List[str]]]:
    test_groups = defaultdict(list)
    for tname in sorted(os.listdir(test_root)):
        if not tname.startswith("test_dataset_"):
            continue
        base = tname.replace("test_dataset_", "").split("_")[0]
        test_groups[base].append(os.path.join(test_root, tname))
    pairs = []
    for tname in sorted(os.listdir(train_root)):
        if not tname.startswith("train_dataset_"):
            continue
        base = tname.replace("train_dataset_", "")
        train_path = os.path.join(train_root, tname)
        pairs.append((train_path, test_groups.get(base, [])))
    return pairs


def save_tsv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def concatenate_output_files(out_dir: str) -> pd.DataFrame:
    preds_files = sorted(glob.glob(os.path.join(out_dir, "*_test_predictions.tsv")))
    seq_files = sorted(glob.glob(os.path.join(out_dir, "*_important_sequences.tsv")))
    dfs = []
    for f in preds_files + seq_files:
        try:
            dfs.append(pd.read_csv(f, sep="\t"))
        except Exception as e:
            print(f"Warning reading {f}: {e}")
    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
    else:
        all_df = pd.DataFrame(columns=["ID","dataset","label_positive_probability","junction_aa","v_call","j_call"])
    for col in ["label_positive_probability","junction_aa","v_call","j_call"]:
        if col in all_df.columns:
            all_df[col] = all_df[col].fillna(-999.0)
    out_path = os.path.join(out_dir, "submissions.csv")
    all_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with shape {all_df.shape}")
    return all_df

# =========================
# Sequence encoding
# =========================

def encode_sequence(seq: str, max_len: int) -> List[int]:
    if not isinstance(seq, str):
        seq = ""
    seq = seq.strip()
    ids = [AA_TO_IDX.get(ch, PAD_IDX) for ch in seq][:max_len]
    if len(ids) < max_len:
        ids += [PAD_IDX] * (max_len - len(ids))
    return ids


def build_repertoire_tensors(
    data_dir: str,
    max_seqs_per_rep: int = 1024,
    max_len: int = 35,
    for_training: bool = True
):
    loader = load_data_generator(data_dir=data_dir)
    metadata_path = os.path.join(data_dir, "metadata.csv")
    has_meta = os.path.exists(metadata_path)

    rep_ids = []
    labels = []
    tensors = []
    rep_seq_dfs = []

    for item in tqdm(loader, desc=f"Building repertoires ({'train' if for_training else 'test'})"):
        if has_meta:
            rep_id, df, label = item
        else:
            rep_file, df = item
            rep_id = os.path.basename(rep_file).replace(".tsv","")
            label = None

        df = df.dropna(subset=["junction_aa"])
        if df.empty:
            continue

        # shuffle and subsample up to max_seqs_per_rep
        if len(df) > max_seqs_per_rep:
            df = df.sample(max_seqs_per_rep, random_state=42)
        else:
            df = df.sample(len(df), random_state=42)

        seq_tensor = [encode_sequence(s, max_len=max_len) for s in df["junction_aa"].tolist()]
        if len(seq_tensor) < max_seqs_per_rep:
            pad_seq = [PAD_IDX] * max_len
            seq_tensor += [pad_seq] * (max_seqs_per_rep - len(seq_tensor))
        seq_tensor = torch.tensor(seq_tensor, dtype=torch.long)  # [S, L]

        rep_ids.append(rep_id)
        tensors.append(seq_tensor.unsqueeze(0))  # [1,S,L]
        rep_seq_dfs.append(df[["junction_aa","v_call","j_call"]].reset_index(drop=True))

        if has_meta:
            labels.append(int(label))

    if not tensors:
        return [], None, None, []

    X = torch.cat(tensors, dim=0)  # [N,S,L]
    if has_meta and for_training:
        y = torch.tensor(labels, dtype=torch.float32)
    else:
        y = None
    return rep_ids, X, y, rep_seq_dfs

# =========================
# Deep MIL model (larger CNN + attention)
# =========================

class CNNSeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_filters=128, kernel_sizes=(5,7,9), dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        convs = []
        for k in kernel_sizes:
            convs.append(nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k//2))
        self.convs = nn.ModuleList(convs)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_dim = num_filters * len(kernel_sizes)

    def forward(self, x):
        # x: [B,L]
        emb = self.embedding(x)       # [B,L,E]
        emb = emb.transpose(1, 2)     # [B,E,L]
        conv_outs = []
        for conv in self.convs:
            h = conv(emb)             # [B,C,L]
            h = self.activation(h)
            h = F.max_pool1d(h, kernel_size=h.size(2)).squeeze(2)  # [B,C]
            conv_outs.append(h)
        h_cat = torch.cat(conv_outs, dim=1)      # [B,out_dim]
        h_cat = self.dropout(h_cat)
        return h_cat


class AttentionMIL(nn.Module):
    def __init__(self, input_dim, att_dim=128, dropout=0.1):
        super().__init__()
        self.att_mlp = nn.Sequential(
            nn.Linear(input_dim, att_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(att_dim, 1)
        )

    def forward(self, seq_repr):
        # seq_repr: [B,S,D]
        logits = self.att_mlp(seq_repr).squeeze(-1)     # [B,S]
        weights = F.softmax(logits, dim=1)              # [B,S]
        rep_repr = torch.bmm(weights.unsqueeze(1), seq_repr).squeeze(1)  # [B,D]
        return rep_repr, weights


class DeepRepertoireNet(nn.Module):
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        embed_dim=64,
        num_filters=128,
        kernel_sizes=(5,7,9),
        att_dim=128,
        hidden_dim=128,
        dropout=0.3
    ):
        super().__init__()
        self.seq_encoder = CNNSeqEncoder(
            vocab_size, embed_dim=embed_dim,
            num_filters=num_filters, kernel_sizes=kernel_sizes,
            dropout=dropout*0.5
        )
        self.att_pool = AttentionMIL(self.seq_encoder.out_dim, att_dim=att_dim, dropout=dropout*0.5)
        self.fc = nn.Sequential(
            nn.Linear(self.seq_encoder.out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: [B,S,L]
        B,S,L = x.shape
        x_flat = x.view(B*S, L)
        seq_repr_flat = self.seq_encoder(x_flat)   # [B*S,D]
        D = seq_repr_flat.shape[1]
        seq_repr = seq_repr_flat.view(B,S,D)       # [B,S,D]
        rep_repr, att_weights = self.att_pool(seq_repr)  # [B,D],[B,S]
        logits = self.fc(rep_repr).squeeze(-1)     # [B]
        return logits, att_weights, seq_repr


class RepertoireDataset(Dataset):
    def __init__(self, X_tensor, y_tensor=None):
        self.X = X_tensor
        self.y = y_tensor

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


def train_one_model(
    X, y,
    num_epochs=30,
    batch_size=8,
    lr=1e-3,
    weight_decay=5e-5,
    device="cuda",
    val_ratio=0.2,
    label_smoothing=0.05,
    seed=42,
    plot_curves=True
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    idx = np.arange(len(y))
    tr_idx, val_idx = train_test_split(idx, test_size=val_ratio, random_state=seed, stratify=y.numpy())
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_ds = RepertoireDataset(X_tr, y_tr)
    val_ds = RepertoireDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = DeepRepertoireNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_auc = -np.inf
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(1, num_epochs+1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits, _, _ = model(xb)
            # label smoothing: y=(1-eps) for positives
            y_smooth = yb * (1.0 - label_smoothing) + 0.5 * label_smoothing
            loss = F.binary_cross_entropy_with_logits(logits, y_smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        model.eval()
        val_losses = []
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits, _, _ = model(xb)
                y_smooth = yb * (1.0 - label_smoothing) + 0.5 * label_smoothing
                loss = F.binary_cross_entropy_with_logits(logits, y_smooth)
                val_losses.append(loss.item())
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(yb.cpu().numpy())

        val_auc = roc_auc_score(all_labels, all_probs)
        mean_tr = float(np.mean(train_losses))
        mean_val = float(np.mean(val_losses))
        history["train_loss"].append(mean_tr)
        history["val_loss"].append(mean_val)
        history["val_auc"].append(val_auc)
        print(f"Epoch {epoch:02d} | train_loss={mean_tr:.4f} | val_loss={mean_val:.4f} | val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    if plot_curves:
        fig, ax1 = plt.subplots(figsize=(6,4))
        ax1.plot(history["train_loss"], label="train loss")
        ax1.plot(history["val_loss"], label="val loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax2 = ax1.twinx()
        ax2.plot(history["val_auc"], color="green", label="val AUC")
        ax2.set_ylabel("AUC")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.title("Training curves (improved model)")
        plt.show()

    return model, history

# =========================
# ImmuneStatePredictor wrapper
# =========================

class ImmuneStatePredictor:
    """
    Deep CNN+attention MIL model, improved and template-compatible.
    """

    def __init__(self, n_jobs: int = 1, device: str = "cpu", **kwargs):
        self.n_jobs = n_jobs
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available; falling back to CPU.")
            device = "cpu"
        self.device = device
        self.model = None
        self.rep_seq_dfs_ = None
        self.train_rep_ids_ = None

        self.max_seqs_per_rep = kwargs.get("max_seqs_per_rep", 1024)
        self.max_len = kwargs.get("max_len", 35)
        self.num_epochs = kwargs.get("num_epochs", 30)
        self.batch_size = kwargs.get("batch_size", 8)
        self.lr = kwargs.get("lr", 1e-3)
        self.weight_decay = kwargs.get("weight_decay", 5e-5)

        self.important_sequences_ = None

    def fit(self, train_dir_path: str):
        print(f"Building tensors and training DeepRepertoireNet for {train_dir_path}...")
        rep_ids, X, y, rep_seq_dfs = build_repertoire_tensors(
            train_dir_path,
            max_seqs_per_rep=self.max_seqs_per_rep,
            max_len=self.max_len,
            for_training=True
        )
        if X is None or y is None:
            raise ValueError("No training data found.")
        self.train_rep_ids_ = rep_ids
        self.rep_seq_dfs_ = rep_seq_dfs

        device = self.device
        X = X.to(device if device == "cuda" else "cpu")

        self.model, _ = train_one_model(
            X, y,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            device=self.device,
            val_ratio=0.2,
            label_smoothing=0.05,
        )

        self.important_sequences_ = self._identify_associated_sequences_internal(train_dir_path)
        print("Training complete.")
        return self

    def predict_proba(self, test_dir_path: str) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not trained yet.")

        print(f"Preparing test repertoires from {test_dir_path}...")
        rep_ids, X, _, _ = build_repertoire_tensors(
            test_dir_path,
            max_seqs_per_rep=self.max_seqs_per_rep,
            max_len=self.max_len,
            for_training=False
        )
        if X is None or len(rep_ids) == 0:
            return pd.DataFrame()

        device = self.device
        X = X.to(device if device == "cuda" else "cpu")

        ds = RepertoireDataset(X)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(device)
                logits, _, _ = self.model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)

        dataset_name = os.path.basename(test_dir_path)
        preds = pd.DataFrame({
            "ID": rep_ids,
            "dataset": [dataset_name] * len(rep_ids),
            "label_positive_probability": all_probs
        })
        preds["junction_aa"] = -999.0
        preds["v_call"] = -999.0
        preds["j_call"] = -999.0
        preds = preds[["ID","dataset","label_positive_probability","junction_aa","v_call","j_call"]]
        print(f"Predicted {len(preds)} repertoires in {test_dir_path}.")
        return preds

    def _identify_associated_sequences_internal(self, train_dir_path: str, top_k: int = 50000) -> pd.DataFrame:
        print("Scoring sequences for label association (improved model)...")
        device = self.device
        model = self.model
        model.eval()

        rep_ids, X, y, rep_seq_dfs = build_repertoire_tensors(
            train_dir_path,
            max_seqs_per_rep=self.max_seqs_per_rep,
            max_len=self.max_len,
            for_training=True
        )
        X = X.to(device if device == "cuda" else "cpu")

        ds = RepertoireDataset(X, y)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        all_scores = []

        with torch.no_grad():
            idx_offset = 0
            for xb, yb in loader:
                xb = xb.to(device)
                logits, att_weights, seq_repr = model(xb)
                B,S = att_weights.shape
                att_np = att_weights.cpu().numpy()
                logits_np = logits.cpu().numpy()

                for i in range(B):
                    global_idx = idx_offset + i
                    rep_df = rep_seq_dfs[global_idx]
                    num_real = min(len(rep_df), S)
                    logit_i = float(logits_np[i])
                    for j in range(num_real):
                        score = att_np[i, j] * logit_i
                        row = rep_df.iloc[j]
                        all_scores.append({
                            "junction_aa": row["junction_aa"],
                            "v_call": row.get("v_call", np.nan),
                            "j_call": row.get("j_call", np.nan),
                            "score": score
                        })
                idx_offset += B

        seq_df = pd.DataFrame(all_scores)
        seq_df = seq_df.groupby(["junction_aa","v_call","j_call"], as_index=False)["score"].mean()
        seq_df = seq_df.sort_values("score", ascending=False).head(top_k)

        dataset_name = os.path.basename(train_dir_path)
        seq_df["dataset"] = dataset_name
        seq_df["ID"] = range(1, len(seq_df)+1)
        seq_df["ID"] = seq_df["dataset"] + "_seq_top_" + seq_df["ID"].astype(str)
        seq_df["label_positive_probability"] = -999.0
        seq_df = seq_df[["ID","dataset","label_positive_probability","junction_aa","v_call","j_call"]]

        # Visualization: sequence length distribution of top hits
        plt.figure(figsize=(6,3))
        seq_df["junction_aa"].str.len().hist(bins=20)
        plt.title("Top sequence length distribution (improved model)")
        plt.xlabel("length")
        plt.ylabel("count")
        plt.show()

        return seq_df

    def identify_associated_sequences(self, train_dir_path: str, top_k: int = 50000) -> pd.DataFrame:
        return self._identify_associated_sequences_internal(train_dir_path, top_k=top_k)

# =========================
# Pipeline helpers
# =========================

def _train_predictor(predictor: ImmuneStatePredictor, train_dir: str):
    print(f"Fitting model on {train_dir} ...")
    predictor.fit(train_dir)


def _generate_predictions(predictor: ImmuneStatePredictor, test_dirs: List[str]) -> pd.DataFrame:
    all_preds = []
    for td in test_dirs:
        print(f"Predicting on {td} ...")
        preds = predictor.predict_proba(td)
        if preds is not None and not preds.empty:
            all_preds.append(preds)
    if all_preds:
        return pd.concat(all_preds, ignore_index=True)
    return pd.DataFrame()


def _save_predictions(predictions: pd.DataFrame, out_dir: str, train_dir: str):
    if predictions.empty:
        raise ValueError("No predictions to save.")
    path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_test_predictions.tsv")
    save_tsv(predictions, path)
    print(f"Saved predictions to {path}")


def _save_important_sequences(predictor: ImmuneStatePredictor, out_dir: str, train_dir: str):
    seqs = predictor.important_sequences_
    if seqs is None or seqs.empty:
        raise ValueError("No important sequences found.")
    path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_important_sequences.tsv")
    save_tsv(seqs, path)
    print(f"Saved important sequences to {path}")


def main(train_dir: str, test_dirs: List[str], out_dir: str, n_jobs: int, device: str):
    validate_dirs_and_files(train_dir, test_dirs, out_dir)
    predictor = ImmuneStatePredictor(
        n_jobs=n_jobs,
        device=device,
        max_seqs_per_rep=1024,
        max_len=35,
        num_epochs=30,
        batch_size=8,
        lr=1e-3,
        weight_decay=5e-5,
    )
    _train_predictor(predictor, train_dir)
    preds = _generate_predictions(predictor, test_dirs)
    _save_predictions(preds, out_dir, train_dir)
    _save_important_sequences(predictor, out_dir, train_dir)


def run():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--test_dirs", required=True, nargs="+")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    args = parser.parse_args()
    main(args.train_dir, args.test_dirs, args.out_dir, args.n_jobs, args.device)


if __name__ == "__main__":
    PATH_DATASET = "/kaggle/input/adaptive-immune-profiling-challenge-2025"
    TRAIN_ROOT = os.path.join(PATH_DATASET, "train_datasets", "train_datasets")
    TEST_ROOT = os.path.join(PATH_DATASET, "test_datasets", "test_datasets")
    OUT_ROOT = "/kaggle/working/results_deep_mil_improved"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    pairs = get_dataset_pairs(TRAIN_ROOT, TEST_ROOT)
    print("Dataset pairs:", pairs)

    for train_path, test_paths in pairs:
        if not test_paths:
            print(f"No test sets for {train_path}, skipping.")
            continue
        main(train_path, test_paths, OUT_ROOT, n_jobs=4, device=device)

    concatenate_output_files(OUT_ROOT)


