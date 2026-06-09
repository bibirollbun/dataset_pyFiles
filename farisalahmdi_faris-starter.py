%%capture
! pip install /kaggle/input/bio-whl/biopython-1.85-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


import numpy as np
from Bio import SeqIO
import torch
from torch.utils.data import Dataset
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from os import rmdir
from torch.utils.data import DataLoader
import pandas as pd


class PositionEmbeddingLayer(nn.Module):
    def __init__(self, sequence_length, vocab_size, output_dim):
        super(PositionEmbeddingLayer, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, output_dim, padding_idx=0)
        self.position_embedding_layer = nn.Embedding(sequence_length, output_dim)

    def forward(self, inputs):
        # inputs shape: (batch, seq_length)
        seq_length = inputs.size(1)
        positions = torch.arange(seq_length, device=inputs.device).unsqueeze(0)  # (1, seq_length)
        embedded = self.embedding_layer(inputs)                            # (batch, seq_length, output_dim)
        pos_emb = self.position_embedding_layer(positions)                 # (1, seq_length, output_dim)
        return embedded + pos_emb

class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, queries, keys, values, mask=None, covariance=None):
        # queries: (batch, heads, seq_len, head_dim)
        scale = math.sqrt(queries.size(-1))
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / scale  # (batch, heads, seq_len, seq_len)
        if covariance is not None:
            scores = scores + covariance.unsqueeze(1)  # make sure covariance is broadcastable
        if mask is not None:
            scores = scores * mask
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, values)

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_k, d_v, d_model):
        """
        h: number of heads
        d_k: output dimension for query and key projection (assumed to be divisible by h)
        d_v: output dimension for value projection (similarly, heads × (d_v/h))
        d_model: final output dimension (from W_o)
        """
        super(MultiHeadAttention, self).__init__()
        self.heads = h
        self.d_k = d_k  # note: d_k should be chosen so that head_dim = d_k // h
        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_v)
        self.W_o = nn.Linear(d_v, d_model)
        self.attention = DotProductAttention()

    def reshape_tensor(self, x, flag):
        # x: (batch, seq_len, d) where d will be split over heads

        if flag:
            batch, seq_len, d = x.size()
            head_dim = d // self.heads
            # reshape to (batch, seq_len, heads, head_dim) then permute to (batch, heads, seq_len, head_dim)
            return x.view(batch, seq_len, self.heads, head_dim).permute(0, 2, 1, 3)
        else:
            batch, h,seq_len, d = x.size()
            # reverse: from (batch, heads, seq_len, head_dim) to (batch, seq_len, d)
            return x.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, d*h)

    def forward(self, query, key, value, attention_mask=None, covariance=None):
        # Linear projections: assume input shape (batch, seq_len, d_model)
        q = self.W_q(query)  # (batch, seq_len, d_k)
        k = self.W_k(key)    # (batch, seq_len, d_k)
        v = self.W_v(value)  # (batch, seq_len, d_v)

        # Reshape for multi-head attention.
        q_reshaped = self.reshape_tensor(q, flag=True)  # (batch, heads, seq_len, d_k/head)
        k_reshaped = self.reshape_tensor(k, flag=True)
        v_reshaped = self.reshape_tensor(v, flag=True)

        # Compute attention output.
        out = self.attention(q_reshaped, k_reshaped, v_reshaped, mask=attention_mask, covariance=covariance)
        # Reverse reshape: (batch, seq_len, d_v)
        output = self.reshape_tensor(out, flag=False)
        return self.W_o(output)

class AddNormalization(nn.Module):
    def __init__(self, d_model):
        super(AddNormalization, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_x):
        return self.layer_norm(x + sublayer_x)

class FeedForward(nn.Module):
    def __init__(self, d_ff, d_model):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = nn.SiLU()  # swish activation

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, h, d_k, d_v, d_model, d_ff, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.add_norm1 = AddNormalization(d_model)
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.add_norm2 = AddNormalization(d_model)

    def forward(self, x, covariance, padding_mask):
        # x: (batch, seq_len, d_model)
        mha_out = self.multihead_attention(x, x, x, attention_mask=padding_mask, covariance=covariance)
        mha_out = self.dropout1(mha_out)
        addnorm_out = self.add_norm1(x, mha_out)
        ff_out = self.feed_forward(addnorm_out)
        ff_out = self.dropout2(ff_out)
        return self.add_norm2(addnorm_out, ff_out)

class Encoder(nn.Module):
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n_layers, dropout_rate):
        super(Encoder, self).__init__()
        self.pos_encoding = PositionEmbeddingLayer(sequence_length, vocab_size, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(4, d_model)
        self.activation = nn.SiLU()
        self.layer_norm = nn.LayerNorm(d_model)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(h, d_k, d_v, d_model, d_ff, dropout_rate) for _ in range(n_layers)
        ])

    def forward(self, input_sequence, covariance, pssm, freq_matrix):
        # Generate a padding mask: positions not equal to zero.
        # (batch, seq_len)
        mask = (input_sequence != 0).float()
        # Compute pairwise mask: (batch, seq_len, seq_len)
        padding_mask = torch.matmul(mask.unsqueeze(2), mask.unsqueeze(1))

        # Positional encoding.
        pos_enc_out = self.pos_encoding(input_sequence)
        x = self.dropout(pos_enc_out)

        # Project extra features and add.
        pssm_proj = self.activation(self.dense(pssm))
        freq_proj = self.activation(self.dense(freq_matrix))
        x = x + pssm_proj + freq_proj
        x = self.layer_norm(x)

        for layer in self.encoder_layers:
            x = layer(x, covariance, padding_mask)
        return x

class RNA3DModel(nn.Module):
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate):
        super(RNA3DModel, self).__init__()
        self.encoder = Encoder(vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)
        self.linear = nn.Linear(d_model, 3)

    def forward(self, inputs):
        X = inputs[0]

        covariance = inputs[1] if len(inputs) > 1 else None
        freq_matrix = inputs[2] if len(inputs) > 2 else None
        pssm = inputs[3] if len(inputs) > 3 else None

        output = self.encoder(X, covariance, pssm, freq_matrix)
        output = self.linear(output)
        return output


def parse_msa_fasta(msa_filepath, alphabet=('A', 'C', 'G', 'U'), pseudocount=1e-6):
    """
    Parse an MSA FASTA file and compute three features:
      - Frequency matrix: shape (L, |alphabet|) 
      - PSSM: log-odds score matrix of shape (L, |alphabet|)
      - Covariance matrix: a scalar covariance between columns, shape (L, L)

    Args:
        msa_filepath (str): Path to the MSA FASTA file.
        alphabet (tuple): Symbols to consider (default for RNA: A, C, G, U).
        pseudocount (float): Small value added to avoid log(0).

    Returns:
        tuple: (frequency_matrix, pssm, covariance)
            frequency_matrix: np.array of shape (L, len(alphabet))
            pssm: np.array of shape (L, len(alphabet))
            covariance: np.array of shape (L, L)
    """
    # Parse all sequences from the MSA file.
    msa_records = list(SeqIO.parse(msa_filepath, "fasta"))
    if len(msa_records) == 0:
        raise ValueError("No sequences found in the provided MSA file.")
    
    # All sequences should be the same length.
    seq_length = len(msa_records[0].seq)
    M = len(msa_records)
    K = len(alphabet)
    
    # Initialize frequency matrix and one-hot encoded array.
    freq_matrix = np.zeros((seq_length, K), dtype=np.float32)
    one_hot = np.zeros((M, seq_length, K), dtype=np.float32)
    
    for m, record in enumerate(msa_records):
        seq = str(record.seq).upper()
        if len(seq) != seq_length:
            raise ValueError("All sequences in the MSA must have the same length.")
        for i, letter in enumerate(seq):
            if letter in alphabet:
                idx = alphabet.index(letter)
                freq_matrix[i, idx] += 1
                one_hot[m, i, idx] = 1.0
            # Optionally: handle gaps ('-') or ambiguous symbols here.
    
    # Normalize frequencies along each position.
    freq_matrix /= (M + 1e-8)
    
    # Compute PSSM:
    # Background probabilities, assuming a uniform background.
    background = np.full((K,), 1.0 / K, dtype=np.float32)
    # Add pseudocount so we never divide by zero.
    freq_with_pc = freq_matrix + pseudocount
    background = background + pseudocount
    # Compute log-odds score. The result is the PSSM.
    pssm = np.log(freq_with_pc / background)
    
    # Compute Covariance Matrix between positions:
    # For each position, we already have a frequency distribution computed via one-hot averages.
    # First, compute the mean one-hot vector per position.
    mean_one_hot = np.mean(one_hot, axis=0)  # shape: (L, K)
    # Compute the difference from the mean for each sequence.
    diff = one_hot - mean_one_hot[None, :, :]  # shape: (M, L, K)
    # Compute covariance between each pair of positions.
    # This gives a scalar covariance value for each pair (i, j).
    covariance = np.einsum('mik,mjk->ij', diff, diff) / (M - 1 + 1e-8)  # shape: (L, L)
    
    return freq_matrix, pssm, covariance


class RNADataset(Dataset):
    def __init__(self, data, tokenizer, freq_matrix=True, pssm=True, covariance=True,
                 max_length=512, mode='train', augment_coords=True, noise_std=0.1):
        """
        data: a pandas DataFrame with columns:
            - "target_id", "sequence", "freq_matrix", "ppsm", "coverience"
            - coordinate columns "x_1", "y_1", "z_1"
        tokenizer: dict mapping sequence tokens to indices.
        freq_matrix, pssm, covariance: flags to include respective features.
        max_length: maximum sequence length for padding/truncation.
        mode: 'train' or 'test'.
        augment_coords: whether to apply random Gaussian noise to the 3D coordinates.
        noise_std: standard deviation for coordinate noise.
        """
        self.data = data.copy()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.freq_matrix = freq_matrix
        self.pssm = pssm
        self.covariance = covariance
        self.mode = mode
        self.augment_coords = augment_coords
        self.noise_std = noise_std

        # Prepare samples aggregated by target_id
        self.target_ids = self.data["target_id"].unique()
        self.samples = []
        for tid in self.target_ids:
            sample = {}
            sample_df = self.data[self.data["target_id"] == tid]
            row = sample_df.iloc[0]

            # Tokenize and pad sequence
            seq = list(row["sequence"])
            tokenized = [self.tokenizer.get(tok, 0) for tok in seq]
            if len(tokenized) < self.max_length:
                tokenized = tokenized + [0] * (self.max_length - len(tokenized))
            else:
                tokenized = tokenized[:self.max_length]
            sample["sequence"] = np.array(tokenized, dtype=np.int32)

            if self.mode == 'train':
                # Coordinates: pad with NaN
                coords = sample_df[["x_1", "y_1", "z_1"]].values
                padded_coords = np.full((self.max_length, 3), np.nan, dtype=np.float32)
                padded_coords[:coords.shape[0], :] = coords
                sample["coords"] = padded_coords

                # Frequency matrix
                if self.freq_matrix:
                    fm = row["freq_matrix"]  # shape (L, K)
                    padded_fm = np.zeros((self.max_length, fm.shape[1]), dtype=np.float32)
                    padded_fm[:fm.shape[0], :] = fm
                    sample["freq_matrix"] = padded_fm

                # PSSM
                if self.pssm:
                    ps = row["ppsm"]  # shape (L, K)
                    padded_ps = np.zeros((self.max_length, ps.shape[1]), dtype=np.float32)
                    padded_ps[:ps.shape[0], :] = ps
                    sample["pssm"] = padded_ps

                # Covariance
                if self.covariance:
                    cov = row["coverience"]  # shape (L, L)
                    padded_cov = np.zeros((self.max_length, self.max_length), dtype=np.float32)
                    padded_cov[:cov.shape[0], :cov.shape[1]] = cov
                    sample["covariance"] = padded_cov
            else:
                if self.freq_matrix:
                    fm = row["freq_matrix"]  # shape (L, K)
                    padded_fm = np.zeros((self.max_length, fm.shape[1]), dtype=np.float32)
                    padded_fm[:fm.shape[0], :] = fm
                    sample["freq_matrix"] = padded_fm

                # PSSM
                if self.pssm:
                    ps = row["ppsm"]  # shape (L, K)
                    padded_ps = np.zeros((self.max_length, ps.shape[1]), dtype=np.float32)
                    padded_ps[:ps.shape[0], :] = ps
                    sample["pssm"] = padded_ps

                # Covariance
                if self.covariance:
                    cov = row["coverience"]  # shape (L, L)
                    padded_cov = np.zeros((self.max_length, self.max_length), dtype=np.float32)
                    padded_cov[:cov.shape[0], :cov.shape[1]] = cov
                    sample["covariance"] = padded_cov

            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        seq_tensor = torch.tensor(sample["sequence"], dtype=torch.int32)

        if self.mode == 'train':
            # Assemble inputs
            inputs = [seq_tensor]
            if self.covariance:
                inputs.append(torch.tensor(sample["covariance"], dtype=torch.float))
            if self.freq_matrix:
                inputs.append(torch.tensor(sample["freq_matrix"], dtype=torch.float))
            if self.pssm:
                inputs.append(torch.tensor(sample["pssm"], dtype=torch.float))

            # Coordinates
            coords = torch.tensor(sample["coords"], dtype=torch.float)

            # Random augmentation: add Gaussian noise
            if self.augment_coords:
                mask = ~torch.isnan(coords[:, 0])
                noise = torch.randn(mask.sum().item(), 3, device=coords.device) * self.noise_std
                coords[mask] = coords[mask] + noise

            return inputs, coords
        else:
            inputs = [seq_tensor]
            if self.covariance:
                inputs.append(torch.tensor(sample["covariance"], dtype=torch.float))
            if self.freq_matrix:
                inputs.append(torch.tensor(sample["freq_matrix"], dtype=torch.float))
            if self.pssm:
                inputs.append(torch.tensor(sample["pssm"], dtype=torch.float))
            return inputs


class DMSELoss(nn.Module):
    def __init__(self):
        super(DMSELoss, self).__init__()

    def forward(self, y_true, y_pred):
        d_true = self._pairwise_distance_matrix(y_true)
        d_pred = self._pairwise_distance_matrix(y_pred)
        
        mask=~torch.isnan(d_true)
        B, N, _ = d_true.shape
        diag_mask = torch.eye(N, device=d_true.device, dtype=torch.bool).unsqueeze(0).expand(B, N, N)
        mask[diag_mask] = False
        
        mse = (d_true[mask] - d_pred[mask]) ** 2
        return mse.mean()

    def pairwise_diff_matrix(self, X):
        return X.unsqueeze(2) - X.unsqueeze(1)  # (batch, seq_len, seq_len, 3)

    def _pairwise_distance_matrix(self, coords):
        diff = self.pairwise_diff_matrix(coords)
        dists = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
        return dists

class DMAELoss(nn.Module):
    def __init__(self):
        super(DMAELoss, self).__init__()

    def forward(self, y_true, y_pred):  
        d_true = self._pairwise_distance_matrix(y_true)
        d_pred = self._pairwise_distance_matrix(y_pred)
        
        mask=~torch.isnan(d_true)
        B, N, _ = d_true.shape
        diag_mask = torch.eye(N, device=d_true.device, dtype=torch.bool).unsqueeze(0).expand(B, N, N)
        mask[diag_mask] = False
        mae = torch.abs(d_true[mask] - d_pred[mask])
        return mae.mean()/10

    def pairwise_diff_matrix(self, X):
        return X.unsqueeze(2) - X.unsqueeze(1)  # (batch, seq_len, seq_len, 3)

    def _pairwise_distance_matrix(self, coords):
        diff = self.pairwise_diff_matrix(coords)
        dists = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
        return dists

class TorsionLoss(nn.Module):
    def __init__(self):
        super(TorsionLoss, self).__init__()

    def _compute_dihedral_angle(self, p0, p1, p2, p3):
        b0 = p1 - p0
        b1 = p2 - p1
        b2 = p3 - p2
        b1_norm = torch.norm(b1, dim=-1, keepdim=True)
        b1_norm_clipped = torch.clamp(b1_norm, min=1e-8)
        b1 = b1 / b1_norm_clipped
        v = b0 - torch.sum(b0 * b1, dim=-1, keepdim=True) * b1
        w = b2 - torch.sum(b2 * b1, dim=-1, keepdim=True) * b1
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        w_norm = torch.norm(w, dim=-1, keepdim=True)
        v = torch.where(v_norm < 1e-8, torch.zeros_like(v), v)
        w = torch.where(w_norm < 1e-8, torch.zeros_like(w), w)
        x = torch.sum(v * w, dim=-1)
        y = torch.sum(torch.cross(b1, v, dim=-1) * w, dim=-1)
        angle = torch.atan2(y + 1e-8, x + 1e-8)
        return angle

    def forward(self, y_true, y_pred):
        mask = (~torch.isnan(y_true[..., 0])).float()  # (batch, seq_len)
        y_true_clean = torch.where(torch.isnan(y_true), torch.zeros_like(y_true), y_true)
        y_pred_clean = torch.where(torch.isnan(y_true), torch.zeros_like(y_pred), y_pred)
        # Create 4-point sliding windows
        p0_true = y_true_clean[:, :-3]
        p1_true = y_true_clean[:, 1:-2]
        p2_true = y_true_clean[:, 2:-1]
        p3_true = y_true_clean[:, 3:]
        p0_pred = y_pred_clean[:, :-3]
        p1_pred = y_pred_clean[:, 1:-2]
        p2_pred = y_pred_clean[:, 2:-1]
        p3_pred = y_pred_clean[:, 3:]
        angle_true = self._compute_dihedral_angle(p0_true, p1_true, p2_true, p3_true)
        angle_pred = self._compute_dihedral_angle(p0_pred, p1_pred, p2_pred, p3_pred)
        mask_torsion = mask[:, :-3] * mask[:, 1:-2] * mask[:, 2:-1] * mask[:, 3:]
        angle_diff = (angle_pred - angle_true + np.pi) % (2 * np.pi) - np.pi
        loss = 0.5 * (1.0 - torch.cos(angle_diff))
        loss = (loss * mask_torsion).sum() / (mask_torsion.sum() + 1e-8)
        return loss


def safe_normalize(x, dim=-1, eps=1e-8):
    norm = torch.norm(x, dim=dim, keepdim=True)
    return x / (norm + eps)

class FAPEloss(nn.Module):
    def __init__(self, Z=10.0, clamp=10.0):
        super().__init__()
        self.Z = Z
        self.clamp = clamp

    def compute_local_frames(self, coords):
        B, N, _ = coords.shape
        R = torch.eye(3, device=coords.device).unsqueeze(0).unsqueeze(0).expand(B, N, 3, 3).clone()

        if N > 2:
            p_prev = coords[:, :-2, :]  # (B, N-2, 3)
            p = coords[:, 1:-1, :]      # (B, N-2, 3)
            p_next = coords[:, 2:, :]   # (B, N-2, 3)

            v1 = p - p_prev             # (B, N-2, 3)
            v2 = p_next - p             # (B, N-2, 3)

            x_axis = safe_normalize(v1, dim=-1)  # (B, N-2, 3)
            z_axis = torch.cross(v1, v2, dim=-1)
            z_axis = safe_normalize(z_axis, dim=-1)
            y_axis = torch.cross(z_axis, x_axis, dim=-1)  # (B, N-2, 3)

            local_R = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # (B, N-2, 3, 3)
            R[:, 1:-1, :, :] = local_R

        T = torch.zeros((B, N, 4, 4), device=coords.device)
        T[:, :, :3, :3] = R
        T[:, :, :3, 3] = coords   
        T[:, :, 3, 3] = 1.0
        return R, T

    def forward(self, y_pred, y_true):
        R_pred, T_pred = self.compute_local_frames(y_pred)
        R_true, T_true = self.compute_local_frames(y_true)


        delta_pred = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)  # (B, N, N, 3)
        delta_true = y_true.unsqueeze(2) - y_true.unsqueeze(1)  # (B, N, N, 3)
        
        X_pred = torch.einsum('b i c d, b i j d -> b i j c', R_pred, delta_pred)
        X_true = torch.einsum('b i c d, b i j d -> b i j c', R_true, delta_true)

        trans_error = torch.norm(X_pred - X_true, dim=-1)  # (B, N, N)
        trans_error = torch.clamp(trans_error, max=self.clamp) / self.Z
        translation_loss = torch.mean(trans_error)

        R_diff = torch.matmul(R_pred.transpose(-2, -1), R_true)  # (B, N, 3, 3)
        trace_val = R_diff.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)  # (B, N)
        angle_error = torch.acos(torch.clamp((trace_val - 1) / 2, min=-1.0, max=1.0))
        rotation_loss = torch.mean(angle_error)

        transformation_loss = torch.mean((T_pred - T_true) ** 2)

        total_loss = translation_loss + rotation_loss + transformation_loss
        return total_loss

class AlignSVDMSELoss(nn.Module):

    def __init__(self, Z=10.0):
        super(AlignSVDMSELoss, self).__init__()
        self.Z = Z

    def forward(self, y_true,y_pred):
        assert y_pred.shape == y_true.shape
        mask=~torch.isnan(y_true.sum(-1))

        y_pred=y_pred[mask]
        y_true=y_true[mask]
        

        centroid_y_pred = y_pred.mean(dim=0, keepdim=True)
        centroid_y_true = y_true.mean(dim=0, keepdim=True)


        y_pred_centered = y_pred - centroid_y_pred.detach()
        y_true_centered = y_true - centroid_y_true

        cov_matrix = y_pred_centered.T @ y_true_centered

        U, S, Vt = torch.svd(cov_matrix)

        R = Vt @ U.T

        if torch.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt @ U.T

        aligned_y_pred = (y_pred_centered @ R.T.detach()) + centroid_y_true.detach()

        return torch.abs(aligned_y_pred-y_true).mean()/self.Z

class dRMAE(nn.Module):
    def __init__(self):
        super(dRMAE, self).__init__()
        self.Z = 10

    def pairwise_diff_matrix(self, X):
        return X.unsqueeze(2) - X.unsqueeze(1)  # (batch, seq_len, seq_len, 3)

    def _pairwise_distance_matrix(self, coords):
        diff = self.pairwise_diff_matrix(coords[:,:,:-1])
        dists = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
        return dists

    def forward(self, y_true,y_pred, epsilon=1e-4,d_clamp=None): 

        y_true_d=self._pairwise_distance_matrix(y_true)
        y_pred_d=self._pairwise_distance_matrix(y_pred)

        mask=~torch.isnan(y_true_d)
        diag_eye = torch.stack([torch.eye(mask.shape[1]).bool() for i in range(mask.shape[0])])
        mask[diag_eye]=False

        rmsd=torch.abs(y_pred_d[mask]-y_true_d[mask])

        return rmsd.mean()/self.Z
class CompositeLoss(nn.Module):
    def __init__(self,
                 torsion_weight=1,
                 dmse_weight=1,
                 dmae_weight=1,
                 fape_weight=1,
                 allignsvd_weight=1,
                 drmae_weight=1):
        super(CompositeLoss, self).__init__()
        self.torsion_weight = torsion_weight
        self.dmse_weight = dmse_weight
        self.fape_weight = fape_weight
        self.allignsvd_weight = allignsvd_weight
        self.drmae_weight = drmae_weight
        self.dmae_weight = dmae_weight

        self.torsion_loss = TorsionLoss()
        # self.dmse_loss = DMSELoss()
        # self.fape_loss = FAPEloss()
        self.allignsvd_loss = AlignSVDMSELoss()
        self.drmae_loss = dRMAE()
        # self.dmae_loss = DMAELoss()

    def forward(self, y_true, y_pred):
        torsion = self.torsion_loss(y_true, y_pred)
        # dmse = self.dmse_loss(y_true, y_pred)
        # fape = self.fape_loss(y_true, y_pred)
        allignsvd = self.allignsvd_loss(y_true,y_pred)
        drmae = self.drmae_loss(y_true,y_pred)
        # dmae = self.dmae_loss(y_true,y_pred)

        return self.allignsvd_weight * allignsvd \
        + self.drmae_weight * drmae \
        + self.torsion_weight * torsion


def compute_rmsd(y_true, y_pred):
    rmsd_values = []
    B = y_true.shape[0]
    for i in range(B):
        yt = y_true[i]  # shape (N, 3)
        yp = y_pred[i]  # shape (N, 3)

        # Check for non-finite values. You can also use torch.nan_to_num to replace them.
        if (not torch.isfinite(yt).all()) or (not torch.isfinite(yp).all()):
            # Skip this sample or set its RMSD to zero.
            rmsd_values.append(torch.tensor(0.0, device=yt.device))
            continue

        # Center the coordinates by subtracting the mean (per sample).
        X = yp - torch.mean(yp, dim=0, keepdim=True)
        Y = yt - torch.mean(yt, dim=0, keepdim=True)

        # Compute the covariance matrix between the centered predicted and true coordinates.
        # This gives a 3x3 matrix.
        C = torch.matmul(X.t(), Y)
        # Check for non-finite values in C.
        if not torch.isfinite(C).all():
            rmsd_values.append(torch.tensor(0.0, device=yt.device))
            continue

        # Compute the SVD of C. We use torch.linalg.svd which is robust and returns (U, S, Vh).
        try:
            U, S, Vt = torch.linalg.svd(C)
        except RuntimeError as e:
            # If SVD fails, skip this sample (or return 0 for it)
            rmsd_values.append(torch.tensor(0.0, device=yt.device))
            continue

        # Reflection fix: if the determinant of (U * Vt) is negative, fix the sign on the last column of U.
        det = torch.det(torch.matmul(U, Vt))
        d = torch.sign(det)
        # Fix the sign of the last column of U if needed.
        U[:, -1] = U[:, -1] * d
        # Compute the optimal rotation matrix.
        R = torch.matmul(U, Vt)

        # Align the predicted coordinates.
        X_aligned = torch.matmul(X, R)
        diff = X_aligned - Y
        # Compute RMSD over all residues.
        rmsd = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=1)))
        rmsd_values.append(rmsd)

    # Compute average RMSD over the batch.
    if len(rmsd_values) > 0:
        return torch.stack(rmsd_values).mean()
    else:
        return torch.tensor(0.0, device=y_true.device)

def compute_tm_score(y_true, y_pred):
    """
    Compute TM-score between predicted and true 3D coordinates.

    This function supports both single-sample input of shape (N, 3)
    and batched input of shape (B, N, 3). If batched, the TM-score is averaged
    over samples.

    Args:
        y_true (ndarray): Ground-truth coordinates. Can be (N, 3) or (B, N, 3).
        y_pred (ndarray): Predicted coordinates. Can be (N, 3) or (B, N, 3).

    Returns:
        float: TM-score (average over batch if batched).
    """
    # If batched input, process each sample individually.
    if y_true.ndim == 3:
        tm_scores = []
        for i in range(y_true.shape[0]):
            tm = compute_tm_score(y_true[i], y_pred[i])
            tm_scores.append(tm)
        return np.mean(tm_scores)

    # Ensure single sample inputs are at least 2D.
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)

    # Remove padded residues. We assume padded rows contain NaNs.
    mask = ~np.isnan(y_true[:, 0])
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    L = len(y_true)
    if L == 0:
        return 0.0  # Nothing to compare.

    # Center the coordinates.
    X = y_pred - np.mean(y_pred, axis=0)
    Y = y_true - np.mean(y_true, axis=0)

    # Compute the covariance matrix.
    C = np.dot(X.T, Y)
    if C.ndim < 2:
        return 0.0

    # Compute SVD in a try/except block to catch convergence issues.
    try:
        U, S, Vt = np.linalg.svd(C)
    except np.linalg.LinAlgError:
        return 0.0

    # Compute optimal rotation R.
    R = np.dot(U, Vt)

    # If the rotation matrix is improper (reflection), fix it.
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(U, Vt)

    # Align predicted coordinates.
    X_aligned = np.dot(X, R)
    dists = np.sqrt(np.sum((X_aligned - Y) ** 2, axis=1))

    # Compute d0 parameter as defined (with a floor value of 0.5).
    if L > 15:
        d0 = 1.24 * ((L - 15) ** (1/3)) - 1.8
    else:
        d0 = 0.5
    d0 = max(d0, 0.5)

    tm = np.sum(1 / (1 + (dists / d0) ** 2)) / L
    return tm


# A, C, G, and U
TOKENIZER = {"A":1, "C":2, "G":3, "U":4,"-":5,"X":6}
ROOT_DIR = "/content/drive/MyDrive/Kaggle/Stanford RNA 3D/"
CUTOFF_DATE = "2020-01-01"
TEST_CUTOFF_DATE = "2022-05-01"
LEARNING_RATE = 0.0001
WD = 0.001
EPOCHS = 50
WARMUP_RATIO = 0.25
LEARNING_RATE_MAX = 0.001
VER = 0.1
TRAIN = True
COVARIENCE = True
PSSM = True
FREQ_MATRIX = True


train_seq = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
val_seq = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv")
train_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")
val_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_labels.csv")
test_seq = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")


train_seq = train_seq.sample(frac=1)
val_seq = val_seq.sample(frac=1)


train_seq[["freq_matrix","ppsm","coverience"]] = train_seq["target_id"].apply(lambda x: parse_msa_fasta("/kaggle/input/stanford-rna-3d-folding/MSA/" + x + ".MSA.fasta")).apply(pd.Series)
val_seq[["freq_matrix","ppsm","coverience"]] = val_seq["target_id"].apply(lambda x: parse_msa_fasta("/kaggle/input/stanford-rna-3d-folding/MSA/" + x + ".MSA.fasta")).apply(pd.Series)


train_labels["target_id"] = train_labels["ID"].str.split("_",expand=True).apply(lambda x: "_".join(x[0:-1]),axis=1)
val_labels["target_id"] = val_labels["ID"].str.split("_",expand=True).apply(lambda x: "_".join(x[0:-1]),axis=1)


to_remove_train = train_labels.groupby("target_id").apply(lambda x: x["x_1"].isna().sum() > (x["resid"].max()/2)).reset_index().rename(columns={0:"to_remove"})
to_remove_val = val_labels.groupby("target_id").apply(lambda x: x["x_1"].isna().sum() > (x["resid"].max()/2)).reset_index().rename(columns={0:"to_remove"})


train_labels = train_labels.merge(to_remove_train[to_remove_train["to_remove"]==False][["target_id"]],on="target_id",how="right")
val_labels = val_labels.merge(to_remove_val[to_remove_val["to_remove"]==False][["target_id"]],on="target_id",how="right")


train_data = train_seq.merge(train_labels,on="target_id",how="right")
val_data = val_seq.merge(val_labels,on="target_id",how="right")


MAX_LENGTH = train_data["resid"].max()


train_data.index = train_data["target_id"]
val_data.index = val_data["target_id"]
test_seq.index = test_seq["target_id"]


MEAN = np.nanmean(train_data[["x_1","y_1","z_1"]].values)
STD = np.nanstd(train_data[["x_1","y_1","z_1"]].values)


# train_data[["x_1","y_1","z_1"]] = (train_data[["x_1","y_1","z_1"]] - MEAN) / STD


train_data = train_data.fillna(np.nan)


train_data_final = train_data[pd.to_datetime(train_data["temporal_cutoff"]) <= pd.to_datetime(CUTOFF_DATE)]
val_data_final = train_data[(pd.to_datetime(train_data["temporal_cutoff"]) > pd.to_datetime(CUTOFF_DATE)) & (pd.to_datetime(train_data["temporal_cutoff"]) <= pd.to_datetime(TEST_CUTOFF_DATE))]


if TRAIN:
    VOCAB_SIZE = 6
    H = 8
    D_K = 64
    D_V = 64
    D_MODEL = 128
    D_FF = 768
    N_LAYERS = 6
    DROPOUT_RATE = 0.1
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    EPOCHS = 50
    
    
    
    train_dataset = RNADataset(train_data_final, TOKENIZER, freq_matrix=True, pssm=True, covariance=True,
                               max_length=MAX_LENGTH, mode='train',augment_coords=False)
    val_dataset = RNADataset(val_data_final, TOKENIZER, freq_matrix=True, pssm=True, covariance=True,
                             max_length=MAX_LENGTH, mode='train',augment_coords=False)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    model = RNA3DModel(VOCAB_SIZE, MAX_LENGTH, H, D_K, D_V, D_MODEL, D_FF, N_LAYERS, DROPOUT_RATE)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*EPOCHS, eta_min=1e-5)
    
    loss_fn = AlignSVDMSELoss()
    loss_fn = loss_fn.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        tm_scores = 0
        rmsd_scores = 0
    
        for batch_idx, (inputs, y_true) in enumerate(train_loader):
            # Move data to the device.
            inputs = tuple(inp.to(device) for inp in inputs if inp is not None)
            y_true = y_true.to(device)
    
            optimizer.zero_grad()
            # Forward pass.
            y_pred = model(inputs)
            # Here y_pred is assumed to have shape (B, seq_len, 3)
            loss = loss_fn(y_true, y_pred)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            rmsd_scores += compute_rmsd(y_true.detach().cpu(), y_pred.detach().cpu())
            tm_scores += compute_tm_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
    
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
        avg_loss = running_loss / len(train_loader)
        avg_rmsd = rmsd_scores / len(train_loader)
        avg_tm = tm_scores / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.4f}, RMSD: {avg_rmsd:.4f}, TM: {avg_tm:.4f}")
    
        # Evaluate on validation set and compute TM-score.
        tm_scores = []
        model.eval()
        with torch.no_grad():
            for inputs, y_true in val_loader:
                inputs = tuple(inp.to(device) for inp in inputs if inp is not None)
                y_true = y_true.to(device)
                y_pred = model(inputs)
                # Convert to numpy and compute TM-score per sample.
                y_true_np = y_true.cpu().numpy()
                y_pred_np = y_pred.cpu().numpy()
                tm = compute_tm_score(y_true_np, y_pred_np)
                tm_scores.append(tm)
        avg_tm = np.mean(tm_scores) if tm_scores else 0.0
        print(f"Epoch [{epoch+1}/{EPOCHS}] Validation TM-score: {avg_tm:.4f}")
        model.train()
    
        scheduler.step()
    
    # Save the model weights
    torch.save(model.state_dict(), f"StanfordRNA3D_model_{VER}.pth")


del val_loader, train_loader, train_dataset, val_dataset,model,inputs,y_true, y_pred


import gc
gc.collect()


import torch
torch.cuda.empty_cache()


torch.cuda.reset_max_memory_allocated()
torch.cuda.reset_peak_memory_stats()


submission = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/sample_submission.csv")


test_seq[["freq_matrix","ppsm","coverience"]] = test_seq["target_id"].apply(lambda x: parse_msa_fasta("/kaggle/input/stanford-rna-3d-folding/MSA/" + x + ".MSA.fasta")).apply(pd.Series)


test_dataset = RNADataset(test_seq, TOKENIZER, freq_matrix=True, pssm=True, covariance=True,
                             max_length=MAX_LENGTH,augment_coords=False, mode='test')
    
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


model = RNA3DModel(VOCAB_SIZE, MAX_LENGTH, H, D_K, D_V, D_MODEL, D_FF, N_LAYERS, DROPOUT_RATE)
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


model.load_state_dict(torch.load('/kaggle/working/StanfordRNA3D_model_0.1.pth',weights_only=True))


import gc
gc.collect()


import torch
torch.cuda.empty_cache()


preds = torch.zeros(len(test_loader),MAX_LENGTH,5,3)
for i in range(5):
    for j,inputs in enumerate(test_loader):
        inputs = tuple(inp.to(device) for inp in inputs if inp is not None)
        with torch.inference_mode():
            pred = model(inputs)
        del inputs
        torch.cuda.empty_cache()
        preds[j,:,i,:] = pred


preds.shape


pred_cords = preds.reshape(-1,MAX_LENGTH,15)


c = 0
for i,(target_id,row) in enumerate(test_seq.iterrows()):
    sequence = list(row["sequence"])
    seq_len = len(sequence)
    for j in range(seq_len):
        submission.loc[c,:] = [target_id+f"_{j+1}",sequence[j],j+1] + pred_cords[i,j,:].tolist()
        c += 1


submission.to_csv("submission.csv",index=False)

