# Cell 1: Imports
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import uuid
from tqdm import tqdm
import random

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from scipy.interpolate import interp1d

torch.manual_seed(42)
np.random.seed(42)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    for col in submission.columns:
        if not pd.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')

    NEGATIVE_PART = -299
    LARGEST_CHUNK = 600
    SMALLEST_CHUNK = 350
    TOTAL_REALIZATIONS = 10
    INFLATION_SIGMA = 600

    sigma_2 = np.ones((LARGEST_CHUNK + NEGATIVE_PART - 1))
    from_ranges = [1, 61, 245]
    to_ranges_excl = [61, 245, 301]
    log_slopes = [1.0406028049510443, 0.0, 7.835345062351012]
    log_offsets = [-6.430669850650689, -2.1617411566043896, -45.24876794412965]

    for growth_mode in range(len(from_ranges)):
        for i in range(from_ranges[growth_mode], to_ranges_excl[growth_mode]):
            sigma_2[i - 1] = np.exp(np.log(i) * log_slopes[growth_mode] + log_offsets[growth_mode])

    sigma_2 *= INFLATION_SIGMA
    cov_matrix_inv_diag = 1. / sigma_2  # Inverse of the diagonal elements

    num_rows = solution.shape[0]
    p = 1. / TOTAL_REALIZATIONS
    ps = np.full((num_rows, TOTAL_REALIZATIONS), p)

    log_p = -np.log(TOTAL_REALIZATIONS)
    inner_product_matrix = np.zeros((num_rows, TOTAL_REALIZATIONS))
    num_columns = LARGEST_CHUNK + NEGATIVE_PART - 1

    # collecting solution and submission in numpy arrays
    full_submission = np.zeros((num_rows, TOTAL_REALIZATIONS, num_columns))
    full_solution = np.zeros((num_rows, TOTAL_REALIZATIONS, num_columns))

    # Iterating through column names
    for k in range(TOTAL_REALIZATIONS):
        # compute misfits in individual columns
        # Create an array filled with zeros
        misfit = np.zeros((num_rows, num_columns))
        for i in range(num_columns):
            if k == 0:
                column_name = str(i + 1)
            else:
                column_name = f"r_{k}_pos_{i + 1}"
            # this needs to be different cov matrix
            misfit[:, i] = solution[column_name].values - submission[column_name].values
            full_submission[:, k, i] = submission[column_name].values
            full_solution[:, k, i] = solution[column_name].values
        misfit_scaled = misfit * cov_matrix_inv_diag
        inner_product = np.sum(misfit_scaled * misfit, axis=1)
        # exp_misfit_cur = np.exp(inner_product)
        # exp_misfit[:, k] = exp_misfit_cur
        inner_product_matrix[:, k] = inner_product #+ log_p

    # Find normalization constant
    # max_vals = 100*np.ones_like(inner_product_matrix[:,0]) # np.zeros_like(inner_product_matrix[:,0]) # np.max(inner_product_matrix, axis=1)
    max_vals = np.max(inner_product_matrix, axis=1)
    # add and subtract max_vals to avoid numerical instability
    product = np.exp(inner_product_matrix + log_p - max_vals[:, None])
    log_sum_exp = max_vals + np.log(np.sum(product, axis=1))

    nll = -log_sum_exp
    computed_score = nll.mean()

    # Check for infinite scores
    if np.isinf(computed_score):
        raise ParticipantVisibleError(f"Your score is {computed_score}, which means there is room for improvement.")

    return computed_score


def extract_features(X, valid_mask):
    """
    X:           (N, 300) array of values (e.g., padded or filled sequences)
    valid_mask:  (N, 300) boolean array, True where values are valid
    """
    features = []
    for row, mask in zip(X, valid_mask):
        valid = row[mask]
        if len(valid) == 0:
            stats = [0.0] * 8
        else:
            mean = valid.mean()
            std = valid.std()
            min_val = valid.min()
            max_val = valid.max()
            count = len(valid) / 300.0
            slope = np.polyfit(np.arange(len(valid)), valid, 1)[0]
            stats = [mean, std, min_val, max_val, count, valid[0], valid[-1], slope]
        features.append(stats)
    return np.array(features, dtype=np.float32)

####################################################################################
# Interpolation strategies
####################################################################################

def extrapolate_slope(X_values, valid_mask, forecast_horizon=300, fit_window=150):
    """
    Extrapolate slope-based forecast from valid values.

    Parameters:
        X_values (ndarray): (N, 300) array of input values (may include filled values).
        valid_mask (ndarray): (N, 300) boolean mask indicating valid values (True = valid).
        forecast_horizon (int): Length of forecast to produce (default = 300).
        fit_window (int): Number of valid tail points to fit the slope on.

    Returns:
        ndarray: (N, forecast_horizon) array of extrapolated slope forecasts.
    """
    results = []
    for seq, mask in zip(X_values, valid_mask):
        valid_vals = seq[mask]
        if len(valid_vals) < 2:
            forecast = np.zeros(forecast_horizon, dtype=np.float32)
        else:
            tail = valid_vals[-fit_window:] if len(valid_vals) >= fit_window else valid_vals
            x_vals = np.arange(-len(tail), 0)  # e.g., [-10, ..., -1]
            y_vals = tail

            model = LinearRegression()
            model.fit(x_vals.reshape(-1, 1), y_vals)

            x_forecast = np.arange(forecast_horizon)
            forecast = model.coef_[0] * x_forecast
        results.append(forecast)

    return np.array(results, dtype=np.float32)


def extrapolate_last_value(x_seq, forecast_horizon=300):
    return np.array([
        np.ones(forecast_horizon) * seq[seq != 0][-1] if np.any(seq != 0) else np.zeros(forecast_horizon)
        for seq in x_seq
    ])

def extrapolate_with_mean(x_seq, forecast_horizon=300):
    return np.array([
        np.ones(forecast_horizon) * seq[seq != 0].mean() if np.any(seq != 0) else np.zeros(forecast_horizon)
        for seq in x_seq
    ])

def compute_anchor_positions(num_anchors, max_len=300):
    if num_anchors == 1:
        return np.array([max_len - 1])  # Just the final step
    return np.linspace(0, max_len - 1, num=num_anchors).astype(int)


###############################################################################
# Fill past values
##############################################################################

def fill_past_values(X_raw, mode="lin", total_len=300):
    """
    Fill the leading missing values in each row of X_raw.

    Parameters:
        X_raw (ndarray): Raw input array with shape (N, 300) and NaNs for missing values.
        mode (str): One of {"lin", "zero", "mean"}.
        total_len (int): Desired total length after filling (default = 300).

    Returns:
        X_filled (ndarray): Array of shape (N, 300) with filled sequences.
    """
    X_filled = []
    for row in X_raw:
        valid_idx = np.where(~np.isnan(row))[0]

        if len(valid_idx) == 0:
            filled = np.zeros(total_len, dtype=np.float32)

        else:
            first_idx = valid_idx[0]
            last_idx = valid_idx[-1]
            first_val = row[first_idx]
            last_val = row[last_idx]

            tail = row[first_idx:].astype(np.float32)
            tail[np.isnan(tail)] = 0.0

            num_missing = total_len - len(tail)

            if mode == "zero":
                extrapolated = [0.0] * num_missing

            elif mode == "mean":
                mean_val = np.nanmean(row)
                extrapolated = [mean_val] * num_missing

            elif mode == "lin":
                slope = (last_val - first_val) / (last_idx - first_idx) if last_idx != first_idx else 0.0
                extrapolated = [first_val - slope * i for i in range(num_missing, 0, -1)]

            else:
                raise ValueError(f"Unknown mode: {mode}. Choose from 'lin', 'zero', 'mean'.")

            filled = np.array(extrapolated + tail.tolist(), dtype=np.float32)

        X_filled.append(filled)

    return np.stack(X_filled)

def scale_with_robust_scaler(X):
    N, L = X.shape
    reshaped = X.reshape(-1, 1)

    scaler = RobustScaler()
    scaled = scaler.fit_transform(reshaped)

    X_scaled = scaled.reshape(N, L)
    return X_scaled.astype(np.float32), scaler



class GeoStatDataset(Dataset):
    def __init__(self, X_padded, X_padded_scaled, X_stats, X_stats_scaled, valid_mask, y, y_scaled):
        self.X_padded = torch.tensor(X_padded).float()
        self.X_padded_scaled = torch.tensor(X_padded_scaled).float()
        self.X_stats = torch.tensor(X_stats).float()
        self.X_stats_scaled = torch.tensor(X_stats_scaled).float()
        self.y = torch.tensor(y).float()
        self.y_scaled = torch.tensor(y_scaled).float()

        slope_preds = extrapolate_slope(X_padded, valid_mask)
        self.slope_preds = torch.tensor(slope_preds).float()
        self.residuals = self.y - self.slope_preds

        slope_preds_scaled = extrapolate_slope(X_padded_scaled, valid_mask)
        self.slope_preds_scaled = torch.tensor(slope_preds_scaled).float()
        self.residuals_scaled = self.y_scaled - self.slope_preds_scaled

        self.match_mask = []
        for xi, yi in zip(X_padded, y):
            valid_xi = xi[xi != 0.0]
            if len(valid_xi) < 2:
                self.match_mask.append(False)
            else:
                input_trend = valid_xi[0] < valid_xi[-1]
                output_trend = yi[0] < yi[-1]
                self.match_mask.append(input_trend == output_trend)
        self.match_mask = torch.tensor(self.match_mask, dtype=torch.bool)

    def __len__(self):
        return len(self.X_stats)

    def __getitem__(self, idx):
        return {
            "x_padded": self.X_padded[idx],
            "x_padded_scaled": self.X_padded_scaled[idx],
            "x_stats": self.X_stats[idx],
            "x_stats_scaled": self.X_stats_scaled[idx],
            "y": self.y[idx],      
            "y_scaled": self.y_scaled[idx],
            "slope_pred": self.slope_preds[idx],
            "residuals": self.residuals[idx], 
            "residuals_scaled": self.residuals_scaled[idx]
        }


def sample_anchors(mean, logvar, num_samples=1):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn((num_samples, *mean.shape), device=mean.device)
    return mean.unsqueeze(0) + eps * std.unsqueeze(0)  # (S, B, 11)

def interpolate_anchor_predictions(anchor_preds, anchor_indices, max_len=300):
    x_full = torch.arange(max_len, device=anchor_preds.device).float()
    anchor_x = anchor_indices.float()  # (A,)

    def _manual_interp(y):  # y: (B, A)
        B, A = y.shape
        x = x_full.unsqueeze(0).expand(B, -1)  # (B, max_len)

        # Find which interval each x belongs to
        idx = torch.searchsorted(anchor_x, x, right=True).clamp(1, A - 1)  # (B, max_len)
        x0 = anchor_x[idx - 1]  # (B, max_len)
        x1 = anchor_x[idx]      # (B, max_len)

        y0 = torch.gather(y, 1, idx - 1)
        y1 = torch.gather(y, 1, idx)

        # Linear interpolation
        weight = (x - x0) / (x1 - x0 + 1e-8)
        return y0 + weight * (y1 - y0)

    if anchor_preds.dim() == 2:  # (B, A)
        return _manual_interp(anchor_preds)

    elif anchor_preds.dim() == 3:  # (S, B, A)
        S, B, A = anchor_preds.shape
        return torch.stack([_manual_interp(anchor_preds[s]) for s in range(S)], dim=0)  # (S, B, max_len)

    else:
        raise ValueError("anchor_preds must be of shape (B, A) or (S, B, A)")


num_anchors = 8
num_samples = 64
max_len = 300


class AnchorMLP(nn.Module):
    def __init__(self, input_dim=8 + num_samples, hidden_dim=64, num_anchors=1, num_layers=4, activation_fn=nn.Tanh):
        super().__init__()

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation_fn())
            in_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, num_anchors)

    def forward(self, x_stats, x_samples):
        net_input = torch.cat((x_stats, x_samples), dim=1)
        features = self.feature_extractor(net_input)
        mean = self.head(features)
        return mean

def train_anchor_mlp(
    model,
    train_loader,
    val_loader,
    anchor_positions,
    samples_positions,
    robust_scaler,
    device="cuda",
    epochs=100,
    lr=1e-3,
    model_path="best_model.pt",
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_nll = float("-inf")
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x_stats = batch["x_stats_scaled"].to(device)
            x_samples = batch["x_padded_scaled"][:, samples_positions].to(device)
            residuals = batch["residuals_scaled"].to(device)

            optimizer.zero_grad()
            mean = model(x_stats, x_samples)

            mean = interpolate_anchor_predictions(mean, torch.tensor(anchor_positions, device=mean.device), max_len=300)
            loss = loss_fn(mean, residuals)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # === Validation ===
        model.eval()
        val_preds, val_targets, all_realizations = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                x_stats = batch["x_stats_scaled"].to(device)
                x_samples = batch["x_padded_scaled"][:, samples_positions].to(device)
                yb = batch["y"].to(device)
                slope_pred = batch["slope_pred"].cpu().numpy()

                mean = model(x_stats, x_samples)
                mean_np = mean.detach().cpu().numpy()
                mean = torch.tensor(robust_scaler.inverse_transform(mean_np)).to(device)

                interp_mean = interpolate_anchor_predictions(mean, torch.tensor(anchor_positions, device=mean.device), max_len=300).detach().cpu().numpy()
                final_pred = slope_pred + interp_mean
                val_preds.append(final_pred)
                final_realizations = np.stack([slope_pred + interp_mean for _ in range(10)], axis=0)  # shape: (10, B, 300)
                all_realizations.append(final_realizations)
                val_targets.append(yb.detach().cpu().numpy())

        # Score
        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        all_realizations = np.concatenate(all_realizations, axis=1)

        cols = [str(i) for i in range(1, 301)]
        submission_blocks = [pd.DataFrame(all_realizations[0], columns=cols)]
        for r in range(1, 10):
            submission_blocks.append(pd.DataFrame(
                all_realizations[r],
                columns=[f"r_{r}_pos_{i}" for i in range(1, 301)]))
        submission_df = pd.concat(submission_blocks, axis=1)
        submission_df.insert(0, "id", np.arange(len(val_preds)))

        solution_blocks = [pd.DataFrame(val_targets, columns=cols)]
        for r in range(1, 10):
            solution_blocks.append(pd.DataFrame(
                val_targets,
                columns=[f"r_{r}_pos_{i}" for i in range(1, 301)]))
        solution_df = pd.concat(solution_blocks, axis=1)
        solution_df.insert(0, "id", np.arange(len(val_targets)))

        nll_score = score(solution_df, submission_df, row_id_column_name="id")
        val_mse = np.mean((val_preds - val_targets) ** 2)

        print(f"Epoch {epoch+1:3d} | Train MSE: {total_loss:.4f} | Val MSE: {val_mse:.4f} | Val NLL: {nll_score:.4f}")

        if nll_score > best_nll:
            best_nll = nll_score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_path)
            print(f"âœ… New best model saved at epoch {best_epoch} with NLL: {best_nll:.4f}")

    print(f"\nğŸ�� Best model was from epoch {best_epoch} with NLL Score: {best_nll:.4f}")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model



class AnchorLSTM(nn.Module):
    def __init__(self, input_dim=1, stat_dim=8, hidden_dim=64,
                 num_samples=64, num_anchors=64, num_layers=2, dropout=0.2):
        super().__init__()

        self.num_anchors = num_anchors
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.stat_proj = nn.Linear(stat_dim, hidden_dim)

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_anchors)
        )

    def forward(self, x_stats, x_samples):
        B = x_samples.size(0)

        x_seq = x_samples.unsqueeze(-1)
        lstm_out, _ = self.lstm(x_seq)
        context = lstm_out[:, -1, :]
        stat_context = self.stat_proj(x_stats)
        combined = context + stat_context
        preds = self.output_proj(combined)
        return preds


def train_anchor_lstm(
    model,
    train_loader,
    val_loader,
    anchor_positions,
    samples_positions,
    robust_scaler,
    device="cuda",
    epochs=100,
    lr=1e-3,
    model_path="best_lstm.pt",
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_nll = float("-inf")
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x_stats = batch["x_stats_scaled"].to(device)
            x_samples = batch["x_padded_scaled"][:, samples_positions].to(device)  # (B, num_samples)
            residuals = batch["residuals_scaled"].to(device)  # (B, 300)

            optimizer.zero_grad()
            res_mean = model(x_stats=x_stats, x_samples=x_samples)  # (B, num_anchors)

            res_mean_interp = interpolate_anchor_predictions(
                res_mean,
                torch.tensor(anchor_positions, device=res_mean.device),
                max_len=300
            )

            loss = loss_fn(res_mean_interp, residuals)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # === Validation ===
        model.eval()
        val_preds, val_targets, all_realizations = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                x_stats = batch["x_stats_scaled"].to(device)
                x_samples = batch["x_padded_scaled"][:, samples_positions].to(device)
                yb = batch["y"].to(device)
                slope_pred = batch["slope_pred"].cpu().numpy()

                res_mean = model(x_stats, x_samples)  # (B, num_anchors)
                res_mean_np = res_mean.detach().cpu().numpy()
                res_mean_np = robust_scaler.inverse_transform(res_mean_np)  # (B, num_anchors)

                interp_residuals = interpolate_anchor_predictions(
                    torch.tensor(res_mean_np, device=device),
                    torch.tensor(anchor_positions, device=device),
                    max_len=300
                ).cpu().numpy()  # (B, 300)

                final_pred = slope_pred + interp_residuals  # (B, 300)
                val_preds.append(final_pred)
                all_realizations.append(np.stack([final_pred] * 10, axis=0))  # (10, B, 300)
                val_targets.append(yb.detach().cpu().numpy())

        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        all_realizations = np.concatenate(all_realizations, axis=1)  # (10, N, 300)

        # Prepare for submission
        cols = [str(i) for i in range(1, 301)]
        submission_blocks = [pd.DataFrame(all_realizations[0], columns=cols)]
        for r in range(1, 10):
            submission_blocks.append(pd.DataFrame(
                all_realizations[r],
                columns=[f"r_{r}_pos_{i}" for i in range(1, 301)]
            ))
        submission_df = pd.concat(submission_blocks, axis=1)
        submission_df.insert(0, "id", np.arange(len(val_preds)))

        solution_blocks = [pd.DataFrame(val_targets, columns=cols)]
        for r in range(1, 10):
            solution_blocks.append(pd.DataFrame(
                val_targets,
                columns=[f"r_{r}_pos_{i}" for i in range(1, 301)]
            ))
        solution_df = pd.concat(solution_blocks, axis=1)
        solution_df.insert(0, "id", np.arange(len(val_targets)))

        # Evaluate
        nll_score = score(solution_df, submission_df, row_id_column_name="id")
        val_mse = np.mean((val_preds - val_targets) ** 2)

        print(f"Epoch {epoch+1:3d} | Train MSE: {total_loss:.4f} | Val MSE: {val_mse:.4f} | Val NLL: {nll_score:.4f}")

        if nll_score > best_nll:
            best_nll = nll_score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_path)
            print(f"âœ… New best model saved at epoch {best_epoch} with NLL: {best_nll:.4f}")

    print(f"\nğŸ�� Best model was from epoch {best_epoch} with NLL Score: {best_nll:.4f}")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model



class AnchorTransformer(nn.Module):
    def __init__(self, input_dim=1, stat_dim=8, hidden_dim=64, num_samples=64, num_anchors=64,
                 nhead=4, num_layers=2, pos_enc_dim=16):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc_dim = pos_enc_dim

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )

        self.positional_proj = nn.Linear(pos_enc_dim, hidden_dim)
        self.fc_input = nn.Linear(hidden_dim + stat_dim + num_samples, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_anchors)  # Only mean

        self.register_buffer("sample_positions", torch.tensor(
            compute_anchor_positions(num_samples, max_len=300), dtype=torch.float32))

    def forward(self, x_stats, x_samples):
        B, A = x_samples.shape
        x_seq = x_samples.unsqueeze(-1)  # (B, A, 1)
        x_proj = self.input_proj(x_seq)  # (B, A, H)

        # Positional encoding
        pos = self.sample_positions.unsqueeze(1)  # (A, 1)
        i = torch.arange(self.pos_enc_dim, device=x_samples.device).unsqueeze(0)
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / self.pos_enc_dim)
        angle_rads = pos * angle_rates
        pos_encoding = torch.zeros_like(angle_rads)
        pos_encoding[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        pos_encoding[:, 1::2] = torch.cos(angle_rads[:, 1::2])
        pos_encoding = self.positional_proj(pos_encoding)
        pos_encoding = pos_encoding.unsqueeze(0).expand(B, -1, -1)  # (B, A, H)

        x_proj = x_proj + pos_encoding
        trans_out = self.transformer(x_proj)  # (B, A, H)
        pooled = trans_out.mean(dim=1)  # (B, H)

        full_input = torch.cat([pooled, x_stats, x_samples], dim=1)
        h = F.relu(self.fc_input(full_input))
        mean = self.head(h)  # (B, num_anchors)
        return mean

def train_anchor_transformer(
    model,
    train_loader,
    val_loader,
    samples_positions,
    anchor_positions,
    robust_scaler,
    device="cuda",
    epochs=100,
    lr=1e-3,
    model_path="best_transformer_model.pt",
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_nll = float("-inf")
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x_stats = batch["x_stats_scaled"].to(device)
            x_samples = batch["x_padded_scaled"][:, samples_positions].to(device)
            residuals = batch["residuals_scaled"].to(device)

            optimizer.zero_grad()
            mean = model(x_stats, x_samples)
            mean = interpolate_anchor_predictions(mean, torch.tensor(anchor_positions, device=device), max_len=300)
            loss = loss_fn(mean, residuals)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # === Validation ===
        model.eval()
        val_preds, val_targets, all_realizations = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                x_stats = batch["x_stats_scaled"].to(device)
                x_samples = batch["x_padded_scaled"][:, samples_positions].to(device)
                yb = batch["y"].to(device)
                slope_pred = batch["slope_pred"].cpu().numpy()

                mean = model(x_stats, x_samples)
                mean_np = mean.detach().cpu().numpy()
                mean = torch.tensor(robust_scaler.inverse_transform(mean_np)).to(device)

                interp_mean = interpolate_anchor_predictions(mean, torch.tensor(anchor_positions, device=device), max_len=300).detach().cpu().numpy()
                final_pred = slope_pred + interp_mean
                val_preds.append(final_pred)

                final_realizations = np.stack([final_pred] * 10, axis=0)  # (10, B, 300)
                all_realizations.append(final_realizations)
                val_targets.append(yb.detach().cpu().numpy())

        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        all_realizations = np.concatenate(all_realizations, axis=1)

        # Format for scoring
        cols = [str(i) for i in range(1, 301)]
        submission_blocks = [pd.DataFrame(all_realizations[0], columns=cols)]
        for r in range(1, 10):
            submission_blocks.append(pd.DataFrame(
                all_realizations[r],
                columns=[f"r_{r}_pos_{i}" for i in range(1, 301)]))
        submission_df = pd.concat(submission_blocks, axis=1)
        submission_df.insert(0, "id", np.arange(len(val_preds)))

        solution_blocks = [pd.DataFrame(val_targets, columns=cols)]
        for r in range(1, 10):
            solution_blocks.append(pd.DataFrame(
                val_targets, columns=[f"r_{r}_pos_{i}" for i in range(1, 301)]))
        solution_df = pd.concat(solution_blocks, axis=1)
        solution_df.insert(0, "id", np.arange(len(val_targets)))

        nll_score = score(solution_df, submission_df, row_id_column_name="id")
        val_mse = np.mean((val_preds - val_targets) ** 2)

        print(f"Epoch {epoch+1:3d} | Train MSE: {total_loss:.4f} | Val MSE: {val_mse:.4f} | Val NLL: {nll_score:.4f}")

        if nll_score > best_nll:
            best_nll = nll_score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_path)
            print(f"âœ… New best model saved at epoch {best_epoch} with NLL: {best_nll:.4f}")

    print(f"\nğŸ�� Best model was from epoch {best_epoch} with NLL Score: {best_nll:.4f}")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model



# Load original train data
DATA_PATH = '/kaggle/input/geology-forecast-challenge-2025/data'
train_df = pd.read_csv(f"{DATA_PATH}/train.csv")

# Define relevant columns
input_cols = [str(i) for i in range(-299, 1)]
target_cols = [str(i) for i in range(1, 301)]
sim_target_blocks = [
    [f"r_{r}_pos_{i}" for i in range(1, 301)] for r in range(1, 10)
]

# Reconstruct rows with proper string column names
expanded_rows = []

for i, row in train_df.iterrows():
    base_input = row[input_cols].values.astype(np.float32)
    geology_id = row["geology_id"]

    # Main realization
    if not row[target_cols].isnull().any():
        expanded_rows.append({
            **{str(col): base_input[j] for j, col in enumerate(input_cols)},  # force string
            **{str(col): row[str(col)] for col in range(1, 301)},             # target cols as str
            "geology_id": geology_id
        })

    # Simulated realizations
    for r_idx, sim_cols in enumerate(sim_target_blocks, 1):
        if all(pd.notna(row[col]) for col in sim_cols):
            expanded_rows.append({
                **{str(col): base_input[j] for j, col in enumerate(input_cols)},
                **{str(i): row[f"r_{r_idx}_pos_{i}"] for i in range(1, 301)},
                "geology_id": geology_id
            })

# Construct with string-typed columns
expanded_train_df = pd.DataFrame(expanded_rows)

# Sort columns to match train_df ordering
ordered_columns = input_cols + target_cols + ["geology_id"]
expanded_train_df = expanded_train_df[ordered_columns]

print("âœ… Columns match train_df format:", expanded_train_df.columns.equals(train_df[ordered_columns].columns))
print(f"âœ… expanded_train_df shape: {expanded_train_df.shape}")


use_expanded = False
if not use_expanded:
    X_raw = train_df[input_cols].values
    y = train_df[target_cols].values.astype(np.float32)
else:
    X_raw = expanded_train_df[input_cols].values
    y = expanded_train_df[target_cols].values.astype(np.float32)

# Extract sequences (remove NaNs), pad with leading zeros
X_seq = [row[~np.isnan(row)].tolist() for row in X_raw]
valid_mask = ~np.isnan(X_raw)  # True where data is valid

# Modify X_padded to extrapolate backwards using slope from first to last defined value
filling_mode = "lin"
X_padded = fill_past_values(X_raw, mode=filling_mode, total_len=300)
X_padded_scaled, robust_scaler = scale_with_robust_scaler(X_padded)
y_scaled = robust_scaler.transform(y.reshape(-1, 1)).reshape(-1, 300)

X_stats = extract_features(X_padded, valid_mask)
X_stats_scaled = extract_features(X_padded_scaled, valid_mask)

anchor_positions = compute_anchor_positions(num_anchors, max_len=300)
samples_positions = compute_anchor_positions(num_samples, max_len=300)
print("Anchors:", anchor_positions)
print("Samples:", samples_positions)

y_anchors = y[:, anchor_positions]

geology_ids = train_df["geology_id"].unique()

train_ids, val_ids = train_test_split(
    geology_ids, test_size=0.2, random_state=42, shuffle=True
)

if not use_expanded:
    train_mask = train_df["geology_id"].isin(train_ids).values
    val_mask   = train_df["geology_id"].isin(val_ids).values
else:
    train_mask = expanded_train_df["geology_id"].isin(train_ids).values
    val_mask   = expanded_train_df["geology_id"].isin(val_ids).values

X_train               = X_padded[train_mask]
X_val                 = X_padded[val_mask]

X_scaled_train        = X_padded_scaled[train_mask]
X_scaled_val          = X_padded_scaled[val_mask]

X_stats_train         = X_stats[train_mask]
X_stats_val           = X_stats[val_mask]

X_stats_scaled_train  = X_stats_scaled[train_mask]
X_stats_scaled_val    = X_stats_scaled[val_mask]

valid_mask_train      = valid_mask[train_mask]
valid_mask_val        = valid_mask[val_mask]

y_train               = y[train_mask]
y_val                 = y[val_mask]

y_train_scaled        = y_scaled[train_mask]
y_val_scaled          = y_scaled[val_mask]

y_train_anchors       = y_anchors[train_mask]
y_val_anchors         = y_anchors[val_mask]

print("X_train shape:", X_train.shape)
print("X_scaled_train shape:", X_scaled_train.shape)
print("X_stats_train shape:", X_stats_train.shape)
print("X_stats_scaled_train shape:", X_stats_scaled_train.shape)
print(y_train.shape)
print(y_train_scaled.shape)


import matplotlib.pyplot as plt
import numpy as np

# Parameters
num_to_plot = 100
n_rows, n_cols = 10, 10
X_axis = np.arange(-299, 1)
Y_axis = np.arange(1, 301)
Full_axis = np.concatenate([X_axis, Y_axis])  # -299 to 300

# Subset data
X_part = X_padded[:num_to_plot]       # (N, 300)
Y_part = y[:num_to_plot]              # (N, 300)
mask_part = valid_mask[:num_to_plot]  # (N, 300)

# Plot
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20), sharex=True, sharey=True)

for i, ax in enumerate(axes.flat):
    x_vals = X_part[i]
    y_vals = Y_part[i]
    mask = mask_part[i]

    # Plot original (valid) X
    ax.plot(X_axis[mask], x_vals[mask], 'b-', label="Original", linewidth=1)
    # Plot filled X
    ax.plot(X_axis[~mask], x_vals[~mask], 'r--', label="Filled", linewidth=1)
    # Plot Y targets
    ax.plot(Y_axis, y_vals, 'g-', label="Target Y", linewidth=1)

    ax.set_title(f"Sample {i}", fontsize=8)
    ax.grid(True)
    ax.tick_params(labelsize=6)

# Add legend to one axis
axes[0, 0].legend(fontsize=8)

fig.suptitle("Sequences with Original X (blue), Filled X (red), and Y (green)", fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()



def determine_trend(x, y):
    """Returns True if input and output trends match (increasing/decreasing)."""
    valid_x = x[x != 0.0]
    if len(valid_x) < 2:
        return False  # Can't determine trend reliably
    input_trend = valid_x[0] < valid_x[-1]
    output_trend = y[0] < y[-1]
    return input_trend == output_trend

# --- Baseline Predictions ---
slope_preds = extrapolate_slope(X_val, valid_mask_val, forecast_horizon=300)
last_val_preds = extrapolate_last_value(X_val, forecast_horizon=300)
mean_val_preds = extrapolate_with_mean(X_val, forecast_horizon=300)

# --- Trend-aware "cheating" slope extrapolation ---
cheating_preds = []
for xi, yi, si in zip(X_val, y_val, slope_preds):
    match = determine_trend(xi, yi)
    if match:
        cheating_preds.append(si)
    else:
        cheating_preds.append(-si)  # flip direction if trend mismatches

cheating_preds = np.array(cheating_preds)

# --- Evaluation ---
mse_slope = mean_squared_error(y_val, slope_preds)
mse_last  = mean_squared_error(y_val, last_val_preds)
mse_mean = mean_squared_error(y_val, mean_val_preds)
mse_cheat = mean_squared_error(y_val, cheating_preds)

print(f"Baseline MSE - Slope Extrapolation: {mse_slope:.4f}")
print(f"Baseline MSE - Last Value Constant: {mse_last:.4f}")
print(f"Baseline MSE - Mean Value Constant: {mse_mean:.4f}")
print(f"Cheating MSE - Trend-aware Slope Flip: {mse_cheat:.4f}")


train_ds = GeoStatDataset(X_train, X_scaled_train, X_stats_train, X_stats_scaled_train, valid_mask_train, y_train, y_train_scaled)
val_ds = GeoStatDataset(X_val, X_scaled_val, X_stats_val, X_stats_scaled_val, valid_mask_val, y_val, y_val_scaled)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)


model_type = "mlp"  # Options: "mlp", "lstm", "transformer"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_type == "mlp":
    model = AnchorMLP(
        input_dim=8 + num_samples,
        hidden_dim=64,
        num_anchors=num_anchors,
        num_layers=8
    ).to(device)

elif model_type == "lstm":
    model = AnchorLSTM(
        input_dim=1,
        hidden_dim=128,
        num_samples=num_samples,
        num_anchors=num_anchors,
        num_layers=8
    ).to(device)

elif model_type == "transformer":
    model = AnchorTransformer(
         input_dim=1, 
         stat_dim=8, 
         hidden_dim=128, 
         num_samples=num_samples, 
         num_anchors=num_anchors, 
         nhead=4, 
         num_layers=2,
         pos_enc_dim=16
    ).to(device)
else:
    raise ValueError(f"Unknown model_type: {model_type}")

model_path = '/kaggle/working/best_model.pt'
best_nll = float("-inf")
best_epoch = -1


epochs = 10
if model_type == "mlp":
    model = train_anchor_mlp(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        anchor_positions=anchor_positions,
        samples_positions=samples_positions,
        robust_scaler=robust_scaler,
        device=device,
        epochs=epochs,
        lr=1e-3,
        model_path=model_path,
    )
elif model_type == "lstm":
    model = train_anchor_lstm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        samples_positions=samples_positions,
        anchor_positions=anchor_positions,
        robust_scaler=robust_scaler,
        device=device,
        epochs=epochs,
        lr=1e-3,
        model_path=model_path,
    )
elif model_type == "transformer":
    model = train_anchor_transformer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        samples_positions=samples_positions,
        anchor_positions=anchor_positions,
        robust_scaler=robust_scaler,
        device=device,
        epochs=epochs,
        lr=1e-3,
        model_path=model_path,
    )


def predict_residuals_with_mean(model, batch, samples_positions, anchor_positions, robust_scaler, device):
    model.eval()
    with torch.no_grad():
        xb = batch["x_stats_scaled"].to(device)
        x_samples = batch["x_padded_scaled"][:, samples_positions].to(device)
        mean = model(xb, x_samples)

        mean_np = mean.detach().cpu().numpy()
        mean = torch.tensor(robust_scaler.inverse_transform(mean_np)).to(device)

        interp_mean = interpolate_anchor_predictions(
            mean, torch.tensor(anchor_positions, device=device), max_len=300
        ).detach().cpu().numpy()

        slope_pred = batch["slope_pred"].cpu().numpy()  # (1, 300)
        final_realization = slope_pred + interp_mean  # (1, 300)

        return final_realization[0]  # (300,)

sample_indices = random.sample(range(len(val_ds)), 30)
fig, axes = plt.subplots(6, 5, figsize=(28, 20), sharex=True, sharey=True)

for idx, ax in zip(sample_indices, axes.flatten()):
    batch = val_ds[idx]
    batch = {k: v.unsqueeze(0) for k, v in batch.items()}
    prediction = predict_residuals_with_mean(model, batch, samples_positions, anchor_positions, robust_scaler, device)
    true = batch["y"].squeeze().numpy()

    ax.plot(true, label="Ground Truth", color="black", linewidth=2)
    ax.plot(prediction, color="blue", linewidth=1.5, label="Prediction")
    ax.set_title(f"Sample {idx}", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True)

# Legend and layout
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=16)
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.suptitle(f"Validation Fit for 30 Samples ({model_type.upper()})", fontsize=22)
plt.show()


import pandas as pd
import numpy as np
import torch

# === CONFIG ===
model_path = "best_model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load test data ---
test_df = pd.read_csv(f"{DATA_PATH}/test.csv")
X_test_raw = test_df[[str(i) for i in range(-299, 1)]].values
valid_mask_test = ~np.isnan(X_test_raw)

# Pad missing data
X_test_padded = fill_past_values(X_test_raw, mode="mean", total_len=300)

# Scale inputs
X_test_padded_scaled = robust_scaler.transform(X_test_padded.reshape(-1, 1)).reshape(-1, 300)

# Extract features and samples
X_test_stats_scaled = torch.tensor(extract_features(X_test_padded_scaled, valid_mask_test), dtype=torch.float32).to(device)
X_test_samples_scaled = torch.tensor(X_test_padded_scaled[:, samples_positions], dtype=torch.float32).to(device)

# Extrapolate slope (in original Z scale)
slope_preds_test = extrapolate_slope(X_test_padded, valid_mask_test, forecast_horizon=300)  # shape: (N, 300)

# --- Load model ---
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# --- Predict residual means ---
with torch.no_grad():
    mean = model(X_test_stats_scaled, X_test_samples_scaled)  # shape: (N, A)
    mean_np = mean.cpu().numpy()
    mean_unscaled = robust_scaler.inverse_transform(mean_np)  # shape: (N, A)
    mean_unscaled_tensor = torch.tensor(mean_unscaled, dtype=torch.float32).to(device)

    # Interpolate residuals to 300 positions
    interp_residuals = interpolate_anchor_predictions(
        mean_unscaled_tensor, torch.tensor(anchor_positions, device=device), max_len=300
    ).cpu().numpy()  # shape: (N, 300)

# --- Compute final Z predictions ---
final_realization = slope_preds_test + interp_residuals  # shape: (N, 300)
final_realizations = np.stack([final_realization] * 10, axis=0)  # shape: (10, N, 300)

# --- Format submission ---
cols = [str(i) for i in range(1, 301)]
submission_blocks = [pd.DataFrame(final_realizations[0], columns=cols)]
for r in range(1, 10):
    df_r = pd.DataFrame(final_realizations[r], columns=[f"r_{r}_pos_{i}" for i in range(1, 301)])
    submission_blocks.append(df_r)

submission_df = pd.concat(submission_blocks, axis=1)
submission_df.insert(0, "geology_id", test_df["geology_id"])

# --- Save ---
submission_df.to_csv("submission.csv", index=False, encoding="utf-8")
print("âœ… Saved submission to submission.csv")


