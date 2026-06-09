%%writefile dtgfn_boosted_production.py
# dtgfn_boosted_production.py – Production-Ready, Final GFN-Boost Pipeline
# ------------------------------------------------------------------------------------------------
# This version incorporates a full suite of fixes for show-stopper bugs, performance,
# and stability, based on a detailed code review. This is the definitive version.
# -------------------------------------------------------------------------------------------------
from __future__ import annotations
import math, random, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set
from collections import deque
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from tqdm import tqdm
import lightgbm as lgb
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

# ---------------------- 0. CLI Args ----------------------------------
def parse_args():
    """Parses command-line arguments."""
    p = argparse.ArgumentParser(description="Train a production-ready Boosted DT-GFN model.")
    p.add_argument("--boosting-lr", type=float, default=0.1, help="Learning rate for the boosting updates.")
    p.add_argument("--top-k-trees", type=int, default=10, help="Number of best trees to average at each boosting step.")
    p.add_argument("train", type=Path, help="Path to training data.")
    p.add_argument("test",  type=Path, help="Path to test data.")
    p.add_argument("--out",       type=Path, default="drw_preds.csv", help="Output path for predictions.")
    p.add_argument("--device",    default="cuda", help="Device for training.")
    p.add_argument("--updates",   type=int, default=50, help="Number of boosting rounds.")
    p.add_argument("--rollouts",  type=int, default=60, help="On-policy trajectories per boosting round.")
    p.add_argument("--batch",     type=int, default=8192, help="Batch size for GFN evaluation.")
    p.add_argument("--bins",      type=int, default=255, help="Number of feature bins.")
    p.add_argument("--lstm-hidden", type=int, default=512, help="Dimension of the LSTM state tracker.")
    p.add_argument("--mlp-layers", type=int, default=12, help="Number of hidden layers in MLP heads.")
    p.add_argument("--mlp-width", type=int, default=256, help="Width of hidden layers in MLP heads.")
    p.add_argument("--max-depth", type=int, default=7, help="Maximum depth for generated trees.")
    p.add_argument("--lr", type=float, default=5e-5, help="Peak learning rate for the GFN policy network schedule.")
    p.add_argument("--beta-start", type=float, default=0.35, help="Starting value for beta (split penalty).")
    p.add_argument("--beta-end", type=float, default=math.log(4), help="Final value for beta (split penalty).")
    p.add_argument("--prior-scale", type=float, default=0.5, help="Scaling factor for the gain-based prior.")
    return p.parse_args()

# ---------------------- 1. Vocab & Tok -------------------------------
@dataclass
class Vocab:
    num_feat: int; num_th: int; num_leaf: int
    PAD: int=0; BOS: int=1; EOS: int=2
    @property
    def split_start(self) -> int: return 3
    def size(self) -> int: return self.split_start + self.num_feat + self.num_th + self.num_leaf

class Tok:
    def __init__(self, v: Vocab): self.v = v
    def _feat(self, i: int) -> int: return self.v.split_start + i
    def _th(self, i: int) -> int: return self.v.split_start + self.v.num_feat + i
    def _leaf(self, i: int) -> int: return self.v.split_start + self.v.num_feat + self.v.num_th + i
    def decode_one(self, tid: int) -> Tuple[str, int]:
        if tid < self.v.split_start: raise ValueError(f"Invalid token id: {tid}")
        rem = tid - self.v.split_start
        if rem < self.v.num_feat: return "feat", rem
        rem -= self.v.num_feat
        if rem < self.v.num_th: return "th", rem
        return "leaf", rem - self.v.num_th
    def decode(self, ids: List[int]) -> List[Tuple[str, int]]:
        return [self.decode_one(i) for i in ids if i not in (self.v.BOS, self.v.EOS)]

# ---------------------- 2. Policy Architecture (LSTM) -------------------------
# No decorator on the class itself
class PolicyPaperMLP(nn.Module):
    def __init__(self, vocab_sz: int, lstm_hidden: int, mlp_layers: int, mlp_width: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz, lstm_hidden)
        self.rnn = nn.LSTM(input_size=lstm_hidden, hidden_size=lstm_hidden, num_layers=1, batch_first=True)
        layers = [nn.Linear(lstm_hidden, mlp_width), nn.ReLU()]
        for _ in range(mlp_layers - 1):
            layers.extend([nn.Linear(mlp_width, mlp_width), nn.ReLU()])
        self.shared_mlp = nn.Sequential(*layers)
        self.head_tok = nn.Linear(mlp_width, vocab_sz)
        self.head_flow = nn.Linear(mlp_width, 1)

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rnn_output, _ = self.rnn(self.embedding(seq))
        shared_output = self.shared_mlp(rnn_output)
        return self.head_tok(shared_output), self.head_flow(shared_output).squeeze(-1)

    @torch.jit.export  # <--- ADD THIS DECORATOR
    def log_prob(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.size(1) < 2: return torch.empty(seq.size(0), 0, device=seq.device)
        logits, _ = self.forward(seq[:, :-1]); return torch.gather(logits.log_softmax(dim=-1), -1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)

    @torch.jit.export  # <--- ADD THIS DECORATOR
    def log_F(self, seq: torch.Tensor) -> torch.Tensor:
        _, flow_values = self.forward(seq); return flow_values

# ---------------------- 3. Losses, Env, and Helpers -------------------
@torch.jit.script
def tb_loss(log_pf, log_pb, log_z, R, prior):
    return ((log_z + log_pf.sum(1) - (torch.log(R) + prior + log_pb.sum(1)))**2).mean()

@torch.jit.script
def fl_loss(logF, log_pf, log_pb, dE):
    return ((logF[:, :-1] + log_pf - (logF[:, 1:] + log_pb - dE))**2).mean()

class DRWEnv:
    def __init__(self, df, feats, target_col, bins, device):
        self.device = device
        self.X_full = self._featurise(df, df, feats, bins)
        self.y_full = torch.tensor(df[target_col].values, dtype=torch.float32, device=device)
        self.y = self.y_full.clone()
    def _featurise(self, df_target, df_source, feats, bins):
        X_binned = []
        for f in feats:
            s = df_source[f].replace([np.inf,-np.inf],np.nan).fillna(df_source[f].median()).values
            qs = np.linspace(0,1,bins+1); edges = np.quantile(s,qs)
            edges = np.unique(edges); edges[0] -= 1e-9; edges[-1] += 1e-9
            s_eval = df_target[f].replace([np.inf,-np.inf],np.nan).fillna(df_source[f].median()).values
            X_binned.append(np.searchsorted(edges,s_eval,side="right")-1)
        return torch.tensor(np.stack(X_binned,1).astype(np.int32), device=self.device)
    def reset(self, n: int):
        self.idxs = torch.from_numpy(np.random.choice(len(self.y),n,replace=False)).to(self.device)
        self.paths, self.open_leaves, self.done = [], 1, False
    def step(self, action: Tuple[str, int]):
        self.paths.append(action); kind, _ = action
        if kind == "feat": self.open_leaves += 1
        elif kind == "leaf": self.open_leaves -= 1
        self.done = (self.open_leaves == 0 or len(self.paths) > 8192)
    def evaluate(self, current_beta):
        prior = -current_beta * sum(1 for k, _ in self.paths if k == "feat")
        y_t = self.y[self.idxs]
        if y_t.numel() == 0: return 0.0, torch.tensor([prior], device=self.device), None, None
        X_batch = self.X_full[self.idxs]
        base_mse = ((y_t - y_t.mean())**2).mean()
        if not self.paths or self.open_leaves != 0:
            return 1 / (1 + base_mse), torch.tensor([prior], device=self.device), None, y_t
        path_iter = iter(self.paths)
        def build():
            try: k,i = next(path_iter)
            except StopIteration: return None
            if k == "feat": return {"f": i, "t": next(path_iter)[1], "L": build(), "R": build()}
            return "leaf_node"
        tree = build()
        if not tree: return 1 / (1 + base_mse), torch.tensor([prior], device=self.device), None, y_t
        pred = torch.empty_like(y_t)
        stack = [(tree, torch.arange(y_t.numel(), device=self.device))]
        while stack:
            node, idx = stack.pop()
            if not idx.numel() or node is None: continue
            if isinstance(node, dict):
                f,t,L,R = node['f'],node['t'],node['L'],node['R']
                mask = X_batch[idx, f] <= t
                stack.extend([(R, idx[~mask]), (L, idx[mask])])
            else:
                pred[idx] = y_t[idx].mean() if idx.numel() > 0 else y_t.mean()
        mse = ((pred - y_t)**2).mean()
        return 1 / (1 + mse), torch.tensor([prior], device=self.device), pred, y_t

def deltaE_split_gain(tokens, tok, env):
    y = env.y[env.idxs]; N = y.numel()
    dE = torch.zeros(tokens.shape[1] - 1, device=y.device)
    if N > 0:
        y2 = y * y; full_mse = (y2.mean() - y.mean() ** 2).item()
    else: full_mse = 0.0
    stack_rows = [torch.arange(N, device=y.device)]; stack_mse = [full_mse]
    action_sequence = tokens[0, 1:-1].tolist(); it = iter(tok.decode(action_sequence))
    token_idx = 0
    for kind, idx in it:
        if kind == "feat":
            token_idx += 1
            try: _, th = next(it)
            except StopIteration: break
            if not stack_rows: break
            parent_rows = stack_rows.pop(); parent_mse = stack_mse.pop()
            fv = env.X_full[env.idxs[parent_rows], idx]; mask = fv <= th
            L_rows = parent_rows[mask]; R_rows = parent_rows[~mask]
            def mse_fn(rows):
                if rows.numel() < 2: return 0.0
                yy = y[rows]; return ((yy*yy).mean() - yy.mean()**2).item()
            mseL, mseR = mse_fn(L_rows), mse_fn(R_rows)
            stack_rows.extend([R_rows, L_rows]); stack_mse.extend([mseR, mseL])
            wL = L_rows.numel(); wR = R_rows.numel(); parent_N = wL + wR
            if parent_N > 0:
                gain = parent_mse - (wL/parent_N * mseL + wR/parent_N * mseR)
                dE[token_idx] = -gain
            token_idx += 1
        else:
            if stack_rows: stack_rows.pop(); stack_mse.pop()
            token_idx += 1
    return dE.unsqueeze(0)

def get_tree_predictor(traj, X_binned, y_target, tok):
    path_iter = iter(tok.decode(traj[1:-1]))
    def build_recursive():
        try: k, i = next(path_iter)
        except StopIteration: return None
        if k == 'feat': return {'type': 'split', 'f': i, 't': next(path_iter)[1], 'L': build_recursive(), 'R': build_recursive()}
        return {'type': 'leaf', 'value': 0}
    tree_structure = build_recursive()
    if tree_structure is None: return lambda X: torch.zeros(X.size(0), device=X.device)
    q = [(tree_structure, torch.arange(X_binned.size(0), device=X_binned.device))]
    while q:
        node, indices = q.pop(0)
        if node['type'] == 'split':
            if not indices.numel() or node.get('L') is None: continue
            mask = X_binned[indices, node['f']] <= node['t']
            q.append((node['L'], indices[mask])); q.append((node['R'], indices[~mask]))
        else:
            node['value'] = y_target[indices].mean().item() if indices.numel() > 0 else y_target.mean().item()
    def predict(X_test):
        out = torch.empty(X_test.size(0), device=X_test.device)
        stack = [(tree_structure, torch.arange(X_test.size(0), device=X_test.device))]
        while stack:
            node, indices = stack.pop()
            if not indices.numel() or not node: continue
            if node['type'] == 'leaf': out[indices] = node['value']
            else:
                mask = X_test[indices, node['f']] <= node['t']
                if node.get('L'): stack.append((node['L'], indices[mask]))
                if node.get('R'): stack.append((node['R'], indices[~mask]))
        return out
    return predict

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity, self.data = capacity, []
    def add(self, r, t, p, idxs):
        self.data.append((r, t, p, idxs)); self.data.sort(key=lambda x:x[0], reverse=True)
        if len(self.data) > self.capacity: self.data.pop()
    def sample(self, k): return random.sample(self.data, min(k, len(self.data)))

def _safe_sample(logits, mask, temperature):
    logits = logits / temperature; masked = torch.where(mask, logits, torch.tensor(-1e9, device=logits.device))
    probs = torch.softmax(masked, dim=-1); return torch.multinomial(probs, 1).item()

def create_gain_bias(df_train, feats, target, tok, bins, prior_scale=0.5) -> torch.Tensor:
    if len(df_train) > 200_000:
        df_sample = df_train.sample(n=200_000, random_state=42)
    else:
        df_sample = df_train
    X_binned = []
    for f in feats:
        s = df_sample[f].replace([np.inf,-np.inf],np.nan).fillna(df_sample[f].median()).values
        qs = np.linspace(0,1,bins+1); edges = np.quantile(s,qs)
        edges = np.unique(edges); edges[0] -= 1e-9; edges[-1] += 1e-9
        X_binned.append(np.searchsorted(edges,s,side="right")-1)
    X_binned = np.stack(X_binned,1)
    y_train = df_sample[target].values
    lgb_train = lgb.Dataset(X_binned, y_train, feature_name=feats)
    params = { 'objective': 'regression_l1', 'metric': 'l1', 'n_estimators': 100, 'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'num_leaves': 1024, 'max_depth': 10, 'verbose': -1, 'n_jobs': -1 }
    gbm = lgb.train(params, lgb_train)
    tree_info = gbm.dump_model()["tree_info"]
    all_splits = []
    # This function is INSIDE create_gain_bias
    def parse_node(node):
        nonlocal bias  # <--- ADD THIS LINE
        if "split_gain" in node and node["split_gain"] > 0:
            f_idx = node["split_feature"]
            gain = node["split_gain"]
            
            tok_id_feat = tok._feat(f_idx)
            if tok_id_feat < tok.v.size():
                bias[tok_id_feat] += gain
    
            try:
                bin_idx = min(int(node["threshold"]), bins - 1)
                tok_id_th = tok._th(bin_idx)
                if tok_id_th < tok.v.size():
                    bias[tok_id_th] += gain
            except ValueError:
                try:
                    categories = [int(c) for c in node["threshold"].split('||')]
                    if len(categories) > 0:
                        distributed_gain = gain / len(categories)
                        for cat_idx in categories:
                            bin_idx = min(cat_idx, bins - 1)
                            tok_id_th = tok._th(bin_idx)
                            if tok_id_th < tok.v.size():
                                bias[tok_id_th] += distributed_gain
                except (ValueError, AttributeError):
                    pass
    
        if "left_child" in node: parse_node(node["left_child"])
        if "right_child" in node: parse_node(node["right_child"])
    gains = np.array([g for _, _, g in all_splits])
    if len(gains) == 0: return torch.zeros(tok.v.size(), dtype=torch.float32)
    mean, std = gains.mean(), gains.std()
    is_valid = np.abs(gains - mean) < 3 * std
    bias = np.zeros(tok.v.size())
    for (f, bin_, gain), is_valid_gain in zip(all_splits, is_valid):
        if is_valid_gain:
            tok_id_feat, tok_id_th = tok._feat(f), tok._th(bin_)
            if tok_id_feat < tok.v.size() and tok_id_th < tok.v.size():
                bias[tok_id_feat] += gain; bias[tok_id_th] += gain
    bias_std = bias.std()
    if bias_std > 1e-6: bias /= bias_std
    bias *= (prior_scale / 2.0)
    return torch.tensor(bias, dtype=torch.float32)

# ------------------------- 4. Main Training & Inference Logic ---------------------------
def train_and_ensemble(args: argparse.Namespace) -> None:
    print("--- Setting up models and environment with PRODUCTION GFN-Boost Pipeline ---")
    df_tr, df_te = pd.read_parquet(args.train), pd.read_parquet(args.test)
    
    feats = sorted(list(set([
        "X863", "X856", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X855", "X174", "X302", "X178", "X168", "X612",
        "buy_qty", "sell_qty", "volume", "X888", "X421", "X333", "X292",
        "bid_qty", "ask_qty",
        "X344", "X137", "X532"
    ])))

    v = Vocab(len(feats), args.bins, 1)
    tok = Tok(v)
    
    env = DRWEnv(df_tr, feats, 'label', args.bins, args.device)
    y_tr_true, X_tr_binned = env.y_full.clone(), env.X_full.clone()
    
    prior_bias = create_gain_bias(df_tr, feats, 'label', tok, args.bins, args.prior_scale).to(args.device)
    
    pf = PolicyPaperMLP(v.size(), args.lstm_hidden, args.mlp_layers, args.mlp_width).to(args.device)
    pb = PolicyPaperMLP(v.size(), args.lstm_hidden, args.mlp_layers, args.mlp_width).to(args.device)
    
    pf = torch.jit.script(pf); pb = torch.jit.script(pb)
    
    with torch.no_grad():
        pf.head_tok.bias.copy_(prior_bias)
        pb.head_tok.bias.copy_(prior_bias)
    
    log_z = torch.zeros((), device=args.device, requires_grad=True)
    optf, optb, optz = torch.optim.AdamW(pf.parameters(), lr=args.lr), torch.optim.AdamW(pb.parameters(), lr=args.lr), torch.optim.Adam([log_z], lr=args.lr/10)
    optimizers = [optf, optb, optz]
    warmup_updates, t_max_val = 10, max(1, args.updates - 10)
    warmup_schedulers = [LambdaLR(opt, lr_lambda=lambda upd: min(1.0, upd / warmup_updates)) for opt in optimizers]
    decay_schedulers = [CosineAnnealingLR(opt, T_max=t_max_val) for opt in optimizers]
    schedulers = [SequentialLR(opt, schedulers=[ws, ds], milestones=[warmup_updates]) for opt, ws, ds in zip(optimizers, warmup_schedulers, decay_schedulers)]
    buf = ReplayBuffer(capacity=10000)
    
    base_prediction = torch.full_like(y_tr_true, y_tr_true.mean())
    boosting_ensemble = []
    
    BETA_ANNEAL_UPDATES, TEMP_ANNEAL_UPDATES, FL_LOSS_ANNEAL_UPDATES = 20.0, 20.0, 20.0
    
    print("--- Starting GFN-Boost Training ---")
    for upd in range(1, args.updates + 1):
        residuals, env.y = y_tr_true - base_prediction, y_tr_true - base_prediction
        progress = min(1.0, (upd - 1) / BETA_ANNEAL_UPDATES)
        current_beta = 0.0#args.beta_start + progress * (args.beta_end - args.beta_start)
        temperature = max(1.0, 5.0 - (upd - 1) * (4.0 / TEMP_ANNEAL_UPDATES))
        lam_fl = 0.1#min(1.0, upd / FL_LOSS_ANNEAL_UPDATES)
        tb_loss_acc, fl_loss_acc, complete_rollouts_this_update = 0, 0, 0
        
        for opt in optimizers: opt.zero_grad()
        
        pbar = tqdm(range(args.rollouts), desc=f"Boosting Round {upd:02d}", leave=False)
        for _ in pbar:
            env.reset(args.batch)
            seq = [v.BOS]
            open_leaf_depths = deque([0]) 
            with torch.no_grad():
                while not env.done:
                    if not open_leaf_depths: break
                    x = torch.tensor([seq], device=args.device)
                    logits, _ = pf.forward(x)
                    logits = logits[0, -1]
                    mask = torch.zeros_like(logits, dtype=torch.bool)
                    current_depth = open_leaf_depths[-1]
                    if env.open_leaves > 0 and current_depth < args.max_depth:
                        mask[v.split_start : v.split_start + v.num_feat] = True
                    if env.open_leaves > 0:
                        mask[v.split_start + v.num_feat + v.num_th:] = True
                    if not mask.any(): break
                    tok1 = _safe_sample(logits, mask, temperature)
                    kind, idx = tok.decode_one(tok1)
                    if kind == 'feat':
                        d = open_leaf_depths.pop(); open_leaf_depths.append(d + 1); open_leaf_depths.append(d + 1)
                    else: open_leaf_depths.pop()
                    env.step((kind, idx)); seq.append(tok1)
                    if kind == 'feat':
                        x2, _ = pf.forward(torch.tensor([seq], device=args.device))
                        logits_th = x2[0, -1]
                        th_mask = torch.zeros_like(logits_th, dtype=torch.bool)
                        th_mask[v.split_start+v.num_feat : v.split_start+v.num_feat+v.num_th] = True
                        tok2 = _safe_sample(logits_th, th_mask, temperature)
                        env.step(tok.decode_one(tok2)); seq.append(tok2)
            seq.append(v.EOS)

            if env.open_leaves != 0: continue
            complete_rollouts_this_update += 1
            R, prior, _, _ = env.evaluate(current_beta)
            buf.add(R.item(), seq, prior.item(), env.idxs.clone())
            
            tokens_fwd = torch.tensor([seq], device=args.device)
            log_pf_fwd = pf.log_prob(tokens_fwd)
            logF = pf.log_F(tokens_fwd)
            tokens_bwd = torch.flip(tokens_fwd, dims=[1])
            log_pb = pb.log_prob(tokens_bwd)
            dE = deltaE_split_gain(tokens_fwd, tok, env)
            
            l_tb = tb_loss(log_pf_fwd, log_pb, log_z, R.unsqueeze(0), prior)
            l_fl = fl_loss(logF, log_pf_fwd, log_pb, dE)
            loss = l_tb + lam_fl * l_fl
            loss.backward()
            tb_loss_acc += l_tb.item(); fl_loss_acc += l_fl.item()

        did_any_backward = complete_rollouts_this_update > 0
        if did_any_backward:
            denom = complete_rollouts_this_update
            torch.nn.utils.clip_grad_norm_(pf.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(pb.parameters(), 1.0)
            torch.nn.utils.clip_grad_value_([log_z], 1.0)
            for opt in optimizers:
                for group in opt.param_groups:
                    for param in group['params']:
                        if param.grad is not None: param.grad /= denom
            for opt in optimizers: opt.step()
        
        if did_any_backward:
            for scheduler in schedulers: scheduler.step()
        
        if buf.data:
            top_k_trajectories = buf.data[:args.top_k_trees]
            top_k_sequences = [t[1] for t in top_k_trajectories]
            boosting_ensemble.append(top_k_sequences)
            avg_residual_preds = torch.zeros_like(base_prediction)
            for best_seq in top_k_sequences:
                predictor = get_tree_predictor(best_seq, X_tr_binned, residuals, tok)
                avg_residual_preds += predictor(X_tr_binned)
            if len(top_k_sequences) > 0:
                avg_residual_preds /= len(top_k_sequences)
            
            base_prediction += args.boosting_lr * avg_residual_preds
            
            final_corr = torch.corrcoef(torch.stack([base_prediction, y_tr_true]))[0,1].item()
            print(f"Boosting Round {upd:02d} | Comp: {complete_rollouts_this_update}/{args.rollouts} | "
                  f"TB: {tb_loss_acc/denom} | FL: {fl_loss_acc/denom} | Train ρ: {final_corr:+.3f}")
        else:
            print(f"Boosting Round {upd:02d} | No complete trees generated.")
        
        buf.data.clear()

    print("\n--- Training finished. Starting Boosted Inference. ---")
    X_te_binned = env._featurise(df_te, df_tr, feats, args.bins)
    final_test_preds = torch.full((len(df_te),), y_tr_true.mean().item(), device=args.device)
    inference_residuals = y_tr_true.clone()
    for i, top_k_in_round in enumerate(tqdm(boosting_ensemble, desc="Building boosted ensemble")):
        avg_train_preds_for_round, avg_test_preds_for_round = torch.zeros_like(y_tr_true), torch.zeros_like(final_test_preds)
        if not top_k_in_round: continue
        
        for tree_seq in top_k_in_round:
            predictor = get_tree_predictor(tree_seq, X_tr_binned, inference_residuals, tok)
            avg_train_preds_for_round += predictor(X_tr_binned)
            avg_test_preds_for_round += predictor(X_te_binned)
            
        avg_train_preds_for_round /= len(top_k_in_round)
        avg_test_preds_for_round /= len(top_k_in_round)

        final_test_preds += args.boosting_lr * avg_test_preds_for_round
        inference_residuals -= args.boosting_lr * avg_train_preds_for_round

    try:
        df_out = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
        df_out['prediction'] = final_test_preds.cpu().numpy()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df_out = pd.DataFrame({'id': df_te.get('id', range(len(df_te))), 'prediction': final_test_preds.cpu().numpy()})
    df_out.to_csv(args.out, index=False)
    print(f"Inference complete. Predictions saved to -> {args.out}")

if __name__ == '__main__':
    args = parse_args()
    train_and_ensemble(args)


# choose sane defaults for Kaggle's 16 GB GPU
!python dtgfn_boosted_production.py /kaggle/input/drw-crypto-market-prediction/train.parquet /kaggle/input/drw-crypto-market-prediction/test.parquet \
    --device cuda    --updates 100   --rollouts 10   --batch 500_000  





