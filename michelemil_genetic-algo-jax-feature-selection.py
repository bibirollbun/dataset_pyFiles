!pip install tensorflow flax scikit-learn matplotlib seaborn polars


import os
import logging
from typing import List, Dict, Tuple
from functools import partial
import math
import time
import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold

# -------------------------
# TPU / JAX Initialization
# -------------------------
try:
    jax.distributed.initialize()
except Exception as e:
    print(f"Distributed init info: {e}")

print(f"TPU Devices: {jax.devices()}")

# -------------------------
# CONFIG
# -------------------------
TRAIN_PATH = "/kaggle/input/detecting-reversal-points-in-us-equities/new_comptetition_data/train.csv"
POP_SIZE = 512
ISLANDS = 4
EVAL_BATCH_SIZE = 512
MAX_TEMPORAL_CANDIDATES = 7000  # Set to 7000 for high-feature run
RIDGE_ALPHA = 1.0
TARGET_SPARSITY = 0.05
INITIAL_P = 0.20
CXPB = 0.6
MUTPB_START = 0.08
MUTPB_END = 0.02
ELITISM = max(32, POP_SIZE // 32)
MAX_SOLVE_CAP = 512
SEED_TOPK = 512
GENERATIONS = 800
MIGRATE_EVERY = 10
MIGRATE_K = 8
FOLDS_FOR_EVAL = 3
RNG_SEED = 42

jax.config.update('jax_default_matmul_precision', 'high')
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ga_pipeline_v2")

# -------------------------
# Utility & JAX Helpers
# -------------------------
def categorize_features_precise(feature_names: List[str]) -> Dict[str, List[str]]:
    groups = {'base': [], 'events': [], 'temporal_relations': [], 'other': []}
    for f in feature_names:
        if f in {'ratio', 'momentum', 'sm_ratio', 'sm_momentum'}:
            groups['base'].append(f)
        elif any(f.startswith(p) for p in ['troughs_', 'peaks_', 'cross_threshold_', 'trending_', 'zone_']):
            groups['events'].append(f)
        elif f.startswith(('occurs_within_', 'happens_within_')):
            groups['temporal_relations'].append(f)
        else:
            groups['other'].append(f)
    return groups

@jit
def compute_feature_scores_jax(X_f32: jnp.ndarray, y_i32: jnp.ndarray) -> jnp.ndarray:
    X_centered = X_f32 - jnp.mean(X_f32, axis=0, keepdims=True)
    y_f32 = y_i32.astype(jnp.float32)
    y_centered = y_f32 - jnp.mean(y_f32)
    cov = jnp.mean(X_centered * y_centered[:, None], axis=0)
    x_std = jnp.std(X_f32, axis=0)
    y_std = jnp.std(y_f32)
    return jnp.abs(cov / (jnp.where(x_std == 0, 1.0, x_std) * jnp.where(y_std == 0, 1.0, y_std)))

@partial(jit, static_argnums=(2,))
def compute_f1_macro_jit(y_true: jnp.ndarray, y_pred: jnp.ndarray, num_classes: int):
    y_true_oh = jax.nn.one_hot(y_true, num_classes)
    y_pred_oh = jax.nn.one_hot(y_pred, num_classes)
    tp = jnp.sum(y_true_oh * y_pred_oh, axis=0)
    fp = jnp.sum((1 - y_true_oh) * y_pred_oh, axis=0)
    fn = jnp.sum(y_true_oh * (1 - y_pred_oh), axis=0)
    precision = jnp.where(tp + fp == 0, 0.0, tp / (tp + fp))
    recall = jnp.where(tp + fn == 0, 0.0, tp / (tp + fn))
    f1 = jnp.where(precision + recall == 0, 0.0, 2 * precision * recall / (precision + recall))
    return jnp.mean(f1)

@partial(jit, static_argnums=(4,5))
def solve_subspace_jit(XtX_full, Xty_full, mask, alpha, num_classes, max_cap, importance_scores):
    weighted = mask.astype(jnp.float32) * importance_scores
    sort_idx = jnp.argsort(weighted, descending=True)
    active_indices = sort_idx[:max_cap]
    is_active = mask[active_indices]

    XtX_sub = XtX_full[active_indices][:, active_indices]
    Xty_sub = Xty_full[active_indices]

    sub_mask_mat = is_active[:, None] * is_active[None, :]
    XtX_active = XtX_sub * sub_mask_mat

    diag_reg = jnp.where(is_active > 0, alpha, 1.0)
    XtX_reg = XtX_active + jnp.diag(diag_reg) + (jnp.eye(max_cap) * (1.0 - is_active[:, None]))

    Xty_active = Xty_sub * is_active[:, None]
    W_sub = jnp.linalg.solve(XtX_reg, Xty_active)
    return W_sub, active_indices

def make_evaluate_batch_optimized(precomputed_folds, num_classes, alpha, max_cap, importance_scores):
    def evaluate_one(mask):
        f1s = []
        for (XtX_tr, Xty_tr, X_val, y_val) in precomputed_folds:
            W_sub, active_indices = solve_subspace_jit(XtX_tr, Xty_tr, mask, alpha, num_classes, max_cap, importance_scores)
            X_val_sub = X_val[:, active_indices]
            logits = X_val_sub @ W_sub
            y_pred = jnp.argmax(logits, axis=1)
            f1s.append(compute_f1_macro_jit(y_val, y_pred, num_classes))
        return jnp.mean(jnp.stack(f1s))
    
    batched = vmap(evaluate_one, in_axes=(0,))
    return jit(batched)

# -------------------------
# GA Classes & Logic
# -------------------------
class GAIsland:
    def __init__(self, rng_key, pop_size, n_genes, init_p, target_p, cxpb, mutpb_start, mutpb_end, elitism, importance_scores):
        self.rng = rng_key
        self.pop_size = pop_size
        self.n_genes = n_genes
        self.cxpb = cxpb
        self.mutpb_start = mutpb_start
        self.mutpb_end = mutpb_end
        self.elitism = elitism
        self.importance_scores = importance_scores
        
        subkey, self.rng = random.split(self.rng)
        rand_init = random.bernoulli(subkey, p=init_p, shape=(pop_size, n_genes))
        self.pop = rand_init.astype(jnp.bool_)
        self.fitness = jnp.zeros(pop_size, dtype=jnp.float32)

    def seeded_with_topk(self, topk_idx, n_seeded):
        n_seeded = min(n_seeded, self.pop_size)
        base = jnp.zeros((n_seeded, self.n_genes), dtype=jnp.bool_)
        base = base.at[:, topk_idx].set(True)
        subkey, self.rng = random.split(self.rng)
        extras = random.bernoulli(subkey, p=0.03, shape=(n_seeded, self.n_genes)).astype(jnp.bool_)
        seeded = jnp.logical_or(base, extras)
        self.pop = self.pop.at[:n_seeded].set(seeded)

    def evaluate_full_pop(self, eval_batch_fn):
        fits = []
        for i in range(0, self.pop_size, EVAL_BATCH_SIZE):
            batch = self.pop[i:i+EVAL_BATCH_SIZE]
            if batch.shape[0] < EVAL_BATCH_SIZE:
                pad = jnp.zeros((EVAL_BATCH_SIZE - batch.shape[0], self.n_genes), dtype=jnp.bool_)
                batch = jnp.concatenate([batch, pad], axis=0)
                res = eval_batch_fn(batch)[:batch.shape[0]-pad.shape[0]]
            else:
                res = eval_batch_fn(batch)
            fits.append(res)
        self.fitness = jnp.concatenate(fits)

    def epoch(self, eval_batch_fn, gen_idx, total_gens):
        subkey, self.rng = random.split(self.rng)
        mutpb = self.mutpb_start * (1 - gen_idx / total_gens) + self.mutpb_end * (gen_idx / total_gens)

        # Tournament
        subkey, sk = random.split(subkey)
        idx = random.randint(sk, (self.pop_size, 3), 0, self.pop_size)
        subkey, sk = random.split(subkey)
        noisy_fitness = self.fitness + (random.normal(sk, self.fitness.shape) * 1e-6)
        winners = jnp.argmax(noisy_fitness[idx], axis=1)
        parents = self.pop[idx[jnp.arange(self.pop_size), winners]]

        # Crossover
        subkey, sk = random.split(subkey)
        cross_mask = random.bernoulli(sk, p=0.5, shape=parents.shape)
        offspring = jnp.where(cross_mask, parents, jnp.roll(parents, 1, axis=0))

        # Mutation
        subkey, sk = random.split(subkey)
        mutation = random.bernoulli(sk, p=mutpb, shape=offspring.shape)
        offspring = jnp.logical_xor(offspring, mutation)

        # Elitism
        elite_idx = jnp.argsort(self.fitness)[-self.elitism:]
        offspring = offspring.at[:self.elitism].set(self.pop[elite_idx])

        self.pop = offspring
        self.evaluate_full_pop(eval_batch_fn)

def run_multi_island_ga(eval_batch_fn, n_genes, importance_scores, total_gens):
    base_key = random.PRNGKey(RNG_SEED)
    islands = []
    for i in range(ISLANDS):
        k = random.fold_in(base_key, i)
        isl = GAIsland(k, POP_SIZE, n_genes, INITIAL_P, TARGET_SPARSITY, CXPB, MUTPB_START, MUTPB_END, ELITISM, importance_scores)
        islands.append(isl)

    # Seed top-k into each island's initial population
    # Note: caller handles specific seeding calls, but islands are initialized here
    
    global_best_score, global_best_mask = -1.0, None
    pbar = tqdm(range(total_gens), desc="GA Evolution")
    
    for gen in pbar:
        gen_best = -1.0
        for isl in islands:
            isl.epoch(eval_batch_fn, gen, total_gens)
            best_idx = int(jnp.argmax(isl.fitness))
            score = float(isl.fitness[best_idx])
            if score > gen_best: gen_best = score
            if score > global_best_score:
                global_best_score = score
                global_best_mask = np.array(isl.pop[best_idx], dtype=np.bool_)

        if (gen + 1) % MIGRATE_EVERY == 0:
            elites_pool = jnp.concatenate([isl.pop[jnp.argsort(isl.fitness)[-MIGRATE_K:]] for isl in islands])
            for i, isl in enumerate(islands):
                start = (i * MIGRATE_K) % elites_pool.shape[0]
                worst_idx = jnp.argsort(isl.fitness)[:MIGRATE_K]
                isl.pop = isl.pop.at[worst_idx].set(elites_pool[start:start+MIGRATE_K])

        avg_active = np.mean([int(jnp.sum(isl.pop)) for isl in islands])
        pbar.set_postfix({"GenBest": f"{gen_best:.4f}", "GlobalBest": f"{global_best_score:.4f}", "AvgActive": f"{avg_active:.1f}"})

    return global_best_mask, global_best_score

# -------------------------
# Main Execution
# -------------------------
def main():
    logger.info("Initializing Data Pipeline...")
    
    # 1. Load schema and identify numeric features
    schema = pl.scan_csv(TRAIN_PATH).collect_schema()
    numeric_feats = [c for c, dt in schema.items() if dt in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Boolean) and c != 'class_label']
    
    # 2. Read data
    df = pl.read_csv(TRAIN_PATH, columns=['class_label'] + numeric_feats)

    # 3. Robust Label Mapping
    def robust_map(x):
        if x is None: return 'None'
        v = str(x).strip().upper()
        if v in ('H','HH','LH','HIGH','HIGHLIGHT'): return 'H'
        if v in ('L','LL','HL','LOW'): return 'L'
        if 'H' in v and 'L' not in v: return 'H'
        if 'L' in v and 'H' not in v: return 'L'
        return 'None'
    
    df = df.with_columns(pl.col('class_label').map_elements(robust_map, return_dtype=pl.Utf8))
    le = LabelEncoder()
    y_encoded = le.fit_transform(df['class_label'].to_numpy())
    num_classes = int(len(le.classes_))
    y_np = np.asarray(y_encoded, dtype=np.int32)
    
    # 4. Impute and Standardize for Initial Importance Scoring
    df_proc = df.with_columns([pl.col(c).cast(pl.Float32).fill_null(0.0) for c in numeric_feats])
    X_initial = df_proc.select(numeric_feats).to_numpy().astype(np.float32)
    
    scaler = StandardScaler()
    X_std_initial = scaler.fit_transform(X_initial)
    
    # 5. Compute Univariate Importance (Global)
    logger.info("Computing global importance scores...")
    importance_scores_full = np.array(compute_feature_scores_jax(jnp.array(X_std_initial), jnp.array(y_np)))
    
    # 6. Group Features and Select Candidates (the "all_star" list)
    groups = categorize_features_precise(numeric_feats)
    score_ser = pd.Series(importance_scores_full, index=numeric_feats)
    
    essential = groups['base'] + groups['events'] + groups['other']
    top_temporal = score_ser[score_ser.index.isin(groups['temporal_relations'])].sort_values(ascending=False).head(MAX_TEMPORAL_CANDIDATES).index.tolist()
    
    all_star = essential + top_temporal
    n_genes = len(all_star)
    logger.info(f"Candidate features for GA: {n_genes}")

    # 7. CRITICAL: Align Importance Scores with Candidate Features
    # This prevents the "Incompatible Shapes" error
    full_score_dict = dict(zip(numeric_feats, importance_scores_full))
    candidate_scores = np.array([full_score_dict[f] for f in all_star], dtype=np.float32)
    
    # Robust Normalization (NumPy 2.0 compatible, avoiding .ptp())
    score_min = candidate_scores.min()
    score_max = candidate_scores.max()
    candidate_scores = (candidate_scores - score_min) / (score_max - score_min + 1e-12)
    
    importance_scores_jnp = jnp.array(candidate_scores)

    # 8. Prepare Scaled Candidate Matrix
    X_candidates = scaler.fit_transform(df_proc.select(all_star).to_numpy().astype(np.float32))
    X_jax_cand = jnp.array(X_candidates)

    # 9. Precompute Folds for GA Evaluation
    skf = StratifiedKFold(n_splits=FOLDS_FOR_EVAL, shuffle=True, random_state=RNG_SEED)
    precomputed_folds = []
    for tr_idx, val_idx in skf.split(X_candidates, y_np):
        y_tr_oh = jax.nn.one_hot(jnp.array(y_np[tr_idx]), num_classes)
        precomputed_folds.append((
            X_jax_cand[tr_idx].T @ X_jax_cand[tr_idx], 
            X_jax_cand[tr_idx].T @ y_tr_oh,
            X_jax_cand[val_idx], 
            jnp.array(y_np[val_idx])
        ))

    # 10. Initialize Evaluator and Warmup
    eval_batch_fn = make_evaluate_batch_optimized(
        precomputed_folds, 
        num_classes, 
        RIDGE_ALPHA, 
        MAX_SOLVE_CAP, 
        importance_scores_jnp
    )
    
    logger.info("Warming JIT...")
    dummy = jnp.zeros((EVAL_BATCH_SIZE, n_genes), dtype=jnp.bool_)
    _ = eval_batch_fn(dummy)

    # 11. Top-K Seeding indices (relative to all_star list)
    topk_idx = jnp.array(np.argsort(candidate_scores)[-SEED_TOPK:][::-1], dtype=jnp.int32)

    # 12. Seed Islands and Run GA
    # We must seed islands manually here since the runner initializes fresh islands
    # NOTE: The previous runner function recreated islands. Let's fix that pattern or manually seed inside the runner.
    # To keep logic clean, I will manually init and seed here, then run.
    
    # Initialize islands locally to seed them, then pass to runner? 
    # The runner `run_multi_island_ga` currently creates its own islands. 
    # Let's adjust the logic slightly: We will let the runner create them, but we need to pass `topk_idx` to it?
    # Actually, simpler: MODIFY run_multi_island_ga to accept topk_idx and seed.
    
    # Re-defining run_multi_island_ga to accept topk_idx for seeding
    def run_ga_seeded(eval_batch_fn, n_genes, importance_scores, total_gens, topk_idx):
        base_key = random.PRNGKey(RNG_SEED)
        islands = []
        for i in range(ISLANDS):
            k = random.fold_in(base_key, i)
            isl = GAIsland(k, POP_SIZE, n_genes, INITIAL_P, TARGET_SPARSITY, CXPB, MUTPB_START, MUTPB_END, ELITISM, importance_scores)
            # SEED HERE
            n_seeded = max(1, POP_SIZE // 10)
            isl.seeded_with_topk(topk_idx, n_seeded)
            islands.append(isl)

        # Initial Eval
        for isl in islands:
            isl.evaluate_full_pop(eval_batch_fn)

        global_best_score, global_best_mask = -1.0, None
        pbar = tqdm(range(total_gens), desc="GA Evolution")
        
        for gen in pbar:
            gen_best = -1.0
            for isl in islands:
                isl.epoch(eval_batch_fn, gen, total_gens)
                best_idx = int(jnp.argmax(isl.fitness))
                score = float(isl.fitness[best_idx])
                if score > gen_best: gen_best = score
                if score > global_best_score:
                    global_best_score = score
                    global_best_mask = np.array(isl.pop[best_idx], dtype=np.bool_)

            if (gen + 1) % MIGRATE_EVERY == 0:
                elites_pool = jnp.concatenate([isl.pop[jnp.argsort(isl.fitness)[-MIGRATE_K:]] for isl in islands])
                for i, isl in enumerate(islands):
                    start = (i * MIGRATE_K) % elites_pool.shape[0]
                    worst_idx = jnp.argsort(isl.fitness)[:MIGRATE_K]
                    isl.pop = isl.pop.at[worst_idx].set(elites_pool[start:start+MIGRATE_K])

            avg_active = np.mean([int(jnp.sum(isl.pop)) for isl in islands])
            pbar.set_postfix({"GenBest": f"{gen_best:.4f}", "GlobalBest": f"{global_best_score:.4f}", "AvgActive": f"{avg_active:.1f}"})

        return global_best_mask, global_best_score

    # Execute
    best_mask, best_score = run_ga_seeded(
        eval_batch_fn, 
        n_genes, 
        importance_scores_jnp, 
        GENERATIONS, 
        topk_idx
    )
    
    # 13. Output Results
    final_features = [all_star[i] for i in range(n_genes) if best_mask[i]]
    pd.Series(final_features).to_csv("final_selected_features.csv", index=False)
    logger.info(f"GA complete. Selected {len(final_features)} features with best F1 {best_score:.6f}")

if __name__ == "__main__":
    main()

