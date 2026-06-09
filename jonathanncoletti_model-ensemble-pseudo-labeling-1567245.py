import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import networkx as nx
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
import gc
import warnings

# optimizations
import cudf
import cugraph
import cupy as cp

warnings.filterwarnings('ignore')

BASE_PATH = './'

# Confidence for cheater
PSEUDO_LABEL_THRESHOLD_HIGH = 0.96  

# Confidence for clean
PSEUDO_LABEL_THRESHOLD_LOW = 0.02   


# model params 
xgb_params = {
    'objective': 'binary:logistic', 'eval_metric': 'logloss', 'tree_method': 'hist',
    'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 1000, 'subsample': 0.8,
    'colsample_bytree': 0.7, 'n_jobs': -1, 'verbosity': 0, 'random_state': 42
}
lgb_params = {
    'objective': 'binary', 'metric': 'binary_logloss', 'num_leaves': 63,
    'learning_rate': 0.03, 'n_estimators': 1000, 'feature_fraction': 0.7,
    'bagging_fraction': 0.8, 'bagging_freq': 3, 'n_jobs': -1, 'verbose': -1, 'random_state': 42
}
cb_params = {
    'loss_function': 'Logloss', 'iterations': 1000, 'learning_rate': 0.04,
    'depth': 6, 'verbose': 0, 'allow_writing_files': False, 'random_seed': 42
}
mlp_params = {
    'hidden_layer_sizes': (64, 32), 'activation': 'relu', 'solver': 'adam',
    'max_iter': 300, 'early_stopping': True, 'random_state': 42
}

# helper functions
def calculate_cost(y_true, y_prob, t1, t2):
    """not accurate to competition: Fast cost calculation (Numpy/CPU)."""
    auto_pass = y_prob < t1
    manual = (y_prob >= t1) & (y_prob < t2)
    auto_block = y_prob >= t2
    
    cost = 0
    cost += np.sum((y_true == 1) & auto_pass) * 600
    cost += np.sum((y_true == 1) & manual) * 5
    cost += np.sum((y_true == 0) & manual) * 150
    cost += np.sum((y_true == 0) & auto_block) * 300
    return cost

def optimize_thresholds_fast(y_true, y_prob):
    """Fast threshold optimization."""
    best_cost = float('inf')
    best_t1, best_t2 = 0.0, 1.0
    
    # Coarse search
    thresholds = np.linspace(0, 1, 51)
    for t1 in thresholds:
        for t2 in thresholds[thresholds > t1]:
            cost = calculate_cost(y_true, y_prob, t1, t2)
            if cost < best_cost:
                best_cost = cost
                best_t1, best_t2 = t1, t2
    
    # Fine search
    fine_range = 0.05
    fine_thresholds = np.linspace(
        max(0, best_t1 - fine_range), 
        min(1, best_t1 + fine_range), 
        21
    )
    for t1 in fine_thresholds:
        for t2 in np.linspace(
            max(t1 + 0.01, best_t2 - fine_range), 
            min(1, best_t2 + fine_range), 
            21
        ):
            cost = calculate_cost(y_true, y_prob, t1, t2)
            if cost < best_cost:
                best_cost = cost
                best_t1, best_t2 = t1, t2
    
    return best_t1, best_t2, best_cost

def get_graph_features_gpu(social_graph_df, all_users_series):
    """Compute graph features using cugraph (GPU)."""
    print("  Preparing GPU Graph...")
    le = LabelEncoder()
    le.fit(all_users_series)
    
    sg_pd = social_graph_df.copy()
    sg_pd['src'] = le.transform(sg_pd['user_a'])
    sg_pd['dst'] = le.transform(sg_pd['user_b'])
    
    gdf_edges = cudf.DataFrame.from_pandas(sg_pd[['src', 'dst']])
    G = cugraph.Graph()
    G.from_cudf_edgelist(gdf_edges, source='src', destination='dst', renumber=False)
    
    print("  Running PageRank (GPU)...")
    pr_df = cugraph.pagerank(G, alpha=0.85, tol=1e-4)
    
    print("  Running Degree Centrality (GPU)...")
    degree_df = G.degrees()
    
    print("  Running Triangle Count (GPU)...")
    tc_df = cugraph.triangle_count(G)
    
    user_ids = np.arange(len(le.classes_))
    results_df = cudf.DataFrame({'vertex': user_ids})
    
    results_df = results_df.merge(pr_df, on='vertex', how='left').fillna(0)
    results_df = results_df.merge(degree_df, on='vertex', how='left').fillna(0)
    results_df = results_df.merge(tc_df, on='vertex', how='left').fillna(0)
    
    results_pd = results_df.to_pandas()
    results_pd['user_hash'] = le.inverse_transform(results_pd['vertex'])
    
    return results_pd.set_index('user_hash'), le, G

def run_risk_propagation_gpu(G, seeds_series, le, all_users_count):
    """Personalized PageRank on GPU."""
    cheater_seeds = seeds_series[seeds_series == 1]
    
    if len(cheater_seeds) == 0:
        return pd.Series(0.0, index=seeds_series.index)

    seed_ids = le.transform(cheater_seeds.index)
    
    pers_df = cudf.DataFrame({
        'vertex': seed_ids,
        'values': cp.ones(len(seed_ids), dtype='float32')
    })
    
    ppr_df = cugraph.pagerank(G, personalization=pers_df, alpha=0.85, tol=1e-4)
    ppr_pd = ppr_df.to_pandas()
    ppr_pd['user_hash'] = le.inverse_transform(ppr_pd['vertex'])
    
    return ppr_pd.set_index('user_hash')['pagerank']



def train_round_1_cv(X, y, X_test, feature_cols):
    """
    Round 1: Standard CV to get OOF predictions (for thresholds) 
    and reliable Test predictions (for pseudo-labeling).
    """
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    oof_stack = np.zeros((len(X), 4))
    test_stack = np.zeros((len(X_test), 4))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    print("  Starting 3-Fold CV...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # 1. XGB
        model_xgb = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_stack[val_idx, 0] = model_xgb.predict_proba(X_val)[:, 1]
        test_stack[:, 0] += model_xgb.predict_proba(X_test)[:, 1] / 3

        # 2. LGB
        model_lgb = lgb.LGBMClassifier(**lgb_params)
        model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_stack[val_idx, 1] = model_lgb.predict_proba(X_val)[:, 1]
        test_stack[:, 1] += model_lgb.predict_proba(X_test)[:, 1] / 3

        # 3. CatBoost
        model_cb = CatBoostClassifier(**cb_params)
        model_cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), 
                     early_stopping_rounds=50, verbose=False)
        oof_stack[val_idx, 2] = model_cb.predict_proba(X_val)[:, 1]
        test_stack[:, 2] += model_cb.predict_proba(X_test)[:, 1] / 3

        # 4. MLP (Scaled data)
        model_mlp = MLPClassifier(**mlp_params)
        model_mlp.fit(X_scaled[train_idx], y_tr)
        oof_stack[val_idx, 3] = model_mlp.predict_proba(X_scaled[val_idx])[:, 1]
        test_stack[:, 3] += model_mlp.predict_proba(X_test_scaled)[:, 1] / 3
        
        gc.collect()

    return oof_stack, test_stack

def train_round_2_full(X_combined, y_combined, weights, X_test, feature_cols):
    """
    Round 2: Full fit on Combined (Original + Pseudo) data using Sample Weights.
    No CV, no early stopping (relies on params).
    """
    print("  Training full models with sample weights...")
    
    test_preds = np.zeros((len(X_test), 4))
    
    # Scale for MLP
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_combined)
    X_test_sc = scaler.transform(X_test)

    # 1. XGBoost (weighted)
    print("    Fitting XGBoost...")
    model_xgb = xgb.XGBClassifier(**xgb_params) # No early stopping
    model_xgb.fit(X_combined, y_combined, sample_weight=weights, verbose=False)
    test_preds[:, 0] = model_xgb.predict_proba(X_test)[:, 1]

    # 2. LightGBM (weighted)
    print("    Fitting LightGBM...")
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(X_combined, y_combined, sample_weight=weights)
    test_preds[:, 1] = model_lgb.predict_proba(X_test)[:, 1]

    # 3. CatBoost (weighted)
    print("    Fitting CatBoost...")
    model_cb = CatBoostClassifier(**cb_params)
    model_cb.fit(X_combined, y_combined, sample_weight=weights, verbose=False)
    test_preds[:, 2] = model_cb.predict_proba(X_test)[:, 1]

    # 4. MLP (Unweighted, as requested)
    print("    Fitting MLP...")
    model_mlp = MLPClassifier(**mlp_params)
    model_mlp.fit(X_sc, y_combined)
    test_preds[:, 3] = model_mlp.predict_proba(X_test_sc)[:, 1]
    
    return test_preds

def main():
    print("=" * 50)
    print("Mercor Cheating Detection - Weighted Pseudo Labeling")
    print("=" * 50)
    
    # 1. Load Data
    print("\n1. Loading data...")
    train = pd.read_csv(BASE_PATH + 'train.csv')
    test = pd.read_csv(BASE_PATH + 'test.csv')
    social_graph = pd.read_csv(BASE_PATH + 'social_graph.csv')
    
    all_users = pd.concat([
        social_graph['user_a'], social_graph['user_b'], 
        train['user_hash'], test['user_hash']
    ]).unique()
    
    # 2. GPU Graph Features
    print("\n2. Computing GPU graph features...")
    graph_stats, le, G = get_graph_features_gpu(social_graph, all_users)
    
    def map_feature(df, feature_name, source_col='pagerank'):
        return df['user_hash'].map(graph_stats[source_col]).fillna(0)

    for df in [train, test]:
        df['pagerank'] = map_feature(df, 'pagerank', 'pagerank')
        deg_col = 'degree' if 'degree' in graph_stats.columns else 'out_degree'
        df['degree'] = map_feature(df, 'degree', deg_col)
        df['clustering'] = map_feature(df, 'clustering', 'counts')
    
    # 3. Risk Propagation
    print("\n3. GPU Risk Propagation...")
    labeled_train = train[train['is_cheating'].notna()]
    y = labeled_train['is_cheating'].values
    oof_risk = pd.Series(index=labeled_train['user_hash'], dtype=float)
    
    kf_feat = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf_feat.split(labeled_train, y)):
        train_fold = labeled_train.iloc[train_idx]
        train_seeds = train_fold.set_index('user_hash')['is_cheating']
        risk_scores = run_risk_propagation_gpu(G, train_seeds, le, len(all_users))
        val_users = labeled_train.iloc[val_idx]['user_hash']
        oof_risk.loc[val_users] = risk_scores.reindex(val_users).fillna(0)
        gc.collect()
    
    train['risk_score'] = train['user_hash'].map(oof_risk).fillna(0)
    
    all_seeds = labeled_train.set_index('user_hash')['is_cheating']
    test_risk_scores = run_risk_propagation_gpu(G, all_seeds, le, len(all_users))
    test['risk_score'] = test['user_hash'].map(test_risk_scores).fillna(0)
    
    # 4. Features
    print("\n4. Feature Engineering...")
    for df in [train, test]:
        df['risk_log'] = np.log1p(df['risk_score'] * 1e5)
        df['risk_degree'] = df['risk_log'] * np.log1p(df['degree'])
        df['risk_pagerank'] = df['risk_score'] * df['pagerank']
        df['risk_f12'] = df['risk_score'] * df['feature_012'].fillna(-1)

    feature_cols = [c for c in train.columns if c.startswith('feature_')] + \
                   ['degree', 'pagerank', 'clustering', 'risk_score', 
                    'risk_degree', 'risk_pagerank', 'risk_log', 'risk_f12']
    
    train[feature_cols] = train[feature_cols].fillna(-1)
    test[feature_cols] = test[feature_cols].fillna(-1)
    
    X = train.loc[labeled_train.index, feature_cols].values
    y = labeled_train['is_cheating'].values
    X_test_orig = test[feature_cols].values

    # 5. Round 1 (CV)
    print("\n5. Initial Ensemble Training (Round 1)...")
    oof_stack_1, test_stack_1 = train_round_1_cv(X, y, X_test_orig, feature_cols)
    
    ensemble_weights = np.array([0.35, 0.30, 0.25, 0.10]) # XGB, LGB, CB, MLP
    test_preds_1 = np.average(test_stack_1, axis=1, weights=ensemble_weights)
    oof_preds_1 = np.average(oof_stack_1, axis=1, weights=ensemble_weights)
    
    print("\n6. Optimizing Thresholds (on Round 1 CV)...")
    best_t1, best_t2, final_cost = optimize_thresholds_fast(y, oof_preds_1)
    print(f"  Best T1={best_t1:.3f}, T2={best_t2:.3f} | Cost={final_cost:.0f}")

    print("\n7. Generating Weighted Pseudo Labels...")
    
    high_conf_idx_cheat = np.where(test_preds_1 > PSEUDO_LABEL_THRESHOLD_HIGH)[0]
    high_conf_idx_clean = np.where(test_preds_1 < PSEUDO_LABEL_THRESHOLD_LOW)[0]
    
    # Cheaters: Weight scales with confidence 0.5 -> 0.9
    pseudo_cheat_weights = np.clip(
        test_preds_1[high_conf_idx_cheat], 0.5, 0.9
    )
    # Clean: Weight scales with confidence, but capped low (0.05 -> 0.3) to handle imbalance
    pseudo_clean_weights = np.clip(
        1 - test_preds_1[high_conf_idx_clean], 0.05, 0.3
    )
    
    orig_weights = np.ones(len(X))
    
    # Combined Weights
    sample_weights = np.hstack([
        orig_weights,
        pseudo_cheat_weights,
        pseudo_clean_weights
    ])
    
    # Create Pseudo Data
    X_pseudo_cheat = X_test_orig[high_conf_idx_cheat]
    y_pseudo_cheat = np.ones(len(high_conf_idx_cheat))
    
    X_pseudo_clean = X_test_orig[high_conf_idx_clean]
    y_pseudo_clean = np.zeros(len(high_conf_idx_clean))
    
    X_combined = np.vstack([X, X_pseudo_cheat, X_pseudo_clean])
    y_combined = np.hstack([y, y_pseudo_cheat, y_pseudo_clean])
    
    print(f"  Orig: {len(X)} | Pseudo Cheat: {len(X_pseudo_cheat)} | Pseudo Clean: {len(X_pseudo_clean)}")
    print(f"  Total Training: {len(X_combined)}")
    
    # 8. Round 2 (Full Fit)
    print("\n8. Retraining Ensemble (Round 2 - Full Fit w/ Weights)...")
    test_stack_2 = train_round_2_full(X_combined, y_combined, sample_weights, X_test_orig, feature_cols)
    
    test_preds_final = np.average(test_stack_2, axis=1, weights=ensemble_weights)

    print("\n9. Saving Submission...")
    submission = pd.DataFrame({
        'user_hash': test['user_hash'],
        'prediction': test_preds_final
    })
    
    out_path = BASE_PATH + 'submission.csv'
    submission.to_csv(out_path, index=False)
    
    # Check stats
    decisions = pd.cut(test_preds_final, 
                       bins=[-np.inf, best_t1, best_t2, np.inf],
                       labels=['auto_pass', 'manual', 'auto_block'])
    
    print(f"\nFinal Decisions:")
    print(decisions.value_counts())
    print(f"\nSubmission saved to: {out_path}")

if __name__ == "__main__":
    main()

