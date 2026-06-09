import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import networkx as nx
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import gc
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
BASE_PATH = '/kaggle/input/mercor-cheating-detection/'

# --- HELPER FUNCTIONS ---

def calculate_cost(y_true, y_prob, t1, t2):
    """Fast cost calculation."""
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
    
    # Step 1: Coarse search
    thresholds = np.linspace(0, 1, 51)  # Reduced from 101 to 51
    for t1 in thresholds:
        for t2 in thresholds[thresholds > t1]:
            cost = calculate_cost(y_true, y_prob, t1, t2)
            if cost < best_cost:
                best_cost = cost
                best_t1, best_t2 = t1, t2
    
    # Step 2: Fine search around best point
    fine_range = 0.1
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

def run_fast_label_propagation(social_graph, seeds, all_users, n_iter=3):
    """
    Fast label propagation with minimal iterations.
    """
    # Create adjacency dictionary (much faster than DataFrame operations)
    adj = {}
    for _, row in social_graph.iterrows():
        adj.setdefault(row['user_a'], []).append(row['user_b'])
        adj.setdefault(row['user_b'], []).append(row['user_a'])
    
    # Initialize scores
    scores = {user: 0.5 for user in all_users}
    scores.update(seeds.to_dict())
    
    # Run propagation
    for _ in range(n_iter):
        new_scores = {}
        for user in all_users:
            if user in seeds:
                new_scores[user] = seeds[user]
            else:
                neighbors = adj.get(user, [])
                if neighbors:
                    neighbor_scores = [scores[n] for n in neighbors]
                    new_scores[user] = 0.5 * scores[user] + 0.5 * np.mean(neighbor_scores)
                else:
                    new_scores[user] = scores[user]
        scores = new_scores
    
    return pd.Series(scores)

def compute_fast_graph_features(social_graph, all_users):
    """
    Compute only the most important graph features quickly.
    """
    print("Computing fast graph features...")
    
    # Build graph
    G = nx.from_pandas_edgelist(social_graph, 'user_a', 'user_b')
    
    # 1. Degree (simplified calculation)
    degree_counts = {}
    for _, row in social_graph.iterrows():
        degree_counts[row['user_a']] = degree_counts.get(row['user_a'], 0) + 1
        degree_counts[row['user_b']] = degree_counts.get(row['user_b'], 0) + 1
    
    # 2. Fast PageRank (with fewer iterations)
    print("  Computing PageRank...")
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=50, tol=1e-4)
    
    # 3. Simple clustering coefficient (for subset of nodes)
    print("  Computing clustering coefficient...")
    clustering = {}
    sample_nodes = list(G.nodes())[:5000]  # Limit computation
    for node in sample_nodes:
        clustering[node] = nx.clustering(G, node)
    
    return {
        'degree': degree_counts,
        'pagerank': pagerank,
        'clustering': clustering
    }

def main():
    print("=" * 50)
    print("Mercor Cheating Detection - Fast Ensemble")
    print("=" * 50)
    
    # 1. Load Data
    print("\n1. Loading data...")
    train = pd.read_csv(BASE_PATH + 'train.csv')
    test = pd.read_csv(BASE_PATH + 'test.csv')
    social_graph = pd.read_csv(BASE_PATH + 'social_graph.csv')
    
    # 2. Basic Degree Features (Fast)
    print("\n2. Computing basic degree features...")
    degree_a = social_graph['user_a'].value_counts()
    degree_b = social_graph['user_b'].value_counts()
    degrees = degree_a.add(degree_b, fill_value=0)
    
    train['degree'] = train['user_hash'].map(degrees).fillna(0)
    test['degree'] = test['user_hash'].map(degrees).fillna(0)
    
    # 3. Fast Graph Features
    print("\n3. Computing fast graph features...")
    all_users = pd.concat([social_graph['user_a'], social_graph['user_b'], 
                          train['user_hash'], test['user_hash']]).unique()
    
    graph_features = compute_fast_graph_features(social_graph, all_users)
    
    train['pagerank'] = train['user_hash'].map(graph_features['pagerank']).fillna(0)
    test['pagerank'] = test['user_hash'].map(graph_features['pagerank']).fillna(0)
    
    train['clustering'] = train['user_hash'].map(graph_features['clustering']).fillna(0)
    test['clustering'] = test['user_hash'].map(graph_features['clustering']).fillna(0)
    
    # 4. Fast Label Propagation (3 folds instead of 5)
    print("\n4. Fast label propagation (3-fold)...")
    
    high_conf_clean = train[train['high_conf_clean'] == 1].set_index('user_hash')
    clean_seeds = pd.Series(0.0, index=high_conf_clean.index)
    
    labeled_train = train[train['is_cheating'].notna()]
    y = labeled_train['is_cheating'].values
    
    oof_risk = pd.Series(index=labeled_train['user_hash'], dtype=float)
    
    # Use only 3 folds for speed
    kf_feat = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf_feat.split(labeled_train, y)):
        print(f"  LP Fold {fold+1}/3...")
        
        train_fold = labeled_train.iloc[train_idx]
        train_seeds = train_fold.set_index('user_hash')['is_cheating']
        seeds = pd.concat([train_seeds, clean_seeds])
        
        # Run fast LP with only 3 iterations
        scores = run_fast_label_propagation(social_graph, seeds, all_users, n_iter=3)
        
        val_users = labeled_train.iloc[val_idx]['user_hash']
        oof_risk.loc[val_users] = scores.loc[val_users]
        
        gc.collect()
    
    train['risk_score'] = train['user_hash'].map(oof_risk).fillna(0.5)
    
    # Test scores using all labeled data
    print("  Generating test risk scores...")
    all_labeled_seeds = labeled_train.set_index('user_hash')['is_cheating']
    seeds = pd.concat([all_labeled_seeds, clean_seeds])
    test_scores = run_fast_label_propagation(social_graph, seeds, all_users, n_iter=3)
    test['risk_score'] = test['user_hash'].map(test_scores).fillna(0.5)
    
    # 5. Create simple interaction features
    print("\n5. Creating interaction features...")
    train['risk_degree'] = train['risk_score'] * train['degree']
    test['risk_degree'] = test['risk_score'] * test['degree']
    
    train['risk_pagerank'] = train['risk_score'] * train['pagerank']
    test['risk_pagerank'] = test['risk_score'] * test['pagerank']
    
    # 6. Prepare features
    print("\n6. Preparing features...")
    feature_cols = [c for c in train.columns if c.startswith('feature_')] + \
                  ['degree', 'pagerank', 'clustering', 'risk_score', 
                   'risk_degree', 'risk_pagerank']
    
    # Fill NaN
    train[feature_cols] = train[feature_cols].fillna(-1)
    test[feature_cols] = test[feature_cols].fillna(-1)
    
    X = train.loc[labeled_train.index, feature_cols].values
    y = labeled_train['is_cheating'].values
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    test_scaled = scaler.transform(test[feature_cols].values)
    
    print(f"\nTraining on {len(X)} samples with {len(feature_cols)} features")
    
    xgb_params = {
        'objective': 'binary:logistic', 
        'eval_metric': 'logloss', 
        'tree_method': 'hist',
        'max_depth': 6, 
        'learning_rate': 0.05, 
        'n_estimators': 800,
        'subsample': 0.8,
        'colsample_bytree': 0.7, 
        'gamma': 0.3, 
        'reg_alpha': 5.0, 
        'reg_lambda': 4.0,
        'min_child_weight': 3, 
        'scale_pos_weight': 2.2, 
        'random_state': 42, 
        'n_jobs': -1, 
        'verbosity': 0
    }
    
    lgb_params = {
        'objective': 'binary', 
        'metric': 'binary_logloss', 
        'boosting_type': 'gbdt',
        'num_leaves': 63,  
        'learning_rate': 0.03, 
        'n_estimators': 800,  
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8, 
        'bagging_freq': 3, 
        'lambda_l1': 2.0, 
        'lambda_l2': 2.0,
        'min_data_in_leaf': 20,
        'scale_pos_weight': 2.2, 
        'random_state': 42, 
        'n_jobs': -1, 
        'verbose': -1
    }

    cb_params = {
        'loss_function': 'Logloss', 
        'iterations': 800,  
        'learning_rate': 0.04, 
        'depth': 6,
        'l2_leaf_reg': 3.0, 
        'scale_pos_weight': 2.2, 
        'random_seed': 42, 
        'verbose': 0, 
        'allow_writing_files': False
    }

    mlp_params = {
        'hidden_layer_sizes': (64, 32),  
        'activation': 'relu', 
        'solver': 'adam',
        'alpha': 0.001, 
        'batch_size': 256, 
        'learning_rate_init': 0.001,
        'max_iter': 300,  
        'early_stopping': True,
        'random_state': 42
    }

    # --- ENSEMBLE TRAINING (3 folds for speed) ---
    print("\n7. Training ensemble models (3-fold)...")
    
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    oof_stack = np.zeros((len(X), 4))
    test_stack = np.zeros((len(test), 4))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"  Model Fold {fold+1}/3")
        
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        X_tr_sc, X_val_sc = X_scaled[train_idx], X_scaled[val_idx]
        
        # 1. XGBoost (with early stopping)
        model_xgb = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_stack[val_idx, 0] = model_xgb.predict_proba(X_val)[:, 1]
        test_stack[:, 0] += model_xgb.predict_proba(test[feature_cols].values)[:, 1] / 3

        # 2. LightGBM
        model_lgb = lgb.LGBMClassifier(**lgb_params)
        model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                     callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_stack[val_idx, 1] = model_lgb.predict_proba(X_val)[:, 1]
        test_stack[:, 1] += model_lgb.predict_proba(test[feature_cols].values)[:, 1] / 3

        # 3. CatBoost
        model_cb = CatBoostClassifier(**cb_params)
        model_cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), 
                    early_stopping_rounds=50, verbose=False)
        oof_stack[val_idx, 2] = model_cb.predict_proba(X_val)[:, 1]
        test_stack[:, 2] += model_cb.predict_proba(test[feature_cols].values)[:, 1] / 3

        # 4. MLP
        model_mlp = MLPClassifier(**mlp_params)
        model_mlp.fit(X_tr_sc, y_tr)
        oof_stack[val_idx, 3] = model_mlp.predict_proba(X_val_sc)[:, 1]
        test_stack[:, 3] += model_mlp.predict_proba(test_scaled)[:, 1] / 3
        
        gc.collect()
    
    # 8. Simple weighted blend (faster than optimization)
    print("\n8. Creating weighted blend...")
    
    # Pre-defined weights based on typical performance
    weights = np.array([0.35, 0.30, 0.25, 0.10])  # XGB, LGB, CB, MLP
    oof_final = np.average(oof_stack, axis=1, weights=weights)
    test_final = np.average(test_stack, axis=1, weights=weights)
    
    # 9. Optimize thresholds
    print("\n9. Optimizing thresholds...")
    best_t1, best_t2, best_cost = optimize_thresholds_fast(y, oof_final)
    print(f"  Optimal: T1={best_t1:.3f}, T2={best_t2:.3f}")
    print(f"  Estimated CV cost: {best_cost:.0f}")
    
    # 10. Create submission
    print("\n10. Creating submission...")
    submission = pd.DataFrame({
        'user_hash': test['user_hash'],
        'prediction': test_final
    })
    
    out_path = './submission.csv'
    submission.to_csv(out_path, index=False)
    
    # Quick analysis
    decisions = pd.cut(test_final, 
                      bins=[-np.inf, best_t1, best_t2, np.inf],
                      labels=['auto_pass', 'manual', 'auto_block'])
    
    print(f"\nDecision distribution:")
    for decision in ['auto_pass', 'manual', 'auto_block']:
        count = (decisions == decision).sum()
        pct = 100 * count / len(test_final)
        print(f"  {decision}: {count} ({pct:.1f}%)")
    
    print(f"\n✓ Submission saved to: {out_path}")
    print("\n" + "=" * 50)
    print("Fast training complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()




