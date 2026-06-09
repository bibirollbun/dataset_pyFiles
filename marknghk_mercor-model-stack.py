# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import gc
import time
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping

# Optional CatBoost import
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")


# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    "SEED": 42,
    "N_FOLDS": 5,
    "INPUT_DIR": "/kaggle/input/mercor-cheating-detection",
    "LP_ITERATIONS": 3,
    "MODELS": {
        "xgb": {
            "n_estimators": 722, "learning_rate": 0.03, "max_depth": 9,
            "subsample": 0.9, "colsample_bytree": 0.8, "gamma": 0.7,
            "min_child_weight": 5, "reg_alpha": 2, "reg_lambda": 1.7,
            "verbosity": 0
        },
        "lgbm": {
            "n_estimators": 761, "learning_rate": 0.03, "num_leaves": 105,
            "max_depth": 9, "min_data_in_leaf": 123, "lambda_l1": 0.02,
            "lambda_l2": 0.005, "feature_fraction": 0.47, "bagging_fraction": 0.81,
            "bagging_freq": 1, "verbose": -1
        },
        "cat": {
            "iterations": 700, "learning_rate": 0.03, "depth": 9,
            "verbose": False
        }
    }
}




# ==============================================================================
# CLASS: Graph Feature Engineering
# ==============================================================================
class GraphFeatureEngineer:
    def __init__(self, train_df, test_df, graph_df):
        self.train = train_df
        self.test = test_df
        self.graph = graph_df
        self.all_users = pd.concat([
            graph_df['source'], graph_df['target'],
            train_df['user_hash'], test_df['user_hash']
        ]).unique()

    def run_fast_label_propagation(self, seeds, n_iter=3):
        """Iterative label propagation on the graph."""
        adj = defaultdict(list)
        for _, row in self.graph.iterrows():
            adj[row['source']].append(row['target'])
            adj[row['target']].append(row['source'])
        
        scores = {user: 0.5 for user in self.all_users}
        scores.update(seeds.to_dict())
        
        for _ in range(n_iter):
            new_scores = {}
            for user in self.all_users:
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

    def compute_structural_features(self):
        """Computes Degree, Component Size, PageRank, etc."""
        print("Building NetworkX graph...")
        G = nx.from_pandas_edgelist(self.graph, "source", "target", create_using=nx.Graph())
        
        features = {}
        features['degree'] = dict(G.degree())
        features['pagerank'] = nx.pagerank(G, alpha=0.85)
        
        comp_map = {}
        for comp in nx.connected_components(G):
            size = len(comp)
            for node in comp:
                comp_map[node] = size
        features['component_size'] = comp_map
        
        # Neighbor Cheat Ratio
        user_to_label = self.train.set_index("user_hash")["is_cheating"].dropna().to_dict()
        nbr_cheat_ratio = {}
        num_labeled_nbrs = {}
        
        for node in G.nodes():
            nbrs = list(G.neighbors(node))
            labeled = [n for n in nbrs if n in user_to_label]
            if labeled:
                nbr_cheat_ratio[node] = np.mean([user_to_label[n] for n in labeled])
                num_labeled_nbrs[node] = len(labeled)
            else:
                nbr_cheat_ratio[node] = 0.0
                num_labeled_nbrs[node] = 0
                
        features['neighbor_cheat_ratio'] = nbr_cheat_ratio
        features['num_labeled_neighbors'] = num_labeled_nbrs
        
        return features

    def compute_risk_scores(self):
        """Runs LP via CV to generate OOF risk scores and test risk scores."""
        print("Running Label Propagation...")
        labeled_train = self.train[self.train['is_cheating'].notna()]
        clean_seeds = pd.Series(0.0, index=self.train[self.train['high_conf_clean'] == 1]['user_hash'].values)
        
        # OOF Risk Scores
        oof_risk = pd.Series(index=labeled_train['user_hash'], dtype=float)
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=CONFIG['SEED'])
        y = labeled_train['is_cheating'].values

        for train_idx, val_idx in kf.split(labeled_train, y):
            train_seeds = labeled_train.iloc[train_idx].set_index('user_hash')['is_cheating']
            seeds = pd.concat([train_seeds, clean_seeds])
            scores = self.run_fast_label_propagation(seeds, n_iter=CONFIG['LP_ITERATIONS'])
            
            val_users = labeled_train.iloc[val_idx]['user_hash']
            oof_risk.loc[val_users] = scores.loc[val_users]
            
        # Test Risk Scores
        all_seeds = pd.concat([labeled_train.set_index('user_hash')['is_cheating'], clean_seeds])
        test_scores = self.run_fast_label_propagation(all_seeds, n_iter=CONFIG['LP_ITERATIONS'])
        
        return oof_risk, test_scores

    def aggregate_neighbor_features(self, feature_cols):
        """Aggregates tabular features of neighbors (Mean/Std)."""
        print("Aggregating neighbor features...")
        all_feat = pd.concat([
            self.train[["user_hash"] + feature_cols], 
            self.test[["user_hash"] + feature_cols]
        ]).drop_duplicates("user_hash").set_index("user_hash")

        # Create bidirectional edges for aggregation
        rev_graph = self.graph.rename(columns={"source": "target", "target": "source"})
        full_edges = pd.concat([self.graph, rev_graph], ignore_index=True)
        full_edges = full_edges.merge(all_feat, left_on="target", right_index=True, how="left")
        
        agg = full_edges.groupby("source")[feature_cols].agg(["mean", "std"])
        agg.columns = [f"nbr_{c[0]}_{c[1]}" for c in agg.columns]
        
        del all_feat, full_edges, rev_graph
        gc.collect()
        return agg




# ==============================================================================
# CLASS: Tabular Feature Engineering
# ==============================================================================
class TabularFeatureEngineer:
    def __init__(self, df, feature_cols):
        self.df = df
        self.feature_cols = feature_cols

    def process(self):
        # Quantiles and Flags
        self.df["f012_is_too_fast"] = (self.df["feature_012"] > self.df["feature_012"].quantile(0.95)).astype(int)
        self.df["f012_bin"] = pd.qcut(self.df["feature_012"], q=5, duplicates='drop').cat.codes
        self.df["f015_bin"] = pd.qcut(self.df["feature_015"], q=7, duplicates='drop').cat.codes
        self.df["f016_is_high"] = (self.df["feature_016"] > self.df["feature_016"].median()).astype(int)
        self.df["f012_in_risky_time"] = self.df["feature_012"] * (1 - self.df["feature_014"])    
        self.df["danger_f004"] = self.df["feature_004"].isin([0.0, 3.0, np.nan]).astype(int)
        self.df["missing_count"] = self.df[self.feature_cols].isin([np.nan]).sum(axis=1)

        # Relative features (comparing to neighbors)
        eps = 1e-5
        for col in self.feature_cols:
            nbr_mean = f"nbr_{col}_mean"
            if nbr_mean in self.df.columns:
                self.df[f"{col}_ratio"] = self.df[col] / (self.df[nbr_mean] + eps)
                self.df[f"{col}_diff"] = self.df[col] - self.df[nbr_mean]

        return self.df


# ==============================================================================
# CLASS: Stacking Ensemble
# ==============================================================================
class StackingEnsemble:
    def __init__(self, features, seed=42, n_folds=5):
        self.features = features
        self.seed = seed
        self.n_folds = n_folds
        self.models = self._get_base_models()

    def _get_base_models(self):
        # FIX: Pass early_stopping_rounds to the constructor for XGBoost
        models = [
            ("xgb", XGBClassifier(
                random_state=self.seed, 
                early_stopping_rounds=50,  # <--- Moved here
                **CONFIG["MODELS"]["xgb"]
            )),
            ("lgbm", LGBMClassifier(random_state=self.seed, **CONFIG["MODELS"]["lgbm"]))
        ]
        if CATBOOST_AVAILABLE:
            models.append(("cat", CatBoostClassifier(random_seed=self.seed, **CONFIG["MODELS"]["cat"])))
        return models

    def train_predict(self, X, y, X_test):
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        
        oof_preds = []
        test_preds = []

        # Train Base Models
        for name, model in self.models:
            print(f"Training {name}...")
            oof = np.zeros(len(X))
            test_pred = np.zeros(len(X_test))

            for tr_idx, va_idx in skf.split(X, y):
                X_tr, X_va = X.iloc[tr_idx][self.features], X.iloc[va_idx][self.features]
                y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

                # Handle specific fit params
                fit_params = {}
                if name == "xgb":
                    # FIX: Removed 'early_stopping_rounds' from here
                    fit_params = {'eval_set': [(X_va, y_va)], 'verbose': 0}
                elif name == "lgbm":
                    fit_params = {'eval_set': [(X_va, y_va)], 'eval_metric': "logloss", 
                                  'callbacks': [early_stopping(50, verbose=False)]}
                elif name == "cat":
                    # CatBoost still accepts early_stopping_rounds in fit()
                    fit_params = {'eval_set': (X_va, y_va), 'early_stopping_rounds': 50, 'verbose': False}

                model.fit(X_tr, y_tr, **fit_params)
                
                oof[va_idx] = model.predict_proba(X_va)[:, 1]
                test_pred += model.predict_proba(X_test[self.features])[:, 1] / self.n_folds

            oof_preds.append(oof)
            test_preds.append(test_pred)

        # Meta Learner (Logistic Regression)
        print("Training Meta Learner...")
        oof_stack = np.column_stack(oof_preds)
        test_stack = np.column_stack(test_preds)
        
        meta_oof = np.zeros(len(X))
        meta_test = np.zeros(len(X_test))
        
        for tr_idx, va_idx in skf.split(oof_stack, y):
            meta = LogisticRegression(random_state=self.seed, max_iter=1000)
            meta.fit(oof_stack[tr_idx], y.iloc[tr_idx])
            meta_oof[va_idx] = meta.predict_proba(oof_stack[va_idx])[:, 1]
            meta_test += meta.predict_proba(test_stack)[:, 1] / self.n_folds

        return meta_oof, meta_test



# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("Loading data...")
    train = pd.read_csv(os.path.join(CONFIG["INPUT_DIR"], "train.csv"))
    test = pd.read_csv(os.path.join(CONFIG["INPUT_DIR"], "test.csv"))
    graph = pd.read_csv(os.path.join(CONFIG["INPUT_DIR"], "social_graph.csv"), names=["source", "target"])
    
    base_feat_cols = [c for c in train.columns if c.startswith("feature_")]

    # 1. Graph Engineering
    gfe = GraphFeatureEngineer(train, test, graph)
    
    # Neighbor Aggregations
    nbr_stats = gfe.aggregate_neighbor_features(base_feat_cols)
    train = train.merge(nbr_stats, left_on="user_hash", right_index=True, how="left").fillna(0)
    test = test.merge(nbr_stats, left_on="user_hash", right_index=True, how="left").fillna(0)
    
    # Risk Scores (Label Propagation)
    oof_risk, test_risk = gfe.compute_risk_scores()
    train['risk_score'] = train['user_hash'].map(oof_risk).fillna(0.5)
    test['risk_score'] = test['user_hash'].map(test_risk).fillna(0.5)
    
    # Structural Features
    struct_feats = gfe.compute_structural_features()
    for df in [train, test]:
        for k, v in struct_feats.items():
            df[k] = df['user_hash'].map(v).fillna(0 if k != 'component_size' else 1)

    # 2. Tabular Feature Engineering
    tfe_train = TabularFeatureEngineer(train, base_feat_cols)
    train = tfe_train.process()
    
    tfe_test = TabularFeatureEngineer(test, base_feat_cols)
    test = tfe_test.process()

    # Define Feature List
    exclude = ['user_hash', 'is_cheating', 'high_conf_clean']
    all_features = [c for c in train.columns if c not in exclude and train[c].dtype in [np.float64, np.int64]]
    print(f"Total features: {len(all_features)}")

    # 3. Model Training
    labeled = train[train["is_cheating"].notnull()].reset_index(drop=True)
    X = labeled.reset_index(drop=True)
    y = labeled["is_cheating"].astype(int).reset_index(drop=True)
    X_test = test.reset_index(drop=True)

    stacker = StackingEnsemble(features=all_features, seed=CONFIG["SEED"], n_folds=CONFIG["N_FOLDS"])
    final_oof, final_test = stacker.train_predict(X, y, X_test)

    print(f"Final AUC: {roc_auc_score(y, final_oof):.5f}")

    # 4. Submission
    submission = pd.DataFrame({"user_hash": test["user_hash"], "prediction": final_test})
    submission.to_csv("submission.csv", index=False)
    print("✅ Saved final submission.")

