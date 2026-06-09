# [Mercor] Smart Cost-Optimization + Graph XGBoost
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.optimize import minimize
import warnings

plt.style.use('ggplot')
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    seed = 42
    n_folds = 5
    initial_thresholds = [0.2, 0.95]

# ==========================================
# 2. DATA LOADING & EDA
# ==========================================
def load_data():
    train = pd.read_csv('/kaggle/input/mercor-cheating-detection/train.csv')
    test = pd.read_csv('/kaggle/input/mercor-cheating-detection/test.csv')
    graph = pd.read_csv('/kaggle/input/mercor-cheating-detection/social_graph.csv')
    
    print(f"Train Shape: {train.shape}")
    print(f"Test Shape: {test.shape}")
    
    return train, test, graph

def plot_feature_distribution(df, feature):
    plt.figure(figsize=(10, 5))
    sns.histplot(np.log1p(df[feature]), kde=True, color='purple')
    plt.title(f'Log-Transformed Distribution of {feature}')
    plt.xlabel(f'Log(1 + {feature})')
    plt.show()

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
def extract_features(train, test, graph):
    print("Generating Features...")
    
    degrees = pd.concat([graph['user_a'], graph['user_b']]).value_counts()
    
    # Map degrees
    train['degree'] = train['user_hash'].map(degrees).fillna(0)
    test['degree'] = test['user_hash'].map(degrees).fillna(0)
    
    # --- Tabular Features ---
    all_df = pd.concat([train.drop(columns=['is_cheating', 'high_conf_clean']), test], axis=0)
    
    # 1. Log Transform Feature 10
    all_df['feature_010_log'] = np.log1p(all_df['feature_010'])
    
    # 2. Interaction Feature
    all_df['f6_x_f14'] = all_df['feature_006'] * all_df['feature_014']
    
    # 3. Missingness Flags
    for col in [c for c in all_df.columns if c.startswith('feature_')]:
        if all_df[col].isnull().any():
            all_df[f'{col}_nan'] = all_df[col].isnull().astype(int)
    
    # Split back
    train_feat = all_df.iloc[:len(train)].copy()
    test_feat = all_df.iloc[len(train):].copy()
    
    # Add Targets back
    train_feat['is_cheating'] = train['is_cheating']
    train_feat['high_conf_clean'] = train['high_conf_clean']
    
    return train_feat, test_feat

# ==========================================
# 4. COST FUNCTION
# ==========================================
def calculate_cost(y_true, y_pred, t1, t2):
    # Vectorized Cost Calculation
    cost = 0
    # Auto-Pass (False Negatives)
    cost += np.sum((y_pred < t1) & (y_true == 1)) * 600
    # Auto-Block (False Positives)
    cost += np.sum((y_pred >= t2) & (y_true == 0)) * 300
    # Manual Review
    mask_review = (y_pred >= t1) & (y_pred < t2)
    cost += np.sum((mask_review) & (y_true == 1)) * 5
    cost += np.sum((mask_review) & (y_true == 0)) * 150
    return cost

def optimize_thresholds(y_true, y_pred):
    def objective(x):
        return calculate_cost(y_true, y_pred, x[0], x[1])
    res = minimize(objective, Config.initial_thresholds, method='Nelder-Mead')
    return res.x

# ==========================================
# 5. TRAINING
# ==========================================
def run_training():
    train, test, graph = load_data()
    
    plot_feature_distribution(train, 'feature_010')
    
    train, test = extract_features(train, test, graph)
    
    train.loc[train['high_conf_clean'] == 1, 'is_cheating'] = 0
    train = train[train['is_cheating'].notna()]
    
    features = [c for c in train.columns if c not in ['user_hash', 'is_cheating', 'high_conf_clean']]
    
    print(f"Training with {len(features)} features on {len(train)} samples.")
    
    skf = StratifiedKFold(n_splits=Config.n_folds, shuffle=True, random_state=Config.seed)
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        tree_method='hist',
        random_state=Config.seed
    )
    
    for fold, (idx_t, idx_v) in enumerate(skf.split(train, train['is_cheating'])):
        X_tr, y_tr = train.iloc[idx_t][features], train.iloc[idx_t]['is_cheating']
        X_va, y_va = train.iloc[idx_v][features], train.iloc[idx_v]['is_cheating']
        
        model.fit(X_tr, y_tr, verbose=False)
        oof_preds[idx_v] = model.predict_proba(X_va)[:, 1]
        test_preds += model.predict_proba(test[features])[:, 1] / Config.n_folds
        print(f"Fold {fold+1} Done.")
        
    # --- VISUALIZATION: FEATURE IMPORTANCE ---
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model, max_num_features=15, height=0.5, color='teal')
    plt.title("Top 15 Features Driving Cheating Detection")
    plt.show()
    
    # --- OPTIMIZATION ---
    print("\nFinding Optimal Business Thresholds...")
    best_t1, best_t2 = optimize_thresholds(train['is_cheating'], oof_preds)
    print(f"Optimal Thresholds: Pass < {best_t1:.3f} | Block >= {best_t2:.3f}")
    
    final_cost = calculate_cost(train['is_cheating'], oof_preds, best_t1, best_t2)
    print(f"Total CV Cost: ${final_cost:,.0f}")
    
    # Submit
    sub = pd.DataFrame({'user_hash': test['user_hash'], 'prediction': test_preds})
    sub.to_csv('submission.csv', index=False)
    print("Saved submission.csv")

if __name__ == "__main__":
    run_training()




