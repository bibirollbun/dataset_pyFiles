from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import numpy as np
import pandas as pd


# =========================
# Configuration
# =========================
class Config:
    TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"

    FEATURES = [
        "X863", "X856", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X855", "X174", "X302", "X178", "X168", "X612", "bid_qty",
        "ask_qty", "buy_qty", "sell_qty", "volume", "X888", "X421", "X333","X817", 
        "X586",  "X292"
    ]

    LABEL_COLUMN = "label"
    N_FOLDS = 3
    RANDOM_STATE = 42

XGB_PARAMS = {
    "tree_method": "hist",
    "device": "gpu",
    "colsample_bylevel": 0.4778,
    "colsample_bynode": 0.3628,
    "colsample_bytree": 0.7107,
    "gamma": 1.7095,
    "learning_rate": 0.02213,
    "max_depth": 20,
    "max_leaves": 12,
    "min_child_weight": 16,
    "n_estimators": 1667,
    "subsample": 0.06567,
    "reg_alpha": 39.3524,
    "reg_lambda": 75.4484,
    "verbosity": 0,
    "random_state": Config.RANDOM_STATE,
    "n_jobs": -1
}

LEARNERS = [
    {"name": "xgb", "Estimator": XGBRegressor, "params": XGB_PARAMS}
]


def add_features(df):
    df['bid_ask_interaction'] = df['bid_qty'] * df['ask_qty']
    df['bid_buy_interaction'] = df['bid_qty'] * df['buy_qty']
    df['bid_sell_interaction'] = df['bid_qty'] * df['sell_qty']
    df['ask_buy_interaction'] = df['ask_qty'] * df['buy_qty']
    df['ask_sell_interaction'] = df['ask_qty'] * df['sell_qty']


    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'])
    df['selling_pressure'] = df['sell_qty'] / (df['volume'])
    df['log_volume'] = np.log1p(df['volume'])

    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'])
    df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'])
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'])
    df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'])
    

    return df

def create_time_decay_weights(n: int, decay: float = 0.9) -> np.ndarray:
    positions = np.arange(n)
    normalized = positions / (n - 1)
    weights = decay ** (1.0 - normalized)
    return weights * n / weights.sum()
    
def load_data():
    train_df = pd.read_parquet(Config.TRAIN_PATH, columns=Config.FEATURES + [Config.LABEL_COLUMN])
    test_df = pd.read_parquet(Config.TEST_PATH, columns=Config.FEATURES)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Loaded data - Train: {train_df.shape}, Test: {test_df.shape}, Submission: {submission_df.shape}")

    train_df = add_features(train_df)
    test_df = add_features(test_df)

    Config.FEATURES += ["log_volume", 'bid_ask_interaction', 'bid_buy_interaction', 'bid_sell_interaction', 'ask_buy_interaction',
                        'ask_sell_interaction']

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), submission_df


def get_model_slices(n_samples: int):
    return [
        {"name": "full_data", "cutoff": 0},
        {"name": "last_90pct", "cutoff": int(0.10 * n_samples)},
        {"name": "last_85pct", "cutoff": int(0.15 * n_samples)},
        {"name": "last_80pct", "cutoff": int(0.20 * n_samples)},

    ]


# =========================
# Training and Evaluation
# =========================
def train_and_evaluate(train_df, test_df):
    n_samples = len(train_df)
    model_slices = get_model_slices(n_samples)

    oof_preds = {
        learner["name"]: {s["name"]: np.zeros(n_samples) for s in model_slices}
        for learner in LEARNERS
    }
    test_preds = {
        learner["name"]: {s["name"]: np.zeros(len(test_df)) for s in model_slices}
        for learner in LEARNERS
    }

    full_weights = create_time_decay_weights(n_samples)
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), start=1):
        print(f"\n--- Fold {fold}/{Config.N_FOLDS} ---")
        X_valid = train_df.iloc[valid_idx][Config.FEATURES]
        y_valid = train_df.iloc[valid_idx][Config.LABEL_COLUMN]

        for s in model_slices:
            cutoff = s["cutoff"]
            slice_name = s["name"]
            subset = train_df.iloc[cutoff:].reset_index(drop=True)
            rel_idx = train_idx[train_idx >= cutoff] - cutoff

            X_train = subset.iloc[rel_idx][Config.FEATURES]
            y_train = subset.iloc[rel_idx][Config.LABEL_COLUMN]
            sw = create_time_decay_weights(len(subset))[rel_idx] if cutoff > 0 else full_weights[train_idx]

            print(f"  Training slice: {slice_name}, samples: {len(X_train)}")

            for learner in LEARNERS:
                model = learner["Estimator"](**learner["params"])
                model.fit(X_train, y_train, sample_weight=sw, eval_set=[(X_valid, y_valid)], verbose=False)

                mask = valid_idx >= cutoff
                if mask.any():
                    idxs = valid_idx[mask]
                    oof_preds[learner["name"]][slice_name][idxs] = model.predict(train_df.iloc[idxs][Config.FEATURES])
                if cutoff > 0 and (~mask).any():
                    oof_preds[learner["name"]][slice_name][valid_idx[~mask]] = oof_preds[learner["name"]]["full_data"][
                        valid_idx[~mask]]


                test_preds[learner["name"]][slice_name] += model.predict(test_df[Config.FEATURES])

    # Normalize test predictions
    for learner_name in test_preds:
        for slice_name in test_preds[learner_name]:
            test_preds[learner_name][slice_name] /= (Config.N_FOLDS-1)

    return oof_preds, test_preds, model_slices


# =========================
# Ensemble & Submission
# =========================
def ensemble_and_submit(train_df, oof_preds, test_preds, submission_df):
    learner_name = 'xgb'
    weights = np.array([1,1,1,1])

    oof_weighted = pd.DataFrame(oof_preds[learner_name]).values @ weights
    test_weighted = pd.DataFrame(test_preds[learner_name]).values @ weights
    score_weighted = pearsonr(train_df[Config.LABEL_COLUMN], oof_weighted)[0]
    print(f"{learner_name.upper()} Weighted Ensemble Pearson: {score_weighted:.4f}")


    submission_df["prediction"] = test_weighted
    submission_df.to_csv("submission.csv", index=False)
    print("Saved: submission.csv")
    print (submission_df.head(10))



if __name__ == "__main__":
    train_df, test_df, submission_df = load_data()
    oof_preds, test_preds, model_slices = train_and_evaluate(train_df, test_df)
    ensemble_and_submit(train_df, oof_preds, test_preds, submission_df)











