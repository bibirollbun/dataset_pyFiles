# world Ğ¿Ğ¾Ğ´Ğ°Ñ€Ğ¸Ñ‚ Ñ�ĞºĞ¾Ğ»ÑŒĞºĞ¾ Ğ¿Ñ€Ğ¾Ñ�Ğ¸ÑˆÑŒ, Ğ·Ğ°Ğ±ĞµÑ€Ñ‘Ñ‚, ĞºĞ¾Ğ³Ğ´Ğ° Ğ½Ğµ Ğ¶Ğ´Ñ‘ÑˆÑŒğŸ§šğŸ�»â€�â™€ï¸�
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr

class Config:
    TRAIN_PATH       = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH        = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH  = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    FEATURES         = [
        "X863","X856","X344","X598","X862","X385","X852","X603",
        "X860","X674","X415","X345","X137","X855","X174","X302",
        "X178","X532","X168","X612", "bid_qty","ask_qty","buy_qty","sell_qty","volume"
    ]
    LABEL_COLUMN     = "label"
    N_FOLDS          = 3
    RANDOM_STATE     = 42

XGB_PARAMS = {
    "tree_method": "hist",  
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 300,
    "random_state": Config.RANDOM_STATE,
    "n_jobs": -1
}

LGBM_PARAMS = {
    "boosting_type": "gbdt",
    "n_jobs": -1,
    "random_state": Config.RANDOM_STATE,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 300,
    "verbose": -1
}

# === ĞœĞ¾Ğ´ĞµĞ»Ğ¸ ===
LEARNERS = [
    {"name": "xgb",  "Estimator": XGBRegressor,  "params": XGB_PARAMS},
    {"name": "lgbm", "Estimator": LGBMRegressor, "params": LGBM_PARAMS}
]

MODEL_SLICES = [
    {"name": "full_data",   "cutoff": 0},
    {"name": "last_75pct",  "cutoff": 0},  #  one day it will be
    {"name": "last_50pct",  "cutoff": 0}
]

# === Ğ¤ÑƒĞ½ĞºÑ†Ğ¸Ğ¸ ===
def create_time_decay_weights(n: int, decay: float = 0.95) -> np.ndarray:
    positions = np.arange(n)
    normalized = positions / float(n - 1)
    weights = decay ** (1.0 - normalized)
    return weights * n / weights.sum()

def load_data():
    train_df = pd.read_parquet(Config.TRAIN_PATH, columns=Config.FEATURES + [Config.LABEL_COLUMN])
    test_df = pd.read_parquet(Config.TEST_PATH, columns=Config.FEATURES)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), submission_df

# the main thing
train_df, test_df, submission_df = load_data()
n_samples = len(train_df)
MODEL_SLICES[1]["cutoff"] = int(0.25 * n_samples)
MODEL_SLICES[2]["cutoff"] = int(0.50 * n_samples)

full_weights = create_time_decay_weights(n_samples)
kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)

# storage for predictions
oof_preds = {
    learner["name"]: {sl["name"]: np.zeros(n_samples) for sl in MODEL_SLICES}
    for learner in LEARNERS
}
test_preds = {
    learner["name"]: {sl["name"]: np.zeros(len(test_df)) for sl in MODEL_SLICES}
    for learner in LEARNERS
}

# training
for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), 1):
    print(f"\n--- Fold {fold}/{Config.N_FOLDS} ---")
    X_valid = train_df.iloc[valid_idx][Config.FEATURES]
    y_valid = train_df.iloc[valid_idx][Config.LABEL_COLUMN]

    for sl in MODEL_SLICES:
        slice_name = sl["name"]
        cutoff = sl["cutoff"]
        subset = train_df.iloc[cutoff:].reset_index(drop=True)
        rel_idx = train_idx[train_idx >= cutoff] - cutoff

        X_train = subset.iloc[rel_idx][Config.FEATURES]
        y_train = subset.iloc[rel_idx][Config.LABEL_COLUMN]
        sw = create_time_decay_weights(len(subset))[rel_idx] if cutoff else full_weights[train_idx]

        for learner in LEARNERS:
            name = learner["name"]
            Estimator = learner["Estimator"]
            model = Estimator(**learner["params"])

            try:
                model.fit(
                    X_train, y_train,
                    sample_weight=sw,
                    eval_set=[(X_valid, y_valid)],
                    early_stopping_rounds=50,
                    verbose=False
                )

                mask = valid_idx >= cutoff
                if mask.any():
                    idxs = valid_idx[mask]
                    oof_preds[name][slice_name][idxs] = model.predict(train_df.iloc[idxs][Config.FEATURES])
                if cutoff > 0 and (~mask).any():
                    oof_preds[name][slice_name][valid_idx[~mask]] = oof_preds[name]["full_data"][valid_idx[~mask]]

                test_preds[name][slice_name] += model.predict(test_df[Config.FEATURES])

            except Exception as e:
                print(f"Error training {name} on {slice_name}: {e}")
                oof_preds[name][slice_name][valid_idx] = y_valid.mean()
                test_preds[name][slice_name] += y_valid.mean()
# need some metrics here(//....
# aggregate
for name in test_preds:
    for slice_name in test_preds[name]:
        test_preds[name][slice_name] /= Config.N_FOLDS

final_test = np.mean([
    np.mean(list(test_preds[name].values()), axis=0)
    for name in test_preds
], axis=0)

submission_df["prediction"] = final_test
submission_df.to_csv("submission.csv", index=False)
print("here we are: submission.csv ")


