def sort_test_df_by_time(df):
    assert len(df.shape) == 2
    assert df.shape[0] == 538150

    n = df.shape[0]
    t = pd.Series(np.arange(n))
    t = t.sample(n=n, random_state=700)

    t = pd.Series(np.arange(n), index=t.to_numpy()).sort_index()
    return df.iloc[t.to_numpy()]


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import warnings

warnings.filterwarnings("ignore")


class Config:
    TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    # use the submission file with reset order
    SUBMISSION_PATH = "/kaggle/input/drw-submission-order-reset/sample_submission_idx.csv"

    FEATURES = list(set([
        "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X137", "X855", "X174", "X302", "X178", "X532", "X168", "X612",
        "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", "X888", "X421", "X333",
    ]))

    LABEL_COLUMN = "label"
    N_FOLDS = 3
    RANDOM_STATE = 42
    lags = [1,3,5,10,20,60,120,180,240,60*24,60*24*2,60*24*3]


def create_time_decay_weights(n: int, decay: float = 0.9) -> np.ndarray:
    positions = np.arange(n)
    normalized = positions / (n - 1)
    weights = decay ** (1.0 - normalized)
    return weights * n / weights.sum()


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

    FEATURES = [
        "X863", "X856", "X598", "X862", "X385", "X603", "X860", "X674", "X415", "X345", "X855", "X174", "X302",
        "X178", "X168", "X612", "X333", "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", 'log_volume',
        'bid_ask_interaction', 'bid_buy_interaction', 'bid_sell_interaction', 'ask_buy_interaction',
        'ask_sell_interaction']

    for col in FEATURES:
        for lag in Config.lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            df[f'{col}_lead_{lag}'] = df[col].shift(-lag)

    df = df.fillna(df.mean())

    return df


def sort_test_df_by_time(df):
    assert len(df.shape) == 2
    assert df.shape[0] == 538150

    n = df.shape[0]
    t = pd.Series(np.arange(n))
    t = t.sample(n=n, random_state=700)

    t = pd.Series(np.arange(n), index=t.to_numpy()).sort_index()
    return df.iloc[t.to_numpy()]

def load_data():
    train_df = pd.read_parquet(Config.TRAIN_PATH, columns=Config.FEATURES + [Config.LABEL_COLUMN])
    test_df = pd.read_parquet(Config.TEST_PATH, columns=Config.FEATURES)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Loaded data - Train: {train_df.shape}, Test: {test_df.shape}, Submission: {submission_df.shape}")

    # restore the test time order
    test_df = sort_test_df_by_time(test_df).reset_index(drop=True)

    train_df = add_features(train_df)
    test_df = add_features(test_df)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), submission_df


# =========================
# Training and Evaluation
# =========================
def train_and_evaluate(train_df, test_df):
    FEATURES = [
        "X863", "X856", "X598", "X862", "X385", "X603", "X860", "X674", "X415", "X345", "X855", "X174", "X302",
        "X178", "X168", "X612", "X333", "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", 'log_volume',
        'bid_ask_interaction', 'bid_buy_interaction', 'bid_sell_interaction', 'ask_buy_interaction',
        'ask_sell_interaction']
    
    lag_features = [f'{f}_lag_{l}' for f in FEATURES for l in Config.lags]
    lead_features = [f'{f}_lead_{l}' for f in FEATURES for l in Config.lags]

    FEATURES += lag_features
    FEATURES += lead_features

    n_samples = len(train_df)
    test_preds = np.zeros(len(test_df))

    full_weights = create_time_decay_weights(n_samples)
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), start=1):


        X_train = train_df.iloc[train_idx][FEATURES]
        y_train = train_df.iloc[train_idx][Config.LABEL_COLUMN]
        sw = full_weights[train_idx]


        model = Ridge(alpha = 0.105)
        model.fit(
            X_train, y_train,
            sample_weight=sw
        )

        test_preds += model.predict(test_df[FEATURES])
    return test_preds



def ensemble_simple(test_preds, submission_df):
    submission_df["prediction"] = test_preds
    submission_df = submission_df.sort_values(by='idx', ascending=True)
    submission_df = submission_df.drop('idx', axis=1)
    submission_df['ID'] = range(1, len(submission_df) + 1)
    submission_df = submission_df.reset_index(drop=True)
    submission_df.to_csv(f"submission.csv", index=False)



if __name__ == "__main__":
    train_df, test_df, submission_df = load_data()
    test_preds = train_and_evaluate(train_df, test_df)
    ensemble_simple(test_preds, submission_df)




