import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr


def feature_engineering(df):
   #10
    df['exp_856P868P855P289'] = np.exp(df['X856'] + df['X868'] + df['X855'] + df['X289'])
    df['exp_860P868P855P289'] = np.exp(df['X860'] + df['X868'] + df['X855'] + df['X289'])
    df['exp_598P868P855P289'] = np.exp(df['X598'] + df['X868'] + df['X855'] + df['X289'])
    df['exp_612P868P855P289'] = np.exp(df['X612'] + df['X868'] + df['X855'] + df['X289'])
    df['exp_289P855P21'] = np.exp(df['X289'] + df['X855'] + df['X21'])
    df['868xexp_289M125'] = df['X868'] * np.exp(df['X289'] - df['X125'])
    #9
    df['exp_603P868P855P289'] = np.exp(df['X603'] + df['X868'] + df['X855'] + df['X289'])
    df['exp_174P868P855P289'] = np.exp(df['X174'] + df['X868'] + df['X855'] + df['X289'])
    df['exp_465P868P855P289'] = np.exp(df['X465'] + df['X868'] + df['X855'] + df['X289'])
    df['exp_125P862P289M125'] = np.exp(df['X125'] + df['X862'] + df['X289'] - df['X125'])
    df['exp_168P868P855P289'] = np.exp(df['X168'] + df['X868'] + df['X855'] + df['X289'])
    df['exp_855P289M125'] = np.exp(df['X855'] + df['X289'] - df['X125'])
    df['exp_302P289M125'] = np.exp(df['X302'] + df['X289'] - df['X125'])
    df['289xexp_289M125'] = df['X289'] * np.exp(df['X289'] - df['X125'])
    #8
    df['exp_862P868P855P289'] = np.exp(df['X862'] + df['X868'] + df['X855'] + df['X289'])
    df['868x868x855x289'] = df['X868'] * df['X868'] * df['X855'] * df['X289']
    df['385xexp_289M125'] = df['X385'] * np.exp(df['X289'] - df['X125'])
    df['exp_862P289M125'] = np.exp(df['X862'] + df['X289'] - df['X125'])
    df['exp_786P289M125'] = np.exp(df['X786'] + df['X289'] - df['X125'])
    df['exp_856P289M125'] = np.exp(df['X856'] + df['X289'] - df['X125'])
    df['852x868x855x289'] = df['X852'] * df['X868'] * df['X855'] * df['X289']
    df['465x862x465']=df['X465']*df['X465']*df['X862']
    df['540x881']=df['X540']*df['X881']
    
    df['bid_ask_interaction'] = df['bid_qty'] * df['ask_qty']
    df['bid_buy_interaction'] = df['bid_qty'] * df['buy_qty']
    df['bid_sell_interaction'] = df['bid_qty'] * df['sell_qty']
    df['ask_buy_interaction'] = df['ask_qty'] * df['buy_qty']
    df['ask_sell_interaction'] = df['ask_qty'] * df['sell_qty']

    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-10)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-10)
    df['log_volume'] = np.log1p(df['volume'])

    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-10)
    df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + 1e-10)
    
    df['ask_buy_interaction_x_X293']=df['X293']*df['ask_buy_interaction']
     # Price Pressure Indicators
    df['net_order_flow'] = df['buy_qty'] - df['sell_qty']
    df['normalized_net_flow'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    df['buying_pressure'] = df['buy_qty'] / (df['volume'] + 1e-10)
    df['volume_weighted_buy'] = df['buy_qty'] * df['volume']
    
    # Liquidity Depth Measures
    df['total_depth'] = df['bid_qty'] + df['ask_qty']
    df['depth_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['relative_spread'] = np.abs(df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['log_depth'] = np.log1p(df['total_depth'])
    
    # Order Flow Toxicity Proxies
    df['kyle_lambda'] = np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['flow_toxicity'] = np.abs(df['order_flow_imbalance']) * df['volume']
    df['aggressive_flow_ratio'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    
    # Market Activity Indicators
    df['volume_depth_ratio'] = df['volume'] / (df['total_depth'] + 1e-10)
    df['activity_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-10)
    df['log_buy_qty'] = np.log1p(df['buy_qty'])
    df['log_sell_qty'] = np.log1p(df['sell_qty'])
    df['log_bid_qty'] = np.log1p(df['bid_qty'])
    df['log_ask_qty'] = np.log1p(df['ask_qty'])
    
    # Microstructure Volatility Proxies
    df['realized_spread_proxy'] = 2 * np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['price_impact_proxy'] = df['net_order_flow'] / (df['total_depth'] + 1e-10)
    df['quote_volatility_proxy'] = np.abs(df['depth_imbalance'])
    
    # Complex Interaction Terms
    df['flow_depth_interaction'] = df['net_order_flow'] * df['total_depth']
    df['imbalance_volume_interaction'] = df['order_flow_imbalance'] * df['volume']
    df['depth_volume_interaction'] = df['total_depth'] * df['volume']
    df['buy_sell_spread'] = np.abs(df['buy_qty'] - df['sell_qty'])
    df['bid_ask_spread'] = np.abs(df['bid_qty'] - df['ask_qty'])
    
    # Information Asymmetry Measures
    df['trade_informativeness'] = df['net_order_flow'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['execution_shortfall_proxy'] = df['buy_sell_spread'] / (df['volume'] + 1e-10)
    df['adverse_selection_proxy'] = df['net_order_flow'] / (df['total_depth'] + 1e-10) * df['volume']
    
    # Market Efficiency Indicators
    df['fill_probability'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['execution_rate'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    df['market_efficiency'] = df['volume'] / (df['bid_ask_spread'] + 1e-10)
    
    # Non-linear Transformations
    df['sqrt_volume'] = np.sqrt(df['volume'])
    df['sqrt_depth'] = np.sqrt(df['total_depth'])
    df['volume_squared'] = df['volume'] ** 2
    df['imbalance_squared'] = df['order_flow_imbalance'] ** 2
    
    # Relative Measures
    df['bid_ratio'] = df['bid_qty'] / (df['total_depth'] + 1e-10)
    df['ask_ratio'] = df['ask_qty'] / (df['total_depth'] + 1e-10)
    df['buy_ratio'] = df['buy_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['sell_ratio'] = df['sell_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    
    # Market Stress Indicators
    df['liquidity_consumption'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    df['market_stress'] = df['volume'] / (df['total_depth'] + 1e-10) * np.abs(df['order_flow_imbalance'])
    df['depth_depletion'] = df['volume'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    
    # Directional Indicators
    df['net_buying_ratio'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    df['directional_volume'] = df['net_order_flow'] * np.log1p(df['volume'])
    df['signed_volume'] = np.sign(df['net_order_flow']) * df['volume']

    #etc
    df['sqrt_volume_div_log_volume'] = df['sqrt_volume'] / (df['log_volume'] + 1e-6)
    df['sqrt_volume_div_activity_intensity'] = df['sqrt_volume'] / (df['activity_intensity'] + 1e-6)
    df['sqrt_volume_mul_fill_probability'] = df['sqrt_volume'] * df['fill_probability']
    df['volume_div_sqrt_volume'] = df['volume'] / (df['sqrt_volume'] + 1e-6)
    df['sqrt_volume_div_fill_probability'] = df['sqrt_volume'] / (df['fill_probability'] + 1e-6)
    df['sqrt_volume_mul_activity_intensity'] = df['sqrt_volume'] * df['activity_intensity']
    df['sqrt_volume_div_log_sell_qty'] = df['sqrt_volume'] / (df['log_sell_qty'] + 1e-6)
    df['log_buy_qty_mul_sqrt_volume'] = df['log_buy_qty'] * df['sqrt_volume']
    df['sqrt_volume_mul_log_buy_qty'] = df['sqrt_volume'] * df['log_buy_qty']
    df['log_volume_mul_sqrt_volume'] = df['log_volume'] * df['sqrt_volume']
    
    df['log_sell_qty_mul_X598'] = df['log_sell_qty'] * df['X598']
    df['log_buy_qty_mul_X598'] = df['log_buy_qty'] * df['X598']
    df['log_volume_mul_X598'] = df['log_volume'] * df['X598']
    
    df['sqrt_volume_mul_X856'] = df['sqrt_volume'] * df['X856']
    
    df['log_sell_qty_mul_X302'] = df['log_sell_qty'] * df['X302']
    df['log_volume_mul_X302'] = df['log_volume'] * df['X302']
    df['log_buy_qty_mul_X302'] = df['log_buy_qty'] * df['X302']
    
    df['log_sell_qty_mul_X292'] = df['log_sell_qty'] * df['X292']
    
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df 


class Config:
    TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"

    FEATURES = [
        "X863", "X856", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X855", "X174", "X302", "X178", "X168", "X612",
        "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", "X888", "X421", "X333",
        'X465','X153','X289','X125','X21',"X868", "X786" ,"X293","X873",'X540','X493','X862',
        'X881','X425','X858',"X292","X817", "X586"
        
        
    ]
    SELECTED_FEATURES=[

        "X863", "X856", "X598", "X862", "X385", "X603", "X860", "X674",
        "X415", "X345", "X855", "X174", "X302", "X178", "X168", "X612",
        "buy_qty", "sell_qty", "volume", 
        "X888", "X421", "X333","X292","X817", 
        "X586",
        'ask_buy_interaction_x_X293',  '868xexp_289M125','exp_786P289M125','exp_856P289M125',
        'exp_612P868P855P289','exp_598P868P855P289',
        'exp_855P289M125',
        '385xexp_289M125','465x862x465','540x881','exp_125P862P289M125','bid_ask_interaction', 'bid_buy_interaction', 'bid_sell_interaction', 'ask_buy_interaction',
        'ask_sell_interaction', "log_volume", 'net_order_flow', 'normalized_net_flow',
        'buying_pressure', 'volume_weighted_buy', 'total_depth', 'depth_imbalance',
        'relative_spread', 'log_depth', 'kyle_lambda', 'flow_toxicity', 'aggressive_flow_ratio',
        'volume_depth_ratio', 'activity_intensity', 'log_buy_qty', 'log_sell_qty',
        'log_bid_qty', 'log_ask_qty', 'realized_spread_proxy', 'price_impact_proxy',
        'quote_volatility_proxy', 'flow_depth_interaction', 'imbalance_volume_interaction',
        'depth_volume_interaction',  'trade_informativeness',
        'execution_shortfall_proxy', 'adverse_selection_proxy', 'fill_probability',
        'execution_rate', 'market_efficiency', 'sqrt_volume', 'sqrt_depth', 'volume_squared',
        'imbalance_squared', 'bid_ratio', 'ask_ratio', 'buy_ratio', 'sell_ratio',
        'liquidity_consumption', 'market_stress', 'depth_depletion', 'net_buying_ratio',
        'directional_volume', 'signed_volume',   
        
    
        #"sqrt_volume_div_activity_intensity",
        "sqrt_volume_mul_fill_probability",
        "volume_div_sqrt_volume",
        #"sqrt_volume_div_fill_probability",
        #"sqrt_volume_mul_activity_intensity",
        #"sqrt_volume_div_log_sell_qty",
        "log_buy_qty_mul_sqrt_volume",
        "sqrt_volume_mul_log_buy_qty",
        "log_volume_mul_sqrt_volume",
        
        #"log_sell_qty_mul_X598",
        #"log_buy_qty_mul_X598",
        #"log_volume_mul_X598",
        "sqrt_volume_mul_X856",
        "log_sell_qty_mul_X302",
        "log_volume_mul_X302",
        "log_buy_qty_mul_X302",
        "log_sell_qty_mul_X292"


                      
    
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
    {"name": "xgb", "Estimator": XGBRegressor, "params": XGB_PARAMS},
]


def create_time_decay_weights(n: int, decay: float = 0.9) -> np.ndarray:
    positions = np.arange(n)
    normalized = positions / (n - 1)
    weights = decay ** (1.0 - normalized)
    return weights * n / weights.sum()

def load_data():
    train_df = pd.read_parquet(Config.TRAIN_PATH, columns=Config.FEATURES + [Config.LABEL_COLUMN])
    test_df = pd.read_parquet(Config.TEST_PATH, columns=Config.FEATURES)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)

    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    print(f"Loaded data - Train: {train_df.shape}, Test: {test_df.shape}, Submission: {submission_df.shape}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), submission_df


#Config.FEATURES += ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]
Config.FEATURES = list(set(Config.FEATURES))  # remove duplicates


def get_model_slices(n_samples: int):
    base_slices = [
        {"name": "full_data", "cutoff": 0, "is_oldest": False, "outlier_adjusted": False},
        {"name": "last_90pct", "cutoff": int(0.10 * n_samples), "is_oldest": False, "outlier_adjusted": False},
        {"name": "last_85pct", "cutoff": int(0.15 * n_samples), "is_oldest": False, "outlier_adjusted": False},
        {"name": "last_80pct", "cutoff": int(0.20 * n_samples), "is_oldest": False, "outlier_adjusted": False},
        {"name": "last_50pct", "cutoff": int(0.50 * n_samples), "is_oldest": False, "outlier_adjusted": False},
        {"name": "oldest_25pct", "cutoff": int(0.25 * n_samples), "is_oldest": True, "outlier_adjusted": False},
    ]
    
    
    return base_slices

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

    # ëª¨ë�¸ ì €ì�¥ìš© ë”•ì…”ë„ˆë¦¬ ì¶”ê°€ (ì˜ˆ: learner_name -> slice_name -> list of models per fold)
    trained_models = {
        learner["name"]: {s["name"]: [] for s in model_slices}
        for learner in LEARNERS
    }

    full_weights = create_time_decay_weights(n_samples)
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), start=1):
        print(f"\n--- Fold {fold}/{Config.N_FOLDS} ---")
        X_valid = train_df.iloc[valid_idx][Config.SELECTED_FEATURES]
        y_valid = train_df.iloc[valid_idx][Config.LABEL_COLUMN]

        for s in model_slices:
            cutoff = s["cutoff"]
            slice_name = s["name"]
            subset = train_df.iloc[cutoff:].reset_index(drop=True)
            rel_idx = train_idx[train_idx >= cutoff] - cutoff

            X_train = subset.iloc[rel_idx][Config.SELECTED_FEATURES]
            y_train = subset.iloc[rel_idx][Config.LABEL_COLUMN]
            sw = create_time_decay_weights(len(subset))[rel_idx] if cutoff > 0 else full_weights[train_idx]

            print(f"  Training slice: {slice_name}, samples: {len(X_train)}")

            for learner in LEARNERS:
                model = learner["Estimator"](**learner["params"])
                model.fit(X_train, y_train, sample_weight=sw, eval_set=[(X_valid, y_valid)], verbose=False)

                # í•™ìŠµë�œ ëª¨ë�¸ ì €ì�¥
                trained_models[learner["name"]][slice_name].append(model)

                mask = valid_idx >= cutoff
                if mask.any():
                    idxs = valid_idx[mask]
                    oof_preds[learner["name"]][slice_name][idxs] = model.predict(train_df.iloc[idxs][Config.SELECTED_FEATURES])
                if cutoff > 0 and (~mask).any():
                    oof_preds[learner["name"]][slice_name][valid_idx[~mask]] = oof_preds[learner["name"]]["full_data"][valid_idx[~mask]]

                test_preds[learner["name"]][slice_name] += model.predict(test_df[Config.SELECTED_FEATURES])

    # Normalize test predictions
    for learner_name in test_preds:
        for slice_name in test_preds[learner_name]:
            test_preds[learner_name][slice_name] /= Config.N_FOLDS

    return oof_preds, test_preds, model_slices, trained_models


manual_weights = {
    "full_data": 1,
    "last_90pct": 1,
    "last_85pct": 1,
    "last_80pct": 0.8,
    "last_50pct": 0.5,
    "oldest_25pct": 1,
}
adjust_weights={
    "full_data": 0.2,
    "last_90pct": 0.2,
    "last_85pct": 0.3,
    "last_80pct": 0.3,
    "last_50pct": 2,
    "oldest_25pct": 0,
    
}

def ensemble_and_submit(train_df, test_df, oof_preds, test_preds, submission_df, 
                        manual_weights=None, adjust_weights=None):
    learner_ensembles = {}

    # âœ… Train ë³¼ë¥¨ ê¸°ì¤€ ìƒ�ìœ„ 20% ì�„ê³„ê°’ ê³„ì‚°
    volume_threshold = train_df["volume"].quantile(0.8)

    # âœ… Test ë³¼ë¥¨ì�´ ìƒ�ìœ„ 20% ì´ˆê³¼ì�¸ì§€ í™•ì�¸
    if test_df["volume"].mean() > volume_threshold:
        print("âš ï¸� Test volume is high â€” using adjust_weights.")
        weights = adjust_weights if adjust_weights is not None else manual_weights
    else:
        print("âœ… Test volume is within normal range â€” using manual_weights.")
        weights = manual_weights

    if weights is None:
        weights = {s: 1.0 for s in next(iter(oof_preds.values())).keys()}
    
    total_weight = sum(weights.values())

    for learner_name in oof_preds:
        oof_weighted = sum(
            weights[s] / total_weight * oof_preds[learner_name][s]
            for s in weights if s in oof_preds[learner_name]
        )
        test_weighted = sum(
            weights[s] / total_weight * test_preds[learner_name][s]
            for s in weights if s in test_preds[learner_name]
        )

        score_weighted = pearsonr(train_df[Config.LABEL_COLUMN], oof_weighted)[0]
        print(f"{learner_name.upper()} Weighted Ensemble Pearson: {score_weighted:.4f}")

        learner_ensembles[learner_name] = {
            "oof_weighted": oof_weighted,
            "test_weighted": test_weighted
        }

    final_oof = np.mean([le["oof_weighted"] for le in learner_ensembles.values()], axis=0)
    final_test = np.mean([le["test_weighted"] for le in learner_ensembles.values()], axis=0)
    final_score = pearsonr(train_df[Config.LABEL_COLUMN], final_oof)[0]

    print(f"\nâœ… FINAL ensemble across learners (weighted): {final_score:.4f}")

    submission_df["prediction"] = final_test
    submission_df.to_csv("submission.csv", index=False)
    print("ğŸ“� Saved: submission.csv")


if __name__ == "__main__":
    train_df, test_df, submission_df = load_data()
    oof_preds, test_preds, model_slices,trained_models = train_and_evaluate(train_df, test_df)
    ensemble_and_submit(train_df, test_df, oof_preds, test_preds, submission_df,manual_weights)



"""
import shap
import matplotlib.pyplot as plt
# ëª¨ë�¸, ë�°ì�´í„° ì¤€ë¹„ ì½”ë“œ (ì˜ˆì‹œ)
learner_name = 'xgb'
fold_index = 0
slice_name_full = 'last_50pct'
model_full = trained_models[learner_name][slice_name_full][fold_index]

X_sample_full = train_df[Config.SELECTED_FEATURES]

# SHAP explainer ë°� ê°’ ê³„ì‚°
explainer_full = shap.TreeExplainer(model_full)
shap_values_full = explainer_full.shap_values(X_sample_full)

# SHAP plot
shap.summary_plot(shap_values_full, X_sample_full, max_display=30)
"""

