!pip install polars==1.31.0

import os
import time
import json
import pickle
import builtins
import numpy as np
import polars as pl
import pandas as pd
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.model_selection import ParameterGrid


CWD = "/home/rayllon/_inception/Kaggle/DRW_Crypto/xgboost/main_xgboost_ensemble_results"


# %%
class ConfigGrid:
    """
    * add_extra_features: Filter outliers below 1% and above 99%
    * filter_outliers → Percentile threshold for filtering features based on avg. feature importance of 3
      xgb of filter_features_model_n_rounds epochs each (only on training set)
    * filter_features: Number of rounds for the feature importance model
    * filter_features_model_n_rounds: Number of rounds for the feature importance model
    * scaler: Options: "standard", "minmax", "robust"
    * n_models: Number of models to train in the ensemble
    * eastop_patience: Early stopping patience each model in the ensemble
    * model_n_rounds: Number of rounds for each model in the ensemble
    * base_params: Base parameters for the XGBoost model
    * params: Fine tuning parameters for the XGBoost model
    * train_data_path: Path to the training data
    * models_save_dir: Directory to save the trained models
    """

    def __init__(
        self,
        add_extra_features,
        filter_outliers,
        filter_features,
        filter_features_model_n_rounds,
        scaler,
        n_models,
        eastop_patience,
        model_n_rounds,
        base_params,
        params,
        train_data_path,
        models_save_dir,
    ):
        self.ADD_EXTRA_FEATURES = add_extra_features
        self.FILTER_OUTLIERS = filter_outliers
        self.FILTER_FEATURES = filter_features
        self.FILTER_FEATURES_MODEL_N_ROUNDS = filter_features_model_n_rounds
        self.SCALER = scaler
        self.N_MODELS = n_models
        self.EASTOP_PATIENCE = eastop_patience
        self.MODEL_N_ROUNDS = model_n_rounds
        self.BASE_PARAMS = base_params
        self.PARAMS = params
        self.TRAIN_DATA_PATH = train_data_path
        self.MODELS_SAVE_DIR = models_save_dir

    def stream_configs(self):
        config_grid_dict_copy = self.__dict__.copy()
        base_params = config_grid_dict_copy.pop("BASE_PARAMS")
        params = config_grid_dict_copy.pop("PARAMS")
        config_dict = {**config_grid_dict_copy, **base_params, **params}

        for key, value in config_dict.items():
            if not isinstance(value, list):
                config_dict[key] = [value]

        grid = ParameterGrid(config_dict)
        for config in grid:
            config_obj = ConfigGrid(
                add_extra_features=config["ADD_EXTRA_FEATURES"],
                filter_outliers=config["FILTER_OUTLIERS"],
                filter_features=config["FILTER_FEATURES"],
                filter_features_model_n_rounds=config["FILTER_FEATURES_MODEL_N_ROUNDS"],
                scaler=config["SCALER"],
                n_models=config["N_MODELS"],
                eastop_patience=config["EASTOP_PATIENCE"],
                model_n_rounds=config["MODEL_N_ROUNDS"],
                base_params={k: v for k, v in config.items() if k in base_params},
                params={k: v for k, v in config.items() if k in params},
                train_data_path=config["TRAIN_DATA_PATH"],
                models_save_dir=config["MODELS_SAVE_DIR"],
            )
            yield config_obj

    def get_config_id(self):
        # id example: config_xgbensemble__extraFeatures__filterOutliers_1_99__filterFeatures_25_250_rounds__scaler_standard__nModels_10__eastopPatience_250__modelNRounds_2500
        filter_outliers_id_test = (
            f"filterOutliers_{self.FILTER_OUTLIERS[0]}_{self.FILTER_OUTLIERS[1]}"
            if self.FILTER_OUTLIERS is not None
            else "filterOutliers_None"
        )
        return f"config_xgbensemble__extraFeatures_{self.ADD_EXTRA_FEATURES}__{filter_outliers_id_test}__filterFeatures_{self.FILTER_FEATURES}_{self.FILTER_FEATURES_MODEL_N_ROUNDS}_rounds__scaler_{self.SCALER}__nModels_{self.N_MODELS}__eastopPatience_{self.EASTOP_PATIENCE}__modelNRounds_{self.MODEL_N_ROUNDS}"


# %%
def add_features_polars(df, test=False):
    # Original features
    df = df.with_columns((df["bid_qty"] * df["ask_qty"]).alias("bid_ask_interaction"))

    df = df.with_columns(
        ((df["buy_qty"] + df["sell_qty"]) / (df["bid_qty"] + df["ask_qty"] + 1e-10)).alias(
            "trade_depth_utilisation"
        )
    )
    df = df.with_columns((df["buy_qty"] / (df["bid_qty"] + 1e-10)).alias("side_depth_consumption_buy"))
    df = df.with_columns((df["sell_qty"] / (df["ask_qty"] + 1e-10)).alias("side_depth_consumption_sell"))

    df = df.with_columns((df["bid_qty"] * df["buy_qty"]).alias("bid_buy_interaction"))
    df = df.with_columns((df["bid_qty"] * df["sell_qty"]).alias("bid_sell_interaction"))
    df = df.with_columns((df["ask_qty"] * df["buy_qty"]).alias("ask_buy_interaction"))
    df = df.with_columns((df["ask_qty"] * df["sell_qty"]).alias("ask_sell_interaction"))

    df = df.with_columns((df["sell_qty"] * df["volume"]).alias("volume_weighted_sell"))
    df = df.with_columns((df["buy_qty"] / (df["sell_qty"] + 1e-10)).alias("buy_sell_ratio"))
    df = df.with_columns(
        (df["buy_qty"] / (df["sell_qty"] + df["buy_qty"] + 1e-10)).alias("taker_buy_sell_fraction")
    )
    df = df.with_columns((df["sell_qty"] / (df["volume"] + 1e-10)).alias("selling_pressure"))
    df = df.with_columns((np.log1p(df["volume"])).alias("log_volume"))

    df = df.with_columns(
        (np.abs(df["buy_qty"] - df["sell_qty"]) / (df["volume"] + 1e-10)).alias("effective_spread_proxy")
    )
    df = df.with_columns(
        ((df["bid_qty"] - df["ask_qty"]) / (df["bid_qty"] + df["ask_qty"] + 1e-10)).alias("bid_ask_imbalance")
    )
    df = df.with_columns(
        ((df["buy_qty"] - df["sell_qty"]) / (df["buy_qty"] + df["sell_qty"] + 1e-10)).alias(
            "order_flow_imbalance"
        )
    )
    df = df.with_columns(((df["bid_qty"] + df["ask_qty"]) / (df["volume"] + 1e-10)).alias("liquidity_ratio"))

    # === NEW MICROSTRUCTURE FEATURES ===

    # Price Pressure Indicators
    df = df.with_columns((df["buy_qty"] - df["sell_qty"]).alias("net_order_flow"))
    df = df.with_columns((df["net_order_flow"] / (df["volume"] + 1e-10)).alias("normalized_net_flow"))
    df = df.with_columns((df["buy_qty"] / (df["volume"] + 1e-10)).alias("buying_pressure"))
    df = df.with_columns((df["buy_qty"] * df["volume"]).alias("volume_weighted_buy"))

    # Liquidity Depth Measures
    df = df.with_columns((df["bid_qty"] + df["ask_qty"]).alias("total_depth"))
    df = df.with_columns(
        ((df["bid_qty"] - df["ask_qty"]) / (df["total_depth"] + 1e-10)).alias("depth_imbalance")
    )
    df = df.with_columns(
        (np.abs(df["bid_qty"] - df["ask_qty"]) / (df["total_depth"] + 1e-10)).alias("relative_spread")
    )
    df = df.with_columns((np.log1p(df["total_depth"])).alias("log_depth"))

    # Order Flow Toxicity Proxies
    df = df.with_columns((np.abs(df["net_order_flow"]) / (df["volume"] + 1e-10)).alias("kyle_lambda"))
    df = df.with_columns((np.abs(df["order_flow_imbalance"]) * df["volume"]).alias("flow_toxicity"))
    df = df.with_columns(
        ((df["buy_qty"] + df["sell_qty"]) / (df["total_depth"] + 1e-10)).alias("aggressive_flow_ratio")
    )

    # Market Activity Indicators
    df = df.with_columns((df["volume"] / (df["total_depth"] + 1e-10)).alias("volume_depth_ratio"))
    df = df.with_columns(
        ((df["buy_qty"] + df["sell_qty"]) / (df["volume"] + 1e-10)).alias("activity_intensity")
    )
    df = df.with_columns((np.log1p(df["buy_qty"])).alias("log_buy_qty"))
    df = df.with_columns((np.log1p(df["sell_qty"])).alias("log_sell_qty"))
    df = df.with_columns((np.log1p(df["bid_qty"])).alias("log_bid_qty"))
    df = df.with_columns((np.log1p(df["ask_qty"])).alias("log_ask_qty"))

    # Microstructure Volatility Proxies
    df = df.with_columns(
        (2 * np.abs(df["net_order_flow"]) / (df["volume"] + 1e-10)).alias("realized_spread_proxy")
    )
    df = df.with_columns((df["net_order_flow"] / (df["total_depth"] + 1e-10)).alias("price_impact_proxy"))
    df = df.with_columns((np.abs(df["depth_imbalance"])).alias("quote_volatility_proxy"))

    # Complex Interaction Terms
    df = df.with_columns((df["net_order_flow"] * df["total_depth"]).alias("flow_depth_interaction"))
    df = df.with_columns((df["order_flow_imbalance"] * df["volume"]).alias("imbalance_volume_interaction"))
    df = df.with_columns((df["total_depth"] * df["volume"]).alias("depth_volume_interaction"))
    df = df.with_columns((np.abs(df["buy_qty"] - df["sell_qty"])).alias("buy_sell_spread"))
    df = df.with_columns((np.abs(df["bid_qty"] - df["ask_qty"])).alias("bid_ask_spread"))

    # Information Asymmetry Measures
    df = df.with_columns(
        (df["net_order_flow"] / (df["bid_qty"] + df["ask_qty"] + 1e-10)).alias("trade_informativeness")
    )
    df = df.with_columns((df["buy_sell_spread"] / (df["volume"] + 1e-10)).alias("execution_shortfall_proxy"))
    df = df.with_columns(
        (df["net_order_flow"] / (df["total_depth"] + 1e-10) * df["volume"]).alias("adverse_selection_proxy")
    )

    # Market Efficiency Indicators
    df = df.with_columns((df["volume"] / (df["buy_qty"] + df["sell_qty"] + 1e-10)).alias("fill_probability"))
    df = df.with_columns(
        ((df["buy_qty"] + df["sell_qty"]) / (df["total_depth"] + 1e-10)).alias("execution_rate")
    )
    df = df.with_columns((df["volume"] / (df["bid_ask_spread"] + 1e-10)).alias("market_efficiency"))

    # Non-linear Transformations
    df = df.with_columns((np.sqrt(df["volume"])).alias("sqrt_volume"))
    df = df.with_columns((np.sqrt(df["total_depth"])).alias("sqrt_depth"))
    df = df.with_columns((df["volume"] ** 2).alias("volume_squared"))
    df = df.with_columns((df["order_flow_imbalance"] ** 2).alias("imbalance_squared"))

    # Relative Measures
    df = df.with_columns((df["bid_qty"] / (df["total_depth"] + 1e-10)).alias("bid_ratio"))
    df = df.with_columns((df["bid_qty"] / (df["ask_qty"] + 1e-10)).alias("bid_ask_depth_ratio"))
    df = df.with_columns((df["ask_qty"] / (df["total_depth"] + 1e-10)).alias("ask_ratio"))
    df = df.with_columns((df["buy_qty"] / (df["buy_qty"] + df["sell_qty"] + 1e-10)).alias("buy_ratio"))
    df = df.with_columns((df["sell_qty"] / (df["buy_qty"] + df["sell_qty"] + 1e-10)).alias("sell_ratio"))

    # Market Stress Indicators
    df = df.with_columns(
        (df["buy_qty"] + df["sell_qty"]) / (df["total_depth"] + 1e-10).alias("liquidity_consumption")
    )
    df = df.with_columns(
        (df["volume"] / (df["total_depth"] + 1e-10) * np.abs(df["order_flow_imbalance"])).alias(
            "market_stress"
        )
    )
    df = df.with_columns((df["volume"] / (df["bid_qty"] + df["ask_qty"] + 1e-10)).alias("depth_depletion"))

    # Directional Indicators
    df = df.with_columns((df["net_order_flow"] / (df["volume"] + 1e-10)).alias("net_buying_ratio"))
    df = df.with_columns((df["net_order_flow"] * np.log1p(df["volume"])).alias("directional_volume"))
    df = df.with_columns((np.sign(df["net_order_flow"]) * df["volume"]).alias("signed_volume"))

    if test:
        full_features_list = list(df.columns)
    else:
        full_features_list = list(df.drop(["label"]).columns)
        
    full_features_list = list(set(full_features_list))  # Ensure unique features

    # Replace infinities and NaNs
    n_nans = np.sum(df.select(pl.all().is_nan().sum()).to_numpy())
    n_infs = np.sum(df.select(pl.all().is_infinite().sum()).to_numpy())
    n_nulls = np.sum(df.select(pl.all().is_null().sum()).to_numpy())
    if (n_nans > 0) or (n_infs > 0) or (n_nulls > 0):
        raise ValueError(
            f"DataFrame contains NaNs: {n_nans}, Infs: {n_infs}, Nulls: {n_nulls}. Please clean the data."
        )

    assert "label" not in full_features_list, "Label should not be in features list"

    return df, full_features_list


def load_train_data(config, final=False):
    features = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]
    features += ["X" + str(i) for i in range(1, 781)]

    train_dtypes = {
        **{feature: pl.Float32 for feature in features},
        "label": pl.Float64,
        # "timestamp": pl.Datetime("ns", None),
    }
    train = pl.scan_parquet(
        config.TRAIN_DATA_PATH,
        low_memory=True,
        schema=train_dtypes,
        extra_columns="ignore",
        cast_options=pl.ScanCastOptions(float_cast="downcast"),
    ).collect()

    ## Below is used for notebook fast testing
    #!#!#!#!#!#!#!#!#!#!#!#!#!#!
    # train = train[:50000]
    #!#!#!#!#!#!#!#!#!#!#!#!#!#!

    # Filter bad rows and columns
    bad_cols = [
        col
        for col in train.columns
        if train.select(
            (pl.col(col).is_nan() | (pl.col(col).is_infinite()) | (pl.col(col).is_null())).any()
        ).item()
    ]
    if len(bad_cols) > 0:
        train = train.drop(bad_cols)
        # drop any rows with NaN, Inf, or Null values
    # Filter out rows containing null, NaN, or infinite
    mask = train.select(
        pl.any_horizontal([pl.all().is_null(), pl.all().is_nan(), pl.all().is_infinite()]).alias(
            "any_invalid"
        )
    )["any_invalid"]
    train = train.filter(~mask)

    # Add extra feature engineering features if specified
    all_features = train.drop("label").columns
    if config.ADD_EXTRA_FEATURES:
        initial_n_features = len(all_features)
        train, all_features = add_features_polars(train)
        print(
            f"→ Extra Features: Added {len(all_features) - initial_n_features} extra features, total: {len(all_features)} features"
        )
        assert "label" not in all_features, "Label should not be in features list"

    train_end_idx = int(len(train) * 0.6)
    if final:
        # X_train, y_train = train[all_features][-train_end_idx:], train["label"][-train_end_idx:]
        X_train, y_train = train[all_features][:], train["label"][:]
        X_val = None
        y_val = None    
    else:
        X_train, y_train = train[all_features][:train_end_idx], train["label"][:train_end_idx]
        X_val, y_val = train[all_features][train_end_idx:], train["label"][train_end_idx:]
    train = None  # Free memory

    # Filter outliers if specified
    if config.FILTER_OUTLIERS is not None:
        # get rows index in y_train that are above the 99 percentile and below the 1 percentile
        y_train = y_train.to_numpy()
        y_train_mask = (y_train > np.percentile(y_train, config.FILTER_OUTLIERS[0])) & (
            y_train < np.percentile(y_train, config.FILTER_OUTLIERS[1])
        )
        print(f"→ Filtering outliers: Deleted a total of {len(y_train) - np.sum(y_train_mask)} outliers")
        X_train = X_train.filter(y_train_mask)
        y_train = y_train[y_train_mask]

    # Apply scaling if specified
    if config.SCALER is not None:
        if config.SCALER == "standard":
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
        elif config.SCALER == "minmax":
            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler()
        elif config.SCALER == "robust":
            from sklearn.preprocessing import RobustScaler

            scaler = RobustScaler()
        columns = X_train.columns
        X_train = scaler.fit_transform(X_train)
        X_train = pl.DataFrame(X_train, schema=columns)
        if not final:
            X_val = scaler.transform(X_val)
            X_val = pl.DataFrame(X_val, schema=columns)
        print(f"→ Scaler: Applied {scaler}")

    if final:
        print(
            f"→ FINAL Data loaded: X_train patterns: {len(X_train)} patterns and {len(all_features)} features"
        )
    else:
        print(
            f"→ Data loaded: Final X_train|X_val patterns: {len(X_train)}|{len(X_val)} patterns and {len(all_features)} features"
        )

    if final:
        return X_train, y_train, X_val, y_val, all_features, scaler
    else:
        return X_train, y_train, X_val, y_val, all_features

def load_test_data(config, scaler):
    features = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]
    features += ["X" + str(i) for i in range(1, 781)]

    test_dtypes = {
        **{feature: pl.Float32 for feature in features},
    }
    test = pl.scan_parquet(
        "/kaggle/input/drw-crypto-market-prediction/test.parquet",
        low_memory=True,
        schema=test_dtypes,
        extra_columns="ignore",
        cast_options=pl.ScanCastOptions(float_cast="downcast"),
    ).collect()

    # Filter bad rows and columns
    bad_cols = [
        col
        for col in test.columns
        if test.select(
            (pl.col(col).is_nan() | (pl.col(col).is_infinite()) | (pl.col(col).is_null())).any()
        ).item()
    ]
    if len(bad_cols) > 0:
        test = test.drop(bad_cols)
        # drop any rows with NaN, Inf, or Null values
    # Filter out rows containing null, NaN, or infinite
    mask = test.select(
        pl.any_horizontal([pl.all().is_null(), pl.all().is_nan(), pl.all().is_infinite()]).alias(
            "any_invalid"
        )
    )["any_invalid"]
    test = test.filter(~mask)

    # Add extra feature engineering features if specified
    all_features = test.columns
    if config.ADD_EXTRA_FEATURES:
        initial_n_features = len(all_features)
        test, all_features = add_features_polars(test, test=True)
        print(
            f"→ Extra Features: Added {len(all_features) - initial_n_features} extra features, total: {len(all_features)} features"
        )
        assert "label" not in all_features, "Label should not be in features list"

        columns = test.columns
        test = scaler.fit_transform(test)
        test = pl.DataFrame(test, schema=columns)

    return test


def performance_report_regression(*, y_train, y_train_pred, y_val, y_val_pred, n_decimals=4):

    # convert arguments to numpy arrays if they are not already
    if not isinstance(y_train, np.ndarray):
        y_train = y_train.to_numpy()
    if not isinstance(y_train_pred, np.ndarray):
        y_train_pred = y_train_pred.to_numpy()
    if not isinstance(y_val, np.ndarray):
        y_val = y_val.to_numpy()
    if not isinstance(y_val_pred, np.ndarray):
        y_val_pred = y_val_pred.to_numpy()

    from sklearn.metrics import (
        r2_score,
        mean_absolute_percentage_error,
        mean_absolute_error,
    )
    from scipy.stats import pearsonr

    epsilon = 1e-10

    train_baseline_preds = np.repeat(y_train.mean(), len(y_train)) + np.random.normal(
        0, 0.00000000001, len(y_train)
    )
    train_pearsonr = pearsonr(y_train, y_train_pred)[0]
    train_r2 = r2_score(y_train, y_train_pred)
    train_baseline_mae = mean_absolute_error(y_train, train_baseline_preds)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mae_imp = (train_mae - train_baseline_mae) / (train_baseline_mae + epsilon)
    train_baseline_mape = mean_absolute_percentage_error(y_train, train_baseline_preds)
    train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
    train_mape_imp = (train_mape - train_baseline_mape) / (train_baseline_mape + epsilon)
    random_directional_accuracy_train = []
    for _ in range(1000):
        random_directional_accuracy_train.append(
            np.mean(np.sign(y_train) == np.sign(np.random.normal(0, 1, len(y_train))))
        )
    random_directional_accuracy_train = np.mean(random_directional_accuracy_train)
    train_directional_accuracy = np.mean(np.sign(y_train) == np.sign(y_train_pred))
    train_directional_accuracy_imp = (train_directional_accuracy - random_directional_accuracy_train) / (
        random_directional_accuracy_train + epsilon
    )

    val_baseline_preds = np.repeat(y_train.mean(), len(y_val)) + np.random.normal(
        0, 0.00000000001, len(y_val)
    )
    val_pearsonr = pearsonr(y_val, y_val_pred)[0]
    val_r2 = r2_score(y_val, y_val_pred)
    val_baseline_mae = mean_absolute_error(y_val, val_baseline_preds)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_mae_imp = (val_mae - val_baseline_mae) / (val_baseline_mae + epsilon)
    val_baseline_mape = mean_absolute_percentage_error(y_val, val_baseline_preds)
    val_mape = mean_absolute_percentage_error(y_val, y_val_pred)
    val_mape_imp = (val_mape - val_baseline_mape) / (val_baseline_mape + epsilon)
    random_directional_accuracy_val = []
    for _ in range(1000):
        random_directional_accuracy_val.append(
            np.mean(np.sign(y_val) == np.sign(np.random.normal(0, 1, len(y_val))))
        )
    random_directional_accuracy_val = np.mean(random_directional_accuracy_val)
    val_directional_accuracy = np.mean(np.sign(y_val) == np.sign(y_val_pred))
    val_directional_accuracy_imp = (val_directional_accuracy - random_directional_accuracy_val) / (
        random_directional_accuracy_val + epsilon
    )

    print(
        "\n=========================================================================================================="
    )
    print("Performance Report:")
    print("·····················")
    print(
        f"Train Pearson r: {train_pearsonr:.{n_decimals}f} →→ Validation Pearson r: {val_pearsonr:.{n_decimals}f}"
    )
    print(f"Train R2: {train_r2:.{n_decimals}f} →→ Validation R2: {val_r2:.{n_decimals}f}")
    print("···")
    print(
        f"Train Baseline MAE: \t{train_baseline_mae:.{n_decimals}f} →→ Validation Baseline MAE: \t{val_baseline_mae:.{n_decimals}f}"
    )
    print(f"Train MAE: \t\t{train_mae:.{n_decimals}f} →→ Validation MAE: \t\t{val_mae:.{n_decimals}f}")
    print(
        f"Train MAE improvement: {(train_mae_imp*100):.{n_decimals//2}f}% →→ Validation MAE improvement:\t{(val_mae_imp*100):.{n_decimals//2}f}%"
    )
    print("···")
    print(
        f"Train Baseline MAPE:  \t{train_baseline_mape:.{n_decimals}f} →→ Validation Baseline MAPE: \t{val_baseline_mape:.{n_decimals}f}"
    )
    print(f"Train MAPE:  \t\t{train_mape:.{n_decimals}f} →→ Validation MAPE: \t\t{val_mape:.{n_decimals}f}")
    print(
        f"Train MAPE improvement: {(train_mape_imp*100):.{n_decimals//2}f}% →→ Validation MAPE improvement:\t{(val_mape_imp*100):.{n_decimals//2}f}%"
    )
    print("···")
    print(
        f"Random Directional Accuracy Train: \t{random_directional_accuracy_train:.{n_decimals}f} →→ Random Directional Accuracy Validation: \t{random_directional_accuracy_val:.{n_decimals}f}"
    )
    print(
        f"Train Directional Accuracy: \t\t{train_directional_accuracy:.{n_decimals}f} →→ Validation Directional Accuracy: \t\t{val_directional_accuracy:.{n_decimals}f}"
    )
    print(
        f"Train Directional Accuracy improvement: {(train_directional_accuracy_imp*100):.{n_decimals//2}f}% →→ Validation Directional Accuracy improvement:\t{val_directional_accuracy_imp:.{n_decimals}f}%"
    )
    print("·····················")
    print(
        "=========================================================================================================="
    )


class EarlyStoppingPearsonCallback(xgb.callback.TrainingCallback):
    def __init__(self, dvalid, y_val, patience=10, maximize=True):
        self.dvalid = dvalid
        self.y_val = y_val
        self.patience = patience
        self.maximize = maximize
        self.best_score = None
        self.best_iteration = 0
        self.counter = 0
        self.stopped_iteration = None

    def before_training(self, model):
        self.start_time = time.time()
        return model

    def after_iteration(self, model, epoch, evals_log):
        y_pred = model.predict(self.dvalid, iteration_range=(0, epoch + 1))
        if len(np.unique(y_pred)) == 1:
            score = 0.0
        else:
            score, _ = pearsonr(self.y_val, y_pred)
        improved = (
            (score > self.best_score if self.best_score is not None else True)
            if self.maximize
            else (score < self.best_score if self.best_score is not None else True)
        )

        if improved:
            self.best_score = score
            self.best_iteration = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:  # and (epoch > self.patience * 2):
                print(
                    f"Early stopping at iteration {epoch + 1}. Best Pearson: {self.best_score:.4f} at iteration {self.best_iteration + 1}"
                )
                self.stopped_iteration = epoch
                model.set_attr(best_iteration=str(self.best_iteration))
                model.set_attr(best_score=str(self.best_score))
                print(f"Total training time: {time.time() - self.start_time:.1f}s")

                return True  # signal to stop training

        if (epoch + 1) % 50 == 0:
            print(
                f"  [{epoch + 1}] Pearson Correlation: {score:.4f} (Best: {self.best_score:.4f}). Elapsed: {time.time() - self.start_time:.1f}s"
            )
        return False

    def after_training(self, model):
        if self.stopped_iteration is not None:
            model.set_attr(best_iteration=str(self.best_iteration))
            model.set_attr(best_score=str(self.best_score))
        return model


class XGBEnsemble:
    def __init__(
        self,
        n_models,
        base_params,
        params,
        model_n_rounds,
        model_early_stopping_patience,
        models_save_dir=None,
    ):
        self.n_models = n_models
        self.base_params = base_params
        self.params = params
        self.model_n_rounds = model_n_rounds
        self.model_early_stopping_patience = model_early_stopping_patience
        self.models_save_dir = models_save_dir
        self.fit_status = "not_fitted"

        if not os.path.exists(self.models_save_dir):
            os.makedirs(self.models_save_dir, exist_ok=True)

    def fit(self, dtrain, dval, y_train, y_val):
        self.models = []
        self.best_n_epochs_per_model = []
        for seed in range(self.n_models):
            self.base_params["random_state"] = seed
            self.base_params["seed"] = seed
            print(f"\n\nTraining model {seed}|{self.n_models - 1} with params:\n {self.params}")
            model = xgb.train(
                params={**self.base_params, **self.params},
                dtrain=dtrain,
                num_boost_round=self.model_n_rounds,
                evals=[(dval, "valid")],
                callbacks=[
                    EarlyStoppingPearsonCallback(
                        dval,
                        y_val,
                        patience=self.model_early_stopping_patience,
                        maximize=True,
                    )
                ],
            )
            try:
                best_iteration = model.best_iteration
            except:
                best_iteration = self.model_n_rounds - 1

            self.best_n_epochs_per_model.append(best_iteration)
            self.models.append(model)

            # preds_train = model.predict(dtrain, iteration_range=(0, best_iteration + 1))
            # preds_val = model.predict(dval, iteration_range=(0, best_iteration + 1))

            # print(f"\n\nTrain finished. Printing performance report...")
            # print("Ensemble single model pearsonr in validation set:", pearsonr(y_val, preds_val)[0])

            if self.models_save_dir:
                model_path = os.path.join(self.models_save_dir, f"xgb_model_seed_{seed}.pkl")
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                print(f"Model {seed} saved to {model_path}")
        self.fit_status = "fitted"
        return self

    def fit_final(self, dtrain):
        if self.fit_status != "fitted":
            raise RuntimeError("Model must be fitted before calling fit_final.")

        self.models = []
        for seed in range(self.n_models):
            self.base_params["random_state"] = seed
            self.base_params["seed"] = seed
            print(
                f"\n Model {seed}: Training final model {seed}|{self.n_models} with {self.best_n_epochs_per_model[seed]} epochs and params:\n {self.params}"
            )
            model = xgb.train(
                params={**self.base_params, **self.params},
                dtrain=dtrain,
                num_boost_round=self.best_n_epochs_per_model[seed],
            )
            print(f"Model {seed}: Train finished.")
            self.models.append(model)

            if self.models_save_dir:
                model_path = os.path.join(self.models_save_dir, f"xgb_model_seed_{seed}.pkl")
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                print(f"Model {seed}: saved to disk at {model_path}")
        self.fit_status = "final_fitted"
        return self

    def load_models(self):
        self.models = []
        for seed in range(self.n_models):
            model_path = os.path.join(self.models_save_dir, f"xgb_model_seed_{seed}.pkl")
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                self.models.append(model)
            else:
                raise FileNotFoundError(f"Model file {model_path} does not exist.")
        return self

    def predict(self, dtest):
        if self.fit_status not in ["fitted", "final_fitted"]:
            raise RuntimeError("Model must be fitted before calling predict.")
        print(f"→ Model: Performing predictions with fit_status: {self.fit_status}")
        preds = []
        for seed, model in enumerate(self.models):
            preds.append(model.predict(dtest, iteration_range=(0, self.best_n_epochs_per_model[seed])))
        return sum(preds) / len(preds)


config_grid = ConfigGrid(
    add_extra_features=True,
    filter_outliers=[None],  # (1, 99)
    filter_features=[25],  # [75, 50, 25, 5, None]
    filter_features_model_n_rounds=[150],
    scaler=["robust"],  # ["standard", "robust"],
    n_models=10, #[10, 20],
    eastop_patience=250,
    model_n_rounds=2500,
    base_params={
        "tree_method": "hist",
        "device": "cpu",
        "verbosity": 0,
        "n_jobs": -1,
        "disable_default_eval_metric": True,
    },
    params={
        "subsample": 0.025,
        "colsample_bytree": 0.5,
        "colsample_bylevel": 0.4,
        "colsample_bynode": 0.35,
        "gamma": 2.0,
        "learning_rate": 0.075,
        "max_leaves": 8,
        "min_child_weight": 16,
        "reg_alpha": 60.0,
        "reg_lambda": 50.0,
    },
    train_data_path="/kaggle/input/drw-crypto-market-prediction/train.parquet",
    models_save_dir=False,
    # models_save_dir="/home/rayllon/_inception/Kaggle/DRW_Crypto/xgboost/ensemble_models",
)

# class TrainedConfig:
#     def __init__(self, model, features, scaler):
#         self.MODEL = model
#         self.FEATURES = features
#         self.SCALER = scaler


configs_with_val_pearsonr = {}

best_config = None
best_config_model = None
best_config_pearsonr = None

for config in config_grid.stream_configs():
    config_id = config.get_config_id()
    print(f"\nRunning config: {config_id}")
    
    X_train, y_train, X_val, y_val, features = load_train_data(config)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)

    # Feature selection
    if config.FILTER_FEATURES is not None:
        print(
            f"→ Filtering features: Filter those with importance above {config.FILTER_FEATURES}% based on avg. importance of 3 xgb models with {config.FILTER_FEATURES_MODEL_N_ROUNDS} epochs each"
        )
        feature_importances = pd.DataFrame(index=dtrain.feature_names)
        for seed in range(50, 53, 1):  # ensure seeds are not 0, 1, 2, 3,... as in the ensemble
            params_random_seed = {
                "random_state": seed,
                "seed": seed,
            }
            model_base = xgb.train(
                params={**config.BASE_PARAMS, **config.PARAMS, **params_random_seed},
                dtrain=dtrain,
                num_boost_round=config.FILTER_FEATURES_MODEL_N_ROUNDS,
            )
            model_feature_importances = model_base.get_score(importance_type="total_gain")
            feature_importances = pd.concat(
                [
                    feature_importances,
                    pd.DataFrame(
                        index=model_feature_importances.keys(),
                        data=model_feature_importances.values(),
                        columns=[f"importance_{seed}"],
                    ),
                ],
                axis=1,
                ignore_index=False,
            )
        feature_importances = feature_importances.fillna(0.0)
        feature_importances["importance_avg"] = feature_importances.filter(like="importance_").mean(axis=1)
        print("Threshold percentile:", np.percentile(feature_importances["importance_avg"], config.FILTER_FEATURES))
        feature_importances = feature_importances[
            feature_importances["importance_avg"]
            > np.percentile(feature_importances["importance_avg"], config.FILTER_FEATURES)
        ]
        features = feature_importances.index.tolist()
        config._FEATURES = features
        X_train = X_train[:, features]
        X_val = X_val[:, features]
        print(f"→ Filtering features: Keeping {len(feature_importances)} features based on avg. importance")
        print(
            f"→ Filtering features: Best selected importance: {feature_importances['importance_avg'].max():.4f}"
        )
        print(
            f"→ Filtering features: Worst selected importance: {feature_importances['importance_avg'].min():.4f}"
        )
        print(f"→ Filtering features: Final X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

    # cast all float64 columns to float32
    X_train = X_train.with_columns(
        [
            pl.col(col).cast(pl.Float32)
            for col, dtype in zip(X_train.columns, X_train.dtypes)
            if dtype == pl.Float64
        ]
    )
    X_val = X_val.with_columns(
        [
            pl.col(col).cast(pl.Float32)
            for col, dtype in zip(X_val.columns, X_val.dtypes)
            if dtype == pl.Float64
        ]
    )

    # Create DMatrix for training and validation sets
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

    # Initialize the ensemble model with the specified configuration
    model = XGBEnsemble(
        n_models=config.N_MODELS,
        base_params=config.BASE_PARAMS,
        params=config.PARAMS,
        model_n_rounds=config.MODEL_N_ROUNDS,
        model_early_stopping_patience=config.EASTOP_PATIENCE,
        models_save_dir=config.MODELS_SAVE_DIR,
    )

    model = model.fit(dtrain=dtrain, dval=dval, y_train=y_train, y_val=y_val)

    try:
        best_iteration = model.best_iteration
        print(f"→ Model: Tran finished with early stopping, best iteration: {best_iteration}")
    except:
        best_iteration = config.MODEL_N_ROUNDS - 1
        print("→ Model: Tran finished with NO early stopping, using last iteration as best.")

    preds_train = model.predict(dtrain)
    preds_val = model.predict(dval)

    print(f"\n\n→ Model: Ensemble train finished. Printing final XGBEnsemble performance report...")
    print(
        f"→ Model: Ensemble Train predictions mean | std: {preds_train.mean():.4f} | {preds_train.std():.4f}"
    )
    for percentile in [5, 25, 50, 75, 95]:
        print(
            f"→ Model: Ensemble Train predictions {percentile} percentile: {np.percentile(preds_train, percentile):.4f} | {percentile}%"
        )
    print(f"→ Model: Ensemble Val predictions mean | std: {preds_val.mean():.4f} | {preds_val.std():.4f}")
    for percentile in [5, 25, 50, 75, 95]:
        print(
            f"→ Model: Ensemble Val predictions {percentile} percentile: {np.percentile(preds_val, percentile):.4f} | {percentile}%"
        )

    performance_report_regression(
        y_val=y_val,
        y_val_pred=preds_val,
        y_train=y_train,
        y_train_pred=preds_train,
        n_decimals=4,
    )

    config_pearsonr = pearsonr(y_val, preds_val)[0]

    if (best_config_pearsonr is None) or (config_pearsonr < best_config_pearsonr):
        best_config = config
        best_config_model = model
        best_config_pearsonr = config_pearsonr
        
    configs_with_val_pearsonr[config_id] = config_pearsonr

    print("→ → → Configuration finished!")


best_config_id = sorted(configs_with_val_pearsonr, reverse=True)[0]
print("Best configuration id is:\n→", best_config_id)
print(".. with a pearsonr in the validation set equal to", round(configs_with_val_pearsonr[best_config_id], 5))


config_id = best_config.get_config_id()
print(f"\nRunning best config: {config_id}")

X_train, y_train, X_val, y_val, features, scaler = load_train_data(best_config, final=True)
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)

# Feature selection
if best_config.FILTER_FEATURES is not None:
    X_train = X_train[:, best_config._FEATURES]
    print(f"→ Filtering features: Final X_train shape: {X_train.shape}")

# cast all float64 columns to float32
X_train = X_train.with_columns(
    [
        pl.col(col).cast(pl.Float32)
        for col, dtype in zip(X_train.columns, X_train.dtypes)
        if dtype == pl.Float64
    ]
)

# Create DMatrix for training and validation sets
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=best_config._FEATURES)

model = model.fit_final(dtrain=dtrain)


X_test = load_test_data(
    config=best_config,
    scaler=scaler
)
X_test = X_test[:, best_config._FEATURES]
dtest = xgb.DMatrix(X_test, feature_names=best_config._FEATURES)
test_preds = model.predict(dtest)


submission = pd.DataFrame(test_preds, columns=["prediction"])
submission.index.name = "ID"
submission = submission.reset_index()
submission["ID"] = submission["ID"] + 1
display(submission)
submission.to_csv("submission.csv", index=False)

