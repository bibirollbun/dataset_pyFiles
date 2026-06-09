import warnings
warnings.filterwarnings("ignore")


# ========= imports =========

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


# Load the data
train_data = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv")

print(f"Training data shape: {train_data.shape}")


# Convert 'month' column to integers, with 1-67 being the training set and 68-79 being the test set

def month_str_to_time(month_str):
    """Convert month string to time integer - EXACT from improved_solution.py"""
    if '-' in month_str:
        year, month = month_str.split('-')
        year = int(year)
        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
    else:
        parts = month_str.split()
        year = int(parts[0])
        month_name = parts[1]
        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }

    month_num = month_map[month_name if '-' not in month_str else month]
    time_int = (year - 2019) * 12 + month_num
    return time_int


# Apply to the dataframe
train_data['time'] = train_data['month'].apply(month_str_to_time)
print(f"Time range: {train_data['time'].min()} to {train_data['time'].max()}")
print(f"Training set: 1-67, Test set: 68-79")


# Build training matrix: amount_new_house_transactions [time x sector_id]
def create_training_matrix(df):
    """
    Convert dataframe to matrix format: [time x sector_id]
    This is much more efficient for calculations
    """
    print("ï¿½ï¿½ï¸� Creating training matrix...")
    
    # Create pivot table: time as rows, sector as columns
    matrix = df.pivot_table(
        index='time', 
        columns='sector', 
        values='amount_new_house_transactions',
        fill_value=0  # Fill missing values with 0
    )
    
    # Ensure all sectors are present (1-96)
    expected_sectors = [f'sector {i}' for i in range(1, 97)]
    for sector in expected_sectors:
        if sector not in matrix.columns:
            matrix[sector] = 0
    
    # Sort columns by sector number
    matrix = matrix.reindex(sorted(matrix.columns, key=lambda x: int(x.split()[-1])), axis=1)
    
    print(f"Matrix shape: {matrix.shape}")
    print(f"Time range: {matrix.index.min()} to {matrix.index.max()}")
    print(f"Sectors: {matrix.columns[0]} to {matrix.columns[-1]}")
    
    return matrix

# Create the matrix
train_matrix = create_training_matrix(train_data)


train_matrix.head(12)


# ========= metrics =========
def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask): return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    eps = np.finfo(np.float64).eps
    return np.mean(2.0 * np.abs(y_true - y_pred) / np.maximum(denom, eps)) * 100

def custom_score(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0: raise ValueError("empty array")
    if (y_true < 0).any(): raise ValueError("negative y-true")
    if (~np.isfinite(y_pred)).any(): raise ValueError("infinite y-pred")

    ape = np.abs(y_true - y_pred) / np.maximum(y_true, eps)
    good_mask = ape <= 1.0
    good_rate = float(good_mask.mean())
    if good_rate < 0.7:
        return {'score': 0.0, 'good_rate': good_rate, 'str': f"score=0.000  good_rate={good_rate:.3f}"}
    mape_over_good = float(np.mean(ape[good_mask]))
    scaled_mape = mape_over_good / max(good_rate, eps)
    score = 1.0 - scaled_mape
    return {'score': float(score), 'good_rate': good_rate, 'str': f"score={score:.3f}  good_rate={good_rate:.3f}"}


# # ========= feature engineering =========
# def create_features_at_time(train_matrix: pd.DataFrame, time_idx: int, max_history=18): # For Zero-Classifier
#     """
#     Create features for all sectors at a specific time t (using history <= t).
#     One row per sector that has at least `max_history` months available by t.

#     Adds volatility features inspired by ARCH/GARCH:
#       - level volatility: std over raw levels (3/6/12)
#       - return volatility: std of log1p-returns (3/6/12)
#       - EWMA conditional variance proxy of returns (alpha=0.3)
#       - coefficient of variation on positive levels (3/6/12)
#       - volatility ratios (short/long)
#       - market-level volatility analogs
#     """
#     df_panel = train_matrix.apply(pd.to_numeric, errors='coerce').sort_index()
#     df_panel.index.name = 'time'

#     total = df_panel.sum(axis=1)
#     total_roll3 = total.rolling(3, min_periods=1).mean().shift(1)   # up to t-1
#     total_roll6 = total.rolling(6, min_periods=1).mean().shift(1)

#     # --- helpers
#     def _log1p_returns(x: pd.Series) -> np.ndarray:
#         x = pd.to_numeric(x, errors='coerce').fillna(0.0).astype(float).values
#         if x.size < 2:
#             return np.array([], dtype=float)
#         lx = np.log1p(np.maximum(x, 0.0))
#         return np.diff(lx)

#     def _ewma_cond_var(ret: np.ndarray, alpha=0.3):
#         # simple GARCH(1,1)-like proxy: v_t = (1-alpha)*v_{t-1} + alpha*r_{t-1}^2
#         if ret.size == 0:
#             return 0.0
#         v = 0.0
#         for r in ret:
#             v = (1.0 - alpha) * v + alpha * (r * r)
#         return float(v)

#     features_list, sector_list = [], []

#     for sector in df_panel.columns:
#         series = df_panel[sector].dropna()
#         # history available up to and including time t
#         hist = series[series.index <= time_idx]
#         if len(hist) < max_history:
#             continue

#         f = {}
#         # ID
#         f['sector_id'] = int(str(sector).replace('sector ', '')) / 100.0

#         # calendar at time t
#         month_t = ((time_idx - 1) % 12) + 1
#         f['month_sin'] = np.sin(2*np.pi*month_t/12)
#         f['month_cos'] = np.cos(2*np.pi*month_t/12)
#         f['quarter'] = ((month_t - 1) // 3 + 1) / 4.0

#         # lags at t (values from <= t)
#         for lag in [1, 2, 3, 6, 12]:
#             if len(hist) >= lag:
#                 val = float(hist.iloc[-lag])
#                 f[f'lag_{lag}_log1p'] = np.log1p(val)
#                 f[f'lag_{lag}_zero'] = int(val == 0)
#             else:
#                 f[f'lag_{lag}_log1p'] = 0.0
#                 f[f'lag_{lag}_zero'] = 1

#         # rolling â€“ level stats
#         for w in [3, 6, 12]:
#             if len(hist) >= w:
#                 roll = hist.tail(w)
#                 f[f'roll_{w}_mean_log1p']   = np.log1p(roll.mean())
#                 f[f'roll_{w}_median_log1p'] = np.log1p(roll.median())
#                 f[f'roll_{w}_zero_rate']    = float((roll == 0).mean())
#                 f[f'roll_{w}_trend']        = np.tanh(np.polyfit(range(w), roll.values, 1)[0]) if roll.std() > 0 else 0.0
#                 # NEW: level volatility
#                 f[f'roll_{w}_std_level']    = float(roll.std())
#             else:
#                 f[f'roll_{w}_mean_log1p']   = 0.0
#                 f[f'roll_{w}_median_log1p'] = 0.0
#                 f[f'roll_{w}_zero_rate']    = 1.0
#                 f[f'roll_{w}_trend']        = 0.0
#                 f[f'roll_{w}_std_level']    = 0.0

#         # sector stats
#         f['sector_zero_rate'] = float((hist == 0).mean())
#         f['sector_activity_consistency'] = 1.0 - f['sector_zero_rate']

#         nz = hist[hist > 0]
#         if len(nz) > 2:
#             mean_pos = nz.mean()
#             std_pos  = nz.std()
#             f['nonzero_mean_log1p'] = np.log1p(mean_pos)
#             f['nonzero_cv']         = np.tanh(std_pos / (mean_pos + 1e-6))
#         else:
#             f['nonzero_mean_log1p'] = 0.0
#             f['nonzero_cv']         = 0.0

#         # --- NEW: return-based volatility features
#         ret = _log1p_returns(hist)
#         for w in [3, 6, 12]:
#             if ret.size >= w:
#                 f[f'ret_std_{w}'] = float(pd.Series(ret).tail(w).std())
#             else:
#                 f[f'ret_std_{w}'] = 0.0

#         # EWMA conditional variance proxy (ARCH-ish)
#         f['ret_cond_var_ewma'] = _ewma_cond_var(ret, alpha=0.3)

#         # CV of levels in short/long windows (positives only), plus ratios
#         def _cv(x: pd.Series):
#             x = x[x > 0]
#             if len(x) < 2 or x.mean() == 0:
#                 return 0.0
#             return float(x.std() / (x.mean() + 1e-6))

#         cv6  = _cv(hist.tail(6))
#         cv12 = _cv(hist.tail(12))
#         f['cv_pos_6']  = np.tanh(cv6)
#         f['cv_pos_12'] = np.tanh(cv12)
#         f['cv_pos_ratio_6_12'] = (cv6 / (cv12 + 1e-6)) if cv12 > 0 else 0.0

#         # short/long volatility ratios (returns & levels)
#         lvl_std6  = f['roll_6_std_level']
#         lvl_std12 = f['roll_12_std_level']
#         f['vol_ratio_level_6_12'] = (lvl_std6 / (lvl_std12 + 1e-6)) if lvl_std12 > 0 else 0.0

#         ret_std6  = f['ret_std_6']
#         ret_std12 = f['ret_std_12']
#         f['vol_ratio_ret_6_12'] = (ret_std6 / (ret_std12 + 1e-6)) if ret_std12 > 0 else 0.0

#         # market context at time t (from shifted rolls)
#         mkt3 = float(total_roll3.loc[time_idx]) if time_idx in total_roll3.index and pd.notna(total_roll3.loc[time_idx]) else 0.0
#         mkt6 = float(total_roll6.loc[time_idx]) if time_idx in total_roll6.index and pd.notna(total_roll6.loc[time_idx]) else 0.0
#         f['market_activity_log1p'] = np.log1p(mkt3)
#         f['market_trend']          = np.tanh((mkt3 - mkt6) / (mkt6 + 1e-6))

#         # NEW: market return volatility proxies
#         mkt_hist = total[total.index <= time_idx]
#         mkt_ret = _log1p_returns(mkt_hist)
#         if mkt_ret.size >= 6:
#             f['market_ret_std_6'] = float(pd.Series(mkt_ret).tail(6).std())
#         else:
#             f['market_ret_std_6'] = 0.0
#         if mkt_ret.size >= 12:
#             f['market_ret_std_12'] = float(pd.Series(mkt_ret).tail(12).std())
#         else:
#             f['market_ret_std_12'] = 0.0
#         f['market_ret_cond_var_ewma'] = _ewma_cond_var(mkt_ret, alpha=0.3)

#         # zero streak (recent)
#         cz = 0
#         for v in hist.tail(6).iloc[::-1]:
#             if v == 0: cz += 1
#             else: break
#         f['consecutive_zeros_norm'] = min(cz / 6.0, 1.0)

#         # interaction
#         f['seasonal_zero_interaction'] = f['month_sin'] * f['sector_zero_rate']

#         features_list.append(f)
#         sector_list.append(sector)

#     features_df = pd.DataFrame(features_list)
#     features_df['sector'] = sector_list
#     features_df['time'] = int(time_idx)
#     return features_df

# def create_training_data_for_horizon(train_matrix: pd.DataFrame, horizon: int, max_history=18, max_train_time=None):
#     """
#     Build supervised table for a given horizon h: X at time t â†’ y at t+h.
#     Rows: (sector, t) with enough history by t; target is panel[t+h, sector].
#     """
#     print(f"Creating training data for horizon={horizon} ...")
#     df_panel = train_matrix.apply(pd.to_numeric, errors='coerce').sort_index()

#     # valid feature times t
#     min_t = int(df_panel.index.min() + max_history)
#     max_t_possible = int(df_panel.index.max() - horizon)
#     max_t = int(min(max_t_possible, max_train_time)) if max_train_time is not None else max_t_possible
#     if max_t < min_t:
#         raise ValueError(f"No valid times for horizon {horizon}. Increase data or reduce horizon.")
#     valid_times = [int(t) for t in df_panel.index if min_t <= t <= max_t]

#     frames = []
#     for t in valid_times:
#         feat_t = create_features_at_time(df_panel, t, max_history=max_history)
#         if feat_t.empty:
#             continue
#         tgt_time = t + horizon
#         # horizon/meta features
#         target_month = ((tgt_time - 1) % 12) + 1
#         feat_t['target_month_sin'] = np.sin(2*np.pi*target_month/12)
#         feat_t['target_month_cos'] = np.cos(2*np.pi*target_month/12)
#         feat_t['target_quarter']   = ((target_month - 1) // 3 + 1) / 4.0
#         feat_t['horizon'] = horizon / 12.0
#         feat_t['target_time'] = int(tgt_time)

#         # attach targets aligned by sector
#         feat_t['target'] = df_panel.loc[tgt_time, feat_t['sector']].values.astype(float)
#         frames.append(feat_t)

#     result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
#     if result.empty:
#         print(f"âš ï¸� Horizon {horizon}: produced 0 samples.")
#         return result

#     print(f"  â†’ {len(result)} rows, {result.filter(regex='^(?!sector$|time$|target$|target_time$)').shape[1]} features")
#     print(f"  zero-rate={np.mean(result['target'].values==0):.1%}  mean={result['target'].mean():.2f}")
#     return result


# ========= Horizon-specific model =========
class HorizonSpecificModel:
    """
    Logistic (zero) + optional isotonic + Ridge(log1p) for positives, trained per horizon.
    """
    def __init__(self, horizon, ridge_alpha=2.0, logistic_C=0.5, use_isotonic=True, verbose=True):
        self.horizon = int(horizon)
        self.ridge_alpha = float(ridge_alpha)
        self.logistic_C  = float(logistic_C)
        self.use_isotonic = bool(use_isotonic)
        self.verbose = bool(verbose)

        self.scaler = None
        self.zero_model = None
        self.count_model = None
        self.iso_ = None
        self.zero_threshold_ = 0.5
        self.mu_cap_ = 1e12
        self.is_fitted = False

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if self.verbose:
            print(f"\nğŸ”§ Train Horizon H={self.horizon}  X={X.shape}  zero-rate={np.mean(y==0):.1%}")

        self.scaler = RobustScaler()
        Xs = self.scaler.fit_transform(X)

        # zero model
        self.zero_model = LogisticRegression(
            C=self.logistic_C, class_weight='balanced', max_iter=500, random_state=42
        )
        self.zero_model.fit(Xs, (y == 0).astype(int))

        # isotonic
        if self.use_isotonic:
            pi_raw = self.zero_model.predict_proba(Xs)[:, 1]
            self.iso_ = IsotonicRegression(out_of_bounds='clip').fit(pi_raw, (y == 0).astype(int))
        else:
            self.iso_ = None

        # count model on positives
        mask_pos = y > 0
        if mask_pos.sum() > 10:
            y_log = np.log1p(y[mask_pos])
            self.count_model = Ridge(alpha=self.ridge_alpha, solver='svd')
            self.count_model.fit(Xs[mask_pos], y_log)
            self.mu_cap_ = float(np.quantile(y[mask_pos], 0.995))
        else:
            self.count_model = None
            self.mu_cap_ = 1e12

        # threshold tuning
        self._calibrate_threshold(Xs, y)
        self.is_fitted = True
        return self

    def _calibrate_threshold(self, Xs, y):
        pi_raw = self.zero_model.predict_proba(Xs)[:, 1]
        p0 = self.iso_.transform(pi_raw) if self.iso_ is not None else pi_raw
        if self.count_model is not None:
            mu = np.clip(np.expm1(self.count_model.predict(Xs)), 0.0, self.mu_cap_)
        else:
            mu = np.zeros(len(Xs), dtype=float)
        best_thr, best_sc = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 19):
            yhat = (1.0 - p0) * mu
            yhat = yhat.copy()
            yhat[p0 >= t] = 0.0
            sc = custom_score(y, yhat)['score']
            if sc > best_sc:
                best_sc, best_thr = sc, float(t)
        self.zero_threshold_ = best_thr
        if self.verbose:
            print(f"  tuned thr={self.zero_threshold_:.2f}  score={best_sc:.3f}")

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        Xs = self.scaler.transform(np.asarray(X, dtype=float))
        pi_raw = self.zero_model.predict_proba(Xs)[:, 1]
        p0 = self.iso_.transform(pi_raw) if self.iso_ is not None else pi_raw
        if self.count_model is not None:
            mu = np.clip(np.expm1(self.count_model.predict(Xs)), 0.0, self.mu_cap_)
        else:
            mu = np.zeros(len(Xs), dtype=float)
        yhat = (1.0 - p0) * mu
        yhat[p0 >= self.zero_threshold_] = 0.0
        return np.maximum(yhat, 0.0)

    def predict_soft(self, X):
        """
        Predict without hard zeroing: return (1 - p0) * mu for every row.
        (Uses the same logistic p0 and Ridge mu as .predict, but DOES NOT
        set entries to 0 when p0 >= threshold.)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        Xs = self.scaler.transform(np.asarray(X, dtype=float))

        # zero probability (calibrated if available)
        pi_raw = self.zero_model.predict_proba(Xs)[:, 1]
        p0 = self.iso_.transform(pi_raw) if self.iso_ is not None else pi_raw

        # positive component
        if self.count_model is not None:
            mu = np.clip(np.expm1(self.count_model.predict(Xs)), 0.0, self.mu_cap_)
        else:
            mu = np.zeros(len(Xs), dtype=float)

        # soft discount only, no hard cut
        yhat = (1.0 - p0) * mu
        return np.maximum(yhat, 0.0)



# ========= Multi-horizon trainer/predictor =========
class MultiHorizonForecaster:
    def __init__(self, horizons=12, ridge_alpha=2.0, logistic_C=0.5, use_isotonic=True, verbose=True, max_history=18):
        self.horizons = list(range(1, horizons + 1))
        self.ridge_alpha = float(ridge_alpha)
        self.logistic_C  = float(logistic_C)
        self.use_isotonic = bool(use_isotonic)
        self.verbose = bool(verbose)
        self.max_history = int(max_history)

        self.models = {}
        self.feature_cols = None

    def fit(self, train_matrix: pd.DataFrame):
        print(f"\nğŸš€ Training {len(self.horizons)} horizon-specific models ...")
        max_train_time = int(train_matrix.index.max())

        # Build training tables per horizon
        tables = {}
        for h in self.horizons:
            tbl = create_training_data_for_horizon(train_matrix, h, max_history=self.max_history, max_train_time=max_train_time)
            if tbl.empty:
                raise ValueError(f"Horizon {h} produced no samples; reduce max_history or check data.")
            tables[h] = tbl

        # Use columns from H=1 as canonical feature set
        self.feature_cols = [c for c in tables[1].columns if c not in ['sector','time','target','target_time']]
        if self.verbose:
            print(f"Feature count: {len(self.feature_cols)}  Samples/H: {len(tables[1])}")

        # Train each horizon model
        for h in self.horizons:
            dfh = tables[h]
            X = dfh[self.feature_cols].astype(float).values
            y = dfh['target'].astype(float).values
            model = HorizonSpecificModel(horizon=h, ridge_alpha=self.ridge_alpha,
                                         logistic_C=self.logistic_C, use_isotonic=self.use_isotonic,
                                         verbose=self.verbose)
            model.fit(X, y)
            self.models[h] = model

        print(f"âœ… Trained {len(self.models)} models.")
        return self

    def _features_for_start(self, train_matrix, start_time):
        feats = create_features_at_time(train_matrix, start_time, max_history=self.max_history)
        if feats.empty:
            raise ValueError(f"No features at time {start_time} (insufficient history).")
        return feats

    def predict(self, train_matrix: pd.DataFrame, start_time: int):
        """
        Predict at horizons from start_time: columns are target months (start_time + h).
        Rows = sectors (only those with sufficient history by start_time).
        """
        if not self.models:
            raise ValueError("Fit the forecaster first.")
        feats_t = self._features_for_start(train_matrix, start_time)
        sectors = feats_t['sector'].values

        preds = {int(start_time + h): None for h in self.horizons}
        for h in self.horizons:
            tgt_time = int(start_time + h)
            dfh = feats_t.copy()
            tgt_month = ((tgt_time - 1) % 12) + 1
            dfh['target_month_sin'] = np.sin(2*np.pi*tgt_month/12)
            dfh['target_month_cos'] = np.cos(2*np.pi*tgt_month/12)
            dfh['target_quarter']   = ((tgt_month - 1) // 3 + 1) / 4.0
            dfh['horizon'] = h / 12.0

            # ensure all features exist, in order
            for c in self.feature_cols:
                if c not in dfh.columns:
                    dfh[c] = 0.0
            X = dfh[self.feature_cols].astype(float).values
            preds[tgt_time] = self.models[h].predict(X)

        # assemble (rows sectors, cols target months)
        out = pd.DataFrame(index=sectors, data={t: preds[t] for t in sorted(preds)}, dtype=float)
        out.index.name = 'sector'
        return out


# # ========= CV with per-horizon tracking =========
# def chronological_cv_with_horizon_tracking(train_matrix, split_configs,
#                                            ridge_alpha=2.0, logistic_C=0.5,
#                                            use_isotonic=True, max_history=18):
#     """
#     split_configs: list of (train_end, test_start, test_end)
#     e.g., [(43, 44, 55), (55, 56, 67)]
#     """
#     rows = []
#     for sid, (train_end, test_start, test_end) in enumerate(split_configs, 1):
#         print("\n" + "="*80)
#         print(f"SPLIT {sid}: Train 1..{train_end}  |  Test {test_start}..{test_end}")
#         print("="*80)

#         panel_tr = train_matrix[train_matrix.index <= train_end]
#         n_h = int(test_end - test_start + 1)

#         forecaster = MultiHorizonForecaster(horizons=n_h, ridge_alpha=ridge_alpha,
#                                             logistic_C=logistic_C, use_isotonic=use_isotonic,
#                                             verbose=False, max_history=max_history)
#         forecaster.fit(panel_tr)
#         preds = forecaster.predict(panel_tr, start_time=train_end)   # rows: sectors with history

#         # Eval per horizon: align by sector intersection
#         print("\nğŸ“Š Per-horizon performance")
#         print(f"{'H':>2} {'Month':>5} {'Score':>7} {'Good%':>7} {'SMAPE%':>8} {'MAE':>8} {'ZeroPred%':>9} {'ZeroAct%':>9}")
#         for h in range(1, n_h+1):
#             tgt_time = train_end + h
#             if tgt_time not in preds.columns: continue
#             # align sectors
#             common = train_matrix.columns.intersection(preds.index)
#             y_true = train_matrix.loc[tgt_time, common].astype(float).values
#             y_pred = preds.loc[common, tgt_time].astype(float).values

#             comp = custom_score(y_true, y_pred)
#             smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
#             mae = mean_absolute_error(y_true, y_pred)
#             zpred = float((y_pred == 0).mean())
#             zact  = float((y_true == 0).mean())

#             print(f"{h:>2} {tgt_time:>5} {comp['score']:>7.3f} {comp['good_rate']*100:>6.1f}% {smape:>8.1f} {mae:>8.2f} {zpred*100:>8.1f}% {zact*100:>8.1f}%")
#             rows.append({
#                 'split': sid, 'train_end': train_end, 'horizon': h, 'target_month': tgt_time,
#                 'custom_score': comp['score'], 'good_rate': comp['good_rate'],
#                 'smape': smape, 'mae': mae, 'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
#                 'zero_rate_pred': zpred, 'zero_rate_actual': zact
#             })

#         # overall on the tested horizon window (stack aligned sectors across months)
#         all_true, all_pred = [], []
#         for h in range(1, n_h+1):
#             tgt_time = train_end + h
#             if tgt_time not in preds.columns: continue
#             common = train_matrix.columns.intersection(preds.index)
#             all_true.append(train_matrix.loc[tgt_time, common].astype(float).values)
#             all_pred.append(preds.loc[common, tgt_time].astype(float).values)
#         if all_true:
#             all_true = np.concatenate(all_true)
#             all_pred = np.concatenate(all_pred)
#             comp = custom_score(all_true, all_pred)
#             smape = symmetric_mean_absolute_percentage_error(all_true, all_pred)
#             print("\n" + "-"*80)
#             print(f"OVERALL SPLIT {sid}:  score={comp['score']:.3f}  good={comp['good_rate']*100:.1f}%  SMAPE={smape:.1f}%")

#     results = pd.DataFrame(rows)
#     if not results.empty:
#         print("\n" + "="*80)
#         print("PERFORMANCE BY HORIZON (meanÂ±std across splits)")
#         print("="*80)
#         summ = results.groupby('horizon').agg(
#             custom_score_mean=('custom_score','mean'),
#             custom_score_std=('custom_score','std'),
#             good_rate_mean=('good_rate','mean'),
#             good_rate_std=('good_rate','std'),
#             smape_mean=('smape','mean'),
#             smape_std=('smape','std'),
#             mae_mean=('mae','mean'),
#             mae_std=('mae','std'),
#         ).round(3)
#         print(summ)
#     return results


# # ========= Final prediction (train 1..67, predict 68..79) =========
# def generate_final_predictions(train_matrix, horizons=12, ridge_alpha=2.0, logistic_C=0.5,
#                                use_isotonic=True, max_history=18):
#     last_train = int(train_matrix.index.max())
#     print("\n" + "="*80)
#     print(f"FINAL: train 1..{last_train}  â†’  predict {last_train+1}..{last_train+horizons}")
#     print("="*80)

#     forecaster = MultiHorizonForecaster(horizons=horizons, ridge_alpha=ridge_alpha,
#                                         logistic_C=logistic_C, use_isotonic=use_isotonic,
#                                         verbose=True, max_history=max_history)
#     forecaster.fit(train_matrix)
#     preds = forecaster.predict(train_matrix, start_time=last_train)

#     print(f"Pred shape: {preds.shape}  months={list(preds.columns)}  zero-rate={(preds.values==0).mean():.1%}")
#     return preds


# --- single-anchor features (one row per sector, built at anchor_time) ---
def build_anchor_features(train_matrix: pd.DataFrame, anchor_time: int, max_history: int = 18) -> pd.DataFrame:
    df_panel = train_matrix.apply(pd.to_numeric, errors='coerce').sort_index()
    assert anchor_time in df_panel.index, "anchor_time must be inside train_matrix index"

    total = df_panel.sum(axis=1)
    total_roll3 = total.rolling(3, min_periods=1).mean().shift(1)
    total_roll6 = total.rolling(6, min_periods=1).mean().shift(1)

    def _log1p_returns(x: pd.Series) -> np.ndarray:
        x = pd.to_numeric(x, errors='coerce').fillna(0.0).astype(float).values
        if x.size < 2:
            return np.array([], dtype=float)
        lx = np.log1p(np.maximum(x, 0.0))
        return np.diff(lx)

    def _ewma_cond_var(ret: np.ndarray, alpha=0.3):
        if ret.size == 0:
            return 0.0
        v = 0.0
        for r in ret:
            v = (1.0 - alpha) * v + alpha * (r * r)
        return float(v)

    rows, sectors = [], []
    for s in df_panel.columns:
        hist = df_panel[s].loc[:anchor_time].dropna()
        if len(hist) < max_history:
            continue

        f = {}
        # id / calendar at anchor_time
        f['sector_id'] = int(str(s).replace('sector ', '')) / 100.0
        m = ((anchor_time - 1) % 12) + 1
        f['month_sin'] = np.sin(2*np.pi*m/12)
        f['month_cos'] = np.cos(2*np.pi*m/12)
        f['quarter']  = ((m - 1)//3 + 1) / 4.0

        # lags
        for lag in [1,2,3,6,12]:
            if len(hist) >= lag:
                v = float(hist.iloc[-lag])
                f[f'lag_{lag}_log1p'] = np.log1p(v)
                f[f'lag_{lag}_zero']  = int(v == 0)
            else:
                f[f'lag_{lag}_log1p'] = 0.0
                f[f'lag_{lag}_zero']  = 1

        # rolling windows ending at anchor_time + volatility
        for w in [3,6,12]:
            if len(hist) >= w:
                roll = hist.tail(w)
                f[f'roll_{w}_mean_log1p']   = np.log1p(roll.mean())
                f[f'roll_{w}_median_log1p'] = np.log1p(roll.median())
                f[f'roll_{w}_zero_rate']    = float((roll == 0).mean())
                f[f'roll_{w}_trend']        = np.tanh(np.polyfit(range(w), roll.values, 1)[0]) if roll.std()>0 else 0.0
                f[f'roll_{w}_std_level']    = float(roll.std())
            else:
                f[f'roll_{w}_mean_log1p']   = 0.0
                f[f'roll_{w}_median_log1p'] = 0.0
                f[f'roll_{w}_zero_rate']    = 1.0
                f[f'roll_{w}_trend']        = 0.0
                f[f'roll_{w}_std_level']    = 0.0

        # sector-level stats
        f['sector_zero_rate'] = float((hist == 0).mean())
        f['sector_activity_consistency'] = 1.0 - f['sector_zero_rate']
        nz = hist[hist > 0]
        if len(nz) > 2:
            mean_pos = nz.mean()
            std_pos  = nz.std()
            f['nonzero_mean_log1p'] = np.log1p(mean_pos)
            f['nonzero_cv']         = np.tanh(std_pos/(mean_pos + 1e-6))
        else:
            f['nonzero_mean_log1p'] = 0.0
            f['nonzero_cv']         = 0.0

        # return-volatility block
        ret = _log1p_returns(hist)
        for w in [3,6,12]:
            if ret.size >= w:
                f[f'ret_std_{w}'] = float(pd.Series(ret).tail(w).std())
            else:
                f[f'ret_std_{w}'] = 0.0
        f['ret_cond_var_ewma'] = _ewma_cond_var(ret, alpha=0.3)

        def _cv(x: pd.Series):
            x = x[x > 0]
            if len(x) < 2 or x.mean() == 0:
                return 0.0
            return float(x.std() / (x.mean() + 1e-6))

        cv6  = _cv(hist.tail(6))
        cv12 = _cv(hist.tail(12))
        f['cv_pos_6']  = np.tanh(cv6)
        f['cv_pos_12'] = np.tanh(cv12)
        f['cv_pos_ratio_6_12'] = (cv6 / (cv12 + 1e-6)) if cv12 > 0 else 0.0

        lvl_std6  = f['roll_6_std_level']
        lvl_std12 = f['roll_12_std_level']
        f['vol_ratio_level_6_12'] = (lvl_std6 / (lvl_std12 + 1e-6)) if lvl_std12 > 0 else 0.0

        ret_std6  = f['ret_std_6']
        ret_std12 = f['ret_std_12']
        f['vol_ratio_ret_6_12'] = (ret_std6 / (ret_std12 + 1e-6)) if ret_std12 > 0 else 0.0

        # market context known at anchor_time
        m3 = float(total_roll3.reindex(df_panel.index).loc[anchor_time] or 0.0)
        m6 = float(total_roll6.reindex(df_panel.index).loc[anchor_time] or 0.0)
        f['market_activity_log1p'] = np.log1p(m3)
        f['market_trend']          = np.tanh((m3 - m6) / (m6 + 1e-6))

        # market volatility
        mkt_hist = total.loc[:anchor_time]
        mkt_ret = _log1p_returns(mkt_hist)
        f['market_ret_std_6'] = float(pd.Series(mkt_ret).tail(6).std()) if mkt_ret.size >= 6 else 0.0
        f['market_ret_std_12'] = float(pd.Series(mkt_ret).tail(12).std()) if mkt_ret.size >= 12 else 0.0
        f['market_ret_cond_var_ewma'] = _ewma_cond_var(mkt_ret, alpha=0.3)

        # recent zero streak
        cz = 0
        for v in hist.tail(6).iloc[::-1]:
            if v == 0: cz += 1
            else: break
        f['consecutive_zeros_norm']   = min(cz/6.0, 1.0)
        f['seasonal_zero_interaction'] = f['month_sin'] * f['sector_zero_rate']

        rows.append(f); sectors.append(s)

    feat = pd.DataFrame(rows)
    feat['sector'] = sectors
    feat['time'] = int(anchor_time)
    return feat


def fit_multi_horizon_and_forecast_final(
    train_matrix: pd.DataFrame,
    horizons: int = 12,
    ridge_alpha: float = 2.0,
    logistic_C: float = 0.5,
    use_isotonic: bool = True,
    max_history: int = 18,
    verbose: bool = True,
    # optional CSV outputs
    save_positive_csv: str | None = None,
    save_shrunk_csv: str | None = None,
    save_final_csv: str | None = None,
    save_probzero_csv: str | None = None,
):
    """
    Train one HorizonSpecificModel per horizon on multi-anchor data (up to t=67-h),
    then predict months 68..79 from the single anchor t=67.

    Saves (if paths provided):
      - positive_only_68_79.csv : raw Î¼Ì‚ from positive head
      - shrunk_nomask_68_79.csv: (1 - p0) * Î¼Ì‚ (no hard zeroing)
      - final_masked_68_79.csv : hard-zeroed final predictions
      - prob_zero_68_79.csv    : calibrated p0 (probability of being zero)
    """
    import numpy as np
    import pandas as pd

    last_train_t = int(train_matrix.index.max())
    assert last_train_t == 67, "This final routine expects the panel ends at 67."

    # ---- 1) build per-horizon training frames ----
    per_h_train = {}
    for h in range(1, horizons + 1):
        df_h = create_training_data_for_horizon(
            train_matrix, horizon=h, max_history=max_history, max_train_time=last_train_t - h
        )
        feat_cols = [c for c in df_h.columns if c not in ('target','sector','time','target_time')]
        per_h_train[h] = (df_h, feat_cols)
        if verbose:
            print(f"H{h:>2}: rows={len(df_h):>5}, features={len(feat_cols):>3}, "
                  f"zero-rate={(df_h['target'].values==0).mean():.1%}")

    feature_cols = per_h_train[1][1]

    # ---- 2) fit models ----
    models = {}
    for h in range(1, horizons + 1):
        df_h, _ = per_h_train[h]
        X = df_h[feature_cols].astype(float).values
        y = df_h['target'].astype(float).values

        m = HorizonSpecificModel(
            horizon=h,
            ridge_alpha=ridge_alpha,
            logistic_C=logistic_C,
            use_isotonic=use_isotonic,
            verbose=False
        )
        m.fit(X, y)
        models[h] = m

    # ---- 3) features at t=67 (single anchor) ----
    base_X = create_features_at_time(
        train_matrix.loc[:last_train_t], time_idx=last_train_t, max_history=max_history
    )
    sectors = base_X['sector'].values

    # ---- 4) predict 68..79 ----
    preds_positive = {}
    preds_shrunk   = {}
    preds_final    = {}
    preds_p0       = {}

    for h in range(1, horizons + 1):
        tgt_t = last_train_t + h
        tgt_month = ((tgt_t - 1) % 12) + 1

        Xh = base_X.copy()
        Xh['target_month_sin'] = np.sin(2*np.pi*tgt_month/12)
        Xh['target_month_cos'] = np.cos(2*np.pi*tgt_month/12)
        Xh['target_quarter']   = ((tgt_month - 1)//3 + 1) / 4.0
        Xh['horizon'] = h / 12.0

        for c in feature_cols:
            if c not in Xh.columns:
                Xh[c] = 0.0
        X = Xh[feature_cols].astype(float).values

        mdl = models[h]
        Xs = mdl.scaler.transform(X)

        # --- p0: probability of zero (no temperature in the original model) ---
        p0_raw = mdl.zero_model.predict_proba(Xs)[:, 1]
        p0 = mdl.iso_.transform(p0_raw) if mdl.iso_ is not None else p0_raw

        # --- Î¼Ì‚ from positive head ---
        if mdl.count_model is not None:
            mu_hat = np.clip(np.expm1(mdl.count_model.predict(Xs)), 0.0, mdl.mu_cap_)
        else:
            mu_hat = np.zeros(len(Xs), dtype=float)

        # shrunk (no masking)
        shrunk = (1.0 - p0) * mu_hat

        # final (apply threshold)
        y_hat = shrunk.copy()
        y_hat[p0 >= mdl.zero_threshold_] = 0.0
        y_hat = np.maximum(y_hat, 0.0)

        preds_positive[tgt_t] = mu_hat
        preds_shrunk[tgt_t]   = shrunk
        preds_final[tgt_t]    = y_hat
        preds_p0[tgt_t]       = p0

    # ---- 5) assemble outputs ----
    pred_wide_positive = pd.DataFrame(preds_positive, index=sectors).sort_index(axis=1)
    pred_wide_shrunk   = pd.DataFrame(preds_shrunk,   index=sectors).sort_index(axis=1)
    pred_wide_final    = pd.DataFrame(preds_final,    index=sectors).sort_index(axis=1)
    prob_zero_wide     = pd.DataFrame(preds_p0,       index=sectors).sort_index(axis=1)

    for df in (pred_wide_positive, pred_wide_shrunk, pred_wide_final, prob_zero_wide):
        df.index.name = 'sector'

    pred_long_final = (
        pred_wide_final.stack().rename("pred").reset_index()
        .rename(columns={"level_1":"time"})
        .sort_values(["time","sector"]).reset_index(drop=True)
    )

    # ---- 6) optional saves ----
    if save_positive_csv:
        pred_wide_positive.to_csv(save_positive_csv, index=True)
    if save_shrunk_csv:
        pred_wide_shrunk.to_csv(save_shrunk_csv, index=True)
    if save_final_csv:
        pred_wide_final.to_csv(save_final_csv, index=True)
    if save_probzero_csv:
        prob_zero_wide.to_csv(save_probzero_csv, index=True)

    # ---- 7) diagnostics ----
    if verbose:
        print("\nFinal diagnostics (train 1..67 â†’ predict 68..79):")
        print(f"  Feature columns: {len(feature_cols)}")
        print(f"  Pred months: {list(pred_wide_final.columns)}")
        print(f"  Final zero-rate overall: {(pred_wide_final.values==0).mean():.1%}")
        print("  Final zero-rate by horizon: " +
              ", ".join([f"{t}:{(pred_wide_final[t].values==0).mean():.1%}" for t in pred_wide_final.columns]))

    return (
        pred_wide_final, pred_long_final, models, feature_cols,
        pred_wide_positive, pred_wide_shrunk, prob_zero_wide
    )


# pred_68_79_wide, pred_68_79_long, models, feature_cols, pos_only_68_79, shrunk_68_79, prob_zero_68_79 = \
#     fit_multi_horizon_and_forecast_final(
#         train_matrix,
#         horizons=12,
#         ridge_alpha=20,
#         logistic_C=0.1,
#         use_isotonic=True,
#         max_history=16,
#         verbose=True,
#         save_positive_csv="positive_only_68_79.csv",
#         save_shrunk_csv="shrunk_nomask_68_79.csv",
#         save_final_csv="final_masked_68_79.csv",
#         save_probzero_csv="prob_zero_68_79.csv"   
#     )

# print("\nHead (probability of zero):")
# print(prob_zero_68_79.iloc[:5, :5])


def create_features_at_time(train_matrix: pd.DataFrame, time_idx: int, max_history=18):
    """
    Create features for all sectors at a specific time t (using history <= t).
    One row per sector that has at least `max_history` months available by t.
    """
    df_panel = train_matrix.apply(pd.to_numeric, errors='coerce').sort_index()
    df_panel.index.name = 'time'

    total = df_panel.sum(axis=1)
    total_roll3 = total.rolling(3, min_periods=1).mean().shift(1)   # up to t-1
    total_roll6 = total.rolling(6, min_periods=1).mean().shift(1)

    features_list, sector_list = [], []

    for sector in df_panel.columns:
        series = df_panel[sector].dropna()
        # history available up to and including time t
        hist = series[series.index <= time_idx]
        if len(hist) < max_history:
            continue

        f = {}
        # ID
        f['sector_id'] = int(str(sector).replace('sector ', '')) / 100.0

        # calendar at time t
        month_t = ((time_idx - 1) % 12) + 1
        f['month_sin'] = np.sin(2*np.pi*month_t/12)
        f['month_cos'] = np.cos(2*np.pi*month_t/12)
        f['quarter'] = ((month_t - 1) // 3 + 1) / 4.0

        # lags at t (values from <= t)
        for lag in [1, 2, 3, 6, 12]:
            if len(hist) >= lag:
                val = float(hist.iloc[-lag])
                f[f'lag_{lag}_log1p'] = np.log1p(val)
                f[f'lag_{lag}_zero'] = int(val == 0)
            else:
                f[f'lag_{lag}_log1p'] = 0.0
                f[f'lag_{lag}_zero'] = 1

        # rolling
        for w in [3, 6, 12]:
            if len(hist) >= w:
                roll = hist.tail(w)
                f[f'roll_{w}_mean_log1p'] = np.log1p(roll.mean())
                f[f'roll_{w}_median_log1p'] = np.log1p(roll.median())
                f[f'roll_{w}_zero_rate'] = float((roll == 0).mean())
                f[f'roll_{w}_trend'] = np.tanh(np.polyfit(range(w), roll.values, 1)[0]) if roll.std() > 0 else 0.0
            else:
                f[f'roll_{w}_mean_log1p'] = 0.0
                f[f'roll_{w}_median_log1p'] = 0.0
                f[f'roll_{w}_zero_rate'] = 1.0
                f[f'roll_{w}_trend'] = 0.0

        # sector stats
        f['sector_zero_rate'] = float((hist == 0).mean())
        f['sector_activity_consistency'] = 1.0 - f['sector_zero_rate']

        nz = hist[hist > 0]
        if len(nz) > 2:
            f['nonzero_mean_log1p'] = np.log1p(nz.mean())
            f['nonzero_cv'] = np.tanh(nz.std() / (nz.mean() + 1e-6))
        else:
            f['nonzero_mean_log1p'] = 0.0
            f['nonzero_cv'] = 0.0

        # market context at time t (from shifted rolls)
        mkt3 = float(total_roll3.loc[time_idx]) if time_idx in total_roll3.index and pd.notna(total_roll3.loc[time_idx]) else 0.0
        mkt6 = float(total_roll6.loc[time_idx]) if time_idx in total_roll6.index and pd.notna(total_roll6.loc[time_idx]) else 0.0
        f['market_activity_log1p'] = np.log1p(mkt3)
        f['market_trend'] = np.tanh((mkt3 - mkt6) / (mkt6 + 1e-6))

        # zero streak
        cz = 0
        for v in hist.tail(6).iloc[::-1]:
            if v == 0: cz += 1
            else: break
        f['consecutive_zeros_norm'] = min(cz / 6.0, 1.0)

        # interaction
        f['seasonal_zero_interaction'] = f['month_sin'] * f['sector_zero_rate']

        features_list.append(f)
        sector_list.append(sector)

    features_df = pd.DataFrame(features_list)
    features_df['sector'] = sector_list
    features_df['time'] = int(time_idx)
    return features_df

def create_training_data_for_horizon(train_matrix: pd.DataFrame, horizon: int, max_history=18, max_train_time=None):
    """
    Build supervised table for a given horizon h: X at time t â†’ y at t+h.
    Rows: (sector, t) with enough history by t; target is panel[t+h, sector].
    """
    print(f"Creating training data for horizon={horizon} ...")
    df_panel = train_matrix.apply(pd.to_numeric, errors='coerce').sort_index()

    # valid feature times t
    min_t = int(df_panel.index.min() + max_history)
    max_t_possible = int(df_panel.index.max() - horizon)
    max_t = int(min(max_t_possible, max_train_time)) if max_train_time is not None else max_t_possible
    if max_t < min_t:
        raise ValueError(f"No valid times for horizon {horizon}. Increase data or reduce horizon.")
    valid_times = [int(t) for t in df_panel.index if min_t <= t <= max_t]

    frames = []
    for t in valid_times:
        feat_t = create_features_at_time(df_panel, t, max_history=max_history)
        if feat_t.empty:
            continue
        tgt_time = t + horizon
        # horizon/meta features
        target_month = ((tgt_time - 1) % 12) + 1
        feat_t['target_month_sin'] = np.sin(2*np.pi*target_month/12)
        feat_t['target_month_cos'] = np.cos(2*np.pi*target_month/12)
        feat_t['target_quarter']   = ((target_month - 1) // 3 + 1) / 4.0
        feat_t['horizon'] = horizon / 12.0
        feat_t['target_time'] = int(tgt_time)

        # attach targets aligned by sector
        feat_t['target'] = df_panel.loc[tgt_time, feat_t['sector']].values.astype(float)
        frames.append(feat_t)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if result.empty:
        print(f"âš ï¸� Horizon {horizon}: produced 0 samples.")
        return result

    print(f"  â†’ {len(result)} rows, {result.filter(regex='^(?!sector$|time$|target$|target_time$)').shape[1]} features")
    print(f"  zero-rate={np.mean(result['target'].values==0):.1%}  mean={result['target'].mean():.2f}")
    return result


# --- single-anchor features (one row per sector, built at anchor_time) ---
def build_anchor_features(train_matrix: pd.DataFrame, anchor_time: int, max_history: int = 18) -> pd.DataFrame:
    df_panel = train_matrix.apply(pd.to_numeric, errors='coerce').sort_index()
    assert anchor_time in df_panel.index, "anchor_time must be inside train_matrix index"

    total = df_panel.sum(axis=1)
    total_roll3 = total.rolling(3, min_periods=1).mean().shift(1)
    total_roll6 = total.rolling(6, min_periods=1).mean().shift(1)

    rows = []
    sectors = []
    for s in df_panel.columns:
        hist = df_panel[s].loc[:anchor_time].dropna()
        if len(hist) < max_history:
            continue

        f = {}
        # id / calendar at anchor_time
        f['sector_id'] = int(str(s).replace('sector ', '')) / 100.0
        m = ((anchor_time - 1) % 12) + 1
        f['month_sin'] = np.sin(2*np.pi*m/12)
        f['month_cos'] = np.cos(2*np.pi*m/12)
        f['quarter']  = ((m - 1)//3 + 1) / 4.0

        # lags relative to anchor_time (use values at anchor_time, anchor_time-1, â€¦)
        for lag in [1,2,3,6,12]:
            if len(hist) >= lag:
                v = float(hist.iloc[-lag])
                f[f'lag_{lag}_log1p'] = np.log1p(v)
                f[f'lag_{lag}_zero']  = int(v == 0)
            else:
                f[f'lag_{lag}_log1p'] = 0.0
                f[f'lag_{lag}_zero']  = 1

        # rolling windows ending at anchor_time
        for w in [3,6,12]:
            if len(hist) >= w:
                roll = hist.tail(w)
                f[f'roll_{w}_mean_log1p']   = np.log1p(roll.mean())
                f[f'roll_{w}_median_log1p'] = np.log1p(roll.median())
                f[f'roll_{w}_zero_rate']    = float((roll == 0).mean())
                f[f'roll_{w}_trend']        = np.tanh(np.polyfit(range(w), roll.values, 1)[0]) if roll.std()>0 else 0.0
            else:
                f[f'roll_{w}_mean_log1p']   = 0.0
                f[f'roll_{w}_median_log1p'] = 0.0
                f[f'roll_{w}_zero_rate']    = 1.0
                f[f'roll_{w}_trend']        = 0.0

        # sector-level stats
        f['sector_zero_rate'] = float((hist == 0).mean())
        f['sector_activity_consistency'] = 1.0 - f['sector_zero_rate']
        nz = hist[hist > 0]
        if len(nz) > 2:
            f['nonzero_mean_log1p'] = np.log1p(nz.mean())
            f['nonzero_cv']         = np.tanh(nz.std()/(nz.mean()+1e-6))
        else:
            f['nonzero_mean_log1p'] = 0.0
            f['nonzero_cv']         = 0.0

        # market context known at anchor_time
        m3 = float(total_roll3.reindex(df_panel.index).loc[anchor_time] or 0.0)
        m6 = float(total_roll6.reindex(df_panel.index).loc[anchor_time] or 0.0)
        f['market_activity_log1p'] = np.log1p(m3)
        f['market_trend']          = np.tanh((m3 - m6) / (m6 + 1e-6))

        # recent zero streak
        cz = 0
        for v in hist.tail(6).iloc[::-1]:
            if v == 0: cz += 1
            else: break
        f['consecutive_zeros_norm']   = min(cz/6.0, 1.0)
        f['seasonal_zero_interaction'] = f['month_sin'] * f['sector_zero_rate']

        rows.append(f); sectors.append(s)

    feat = pd.DataFrame(rows)
    feat['sector'] = sectors
    feat['time'] = int(anchor_time)
    return feat


# ========= metric-aligned post-processing helpers =========
def comp_metric(y_true, y_pred, eps=1e-12):
    """
    Competition metric:
      - Score=0 if >30% samples have APE > 1 (i.e., >100%).
      - Else: score = 1 - mean(APE over good) / good_fraction.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ape = np.abs((y_pred - y_true) / np.maximum(y_true, eps))
    frac_bad = (ape > 1.0).mean()
    if frac_bad > 0.30:
        return 0.0
    good_mask = (ape <= 1.0)
    good_frac = good_mask.mean()
    if good_frac <= 0.0:
        return 0.0
    scaled_mape = ape[good_mask].mean() / good_frac
    return float(1.0 - scaled_mape)


def tiny_positive_to_zero(y_pred, lag1, zero_streak6_norm, tau=0.5):
    """
    Turn tiny positives into zeros when the recent history is mostly zeros.
    - lag1: last observed value at anchor t for each sector
    - zero_streak6_norm: the 'consecutive_zeros_norm' feature (0..1). We convert to 0..6 internally.
    - tau: predictions below this (in absolute) are snapped to 0 when the rule is triggered.
    """
    y_pred = np.asarray(y_pred, float).copy()
    zstreak6 = np.round(np.asarray(zero_streak6_norm, float) * 6.0)
    hard_zero = (lag1 == 0) & (zstreak6 >= 4) & (y_pred < tau)
    y_pred[hard_zero] = 0.0
    return y_pred


def _tune_shrink_and_clip(y_true, y_pred_raw,
                          s_grid=np.linspace(0.70, 1.00, 7),
                          q_caps=(0.95, 0.975, 0.99, 0.995)):
    """
    Grid-search a global shrink (multiplicative) and an absolute cap for a single month/horizon.
    Returns (best_s, best_cap, best_score).
    """
    y_true = np.asarray(y_true, float)
    y_raw  = np.asarray(y_pred_raw, float)
    best = (1.0, np.inf, -1.0)

    preds_pos = y_raw[np.isfinite(y_raw)]
    preds_pos = preds_pos[preds_pos > 0]
    caps = [np.inf]
    if preds_pos.size > 20:
        caps.extend([float(np.quantile(preds_pos, q)) for q in q_caps])

    for s in s_grid:
        base = y_raw * s
        for cap in caps:
            y_pp = np.clip(base, 0.0, cap)
            sc = comp_metric(y_true, y_pp)
            if sc > best[2]:
                best = (float(s), float(cap), float(sc))
    return best  # (s, cap, score)


def apply_postprocess(y_pred_raw, s, cap):
    """Apply multiplicative shrink and absolute upper cap."""
    return np.clip(np.asarray(y_pred_raw, float) * float(s), 0.0, float(cap))


def ratio_guard(y_pred, last_val, max_ratio=3.0):
    """
    Additional relative cap vs. last observed value.
    Keeps predictions â‰¤ max_ratio * lag1 (where lag1>0); otherwise no extra cap.
    """
    y_pred = np.asarray(y_pred, float).copy()
    last_val = np.asarray(last_val, float)
    rel_cap = np.where(last_val > 0, max_ratio * last_val, np.inf)
    return np.minimum(y_pred, rel_cap)


def safe_backoff(y_true, y_pred, start=1.0, step=0.05, floor=0.70):
    """
    If the catastrophic rate (>100% APE) is still near the 30% gate on validation,
    progressively shrink until it's comfortably below the gate.
    Returns (y_pp, used_scale).
    """
    y_pred = np.asarray(y_pred, float)
    s = float(start)
    eps = 1e-12
    while s >= floor:
        y_pp = np.maximum(y_pred * s, 0.0)
        ape = np.abs((y_pp - y_true) / np.maximum(y_true, eps))
        if (ape > 1.0).mean() <= 0.28:   # small margin
            return y_pp, s
        s -= step
    return np.maximum(y_pred * floor, 0.0), floor


def generate_final_predictions(
    train_matrix,
    horizons=12,
    ridge_alpha=2.0,
    logistic_C=0.5,
    use_isotonic=True,
    max_history=18,
):
    last_train = int(train_matrix.index.max())
    print("\n" + "="*80)
    print(f"FINAL: train 1..{last_train}  â†’  predict {last_train+1}..{last_train+horizons}")
    print("="*80)

    # Train horizon models on all available data
    forecaster = MultiHorizonForecaster(
        horizons=horizons, ridge_alpha=ridge_alpha,
        logistic_C=logistic_C, use_isotonic=use_isotonic,
        verbose=True, max_history=max_history
    )
    forecaster.fit(train_matrix)

    # Learn (s, cap) per horizon using the last feasible anchor t* = last_train - h
    tuned_hard = {}   # for standard path (with hard zeroing)
    tuned_soft = {}   # for soft path (no hard zeroing)
    for h in range(1, horizons + 1):
        anchor_t = last_train - h
        if anchor_t not in train_matrix.index:
            tuned_hard[h] = (1.0, np.inf)
            tuned_soft[h] = (1.0, np.inf)
            continue

        # --- features/guards at anchor_t
        feat_anchor = create_features_at_time(train_matrix, time_idx=anchor_t, max_history=max_history)
        sectors = feat_anchor['sector'].values
        feat_anchor = feat_anchor.set_index('sector')
        zstreak_norm_vec = feat_anchor['consecutive_zeros_norm'].astype(float).reindex(sectors).to_numpy()
        lag1_vec = train_matrix.loc[anchor_t, sectors].astype(float).to_numpy()

        # --- build meta-cols exactly as at predict time
        tgt_time = anchor_t + h
        dfh = feat_anchor.copy()
        tgt_month = ((tgt_time - 1) % 12) + 1
        dfh['target_month_sin'] = np.sin(2*np.pi*tgt_month/12)
        dfh['target_month_cos'] = np.cos(2*np.pi*tgt_month/12)
        dfh['target_quarter']   = ((tgt_month - 1) // 3 + 1) / 4.0
        dfh['horizon'] = h / 12.0
        for c in forecaster.feature_cols:
            if c not in dfh.columns:
                dfh[c] = 0.0
        Xh = dfh[forecaster.feature_cols].astype(float).values

        # --- raw predictions: hard vs. soft
        y_raw_hard = forecaster.models[h].predict(Xh)         # uses hard zeroing
        y_raw_soft = forecaster.models[h].predict_soft(Xh)    # soft, no hard zeroing

        # --- ground truth
        y_true = train_matrix.loc[tgt_time, sectors].astype(float).to_numpy()

        # --- tune for hard path
        y_step_h = tiny_positive_to_zero(y_raw_hard, lag1_vec, zstreak_norm_vec, tau=0.5)
        s_h, cap_h, _ = _tune_shrink_and_clip(y_true, y_step_h)
        tuned_hard[h] = (s_h, cap_h)

        # --- tune for soft path (same post-proc recipe)
        y_step_s = tiny_positive_to_zero(y_raw_soft, lag1_vec, zstreak_norm_vec, tau=0.5)
        s_s, cap_s, _ = _tune_shrink_and_clip(y_true, y_step_s)
        tuned_soft[h] = (s_s, cap_s)

    # --- features at the real test anchor (last_train)
    feats_now = create_features_at_time(train_matrix, time_idx=last_train, max_history=max_history)
    sectors_now = feats_now['sector'].values
    feats_now = feats_now.set_index('sector')
    zstreak_norm_now = feats_now['consecutive_zeros_norm'].astype(float).reindex(sectors_now).to_numpy()
    lag1_now = train_matrix.loc[last_train, sectors_now].astype(float).to_numpy()

    preds_hard = {}
    preds_soft = {}
    for h in range(1, horizons + 1):
        tgt_time = last_train + h
        dfh = feats_now.copy()
        tgt_month = ((tgt_time - 1) % 12) + 1
        dfh['target_month_sin'] = np.sin(2*np.pi*tgt_month/12)
        dfh['target_month_cos'] = np.cos(2*np.pi*tgt_month/12)
        dfh['target_quarter']   = ((tgt_month - 1) // 3 + 1) / 4.0
        dfh['horizon'] = h / 12.0
        for c in forecaster.feature_cols:
            if c not in dfh.columns:
                dfh[c] = 0.0
        Xh = dfh[forecaster.feature_cols].astype(float).values

        # --- standard path (hard zeroing inside .predict)
        y_raw_h = forecaster.models[h].predict(Xh)
        y_step_h = tiny_positive_to_zero(y_raw_h, lag1_now, zstreak_norm_now, tau=0.5)
        s_h, cap_h = tuned_hard.get(h, (1.0, np.inf))
        y_pp_h = apply_postprocess(y_step_h, s_h, cap_h)
        y_pp_h = ratio_guard(y_pp_h, lag1_now, max_ratio=3.0)
        preds_hard[tgt_time] = y_pp_h

        # --- soft path (no hard zeroing)
        y_raw_s = forecaster.models[h].predict_soft(Xh)
        y_step_s = tiny_positive_to_zero(y_raw_s, lag1_now, zstreak_norm_now, tau=0.5)
        s_s, cap_s = tuned_soft.get(h, (1.0, np.inf))
        y_pp_s = apply_postprocess(y_step_s, s_s, cap_s)
        y_pp_s = ratio_guard(y_pp_s, lag1_now, max_ratio=3.0)
        preds_soft[tgt_time] = y_pp_s

    preds_df      = pd.DataFrame(preds_hard, index=sectors_now).sort_index(axis=1)
    preds_df_soft = pd.DataFrame(preds_soft, index=sectors_now).sort_index(axis=1)

    print(f"[hard] shape={preds_df.shape}      zero-rate={(preds_df.values==0).mean():.1%}")
    print(f"[soft] shape={preds_df_soft.shape} zero-rate={(preds_df_soft.values==0).mean():.1%}")

    # Return both for analysis
    return preds_df, preds_df_soft



# 0) choose the horizon to submit (next 12 months)
H = 12
ridge_alpha  = 0.73
logistic_C   = 0.07
use_isotonic = True
max_history  = 16

# 1) generate forecasts (post-processed + tuned per horizon)
final_preds, final_preds_positive = generate_final_predictions(
    train_matrix=train_matrix,
    horizons=H,
    ridge_alpha=ridge_alpha,
    logistic_C=logistic_C,
    use_isotonic=use_isotonic,
    max_history=max_history,
)


final_preds_positive.tail(50)


final_preds.tail(50)


def adjust_predictions(final_preds, final_preds_positive, adjustment_rules):
    """
    Manually adjust predictions by sector, replacing zeros with positive predictions
    or positive predictions with zeros.
    
    Parameters:
    -----------
    final_preds : pd.DataFrame
        Original predictions dataframe (with zeros from logistic regression)
    final_preds_positive : pd.DataFrame
        Predictions dataframe without zero classification (all positive)
    adjustment_rules : dict
        Dictionary specifying adjustments by sector. Format:
        {
            'sector_name': {
                'use_positive': [list of column indices or names],  # Replace zeros with positive
                'use_zero': [list of column indices or names]       # Replace positive with zeros
            }
        }
    
    Returns:
    --------
    pd.DataFrame
        Adjusted predictions dataframe
    
    Example:
    --------
    adjustment_rules = {
        'sector 1': {
            'use_positive': [68, 69, 70],  # Use positive predictions for these columns
            'use_zero': []                  # Keep current predictions
        },
        'sector 12': {
            'use_positive': [],             # Keep current predictions (zeros)
            'use_zero': [71, 72]            # Force these to zero
        }
    }
    
    adjusted = adjust_predictions(final_preds, final_preds_positive, adjustment_rules)
    """
    
    # Create a copy to avoid modifying the original
    adjusted_preds = final_preds.copy()
    
    # Process each sector's adjustment rules
    for sector, rules in adjustment_rules.items():
        if sector not in adjusted_preds.index:
            print(f"Warning: {sector} not found in dataframe index")
            continue
        
        # Replace zeros with positive predictions
        if 'use_positive' in rules and rules['use_positive']:
            for col in rules['use_positive']:
                if col in adjusted_preds.columns:
                    adjusted_preds.loc[sector, col] = final_preds_positive.loc[sector, col]
                else:
                    print(f"Warning: Column {col} not found")
        
        # Replace positive predictions with zeros
        if 'use_zero' in rules and rules['use_zero']:
            for col in rules['use_zero']:
                if col in adjusted_preds.columns:
                    adjusted_preds.loc[sector, col] = 0.0
                else:
                    print(f"Warning: Column {col} not found")
    
    return adjusted_preds


def compare_adjustments(original, adjusted, sector):
    """
    Compare original and adjusted predictions for a specific sector.
    
    Parameters:
    -----------
    original : pd.DataFrame
        Original predictions
    adjusted : pd.DataFrame
        Adjusted predictions
    sector : str
        Sector name to compare
    
    Returns:
    --------
    pd.DataFrame
        Comparison showing differences
    """
    if sector not in original.index or sector not in adjusted.index:
        print(f"Sector {sector} not found")
        return None
    
    comparison = pd.DataFrame({
        'Original': original.loc[sector],
        'Adjusted': adjusted.loc[sector],
        'Changed': original.loc[sector] != adjusted.loc[sector]
    })
    
    # Show only columns that changed
    changed_cols = comparison[comparison['Changed']].index
    
    if len(changed_cols) > 0:
        print(f"\nChanges in {sector}:")
        print(comparison[comparison['Changed']][['Original', 'Adjusted']])
        print(f"\nTotal changes: {len(changed_cols)}")
    else:
        print(f"No changes in {sector}")
    
    return comparison


# Define which sectors and columns to adjust
adjustment_rules = {
    'sector 12': {
        'use_positive': [68, 69, 71],  # Replace zeros with positive values
        'use_zero': []  # No forced zeros
    },
    'sector 19': {
        'use_positive': [72, 76, 78, 79],
        'use_zero': []  # Force these to be zero
    },
    'sector 33': {
        'use_positive': [],
        'use_zero': [69, 70, 71, 72, 74, 75, 76, 77]  # Force these to be zero
    },
    'sector 53': {
        'use_positive': [68],
        'use_zero': []  # Force these to be zero
    },
    'sector 58': {
        'use_positive': [68],
        'use_zero': []  # Force these to be zero
    },
    'sector 89': {
        'use_positive': [69, 71, 72],
        'use_zero': []  # Force these to be zero
    },
    'sector 70': {
        'use_positive': [78, 79],
        'use_zero': []  # Force these to be zero
    },
    'sector 17': {
        'use_positive': [78, 79],
        'use_zero': []  # Force these to be zero
    },
    'sector 72': {
        'use_positive': [69, 71],
        'use_zero': []  # Force these to be zero
    },
    'sector 73': {
        'use_positive': [68, 71, 72, 73],
        'use_zero': []  # Force these to be zero
    },
    'sector 75': {
        'use_positive': [69, 72],
        'use_zero': []  # Force these to be zero
    }
}

# Apply the adjustments
adjusted_preds = adjust_predictions(final_preds, final_preds_positive, adjustment_rules)

# Check what changed
# compare_adjustments(final_preds, adjusted_preds, 'sector 12')


adjusted_preds.tail(50)


# ========= seasonal nudger (extended) =========
def _month_from_time(t: int) -> int:
    return ((int(t) - 1) % 12) + 1

def _trimmed_mean(a, trim_prop=0.1):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0: return np.nan
    k = int(np.floor(trim_prop * a.size))
    if k <= 0: return float(np.mean(a))
    a_sorted = np.sort(a)
    return float(np.mean(a_sorted[k: a.size - k]))

def _winsor_mean(a, q=0.10):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0: return np.nan
    lo, hi = np.quantile(a, q), np.quantile(a, 1 - q)
    a_w = np.clip(a, lo, hi)
    return float(np.mean(a_w))

def _quantile_blend(a, q_lo=0.35, q_hi=0.65):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0: return np.nan
    lo = np.quantile(a, q_lo)
    hi = np.quantile(a, q_hi)
    return float((lo + hi) / 2.0)

def _ema_on_pairs(pairs, alpha=0.35):
    """
    pairs: list of (time, ratio) sorted by time
    """
    if len(pairs) == 0: return np.nan
    m = None
    for _, r in pairs:
        r = float(r)
        if not np.isfinite(r) or r <= 0: 
            continue
        if m is None:
            m = r
        else:
            m = alpha * r + (1 - alpha) * m
    return float(m) if (m is not None) else np.nan

def build_seasonal_nudger(train_matrix: pd.DataFrame,
                          window: int = 12,
                          method: str = "harmonic",
                          min_obs_per_month: int = 1,
                          # method hyperparams:
                          trim_prop: float = 0.10,
                          winsor_q: float = 0.10,
                          q_lo: float = 0.35, q_hi: float = 0.65,
                          ema_alpha: float = 0.35) -> dict:
    """
    Compute raw per-(sector, month) factors using the last `window` months.
    method:
      - 'harmonic'      : robust when ratios can get large (positives only)
      - 'median'        : very robust
      - 'mean'          : simple average
      - 'trimmed_mean'  : mean after trimming tails (trim_prop)
      - 'winsor_mean'   : mean after winsorizing to [q, 1-q]
      - 'quantile_blend': average of two quantiles (q_lo, q_hi)
      - 'ema'           : recency-weighted average over window (ema_alpha)
    All methods normalize factors within a sector to have mean â‰ˆ 1 across the 12 months.
    """
    df = train_matrix.apply(pd.to_numeric, errors='coerce').sort_index()
    last_t = int(df.index.max())
    win_idx = df.index[df.index > (last_t - window)]
    win = df.loc[win_idx].copy()

    # sector baseline (positives only) over the window
    sector_base = win.replace(0, np.nan).mean(axis=0)

    # collect ratios
    # for EMA we need time, so store (t, ratio); otherwise just ratios
    buckets = { (s, m): [] for s in df.columns for m in range(1, 13) }
    buckets_time = { (s, m): [] for s in df.columns for m in range(1, 13) }

    for t in win.index:
        m = _month_from_time(t)
        row = win.loc[t]
        for s, y in row.items():
            b = float(sector_base.get(s, np.nan))
            if not np.isfinite(b) or b <= 0:
                continue
            r = float(y) / b
            if r > 0 and np.isfinite(r):
                buckets[(s, m)].append(r)
                buckets_time[(s, m)].append((int(t), r))

    # aggregate
    factors_raw = {}
    for key in buckets.keys():
        arr = buckets[key]
        if len(arr) < min_obs_per_month:
            f = 1.0
        else:
            a = np.asarray(arr, float)
            a = a[np.isfinite(a) & (a > 0)]
            if a.size == 0:
                f = 1.0
            else:
                if method == "harmonic":
                    f = float(a.size / np.sum(1.0 / np.maximum(a, 1e-12)))
                elif method == "median":
                    f = float(np.median(a))
                elif method == "mean":
                    f = float(np.mean(a))
                elif method == "trimmed_mean":
                    f = _trimmed_mean(a, trim_prop=trim_prop)
                elif method == "winsor_mean":
                    f = _winsor_mean(a, q=winsor_q)
                elif method == "quantile_blend":
                    f = _quantile_blend(a, q_lo=q_lo, q_hi=q_hi)
                elif method == "ema":
                    pairs = sorted(buckets_time[key], key=lambda x: x[0])
                    f = _ema_on_pairs(pairs, alpha=ema_alpha)
                    if not np.isfinite(f): f = 1.0
                else:
                    raise ValueError(f"Unknown method: {method}")
        factors_raw[key] = float(f if np.isfinite(f) and f > 0 else 1.0)

    # normalize within sector to mean â‰ˆ 1
    for s in df.columns:
        vals = np.array([factors_raw[(s, m)] for m in range(1, 13)], dtype=float)
        good = np.isfinite(vals) & (vals > 0)
        scale = np.mean(vals[good]) if np.any(good) else 1.0
        scale = scale if (np.isfinite(scale) and scale > 0) else 1.0
        for m in range(1, 13):
            factors_raw[(s, m)] = float(factors_raw[(s, m)] / scale if scale != 0 else 1.0)
    return factors_raw

def shrink_seasonal_factors(factors_raw: dict,
                            strength: float = 0.30,
                            clip: float = 0.10) -> dict:
    """
    Shrink toward 1.0 and cap to [1-clip, 1+clip] to prevent over-corrections.
    """
    lo, hi = 1.0 - clip, 1.0 + clip
    out = {}
    for key, f in factors_raw.items():
        f = float(f) if np.isfinite(f) else 1.0
        f_sh = 1.0 + strength * (f - 1.0)
        out[key] = float(np.clip(f_sh, lo, hi))
    return out

def apply_seasonal_nudge_to_predictions(features_like_df: pd.DataFrame,
                                        y_pred: np.ndarray,
                                        factors: dict | None) -> np.ndarray:
    y = np.asarray(y_pred, dtype=float).copy()
    if not factors:
        return y
    sec = features_like_df['sector'].values
    tim = features_like_df['time'].values
    for i, (s, t, v) in enumerate(zip(sec, tim, y)):
        if v > 0:
            m = _month_from_time(int(t))
            y[i] = float(v * factors.get((s, m), 1.0))
    return y

def apply_seasonal_nudge_to_matrix(preds_matrix: pd.DataFrame,
                                   factors: dict | None) -> pd.DataFrame:
    if not factors:
        return preds_matrix.copy()
    out = preds_matrix.copy()
    for t in out.index:
        m = _month_from_time(int(t))
        for s in out.columns:
            v = float(out.loc[t, s])
            if v > 0:
                out.loc[t, s] = float(v * factors.get((s, m), 1.0))
    return out


future_preds = adjusted_preds.T.copy()


method  = "trimmed_mean"   # e.g. trimmed mean performed best in your table
window  = 12               # last 12 months
lam     = 0.30             # nudge strength (try a few)
clip    = 0.10             # cap multipliers to [0.90, 1.10]

factors_raw = build_seasonal_nudger(
    train_matrix, window=window, method=method,
    trim_prop=0.10, winsor_q=0.10, ema_alpha=0.35
)
factors = shrink_seasonal_factors(factors_raw, strength=lam, clip=clip)

future_preds_nudged = apply_seasonal_nudge_to_matrix(future_preds, factors)


future_preds_nudged.head(12)


def create_submission_from_predictions_v2(predictions_df):
    """Convert predictions to submission format"""
    
    submission_data = []
    
    for time in predictions_df.index:
        for sector in predictions_df.columns:
            # Convert time back to month string
            year = 2019 + (time - 1) // 12
            month_num = ((time - 1) % 12) + 1
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            month_str = f"{year} {month_names[month_num-1]}"
            
            # Create test ID
            test_id = f"{month_str}_{sector}"
            
            # Get prediction value
            prediction = predictions_df.loc[time, sector]
            
            submission_data.append({
                'id': test_id,
                'new_house_transaction_amount': prediction
            })
    
    return pd.DataFrame(submission_data)


almost_preds = create_submission_from_predictions_v2(future_preds_nudged)


almost_preds.head(12)


sector_42 = pd.read_csv("/kaggle/input/sector-42/submission (32).csv")


# Filter sector_42 dataframe to only get sector 42 rows
sector_42_only = sector_42[sector_42['id'].str.endswith('sector 42')].copy()

print(f"Rows in sector_42 for sector 42 only: {len(sector_42_only)}")

# Now replace in almost_preds
mask = almost_preds['id'].str.endswith('sector 42')
print(f"Rows in almost_preds for sector 42: {mask.sum()}")

# Create a dictionary mapping from id to value
sector_42_dict = dict(zip(sector_42_only['id'], sector_42_only['new_house_transaction_amount']))

# Update using the dictionary to ensure correct matching by id
for idx in almost_preds[mask].index:
    id_value = almost_preds.loc[idx, 'id']
    if id_value in sector_42_dict:
        almost_preds.loc[idx, 'new_house_transaction_amount'] = sector_42_dict[id_value]

# Verify the replacement
print("\nUpdated sector 42 predictions:")
print(almost_preds[mask].sort_values('id'))


submission = almost_preds.copy()


submission.to_csv("submission.csv", index=False)

