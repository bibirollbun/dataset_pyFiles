import os
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from joblib import dump, load


# Boosting libs
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor


warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)


@dataclass
class Config:
    train_path: str = "/kaggle/input/playground-series-s5e10/train.csv"
    test_path: str = "/kaggle/input/playground-series-s5e10/test.csv"
    submission_path: str = "/kaggle/input/playground-series-s5e10/sample_submission.csv"
    output_submission: str = "submission.csv"
    n_splits: int = 5
    random_state: int = SEED
    target_col: str = "accident_risk"
    id_col: str = "id"
    models_dir: str = "models"
    save_models: bool = True


class DataLoader:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print("Loading data...")
        train = pd.read_csv(self.cfg.train_path)
        test = pd.read_csv(self.cfg.test_path)
        sample = pd.read_csv(self.cfg.submission_path)
        print(f"Train shape: {train.shape}")
        print(f"Test shape: {test.shape}")
        return train, test, sample


class Preprocessor:
    """Handle missing values and encode categorical features."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cat_cols: List[str] = []
        self.num_cols: List[str] = []
        self.imputer_num = SimpleImputer(strategy="median")
        self.imputer_cat = SimpleImputer(strategy="most_frequent")
        self.encoder = None
        self.scaler = StandardScaler()

    def _infer_columns(self, df: pd.DataFrame):
        all_cols = df.columns.tolist()
        if self.cfg.id_col in all_cols:
            all_cols.remove(self.cfg.id_col)
        if self.cfg.target_col in all_cols:
            all_cols.remove(self.cfg.target_col)
        # Heuristic: object dtype -> categorical, otherwise numeric
        self.cat_cols = [c for c in all_cols if df[c].dtype == "object" or df[c].dtype.name == 'category']
        self.num_cols = [c for c in all_cols if c not in self.cat_cols]
        print(f"Inferred categorical cols: {self.cat_cols}")
        print(f"Inferred numeric cols: {self.num_cols}")

    def fit(self, df: pd.DataFrame):
        self._infer_columns(df)
        # Fit imputers
        if len(self.num_cols) > 0:
            self.imputer_num.fit(df[self.num_cols])
        if len(self.cat_cols) > 0:
            self.imputer_cat.fit(df[self.cat_cols])
            # Fit ordinal encoder for categories (safe for unseen values)
            self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            tmp = df[self.cat_cols].fillna("__NA__")
            self.encoder.fit(tmp)
        # Fit scaler on numeric for MLP
        Xnum = pd.DataFrame(self.imputer_num.transform(df[self.num_cols]) if len(self.num_cols)>0 else np.zeros((len(df),0)), columns=self.num_cols if len(self.num_cols)>0 else [])
        if len(self.num_cols) > 0:
            self.scaler.fit(Xnum)
        print("Preprocessor fitted.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Impute numeric
        if len(self.num_cols) > 0:
            df[self.num_cols] = self.imputer_num.transform(df[self.num_cols])
        # Impute and encode cat
        if len(self.cat_cols) > 0:
            df[self.cat_cols] = self.imputer_cat.transform(df[self.cat_cols])
            df[self.cat_cols] = pd.DataFrame(self.encoder.transform(df[self.cat_cols]), columns=self.cat_cols, index=df.index)
        # Safety: engineered features (created after fit) may have NaNs — fill remaining missing values
        # We use a simple strategy: fill numeric NaNs with 0, and any remaining object/category NaNs with a placeholder -1
        for col in df.columns:
            if df[col].dtype.kind in 'biufc':  # numeric
                if df[col].isnull().any():
                    df[col] = df[col].fillna(0)
            else:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(-1)
        return df

    def transform_for_mlp(self, df: pd.DataFrame) -> np.ndarray:
        df_t = self.transform(df)
        # For MLP: use numeric + encoded cat numeric values scaled
        arr = np.hstack([
            df_t[self.num_cols].values if len(self.num_cols) > 0 else np.zeros((len(df_t), 0)),
            df_t[self.cat_cols].values if len(self.cat_cols) > 0 else np.zeros((len(df_t), 0))
        ])
        return self.scaler.transform(arr) if len(self.num_cols)>0 else arr


class FeatureEngineer:
    """Create simple interaction features and cyclic encoding for time_of_day if present."""

    def __init__(self):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Example: encode time_of_day cyclically if present and numeric
        if 'time_of_day' in df.columns:
            try:
                # assume in hours or fraction; scale to 24
                tod = pd.to_numeric(df['time_of_day'], errors='coerce')
                df['tod_sin'] = np.sin(2 * np.pi * tod / 24)
                df['tod_cos'] = np.cos(2 * np.pi * tod / 24)
            except Exception:
                pass
        # Simple interaction: lanes * speed_limit
        if 'num_lanes' in df.columns and 'speed_limit' in df.columns:
            df['lanes_x_speed'] = pd.to_numeric(df['num_lanes'], errors='coerce').fillna(0) * pd.to_numeric(df['speed_limit'], errors='coerce').fillna(0)
        return df


class ModelTrainer:
    def __init__(self, cfg: Config, preprocessor: Preprocessor, fe: FeatureEngineer):
        self.cfg = cfg
        self.preprocessor = preprocessor
        self.fe = fe
        self.folds = KFold(n_splits=self.cfg.n_splits, shuffle=True, random_state=self.cfg.random_state)
        os.makedirs(self.cfg.models_dir, exist_ok=True)

    def _rmse(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)

    def _get_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.fe.transform(df)
        df_t = self.preprocessor.transform(df)
        # keep id and target separately, return feature dataframe
        features = [c for c in df_t.columns if c not in [self.cfg.id_col, self.cfg.target_col]]
        return df_t[features]

    def train_and_evaluate(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
        X = self._get_feature_matrix(train_df)
        y = train_df[self.cfg.target_col].values
        X_test_full = self._get_feature_matrix(test_df)

        oof_preds = np.zeros(len(train_df))
        test_preds_models = { 'xgb': np.zeros(len(test_df)), 'lgb': np.zeros(len(test_df)), 'cat': np.zeros(len(test_df)), 'mlp': np.zeros(len(test_df)) }
        val_scores = { 'xgb': [], 'lgb': [], 'cat': [], 'mlp': [] }

        for fold, (tr_idx, val_idx) in enumerate(self.folds.split(X, y)):
            print(f"--- Fold {fold+1} / {self.cfg.n_splits} ---")
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            # XGBoost
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_val)
            dtest = xgb.DMatrix(X_test_full)
            xgb_params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'seed': self.cfg.random_state,
                'learning_rate': 0.05,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'nthread': 4,
            }
            xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=2000, evals=[(dtrain,'train'),(dval,'val')], early_stopping_rounds=50, verbose_eval=100)
            # handle xgboost predict API differences across versions
            best_it = getattr(xgb_model, 'best_iteration', None)
            if best_it is None:
                xgb_val_pred = xgb_model.predict(dval)
                xgb_test_pred = xgb_model.predict(dtest)
            else:
                # predict using iteration_range up to best_iteration
                xgb_val_pred = xgb_model.predict(dval, iteration_range=(0, best_it))
                xgb_test_pred = xgb_model.predict(dtest, iteration_range=(0, best_it))
            val_rmse = self._rmse(y_val, xgb_val_pred)
            print(f"XGB fold {fold} RMSE: {val_rmse:.6f}")
            val_scores['xgb'].append(val_rmse)
            test_preds_models['xgb'] += xgb_test_pred / self.cfg.n_splits

            # LightGBM
            lgb_train = lgb.Dataset(X_tr, label=y_tr)
            lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
            lgb_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'seed': self.cfg.random_state,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbosity': -1
            }
            # train LightGBM using callbacks (compatible with newer versions)
            lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=2000, valid_sets=[lgb_train, lgb_val], callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)])
            lgb_val_pred = lgb_model.predict(X_val, num_iteration=getattr(lgb_model, 'best_iteration', None))
            lgb_test_pred = lgb_model.predict(X_test_full, num_iteration=getattr(lgb_model, 'best_iteration', None))
            val_rmse = self._rmse(y_val, lgb_val_pred)
            print(f"LGB fold {fold} RMSE: {val_rmse:.6f}")
            val_scores['lgb'].append(val_rmse)
            test_preds_models['lgb'] += lgb_test_pred / self.cfg.n_splits

            # CatBoost
            cat_model = CatBoostRegressor(
                iterations=2000,
                learning_rate=0.05,
                depth=6,
                eval_metric='RMSE',
                random_seed=self.cfg.random_state,
                early_stopping_rounds=50,
                verbose=100
            )
            cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
            cat_val_pred = cat_model.predict(X_val)
            cat_test_pred = cat_model.predict(X_test_full)
            val_rmse = self._rmse(y_val, cat_val_pred)
            print(f"Cat fold {fold} RMSE: {val_rmse:.6f}")
            val_scores['cat'].append(val_rmse)
            test_preds_models['cat'] += cat_test_pred / self.cfg.n_splits

            # MLP (sklearn) - scale features
            # Note: if scaler wasn't fitted (no numeric cols), fallback to raw
            try:
                X_tr_mlp = self.preprocessor.scaler.transform(X_tr)
                X_val_mlp = self.preprocessor.scaler.transform(X_val)
                X_test_mlp = self.preprocessor.scaler.transform(X_test_full)
            except Exception:
                X_tr_mlp = X_tr.values
                X_val_mlp = X_val.values
                X_test_mlp = X_test_full.values

            mlp = MLPRegressor(hidden_layer_sizes=(128,64), learning_rate_init=1e-3, random_state=self.cfg.random_state, max_iter=500, early_stopping=True, verbose=False)
            mlp.fit(X_tr_mlp, y_tr)
            mlp_val_pred = mlp.predict(X_val_mlp)
            mlp_test_pred = mlp.predict(X_test_mlp)
            val_rmse = self._rmse(y_val, mlp_val_pred)
            print(f"MLP fold {fold} RMSE: {val_rmse:.6f}")
            val_scores['mlp'].append(val_rmse)
            test_preds_models['mlp'] += mlp_test_pred / self.cfg.n_splits

            # OOF stacking baseline: average of boosting models for this fold
            fold_pred = (xgb_val_pred + lgb_val_pred + cat_val_pred + mlp_val_pred) / 4.0
            oof_preds[val_idx] = fold_pred

            # Save fold models
            if self.cfg.save_models:
                try:
                    dump(xgb_model, os.path.join(self.cfg.models_dir, f"xgb_fold{fold}.joblib"))
                except Exception:
                    pass
                try:
                    dump(lgb_model, os.path.join(self.cfg.models_dir, f"lgb_fold{fold}.joblib"))
                except Exception:
                    pass
                try:
                    cat_model.save_model(os.path.join(self.cfg.models_dir, f"cat_fold{fold}.cbm"))
                except Exception:
                    pass
                try:
                    dump(mlp, os.path.join(self.cfg.models_dir, f"mlp_fold{fold}.joblib"))
                except Exception:
                    pass

        # Compute overall OOF RMSE
        oof_rmse = self._rmse(y, oof_preds)
        print(f"OOF RMSE (simple average stack): {oof_rmse:.6f}")

        # Show validation scores summary
        for m in val_scores:
            print(f"{m} mean RMSE: {np.mean(val_scores[m]):.6f} +/- {np.std(val_scores[m]):.6f}")

        # Simple ensemble: weighted average by inverse mean val rmse
        means = {m: np.mean(val_scores[m]) for m in val_scores}
        inv = {m: 1.0 / means[m] for m in means}
        total_inv = sum(inv.values())
        weights = {m: inv[m] / total_inv for m in inv}
        print(f"Ensemble weights: {weights}")

        blended_test = sum(test_preds_models[m] * weights[m] for m in test_preds_models)

        # Prepare submission
        submission = pd.DataFrame({self.cfg.id_col: test_df[self.cfg.id_col].values, self.cfg.target_col: blended_test})
        # Clip predictions to [0,1]
        submission[self.cfg.target_col] = submission[self.cfg.target_col].clip(0,1)
        submission.to_csv(self.cfg.output_submission, index=False)
        print(f"Wrote submission to {self.cfg.output_submission}")

        # Return OOF dataframe for further analysis
        oof_df = train_df[[self.cfg.id_col, self.cfg.target_col]].copy()
        oof_df['oof_pred'] = oof_preds
        return oof_df


def run_all(train_path="/kaggle/input/playground-series-s5e10/train.csv", test_path='/kaggle/input/playground-series-s5e10/test.csv', submission_path='/kaggle/input/playground-series-s5e10/sample_submission.csv'):
    cfg = Config(train_path=train_path, test_path=test_path, submission_path=submission_path)
    loader = DataLoader(cfg)
    train, test, sample = loader.load()

    # Quick safety checks: ensure id col exists
    if cfg.id_col not in train.columns or cfg.id_col not in test.columns:
        raise ValueError(f"Dataset must contain '{cfg.id_col}' column")

    pre = Preprocessor(cfg)
    pre.fit(train)

    fe = FeatureEngineer()

    trainer = ModelTrainer(cfg, pre, fe)
    oof_df = trainer.train_and_evaluate(train, test)

    print("Done.")
    return oof_df


oof = run_all()

