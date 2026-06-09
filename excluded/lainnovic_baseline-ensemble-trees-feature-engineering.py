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


print(" all the model params below are from optuna using 10 trials to create baseline and to understand feature importance")


!pip install xgboost lightgbm
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'



import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import RidgeCV

class BPMEnsemblePredictor:
    def __init__(self):
        # âœ… Always use best tuned params (no FAST mode anymore)
        self.xgb_params = {
            'n_estimators': 2500,
            'max_depth': 4,
            'learning_rate': 0.025149991569945206,
            'subsample': 0.878592857091884,
            'colsample_bytree': 0.6407511396935177,
            'reg_alpha': 0.2562630537926758,
            'reg_lambda': 3.3078061640685643,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'gpu_id': 0
        }
        self.lgb_params = {
            'n_estimators': 1000,
            'max_depth': 3,
            'learning_rate': 0.27849647461583216,
            'subsample': 0.7792653428855618,
            'colsample_bytree': 0.6589220743356489,
            'reg_alpha': 9.384640347769494,
            'reg_lambda': 0.1165925847370386,
            'num_leaves': 100,
            'min_child_samples': 48,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            'device': 'gpu',
            'metric': 'rmse'
        }
        self.rf_params = {
            'n_estimators': 410,
            'max_depth': 7,
            'min_samples_split': 6,
            'min_samples_leaf': 3,
            'max_features': 'log2',
            'random_state': 42,
            'n_jobs': -1
        }
        self.cat_params = {
            'iterations': 2500,
            'learning_rate': 0.05,
            'depth': 6,
            'random_seed': 42,
            'loss_function': 'RMSE',
            'verbose': 0,
            'task_type': 'GPU',
            'devices': '0'
        }
        self.n_splits = 5
        self.early_stop = 50
        self.meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=5)
        self.fitted_base_models = {}
        self.duration_stats = None
        
    def engineer_features(self, df):
        df = df.copy()
        df["Energy_Rhythm"] = df["Energy"] / (df["RhythmScore"] + 1e-6)
        df["Loudness_Acoustic"] = df["AudioLoudness"] * df["AcousticQuality"]
        df["Mood_Instrumental"] = df["MoodScore"] * df["InstrumentalScore"]
        df["LogDuration"] = np.log1p(df["TrackDurationMs"])
        df["Energy_Loudness"] = df["Energy"] * df["AudioLoudness"]
        df["Vocal_Acoustic"] = df["VocalContent"] * df["AcousticQuality"]
        df["Live_Energy"] = df["LivePerformanceLikelihood"] * df["Energy"]
        df["Rhythm_Mood"] = df["RhythmScore"] * df["MoodScore"]
        df["Energy_Squared"] = df["Energy"] ** 2
        df["RhythmScore_Squared"] = df["RhythmScore"] ** 2
        df["Duration_Energy"] = df["TrackDurationMs"] * df["Energy"]
        df["Duration_Category"] = pd.cut(df["TrackDurationMs"],
                                         bins=5, labels=['very_short', 'short', 'medium', 'long', 'very_long'])
        df["Duration_Category"] = df["Duration_Category"].astype('category').cat.codes
        try:
            df["Duration_Bin"] = pd.qcut(df["TrackDurationMs"], 10, labels=False, duplicates='drop')
        except Exception:
            df["Duration_Bin"] = pd.cut(df["TrackDurationMs"], bins=10, labels=False)
            df["Duration_Bin"] = df["Duration_Bin"].astype(int)
        df["Energy_rank"] = df["Energy"].rank(pct=True)
        df["Rhythm_rank"] = df["RhythmScore"].rank(pct=True)
        if (hasattr(self, 'duration_stats') and self.duration_stats is not None):
            df = df.merge(self.duration_stats, how='left', left_on='Duration_Bin', right_index=True)
            for c in ['dur_bpm_mean', 'dur_bpm_median', 'dur_bpm_std']:
                if c in df.columns:
                    df[c] = df[c].fillna(self.duration_stats[c].mean())
                else:
                    df[c] = self.duration_stats[c].mean()
        else:
            df['dur_bpm_mean'] = 0.0
            df['dur_bpm_median'] = 0.0
            df['dur_bpm_std'] = 0.0
        return df
    
    def prepare_features(self, df, is_train=True):
        df_eng = self.engineer_features(df)
        feature_cols = [col for col in df_eng.columns if col not in ['id', 'BeatsPerMinute']]
        X = df_eng[feature_cols]
        if is_train:
            y = df_eng['BeatsPerMinute']
            return X, y
        else:
            return X
    
    def train_models(self, X_train, y_train):
        print("ğŸ�¯ Training stacking ensemble (log1p target) â€” FULL tuned mode")
        total_start_time = time.time()
        tmp = X_train.copy()
        tmp['BeatsPerMinute'] = y_train.values
        duration_stats = tmp.groupby('Duration_Bin')['BeatsPerMinute'].agg(['mean', 'median', 'std'])
        duration_stats = duration_stats.rename(columns={'mean': 'dur_bpm_mean', 'median': 'dur_bpm_median', 'std': 'dur_bpm_std'})
        self.duration_stats = duration_stats
        X_train = X_train.merge(self.duration_stats, how='left', left_on='Duration_Bin', right_index=True)
        for c in ['dur_bpm_mean', 'dur_bpm_median', 'dur_bpm_std']:
            if c in X_train.columns:
                X_train[c] = X_train[c].fillna(X_train[c].mean())
            else:
                X_train[c] = 0.0
        y_train_log = np.log1p(y_train)
        base_learners = [
            ('xgb', xgb.XGBRegressor(**self.xgb_params)),
            ('lgb', lgb.LGBMRegressor(**self.lgb_params)),
            ('cat', CatBoostRegressor(**self.cat_params)),
            ('rf', RandomForestRegressor(**self.rf_params))
        ]
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        oof_preds = np.zeros((X_train.shape[0], len(base_learners)))
        for idx, (name, _) in enumerate(base_learners):
            print(f"ğŸ”„ OOF training base model: {name}")
            oof = np.zeros(X_train.shape[0])
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                t0 = time.time()
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]
                if name == 'xgb':
                    m = xgb.XGBRegressor(**self.xgb_params)
                    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=self.early_stop, verbose=False)
                elif name == 'lgb':
                    m = lgb.LGBMRegressor(**self.lgb_params)
                    m.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(stopping_rounds=self.early_stop)]
                    )
                elif name == 'cat':
                    m = CatBoostRegressor(**self.cat_params)
                    m.fit(
                        X_tr, y_tr,
                        eval_set=(X_val, y_val),
                        early_stopping_rounds=self.early_stop,
                        verbose=False
                    )
                else:
                    m = RandomForestRegressor(**self.rf_params)
                    m.fit(X_tr, y_tr)
                oof[val_idx] = m.predict(X_val)
                dt = time.time() - t0
                print(f"   Fold {fold + 1}/{self.n_splits} done for {name} in {dt/60:.2f}m")
            oof_preds[:, idx] = oof
            print(f"   Finished OOF for {name}\n")
        print("ğŸ”§ Fitting RidgeCV meta-learner on OOF preds (log-space)...")
        self.meta.fit(oof_preds, y_train_log)
        print("ğŸš€ Training final base models on FULL training set (log-space)...")
        X_full = X_train.copy()
        y_full = y_train_log.copy()
        self.fitted_base_models = {}
        for name, _ in base_learners:
            print(f"   Final fit: {name}")
            if name == 'xgb':
                m = xgb.XGBRegressor(**self.xgb_params)
                m.fit(X_full, y_full, eval_set=[(X_full, y_full)], verbose=False)
            elif name == 'lgb':
                m = lgb.LGBMRegressor(**self.lgb_params)
                m.fit(X_full, y_full)
            elif name == 'cat':
                m = CatBoostRegressor(**self.cat_params)
                m.fit(X_full, y_full, verbose=False)
            else:
                m = RandomForestRegressor(**self.rf_params)
                m.fit(X_full, y_full)
            self.fitted_base_models[name] = m
        stacked_oof_log = self.meta.predict(oof_preds)
        stacked_oof_orig = np.expm1(stacked_oof_log)
        stacked_rmse = np.sqrt(mean_squared_error(y_train.values, stacked_oof_orig))
        total_time = time.time() - total_start_time
        print(f"\nğŸ�¯ STACKED OOF RESULTS:")
        print(f"   Stacked OOF RMSE (original scale): {stacked_rmse:.4f}")
        print(f"â�±ï¸� Training pipeline time: {total_time/60:.2f} minutes")
        return {
            'stacked_oof_rmse': stacked_rmse,
            'total_time_minutes': total_time / 60.0
        }
    
    def predict(self, X_test):
        print("ğŸ”® Making stacked ensemble predictions...")
        X_test = X_test.copy()
        if self.duration_stats is not None:
            X_test = X_test.merge(self.duration_stats, how='left', left_on='Duration_Bin', right_index=True)
            for c in ['dur_bpm_mean', 'dur_bpm_median', 'dur_bpm_std']:
                if c in X_test.columns:
                    X_test[c] = X_test[c].fillna(self.duration_stats[c].mean())
                else:
                    X_test[c] = self.duration_stats[c].mean()
        else:
            X_test['dur_bpm_mean'] = 0.0
            X_test['dur_bpm_median'] = 0.0
            X_test['dur_bpm_std'] = 0.0
        base_preds = []
        for name in ['xgb', 'lgb', 'cat', 'rf']:
            model = self.fitted_base_models.get(name, None)
            if model is None:
                raise RuntimeError(f"Model {name} not trained / not found in fitted_base_models.")
            pred_log = model.predict(X_test)
            base_preds.append(pred_log.reshape(-1, 1))
        base_stack = np.hstack(base_preds)
        meta_pred_log = self.meta.predict(base_stack)
        final_predictions = np.expm1(meta_pred_log)
        print("âœ… Predictions completed!")
        return final_predictions

def main():
    print("ğŸ�µ BPM Prediction Ensemble Pipeline (Stacking with log-target, FULL tuned mode)")
    print("=" * 60)
    print("Loading data...")
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
    sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
    print(f"ğŸ“Š Train shape: {train_df.shape}")
    print(f"ğŸ“Š Test shape: {test_df.shape}")
    ensemble = BPMEnsemblePredictor()
    
    print("\nğŸ”§ Preparing training features...")
    X_train, y_train = ensemble.prepare_features(train_df, is_train=True)
    print(f"âœ… Training features shape: {X_train.shape}")
    print(f"ğŸ“‹ Feature columns: {list(X_train.columns)}")
    
    scores = ensemble.train_models(X_train, y_train)
    
    print("\nğŸ”§ Preparing test features...")
    X_test = ensemble.prepare_features(test_df, is_train=False)
    print(f"âœ… Test features shape: {X_test.shape}")
    
    predictions = ensemble.predict(X_test)
    
    submission = sample_submission.copy()
    submission['BeatsPerMinute'] = predictions
    submission.to_csv('submission.csv', index=False)
    
    print(f"\nğŸ’¾ Submission saved to 'submission.csv'")
    print(f"ğŸ“ˆ Prediction statistics:")
    print(f"  Min: {predictions.min():.2f}")
    print(f"  Max: {predictions.max():.2f}")
    print(f"  Mean: {predictions.mean():.2f}")
    print(f"  Std: {predictions.std():.2f}")
    
    print(f"\nğŸ�‰ Pipeline completed successfully!")
    print(f"â�±ï¸� Total time: {scores['total_time_minutes']:.1f} minutes")
    
    return ensemble, submission, scores

if __name__ == "__main__":
    ensemble, submission, scores = main()


