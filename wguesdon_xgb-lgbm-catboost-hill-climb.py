import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from scipy.optimize import differential_evolution
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
np.random.seed(SEED)
DATA_PATH = '/kaggle/input/predicting-the-price-of-diamond/'

class DiamondPricePredictor:
    """Advanced diamond price prediction with ensemble learning and GPU acceleration.
    
    Attributes:
        n_folds: Number of folds for cross-validation.
        use_gpu: Whether to use GPU acceleration.
        models: Dictionary storing trained models.
        label_encoders: Dictionary of label encoders for categorical features.
        ensemble_weights: Optimized weights for model ensemble.
        cut_order: Ordered cut quality mapping.
        color_order: Ordered color mapping.
        clarity_order: Ordered clarity mapping.
    """
    
    def __init__(self, n_folds=5, use_gpu=True):
        """Initializes the DiamondPricePredictor.
        
        Args:
            n_folds: Number of cross-validation folds.
            use_gpu: Whether to use GPU acceleration.
        """
        self.n_folds = n_folds
        self.use_gpu = use_gpu
        self.models = {}
        self.label_encoders = {}
        self.ensemble_weights = None
        
        self.cut_order = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
        self.color_order = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
        self.clarity_order = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}
        
    def load_data(self):
        """Loads and prepares the data from Kaggle input directory.
        
        Returns:
            Self for method chaining.
        """
        print("Loading data...")
        self.train_df = pd.read_csv(f'{DATA_PATH}train.csv')
        self.test_df = pd.read_csv(f'{DATA_PATH}test.csv')
        self.submission_df = pd.read_csv(f'{DATA_PATH}submission.csv')
        
        print(f"Train shape: {self.train_df.shape}")
        print(f"Test shape: {self.test_df.shape}")
        
        self.X_train = self.train_df.drop(['id', 'price'], axis=1)
        self.y_train = np.log1p(self.train_df['price'])
        self.X_test = self.test_df.drop(['id'], axis=1)
        
        return self
    
    def feature_engineering(self, df):
        """Creates comprehensive features from existing ones.
        
        Args:
            df: Input dataframe with diamond features.
            
        Returns:
            DataFrame with engineered features.
        """
        df = df.copy()
        
        df['cut_num'] = df['cut'].map(self.cut_order).fillna(0)
        df['color_num'] = df['color'].map(self.color_order).fillna(0)
        df['clarity_num'] = df['clarity'].map(self.clarity_order).fillna(0)
        
        df['quality_score'] = df['cut_num'] * df['color_num'] * df['clarity_num']
        df['quality_avg'] = (df['cut_num'] + df['color_num'] + df['clarity_num']) / 3
        df['premium_score'] = df['cut_num'] * df['clarity_num']
        
        df['volume'] = df['x'] * df['y'] * df['z']
        df['volume_carat_ratio'] = df['volume'] / (df['carat'] + 0.001)
        df['density'] = df['carat'] / (df['volume'] + 0.001)
        
        df['table_depth_ratio'] = df['table'] / (df['depth'] + 0.001)
        df['table_squared'] = df['table'] ** 2
        df['depth_squared'] = df['depth'] ** 2
        
        df['xy_ratio'] = df['x'] / (df['y'] + 0.001)
        df['xz_ratio'] = df['x'] / (df['z'] + 0.001)
        df['yz_ratio'] = df['y'] / (df['z'] + 0.001)
        
        df['avg_dim'] = (df['x'] + df['y'] + df['z']) / 3
        df['max_dim'] = df[['x', 'y', 'z']].max(axis=1)
        df['min_dim'] = df[['x', 'y', 'z']].min(axis=1)
        df['dim_range'] = df['max_dim'] - df['min_dim']
        
        df['surface_area'] = 2 * (df['x']*df['y'] + df['x']*df['z'] + df['y']*df['z'])
        df['surface_volume_ratio'] = df['surface_area'] / (df['volume'] + 0.001)
        
        df['carat_squared'] = df['carat'] ** 2
        df['carat_cubed'] = df['carat'] ** 3
        df['carat_sqrt'] = np.sqrt(df['carat'])
        df['carat_log'] = np.log1p(df['carat'])
        
        df['carat_per_dim'] = df['carat'] / (df['avg_dim'] + 0.001)
        
        ideal_depth = 61.5
        ideal_table = 57.5
        df['depth_deviation'] = np.abs(df['depth'] - ideal_depth)
        df['table_deviation'] = np.abs(df['table'] - ideal_table)
        df['ideal_deviation'] = df['depth_deviation'] + df['table_deviation']
        
        df['dimension_std'] = df[['x', 'y', 'z']].std(axis=1)
        df['dimension_var'] = df[['x', 'y', 'z']].var(axis=1)
        
        df['is_perfect_cut'] = ((df['depth'] >= 60) & (df['depth'] <= 63) & 
                                (df['table'] >= 56) & (df['table'] <= 59)).astype(int)
        
        df['carat_bin'] = pd.cut(df['carat'], 
                                 bins=[0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 10.0],
                                 labels=False, include_lowest=True).fillna(0)
        
        df['volume_bin'] = pd.qcut(df['volume'], q=10, labels=False, duplicates='drop').fillna(0)
        
        df['x_y_diff'] = np.abs(df['x'] - df['y'])
        df['symmetry_score'] = 1 / (1 + df['x_y_diff'] + df['dimension_std'])
        
        df['brilliance_score'] = (df['cut_num'] * df['clarity_num']) / (df['depth_deviation'] + 1)
        
        df['price_indicator'] = (df['carat_squared'] * df['quality_score'] * df['volume']) / 100
        
        return df
    
    def create_interactions(self, df):
        """Creates interaction features between key variables.
        
        Args:
            df: DataFrame with features.
            
        Returns:
            DataFrame with interaction features.
        """
        df['carat_cut_interaction'] = df['carat'] * df['cut_num']
        df['carat_color_interaction'] = df['carat'] * df['color_num']
        df['carat_clarity_interaction'] = df['carat'] * df['clarity_num']
        df['volume_quality_interaction'] = df['volume'] * df['quality_score']
        df['surface_quality_interaction'] = df['surface_area'] * df['quality_avg']
        
        return df
    
    def prepare_features(self):
        """Prepares features with comprehensive engineering.
        
        Returns:
            Self for method chaining.
        """
        print("Engineering features...")
        
        self.X_train_fe = self.feature_engineering(self.X_train)
        self.X_train_fe = self.create_interactions(self.X_train_fe)
        
        self.X_test_fe = self.feature_engineering(self.X_test)
        self.X_test_fe = self.create_interactions(self.X_test_fe)
        
        cols_to_drop = ['cut', 'color', 'clarity']
        self.X_train_fe = self.X_train_fe.drop(columns=cols_to_drop)
        self.X_test_fe = self.X_test_fe.drop(columns=cols_to_drop)
        
        self.X_train_encoded = self.X_train_fe
        self.X_test_encoded = self.X_test_fe
        
        print(f"Final feature count: {self.X_train_encoded.shape[1]}")
        
        return self
    
    def optimize_xgboost_params(self, X_train, y_train, X_val, y_val):
        """Optimizes XGBoost hyperparameters using Optuna.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.
            
        Returns:
            Best parameters dictionary.
        """
        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'max_depth': trial.suggest_int('max_depth', 6, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'n_estimators': 1000,
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': SEED,
                'tree_method': 'gpu_hist' if self.use_gpu else 'auto',
                'predictor': 'gpu_predictor' if self.use_gpu else 'auto',
                'gpu_id': 0,
            }
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     early_stopping_rounds=50,
                     verbose=False)
            
            preds = model.predict(X_val)
            return r2_score(y_val, preds)
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        
        return study.best_params
    
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """Trains XGBoost model with GPU support.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Optional validation features.
            y_val: Optional validation target.
            params: Optional parameters dictionary.
            
        Returns:
            Trained XGBoost model.
        """
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 10,
                'learning_rate': 0.02,
                'n_estimators': 3000,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'reg_alpha': 0.5,
                'reg_lambda': 2,
                'min_child_weight': 3,
                'random_state': SEED,
                'tree_method': 'gpu_hist' if self.use_gpu else 'auto',
                'predictor': 'gpu_predictor' if self.use_gpu else 'auto',
                'gpu_id': 0,
                'n_jobs': -1
            }
        else:
            base_params = {
                'objective': 'reg:squarederror',
                'n_estimators': 3000,
                'random_state': SEED,
                'tree_method': 'gpu_hist' if self.use_gpu else 'auto',
                'predictor': 'gpu_predictor' if self.use_gpu else 'auto',
                'gpu_id': 0,
                'n_jobs': -1
            }
            params = {**base_params, **params}
        
        model = xgb.XGBRegressor(**params)
        
        if X_val is not None:
            model.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     early_stopping_rounds=100,
                     verbose=False)
        else:
            model.fit(X_train, y_train)
        
        return model
    
    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None):
        """Trains LightGBM model with optimized parameters.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Optional validation features.
            y_val: Optional validation target.
            
        Returns:
            Trained LightGBM model.
        """
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 100,
            'learning_rate': 0.02,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 5,
            'reg_alpha': 0.5,
            'reg_lambda': 2,
            'min_child_samples': 20,
            'n_estimators': 3000,
            'random_state': SEED,
            'device': 'gpu' if self.use_gpu else 'cpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'n_jobs': -1,
            'verbose': -1
        }
        
        model = lgb.LGBMRegressor(**params)
        
        if X_val is not None:
            model.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        else:
            model.fit(X_train, y_train)
        
        return model
    
    def train_catboost(self, X_train, y_train, X_val=None, y_val=None):
        """Trains CatBoost model with optimized parameters.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Optional validation features.
            y_val: Optional validation target.
            
        Returns:
            Trained CatBoost model.
        """
        params = {
            'iterations': 3000,
            'learning_rate': 0.02,
            'depth': 10,
            'l2_leaf_reg': 5,
            'loss_function': 'RMSE',
            'random_seed': SEED,
            'task_type': 'GPU' if self.use_gpu else 'CPU',
            'devices': '0' if self.use_gpu else None,
            'verbose': False
        }
        
        if not self.use_gpu:
            params.pop('devices')
        
        model = cb.CatBoostRegressor(**params)
        
        if X_val is not None:
            model.fit(X_train, y_train,
                     eval_set=(X_val, y_val),
                     early_stopping_rounds=100,
                     verbose=False)
        else:
            model.fit(X_train, y_train, verbose=False)
        
        return model
    
    def cross_validate(self, optimize_hyperparams=False):
        """Performs cross-validation for all models.
        
        Args:
            optimize_hyperparams: Whether to optimize hyperparameters.
        
        Returns:
            Self for method chaining.
        """
        print("\nPerforming cross-validation...")
        
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=SEED)
        
        oof_preds = {
            'xgboost': np.zeros(len(self.X_train_encoded)),
            'lightgbm': np.zeros(len(self.X_train_encoded)),
            'catboost': np.zeros(len(self.X_train_encoded))
        }
        
        test_preds = {
            'xgboost': [],
            'lightgbm': [],
            'catboost': []
        }
        
        best_xgb_params = None
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.X_train_encoded), 1):
            print(f"\nFold {fold}/{self.n_folds}")
            
            X_tr, X_val = self.X_train_encoded.iloc[train_idx], self.X_train_encoded.iloc[val_idx]
            y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            
            if optimize_hyperparams and fold == 1:
                print("  Optimizing XGBoost hyperparameters...")
                best_xgb_params = self.optimize_xgboost_params(X_tr, y_tr, X_val, y_val)
                print(f"  Best params: {best_xgb_params}")
            
            print("  Training XGBoost...")
            xgb_model = self.train_xgboost(X_tr, y_tr, X_val, y_val, best_xgb_params)
            oof_preds['xgboost'][val_idx] = xgb_model.predict(X_val)
            test_preds['xgboost'].append(xgb_model.predict(self.X_test_encoded))
            
            print("  Training LightGBM...")
            lgb_model = self.train_lightgbm(X_tr, y_tr, X_val, y_val)
            oof_preds['lightgbm'][val_idx] = lgb_model.predict(X_val)
            test_preds['lightgbm'].append(lgb_model.predict(self.X_test_encoded))
            
            print("  Training CatBoost...")
            cb_model = self.train_catboost(X_tr, y_tr, X_val, y_val)
            oof_preds['catboost'][val_idx] = cb_model.predict(X_val)
            test_preds['catboost'].append(cb_model.predict(self.X_test_encoded))
            
            for name, preds in oof_preds.items():
                if preds[val_idx].sum() != 0:
                    score = r2_score(y_val, preds[val_idx])
                    print(f"  {name} R2: {score:.4f}")
        
        self.test_preds_avg = {}
        for name, preds_list in test_preds.items():
            self.test_preds_avg[name] = np.mean(preds_list, axis=0)
        
        print("\nOverall CV Scores:")
        self.cv_scores = {}
        for name, preds in oof_preds.items():
            score = r2_score(self.y_train, preds)
            self.cv_scores[name] = score
            print(f"  {name}: R2 = {score:.4f}")
        
        self.oof_preds = oof_preds
        self.best_xgb_params = best_xgb_params
        
        return self
    
    def hill_climbing_ensemble(self, max_iterations=300):
        """Performs advanced hill climbing for ensemble optimization.
        
        Args:
            max_iterations: Maximum iterations for optimization.
            
        Returns:
            Self for method chaining.
        """
        print("\nPerforming hill climbing ensemble optimization...")
        
        def objective(weights):
            """Objective function to minimize (negative R2).
            
            Args:
                weights: Model weights array.
                
            Returns:
                Negative R2 score.
            """
            weights = weights / weights.sum()
            ensemble_pred = sum(w * self.oof_preds[m] 
                              for w, m in zip(weights, self.oof_preds.keys()))
            return -r2_score(self.y_train, ensemble_pred)
        
        n_models = len(self.oof_preds)
        best_weights = np.ones(n_models) / n_models
        best_score = -objective(best_weights)
        
        for restart in range(15):
            if restart == 0:
                current_weights = best_weights.copy()
            else:
                current_weights = np.random.dirichlet(np.ones(n_models) * 0.5)
            
            current_score = -objective(current_weights)
            
            for iteration in range(max_iterations // 15):
                for _ in range(30):
                    perturbation = np.random.normal(0, 0.01, n_models)
                    new_weights = current_weights + perturbation
                    new_weights = np.clip(new_weights, 0.001, 1)
                    new_weights = new_weights / new_weights.sum()
                    
                    new_score = -objective(new_weights)
                    
                    if new_score > current_score:
                        current_weights = new_weights
                        current_score = new_score
                        
                        if new_score > best_score:
                            best_weights = new_weights
                            best_score = new_score
        
        bounds = [(0.001, 1)] * n_models
        result = differential_evolution(objective, bounds, seed=SEED, maxiter=200, popsize=30)
        
        de_weights = result.x / result.x.sum()
        de_score = -result.fun
        
        if de_score > best_score:
            best_weights = de_weights
            best_score = de_score
        
        self.ensemble_weights = best_weights
        
        print(f"\nOptimal ensemble weights:")
        for name, weight in zip(self.oof_preds.keys(), self.ensemble_weights):
            print(f"  {name}: {weight:.4f}")
        print(f"Ensemble CV R2 Score: {best_score:.4f}")
        
        return self
    
    def train_final_models(self):
        """Trains final models on full training data.
        
        Returns:
            Self for method chaining.
        """
        print("\nTraining final models on full data...")
        
        self.final_models = {}
        
        print("Training final XGBoost...")
        self.final_models['xgboost'] = self.train_xgboost(
            self.X_train_encoded, self.y_train, params=self.best_xgb_params
        )
        
        print("Training final LightGBM...")
        self.final_models['lightgbm'] = self.train_lightgbm(
            self.X_train_encoded, self.y_train
        )
        
        print("Training final CatBoost...")
        self.final_models['catboost'] = self.train_catboost(
            self.X_train_encoded, self.y_train
        )
        
        return self
    
    def make_predictions(self):
        """Makes final ensemble predictions with inverse transform.
        
        Returns:
            Tuple of predictions.
        """
        print("\nMaking final predictions...")
        
        final_preds = {}
        for name, model in self.final_models.items():
            final_preds[name] = model.predict(self.X_test_encoded)
        
        if self.ensemble_weights is not None:
            ensemble_pred = sum(w * final_preds[m] 
                              for w, m in zip(self.ensemble_weights, final_preds.keys()))
        else:
            ensemble_pred = np.mean(list(final_preds.values()), axis=0)
        
        simple_avg_pred = np.mean(list(final_preds.values()), axis=0)
        
        ensemble_pred = np.expm1(ensemble_pred)
        simple_avg_pred = np.expm1(simple_avg_pred)
        
        individual_preds = {}
        for name, preds in final_preds.items():
            individual_preds[name] = np.expm1(preds)
        
        return ensemble_pred, simple_avg_pred, individual_preds
    
    def create_submission(self, predictions, filename='submission_ensemble.csv'):
        """Creates submission file.
        
        Args:
            predictions: Array of price predictions.
            filename: Output filename.
            
        Returns:
            Submission DataFrame.
        """
        submission = self.submission_df.copy()
        submission['price'] = predictions
        submission.to_csv(filename, index=False)
        print(f"\nSubmission saved to {filename}")
        
        print(f"Prediction statistics:")
        print(f"  Mean: {predictions.mean():.2f}")
        print(f"  Std:  {predictions.std():.2f}")
        print(f"  Min:  {predictions.min():.2f}")
        print(f"  Max:  {predictions.max():.2f}")
        
        return submission

def main():
    """Main execution function with advanced optimization.
    
    Returns:
        Trained DiamondPricePredictor instance.
    """
    predictor = DiamondPricePredictor(n_folds=5, use_gpu=True)
    
    predictor.load_data()
    predictor.prepare_features()
    predictor.cross_validate(optimize_hyperparams=True)
    predictor.hill_climbing_ensemble(max_iterations=300)
    predictor.train_final_models()
    
    ensemble_pred, simple_avg_pred, individual_preds = predictor.make_predictions()
    
    predictor.create_submission(ensemble_pred, 'submission_hill_climbing.csv')
    predictor.create_submission(simple_avg_pred, 'submission_simple_avg.csv')
    
    for name, preds in individual_preds.items():
        predictor.create_submission(preds, f'submission_{name}.csv')
    
    print("\n" + "="*50)
    print("Training complete! Files created:")
    print("  - submission_hill_climbing.csv (main submission)")
    print("  - submission_simple_avg.csv")
    print("  - submission_xgboost.csv")
    print("  - submission_lightgbm.csv")
    print("  - submission_catboost.csv")
    
    return predictor

if __name__ == "__main__":
    predictor = main()

