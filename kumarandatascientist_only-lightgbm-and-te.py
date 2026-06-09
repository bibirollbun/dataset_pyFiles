import pandas as pd
import numpy as np
import time
import warnings
warnings.simplefilter('ignore')

# GPU‐accelerated Target Encoder
from cuml.preprocessing import TargetEncoder

# Model selection, metrics, and utilities
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import pandas.api.types as ptypes  # For checking numeric types
from tqdm.auto import tqdm
import optuna

###############################################################################
# LightGBM-Only Pipeline with Extra Feature Engineering (Fixed Frequency Encoding)
###############################################################################
class LightGBMPipeline:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame,
                 target: str, features: list, cats: list,
                 te_params: dict = None,
                 sample_frac: float = 0.5,
                 random_state: int = 42):
        self.train = train.copy()
        self.test = test.copy()
        self.target = target
        self.features = features
        self.cats = cats
        self.sample_frac = sample_frac
        self.random_state = random_state

        # Default target encoder parameters if none provided.
        if te_params is None:
            te_params = {'n_folds': 25, 'smooth': 20, 'split_method': 'random', 'stat': 'mean'}
        self.te_params = te_params
        self.TE = TargetEncoder(**self.te_params)

        self.lgb_best_params = None
        self.best_cv_rmse_lgb = None
        self.model_lgbm = None
        self.all_features = []  # will be updated during preprocessing

        self.metrics_log = []

    def log(self, message: str):
        print(message)
        self.metrics_log.append(message)

    def save_metrics_log(self, filename="ml_model_metrics_analysis.txt"):
        with open(filename, "w") as f:
            for message in self.metrics_log:
                f.write(message + "\n")
        self.log(f"Metrics log saved to {filename}.")

    def preprocess_data(self):
        start_time = time.time()

        # 1. Apply Target Encoding for each original feature.
        for col in self.features:
            te_col = f"TE_{col}"
            self.train[te_col] = self.TE.fit_transform(self.train[col], self.train[self.target])
            self.test[te_col] = self.TE.transform(self.test[col])
        # Keep track of original + TE features.
        self.all_features = self.features + [f"TE_{col}" for col in self.features]

        # 2. Ensure categorical columns are set properly.
        self.train[self.cats] = self.train[self.cats].fillna('Missing').astype('category')
        self.test[self.cats] = self.test[self.cats].fillna('Missing').astype('category')

        # 3. (Optional) Create interaction features if the total isn’t huge.
        if len(self.cats) * len(self.features) < 50:
            for cat_col in self.cats:
                for num_col in self.features:
                    inter_col = f'{cat_col}_x_{num_col}'
                    self.train[inter_col] = self.train[cat_col].astype(str) + '_' + self.train[num_col].astype(str)
                    self.test[inter_col] = self.test[cat_col].astype(str) + '_' + self.test[num_col].astype(str)
                    # Convert the new interaction feature to numeric codes.
                    self.train[inter_col] = self.train[inter_col].astype('category').cat.codes
                    self.test[inter_col] = self.test[inter_col].astype('category').cat.codes
                    self.all_features.append(inter_col)

        # 4. EXTRA FEATURE ENGINEERING

        # 4.1 Frequency Encoding for categorical features.
        # Convert the categorical columns to string before mapping to avoid the TypeError.
        for col in self.cats:
            freq_enc_col = f'freq_{col}'
            # Calculate frequency based on string representation.
            freq = self.train[col].astype(str).value_counts(normalize=True)
            self.train[freq_enc_col] = self.train[col].astype(str).map(freq)
            self.test[freq_enc_col] = self.test[col].astype(str).map(freq).fillna(0)
            self.all_features.append(freq_enc_col)

        # 4.2 Polynomial Features for numeric columns (squared and log transformations).
        # Loop over the original features.
        for col in self.features:
            if ptypes.is_numeric_dtype(self.train[col]):
                # Squared feature
                squared_col = f'{col}_squared'
                self.train[squared_col] = self.train[col] ** 2
                self.test[squared_col] = self.test[col] ** 2
                self.all_features.append(squared_col)
                
                # Log-transformation (only if all values are positive)
                if (self.train[col] > 0).all():
                    log_col = f'log_{col}'
                    self.train[log_col] = np.log1p(self.train[col])
                    self.test[log_col] = np.log1p(self.test[col])
                    self.all_features.append(log_col)

        # 5. Convert non-numeric features (in our feature set) to numeric codes.
        for col in self.all_features:
            if not ptypes.is_numeric_dtype(self.train[col]):
                self.train[col] = self.train[col].astype('category').cat.codes
                self.test[col] = self.test[col].astype('category').cat.codes

        # 6. Fill missing values (using the median) and scale features.
        for col in self.all_features:
            median_val = self.train[col].median()
            self.train[col] = self.train[col].fillna(median_val)
            self.test[col] = self.test[col].fillna(median_val)
        scaler = StandardScaler()
        self.train[self.all_features] = scaler.fit_transform(self.train[self.all_features])
        self.test[self.all_features] = scaler.transform(self.test[self.all_features])

        elapsed = time.time() - start_time
        self.log(f"Preprocessing complete. (Time taken: {elapsed:.2f} sec)")
        self.log(f"All features created: {self.all_features}")

    def tune_lightgbm(self, n_trials: int = 10):
        """
        Use Optuna to tune hyperparameters for LightGBM on a subsample with 2-fold CV.
        """
        self.log("Starting hyperparameter tuning with Optuna for LightGBM...")
        # Use a smaller subsample to speed up tuning.
        train_sample = self.train.sample(frac=0.2, random_state=self.random_state)

        def objective_lgb(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "n_estimators": trial.suggest_int("n_estimators", 300, 800),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
                "device": "gpu"
            }
            cv = KFold(n_splits=2, shuffle=True, random_state=self.random_state)
            cv_scores = []
            for train_idx, val_idx in cv.split(train_sample):
                X_train_cv = train_sample.iloc[train_idx][self.all_features]
                y_train_cv = train_sample.iloc[train_idx][self.target]
                X_val_cv = train_sample.iloc[val_idx][self.all_features]
                y_val_cv = train_sample.iloc[val_idx][self.target]

                model = lgb.LGBMRegressor(random_state=self.random_state, **params)
                model.fit(
                    X_train_cv, y_train_cv,
                    eval_set=[(X_val_cv, y_val_cv)],
                    eval_metric='rmse',
                    callbacks=[lgb.early_stopping(20)]
                )
                preds = model.predict(X_val_cv)
                rmse = np.sqrt(mean_squared_error(y_val_cv, preds))
                cv_scores.append(rmse)
            return np.mean(cv_scores)

        study_lgb = optuna.create_study(direction="minimize")
        study_lgb.optimize(objective_lgb, n_trials=n_trials)
        self.lgb_best_params = study_lgb.best_trial.params
        self.best_cv_rmse_lgb = study_lgb.best_value
        self.log(f"LightGBM tuning complete. Best params: {self.lgb_best_params}")
        self.log(f"Best CV RMSE (LightGBM): {self.best_cv_rmse_lgb:.4f}")

    def train_final_model(self, early_stopping_rounds: int = 20):
        """
        Train the final LightGBM model using a hold-out split.
        """
        start_time = time.time()
        X_train, X_val, y_train, y_val = train_test_split(
            self.train[self.all_features], self.train[self.target],
            test_size=0.2, random_state=self.random_state
        )

        self.log("Training final LightGBM model...")
        lgb_params = self.lgb_best_params.copy() if self.lgb_best_params is not None else {}
        lgb_params["device"] = "gpu"  # Ensure GPU is used if available
        self.model_lgbm = lgb.LGBMRegressor(random_state=self.random_state, **lgb_params)
        self.model_lgbm.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(early_stopping_rounds)]
        )
        preds_lgbm = self.model_lgbm.predict(X_val)
        rmse_lgbm = np.sqrt(mean_squared_error(y_val, preds_lgbm))
        self.log(f"Final model hold-out RMSE: {rmse_lgbm:.4f}")
        elapsed = time.time() - start_time
        self.log(f"Final model training complete. (Time taken: {elapsed:.2f} sec)")

    def predict_test(self):
        """
        Use the final LightGBM model to predict on the test set.
        """
        start_time = time.time()
        test_preds = self.model_lgbm.predict(self.test[self.all_features])
        elapsed = time.time() - start_time
        self.log(f"Test prediction complete. (Time taken: {elapsed:.2f} sec)")
        return test_preds

    def save_submission(self, predictions, filename="submission.csv"):
        start_time = time.time()
        sub = pd.DataFrame({ "id": self.test.index, self.target: predictions })
        sub.to_csv(filename, index=False)
        elapsed = time.time() - start_time
        self.log(f"Submission saved to {filename}. (Time taken: {elapsed:.2f} sec)")

    def run_pipeline(self):
        overall_start = time.time()
        steps = [
            ("Preprocessing Data", self.preprocess_data),
            ("Hyperparameter Tuning (LightGBM)", self.tune_lightgbm),
            ("Training Final Model (LightGBM)", self.train_final_model),
        ]
        
        with tqdm(total=len(steps), desc="Pipeline Steps", unit="step") as pbar:
            self.log("Starting pipeline execution...")
            for step_name, step_func in steps:
                self.log(f"Starting step: {step_name}")
                step_func()
                pbar.update(1)
                self.log(f"Completed step: {step_name}")
        
        self.log("Predicting Test Set...")
        predictions = self.predict_test()
        self.log("Saving Submission...")
        self.save_submission(predictions)
        self.save_metrics_log()
        
        overall_elapsed = time.time() - overall_start
        self.log("Pipeline execution complete.")
        self.log(f"Total pipeline time: {overall_elapsed:.2f} sec")



###############################################################################
# TE & LGBM pipeline Usage
###############################################################################

if __name__ == "__main__":
    # Load the datasets.
    train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv", index_col='id')
    train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col='id')
    train = pd.concat([train, train_extra], axis=0, ignore_index=True)
    test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv", index_col='id')

    # Define target and feature columns.
    target = "Price"
    features = [col for col in train.columns if col != target]
    
    # Define categorical columns (example: all columns except Price and Weight Capacity)
    cats = [col for col in train.columns if col not in [target, "Weight Capacity (kg)"]]
    
    # Initialize and run the pipeline.
    pipeline = LightGBMPipeline(train=train, test=test, target=target, features=features, cats=cats)
    pipeline.run_pipeline()



import pandas as pd

df = pd.read_csv('submission.csv')
df.head(10)


df['Price']= df['Price']*0.99541193271848074
df.to_csv('submission.csv',index=False)
df.head(10)




