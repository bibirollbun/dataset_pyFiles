import os
import random
import numpy as np
import pandas as pd
from cuml.preprocessing import TargetEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
import optuna
import pickle
from joblib import Parallel, delayed
from sklearn.base import clone

class TwoBaselineMLModelWorkflow:
    """
    A two-baseline ML workflow that:
      - Loads and concatenates training data from two sources.
      - Computes a simple mean-based baseline.
      - Performs advanced feature engineering:
            * For extra categorical features (e.g., "Brand", "Material", etc.):
                  - Creates numeric code versions.
                  - Creates count encoding features.
                  - Creates target‑encoded versions.
                  - Creates interaction features between "Weight Capacity (kg)" and the target‑encoded values.
            * For "Weight Capacity (kg)":
                  - Creates polynomial features (square and cube).
                  - Creates its target‑encoded version (te_weight) and target‑encoded polynomial features (te_sq, te_cube).
      - Builds a final feature set from all these engineered features.
      - Tunes a regularized linear regression model (Ridge, Lasso, or ElasticNet) using (a sample of) CV and Optuna.
      - Trains the final model on full training data.
      - Processes the test data with the same transformations and writes a submission CSV.
      - Saves the entire pretrained pipeline to a file and can later load it for inference.
      
    TQDM progress bars report progress for each major step.
    """
    
    def __init__(self, train_file, train_extra_file, test_file, submission_file):
        self.train_file = train_file
        self.train_extra_file = train_extra_file
        self.test_file = test_file
        self.submission_file = submission_file
        self.SEED = 42
        
        # Placeholders for data and models
        self.train = None
        self.test = None
        self.final_model = None
        
        # Fixed target encoding hyperparameters (from previous tuning)
        self.best_n_folds = 28
        self.best_smooth = 21.57
        self.best_split_method = 'continuous'
        
        # Transformers & Encoders for weight capacity feature engineering
        self.TE_weight = None
        self.poly2 = PolynomialFeatures(degree=2, include_bias=False)
        self.poly3 = PolynomialFeatures(degree=3, include_bias=False)
        self.TE_sq = None
        self.TE_cube = None
        
        # Extra categorical features to process
        self.cat_features = ['Brand', 'Material', 'Size', 'Compartments', 
                             'Laptop Compartment', 'Waterproof', 'Style', 'Color']
        # Dictionary to hold target encoders for each extra categorical feature
        self.cat_TE = {}
    
    def load_data(self):
        """Load and combine training data from two files, and load test data."""
        tqdm.write("Loading training data...")
        train_main = pd.read_csv(self.train_file)
        tqdm.write(f"Train shape: {train_main.shape}")
        
        train_extra = pd.read_csv(self.train_extra_file)
        tqdm.write(f"Extra Train shape: {train_extra.shape}")
        
        self.train = pd.concat([train_main, train_extra], axis=0, ignore_index=True)
        tqdm.write(f"Combined Train shape: {self.train.shape}")
        
        self.test = pd.read_csv(self.test_file)
        tqdm.write(f"Test shape: {self.test.shape}")
    
    def baseline1(self):
        """Compute a simple baseline using the training Price mean."""
        tqdm.write("Running Baseline 1: Train Mean baseline ...")
        train_mean = self.train['Price'].mean()
        self.train['pred_baseline'] = train_mean
        rmse = np.sqrt(np.mean((self.train['Price'] - self.train['pred_baseline']) ** 2))
        tqdm.write(f"Validation RMSE using Train Mean = {rmse:.4f}")
    
    def optimized_baseline(self):
        """
        Run the optimized baseline workflow:
          1. Feature Engineering & Target Encoding of "Weight Capacity (kg)" (with polynomial features)
             and extra categorical features.
          2. Hyperparameter tuning (via Optuna) for a regularized linear regression model.
          3. Final model training on full training data.
        """
        tqdm.write("Starting Optimized Baseline workflow ...")
        # --- Step 1: Feature Engineering on Training Data ---
        tqdm.write("Step 1: Feature Engineering & Target Encoding on training data")
        
        # Impute missing values for "Weight Capacity (kg)"
        if self.train["Weight Capacity (kg)"].isna().sum() > 0:
            self.train["Weight Capacity (kg)"] = self.train["Weight Capacity (kg)"].fillna(
                self.train["Weight Capacity (kg)"].mean()
            )
        
        # Process extra categorical features:
        for col in self.cat_features:
            self.train[col] = self.train[col].fillna("Missing").astype(str)
            # Numeric code version
            self.train[col + "_code"] = self.train[col].astype('category').cat.codes
            # Count encoding
            self.train[col + "_count"] = self.train[col].map(self.train[col].value_counts())
            # Target encoding
            te = TargetEncoder(n_folds=self.best_n_folds, smooth=self.best_smooth,
                               split_method=self.best_split_method, stat='mean')
            self.train["TE_" + col] = te.fit_transform(self.train[col], self.train['Price'])
            self.cat_TE[col] = te
            # Interaction: Multiply original Weight Capacity with target encoded categorical feature
            self.train["WCap_x_TE_" + col] = self.train["Weight Capacity (kg)"] * self.train["TE_" + col]
        
        # Process "Weight Capacity (kg)":
        # (a) Target encode the original "Weight Capacity (kg)"
        self.TE_weight = TargetEncoder(n_folds=self.best_n_folds,
                                       smooth=self.best_smooth,
                                       split_method=self.best_split_method,
                                       stat='mean')
        self.train['te_weight'] = self.TE_weight.fit_transform(self.train['Weight Capacity (kg)'], self.train['Price'])
        
        # (b) Generate polynomial features for "Weight Capacity (kg)"
        weight_poly2 = self.poly2.fit_transform(self.train[['Weight Capacity (kg)']])
        weight_poly3 = self.poly3.fit_transform(self.train[['Weight Capacity (kg)']])
        self.train['WeightCapacity_sq'] = weight_poly2[:, 1]      # square
        self.train['WeightCapacity_cube'] = weight_poly3[:, 2]      # cube
        
        # (c) Target encode the squared and cubic features
        self.TE_sq = TargetEncoder(n_folds=self.best_n_folds,
                                   smooth=self.best_smooth,
                                   split_method=self.best_split_method,
                                   stat='mean')
        self.TE_cube = TargetEncoder(n_folds=self.best_n_folds,
                                     smooth=self.best_smooth,
                                     split_method=self.best_split_method,
                                     stat='mean')
        self.train['te_sq'] = self.TE_sq.fit_transform(self.train['WeightCapacity_sq'], self.train['Price'])
        self.train['te_cube'] = self.TE_cube.fit_transform(self.train['WeightCapacity_cube'], self.train['Price'])
        
        # --- Prepare Final Feature Matrix ---
        orig_cat_features = [col + "_code" for col in self.cat_features]
        count_cat_features = [col + "_count" for col in self.cat_features]
        weight_feature = ["Weight Capacity (kg)"]
        te_cat_features = ["TE_" + col for col in self.cat_features]
        inter_cat_features = ["WCap_x_TE_" + col for col in self.cat_features]
        weight_features = ["te_weight", "te_sq", "te_cube"]
        
        final_features = orig_cat_features + count_cat_features + weight_feature + te_cat_features + inter_cat_features + weight_features
        tqdm.write("Final features used for model training:")
        tqdm.write(str(final_features))
        
        self.X = self.train[final_features]
        self.y = self.train['Price']
        
        # --- Step 2: Hyperparameter Tuning using Optuna (Parallelized CV on a sample) ---
        tqdm.write("Step 2: Hyperparameter Tuning with Optuna ...")
        def objective(trial):
            regressor_choice = trial.suggest_categorical("regressor", ["ridge", "lasso", "elasticnet"])
            alpha = trial.suggest_float("alpha", 1e-3, 1e3, log=True)
            if regressor_choice == "elasticnet":
                l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=self.SEED, max_iter=10000)
            elif regressor_choice == "ridge":
                model = Ridge(alpha=alpha, random_state=self.SEED, max_iter=10000)
            else:
                model = Lasso(alpha=alpha, random_state=self.SEED, max_iter=10000)
            
            # Use a sample of the data for tuning to speed up execution
            sample_frac = 0.1
            X_sample = self.X.sample(frac=sample_frac, random_state=self.SEED)
            y_sample = self.y.loc[X_sample.index]
            
            kf = KFold(n_splits=3, shuffle=True, random_state=self.SEED)
            
            def evaluate_fold(train_index, val_index):
                X_train_cv = X_sample.iloc[train_index]
                X_val_cv = X_sample.iloc[val_index]
                y_train_cv = y_sample.iloc[train_index]
                y_val_cv = y_sample.iloc[val_index]
                local_model = clone(model)
                local_model.fit(X_train_cv, y_train_cv)
                preds = local_model.predict(X_val_cv)
                return np.sqrt(mean_squared_error(y_val_cv, preds))
            
            cv_rmse = Parallel(n_jobs=-1, backend="threading")(
                delayed(evaluate_fold)(train_index, val_index) for train_index, val_index in kf.split(X_sample)
            )
            return np.mean(cv_rmse)
        
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.SEED))
        study.optimize(objective, n_trials=20, show_progress_bar=True)
        
        tqdm.write("Best hyperparameters:")
        best_trial = study.best_trial
        for key, value in best_trial.params.items():
            tqdm.write(f"  {key}: {value}")
        self.best_params = best_trial.params
        
        # --- Step 3: Final Model Training on Full Training Data ---
        tqdm.write("Step 3: Training final model on full training data ...")
        regressor_choice = self.best_params["regressor"]
        alpha = self.best_params["alpha"]
        if regressor_choice == "elasticnet":
            l1_ratio = self.best_params["l1_ratio"]
            self.final_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=self.SEED, max_iter=10000)
        elif regressor_choice == "ridge":
            self.final_model = Ridge(alpha=alpha, random_state=self.SEED, max_iter=10000)
        else:
            self.final_model = Lasso(alpha=alpha, random_state=self.SEED, max_iter=10000)
        
        self.final_model.fit(self.X, self.y)
        final_train_pred = self.final_model.predict(self.X)
        final_train_rmse = np.sqrt(mean_squared_error(self.y, final_train_pred))
        tqdm.write(f"Final training RMSE: {final_train_rmse:.4f}")
        
        # Save the pretrained pipeline
        self.save_pipeline()
    
    def process_test_and_generate_submission(self):
        """
        Process the test data using the same feature engineering and target encoding
        steps, predict Price using the final model, and generate the submission file.
        """
        tqdm.write("Processing test data and generating submission ...")
        test = pd.read_csv(self.test_file)
        
        if test["Weight Capacity (kg)"].isna().sum() > 0:
            test["Weight Capacity (kg)"].fillna(self.train["Weight Capacity (kg)"].mean(), inplace=True)
        
        for col in self.cat_features:
            test[col] = test[col].fillna("Missing").astype(str)
            test[col + "_code"] = test[col].astype('category').cat.codes
            test[col + "_count"] = test[col].map(self.train[col].value_counts())
            test["TE_" + col] = self.cat_TE[col].transform(test[col])
            test["WCap_x_TE_" + col] = test["Weight Capacity (kg)"] * test["TE_" + col]
        
        test['te_weight'] = self.TE_weight.transform(test['Weight Capacity (kg)'])
        weight_poly2_test = self.poly2.transform(test[['Weight Capacity (kg)']])
        weight_poly3_test = self.poly3.transform(test[['Weight Capacity (kg)']])
        test['WeightCapacity_sq'] = weight_poly2_test[:, 1]
        test['WeightCapacity_cube'] = weight_poly3_test[:, 2]
        test['te_sq'] = self.TE_sq.transform(test['WeightCapacity_sq'])
        test['te_cube'] = self.TE_cube.transform(test['WeightCapacity_cube'])
        
        orig_cat_features = [col + "_code" for col in self.cat_features]
        count_cat_features = [col + "_count" for col in self.cat_features]
        weight_feature = ["Weight Capacity (kg)"]
        te_cat_features = ["TE_" + col for col in self.cat_features]
        inter_cat_features = ["WCap_x_TE_" + col for col in self.cat_features]
        weight_features = ["te_weight", "te_sq", "te_cube"]
        final_features = orig_cat_features + count_cat_features + weight_feature + te_cat_features + inter_cat_features + weight_features
        
        X_test = test[final_features]
        test_pred = self.final_model.predict(X_test)
        
        sub = pd.DataFrame({
            "id": test["id"] if "id" in test.columns else test.index,
            "Price": test_pred
        })
        sub.to_csv(self.submission_file, index=False)
        tqdm.write(f"Submission file saved to {self.submission_file}")
    
    def save_pipeline(self, filename="twobaseline_optimized_backpack.pth"):
        """Save the entire pipeline (the workflow instance) to a file."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        tqdm.write(f"Pretrained pipeline saved to {filename}.")
    
    @staticmethod
    def load_pipeline(filename="twobaseline_optimized_backpack.pth"):
        """Load the pipeline from a saved file."""
        with open(filename, "rb") as f:
            pipeline = pickle.load(f)
        tqdm.write(f"Pretrained pipeline loaded from {filename}.")
        return pipeline
    
    def run_workflow(self):
        """Run the complete two-baseline workflow with progress reporting."""
        overall_steps = [
            "Load Data", 
            "Baseline 1 (Train Mean)", 
            "Optimized Baseline: Feature Engineering, Hyperparameter Tuning & Final Model Training", 
            "Test Processing & Submission Generation"
        ]
        with tqdm(total=len(overall_steps), desc="Overall Workflow") as pbar:
            self.load_data()
            pbar.update(1)
            
            self.baseline1()
            pbar.update(1)
            
            self.optimized_baseline()
            pbar.update(1)
            
            self.process_test_and_generate_submission()
            pbar.update(1)
        
        tqdm.write("Workflow complete.")




# -----------------------------------------------------------------------------------
# TwoBaselineMLModelWorkflow usage: include TE,FE, polynomial FE, optuna, Kfold,oof
# -----------------------------------------------------------------------------------
if __name__ == "__main__":
    
    train_file = "/kaggle/input/playground-series-s5e2/train.csv"
    train_extra_file = "/kaggle/input/playground-series-s5e2/training_extra.csv"
    test_file = "/kaggle/input/playground-series-s5e2/test.csv"
    submission_file = "submission.csv"
    
    pretrained_filename = "twobaseline_optimized_backpack.pth"
    
    # If a pretrained pipeline exists, load it and generate test predictions.
    if os.path.exists(pretrained_filename):
        workflow = TwoBaselineMLModelWorkflow.load_pipeline(pretrained_filename)
        workflow.process_test_and_generate_submission()
    else:
        workflow = TwoBaselineMLModelWorkflow(train_file, train_extra_file, test_file, submission_file)
        workflow.run_workflow()








