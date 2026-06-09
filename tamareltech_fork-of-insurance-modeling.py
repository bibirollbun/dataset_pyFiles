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


# Install only necessary packages
!pip install numpy pandas scikit-learn flaml

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error
from flaml import AutoML
import time
import logging
import warnings
warnings.filterwarnings('ignore')

# Minimal logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

class EfficientPremiumPredictor:
    """Streamlined insurance premium prediction using FLAML AutoML"""
    
    def __init__(self, input_path, output_path='./'):
        self.input_path = input_path
        self.output_path = output_path
        self.target_column = 'Premium Amount'
        self.id_column = 'id'
        self.start_time = time.time()
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        # Configure time budget (in seconds)
        self.flaml_time_budget = 1200  # 20 minutes total
    
    def transform_target(self, y, inverse=False):
        """Transform target variable to better optimize for RMSLE"""
        if inverse:
            # Convert back from log space to original space
            return np.exp(y) - 1
        else:
            # Convert to log space for training
            # Adding 1 to handle potential zero values
            return np.log1p(y)
    
    def estimate_prediction_uncertainty(self, automl, X_test, predictions, confidence=0.95):
        """Estimate uncertainty in premium predictions"""
        logger.info("Estimating prediction uncertainty...")
        
        # For tree-based models, use quantile regression approach
        if automl.best_estimator in ["lgbm", "xgboost", "rf", "catboost"]:
            # Create dataframe for uncertainty analysis
            uncertainty_df = pd.DataFrame({
                self.id_column: self.test_ids,
                'Prediction': predictions
            })
            
            # Estimate prediction variance using model-specific approach
            if hasattr(automl.model, 'estimator'):
                model = automl.model.estimator
                
                # For tree-based models, use tree variance as uncertainty proxy
                if hasattr(model, 'apply'):
                    # Get leaf indices for each sample
                    leaf_indices = model.apply(X_test)
                    
                    # Map each leaf to predictions
                    leaf_to_predictions = {}
                    train_leaves = model.apply(self.X_train)
                    
                    # Calculate variance within each leaf
                    variances = []
                    for i, leaf_id in enumerate(leaf_indices):
                        # Use string representation of leaf_id as dictionary key
                        leaf_key = str(leaf_id)
                        if leaf_key not in leaf_to_predictions:
                            # Find training samples in this leaf
                            mask = np.array([str(l) == leaf_key for l in train_leaves])
                            if np.any(mask):
                                # Get transformed predictions for these samples
                                leaf_predictions = self.transform_target(
                                    automl.predict(self.X_train[mask]), 
                                    inverse=True
                                )
                                leaf_to_predictions[leaf_key] = leaf_predictions
                        
                        # Get variance for this prediction
                        if leaf_key in leaf_to_predictions and len(leaf_to_predictions[leaf_key]) > 1:
                            variances.append(np.var(leaf_to_predictions[leaf_key]))
                        else:
                            # Fall back to overall variance if leaf has insufficient samples
                            variances.append(np.var(predictions))
                    
                    # Add variance-based confidence intervals
                    z_score = 1.96  # 95% confidence
                    std_devs = np.sqrt(variances)
                    uncertainty_df['Lower_Bound'] = predictions - z_score * std_devs
                    uncertainty_df['Upper_Bound'] = predictions + z_score * std_devs
                    
                    # Ensure lower bound is positive
                    uncertainty_df['Lower_Bound'] = np.maximum(uncertainty_df['Lower_Bound'], 0.001)
                    
                    # Save uncertainty estimates
                    uncertainty_path = os.path.join(self.output_path, 'prediction_uncertainty.csv')
                    uncertainty_df.to_csv(uncertainty_path, index=False)
                    logger.info(f"Prediction uncertainty estimates saved to {uncertainty_path}")
                    
                    return uncertainty_df
        
        logger.warning("Uncertainty estimation not available for this model type")
        return None
    
    def analyze_feature_importance(self, automl):
        """Extract and analyze feature importance from the trained model"""
        logger.info("Analyzing feature importance...")
        
        # Check if the model supports feature importance
        if hasattr(automl.model, 'feature_importances_') or hasattr(automl.model.estimator, 'feature_importances_'):
            # Extract feature importances from the appropriate location
            if hasattr(automl.model, 'feature_importances_'):
                importances = automl.model.feature_importances_
            else:
                importances = automl.model.estimator.feature_importances_
                
            # Create a DataFrame for easier analysis and visualization
            importance_df = pd.DataFrame({
                'Feature': self.X_train.columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Log top and bottom features
            logger.info("Top 10 most influential features:")
            for i, row in importance_df.head(10).iterrows():
                logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")
                
            logger.info("Least influential features:")
            for i, row in importance_df.tail(5).iterrows():
                logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")
            
            # Save feature importance to CSV
            importance_path = os.path.join(self.output_path, 'feature_importance.csv')
            importance_df.to_csv(importance_path, index=False)
            logger.info(f"Feature importance saved to {importance_path}")
            
            return importance_df
        else:
            logger.warning("Model does not provide feature importance information")
            return None
        
    def load_and_prepare_data(self):
        """Efficiently load and prepare data for modeling"""
        logger.info("Loading and preparing data...")
        
        # Load datasets
        train = pd.read_csv(os.path.join(self.input_path, 'train.csv'))
        test = pd.read_csv(os.path.join(self.input_path, 'test.csv'))
        
        logger.info(f"Loaded - Train: {train.shape}, Test: {test.shape}")
        
        # Extract target and IDs
        self.y_train = train[self.target_column].copy()
        self.test_ids = test[self.id_column].copy()
        
        # Basic feature engineering (only the most impactful features)
        train_processed = train.copy()
        test_processed = test.copy()
        
        # Age transformations
        if 'Age' in train.columns:
            train_processed['Age_Squared'] = train_processed['Age'] ** 2
            test_processed['Age_Squared'] = test_processed['Age'] ** 2
            
            # Log transform age
            train_processed['Log_Age'] = np.log1p(train_processed['Age'])
            test_processed['Log_Age'] = np.log1p(test_processed['Age'])
        
        # Income transformations
        if 'Annual Income' in train.columns:
            train_processed['Log_Income'] = np.log1p(train_processed['Annual Income'])
            test_processed['Log_Income'] = np.log1p(test_processed['Annual Income'])
        
        # Health transformations
        if 'Health Score' in train.columns:
            max_health = train_processed['Health Score'].max()
            train_processed['Inverse_Health'] = max_health - train_processed['Health Score']
            test_processed['Inverse_Health'] = max_health - test_processed['Health Score']
        
        # Family size
        if 'Number of Dependents' in train.columns:
            train_processed['Family_Size'] = train_processed['Number of Dependents'].fillna(0) + 1
            test_processed['Family_Size'] = test_processed['Number of Dependents'].fillna(0) + 1
            
            # Income per person
            if 'Annual Income' in train.columns:
                train_processed['Income_Per_Person'] = train_processed['Annual Income'] / train_processed['Family_Size']
                test_processed['Income_Per_Person'] = test_processed['Annual Income'] / test_processed['Family_Size']
                
                # Replace infinities with NaN
                train_processed['Income_Per_Person'].replace([np.inf, -np.inf], np.nan, inplace=True)
                test_processed['Income_Per_Person'].replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Interaction feature
        if all(col in train.columns for col in ['Age', 'Health Score']):
            train_processed['Age_Health'] = train_processed['Age'] * train_processed['Health Score']
            test_processed['Age_Health'] = test_processed['Age'] * test_processed['Health Score']
        
        # Prepare for modeling
        train_processed = train_processed.drop([self.id_column, self.target_column], axis=1)
        test_processed = test_processed.drop([self.id_column], axis=1)
        
        # Identify categorical columns
        categorical_cols = []
        for col in train_processed.columns:
            if train_processed[col].dtype == 'object' or col in [
                'Gender', 'Marital Status', 'Education Level', 'Occupation', 'Location'
            ]:
                categorical_cols.append(col)
        
        # Efficient preprocessing
        for col in categorical_cols:
            # Fill missing values
            train_processed[col] = train_processed[col].fillna('Unknown')
            test_processed[col] = test_processed[col].fillna('Unknown')
            
            # Label encoding
            le = LabelEncoder()
            # Combine train and test values
            all_values = pd.concat([train_processed[col], test_processed[col]]).astype(str)
            le.fit(all_values)
            
            # Transform both datasets
            train_processed[col] = le.transform(train_processed[col].astype(str))
            test_processed[col] = le.transform(test_processed[col].astype(str))
        
        # Handle numerical missing values
        for col in train_processed.columns:
            if col not in categorical_cols:
                # Simple median imputation
                median_val = train_processed[col].median()
                train_processed[col] = train_processed[col].fillna(median_val)
                test_processed[col] = test_processed[col].fillna(median_val)
        
        # Store processed data
        self.X_train = train_processed
        self.X_test = test_processed
        
        logger.info(f"Data preparation complete with {train_processed.shape[1]} features")
        return True
    
    def train_and_predict(self):
        """Efficiently train FLAML AutoML and generate predictions"""
        logger.info(f"Training FLAML with {self.flaml_time_budget} seconds budget...")
        
        # Transform target for training (log transformation)
        logger.info("Applying log transformation to target variable for RMSLE optimization")
        y_train_transformed = self.transform_target(self.y_train)
        
        # Create and configure FLAML with simplified settings
        automl = AutoML()
        
        # Simplified settings based on user's preference
        settings = {
            "task": "regression",
            "time_budget": self.flaml_time_budget,
            "metric": "rmse",  # Now optimizing RMSE on log-transformed target
            "eval_method": "cv",
            "n_splits": 5,
        }
        
        # Train the model on transformed target
        automl.fit(
            X_train=self.X_train,
            y_train=y_train_transformed,
            **settings
        )
        
        # Log basic model info
        logger.info(f"Training complete. Best model: {automl.best_estimator}")

        # Add feature importance analysis
        feature_importance = self.analyze_feature_importance(automl)
        
        # Generate predictions (still in log space)
        logger.info("Generating predictions...")
        log_predictions = automl.predict(self.X_test)
        
        # Transform predictions back to original scale
        logger.info("Transforming predictions back to original scale")
        predictions = self.transform_target(log_predictions, inverse=True)
        
        # Apply safeguards (simpler now that we're already handling log transformation)
        predictions = np.maximum(predictions, 0.001)
        
        # Apply mild quantile-based outlier capping
        upper_quantile = np.percentile(predictions, 99.5)
        predictions = np.minimum(predictions, upper_quantile)
        
        # Generate prediction uncertainty estimates
        logger.info("Estimating prediction uncertainty...")
        uncertainty_estimates = self.estimate_prediction_uncertainty(automl, self.X_test, predictions)
        
        # Create submission file
        submission = pd.DataFrame({
            self.id_column: self.test_ids,
            self.target_column: predictions
        })
        
        # Save to CSV
        submission_path = os.path.join(self.output_path, 'submission.csv')
        submission.to_csv(submission_path, index=False)
        
        logger.info(f"Submission file created at {submission_path}")
        
        # Calculate total runtime
        total_time = (time.time() - self.start_time) / 60
        logger.info(f"Complete in {total_time:.2f} minutes")
        
        return True

    def run(self):
        """Execute the streamlined pipeline"""
        if self.load_and_prepare_data():
            return self.train_and_predict()
        return False


# Run the efficient pipeline
if __name__ == "__main__":
    input_path = '/kaggle/input/premiumpulse-risk-modeling/'
    output_path = '/kaggle/working/'
    
    predictor = EfficientPremiumPredictor(input_path, output_path)
    predictor.run()

