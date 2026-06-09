import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore
from IPython.display import clear_output
from sklearn.model_selection import *
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import catboost as cb
from lightgbm import LGBMRegressor
import lightgbm as lgb
from tqdm import tqdm
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple


# Install any additional packages if required (uncomment if necessary)
!pip install lightgbm xgboost catboost
!git clone https://github.com/muhammadabdullah0303/AbdML


!pip install -qq lifelines


import sys
sys.path.append('/kaggle/working/repository')

from AbdML.main import AbdBase


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SalesPredictor:
    def __init__(self, 
                 data_dir: str = '/kaggle/input/playground-series-s5e1/',
                 seed: int = 114520,
                 n_splits: int = 6,
                 repository_path: str = '/kaggle/working/repository'):
        """Initialize the Sales Predictor with configuration parameters."""
        self.data_dir = Path(data_dir)
        self.seed = seed
        self.n_splits = n_splits
        self.repository_path = Path(repository_path)
        self._setup_environment()
        
    def _setup_environment(self) -> None:
        """Setup the environment and import custom modules."""
        try:
            sys.path.append(str(self.repository_path))
            from AbdML.main import AbdBase
            self.AbdBase = AbdBase
            logger.info("Environment setup completed successfully")
        except ImportError as e:
            logger.error(f"Failed to import AbdML: {e}")
            raise

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and preprocess the data."""
        try:
            train = pd.read_csv(self.data_dir / 'train.csv')
            test = pd.read_csv(self.data_dir / 'test.csv')
            sample = pd.read_csv(self.data_dir / 'sample_submission.csv')
            
            # Preprocess data
            train = train.dropna(subset=['num_sold'])
            self.train_ids = train['id']
            self.test_ids = test['id']
            train = train.drop('id', axis=1)
            test = test.drop('id', axis=1)
            
            logger.info("Data loaded and preprocessed successfully")
            return train, test, sample
            
        except FileNotFoundError as e:
            logger.error(f"Error loading data files: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during data loading: {e}")
            raise

    def prepare_features(self, train: pd.DataFrame, test: pd.DataFrame) -> 'AbdBase':
        """Prepare features and initialize AbdBase."""
        cat_c = ['country', 'store', 'product', 'month_name', 'day_of_week']
        ohe_cols = {'cat_c': cat_c}
        
        try:
            base = self.AbdBase(
                train_data=train,
                test_data=test,
                target_column='num_sold',
                gpu=False,
                handle_date=True,
                problem_type="regression",
                metric="mape",
                seed=self.seed,
                n_splits=self.n_splits,
                early_stop=True,
                num_classes=0,
                cat_features=None,
                ohe_fe=ohe_cols,
                fold_type='GKF'
            )
            logger.info("Feature preparation completed")
            return base
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise

    def train_model(self, base: 'AbdBase') -> Tuple[list, np.ndarray]:
        """Train the LGBM model with specified parameters."""
        params = {
            'n_estimators': 850,
            'max_depth': 5,
            'colsample_bytree': 0.41,
            'subsample': 0.52,
            'learning_rate': 0.09,
            'min_child_samples': 90
        }
        
        try:
            rLGBM = base.Train_ML(params, 'LGBM', e_stop=250, y_log=True, g_col='group')
            logger.info("Model training completed successfully")
            return rLGBM
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

    def generate_submission(self, 
                          sample: pd.DataFrame, 
                          predictions: np.ndarray, 
                          output_path: str = "submission.csv") -> pd.DataFrame:
        """Generate and save submission file."""
        try:
            submission = sample.copy()
            submission["num_sold"] = predictions
            submission.to_csv(output_path, index=False)
            logger.info(f"Submission file saved to {output_path}. Shape: {submission.shape}")
            return submission
            
        except Exception as e:
            logger.error(f"Error generating submission: {e}")
            raise

    def run(self) -> pd.DataFrame:
        """Execute the complete prediction pipeline."""
        train, test, sample = self.load_data()
        base = self.prepare_features(train, test)
        
        # Display sample of prepared features
        logger.info("Training data sample:")
        print(base.X_train.head())
        logger.info("Test data sample:")
        print(base.X_test.head())
        
        rLGBM = self.train_model(base)
        submission = self.generate_submission(sample, rLGBM[1])
        return submission


# Main execution
def main():
    """Main execution function."""
    try:
        predictor = SalesPredictor()
        submission = predictor.run()
        print(submission.head())
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise

# Run the pipeline
main()

