# Import Essential Libraries with Comprehensive Error Handling
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import time

# Advanced Scikit-Learn Modules
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# Logging and Warning Management
import logging
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

class NeuroClassificationPipeline:
    def __init__(self, train_path, test_path, solution_path):
        """
        Initialize the neuroimaging classification pipeline

        Parameters:
        - train_path: Path to training functional connectome matrices
        - test_path: Path to test functional connectome matrices
        - solution_path: Path to training solutions
        """
        # Robust Data Loading with Error Handling
        try:
            # Load training data
            logging.info("Loading training data...")
            self.train = pd.read_csv(train_path, index_col="participant_id")
            # Load test data
            logging.info("Loading test data...")
            self.test = pd.read_csv(test_path, index_col="participant_id")
            # Load training solutions
            logging.info("Loading training solutions...")
            self.y_train = pd.read_excel(solution_path)
        except Exception as e:
            logging.error(f"Data loading failed: {e}")
            raise

        # Logging Dataset Characteristics
        self._log_dataset_info()

    def _log_dataset_info(self):
        """Detailed Dataset Information Logging"""
        # Log the shape of the training data
        logging.info(f"Training Data Shape: {self.train.shape}")
        # Log the shape of the test data
        logging.info(f"Test Data Shape: {self.test.shape}")
        # Log the number of missing values in the training data
        logging.info("\nMissing Values:")
        logging.info(f"Train Missing: {self.train.isnull().sum().sum()}")
        # Log the number of missing values in the test data
        logging.info(f"Test Missing: {self.test.isnull().sum().sum()}")

        # Target Distribution Visualization
        plt.figure(figsize=(12, 4))

        # Plot ADHD Outcome Distribution
        plt.subplot(121)
        self.y_train['ADHD_Outcome'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title('ADHD Outcome Distribution')

        # Plot Sex Distribution
        plt.subplot(122)
        self.y_train['Sex_F'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title('Sex Distribution')

        plt.tight_layout()
        plt.savefig('target_distribution.png')
        logging.info("Target distribution visualization saved as 'target_distribution.png'")

    def advanced_feature_engineering(self, df):
        """
        Advanced Feature Engineering with Multiple Statistical Features

        Args:
            df (pd.DataFrame): Input connectivity matrix

        Returns:
            pd.DataFrame: Enhanced feature matrix
        """
        logging.info("Starting feature engineering...")
        features = df.copy()

        # Calculate mean of each row
        features['row_mean'] = df.mean(axis=1)
        # Calculate standard deviation of each row
        features['row_std'] = df.std(axis=1)
        # Calculate skewness of each row
        features['row_skew'] = df.skew(axis=1)
        # Calculate kurtosis of each row
        features['row_kurtosis'] = df.kurtosis(axis=1)
        # Calculate median of each row
        features['row_median'] = df.median(axis=1)
        # Calculate maximum value of each row
        features['row_max'] = df.max(axis=1)
        # Calculate minimum value of each row
        features['row_min'] = df.min(axis=1)
        # Calculate range of each row (max - min)
        features['row_range'] = features['row_max'] - features['row_min']

        logging.info("Feature engineering completed.")
        return features

    def train_model(self, X, y, model_type='adhd'):
        """
        Advanced Model Training with Cross-Validation

        Args:
            X (np.array): Feature matrix
            y (pd.Series): Target variable
            model_type (str): 'adhd' or 'sex'

        Returns:
            Trained LightGBM model
        """
        logging.info(f"Training {model_type.upper()} model...")
        # Advanced LightGBM Parameters
        lgb_params = {
            'n_estimators': 100,  # Reduced number of estimators for faster training
            'learning_rate': 0.02,
            'max_depth': 7,
            'num_leaves': 31,
            'random_state': 42,
            'n_jobs': -1,
            'metric': 'f1'
        }

        # Create Pipeline with Feature Selection
        pipeline = Pipeline([
            ('scaler', RobustScaler()),  # Scale the features
            ('feature_selection', SelectKBest(f_classif, k=20)),  # Select top 20 features
            ('classifier', lgb.LGBMClassifier(**lgb_params))  # Train LightGBM classifier
        ])

        # Stratified K-Fold Cross-Validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced number of splits for faster training
        start_time = time.time()
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1', n_jobs=-1)
        elapsed_time = time.time() - start_time

        # Log cross-validation scores
        logging.info(f"{model_type.upper()} Model Cross-Val Scores: {cv_scores}")
        logging.info(f"Mean CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        logging.info(f"Cross-validation completed in {elapsed_time:.2f} seconds.")

        # Final Model Training
        pipeline.fit(X, y)
        logging.info(f"{model_type.upper()} model training completed.")
        return pipeline

    def predict(self, pipeline, X_test):
        """
        Make predictions using trained pipeline

        Args:
            pipeline: Trained sklearn pipeline
            X_test: Test feature matrix

        Returns:
            Predictions
        """
        logging.info("Making predictions...")
        return pipeline.predict(X_test)

    def evaluate_model(self, pipeline, X_val, y_val):
        """
        Evaluate the model using the validation set

        Args:
            pipeline: Trained sklearn pipeline
            X_val: Validation feature matrix
            y_val: Validation target variable

        Returns:
            F1 score
        """
        logging.info("Evaluating the model...")
        y_pred = pipeline.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='weighted')
        logging.info(f"Validation F1 Score: {f1:.4f}")
        return f1

    def run_pipeline(self):
        """
        Execute Complete Classification Pipeline
        """
        logging.info("Starting the classification pipeline...")
        # Feature Engineering
        X_train = self.advanced_feature_engineering(self.train)
        X_test = self.advanced_feature_engineering(self.test)

        # Split the training data into training and validation sets to simulate the competition's evaluation process
        X_tr, X_val, y_tr_adhd, y_val_adhd, y_tr_sex, y_val_sex = train_test_split(
            X_train,
            self.y_train['ADHD_Outcome'],
            self.y_train['Sex_F'],
            test_size=0.51,  # Simulate the 51% validation set
            random_state=42
        )

        # Train Models
        logging.info("Training models...")
        adhd_model = self.train_model(X_tr, y_tr_adhd, 'adhd')
        sex_model = self.train_model(X_tr, y_tr_sex, 'sex')

        # Evaluate Models on the validation set (51%)
        adhd_f1_val = self.evaluate_model(adhd_model, X_val, y_val_adhd)
        sex_f1_val = self.evaluate_model(sex_model, X_val, y_val_sex)

        # Estimate the final score
        estimated_score = (adhd_f1_val + sex_f1_val) / 2
        logging.info(f"Estimated Final Score: {estimated_score:.4f}")

        # Predictions for the full test set
        test_preds_adhd = self.predict(adhd_model, X_test)
        test_preds_sex = self.predict(sex_model, X_test)

        # Create Submission
        submission = pd.DataFrame({
            'participant_id': self.test.index,
            'ADHD_Outcome': test_preds_adhd,
            'Sex_F': test_preds_sex
        })
        submission.to_csv('advanced_submission.csv', index=False)
        logging.info("Advanced Submission Created Successfully!")

# Main Execution
if __name__ == "__main__":
    # Initialize the pipeline with the provided data paths
    logging.info("Initializing the pipeline...")
    pipeline = NeuroClassificationPipeline(
        train_path="/kaggle/input/widsdatathon2025/TRAIN/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv",
        test_path="/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv",
        solution_path="/kaggle/input/widsdatathon2025/TRAIN/TRAINING_SOLUTIONS.xlsx"
    )
    # Run the complete classification pipeline
    pipeline.run_pipeline()


