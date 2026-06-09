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


!pip install autogluon


import pandas as pd
import numpy as np
import autogluon.multimodal as agmm
import torch
from sklearn.model_selection import train_test_split
import os
from tqdm.auto import tqdm
import gc
from sklearn.metrics import classification_report, confusion_matrix
import logging
import warnings
from typing import Dict, Tuple, Optional

class Config:
    """Configuration class with model and training parameters"""
    # Paths
    BASE_PATH = '/kaggle/input/cidaut-ai-fake-scene-classification-2024'
    TRAIN_PATH = os.path.join(BASE_PATH, 'Train')
    TEST_PATH = os.path.join(BASE_PATH, 'Test')
    OUTPUT_PATH = '.'
    
    # Model parameters
    SEED = 42
    VAL_SIZE = 0.2
    
    # Enhanced AutoGluon configuration
    HYPERPARAMETERS = {
    'model': {
        'names': ['timm_image'],
        'dropout_rate': 0.2,
        'timm_image': {
            'checkpoint_name': 'convnext_xlarge_384_in22ft1k',  # Changed to ConvNeXt
            'use_pretrained': True,
            'freeze_backbone': False
        }
    },
    'optimization': {
        'learning_rate': 5e-5,    # More conservative
        'weight_decay': 1e-2,     # Stronger regularization
        'max_epochs': 15,         # Reduced to prevent overfitting
        'label_smoothing': 0.01,  # Minimal smoothing
        'loss_function': 'binary_cross_entropy',  # Simpler loss
        'optimizer': 'adamw',
        'lr_scheduler': 'cosine',
        'warmup_epochs': 1
    },
    'data': {
        'image_size': 384,        # More memory efficient
        'augmentation': {
            'random_resized_crop': True,
            'random_horizontal_flip': 0.5,
            'color_jitter': 0.1,  # Minimal color augmentation
            # Removed complex augmentations
        },
        'batch_size': 8
    }
}
def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )

def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
class ImageClassifier:
    def __init__(self, config: Config):
        self.config = config
        setup_logging()
        seed_everything(config.SEED)
        self.logger = logging.getLogger(__name__)
        self.setup_data()
        
    def setup_data(self) -> None:
        """Load and prepare data with error handling"""
        self.logger.info("Loading data...")
        try:
            self.train = pd.read_csv(os.path.join(self.config.BASE_PATH, 'train.csv'))
            self.test = pd.read_csv(os.path.join(self.config.BASE_PATH, 'sample_submission.csv'))
            self.sub = pd.read_csv(os.path.join(self.config.BASE_PATH, 'sample_submission.csv'))
            
            # Validate data
            required_columns = {'image', 'label'}
            if not required_columns.issubset(self.train.columns):
                raise ValueError(f"Training data missing required columns: {required_columns}")
            
            # Add full paths with validation
            self.train['image'] = self.train['image'].apply(
                lambda x: os.path.join(self.config.TRAIN_PATH, x)
            )
            self.test['image'] = self.test['image'].apply(
                lambda x: os.path.join(self.config.TEST_PATH, x)
            )
            
            # Validate image paths
            missing_train = [f for f in self.train['image'] if not os.path.exists(f)]
            if missing_train:
                self.logger.warning(f"Missing {len(missing_train)} training images")
            
            # Create stratified validation split
            self.train_data, self.val_data = train_test_split(
                self.train, 
                test_size=self.config.VAL_SIZE,
                random_state=self.config.SEED,
                stratify=self.train['label']
            )
            
            self.logger.info(f"Train size: {len(self.train_data)}, Validation size: {len(self.val_data)}")
            
        except Exception as e:
            self.logger.error(f"Error in data setup: {str(e)}")
            raise
            
    def train_model(self) -> None:
        """Train the model with validation and metrics tracking"""
        self.logger.info("Training model...")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            # Initialize predictor with binary classification
            self.predictor = agmm.MultiModalPredictor(
                label='label',
                path=os.path.join(self.config.OUTPUT_PATH, 'modelnxz497'),
                problem_type='binary'  # Changed from 'multiclass' to 'binary'
            )
            
            # Train with validation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.predictor.fit(
                    train_data=self.train_data,
                    tuning_data=self.val_data,
                    hyperparameters=self.config.HYPERPARAMETERS,
                    time_limit=7200  # 2 hour time limit
                )
            
            # Evaluate on validation set
            val_predictions = self.predictor.predict(self.val_data)
            val_proba = self.predictor.predict_proba(self.val_data)
            
            # Generate detailed metrics
            report = classification_report(
                self.val_data['label'], 
                val_predictions,
                output_dict=True
            )
            
            self.logger.info("\nValidation Results:")
            self.logger.info(f"\nClassification Report:\n{pd.DataFrame(report).T}")
            
            # Save validation predictions for analysis
            val_results = pd.DataFrame({
                'true_label': self.val_data['label'],
                'predicted_label': val_predictions
            })
            val_results.to_csv('validation_results.csv', index=False)
            
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise
            
    def predict(self) -> None:
        """Generate and save predictions with error handling"""
        self.logger.info("Generating predictions...")
        try:
            predictions = self.predictor.predict(self.test)
            probabilities = self.predictor.predict_proba(self.test)
            
            # Create submission file with probabilities
            self.sub['label'] = predictions
            for i, col in enumerate(probabilities.columns):
                self.sub[f'prob_{col}'] = probabilities.iloc[:, i]
            
            submission_path = os.path.join(self.config.OUTPUT_PATH, 'submissionuv19x.csv')
            self.sub.to_csv(submission_path, index=False)
            
            # Log prediction distribution
            value_counts = pd.Series(predictions).value_counts()
            self.logger.info("\nPrediction Distribution:")
            for label, count in value_counts.items():
                self.logger.info(f"{label}: {count} ({count/len(predictions)*100:.2f}%)")
                
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise
    
    def run(self) -> None:
        """Run the complete pipeline with timing"""
        try:
            start_time = pd.Timestamp.now()
            self.logger.info(f"Starting pipeline at {start_time}")
            
            self.train_model()
            self.predict()
            
            end_time = pd.Timestamp.now()
            duration = end_time - start_time
            self.logger.info(f"Pipeline completed in {duration}")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

if __name__ == "__main__":
    classifier = ImageClassifier(Config)
    classifier.run()


!rm -rf /kaggle/working/*


import os
import torch
torch.cuda.empty_cache()


30 epochs -9418
40 epochs -9392
35 epochs -




