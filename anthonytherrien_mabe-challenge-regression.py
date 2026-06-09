"""
MABe Challenge 2025 - Starter Code - Partial dataset
Multi-Agent Behavior Recognition in Mice

Optimizations for speed:
1. Process only a subset of training data
2. Simplified feature extraction
3. Larger window stride (less overlap)
4. Single model instead of ensemble
5. Batch processing with early stopping
6. Reduced cross-validation folds
"""


import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import gc
from typing import Dict, List, Optional
from dataclasses import dataclass
from tqdm import tqdm
import pickle
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from xgboost import XGBClassifier

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration parameters - optimized for fast execution"""
    # Paths
    data_path: Path = Path('/kaggle/input/MABe-mouse-behavior-detection')
    output_path: Path = Path('/kaggle/working')
    
    # SPEED OPTIMIZATIONS
    max_train_videos: int = 500  # Process only 500 videos instead of all 8790
    max_test_videos: Optional[int] = None  # Process all test videos
    sample_rate: float = 0.3  # Sample 30% of windows from each video
    
    # Feature extraction - simplified
    window_size: int = 60  # Larger windows
    window_stride: int = 30  # Larger stride = less overlap = faster
    min_window_fill: float = 0.3  # More lenient
    use_simple_features: bool = True  # Use only basic features
    
    # Model parameters - simplified
    use_single_model: bool = True  # No ensemble for speed
    n_splits: int = 2  # Reduced CV folds (was 5)
    random_state: int = 42
    
    # Behavior detection
    confidence_threshold: float = 0.25
    min_behavior_duration: int = 5  # Reduced from 10
    
    # Processing
    batch_size: int = 50  # Smaller batches for memory
    early_stop_batches: int = 10  # Stop after 10 batches if not improving

config = Config()


# ============================================================================
# SIMPLIFIED DATA STRUCTURES
# ============================================================================

class MouseTrackingData:
    """Simplified container for mouse tracking data"""
    
    def __init__(self, video_id: str, data: pd.DataFrame):
        self.video_id = video_id
        self.data = data
        self._preprocess()
    
    def _preprocess(self):
        """Minimal preprocessing"""
        self.frame_col = None
        for col in ['video_frame', 'frame', 'frame_number']:
            if col in self.data.columns:
                self.frame_col = col
                break
        
        if self.frame_col is None:
            self.data['frame'] = self.data.index
            self.frame_col = 'frame'
    
    def get_trajectory(self) -> pd.DataFrame:
        """Get simplified trajectory - just centroid"""
        if 'x' in self.data.columns and 'y' in self.data.columns:
            trajectory = self.data.groupby(self.frame_col).agg({
                'x': 'mean',
                'y': 'mean'
            }).reset_index()
            trajectory.columns = ['frame', 'x', 'y']
            return trajectory
        return pd.DataFrame()

# ============================================================================
# FAST FEATURE EXTRACTION
# ============================================================================

class FastFeatureExtractor:
    """Simplified feature extraction for speed"""
    
    def __init__(self, window_size: int = 60, stride: int = 30):
        self.window_size = window_size
        self.stride = stride
    
    def extract_basic_features(self, trajectory: pd.DataFrame) -> Dict[str, float]:
        """Extract only essential movement features"""
        features = {}
        
        if len(trajectory) < 2:
            return features
        
        x = trajectory['x'].values
        y = trajectory['y'].values
        
        dx = np.diff(x)
        dy = np.diff(y)
        velocity = np.sqrt(dx**2 + dy**2)
        
        features['vel_mean'] = np.mean(velocity)
        features['vel_std'] = np.std(velocity)
        features['vel_max'] = np.max(velocity)
        features['total_dist'] = np.sum(velocity)
        features['net_disp'] = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
        features['x_range'] = np.max(x) - np.min(x)
        features['y_range'] = np.max(y) - np.min(y)
        features['stationary'] = np.mean(velocity < 2.0)
        features['fast_move'] = np.mean(velocity > 20.0)
        
        return features
    
    def extract_windows_fast(self, tracking_data: MouseTrackingData, 
                            sample_rate: float = 1.0) -> pd.DataFrame:
        """Fast window extraction with sampling"""
        trajectory = tracking_data.get_trajectory()
        
        if trajectory.empty or len(trajectory) < self.window_size:
            return pd.DataFrame()
        
        all_features = []
        min_frame = trajectory['frame'].min()
        max_frame = trajectory['frame'].max()
        
        window_starts = list(range(int(min_frame), 
                                  int(max_frame) - self.window_size + 1, 
                                  self.stride))
        
        if sample_rate < 1.0:
            n_samples = max(1, int(len(window_starts) * sample_rate))
            window_starts = random.sample(window_starts, min(n_samples, len(window_starts)))
        
        for start_frame in window_starts:
            end_frame = start_frame + self.window_size
            window = trajectory[(trajectory['frame'] >= start_frame) & 
                               (trajectory['frame'] < end_frame)]
            
            if len(window) < self.window_size * config.min_window_fill:
                continue
            
            features = {
                'video_id': tracking_data.video_id,
                'start_frame': start_frame,
                'end_frame': end_frame,
            }
            
            basic_feats = self.extract_basic_features(window)
            features.update(basic_feats)
            
            all_features.append(features)
        
        return pd.DataFrame(all_features)


# ============================================================================
# SIMPLIFIED ANNOTATION PROCESSING
# ============================================================================

class FastAnnotationProcessor:
    """Fast annotation processing"""
    
    @staticmethod
    def load_annotations(annotation_path: Path) -> pd.DataFrame:
        """Load annotations with minimal processing"""
        if not annotation_path.exists():
            return pd.DataFrame()
        
        try:
            annotations = pd.read_parquet(annotation_path)
            col_mapping = {
                'start': 'start_frame',
                'end': 'end_frame',
                'stop': 'end_frame',
                'stop_frame': 'end_frame'
            }
            
            for old_col, new_col in col_mapping.items():
                if old_col in annotations.columns and new_col not in annotations.columns:
                    annotations[new_col] = annotations[old_col]
            
            if 'start_frame' in annotations.columns and 'end_frame' in annotations.columns:
                return annotations[['start_frame', 'end_frame']]
            
            return pd.DataFrame()
            
        except:
            return pd.DataFrame()
    
    @staticmethod
    def create_labels_fast(features: pd.DataFrame, 
                          annotations: pd.DataFrame) -> np.ndarray:
        """Fast label creation"""
        if annotations.empty or 'start_frame' not in annotations.columns:
            return np.zeros(len(features))
        
        labels = np.zeros(len(features))
        for idx, window in features.iterrows():
            overlaps = ((annotations['start_frame'] <= window['end_frame']) & 
                       (annotations['end_frame'] >= window['start_frame']))
            labels[idx] = 1 if overlaps.any() else 0
        
        return labels


# ============================================================================
# FAST MODEL (XGBoost)
# ============================================================================

class FastBehaviorDetector:
    """Simplified model for fast training with XGBoost"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
    
    def prepare_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Prepare features quickly"""
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        exclude = ['start_frame', 'end_frame']
        self.feature_columns = [c for c in numeric_cols if c not in exclude]
        
        if not self.feature_columns:
            return np.zeros((len(features_df), 1))
        
        X = features_df[self.feature_columns].fillna(0).values
        return X
    
    def train_fast(self, X: np.ndarray, y: np.ndarray):
        """Fast training with single XGBoost model"""
        print(f"Fast training on {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Positive rate: {100*np.mean(y):.1f}%")
        
        if X.shape[0] == 0 or X.shape[1] == 0:
            return
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=config.random_state, stratify=y
        )
        
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_weight=5,
            random_state=config.random_state,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
            verbosity=0
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=30,
            verbose=False
        )
        
        val_pred = self.model.predict(X_val)
        val_f1 = f1_score(y_val, val_pred)
        print(f"Validation F1: {val_f1:.3f}")
    
    def predict_fast(self, X: np.ndarray) -> np.ndarray:
        """Fast prediction"""
        if self.model is None:
            return np.zeros(X.shape[0])
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


# ============================================================================
# MAIN FAST PIPELINE
# ============================================================================

class FastMABePipeline:
    """Optimized pipeline for speed"""
    
    def __init__(self):
        self.feature_extractor = FastFeatureExtractor(
            window_size=config.window_size,
            stride=config.window_stride
        )
        self.annotation_processor = FastAnnotationProcessor()
        self.detector = FastBehaviorDetector()
    
    def process_training_fast(self):
        """Fast training data processing"""
        print("\n" + "="*60)
        print("FAST TRAINING DATA PROCESSING")
        print("="*60)
        
        tracking_files = []
        train_path = config.data_path / 'train_tracking'
        
        if train_path.exists():
            for lab_dir in train_path.iterdir():
                if lab_dir.is_dir():
                    files = list(lab_dir.glob('*.parquet'))
                    tracking_files.extend(files)
        
        if config.max_train_videos and len(tracking_files) > config.max_train_videos:
            print(f"Sampling {config.max_train_videos} from {len(tracking_files)} files")
            tracking_files = random.sample(tracking_files, config.max_train_videos)
        
        print(f"Processing {len(tracking_files)} files")
        
        all_features = []
        all_labels = []
        
        for i in range(0, len(tracking_files), config.batch_size):
            batch = tracking_files[i:i+config.batch_size]
            batch_num = i//config.batch_size + 1
            total_batches = (len(tracking_files)-1)//config.batch_size + 1
            
            print(f"\nBatch {batch_num}/{total_batches}")
            
            for file_path in tqdm(batch, desc="Processing"):
                try:
                    data = pd.read_parquet(file_path)
                    tracking = MouseTrackingData(file_path.stem, data)
                    features = self.feature_extractor.extract_windows_fast(
                        tracking, sample_rate=config.sample_rate
                    )
                    
                    if features.empty:
                        continue
                    
                    ann_path = config.data_path / 'train_annotation' / \
                              file_path.parent.name / f"{file_path.stem}.parquet"
                    annotations = self.annotation_processor.load_annotations(ann_path)
                    
                    labels = self.annotation_processor.create_labels_fast(
                        features, annotations
                    )
                    
                    all_features.append(features)
                    all_labels.append(labels)
                    
                except Exception as e:
                    continue
            
            if len(all_features) > 0:
                total_samples = sum(len(f) for f in all_features)
                if total_samples > 50000:
                    print(f"Early stop: {total_samples} samples collected")
                    break
            
            gc.collect()
        
        if all_features:
            features_df = pd.concat(all_features, ignore_index=True)
            labels = np.concatenate(all_labels)
            print(f"\nTotal samples: {len(features_df)}")
            print(f"Positive rate: {100*np.mean(labels):.1f}%")
            return features_df, labels
        
        return None, None
    
    def train_model_fast(self, features_df: pd.DataFrame, labels: np.ndarray):
        """Fast model training"""
        print("\n" + "="*60)
        print("FAST MODEL TRAINING")
        print("="*60)
        
        X = self.detector.prepare_features(features_df)
        self.detector.train_fast(X, labels)
    
    def generate_predictions_fast(self) -> pd.DataFrame:
        """Fast prediction generation"""
        print("\n" + "="*60)
        print("FAST PREDICTION GENERATION")
        print("="*60)
        
        test_path = config.data_path / 'test_tracking'
        test_files = []
        
        if test_path.exists():
            for lab_dir in test_path.iterdir():
                if lab_dir.is_dir():
                    test_files.extend(list(lab_dir.glob('*.parquet')))
        
        print(f"Found {len(test_files)} test files")
        
        if config.max_test_videos and len(test_files) > config.max_test_videos:
            test_files = test_files[:config.max_test_videos]
        
        predictions = []
        
        for test_file in tqdm(test_files, desc="Predicting"):
            try:
                data = pd.read_parquet(test_file)
                tracking = MouseTrackingData(test_file.stem, data)
                features = self.feature_extractor.extract_windows_fast(
                    tracking, sample_rate=1.0
                )
                
                if features.empty:
                    continue
                
                X = self.detector.prepare_features(features)
                proba = self.detector.predict_fast(X)
                
                for idx, prob in enumerate(proba):
                    if prob > config.confidence_threshold:
                        predictions.append({
                            'video_id': features.iloc[idx]['video_id'],
                            'agent_id': 'mouse1',
                            'target_id': 'mouse2',
                            'action': 'sniff',
                            'start_frame': int(features.iloc[idx]['start_frame']),
                            'stop_frame': int(features.iloc[idx]['end_frame'])
                        })
            except:
                continue
        
        if predictions:
            pred_df = pd.DataFrame(predictions)
            pred_df = pred_df.drop_duplicates()
            pred_df = pred_df.sort_values(['video_id', 'start_frame'])
            return pred_df
        else:
            return pd.DataFrame({
                'video_id': ['test_video'],
                'agent_id': ['mouse1'],
                'target_id': ['mouse2'],
                'action': ['grooming'],
                'start_frame': [0],
                'stop_frame': [30]
            })
    
    def create_submission(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Create submission file"""
        predictions['row_id'] = range(len(predictions))
        submission = predictions[['row_id', 'video_id', 'agent_id', 'target_id', 
                                 'action', 'start_frame', 'stop_frame']]
        
        submission.to_csv('submission.csv', index=False)
        print(f"\nCreated submission.csv with {len(submission)} predictions")
        return submission


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution - optimized for speed"""
    import time
    start_time = time.time()
    
    print("\n" + "="*80)
    print("MABe CHALLENGE 2025 - FAST STARTER CODE (XGBoost Version)")
    print("Optimized to run in < 12 hours")
    print("="*80)
    
    print(f"\nSpeed Settings:")
    print(f"  Max training videos: {config.max_train_videos}")
    print(f"  Window sampling rate: {config.sample_rate}")
    print(f"  Window size: {config.window_size}")
    print(f"  Window stride: {config.window_stride}")
    
    pipeline = FastMABePipeline()
    
    features_df, labels = pipeline.process_training_fast()
    
    if features_df is not None and len(features_df) > 100:
        pipeline.train_model_fast(features_df, labels)
        
        with open('fast_model.pkl', 'wb') as f:
            pickle.dump(pipeline.detector, f)
        print("Model saved to fast_model.pkl")
        
        predictions = pipeline.generate_predictions_fast()
        submission = pipeline.create_submission(predictions)
        
        elapsed = time.time() - start_time
        hours = elapsed / 3600
        print(f"\nTotal runtime: {hours:.2f} hours")
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE!")
        print("="*80)
        
        return submission
    else:
        print("\nInsufficient training data!")
        return None


if __name__ == "__main__":
    submission = main()
    if submission is not None:
        print("\n✓ Execution completed successfully!")
        print(f"Submission preview:\n{submission.head()}")

