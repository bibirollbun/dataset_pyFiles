import numpy as np
import pandas as pd
import os
import gc
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Fix for PyTorch 2.6 weights_only issue
try:
    from torch.serialization import add_safe_globals
    add_safe_globals([np.core.multiarray.scalar])
except:
    pass

class MouseDataset:
    """Mouse dataset handler"""
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.tracking_files = []
        
    def discover_files(self):
        """Discover dataset files"""
        print("ğŸ”� DISCOVERING FILES...")
        
        tracking_path = self.dataset_path / 'train_tracking'
        
        for lab_folder in tracking_path.iterdir():
            if lab_folder.is_dir():
                parquet_files = list(lab_folder.glob("*.parquet"))
                self.tracking_files.extend(parquet_files)
        
        print(f"ğŸ“� Found {len(self.tracking_files)} tracking files")
        return self.tracking_files

class FeatureEngineer:
    """Feature engineering with proper mouse handling"""
    
    def __init__(self):
        self.feature_cache = {}
        
    def compute_features(self, video_data):
        """Compute features for mouse behavior"""
        if video_data.empty:
            return None
            
        features = {}
        
        # Sort by frame
        video_data = video_data.sort_values('video_frame')
        
        # Get ALL mice from the data
        all_mice = sorted(video_data['mouse_id'].unique())
        features['total_mice'] = len(all_mice)
        
        print(f"   Detected mice: {all_mice}")  # Debug info
        
        # Features for each mouse
        for mouse_id in all_mice:
            mouse_data = video_data[video_data['mouse_id'] == mouse_id]
            
            if len(mouse_data) > 1:
                mouse_data = mouse_data.sort_values('video_frame')
                x_pos = mouse_data['x'].values
                y_pos = mouse_data['y'].values
                
                # Movement features
                if len(x_pos) > 1:
                    dx = np.diff(x_pos)
                    dy = np.diff(y_pos)
                    
                    speed = np.sqrt(dx**2 + dy**2)
                    
                    features[f'mouse_{mouse_id}_speed_mean'] = np.mean(speed)
                    features[f'mouse_{mouse_id}_speed_max'] = np.max(speed)
                    features[f'mouse_{mouse_id}_x_mean'] = np.mean(x_pos)
                    features[f'mouse_{mouse_id}_y_mean'] = np.mean(y_pos)
                    
                    # Distance traveled
                    total_distance = np.sum(np.sqrt(dx**2 + dy**2))
                    features[f'mouse_{mouse_id}_distance'] = total_distance
        
        # Social interactions between ALL mice
        if len(all_mice) >= 2:
            for i in range(len(all_mice)):
                for j in range(i+1, len(all_mice)):
                    mouse1, mouse2 = all_mice[i], all_mice[j]
                    social_features = self._compute_social_interaction(video_data, mouse1, mouse2)
                    features.update(social_features)
        
        return features
    
    def _compute_social_interaction(self, video_data, mouse1, mouse2):
        """Compute social interaction features"""
        features = {}
        
        m1_data = video_data[video_data['mouse_id'] == mouse1].sort_values('video_frame')
        m2_data = video_data[video_data['mouse_id'] == mouse2].sort_values('video_frame')
        
        if len(m1_data) > 0 and len(m2_data) > 0:
            common_frames = set(m1_data['video_frame']).intersection(set(m2_data['video_frame']))
            if len(common_frames) > 0:
                distances = []
                for frame in sorted(common_frames)[:50]:
                    m1_frame = m1_data[m1_data['video_frame'] == frame]
                    m2_frame = m2_data[m2_data['video_frame'] == frame]
                    
                    if len(m1_frame) > 0 and len(m2_frame) > 0:
                        dist = np.sqrt(
                            (m1_frame['x'].mean() - m2_frame['x'].mean())**2 +
                            (m1_frame['y'].mean() - m2_frame['y'].mean())**2
                        )
                        distances.append(dist)
                
                if distances:
                    features[f'dist_{mouse1}_{mouse2}_mean'] = np.mean(distances)
                    features[f'dist_{mouse1}_{mouse2}_min'] = np.min(distances)
        
        return features

class MouseBehaviorDataset(Dataset):
    """PyTorch Dataset for mouse behavior"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class MouseBehaviorNN(nn.Module):
    """Neural Network for mouse behavior classification"""
    
    def __init__(self, input_size, num_classes):
        super(MouseBehaviorNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

class MouseBehaviorPipeline:
    """Complete pipeline for mouse behavior detection"""
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.dataset_handler = MouseDataset(dataset_path)
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"ğŸš€ Using device: {self.device}")
        
    def process_training_data(self):
        """Process training data"""
        print("ğŸ�¯ PROCESSING TRAINING DATA...")
        
        tracking_files = self.dataset_handler.discover_files()
        
        all_features = []
        all_labels = []
        
        for file_path in tqdm(tracking_files[:2000], desc="Processing videos"):
            try:
                df = pd.read_parquet(file_path)
                
                # Sample frames for efficiency
                if len(df) > 1000:
                    df = df.iloc[::10].copy()
                
                # Compute features
                lab = file_path.parent.name
                features = self.feature_engineer.compute_features(df)
                
                if features:
                    # Map lab to behavior
                    lab_behavior_map = {
                        'NiftyGoldfinch': 'sniff',
                        'GroovyShrew': 'rear', 
                        'CalMS21_supplemental': 'sniff',
                        'CalMS21_task1': 'sniffgenital',
                        'CalMS21_task2': 'attack',
                        'AdaptableSnail': 'attack',
                        'SparklingTapir': 'attack',
                        'CRIM13': 'approach',
                        'UppityFerret': 'sniffgenital',
                    }
                    label = lab_behavior_map.get(lab, 'sniff')
                    
                    all_features.append(features)
                    all_labels.append(label)
                
                del df
                
            except Exception as e:
                continue
        
        print(f"âœ… Processed {len(all_features)} training samples")
        return all_features, all_labels
    
    def prepare_features(self, all_features, all_labels):
        """Prepare features for training"""
        print("ğŸ› ï¸� PREPARING FEATURES...")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features).fillna(0)
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        # Get feature columns
        feature_columns = [col for col in features_df.columns if col != 'total_mice']
        X = self.scaler.fit_transform(features_df[feature_columns])
        y = self.label_encoder.fit_transform(all_labels)
        
        print(f"ğŸ“Š Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"ğŸ�¯ Behaviors: {list(self.label_encoder.classes_)}")
        
        return X, y, feature_columns
    
    def train_model(self, X, y):
        """Train the model"""
        print("ğŸ§  TRAINING MODEL...")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"ğŸ“Š Training: {X_train.shape[0]}, Validation: {X_val.shape[0]}")
        
        train_dataset = MouseBehaviorDataset(X_train, y_train)
        val_dataset = MouseBehaviorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        model = MouseBehaviorNN(X.shape[1], len(np.unique(y))).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        best_val_f1 = 0
        best_model = None
        
        for epoch in range(50):
            # Training
            model.train()
            train_loss = 0
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = model(batch_features)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())
            
            val_f1 = f1_score(all_labels, all_preds, average='weighted')
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model = model.state_dict().copy()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/50: Train Loss: {train_loss/len(train_loader):.4f}, Val F1: {val_f1:.4f}")
        
        if best_model:
            model.load_state_dict(best_model)
            torch.save(model.state_dict(), 'best_mouse_model.pth')
        
        print(f"ğŸ�¯ Best Validation F1: {best_val_f1:.4f}")
        return model
    
    def create_submission(self, model, feature_columns):
        """Create submission.csv with proper format"""
        print("ğŸ“¤ CREATING SUBMISSION.CSV...")
        
        test_path = self.dataset_path / 'test_tracking'
        if not test_path.exists():
            print("â�Œ Test data not found!")
            return None
        
        test_files = list(test_path.rglob("*.parquet"))
        print(f"ğŸ�¯ Found {len(test_files)} test videos")
        
        all_predictions = []
        
        for file_path in test_files:
            try:
                df = pd.read_parquet(file_path)
                video_id = file_path.stem  # Just the filename
                
                # Get ALL mice from test data
                all_mice = sorted(df['mouse_id'].unique())
                print(f"   Video {video_id}: Mice detected - {all_mice}")
                
                # Process the entire video or in segments
                total_frames = df['video_frame'].max()
                
                # Create multiple segments
                for segment_num in range(5):
                    start_frame = segment_num * 2000
                    end_frame = min((segment_num + 1) * 2000, total_frames)
                    
                    if end_frame - start_frame < 500:
                        continue
                    
                    segment_data = df[(df['video_frame'] >= start_frame) & 
                                     (df['video_frame'] < end_frame)]
                    
                    if len(segment_data) < 100:
                        continue
                    
                    # Sample frames
                    if len(segment_data) > 500:
                        segment_data = segment_data.iloc[::5].copy()
                    
                    # Compute features
                    features = self.feature_engineer.compute_features(segment_data)
                    
                    if features:
                        # Prepare for prediction
                        features_df = pd.DataFrame([features]).fillna(0)
                        features_df = features_df.replace([np.inf, -np.inf], 0)
                        
                        # Ensure all feature columns exist
                        for col in feature_columns:
                            if col not in features_df.columns:
                                features_df[col] = 0
                        
                        X_test = self.scaler.transform(features_df[feature_columns])
                        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
                        
                        # Predict
                        model.eval()
                        with torch.no_grad():
                            outputs = model(X_test_tensor)
                            _, prediction = torch.max(outputs, 1)
                            behavior = self.label_encoder.inverse_transform(prediction.cpu().numpy())[0]
                        
                        # Create predictions for ALL mouse pairs
                        for i in range(len(all_mice)):
                            for j in range(len(all_mice)):
                                if i != j:  # Different mice
                                    agent_id = f"mouse{all_mice[i]}"
                                    target_id = f"mouse{all_mice[j]}"
                                    
                                    all_predictions.append({
                                        'row_id': len(all_predictions),
                                        'video_id': video_id,
                                        'agent_id': agent_id,
                                        'target_id': target_id,
                                        'action': behavior,
                                        'start_frame': int(start_frame),
                                        'stop_frame': int(end_frame)
                                    })
                
                del df
                gc.collect()
                
            except Exception as e:
                print(f"âš ï¸� Error processing {file_path}: {e}")
                continue
        
        # Create DataFrame with correct column order
        submission_df = pd.DataFrame(all_predictions)
        
        # Ensure correct column order
        submission_df = submission_df[['row_id', 'video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']]
        
        # Save to CSV
        submission_path = '/kaggle/working/submission.csv'
        submission_df.to_csv(submission_path, index=False)
        
        print(f"âœ… SUBMISSION.CSV CREATED: {submission_df.shape}")
        print(f"ğŸ“� Saved: {submission_path}")
        
        print("\nğŸ“Š Submission preview:")
        print(submission_df.head())
        
        print("\nğŸ“ˆ Behavior distribution:")
        print(submission_df['action'].value_counts())
        
        print(f"\nğŸ�­ Unique agent IDs: {sorted(submission_df['agent_id'].unique())}")
        print(f"ğŸ�­ Unique target IDs: {sorted(submission_df['target_id'].unique())}")
        
        return submission_df

    def run_complete_pipeline(self):
        """Run the complete pipeline"""
        print("=" * 80)
        print("ğŸš€ MOUSE BEHAVIOR DETECTION PIPELINE")
        print("ğŸ�¯ COMPLETE TRAINING + SUBMISSION")
        print("=" * 80)
        
        try:
            # Step 1: Process training data
            print("\n1. ğŸ“¥ PROCESSING TRAINING DATA")
            all_features, all_labels = self.process_training_data()
            
            if not all_features:
                print("â�Œ No training data processed!")
                return 0
            
            # Step 2: Prepare features
            print("\n2. ğŸ› ï¸� PREPARING FEATURES")
            X, y, feature_columns = self.prepare_features(all_features, all_labels)
            
            # Step 3: Train model
            print("\n3. ğŸ§  TRAINING MODEL")
            model = self.train_model(X, y)
            
            # Step 4: Create submission
            print("\n4. ğŸ“¤ CREATING SUBMISSION")
            submission = self.create_submission(model, feature_columns)
            
            print("\n" + "=" * 80)
            print("ğŸ�‰ PIPELINE COMPLETED SUCCESSFULLY!")
            
            if submission is not None:
                print("ğŸ“� Submission saved: /kaggle/working/submission.csv")
                print(f"ğŸ“Š Total predictions: {len(submission)}")
            
            print("=" * 80)
            return 1
            
        except Exception as e:
            print(f"â�Œ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return 0

# Run the pipeline
if __name__ == "__main__":
    DATASET_PATH = "/kaggle/input/MABe-mouse-behavior-detection"
    
    print("ğŸš€ STARTING MOUSE BEHAVIOR PIPELINE")
    print("=" * 80)
    
    pipeline = MouseBehaviorPipeline(DATASET_PATH)
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("âœ… PIPELINE COMPLETED SUCCESSFULLY!")
        print("ğŸ�¯ Your submission.csv is ready!")
    else:
        print("â�Œ Pipeline failed")

