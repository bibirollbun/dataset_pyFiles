
"""
Mouse Behavior Detection Competition - Simple Metadata-Only Baseline
A minimal baseline that uses only metadata to create valid predictions
"""

import pandas as pd
import numpy as np
import json
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

class SimpleMouseBehaviorPredictor:
    def __init__(self):
        self.lab_behaviors = {}
        self.behavior_durations = {}
        
    def load_data(self, data_dir: str = '/kaggle/input/MABe-mouse-behavior-detection'):
        """Load training and test data"""
        print("Loading metadata...")
        self.train_df = pd.read_csv(f'{data_dir}/train.csv')
        self.test_df = pd.read_csv(f'{data_dir}/test.csv')
        
        print(f"Training videos: {len(self.train_df)}")
        print(f"Test videos: {len(self.test_df)}")
        
        # Analyze behaviors from training metadata
        self.analyze_training_behaviors()
        
    def analyze_training_behaviors(self):
        """Analyze behavior patterns from training metadata only"""
        print("Analyzing behavior patterns from metadata...")
        
        lab_behavior_counts = {}
        
        for idx, row in self.train_df.iterrows():
            lab_id = row['lab_id']
            
            if lab_id not in lab_behavior_counts:
                lab_behavior_counts[lab_id] = {}
            
            try:
                behaviors_labeled = json.loads(row['behaviors labeled'])
                
                for behavior_key in behaviors_labeled:
                    # Parse behavior key: "agent_id,target_id,action"
                    parts = behavior_key.split(',')
                    if len(parts) == 3:
                        action = parts[2]
                        if action not in lab_behavior_counts[lab_id]:
                            lab_behavior_counts[lab_id][action] = 0
                        lab_behavior_counts[lab_id][action] += 1
                        
            except:
                continue
        
        # Get most common behaviors per lab
        self.lab_behaviors = {}
        for lab_id, behavior_counts in lab_behavior_counts.items():
            if behavior_counts:
                # Sort by frequency
                sorted_behaviors = sorted(behavior_counts.items(), key=lambda x: x[1], reverse=True)
                self.lab_behaviors[lab_id] = [behavior for behavior, count in sorted_behaviors]
            
        print("Behavior analysis complete!")
        for lab, behaviors in self.lab_behaviors.items():
            print(f"{lab}: {len(behaviors)} behaviors - {behaviors[:3]}{'...' if len(behaviors) > 3 else ''}")
            
    def get_default_duration(self, action: str) -> tuple:
        """Get default start/stop frames for different actions"""
        # Default durations based on typical behavior patterns
        duration_map = {
            'sniff': (30, 60),
            'attack': (15, 45),
            'chase': (45, 90),
            'mount': (30, 90),
            'groom': (60, 120),
            'approach': (20, 50),
            'escape': (15, 45),
            'rear': (20, 60),
            'avoid': (20, 50),
            'defend': (20, 60),
        }
        
        # Default for unknown actions
        default_duration = (30, 80)
        
        for key_action, (min_dur, max_dur) in duration_map.items():
            if key_action in action.lower():
                return (min_dur, max_dur)
                
        return default_duration
    
    def predict_behaviors_for_video(self, video_info: pd.Series) -> List[Dict]:
        """Generate behavior predictions for a single video using only metadata"""
        predictions = []
        
        video_id = video_info['video_id']
        lab_id = video_info['lab_id']
        
        print(f"Processing video {video_id} from lab {lab_id}")
        
        # Parse behaviors that should be labeled
        try:
            behaviors_labeled_str = str(video_info['behaviors labeled'])
            if behaviors_labeled_str and behaviors_labeled_str != 'nan':
                behaviors_labeled = json.loads(behaviors_labeled_str)
            else:
                behaviors_labeled = []
        except Exception as e:
            print(f"Error parsing behaviors for video {video_id}: {e}")
            behaviors_labeled = []
            
        # If no specific behaviors are labeled, create basic predictions using lab patterns
        if not behaviors_labeled:
            lab_behaviors = self.lab_behaviors.get(lab_id, ['sniff'])
            if lab_behaviors:
                # Create a simple prediction with most common behavior for this lab
                action = lab_behaviors[0]
                min_dur, max_dur = self.get_default_duration(action)
                
                # Use video duration info if available
                try:
                    fps = float(video_info['frames per second'])
                    duration_sec = float(video_info['video duration (sec)'])
                    total_frames = int(fps * duration_sec)
                    
                    # Place prediction in middle third of video
                    start_frame = total_frames // 3
                    end_frame = min(start_frame + max_dur, total_frames - 10)
                except:
                    # Fallback frame values
                    start_frame = 100
                    end_frame = 200
                
                predictions.append({
                    'video_id': video_id,
                    'agent_id': 1,  # Default mouse IDs
                    'target_id': 2,
                    'action': action,
                    'start_frame': start_frame,
                    'stop_frame': end_frame
                })
            
            return predictions
        
        # Process each labeled behavior
        for behavior_key in behaviors_labeled:
            try:
                parts = behavior_key.split(',')
                if len(parts) != 3:
                    continue
                    
                agent_id = int(parts[0])
                target_id = int(parts[1])
                action = parts[2]
                
                # Get reasonable duration for this action
                min_dur, max_dur = self.get_default_duration(action)
                duration = np.random.randint(min_dur, max_dur + 1)
                
                # Calculate frame positions based on video info
                try:
                    fps = float(video_info['frames per second'])
                    duration_sec = float(video_info['video duration (sec)'])
                    total_frames = int(fps * duration_sec)
                    
                    # Randomly place behavior in video (avoid first and last 10%)
                    earliest_start = int(0.1 * total_frames)
                    latest_start = int(0.9 * total_frames) - duration
                    
                    if latest_start <= earliest_start:
                        start_frame = earliest_start
                    else:
                        start_frame = np.random.randint(earliest_start, latest_start)
                        
                    stop_frame = min(start_frame + duration, total_frames - 1)
                    
                except Exception as e:
                    print(f"Error calculating frames for video {video_id}: {e}")
                    # Fallback frame calculation
                    start_frame = np.random.randint(50, 500)
                    stop_frame = start_frame + duration
                
                predictions.append({
                    'video_id': video_id,
                    'agent_id': agent_id,
                    'target_id': target_id,
                    'action': action,
                    'start_frame': int(start_frame),
                    'stop_frame': int(stop_frame)
                })
                
            except Exception as e:
                print(f"Error processing behavior {behavior_key} for video {video_id}: {e}")
                continue
                
        return predictions
        
    def generate_predictions(self) -> pd.DataFrame:
        """Generate predictions for all test videos"""
        print("Generating predictions...")
        
        all_predictions = []
        
        for idx, row in self.test_df.iterrows():
            predictions = self.predict_behaviors_for_video(row)
            all_predictions.extend(predictions)
            
        print(f"Generated {len(all_predictions)} predictions")
        
        # Ensure we have at least some predictions
        if not all_predictions:
            print("Creating fallback predictions...")
            
            for idx, row in self.test_df.iterrows():
                all_predictions.append({
                    'video_id': row['video_id'],
                    'agent_id': 1,
                    'target_id': 2,
                    'action': 'sniff',
                    'start_frame': 100,
                    'stop_frame': 200
                })
        
        # Create DataFrame
        predictions_df = pd.DataFrame(all_predictions)
        predictions_df['row_id'] = range(len(predictions_df))
        
        return predictions_df
        
    def create_submission(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Create submission file in correct format"""
        required_columns = ['row_id', 'video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']
        
        # Ensure all columns exist
        for col in required_columns:
            if col not in predictions_df.columns:
                print(f"Missing required column: {col}")
                return pd.DataFrame(columns=required_columns)
        
        submission = predictions_df[required_columns].copy()
        
        # Ensure correct data types
        submission['row_id'] = submission['row_id'].astype(int)
        submission['video_id'] = submission['video_id'].astype(int)
        submission['agent_id'] = submission['agent_id'].astype(int)
        submission['target_id'] = submission['target_id'].astype(int)
        submission['start_frame'] = submission['start_frame'].astype(int)
        submission['stop_frame'] = submission['stop_frame'].astype(int)
        
        return submission

def main():
    """Main pipeline execution"""
    # Data directory - adjust path as needed
    data_dir = '/kaggle/input/MABe-mouse-behavior-detection'
    
    try:
        # Initialize predictor
        predictor = SimpleMouseBehaviorPredictor()
        
        # Load data and analyze
        predictor.load_data(data_dir)
        
        # Generate predictions
        predictions_df = predictor.generate_predictions()
        
        # Create submission
        submission_df = predictor.create_submission(predictions_df)
        
        print(f"\nSubmission shape: {submission_df.shape}")
        print(f"Unique videos: {submission_df['video_id'].nunique()}")
        print(f"Unique actions: {submission_df['action'].nunique()}")
        print("\nAction distribution:")
        print(submission_df['action'].value_counts())
        
        print("\nSubmission sample:")
        print(submission_df.head())
        
        # Validate submission
        if len(submission_df) == 0:
            raise ValueError("Empty submission!")
            
        # Save submission
        submission_df.to_csv('submission.csv', index=False)
        print("\nSubmission saved as 'submission.csv'")
        
        return submission_df
        
    except Exception as e:
        print(f"Error in main pipeline: {e}")
        
        # Create absolute minimal fallback
        print("Creating minimal fallback submission...")
        
        try:
            test_df = pd.read_csv(f'{data_dir}/test.csv')
            
            fallback_predictions = []
            for idx, row in test_df.iterrows():
                fallback_predictions.append({
                    'row_id': idx,
                    'video_id': row['video_id'],
                    'agent_id': 1,
                    'target_id': 2,
                    'action': 'sniff',
                    'start_frame': 100,
                    'stop_frame': 200
                })
            
            fallback_df = pd.DataFrame(fallback_predictions)
            fallback_df.to_csv('submission.csv', index=False)
            print("Minimal fallback submission saved!")
            
        except Exception as fallback_error:
            print(f"Even fallback failed: {fallback_error}")
            
            # Ultimate fallback
            ultra_minimal = pd.DataFrame({
                'row_id': [0],
                'video_id': [1],
                'agent_id': [1],
                'target_id': [2],
                'action': ['sniff'],
                'start_frame': [100],
                'stop_frame': [200]
            })
            ultra_minimal.to_csv('submission.csv', index=False)
            print("Ultra-minimal submission created!")

if __name__ == "__main__":
    main()




