"""
Big Data Bowl 2026 - Expected Completion Percentage (xCP) Model v5.2
====================================================================
This code has the following capabilities:
- Ingests BDB 2026 Input & Output Data
- Processes data (including calculating speed, direction, and acceleration)
- Builds Hierarchical xCP Model to predict % completion throughout the throw
- Assigns responsibility for changes in xCP to defenders based on location and attention

Author: @BRosenkranz
Date: December 2025
"""

import time
# Record the start time
start_time = time.time()

# SECTION 0: IMPORTS AND CONFIGURATION
# ============================================================================

#General Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import gc
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from tqdm import tqdm
import copy
import pickle
from scipy.signal import savgol_filter
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score, log_loss, accuracy_score, 
    precision_recall_curve, average_precision_score,
    roc_curve, confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


#Configuration Settings
@dataclass
class Config:
    """Model and training configuration"""
    
    # Temporal context settings
    NUM_HISTORICAL_FRAMES: int = 5  # Number of frames to look back for temporal context
    
    #Flag if model should be trained on all post-throw frames or just release frame
    USE_MULTIFRAME_TRAINING: bool = True  # Train on all frames (not just release)

    #Number of frames to exclude from end of prediction (0 to disable)
    EXCLUDE_LATE_FRAMES_FROM_TRAINING: int = 2 

    #Set to true if want to duplicate certain plays across Y-axis
    USE_AUGMENTATION: bool = False
    
    # Model architecture
    REC_FEATURE_DIM: int = 11  # Receiver features
    OTHER_FEATURE_DIM: int = 11  # Defender features
    D_MODEL: int = 256
    N_HEADS: int = 8
    DROPOUT: float = 0.3
    TOPK_DEFENDERS: int = 3  # Number of top defenders to pool
    
    # Training settings
    BATCH_SIZE: int = 128
    LEARNING_RATE: float = 2e-4
    WEIGHT_DECAY: float = 0.01
    NUM_EPOCHS: int = 75
    EARLY_STOPPING_PATIENCE: int = 10    

    # Random seed
    RANDOM_SEED: int = 55

# Initialize configuration
config = Config()

# Set random seeds for reproducibility
np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 50)
print("BIG DATA BOWL 2026 - XCP MODEL")
print("Expected Completion Percentage at Pass Release")
print("=" * 50)
print(f"\nUsing device: {device}")

print("\n Model Config:")
print(f"  Temporal Context:      {config.NUM_HISTORICAL_FRAMES} frames")
print(f"  Receiver Features:     {config.REC_FEATURE_DIM}")
print(f"  Defender Features:     {config.OTHER_FEATURE_DIM}")
print(f"  Model Dimension:       {config.D_MODEL}")
print(f"  Top-K Defenders:       {config.TOPK_DEFENDERS}")
print(f"  Dropout:               {config.DROPOUT}")
print(f"  Weight Decay:          {config.WEIGHT_DECAY}")


# 1: DATA LOADING AND PREPROCESSING
# ============================================================================

#Do calculations based on speed, orientation, etc.
def calculate_physics_from_positions(df, frame_rate=10, smooth_window=5):
    """
    Calculate speed, acceleration, and direction from x,y coordinates.
    
    Uses Savitzky-Golay smoothing to reduce noise in the derivative calculations.
    Frame 0 assumes standing still, then backward differencing is used for velocity
    and acceleration calculations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: game_id, play_id, nfl_id, frame_id, x, y
    frame_rate : int
        Frames per second (default 10 for NFL tracking data)
    smooth_window : int
        Window size for Savitzky-Golay smoothing (must be odd, 3-7 recommended)
        
    Returns:
    --------
    pd.DataFrame with added columns: s (speed), a (acceleration), dir (direction)
    """
    
    if smooth_window % 2 == 0:
        smooth_window += 1  # Ensure odd window
    
    dt = 1.0 / frame_rate  # Time between frames
    
    # Make a copy to avoid modifying original
    result_df = df.copy()
    
    # Initialize new columns
    result_df['s'] = 0.0
    result_df['a'] = 0.0
    result_df['dir'] = 0.0
    
    # Process each player in each play separately
    for (game_id, play_id, nfl_id), group in result_df.groupby(['game_id', 'play_id', 'nfl_id']):
        if len(group) < 2:
            continue
            
        # Sort by frame_id to ensure correct temporal order
        group = group.sort_values('frame_id')
        indices = group.index
        
        x = group['x'].values
        y = group['y'].values
        n_frames = len(x)
        
        # Apply smoothing if we have enough frames
        if n_frames >= smooth_window:
            try:
                x_smooth = savgol_filter(x, smooth_window, 2, mode='nearest')
                y_smooth = savgol_filter(y, smooth_window, 2, mode='nearest')
            except:
                # Fallback to original if smoothing fails
                x_smooth = x
                y_smooth = y
        else:
            x_smooth = x
            y_smooth = y
        
        # Calculate velocity components using backward differencing
        vx = np.zeros(n_frames)
        vy = np.zeros(n_frames)
        
        for i in range(n_frames):
            if i == 0:
                # Frame 0: assume standing still
                vx[i] = 0.0
                vy[i] = 0.0
            else:
                vx[i] = (x_smooth[i] - x_smooth[i-1]) / dt
                vy[i] = (y_smooth[i] - y_smooth[i-1]) / dt

        
        # Calculate speed (magnitude of velocity)
        speed = np.sqrt(vx**2 + vy**2)
        

        # Smooth velocity before calculating acceleration ***
        if n_frames >= smooth_window:
            try:
                vx = savgol_filter(vx, smooth_window, 2, mode='nearest')
                vy = savgol_filter(vy, smooth_window, 2, mode='nearest')
            except:
                pass
        
        # Calculate direction (angle of velocity vector in degrees)
        direction = np.degrees(np.arctan2(vy, vx))

        #Convert to NFL Convention
        #direction = (90 - direction) % 360
        
        # Handle frame 0: use direction from frame 1 if available
        if n_frames > 1 and speed[0] < 0.01:
            direction[0] = direction[1]

        # Calculate acceleration components (change in velocity vector)
        ax = np.zeros(n_frames)
        ay = np.zeros(n_frames)
        
        for i in range(n_frames):
            if i == 0:
                ax[i] = 0.0
                ay[i] = 0.0
            else:
                # Backward difference in velocity components
                ax[i] = (vx[i] - vx[i-1]) / dt
                ay[i] = (vy[i] - vy[i-1]) / dt
        
        # Calculate acceleration magnitude (includes both tangential and centripetal components)
        accel = np.sqrt(ax**2 + ay**2)

        # Cap at realistic maximum 
        accel = np.clip(accel, 0, 10.0)
        
        # Apply light smoothing to acceleration to reduce noise
        if n_frames >= 7: 
            accel_smooth = np.zeros(n_frames)
            for i in range(n_frames):
                if i < 2:
                    accel_smooth[i] = accel[i]
                elif i >= n_frames - 2:
                    accel_smooth[i] = accel[i]
                else:
                    # 5-point moving average
                    accel_smooth[i] = (accel[i-2] + accel[i-1] + accel[i] + accel[i+1] + accel[i+2]) / 5
            accel = accel_smooth
        elif n_frames >= 3:
            # Fallback to 3-point if not enough frames
            accel_smooth = np.zeros(n_frames)
            for i in range(n_frames):
                if i == 0:
                    accel_smooth[i] = accel[i]
                elif i == n_frames - 1:
                    accel_smooth[i] = accel[i]
                else:
                    accel_smooth[i] = (accel[i-1] + accel[i] + accel[i+1]) / 3
            accel = accel_smooth
        
        # Assign back to dataframe
        result_df.loc[indices, 's'] = speed
        result_df.loc[indices, 'a'] = accel
        result_df.loc[indices, 'dir'] = direction
    
    return result_df

class NFLDataProcessor:
    """Handles data loading and initial preprocessing"""
    
    def __init__(self, config: Config):
        self.config = config
        self.base_path = Path('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final')
        self.train_path = self.base_path / 'train'
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and merge input + output tracking data with continuous frame numbering.
        Adds QB to output frames (frozen at release position)
        """
        print("LOADING AND MERGING INPUT + OUTPUT DATA")
        
        # Load supplementary data
        supplementary_df = pd.read_csv(self.base_path / 'supplementary_data.csv')
        
        # Filter valid plays
        supplementary_df = supplementary_df[
            (supplementary_df['play_nullified_by_penalty'] != 'Y') &
            (supplementary_df['pass_result'].isin(['C', 'I', 'IN']))
        ]
        
        # Create completion target
        supplementary_df['completion_target'] = (supplementary_df['pass_result'] == 'C').astype(int)
        
        print(f"\nValid plays: {len(supplementary_df):,}")
        print(f"  Completions: {(supplementary_df['completion_target']==1).sum():,} ({(supplementary_df['completion_target']==1).mean():.1%})")
        print(f"  Incompletions (incl. interceptions): {(supplementary_df['completion_target']==0).sum():,} ({(supplementary_df['completion_target']==0).mean():.1%})")
        
        # Load and merge all weeks
        weeks_to_load = list(range(1, 19))
        all_weeks = []
        
        for week in tqdm(weeks_to_load, desc="Loading weeks"):
            week_str = f"{week:02d}"
            input_file = self.train_path / f'input_2023_w{week_str}.csv'
            output_file = self.train_path / f'output_2023_w{week_str}.csv'
            
            if input_file.exists() and output_file.exists():
                week_data = self._merge_input_output(input_file, output_file, week)
                all_weeks.append(week_data)
                print(f"  Week {week:2d}: {len(week_data):>8,} total frames (input + output)")
        
        # Combine all weeks
        tracking_df = pd.concat(all_weeks, ignore_index=True)
        
        print("\n" + "=" * 50)
        print(f"TOTAL DATASET: {len(tracking_df):,} frames")
        print(f"  Input frames: {(tracking_df['data_source'] == 'input').sum():,}")
        print(f"  Output frames: {(tracking_df['data_source'] == 'output').sum():,}")
        print(f"  Release frames (for training): {tracking_df['is_release_frame'].sum():,}")
        print("=" * 50)
        
        return tracking_df, supplementary_df

    def _merge_input_output(self, input_path, output_path, week_num):
        """
        Merge input and output files for a single week with continuous frame numbering.
        
        Key steps:
        1. Load input and output files
        2. Merge with continuous frame_id numbering
        3. Calculate physics on MERGED data (preserves temporal continuity)
        4. Mark data sources and release frames
        5. Add QB to output frames (after physics calculated)
        6. Add metadata
        """
        #Load data
        input_df = pd.read_csv(input_path)
        output_df = pd.read_csv(output_path)

        #Merge with continuous frame numbering
        input_df = input_df.copy()
        output_df = output_df.copy()
        input_df['data_source'] = 'input'
        output_df['data_source'] = 'output'
        
        #Get max input frame per play (for renumbering output frames)
        max_input_frames = input_df.groupby(['game_id', 'play_id'])['frame_id'].max().reset_index()
        max_input_frames.columns = ['game_id', 'play_id', 'release_frame_id']
        
        #Merge release_frame_id into output_df and renumber frames
        output_df = output_df.merge(max_input_frames, on=['game_id', 'play_id'], how='left')
        output_df['frame_id'] = output_df['frame_id'] + output_df['release_frame_id'].fillna(0).astype(int)
        output_df = output_df.drop(columns=['release_frame_id'])
        
        #Combine input and output
        merged_df = pd.concat([input_df, output_df], ignore_index=True)
        
        #Calculate physics on merged data
        merged_df = calculate_physics_from_positions(merged_df, frame_rate=10, smooth_window=5)
        
        #Mark release frames
        merged_df = merged_df.merge(max_input_frames, on=['game_id', 'play_id'], how='left')
        merged_df['is_release_frame'] = (merged_df['frame_id'] == merged_df['release_frame_id'])
        
        #Carry forward player metadata (role and side) to output frames
        print(f"Bring forward metadata to output frames...")

        #Prepare metadata from input
        metadata_cols = ['game_id', 'play_id', 'nfl_id', 'player_role', 'player_side', 
                         'player_name', 'player_position', 'play_direction', 'absolute_yardline_number', 'player_to_predict']
        available_cols = [col for col in metadata_cols if col in input_df.columns]
        
        input_metadata = input_df[available_cols].drop_duplicates()
        
        #Only update output frames
        output_mask = merged_df['data_source'] == 'output'
        output_rows = merged_df[output_mask].copy()
        
        #Drop metadata columns if they exist (they'll be re-added from merge)
        cols_to_drop = [col for col in available_cols if col in output_rows.columns and col not in ['game_id', 'play_id', 'nfl_id']]
        if cols_to_drop:
            output_rows = output_rows.drop(columns=cols_to_drop)
        
        #Merge metadata
        output_rows = output_rows.merge(
            input_metadata,
            on=['game_id', 'play_id', 'nfl_id'],
            how='left'
        )
        
        #Put back into merged_df
        merged_df = pd.concat([
            merged_df[~output_mask],  # Keep input rows as-is
            output_rows                # Output rows with metadata
        ], ignore_index=True)
        
        # Add QB to output frames after physics calc
        qb_rows_to_add = []
        
        for (game_id, play_id) in merged_df[['game_id', 'play_id']].drop_duplicates().values:
            play_data = merged_df[
                (merged_df['game_id'] == game_id) & 
                (merged_df['play_id'] == play_id)
            ]
            
            #Find QB at release frame
            qb_at_release = play_data[
                (play_data['is_release_frame']) & 
                (play_data['player_role'] == 'Passer')
            ]
            
            if len(qb_at_release) == 0:
                continue
            
            qb_at_release = qb_at_release.iloc[0]
            
            #Check if QB already exists in output frames (safety check)
            qb_in_output = play_data[
                (play_data['data_source'] == 'output') &
                (play_data['nfl_id'] == qb_at_release['nfl_id'])
            ]
            
            if len(qb_in_output) > 0:
                #If QB exists - just update positions to freeze them
                qb_output_mask = (
                    (merged_df['game_id'] == game_id) &
                    (merged_df['play_id'] == play_id) &
                    (merged_df['nfl_id'] == qb_at_release['nfl_id']) &
                    (merged_df['data_source'] == 'output')
                )
                
                merged_df.loc[qb_output_mask, 'x'] = qb_at_release['x']
                merged_df.loc[qb_output_mask, 'y'] = qb_at_release['y']
                merged_df.loc[qb_output_mask, 's'] = qb_at_release['s']
                merged_df.loc[qb_output_mask, 'a'] = qb_at_release['a']
                merged_df.loc[qb_output_mask, 'dir'] = qb_at_release['dir']
            else:
                #If QB doesn't exist - create rows for each frame in output data
                output_frames = play_data[play_data['data_source'] == 'output']['frame_id'].unique()
                
                for frame_id in output_frames:
                    qb_row = qb_at_release.to_dict()
                    qb_row['frame_id'] = frame_id
                    qb_row['data_source'] = 'output'
                    qb_row['is_release_frame'] = False
                    qb_rows_to_add.append(qb_row)
        
        #Add all QB rows back
        if qb_rows_to_add:
            qb_df = pd.DataFrame(qb_rows_to_add)
            merged_df = pd.concat([merged_df, qb_df], ignore_index=True)
            print(f"    Added {len(qb_rows_to_add)} QB rows to output frames")
        
        #Drop orientation column if exists (can't calculate in output data, so removing for consistency)
        if 'o' in merged_df.columns:
            merged_df = merged_df.drop(columns=['o'])
        
        merged_df['week'] = week_num
        
        return merged_df


# 2: FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """
    Extracts features at the moment of pass release (or all frames if multi-frame enabled).
    
    Features are context-based (positioning and movement) rather than player-identity based.
    Temporal context is captured by looking back NUM_HISTORICAL_FRAMES from release.
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        #Track feature statistics for diagnostics
        self.feature_stats = {
            'receiver': {'count': 0, 'nan_count': 0, 'inf_count': 0},
            'defender': {'count': 0, 'nan_count': 0, 'inf_count': 0}
        }
        
        #Track processing for multi-frame mode
        self.frames_processed = 0
        self.frames_skipped_no_history = 0
        self.frames_skipped_late_frames = 0
        
    def process_play(self, play_data: pd.DataFrame):
        """
        Process a play and return features.
        
        If USE_MULTIFRAME_TRAINING=True: Returns List[Dict] with features for all valid frames
        If USE_MULTIFRAME_TRAINING=False: Returns Dict with features for release frame only
        
        Returns a dictionary (or list of dictionaries) with receiver and defender features.
        """
        #Identify release frame and target
        release_frame = play_data[play_data['frame_id'] == play_data['frame_id'].max()]
        
        qb = release_frame[release_frame['player_role'] == 'Passer']
        target = release_frame[release_frame['player_role'] == 'Targeted Receiver']
        
        if qb.empty or target.empty:
            raise ValueError("Missing QB or Target Receiver")
        
        target_id = target.iloc[0]['nfl_id']
        
        #Standardize plays so all go from left->right
        play_dir = play_data.iloc[0]['play_direction']
        play_data = self._standardize_direction(play_data, play_dir)
        
        #Get all frames from snap to end
        frames_data = play_data.copy()
        
        #If Multi-Frame is true, extract features for all valid frames
        if self.config.USE_MULTIFRAME_TRAINING:
            return self._extract_all_frame_features(frames_data, target_id)
        else:
            #Otherwise extract features at release frame only
            return self._extract_release_features(frames_data, target_id)

    def _extract_all_frame_features(self, frames_data: pd.DataFrame, target_id: int) -> List[Dict]:
        """
        NEW METHOD: Extract features for all frames in a play.
        
        Skips frames without sufficient history (first NUM_HISTORICAL_FRAMES - 1 frames).
        Each frame gets features from its temporal window.
        """
        all_frame_features = []
        
        #Get all unique frame IDs
        all_frame_ids = sorted(frames_data['frame_id'].unique())
        
        #Find release frame for metadata
        release_frames = frames_data[frames_data['is_release_frame']]
        if len(release_frames) == 0:
            return []
        release_frame_id = release_frames['frame_id'].iloc[0]
        
        #Process each frame that has sufficient history - Skip first (historical_frames - 1) frames since they don't have enough lookback
        min_frame_id = all_frame_ids[0] + (self.config.NUM_HISTORICAL_FRAMES - 1)
        
        for current_frame_id in all_frame_ids:
            #Skip if not enough history
            if current_frame_id < min_frame_id:
                self.frames_skipped_no_history += 1
                continue

            #Only keep frames starting from the release - out of scope for now given we're looking at ball-in-air movement
            if current_frame_id < release_frame_id:
                continue
                                    
            try:
                #Get temporal window for this frame
                temporal_window = self._get_temporal_frames(frames_data, current_frame_id)
                
                #Verify we have enough frames
                if len(temporal_window['frame_id'].unique()) < self.config.NUM_HISTORICAL_FRAMES:
                    self.frames_skipped_no_history += 1
                    continue
                
                #Extract features for this frame
                frame_features = self._extract_release_features(
                    temporal_window, 
                    target_id,
                    force_frame_id=current_frame_id  # Tell method which frame we're targeting
                )
                
                #Add metadata for this frame
                frame_features['frame_id'] = current_frame_id
                frame_features['is_release_frame'] = (current_frame_id == release_frame_id)
                frame_features['frames_from_release'] = current_frame_id - release_frame_id
                
                all_frame_features.append(frame_features)
                self.frames_processed += 1
                
            except Exception as e:
                # Skip frames with errors
                continue
        
        return all_frame_features

    def _standardize_direction(self, df: pd.DataFrame, play_direction: str) -> pd.DataFrame:
        """
        Standardize play direction so all plays move left-to-right.
        
        Adjusts direction angles to point toward endzone (0° = toward endzone),
        then flips coordinates for plays moving left.
        """
        df = df.copy()
    
        # Normalize arctan2 output (-180 to 180) to 0-360 range
        df['dir'] = df['dir'] % 360 
        
        # Rotate field 180 degrees if play goes left
        if play_direction == 'left':
            df['x'] = 120 - df['x']
            df['y'] = 53.3 - df['y']           
            df['dir'] = (df['dir'] + 180) % 360 
        
        return df

    def _get_temporal_frames(self, play_data: pd.DataFrame, current_frame_id: int) -> pd.DataFrame:
        """
        Get the last NUM_HISTORICAL_FRAMES frames up to and including current_frame_id.
        """
        start_frame = max(1, current_frame_id - self.config.NUM_HISTORICAL_FRAMES + 1)
        return play_data[play_data['frame_id'].between(start_frame, current_frame_id)].copy()

    def _extract_release_features(self, frames: pd.DataFrame, target_id: int, force_frame_id: int = None) -> Dict:
        """
        Extract features at specified frame with temporal context. 
        """
        features = {'receiver': None, 'others': []}
        
        #Determine which frame we're extracting features for
        if force_frame_id is not None:
            release_frame_id = force_frame_id
        else:
            release_frame_id = frames['frame_id'].max()
        
        #Get time window for feature extraction
        temporal_frames = self._get_temporal_frames(frames, release_frame_id)
        
        #Only include players we have data for in all frames (player_to_predict == True OR Passer)
        eligible_players = temporal_frames[
            (temporal_frames['player_to_predict'] == True) | 
            (temporal_frames['player_role'] == 'Passer')
        ].copy()
        
        player_groups = eligible_players.groupby('nfl_id')
        
        if target_id not in player_groups.groups:
            raise ValueError("Target ID not found")
        
        target_frames = player_groups.get_group(target_id)
        
        #Calculate QB states for each frame to serve as reference
        qb_frames = temporal_frames[temporal_frames['player_role'] == 'Passer']
        qb_states = {}
        
        if not qb_frames.empty:
            for _, qf in qb_frames.iterrows():
                rad_dir = np.radians(qf['dir'])
                vx = qf['s'] * np.cos(rad_dir)
                retreat_speed = -vx if vx < 0 else 0  # Negative x-velocity = retreating
                
                qb_states[qf['frame_id']] = {
                    's': qf['s'],
                    'retreat': retreat_speed,
                    'a': qf['a'],
                    'x': qf['x'],
                    'y': qf['y']
                }
        
        #Now extract target receiver features
        rec_feats = []
        rec_states = [] 
        
        for _, frame in target_frames.iterrows():
            fid = frame['frame_id']
            qs = qb_states.get(fid, {'s': 0, 'retreat': 0, 'a': 0, 'x': 0, 'y': 0})
            
            # Position relative to QB
            x_rel_qb = frame['x'] - qs['x']
            y_rel_qb = frame['y'] - qs['y']
            
            # Velocity components
            vx = frame['s'] * np.cos(np.radians(frame['dir']))
            vy = frame['s'] * np.sin(np.radians(frame['dir']))
            
            # Store receiver for defender calculations
            rec_states.append({'x': frame['x'], 'y': frame['y'], 'vx': vx, 'vy': vy})
            
            # Distance to QB
            rec_to_qb_dist = np.sqrt((frame['x'] - qs['x'])**2 + (frame['y'] - qs['y'])**2)
            
            # Radial velocity (velocity movements toward/away from QB)
            if rec_to_qb_dist > 0.1:
                qb_radial_velocity = ((frame['x'] - qs['x']) * vx + (frame['y'] - qs['y']) * vy) / rec_to_qb_dist
            else:
                qb_radial_velocity = 0.0
            
            # Receiver feature vector (11 features)
            f_vec = [
                x_rel_qb,           # x position relative to QB
                y_rel_qb,           # y position relative to QB
                frame['s'],         # receiver speed
                frame['a'],         # receiver acceleration
                vx,                 # receiver x-velocity
                vy,                 # receiver y-velocity
                qs['s'],            # QB speed
                qs['retreat'],      # QB retreat speed
                qs['a'],            # QB acceleration
                rec_to_qb_dist,     # distance to QB
                qb_radial_velocity  # velocity toward/away from QB
            ]
            
            rec_feats.append(f_vec)
            
            # Track stats
            self.feature_stats['receiver']['count'] += 1
            if np.any(np.isnan(f_vec)):
                self.feature_stats['receiver']['nan_count'] += 1
            if np.any(np.isinf(f_vec)):
                self.feature_stats['receiver']['inf_count'] += 1
        
        features['receiver'] = {'id': target_id, 'frames': rec_feats}
        
        # Process Defenders (built to also include other offensive players if needed, but exluded for now given model projects beyond pass release)
        for pid, p_frames in player_groups:
            if pid == target_id:
                continue
            if p_frames.iloc[0]['player_role'] == 'Passer':
                continue
            
            other_feats = []
            role_code = self._encode_role(p_frames.iloc[0]['player_role'])
            
            for i, (_, frame) in enumerate(p_frames.iterrows()):
                if i >= len(rec_states):
                    break
                
                r_state = rec_states[i]
                fid = frame['frame_id']
                qs = qb_states.get(fid, {'x': 0, 'y': 0})
                
                # Calc Physics
                vx = frame['s'] * np.cos(np.radians(frame['dir']))
                vy = frame['s'] * np.sin(np.radians(frame['dir']))
                
                # Position relative to receiver
                dx = frame['x'] - r_state['x']
                dy = frame['y'] - r_state['y']
                dist = np.sqrt(dx**2 + dy**2)
                
                # Relative velocity calc
                dvx = vx - r_state['vx']
                dvy = vy - r_state['vy']
                
                # Closing speed (to help start model going in right direction with spatial relationships)
                closing_speed = 0
                if dist > 0:
                    closing_speed = -1 * (dvx * dx + dvy * dy) / dist
                
                # Proj future separation (1.5 seconds ahead)
                future_time = 1.5
                fut_def_x = frame['x'] + (vx * future_time)
                fut_def_y = frame['y'] + (vy * future_time)
                fut_rec_x = r_state['x'] + (r_state['vx'] * future_time)
                fut_rec_y = r_state['y'] + (r_state['vy'] * future_time)
                future_sep = np.sqrt((fut_def_x - fut_rec_x)**2 + (fut_def_y - fut_rec_y)**2)
                
                # Defender Advantage - closing speed with directional alignment
                pursuit_angle = np.arctan2(dy, dx) if dist > 0 else 0
                dir_alignment = np.cos(np.radians(frame['dir']) - pursuit_angle)
                defender_advantage = closing_speed * dir_alignment
                
                # Time to Collision - how long until defender reaches receiver
                if dist > 0 and closing_speed > 0:
                    time_to_intercept = min(dist / closing_speed, 10.0)
                else:
                    time_to_intercept = 10.0
                
                # In Passing Lane: perpendicular distance from QB-receiver line - not perfect, assumes need to be within 5 yards of QB or receiver to impact throw
                def_to_qb_dist = np.sqrt((qs['x'] - frame['x'])**2 + (qs['y'] - frame['y'])**2)
                is_near_action = (def_to_qb_dist < 5.0) or (dist < 5.0)
                
                if is_near_action:
                    qb_to_rec_x = r_state['x'] - qs['x']
                    qb_to_rec_y = r_state['y'] - qs['y']
                    qb_to_rec_dist = np.sqrt(qb_to_rec_x**2 + qb_to_rec_y**2)
                    
                    def_to_qb_x = qs['x'] - frame['x']
                    def_to_qb_y = qs['y'] - frame['y']
                    
                    cross_prod = abs((qb_to_rec_x * def_to_qb_y) - (qb_to_rec_y * def_to_qb_x))
                    perpendicular_dist = cross_prod / qb_to_rec_dist if qb_to_rec_dist > 0 else 999
                    
                    in_passing_lane = 1.0 if perpendicular_dist < 5.0 else 0.0
                else:
                    in_passing_lane = 0.0
                
                # Defender feature vector (11 features)
                f_vec = [
                    dx,                    # x distance to receiver
                    dy,                    # y distance to receiver
                    dist,                  # total distance to receiver
                    vx,                    # defender x-velocity
                    vy,                    # defender y-velocity
                    closing_speed,         # closing speed on receiver
                    future_sep,            # projected future separation
                    float(role_code),      # role encoding
                    defender_advantage,    # advantage metric
                    time_to_intercept,     # time to intercept
                    in_passing_lane        # binary passing lane indicator
                ]
                
                other_feats.append(f_vec)
                
                # Track statistics
                self.feature_stats['defender']['count'] += 1
                if np.any(np.isnan(f_vec)):
                    self.feature_stats['defender']['nan_count'] += 1
                if np.any(np.isinf(f_vec)):
                    self.feature_stats['defender']['inf_count'] += 1
            
            features['others'].append({
                'id': pid, 
                'frames': other_feats,
                'role': p_frames.iloc[0]['player_role']
            })
        
        # Add metadata for potential visualization use (only for release frame mode)
        if not self.config.USE_MULTIFRAME_TRAINING:
            features['frame_id'] = release_frame_id
            features['is_release_frame'] = True
            features['frames_before_release'] = 0
        
        return features

    def _encode_role(self, role: str) -> int:
        """Encode player role as integer"""
        if role == 'Defensive Coverage':
            return 1
        if role == 'Defensive Line':
            return 2
        return 0  # Other receivers


    def print_diagnostics(self):
        """Print feature extraction Notes"""
        print("\n" + "=" * 50)
        print("FEATURE EXTRACTION DIAGNOSTICS")
        print("=" * 50)
        
        if self.config.USE_MULTIFRAME_TRAINING:
            print(f"\nMulti-frame Training Mode:")
            print(f"  Frames processed: {self.frames_processed:,}")
            print(f"  Frames skipped (no history): {self.frames_skipped_no_history:,}")
            
            if self.config.EXCLUDE_LATE_FRAMES_FROM_TRAINING > 0:
                print(f"  Frames skipped (late frames): {self.frames_skipped_late_frames:,}")
                total_skipped = self.frames_skipped_no_history + self.frames_skipped_late_frames
                total_possible = self.frames_processed + total_skipped
                pct_excluded = self.frames_skipped_late_frames / total_possible * 100 if total_possible > 0 else 0
                print(f"  Late frame exclusion: {self.config.EXCLUDE_LATE_FRAMES_FROM_TRAINING} frames ({pct_excluded:.1f}%)")

        #Check for errors in processing
        print("\nReceiver Features:")
        print(f"  Total extracted: {self.feature_stats['receiver']['count']:,}")
        print(f"  NaN occurrences: {self.feature_stats['receiver']['nan_count']:,}")
        print(f"  Inf occurrences: {self.feature_stats['receiver']['inf_count']:,}")
        
        print("\nDefender Features:")
        print(f"  Total extracted: {self.feature_stats['defender']['count']:,}")
        print(f"  NaN occurrences: {self.feature_stats['defender']['nan_count']:,}")
        print(f"  Inf occurrences: {self.feature_stats['defender']['inf_count']:,}")
        
        if self.feature_stats['receiver']['nan_count'] > 0 or self.feature_stats['defender']['nan_count'] > 0:
            print("\n WARNING: Oops, NaN values detected in features")
        if self.feature_stats['receiver']['inf_count'] > 0 or self.feature_stats['defender']['inf_count'] > 0:
            print("\n WARNING: Oops, Infinite values detected in features")


# SECTION 3: DATA AUGMENTATION
# ============================================================================

def augment_minority_class(features: List[Dict], targets: pd.DataFrame) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Augment incomplete passes by mirroring across the y-axis.This creates physically valid variations by flipping the field horizontally, helping to balance the dataset and improving model generalization.
    Warning: Has potential for data leakage if not transformed careful (validate via "test data" to ensure no overfitting)
    """
    augmented_features = []
    augmented_targets = []
    
    for feat, (_, target_row) in zip(features, targets.iterrows()):
        augmented_features.append(feat)
        augmented_targets.append(target_row)
        
        # If incomplete, add mirrored version
        if target_row['completion_target'] == 0:
            mirrored_feat = copy.deepcopy(feat)
            
            # Flip receiver y components
            for rec_frame in mirrored_feat['receiver']['frames']:
                rec_frame[1] = -rec_frame[1]   # y_rel_qb
                rec_frame[5] = -rec_frame[5]   # vy
            
            # Flip defender y components
            for other in mirrored_feat['others']:
                for def_frame in other['frames']:
                    def_frame[1] = -def_frame[1]   # dy
                    def_frame[4] = -def_frame[4]   # vy
            
            # Preserve metadata
            if 'frame_id' in mirrored_feat:
                mirrored_feat['frame_id'] = feat['frame_id']
            if 'is_release_frame' in mirrored_feat:
                mirrored_feat['is_release_frame'] = feat['is_release_frame']
            if 'frames_before_release' in mirrored_feat:
                mirrored_feat['frames_before_release'] = feat['frames_before_release']
            if 'frames_from_release' in mirrored_feat:
                mirrored_feat['frames_from_release'] = feat['frames_from_release']
            
            augmented_features.append(mirrored_feat)
            augmented_targets.append(target_row)
    
    augmented_targets_df = pd.DataFrame(augmented_targets).reset_index(drop=True)
    
    return augmented_features, augmented_targets_df


# 4: MODEL ARCHITECTURE
# ============================================================================

class CrossAttentionFusion(nn.Module):
    """
    Cross-attention mechanism where receiver attends to all defenders. Attention weights represent each defender's "importance" to the prediction.
    """
    
    def __init__(self, rec_dim, def_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.rec_dim = rec_dim
        self.def_dim = def_dim
        self.num_heads = num_heads
        self.head_dim = rec_dim // num_heads
        
        assert rec_dim % num_heads == 0, "rec_dim must be divisible by num_heads"
        
        # Project receiver embedding to query
        self.q_proj = nn.Linear(rec_dim, rec_dim)
        
        # Project defender embeddings to key and value
        self.k_proj = nn.Linear(def_dim, rec_dim)
        self.v_proj = nn.Linear(def_dim, rec_dim)
        
        # Output projection
        self.out_proj = nn.Linear(rec_dim, rec_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, rec_emb, def_emb, def_mask=None, return_attention=False):
        """
        Args:
            rec_emb: (batch_size, rec_dim) - receiver embedding
            def_emb: (batch_size, num_defenders, def_dim) - defender embeddings
            def_mask: (batch_size, num_defenders) - True for valid defenders, False for padding
            return_attention: whether to return attention weights
            
        Returns:
            context: (batch_size, rec_dim) - context vector from attending to defenders
            attn_weights: (batch_size, num_defenders) - attention weights (only if return_attention=True)
        """
        batch_size = rec_emb.size(0)
        num_defenders = def_emb.size(1)
        
        # Project to Q, K, V
        Q = self.q_proj(rec_emb).unsqueeze(1)  # (batch, 1, rec_dim)
        K = self.k_proj(def_emb)  # (batch, num_def, rec_dim)
        V = self.v_proj(def_emb)  # (batch, num_def, rec_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_defenders, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_defenders, self.num_heads, self.head_dim).transpose(1, 2)
        #Now: Q (batch, heads, 1, head_dim), K/V (batch, heads, num_def, head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # scores: (batch, heads, 1, num_def)
        
        # Apply mask for padding
        if def_mask is not None:
            # Expand mask for heads and query dimension
            mask_expanded = def_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, num_def)
            scores = scores.masked_fill(~mask_expanded, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights_per_head = F.softmax(scores, dim=-1)  # (batch, heads, 1, num_def)
        
        # Handle all-masked case (no valid defenders)
        attn_weights_per_head = torch.nan_to_num(attn_weights_per_head, nan=0.0)
        
        # Apply dropout
        attn_weights_per_head = self.dropout(attn_weights_per_head)
        
        # Compute context
        context = torch.matmul(attn_weights_per_head, V)  # (batch, heads, 1, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, self.rec_dim)
        context = self.out_proj(context.squeeze(1))  # (batch, rec_dim)
        
        if return_attention:
            # Average attention across heads for interpretability
            attn_weights = attn_weights_per_head.mean(dim=1).squeeze(1)
            return context, attn_weights
        
        return context

class HierarchicalXCP(nn.Module):
    """
    Hierarchical Expected Completion Percentage Model with Cross-Attention.
    
    Steps:
    1. Encode receiver temporal features
    2. Encode defender temporal features  
    3. Cross-attention: receiver attends to all defenders
    4. Fuse receiver embedding with attention-weighted context
    5. Predict completion probability
    """
    
    def __init__(self, config: Config):
        super().__init__()
        
        self.config = config
        
        # Calculate input dimensions (temporal features flattened)
        self.rec_input_dim = config.NUM_HISTORICAL_FRAMES * config.REC_FEATURE_DIM
        self.other_input_dim = config.NUM_HISTORICAL_FRAMES * config.OTHER_FEATURE_DIM
        self.d_model = config.D_MODEL
        
        # Receiver encoder
        self.rec_encoder = nn.Sequential(
            nn.Linear(self.rec_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(256, self.d_model),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU()
        )
        
        # Defender encoder
        self.def_encoder_body = nn.Sequential(
            nn.Linear(self.other_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(128, self.d_model)
        )
        
        # Defender batch norm (applied after reshaping)
        self.def_bn = nn.BatchNorm1d(self.d_model)
        
        # Cross-attention
        self.cross_attention = CrossAttentionFusion(
            rec_dim=self.d_model,
            def_dim=self.d_model,
            num_heads=config.N_HEADS,
            dropout=config.DROPOUT
        )
        
        # Fusion: combines receiver embedding with attention context
        fusion_input_dim = self.d_model * 2  
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, 1)
        )
        
        # Store attention weights during forward pass (for inference)
        self._last_attention_weights = None
    
    def forward(self, rec_feats, others_feats, return_attention=False):
        """
        Forward pass.
        
        Args:
            rec_feats: (batch_size, rec_input_dim) - receiver temporal features
            others_feats: (batch_size, max_defenders, other_input_dim) - defender features
            return_attention: if True, also return attention weights
        
        Returns:
            logits: (batch_size,) - completion probability logits
            attn_weights: (batch_size, max_defenders) - attention weights (if return_attention=True)
        """
        batch_size = rec_feats.size(0)
        num_defenders = others_feats.size(1)
        
        # Encode receiver
        rec_emb = self.rec_encoder(rec_feats)  # (batch_size, d_model)
        
        # Encode defenders
        if num_defenders > 0:
            # Reshape for batch processing through encoder
            others_flat = others_feats.view(-1, self.other_input_dim)
            
            # Check for all-zero defenders (padding)
            non_zero_mask = (others_flat.abs().sum(dim=-1) > 0)
            
            # Encode all defenders
            def_emb_flat = self.def_encoder_body(others_flat)
            
            # Apply batch norm only to non-padding defenders
            if non_zero_mask.any():
                def_emb_flat[non_zero_mask] = self.def_bn(def_emb_flat[non_zero_mask])
            
            # Add final ReLU
            def_emb_flat = F.relu(def_emb_flat)
            
            # Reshape back
            def_emb = def_emb_flat.view(batch_size, num_defenders, self.d_model)
            
            # Create mask for valid defenders (non-padding)
            def_mask = (others_feats.abs().sum(dim=-1) > 0)  # (batch, num_def)
            
            # Cross-attention
            context, attn_weights = self.cross_attention(
                rec_emb, def_emb, def_mask, return_attention=True
            )
            
            # Store for later retrieval
            self._last_attention_weights = attn_weights.detach()
            
        else:
            # No defenders - use zero context
            context = torch.zeros(batch_size, self.d_model, device=rec_feats.device)
            attn_weights = torch.zeros(batch_size, 0, device=rec_feats.device)
            self._last_attention_weights = attn_weights
        
        # Combine receiver embedding with attention context
        combined = torch.cat([rec_emb, context], dim=1)
        logits = self.fusion(combined).squeeze(-1)
        
        if return_attention:
            return logits, attn_weights
        
        return logits
    
    def get_last_attention_weights(self):
        #Retrieve attention weights from most recent forward pass.
        return self._last_attention_weights


# 5: DATA SCALING
# ============================================================================

class FootballDataset(Dataset):
    """
    Handles feature scaling, batching, and tracks player IDs for attribution
    """
    
    def __init__(self, features: List[Dict], targets: pd.DataFrame, config: Config, 
                 scaler=None, is_training=True):
        self.features = features
        self.targets = targets
        self.config = config
        self.is_training = is_training
        self.max_defenders = 10  # Maximum defenders to include
        
        # Fit or use existing scaler
        if is_training and scaler is None:
            self.scaler = self._fit_scaler(features)
        else:
            self.scaler = scaler
    
    def _fit_scaler(self, features):
        """Fit RobustScaler on training data"""
        all_rec_frames = []
        all_other_frames = []
        
        for play in features:
            for frame in play['receiver']['frames']:
                all_rec_frames.append(frame)
            
            for other in play['others']:
                for frame in other['frames']:
                    all_other_frames.append(frame)
        
        scaler = {
            'receiver': RobustScaler(quantile_range=(5, 95)),
            'others': RobustScaler(quantile_range=(5, 95))
        }
        
        if all_rec_frames:
            scaler['receiver'].fit(all_rec_frames)
        if all_other_frames:
            scaler['others'].fit(all_other_frames)
        
        return scaler
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        play = self.features[idx]
        target = self.targets.iloc[idx]
        
        # Process receiver features
        rec_frames = np.array(play['receiver']['frames'])
        if hasattr(self.scaler['receiver'], 'transform'):
            rec_frames = self.scaler['receiver'].transform(rec_frames)
        
        # Pad or truncate to expected number of frames
        expected_rec_frames = self.config.NUM_HISTORICAL_FRAMES
        actual_rec_frames = len(rec_frames)
        
        if actual_rec_frames < expected_rec_frames:
            padding = np.zeros((expected_rec_frames - actual_rec_frames, rec_frames.shape[1]))
            rec_frames = np.vstack([padding, rec_frames])
        elif actual_rec_frames > expected_rec_frames:
            rec_frames = rec_frames[-expected_rec_frames:]
        
        rec_flat = rec_frames.flatten()
        
        # Process defender features and track IDs/roles
        other_feats_list = []
        other_ids = []
        other_roles = []
        
        for other in play['others']:
            other_frames = np.array(other['frames'])
            actual_other_frames = len(other_frames)
            
            # Pad or truncate to match receiver frames
            if actual_other_frames < expected_rec_frames:
                padding = np.zeros((expected_rec_frames - actual_other_frames, other_frames.shape[1]))
                other_frames = np.vstack([padding, other_frames])
            elif actual_other_frames > expected_rec_frames:
                other_frames = other_frames[-expected_rec_frames:]
            
            if hasattr(self.scaler['others'], 'transform'):
                other_frames = self.scaler['others'].transform(other_frames)
            
            other_feats_list.append(other_frames.flatten())
            other_ids.append(other['id'])
            other_roles.append(other.get('role', 'Unknown'))
        
        # Limit to max defenders and pad
        if len(other_feats_list) > self.max_defenders:
            other_feats_list = other_feats_list[:self.max_defenders]
            other_ids = other_ids[:self.max_defenders]
            other_roles = other_roles[:self.max_defenders]
        
        feat_dim = self.config.NUM_HISTORICAL_FRAMES * self.config.OTHER_FEATURE_DIM
        others_array = np.zeros((self.max_defenders, feat_dim))
        
        # Pad IDs and roles
        padded_ids = [-1] * self.max_defenders  # -1 indicates padding
        padded_roles = ['Padding'] * self.max_defenders
        
        for i, feats in enumerate(other_feats_list):
            if len(feats) == feat_dim:
                others_array[i] = feats
                padded_ids[i] = other_ids[i]
                padded_roles[i] = other_roles[i]
        
        # Get metadata if available
        is_release = play.get('is_release_frame', True)
        
        return {
            'receiver': torch.FloatTensor(rec_flat),
            'others': torch.FloatTensor(others_array),
            'completion_target': torch.FloatTensor([target['completion_target']]),
            'is_release_frame': is_release,
            'other_ids': padded_ids,  # List of nfl_ids (-1 for padding)
            'other_roles': padded_roles,  # List of roles ('Padding' for padding)
            'receiver_id': play['receiver']['id'],
            'game_id': target['game_id'],
            'play_id': target['play_id']
        }
        


# 6: MODEL TRAINING COMPONENTS
# ============================================================================

def create_weighted_loss(targets_df, device):
    """
    Create BCE loss with class weighting to handle imbalanced data. Weights the positive class (completions) based on class frequencies.
    """
    n_complete = (targets_df['completion_target'] == 1).sum()
    n_incomplete = (targets_df['completion_target'] == 0).sum()
    
    pos_weight = torch.tensor([n_incomplete / n_complete], device=device)
    
    print(f"\nClass distribution:")
    print(f"  Completions: {n_complete} ({n_complete/(n_complete+n_incomplete):.1%})")
    print(f"  Incompletions: {n_incomplete} ({n_incomplete/(n_complete+n_incomplete):.1%})")
    print(f"  Positive class weight: {pos_weight.item():.3f}")
    
    #Currently not using weighted - model is converging
    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) #Utilize instead if want weighting

    return criterion

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        rec_feats = batch['receiver'].to(device)
        others_feats = batch['others'].to(device)
        comp_target = batch['completion_target'].to(device).squeeze()
        
        optimizer.zero_grad()
        
        logits = model(rec_feats, others_feats)
        loss = criterion(logits, comp_target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        batches += 1
    
    return total_loss / batches

def evaluate_model(model, dataloader, device, title="Model Evaluation", create_plots=True):
    #Comprehensive model evaluation with metrics and visualizations.
    model.eval()
    all_comp_preds = []
    all_comp_targets = []
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            rec_feats = batch['receiver'].to(device)
            others_feats = batch['others'].to(device)
            
            comp_logits = model(rec_feats, others_feats)
            comp_probs = torch.sigmoid(comp_logits).cpu().numpy()
            
            all_comp_preds.extend(comp_probs)
            all_comp_targets.extend(batch['completion_target'].cpu().numpy())
            all_logits.extend(comp_logits.cpu().numpy())
    
    y_true = np.array(all_comp_targets).flatten()
    y_pred = np.array(all_comp_preds).flatten()
    logits_array = np.array(all_logits).flatten()
    
    # Calculate metrics
    auc = roc_auc_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)
    logloss = log_loss(y_true, y_pred)

    # Find optimal threshold using Youden's J statistic (Sensitivity + Specificity - 1). This balances the model's ability to catch completions AND incompletions
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred)
    j_scores = tpr + (1 - fpr) - 1
    
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds_roc[optimal_idx]
    
    y_pred_binary = (y_pred >= optimal_threshold).astype(int)
    acc = accuracy_score(y_true, y_pred_binary)
    
    # Check on Class separation
    complete_preds = y_pred[y_true == 1]
    incomplete_preds = y_pred[y_true == 0]
    separation = complete_preds.mean() - incomplete_preds.mean()
    
    # Print metrics
    print(f"\n{title}:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  Log Loss: {logloss:.4f}")
    print(f"  Accuracy: {acc:.1%}")
    print(f"  Class Separation: {separation:.4f}")
    print(f"  Prediction Std: {y_pred.std():.4f}")
    
    if create_plots:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        axes[0, 0].plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        axes[0, 0].plot([0, 1], [0, 1], 'k--')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Prediction Distribution
        axes[0, 1].hist(complete_preds, bins=50, alpha=0.5, label='Complete', density=True)
        axes[0, 1].hist(incomplete_preds, bins=50, alpha=0.5, label='Incomplete', density=True)
        axes[0, 1].set_xlabel('Predicted Probability')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Prediction Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Calibration Curve
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
        axes[1, 0].plot(prob_pred, prob_true, 'o-', label='Model')
        axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Perfect')
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('True Probability')
        axes[1, 0].set_title('Calibration Curve')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred_binary)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 1], cmap='Blues')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('True')
        axes[1, 1].set_title('Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig('/kaggle/working/evaluation_plots.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    return {
        'auc': auc,
        'brier': brier,
        'logloss': logloss,
        'accuracy': acc,
        'separation': separation,
        'pred_std': y_pred.std()
    }

def calibrate_with_temperature(model, dataloader, device, initial_temp=1.0):
    """
    Apply temperature scaling for calibration.
    """
    print("\n" + "=" * 50)
    print("TEMPERATURE SCALING CALIBRATION")
    print("=" * 50)
    
    # Collect logits and targets
    all_logits = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting logits"):
            rec_feats = batch['receiver'].to(device)
            others_feats = batch['others'].to(device)
            logits = model(rec_feats, others_feats)
            
            all_logits.append(logits)
            all_targets.append(batch['completion_target'].to(device).squeeze())
    
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    
    # Optimize temperature
    temperature = nn.Parameter(torch.ones(1, device=device) * initial_temp)
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
    
    criterion = nn.BCEWithLogitsLoss()
    
    def eval():
        optimizer.zero_grad()
        loss = criterion(all_logits / temperature, all_targets)
        loss.backward()
        return loss
    
    optimizer.step(eval)
    
    optimal_temp = temperature.item()
    print(f"\nOptimal temperature: {optimal_temp:.4f}")
    
    # Apply temperature scaling to model
    original_forward = model.forward
    
    def calibrated_forward(rec_feats, others_feats, return_attention=False):
        #Temperature-scaled forward that supports attention
        if return_attention:
            logits, attn_weights = original_forward(rec_feats, others_feats, return_attention=True)
            return logits / optimal_temp, attn_weights
        else:
            logits = original_forward(rec_feats, others_feats, return_attention=False)
            return logits / optimal_temp
    
    model.forward = calibrated_forward
    
    return optimal_temp



def run_comprehensive_diagnostics(model, dataloader, targets_df, device):
    #Run Full diagnostic analysis.
    
    print("\n" + "=" * 50)
    print("Run full diagnostics")
    print("=" * 50)
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running diagnostics"):
            rec_feats = batch['receiver'].to(device)
            others_feats = batch['others'].to(device)
            
            logits = model(rec_feats, others_feats)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_preds.extend(probs)
            all_targets.extend(batch['completion_target'].cpu().numpy())
    
    y_true = np.array(all_targets).flatten()
    y_pred = np.array(all_preds).flatten()
    
    # Calculate metrics
    complete_mask = y_true == 1
    incomplete_mask = y_true == 0
    
    complete_preds = y_pred[complete_mask]
    incomplete_preds = y_pred[incomplete_mask]
    
    # Check Separation
    separation = complete_preds.mean() - incomplete_preds.mean()
    
    # Check Spread
    pred_std = y_pred.std()
    
    # Log loss
    logloss = log_loss(y_true, y_pred)
    
    # Compare to baseline
    baseline_pred = np.full_like(y_pred, y_true.mean())
    baseline_logloss = log_loss(y_true, baseline_pred)
    improvement = ((baseline_logloss - logloss) / baseline_logloss) * 100
    
    print(f"\nClass Separation: {separation:.4f}")
    print(f"  Complete mean: {complete_preds.mean():.4f}")
    print(f"  Incomplete mean: {incomplete_preds.mean():.4f}")
    
    print(f"\nPrediction Spread:")
    print(f"  Std: {pred_std:.4f}")
    print(f"  Min: {y_pred.min():.4f}")
    print(f"  Max: {y_pred.max():.4f}")
    
    print(f"\nLog Loss:")
    print(f"  Model: {logloss:.4f}")
    print(f"  Baseline: {baseline_logloss:.4f}")
    print(f"  Improvement: {improvement:.1f}%")
    
    # Create comprehensive plot
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Distribution by outcome
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(complete_preds, bins=50, alpha=0.5, label='Complete', density=True)
    ax1.hist(incomplete_preds, bins=50, alpha=0.5, label='Incomplete', density=True)
    ax1.axvline(complete_preds.mean(), color='blue', linestyle='--', alpha=0.8)
    ax1.axvline(incomplete_preds.mean(), color='orange', linestyle='--', alpha=0.8)
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Prediction Distribution (Separation: {separation:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Calibration curve
    ax2 = fig.add_subplot(gs[0, 1])
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
    ax2.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=8)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('True Probability')
    ax2.set_title('Calibration Curve')
    ax2.grid(True, alpha=0.3)
    
    # ROC Curve
    ax3 = fig.add_subplot(gs[0, 2])
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    ax3.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.4f}')
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curve')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Prediction bins
    ax4 = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(y_pred, bins) - 1
    bin_means = [y_true[bin_indices == i].mean() if (bin_indices == i).sum() > 0 else 0 
                 for i in range(len(bins)-1)]
    bin_counts = [(bin_indices == i).sum() for i in range(len(bins)-1)]
    ax4.bar(range(len(bin_means)), bin_means, alpha=0.7)
    ax4.set_xlabel('Prediction Bin')
    ax4.set_ylabel('Actual Completion Rate')
    ax4.set_title('Completion Rate by Prediction Bin')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Confidence vs accuracy
    ax5 = fig.add_subplot(gs[1, 1])
    confidence = np.abs(y_pred - 0.5) * 2
    conf_bins = np.linspace(0, 1, 11)
    conf_bin_indices = np.digitize(confidence, conf_bins) - 1
    binary_preds = (y_pred > 0.5).astype(int)
    conf_accuracies = [accuracy_score(y_true[conf_bin_indices == i], binary_preds[conf_bin_indices == i])
                       if (conf_bin_indices == i).sum() > 10 else np.nan
                       for i in range(len(conf_bins)-1)]
    ax5.plot(conf_bins[:-1], conf_accuracies, 'o-', linewidth=2, markersize=8)
    ax5.set_xlabel('Confidence')
    ax5.set_ylabel('Accuracy')
    ax5.set_title('Accuracy vs Confidence')
    ax5.grid(True, alpha=0.3)
    
    # Error analysis
    ax6 = fig.add_subplot(gs[1, 2])
    errors = np.abs(y_true - y_pred)
    ax6.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax6.axvline(errors.mean(), color='red', linestyle='--', label=f'Mean: {errors.mean():.3f}')
    ax6.set_xlabel('Absolute Error')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Error Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Precision-Recall
    ax7 = fig.add_subplot(gs[2, 0])
    precisions, recalls, _ = precision_recall_curve(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred)
    ax7.plot(recalls, precisions, linewidth=2, label=f'AP = {avg_precision:.4f}')
    ax7.set_xlabel('Recall')
    ax7.set_ylabel('Precision')
    ax7.set_title('Precision-Recall Curve')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Quantile analysis
    ax8 = fig.add_subplot(gs[2, 1])
    quantiles = np.percentile(y_pred, [10, 25, 50, 75, 90])
    quantile_labels = ['10%', '25%', '50%', '75%', '90%']
    ax8.barh(quantile_labels, quantiles, alpha=0.7)
    ax8.set_xlabel('Predicted Probability')
    ax8.set_title('Prediction Quantiles')
    ax8.grid(True, alpha=0.3, axis='x')
    
    # Summary statistics
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    summary_text = f"""
    SUMMARY STATISTICS
    
    Samples: {len(y_true):,}
    Completions: {complete_mask.sum():,} ({complete_mask.mean():.1%})
    Incompletions: {incomplete_mask.sum():,} ({incomplete_mask.mean():.1%})
    
    AUC: {auc:.4f}
    Brier Score: {brier_score_loss(y_true, y_pred):.4f}
    Log Loss: {logloss:.4f}
    
    Class Separation: {separation:.4f}
    Prediction Std: {pred_std:.4f}
    
    Complete Mean: {complete_preds.mean():.4f}
    Incomplete Mean: {incomplete_preds.mean():.4f}
    
    Improvement over baseline: {improvement:.1f}%
    """
    ax9.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center')
    
    plt.savefig('/kaggle/working/comprehensive_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Calculate accuracy using Youden's J
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred)
    j_scores = tpr + (1 - fpr) - 1
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds_roc[optimal_idx]
    y_pred_binary = (y_pred >= optimal_threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred_binary)
    
    return {
        'separation': separation,
        'pred_std': pred_std,
        'logloss': logloss,
        'improvement': improvement,
        'accuracy': accuracy,
        'optimal_threshold': optimal_threshold
    }


# 7. PHYSICS VALIDATION MODULE
# ============================================================================

def validate_physics_calculations(input_df: pd.DataFrame, calculated_df: pd.DataFrame, 
                                   sample_plays: int = 50, verbose: bool = True) -> Dict:
    """
    Validate calculated physics (s, a, dir) against original tracking data - input frames have original metrics
        
    """
    print("\n" + "=" * 50)
    print("Physics Validation Code")
    print("=" * 50)
    
    # Get unique plays and sample
    unique_plays = input_df[['game_id', 'play_id']].drop_duplicates()
    if len(unique_plays) > sample_plays:
        sample = unique_plays.sample(n=sample_plays, random_state=42)
    else:
        sample = unique_plays
    
    print(f"\nValidating {len(sample)} plays...")
    
    # Collect comparison data
    comparisons = []
    
    for _, row in sample.iterrows():
        game_id, play_id = row['game_id'], row['play_id']
        
        # Get original NFL data for this play
        orig = input_df[
            (input_df['game_id'] == game_id) & 
            (input_df['play_id'] == play_id)
        ][['game_id', 'play_id', 'nfl_id', 'frame_id', 's', 'a', 'dir']].copy()
        orig.columns = ['game_id', 'play_id', 'nfl_id', 'frame_id', 's_nfl', 'a_nfl', 'dir_nfl']

        # Convert BDB (CW, 0=N) to Math (CCW, 0=E) for comparison
        orig['dir_nfl'] = (90 - orig['dir_nfl']) % 360
        
        # Shift to match output (-180 to 180 range)
        orig.loc[orig['dir_nfl'] > 180, 'dir_nfl'] -= 360
        
        # Get calculated data (input frames only)
        calc = calculated_df[
            (calculated_df['game_id'] == game_id) & 
            (calculated_df['play_id'] == play_id) &
            (calculated_df['data_source'] == 'input')
        ][['game_id', 'play_id', 'nfl_id', 'frame_id', 's', 'a', 'dir']].copy()
        calc.columns = ['game_id', 'play_id', 'nfl_id', 'frame_id', 's_calc', 'a_calc', 'dir_calc']
        
        # Merge on exact frame
        merged = orig.merge(calc, on=['game_id', 'play_id', 'nfl_id', 'frame_id'], how='inner')
        
        if len(merged) > 0:
            comparisons.append(merged)
    
    if not comparisons:
        print("ERROR: No matching frames found for comparison!")
        return {}
    
    all_comparisons = pd.concat(comparisons, ignore_index=True)
    
    # Skip frame 0 (we assume standing still, but there may be real values there)
    all_comparisons = all_comparisons[all_comparisons['frame_id'] > 1]
    
    print(f"Total frame-player observations: {len(all_comparisons):,}")
    
    # Calculate errors
    all_comparisons['s_error'] = all_comparisons['s_calc'] - all_comparisons['s_nfl']
    all_comparisons['a_error'] = all_comparisons['a_calc'] - all_comparisons['a_nfl']
    
    # Direction error (manage wraparound)
    dir_diff = all_comparisons['dir_calc'] - all_comparisons['dir_nfl']
    all_comparisons['dir_error'] = np.where(
        dir_diff > 180, dir_diff - 360,
        np.where(dir_diff < -180, dir_diff + 360, dir_diff)
    )
    
    # Calculate statistics
    results = {}
    
    for var in ['s', 'a', 'dir']:
        error_col = f'{var}_error'
        nfl_col = f'{var}_nfl'
        calc_col = f'{var}_calc'
        
        errors = all_comparisons[error_col]
        
        results[var] = {
            'mean_error': errors.mean(),
            'std_error': errors.std(),
            'mae': errors.abs().mean(),
            'max_abs_error': errors.abs().max(),
            'correlation': all_comparisons[nfl_col].corr(all_comparisons[calc_col]),
            'pct_within_5pct': (errors.abs() <= all_comparisons[nfl_col].abs() * 0.05).mean() * 100,
            'pct_within_10pct': (errors.abs() <= all_comparisons[nfl_col].abs() * 0.10).mean() * 100,
        }
    
    if verbose:
        print("Speed (yards/second)")
        print(f"  Mean Error:      {results['s']['mean_error']:+.4f}")
        print(f"  Std Error:       {results['s']['std_error']:.4f}")
        print(f"  MAE:             {results['s']['mae']:.4f}")
        print(f"  Max Abs Error:   {results['s']['max_abs_error']:.4f}")
        print(f"  Correlation:     {results['s']['correlation']:.4f}")
        print(f"  Within 5%:       {results['s']['pct_within_5pct']:.1f}%")
        print(f"  Within 10%:      {results['s']['pct_within_10pct']:.1f}%")
        
        print("Acceleration (yards/second²)")
        print(f"  Mean Error:      {results['a']['mean_error']:+.4f}")
        print(f"  Std Error:       {results['a']['std_error']:.4f}")
        print(f"  MAE:             {results['a']['mae']:.4f}")
        print(f"  Max Abs Error:   {results['a']['max_abs_error']:.4f}")
        print(f"  Correlation:     {results['a']['correlation']:.4f}")
        
        print("Direction (degrees)")
        print(f"  Mean Error:      {results['dir']['mean_error']:+.2f}°")
        print(f"  Std Error:       {results['dir']['std_error']:.2f}°")
        print(f"  MAE:             {results['dir']['mae']:.2f}°")
        print(f"  Max Abs Error:   {results['dir']['max_abs_error']:.2f}°")
        print(f"  Correlation:     {results['dir']['correlation']:.4f}")
        
    return results, all_comparisons


def plot_physics_validation(comparisons_df: pd.DataFrame):
    #Create diagnostic plots for physics validation.
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Speed: scatter plot
    ax = axes[0, 0]
    ax.scatter(comparisons_df['s_nfl'], comparisons_df['s_calc'], alpha=0.1, s=1)
    max_s = max(comparisons_df['s_nfl'].max(), comparisons_df['s_calc'].max())
    ax.plot([0, max_s], [0, max_s], 'r--', label='Perfect')
    ax.set_xlabel('NFL Speed (y/s)')
    ax.set_ylabel('Calculated Speed (y/s)')
    ax.set_title(f"Speed: r={comparisons_df['s_nfl'].corr(comparisons_df['s_calc']):.3f}")
    ax.legend()
    
    # Speed: error distribution
    ax = axes[0, 1]
    ax.hist(comparisons_df['s_error'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel('Speed Error (calc - NFL)')
    ax.set_ylabel('Frequency')
    ax.set_title(f"Speed Error: μ={comparisons_df['s_error'].mean():.3f}")
    
    # Acceleration: scatter plot
    ax = axes[0, 2]
    ax.scatter(comparisons_df['a_nfl'], comparisons_df['a_calc'], alpha=0.1, s=1)
    min_a = min(comparisons_df['a_nfl'].min(), comparisons_df['a_calc'].min())
    max_a = max(comparisons_df['a_nfl'].max(), comparisons_df['a_calc'].max())
    ax.plot([min_a, max_a], [min_a, max_a], 'r--', label='Perfect')
    ax.set_xlabel('NFL Acceleration (y/s²)')
    ax.set_ylabel('Calculated Acceleration (y/s²)')
    ax.set_title(f"Accel: r={comparisons_df['a_nfl'].corr(comparisons_df['a_calc']):.3f}")
    ax.legend()
    
    # Direction: scatter plot  
    ax = axes[1, 0]
    ax.scatter(comparisons_df['dir_nfl'], comparisons_df['dir_calc'], alpha=0.1, s=1)
    ax.plot([0, 360], [0, 360], 'r--', label='Perfect')
    ax.set_xlabel('NFL Direction (°)')
    ax.set_ylabel('Calculated Direction (°)')
    ax.set_title(f"Direction: r={comparisons_df['dir_nfl'].corr(comparisons_df['dir_calc']):.3f}")
    ax.legend()
    
    # Direction: error distribution
    ax = axes[1, 1]
    ax.hist(comparisons_df['dir_error'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel('Direction Error (°)')
    ax.set_ylabel('Frequency')
    ax.set_title(f"Dir Error: μ={comparisons_df['dir_error'].mean():.1f}°")
    
    # Error vs speed (do errors get worse at high speed?)
    ax = axes[1, 2]
    ax.scatter(comparisons_df['s_nfl'], comparisons_df['s_error'].abs(), alpha=0.1, s=1)
    ax.set_xlabel('NFL Speed (y/s)')
    ax.set_ylabel('Absolute Speed Error')
    ax.set_title('Speed Error vs Speed')
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/physics_validation.png', dpi=150, bbox_inches='tight')
    plt.show()


# 8. MAIN EXECUTION
# ============================================================================

print("\n" + "=" * 50)
print("Starting Processing")
print("=" * 50)

# Load data
data_loader = NFLDataProcessor(config)
input_data, supplementary_df = data_loader.load_data()

# Merge tracking and target data
print("\nMerging tracking and supplementary data...")
merged_data = input_data.merge(
    supplementary_df[['game_id', 'play_id', 'completion_target']],
    on=['game_id', 'play_id'],
    how='inner'
)
print("Merge complete")

#Filter to only frames needed
if config.USE_MULTIFRAME_TRAINING:
    print("Using frames - release and post-release")
else:
    print("Using release frames only")

print(f"Total frames in full dataset: {len(merged_data):,}")
print(f"  Input frames (pre-release): {(merged_data['data_source'] == 'input').sum():,}")
print(f"  Output frames (post-release): {(merged_data['data_source'] == 'output').sum():,}")
print(f"  Release frames: {merged_data['is_release_frame'].sum():,}")

# Keep full dataset for later if needed
full_tracking_df = merged_data.copy()
full_tracking_df.to_csv('/kaggle/working/full_tracking_data.csv', index=False) 
    

# Set Flag: Use all frames or just release frames based on config
if config.USE_MULTIFRAME_TRAINING:
    training_df = merged_data.copy()
    print(f"\n Multi-frame mode: Using all {len(training_df):,} frames for training")
else:
    training_df = merged_data[merged_data['is_release_frame'] == True].copy()
    print(f"\n Single-frame mode: Using {len(training_df):,} release frames for training")

print(f"  Unique plays: {training_df[['game_id', 'play_id']].drop_duplicates().shape[0]:,}")
print(f"  Completion rate: {training_df['completion_target'].mean():.1%}")


######################################
#CONFIRM PHYSICS CALCS

# Quick physics validation on week 1
print("\n" + "=" * 50)
print("VALIDATING PHYSICS CALCULATIONS")
print("=" * 50)

# Load raw week 1 input for comparison
raw_input_w1 = pd.read_csv(data_loader.train_path / 'input_2023_w01.csv')

validation_results, comparison_data = validate_physics_calculations(
    input_df=raw_input_w1,
    calculated_df=merged_data[merged_data['week'] == 1],
    sample_plays=50,
    verbose=True
)

plot_physics_validation(comparison_data)


######################################
#FEATURE ENGINEERING

print("\n" + "=" * 50)
print("FEATURE ENGINEERING")
print("=" * 50)

engineer = FeatureEngineer(config)

# Test coordinate standardization
test_play = merged_data[(merged_data['game_id'] == merged_data['game_id'].iloc[0]) & 
                         (merged_data['play_id'] == merged_data['play_id'].iloc[0])]
print("\nCoordinate standardization test:")
print(f"  Original dir range: {test_play['dir'].min():.1f}° to {test_play['dir'].max():.1f}°")

play_dir = test_play.iloc[0]['play_direction']
rotated = engineer._standardize_direction(test_play, play_dir)
print(f"  Standardized dir range: {rotated['dir'].min():.1f}° to {rotated['dir'].max():.1f}°")

# Pre-group tracking data by play for efficient lookup
print("\nGrouping tracking data by play...")
play_groups = training_df.groupby(['game_id', 'play_id'])
available_play_keys = set(play_groups.groups.keys())

# Filter supplementary data to only loaded plays
print("Filtering supplementary data to loaded plays...")
supp_index = pd.Index(list(zip(supplementary_df['game_id'], supplementary_df['play_id'])))
valid_plays_df = supplementary_df[supp_index.isin(available_play_keys)].copy()

print(f"Processing {len(valid_plays_df)} plays...")

# Extract features for all plays
all_features = []
valid_play_info = []

for idx, row in tqdm(valid_plays_df.iterrows(), total=len(valid_plays_df), desc="Extracting features"):
    try:
        # Retrieve pre-grouped tracking data
        play_data = play_groups.get_group((row['game_id'], row['play_id']))
        
        # Extract features (returns Dict or List[Dict] depending on mode)
        play_features = engineer.process_play(play_data)
        
        # Handle multi-frame mode (list of dicts) vs single-frame mode (single dict)
        if config.USE_MULTIFRAME_TRAINING and isinstance(play_features, list):
            # Multi-frame: add each frame separately
            for frame_feat in play_features:
                all_features.append(frame_feat)
                
                # Create target row for this frame
                target_row = row.copy()
                target_row['game_id'] = row['game_id']
                target_row['play_id'] = row['play_id']
                if 'frame_id' in frame_feat:
                    target_row['frame_id'] = frame_feat['frame_id']
                if 'is_release_frame' in frame_feat:
                    target_row['is_release_frame'] = frame_feat['is_release_frame']
                if 'frames_from_release' in frame_feat:
                    target_row['frames_from_release'] = frame_feat['frames_from_release']
                valid_play_info.append(target_row)
        else:
            # Single-frame: only add once
            all_features.append(play_features)
            target_row = row.copy()
            target_row['game_id'] = row['game_id']
            target_row['play_id'] = row['play_id']
            valid_play_info.append(target_row)
        
    except Exception as e:
        continue

# Reconstruct targets dataframe
targets = pd.DataFrame(valid_play_info).reset_index(drop=True)

# Print diagnostics
engineer.print_diagnostics()

print(f"\n Successfully processed {len(all_features)} training examples")
if config.USE_MULTIFRAME_TRAINING:
    print(f"  From {len(valid_plays_df)} unique plays")
    print(f"  Average frames per play: {len(all_features) / len(valid_plays_df):.1f}")
print(f"  Completion rate: {targets['completion_target'].mean():.1%}")

# Show breakdown by frame type if multi-frame
if config.USE_MULTIFRAME_TRAINING and 'frames_from_release' in targets.columns:
    print(f"\nFrame type breakdown:")
    print(f"  Pre-release: {(targets['frames_from_release'] < 0).sum():,}")
    print(f"  Release: {targets.get('is_release_frame', pd.Series([False])).sum():,}")
    print(f"  Post-release: {(targets['frames_from_release'] > 0).sum():,}")

#######################################
#Create Training & Validation Split, then run augmentation (if needed)

print("\n" + "=" * 50)
print("CREATING TRAIN/VAL SPLIT")
print("=" * 50)

# Group indices by play to ensure all frames of a play go to same set (to avoid data leakage)
play_to_indices = {}
for idx, row in targets.iterrows():
    play_key = (row['game_id'], row['play_id'])
    if play_key not in play_to_indices:
        play_to_indices[play_key] = []
    play_to_indices[play_key].append(idx)

unique_plays = list(play_to_indices.keys())

# Get the target for stratifying (use the first frame's target for the play)
play_targets_list = [targets.iloc[indices[0]]['completion_target'] for indices in play_to_indices.values()]

# Split unique plays into Train and Validation sets
train_plays_keys, val_plays_keys = train_test_split(
    unique_plays,
    test_size=0.2,
    random_state=config.RANDOM_SEED,
    stratify=play_targets_list
)

train_plays_set = set(train_plays_keys)

# Assign feature indices to Train or Validation based on the play split
train_indices = []
val_indices = []

for play_key, indices in play_to_indices.items():
    if play_key in train_plays_set:
        train_indices.extend(indices)
    else:
        val_indices.extend(indices)

# Create the initial Train and Validation sets (pre-aug)
train_features = [all_features[i] for i in train_indices]
val_features = [all_features[i] for i in val_indices]

train_targets = targets.iloc[train_indices].reset_index(drop=True)
val_targets = targets.iloc[val_indices].reset_index(drop=True)

print(f"Train set (Raw): {len(train_features):,} examples from {len(train_plays_keys):,} plays")
print(f"  Completion rate: {train_targets['completion_target'].mean():.1%}")

print(f"Validation set (Real): {len(val_features):,} examples from {len(val_plays_keys):,} plays")
print(f"  Completion rate: {val_targets['completion_target'].mean():.1%}")

######################################
#Apply Data Augmentation to Training Set Only (to avoid data leakage)

if config.USE_AUGMENTATION:
    print("\n" + "=" * 50)
    print("APPLYING DATA AUGMENTATION (TRAIN ONLY)")
    print("=" * 50)
    
    print(f"Original Train size: {len(train_features)} examples")
    
    # Augment training features and targets
    train_features, train_targets = augment_minority_class(train_features, train_targets)
    
    print(f"Augmented Train size: {len(train_features)} examples")
    print(f"New Train Completion Rate: {train_targets['completion_target'].mean():.1%}")
    print(f"Validation Completion Rate (Unchanged): {val_targets['completion_target'].mean():.1%}")
    
# Need to pass to the full analysis function later
val_plays = val_plays_keys

######################################
#Create Datasets and Loaders

print("\n" + "=" * 50)
print("CREATING DATASETS")
print("=" * 50)

train_dataset = FootballDataset(train_features, train_targets, config, is_training=True)
val_dataset = FootballDataset(val_features, val_targets, config, scaler=train_dataset.scaler, is_training=False)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

######################################
#Initialize Model 

print("\n" + "=" * 50)
print("INITIALIZING MODEL")
print("=" * 50)

model = HierarchicalXCP(config).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


######################################
# Do Training

print("\n" + "=" * 50)
print("TRAINING")
print("=" * 50)

#Setup
comp_criterion = create_weighted_loss(train_targets, device)
optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

best_auc = 0
patience_counter = 0
metrics_history = []

for epoch in range(config.NUM_EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, comp_criterion, device)
    
    print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} - Loss: {train_loss:.4f}")
    
    # Evaluate every 3 epochs
    if (epoch + 1) % 3 == 0:
        metrics = evaluate_model(model, val_loader, device, 
                                title=f"Epoch {epoch+1} Validation",
                                create_plots=True)
        
        metrics['epoch'] = epoch + 1
        metrics['train_loss'] = train_loss
        metrics_history.append(metrics)
        
        scheduler.step(metrics['auc'])
        
        # Save best model
        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auc': best_auc,
                'scaler': train_dataset.scaler,
                'config': config,
                'metrics': metrics
            }, '/kaggle/working/best_model.pth')
            print(f" New best model saved (AUC: {best_auc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print("\n Early stopping triggered")
            break

#Save training metrics
metrics_df = pd.DataFrame(metrics_history)
metrics_df.to_csv('/kaggle/working/training_metrics.csv', index=False)
print(f"\n Saved training metrics to /kaggle/working/training_metrics.csv")

print(f"\n{'='*50}")
print(f"Training Complete")
print(f"Best validation AUC: {best_auc:.4f}")
print(f"{'='*50}")

######################################
#Load Best Model from Training

print("\nLoading best model for calibration and final evaluation...")
checkpoint = torch.load('/kaggle/working/best_model.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

# Apply temperature scaling
optimal_temperature = calibrate_with_temperature(model, val_loader, device)

# Save calibrated model
torch.save({
    'model_state_dict': model.state_dict(),
    'temperature': optimal_temperature,
    'scaler': checkpoint['scaler'],
    'config': checkpoint['config'],
    'auc': best_auc
}, '/kaggle/working/best_model_calibrated.pth')

print("\n Saved calibrated model to /kaggle/working/best_model_calibrated.pth")

#Save Scaler for Inference
scaler = checkpoint['scaler']

#Print Final Diagnostics
diag_results = run_comprehensive_diagnostics(model, val_loader, val_targets, device)

######################################
#Overall Summary

print("\n" + "=" * 50)
print("FINAL RESULTS SUMMARY")
print("=" * 50)

print(f"\nModel Configuration:")
print(f"  Temporal Context:      {config.NUM_HISTORICAL_FRAMES} frames")
print(f"  Receiver Features:     {config.REC_FEATURE_DIM}")
print(f"  Defender Features:     {config.OTHER_FEATURE_DIM}")
print(f"  Top-K Defenders:       {config.TOPK_DEFENDERS}")

print(f"\nPerformance:")
print(f"  Best AUC:              {best_auc:.4f}")
print(f"  Class Separation:      {diag_results['separation']:.4f}")
print(f"  Prediction Std:        {diag_results['pred_std']:.4f}")
print(f"  Log Loss:              {diag_results['logloss']:.4f}")
print(f"  Improvement over baseline: {diag_results['improvement']:.1f}%")
print(f"  Accuracy:              {diag_results['accuracy']:.1%} (threshold: {diag_results['optimal_threshold']:.3f})")


print(f"\nFiles Saved:")
print(f"  Best model:            /kaggle/working/best_model.pth")
print(f"  Calibrated model:      /kaggle/working/best_model_calibrated.pth")
print(f"  Training metrics:      /kaggle/working/training_metrics.csv")
print(f"  Diagnostics plot:      /kaggle/working/comprehensive_diagnostics.png")


# Record the end time
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Execution time at this point: {elapsed_time:.4f} seconds")


# 9. Validation Analysis
# ============================================================================

print("\n" + "=" * 50)
print("Do Validation Analysis")
print("=" * 50)

# Generate predictions on validation set
print("\nGenerating predictions on validation set...")
model.eval()
val_predictions = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Predicting"):
        rec_feats = batch['receiver'].to(device)
        others_feats = batch['others'].to(device)
        
        logits = model(rec_feats, others_feats)
        probs = torch.sigmoid(logits).cpu().numpy()
        
        val_predictions.extend(probs)

# Create predictions dataframe aligned with val_targets
val_results = val_targets.copy()
val_results['xCP'] = val_predictions
val_results['predicted_completion'] = (np.array(val_predictions) >= 0.5).astype(int)
val_results['correct'] = (val_results['predicted_completion'] == val_results['completion_target']).astype(int)

print(f" Generated {len(val_predictions):,} predictions")

######################################
#Overall Metrics

print("\n" + "-" * 50)
print("OVERALL VALIDATION METRICS")
print("-" * 50)

overall_auc = roc_auc_score(val_results['completion_target'], val_results['xCP'])
overall_acc = accuracy_score(val_results['completion_target'], val_results['predicted_completion'])
overall_brier = brier_score_loss(val_results['completion_target'], val_results['xCP'])

print(f"\nAll Frames Combined:")
print(f"  Samples:    {len(val_results):,}")
print(f"  AUC:        {overall_auc:.4f}")
print(f"  Accuracy:   {overall_acc:.1%}")
print(f"  Brier:      {overall_brier:.4f}")

######################################
# By Frame Type

if config.USE_MULTIFRAME_TRAINING and 'frames_from_release' in val_results.columns:
    print("\n" + "-" * 50)
    print("BREAKDOWN BY FRAME TYPE")
    print("-" * 50)
    
    # Define frame categories
    val_results['frame_type'] = 'other'
    val_results.loc[val_results['frames_from_release'] < 0, 'frame_type'] = 'pre_release'
    val_results.loc[val_results.get('is_release_frame', False) == True, 'frame_type'] = 'release'
    val_results.loc[val_results['frames_from_release'] > 0, 'frame_type'] = 'post_release'
    
    for frame_type in ['pre_release', 'release', 'post_release']:
        subset = val_results[val_results['frame_type'] == frame_type]
        
        if len(subset) > 0:
            auc = roc_auc_score(subset['completion_target'], subset['xCP'])
            acc = accuracy_score(subset['completion_target'], subset['predicted_completion'])
            brier = brier_score_loss(subset['completion_target'], subset['xCP'])
            
            print(f"\n{frame_type.upper().replace('_', ' ')}:")
            print(f"  Samples:    {len(subset):,}")
            print(f"  AUC:        {auc:.4f}")
            print(f"  Accuracy:   {acc:.1%}")
            print(f"  Brier:      {brier:.4f}")


# Aggregate predictions to play level
print("\n" + "-" * 50)
print("AGGREGATING TO PLAY LEVEL")
print("-" * 50)

if config.USE_MULTIFRAME_TRAINING:
    release_predictions = val_results[val_results.get('is_release_frame', False) == True].copy()
    
    if len(release_predictions) > 0:
        play_level_release = release_predictions[['game_id', 'play_id', 'completion_target', 'xCP']].copy()
        play_level_release = play_level_release.rename(columns={'xCP': 'xCP_at_release'})
        
    else:
        play_level_release = None
    
    play_level_avg = val_results.groupby(['game_id', 'play_id']).agg({
        'completion_target': 'first',
        'xCP': 'mean'
    }).reset_index()
    play_level_avg = play_level_avg.rename(columns={'xCP': 'xCP_avg_all_frames'})
    
    
    if play_level_release is not None:
        play_level_preds = play_level_release.merge(
            play_level_avg[['game_id', 'play_id', 'xCP_avg_all_frames']], 
            on=['game_id', 'play_id'], 
            how='outer'
        )
    else:
        play_level_preds = play_level_avg
        play_level_preds['xCP_at_release'] = np.nan
else:
    play_level_preds = val_results[['game_id', 'play_id', 'completion_target', 'xCP']].copy()
    play_level_preds = play_level_preds.rename(columns={'xCP': 'xCP_at_release'})

###################################
#Merge with supplementary data for analysis

print("\n" + "-" * 50)
print("MERGING WITH SUPPLEMENTARY DATA")
print("-" * 50)

# Drop completion_target from predictions to avoid duplicate columns
cols_to_merge = [col for col in play_level_preds.columns if col not in ['completion_target']]
play_level_preds_clean = play_level_preds[cols_to_merge].copy()

# Create wider supplementary dataframe with preds
supplementary_with_preds = supplementary_df.merge(
    play_level_preds_clean,
    on=['game_id', 'play_id'],
    how='left'
)

# Add flag for validation set
supplementary_with_preds['in_validation_set'] = supplementary_with_preds['xCP_at_release'].notna()

print(f"\nMerge Results:")
print(f"  Total plays in supplementary: {len(supplementary_df):,}")
print(f"  Plays with predictions:       {supplementary_with_preds['in_validation_set'].sum():,}")
print(f"  Validation plays:             {len(play_level_preds):,}")
print(f"\nPrediction columns added to supplementary:")
for col in cols_to_merge:
    if col not in ['game_id', 'play_id']:
        print(f"  - {col}")

###########################
#Analysis by Play Types

print("\n" + "-" * 50)
print("Validation Set Analysis by Play Types")
print("-" * 50)

val_only = supplementary_with_preds[supplementary_with_preds['in_validation_set']].copy()

# By pass length
if 'pass_length' in val_only.columns:
    val_only['pass_length_bin'] = pd.cut(val_only['pass_length'], 
                                          bins=[-100, 0, 10, 20, 100], 
                                          labels=['Behind LOS', 'Short (0-10)', 'Medium (10-20)', 'Deep (20+)'])
    
    print("\nBy Pass Length:")
    for length_bin in val_only['pass_length_bin'].dropna().unique():
        subset = val_only[val_only['pass_length_bin'] == length_bin]
        if len(subset) > 10:
            auc = roc_auc_score(subset['completion_target'], subset['xCP_at_release'])
            actual_comp_rate = subset['completion_target'].mean()
            avg_xcp = subset['xCP_at_release'].mean()
            print(f"  {length_bin:15s}: {len(subset):5,} plays | AUC: {auc:.3f} | Actual: {actual_comp_rate:.1%} | xCP: {avg_xcp:.1%}")

# By coverage type
if 'team_coverage_man_zone' in val_only.columns:
    print("\nBy Coverage Type:")
    for coverage in val_only['team_coverage_man_zone'].dropna().unique():
        subset = val_only[val_only['team_coverage_man_zone'] == coverage]
        if len(subset) > 10:
            auc = roc_auc_score(subset['completion_target'], subset['xCP_at_release'])
            actual_comp_rate = subset['completion_target'].mean()
            avg_xcp = subset['xCP_at_release'].mean()
            print(f"  {coverage:15s}: {len(subset):5,} plays | AUC: {auc:.3f} | Actual: {actual_comp_rate:.1%} | xCP: {avg_xcp:.1%}")


# 10. SAVE RESULTS
# ============================================================================

print("\n" + "-" * 50)
print("Saving Results")
print("-" * 50)

# Save frame-level predictions
if config.USE_MULTIFRAME_TRAINING:
    val_results.to_csv('/kaggle/working/validation_frame_predictions.csv', index=False)
    print(f" Saved frame-level predictions: /kaggle/working/validation_frame_predictions.csv")
    print(f"  Rows: {len(val_results):,}")
    print(f"  Columns: {len(val_results.columns)}")

# Save enriched supplementary data (with all play-level predictions)
supplementary_with_preds.to_csv('/kaggle/working/supplementary_with_predictions.csv', index=False)
print(f" Saved supplementary with predictions: /kaggle/working/supplementary_with_predictions.csv")
print(f"  Rows: {len(supplementary_with_preds):,}")
print(f"  Total columns: {len(supplementary_with_preds.columns)}")
print(f"  Validation plays: {supplementary_with_preds['in_validation_set'].sum():,}")

############### 
#Review xCP Distribution by Outcome

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# xCP distribution by outcome
ax1 = axes[0, 0]
completions = val_only[val_only['completion_target'] == 1]['xCP_at_release']
incompletions = val_only[val_only['completion_target'] == 0]['xCP_at_release']
ax1.hist(completions, bins=50, alpha=0.5, label='Completions', density=True, color='green')
ax1.hist(incompletions, bins=50, alpha=0.5, label='Incompletions', density=True, color='red')
ax1.axvline(completions.mean(), color='green', linestyle='--', linewidth=2, label=f'Comp Mean: {completions.mean():.2f}')
ax1.axvline(incompletions.mean(), color='red', linestyle='--', linewidth=2, label=f'Incomp Mean: {incompletions.mean():.2f}')
ax1.set_xlabel('xCP at Release')
ax1.set_ylabel('Density')
ax1.set_title('xCP Distribution by Actual Outcome')
ax1.legend()
ax1.grid(True, alpha=0.3)

# xCP vs Actual Completion Rate
ax2 = axes[0, 1]
bins = np.linspace(0, 1, 11)
val_only['xcp_bin'] = pd.cut(val_only['xCP_at_release'], bins=bins)
bin_stats = val_only.groupby('xcp_bin').agg({
    'completion_target': ['mean', 'count']
}).reset_index()
bin_centers = [interval.mid for interval in bin_stats['xcp_bin']]
actual_rates = bin_stats[('completion_target', 'mean')]
counts = bin_stats[('completion_target', 'count')]

ax2.scatter(bin_centers, actual_rates, s=counts*2, alpha=0.6)
ax2.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
ax2.set_xlabel('xCP at Release (Bin Center)')
ax2.set_ylabel('Actual Completion Rate')
ax2.set_title('Model Calibration (size = sample count)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Pass length vs xCP
if 'pass_length' in val_only.columns:
    ax3 = axes[1, 0]
    completions_df = val_only[val_only['completion_target'] == 1]
    incompletions_df = val_only[val_only['completion_target'] == 0]
    ax3.scatter(completions_df['pass_length'], completions_df['xCP_at_release'], 
                alpha=0.3, s=10, color='green', label='Completions')
    ax3.scatter(incompletions_df['pass_length'], incompletions_df['xCP_at_release'], 
                alpha=0.3, s=10, color='red', label='Incompletions')
    ax3.set_xlabel('Pass Length (yards)')
    ax3.set_ylabel('xCP at Release')
    ax3.set_title('xCP vs Pass Length')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
else:
    ax3 = axes[1, 0]
    ax3.axis('off')

# Most surprising predictions (high xCP incompletions and low xCP completions)
ax4 = axes[1, 1]
val_only['surprise'] = np.abs(val_only['xCP_at_release'] - val_only['completion_target'])
top_surprises = val_only.nlargest(20, 'surprise')

colors = ['red' if ct == 0 else 'green' for ct in top_surprises['completion_target']]
ax4.barh(range(len(top_surprises)), top_surprises['surprise'], color=colors, alpha=0.6)
ax4.set_xlabel('Absolute Error')
ax4.set_ylabel('Play Index (top 20 most surprising)')
ax4.set_title('Most Surprising Predictions\n(Red=Incompletion, Green=Completion)')
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/kaggle/working/validation_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print(" Saved visualization: /kaggle/working/validation_analysis.png")

#################### 
#Print Summary Stats 

print("\n" + "=" * 50)
print("Validation Analysis Key Results")
print("=" * 50)

print(f"\nKey Findings:")
print(f"  Validation plays:              {len(val_only):,}")
print(f"  Overall AUC:                   {overall_auc:.4f}")
print(f"  Overall Accuracy:              {overall_acc:.1%}")
print(f"  Actual completion rate:        {val_only['completion_target'].mean():.1%}")
print(f"  Mean predicted xCP:            {val_only['xCP_at_release'].mean():.1%}")

print(f"\nFiles Generated:")
if config.USE_MULTIFRAME_TRAINING:
    print(f"  1. validation_frame_predictions.csv     - Frame-by-frame predictions (all frames)")
print(f"  2. supplementary_with_predictions.csv   - Full supplementary data + all play-level predictions")
print(f"  3. validation_analysis.png              - Visualization")


# 11. INFERENCE EXPORTS AND LEADERBOARD CREATION
# Runs process on all frames (for visualization if needed) & calculates leaderboard on all weeks
# ============================================================================

#######
#Updated Feature Engineer to remove pre-release filter for visualization if needed

class InferenceFeatureEngineer(FeatureEngineer):
    def _extract_all_frame_features(self, frames_data: pd.DataFrame, target_id: int) -> List[Dict]:
        """
        Extract features for ALL frames in a play, including pre-release.
        """
        all_frame_features = []
        all_frame_ids = sorted(frames_data['frame_id'].unique())
        
        # Find release frame for metadata
        release_frames = frames_data[frames_data['is_release_frame']]
        if len(release_frames) == 0:
            return []
        release_frame_id = release_frames['frame_id'].iloc[0]
        
        # Process each frame that has sufficient history
        min_frame_id = all_frame_ids[0] + (self.config.NUM_HISTORICAL_FRAMES - 1)
        
        for current_frame_id in all_frame_ids:
            if current_frame_id < min_frame_id:
                self.frames_skipped_no_history += 1
                continue
                                    
            try:
                temporal_window = self._get_temporal_frames(frames_data, current_frame_id)
                
                if len(temporal_window['frame_id'].unique()) < self.config.NUM_HISTORICAL_FRAMES:
                    self.frames_skipped_no_history += 1
                    continue
                
                frame_features = self._extract_release_features(
                    temporal_window, 
                    target_id,
                    force_frame_id=current_frame_id
                )
                
                frame_features['frame_id'] = current_frame_id
                frame_features['is_release_frame'] = (current_frame_id == release_frame_id)
                frame_features['frames_from_release'] = current_frame_id - release_frame_id
                
                all_frame_features.append(frame_features)
                self.frames_processed += 1
                
            except Exception as e:
                continue
        
        return all_frame_features

########################
# DATA PREPARATION
def prepare_batch_for_inference_module(features_list: List[Dict], scaler: Dict, config):
    #Convert features to model inputs (Local version)
    
    batch_data = []
    
    for features in features_list:
        # Receiver
        rec_frames = np.array(features['receiver']['frames'])
        if hasattr(scaler['receiver'], 'transform'):
            rec_frames = scaler['receiver'].transform(rec_frames)
        
        expected_frames = config.NUM_HISTORICAL_FRAMES
        actual_frames = len(rec_frames)
        
        if actual_frames < expected_frames:
            padding = np.zeros((expected_frames - actual_frames, rec_frames.shape[1]))
            rec_frames = np.vstack([padding, rec_frames])
        elif actual_frames > expected_frames:
            rec_frames = rec_frames[-expected_frames:]
        
        rec_tensor = torch.FloatTensor(rec_frames.flatten())
        
        # Defenders
        max_defenders = 10
        others_list = []
        
        for other in features['others']:
            other_frames = np.array(other['frames'])
            if hasattr(scaler['others'], 'transform'):
                other_frames = scaler['others'].transform(other_frames)
            
            if len(other_frames) < expected_frames:
                padding = np.zeros((expected_frames - len(other_frames), other_frames.shape[1]))
                other_frames = np.vstack([padding, other_frames])
            elif len(other_frames) > expected_frames:
                other_frames = other_frames[-expected_frames:]
            
            others_list.append(other_frames.flatten())
        
        if others_list:
            others_array = np.array(others_list[:max_defenders])
            if len(others_array) < max_defenders:
                padding = np.zeros((max_defenders - len(others_array), others_array.shape[1]))
                others_array = np.vstack([others_array, padding])
        else:
            others_array = np.zeros((max_defenders, expected_frames * config.OTHER_FEATURE_DIM))
        
        others_tensor = torch.FloatTensor(others_array)
        
        batch_data.append({
            'receiver': rec_tensor,
            'others': others_tensor,
            'frame_id': features['frame_id']
        })
    
    return batch_data

###################
# INFERENCE LOGIC

def run_inference_on_plays_module(model, tracking_data: pd.DataFrame, scaler: Dict, 
                                  config, plays_to_process: List[Tuple[int, int]], device):
    """Run xCP inference on specified plays using existing environment objects."""
    engineer = InferenceFeatureEngineer(config)
    model.eval()
    
    all_predictions = []
    
    for game_id, play_id in tqdm(plays_to_process, desc="Running inference"):
        play_mask = (tracking_data['game_id'] == game_id) & (tracking_data['play_id'] == play_id)
        play_data = tracking_data[play_mask].copy()
        
        if play_data.empty:
            continue
        
        try:
            frame_features = engineer.process_play(play_data)
            
            if not frame_features:
                continue
            
            batch_data = prepare_batch_for_inference_module(frame_features, scaler, config)
            
            with torch.no_grad():
                for i, item in enumerate(batch_data):
                    rec_feats = item['receiver'].unsqueeze(0).to(device)
                    others_feats = item['others'].unsqueeze(0).to(device)
                    
                    logits, attn_weights = model(rec_feats, others_feats, return_attention=True)
                    xcp = torch.sigmoid(logits).item()
                    
                    attn_weights_np = attn_weights.cpu().numpy()[0]
                    defender_ids = [d['id'] for d in frame_features[i]['others']]
                    
                    pred_dict = {
                        'game_id': game_id,
                        'play_id': play_id,
                        'frame_id': item['frame_id'],
                        'frames_from_release': frame_features[i]['frames_from_release'],
                        'is_release_frame': frame_features[i]['is_release_frame'],
                        'xcp': xcp
                    }
                    
                    for j, (def_id, attn_weight) in enumerate(zip(defender_ids, attn_weights_np)):
                        pred_dict[f'defender_{j+1}_id'] = def_id
                        pred_dict[f'defender_{j+1}_attention'] = float(attn_weight)
                    
                    all_predictions.append(pred_dict)
        
        except Exception as e:
            continue
    
    return pd.DataFrame(all_predictions)

###################
# EXPORT FUNCTIONS

def create_comprehensive_export(tracking_data: pd.DataFrame, 
                                xcp_predictions: pd.DataFrame,
                                supplementary: pd.DataFrame) -> pd.DataFrame:
    """Create a comprehensive CSV export with all relevant data"""
    print("\nCreating comprehensive export...")
    
    merged = tracking_data.merge(
        xcp_predictions,
        on=['game_id', 'play_id', 'frame_id'],
        how='left',
        suffixes=('_tracking', '_pred')
    )
    
    # Consolidate metadata columns
    for col in ['frames_from_release', 'is_release_frame']:
        pred_col = f'{col}_pred'
        track_col = f'{col}_tracking'
        
        if pred_col in merged.columns:
            merged[col] = merged[pred_col]
            merged.drop([pred_col], axis=1, inplace=True, errors='ignore')
            if track_col in merged.columns:
                merged.drop([track_col], axis=1, inplace=True)
        elif track_col in merged.columns:
            merged[col] = merged[track_col]
            merged.drop([track_col], axis=1, inplace=True)
        elif col == 'frames_from_release' and 'is_release_frame' in merged.columns:
             # Fallback calculation
             merged[col] = merged.groupby(['game_id', 'play_id'])['frame_id'].transform(
                lambda x: x - x[merged.loc[x.index, 'is_release_frame']].iloc[0] if any(merged.loc[x.index, 'is_release_frame']) else 0
            )
        else:
            merged[col] = 0 if col == 'frames_from_release' else False
    
    play_info = supplementary[[
        'game_id', 'play_id', 'quarter', 'down', 'yards_to_go',
        'pass_result', 'pass_length', 'offense_formation', 
        'team_coverage_type', 'yards_gained', 'play_description'
    ]].drop_duplicates()
    
    merged = merged.merge(play_info, on=['game_id', 'play_id'], how='left')
    
    export_cols = [
        'game_id', 'play_id', 'frame_id', 'nfl_id',
        'xcp',
        'is_release_frame', 'frames_from_release',
        'player_name', 'player_position', 'player_side', 'player_role',
        'x', 'y', 's', 'a', 'dir',
        'quarter', 'down', 'yards_to_go', 'pass_result', 'pass_length',
        'offense_formation', 'team_coverage_type', 'yards_gained',
        'play_description'
    ]
    
    export_cols = [col for col in export_cols if col in merged.columns]
    export_df = merged[export_cols].copy()
    
    role_order = {'Passer': 0, 'Targeted Receiver': 1, 'Other Route Runner': 2, 'Defensive Coverage': 3}
    export_df['role_sort'] = export_df['player_role'].map(role_order).fillna(4)
    export_df = export_df.sort_values(['game_id', 'play_id', 'frame_id', 'role_sort'])
    export_df = export_df.drop('role_sort', axis=1)
    
    print(f"Created export with {len(export_df)} rows")
    return export_df

def create_play_summary_export(xcp_predictions: pd.DataFrame,
                               tracking_data: pd.DataFrame,
                               supplementary: pd.DataFrame,
                               frames_from_end: int = 2) -> pd.DataFrame:
    """Create a play-level summary export"""
    print("\nCreating play-level summary...")
    
    play_groups = xcp_predictions.groupby(['game_id', 'play_id'])
    
    summaries = []
    for (game_id, play_id), group in play_groups:
        # Get release frame xCP safely
        if 'frames_from_release' in group.columns and (group['frames_from_release'] == 0).any():
            release_xcp = group[group['frames_from_release'] == 0]['xcp'].iloc[0]
        else:
            # Fallback
            release_xcp = group['xcp'].iloc[0]
        
        # Calculate final frame index
        final_frame_idx = -(frames_from_end + 1)
        if abs(final_frame_idx) > len(group):
            final_frame_idx = 0
        
        final_xcp = group['xcp'].iloc[final_frame_idx]
        
        summary = {
            'game_id': game_id,
            'play_id': play_id,
            'num_frames': len(group),
            'xcp_at_release': release_xcp,
            'xcp_min': group['xcp'].min(),
            'xcp_max': group['xcp'].max(),
            'xcp_final': final_xcp,
            'xcp_change': final_xcp - release_xcp,
            'xcp_volatility': group['xcp'].std()
        }
        summaries.append(summary)
    
    summary_df = pd.DataFrame(summaries)
    
    # Add ball landing coordinates
    ball_landing = tracking_data[['game_id', 'play_id', 'ball_land_x', 'ball_land_y']].drop_duplicates(subset=['game_id', 'play_id'], keep='first')
    summary_df = summary_df.merge(ball_landing, on=['game_id', 'play_id'], how='left')
    
    # Add defender names
    defender_names = tracking_data[
        (tracking_data['player_side'] == 'Defense') & 
        (tracking_data['frame_id'] == 1)
    ][['game_id', 'play_id', 'player_name']].copy()
    
    defenders_per_play = defender_names.groupby(['game_id', 'play_id'])['player_name'].apply(
        lambda x: ', '.join(sorted(x))
    ).reset_index()
    defenders_per_play.columns = ['game_id', 'play_id', 'defenders']
    summary_df = summary_df.merge(defenders_per_play, on=['game_id', 'play_id'], how='left')
    
    # Merge supplementary
    supp_cols = [col for col in supplementary.columns if col not in summary_df.columns or col in ['game_id', 'play_id']]
    play_info = supplementary[supp_cols].drop_duplicates()
    summary_df = summary_df.merge(play_info, on=['game_id', 'play_id'], how='left')
    
    print(f"Created summary for {len(summary_df)} plays")
    return summary_df

def create_defender_attribution_export(xcp_predictions: pd.DataFrame,
                                       tracking_data: pd.DataFrame) -> pd.DataFrame:
    
    #Create a defender attribution export in long format
    print("\nCreating defender attribution export...")
    
    attribution_rows = []
    
    for _, row in xcp_predictions.iterrows():
        for i in range(1, 11):
            def_id_col = f'defender_{i}_id'
            def_attn_col = f'defender_{i}_attention'
            
            if def_id_col in row and def_attn_col in row:
                defender_id = row[def_id_col]
                attention = row[def_attn_col]
                
                if pd.notna(defender_id) and defender_id != 0 and attention > 0:
                    attribution_rows.append({
                        'game_id': row['game_id'],
                        'play_id': row['play_id'],
                        'frame_id': row['frame_id'],
                        'defender_id': int(defender_id),
                        'attention_weight': float(attention),
                        'xcp': row['xcp']
                    })
    
    attribution_df = pd.DataFrame(attribution_rows)
    
    if len(attribution_df) > 0:
        defender_info = tracking_data[
            tracking_data['player_role'] == 'Defensive Coverage'
        ][['game_id', 'play_id', 'frame_id', 'nfl_id', 'player_name', 'x', 'y']].drop_duplicates()
        
        attribution_df = attribution_df.merge(
            defender_info,
            left_on=['game_id', 'play_id', 'frame_id', 'defender_id'],
            right_on=['game_id', 'play_id', 'frame_id', 'nfl_id'],
            how='left'
        ).drop('nfl_id', axis=1)
        
        attribution_df = attribution_df.sort_values(
            ['game_id', 'play_id', 'frame_id', 'attention_weight'],
            ascending=[True, True, True, False]
        )
    
    print(f"Created {len(attribution_df)} defender attribution records")
    return attribution_df

#######################
# LEADERBOARD CALCULATION

def calculate_leaderboards_all_weeks_module(
    model, tracking_data, supplementary, scaler, config, device,
    frames_from_end: int = 2,
    frames_after_release: int = 5,
    use_frames_after_release: bool = False,
    use_final_frame_attention: bool = True):
    #Calculate defender reaction leaderboards across ALL weeks found in tracking_data.
    
    print("\n" + "="*50)
    print("DEFENDER REACTION LEADERBOARDS - ALL WEEKS")
    
    if use_frames_after_release:
        print(f"Comparison Point: {frames_after_release} frames AFTER RELEASE")
    else:
        print(f"Comparison Point: {frames_from_end} frames FROM END")
        
    print(f"Attention Source: {'COMPARISON FRAME' if use_final_frame_attention else 'RELEASE FRAME'}")
    
    # Verify weeks
    available_weeks = sorted(tracking_data['week'].unique())
    print(f"Weeks found in dataset: {available_weeks}")
    print("="*80)
    
    # Get unique plays
    all_plays = tracking_data[['game_id', 'play_id']].drop_duplicates()
    print(f"\nProcessing {len(all_plays)} plays across all weeks...")
    
    engineer = InferenceFeatureEngineer(config)
    play_results = []
    
    for _, row in tqdm(all_plays.iterrows(), total=len(all_plays), desc="Processing plays"):
        game_id = row['game_id']
        play_id = row['play_id']
        
        play_mask = (tracking_data['game_id'] == game_id) & (tracking_data['play_id'] == play_id)
        play_data = tracking_data[play_mask].copy()
        
        if play_data.empty: continue
        
        release_frames = play_data[play_data['is_release_frame'] == True]
        if release_frames.empty: continue

        release_frame_id = release_frames['frame_id'].iloc[0]
        max_frame_id = play_data['frame_id'].max()
        
        if use_frames_after_release:
            final_frame_id = release_frame_id + frames_after_release
            if final_frame_id > max_frame_id: final_frame_id = max_frame_id
        else:
            final_frame_id = max_frame_id - frames_from_end
            if final_frame_id < release_frame_id: final_frame_id = max_frame_id
            
        try:
            frame_features = engineer.process_play(play_data)
            if not frame_features: continue
            
            release_feat = None
            final_feat = None
            for feat in frame_features:
                if feat['frame_id'] == release_frame_id: release_feat = feat
                if feat['frame_id'] == final_frame_id: final_feat = feat
            
            if not release_feat or not final_feat: continue
            
            xcp_release = None
            xcp_final = None
            target_defenders = {}
            
            with torch.no_grad():
                # Release
                batch_data = prepare_batch_for_inference_module([release_feat], scaler, config)
                logits_rel, attn_weights_rel = model(
                    batch_data[0]['receiver'].unsqueeze(0).to(device),
                    batch_data[0]['others'].unsqueeze(0).to(device),
                    return_attention=True
                )
                xcp_release = torch.sigmoid(logits_rel).item()
                
                # Comparison
                batch_data = prepare_batch_for_inference_module([final_feat], scaler, config)
                logits_fin, attn_weights_fin = model(
                    batch_data[0]['receiver'].unsqueeze(0).to(device),
                    batch_data[0]['others'].unsqueeze(0).to(device),
                    return_attention=True
                )
                xcp_final = torch.sigmoid(logits_fin).item()
                
                if use_final_frame_attention:
                    target_attn = attn_weights_fin
                    target_def_ids = [d['id'] for d in final_feat['others']]
                else:
                    target_attn = attn_weights_rel
                    target_def_ids = [d['id'] for d in release_feat['others']]
                
                attn_weights_np = target_attn.cpu().numpy()[0]
                for def_id, attn_weight in zip(target_def_ids, attn_weights_np):
                    if attn_weight > 0:
                        target_defenders[int(def_id)] = float(attn_weight)
            
            play_results.append({
                'game_id': game_id,
                'play_id': play_id,
                'xcp_release': xcp_release,
                'xcp_final': xcp_final,
                'xcp_change': xcp_final - xcp_release,
                'target_defenders': target_defenders
            })
            
        except Exception:
            continue
    
    print(f"\nSuccessfully processed {len(play_results)} plays")
    
    # Build contribution data
    contributions = []
    
    for play in play_results:
        xcp_change = play['xcp_change']
        for defender_id, attention in play['target_defenders'].items():
            contrib_val = xcp_change * attention
            contributions.append({
                'game_id': play['game_id'],
                'play_id': play['play_id'],
                'defender_id': defender_id,
                'xcp_change': xcp_change,
                'attention_weight': attention,
                'contribution': contrib_val,
                # Impact Rate Flag: Decreased xCP by more than 10% (-0.10 or lower)
                'is_impact_10': 1 if contrib_val <= -0.10 else 0
            })
    
    contribution_df = pd.DataFrame(contributions)
    
    # Metadata merges
    defender_info = tracking_data[['nfl_id', 'player_name', 'player_position']].drop_duplicates()
    contribution_df = contribution_df.merge(
        defender_info, left_on='defender_id', right_on='nfl_id', how='left'
    ).drop('nfl_id', axis=1)
    
    defender_teams = supplementary[['game_id', 'play_id', 'defensive_team']].drop_duplicates()
    contribution_df = contribution_df.merge(defender_teams, on=['game_id', 'play_id'], how='left')
    
    defender_primary_team = contribution_df.groupby('defender_id')['defensive_team'].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    ).reset_index()
    defender_primary_team.columns = ['defender_id', 'team']
    
    coverage_info = supplementary[['game_id', 'play_id', 'team_coverage_man_zone']].drop_duplicates()
    contribution_df = contribution_df.merge(coverage_info, on=['game_id', 'play_id'], how='left')
    
    # Helper for leaderboard generation
    def make_leaderboard(df_subset, filename_suffix):
        # Aggregate logic
        lb = df_subset.groupby(['defender_id', 'player_name', 'player_position']).agg({
            'contribution': ['sum', 'median'],  # Sum for Total rCPA, Median for Median rCPA
            'game_id': 'count',                 # Total plays
            'is_impact_10': 'mean'              # Mean of 0/1 gives the Rate
        }).reset_index()
        
        # Flatten MultiIndex columns and apply new names
        lb.columns = [
            'defender_id', 'player_name', 'player_position', 
            'Total rCPA', 'Median rCPA', 
            'num_plays', 
            'Impact Rate'
        ]
        
        # Add Team
        lb = lb.merge(defender_primary_team, on='defender_id', how='left')
        
        # Calculate Mean rCPA
        lb['Mean rCPA'] = lb['Total rCPA'] / lb['num_plays']
        
        # Reorder and Sort
        cols = [
            'defender_id', 'player_name', 'team', 'player_position', 
            'Total rCPA', 'num_plays', 
            'Mean rCPA', 'Median rCPA',
            'Impact Rate'
        ]
        lb = lb[cols].sort_values('Total rCPA', ascending=True)
        
        lb.to_csv(f'/kaggle/working/defender_leaderboard_{filename_suffix}.csv', index=False)
        return lb
    
    # Generate Leaderboards
    overall = make_leaderboard(contribution_df, 'overall')
    
    man_df = contribution_df[contribution_df['team_coverage_man_zone'].str.contains('MAN', case=False, na=False)]
    man = make_leaderboard(man_df, 'man')
    
    zone_df = contribution_df[contribution_df['team_coverage_man_zone'].str.contains('ZONE', case=False, na=False)]
    zone = make_leaderboard(zone_df, 'zone')
    
    print(f"Leaderboards created & saved:")
    print(f"  Overall: {len(overall)} defenders")
    print(f"  Man: {len(man)} defenders")
    print(f"  Zone: {len(zone)} defenders")
    
    return {'overall': overall, 'man': man, 'zone': zone}


######################
# MAIN MODULE WRAPPER

def run_analysis_module(model, tracking_data, supplementary_data, scaler, config, 
                        weeks_to_analyze: List[int], device, frames_from_end: int = 2):
    print("\n" + "="*50)
    print("STARTING ANALYSIS")
    print("="*50)
    
    # Filter to selected weeks for detailed analysis
    week_mask = supplementary_data['week'].isin(weeks_to_analyze)
    selected_plays = supplementary_data[week_mask][['game_id', 'play_id']].drop_duplicates()
    print(f"Analyzing {len(selected_plays)} plays in weeks {weeks_to_analyze}")
    
    tracking_subset = tracking_data.merge(selected_plays, on=['game_id', 'play_id'], how='inner')
    plays_to_process = list(selected_plays.itertuples(index=False, name=None))
    
    # 1. Inference
    xcp_predictions = run_inference_on_plays_module(
        model, tracking_subset, scaler, config, plays_to_process, device
    )
    print(f"\nGenerated {len(xcp_predictions)} frame-level predictions")
    
    # 2. Exports
    xcp_predictions.to_csv('/kaggle/working/xcp_predictions.csv', index=False)
    
    comprehensive = create_comprehensive_export(tracking_subset, xcp_predictions, supplementary_data)
    comprehensive.to_csv('/kaggle/working/xcp_comprehensive_export.csv', index=False)
    
    summary = create_play_summary_export(xcp_predictions, tracking_subset, supplementary_data, frames_from_end)
    summary.to_csv('/kaggle/working/xcp_play_summary.csv', index=False)
    
    attribution = create_defender_attribution_export(xcp_predictions, tracking_subset)
    attribution.to_csv('/kaggle/working/defender_attribution.csv', index=False)
    
    print("\n" + "="*50)
    print("MODULE COMPLETE")
    print("="*50)

# ============================================================================
# EXECUTION
# ============================================================================

# Define parameters
WEEKS_TO_ANALYZE_MODULE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
#WEEKS_TO_ANALYZE_MODULE = [1, 2, 3, 4] #Analyze just a few

#Set to True if you want to process leaderboards (increases compute time), otherwise just exports all frames for visualization
CALCULATE_LEADERBOARDS = True

#Mark default cutoff from end to calculate change in xCP. Default is 2 to remove the last few frames when players sometimes let up
FRAMES_FROM_END = 2

# If we're measuring compared to a specific frame after release, set to True and set the # of frames (default rCPA is .5 seconds) 
USE_FRAMES_AFTER_RELEASE = True 
FRAMES_AFTER_RELEASE = 5

# Determine whether we get defender attention from final frame (True) or release frame (False)
USE_FINAL_FRAME_ATTENTION = True 


##### Run Full Inference (uses subset of weeks for detailed frames)
run_analysis_module(
    model, full_tracking_df, supplementary_df, scaler, config,
    WEEKS_TO_ANALYZE_MODULE, device, frames_from_end=FRAMES_FROM_END
)

##### Calculate Leaderboards (uses all weeks)
if CALCULATE_LEADERBOARDS:
    leaderboards = calculate_leaderboards_all_weeks_module(
        model, full_tracking_df, supplementary_df, scaler, config, device,
        # Config options
        frames_from_end=FRAMES_FROM_END,
        frames_after_release=FRAMES_AFTER_RELEASE,
        use_frames_after_release=USE_FRAMES_AFTER_RELEASE,
        use_final_frame_attention=USE_FINAL_FRAME_ATTENTION
    )

