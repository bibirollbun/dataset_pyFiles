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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =============================================================================
# DATA MANAGER FOR PERSISTENCE
# =============================================================================

class NFLDataManager:
    def __init__(self, base_path="outputs/working/"):
        self.base_path = Path(base_path)
        self.features_path = self.base_path / "features"
        self.models_path = self.base_path / "models"
        self.metadata_path = self.base_path / "metadata"
        self.eda_path = self.base_path / "eda"
        
        # Create directories
        self.features_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        self.eda_path.mkdir(parents=True, exist_ok=True)
    
    def save_features(self, features_df, feature_set_name, week=None):
        """Save engineered features to disk"""
        filename = f"{feature_set_name}"
        if week:
            filename += f"_week_{week}"
        filename += ".parquet"
        
        file_path = self.features_path / filename
        features_df.to_parquet(file_path, index=False)
        print(f"âœ… Features saved to: {file_path}")
        
        # Save metadata
        metadata = {
            'feature_set_name': feature_set_name,
            'week': week,
            'shape': features_df.shape,
            'columns': features_df.columns.tolist(),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        metadata_file = self.metadata_path / f"{feature_set_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_features(self, feature_set_name, week=None):
        """Load saved features from disk"""
        filename = f"{feature_set_name}"
        if week:
            filename += f"_week_{week}"
        filename += ".parquet"
        
        file_path = self.features_path / filename
        if file_path.exists():
            features_df = pd.read_parquet(file_path)
            print(f"âœ… Features loaded from: {file_path}")
            return features_df
        else:
            print(f"â�Œ Feature file not found: {file_path}")
            return None

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class NFLFeatureEngineer:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        
    def engineer_features(self, input_dfs, output_dfs, feature_set_name):
        """Main feature engineering pipeline with EDA column preservation"""
        print("ğŸš€ Starting Feature Engineering Pipeline")
        print("="*60)
        
        all_features = []
        
        for week_name, input_df in input_dfs.items():
            print(f"\nğŸ“� Processing {week_name}...")
            
            # Get corresponding output data
            output_df = output_dfs.get(week_name)
            if output_df is None:
                print(f"  âš ï¸� No output data for {week_name}, skipping...")
                continue
            
            # Merge input and output data
            try:
                merged_data = input_df.merge(
                    output_df[['game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y']],
                    on=['game_id', 'play_id', 'nfl_id', 'frame_id'],
                    suffixes=('', '_target'),
                    how='inner'
                )
                print(f"  âœ… Successfully merged data: {merged_data.shape}")
            except Exception as e:
                print(f"  â�Œ Merge failed: {e}")
                continue
            
            # Only use players marked for prediction if the column exists
            if 'player_to_predict' in merged_data.columns:
                before_filter = len(merged_data)
                merged_data = merged_data[merged_data['player_to_predict'] == True]
                after_filter = len(merged_data)
                print(f"  ğŸ”� Filtered to players to predict: {after_filter}/{before_filter}")
            
            # Engineer features
            features = self._create_all_features(merged_data)
            
            # STEP: Add ALL original columns that might be needed for EDA
            # This ensures we have the raw data for visualization and analysis
            eda_columns = [
                'player_side', 'player_position', 'player_role', 'player_name',
                'play_direction', 'absolute_yardline_number', 
                'player_height', 'player_weight', 'player_birth_date'
            ]
            
            for col in eda_columns:
                if col in merged_data.columns and col not in features.columns:
                    features[col] = merged_data[col]
                    print(f"  ğŸ“Š Preserved EDA column: {col}")
            
            # Add identifiers and targets
            identifier_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id']
            for col in identifier_cols:
                if col in merged_data.columns:
                    features[col] = merged_data[col]
            
            if 'x_target' in merged_data.columns and 'y_target' in merged_data.columns:
                features['x_target'] = merged_data['x_target']
                features['y_target'] = merged_data['y_target']
            
            all_features.append(features)
            
            # Save weekly features
            week_num = week_name.split('_')[-1]
            self.data_manager.save_features(
                features, 
                f"{feature_set_name}_week_{week_num}",
                week=week_num
            )
        
        # Save combined features
        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            self.data_manager.save_features(combined_features, f"{feature_set_name}_combined")
            print(f"\nâœ… Combined features shape: {combined_features.shape}")
            
            # Print preserved columns for EDA
            eda_preserved = [col for col in combined_features.columns if col in [
                'player_side', 'player_position', 'player_role', 'player_name',
                'play_direction', 'absolute_yardline_number'
            ]]
            print(f"ğŸ“Š Preserved EDA columns: {len(eda_preserved)}")
            
            feature_count = len([col for col in combined_features.columns if col not in [
                'game_id', 'play_id', 'nfl_id', 'frame_id', 'x_target', 'y_target'
            ] + eda_preserved])
            print(f"âœ… Total engineered features: {feature_count}")
            return combined_features
        
        print("â�Œ No features were created")
        return None
    
    def _create_all_features(self, df):
        """Create all feature types with proper column handling"""
        features = pd.DataFrame(index=df.index)
        
        print("  Creating feature categories:")
        
        # STEP 1: First add identifier columns to features so we can merge later
        identifier_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id']
        for col in identifier_cols:
            if col in df.columns:
                features[col] = df[col]
        
        # STEP 2: Create features in order of dependency
        # 1. Basic features (no dependencies)
        try:
            features = self._create_player_features(df, features)
            print("    âœ… Player features")
        except Exception as e:
            print(f"    â�Œ Player features failed: {e}")
        
        # 2. Movement features  
        try:
            features = self._create_movement_features(df, features)
            print("    âœ… Movement features")
        except Exception as e:
            print(f"    â�Œ Movement features failed: {e}")
        
        # 3. Advanced Physics Features (depend on movement features)
        try:
            features = self._create_advanced_physics_features(df, features)
            print("    âœ… Advanced physics features")
        except Exception as e:
            print(f"    â�Œ Advanced physics features failed: {e}")
        
        # 4. Play Context Features
        try:
            features = self._create_play_context_features(df, features)
            print("    âœ… Play context features")
        except Exception as e:
            print(f"    â�Œ Play context features failed: {e}")
        
        # 5. Football Intelligence Features
        try:
            features = self._create_football_features(df, features)
            print("    âœ… Football intelligence features")
        except Exception as e:
            print(f"    â�Œ Football intelligence features failed: {e}")
        
        # STEP 3: Create complex features that require grouping and merging
        # 6. Player Awareness Features (requires grouping by play)
        try:
            features = self._create_player_awareness_features(df, features)
            print("    âœ… Player awareness features")
        except Exception as e:
            print(f"    â�Œ Player awareness features failed: {e}")
        
        # 7. Player Maximum Features (requires grouping by player and play)
        try:
            features = self._create_player_max_features(df, features)
            print("    âœ… Player maximum features")
        except Exception as e:
            print(f"    â�Œ Player maximum features failed: {e}")
        
        return features
    
    def _create_player_features(self, df, features):
        """Player demographic and physical attributes"""
        # Physical attributes
        if 'player_height' in df.columns:
            features['height_inches'] = df['player_height'].apply(self._convert_height_to_inches)
        
        if 'player_weight' in df.columns:
            features['weight'] = df['player_weight']
            
        # Position features (one-hot encoding)
        if 'player_position' in df.columns:
            position_dummies = pd.get_dummies(df['player_position'], prefix='pos')
            features = pd.concat([features, position_dummies], axis=1)
            
        # Role features
        if 'player_role' in df.columns:
            role_dummies = pd.get_dummies(df['player_role'], prefix='role')
            features = pd.concat([features, role_dummies], axis=1)
            
        # Side features
        if 'player_side' in df.columns:
            features['is_offense'] = (df['player_side'] == 'Offense').astype(int)
            
        return features
    
    def _create_movement_features(self, df, features):
        """Movement and physics-based features"""
        # Current movement state
        movement_cols = ['x', 'y', 's', 'a', 'o', 'dir']
        for col in movement_cols:
            if col in df.columns:
                features[col] = df[col]
                
        # Convert orientation and direction to radians for trigonometric features
        if 'o' in df.columns:
            features['o_rad'] = np.radians(df['o'])
            features['o_sin'] = np.sin(features['o_rad'])
            features['o_cos'] = np.cos(features['o_rad'])
            
        if 'dir' in df.columns:
            features['dir_rad'] = np.radians(df['dir'])
            features['dir_sin'] = np.sin(features['dir_rad'])
            features['dir_cos'] = np.cos(features['dir_rad'])
            
        # Velocity components
        if all(col in df.columns for col in ['s', 'dir']):
            features['velocity_x'] = df['s'] * np.cos(np.radians(df['dir']))
            features['velocity_y'] = df['s'] * np.sin(np.radians(df['dir']))
            
        # Movement quality metrics
        if 's' in df.columns:
            features['speed_squared'] = df['s'] ** 2
            features['is_moving'] = (df['s'] > 0.5).astype(int)
            
        return features
    
    def _create_play_context_features(self, df, features):
        """Play-level context features"""
        # Field position
        if 'absolute_yardline_number' in df.columns:
            features['yardline'] = df['absolute_yardline_number']
            features['field_third'] = pd.cut(df['absolute_yardline_number'], 
                                           bins=[0, 40, 80, 120], 
                                           labels=[0, 1, 2]).astype(float)
            
        # Play direction
        if 'play_direction' in df.columns:
            features['moving_left'] = (df['play_direction'] == 'left').astype(int)
            
        # Ball landing position (CRITICAL FEATURE)
        if all(col in df.columns for col in ['ball_land_x', 'ball_land_y']):
            features['ball_land_x'] = df['ball_land_x']
            features['ball_land_y'] = df['ball_land_y']
            
            # Distance to ball landing spot
            if all(col in df.columns for col in ['x', 'y']):
                features['dist_to_ball_land'] = np.sqrt(
                    (df['x'] - df['ball_land_x'])**2 + 
                    (df['y'] - df['ball_land_y'])**2
                )
                
        # Frame prediction context
        if 'num_frames_output' in df.columns:
            features['frames_to_predict'] = df['num_frames_output']
            
        return features
    
    def _create_football_features(self, df, features):
        """Football-specific strategic features"""
        # Route concepts based on player role
        if 'player_role' in df.columns:
            features['is_targeted_receiver'] = (df['player_role'] == 'Targeted Receiver').astype(int)
            features['is_passer'] = (df['player_role'] == 'Passer').astype(int)
            
        if 'player_position' in df.columns:
            features['is_wide_receiver'] = (df['player_position'] == 'WR').astype(int)
            
        # Field awareness
        if all(col in df.columns for col in ['x', 'y']):
            features['dist_to_sideline'] = np.minimum(df['y'], 53.3 - df['y'])
            features['dist_to_endzone'] = 120 - df['x']  # assuming offense moving right
            
        return features
    
    def _convert_height_to_inches(self, height_str):
        """Convert height string to inches"""
        if pd.isna(height_str):
            return np.nan
        try:
            feet, inches = height_str.split('-')
            return int(feet) * 12 + int(inches)
        except:
            return np.nan

    def _create_advanced_physics_features(self, df, features):
        """Advanced physics and player awareness features"""
        # Momentum features (weight * velocity)
        if all(col in df.columns for col in ['player_weight', 'velocity_x', 'velocity_y']):
            features['momentum_x'] = df['player_weight'] * df['velocity_x']
            features['momentum_y'] = df['player_weight'] * df['velocity_y']
            features['momentum_magnitude'] = np.sqrt(features['momentum_x']**2 + features['momentum_y']**2)
    
        # Kinetic Energy features (0.5 * mass * velocity^2)
        if all(col in df.columns for col in ['player_weight', 's']):
            # Convert weight to mass (simplified: weight in lbs / 32.2 ft/sÂ² for mass in slugs)
            # For football purposes, we can use weight directly as proportional to mass
            features['kinetic_energy'] = 0.5 * df['player_weight'] * (df['s'] ** 2)
    
        return features

    def _create_player_awareness_features(self, df, features):
        """Player awareness and vision features - FIXED VERSION"""
        # Check if we have all required columns in the source data
        required_cols = ['game_id', 'play_id', 'player_side', 'player_role', 'x', 'y', 'o', 'nfl_id']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"    âš ï¸� Missing columns for player awareness: {missing_cols}")
            return features
        
        try:
            print("    Computing player awareness features...")
            awareness_features = []
            processed_plays = 0
            
            for (game_id, play_id), play_df in df.groupby(['game_id', 'play_id']):
                processed_plays += 1
                
                # Find QB position
                qb_data = play_df[(play_df['player_role'] == 'Passer') | 
                                (play_df['player_position'] == 'QB')]
                if len(qb_data) > 0:
                    qb_x = qb_data['x'].iloc[0]
                    qb_y = qb_data['y'].iloc[0]
                else:
                    qb_x, qb_y = np.nan, np.nan
                
                # Find targeted receivers
                targeted_receivers = play_df[play_df['player_role'] == 'Targeted Receiver']
                
                for _, player in play_df.iterrows():
                    awareness_data = {
                        'game_id': game_id,
                        'play_id': play_id,
                        'nfl_id': player['nfl_id'],
                        'frame_id': player['frame_id']
                    }
                    
                    # Eyes on QB (for all players)
                    if not pd.isna(qb_x) and not pd.isna(qb_y) and not pd.isna(player['o']):
                        dx = qb_x - player['x']
                        dy = qb_y - player['y']
                        angle_to_qb = np.degrees(np.arctan2(dy, dx)) % 360
                        orientation_diff = abs((angle_to_qb - player['o'] + 180) % 360 - 180)
                        awareness_data['eyes_on_qb'] = 1 if orientation_diff <= 45 else 0
                    else:
                        awareness_data['eyes_on_qb'] = 0
                    
                    # Eyes on Receiver (for defensive players)
                    awareness_data['eyes_on_receiver'] = 0
                    if (player['player_side'] == 'Defense' and 
                        len(targeted_receivers) > 0 and 
                        not pd.isna(player['o'])):
                        
                        min_receiver_dist = float('inf')
                        closest_receiver_angle = None
                        
                        for _, receiver in targeted_receivers.iterrows():
                            if pd.isna(receiver['x']) or pd.isna(receiver['y']):
                                continue
                            dist = np.sqrt((receiver['x'] - player['x'])**2 + 
                                        (receiver['y'] - player['y'])**2)
                            if dist < min_receiver_dist:
                                min_receiver_dist = dist
                                dx = receiver['x'] - player['x']
                                dy = receiver['y'] - player['y']
                                closest_receiver_angle = np.degrees(np.arctan2(dy, dx)) % 360
                        
                        if closest_receiver_angle is not None:
                            orientation_diff = abs((closest_receiver_angle - player['o'] + 180) % 360 - 180)
                            awareness_data['eyes_on_receiver'] = 1 if orientation_diff <= 45 else 0
                    
                    awareness_features.append(awareness_data)
            
            if awareness_features:
                awareness_df = pd.DataFrame(awareness_features)
                # Now merge using all identifier columns including frame_id
                merge_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id']
                
                # Check if we have the merge columns in both DataFrames
                if (all(col in features.columns for col in merge_cols) and 
                    all(col in awareness_df.columns for col in merge_cols)):
                    
                    # Merge on all identifier columns to ensure 1:1 mapping
                    features = features.merge(awareness_df, on=merge_cols, how='left')
                    print(f"    âœ… Added player awareness features for {processed_plays} plays")
                else:
                    missing_in_features = [col for col in merge_cols if col not in features.columns]
                    missing_in_awareness = [col for col in merge_cols if col not in awareness_df.columns]
                    print(f"    â�Œ Cannot merge awareness - missing in features: {missing_in_features}, missing in awareness: {missing_in_awareness}")
            
        except Exception as e:
            print(f"    â�Œ Error in player awareness features: {e}")
            import traceback
            traceback.print_exc()
        
        return features

    def _create_player_max_features(self, df, features):
        """Player maximum performance features - FIXED VERSION"""
        try:
            # Check if we have the required columns
            required_cols = ['game_id', 'play_id', 'nfl_id', 's', 'a']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"    âš ï¸� Missing columns for player max features: {missing_cols}")
                return features
                
            print("    Computing player maximum features...")
            max_features = []
            processed_players = 0
            
            for (game_id, play_id, nfl_id), player_play_df in df.groupby(['game_id', 'play_id', 'nfl_id']):
                processed_players += 1
                
                max_data = {
                    'game_id': game_id,
                    'play_id': play_id,
                    'nfl_id': nfl_id,
                    'max_speed_this_play': player_play_df['s'].max(),
                    'max_acceleration_this_play': player_play_df['a'].max()
                }
                
                # Add max kinetic energy and momentum if available
                if 'kinetic_energy' in player_play_df.columns:
                    max_data['max_kinetic_energy_this_play'] = player_play_df['kinetic_energy'].max()
                if 'momentum_magnitude' in player_play_df.columns:
                    max_data['max_momentum_this_play'] = player_play_df['momentum_magnitude'].max()
                    
                max_features.append(max_data)
            
            if max_features:
                max_df = pd.DataFrame(max_features)
                # Merge back to features
                merge_cols = ['game_id', 'play_id', 'nfl_id']
                
                if (all(col in features.columns for col in merge_cols) and 
                    all(col in max_df.columns for col in merge_cols)):
                    
                    features = features.merge(max_df, on=merge_cols, how='left')
                    print(f"    âœ… Added player maximum features for {processed_players} players")
                else:
                    missing_in_features = [col for col in merge_cols if col not in features.columns]
                    missing_in_max = [col for col in merge_cols if col not in max_df.columns]
                    print(f"    â�Œ Cannot merge max features - missing in features: {missing_in_features}, missing in max: {missing_in_max}")
            
        except Exception as e:
            print(f"    â�Œ Error in player max features: {e}")
        
        return features

# =============================================================================
# COMPREHENSIVE EDA FOR ENGINEERED FEATURES
# =============================================================================

class NFLFeatureEDA:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        
    def create_comprehensive_eda(self, features_df, feature_set_name):
        """Create comprehensive EDA for engineered features"""
        print("ğŸ“Š Creating Comprehensive EDA...")
        print("="*60)
        
        # 1. Feature Overview
        self._feature_overview(features_df, feature_set_name)
        
        # 2. Target Analysis
        self._target_analysis(features_df, feature_set_name)
        
        # 3. Feature Correlations
        self._feature_correlations(features_df, feature_set_name)
        
        # 4. Football-Specific Analysis
        self._football_analysis(features_df, feature_set_name)

        # 5. Advanced Features Analysis (NEW)
        self._create_advanced_features_analysis(features_df, feature_set_name)
        
        # 6. Feature Importance Preview
        self._feature_importance_preview(features_df, feature_set_name)

       
    
        print(f"âœ… EDA complete! Check {self.data_manager.eda_path} for visualizations")
    
    def _feature_overview(self, features_df, feature_set_name):
        """Overview of engineered features"""
        print("   Creating feature overview...")
        
        feature_columns = [col for col in features_df.columns 
                          if col not in ['game_id', 'play_id', 'nfl_id', 'frame_id', 'x_target', 'y_target']]
        
        # Feature categories
        categories = {
            'Player Features': [col for col in feature_columns if col.startswith('pos_') or col.startswith('role_') or col in ['height_inches', 'weight', 'is_offense']],
            'Movement Features': [col for col in feature_columns if col in ['x', 'y', 's', 'a', 'o', 'dir', 'velocity_x', 'velocity_y', 'speed_squared', 'is_moving'] or 'sin' in col or 'cos' in col or 'rad' in col],
            'Play Context': [col for col in feature_columns if col in ['yardline', 'field_third', 'moving_left', 'ball_land_x', 'ball_land_y', 'dist_to_ball_land', 'frames_to_predict']],
            'Football Intelligence': [col for col in feature_columns if col in ['is_targeted_receiver', 'is_passer', 'is_wide_receiver', 'dist_to_sideline', 'dist_to_endzone']]
        }
        
        # Plot feature categories
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Feature count by category
        category_counts = {k: len(v) for k, v in categories.items()}
        ax1.bar(category_counts.keys(), category_counts.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_title('Feature Count by Category', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Features')
        ax1.tick_params(axis='x', rotation=45)
        
        # Data types
        dtypes_count = features_df[feature_columns].dtypes.value_counts()
        ax2.pie(dtypes_count.values, labels=dtypes_count.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
        ax2.set_title('Feature Data Types', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.data_manager.eda_path / f"{feature_set_name}_feature_overview.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Missing values analysis
        missing_values = features_df[feature_columns].isnull().sum()
        if missing_values.sum() > 0:
            plt.figure(figsize=(10, 6))
            missing_values[missing_values > 0].sort_values().plot(kind='barh', color='coral')
            plt.title('Missing Values by Feature', fontsize=14, fontweight='bold')
            plt.xlabel('Number of Missing Values')
            plt.tight_layout()
            plt.savefig(self.data_manager.eda_path / f"{feature_set_name}_missing_values.png", dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("   âœ… No missing values in features")
    
    def _target_analysis(self, features_df, feature_set_name):
        """Analysis of target variables"""
        print("   Creating target analysis...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Target Variable Analysis', fontsize=16, fontweight='bold')
        
        # Target distributions
        axes[0,0].hist(features_df['x_target'].dropna(), bins=50, alpha=0.7, color='blue')
        axes[0,0].set_title('Target X Distribution')
        axes[0,0].set_xlabel('X Coordinate')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].hist(features_df['y_target'].dropna(), bins=50, alpha=0.7, color='red')
        axes[0,1].set_title('Target Y Distribution')
        axes[0,1].set_xlabel('Y Coordinate')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].grid(True, alpha=0.3)
        
        # 2D scatter of targets
        speed_val = features_df.get('s', 0)
        scatter = axes[0,2].scatter(features_df['x_target'], features_df['y_target'], 
                                   c=speed_val, alpha=0.5, s=1, cmap='viridis')
        axes[0,2].set_title('Target Locations Colored by Speed')
        axes[0,2].set_xlabel('X Target')
        axes[0,2].set_ylabel('Y Target')
        axes[0,2].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0,2], label='Speed')
        
        # Field visualization with targets
        self._draw_football_field(axes[1,0])
        axes[1,0].scatter(features_df['x_target'], features_df['y_target'], alpha=0.1, s=1, color='purple')
        axes[1,0].set_title('Target Locations on Field')
        
        # Speed vs target distance
        if 's' in features_df.columns and 'x' in features_df.columns:
            initial_x = features_df['x']
            initial_y = features_df['y']
            distance_moved = np.sqrt(
                (features_df['x_target'] - initial_x)**2 + 
                (features_df['y_target'] - initial_y)**2
            )
            axes[1,1].scatter(features_df['s'], distance_moved, alpha=0.5, s=1, color='green')
            axes[1,1].set_xlabel('Initial Speed (yards/s)')
            axes[1,1].set_ylabel('Distance Moved (yards)')
            axes[1,1].set_title('Speed vs Distance Moved')
            axes[1,1].grid(True, alpha=0.3)
        
        # Target statistics
        axes[1,2].axis('off')
        stats_text = f"""
        Target Statistics:
        X Target:
          Mean: {features_df['x_target'].mean():.2f}
          Std:  {features_df['x_target'].std():.2f}
          Min:  {features_df['x_target'].min():.2f}
          Max:  {features_df['x_target'].max():.2f}
        
        Y Target:
          Mean: {features_df['y_target'].mean():.2f}
          Std:  {features_df['y_target'].std():.2f}
          Min:  {features_df['y_target'].min():.2f}
          Max:  {features_df['y_target'].max():.2f}
        """
        axes[1,2].text(0.1, 0.9, stats_text, transform=axes[1,2].transAxes, fontfamily='monospace',
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.data_manager.eda_path / f"{feature_set_name}_target_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _feature_correlations(self, features_df, feature_set_name):
        """Feature correlation analysis"""
        print("   Creating correlation analysis...")
        
        feature_columns = [col for col in features_df.columns 
                          if col not in ['game_id', 'play_id', 'nfl_id', 'frame_id', 'x_target', 'y_target']]
        
        # Select top 20 features for correlation matrix (for readability)
        numeric_features = features_df[feature_columns].select_dtypes(include=[np.number]).columns
        if len(numeric_features) > 20:
            # Use features with highest variance
            variances = features_df[numeric_features].var().sort_values(ascending=False)
            top_features = variances.head(20).index
        else:
            top_features = numeric_features
        
        # Correlation with targets
        target_correlations = pd.DataFrame({
            'x_target_corr': features_df[top_features].corrwith(features_df['x_target']).abs(),
            'y_target_corr': features_df[top_features].corrwith(features_df['y_target']).abs()
        }).sort_values('x_target_corr', ascending=False)
        
        # Plot correlation with targets
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        fig.suptitle('Feature Correlation Analysis', fontsize=16, fontweight='bold')
        
        target_correlations.head(15).plot(kind='barh', ax=ax1, color=['skyblue', 'lightcoral'])
        ax1.set_title('Top 15 Features Correlated with Targets (Absolute)')
        ax1.set_xlabel('Absolute Correlation')
        ax1.legend(['X Target', 'Y Target'])
        
        # Correlation heatmap
        corr_matrix = features_df[top_features].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, center=0, cmap='coolwarm', 
                   square=True, ax=ax2, cbar_kws={"shrink": .8})
        ax2.set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(self.data_manager.eda_path / f"{feature_set_name}_correlations.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print top correlations
        print("\nğŸ”� Top 10 features correlated with X target:")
        print(target_correlations.head(10)['x_target_corr'])
        
        print("\nğŸ”� Top 10 features correlated with Y target:")
        print(target_correlations.head(10)['y_target_corr'])
    
    def _football_analysis(self, features_df, feature_set_name):
        """Football-specific analysis"""
        print("   Creating football analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Football-Specific Feature Analysis', fontsize=16, fontweight='bold')
        
        # 1. Player Role Analysis
        if 'role_Targeted Receiver' in features_df.columns:
            role_cols = [col for col in features_df.columns if col.startswith('role_')]
            
            self._draw_football_field(axes[0,0])
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            for i, role_col in enumerate(role_cols[:5]):  # Top 5 roles
                role_data = features_df[features_df[role_col] == 1]
                if len(role_data) > 0:
                    axes[0,0].scatter(role_data['x_target'], role_data['y_target'], 
                                     alpha=0.3, s=10, color=colors[i % len(colors)], 
                                     label=role_col.replace('role_', ''))
            axes[0,0].legend()
            axes[0,0].set_title('Target Locations by Player Role')
        
        # 2. Ball Landing Impact
        if 'dist_to_ball_land' in features_df.columns:
            # Bin distance to ball landing and plot average target locations
            features_df['dist_bin'] = pd.cut(features_df['dist_to_ball_land'], bins=5)
            
            self._draw_football_field(axes[0,1])
            colors = plt.cm.viridis(np.linspace(0, 1, 5))
            for i, (bin_val, color) in enumerate(zip(features_df['dist_bin'].cat.categories, colors)):
                bin_data = features_df[features_df['dist_bin'] == bin_val]
                if len(bin_data) > 0:
                    axes[0,1].scatter(bin_data['x_target'].mean(), bin_data['y_target'].mean(), 
                                     s=100, color=color, label=f'Bin {i+1}')
            axes[0,1].legend()
            axes[0,1].set_title('Avg Target Locations by Distance to Ball')
        
        # 3. Speed analysis by position
        if 's' in features_df.columns:
            speed_by_pos = []
            pos_labels = []
            pos_cols = [col for col in features_df.columns if col.startswith('pos_')]
            
            for pos_col in pos_cols[:6]:  # Top 6 positions
                pos_data = features_df[features_df[pos_col] == 1]['s'].dropna()
                if len(pos_data) > 0:
                    speed_by_pos.append(pos_data)
                    pos_labels.append(pos_col.replace('pos_', ''))
            
            if speed_by_pos:
                box_plot = axes[1,0].boxplot(speed_by_pos, labels=pos_labels, patch_artist=True)
                # Add colors to boxes
                colors = plt.cm.Set3(np.linspace(0, 1, len(speed_by_pos)))
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)
                axes[1,0].set_title('Speed Distribution by Position')
                axes[1,0].set_ylabel('Speed (yards/s)')
                axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Field position impact
        if 'field_third' in features_df.columns:
            self._draw_football_field(axes[1,1])
            colors = ['red', 'blue', 'green']
            for third in sorted(features_df['field_third'].unique()):
                if not pd.isna(third):
                    third_data = features_df[features_df['field_third'] == third]
                    axes[1,1].scatter(third_data['x_target'], third_data['y_target'], 
                                     alpha=0.3, s=10, color=colors[int(third)], 
                                     label=f'Third {int(third)+1}')
            axes[1,1].legend()
            axes[1,1].set_title('Target Locations by Field Third')
        
        plt.tight_layout()
        plt.savefig(self.data_manager.eda_path / f"{feature_set_name}_football_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _feature_importance_preview(self, features_df, feature_set_name):

        """Preview feature importance using simple models"""
        print("   Creating feature importance preview...")
    
        feature_columns = [col for col in features_df.columns 
                      if col not in ['game_id', 'play_id', 'nfl_id', 'frame_id', 'x_target', 'y_target']]
    
        numeric_features = features_df[feature_columns].select_dtypes(include=[np.number])
    
        if len(numeric_features.columns) == 0:
            print("   âš ï¸� No numeric features for importance analysis")
            return
    
        # Prepare data
        X = numeric_features.fillna(numeric_features.median())
        y_x = features_df['x_target']
        y_y = features_df['y_target']
    
        # Remove rows with missing targets
        valid_idx = ~(y_x.isna() | y_y.isna())
        X = X[valid_idx]
        y_x = y_x[valid_idx]
        y_y = y_y[valid_idx]
    
        if len(X) < 100:
            print("   âš ï¸� Not enough data for feature importance analysis")
            return
    
        # Train simple models
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        fig.suptitle('Feature Importance Preview (Random Forest)', fontsize=16, fontweight='bold')
    
        # Store scores
        scores = {}
    
        for idx, (y, coord_name) in enumerate([(y_x, 'X'), (y_y, 'Y')]):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
        
            # Get feature importance
            importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
            }).sort_values('importance', ascending=True)
        
            # Plot top 15 features
            ax = ax1 if idx == 0 else ax2
            bars = ax.barh(importance.tail(15)['feature'], importance.tail(15)['importance'], 
                      color=plt.cm.viridis(np.linspace(0, 1, 15)))
            ax.set_title(f'Top 15 Features for {coord_name} Coordinate')
            ax.set_xlabel('Feature Importance')
        
            # Add value annotations on bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                   ha='left', va='center', fontsize=8)
        
            # Calculate and store score
            score = rf.score(X_test, y_test)
            scores[coord_name] = score
    
        plt.tight_layout()
        plt.savefig(self.data_manager.eda_path / f"{feature_set_name}_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.show()
    
        # Print model performance
        print(f"   ğŸ“ˆ Random Forest RÂ² Score - X: {scores['X']:.3f}, Y: {scores['Y']:.3f}")
    
    def _draw_football_field(self, ax):
        """Draw a football field background"""
        field_length = 120
        field_width = 53.3
        
        # Create rectangle for the field
        rect = plt.Rectangle((0, 0), field_length, field_width, 
                           linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.3)
        ax.add_patch(rect)
        
        # Add yard lines
        for x in range(10, field_length, 10):
            ax.axvline(x=x, color='white', alpha=0.5, linestyle='-', linewidth=1)
        
        # Add 50-yard line
        ax.axvline(x=60, color='white', alpha=1, linestyle='-', linewidth=2)
        
        # Set limits and aspect ratio
        ax.set_xlim(0, field_length)
        ax.set_ylim(0, field_width)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    def _create_advanced_features_analysis(self, features_df, feature_set_name):
        """Analysis of advanced physics and awareness features - ROBUST VERSION"""
        print("   Creating advanced features analysis...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Advanced Features Analysis', fontsize=16, fontweight='bold')
        
        # 1. Momentum Analysis
        if 'momentum_magnitude' in features_df.columns:
            axes[0,0].hist(features_df['momentum_magnitude'].dropna(), bins=50, alpha=0.7, color='blue')
            axes[0,0].set_title('Momentum Magnitude Distribution')
            axes[0,0].set_xlabel('Momentum (lbÂ·yards/s)')
            axes[0,0].set_ylabel('Frequency')
            axes[0,0].grid(True, alpha=0.3)
        else:
            axes[0,0].text(0.5, 0.5, 'Momentum data\nnot available', 
                        ha='center', va='center', transform=axes[0,0].transAxes)
            axes[0,0].set_title('Momentum Magnitude Distribution')
        
        # 2. Kinetic Energy Analysis
        if 'kinetic_energy' in features_df.columns:
            axes[0,1].hist(features_df['kinetic_energy'].dropna(), bins=50, alpha=0.7, color='red')
            axes[0,1].set_title('Kinetic Energy Distribution')
            axes[0,1].set_xlabel('Kinetic Energy')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].grid(True, alpha=0.3)
        else:
            axes[0,1].text(0.5, 0.5, 'Kinetic Energy data\nnot available', 
                        ha='center', va='center', transform=axes[0,1].transAxes)
            axes[0,1].set_title('Kinetic Energy Distribution')
        
        # 3. Eyes on QB by Player Role
        if 'eyes_on_qb' in features_df.columns and 'player_role' in features_df.columns:
            qb_awareness = features_df.groupby('player_role')['eyes_on_qb'].mean().sort_values(ascending=False)
            qb_awareness.head(10).plot(kind='bar', ax=axes[0,2], color='green')
            axes[0,2].set_title('Eyes on QB by Player Role')
            axes[0,2].set_ylabel('Percentage with Eyes on QB')
            axes[0,2].tick_params(axis='x', rotation=45)
        else:
            axes[0,2].text(0.5, 0.5, 'Eyes on QB data\nnot available', 
                        ha='center', va='center', transform=axes[0,2].transAxes)
            axes[0,2].set_title('Eyes on QB by Player Role')
        
        # 4. Eyes on Receiver by Defensive Position
        if ('eyes_on_receiver' in features_df.columns and 
            'player_position' in features_df.columns and 
            'player_side' in features_df.columns):
            
            defense_data = features_df[features_df['player_side'] == 'Defense']
            if len(defense_data) > 0:
                receiver_awareness = defense_data.groupby('player_position')['eyes_on_receiver'].mean().sort_values(ascending=False)
                receiver_awareness.plot(kind='bar', ax=axes[1,0], color='orange')
                axes[1,0].set_title('Eyes on Receiver by Defensive Position')
                axes[1,0].set_ylabel('Percentage with Eyes on Receiver')
                axes[1,0].tick_params(axis='x', rotation=45)
            else:
                axes[1,0].text(0.5, 0.5, 'No defensive player data\navailable', 
                            ha='center', va='center', transform=axes[1,0].transAxes)
                axes[1,0].set_title('Eyes on Receiver by Defensive Position')
        else:
            axes[1,0].text(0.5, 0.5, 'Eyes on Receiver data\nnot available', 
                        ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('Eyes on Receiver by Defensive Position')
        
        # 5. Maximum Speed vs Kinetic Energy
        if all(col in features_df.columns for col in ['max_speed_this_play', 'kinetic_energy']):
            axes[1,1].scatter(features_df['max_speed_this_play'], features_df['kinetic_energy'], 
                            alpha=0.5, s=1, color='purple')
            axes[1,1].set_xlabel('Maximum Speed (yards/s)')
            axes[1,1].set_ylabel('Kinetic Energy')
            axes[1,1].set_title('Max Speed vs Kinetic Energy')
            axes[1,1].grid(True, alpha=0.3)
        else:
            axes[1,1].text(0.5, 0.5, 'Speed/Energy comparison\ndata not available', 
                        ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Max Speed vs Kinetic Energy')
        
        # 6. Awareness Impact on Target Accuracy
        if all(col in features_df.columns for col in ['eyes_on_qb', 'eyes_on_receiver', 'dist_to_ball_land']):
            # Create combined awareness score
            features_df['total_awareness'] = features_df['eyes_on_qb'] + features_df['eyes_on_receiver']
            awareness_accuracy = features_df.groupby('total_awareness')['dist_to_ball_land'].mean()
            axes[1,2].plot(awareness_accuracy.index, awareness_accuracy.values, marker='o', linewidth=2, markersize=8)
            axes[1,2].set_xlabel('Total Awareness Score')
            axes[1,2].set_ylabel('Avg Distance to Ball Landing')
            axes[1,2].set_title('Awareness vs Positioning Accuracy')
            axes[1,2].grid(True, alpha=0.3)
        else:
            axes[1,2].text(0.5, 0.5, 'Awareness vs Accuracy data\nnot available', 
                        ha='center', va='center', transform=axes[1,2].transAxes)
            axes[1,2].set_title('Awareness vs Positioning Accuracy')
        
        plt.tight_layout()
        plt.savefig(self.data_manager.eda_path / f"{feature_set_name}_advanced_features.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print advanced feature insights
        print("\nğŸ”¬ Advanced Features Insights:")
        
        if 'eyes_on_qb' in features_df.columns:
            qb_awareness_rate = features_df['eyes_on_qb'].mean()
            print(f"   ğŸ‘€ Overall Eyes on QB: {qb_awareness_rate:.1%}")
        else:
            print("   ğŸ‘€ Eyes on QB: Not available")
        
        if 'eyes_on_receiver' in features_df.columns and 'player_side' in features_df.columns:
            defense_df = features_df[features_df['player_side'] == 'Defense']
            if len(defense_df) > 0:
                receiver_awareness_rate = defense_df['eyes_on_receiver'].mean()
                print(f"   ğŸ‘€ Defensive Eyes on Receiver: {receiver_awareness_rate:.1%}")
            else:
                print("   ğŸ‘€ Defensive Eyes on Receiver: No defensive players found")
        else:
            print("   ğŸ‘€ Defensive Eyes on Receiver: Not available")
        
        if 'momentum_magnitude' in features_df.columns:
            avg_momentum = features_df['momentum_magnitude'].mean()
            print(f"   ğŸš€ Average Momentum: {avg_momentum:.1f} lbÂ·yards/s")
        else:
            print("   ğŸš€ Momentum: Not available")
        
        if 'kinetic_energy' in features_df.columns:
            avg_energy = features_df['kinetic_energy'].mean()
            print(f"   âš¡ Average Kinetic Energy: {avg_energy:.1f}")
        else:
            print("   âš¡ Kinetic Energy: Not available")

# =============================================================================
# DATA LOADING FUNCTION
# =============================================================================

def load_nfl_data(data_path, sample_weeks=2):
    """Load NFL competition data"""
    print("ğŸ“‚ Loading NFL Big Data Bowl 2026 Data...")
    print("="*60)
    
    data_path = Path(data_path)
    train_input = {}
    train_output = {}
    
    for week in range(1, sample_weeks + 1):
        input_file = data_path / f'input_2023_w{week:02d}.csv'
        output_file = data_path / f'output_2023_w{week:02d}.csv'
        
        if input_file.exists():
            train_input[f'week_{week}'] = pd.read_csv(input_file)
            print(f"âœ… Loaded: input_2023_w{week:02d}.csv")
            
        if output_file.exists():
            train_output[f'week_{week}'] = pd.read_csv(output_file)
            print(f"âœ… Loaded: output_2023_w{week:02d}.csv")
    
    print(f"\nğŸ“Š Summary:")
    print(f"   Input weeks: {len(train_input)}")
    print(f"   Output weeks: {len(train_output)}")
    
    if train_input:
        sample_df = list(train_input.values())[0]
        print(f"   Sample input shape: {sample_df.shape}")
        print(f"   Sample columns: {list(sample_df.columns)[:10]}...")
    
    return train_input, train_output

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Configuration
    DATA_PATH = "/kaggle/input/nfl-big-data-bowl-2026-prediction/train"
    SAMPLE_WEEKS = 2 # change as needed
    FEATURE_SET_NAME = "baseline_features_v1"
    
    print("ğŸ�ˆ NFL Big Data Bowl 2026 - Complete Feature Engineering & EDA")
    print("="*60)
    
    # Step 1: Load Data
    train_input, train_output = load_nfl_data(DATA_PATH, SAMPLE_WEEKS)
    
    if not train_input:
        print("â�Œ No data loaded. Please check the data path.")
        exit()
    
    # Step 2: Initialize Components
    data_manager = NFLDataManager()
    feature_engineer = NFLFeatureEngineer(data_manager)
    eda_analyzer = NFLFeatureEDA(data_manager)
    
    # Step 3: Engineer Features
    combined_features = feature_engineer.engineer_features(
        train_input, train_output, FEATURE_SET_NAME
    )
    
    if combined_features is not None:
        # Step 4: Run Comprehensive EDA
        eda_analyzer.create_comprehensive_eda(combined_features, FEATURE_SET_NAME)
        
        # Step 5: Final Summary
        feature_columns = [col for col in combined_features.columns 
                          if col not in ['game_id', 'play_id', 'nfl_id', 'frame_id', 'x_target', 'y_target']]
        
        print("\n" + "="*60)
        print("ğŸ�‰ FEATURE ENGINEERING & EDA COMPLETE!")
        print("="*60)
        print(f"ğŸ“ˆ Total engineered features: {len(feature_columns)}")
        print(f"ğŸ“Š Total samples: {len(combined_features)}")
        print(f"ğŸ’¾ Features saved to: {data_manager.features_path}")
        print(f"ğŸ“‹ EDA visualizations saved to: {data_manager.eda_path}")
        
        # Feature category breakdown
        categories = {
            'Player Features': [col for col in feature_columns if col.startswith('pos_') or col.startswith('role_') or col in ['height_inches', 'weight', 'is_offense']],
            'Movement Features': [col for col in feature_columns if col in ['x', 'y', 's', 'a', 'o', 'dir', 'velocity_x', 'velocity_y', 'speed_squared', 'is_moving'] or 'sin' in col or 'cos' in col or 'rad' in col],
            'Play Context': [col for col in feature_columns if col in ['yardline', 'field_third', 'moving_left', 'ball_land_x', 'ball_land_y', 'dist_to_ball_land', 'frames_to_predict']],
            'Football Intelligence': [col for col in feature_columns if col in ['is_targeted_receiver', 'is_passer', 'is_wide_receiver', 'dist_to_sideline', 'dist_to_endzone']]
        }
        
        print(f"\nğŸ“‚ Feature Categories:")
        for category, features in categories.items():
            print(f"   {category}: {len(features)} features")
            
    else:
        print("â�Œ Feature engineering failed!")

