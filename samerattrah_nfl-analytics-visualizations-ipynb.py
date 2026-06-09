import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# Ensure backend acts non-interactively
import matplotlib
matplotlib.use('Agg')

class GlobalPositionAnalyzer:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def convert_height(self, height_str):
        if pd.isna(height_str): return None
        try:
            if isinstance(height_str, str) and '-' in height_str:
                feet, inches = map(int, height_str.split('-'))
                return feet * 12 + inches
            return float(height_str)
        except:
            return None

    def load_all_data(self):
        files = glob.glob(os.path.join(self.data_dir, "input_*.csv"))
        files.sort()
        
        if not files:
            print("No input files found!")
            return None

        print(f"Found {len(files)} week files. Aggregating data...")
        
        data_frames = []
        cols_to_use = [
            'nfl_id', 'play_id', 'player_position', 'player_turnover_metrics', # Basic IDs and grouping
            'player_height', 'player_weight', # Static Attrs
            's', 'a', 'o', 'dir', # Dynamic Attrs
            'player_role', 'player_side', # Context Attrs
            'x', 'y', 'ball_land_x', 'ball_land_y' # Location Attrs
        ]
        
        # Robustly handle columns that might be missing in some files or named differently if needed
        # Based on previous file checks, these should exist. But let's be safe.
        
        for f in files:
            week = os.path.basename(f)
            print(f"Reading {week}...", end='\r')
            try:
                # Read only first row to check cols
                header = pd.read_csv(f, nrows=0).columns.tolist()
                use_cols = [c for c in cols_to_use if c in header]
                
                df_chunk = pd.read_csv(f, usecols=use_cols)
                
                # Convert height immediately to save processing later and standardise
                if 'player_height' in df_chunk.columns:
                     df_chunk['player_height_inches'] = df_chunk['player_height'].apply(self.convert_height)
                
                # Optimise types
                for col in ['player_position', 'player_role', 'player_side']:
                    if col in df_chunk.columns:
                        df_chunk[col] = df_chunk[col].astype('category')
                
                data_frames.append(df_chunk)
            except Exception as e:
                print(f"\nFailed to read {week}: {e}")

        print("\nConcatenating all weeks...")
        full_df = pd.concat(data_frames, ignore_index=True)
        print(f"Total records: {len(full_df)}")
        return full_df

    def plot_player_vs_target_locations(self, df, output_sub):
        """
        Generates scatter plots for each position comparing player location (x, y) 
        vs ball landing location (ball_land_x, ball_land_y).
        """
        if not all(col in df.columns for col in ['x', 'y', 'ball_land_x', 'ball_land_y', 'player_position']):
            print("Missing location columns. Skipping location analysis.")
            return

        print("Generating player vs target location plots...")
        loc_output_dir = os.path.join(output_sub, 'location_analysis')
        os.makedirs(loc_output_dir, exist_ok=True)

        positions = df['player_position'].unique()
        
        for pos in positions:
            if pd.isna(pos): continue
            
            # Filter data for this position
            pos_df = df[df['player_position'] == pos].copy()
            
            # Drop NaNs for plotting
            pos_df = pos_df.dropna(subset=['x', 'y', 'ball_land_x', 'ball_land_y'])
            
            if pos_df.empty:
                continue

            # Subsample if too large to prevent memory issues/overplotting
            if len(pos_df) > 50000:
                plot_data = pos_df.sample(50000, random_state=42)
            else:
                plot_data = pos_df

            plt.figure(figsize=(12, 6))
            
            # Plot Player Locations
            plt.scatter(plot_data['x'], plot_data['y'], 
                        alpha=0.2, s=5, label='Player Location', color='blue')
            
            # Plot Ball Landing Locations
            # Note: Ball landing might be the same for all players in a play, 
            # so this distribution represents the target spots relevant to this position's plays.
            plt.scatter(plot_data['ball_land_x'], plot_data['ball_land_y'], 
                        alpha=0.2, s=5, label='Ball Landing (Perfect) Location', color='red', marker='x')

            plt.title(f'Player vs Ball Landing Locations - Position: {pos}')
            plt.xlabel('X Coordinate (yards)')
            plt.ylabel('Y Coordinate (yards)')
            plt.legend()
            
            # Draw Field Borders (approximate)
            plt.axhline(0, color='gray', linestyle='--')
            plt.axhline(53.3, color='gray', linestyle='--')
            plt.axvline(0, color='gray', linestyle='--')
            plt.axvline(120, color='gray', linestyle='--')
            
            plt.tight_layout()
            plt.savefig(os.path.join(loc_output_dir, f'location_vs_target_{pos}.png'))
            plt.close()
            print(f"Generated location plot for {pos}")

    def analyze_global(self):
        df = self.load_all_data()
        if df is None or df.empty:
            print("No data to analyze.")
            return

        print("Generating global visualizations...")
        
        output_sub = self.output_dir
        os.makedirs(output_sub, exist_ok=True)

        # --- Location Analysis ---
        self.plot_player_vs_target_locations(df, output_sub)

        # --- Static Features (per player) ---
        # unique player check
        if 'nfl_id' in df.columns:
             df_player_static = df.drop_duplicates(subset=['nfl_id'])
        else:
             df_player_static = df

        # 1. Height vs Position
        if 'player_height_inches' in df_player_static.columns:
            plt.figure(figsize=(16, 8))
            sns.boxplot(x='player_position', y='player_height_inches', data=df_player_static)
            plt.title('Player Height by Position (All Weeks)')
            plt.xlabel('Position')
            plt.ylabel('Height (inches)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_sub, 'global_height_by_position.png'))
            plt.close()

        # 2. Weight vs Position
        if 'player_weight' in df_player_static.columns:
            plt.figure(figsize=(16, 8))
            sns.boxplot(x='player_position', y='player_weight', data=df_player_static)
            plt.title('Player Weight by Position (All Weeks)')
            plt.xlabel('Position')
            plt.ylabel('Weight (lbs)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_sub, 'global_weight_by_position.png'))
            plt.close()

        # --- Dynamic Features (per frame) ---
        # 3. Speed vs Position
        if 's' in df.columns:
            plt.figure(figsize=(16, 8))
            sns.boxplot(x='player_position', y='s', data=df)
            plt.title('Speed by Position (All Weeks)')
            plt.xlabel('Position')
            plt.ylabel('Speed (yards/s)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_sub, 'global_speed_by_position.png'))
            plt.close()

        # 4. Acceleration vs Position
        if 'a' in df.columns:
            plt.figure(figsize=(16, 8))
            sns.boxplot(x='player_position', y='a', data=df)
            plt.title('Acceleration by Position (All Weeks)')
            plt.xlabel('Position')
            plt.ylabel('Acceleration (yards/s^2)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_sub, 'global_acceleration_by_position.png'))
            plt.close()
        
        # 5. Orientation (o) vs Position
        if 'o' in df.columns:
            plt.figure(figsize=(16, 8))
            sns.boxplot(x='player_position', y='o', data=df)
            plt.title('Orientation (o) by Position (All Weeks)')
            plt.xlabel('Position')
            plt.ylabel('Orientation (degrees)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_sub, 'global_orientation_by_position.png'))
            plt.close()

        # 6. Direction (dir) vs Position
        if 'dir' in df.columns:
            plt.figure(figsize=(16, 8))
            sns.boxplot(x='player_position', y='dir', data=df)
            plt.title('Direction (dir) by Position (All Weeks)')
            plt.xlabel('Position')
            plt.ylabel('Direction (degrees)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_sub, 'global_direction_by_position.png'))
            plt.close()

        # --- Categorical/Play Features ---
        # Drop duplicates for role/side analysis per play per player to avoid frame bias
        if 'play_id' in df.columns and 'nfl_id' in df.columns:
            df_play_static = df.drop_duplicates(subset=['play_id', 'nfl_id'])
        else:
            df_play_static = df

        # 7. Role Distribution
        if 'player_role' in df.columns:
            plt.figure(figsize=(16, 8))
            ct = pd.crosstab(df_play_static['player_position'], df_play_static['player_role'])
            ct.plot(kind='bar', stacked=True, figsize=(16, 8), cmap='viridis')
            plt.title('Player Role Distribution by Position (All Weeks)')
            plt.xlabel('Position')
            plt.ylabel('Count of Plays')
            plt.xticks(rotation=45)
            plt.legend(title='Player Role', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_sub, 'global_role_by_position.png'))
            plt.close()

        # 8. Side Distribution
        if 'player_side' in df.columns:
            plt.figure(figsize=(16, 8))
            ct = pd.crosstab(df_play_static['player_position'], df_play_static['player_side'])
            ct.plot(kind='bar', stacked=True, figsize=(16, 8), cmap='Set2')
            plt.title('Player Side Distribution by Position (All Weeks)')
            plt.xlabel('Position')
            plt.ylabel('Count of Plays')
            plt.xticks(rotation=45)
            plt.legend(title='Player Side', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_sub, 'global_side_by_position.png'))
            plt.close()

        print("Global analysis complete.")

if __name__ == "__main__":
    INPUT_DIR = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train"
    OUTPUT_DIR = "/kaggle/working/global_distributions"
    
    analyzer = GlobalPositionAnalyzer(INPUT_DIR, OUTPUT_DIR)
    analyzer.analyze_global()



import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys

# Ensure backend acts non-interactively
import matplotlib
matplotlib.use('Agg')

class OutputDistributionAnalyzer:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def analyze_global(self):
        input_files = glob.glob(os.path.join(self.data_dir, "input_*.csv"))
        input_files.sort()
        
        if not input_files:
            print("No input files found!")
            return

        print(f"Found {len(input_files)} week files. Aggregating output targets...")
        
        # Structure: {position: {'end_x': [], 'end_y': []}}
        position_targets = {} 

        for f in input_files:
            week_name = os.path.basename(f).replace('input_', '').replace('.csv', '')
            output_file = os.path.join(self.data_dir, f"output_{week_name}.csv")
            
            if not os.path.exists(output_file):
                continue

            print(f"Processing {week_name}...", end='\r')

            try:
                # 1. Read Output Data -> Get FINAL position per play
                # We assume the last frame in output is the target 'final' position
                df_out = pd.read_csv(output_file)
                # Sort to ensure we get last frame
                df_out = df_out.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
                
                # Get last frame for each player/play
                final_pos = df_out.groupby(['game_id', 'play_id', 'nfl_id'], as_index=False).last()
                final_pos = final_pos.rename(columns={'x': 'end_x', 'y': 'end_y'})
                final_pos = final_pos[['game_id', 'play_id', 'nfl_id', 'end_x', 'end_y']]

                # 2. Read Input Data -> Get POSITION info
                # We only want to map nfl_id/play/game to position
                use_cols = ['game_id', 'play_id', 'nfl_id', 'player_position', 'player_to_predict']
                df_in = pd.read_csv(f, usecols=lambda c: c in use_cols)
                
                # Filter for players to predict
                if 'player_to_predict' in df_in.columns:
                    df_in['player_to_predict'] = df_in['player_to_predict'].astype(str)
                    df_in = df_in[df_in['player_to_predict'] == 'True']
                
                # We just need the unique mapping per play
                df_in = df_in[['game_id', 'play_id', 'nfl_id', 'player_position']].drop_duplicates()

                # 3. Merge to associate Position with Final Output
                merged_df = pd.merge(final_pos, df_in, on=['game_id', 'play_id', 'nfl_id'], how='inner')
                
                # 4. Store by Position
                for pos, group in merged_df.groupby('player_position'):
                    if pos not in position_targets:
                        position_targets[pos] = {'end_x': [], 'end_y': []}
                    
                    position_targets[pos]['end_x'].extend(group['end_x'].tolist())
                    position_targets[pos]['end_y'].extend(group['end_y'].tolist())

            except Exception as e:
                print(f"\nError processing {week_name}: {e}")
        
        print("\nAll weeks processed. Generating output distribution plots...")
        
        # Style settings
        plt.style.use('seaborn-v0_8-darkgrid') 
        
        for pos, data in position_targets.items():
            if not data['end_x']: continue
            
            print(f"Plotting {pos}...", end='\r')
            
            ex = np.array(data['end_x'])
            ey = np.array(data['end_y'])
            
            # Subsample if massive
            count = len(ex)
            if count > 50000:
                indices = np.random.choice(count, 50000, replace=False)
                ex, ey = ex[indices], ey[indices]
            
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.set_facecolor('#f0f0f0') 
            
            # Scatter Plot of Final Destinations
            # Use alpha and distinct color (e.g., Red or heatmap like)
            ax.scatter(ex, ey, c='red', s=15, alpha=0.3, label='Final Output Position')
            
            # KDE plot (optional density - can be slow for many points, stick to scatter or maybe hexbin)
            # ax.hexbin(ex, ey, gridsize=30, cmap='Reds', mincnt=1)
            
            # Field Markings
            ax.axhline(0, color='black', linewidth=1)
            ax.axhline(53.3, color='black', linewidth=1)
            ax.axvline(0, color='black', linewidth=1)
            ax.axvline(120, color='black', linewidth=1)
            for x in range(10, 110, 10):
                ax.axvline(x, color='gray', linestyle=':', alpha=0.5)
            
            ax.set_title(f'Final Output Positions Distribution - Position: {pos}', fontsize=16, pad=15)
            ax.set_xlabel('Field X (yards)', fontsize=12)
            ax.set_ylabel('Field Y (yards)', fontsize=12)
            ax.set_aspect('equal')
            ax.set_xlim(-5, 125)
            ax.set_ylim(-5, 60)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'output_distribution_{pos}.png'), dpi=150)
            plt.close()
            
        print("\nOutput distribution analysis complete.")

if __name__ == "__main__":
    INPUT_DIR = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train"
    OUTPUT_DIR = "/kaggle/working/output_distributions"
    
    analyzer = OutputDistributionAnalyzer(INPUT_DIR, OUTPUT_DIR)
    analyzer.analyze_global()



import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys

# Ensure backend acts non-interactively
import matplotlib
matplotlib.use('Agg')

class MotionVsFinalAnalyzer:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def analyze_global(self):
        input_files = glob.glob(os.path.join(self.data_dir, "input_*.csv"))
        input_files.sort()
        
        if not input_files:
            print("No input files found!")
            return

        print(f"Found {len(input_files)} week files. Pairing trajectories...")
        
        # Structure: {position: {'start_x': [], 'start_y': [], 'end_x': [], 'end_y': []}}
        position_vectors = {} 

        for f in input_files:
            week_name = os.path.basename(f).replace('input_', '').replace('.csv', '')
            output_file = os.path.join(self.data_dir, f"output_{week_name}.csv")
            
            if not os.path.exists(output_file):
                continue

            print(f"Processing {week_name}...", end='\r')

            try:
                # 1. Read Output Data -> Get FINAL position per play
                # We assume the last frame in output is the target 'final' position
                df_out = pd.read_csv(output_file)
                # Sort to ensure we get last frame
                df_out = df_out.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
                
                # Get last frame for each player/play
                final_pos = df_out.groupby(['game_id', 'play_id', 'nfl_id'], as_index=False).last()
                final_pos = final_pos.rename(columns={'x': 'end_x', 'y': 'end_y'})
                final_pos = final_pos[['game_id', 'play_id', 'nfl_id', 'end_x', 'end_y']]

                # 2. Read Input Data -> Get LAST KNOWN position (Start of prediction)
                # We only care about player_to_predict=True
                use_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id', 'player_position', 'player_to_predict', 'x', 'y']
                df_in = pd.read_csv(f, usecols=lambda c: c in use_cols)
                
                # Standardize boolean col
                if 'player_to_predict' in df_in.columns:
                    df_in['player_to_predict'] = df_in['player_to_predict'].astype(str)
                    df_in = df_in[df_in['player_to_predict'] == 'True']

                # Get the last frame from input (the 'current' state before prediction)
                df_in = df_in.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
                start_pos = df_in.groupby(['game_id', 'play_id', 'nfl_id'], as_index=False).last()
                start_pos = start_pos.rename(columns={'x': 'start_x', 'y': 'start_y'})
                start_pos = start_pos[['game_id', 'play_id', 'nfl_id', 'player_position', 'start_x', 'start_y']]

                # 3. Merge to create vectors (Start -> End)
                # Inner join ensures we have both start and end for the same play/player
                vectors_df = pd.merge(start_pos, final_pos, on=['game_id', 'play_id', 'nfl_id'], how='inner')
                
                # 4. Store by Position
                for pos, group in vectors_df.groupby('player_position'):
                    if pos not in position_vectors:
                        position_vectors[pos] = {'start_x': [], 'start_y': [], 'end_x': [], 'end_y': []}
                    
                    # Store data
                    # To avoid memory explosion, we can keep subsampling during collection if needed,
                    # but lists of floats are reasonably efficient for ~20 weeks.
                    position_vectors[pos]['start_x'].extend(group['start_x'].tolist())
                    position_vectors[pos]['start_y'].extend(group['start_y'].tolist())
                    position_vectors[pos]['end_x'].extend(group['end_x'].tolist())
                    position_vectors[pos]['end_y'].extend(group['end_y'].tolist())

            except Exception as e:
                print(f"\nError processing {week_name}: {e}")
        
        print("\nAll weeks processed. Generating vector plots...")
        
        plot_dir = self.output_dir
        os.makedirs(plot_dir, exist_ok=True)
        
        # Style settings for "readable" and "premium" look
        plt.style.use('seaborn-v0_8-darkgrid') 
        # Or manually set style
        
        for pos, data in position_vectors.items():
            if not data['start_x']: continue
            
            print(f"Plotting {pos}...", end='\r')
            
            # Convert to numpy for easier handling
            sx = np.array(data['start_x'])
            sy = np.array(data['start_y'])
            ex = np.array(data['end_x'])
            ey = np.array(data['end_y'])
            
            count = len(sx)
            
            # Smart Subsampling
            # If too many points, randomly sample N points to keep plot readable
            MAX_ARROWS = 800
            if count > MAX_ARROWS:
                indices = np.random.choice(count, MAX_ARROWS, replace=False)
                sx, sy, ex, ey = sx[indices], sy[indices], ex[indices], ey[indices]
            
            # Calculate vectors
            u = ex - sx
            v = ey - sy
            mag = np.sqrt(u**2 + v**2) # Distance traveled
            
            # Create Plot
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot Field Background Context (optional, but helps readability)
            ax.set_facecolor('#f0f0f0') # Light gray background
            
            # Quiver Plot (Arrows)
            # Color by Magnitude (Distance) to show "explosiveness" or "long runs"
            # Cmap: 'plasma' (purple-orange-yellow) stands out well on light bg
            q = ax.quiver(sx, sy, u, v, mag, 
                          angles='xy', scale_units='xy', scale=1, 
                          cmap='plasma', alpha=0.8, width=0.003, headwidth=4, headlength=5)
            
            # Add Scatter for endpoints to mark the final spot firmly
            # ax.scatter(ex, ey, c='black', s=10, alpha=0.5, marker='.', zorder=2)
            
            # Colorbar
            cbar = plt.colorbar(q, ax=ax)
            cbar.set_label('Distance Traveled (yards)')
            
            # Field Markings
            ax.axhline(0, color='black', linewidth=1)
            ax.axhline(53.3, color='black', linewidth=1)
            ax.axvline(0, color='black', linewidth=1)
            ax.axvline(120, color='black', linewidth=1)
            # Yard lines
            for x in range(10, 110, 10):
                ax.axvline(x, color='gray', linestyle=':', alpha=0.5)
            
            # Labels
            ax.set_title(f'Player Motion Vectors: Start to Final Position - Position: {pos}', fontsize=16, pad=15)
            ax.set_xlabel('Field X (yards)', fontsize=12)
            ax.set_ylabel('Field Y (yards)', fontsize=12)
            
            # Fix aspect ratio to represent real field
            ax.set_aspect('equal')
            
            # Interactive-like limits
            ax.set_xlim(-5, 125)
            ax.set_ylim(-5, 60)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'vector_motion_{pos}.png'), dpi=150)
            plt.close()
            
        print("\nVector analysis complete.")

if __name__ == "__main__":
    INPUT_DIR = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train"
    OUTPUT_DIR = "/kaggle/working/motion_vectors"
    
    analyzer = MotionVsFinalAnalyzer(INPUT_DIR, OUTPUT_DIR)
    analyzer.analyze_global()



"""Sequence visualization script organized by player position.

This script generates per-week trajectory plots for each position,
showing player motion paths, final predicted positions, and ball landing locations.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ensure backend acts non-interactively
import matplotlib
matplotlib.use('Agg')


class SequenceVisualizerByPosition:
    """Handles the generation of sequence-level tracking visualizations,
    organized by Player Position.
    """

    def __init__(self, data_dir, output_dir):
        """Initialize the visualizer.

        Args:
            data_dir: Directory containing input_*.csv and output_*.csv files.
            output_dir: Directory to save generated plots.
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def get_weeks(self):
        """Discover available weeks from input files.

        Returns:
            Sorted list of week identifiers (e.g., ['2023_w01', '2023_w02', ...]).
        """
        input_files = glob.glob(os.path.join(self.data_dir, "input_*.csv"))
        weeks = []
        for f in input_files:
            week_name = os.path.basename(f).replace('input_', '').replace('.csv', '')
            weeks.append(week_name)
        return sorted(weeks)

    def read_input(self, week):
        """Read input CSV for a specific week.

        Args:
            week: Week identifier string.

        Returns:
            DataFrame with input data.
        """
        file_path = os.path.join(self.data_dir, f"input_{week}.csv")
        return pd.read_csv(file_path)

    def read_output(self, week):
        """Read output CSV for a specific week.

        Args:
            week: Week identifier string.

        Returns:
            DataFrame with output data, or None if file doesn't exist.
        """
        file_path = os.path.join(self.data_dir, f"output_{week}.csv")
        if not os.path.exists(file_path):
            return None
        return pd.read_csv(file_path)

    def generate_all_weeks(self, limit_per_position=5):
        """Iterate through all available weeks and generate sequence plots.

        Args:
            limit_per_position: Max sequences to process per position per week.
        """
        weeks = self.get_weeks()
        if not weeks:
            print("No weeks found in data directory.")
            return

        print(f"Found {len(weeks)} weeks of data. Starting sequence generation...")
        print(f"Limit per position: {limit_per_position}")

        for week in tqdm(weeks, desc="Processing Weeks"):
            try:
                self.visualize_week(week, limit_per_pos=limit_per_position)
            except Exception as e:
                print(f"Error processing week {week}: {e}")

    def visualize_week(self, week, limit_per_pos=5):
        """Generate sequence plots for a specific week, grouping by position.

        Args:
            week: Week identifier string.
            limit_per_pos: Max sequences per position.
        """
        try:
            input_df = self.read_input(week)
            output_df = self.read_output(week)
        except Exception as e:
            print(f"Error reading data for week {week}: {e}")
            return

        if output_df is None:
            print(f"No output file found for week {week}, skipping.")
            return

        # Filter for players to predict
        if 'player_to_predict' not in input_df.columns:
            print("'player_to_predict' column not found.")
            return

        # Handle string/bool conversion
        if input_df['player_to_predict'].dtype == 'object':
            target_players = input_df[
                input_df['player_to_predict'].astype(str) == 'True'
            ]
        else:
            target_players = input_df[input_df['player_to_predict'] == True]

        # Prepare directory for this week
        week_dir = os.path.join(self.output_dir, f'week_{week}')
        os.makedirs(week_dir, exist_ok=True)

        # Get unique positions
        unique_positions = target_players['player_position'].dropna().unique()

        for pos in unique_positions:
            # Filter distinct sequences for this position
            pos_df = target_players[target_players['player_position'] == pos]
            # Group by sequence
            grouped = pos_df.groupby(['game_id', 'play_id', 'nfl_id'])

            # Get list of sequences
            sequences = list(grouped.groups.keys())

            # Limit
            if limit_per_pos:
                sequences = sequences[:limit_per_pos]

            if not sequences:
                continue

            # Create position subdirectory
            pos_dir = os.path.join(week_dir, pos)
            os.makedirs(pos_dir, exist_ok=True)

            for (game_id, play_id, nfl_id) in sequences:
                try:
                    # 1. Input Data
                    input_seq = grouped.get_group((game_id, play_id, nfl_id))

                    # 2. Output Data (Final Pos)
                    output_seq = output_df[
                        (output_df['game_id'] == game_id) &
                        (output_df['play_id'] == play_id) &
                        (output_df['nfl_id'] == nfl_id)
                    ]

                    if output_seq.empty:
                        continue

                    final_output_pos = output_seq.iloc[-1]

                    # 3. Ball Landing
                    game_row = input_seq.iloc[0]
                    ball_x = game_row.get('ball_land_x', float('nan'))
                    ball_y = game_row.get('ball_land_y', float('nan'))

                    self.plot_sequence(
                        input_seq,
                        final_output_pos,
                        ball_x, ball_y,
                        game_id, play_id, nfl_id,
                        pos,
                        pos_dir
                    )

                except Exception as e:
                    print(f"Error plotting {game_id}-{play_id}-{nfl_id}: {e}")

    def plot_sequence(self, input_df, output_final_row, ball_x, ball_y,
                      game_id, play_id, nfl_id, position, output_dir):
        """Create a plot for the sequence focusing on position context.

        Args:
            input_df: DataFrame with input trajectory data.
            output_final_row: Series with final output position.
            ball_x: Ball landing x-coordinate.
            ball_y: Ball landing y-coordinate.
            game_id: Game identifier.
            play_id: Play identifier.
            nfl_id: Player identifier.
            position: Player position string.
            output_dir: Directory to save the plot.
        """
        plt.figure(figsize=(14, 7))

        # Field Context
        plt.xlim(0, 120)
        plt.ylim(0, 53.3)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.axhline(0, color='black', linewidth=1)
        plt.axhline(53.3, color='black', linewidth=1)
        plt.axvline(0, color='black', linewidth=1)
        plt.axvline(120, color='black', linewidth=1)

        # Yard Lines
        for x in range(10, 110, 10):
            plt.axvline(x, color='gray', linestyle=':', alpha=0.5)

        # 1. Input Trajectory
        plt.plot(
            input_df['x'], input_df['y'],
            'b-', label='Motion Path', alpha=0.6, linewidth=2
        )
        plt.scatter(
            input_df['x'], input_df['y'],
            c=range(len(input_df)), cmap='Blues', s=20, alpha=0.8, edgecolor='none'
        )

        # Start
        plt.plot(
            input_df['x'].iloc[0], input_df['y'].iloc[0],
            'go', label='Start', markersize=8
        )

        # End (Input)
        plt.plot(
            input_df['x'].iloc[-1], input_df['y'].iloc[-1],
            'bs', label='End Input', markersize=8
        )

        # 2. Final Output Position
        plt.plot(
            output_final_row['x'], output_final_row['y'],
            'rx', label='Final Prediction Target', markersize=12, markeredgewidth=3
        )

        # 3. Ball Landing
        if pd.notna(ball_x) and pd.notna(ball_y):
            plt.plot(
                ball_x, ball_y,
                'y*', label='Ball Landing', markersize=15, markeredgecolor='black'
            )

        plt.title(
            f'Position Analysis: {position} | Game {game_id} Play {play_id} '
            f'Player {nfl_id}',
            fontsize=14
        )
        plt.xlabel('Field X (yards)')
        plt.ylabel('Field Y (yards)')
        plt.legend(loc='upper right')

        # Add text box for Role/Position details
        role = input_df['player_role'].iloc[0] if 'player_role' in input_df.columns else 'Unknown'
        info_text = f"Position: {position}\nRole: {role}"
        plt.text(
            2, 51, info_text,
            fontsize=10, bbox=dict(facecolor='white', alpha=0.9, boxstyle='round')
        )

        filename = f"{position}_seq_{game_id}_{play_id}_{nfl_id}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=100)
        plt.close()


def main():
    """Main entry point for the sequence visualizer."""
    # Data directory containing input_*.csv and output_*.csv
    data_dir = (
        '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train'
    )

    # Output directory for plots
    output_dir = (
        '/kaggle/working/sequence_by_position'
    )

    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return

    print("Initializing Position-Based Sequence Visualizer...")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    try:
        visualizer = SequenceVisualizerByPosition(data_dir, output_dir)

        # Run for all weeks, limited to 5 examples per position per week
        visualizer.generate_all_weeks(limit_per_position=5)

        print(f"\nVisualizations saved to {visualizer.output_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()



import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# Ensure backend acts non-interactively
import matplotlib
matplotlib.use('Agg')

class PositionAnalyzer:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def convert_height(self, height_str):
        if pd.isna(height_str): return None
        try:
            if isinstance(height_str, str) and '-' in height_str:
                feet, inches = map(int, height_str.split('-'))
                return feet * 12 + inches
            return float(height_str)
        except:
            return None

    def analyze_week(self, file_path):
        week_name = os.path.basename(file_path).replace('input_', '').replace('.csv', '')
        print(f"Processing {week_name}...")
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            return

        # Preprocessing
        df['player_height_inches'] = df['player_height'].apply(self.convert_height)
        
        # Determine output directory for this week
        week_out = os.path.join(self.output_dir, week_name)
        os.makedirs(week_out, exist_ok=True)
        
        # --- Static Features (per player) ---
        # Height and Weight should be unique per player, but let's take one entry per nfl_id
        df_player_static = df.drop_duplicates(subset=['nfl_id']).copy()

        # 1. Height vs Position
        plt.figure(figsize=(16, 8))
        sns.boxplot(x='player_position', y='player_height_inches', data=df_player_static)
        plt.title(f'Player Height by Position ({week_name})')
        plt.xlabel('Position')
        plt.ylabel('Height (inches)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(week_out, 'height_by_position.png'))
        plt.close()

        # 2. Weight vs Position
        plt.figure(figsize=(16, 8))
        sns.boxplot(x='player_position', y='player_weight', data=df_player_static)
        plt.title(f'Player Weight by Position ({week_name})')
        plt.xlabel('Position')
        plt.ylabel('Weight (lbs)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(week_out, 'weight_by_position.png'))
        plt.close()

        # --- Dynamic Features (per frame) ---
        # S, A, Dir, O, X, Y
        
        # 3. Speed vs Position
        plt.figure(figsize=(16, 8))
        sns.boxplot(x='player_position', y='s', data=df)
        plt.title(f'Speed by Position ({week_name})')
        plt.xlabel('Position')
        plt.ylabel('Speed (yards/s)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(week_out, 'speed_by_position.png'))
        plt.close()

        # 4. Acceleration vs Position
        plt.figure(figsize=(16, 8))
        sns.boxplot(x='player_position', y='a', data=df)
        plt.title(f'Acceleration by Position ({week_name})')
        plt.xlabel('Position')
        plt.ylabel('Acceleration (yards/s^2)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(week_out, 'acceleration_by_position.png'))
        plt.close()
        
        # 5. Orientation (o) vs Position
        if 'o' in df.columns:
            plt.figure(figsize=(16, 8))
            sns.boxplot(x='player_position', y='o', data=df)
            plt.title(f'Orientation (o) by Position ({week_name})')
            plt.xlabel('Position')
            plt.ylabel('Orientation (degrees)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(week_out, 'orientation_by_position.png'))
            plt.close()

        # 6. Direction (dir) vs Position
        if 'dir' in df.columns:
            plt.figure(figsize=(16, 8))
            sns.boxplot(x='player_position', y='dir', data=df)
            plt.title(f'Direction (dir) by Position ({week_name})')
            plt.xlabel('Position')
            plt.ylabel('Direction (degrees)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(week_out, 'direction_by_position.png'))
            plt.close()

        # --- Categorical/Play Features ---
        # Role and Side might vary by play, so drop duplicates by (nfl_id, play_id)
        df_play_static = df.drop_duplicates(subset=['play_id', 'nfl_id']).copy()
        
        # 7. Role Distribution
        if 'player_role' in df.columns:
            plt.figure(figsize=(16, 8))
            # Create a cross-tabulation
            ct = pd.crosstab(df_play_static['player_position'], df_play_static['player_role'])
            # Normalize to get percentages if desired, or just raw counts. Raw counts shows volume.
            ct.plot(kind='bar', stacked=True, figsize=(16, 8), cmap='viridis')
            plt.title(f'Player Role Distribution by Position ({week_name})')
            plt.xlabel('Position')
            plt.ylabel('Count of Plays')
            plt.xticks(rotation=45)
            plt.legend(title='Player Role', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(week_out, 'role_by_position.png'))
            plt.close()

        # 8. Side Distribution
        if 'player_side' in df.columns:
            plt.figure(figsize=(16, 8))
            ct = pd.crosstab(df_play_static['player_position'], df_play_static['player_side'])
            ct.plot(kind='bar', stacked=True, figsize=(16, 8), cmap='Set2')
            plt.title(f'Player Side Distribution by Position ({week_name})')
            plt.xlabel('Position')
            plt.ylabel('Count of Plays')
            plt.xticks(rotation=45)
            plt.legend(title='Player Side', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(week_out, 'side_by_position.png'))
            plt.close()

        print(f"Finished {week_name}.")

    def run_all(self):
        # Look for input_*.csv files
        files = glob.glob(os.path.join(self.data_dir, "input_*.csv"))
        files.sort()
        
        if not files:
            print("No input files found!")
            return

        print(f"Found {len(files)} week files. Starting processing...")
        for f in files:
            self.analyze_week(f)

if __name__ == "__main__":
    # Define paths
    INPUT_DIR = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train"
    OUTPUT_DIR = "/kaggle/working/weekly_distributions"
    
    analyzer = PositionAnalyzer(INPUT_DIR, OUTPUT_DIR)
    analyzer.run_all()


