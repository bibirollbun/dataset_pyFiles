%pip install pandera


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Circle, Ellipse
import seaborn as sns
import os
import glob
import re
import gc
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from datetime import datetime
import pandera.pandas as pa
from pandera.typing import Series
from IPython.display import HTML
from pydantic import BaseModel
from typing import Optional
from typing import Generator, Tuple, List
from IPython.display import display, HTML
import pandas as pd
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Circle, Ellipse
import os
from IPython.display import Image, display


class DataPipelineConfig(BaseModel):
    DATA_DIR: str = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train"
    SUPP_FILE: str = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv"
    OUTPUT_DIR: str = "/kaggle/working/"

class VisPipelineConfig(BaseModel):
    OUTPUT_DIR: str = "/kaggle/working/visuals"
    TRACKING_FILE: str = "/kaggle/working/master_animation_data.csv"
    SUMMARY_FILE: str = "/kaggle/working/eraser_analysis_summary.csv"

# Default config 
vis_config = VisPipelineConfig()
data_config = DataPipelineConfig()


# helper methods
def print_header(title, subtitle=""):
    html = f"""
    <div style="background-color:#1a1a2e; padding:15px; border-radius:5px; margin-top:30px; margin-bottom:10px; border-left: 5px solid #00ff88;">
        <h3 style="color:white; margin:0; font-family:sans-serif;">{title}</h3>
        <p style="color:#a5b1c2; margin:0; font-size:12px;">{subtitle}</p>
    </div>
    """
    display(HTML(html))


class RawSuppSchema(pa.DataFrameModel):
    """
    Validates 'supplementary_data.csv'.
    Contains Game Context, Play Types
    """

    game_id: Series[int] = pa.Field(coerce=True)
    play_id: Series[int] = pa.Field(coerce=True)
    

    week: Series[int] = pa.Field(coerce=True)
    home_team_abbr: Series[str]
    visitor_team_abbr: Series[str]
    

    down: Series[int] = pa.Field(coerce=True, ge=1, le=4)
    yards_to_go: Series[int] = pa.Field(coerce=True)
    possession_team: Series[str]
    yardline_side: Series[str] = pa.Field(nullable=True)
    yardline_number: Series[int] = pa.Field(ge=0, le=50) 
    defensive_team: Series[str] = pa.Field(nullable=True)
    

    pre_snap_home_team_win_probability: Series[float] = pa.Field(nullable=True)
    pre_snap_visitor_team_win_probability: Series[float] = pa.Field(nullable=True)


    play_nullified_by_penalty: Series[str] = pa.Field(nullable=True)
    dropback_type: Series[str] = pa.Field(nullable=True) 
    team_coverage_man_zone: Series[str] = pa.Field(nullable=True)
    team_coverage_type: Series[str] = pa.Field(nullable=True) 
    pass_result: Series[str] = pa.Field(nullable=True) 
    pass_length: Series[int] = pa.Field(nullable=True)
    route_of_targeted_receiver: Series[str] = pa.Field(nullable=True)
    yards_gained: Series[int] = pa.Field(coerce=True, nullable=True)
    expected_points_added: Series[float] = pa.Field(coerce=True, nullable=True)

    class Config:
        strict = 'filter' 


class RawTrackingSchema(pa.DataFrameModel):
    """
    Validates 'input_wXX.csv' files (Pre-Throw) frame.
    """
    game_id: Series[int] = pa.Field(coerce=True)
    play_id: Series[int] = pa.Field(coerce=True)
    frame_id: Series[int] = pa.Field(coerce=True, ge=1)
    nfl_id: Series[float] = pa.Field(coerce=True, nullable=True) # Nullable for Ball

    play_direction: Series[str] 
    player_name: Series[str]
    absolute_yardline_number: Series[int] = pa.Field(ge=0, le=120, nullable=True)
    
    player_role: Series[str] = pa.Field(nullable=True)
    player_position: Series[str] = pa.Field(nullable=True)
    
    x: Series[float] = pa.Field(ge=0, nullable=True)
    y: Series[float] = pa.Field(ge=0, nullable=True)
    s: Series[float] = pa.Field(ge=0, nullable=True) 
    
    ball_land_x: Series[float] = pa.Field(nullable=True)
    ball_land_y: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = 'filter' 


class OutputTrackingSchema(pa.DataFrameModel):
    """
    Validates 'output_wXX.csv' files (Post-Throw) frame.
    """
    game_id: Series[int] = pa.Field(coerce=True)
    play_id: Series[int] = pa.Field(coerce=True)
    nfl_id: Series[float] = pa.Field(coerce=True, nullable=True)
    frame_id: Series[int] = pa.Field(coerce=True)
    
    x: Series[float] = pa.Field(ge=0, nullable=True)
    y: Series[float] = pa.Field(ge=0, nullable=True)

    class Config:
        strict = 'filter'


class PreprocessedSchema(RawTrackingSchema, RawSuppSchema):
    """
    Validates the output of 'preprocessing.py'.
    """
    phase: Series[str] = pa.Field(isin=["pre_throw", "post_throw"])
    yards_from_own_goal: Series[int] = pa.Field(ge=0, le=100, nullable=True)
    possession_win_prob: Series[float] = pa.Field(ge=0, le=1, nullable=True)

    class Config:
        strict = 'filter'


class PhysicsSchema(PreprocessedSchema):
    """
    Validates the output of 'physics_engine.py'.
    """
    s_derived: Series[float] = pa.Field(nullable=True, coerce=True) # Speed
    a_derived: Series[float] = pa.Field(nullable=True, coerce=True) # Accel
        
    class Config:
        strict = 'filter'


class AggregationScoresSchema(pa.DataFrameModel):
    """
    Validates the 'Score' subset merged onto the animation frames.
    """
    game_id: Series[int] = pa.Field(coerce=True)
    play_id: Series[int] = pa.Field(coerce=True)
    nfl_id: Series[float] = pa.Field(coerce=True)

    dist_at_throw: Series[float] = pa.Field(ge=0, nullable=True)
    void_type: Series[str] = pa.Field(isin=["High Void", "Tight Window", "Neutral"], nullable=True)

    vis_score: Series[float] = pa.Field(nullable=True)  
    ceoe_score: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = 'filter'


class FullPlayAnimationSchema(PhysicsSchema, AggregationScoresSchema): 
    """
    Validates the play animation frames.
    """
    class Config:
        strict = 'filter'


class ContextSchema(pa.DataFrameModel):
    """
    Validates the output of the ContextEngine.
    """
    game_id: Series[int] = pa.Field(coerce=True)
    play_id: Series[int] = pa.Field(coerce=True)
    
    target_nfl_id: Series[float] = pa.Field(nullable=True)
    nearest_def_nfl_id: Series[float] = pa.Field(nullable=True)
    dist_at_throw: Series[float] = pa.Field(ge=0, nullable=True) 
    void_type: Series[str] = pa.Field(isin=["High Void", "Tight Window", "Neutral"], nullable=True)

    class Config:
        strict = 'filter'


class EraserMetricsSchema(pa.DataFrameModel):
    """
    Validates the the output of EraserEngine.
    """
    game_id: Series[int] = pa.Field(coerce=True)
    play_id: Series[int] = pa.Field(coerce=True)
    nfl_id: Series[float] = pa.Field(coerce=True)
    
    p_dist_at_throw: Series[float] = pa.Field(ge=0, nullable=True)
    dist_at_arrival: Series[float] = pa.Field() 
    distance_closed: Series[float] = pa.Field() 
    avg_closing_speed: Series[float] = pa.Field() 
    
    vis_score: Series[float] = pa.Field() 

    class Config:
        strict = 'filter'


class BenchMarkingSchema(pa.DataFrameModel):
    """
    Validates the Player Metadata (Static info) to be attached to the final report.
    """
    game_id: Series[int] = pa.Field(coerce=True)
    play_id: Series[int] = pa.Field(coerce=True)
    nfl_id: Series[float] = pa.Field(coerce=True)
    
    player_role: Series[str] = pa.Field()
    player_name: Series[str] = pa.Field(nullable=True)
    player_position: Series[str] = pa.Field()
    week: Series[int] = pa.Field(coerce=True) 
    down: Series[int] = pa.Field(coerce=True)
    team_coverage_type: Series[str]
    pass_result: Series[str] = pa.Field(nullable=True)
    yards_gained: Series[int] = pa.Field(coerce=True, nullable=True)
    pass_length: Series[int] = pa.Field(coerce=True, nullable=True)
    expected_points_added: Series[float] = pa.Field(coerce=True, nullable=True)

    class Config:
        strict = 'filter'


class AnalysisReportSchema(pa.DataFrameModel):
    """
    Validates the Player Metadata (Static info) to be attached to the final report.
    """
    game_id: Series[int]
    play_id: Series[int]
    nfl_id: Series[float]
    
    player_position: Series[str] = pa.Field(nullable=False)
    player_name: Series[str] = pa.Field(nullable=True)
    player_role: Series[str]
    team_coverage_type: Series[str]
    down: Series[int]
    pass_result: Series[str] = pa.Field(nullable=True)
    dist_at_throw: Series[float] = pa.Field(ge=0, nullable=True)
    dist_at_arrival: Series[float] = pa.Field(ge=0, nullable=True)
    yards_gained: Series[int] = pa.Field(coerce=True, nullable=True)
    pass_length: Series[int] = pa.Field(coerce=True, nullable=True)
    expected_points_added: Series[float] = pa.Field(coerce=True, nullable=True)

    void_type: Series[str] = pa.Field(isin=["High Void", "Tight Window", "Neutral"], nullable=True)
    
    vis_score: Series[float]
    avg_closing_speed: Series[float]
    p_dist_at_throw: Series[float] = pa.Field(ge=0, nullable=True)

    ceoe_score: Series[float] = pa.Field(nullable=False)

    class Config:
        strict = 'filter' 

class EraserEngine:
    def __init__(self):
        self.output_schema = EraserMetricsSchema

    def calculate_eraser(self, df: pd.DataFrame, context_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates how distinct defenders close space on the targeted receiver.
        """
        # Filter for Post-Throw Phase only
        df_post = df[df['phase'] == 'post_throw'].copy()
        
        # Isolate the Targeted Receiver's path
        # We need the receiver's X,Y for every frame to compare against defenders
        targets = df_post[df_post['player_role'] == 'Targeted Receiver'][
            ['game_id', 'play_id', 'frame_id', 'x', 'y']
        ].rename(columns={'x': 't_x', 'y': 't_y'})

        # Isolate Defenders
        defenders = df_post[df_post['player_role'] == 'Defensive Coverage'][
            ['game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y']
        ]

        # Merge Defender + Target on (Game, Play, Frame)
        merged = defenders.merge(targets, on=['game_id', 'play_id', 'frame_id'], how='inner')

        # Calculate Dynamic Separation (Distance to Target)
        merged['dist_to_target'] = np.sqrt(
            (merged['x'] - merged['t_x'])**2 + 
            (merged['y'] - merged['t_y'])**2
        )

        def grade_defender(group):
            group = group.sort_values('frame_id')

            d_start = group['dist_to_target'].iloc[0] # Distance at Throw
            d_end = group['dist_to_target'].iloc[-1]  # Distance at Arrival
            
            # Metric 1: VIS (Void Improvement Score)
            # Positive = Good (Closed gap), Negative = Bad (Lost gap)
            vis = d_start - d_end
            
            # Metric 2: Closing Speed (Rate of Change)
            # Calculate distance change per frame
            # We multiply by -1 because getting closer (dist going down) is positive speed
            dist_change = group['dist_to_target'].diff() * -1
            
            # Convert to Yards/Second (1 frame = 0.1s)
            speeds = dist_change * 10 
            avg_speed = speeds.mean()
            
            return pd.Series({
                'p_dist_at_throw': d_start,
                'dist_at_arrival': d_end,
                'distance_closed': max(0, vis),
                'vis_score': vis,
                'avg_closing_speed': avg_speed
            })

        # Apply grouping per player per play
        metrics = merged.groupby(['game_id', 'play_id', 'nfl_id']).apply(grade_defender).reset_index()

        return self.output_schema.validate(metrics)


class BenchmarkingEngine:
    """
    benchmarking defender performance using context and physics data.
    Calculates CEOE (Closing Efficiency Over Expectation) for each play/defender.
    """
    def __init__(self):
        self.bench_schema = BenchMarkingSchema
        self.report_schema = AnalysisReportSchema

    def calculate_ceoe(self, df_metrics: pd.DataFrame, 
                       df_context: pd.DataFrame, df_physics: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate CEOE (Closing Efficiency Over Expectation) for defenders.
        CEOE = Player's avg closing speed - Positional/contextual average.
        """
        meta_cols = list(self.bench_schema.to_schema().columns.keys())
        df_meta = self.bench_schema.validate(df_physics[meta_cols])
        df_meta = df_meta.drop_duplicates()

        df_final = df_metrics.merge(df_meta, on=['game_id', 'play_id', 'nfl_id'], how='left')
        df_final = df_final.merge(
            df_context[['game_id', 'play_id', 'void_type', 'dist_at_throw']], 
            on=['game_id', 'play_id'], 
            how='left'
        )

        # Calculate positional/contextual average closing speed
        benchmarks = df_final.groupby(
            ['player_position', 'void_type'])['avg_closing_speed'].transform('mean')

        df_final['ceoe_score'] = df_final['avg_closing_speed'] - benchmarks
        df_final['ceoe_score'] = df_final['ceoe_score'].fillna(0.0)
        
        return self.report_schema.validate(df_final)


class DataLoader:
    def __init__(self, data_dir: str, supp_file: str):
        """
        Scans the directory for files but DOES NOT load them yet.
        """
        self.data_dir = data_dir
        self.supp_file = supp_file

        self.input_files = sorted(glob.glob(os.path.join(self.data_dir, 'input_*.csv')))
        self.output_files = glob.glob(os.path.join(self.data_dir, 'output_*.csv'))
        
        self.output_map = {}
        for f in self.output_files:
            match = re.search(r'w(\d{2})', f)
            if not match:
                continue
            self.output_map[match.group(1)] = f

    def load_supplementary(self) -> pd.DataFrame:
        """
        Loads the single Supplementary file.
        """
        if not os.path.exists(self.supp_file):
            raise FileNotFoundError(f"Missing Supp File: {self.supp_file}")
            
        df = pd.read_csv(self.supp_file, low_memory=False)
            
        return RawSuppSchema.validate(df)

    def stream_weeks(self) -> Generator[Tuple[str, pd.DataFrame, pd.DataFrame], None, None]:
        """
        The Lazy Loader.
        Yields: (week_num, input_df, output_df)
        """

        for input_path in self.input_files:

            # Extract Week Number
            match = re.search(r'w(\d{2})', input_path)
            
            if not match: continue
            week_num = match.group(1)
            
            output_path = self.output_map.get(week_num)

            print(f"Streaming Week {week_num}...")
            
            # Load from Disk
            input_raw = pd.read_csv(input_path, low_memory=False)
            output_raw = pd.read_csv(output_path, low_memory=False)
            
            input_raw['nfl_id'] = pd.to_numeric(input_raw['nfl_id'], errors='coerce')
            output_raw['nfl_id'] = pd.to_numeric(output_raw['nfl_id'], errors='coerce')

            # VALIDATE
            input_valid = RawTrackingSchema.validate(input_raw)
            output_valid = OutputTrackingSchema.validate(output_raw)
            
            # Yield the clean, validated data to the Orchestrator
            yield week_num, input_valid, output_valid


class DataExporter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.report_schema = AnalysisReportSchema
        self.animation_schema = AggregationScoresSchema
        self.full_animation = FullPlayAnimationSchema

    def export_results(self, df_summary: pd.DataFrame, df_frames: pd.DataFrame):
        """
        1. Validates & Saves the Analytical Report.
        2. Validates & Merges Scores for Animation.
        3. Saves the Master Animation File.
        """
        print(f"   -> Output Directory: {self.output_dir}")

        self.report_schema.validate(df_summary)

        summary_path = os.path.join(self.output_dir, 'eraser_analysis_summary.csv')
        df_summary.to_csv(summary_path, index=False)
        print(f"   -> Saved Eraser Analysis Report to {summary_path}")

        # Define the subset of columns to attach to the visualizer
        score_cols = list(self.animation_schema.to_schema().columns.keys())
        flags_to_merge = self.animation_schema.validate(df_summary[score_cols])

        # MERGE: Left join the scores onto the massive physics dataframe
        # This repeats the score for every frame of the play
        df_animation = df_frames.merge(
            flags_to_merge, 
            on=['game_id', 'play_id', 'nfl_id'], 
            how='left'
        )
        
        self.full_animation.validate(df_animation)

        final_path = os.path.join(self.output_dir, 'master_animation_data.csv')
        df_animation.to_csv(final_path, index=False)
        
        print(f"   -> Saved Animation Master File to {final_path}")


class PhysicsEngine:
    def __init__(self):
        self.output_schema = PhysicsSchema

    def derive_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Savitzky-Golay filter to calculate generic Speed (s) and Acceleration (a).
        REMOVED: Direction (dir) calculation, as we now use specific vectors in Phase B.
        """
        # Ensure temporal ordering for the filter
        df = df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
        
        # SAVITZKY-GOLAY PARAMETERS
        WINDOW = 7 # 0.7 seconds
        POLY = 2   # Quadratic fit
        
        def calculate_sg(group):

            if len(group) < WINDOW:
                dx = group['x'].diff() 
                dy = group['y'].diff()
                
                dist = np.sqrt(dx**2 + dy**2)
                s = dist / 0.1 
                
                # Acceleration is diff of speed
                a = s.diff().fillna(0) / 0.1
                
                return pd.DataFrame(
                    {'s_derived': s, 'a_derived': a}, 
                    index=group.index)
            
            # First Derivative (Velocity)
            vx = savgol_filter(group['x'], window_length=WINDOW, polyorder=POLY, deriv=1, delta=0.1)
            vy = savgol_filter(group['y'], window_length=WINDOW, polyorder=POLY, deriv=1, delta=0.1)
            
            # Second Derivative (Acceleration)
            ax = savgol_filter(group['x'], window_length=WINDOW, polyorder=POLY, deriv=2, delta=0.1)
            ay = savgol_filter(group['y'], window_length=WINDOW, polyorder=POLY, deriv=2, delta=0.1)
            
            # Magnitudes (Scalar)
            s = np.sqrt(vx**2 + vy**2)
            a = np.sqrt(ax**2 + ay**2)
            
            return pd.DataFrame({
                's_derived': s,
                'a_derived': a
            }, index=group.index)

        # Apply grouping
        # Only apply to players (nfl_id is not null)
        mask_players = df['nfl_id'].notna()
        
        physics_cols = df[mask_players].groupby(
            ['game_id', 'play_id', 'nfl_id'], group_keys=False).apply(calculate_sg, include_groups=False)

        # Map back to original DataFrame
        df.loc[physics_cols.index, 's_derived'] = physics_cols['s_derived']
        df.loc[physics_cols.index, 'a_derived'] = physics_cols['a_derived']

        return self.output_schema.validate(df)

class DataPreProcessor:
    def __init__(self):
        self.output_schema = PreprocessedSchema
        self.keep_cols = list(self.output_schema.to_schema().columns.keys())

    def filter_context(self, supp_df):
        """
        Filters the supplementary dataframe and performs 'Lightweight Feature Engineering'..
        """

        # Calculate Possession Win Probability
        supp_df['possession_win_prob'] = np.where(
            supp_df['possession_team'] == supp_df['home_team_abbr'],
            supp_df['pre_snap_home_team_win_probability'],
            supp_df['pre_snap_visitor_team_win_probability'],
        )
        
        # Calculate Normalized Field Position (0-100 Scale)
        # Logic: If on own side, use number. If on opp side, use 100 - number.
        supp_df['yards_from_own_goal'] = np.where(
            supp_df['yardline_side'] == supp_df['possession_team'],
            supp_df['yardline_number'],           
            100 - supp_df['yardline_number']      
        )

        # valid mask filters
        valid_mask = (
            (supp_df['team_coverage_man_zone'].astype(str).str.contains('Zone', case=False, na=False)) &
            (supp_df['pass_result'].isin(['C', 'I', 'IN'])) &
            (supp_df['team_coverage_type'] != 'COVER_6_ZONE') &
            (~supp_df['dropback_type'].str.upper().isin([
                'SCRAMBLE', 'SCRAMBLE_ROLLOUT_LEFT', 'SCRAMBLE_ROLLOUT_RIGHT', 'QB_DRAW'])) &
            (supp_df['play_nullified_by_penalty'] != 'Y')
        )

        # remove trick / cheap plays
        screen_shovel_mask = (
            supp_df['route_of_targeted_receiver'].astype(str).str.upper().str.contains('SCREEN', na=False) | 
            
            # Physics Check: Ball caught behind or at LOS (Shovels/Swings)
            (supp_df['pass_length'] <= 0) | 
            
            # Check-downs (Flat routes < 3 yards)
            (
                (supp_df['route_of_targeted_receiver'].astype(str).str.upper() == 'FLAT') & 
                (supp_df['pass_length'] < 3)
            )
        )
        
        # base situations
        base_situation_mask = (           
            (supp_df['down'].isin([1, 2])) &
            
            # Competitive Game (Neutral Script)
            (supp_df['possession_win_prob'].between(0.20, 0.80)) & 

            # On Schedule (Not 1st & 20 or 2nd & 18)
            (supp_df['yards_to_go'] <= 10) &
            
            # We use our new engineered column here
            (supp_df['yards_from_own_goal'].between(20, 80))
        )
        
        final_valid_mask = (
            valid_mask &
            (~screen_shovel_mask) &
            base_situation_mask
        )

        return supp_df[final_valid_mask].copy()

    def _stitch_tracking_data(self, input_df, output_df, valid_keys):
        """
        Pure Logic. Merges Pre-Throw and Post-Throw data.
        """
        # Filter Input
        input_df['key_tuple'] = list(zip(input_df.game_id, input_df.play_id))
        input_df = input_df[input_df['key_tuple'].isin(valid_keys)].drop(columns=['key_tuple'])
        input_df['phase'] = 'pre_throw'
        
        if output_df.empty: return input_df

        # Filter Output
        output_df['key_tuple'] = list(zip(output_df.game_id, output_df.play_id))
        output_df = output_df[output_df['key_tuple'].isin(valid_keys)].drop(columns=['key_tuple'])
        
        # Logic: Tag first frame of Output as pass_forward
        output_df['event'] = None
        output_df.loc[output_df['frame_id'] == 1, 'event'] = 'pass_forward'
        
        # TODO: move this to schema
        # Metadata Propagation (Players missing in Output get this from Input)
        meta_cols = ['game_id', 'play_id', 'nfl_id', 'player_name', 'jersey_number', 'player_position', 
                     'player_role', 'player_side', 'play_direction', 'absolute_yardline_number', 
                     'ball_land_x', 'ball_land_y']
        
        avail_cols = [c for c in meta_cols if c in input_df.columns]
        player_meta = input_df[avail_cols].drop_duplicates(subset=['game_id', 'play_id', 'nfl_id'])
        
        # Frame Offset Calculation
        play_offsets = input_df.groupby(['game_id', 'play_id'])['frame_id'].max().reset_index()
        play_offsets.columns = ['game_id', 'play_id', 'offset']
        
        output_df = output_df.merge(player_meta, on=['game_id', 'play_id', 'nfl_id'], how='left')
        output_df = output_df.merge(play_offsets, on=['game_id', 'play_id'], how='left')
        
        # Apply Offset
        output_df['frame_id'] = output_df['frame_id'] + output_df['offset'].fillna(0)
        output_df['phase'] = 'post_throw'
        
        df = pd.concat([input_df, output_df.drop(columns=['offset'])], ignore_index=True)

        return df

    def _normalize_coordinates(self, df):
        """
        Standardizes field geometry to Left->Right drive direction.
        """
        if 'play_direction' not in df.columns: return df
        mask = df['play_direction'].str.lower() == 'left'
        
        for col in ['x', 'ball_land_x']:
            if col in df.columns: df.loc[mask, col] = 120 - df.loc[mask, col]

        for col in ['y', 'ball_land_y']:
            if col in df.columns: df.loc[mask, col] = 53.3 - df.loc[mask, col]

        return df

    def _clean_and_deduplicate(self, df):
        """
        Ensures strict temporal ordering and removes duplicate frames at the stitch point.
        """
        df['phase_rank'] = df['phase'].apply(lambda x: 1 if x == 'pre_throw' else 2)        
        
        df = df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id', 'phase_rank'])        
        
        df = df.drop_duplicates(subset=['game_id', 'play_id', 'nfl_id', 'frame_id'], keep='last')
        
        return df.drop(columns=['phase_rank'])

    def process_single_week(self, week_num, input_df, output_df, context_df):
        """
        Internal logic for a single week.
        """
        valid_keys = set(zip(context_df.game_id, context_df.play_id))

        week_df = self._stitch_tracking_data(input_df, output_df, valid_keys)

        week_df = week_df.merge(context_df, on=['game_id', 'play_id'], how='inner')

        week_df = self._normalize_coordinates(week_df)

        week_df['los_x'] = week_df['ball_land_x'] - week_df['pass_length']
        week_df['week'] = int(week_num)

        week_df = self._clean_and_deduplicate(week_df)

        return self.output_schema.validate(week_df)

    def run(self, data_stream: Generator[Tuple[str, pd.DataFrame, pd.DataFrame], None, None], 
            raw_context_df: pd.DataFrame) -> pd.DataFrame:
        """
        MAIN ENTRY POINT.
        """
        clean_context = self.filter_context(raw_context_df)
        
        processed_chunks: List[pd.DataFrame] = []
        for week_num, input_df, output_df in data_stream:
            
            clean_week_df = self.process_single_week(week_num, input_df, output_df, clean_context)

            if not clean_week_df.empty:
                processed_chunks.append(clean_week_df)

            del input_df, output_df
            gc.collect()

        if not processed_chunks:
            return pd.DataFrame()
        
        return pd.concat(processed_chunks, ignore_index=True)

class ContextEngine:
    def __init__(self):
        self.output_schema = ContextSchema

    def calculate_void_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        capture ALL players at the moment of the throw.
        """
        # Filter for Pre-Throw phase
        pre_throw_mask = df['phase'] == 'pre_throw'
        df_pre = df[pre_throw_mask].copy()

        # we calculate the MAX frame_id for every play
        last_frame_ids = df_pre.groupby(['game_id', 'play_id'])['frame_id'].transform('max')
        
        # We keep rows where the frame_id matches the last frame of that play
        throw_frames = df_pre[df_pre['frame_id'] == last_frame_ids].copy()

        # Split into Target vs. Defenders
        targets = throw_frames[throw_frames['player_role'].astype(str).str.strip() == 'Targeted Receiver'][
            ['game_id', 'play_id', 'nfl_id', 'x', 'y', 'week']
        ].rename(columns={'nfl_id': 'target_nfl_id', 'x': 't_x', 'y': 't_y'})

        defenders = throw_frames[throw_frames['player_role'].astype(str).str.strip() == 'Defensive Coverage'][
            ['game_id', 'play_id', 'nfl_id', 'x', 'y']
        ].rename(columns={'nfl_id': 'def_nfl_id', 'x': 'd_x', 'y': 'd_y'})

        merged = defenders.merge(targets, on=['game_id', 'play_id'], how='inner')

        merged['dist'] = np.sqrt(
            (merged['d_x'] - merged['t_x'])**2 + 
            (merged['d_y'] - merged['t_y'])**2
        )

        # Find the Nearest Neighbor
        min_dists = merged.loc[merged.groupby(['game_id', 'play_id'])['dist'].idxmin()]

        context_df = min_dists[['game_id', 'play_id', 'week', 'target_nfl_id', 'def_nfl_id', 'dist']].copy()
        
        context_df = context_df.rename(columns={
            'def_nfl_id': 'nearest_def_nfl_id', 
            'dist': 'dist_at_throw'
        })

        # Apply Classification Labels
        conditions = [
            (context_df['dist_at_throw'] > 5.0),
            (context_df['dist_at_throw'] < 2.0)
        ]
        choices = ['High Void', 'Tight Window']
        context_df['void_type'] = np.select(conditions, choices, default='Neutral')

        return self.output_schema.validate(context_df)

def run_data_pipeline(DATA_DIR=None, SUPP_FILE=None, OUTPUT_DIR=None):
    start_time = datetime.now()

    # Use provided arguments, else fall back to config.py values
    cfg = DataPipelineConfig(
        DATA_DIR=DATA_DIR or data_config.DATA_DIR,
        SUPP_FILE=SUPP_FILE or data_config.SUPP_FILE,
        OUTPUT_DIR=OUTPUT_DIR or data_config.OUTPUT_DIR
    )

    # 1. LOAD
    print(f"[1/7] Initializing Data Loader ({datetime.now().strftime('%H:%M:%S')})...")
    loader = DataLoader(cfg.DATA_DIR, cfg.SUPP_FILE)
    raw_supp = loader.load_supplementary()
    raw_tracking = loader.stream_weeks()

    # 2. PREPROCESS
    print("[2/7] Preprocessing & Stitching frames...")
    processor = DataPreProcessor()
    df_clean = processor.run(data_stream=raw_tracking, raw_context_df=raw_supp)

    # 3. PHYSICS
    print("[3/7] Running Physics Engine (Kinematics)...")
    physics_engine = PhysicsEngine()
    df_physics = physics_engine.derive_metrics(df_clean)
    
    del df_clean
    gc.collect() 

    # 4. CONTEXT
    # TODO: Note this is changing the dataframe entirely - df_physics is our animation dataset.
    print("[4/7] Phase A: Calculating Void Context (S_throw)...")
    context_engine = ContextEngine()
    df_context = context_engine.calculate_void_context(df_physics)
    
    # Debugging
    print(f"   -> Identified Voids for {df_context.shape[0]} plays.")

    # 5. ERASER
    print("[5/7] Phase B: Calculating Eraser Metrics (VIS)...")
    eraser_engine = EraserEngine()
    df_metrics = eraser_engine.calculate_eraser(df_physics, df_context)

    # 6. BENCHMARKING
    print("[6/7] Phase C: Benchmarking (CEOE)...")
    benchmarker = BenchmarkingEngine()
    df_final = benchmarker.calculate_ceoe(
        df_metrics=df_metrics, 
        df_context=df_context, 
        df_physics=df_physics
    )

    # 7. EXPORT
    print("[7/7] Phase D: Exporting Results...")
    exporter = DataExporter(cfg.OUTPUT_DIR)
    exporter.export_results(
        df_summary=df_final, 
        df_frames=df_physics
    )
    
    duration = datetime.now() - start_time
    print(f"PIPELINE FINISHED in {duration}")



# Execute Pipeline
run_data_pipeline(
    DATA_DIR=data_config.DATA_DIR, 
    SUPP_FILE=data_config.SUPP_FILE, 
    OUTPUT_DIR=data_config.OUTPUT_DIR
)


NFL_TEAM_COLORS = {
    'BAL': {'primary': '#241773', 'secondary': '#000000', 'alternate': '#9E7C0C'},
    'CIN': {'primary': '#FB4F14', 'secondary': '#000000'},
    'CLE': {'primary': '#311D00', 'secondary': '#FF3C00'},
    'PIT': {'primary': '#FFB612', 'secondary': '#101820', 'alternate': '#003087'},
    'BUF': {'primary': '#00338D', 'secondary': '#C60C30'},
    'MIA': {'primary': '#008E97', 'secondary': '#FC4C02', 'alternate': '#005778'},
    'NE': {'primary': '#002244', 'secondary': '#C60C30', 'alternate': '#B0B7BC'},
    'NYJ': {'primary': '#125740', 'secondary': '#000000', 'alternate': '#FFFFFF'},
    'HOU': {'primary': '#03202F', 'secondary': '#A71930'},
    'IND': {'primary': '#002C5F', 'secondary': '#A2AAAD'},
    'JAX': {'primary': '#101820', 'secondary': '#D7A22A', 'alternate': '#9F792C'},
    'TEN': {'primary': '#0C2340', 'secondary': '#4B92DB', 'alternate': '#C8102E'},
    'DEN': {'primary': '#FB4F14', 'secondary': '#002244'},
    'KC': {'primary': '#E31837', 'secondary': '#FFB81C'},
    'LV': {'primary': '#000000', 'secondary': '#A5ACAF'},
    'LAC': {'primary': '#0080C6', 'secondary': '#FFC20E', 'alternate': '#FFFFFF'},
    'CHI': {'primary': '#0B162A', 'secondary': '#C83803'},
    'DET': {'primary': '#0076B6', 'secondary': '#B0B7BC', 'alternate': '#000000'},
    'GB': {'primary': '#203731', 'secondary': '#FFB612'},
    'MIN': {'primary': '#4F2683', 'secondary': '#FFC62F'},
    'DAL': {'primary': '#003594', 'secondary': '#041E42', 'alternate': '#869397'},
    'NYG': {'primary': '#0B2265', 'secondary': '#A71930', 'alternate': '#A5ACAF'},
    'PHI': {'primary': '#004C54', 'secondary': '#A5ACAF', 'alternate': '#000000'},
    'WAS': {'primary': '#5A1414', 'secondary': '#FFB612'},
    'ATL': {'primary': '#A71930', 'secondary': '#000000', 'alternate': '#A5ACAF'},
    'CAR': {'primary': '#0085CA', 'secondary': '#101820', 'alternate': '#BFC0BF'},
    'NO': {'primary': '#D3BC8D', 'secondary': '#101820'},
    'TB': {'primary': '#D50A0A', 'secondary': '#FF7900', 'alternate': '#B1BABF'},
    'ARI': {'primary': '#97233F', 'secondary': '#000000', 'alternate': '#FFB612'},
    'LAR': {'primary': '#003594', 'secondary': '#FFA300', 'alternate': '#FF8200'},
    'SF': {'primary': '#AA0000', 'secondary': '#B3995D'},
    'SEA': {'primary': '#002244', 'secondary': '#69BE28', 'alternate': '#A5ACAF'}
}

class AnimationEngine:
    def __init__(self, summary_df, frames_df, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
       
        self.summary_df = summary_df
        self.frames_df = frames_df

    def _draw_field(self, ax):
        """Sets up the static NFL-style field background with enhanced details."""
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 53.3)
        ax.set_facecolor('#2e7d32')  
        ax.set_aspect('equal')
        ax.axis('off')
       
        # Field outline with thicker sidelines
        field_rect = patches.Rectangle((10, 0), 100, 53.3, linewidth=3,
                                         edgecolor='white', facecolor='#3d8b40', zorder=0)
        ax.add_patch(field_rect)
       
        # End Zones with diagonal stripes for texture
        ez_left = patches.Rectangle((0, 0), 10, 53.3, facecolor='#1b5e20', edgecolor='white', linewidth=2, zorder=0)
        ez_right = patches.Rectangle((110, 0), 10, 53.3, facecolor='#1b5e20', edgecolor='white', linewidth=2, zorder=0)
        ax.add_patch(ez_left)
        ax.add_patch(ez_right)
        # Add subtle diagonal lines in end zones for NFL feel
        for y in np.arange(0, 53.3, 2):
            ax.plot([0, 10], [y, y+2], color='white', alpha=0.1, lw=1)
            ax.plot([110, 120], [y, y+2], color='white', alpha=0.1, lw=1)
        ax.text(5, 26.65, 'END\nZONE', ha='center', va='center', fontsize=8,
                fontweight='bold', color='white', alpha=0.6, rotation=90)
        ax.text(115, 26.65, 'END\nZONE', ha='center', va='center', fontsize=8,
                fontweight='bold', color='white', alpha=0.6, rotation=270)
       
        # Yard lines (every 5 yards, bold every 10)
        for x in range(10, 111, 5):
            lw = 3 if x % 10 == 0 else 1
            alpha = 1.0 if x % 10 == 0 else 0.7
            ax.axvline(x, color='white', linestyle='-', linewidth=lw, alpha=alpha, zorder=1)
       
        # Yard numbers with rotation for sideline view
        for x in range(20, 110, 10):
            num = (x - 10) if x <= 60 else (110 - x)
            # Top numbers (rotated)
            ax.text(x, 48, str(num), color='white', ha='center', va='center',
                    fontsize=14, fontweight='bold', alpha=0.8, rotation=180)
            # Bottom numbers
            ax.text(x, 5, str(num), color='white', ha='center', va='center',
                    fontsize=14, fontweight='bold', alpha=0.8)
       
        # Hash marks (NFL style, more accurate spacing)
        hash_y_top = 39.3  
        hash_y_bot = 14.0
        for x in range(10, 111, 1):
            ax.plot([x, x], [hash_y_top, hash_y_top + 0.5], color='white', linewidth=1, alpha=0.8, zorder=1)
            ax.plot([x, x], [hash_y_bot - 0.5, hash_y_bot], color='white', linewidth=1, alpha=0.8, zorder=1)
       
        # Add NFL logo at center for elite feel
        ax.text(60, 26.65, 'NFL', ha='center', va='center', fontsize=24, fontweight='bold',
                color='white', alpha=0.15)
        # Goal posts (simplified Y-shape projection)
        gp_color = '#FFD700'  
       
        # Left goal post
        ax.plot([0, 0], [23.65, 29.65], color=gp_color, lw=4)  
        ax.plot([-1, 1], [26.65, 26.65], color=gp_color, lw=4)
       
        # Right goal post
        ax.plot([120, 120], [23.65, 29.65], color=gp_color, lw=4)
        ax.plot([119, 121], [26.65, 26.65], color=gp_color, lw=4)

    def generate_video(self, game_id, play_id, eraser_id, filename="play_animation.mp4"):
        print(f"   [Animator] Rendering video for {game_id}-{play_id}...")
       
        # Get Play Data
        play_frames = self.frames_df[
            (self.frames_df['game_id'] == game_id) &
            (self.frames_df['play_id'] == play_id)
        ].sort_values('frame_id')
       
        if play_frames.empty: return

        # Actors & Context
        target_row = play_frames[play_frames['player_role'] == 'Targeted Receiver']
        target_id = target_row['nfl_id'].iloc[0]
        target_name = target_row['player_name'].iloc[0] if 'player_name' in target_row.columns else "WR"
       
        # Get QB info
        qb_row = play_frames[play_frames['player_role'] == 'Passer']
        qb_id = qb_row['nfl_id'].iloc[0] if not qb_row.empty else None
        qb_name = qb_row['player_name'].iloc[0] if not qb_row.empty and 'player_name' in qb_row.columns else "QB"
       
        summary_row = self.summary_df[
            (self.summary_df['game_id'] == game_id) &
            (self.summary_df['play_id'] == play_id)
        ]
       
        # Get Scores & Names
        vis_score = 0.0
        start_dist = 0.0
        end_dist = 0.0
        context_id = None
        eraser_name = "DEF"
        context_name = "DEF"
       
        if not summary_row.empty:
            eraser_row = summary_row[summary_row['nfl_id'] == eraser_id]
           
            if not eraser_row.empty:
                vis_score = eraser_row.iloc[0]['vis_score']
                start_dist = eraser_row.iloc[0]['p_dist_at_throw']
                end_dist = eraser_row.iloc[0].get('p_dist_at_arrival', start_dist - vis_score)
               
                if 'player_name' in eraser_row.columns:
                    eraser_name = eraser_row.iloc[0]['player_name']
           
            # Get context defender (closest at throw)
            context_idx = summary_row['p_dist_at_throw'].idxmin()
            context_id = summary_row.loc[context_idx]['nfl_id']
            if 'player_name' in summary_row.columns:
                context_name = summary_row.loc[context_idx]['player_name']

        # Get play metadata
        meta = play_frames.iloc[0]
        pass_result = meta.get('pass_result', 'Unknown')
        yards_gained = meta.get('yards_gained', 0)
        off_team = meta['possession_team']
        def_team = meta['defensive_team']
       
        # Team colors
        off_color = NFL_TEAM_COLORS.get(off_team, {'primary': '#c0392b'})['primary']
        def_color = NFL_TEAM_COLORS.get(def_team, {'primary': '#2980b9'})['primary']
        off_secondary = NFL_TEAM_COLORS.get(off_team, {'secondary': '#ffffff'}).get('secondary', '#ffffff')
        def_secondary = NFL_TEAM_COLORS.get(def_team, {'secondary': '#ffffff'}).get('secondary', '#ffffff')
       
        # BALL TRAJECTORY LOGIC
        unique_frames = sorted(play_frames['frame_id'].unique())
        start_frame = play_frames['frame_id'].min()
        end_frame = play_frames['frame_id'].max()
       
        # Identify the throw frame (first frame of post_throw phase)
        if 'phase' in play_frames.columns:
            post_throw_data = play_frames[play_frames['phase'] == 'post_throw']
            if not post_throw_data.empty:
                throw_frame = post_throw_data['frame_id'].min()
            else:
                throw_frame = start_frame
        else:
            throw_frame = start_frame
       
        # Pause at throw frame for emphasis (repeat frame 5 times at 10 fps ~0.5s pause)
        frames_list = []
        for f in unique_frames:
            frames_list.append(f)
            if f == throw_frame:
                frames_list.extend([f] * 4)  # Add pause
       
        # Get post-throw frames for ball interpolation
        post_throw_frames = sorted([f for f in unique_frames if f >= throw_frame])
        post_throw_steps = len(post_throw_frames)
       
        # Ball landing position
        bx_end, by_end = play_frames.iloc[0]['ball_land_x'], play_frames.iloc[0]['ball_land_y']
       
        # Get QB position at throw frame
        passer_at_throw = play_frames[
            (play_frames['frame_id'] == throw_frame) &
            (play_frames['player_role'] == 'Passer')
        ]
        if not passer_at_throw.empty:
            bx_throw, by_throw = passer_at_throw.iloc[0]['x'], passer_at_throw.iloc[0]['y']
        else:
            passer_early = play_frames[play_frames['player_role'] == 'Passer'].sort_values('frame_id')
            if not passer_early.empty:
                bx_throw, by_throw = passer_early.iloc[0]['x'], passer_early.iloc[0]['y']
            else:
                bx_throw, by_throw = bx_end - 15, 26.65
       
        # Interpolate ball trajectory ONLY for post-throw frames (slight curve for visual elite feel)
        if post_throw_steps > 1:
            t = np.linspace(0, 1, post_throw_steps)
            ball_x_flight = bx_throw + (bx_end - bx_throw) * t
            ball_y_flight = by_throw + (by_end - by_throw) * t + np.sin(t * np.pi) * 1.5  # Slight arc for aesthetics
        else:
            ball_x_flight = [bx_end]
            ball_y_flight = [by_end]
       
        # Build ball position dictionary
        ball_pos_dict = {}
        for f in unique_frames:
            if f < throw_frame:
                passer_at_f = play_frames[
                    (play_frames['frame_id'] == f) &
                    (play_frames['player_role'] == 'Passer')
                ]
                if not passer_at_f.empty:
                    ball_pos_dict[f] = (passer_at_f.iloc[0]['x'], passer_at_f.iloc[0]['y'])
                else:
                    ball_pos_dict[f] = (bx_throw, by_throw)
            else:
                idx = post_throw_frames.index(f)
                ball_pos_dict[f] = (ball_x_flight[idx], ball_y_flight[idx])

        # Cache last known position for each player to handle "ghost" players
        # who disappear in post_throw frames (>8yds from catch point in output data)
        all_player_ids = play_frames['nfl_id'].unique()
        player_cache = {}
       
        # Pre-build cache by iterating through frames in order
        player_positions_by_frame = {}
        for f in unique_frames:
            snap = play_frames[play_frames['frame_id'] == f]
            frame_positions = {}
           
            for pid in all_player_ids:
                player_row = snap[snap['nfl_id'] == pid]
                if not player_row.empty:
                   
                    # Update cache with current position
                    player_cache[pid] = {
                        'x': player_row.iloc[0]['x'],
                        'y': player_row.iloc[0]['y'],
                        'role': player_row.iloc[0]['player_role'],
                        's_derived': player_row.iloc[0].get('s_derived', 0),
                        'phase': player_row.iloc[0]['phase']
                    }
               
                # Use cached position (current or last known)
                if pid in player_cache:
                    frame_positions[pid] = player_cache[pid].copy()
           
            player_positions_by_frame[f] = frame_positions

        # Setup Figure
        fig, ax = plt.subplots(figsize=(14, 7))
        self._draw_field(ax)
       
        # Context Box (Top Left) - Teams, Down & Distance, Coverage
        yd_str = f"{meta['yardline_side']} {int(meta['yardline_number'])}"
        cov_str = str(meta['team_coverage_type']).replace('_', ' ').title()
        context_text = f"{off_team} vs {def_team}\n{int(meta['down'])} & {int(meta['yards_to_go'])} | {yd_str}\n{cov_str}"
        at_context = AnchoredText(context_text, loc='upper left',
                                  prop=dict(size=11, fontweight='bold', family='monospace'), frameon=True)
        at_context.patch.set_boxstyle("round,pad=0.4")
        at_context.patch.set_facecolor('#1a1a2e')
        at_context.patch.set_edgecolor('white')
        at_context.patch.set_alpha(0.95)
       
        for txt in at_context.txt.get_children():
            txt.set_color('white')
       
        ax.add_artist(at_context)
       
        # Metric Box (Top Center) - VIS Score with start/end
        sign = "+" if vis_score > 0 else ""
        metric_text = f"START: {start_dist:.1f} yds\nVIS: {sign}{vis_score:.1f} yds\nEND: {end_dist:.1f} yds"
        at_metric = AnchoredText(metric_text, loc='upper center',
                                 prop=dict(size=12, fontweight='bold', family='monospace'), frameon=True)
       
        at_metric.patch.set_boxstyle("round,pad=0.4")
        at_metric.patch.set_facecolor('#16213e')
        at_metric.patch.set_edgecolor('#00ff88' if vis_score > 0 else '#ff4444')
        at_metric.patch.set_linewidth(3)
       
        for txt in at_metric.txt.get_children():
            txt.set_color('#00ff88' if vis_score > 0 else '#ff4444')
       
        ax.add_artist(at_metric)
       
        # Outcome Badge (Top Right area) - shows COMPLETE/INCOMPLETE with yards
        outcome_color = '#27ae60' if pass_result == 'C' else '#c0392b'
        outcome_text = f'COMPLETE âœ“ +{yards_gained} yds' if pass_result == 'C' else 'INCOMPLETE âœ—'
        outcome_label = ax.text(118, 50, outcome_text, ha='right', va='top', fontsize=11,
                                fontweight='bold', color='white',
                                bbox=dict(facecolor=outcome_color, edgecolor='white',
                                          pad=6, boxstyle='round,pad=0.4'))
       
        # Timer (Right side)
        timer_text = ax.text(118, 42, '', ha='right', fontsize=13, fontweight='bold', color='white',
                             bbox=dict(facecolor='#2c3e50', edgecolor='white', pad=4, boxstyle='round,pad=0.3'))
       
        # Phase Label (dynamic color based on phase)
        phase_label = ax.text(118, 35, '', ha='right', fontsize=11, fontweight='bold', color='white',
                              bbox=dict(facecolor='#3498db', edgecolor='white', pad=5, boxstyle='round,pad=0.3'))
        phase_colors = {
            'pre_throw': '#3498db',  # Blue
            'post_throw': '#e74c3c',  # Red
            'pre_snap': '#95a5a6',  # Gray
            'post_snap': '#f39c12',  # Orange
            'unknown': '#7f8c8d'
        }
       
        # Speed indicator for eraser
        speed_label = ax.text(118, 28, '', ha='right', fontsize=10, fontweight='bold', color='white',
                              bbox=dict(facecolor='#8e44ad', edgecolor='white', pad=4, boxstyle='round,pad=0.3'))
       
        # Ball Landing Spot (X marker - visible entire animation, with glow)
        ax.scatter([bx_end], [by_end], c='#ffff00', s=400, marker='X',
                   edgecolors='#ff6600', linewidths=3, zorder=2, alpha=0.9)
       
        # Add subtle circle for target area
        target_circle = Circle((bx_end, by_end), 1.5, color='#ffff00', alpha=0.2, zorder=1)
        ax.add_patch(target_circle)
        ax.text(bx_end, by_end - 2.5, 'TARGET', ha='center', va='top', fontsize=7,
                fontweight='bold', color='#ffff00', alpha=0.9)
       
        # Other defenders (team color circles with shadow)
        scat_def_others = ax.scatter([], [], c=def_color, s=200, marker='o',
                                     edgecolors=def_secondary, linewidths=2, zorder=3, alpha=0.7)
       
        # Shadow for depth
        scat_def_shadow = ax.scatter([], [], c='black', s=200, marker='o', alpha=0.2, zorder=2)
       
       
        # Other offense (team color circles with shadow)
        scat_off_others = ax.scatter([], [], c=off_color, s=200, marker='o',
                                     edgecolors=off_secondary, linewidths=2, zorder=3, alpha=0.7)
        scat_off_shadow = ax.scatter([], [], c='black', s=200, marker='o', alpha=0.2, zorder=2)
       
        # QB (diamond - distinctive, with team accent)
        qb_color = off_color  # Use offense primary
        scat_qb = ax.scatter([], [], c=qb_color, s=350, marker='D',
                              edgecolors=off_secondary, linewidths=3, zorder=5)
       
        scat_qb_shadow = ax.scatter([], [], c='black', s=350, marker='D', alpha=0.2, zorder=4)
        qb_label = ax.text(0, 0, '', ha='center', va='bottom', fontsize=8,
                           fontweight='bold', color='white',
                           bbox=dict(facecolor=qb_color, edgecolor='none', pad=2, alpha=0.9))
       
        # Target Receiver (star - stands out, with glow)
        target_color = '#ff6b35'  # Keep orange for highlight
        scat_target = ax.scatter([], [], c=target_color, s=450, marker='*',
                                  edgecolors='white', linewidths=2, zorder=6)
        scat_target_glow = ax.scatter([], [], c=target_color, s=600, marker='*', alpha=0.3, zorder=5)
        target_label = ax.text(0, 0, '', ha='center', va='bottom', fontsize=8,
                               fontweight='bold', color='white',
                               bbox=dict(facecolor=target_color, edgecolor='none', pad=2, alpha=0.9))
       
        # Eraser (bright green, largest circle with dark edge)
        eraser_color = '#00ff88'
        scat_eraser = ax.scatter([], [], c=eraser_color, s=400, marker='o',
                                  edgecolors='#004d40', linewidths=3, zorder=7)
        scat_eraser_glow = ax.scatter([], [], c=eraser_color, s=550, marker='o', alpha=0.3, zorder=6)
        eraser_label = ax.text(0, 0, '', ha='center', va='bottom', fontsize=8,
                               fontweight='bold', color='black',
                               bbox=dict(facecolor=eraser_color, edgecolor='none', pad=2, alpha=0.9))
       
        # Context Defender (triangle - closest at throw)
        context_color = '#00bcd4'
        scat_context = ax.scatter([], [], c=context_color, s=300, marker='^',
                                   edgecolors='white', linewidths=2, zorder=5)
        scat_context_shadow = ax.scatter([], [], c='black', s=300, marker='^', alpha=0.2, zorder=4)
        context_label = ax.text(0, 0, '', ha='center', va='bottom', fontsize=8,
                                fontweight='bold', color='white',
                                bbox=dict(facecolor=context_color, edgecolor='none', pad=2, alpha=0.9))
       
        # Ball (football-shaped ellipse for more realistic NFL look)
        ball_color = '#8B4513'
        football_patch = Ellipse((0, 0), width=1.2, height=0.5, color=ball_color, edgecolor='white', linewidth=2, zorder=10)
        ax.add_patch(football_patch)
        # No rotation for simplicity, but could add based on velocity if desired
       
        # Void line (dashed line between eraser and target with distance)
        line_void, = ax.plot([], [], color='#ffff00', linestyle='--', linewidth=2, alpha=0.8, zorder=4)
        text_void = ax.text(0, 0, '', ha='center', va='center', fontsize=10, fontweight='bold',
                            color='black', bbox=dict(facecolor='#ffff00', alpha=0.9, edgecolor='none', pad=3))
       
        trail_length = 30 # TODO: Adjust
        target_history = []
        eraser_history = []
        context_history = []
        qb_history = []
        ball_history = []
       
        target_trail, = ax.plot([], [], color=target_color, lw=3, alpha=0.5, zorder=2, label='Target Path')
        eraser_trail, = ax.plot([], [], color=eraser_color, lw=3, alpha=0.5, zorder=2, label='Eraser Path')
        context_trail, = ax.plot([], [], color=context_color, lw=2, alpha=0.5, zorder=2, label='Context Path')
        qb_trail, = ax.plot([], [], color=qb_color, lw=2, alpha=0.5, zorder=2, label='QB Path')
        ball_trail, = ax.plot([], [], color=ball_color, lw=1.5, linestyle='-', alpha=0.4, zorder=1, label='Ball Path')
       
        legend_elements = [
            plt.scatter([], [], c=eraser_color, s=150, marker='o', edgecolors='#004d40', linewidths=2, label='Eraser'),
            plt.scatter([], [], c=target_color, s=150, marker='*', edgecolors='white', linewidths=2, label='Target WR'),
            plt.scatter([], [], c=context_color, s=100, marker='^', edgecolors='white', linewidths=2, label='Nearest Def'),
            plt.scatter([], [], c=qb_color, s=100, marker='D', edgecolors='white', linewidths=2, label='QB'),
            patches.Ellipse((0,0), 1, 0.5, color=ball_color, label='Ball'),
            plt.scatter([], [], c='#ffff00', s=100, marker='X', edgecolors='#ff6600', linewidths=2, label='Ball Target'),
            plt.Line2D([], [], color='gray', lw=2, alpha=0.5, label='Player Trails')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
                  framealpha=0.95, facecolor='#1a1a2e', edgecolor='white', labelcolor='white',
                  ncol=2)

        # Update Loop
        def update(frame_num):
            # Use actual frame_id (since frames_list may have duplicates)
            actual_frame = frame_num  # frames_list[frame_idx] but since ani uses range(len(frames_list))
           
            # Wait, ani = FuncAnimation(..., frames=frames_list, ...) so frame_num is the value from frames_list, i.e., the frame_id
            f = frame_num  
           
            # Timer relative to throw frame
            time_sec = (f - throw_frame) * 0.1
            if f < throw_frame:
                timer_text.set_text(f"T {time_sec:.1f}s")
                phase_label.set_text("â�³ PRE THROW")
                phase_label.get_bbox_patch().set_facecolor('#3498db')
            else:
                timer_text.set_text(f"T +{time_sec:.1f}s")
                phase_label.set_text("ğŸ�ˆ BALL IN AIR")
                phase_label.get_bbox_patch().set_facecolor('#e74c3c')
           
            # Ball position
            if f in ball_pos_dict:
                bx, by = ball_pos_dict[f]
                football_patch.set_center((bx, by))
                ball_history.append((bx, by))
                if len(ball_history) > trail_length:
                    ball_history.pop(0)
                ball_trail.set_data(*zip(*ball_history))
           
            # Build position lists from cache
            def_others_pos = []
            off_others_pos = []
            def_shadow_pos = []
            off_shadow_pos = []
           
            excluded_ids = {eraser_id, target_id, context_id}
            if qb_id:
                excluded_ids.add(qb_id)
           
            frame_positions = player_positions_by_frame.get(f, {})
            for pid, pos in frame_positions.items():
                if pid in excluded_ids:
                    continue
                role = pos.get('role', '')
                px, py = pos['x'], pos['y']
                if role in ['Coverage Defender', 'Pass Rusher', 'Defender']:
                    def_others_pos.append([px, py])
                    def_shadow_pos.append([px + 0.2, py - 0.2])  # Offset shadow
                elif role not in ['Passer']:
                    off_others_pos.append([px, py])
                    off_shadow_pos.append([px + 0.2, py - 0.2])
           
            scat_def_others.set_offsets(np.array(def_others_pos) if def_others_pos else np.empty((0, 2)))
            scat_def_shadow.set_offsets(np.array(def_shadow_pos) if def_shadow_pos else np.empty((0, 2)))
            scat_off_others.set_offsets(np.array(off_others_pos) if off_others_pos else np.empty((0, 2)))
            scat_off_shadow.set_offsets(np.array(off_shadow_pos) if off_shadow_pos else np.empty((0, 2)))
           
            # QB (from cache)
            if qb_id and qb_id in frame_positions:
                qb_pos = frame_positions[qb_id]
                qx, qy = qb_pos['x'], qb_pos['y']
                scat_qb.set_offsets([[qx, qy]])
                scat_qb_shadow.set_offsets([[qx + 0.2, qy - 0.2]])
                qb_label.set_position((qx, qy + 2.5))
                q_name = str(qb_name).split()[-1][:8] if qb_name else "QB"
                qb_label.set_text(q_name)
                qb_history.append((qx, qy))
                if len(qb_history) > trail_length:
                    qb_history.pop(0)
                qb_trail.set_data(*zip(*qb_history))
            else:
                scat_qb.set_offsets(np.empty((0, 2)))
                scat_qb_shadow.set_offsets(np.empty((0, 2)))
           
            # Target (from cache)
            if target_id in frame_positions:
                target_pos = frame_positions[target_id]
                tx, ty = target_pos['x'], target_pos['y']
                scat_target.set_offsets([[tx, ty]])
                scat_target_glow.set_offsets([[tx, ty]])
                target_label.set_position((tx, ty + 2.5))
                t_name = str(target_name).split()[-1][:8] if target_name else "WR"
                target_label.set_text(t_name)
                target_history.append((tx, ty))
                if len(target_history) > trail_length:
                    target_history.pop(0)
                target_trail.set_data(*zip(*target_history))
           
            # Context defender (from cache, with name label)
            if context_id and context_id in frame_positions and context_id != eraser_id:
                context_pos = frame_positions[context_id]
                cx, cy = context_pos['x'], context_pos['y']
                scat_context.set_offsets([[cx, cy]])
                scat_context_shadow.set_offsets([[cx + 0.2, cy - 0.2]])
                context_label.set_position((cx, cy + 2.5))
                c_name = str(context_name).split()[-1][:8] if context_name else "DEF"
                context_label.set_text(c_name)
                context_history.append((cx, cy))
                if len(context_history) > trail_length:
                    context_history.pop(0)
                context_trail.set_data(*zip(*context_history))
            else:
                scat_context.set_offsets(np.empty((0, 2)))
                scat_context_shadow.set_offsets(np.empty((0, 2)))
                context_label.set_text('')
               
            # Eraser (from cache)
            if eraser_id in frame_positions:
                eraser_pos = frame_positions[eraser_id]
                ex, ey = eraser_pos['x'], eraser_pos['y']
                scat_eraser.set_offsets([[ex, ey]])
                scat_eraser_glow.set_offsets([[ex, ey]])
                eraser_label.set_position((ex, ey + 2.5))
                e_name = str(eraser_name).split()[-1][:8] if eraser_name else "DEF"
                eraser_label.set_text(e_name)
               
                # Speed display
                speed_yps = eraser_pos.get('s_derived', 0)
                speed_mph = speed_yps * 2.045 if pd.notna(speed_yps) else 0
                speed_label.set_text(f"âš¡ {speed_mph:.1f} mph")
               
                # Void line
                if target_id in frame_positions:
                    line_void.set_data([ex, tx], [ey, ty])
                    dist = np.sqrt((ex-tx)**2 + (ey-ty)**2)
                    text_void.set_position(((ex+tx)/2, (ey+ty)/2))
                    text_void.set_text(f"{dist:.1f} yds")
                eraser_history.append((ex, ey))
                if len(eraser_history) > trail_length:
                    eraser_history.pop(0)
                eraser_trail.set_data(*zip(*eraser_history))
           
            return (scat_def_others, scat_def_shadow, scat_off_others, scat_off_shadow, scat_target, scat_target_glow,
                    scat_eraser, scat_eraser_glow, scat_context, scat_context_shadow, scat_qb, scat_qb_shadow,
                    football_patch, line_void, text_void, timer_text, phase_label, speed_label,
                    target_label, eraser_label, qb_label, context_label,
                    target_trail, eraser_trail, context_trail, qb_trail, ball_trail)

        ani = animation.FuncAnimation(fig, update, frames=frames_list, interval=100, blit=True)
       
        save_path = os.path.join(self.output_dir, filename)
        ani.save(save_path, writer='pillow', fps=10)
        plt.close()

        return save_path

class DataLoader:
    def __init__(self, summary_path, frames_path):
        self.summary_path = summary_path
        self.frames_path = frames_path
        self.summary_df = None
        self.frames_df = None

    def load_data(self):
        print(f"   [Loader] Loading Summary Data...")
        self.summary_df = pd.read_csv(self.summary_path)
        self.frames_df = pd.read_csv(self.frames_path)
        
        return self.summary_df, self.frames_df

class StoryDataEngine:
    def __init__(self, summary_df: str, frames_df: str, seed=42):
        self.summary_df = summary_df
        self.frames_df = frames_df
        self.seed = seed
        
    def cast_archetypes(self):
        """
        UPDATED: Selects candidates based on the new VIS thresholds.
        Returns keys: 'The Eraser', 'The Rally', 'The Blanket', 'The Liability'
        """
        df = self.summary_df
        
        # THE ERASER (Elite Recovery)
        # Criteria: Deep start (>10), Massive VIS (>3)
        eraser_pool = df[
            (df['p_dist_at_throw'] > 10) & 
            (df['vis_score'] > 3.0) & 
            (df['dist_at_arrival'] < 4) 
        ]
        eraser = self._select_candidate(eraser_pool, sort_col='vis_score', ascending=False)
        
        # THE RALLY (Standard Zone / The Blob)
        # Criteria: Deep start (>8), Moderate VIS (0.5 to 2.0)
        rally_pool = df[
            (df['p_dist_at_throw'] > 8) & 
            (df['vis_score'].between(0.5, 2.0)) &
            (df['dist_at_arrival'] < 8)
        ]
        rally = self._select_candidate(rally_pool, sort_col='vis_score', ascending=False)
        
        # THE BLANKET (Maintenance)
        # Criteria: Tight start (<3), VIS near zero (-0.5 to 0.5)
        blanket_pool = df[
            (df['p_dist_at_throw'] < 3) & 
            (df['vis_score'].between(-0.5, 0.5)) &
            (df['dist_at_arrival'] < 3)
        ]
        blanket = self._select_candidate(blanket_pool, sort_col='dist_at_arrival', ascending=True)
        
        # THE LIABILITY (True Failure)
        # Criteria: Any start, Negative VIS (Lost Ground)
        liability_pool = df[
            (df['vis_score'] < -1.0) & 
            (df['dist_at_arrival'] > 2.0)
        ]
        liability = self._select_candidate(liability_pool, sort_col='vis_score', ascending=True)

        return {
            'The Eraser': self._extract_meta(eraser, "The Eraser (Elite)"),
            'The Rally': self._extract_meta(rally, "The Rally (Zone)"),
            'The Blanket': self._extract_meta(blanket, "The Blanket (Lockdown)"),
            'The Liability': self._extract_meta(liability, "The Liability (Lost Gap)")
        }

    def _select_candidate(self, df, sort_col, ascending, top_n=1):
        if df.empty: return pd.DataFrame()
        sorted_df = df.sort_values(sort_col, ascending=ascending)
        df = sorted_df.head(top_n)
        print(df.get("player_name", "Unknown Player"))
        return df

    def _extract_meta(self, row, label):
        if row.empty: return None
        return {
            'game_id': int(row.iloc[0]['game_id']),
            'play_id': int(row.iloc[0]['play_id']),
            'nfl_id': float(row.iloc[0]['nfl_id']),
            'player_name': str(row.iloc[0].get('player_name', 'Unknown')),
            'vis_score': float(row.iloc[0]['vis_score']),
            'label': label
        }

    def get_position_contrast(self, position='FS', min_snaps=15):
        """
        Apple-to-apple comparison: Top vs Bottom player for the same position.
        
        1. Ranks players by avg CEOE within the position
        2. Top player â†’ their best VIS play (biggest close)
        3. Bottom player â†’ their worst VIS play (biggest loss)
        """
        df = self.summary_df
        pos_df = df[df['player_position'] == position].copy()
        
        if pos_df.empty:
            print(f"No plays found for position: {position}")
            return {'top': None, 'bottom': None}
        
        # Rank players by average CEOE
        player_ranks = pos_df.groupby('nfl_id').agg(
            player_name=('player_name', 'first'),
            avg_ceoe=('ceoe_score', 'mean'),
            snaps=('play_id', 'count')
        ).reset_index()
        
        # Filter for minimum snaps
        qualified = player_ranks[player_ranks['snaps'] >= min_snaps]
        
        if len(qualified) < 2:
            print(f"Not enough qualified players for {position} (need 2, have {len(qualified)})")
            return {'top': None, 'bottom': None}
        
        top_player = qualified.nlargest(1, 'avg_ceoe').iloc[0]
        bottom_player = qualified.nsmallest(1, 'avg_ceoe').iloc[0]
        
        top_id = top_player['nfl_id']
        bottom_id = bottom_player['nfl_id']

        # Find their best/worst plays
        top_plays = pos_df[pos_df['nfl_id'] == top_id]
        bottom_plays = pos_df[pos_df['nfl_id'] == bottom_id]
        
        # Top player's BEST play (highest VIS = biggest close)
        top_best = top_plays.nlargest(1, 'vis_score')
        
        # Bottom player's WORST play (lowest VIS = biggest loss)
        bottom_worst = bottom_plays.nsmallest(1, 'vis_score')
        
        print(f"Top's best play: VIS = {top_best.iloc[0]['vis_score']:.2f}")
        print(f"Bottom's worst play: VIS = {bottom_worst.iloc[0]['vis_score']:.2f}")
        
        return {
            'top': self._extract_meta(top_best, f"Top {position} Eraser"),
            'bottom': self._extract_meta(bottom_worst, f"Bottom {position} Eraser")
        }

    def get_archetype_contrast(self, min_snaps=15):
        """
        Role contrast: Top FS Eraser vs Top CB Lockdown.
        
        Shows WHY different positions should be graded differently:
        - FS: Closes big gaps (high VIS)
        - CB: Stays tight (low dist_at_arrival)
        """
        df = self.summary_df
        
        fs_df = df[df['player_position'] == 'FS'].copy()
        fs_ranks = fs_df.groupby('nfl_id').agg(
            player_name=('player_name', 'first'),
            avg_ceoe=('ceoe_score', 'mean'),
            snaps=('play_id', 'count')
        ).reset_index()
        fs_qualified = fs_ranks[fs_ranks['snaps'] >= min_snaps]
        
        if fs_qualified.empty:
            print("No qualified FS players")
            return {'eraser': None, 'lockdown': None}
        
        top_fs = fs_qualified.nlargest(1, 'avg_ceoe').iloc[0]
        top_fs_plays = fs_df[fs_df['nfl_id'] == top_fs['nfl_id']]
        eraser_play = top_fs_plays.nlargest(1, 'vis_score')
        
        cb_df = df[df['player_position'] == 'CB'].copy()
        cb_ranks = cb_df.groupby('nfl_id').agg(
            player_name=('player_name', 'first'),
            avg_arrival=('dist_at_arrival', 'mean'),
            snaps=('play_id', 'count')
        ).reset_index()
        cb_qualified = cb_ranks[cb_ranks['snaps'] >= min_snaps]
        
        if cb_qualified.empty:
            print("No qualified CB players")
            return {'eraser': None, 'lockdown': None}
        
        top_cb = cb_qualified.nsmallest(1, 'avg_arrival').iloc[0]  
        top_cb_plays = cb_df[cb_df['nfl_id'] == top_cb['nfl_id']]
        lockdown_play = top_cb_plays.nsmallest(1, 'dist_at_arrival') 

        return {
            'eraser': self._extract_meta(eraser_play, f"Top FS Eraser: {top_fs['player_name']}"),
            'lockdown': self._extract_meta(lockdown_play, f"Top CB Lockdown: {top_cb['player_name']}")
        }

    def get_play_frames(self, play_meta):
        if not play_meta: return pd.DataFrame()
        
        return self.frames_df[
            (self.frames_df['game_id'] == play_meta['game_id']) & 
            (self.frames_df['play_id'] == play_meta['play_id'])
        ].copy()


sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.family'] = 'sans-serif'

class StoryVisualEngine:
    def __init__(self, summary_df, frames_df, output_dir):
        self.summary_df = summary_df
        self.frames_df = frames_df
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.quad_colors = {
            'The Eraser': '#1e8449',      # Dark Green (Elite)
            'The Rally': '#82e0aa',       # Light Green (Active Zone)
            'The Blanket': '#95a5a6',     # Grey (Maintenance)
            'The Liability': '#e74c3c',   # Red (Lost Ground)
            'Neutral': '#ecf0f1'
        }

    def plot_eraser_landscape(self, cast_dict):
        df = self.summary_df.copy()
        
        conditions = [
            (df['vis_score'] > 2.0),                     # The Eraser
            (df['vis_score'].between(0.5, 2.0)),         # The Rally
            (df['vis_score'].between(-0.5, 0.5)),        # The Blanket
            (df['vis_score'] < -0.5)                     # The Liability
        ]

        choices = ['The Eraser', 'The Rally', 'The Blanket', 'The Liability']
        df['quadrant_plot'] = np.select(conditions, choices, default='Neutral')
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Main Scatter
        hue_order = ['The Eraser', 'The Rally', 'The Blanket', 'The Liability']
        
        sns.scatterplot(
            data=df, x='p_dist_at_throw', y='dist_at_arrival',
            hue='quadrant_plot', palette=self.quad_colors,
            hue_order=hue_order,
            alpha=0.6, s=40, linewidth=0, ax=ax
        )
        
        # Reference Lines
        ax.plot([0, 25], [0, 25], 'k--', lw=2, alpha=0.3, label='Break Even (VIS=0)')
        
        # Add the Threshold visual aids (The "Buffer Zone")
        x_vals = np.array([0, 25])
        ax.plot(x_vals, x_vals - 0.5, color='gray', linestyle=':', lw=1, alpha=0.5)
        ax.plot(x_vals, x_vals + 0.5, color='gray', linestyle=':', lw=1, alpha=0.5)

        # Annotations
        for role, play_meta in cast_dict.items():
            if play_meta is None: continue
            
            # Find coordinates in summary
            row = df[(df['game_id'] == play_meta['game_id']) & 
                     (df['play_id'] == play_meta['play_id']) & 
                     (df['nfl_id'] == play_meta['nfl_id'])]
            
            if not row.empty:
                sx = row.iloc[0]['p_dist_at_throw']
                sy = row.iloc[0]['dist_at_arrival']
                
                # Annotate
                ax.annotate(play_meta['label'], 
                            (sx, sy), xytext=(sx+2, sy+2),
                            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                            fontsize=11, fontweight='bold', 
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

        ax.set_title('The Eraser Landscape: Recovery vs. Result', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Start Distance (The Mess)', fontsize=14, fontweight='bold')
        ax.set_ylabel('End Distance (The Finish)', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.legend(title='Defensive Role', loc='upper right')
        
        output_path = os.path.join(self.output_dir, 'V1_Eraser_Landscape.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_race_charts(self, cast_dict):        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12)) 
        axes = axes.flatten()
        
        quad_order = ['The Eraser', 'The Rally', 'The Blanket', 'The Liability']

        # Legend Elements
        legend_elements = [
            plt.Line2D([0], [0], color='grey', lw=3, label='Defender Path'),
            plt.Line2D([0], [0], color='gray', linestyle=':', label='Closed (1.5y)'),
            plt.Line2D([0], [0], marker='o', color='grey', label='Throw', markersize=10, linestyle='None'),
            plt.Line2D([0], [0], marker='X', color='grey', label='Arrival', markersize=10, linestyle='None')
        ]

        for i, quad_name in enumerate(quad_order):
            ax = axes[i]
            play_meta = cast_dict.get(quad_name)
            
            # Default handling if play_meta is missing
            if play_meta is None:
                ax.text(0.5, 0.5, f"No Data for {quad_name}", ha='center', va='center')
                continue

            color = self.quad_colors.get(quad_name, 'grey')

            # Title with VIS Score
            vis_val = play_meta['vis_score']
            sign = "+" if vis_val > 0 else ""
            ax.set_title(f"{quad_name}\n(VIS: {sign}{vis_val:.1f} yds)", 
                         fontsize=16, fontweight='bold', color=color)

            # Get Tracking Data
            play_df = self.frames_df[
                (self.frames_df['game_id'] == play_meta['game_id']) & 
                (self.frames_df['play_id'] == play_meta['play_id'])
            ]
            
            def_track = play_df[play_df['nfl_id'] == play_meta['nfl_id']].sort_values('frame_id')
            target_track = play_df[play_df['player_role'] == 'Targeted Receiver'].sort_values('frame_id')

            if def_track.empty or target_track.empty:
                continue
                    
            merged = pd.merge(def_track, target_track, on='frame_id', suffixes=('_d', '_t'))
            merged['dist'] = np.sqrt((merged['x_d'] - merged['x_t'])**2 + (merged['y_d'] - merged['y_t'])**2)
            merged['time_sec'] = (merged['frame_id'] - merged['frame_id'].min()) * 0.1

            # Spline Smoothing
            time_arr = merged['time_sec'].values
            dist_arr = merged['dist'].values
            if len(time_arr) >= 4:
                try:
                    spline = UnivariateSpline(time_arr, dist_arr, s=50)
                    smooth_dist = spline(time_arr)
                except:
                    smooth_dist = dist_arr
            else:
                smooth_dist = dist_arr
            merged['smooth_dist'] = smooth_dist
            
            max_dist = merged['smooth_dist'].max()
            max_time = merged['time_sec'].max()

            # PLOT LOGIC
            ax.axhspan(0, 1.5, color='#d5d5d5', alpha=0.4, zorder=0)
            ax.text(max_time * 0.95, 0.75, 'CONTESTED\nZONE', ha='right', va='center',
                   fontsize=8, color='#666666', style='italic', alpha=0.8)
            
            ax.plot(merged['time_sec'], merged['smooth_dist'], lw=4, color=color, alpha=0.9)
            ax.fill_between(merged['time_sec'], merged['smooth_dist'], 0, color=color, alpha=0.1)

            start = merged.iloc[0]
            end = merged.iloc[-1]
            
            ax.scatter(start['time_sec'], start['smooth_dist'], color=color, s=150, marker='o', zorder=5, edgecolors='white', lw=2)
            ax.annotate('THROW', (start['time_sec'], start['smooth_dist']), xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold', color=color)

            ax.scatter(end['time_sec'], end['smooth_dist'], color=color, s=150, marker='X', zorder=5, edgecolors='white', lw=2)
            ax.annotate('ARRIVAL', (end['time_sec'], end['smooth_dist']), xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold', color=color)

            # Flight Time (Save for footer)
            play_meta['flight_time'] = end['time_sec'] - start['time_sec']

            # Formatting
            ax.axhline(1.5, color='gray', linestyle=':', lw=2)
            ax.axhline(0, color='black', lw=1)
            ax.set_ylim(-0.5, max(max_dist * 1.15, 5)) 
            ax.set_xlim(0, max_time + 0.3)
            ax.grid(True, alpha=0.3)
            
            if i in [2, 3]: ax.set_xlabel('Seconds After Throw', fontsize=12, fontweight='bold')
            if i in [0, 2]: ax.set_ylabel('Separation (Yards)', fontsize=12, fontweight='bold')

        # Global Legend
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=11)
        
        # Flight Time annotation
        flight_parts = []
        for quad_name in quad_order:
            meta = cast_dict.get(quad_name)
            if meta and 'flight_time' in meta:
                flight_parts.append(f"{quad_name}: {meta['flight_time']:.1f}s")
        
        if flight_parts:
            flight_text = "Ball Flight Time â€” " + " | ".join(flight_parts)
            fig.text(0.5, -0.06, flight_text, ha='center', va='top', fontsize=10, color='#555555', style='italic')
        
        plt.tight_layout(rect=[0, 0.02, 1, 1])
        output_path = os.path.join(self.output_dir, 'V2_Race_Charts.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_coverage_heatmap(self):
        """
        Average VIS by Route Depth vs Coverage Type.
        """
        df = self.summary_df.copy()

        # Binning
        bins = [-5, 5, 10, 15, 25, 100]
        labels = ['Behind LOS', 'Short (0-5)', 'Medium (5-10)', 'Int (10-15)', 'Deep (15+)']
        df['depth_band'] = pd.cut(df['pass_length'], bins=bins, labels=labels)

        # Filter
        main_coverages = ['COVER_1', 'COVER_2_MAN', 'COVER_2_ZONE', 'COVER_3_ZONE', 'COVER_4_ZONE', 'COVER_6_ZONE']
        df = df[df['team_coverage_type'].isin(main_coverages)]

        # Pivot
        grouped = df.groupby(['team_coverage_type', 'depth_band'], observed=False)
        heatmap_data = grouped['vis_score'].mean()
        counts = grouped.size()
        heatmap_data = heatmap_data.where(counts >= 10).unstack()

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        cmap = sns.diverging_palette(10, 130, as_cmap=True) # Red-Green
        
        sns.heatmap(
            heatmap_data, annot=True, fmt=".1f", cmap=cmap,
            center=0, vmin=-2, vmax=4, linewidths=.5,
            cbar_kws={'label': 'Average VIS (Yards Erased)'}, ax=ax
        )

        ax.set_title('Where Defenses Erase Space: Scheme vs. Depth', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Target Depth (Air Yards)', fontsize=14)
        ax.set_ylabel('Coverage Scheme', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        output_path = os.path.join(self.output_dir, 'V3_Coverage_Heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_effort_impact_chart(self):
        """
        Slope chart showing Effort â†’ Outcome (EPA/YAC saved).
        Ties the story together: Context â†’ Effort â†’ Result.
        """
        df = self.summary_df.copy()
        
        completed = df[df['pass_result'] == 'C'].copy()

        # Derive YAC
        completed['yac'] = completed['yards_gained'] - completed['pass_length']
        
        # Create Start Distance bands
        dist_bins = [0, 3, 6, 10, 100]
        dist_labels = ['Tight\n(0-3 yds)', 'Medium\n(3-6 yds)', 'High Void\n(6-10 yds)', 'Exempt\n(10+ yds)']
        completed['start_band'] = pd.cut(completed['p_dist_at_throw'], bins=dist_bins, labels=dist_labels)
        
        # Calculate VIS quartiles WITHIN each start band
        def get_quartile_label(group):
            q25 = group['vis_score'].quantile(0.25)
            q75 = group['vis_score'].quantile(0.75)
            conditions = [
                group['vis_score'] <= q25,
                group['vis_score'] >= q75
            ]
            choices = ['Low Effort', 'High Effort']
            group['effort_bucket'] = np.select(conditions, choices, default='Middle')
            return group
        
        completed = completed.groupby('start_band', group_keys=False, observed=False).apply(get_quartile_label)
        
        extremes = completed[completed['effort_bucket'].isin(['Low Effort', 'High Effort'])]
        impact_data = extremes.groupby(['start_band', 'effort_bucket'], observed=False).agg(
            avg_epa=('expected_points_added', 'mean'),
            avg_yac=('yac', 'mean'),
            count=('play_id', 'count')
        ).reset_index()
        
        # Create the figure with two subplots (EPA and YAC)
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        bands = ['Tight\n(0-3 yds)', 'Medium\n(3-6 yds)', 'High Void\n(6-10 yds)', 'Exempt\n(10+ yds)']
        x_positions = np.arange(len(bands))
        
        for ax_idx, (metric, title, ylabel) in enumerate([
            ('avg_epa', 'EPA Impact: Effort Saves Points', 'Expected Points Added'),
            ('avg_yac', 'YAC Impact: Effort Limits Damage', 'Yards After Catch')
        ]):
            ax = axes[ax_idx]
            
            # Get data for each effort level
            low_effort = []
            high_effort = []
            savings = []
            
            for band in bands:
                low_row = impact_data[(impact_data['start_band'] == band) & (impact_data['effort_bucket'] == 'Low Effort')]
                high_row = impact_data[(impact_data['start_band'] == band) & (impact_data['effort_bucket'] == 'High Effort')]
                
                low_val = low_row[metric].values[0] if not low_row.empty else np.nan
                high_val = high_row[metric].values[0] if not high_row.empty else np.nan
                
                low_effort.append(low_val)
                high_effort.append(high_val)
                savings.append(low_val - high_val if pd.notna(low_val) and pd.notna(high_val) else np.nan)
            
            # Plot bars - calmer colors
            bar_width = 0.35
            bars_low = ax.bar(x_positions - bar_width/2, low_effort, bar_width, 
                             label='Low Effort (Q1)', color='#d98880', alpha=0.85, edgecolor='white', linewidth=1.5)
            bars_high = ax.bar(x_positions + bar_width/2, high_effort, bar_width, 
                              label='High Effort (Q4)', color='#7dcea0', alpha=0.85, edgecolor='white', linewidth=1.5)
            
            # Add value labels on bars
            for bar in bars_low:
                height = bar.get_height()
                if pd.notna(height):
                    ax.annotate(f'{height:.2f}',
                               xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=9, fontweight='bold', color='#943126')
            
            for bar in bars_high:
                height = bar.get_height()
                if pd.notna(height):
                    ax.annotate(f'{height:.2f}',
                               xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1e8449')
            
            # Add savings labels (no arrows, positioned at bottom of chart to avoid overlap)
            max_height = max([v for v in low_effort + high_effort if pd.notna(v)]) if any(pd.notna(v) for v in low_effort + high_effort) else 2
            
            for i, saved in enumerate(savings):
                if pd.notna(saved) and saved > 0:
                    unit = 'EPA' if metric == 'avg_epa' else 'YAC'
                    # Position label below x-axis to avoid overlap with title
                    ax.text(x_positions[i], -0.15 if metric == 'avg_epa' else -0.4, 
                           f'Saved: {saved:.2f} {unit}', 
                           ha='center', va='top', fontsize=9, fontweight='bold',
                           color='#1a5276', 
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='#d5f5e3', edgecolor='#1e8449', alpha=0.9))
            
            # Formatting
            ax.set_xticks(x_positions)
            ax.set_xticklabels(bands, fontsize=10)
            ax.set_xlabel('Starting Distance Band', fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
            ax.legend(loc='upper right', fontsize=9)
            ax.axhline(0, color='black', linewidth=0.5)
            ax.grid(axis='y', alpha=0.3)
            
            # Set y-limits with padding for labels
            if metric == 'avg_epa':
                ax.set_ylim(bottom=-0.6, top=max_height + 0.5)
            else:
                ax.set_ylim(bottom=-1.0, top=max_height + 0.8)
        
        # Overall title
        fig.suptitle('The Payoff of Eraser: High-Effort Defenders Save Points & Yards',
                     fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leave room for labels and title
        output_path = os.path.join(self.output_dir, 'V4_Effort_Impact_Chart.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

class TableGenerator:
    def __init__(self, suumary_df: str):
        self.df = suumary_df
        
        # We exclude Pass Rushers who occasionally drop into coverage
        self.df = self.df[self.df['player_role'].isin([
            'Defensive Coverage', 'Cornerback', 'Safety', 'Linebacker'])]

    def generate_quadrant_counts(self):
        """
        TABLE 2: Quadrant Counts Table
        Buckets plays into the 4 Matrix Outcomes.
        """
        df = self.df.copy()

        # Define Thresholds
        OPEN_THRESH = 3.0
        CLOSED_THRESH = 1.5

        conditions = [
            (df['dist_at_throw'] >= OPEN_THRESH) & (df['dist_at_arrival'] <= CLOSED_THRESH), # Open -> Closed
            (df['dist_at_throw'] < OPEN_THRESH) & (df['dist_at_arrival'] <= CLOSED_THRESH),  # Tight -> Closed
            (df['dist_at_throw'] < OPEN_THRESH) & (df['dist_at_arrival'] > CLOSED_THRESH),  # Tight -> Open
            (df['dist_at_throw'] >= OPEN_THRESH) & (df['dist_at_arrival'] > CLOSED_THRESH)  # Open -> Open
        ]
        
        choices = ['Eraser (The Cleanup)', 'Lockdown (The Blanket)', 'Lost Step (The Beat)', 'Liability (The Void)']
        
        df['quadrant'] = np.select(conditions, choices, default='Neutral/Zone Drift')
        
        summary = df.groupby('quadrant').agg(
            play_count=('play_id', 'count'),
            avg_vis=('vis_score', 'mean'),
            avg_ceoe=('ceoe_score', 'mean')
        ).reset_index()

        summary['avg_vis'] = summary['avg_vis'].round(2)
        summary['avg_ceoe'] = summary['avg_ceoe'].round(3)

        return summary.sort_values('avg_vis', ascending=False)

    def generate_shrunk_leaderboard(self, min_snaps=15, prior_m=20):
        """
        Bayesian Shrinkage with Names.
        """
        # Positional Priors
        pos_stats = self.df.groupby('player_position')['ceoe_score'].mean().to_dict()

        # Add player_name to grouping
        group_cols = ['nfl_id', 'player_position', 'player_role']
        if 'player_name' in self.df.columns:
            group_cols.insert(1, 'player_name')

        player_stats = self.df.groupby(group_cols).agg(
            snaps=('play_id', 'count'),
            raw_ceoe=('ceoe_score', 'mean'),
            avg_vis=('vis_score', 'mean'),
            avg_start=('p_dist_at_throw', 'mean')
        ).reset_index()

        # Shrinkage
        def apply_shrinkage(row):
            prior_mu = pos_stats.get(row['player_position'], 0.0)
            n = row['snaps']
            m = prior_m
            shrunk = ((n * row['raw_ceoe']) + (m * prior_mu)) / (n + m)
            return shrunk

        player_stats['shrunk_ceoe'] = player_stats.apply(apply_shrinkage, axis=1)

        qualified = player_stats[player_stats['snaps'] >= min_snaps].copy()     
        qualified['shrunk_ceoe'] = qualified['shrunk_ceoe'].round(3)
        qualified['raw_ceoe'] = qualified['raw_ceoe'].round(3)
        qualified['avg_vis'] = qualified['avg_vis'].round(2)
        qualified['avg_start'] = qualified['avg_start'].round(1)

        top_erasers = qualified.sort_values('shrunk_ceoe', ascending=False).head(10)
        
        return top_erasers

    def generate_damage_control_validation(self):
        """
        Damage Control Validation (YAC & EPA).

        Hypothesis: On COMPLETED passes, higher VIS (better closing) 
        should correlate with LOWER YAC and LOWER EPA (better for defense).
        """
        df = self.df.copy()
        
        # Filter for Completions Only (Where YAC exists)
        completed = df[df['pass_result'] == 'C'].copy()

        # Derive YAC
        # YAC = Total Yards - Air Yards
        completed['yac'] = completed['yards_gained'] - completed['pass_length']
        
        # Bin Start Distance (Context Control)
        # We only care about Medium/High Voids where YAC is a threat.
        bins = [3, 6, 10, 100]
        labels = ['Medium (3-6)', 'High Void (6-10)', 'Deep (10+)']
        completed['start_band'] = pd.cut(completed['p_dist_at_throw'], bins=bins, labels=labels)
        
        # Bin VIS Score (The Independent Variable)
        # Low Effort vs. High Effort closing
        vis_bins = [-np.inf, 0, 3, np.inf]
        vis_labels = ['Negative (Lost Gap)', 'Moderate (0-3)', 'High Eraser (3+)']
        completed['vis_bucket'] = pd.cut(completed['vis_score'], bins=vis_bins, labels=vis_labels)

        damage_control = completed.groupby(['start_band', 'vis_bucket'], observed=False).agg(
            count=('play_id', 'count'),
            avg_yac=('yac', 'mean'),
            avg_epa=('expected_points_added', 'mean') # Lower is better for defense
        ).reset_index()

        # Pivot for YAC (The Primary Proof)
        yac_pivot = damage_control.pivot(index='start_band', columns='vis_bucket', values='avg_yac')
        
        # Calculate the "Savings" (Difference between Negative VIS and High Eraser)
        yac_pivot['YAC_Savings'] = yac_pivot['Negative (Lost Gap)'] - yac_pivot['High Eraser (3+)']
        
        return yac_pivot.round(2)

    def generate_epa_savings(self):
        """
        EPA Savings Table (Quartile Approach).

        Shows how much Expected Points high-effort defenders save vs low-effort defenders.
        
        Focuses on COMPLETED passes where EPA damage occurs.
        """
        df = self.df.copy()
        
        completed = df[df['pass_result'] == 'C'].copy()
        
        # Create Start Distance bands
        dist_bins = [0, 3, 6, 10, 100]
        dist_labels = ['Tight (0-3)', 'Medium (3-6)', 'High Void (6-10)', 'Exempt (10+)']
        completed['start_band'] = pd.cut(completed['p_dist_at_throw'], bins=dist_bins, labels=dist_labels)
        
        # Calculate VIS quartiles WITHIN each start band
        # This avoids NaNs by making "effort" relative to what's possible from each start position
        def get_quartile_label(group):
            q25 = group['vis_score'].quantile(0.25)
            q75 = group['vis_score'].quantile(0.75)
            
            conditions = [
                group['vis_score'] <= q25,
                group['vis_score'] >= q75
            ]
            choices = ['Low Effort (Q1)', 'High Effort (Q4)']
            group['effort_bucket'] = np.select(conditions, choices, default='Middle (Q2-Q3)')
            return group
        
        completed = completed.groupby('start_band', group_keys=False, observed=False).apply(get_quartile_label)
        
        # Filter to only Q1 and Q4 for clean comparison
        extremes = completed[completed['effort_bucket'].isin(['Low Effort (Q1)', 'High Effort (Q4)'])]
        
        epa_table = extremes.groupby(['start_band', 'effort_bucket'], observed=False).agg(
            play_count=('play_id', 'count'),
            avg_epa=('expected_points_added', 'mean')
        ).reset_index()
        
        # Pivot for clear comparison
        epa_pivot = epa_table.pivot(index='start_band', columns='effort_bucket', values='avg_epa')
        
        # Calculate EPA Saved (Low Effort EPA - High Effort EPA)
        # Positive = High effort defenders saved points
        if 'Low Effort (Q1)' in epa_pivot.columns and 'High Effort (Q4)' in epa_pivot.columns:
            epa_pivot['EPA_Saved'] = epa_pivot['Low Effort (Q1)'] - epa_pivot['High Effort (Q4)']
        
        count_pivot = epa_table.pivot(index='start_band', columns='effort_bucket', values='play_count')
        epa_pivot['Plays_Compared'] = count_pivot.sum(axis=1)
        
        col_order = ['Low Effort (Q1)', 'High Effort (Q4)', 'EPA_Saved', 'Plays_Compared']
        epa_pivot = epa_pivot[[c for c in col_order if c in epa_pivot.columns]]
        
        return epa_pivot.round(3)

    def generate_position_breakdown(self):
        """
        Position Breakdown - "Differe Players Roles"

        Shows which position groups are best suited for the Eraser role.
        """
        df = self.df.copy()
        
        # Define Eraser criteria (derived from start/end distances, not void_type)
        OPEN_THRESH = 6.0
        CLOSED_THRESH = 2.0
        df['is_eraser_play'] = (df['p_dist_at_throw'] >= OPEN_THRESH) & (df['dist_at_arrival'] <= CLOSED_THRESH)
        
        position_stats = df.groupby('player_position').agg(
            play_count=('play_id', 'count'),
            avg_start_dist=('p_dist_at_throw', 'mean'),
            avg_end_dist=('dist_at_arrival', 'mean'),
            avg_vis=('vis_score', 'mean'),
            eraser_plays=('is_eraser_play', 'sum')
        ).reset_index()
        
        # Calculate Eraser Rate (% of plays where they achieved Eraser outcome)
        position_stats['eraser_rate'] = (position_stats['eraser_plays'] / position_stats['play_count'] * 100).round(1)
        
        # Filter for positions with meaningful sample size
        position_stats = position_stats[position_stats['play_count'] >= 50].copy()
        
        # Derive Eraser Archetype based on behavior patterns
        def assign_archetype(row):
            avg_start = row['avg_start_dist']
            avg_vis = row['avg_vis']
            eraser_rate = row['eraser_rate']
            
            # Primary Eraser: Deep starters who close aggressively
            if avg_start >= 8 and avg_vis >= 1.5:
                return "ğŸŸ¢ Primary Eraser"
            # Secondary Eraser: Medium depth with good closing
            elif avg_start >= 6 and avg_vis >= 1.0:
                return "ğŸ”µ Secondary Eraser"
            # Lockdown: Tight coverage specialists (low start = already close)
            elif avg_start < 5 and avg_vis < 0.5:
                return "ğŸŸ¡ Lockdown Focus"
            # Situational: High eraser rate despite moderate metrics
            elif eraser_rate >= 5:
                return "ğŸŸ  Situational Eraser"
            else:
                return "âšª Zone Support"
        
        position_stats['archetype'] = position_stats.apply(assign_archetype, axis=1)
        
        position_stats['avg_start_dist'] = position_stats['avg_start_dist'].round(1)
        position_stats['avg_end_dist'] = position_stats['avg_end_dist'].round(1)
        position_stats['avg_vis'] = position_stats['avg_vis'].round(2)
        
        # Select and order columns for output
        # TODO: bring them from schema.
        output_cols = ['player_position', 'play_count', 'avg_start_dist', 'avg_end_dist',
                       'avg_vis', 'eraser_rate', 'archetype']
        
        return position_stats[output_cols].sort_values('avg_start_dist', ascending=False)

    def generate_void_effect_size(self):
        """
        Void Effect Size Analysis
        Shows completion %, EPA, and YAC by S_throw band with effect size 
        (Î” from Tight baseline) to quantify the jump in difficulty.
        """
        df = self.df.copy()
        
        # Define S_throw bands based on dist_at_throw (original separation)
        dist_col = 'p_dist_at_throw' if 'p_dist_at_throw' in df.columns else 'dist_at_throw'
        
        bins = [0, 2, 6, 10, float('inf')]
        labels = ['Tight (0-2 yds)', 'Medium (3-6 yds)', 'High Void (6-10 yds)', 'Deep (10+ yds)']
        df['start_band'] = pd.cut(df[dist_col], bins=bins, labels=labels, include_lowest=True)
        
        # Derive YAC for completions (yards_gained - pass_length)
        df['yac'] = df['yards_gained'] - df['pass_length']
        
        # Calculate metrics by band
        band_stats = df.groupby('start_band', observed=False).agg(
            play_count=('play_id', 'count'),
            completions=('pass_result', lambda x: (x == 'C').sum()),
            avg_epa=('expected_points_added', 'mean')
        ).reset_index()
        
        # Calculate YAC separately (only for completions)
        completed = df[df['pass_result'] == 'C']
        yac_by_band = completed.groupby('start_band', observed=False)['yac'].mean().reset_index()
        yac_by_band.columns = ['start_band', 'avg_yac']

        band_stats = band_stats.merge(yac_by_band, on='start_band', how='left')
        
        # Calculate completion percentage
        band_stats['completion_pct'] = (band_stats['completions'] / band_stats['play_count'] * 100).round(1)
        
        # Calculate effect size (Î” from Tight baseline)
        tight_completion = band_stats.loc[band_stats['start_band'] == 'Tight (0-2 yds)', 'completion_pct'].values
        if len(tight_completion) > 0:
            tight_baseline = tight_completion[0]
            band_stats['delta_from_tight'] = band_stats['completion_pct'] - tight_baseline
            band_stats['delta_from_tight'] = band_stats['delta_from_tight'].apply(
                lambda x: f"+{x:.1f}pp" if x > 0 else ("â€”" if x == 0 else f"{x:.1f}pp")
            )
        else:
            band_stats['delta_from_tight'] = "â€”"
        
        # Format other columns
        band_stats['avg_epa'] = band_stats['avg_epa'].round(2)
        band_stats['avg_yac'] = band_stats['avg_yac'].round(1)
        band_stats['completion_pct'] = band_stats['completion_pct'].apply(lambda x: f"{x}%")
        
        output = band_stats[['start_band', 'play_count', 'completion_pct', 'delta_from_tight', 'avg_epa', 'avg_yac']]
        output.columns = ['S_throw Band', 'Play Count', 'Completion %', 'Î” from Tight', 'Avg EPA Allowed', 'Avg YAC']
        
        return output

def run_visual_pipeline(SUMMARY_FILE=None, TRACKING_FILE=None, OUTPUT_DIR=None):

    vis_cfg = VisPipelineConfig(
        SUMMARY_FILE=SUMMARY_FILE or vis_config.SUMMARY_FILE,
        TRACKING_FILE=TRACKING_FILE or vis_config.TRACKING_FILE,
        OUTPUT_DIR=OUTPUT_DIR or vis_config.OUTPUT_DIR
    )
    
    summary_path = vis_cfg.SUMMARY_FILE
    tracking_path = vis_cfg.TRACKING_FILE
    output_dir = vis_cfg.OUTPUT_DIR

    loader = DataLoader(summary_path, tracking_path)
    summary_df, frames_df = loader.load_data()

    return summary_df, frames_df, output_dir


# Initiate Visual Pipeline
summary_df, frames_df, output_dir = run_visual_pipeline(
    OUTPUT_DIR=vis_config.OUTPUT_DIR,
    TRACKING_FILE=vis_config.TRACKING_FILE,
    SUMMARY_FILE=vis_config.SUMMARY_FILE
)


# Initiate Table Generator class.
gen = TableGenerator(summary_df)


# --- TABLE 1: SHRUNK LEADERBOARD ---
df_leaderboard = gen.generate_shrunk_leaderboard()
print_header("1. The Eraser Leaderboard", "Bayesian Shrinkage (m=20) | Top 10 Defenders")

st_leader = (df_leaderboard.style
             .format({'shrunk_ceoe': '{:+.3f}', 'raw_ceoe': '{:+.3f}', 'avg_vis': '{:+.2f}', 'avg_start': '{:.1f}'})
             .set_properties(**{'text-align': 'center'})
             .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
)
display(st_leader)


# --- TABLE 2: QUADRANT SUMMARY ---
df_quad = gen.generate_quadrant_counts()
print_header("2. Quadrant Matrix Summary", "Distribution of Plays across the 4 Eraser Archetypes")

st_quad = (df_quad.style
           .format({'avg_vis': '{:+.2f}', 'avg_ceoe': '{:+.3f}'})
           .set_properties(**{'text-align': 'center'})
           .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
           .hide(axis='index')
)
display(st_quad)


# --- TABLE 3: DAMAGE CONTROLE ---
df_yac = gen.generate_damage_control_validation()
print_header("3. Damage Control: YAC Analysis", "Does High VIS reduce Yards After Catch?")

savings_col = 'YAC_Savings' if 'YAC_Savings' in df_yac.columns else None

format_dict = {col: '{:.2f}' for col in df_yac.select_dtypes(include=['float', 'int']).columns}
if savings_col:
    format_dict[savings_col] = '{:+.2f}'

st_yac = (df_yac.style
          .format(format_dict)
          .set_properties(**{'text-align': 'center'})
          .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
)
display(st_yac)


# --- TABLE 4: EPA SAVINGS ---
df_epa = gen.generate_epa_savings()
print_header("4. EPA Savings", "Expected Points Saved by High-Effort (Q4) vs Low-Effort (Q1) Defenders")

st_epa = (df_epa.style
          .format({'EPA_Saved': '{:+.3f}', 'Low Effort (Q1)': '{:.3f}', 'High Effort (Q4)': '{:.3f}'})
          .set_properties(**{'text-align': 'center'})
          .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
)
display(st_epa)


# --- TABLE 5: Position Breakdown ---
df_pos = gen.generate_position_breakdown()
print_header("5. Position Breakdown", "Which positions generate the most Eraser events?")

st_pos = (df_pos.style
          .format({'eraser_rate': '{:.1f}%', 'avg_start_dist': '{:.1f}', 'avg_end_dist': '{:.1f}', 'avg_vis': '{:+.2f}'})
          .set_properties(**{'text-align': 'center'})
          .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
          .hide(axis='index')
)
display(st_pos)


# --- TABLE 6: VOID EFFECT SIZE ---
df_void = gen.generate_void_effect_size()
print_header("6. The Void Effect", "Impact of Starting Distance on Completion % and EPA")

st_void = (df_void.style
           .format({'Avg EPA Allowed': '{:.2f}', 'Avg YAC': '{:.1f}'})
           .set_properties(**{'text-align': 'center'})
           .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
           .hide(axis='index')
)
display(st_void)


# Prepare Data for Visualization. 

# Start Story Engine
story = StoryDataEngine(summary_df, frames_df)

# Get Comparisons
fs_contrast = story.get_position_contrast('FS')
cast_dict = story.cast_archetypes()

# Start Visualize Engine.
viz = StoryVisualEngine(summary_df, frames_df, output_dir)


# Effort Impact Chart: EPA and YAC by defender effort quartile
viz.plot_effort_impact_chart()


# The Eraser Landscape. VIS solves the "Apples to Oranges" evaluation problem. 
viz.plot_eraser_landscape(cast_dict)


# Animation Engine (Video Rendering)
animator = AnimationEngine(summary_df, frames_df, output_dir)

# Render Top FS Eraser
if fs_contrast['top']:
    top_gif_path = animator.generate_video(
        game_id=fs_contrast['top']['game_id'], 
        play_id=fs_contrast['top']['play_id'], 
        eraser_id=fs_contrast['top']['nfl_id'], 
        filename="Figure_Top_FS_Eraser.gif" 
    )

# Render Bottom FS Eraser
if fs_contrast['bottom']:
    bottom_gif_path = animator.generate_video(
        game_id=fs_contrast['bottom']['game_id'], 
        play_id=fs_contrast['bottom']['play_id'], 
        eraser_id=fs_contrast['bottom']['nfl_id'], 
        filename="Figure_Bottom_FS_Eraser.gif" 
    )


print(f"Displaying animation for Top Eraser Performer")
display(Image(filename=top_gif_path))


print(f"Displaying animation for Bottom Eraser Performer")
display(Image(filename=bottom_gif_path))

