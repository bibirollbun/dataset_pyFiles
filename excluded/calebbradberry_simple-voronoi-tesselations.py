# NFL Coach's Pre-Snap Analysis Notebook
# ====================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# Load data files
data = {
    'games': pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/games.csv'),
    'plays': pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/plays.csv'),
    'players': pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/players.csv'),
    'player_play': pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/player_play.csv'),
    'tracking': pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_1.csv')
}

print("Data loaded successfully")
for name, df in data.items():
    print(f"{name}: {df.shape}")

@dataclass
class FieldConstants:
    """Standard NFL field dimensions"""
    LENGTH: float = 120.0
    WIDTH: float = 53.3
    HASH_MARKS: List[float] = (0.2, WIDTH - 0.2)

class PreSnapAnalyzer:
    """Analyzes pre-snap formations and spatial control"""
    
    def __init__(self, tracking_data: pd.DataFrame, plays_data: pd.DataFrame,
                 players_data: pd.DataFrame, player_play_data: pd.DataFrame):
        """Initialize with required data sources"""
        self.tracking = tracking_data
        self.plays = plays_data
        self.players = players_data
        self.player_play = player_play_data
        self.field = FieldConstants()
    
    def get_pre_snap_frame(self, game_id: int, play_id: int) -> pd.DataFrame:
        """Get the last frame before snap for a specific play"""
        play_data = self.tracking[
            (self.tracking['gameId'] == game_id) &
            (self.tracking['playId'] == play_id) &
            (self.tracking['frameType'] == 'BEFORE_SNAP')
        ]
        
        if not play_data.empty:
            return play_data[play_data['frameId'] == play_data['frameId'].max()]
        return pd.DataFrame()

    def calculate_voronoi_control(self, frame: pd.DataFrame, offense_team: str) -> Dict:
        """Calculate spatial control using Voronoi tessellation"""
        try:
            # Get player positions
            positions = frame[['x', 'y']].values
            
            # Add boundary points to constrain Voronoi regions
            bound_box = np.array([
                [0, 0], [0, self.field.WIDTH],
                [self.field.LENGTH, 0], [self.field.LENGTH, self.field.WIDTH],
                [-10, -10], [-10, self.field.WIDTH + 10],
                [self.field.LENGTH + 10, -10], [self.field.LENGTH + 10, self.field.WIDTH + 10]
            ])
            positions_with_bounds = np.vstack([positions, bound_box])
            
            # Calculate Voronoi
            vor = Voronoi(positions_with_bounds)
            
            # Calculate areas for each player (excluding boundary points)
            areas = []
            for i, region in enumerate(vor.regions):
                if -1 not in region and len(region) > 0 and i < len(positions):
                    polygon = [vor.vertices[j] for j in region]
                    area = self._polygon_area(polygon)
                    areas.append(area)
                else:
                    areas.append(0)
                    
            # Split areas between offense and defense
            team_masks = frame['club'] == offense_team
            offense_areas = np.array(areas)[:len(frame)][team_masks]
            defense_areas = np.array(areas)[:len(frame)][~team_masks]
            
            # Calculate control metrics
            total_area = sum(areas[:len(frame)])  # Exclude boundary areas
            offense_control = np.sum(offense_areas) / total_area * 100 if total_area > 0 else 50
            
            return {
                'offense_control': offense_control,
                'offense_areas': offense_areas.tolist(),
                'defense_areas': defense_areas.tolist(),
                'avg_offense_area': np.mean(offense_areas) if len(offense_areas) > 0 else 0,
                'avg_defense_area': np.mean(defense_areas) if len(defense_areas) > 0 else 0
            }
        except Exception as e:
            print(f"Error in Voronoi calculation: {str(e)}")
            return {
                'offense_control': 50,
                'offense_areas': [],
                'defense_areas': [],
                'avg_offense_area': 0,
                'avg_defense_area': 0
            }
    
    def _polygon_area(self, vertices):
        """Calculate area of a polygon using shoelace formula"""
        x = [v[0] for v in vertices]
        y = [v[1] for v in vertices]
        return 0.5 * abs(sum(i * j for i, j in zip(x, y[1:] + y[:1])) -
                        sum(i * j for i, j in zip(x[1:] + x[:1], y)))

    def analyze_formation(self, frame: pd.DataFrame, play_info: pd.Series) -> Dict:
        """Analyze offensive formation and spatial control"""
        # Get player play data for additional context
        play_players = self.player_play[
            (self.player_play['gameId'] == play_info['gameId']) &
            (self.player_play['playId'] == play_info['playId'])
        ]
        
        # Split into offense and defense
        offense = frame[frame['club'] == play_info['possessionTeam']]
        defense = frame[frame['club'] == play_info['defensiveTeam']]
        
        # Calculate spatial metrics
        offensive_spread = np.std(offense['y'])
        defensive_spread = np.std(defense['y'])
        
        # Calculate average distances
        off_center = offense['y'].mean()
        def_center = defense['y'].mean()
        
        # Get motion indicators
        motion_players = play_players[play_players['inMotionAtBallSnap'] == True]
        
        # Calculate Voronoi control
        voronoi_metrics = self.calculate_voronoi_control(frame, play_info['possessionTeam'])
        
        return {
            'formation': play_info['offenseFormation'],
            'offensive_spread': offensive_spread,
            'defensive_spread': defensive_spread,
            'off_def_alignment': abs(off_center - def_center),
            'personnel_count': len(offense),
            'motion_players': len(motion_players),
            'pass_rush_count': len(play_players[play_players['wasInitialPassRusher'] == True]),
            'spatial_control': voronoi_metrics['offense_control'],
            'avg_offensive_space': voronoi_metrics['avg_offense_area'],
            'avg_defensive_space': voronoi_metrics['avg_defense_area']
        }

    def visualize_formation(self, game_id: int, play_id: int, show_voronoi: bool = True):
        """Create coach-friendly visualization of pre-snap formation with optional Voronoi regions"""
        frame = self.get_pre_snap_frame(game_id, play_id)
        play_info = self.plays[
            (self.plays['gameId'] == game_id) &
            (self.plays['playId'] == play_id)
        ].iloc[0]
        
        if frame.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Draw field
        ax.add_patch(plt.Rectangle((0, 0), self.field.LENGTH, self.field.WIDTH,
                                 color='darkgreen', alpha=0.3))
        
        # Plot players
        offense = frame[frame['club'] == play_info['possessionTeam']]
        defense = frame[frame['club'] == play_info['defensiveTeam']]
        
        # Plot with player roles
        play_players = self.player_play[
            (self.player_play['gameId'] == game_id) &
            (self.player_play['playId'] == play_id)
        ]
        
        # Add Voronoi regions if requested
        if show_voronoi:
            # Calculate Voronoi regions
            positions = frame[['x', 'y']].values
            bound_box = np.array([
                [0, 0], [0, self.field.WIDTH],
                [self.field.LENGTH, 0], [self.field.LENGTH, self.field.WIDTH],
                [-10, -10], [-10, self.field.WIDTH + 10],
                [self.field.LENGTH + 10, -10], [self.field.LENGTH + 10, self.field.WIDTH + 10]
            ])
            positions_with_bounds = np.vstack([positions, bound_box])
            
            try:
                vor = Voronoi(positions_with_bounds)
                
                # Plot regions
                for i, region in enumerate(vor.regions):
                    if -1 not in region and len(region) > 0 and i < len(positions):
                        polygon = [vor.vertices[j] for j in region]
                        if len(polygon) > 2:  # Valid polygon
                            is_offense = frame.iloc[i]['club'] == play_info['possessionTeam']
                            color = 'blue' if is_offense else 'red'
                            ax.fill(*zip(*polygon), alpha=0.2, color=color)
            except:
                print("Could not calculate Voronoi regions")
        
        # Plot motion players differently
        motion_players = play_players[play_players['inMotionAtBallSnap'] == True]
        
        # Offensive players
        ax.scatter(offense['x'], offense['y'], 
                  color='blue', s=100, label='Offense',
                  alpha=[1.0 if p not in motion_players['nflId'].values else 0.5 
                        for p in offense['nflId']])
        
        # Defensive players with pass rushers highlighted
        pass_rushers = play_players[play_players['wasInitialPassRusher'] == True]
        ax.scatter(defense['x'], defense['y'],
                  color='red', s=100, label='Defense',
                  alpha=[1.0 if p in pass_rushers['nflId'].values else 0.5 
                        for p in defense['nflId']])
        
        # Add jersey numbers
        for _, player in frame.iterrows():
            if pd.notna(player['jerseyNumber']):
                ax.annotate(str(int(player['jerseyNumber'])),
                           (player['x'], player['y']),
                           color='white',
                           ha='center',
                           va='center')
        
        # Add play information
        title = (f"Formation: {play_info['offenseFormation']}\n"
                f"Players in Motion: {len(motion_players)}\n"
                f"Result: {play_info['playDescription'][:50]}...")
        ax.set_title(title)
        
        ax.set_xlim(-5, self.field.LENGTH + 5)
        ax.set_ylim(-5, self.field.WIDTH + 5)
        ax.legend()
        
        return fig

    def analyze_game(self, game_id: int):
        """Analyze all plays for a specific game"""
        # Get game plays
        game_plays = self.plays[self.plays['gameId'] == game_id]
        
        print("\nGame Summary:")
        print(f"Total plays: {len(game_plays)}")
        print(f"Unique formations: {game_plays['offenseFormation'].nunique()}")
        
        # Formation stats
        formation_stats = game_plays.groupby('offenseFormation').agg({
            'yardsGained': ['count', 'mean', 'sum'],
            'playId': 'count'
        }).round(2)
        
        print("\nFormation Stats:")
        print(formation_stats)
        
        # Find successful plays
        big_plays = game_plays[game_plays['yardsGained'] >= 10]
        if not big_plays.empty:
            print(f"\nAnalyzing {len(big_plays)} big plays (10+ yards)...")
            
            for _, play in big_plays.head(3).iterrows():
                print(f"\nPlay {play['playId']}:")
                print(f"Formation: {play['offenseFormation']}")
                print(f"Yards gained: {play['yardsGained']}")
                
                # Get detailed formation analysis
                frame = self.get_pre_snap_frame(play['gameId'], play['playId'])
                if not frame.empty:
                    analysis = self.analyze_formation(frame, play)
                    print("\nFormation Analysis:")
                    print(f"Offensive Spread: {analysis['offensive_spread']:.2f} yards")
                    print(f"Players in Motion: {analysis['motion_players']}")
                    print(f"Pass Rushers: {analysis['pass_rush_count']}")
                
                fig = self.visualize_formation(play['gameId'], play['playId'])
                if fig:
                    plt.show()

# Create analyzer instance
analyzer = PreSnapAnalyzer(
    tracking_data=data['tracking'],
    plays_data=data['plays'],
    players_data=data['players'],
    player_play_data=data['player_play']
)

# Example usage:
game_id = data['tracking']['gameId'].iloc[0]  # Get first game
print(f"\nAnalyzing game {game_id}")
analyzer.analyze_game(game_id)


# Basic analysis with Voronoi visualization
analyzer.analyze_game(game_id)

# Or for a specific play without Voronoi regions
#analyzer.visualize_formation(game_id, play_id, show_voronoi=False)

