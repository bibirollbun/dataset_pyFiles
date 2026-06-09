import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')

# Bootstrap settings
N_BOOTSTRAP = 1000
RANDOM_SEED = 42

print("Libraries loaded!")


# Configuration
DATA_DIR = Path('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train')
FPS = 10
WEEKS = [f'{i:02d}' for i in range(1, 19)]

print(f"Data directory: {DATA_DIR}")


# Load supplementary data
supp_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv', low_memory=False)
print(f"Supplementary data: {len(supp_df):,} plays")


def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def analyze_closure_for_play(receiver_tracking, defender_tracking, flight_frames):
    """
    Calculate Closure Ratio for all defenders on a play.
    Returns list of dicts with closure metrics for each defender.
    """
    if len(receiver_tracking) == 0 or len(defender_tracking) == 0:
        return None
    
    receiver_tracking = receiver_tracking.sort_values('frame_id')
    frames = sorted(receiver_tracking['frame_id'].unique())
    first_frame = frames[0]
    
    rec_frames = receiver_tracking[['frame_id', 'x', 'y']].copy()
    rec_frames.columns = ['frame_id', 'rec_x', 'rec_y']
    
    defender_results = []
    
    for nfl_id, def_group in defender_tracking.groupby('nfl_id'):
        def_group = def_group.sort_values('frame_id')
        def_frames = def_group[['frame_id', 'x', 'y', 'player_name', 'player_position']].copy()
        def_frames.columns = ['frame_id', 'def_x', 'def_y', 'defender_name', 'defender_position']
        
        merged = rec_frames.merge(def_frames, on='frame_id', how='inner')
        
        if len(merged) < 2:
            continue
        
        merged['separation'] = calculate_distance(
            merged['rec_x'], merged['rec_y'],
            merged['def_x'], merged['def_y']
        )
        
        start_separation = merged.iloc[0]['separation']
        end_separation = merged.iloc[-1]['separation']
        min_separation = merged['separation'].min()
        
        min_sep_frame = merged.loc[merged['separation'].idxmin(), 'frame_id']
        min_sep_frame_pct = (min_sep_frame - first_frame) / max(flight_frames - 1, 1)
        
        if start_separation > 0:
            closure_ratio = (start_separation - min_separation) / start_separation
            closure_ratio = max(0, min(1, closure_ratio))
        else:
            closure_ratio = 0
        
        defender_results.append({
            'defender_nfl_id': nfl_id,
            'defender_name': merged.iloc[0]['defender_name'],
            'defender_position': merged.iloc[0]['defender_position'],
            'start_separation': start_separation,
            'end_separation': end_separation,
            'min_separation': min_separation,
            'min_sep_frame_pct': min_sep_frame_pct,
            'closure_ratio': closure_ratio
        })
    
    return defender_results if len(defender_results) > 0 else None


print("Processing functions defined!")


def process_week(week, data_dir, supp_df):
    """Process one week of data for Closure Ratio analysis."""
    input_df = pd.read_csv(data_dir / f'input_2023_w{week}.csv')
    output_df = pd.read_csv(data_dir / f'output_2023_w{week}.csv')
    
    passers = input_df[input_df['player_role'] == 'Passer']
    throw_info = passers.loc[passers.groupby(['game_id', 'play_id'])['frame_id'].idxmax()]
    throw_info = throw_info[['game_id', 'play_id', 'num_frames_output', 'ball_land_x', 'ball_land_y']].copy()
    throw_info.columns = ['game_id', 'play_id', 'flight_frames', 'land_x', 'land_y']
    
    throw_info = throw_info.merge(
        supp_df[['game_id', 'play_id', 'pass_length', 'pass_result',
                 'route_of_targeted_receiver', 'team_coverage_type', 'team_coverage_man_zone']],
        on=['game_id', 'play_id'], how='left'
    )
    throw_info = throw_info.rename(columns={
        'route_of_targeted_receiver': 'route',
        'team_coverage_type': 'coverage_type',
        'team_coverage_man_zone': 'man_zone'
    })
    
    player_info = input_df[['game_id', 'play_id', 'nfl_id', 'player_role',
                            'player_name', 'player_position']].drop_duplicates()
    receivers = player_info[player_info['player_role'] == 'Targeted Receiver']
    defenders = player_info[player_info['player_role'] == 'Defensive Coverage']
    
    all_matchups = []
    
    for _, play in throw_info.iterrows():
        game_id, play_id = play['game_id'], play['play_id']
        flight_frames = play['flight_frames']
        
        play_receivers = receivers[(receivers['game_id'] == game_id) & (receivers['play_id'] == play_id)]
        if len(play_receivers) == 0:
            continue
        
        receiver_nfl_id = play_receivers.iloc[0]['nfl_id']
        receiver_name = play_receivers.iloc[0]['player_name']
        receiver_position = play_receivers.iloc[0]['player_position']
        
        receiver_tracking = output_df[
            (output_df['game_id'] == game_id) & (output_df['play_id'] == play_id) &
            (output_df['nfl_id'] == receiver_nfl_id)
        ]
        
        play_defenders = defenders[(defenders['game_id'] == game_id) & (defenders['play_id'] == play_id)]
        if len(play_defenders) == 0:
            continue
        
        defender_tracking = output_df[
            (output_df['game_id'] == game_id) & (output_df['play_id'] == play_id) &
            (output_df['nfl_id'].isin(play_defenders['nfl_id']))
        ].merge(play_defenders[['nfl_id', 'player_name', 'player_position']], on='nfl_id')
        
        closure_results = analyze_closure_for_play(receiver_tracking, defender_tracking, flight_frames)
        
        if closure_results is None:
            continue
        
        for result in closure_results:
            result['game_id'] = game_id
            result['play_id'] = play_id
            result['receiver_nfl_id'] = receiver_nfl_id
            result['receiver_name'] = receiver_name
            result['receiver_position'] = receiver_position
            result['pass_result'] = play['pass_result']
            result['pass_length'] = play['pass_length']
            result['route'] = play['route']
            result['coverage_type'] = play['coverage_type']
            result['man_zone'] = play['man_zone']
            result['flight_frames'] = flight_frames
            result['is_complete'] = play['pass_result'] == 'C'
            result['is_interception'] = play['pass_result'] == 'IN'
            result['week'] = int(week)
            
            all_matchups.append(result)
    
    return pd.DataFrame(all_matchups)


print("Week processing function defined!")


all_results = []

for week in WEEKS:
    print(f"Processing week {week}...", end=" ")
    try:
        week_results = process_week(week, DATA_DIR, supp_df)
        all_results.append(week_results)
        print(f"{len(week_results):,} matchups")
    except Exception as e:
        print(f"ERROR: {e}")

matchup_df = pd.concat(all_results, ignore_index=True)
print(f"\nTotal matchups: {len(matchup_df):,}")
print(f"Unique plays: {matchup_df.groupby(['game_id', 'play_id']).ngroups:,}")


print("=" * 70)
print("CREATING PLAY-LEVEL DATASET")
print("=" * 70)

# Select primary defender by START separation (closest at release)
play_df = matchup_df.loc[
    matchup_df.groupby(['game_id', 'play_id'])['start_separation'].idxmin()
].copy()

print(f"\nMatchup-level: {len(matchup_df):,} observations (INFLATED - not used for inference)")
print(f"Play-level: {len(play_df):,} observations (PRIMARY - used for all statistical tests)")
print(f"Avg defenders per play: {len(matchup_df) / len(play_df):.1f}")

print(f"\nPlay-level summary:")
print(f"  Mean closure ratio: {play_df['closure_ratio'].mean()*100:.1f}%")
print(f"  Completion rate: {play_df['is_complete'].mean()*100:.1f}%")
print(f"  Interception rate: {play_df['is_interception'].mean()*100:.1f}%")


def bootstrap_ci(df, stat_func, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED):
    """
    Calculate bootstrap confidence interval by resampling plays.
    
    Parameters:
    -----------
    df : DataFrame with play-level data
    stat_func : function that takes df and returns a statistic
    n_bootstrap : number of bootstrap iterations
    seed : random seed
    
    Returns:
    --------
    dict with 'mean', 'std', 'ci_lower', 'ci_upper'
    """
    np.random.seed(seed)
    n_plays = len(df)
    
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        # Resample plays with replacement
        sample = df.sample(n=n_plays, replace=True)
        bootstrap_stats.append(stat_func(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    return {
        'mean': np.mean(bootstrap_stats),
        'std': np.std(bootstrap_stats),
        'ci_lower': np.percentile(bootstrap_stats, 2.5),
        'ci_upper': np.percentile(bootstrap_stats, 97.5)
    }


print("Bootstrap function defined!")


print("=" * 70)
print("CLOSURE RATIO VS OUTCOMES (PLAY-LEVEL)")
print("=" * 70)

# Create closure bins
play_df['closure_bin'] = pd.cut(
    play_df['closure_ratio'],
    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
)

print(f"\n{'Closure Ratio':<15} {'Comp %':>10} {'INT %':>10} {'n':>10}")
print("-" * 50)

for bin_label in ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']:
    subset = play_df[play_df['closure_bin'] == bin_label]
    print(f"{bin_label:<15} {subset['is_complete'].mean()*100:>10.1f} "
          f"{subset['is_interception'].mean()*100:>10.1f} {len(subset):>10,}")


# Bootstrap CI for completion spread
print("\n--- Bootstrap Confidence Intervals ---")

def calc_spread(df):
    """Calculate completion spread between low and high closure."""
    low = df[df['closure_ratio'] < 0.2]['is_complete'].mean()
    high = df[df['closure_ratio'] >= 0.8]['is_complete'].mean()
    return (low - high) * 100  # percentage points

spread_ci = bootstrap_ci(play_df, calc_spread)

print(f"\nCompletion Spread (0-20% vs 80-100% closure):")
print(f"  Point estimate: {calc_spread(play_df):.1f} pp")
print(f"  Bootstrap mean: {spread_ci['mean']:.1f} pp")
print(f"  95% CI: [{spread_ci['ci_lower']:.1f}, {spread_ci['ci_upper']:.1f}] pp")
print(f"  SE: {spread_ci['std']:.1f} pp")

if spread_ci['ci_lower'] > 0:
    print(f"\n✓ 95% CI excludes zero → Effect is statistically significant")


# Chi-square test on play-level data
print("\n--- Statistical Significance (Play-Level) ---")

contingency = pd.crosstab(play_df['closure_ratio'] >= 0.6, play_df['is_complete'])
chi2, p, _, _ = stats.chi2_contingency(contingency)

# Effect size (Cohen's h)
high = play_df[play_df['closure_ratio'] >= 0.6]['is_complete'].mean()
low = play_df[play_df['closure_ratio'] < 0.6]['is_complete'].mean()
cohens_h = 2 * (np.arcsin(np.sqrt(low)) - np.arcsin(np.sqrt(high)))

print(f"\nChi-square test (60%+ closure vs completion):")
print(f"  χ² = {chi2:.1f}")
print(f"  p-value = {p:.2e}")
print(f"  Effect size (Cohen's h) = {cohens_h:.2f}")


print("=" * 70)
print("WITHIN-CONTESTED PLAYS ANALYSIS")
print("=" * 70)

contested = play_df[play_df['min_separation'] < 1.5].copy()

print(f"\nContested plays (min_sep < 1.5 yards): {len(contested):,} ({100*len(contested)/len(play_df):.1f}%)")
print(f"Contested completion rate: {contested['is_complete'].mean()*100:.1f}%")

# Within contested, does closure ratio still predict?
contested['closure_bin'] = pd.cut(
    contested['closure_ratio'],
    bins=[0, 0.4, 0.6, 0.8, 1.0],
    labels=['0-40%', '40-60%', '60-80%', '80-100%']
)

print(f"\n*** WITHIN CONTESTED PLAYS ***")
print(f"{'Closure Ratio':<15} {'Comp %':>10} {'INT %':>10} {'n':>10}")
print("-" * 50)

for bin_label in ['0-40%', '40-60%', '60-80%', '80-100%']:
    subset = contested[contested['closure_bin'] == bin_label]
    if len(subset) > 0:
        print(f"{bin_label:<15} {subset['is_complete'].mean()*100:>10.1f} "
              f"{subset['is_interception'].mean()*100:>10.1f} {len(subset):>10,}")


# Bootstrap CI for within-contested spread
def calc_contested_spread(df):
    """Calculate spread within contested plays."""
    contested = df[df['min_separation'] < 1.5]
    if len(contested) < 50:
        return np.nan
    low = contested[contested['closure_ratio'] < 0.4]['is_complete'].mean()
    high = contested[contested['closure_ratio'] >= 0.8]['is_complete'].mean()
    return (low - high) * 100

contested_ci = bootstrap_ci(play_df, calc_contested_spread)

print(f"\n--- Bootstrap CI for Within-Contested Spread ---")
print(f"Spread (0-40% vs 80-100% within contested):")
print(f"  Point estimate: {calc_contested_spread(play_df):.1f} pp")
print(f"  95% CI: [{contested_ci['ci_lower']:.1f}, {contested_ci['ci_upper']:.1f}] pp")

if contested_ci['ci_lower'] > 0:
    print(f"\n✓ Closure ratio predicts outcomes WITHIN contested plays!")
    print(f"  This confirms the metric measures SKILL, not just 'was play contested'")


print("=" * 70)
print("EXPECTED COMPLETION MODEL")
print("=" * 70)

# Improved model: start_sep, pass_length, flight_frames, end_sep, route, man/zone
features = ['start_separation', 'pass_length', 'flight_frames', 'end_separation']
model_df = play_df.dropna(subset=features + ['is_complete', 'route', 'man_zone']).copy()

route_dummies = pd.get_dummies(model_df['route'], prefix='route', drop_first=True)
mz_dummies = pd.get_dummies(model_df['man_zone'], prefix='cov', drop_first=True)

X = pd.concat([
    model_df[features].reset_index(drop=True),
    route_dummies.reset_index(drop=True),
    mz_dummies.reset_index(drop=True)
], axis=1)
y = model_df['is_complete'].reset_index(drop=True)

# Cross-validation
model = LogisticRegression(max_iter=1000)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

print(f"\nModel: Logistic Regression")
print(f"Features: start_sep, pass_length, flight_frames, end_sep, route, man/zone")
print(f"5-fold CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

# Fit and predict
model.fit(X, y)
model_df = model_df.reset_index(drop=True)
model_df['expected_comp'] = model.predict_proba(X)[:, 1]


# Compare actual vs expected by closure bin
model_df['closure_bin'] = pd.cut(
    model_df['closure_ratio'],
    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
)

print(f"\n{'Closure':<12} {'Actual':>10} {'Expected':>10} {'Diff':>10} {'n':>10}")
print("-" * 55)

for bin_label in ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']:
    subset = model_df[model_df['closure_bin'] == bin_label]
    if len(subset) > 0:
        actual = subset['is_complete'].mean() * 100
        expected = subset['expected_comp'].mean() * 100
        diff = actual - expected
        print(f"{bin_label:<12} {actual:>10.1f} {expected:>10.1f} {diff:>+10.1f} {len(subset):>10,}")

# Key stat
high_closure = model_df[model_df['closure_ratio'] >= 0.8]
outperformance = (high_closure['is_complete'].mean() - high_closure['expected_comp'].mean()) * 100

print(f"\nKey Finding: 80%+ closure defenders allow {high_closure['is_complete'].mean()*100:.1f}% completions")
print(f"             vs {high_closure['expected_comp'].mean()*100:.1f}% expected")
print(f"             = {outperformance:.1f} pp better than expected")


print("=" * 70)
print("RANDOM SPLIT-HALF RELIABILITY")
print("=" * 70)

MIN_MATCHUPS = 15
N_SPLITS = 100

closure_correlations = []
comp_correlations = []

np.random.seed(RANDOM_SEED)

for i in range(N_SPLITS):
    # Random split of plays
    plays = play_df[['game_id', 'play_id']].drop_duplicates().copy()
    plays['split'] = np.random.choice([0, 1], size=len(plays))
    
    play_split = play_df.merge(plays, on=['game_id', 'play_id'])
    
    half_a = play_split[play_split['split'] == 0]
    half_b = play_split[play_split['split'] == 1]
    
    agg_a = half_a.groupby('defender_nfl_id').agg({
        'closure_ratio': 'mean', 'is_complete': 'mean', 'game_id': 'count'
    }).rename(columns={'closure_ratio': 'closure_a', 'is_complete': 'comp_a', 'game_id': 'n_a'})
    
    agg_b = half_b.groupby('defender_nfl_id').agg({
        'closure_ratio': 'mean', 'is_complete': 'mean', 'game_id': 'count'
    }).rename(columns={'closure_ratio': 'closure_b', 'is_complete': 'comp_b', 'game_id': 'n_b'})
    
    merged = agg_a.merge(agg_b, left_index=True, right_index=True)
    merged = merged[(merged['n_a'] >= MIN_MATCHUPS) & (merged['n_b'] >= MIN_MATCHUPS)]
    
    if len(merged) >= 20:
        closure_correlations.append(merged['closure_a'].corr(merged['closure_b']))
        comp_correlations.append(merged['comp_a'].corr(merged['comp_b']))

closure_correlations = np.array(closure_correlations)
comp_correlations = np.array(comp_correlations)

print(f"\nClosure Ratio Reliability ({N_SPLITS} random splits):")
print(f"  Mean r = {np.mean(closure_correlations):.3f}")
print(f"  95% CI: [{np.percentile(closure_correlations, 2.5):.3f}, {np.percentile(closure_correlations, 97.5):.3f}]")
print(f"  Variance explained: {np.mean(closure_correlations)**2*100:.1f}%")

print(f"\nCompletion Rate Reliability:")
print(f"  Mean r = {np.mean(comp_correlations):.3f}")
print(f"  95% CI: [{np.percentile(comp_correlations, 2.5):.3f}, {np.percentile(comp_correlations, 97.5):.3f}]")
print(f"  Variance explained: {np.mean(comp_correlations)**2*100:.1f}%")

ratio = np.mean(closure_correlations)**2 / np.mean(comp_correlations)**2
print(f"\nClosure Ratio explains {ratio:.1f}x more variance than completion rate")


print("=" * 70)
print("CLOSURE RATIO BY POSITION")
print("=" * 70)

position_map = {
    'FS': 'Safety', 'SS': 'Safety', 'S': 'Safety',
    'CB': 'Cornerback', 'DB': 'Cornerback',
    'ILB': 'Linebacker', 'MLB': 'Linebacker', 'LB': 'Linebacker', 'OLB': 'Linebacker'
}

play_df['position_group'] = play_df['defender_position'].map(position_map)

pos_stats = play_df.groupby('position_group').agg({
    'closure_ratio': 'mean',
    'is_complete': 'mean',
    'game_id': 'count'
}).rename(columns={'game_id': 'n'}).sort_values('closure_ratio', ascending=False)

print(f"\n{'Position':<15} {'Avg Closure':>12} {'Comp% Against':>15} {'n':>10}")
print("-" * 55)
for pos, row in pos_stats.iterrows():
    if pd.notna(pos):
        print(f"{pos:<15} {row['closure_ratio']*100:>11.1f}% {row['is_complete']*100:>14.1f}% {int(row['n']):>10,}")


print("=" * 70)
print("TOP DEFENDERS BY CLOSURE RATIO")
print("=" * 70)

MIN_PLAYS = 50

player_stats = play_df.groupby(['defender_nfl_id', 'defender_name', 'defender_position']).agg({
    'closure_ratio': 'mean',
    'is_complete': 'mean',
    'is_interception': 'sum',
    'game_id': 'count'
}).reset_index().rename(columns={'game_id': 'n_plays', 'is_interception': 'interceptions'})

player_stats = player_stats[player_stats['n_plays'] >= MIN_PLAYS]
player_stats = player_stats.sort_values('closure_ratio', ascending=False)

print(f"\nTop 15 (min {MIN_PLAYS} plays as primary defender):")
print(f"{'Rank':<5} {'Player':<25} {'Pos':<5} {'Closure':>10} {'Comp%':>10} {'INT':>5} {'n':>8}")
print("-" * 75)

for i, (_, row) in enumerate(player_stats.head(15).iterrows(), 1):
    print(f"{i:<5} {row['defender_name']:<25} {row['defender_position']:<5} "
          f"{row['closure_ratio']*100:>9.1f}% {row['is_complete']*100:>9.1f}% "
          f"{int(row['interceptions']):>5} {int(row['n_plays']):>8}")


# Export datasets
matchup_df.to_csv('closure_ratio_matchup_data.csv', index=False)
play_df.to_csv('closure_ratio_play_level.csv', index=False)
player_stats.to_csv('closure_ratio_player_stats.csv', index=False)

print("Exported:")
print(f"  closure_ratio_matchup_data.csv ({len(matchup_df):,} matchups - for reference only)")
print(f"  closure_ratio_play_level.csv ({len(play_df):,} plays - PRIMARY DATASET)")
print(f"  closure_ratio_player_stats.csv ({len(player_stats):,} players)")


print("=" * 70)
print("SUMMARY STATISTICS FOR WRITEUP")
print("=" * 70)

high_80 = play_df[play_df['closure_ratio'] >= 0.8]
low_20 = play_df[play_df['closure_ratio'] < 0.2]

print(f"""
METHODOLOGY:
  Analysis level: PLAY-LEVEL ({len(play_df):,} plays)
  Primary defender: Selected by start separation (closest at release)
  Uncertainty: Bootstrap CIs (n={N_BOOTSTRAP}, resampling plays)
  Expected completion model: AUC = {cv_scores.mean():.3f}

KEY FINDINGS:

  1. Completion Spread:
     0-20% closure: {low_20['is_complete'].mean()*100:.1f}% completion
     80-100% closure: {high_80['is_complete'].mean()*100:.1f}% completion
     Spread: {calc_spread(play_df):.1f} pp [95% CI: {spread_ci['ci_lower']:.1f}-{spread_ci['ci_upper']:.1f}]

  2. Within-Contested Analysis (min_sep < 1.5 yards):
     N = {len(contested):,} plays ({100*len(contested)/len(play_df):.1f}%)
     Spread: {calc_contested_spread(play_df):.1f} pp [95% CI: {contested_ci['ci_lower']:.1f}-{contested_ci['ci_upper']:.1f}]
     → Closure ratio predicts WITHIN contested plays

  3. Expected Completion Adjustment:
     80%+ closure actual: {high_closure['is_complete'].mean()*100:.1f}%
     80%+ closure expected: {high_closure['expected_comp'].mean()*100:.1f}%
     Outperformance: {outperformance:.1f} pp

  4. Split-Half Reliability (100 random splits):
     Closure Ratio: r = {np.mean(closure_correlations):.2f} [95% CI: {np.percentile(closure_correlations, 2.5):.2f}-{np.percentile(closure_correlations, 97.5):.2f}]
     Completion Rate: r = {np.mean(comp_correlations):.2f} [95% CI: {np.percentile(comp_correlations, 2.5):.2f}-{np.percentile(comp_correlations, 97.5):.2f}]
     Closure explains {ratio:.1f}x more variance

STATISTICAL TESTS (Play-Level):
  Chi-square (60%+ closure): χ² = {chi2:.1f}, p = {p:.2e}
  Effect size (Cohen's h): {cohens_h:.2f}
""")




