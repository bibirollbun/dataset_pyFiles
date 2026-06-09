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
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

# LOAD
supplementary = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv')

# FUNCTIONS
def enhanced_pursuit_analysis(input_df, output_df, supplementary_df):
    all_plays = []
    for (game_id, play_id) in input_df[['game_id', 'play_id']].drop_duplicates().values:
        try:
            input_play = input_df[(input_df['game_id'] == game_id) & (input_df['play_id'] == play_id)]
            output_play = output_df[(output_df['game_id'] == game_id) & (output_df['play_id'] == play_id)]
            ball_x, ball_y = input_play['ball_land_x'].iloc[0], input_play['ball_land_y'].iloc[0]
            input_play, output_play = input_play.copy(), output_play.copy()
            input_play['dist_to_ball'] = np.sqrt((input_play['x'] - ball_x)**2 + (input_play['y'] - ball_y)**2)
            output_play['dist_to_ball'] = np.sqrt((output_play['x'] - ball_x)**2 + (output_play['y'] - ball_y)**2)
            defenders_input = input_play[input_play['player_role'] == 'Defensive Coverage']
            defenders_output = output_play[output_play['nfl_id'].isin(defenders_input['nfl_id'])]
            throw_moment = defenders_input[defenders_input['frame_id'] == defenders_input['frame_id'].max()]
            for defender_id in throw_moment['nfl_id'].unique():
                df = defenders_output[defenders_output['nfl_id'] == defender_id].sort_values('frame_id')
                if len(df) > 0:
                    dist_at_throw = throw_moment[throw_moment['nfl_id'] == defender_id]['dist_to_ball'].values[0]
                    min_dist = df['dist_to_ball'].min()
                    pos = df[['x', 'y']].values
                    actual_dist = np.sum(np.sqrt(np.sum(np.diff(pos, axis=0)**2, axis=1)))
                    context = supplementary_df[(supplementary_df['game_id'] == game_id) & (supplementary_df['play_id'] == play_id)]
                    all_plays.append({
                        'game_id': game_id, 'play_id': play_id, 'nfl_id': defender_id,
                        'player_name': throw_moment[throw_moment['nfl_id'] == defender_id]['player_name'].values[0],
                        'min_dist': min_dist,
                        'distance_closed': dist_at_throw - df['dist_to_ball'].iloc[-1],
                        'frames_to_closest': df[df['dist_to_ball'] == min_dist]['frame_id'].iloc[0] - df['frame_id'].min(),
                        'pass_result': context['pass_result'].values[0] if len(context) > 0 else None,
                        'coverage_type': context['team_coverage_man_zone'].values[0] if len(context) > 0 else None,
                        'pass_length': context['pass_length'].values[0] if len(context) > 0 else None
                    })
        except: pass
    return pd.DataFrame(all_plays)

def calculate_dbhi(df):
    df['proximity_score'] = 100 * np.exp(-df['min_dist'] / 2.0)
    df['distance_closed_score'] = np.clip((df['distance_closed'] - df['distance_closed'].quantile(0.1)) / (df['distance_closed'].quantile(0.9) - df['distance_closed'].quantile(0.1)) * 100, 0, 100)
    max_frames = df['frames_to_closest'].quantile(0.9)
    df['reaction_score'] = 100 * (1 - df['frames_to_closest'] / max_frames).clip(0, 1)
    df['dbhi'] = 0.70 * df['proximity_score'] + 0.20 * df['distance_closed_score'] + 0.10 * df['reaction_score']
    return df

def process_all_weeks(sup_df):
    all_weeks = []
    for week in range(1, 19):
        try:
            print(f'Week {week}...', end=' ')
            inp = pd.read_csv(f'/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/input_2023_w{week:02d}.csv')
            out = pd.read_csv(f'/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/output_2023_w{week:02d}.csv')
            week_data = enhanced_pursuit_analysis(inp, out, sup_df)
            week_data['week'] = week
            all_weeks.append(week_data)
            print('✓')
        except Exception as e: print(f'Error')
    return pd.concat(all_weeks, ignore_index=True)

# PROCESS
print('Processing 2023 season...')
all_season_pursuit = process_all_weeks(supplementary)
print(f'\n✓ Pursuits: {len(all_season_pursuit):,}')
print(f'✓ Players: {all_season_pursuit["nfl_id"].nunique()}')
print(f'✓ Plays: {all_season_pursuit[["game_id", "play_id"]].drop_duplicates().shape[0]:,}')

# CALCULATE DBHI
print('\nCalculating DBHI...')
all_season_pursuit = calculate_dbhi(all_season_pursuit)
print('✓ Done')

# RESULTS
print('\n' + '='*70)
print('DBHI BY PASS OUTCOME')
print('='*70)
print(all_season_pursuit.groupby('pass_result')['dbhi'].agg(['count', 'mean', 'median']).round(2))

closest_defenders = all_season_pursuit.loc[all_season_pursuit.groupby(['game_id', 'play_id'])['min_dist'].idxmin()]
print('\nCLOSEST DEFENDER DISTANCE:')
print(closest_defenders.groupby('pass_result')['min_dist'].agg(['count', 'mean', 'median']).round(2))

print('\nCLOSEST DEFENDER DBHI:')
print(closest_defenders.groupby('pass_result')['dbhi'].agg(['mean', 'median']).round(2))

ps = all_season_pursuit.groupby(['nfl_id', 'player_name']).agg({'dbhi': ['mean', 'max', 'count'], 'min_dist': 'mean', 'distance_closed': 'mean', 'pass_result': lambda x: (x == 'IN').sum()}).round(2)
ps.columns = ['avg_dbhi', 'max_dbhi', 'opps', 'avg_min_dist', 'avg_dist_closed', 'ints']
ps_f = ps[ps['opps'] >= 50].sort_values('avg_dbhi', ascending=False)
print('\nTOP 10 BALL HAWKS:')
print(ps_f.head(10))

cov_data = all_season_pursuit[all_season_pursuit['coverage_type'].notna()]
print('\nDBHI BY COVERAGE:')
print(cov_data.groupby('coverage_type')['dbhi'].agg(['mean', 'count']).round(2))

# PLOTS
fig, ax = plt.subplots(figsize=(10, 6))
closest_defenders[closest_defenders['pass_result'].isin(['C', 'I', 'IN'])].boxplot(column='min_dist', by='pass_result', ax=ax)
ax.set_title('Closest Defender Distance by Outcome', fontsize=13, fontweight='bold')
ax.set_xlabel('Pass Result', fontsize=11)
ax.set_ylabel('Distance (yards)', fontsize=11)
ax.axhline(y=2, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
closest_defenders[closest_defenders['pass_result'].isin(['C', 'I', 'IN'])].boxplot(column='dbhi', by='pass_result', ax=ax)
ax.set_title('DBHI by Pass Outcome', fontsize=13, fontweight='bold')
ax.set_xlabel('Pass Result', fontsize=11)
ax.set_ylabel('DBHI', fontsize=11)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(12, 8))
top_20 = ps_f.head(20)
bars = ax.barh(range(20), top_20['avg_dbhi'].values)
colors = top_20['ints'].values
norm = plt.Normalize(vmin=0, vmax=colors.max())
for i, (bar, color) in enumerate(zip(bars, colors)):
    bar.set_color(plt.cm.RdYlGn(norm(color)))
ax.set_yticks(range(20))
ax.set_yticklabels(top_20.index.get_level_values('player_name').values, fontsize=9)
ax.set_xlabel('Avg DBHI', fontsize=11)
ax.set_title('Top 20 Ball Hawks', fontsize=13, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
cb = closest_defenders[closest_defenders['coverage_type'].notna()]
cb.boxplot(column='min_dist', by='coverage_type', ax=axes[0])
axes[0].set_title('Distance: Man vs Zone', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Distance (yards)', fontsize=11)
axes[0].axhline(y=2, color='red', linestyle='--', linewidth=2, alpha=0.7)
cb.boxplot(column='dbhi', by='coverage_type', ax=axes[1])
axes[1].set_title('DBHI: Man vs Zone', fontsize=12, fontweight='bold')
axes[1].set_ylabel('DBHI', fontsize=11)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

all_season_pursuit['pass_dist_cat'] = pd.cut(all_season_pursuit['pass_length'], bins=[-50, 0, 10, 20, 100], labels=['Behind LOS', 'Short (0-10)', 'Medium (10-20)', 'Deep (20+)'])
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
all_season_pursuit[all_season_pursuit['pass_dist_cat'].notna()].boxplot(column='dbhi', by='pass_dist_cat', ax=axes[0])
axes[0].set_title('DBHI by Pass Distance', fontsize=12, fontweight='bold')
axes[0].set_ylabel('DBHI', fontsize=10)
plt.sca(axes[0])
plt.xticks(rotation=45)
cd = all_season_pursuit.loc[all_season_pursuit.groupby(['game_id', 'play_id'])['min_dist'].idxmin()]
cd.boxplot(column='min_dist', by='pass_dist_cat', ax=axes[1])
axes[1].set_title('Closest Distance by Pass Length', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Distance (yards)', fontsize=10)
axes[1].axhline(y=2, color='red', linestyle='--', linewidth=2, alpha=0.7)
plt.sca(axes[1])
plt.xticks(rotation=45)
ds = cd.groupby('pass_dist_cat')['pass_result'].apply(lambda x: pd.Series({'Complete': (x == 'C').sum() / len(x) * 100, 'Incomplete': (x == 'I').sum() / len(x) * 100, 'Interception': (x == 'IN').sum() / len(x) * 100}))
ds.plot(kind='bar', stacked=True, ax=axes[2], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[2].set_title('Outcomes by Distance', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Percentage', fontsize=10)
axes[2].legend(title='Outcome')
plt.sca(axes[2])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# EXPORT
ps_f.to_csv('dbhi_player_rankings.csv')
all_season_pursuit.to_csv('all_pursuit_data.csv', index=False)
closest_defenders.to_csv('closest_defender_analysis.csv', index=False)
print('\n✓ Analysis complete!')


