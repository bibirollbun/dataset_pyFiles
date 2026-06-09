# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import duckdb
con = duckdb.connect(database=":memory:")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib.pyplot as plt
# load tracking data (replace with your real file)

import glob
# Grab all weekly input files
path = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train"
input_files = sorted(glob.glob(f"{path}/input_2023_w*.csv"))
output_files = sorted(glob.glob(f"{path}/output_2023_w*.csv"))
input_df = pd.concat([pd.read_csv(f) for f in input_files], ignore_index=True)
output_df = pd.concat([pd.read_csv(f) for f in output_files], ignore_index=True)

print(input_df.shape, output_df.shape)


def create_df_gcr(input_df):
    df = con.execute("""SELECT game_id, play_id, nfl_id, type,
                        CASE WHEN type = 'input' THEN frame_id ELSE frame_id + max_inpt_frame END AS frame_id, x, y, player_role, player_side, player_name, player_position, 
                        ball_land_x, ball_land_y
                        FROM (
                            SELECT *,
                                   MAX(frame_id) OVER (PARTITION BY game_id, play_id) AS max_inpt_frame
                            FROM (
                                SELECT game_id, play_id, nfl_id, frame_id, x, y, 'input' AS type, player_role, player_side, player_name, player_position, ball_land_x, ball_land_y
                                FROM input_df WHERE player_to_predict = 'True'
                                UNION ALL
                                SELECT game_id, play_id, nfl_id, frame_id, x, y, 'output' AS type, NULL as player_role, NULL as player_side, NULL as player_name, NULL as player_position, 
                                       NULL as ball_land_x, NULL as ball_land_y
                                FROM output_df
                            ) t
                        ) t2
                        ORDER BY game_id, play_id, nfl_id, frame_id""").fetchdf()
    for col in ['player_role', 'player_side', 'player_name', 'player_position', 'ball_land_x', 'ball_land_y']:
        df[col] = df[col].ffill()
    temp_df = input_df[['game_id', 'play_id', 'nfl_id', 'frame_id','s']]
    df = df.merge(temp_df, on=['game_id', 'play_id', 'nfl_id', 'frame_id'], how='left')
    # ----------------------------------------------------
    # 1. Keep only defenders
    # ----------------------------------------------------
    def_df = df[df['player_side'] == 'Defense'].copy()

    # Precompute distance to ball at each frame
    def_df['dist_to_ball'] = np.sqrt(
        (def_df['x'] - def_df['ball_land_x'])**2 +
        (def_df['y'] - def_df['ball_land_y'])**2
    )

    # ----------------------------------------------------
    # 2. Compute throw frame (first) and arrival frame (last)
    # ----------------------------------------------------
    def_throw = (
        def_df[def_df['type'] == 'input']
        .sort_values('frame_id')
        .groupby(['game_id', 'play_id', 'nfl_id'])
        .last()
        .reset_index()
        .rename(columns={
            'frame_id': 'frame_throw',
            'x': 'x_throw',
            'y': 'y_throw'
        })
    )
    
    # last output frame (arrival)
    def_arrive = (
        def_df[def_df['type'] == 'output']
        .sort_values('frame_id')
        .groupby(['game_id', 'play_id', 'nfl_id'])
        .last()
        .reset_index()
        .rename(columns={
            'frame_id': 'frame_arrive',
            'x': 'x_arrive',
            'y': 'y_arrive'
        })
    )

    # ----------------------------------------------------
    # 3. Build the clean throw â†’ arrival table
    # ----------------------------------------------------
    def_gap = def_throw[['game_id','play_id','nfl_id','ball_land_x','ball_land_y']].copy()
    def_gap = def_gap.rename(columns={})

    # Add throw info
    def_gap['x_throw']      = def_throw['x_throw']
    def_gap['y_throw']      = def_throw['y_throw']
    def_gap['frame_throw']  = def_throw['frame_throw']
    def_gap['d_throw']      = np.sqrt(
        (def_gap['x_throw'] - def_gap['ball_land_x'])**2 +
        (def_gap['y_throw'] - def_gap['ball_land_y'])**2
    )

    # Add arrival info
    def_gap['x_arrive']     = def_arrive['x_arrive']
    def_gap['y_arrive']     = def_arrive['y_arrive']
    def_gap['frame_arrive'] = def_arrive['frame_arrive']
    def_gap['d_arrive']     = np.sqrt(
        (def_gap['x_arrive'] - def_gap['ball_land_x'])**2 +
        (def_gap['y_arrive'] - def_gap['ball_land_y'])**2
    )

    # ----------------------------------------------------
    # 4. Compute gap closing rate
    # ----------------------------------------------------
    def_gap['frames_in_air'] = def_gap['frame_arrive'] - def_gap['frame_throw']
    def_gap['time_in_air']   = def_gap['frames_in_air'] / 10  # FPS = 10

    # Avoid divide-by-zero
    def_gap = def_gap[def_gap['time_in_air'] > 0].copy()
    def_gap['gap_closing_rate'] = (def_gap['d_throw'] - def_gap['d_arrive']) / def_gap['time_in_air']

    def_gap = def_gap[def_gap['gap_closing_rate'] > 0].copy()
    
    # ----------------------------------------------------
    # 5. GCR Labels
    # ----------------------------------------------------
    # CODE: incomp.gap_closing_rate.describe(percentiles=[.25, .5])
    # USED based off histogram below 25 percentile and 50 percentile
    def_gap['GCR_label'] = pd.cut(
        def_gap['gap_closing_rate'],
        bins=[0, 2.36, 4.05, 99],
        labels=['Slow Close', 'Average Close', 'Fast Close']
    )

    # ----------------------------------------------------
    # 6. Merge supplementary context
    # ----------------------------------------------------
    supplementary = pd.read_csv(
        "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv"
    )

    # Only join columns that exist
    gcr_merged = def_gap.merge(
        supplementary,
        on=['game_id','play_id'],
        how='left'
    )

    # ----------------------------------------------------
    # 7. Add player top speed + label
    # ----------------------------------------------------
    top_speed = (
        df.groupby(['game_id', 'play_id', 'nfl_id'])['s']
        .max()
        .reset_index()
        .rename(columns={'s': 'top_speed'})
    )

    gcr_merged = gcr_merged.merge(top_speed, on=['game_id','play_id','nfl_id'], how='left')

    gcr_merged['top_speed_label'] = pd.cut(
        gcr_merged['top_speed'],
        bins=[0.22, 3.25, 4.26, 5.47, 10.1],
        labels=['1', '2', '3', '4']
    )

    gcr_merged = gcr_merged[gcr_merged.team_coverage_type != 'PREVENT']

    return gcr_merged.drop_duplicates().reset_index(drop=True)
gcr_merged = create_df_gcr(input_df)
gcr_merged


gcr_merged['is_complete'] = (gcr_merged['pass_result'] == 'C')
incomp = gcr_merged[gcr_merged["is_complete"] == 0]

# Freedman-Diaconis rule
data = incomp['gap_closing_rate']
iqr = np.percentile(data, 75) - np.percentile(data, 25)
bin_width = 2 * iqr / np.cbrt(len(data))
num_bins = int((data.max() - data.min()) / bin_width)

print('bins:', num_bins)
counts, bin_edges = np.histogram(incomp['gap_closing_rate'], bins=num_bins)

hist_df = pd.DataFrame({
    "bin_start": bin_edges[:-1],
    "bin_end": bin_edges[1:],
    "count": counts
})

hist_cnt = hist_df['count'].sum()

hist_df["cuml_count"] = hist_df["count"].cumsum()
hist_df['pct'] = hist_df['cuml_count'] / hist_cnt

hist_df.sort_values('count', ascending=False)
display(hist_df.head(15))

sns.histplot(incomp['gap_closing_rate'], bins=num_bins, kde=True)
plt.title("Distribution of Gap Closing Rate Across All Defenders")
plt.xlabel("Gap Closing Rate (yards/sec)")
plt.ylabel("Frequency")
plt.show()


supplementary = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv")
df = gcr_merged.copy()
df = con.execute("""SELECT df.*, t2.yardline_number, t2.yardline_side FROM df JOIN supplementary t2 ON df.game_id = t2.game_id AND df.play_id = t2.play_id
                    WHERE df.play_action = 'False' AND df.offense_formation = 'SHOTGUN'""").fetchdf()
df['abs_yardline'] = np.where(df.possession_team != df.yardline_side,100 - df.yardline_number, df.yardline_number)
df['distance_label'] = pd.cut(
    df['yards_to_go'],
    bins=[-1, 3.5, 6.5, 100],
    labels=['Short', 'Medium', 'Long']
)
df['field_third'] = pd.cut(
    df['abs_yardline'],
    bins=[-1, 33, 66, 100],
    labels=['own', 'middle', 'opposition']
)
display(con.execute("""SELECT GCR_label, MIN(gap_closing_rate), MAX(gap_closing_rate), COUNT(*) as num_of_plays FROM df GROUP BY GCR_label""").fetchdf())


display(con.execute(f"""SELECT 
                        GCR_label,
                        pass_result,
                        COUNT(*) as num_of_plays,
                        SUM(COUNT(*)) OVER (PARTITION BY GCR_label) AS GCR_label_totals,
                        COUNT(*) / GCR_label_totals as pct
                        FROM df
                        GROUP BY GCR_label, pass_result
                        ORDER BY GCR_label, pass_result
                        """).fetchdf())


df2 = con.execute(f"""SELECT t1.*, 
                      MAX(t2.s) OVER (PARTITION BY t1.game_id, t1.play_id, t1.nfl_id) as top_speed,
                      MAX(t2.a) OVER (PARTITION BY t1.game_id, t1.play_id, t1.nfl_id) as top_acceleration
                      FROM gcr_merged t1 
                      JOIN input_df t2 ON t1.game_id = t2.game_id AND t1.play_id = t2.play_id AND t1.nfl_id = t2.nfl_id
                      """).fetchdf()
df2['top_speed_label'] = pd.cut(
    df2['top_speed'],
    bins=[0.13, 3.33, 4.36, 5.53, 9.75],
    labels=['1', '2', '3', '4']
)

df2 = df2[['game_id', 'play_id', 'nfl_id', 'top_speed', 'top_speed_label']].reset_index(drop=True).drop_duplicates()
df = con.execute("""SELECT * FROM df JOIN df2 ON df.game_id = df2.game_id AND df.play_id = df2.play_id AND df.nfl_id = df2.nfl_id
                    """).fetchdf()
df = df.drop(['game_id_1', 'play_id_1', 'nfl_id_1'], axis=1)

print('Slower top speed = slower gap closing rate')
print('Faster top speed = faster gap closing rate')
display(con.execute(f"""SELECT 
                        GCR_label,
                        top_speed_label,
                        COUNT(*) as num_of_plays,
                        SUM(COUNT(*)) OVER (PARTITION BY GCR_label) AS GCR_label_totals,
                        COUNT(*) / GCR_label_totals as pct
                        FROM df
                        GROUP BY GCR_label, top_speed_label
                        ORDER BY GCR_label, top_speed_label
                        """).fetchdf())


df['is_complete'] = (df['pass_result'] == 'C')
incomp = df[(df["is_complete"] == 0) & (df['gap_closing_rate'] >= 4.05)]

# Freedman-Diaconis rule
data = incomp['top_speed']
iqr = np.percentile(data, 75) - np.percentile(data, 25)
bin_width = 2 * iqr / np.cbrt(len(data))
num_bins = int((data.max() - data.min()) / bin_width)

print('bins:', num_bins)
counts, bin_edges = np.histogram(incomp['top_speed'], bins=num_bins)

hist_df = pd.DataFrame({
    "bin_start": bin_edges[:-1],
    "bin_end": bin_edges[1:],
    "count": counts
})

hist_cnt = hist_df['count'].sum()

hist_df["cuml_count"] = hist_df["count"].cumsum()
hist_df['pct'] = hist_df['cuml_count'] / hist_cnt

hist_df.sort_values('cuml_count')
display(hist_df)

sns.histplot(incomp['top_speed'], bins=num_bins, kde=True)
plt.title("Distribution of Top Speed Across All Defenders")
plt.xlabel("Top Speed (yards/sec)")
plt.ylabel("Frequency")
plt.show()


print('Fast min top_speed:', hist_df[(hist_df['pct'] >= 0.50)].bin_end.min())
display(con.execute(f"""WITH sub AS (SELECT *, CASE WHEN gap_closing_rate >= 4.05 THEN 'Fast' ELSE 'Slow' END as GCR_lbl FROM df) 

                        SELECT 
                        GCR_lbl,
                        pass_result,
                        COUNT(*) as num_of_plays,
                        SUM(COUNT(*)) OVER (PARTITION BY GCR_lbl) AS GCR_label_totals,
                        COUNT(*) / GCR_label_totals as pct
                        FROM sub
                        GROUP BY GCR_lbl, pass_result
                        ORDER BY GCR_lbl DESC, pass_result
                        """).fetchdf())


for i in sorted(df.top_speed_label.unique().tolist()): 
    print(i, df[(df.top_speed_label == i)].shape[0], 'plays')
    display(con.execute(f"""WITH sub AS (SELECT *, CASE WHEN gap_closing_rate >= 4.05 THEN 'Fast' ELSE 'Slow' END as GCR_lbl FROM df) 

                            SELECT 
                            GCR_lbl,
                            pass_result,
                            COUNT() as num_of_plays,
                            SUM(COUNT()) OVER (PARTITION BY GCR_lbl) AS GCR_label_totals,
                            COUNT() / GCR_label_totals as pct
                            FROM sub
                            WHERE top_speed_label = '{i}'
                            GROUP BY GCR_lbl, pass_result
                            ORDER BY GCR_lbl DESC, pass_result
                            """).fetchdf())


df_coverage_heatmap = pd.DataFrame()
for i in df.team_coverage_type.unique():
    print(i, df[(df.team_coverage_type == i)].shape[0], 'plays')
    df_temp = con.execute(f"""WITH sub AS (SELECT *, CASE WHEN gap_closing_rate >= 4.05 THEN 'Fast' ELSE 'Slow' END as GCR_lbl FROM df) 
    
                            SELECT 
                            GCR_lbl,
                            pass_result,
                            COUNT(*) as num_of_plays,
                            SUM(COUNT(*)) OVER (PARTITION BY GCR_lbl) AS GCR_label_totals,
                            COUNT(*) / GCR_label_totals as pct
                            FROM sub
                            WHERE team_coverage_type = '{i}'
                            GROUP BY GCR_lbl, pass_result
                            ORDER BY GCR_lbl DESC, pass_result
                            """).fetchdf()
    display(df_temp)
    df_temp.insert(0, 'play_type', i)
    df_coverage_heatmap = pd.concat([df_coverage_heatmap, df_temp]).reset_index(drop=True)
df_coverage_heatmap.set_index('play_type', inplace=True)


# Pivot the dataframe
pivot = df_coverage_heatmap.pivot_table(
    index="play_type",
    columns=["GCR_lbl", "pass_result"],
    values="pct"
)

# Flatten columns (optional)
pivot.columns = [f"{g}_{p}" for g, p in pivot.columns]

# Explicit order
desired_order = ["Slow_C", "Fast_C", "Slow_I", "Fast_I", "Slow_IN", "Fast_IN"]

# Reorder
pivot = pivot.reindex(columns=desired_order)

# Plot
plt.figure(figsize=(12, 8))
sns.heatmap(pivot, annot=True, cmap="Reds_r")
plt.title("Coverage Heatmap by Play Type")
plt.show()



gcr_merged["gcr_bin"] = (gcr_merged["gap_closing_rate"] >= 4.05).map({True: "Fast", False: "Slow"})

gcr_outcome = (
    gcr_merged.groupby(["gcr_bin", "pass_result"])
    .size()
    .groupby(level=0)
    .apply(lambda x: x / x.sum())
    .unstack(fill_value=0)
)

gcr_outcome.index = gcr_outcome.index.droplevel(1)

gcr_outcome.index.name = "gcr_bin"
gcr_outcome = gcr_outcome.reindex(["Slow", "Fast"])

ax = gcr_outcome.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Pass Outcome Distribution by GCR Category")
plt.ylabel("Proportion of Throws")
plt.xlabel("GCR Category")
plt.legend(title="Pass Result")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,8))

hb = plt.hexbin(
    df["gap_closing_rate"],
    df["top_speed"],
    C=(df["pass_result"] == "I") | (df["pass_result"] == "IN"),
    reduce_C_function=np.mean,
    gridsize=35,
    cmap="Reds"
)

plt.axvline(4.05, color="black", ls="--")
plt.axhline(5.11, color="blue", ls="--")

plt.colorbar(hb, label="Probability of Negative Outcome (I or IN)")
plt.xlabel("Gap Closing Rate (yds/s)")
plt.ylabel("Top Speed (yds/s)")
plt.title("Likelihood of Negative Outcome by GCR and Top Speed (Hexbin)")

plt.show()

