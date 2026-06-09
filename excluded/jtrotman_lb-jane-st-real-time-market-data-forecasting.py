import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'iframe'


title = "Jane Street Real-Time Market Data Forecasting"
subtitle = "Predict financial market responders using real-world data."
slug = "jane-street-real-time-market-data-forecasting"
medal_colors = ['Gold', 'Silver', 'Chocolate']
medal_names = ['GOLD', 'SILVER', 'BRONZE']


MAX_TEAM_SIZE = 5

def read_lb(jfile):
    json_dict = pd.read_json(jfile)
    df = json_dict.publicLeaderboard.apply(pd.Series).set_index("teamId").drop("inTheMoney", axis=1)
    teams_df = json_dict.teams.apply(pd.Series).set_index("teamId")
    df.displayScore = df.displayScore.astype(float)
    df = df.join(teams_df[['teamName', 'submissionCount']])
    return df

def read_team_members_df(jfile):
    json_dict = pd.read_json(jfile)
    df = json_dict.teams.apply(pd.Series).set_index("teamId")
    dfs = [df.teamMembers.str[i].apply(pd.Series).add_prefix(f'user{i}_') for i in range(MAX_TEAM_SIZE)]
    members_df = pd.concat(dfs, axis=1)
    # squeeze this in too:
    members_df['lastSubmissionDate'] = df['lastSubmissionDate']
    return members_df


# p for public lbs
# r for rescore (private) lbs
base = '/kaggle/input/jane-street-real-time-market-data-forecasting-lbs/'
dfs = [
    ( 'p1', read_lb(base + 'jane-street-real-time-market-data-forecasting-250202.json') ),
    ( 'p2', read_lb(base + 'jane-street-real-time-market-data-forecasting-250205.json') ),
    ( 'r1', read_lb(base + 'jane-street-real-time-market-data-forecasting-250211.json') ),
    ( 'r2', read_lb(base + 'jane-street-real-time-market-data-forecasting-250310.json') ),
    ( 'r3', read_lb(base + 'jane-street-real-time-market-data-forecasting-250408.json') ),
    ( 'r4', read_lb(base + 'jane-street-real-time-market-data-forecasting-250512.json') ),
    ( 'r5', read_lb(base + 'jane-street-real-time-market-data-forecasting-250616.json') ),
    ( 'r6', read_lb(base + 'jane-street-real-time-market-data-forecasting-250714.json') ),
]


team_members_df = read_team_members_df(base + 'jane-street-real-time-market-data-forecasting-250202.json')
team_members_df.shape


dfs = [ (tag, df.add_suffix(f"_{tag}")) for tag, df in dfs ]
score_cols = [ (f"displayScore_{tag}") for tag, df in dfs ]
rank_cols = [ (f"rank_{tag}") for tag, df in dfs ]
medal_cols = [ (f"medal_{tag}") for tag, df in dfs ]
subCount_cols = [ (f"submissionCount_{tag}") for tag, df in dfs ]
subId_cols = [ (f"submissionId_{tag}") for tag, df in dfs ]
user_id_cols = [f'user{i}_id' for i in range(MAX_TEAM_SIZE)]


[df.shape for tag, df in dfs]


uni = pd.concat([df for tag, df in dfs] + [team_members_df], axis=1)
uni['teamSize'] = uni[user_id_cols].count(axis=1)
uni['label'] = uni.teamName_r6 + " (" + uni.rank_r6.map(lambda v: f'{v:,.0f}') + ")"
uni['lastSubmissionDate'] = pd.to_datetime(uni['lastSubmissionDate'], format='mixed')
uni.shape


tmp = uni.sort_values('rank_r6').set_index('label')
(tmp[rank_cols].head(50).style
 .format(precision=0)
 .background_gradient(subset=rank_cols[:2], axis=0, cmap='Greens_r')
 .background_gradient(subset=rank_cols[2:], axis=0, cmap='Oranges_r'))


score_counts = uni[score_cols].count()
score_counts.to_frame("Scored Teams").assign(Diff=score_counts.diff())


uni[score_cols].count().plot.bar(title="Team Counts")
plt.xticks(rotation=45);


(uni.groupby(list(score_cols[2:]), dropna=False).submissionId_p1.size()
 .sort_values(ascending=False)
 .to_frame("Num Teams").head(55))


team_stats_df = uni[score_cols].max(axis=1).to_frame("best")
team_stats_df['subCount'] = uni.submissionCount_p1
team_stats_df['teamSize'] = uni.teamSize
team_stats_df['finalRank'] = uni.rank_r6
team_stats_df['lastSubmissionDate'] = uni.lastSubmissionDate
team_stats_df['scoreCount'] = uni.groupby(list(score_cols[2:]), dropna=False).submissionId_p1.transform('size')
team_stats_df['teamName'] = uni['teamName_p1']
team_stats_df['medal'] = uni['medal_r6']
team_stats_df['finalSubCount'] = uni['submissionCount_r6']
team_stats_df['finalScore'] = uni['displayScore_r6']
uni['scoreCount'] = uni.groupby(list(score_cols[2:]), dropna=False).submissionId_p1.transform('size')
team_stats_df.count()


tmp = team_stats_df.query('best>=0')
tmp.plot.scatter('subCount', 'best', logx=True, s=3, figsize=(8,8),
                 title="Submission Count vs Best Score relationship",
                 c=np.where(tmp.scoreCount == 1, 'r', 'b'));


team_stats_df.lastSubmissionDate.dt.date.value_counts().sort_index().tail()


team_stats_df.lastSubmissionDate.dt.date.value_counts().sort_index().plot(figsize=(8,4))
plt.title(f'Last Submission Date for Teams')
plt.xticks(rotation=45);


team_stats_df.lastSubmissionDate.dt.date.value_counts().sort_index().tail(9).plot.bar(figsize=(8,4))
plt.title(f'{title} - lastSubmissionDate')
plt.xticks(rotation=45);


tmp = team_stats_df.query('best>=0')
tmp.plot.scatter('lastSubmissionDate', 'best', s=3, figsize=(9,6),
                 title=f"{title}\nLast Submission Date vs Best Score Relationship",
                 c=np.where(tmp.scoreCount == 1, 'r', 'b'));
plt.xticks(rotation=45);


tmp = team_stats_df.query('best>=0 and finalRank>0').fillna({'medal': 'N/A'})
fig = px.scatter(tmp, 'lastSubmissionDate', 'best',
           hover_name='teamName',
           hover_data={
               'finalRank': True,
               'subCount': True,
               'finalSubCount': True,
               'teamSize': True,
           },
           symbol='medal',
           #symbol_map={
           #   'GOLD': 'circle',
           #   'SILVER': 'triangle-up',
           #   'BRONZE': 'square',
           #   'N/A': 'x',
           #},
           title='Last Submission Date vs Final Score Relationship',
           color='scoreCount')
fig.update_traces(showlegend=False, selector=dict(mode="markers"))


team_stats_df.plot.scatter('finalRank', 'teamSize', s=3, figsize=(10,2),
                           title="Team Sizes",
                           c=np.where(team_stats_df.scoreCount == 1, 'r', 'b'));


tmp = uni.query(medal_cols[-1]+"=='GOLD'").sort_values('rank_r6')
plt.figure(figsize=(12,7))
pd.plotting.parallel_coordinates(tmp, 'label', score_cols, colormap='tab20')
legend_opts = dict(bbox_to_anchor=(1.02, 0, 0.3, 1),
                   loc="upper right",
                   ncol=1,
                   shadow=True,
                   edgecolor="black",
                   mode="expand",
                   borderaxespad=0.)
plt.legend(**legend_opts)
plt.title(f'{title} - Gold Scores Over Time')
plt.xticks(rotation=45);


tmp = uni[~uni.medal_r6.isna()]
plt.figure(figsize=(12,7))
pd.plotting.parallel_coordinates(tmp, 'medal_r6', score_cols, colormap='tab20')
plt.legend().remove()
plt.title(f'{title} - Medal Scores Over Time')
plt.xticks(rotation=45);


train = pd.read_parquet('/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet',
                        columns=['responder_6','weight','date_id'])
train.shape


day_denom_sums = (train.weight * train.responder_6**2).groupby(train.date_id).sum()


day_denom_sums.tail(120).plot(title='Training Data Day $w*y^2$ Sums');


day_denom_sums_20 = day_denom_sums.groupby(day_denom_sums.index//20).sum()
day_denom_sums_20.index *= 20
day_denom_sums_20.tail(6).plot.bar(title='Training Data 20-Day $w*y^2$ Sums');


day_denom_sums_20.tail(24).plot.bar(title='Training Data 20-Day $w*y^2$ Sums');


# Compute per-month R2 scores
def monthly_scores(R2_matrix, d):
    D = np.cumsum(d)
    cum_num = (1 - R2_matrix) * D[None, :]
    num_diff = np.diff(np.concatenate([np.zeros((R2_matrix.shape[0], 1)), cum_num], axis=1), axis=1)
    per_month_R2 = 1 - num_diff / d
    return per_month_R2


tmp = uni.dropna(subset=score_cols).sort_values('rank_r6')
monthly_scores_df = pd.DataFrame(monthly_scores(tmp[score_cols[2:]].clip(lower=0).values, np.ones(6)))
monthly_scores_df.index = tmp.label.values


monthly_scores_df.head()


monthly_scores_df.stack().describe()


monthly_scores_df.head(20).corr().style.background_gradient()


tmp = monthly_scores_df.head(50)
grad_cols = list(tmp.columns)
tmp = tmp.assign(mean=tmp.mean(1), std=tmp.std(1))
(tmp.style.background_gradient(axis=1, subset=grad_cols)
 .background_gradient(cmap='Wistia', subset=['mean'])
 .bar(subset=['std'], color='skyblue', height=80)
 .set_caption("Highlighting Best Month per Team")
)


vertical_headers_table_styles = [
    dict(selector="th.col_heading",
         props=[("writing-mode", "vertical-rl"), 
                ("text-orientation", "mixed"),   # Ensures upright text
                ("vertical-align", "bottom"),   # So text starts from bottom
         ])
]
monthly_scores_df.head(20).T.corr().style.format(precision=2).set_table_styles(
    vertical_headers_table_styles).background_gradient(axis=None, vmin=-1, vmax=1)


%%capture --no-display
sns.clustermap(monthly_scores_df.head(50).T.corr(), figsize=(15,15), cmap='RdYlGn')
fig = plt.title("Correlation Between Top 50 Teams")


monthly_scores_df.rank(axis=0, ascending=False).head(50).astype(int)


tmp = monthly_scores_df.rank(axis=0, ascending=False)
tmp[tmp.min(1)<=10].astype(int)


(monthly_scores_df[tmp.min(1)<=10].style.background_gradient(axis=1)
 .set_caption("Highlighting Best Month per Team"))


uni[uni.teamName_p1.fillna("").str.contains("AngadY")][score_cols[2:]]


uni[uni.teamName_p1.fillna("").str.contains("Japneet Singh")][score_cols[2:]]


colors = uni.medal_r6.map(dict(zip(medal_names, medal_colors))).fillna('deepskyblue')
shakeup = (uni.rank_r6 - uni.rank_p2).abs().mean() / uni.rank_r6.count()
max_rank = uni[rank_cols].max().max()
plt.figure(figsize=(12, 12))
plt.scatter(uni.rank_p1, uni.rank_r6, c=colors, s=3)
plt.title(f'{title} - Shake-up {shakeup:.3f}')
plt.xlabel('Public')
plt.ylabel('Private')
plt.plot((0, max_rank), (0, max_rank), c='k', ls='--', lw=1, alpha=.5);


# Plot bronze teams last otherwise they are almost all hidden
# Also plot smaller submission counts later so they are not covered by bigger points
sortKey = uni.medal_r6.fillna("X") + " " + uni.submissionCount_p1.apply(lambda v: f'{v:3.0f}')
order = sortKey.argsort()[::-1]
tmp = uni.iloc[order]
colors = tmp.medal_r6.map(dict(zip(medal_names, medal_colors))).fillna('deepskyblue')
scoreRange = (-0.001, 0.0145)
plt.figure(figsize=(12, 12))
plt.scatter(tmp.displayScore_p1, tmp.displayScore_r6, c=colors, ec='k', lw=.3, s=tmp.submissionCount_p1/2)
plt.title(f'{title} - Scores')
plt.xlabel('Public')
plt.ylabel('Private')
plt.ylim(*scoreRange);
plt.xlim(*scoreRange);
plt.plot(scoreRange, scoreRange, c='k', ls='--', lw=1, alpha=.5);


tmp = uni.assign(delta=uni.rank_p2 - uni.rank_r6)
cols = ['teamName_p1', 'teamSize', 'medal_r6',
        'rank_p2', 'rank_r6', 'delta',
        'displayScore_p2', 'displayScore_r6',
        'scoreCount']
(tmp.nlargest(30, 'delta')[cols].style
 .format(na_rep='')
 .format(precision=0, subset=['rank_p2', 'rank_r6', 'delta'])
 .bar(subset=['displayScore_r6'], height=80, color='skyblue'))


silver_teams_df = uni[(uni.medal_r6=="SILVER") & (uni.teamSize>=4)].sort_values('rank_r6', ascending=True)
silver_teams_df.shape


silver_teams_df.lastSubmissionDate.dt.date.value_counts()


silver_teams_df.submissionCount_p1.describe()


cols = ['rank_r6', 'teamName_p1', 'teamSize', 'scoreCount', 'submissionCount_p1', 'displayScore_r6', 'lastSubmissionDate', ]
(silver_teams_df[cols].style
 .format({'lastSubmissionDate': "{:%Y.%m.%d}"})
 .format(precision=0, subset=['rank_r6', 'scoreCount', 'submissionCount_p1']))


users_df = silver_teams_df[user_id_cols].stack().astype(int).to_frame("user_id")
users_df.shape


user_act_counts_df = pd.read_csv('/kaggle/input/meta-kaggle-count-user-activities/ActiveUsers.csv')
user_act_counts_df.shape


user_act_counts_df.columns


count_cols = list(user_act_counts_df.columns[user_act_counts_df.columns.str.startswith("Count_")])
# count_cols.remove("Count_UserAchievements_UserId")


users_df = users_df.join(user_act_counts_df.set_index("Id"), on='user_id', how='left')


users_df.describe().T


users_df.PerformanceTier.value_counts().sort_index()


show = [ 'UserName', 'PerformanceTier', 'Age', ]
(users_df.dropna(subset=['UserName']).query("PerformanceTier==0")[show + count_cols].style
 .format(precision=0)
 .set_table_styles(vertical_headers_table_styles)
 .background_gradient(axis=None))


show = [ 'UserName', 'PerformanceTier', 'Age', ]
(users_df.dropna(subset=['UserName'])
 .query("PerformanceTier==1")
 .sample(n=50, random_state=42)[show + count_cols].style
 .format(precision=0)
 .set_table_styles(vertical_headers_table_styles)
 .background_gradient(axis=None))


show = [ 'UserName', 'PerformanceTier', 'Age', ]
(users_df.dropna(subset=['UserName'])
 .query("PerformanceTier>=2")[show + count_cols].style
 .format(precision=0)
 .set_table_styles(vertical_headers_table_styles)
 .background_gradient(axis=None))


teams_with_count_sums_per_user_df = users_df[count_cols].fillna(0).sum(1).unstack()
teams_with_count_sums_per_user_df = teams_with_count_sums_per_user_df.join(silver_teams_df[['teamSize', 'rank_r6', 'teamName_p1']])
(teams_with_count_sums_per_user_df.style
 .background_gradient(subset=user_id_cols, axis=1)
 .format(subset=user_id_cols, na_rep='')
 .format(precision=0)
 .map(lambda x: 'background: white' if pd.isnull(x) else '')
 .set_caption("Highlighting Activity Counts per Member"))


((teams_with_count_sums_per_user_df[user_id_cols] == 0).sum(1).rename("# Inexperienced")
 .value_counts().to_frame("Num Teams"))


plt.rc("axes", edgecolor='#606060')
plt.rc("axes", xmargin=0.01)


tmp = uni.sort_values('displayScore_r6', ascending=False)
plt.figure(figsize=(9,5))
plt.plot(tmp.displayScore_r6.values, c='k', lw=1)

for name, color_code in zip(medal_names, medal_colors):
    span = np.where(tmp.medal_r6==name)[0]
    plt.axvspan(span.min()-.5, span.max()+.5, color=color_code, ec='none', alpha=0.2)
plt.ylim(-.001, .015);
plt.title(f'{title} Score Distribution');


plt.figure(figsize=(9, 5))
plt.plot(tmp.displayScore_r6.values, c='k', lw=1);
for name, color_code in zip(medal_names, medal_colors):
    span = np.where(tmp.medal_r6==name)[0]
    plt.axvspan(span.min()-.5, span.max()+.5, color=color_code, ec='none', alpha=0.2)
plt.ylim(.007, .015);
plt.xlim(-3, span.max());
plt.title(f'{title} Score Distribution - Medal Zone');

