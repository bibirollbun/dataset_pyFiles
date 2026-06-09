import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

!pip install --upgrade plotly

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("Libraries loaded successfully!")


# Load both weeks' data for comparison
df_w2 = pd.read_csv('/kaggle/input/division-b-leaderboard-w3/Leaderboard_B_W2.csv')
df_w3 = pd.read_csv('/kaggle/input/division-b-leaderboard-w3/Leaderboard_B_W3.csv')

print(f"Week 2 leaderboard: {len(df_w2)} teams")
print(f"Week 3 leaderboard: {len(df_w3)} teams")
print(f"New teams this week: {len(df_w3) - len(df_w2)}")

print("\nğŸ“Š WEEK 3 TOP 5 TEAMS:")
print("=" * 60)
top_5 = df_w3.head()
for _, row in top_5.iterrows():
    print(f"{row['Rank']:2d}. {row['TeamName'][:25]:<25} | Score: {row['Score']:.6f}")



# Week-over-week metrics
metrics_comparison = {
    'Metric': ['Total Entrants', 'Active Participants', 'Competing Teams', 'Total Submissions'],
    'Week 2': [86, 8, 6, 14],
    'Week 3': [158, 21, 17, 47],
    'Growth': [72, 13, 11, 33],
    'Growth %': [83.7, 162.5, 183.3, 235.7]
}

metrics_df = pd.DataFrame(metrics_comparison)

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Growth in Numbers', 'Growth Percentage', 'Submission Trends', 'Team Activity'),
    specs=[[{'type': 'bar'}, {'type': 'bar'}], 
           [{'type': 'scatter'}, {'type': 'histogram'}]]
)


# Growth in absolute numbers
fig.add_trace(
    go.Bar(x=metrics_df['Metric'], y=metrics_df['Week 2'], name='Week 2', marker_color='lightblue'),
    row=1, col=1
)
fig.add_trace(
    go.Bar(x=metrics_df['Metric'], y=metrics_df['Week 3'], name='Week 3', marker_color='darkblue'),
    row=1, col=1
)

# Growth percentage
fig.add_trace(
    go.Bar(x=metrics_df['Metric'], y=metrics_df['Growth %'], 
           name='Growth %', marker_color='green', showlegend=False),
    row=1, col=2
)

# Submission trends
weeks = ['Week 2', 'Week 3']
submissions = [14, 47]
fig.add_trace(
    go.Scatter(x=weeks, y=submissions, mode='lines+markers', 
               name='Submissions', line=dict(width=4), showlegend=False),
    row=2, col=1
)

# Team activity distribution
fig.add_trace(
    go.Histogram(x=df_w3['SubmissionCount'], nbinsx=15, 
                 name='Team Activity', marker_color='orange', showlegend=False),
    row=2, col=2
)

fig.update_layout(
    height=800,
    title_text="ğŸ“ˆ Week 2 â†’ Week 3 Growth Analysis",
    showlegend=True
)

fig.show()

print("ğŸš€ GROWTH HIGHLIGHTS:")
print(f"   â€¢ Submissions surged by {metrics_df.loc[3, 'Growth %']:.1f}% - highest growth metric")
print(f"   â€¢ Active participants increased by {metrics_df.loc[1, 'Growth %']:.1f}%")
print(f"   â€¢ Competition intensity: {(47/17):.1f} submissions per team avg")


# Merge dataframes to track position changes
df_w2['Week'] = 2
df_w3['Week'] = 3


# Find teams that existed in both weeks
w2_teams = set(df_w2['TeamName'])
w3_teams = set(df_w3['TeamName'])
continuing_teams = w2_teams.intersection(w3_teams)
new_teams = w3_teams - w2_teams

print(f"ğŸ“Š LEADERBOARD DYNAMICS:")
print(f"   â€¢ Continuing teams: {len(continuing_teams)}")
print(f"   â€¢ New teams this week: {len(new_teams)}")
print(f"   â€¢ Teams that stopped competing: {len(w2_teams - w3_teams)}")


# Score improvements for continuing teams
improvements = []
for team in continuing_teams:
    w2_score = df_w2[df_w2['TeamName'] == team]['Score'].iloc[0]
    w3_score = df_w3[df_w3['TeamName'] == team]['Score'].iloc[0]
    w2_rank = df_w2[df_w2['TeamName'] == team]['Rank'].iloc[0]
    w3_rank = df_w3[df_w3['TeamName'] == team]['Rank'].iloc[0]
    
    improvements.append({
        'TeamName': team,
        'W2_Score': w2_score,
        'W3_Score': w3_score,
        'Score_Improvement': w3_score - w2_score,
        'W2_Rank': w2_rank,
        'W3_Rank': w3_rank,
        'Rank_Change': w2_rank - w3_rank  # Positive = improved rank
    })

improvements_df = pd.DataFrame(improvements)
top_improvers = improvements_df.nlargest(len(improvements_df), 'Score_Improvement')  

# Show all
print(f"\nğŸ�† TOP SCORE IMPROVERS (Week 2 â†’ Week 3):")
print("=" * 70)
for _, row in top_improvers.iterrows():
    print(f"{row['TeamName'][:20]:<20} | +{row['Score_Improvement']:.6f} | Rank: {row['W2_Rank']} â†’ {row['W3_Rank']}")



# Visualization of score improvements
fig = go.Figure()

# Continuing teams
fig.add_trace(go.Scatter(
    x=improvements_df['W2_Score'],
    y=improvements_df['W3_Score'],
    mode='markers',
    marker=dict(size=8, color='blue', opacity=0.7),
    name='Continuing Teams',
    text=improvements_df['TeamName'],
    hovertemplate='<b>%{text}</b><br>Week 2: %{x:.6f}<br>Week 3: %{y:.6f}<extra></extra>'
))


# New teams
new_teams_data = df_w3[df_w3['TeamName'].isin(new_teams)]
fig.add_trace(go.Scatter(
    x=[0.5] * len(new_teams_data),  # Placeholder x-value for new teams
    y=new_teams_data['Score'],
    mode='markers',
    marker=dict(size=8, color='red', opacity=0.7),
    name='New Teams',
    text=new_teams_data['TeamName'],
    hovertemplate='<b>%{text}</b><br>New Team<br>Score: %{y:.6f}<extra></extra>'
))


# Add diagonal line for reference (no improvement)
fig.add_trace(go.Scatter(
    x=[0.5, 1.0],
    y=[0.5, 1.0],
    mode='lines',
    line=dict(dash='dash', color='gray'),
    name='No Improvement Line',
    showlegend=False
))

fig.update_layout(
    title="ğŸ�¯ Team Performance Evolution (Week 2 â†’ Week 3)",
    xaxis_title="Week 2 Score",
    yaxis_title="Week 3 Score",
    height=600
)

fig.show()


# Convert last submission date for analysis
df_w3['LastSubmissionDate'] = pd.to_datetime(df_w3['LastSubmissionDate'])
df_w3['DaysAgo'] = (datetime.now() - df_w3['LastSubmissionDate']).dt.days


# Current standings visualization
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Top 15 Teams Performance', 'Score Distribution'),
    specs=[[{'type': 'bar'}, {'type': 'histogram'}]]
)

top_15 = df_w3.head(15)
fig.add_trace(
    go.Bar(
        x=top_15['Score'],
        y=top_15['TeamName'],
        orientation='h',
        marker_color='lightcoral',
        text=[f"#{rank}" for rank in top_15['Rank']],
        textposition='inside'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Histogram(
        x=df_w3['Score'],
        nbinsx=20,
        marker_color='skyblue',
        opacity=0.7
    ),
    row=1, col=2
)

fig.update_layout(
    height=600,
    title_text="ğŸ“‹ Grand X-Ray Slam Division B - Week 3 Current Standings",
    showlegend=False
)

fig.show()

print(f"\nğŸ“ˆ WEEK 3 PERFORMANCE METRICS:")
print(f"   Leading Score: {df_w3['Score'].max():.6f}")
print(f"   Average Score: {df_w3['Score'].mean():.6f}")
print(f"   Score Range: {df_w3['Score'].min():.6f} - {df_w3['Score'].max():.6f}")
print(f"   Most submissions: {df_w3['SubmissionCount'].max()} by {df_w3.loc[df_w3['SubmissionCount'].idxmax(), 'TeamName']}")



# Recent activity analysis
recent_submissions = df_w3[df_w3['DaysAgo'] <= 2]
active_teams = len(recent_submissions)

print(f"ğŸ”¥ WEEK 3 ACTIVITY HIGHLIGHTS:")
print("=" * 50)
print(f"   â€¢ {active_teams} teams submitted in last 48h")
print(f"   â€¢ Leading team: {df_w3.iloc[0]['TeamName']}")
print(f"   â€¢ Current best score: {df_w3.iloc[0]['Score']:.6f}")
print(f"   â€¢ Average submissions per team: {df_w3['SubmissionCount'].mean():.1f}")


# Competition intensity calculation
intensity_score = (df_w3['SubmissionCount'].sum() / len(df_w3)) * (active_teams / len(df_w3))
print(f"   â€¢ Competition Intensity: {intensity_score:.2f}/1.0")

# Activity level categorization
df_w3['ActivityLevel'] = df_w3['SubmissionCount'].apply(
    lambda x: 'High (5+)' if x >= 5 else 'Medium (2-4)' if x >= 2 else 'Low (1)'
)


# Team performance landscape
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=list(range(1, len(df_w3)+1)),
    y=df_w3['Score'],
    mode='markers+lines',
    marker=dict(
        size=df_w3['SubmissionCount']*2,
        color=df_w3['SubmissionCount'],
        colorscale='viridis',
        showscale=True,
        colorbar=dict(title="Submissions")
    ),
    line=dict(width=1, color='lightgray'),
    text=df_w3['TeamName'],
    hovertemplate='<b>%{text}</b><br>Rank: %{x}<br>Score: %{y:.6f}<br>Submissions: %{marker.color}<extra></extra>'
))

fig.update_layout(
    title='ğŸ—ºï¸� Team Performance Landscape (Bubble size = Submissions)',
    xaxis_title='Team Rank',
    yaxis_title='Score',
    height=500,
    hovermode='closest'
)

fig.show()



# Activity distribution
activity_counts = df_w3['ActivityLevel'].value_counts()
fig_activity = px.pie(values=activity_counts.values, names=activity_counts.index,
                     title="Team Activity Distribution",
                     color_discrete_map={'High (5+)': 'red', 'Medium (2-4)': 'orange', 'Low (1)': 'lightblue'})
fig_activity.show()


# Timeline data
timeline_data = {
    'Phase': ['Competition Start', 'Week 1 Complete', 'Week 2 Complete', 'Current (Week 3)', 'Week 3 Complete', 'Final Submission', 'Results'],
    'Date': ['Aug 23', 'Aug 30', 'Sep 6', 'Sep 8', 'Sep 13', 'Oct 15', 'Oct 17'],
    'Status': ['Complete', 'Complete', 'Complete', 'Active', 'Upcoming', 'Upcoming', 'Upcoming']
}

timeline_df = pd.DataFrame(timeline_data)

fig = go.Figure()
colors = {'Complete': 'green', 'Active': 'blue', 'Upcoming': 'gray'}

for status in timeline_df['Status'].unique():
    mask = timeline_df['Status'] == status
    fig.add_trace(go.Scatter(
        x=timeline_df[mask]['Date'],
        y=timeline_df[mask]['Phase'],
        mode='markers',
        marker=dict(size=15, color=colors[status]),
        name=status,
        text=timeline_df[mask]['Status'],
        textposition='middle right'
    ))

fig.update_layout(
    title='ğŸ“… Competition Timeline',
    xaxis_title='Date',
    height=300,
    showlegend=True
)

fig.show()


# Progress calculation
total_days = 53  # Aug 23 to Oct 15
days_passed = 16  # Aug 23 to Sep 8
days_left = total_days - days_passed 
progress_pct = (days_passed / total_days) * 100

# Progress bar
fig_progress = go.Figure(go.Bar(
    x=[total_days],
    y=["Competition Progress"],
    orientation="h",
    marker=dict(color="lightgray"),
    width=0.5,
    showlegend=False
))

fig_progress.add_trace(go.Bar(
    x=[days_passed],
    y=["Competition Progress"],
    orientation="h",
    text=[f"{progress_pct:.1f}% complete"],
    textposition="inside",
    marker=dict(color="green"),
    width=0.5,
    name="Days Passed"
))

fig_progress.update_layout(
    title=f"Competition Progress (Day {days_passed} of {total_days})",
    barmode='overlay',
    xaxis=dict(range=[0, total_days], title="Days"),
    yaxis=dict(showticklabels=False),
    height=200
)

fig_progress.show()

print(f"â�° REMAINING TIME:")
print(f"   â€¢ Days until deadline: {days_left} days")
print(f"   â€¢ Competition progress: {progress_pct:.1f}% complete")
print(f"   â€¢ Time remaining: {(days_left/7):.1f} weeks")


# Score progression analysis
print(f"ğŸ”� PERFORMANCE INSIGHTS:")
print("=" * 50)

# Compare week 2 vs week 3 top scores
w2_top_score = df_w2['Score'].max()
w3_top_score = df_w3['Score'].max()
score_improvement = w3_top_score - w2_top_score

print(f"   â€¢ Top score improvement: +{score_improvement:.6f} ({(score_improvement/w2_top_score*100):.2f}%)")
print(f"   â€¢ Week 2 best: {w2_top_score:.6f}")
print(f"   â€¢ Week 3 best: {w3_top_score:.6f}")

# Submission efficiency analysis
df_w3['Efficiency'] = df_w3['Score'] / df_w3['SubmissionCount']
top_efficient = df_w3.nlargest(5, 'Efficiency')

print(f"\nğŸ�¯ MOST EFFICIENT TEAMS (Score per Submission):")
for _, row in top_efficient.iterrows():
    print(f"   {row['TeamName'][:20]:<20} | {row['Efficiency']:.6f} | ({row['Score']:.6f}/{row['SubmissionCount']} subs)")

# Competition trends
avg_score_w2 = df_w2['Score'].mean()
avg_score_w3 = df_w3['Score'].mean()
avg_improvement = avg_score_w3 - avg_score_w2

print(f"\nğŸ“Š OVERALL TRENDS:")
print(f"   â€¢ Average score improvement: +{avg_improvement:.6f}")
print(f"   â€¢ Score spread (std): {df_w3['Score'].std():.6f}")
print(f"   â€¢ Teams above 0.90: {len(df_w3[df_w3['Score'] > 0.90])}")
print(f"   â€¢ Teams above 0.85: {len(df_w3[df_w3['Score'] > 0.85])}")


print(f"ğŸ”® WEEK 4 PREDICTIONS:")
print("=" * 50)

# Growth rate calculations
entrant_growth_rate = (158 - 86) / 86
team_growth_rate = (17 - 6) / 6
submission_growth_rate = (47 - 14) / 14

# Project Week 4 numbers
projected_entrants = int(158 * (1 + entrant_growth_rate * 0.7))  # Assuming growth slows
projected_teams = int(17 * (1 + team_growth_rate * 0.6))
projected_submissions = int(47 * (1 + submission_growth_rate * 0.5))

print(f"ğŸ“ˆ PROJECTED WEEK 4 NUMBERS:")
print(f"   â€¢ Estimated entrants: {projected_entrants:,}")
print(f"   â€¢ Estimated teams: {projected_teams}")
print(f"   â€¢ Estimated submissions: {projected_submissions}")

print(f"\nğŸ�¯ KEY AREAS TO WATCH:")
print(f"   â€¢ Score plateau detection: Will teams break {w3_top_score:.6f}?")
print(f"   â€¢ Ensemble methods: Expected surge in advanced techniques")
print(f"   â€¢ Late-entry performance: How new teams adapt quickly")
print(f"   â€¢ Submission efficiency: Quality vs quantity strategies")

print(f"\nğŸš€ UPCOMING MILESTONES:")
print(f"   â€¢ Target: Break 0.95 barrier")
print(f"   â€¢ Watch: Top 10 teams' submission patterns")
print(f"   â€¢ Focus: Cross-validation optimization phase")
print(f"   â€¢ Trend: Expected ensemble model implementations")

