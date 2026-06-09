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


df = pd.read_csv('/kaggle/input/division-a-leaderboard/leaderboard.csv')

print(f"Leaderboard loaded: {len(df)} teams")
print(f"Data shape: {df.shape}")

print("\n TOP 5 TEAMS:")
print("=" * 50)
top_5 = df.head()
for _, row in top_5.iterrows():
    print(f"{row['Rank']:2d}. {row['TeamName'][:20]:<20} | Score: {row['Score']:.4f}")


df['LastSubmissionDate'] = pd.to_datetime(df['LastSubmissionDate'])
df['DaysAgo'] = (datetime.now() - df['LastSubmissionDate']).dt.days

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Top 15 Teams Performance', 'Score Distribution'),
    specs=[[{'type': 'bar'}, {'type': 'histogram'}]]
)

top_15 = df.head(15)
fig.add_trace(
    go.Bar(
        x=top_15['Score'],
        y=top_15['TeamName'],
        orientation='h',
        marker_color='lightblue',
        text=[f"#{rank}" for rank in top_15['Rank']],
        textposition='inside'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Histogram(
        x=df['Score'],
        nbinsx=20,
        marker_color='lightcoral',
        opacity=0.7
    ),
    row=1, col=2
)

fig.update_layout(
    height=600,
    title_text=" Grand X-Ray Slam Division A - Current Standings",
    showlegend=False
)

fig.update_xaxes(title_text="Score", row=1, col=1)
fig.update_yaxes(title_text="Team", row=1, col=1)
fig.update_xaxes(title_text="Score", row=1, col=2)
fig.update_yaxes(title_text="Frequency", row=1, col=2)

fig.show()

print(f"\n PERFORMANCE METRICS:")
print(f" Leading Score: {df['Score'].max():.6f}")
print(f" Average Score: {df['Score'].mean():.6f}")
print(f" Score Range: {df['Score'].min():.6f} - {df['Score'].max():.6f}")


fig = plt.figure(figsize=(16,10))
fig.suptitle(' Competition Activity Insights', fontsize=16, fontweight='bold')

ax1 = fig.add_subplot(2,2,1)
ax1.hist(df['SubmissionCount'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
ax1.set_title('Submissions per Team')
ax1.set_xlabel('Number of Submissions')
ax1.set_ylabel('Teams')
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(2,2,2)
df['ActivityLevel'] = df['SubmissionCount'].apply(
    lambda x: 'High (3+)' if x >= 3 else 'Medium (2)' if x == 2 else 'Low (1)'
)
activity_counts = df['ActivityLevel'].value_counts()
ax2.pie(activity_counts.values, labels=activity_counts.index, autopct='%1.1f%%', 
        colors=['lightgreen', 'orange', 'lightcoral'])
ax2.set_title('Team Activity Levels')

ax3 = fig.add_subplot(2,1,2)
ax3.scatter(df['SubmissionCount'], df['Score'], alpha=0.6, color='purple', s=80)
ax3.set_xlabel('Number of Submissions')
ax3.set_ylabel('Score')
ax3.set_title('Score vs Submission Count')
ax3.grid(True, alpha=0.3)

z = np.polyfit(df['SubmissionCount'], df['Score'], 1)
p = np.poly1d(z)
ax3.plot(df['SubmissionCount'], p(df['SubmissionCount']), "r--", alpha=0.8)

plt.tight_layout()
plt.show()



recent_submissions = df[df['DaysAgo'] <= 2]
active_teams = len(recent_submissions)

print(" WEEK 2 HIGHLIGHTS")
print("=" * 40)
print(f" {active_teams} teams submitted in last 48h")
print(f" Leading team: {df.iloc[0]['TeamName']}")
print(f" Current best score: {df.iloc[0]['Score']:.6f}")
print(f" Most active team: {df.loc[df['SubmissionCount'].idxmax(), 'TeamName']} ({df['SubmissionCount'].max()} submissions)")

intensity_score = (df['SubmissionCount'].sum() / len(df)) * (active_teams / len(df))
print(f" Competition Intensity: {intensity_score:.2f}/1.0")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=list(range(1, len(df)+1)),
    y=df['Score'],
    mode='markers+lines',
    marker=dict(
        size=df['SubmissionCount']*3,
        color=df['SubmissionCount'],
        colorscale='viridis',
        showscale=True,
        colorbar=dict(title="Submissions")
    ),
    line=dict(width=1, color='lightgray'),
    text=df['TeamName'],
    hovertemplate='<b>%{text}</b><br>Rank: %{x}<br>Score: %{y:.6f}<extra></extra>'
))

fig.update_layout(
    title=' Team Performance Landscape (Bubble size = Submissions)',
    xaxis_title='Team Rank',
    yaxis_title='Score',
    height=500,
    hovermode='closest'
)

fig.show()


timeline_data = {
    'Phase': ['Competition Start', 'Week 1 Complete', 'Current (Week 2)', 'Final Submission', 'Results'],
    'Date': ['Aug 21', 'Aug 28', 'Sep 1', 'Oct 15', 'Oct 17'],
    'Status': ['Complete', 'Complete', 'Active', 'Upcoming', 'Upcoming']
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
    title=' Competition Timeline',
    xaxis_title='Date',
    height=300,
    showlegend=True
)
fig.show()


total_days = 55
days_passed = 11
days_left = total_days - days_passed 
progress_pct = (days_passed / total_days) * 100

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
    title="Competition Progress (Day 1 → Day 55)",
    barmode='overlay',
    xaxis=dict(range=[0, total_days], title="Days"),
    yaxis=dict(showticklabels=False),
    height=200
)

fig_progress.show()

print(" REMAINING TIME:")
print(f" Days until deadline: {days_left} days")
print(f" Competition progress: {progress_pct:.1f}% complete")





