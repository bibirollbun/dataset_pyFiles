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


# Load all three weeks' data for comprehensive analysis
df_w2 = pd.read_csv('/kaggle/input/division-a-leaderboard-w4/Leaderboard_A_W2.csv')
df_w3 = pd.read_csv('/kaggle/input/division-a-leaderboard-w4/Leaderboard_A_W3.csv')
df_w4 = pd.read_csv('/kaggle/input/division-a-leaderboard-w4/Leaderboard_A_W4.csv')

print(f"Week 2 leaderboard: {len(df_w2)} teams")
print(f"Week 3 leaderboard: {len(df_w3)} teams")
print(f"Week 4 leaderboard: {len(df_w4)} teams")
print(f"New teams this week: {len(df_w4) - len(df_w3)}")

print("\nğŸ�† WEEK 4 TOP 5 TEAMS:")
print("=" * 60)
top_5 = df_w4.head()
for _, row in top_5.iterrows():
    print(f"{row['Rank']:2d}. {row['TeamName'][:25]:<25} | Score: {row['Score']:.6f}")


# Three-week comparison metrics
metrics_comparison = {
    'Metric': ['Total Entrants', 'Active Participants', 'Competing Teams', 'Total Submissions'],
    'Week 2': [217, 34, 34, 99],
    'Week 3': [320, 63, 61, 256], 
    'Week 4': [421, 85, 81, 391],
    'W2-W3 Growth': [103, 29, 27, 157],
    'W3-W4 Growth': [101, 22, 20, 135],
    'W2-W3 Growth %': [47.5, 85.3, 79.4, 158.6],
    'W3-W4 Growth %': [31.6, 34.9, 32.8, 52.7]
}

metrics_df = pd.DataFrame(metrics_comparison)

# Create comprehensive visualization
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=('Weekly Progression', 'Growth Rate Trends', 
                   'Submission Acceleration', 'Team Expansion', 
                   'Activity Heatmap', 'Competition Maturity'),
    specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
           [{'type': 'bar'}, {'type': 'scatter'}],
           [{'type': 'histogram'}, {'type': 'scatter'}]]
)

# Weekly progression
weeks = ['Week 2', 'Week 3', 'Week 4']
for i, metric in enumerate(['Total Entrants', 'Active Participants', 'Competing Teams', 'Total Submissions']):
    values = [metrics_df.iloc[i]['Week 2'], metrics_df.iloc[i]['Week 3'], metrics_df.iloc[i]['Week 4']]
    fig.add_trace(
        go.Scatter(x=weeks, y=values, mode='lines+markers', name=metric, line=dict(width=3)),
        row=1, col=1
    )

# Growth rate trends
growth_weeks = ['W2-W3', 'W3-W4']
fig.add_trace(
    go.Scatter(x=growth_weeks, y=[158.6, 52.7], mode='lines+markers+text',
               text=['158.6%', '52.7%'], textposition='top center',
               name='Submission Growth %', line=dict(width=4, color='red')),
    row=1, col=2
)

# Submission acceleration
fig.add_trace(
    go.Bar(x=weeks, y=[99, 256, 391], name='Weekly Submissions', 
           marker_color=['lightblue', 'blue', 'darkblue']),
    row=2, col=1
)

# Team expansion over time
fig.add_trace(
    go.Scatter(x=weeks, y=[34, 61, 81], mode='lines+markers+text',
               text=[34, 61, 81], textposition='top center',
               name='Team Count', line=dict(width=5, color='green')),
    row=2, col=2
)

# Current week activity distribution
fig.add_trace(
    go.Histogram(x=df_w4['SubmissionCount'], nbinsx=20, 
                 name='W4 Team Activity', marker_color='orange'),
    row=3, col=1
)

# Competition maturity indicator
maturity_weeks = ['Week 2', 'Week 3', 'Week 4']
intensity = [99/34, 256/61, 391/81]  # Submissions per team
fig.add_trace(
    go.Scatter(x=maturity_weeks, y=intensity, mode='lines+markers+text',
               text=[f'{i:.1f}' for i in intensity], textposition='top center',
               name='Subs/Team Ratio', line=dict(width=4, color='purple')),
    row=3, col=2
)

fig.update_layout(
    height=1000,
    title_text="ğŸ“Š Division A: Three-Week Evolution Analysis",
    showlegend=True
)

fig.show()

print("ğŸ“ˆ GROWTH PATTERN ANALYSIS:")
print(f"   â€¢ Week 2-3: Explosive growth phase (+158.6% submissions)")
print(f"   â€¢ Week 3-4: Stabilizing growth phase (+52.7% submissions)")
print(f"   â€¢ Current intensity: {391/81:.1f} submissions per team")
print(f"   â€¢ Growth rate moderation indicates competition maturation")


# Track teams across all three weeks
w2_teams = set(df_w2['TeamName'])
w3_teams = set(df_w3['TeamName'])
w4_teams = set(df_w4['TeamName'])

# Find different team categories
continuing_teams = w2_teams.intersection(w3_teams).intersection(w4_teams)
w3_w4_continuing = w3_teams.intersection(w4_teams)
new_w4_teams = w4_teams - w3_teams

print(f"ğŸ“Š TEAM EVOLUTION BREAKDOWN:")
print(f"   â€¢ Teams competing all 3 weeks: {len(continuing_teams)}")
print(f"   â€¢ Teams from Week 3 continuing: {len(w3_w4_continuing)}")
print(f"   â€¢ Brand new teams in Week 4: {len(new_w4_teams)}")
print(f"   â€¢ Teams that stopped after Week 3: {len(w3_teams - w4_teams)}")

# Score improvements for Week 3-4 continuing teams
improvements = []
for team in w3_w4_continuing:
    try:
        w3_score = df_w3[df_w3['TeamName'] == team]['Score'].iloc[0]
        w4_score = df_w4[df_w4['TeamName'] == team]['Score'].iloc[0]
        w3_rank = df_w3[df_w3['TeamName'] == team]['Rank'].iloc[0]
        w4_rank = df_w4[df_w4['TeamName'] == team]['Rank'].iloc[0]
        
        improvements.append({
            'TeamName': team,
            'W3_Score': w3_score,
            'W4_Score': w4_score,
            'Score_Improvement': w4_score - w3_score,
            'W3_Rank': w3_rank,
            'W4_Rank': w4_rank,
            'Rank_Change': w3_rank - w4_rank  # Positive = improved rank
        })
    except:
        continue

improvements_df = pd.DataFrame(improvements)
top_improvers = improvements_df.nlargest(5, 'Score_Improvement')
biggest_climbers = improvements_df.nlargest(5, 'Rank_Change')

print(f"\nğŸ�† TOP 5 SCORE IMPROVERS (Week 3 â†’ Week 4):")
print("=" * 70)
for _, row in top_improvers.iterrows():
    print(f"{row['TeamName'][:20]:<20} | +{row['Score_Improvement']:.6f} | Rank: {row['W3_Rank']} â†’ {row['W4_Rank']}")

print(f"\nğŸš€ BIGGEST RANK CLIMBERS (Week 3 â†’ Week 4):")
print("=" * 70)
for _, row in biggest_climbers.iterrows():
    print(f"{row['TeamName'][:20]:<20} | Rank: {row['W3_Rank']} â†’ {row['W4_Rank']} (+{row['Rank_Change']} positions)")

# Visualization of score evolution
fig = go.Figure()

# Continuing teams progression
fig.add_trace(go.Scatter(
    x=improvements_df['W3_Score'],
    y=improvements_df['W4_Score'],
    mode='markers',
    marker=dict(size=10, color='blue', opacity=0.7),
    name='Continuing Teams',
    text=improvements_df['TeamName'],
    hovertemplate='<b>%{text}</b><br>Week 3: %{x:.6f}<br>Week 4: %{y:.6f}<extra></extra>'
))

# New teams
new_teams_data = df_w4[df_w4['TeamName'].isin(new_w4_teams)]
fig.add_trace(go.Scatter(
    x=[0.5] * len(new_teams_data),
    y=new_teams_data['Score'],
    mode='markers',
    marker=dict(size=10, color='red', opacity=0.7),
    name='New Teams (Week 4)',
    text=new_teams_data['TeamName'],
    hovertemplate='<b>%{text}</b><br>New Team<br>Score: %{y:.6f}<extra></extra>'
))

# Add improvement reference line
fig.add_trace(go.Scatter(
    x=[0.5, 1.0],
    y=[0.5, 1.0],
    mode='lines',
    line=dict(dash='dash', color='gray'),
    name='No Improvement Line',
    showlegend=False
))

fig.update_layout(
    title="ğŸ�¯ Team Performance Evolution (Week 3 â†’ Week 4)",
    xaxis_title="Week 3 Score",
    yaxis_title="Week 4 Score",
    height=600
)

fig.show()


# Convert last submission date for analysis
df_w4['LastSubmissionDate'] = pd.to_datetime(df_w4['LastSubmissionDate'])
df_w4['DaysAgo'] = (datetime.now() - df_w4['LastSubmissionDate']).dt.days

# Current standings visualization
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Top 20 Teams Performance', 'Score Distribution Evolution',
                   'Submission Activity Levels', 'Performance Landscape'),
    specs=[[{'type': 'bar'}, {'type': 'histogram'}],
           [{'type': 'pie'}, {'type': 'scatter'}]]
)

# Top 20 teams
top_20 = df_w4.head(20)
fig.add_trace(
    go.Bar(
        x=top_20['Score'],
        y=top_20['TeamName'],
        orientation='h',
        marker_color='lightcoral',
        text=[f"#{rank}" for rank in top_20['Rank']],
        textposition='inside'
    ),
    row=1, col=1
)

# Score distribution comparison
fig.add_trace(
    go.Histogram(x=df_w2['Score'], nbinsx=20, name='Week 2', opacity=0.7, marker_color='lightblue'),
    row=1, col=2
)
fig.add_trace(
    go.Histogram(x=df_w3['Score'], nbinsx=20, name='Week 3', opacity=0.7, marker_color='blue'),
    row=1, col=2
)
fig.add_trace(
    go.Histogram(x=df_w4['Score'], nbinsx=20, name='Week 4', opacity=0.7, marker_color='darkblue'),
    row=1, col=2
)

# Activity levels
df_w4['ActivityLevel'] = df_w4['SubmissionCount'].apply(
    lambda x: 'High (7+)' if x >= 7 else 'Medium (3-6)' if x >= 3 else 'Low (1-2)'
)
activity_counts = df_w4['ActivityLevel'].value_counts()
fig.add_trace(
    go.Pie(labels=activity_counts.index, values=activity_counts.values,
           marker_colors=['red', 'orange', 'lightblue']),
    row=2, col=1
)

# Performance landscape
fig.add_trace(
    go.Scatter(
        x=list(range(1, len(df_w4)+1)),
        y=df_w4['Score'],
        mode='markers',
        marker=dict(
            size=df_w4['SubmissionCount']*1.5,
            color=df_w4['SubmissionCount'],
            colorscale='viridis',
            showscale=True
        ),
        text=df_w4['TeamName'],
        hovertemplate='<b>%{text}</b><br>Rank: %{x}<br>Score: %{y:.6f}<br>Submissions: %{marker.color}<extra></extra>'
    ),
    row=2, col=2
)

fig.update_layout(
    height=900,
    title_text="ğŸ“ˆ Week 4 Comprehensive Performance Analysis",
    showlegend=True
)

fig.show()

print(f"\nğŸ“Š WEEK 4 PERFORMANCE METRICS:")
print(f"   â€¢ Leading Score: {df_w4['Score'].max():.6f}")
print(f"   â€¢ Average Score: {df_w4['Score'].mean():.6f}")
print(f"   â€¢ Score Range: {df_w4['Score'].min():.6f} - {df_w4['Score'].max():.6f}")
print(f"   â€¢ Most submissions: {df_w4['SubmissionCount'].max()} by {df_w4.loc[df_w4['SubmissionCount'].idxmax(), 'TeamName']}")
print(f"   â€¢ Teams above 0.90: {len(df_w4[df_w4['Score'] > 0.90])}")
print(f"   â€¢ Teams above 0.85: {len(df_w4[df_w4['Score'] > 0.85])}")


# Timeline data
timeline_data = {
    'Phase': ['Competition Start', 'Week 1 Complete', 'Week 2 Complete', 'Week 3 Complete', 'Current (Week 4)', 'Week 4 Complete', 'Final Submission', 'Results'],
    'Date': ['Aug 21', 'Aug 28', 'Sep 4', 'Sep 11', 'Sep 15', 'Sep 18', 'Oct 10', 'Oct 12'],
    'Status': ['Complete', 'Complete', 'Complete', 'Complete', 'Active', 'Upcoming', 'Upcoming', 'Upcoming']
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
    title='ğŸ“… Competition Timeline - Halfway Point Reached',
    xaxis_title='Date',
    height=300,
    showlegend=True
)
fig.show()

# Progress calculation
total_days = 50
days_passed = 25  
days_left = total_days - days_passed 
progress_pct = (days_passed / total_days) * 100

# Progress bar with milestone marker
fig_progress = go.Figure()

fig_progress.add_trace(go.Bar(
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
    text=[f"ğŸ�¯ HALFWAY POINT: {progress_pct:.0f}% complete"],
    textposition="inside",
    marker=dict(color="gold"),
    width=0.5,
    name="Days Passed"
))

# Add milestone marker
fig_progress.add_vline(x=25, line_dash="dash", line_color="red", 
                      annotation_text="HALFWAY MILESTONE", annotation_position="top")

fig_progress.update_layout(
    title=f"ğŸ�� Competition Progress: Day {days_passed} of {total_days} (HALFWAY POINT!)",
    barmode='overlay',
    xaxis=dict(range=[0, total_days], title="Days"),
    yaxis=dict(showticklabels=False),
    height=200
)

fig_progress.show()

print(f"ğŸ�� HALFWAY MILESTONE ACHIEVED:")
print(f"   â€¢ Days completed: {days_passed} of {total_days}")
print(f"   â€¢ Days remaining: {days_left} days")
print(f"   â€¢ Time remaining: {(days_left/7):.1f} weeks")
print(f"   â€¢ Competition intensity: Peak performance phase begins")


# Advanced performance insights
print(f"ğŸ”� ADVANCED WEEK 4 INSIGHTS:")
print("=" * 50)

# Score progression analysis across all weeks
w2_top_score = df_w2['Score'].max()
w3_top_score = df_w3['Score'].max() 
w4_top_score = df_w4['Score'].max()

w2_w3_improvement = w3_top_score - w2_top_score
w3_w4_improvement = w4_top_score - w3_top_score

print(f"   â€¢ Week 2-3 top score improvement: +{w2_w3_improvement:.6f}")
print(f"   â€¢ Week 3-4 top score improvement: +{w3_w4_improvement:.6f}")
print(f"   â€¢ Current leading score: {w4_top_score:.6f}")

# Submission efficiency analysis
df_w4['Efficiency'] = df_w4['Score'] / df_w4['SubmissionCount']
top_efficient = df_w4.nlargest(5, 'Efficiency')

print(f"\nğŸ�¯ MOST EFFICIENT TEAMS (Score per Submission):")
for _, row in top_efficient.iterrows():
    print(f"   {row['TeamName'][:20]:<20} | {row['Efficiency']:.6f} | ({row['Score']:.6f}/{row['SubmissionCount']} subs)")

# Competition intensity trends
avg_scores = [df_w2['Score'].mean(), df_w3['Score'].mean(), df_w4['Score'].mean()]
print(f"\nğŸ“Š COMPETITION TRENDS:")
print(f"   â€¢ Average score progression: {avg_scores[0]:.6f} â†’ {avg_scores[1]:.6f} â†’ {avg_scores[2]:.6f}")
print(f"   â€¢ Score improvement rate is {'accelerating' if avg_scores[2]-avg_scores[1] > avg_scores[1]-avg_scores[0] else 'decelerating'}")
print(f"   â€¢ Top 10 average: {df_w4.head(10)['Score'].mean():.6f}")
print(f"   â€¢ Score standard deviation: {df_w4['Score'].std():.6f}")

# Recent activity analysis
recent_submissions = df_w4[df_w4['DaysAgo'] <= 2]
active_teams = len(recent_submissions)

print(f"\nğŸ”¥ RECENT ACTIVITY (Last 48h):")
print(f"   â€¢ Teams with recent submissions: {active_teams}")
print(f"   â€¢ Activity rate: {(active_teams/len(df_w4)*100):.1f}% of teams")
print(f"   â€¢ Current competition intensity: {(391/81):.1f} submissions per team")


print(f"ğŸ”® WEEK 5 PREDICTIONS & SECOND HALF OUTLOOK:")
print("=" * 60)

# Growth rate calculations with trend analysis
entrant_growth_w3_w4 = (421 - 320) / 320
team_growth_w3_w4 = (81 - 61) / 61
submission_growth_w3_w4 = (391 - 256) / 256

# Compare with previous week growth rates
entrant_growth_w2_w3 = (320 - 217) / 217
team_growth_w2_w3 = (61 - 34) / 34
submission_growth_w2_w3 = (256 - 99) / 99

print(f"ğŸ“ˆ GROWTH RATE EVOLUTION:")
print(f"   â€¢ Entrant growth: W2-W3: {entrant_growth_w2_w3:.1%} â†’ W3-W4: {entrant_growth_w3_w4:.1%}")
print(f"   â€¢ Team growth: W2-W3: {team_growth_w2_w3:.1%} â†’ W3-W4: {team_growth_w3_w4:.1%}")
print(f"   â€¢ Submission growth: W2-W3: {submission_growth_w2_w3:.1%} â†’ W3-W4: {submission_growth_w3_w4:.1%}")

# Project Week 5 numbers with refined modeling
projected_entrants = int(421 * (1 + entrant_growth_w3_w4 * 0.8))
projected_teams = int(81 * (1 + team_growth_w3_w4 * 0.7))
projected_submissions = int(391 * (1 + submission_growth_w3_w4 * 0.9))  # Higher factor as teams push harder

print(f"\nğŸ�¯ PROJECTED WEEK 5 NUMBERS:")
print(f"   â€¢ Estimated entrants: {projected_entrants:,}")
print(f"   â€¢ Estimated teams: {projected_teams}")
print(f"   â€¢ Estimated submissions: {projected_submissions}")

print(f"\nğŸ�� SECOND HALF PHASE PREDICTIONS:")
print(f"   â€¢ Competition intensity: PEAK performance phase")
print(f"   â€¢ Score improvements: Expect smaller, harder-fought gains")
print(f"   â€¢ Team strategies: Ensemble methods and fine-tuning focus")
print(f"   â€¢ New entrants: Likely to face steeper learning curve")

print(f"\nğŸ�² KEY MILESTONES TO WATCH:")
print(f"   â€¢ Target: Breaking {w4_top_score:.6f} barrier")
print(f"   â€¢ Trend: Cross-validation optimization surge expected")
print(f"   â€¢ Focus: Quality over quantity in submissions")
print(f"   â€¢ Watch: Top 15 teams' submission frequency patterns")

