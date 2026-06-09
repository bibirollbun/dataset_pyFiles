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
from datetime import datetime

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("ğŸ“Š Grand X-Ray Slam Division A - Final Analysis")
print("=" * 60)
print("Libraries loaded successfully!")


# ============================================
# DATA PATHS - UPDATE THESE FOR FINAL VERSION
# ============================================
PUBLIC_LEADERBOARD_PATH = '/kaggle/input/division-a-final-leaderboard/division-a-public_leaderboard.csv'
PRIVATE_LEADERBOARD_PATH = '/kaggle/input/division-a-final-leaderboard/division-a-private_leaderboar.csv'

# ============================================
# PARTICIPATION DATA - UPDATE FOR FINAL VERSION
# ============================================
FINAL_STATS = {
    'Total Entrants': 765,
    'Active Participants': 214,
    'Competing Teams': 192,
    'Total Submissions': 1197,
    'Competition Duration': '50 days',
    'Start Date': 'August 21, 2025',
    'End Date': 'October 10, 2025'
}

# Historical participation data (for trend analysis)
WEEKLY_PROGRESSION = {
    'Week 2': {'Entrants': 217, 'Participants': 34, 'Teams': 34, 'Submissions': 99},
    'Week 4': {'Entrants': 421, 'Participants': 85, 'Teams': 81, 'Submissions': 391},
    'Final': {'Entrants': 650, 'Participants': 152, 'Teams': 141, 'Submissions': 845}
}

print("\nâœ… Configuration loaded - Ready for analysis")


# Load public and private leaderboards
df_public = pd.read_csv(PUBLIC_LEADERBOARD_PATH)
df_private = pd.read_csv(PRIVATE_LEADERBOARD_PATH)

print(f"ğŸ“‹ Public Leaderboard: {len(df_public)} teams")
print(f"ğŸ”’ Private Leaderboard: {len(df_private)} teams")
print(f"âœ… Data integrity: {'PASS' if len(df_public) == len(df_private) else 'FAIL'}")

print("\nğŸ�† TOP 10 - PRIVATE LEADERBOARD (FINAL STANDINGS):")
print("=" * 80)
top_10_private = df_private.head(10)
for _, row in top_10_private.iterrows():
    medal = "ğŸ¥‡" if row['Rank'] == 1 else "ğŸ¥ˆ" if row['Rank'] == 2 else "ğŸ¥‰" if row['Rank'] == 3 else "  "
    print(f"{medal} {row['Rank']:2d}. {row['TeamName'][:30]:<30} | Score: {row['Score']:.6f} | Subs: {row['SubmissionCount']}")




# Merge public and private leaderboards for comparison
df_comparison = df_public.merge(
    df_private,
    on='TeamName',
    suffixes=('_public', '_private')
)

# Calculate rank changes
df_comparison['Rank_Change'] = df_comparison['Rank_public'] - df_comparison['Rank_private']
df_comparison['Score_Change'] = df_comparison['Score_private'] - df_comparison['Score_public']

print("ğŸ”„ PUBLIC vs PRIVATE LEADERBOARD SHAKE-UP ANALYSIS")
print("=" * 80)

# Statistics
print(f"\nğŸ“Š OVERALL STATISTICS:")
print(f"   â€¢ Teams that improved rank: {len(df_comparison[df_comparison['Rank_Change'] > 0])}")
print(f"   â€¢ Teams that dropped rank: {len(df_comparison[df_comparison['Rank_Change'] < 0])}")
print(f"   â€¢ Teams that maintained rank: {len(df_comparison[df_comparison['Rank_Change'] == 0])}")
print(f"   â€¢ Average rank change: {df_comparison['Rank_Change'].abs().mean():.2f} positions")
print(f"   â€¢ Maximum rank improvement: +{df_comparison['Rank_Change'].max()} positions")
print(f"   â€¢ Maximum rank drop: {df_comparison['Rank_Change'].min()} positions")

# Biggest movers UP
print(f"\nğŸš€ TOP 10 BIGGEST CLIMBERS (Public â†’ Private):")
print("=" * 80)
biggest_climbers = df_comparison.nlargest(10, 'Rank_Change')
for _, row in biggest_climbers.iterrows():
    print(f"   {row['TeamName'][:30]:<30} | {row['Rank_public']:3d} â†’ {row['Rank_private']:3d} (+{row['Rank_Change']} positions)")

# Biggest movers DOWN
print(f"\nğŸ“‰ TOP 10 BIGGEST DROPS (Public â†’ Private):")
print("=" * 80)
biggest_drops = df_comparison.nsmallest(10, 'Rank_Change')
for _, row in biggest_drops.iterrows():
    print(f"   {row['TeamName'][:30]:<30} | {row['Rank_public']:3d} â†’ {row['Rank_private']:3d} ({row['Rank_Change']} positions)")

# Top 10 stability
top_10_changes = df_comparison[df_comparison['Rank_private'] <= 10].sort_values('Rank_private')
print(f"\nğŸ�† TOP 10 STABILITY ANALYSIS:")
print("=" * 80)
for _, row in top_10_changes.iterrows():
    change_text = f"+{row['Rank_Change']}" if row['Rank_Change'] > 0 else f"{row['Rank_Change']}" if row['Rank_Change'] < 0 else "No change"
    print(f"   #{row['Rank_private']:2d} {row['TeamName'][:25]:<25} | Public: #{row['Rank_public']:3d} ({change_text})")


# Create comprehensive shake-up visualization
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Rank Changes Distribution', 'Top 20 Position Changes',
                   'Score Improvements (Public vs Private)', 'Rank Change vs Final Position'),
    specs=[[{'type': 'histogram'}, {'type': 'scatter'}],
           [{'type': 'scatter'}, {'type': 'scatter'}]]
)

# Rank changes distribution
fig.add_trace(
    go.Histogram(x=df_comparison['Rank_Change'], nbinsx=30,
                 marker_color='steelblue', name='Rank Changes'),
    row=1, col=1
)

# Top 20 position changes
top_20_comparison = df_comparison[
    (df_comparison['Rank_public'] <= 20) | (df_comparison['Rank_private'] <= 20)
].sort_values('Rank_private')

for _, row in top_20_comparison.iterrows():
    color = 'green' if row['Rank_Change'] > 0 else 'red' if row['Rank_Change'] < 0 else 'gray'
    fig.add_trace(
        go.Scatter(
            x=['Public', 'Private'],
            y=[row['Rank_public'], row['Rank_private']],
            mode='lines+markers',
            name=row['TeamName'][:15],
            line=dict(color=color, width=2),
            showlegend=False,
            hovertemplate=f"<b>{row['TeamName']}</b><br>%{{y}}<extra></extra>"
        ),
        row=1, col=2
    )

# Score improvements
fig.add_trace(
    go.Scatter(
        x=df_comparison['Score_public'],
        y=df_comparison['Score_private'],
        mode='markers',
        marker=dict(
            size=8,
            color=df_comparison['Rank_Change'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Rank Change", x=0.46)
        ),
        text=df_comparison['TeamName'],
        hovertemplate='<b>%{text}</b><br>Public: %{x:.6f}<br>Private: %{y:.6f}<extra></extra>',
        showlegend=False
    ),
    row=2, col=1
)

# Add diagonal reference line for score comparison
fig.add_trace(
    go.Scatter(x=[0.5, 1.0], y=[0.5, 1.0], mode='lines',
               line=dict(dash='dash', color='gray'),
               showlegend=False),
    row=2, col=1
)

# Rank change vs final position
fig.add_trace(
    go.Scatter(
        x=df_comparison['Rank_private'],
        y=df_comparison['Rank_Change'],
        mode='markers',
        marker=dict(size=6, color='purple', opacity=0.6),
        text=df_comparison['TeamName'],
        hovertemplate='<b>%{text}</b><br>Final Rank: %{x}<br>Rank Change: %{y}<extra></extra>',
        showlegend=False
    ),
    row=2, col=2
)

fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)

fig.update_xaxes(title_text="Rank Change", row=1, col=1)
fig.update_xaxes(title_text="Leaderboard", row=1, col=2)
fig.update_xaxes(title_text="Public Score", row=2, col=1)
fig.update_xaxes(title_text="Final Private Rank", row=2, col=2)

fig.update_yaxes(title_text="Frequency", row=1, col=1)
fig.update_yaxes(title_text="Rank (lower is better)", row=1, col=2)
fig.update_yaxes(title_text="Private Score", row=2, col=1)
fig.update_yaxes(title_text="Rank Change", row=2, col=2)

fig.update_layout(
    height=900,
    title_text="ğŸ”„ Public vs Private Leaderboard Comprehensive Analysis",
    showlegend=False
)

fig.show()

# Statistical analysis
print("\nğŸ“ˆ SHAKE-UP IMPACT ANALYSIS:")
print("=" * 60)
print(f"   â€¢ Correlation (Public Rank vs Private Rank): {df_comparison['Rank_public'].corr(df_comparison['Rank_private']):.4f}")
print(f"   â€¢ Score correlation: {df_comparison['Score_public'].corr(df_comparison['Score_private']):.4f}")
print(f"   â€¢ Teams with >10 position change: {len(df_comparison[df_comparison['Rank_Change'].abs() > 10])}")
print(f"   â€¢ Top 10 teams that changed: {len(top_10_changes[top_10_changes['Rank_Change'] != 0])}/10")


# Prepare growth data
weeks = ['Week 2\n(Day 11)', 'Week 4\n(Day 25)', 'Final\n(Day 50)']
entrants = [217, 421, 650]
teams = [34, 81, 141]
submissions = [99, 391, 845]

# Create growth visualization
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Participation Growth', 'Team Expansion',
                   'Submission Activity', 'Final Distribution'),
    specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
           [{'type': 'bar'}, {'type': 'pie'}]]
)

# Entrants & Participants growth
fig.add_trace(
    go.Scatter(x=weeks, y=entrants, mode='lines+markers+text',
               text=entrants, textposition='top center',
               name='Total Entrants', line=dict(width=4, color='blue')),
    row=1, col=1
)

# Team growth
fig.add_trace(
    go.Scatter(x=weeks, y=teams, mode='lines+markers+text',
               text=teams, textposition='top center',
               name='Competing Teams', line=dict(width=4, color='green')),
    row=1, col=2
)

# Submission activity
fig.add_trace(
    go.Bar(x=weeks, y=submissions, name='Total Submissions',
           marker_color=['lightblue', 'blue', 'darkblue'],
           text=submissions, textposition='outside'),
    row=2, col=1
)

# Final activity distribution
df_private['ActivityLevel'] = df_private['SubmissionCount'].apply(
    lambda x: 'High (10+)' if x >= 10 else 'Medium (5-9)' if x >= 5 else 'Low (1-4)'
)
activity_counts = df_private['ActivityLevel'].value_counts()
fig.add_trace(
    go.Pie(labels=activity_counts.index, values=activity_counts.values,
           marker_colors=['red', 'orange', 'lightblue']),
    row=2, col=2
)

fig.update_layout(
    height=900,
    title_text="ğŸ“Š Competition Growth Timeline & Final Statistics",
    showlegend=True
)

fig.show()

# Growth statistics
print("\nğŸš€ GROWTH STATISTICS:")
print("=" * 60)
print(f"   â€¢ Entrant growth (Week 2 â†’ Final): {((650-217)/217*100):.1f}%")
print(f"   â€¢ Team growth (Week 2 â†’ Final): {((141-34)/34*100):.1f}%")
print(f"   â€¢ Submission growth (Week 2 â†’ Final): {((845-99)/99*100):.1f}%")
print(f"   â€¢ Average submissions per team: {845/141:.1f}")
print(f"   â€¢ Final participation rate: {(152/650*100):.1f}%")


# Performance statistics
print("ğŸ�¯ FINAL PERFORMANCE ANALYSIS")
print("=" * 80)

print(f"\nğŸ“Š SCORE DISTRIBUTION:")
print(f"   â€¢ Winning Score: {df_private['Score'].max():.6f}")
print(f"   â€¢ Average Score: {df_private['Score'].mean():.6f}")
print(f"   â€¢ Median Score: {df_private['Score'].median():.6f}")
print(f"   â€¢ Score Range: {df_private['Score'].min():.6f} - {df_private['Score'].max():.6f}")
print(f"   â€¢ Standard Deviation: {df_private['Score'].std():.6f}")

print(f"\nğŸ”¥ SUBMISSION ACTIVITY:")
print(f"   â€¢ Most submissions: {df_private['SubmissionCount'].max()} by {df_private.loc[df_private['SubmissionCount'].idxmax(), 'TeamName']}")
print(f"   â€¢ Average submissions: {df_private['SubmissionCount'].mean():.1f}")
print(f"   â€¢ Median submissions: {df_private['SubmissionCount'].median():.0f}")
print(f"   â€¢ Teams with 1 submission: {len(df_private[df_private['SubmissionCount'] == 1])}")
print(f"   â€¢ Teams with 10+ submissions: {len(df_private[df_private['SubmissionCount'] >= 10])}")

print(f"\nğŸ�† PERFORMANCE TIERS:")
print(f"   â€¢ Elite (0.95+): {len(df_private[df_private['Score'] >= 0.95])} teams")
print(f"   â€¢ Expert (0.90-0.95): {len(df_private[(df_private['Score'] >= 0.90) & (df_private['Score'] < 0.95)])} teams")
print(f"   â€¢ Advanced (0.85-0.90): {len(df_private[(df_private['Score'] >= 0.85) & (df_private['Score'] < 0.90)])} teams")
print(f"   â€¢ Intermediate (0.80-0.85): {len(df_private[(df_private['Score'] >= 0.80) & (df_private['Score'] < 0.85)])} teams")
print(f"   â€¢ Developing (<0.80): {len(df_private[df_private['Score'] < 0.80])} teams")

# Efficiency analysis
df_private['Efficiency'] = df_private['Score'] / df_private['SubmissionCount']
top_efficient = df_private.nlargest(5, 'Efficiency')

print(f"\nâš¡ MOST EFFICIENT TEAMS (Score per Submission):")
print("=" * 80)
for _, row in top_efficient.iterrows():
    print(f"   {row['TeamName'][:30]:<30} | {row['Efficiency']:.6f} | ({row['Score']:.6f}/{row['SubmissionCount']} subs)")


# Create final performance dashboard
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Final Score Distribution', 'Submissions vs Performance',
                   'Top 30 Teams', 'Efficiency Leaders'),
    specs=[[{'type': 'histogram'}, {'type': 'scatter'}],
           [{'type': 'bar'}, {'type': 'bar'}]]
)

# Score distribution
fig.add_trace(
    go.Histogram(x=df_private['Score'], nbinsx=30,
                 marker_color='steelblue', name='Score Distribution'),
    row=1, col=1
)

# Submissions vs Performance scatter
fig.add_trace(
    go.Scatter(
        x=df_private['SubmissionCount'],
        y=df_private['Score'],
        mode='markers',
        marker=dict(
            size=8,
            color=df_private['Rank'],
            colorscale='Viridis_r',
            showscale=True,
            colorbar=dict(title="Rank", x=1.15)
        ),
        text=df_private['TeamName'],
        hovertemplate='<b>%{text}</b><br>Subs: %{x}<br>Score: %{y:.6f}<br>Rank: %{marker.color}<extra></extra>',
        showlegend=False
    ),
    row=1, col=2
)

# Top 30 teams
top_30 = df_private.head(30)
fig.add_trace(
    go.Bar(
        x=top_30['Score'],
        y=top_30['TeamName'],
        orientation='h',
        marker_color='coral',
        text=[f"#{r}" for r in top_30['Rank']],
        textposition='inside',
        showlegend=False
    ),
    row=2, col=1
)

# Top 10 efficient teams
top_10_efficient = df_private.nlargest(10, 'Efficiency')
fig.add_trace(
    go.Bar(
        x=top_10_efficient['Efficiency'],
        y=top_10_efficient['TeamName'],
        orientation='h',
        marker_color='lightgreen',
        showlegend=False
    ),
    row=2, col=2
)

fig.update_xaxes(title_text="Score", row=1, col=1)
fig.update_xaxes(title_text="Submission Count", row=1, col=2)
fig.update_xaxes(title_text="Score", row=2, col=1)
fig.update_xaxes(title_text="Efficiency (Score/Submission)", row=2, col=2)

fig.update_yaxes(title_text="Frequency", row=1, col=1)
fig.update_yaxes(title_text="Score", row=1, col=2)

fig.update_layout(
    height=1000,
    title_text="ğŸ�† Final Performance Dashboard - Division A Champions",
    showlegend=False
)

fig.show()

