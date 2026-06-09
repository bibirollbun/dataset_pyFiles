import pandas as pd
import numpy as np

!pip install --upgrade plotly

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

print("ğŸ�† GRAND X-RAY SLAM DIVISION B - FINAL RESULTS")
print("=" * 70)


# Load leaderboard data
PUBLIC_LEADERBOARD_PATH = '/kaggle/input/final-leaderboard/division-b-public_leaderboard.csv'
PRIVATE_LEADERBOARD_PATH = '/kaggle/input/final-leaderboard/division-b-private_leaderboard.csv'

df_public = pd.read_csv(PUBLIC_LEADERBOARD_PATH)
df_private = pd.read_csv(PRIVATE_LEADERBOARD_PATH)

# Display competition statistics
print("\nğŸ“Š COMPETITION STATISTICS")
print("-" * 70)
print(f"Total Entrants:        558")
print(f"Active Participants:   124")
print(f"Competing Teams:       113")
print(f"Total Submissions:     537")
print(f"Prize Pool:            $1,500 USD")
print(f"Average Submissions:   {537/113:.1f} per team")
print("=" * 70)


# Display Division B Champions
print("\nğŸ�‰ DIVISION B CHAMPIONS")
print("=" * 70)

# Get top 3 from private leaderboard
top_3_private = df_private.head(3)

for idx, row in top_3_private.iterrows():
    rank = row['Rank']
    if rank == 1:
        medal = "ğŸ¥‡"
        prize = "$750"
    elif rank == 2:
        medal = "ğŸ¥ˆ"
        prize = "$500"
    elif rank == 3:
        medal = "ğŸ¥‰"
        prize = "$250"
    
    print(f"\n{medal} Place {rank}: {row['TeamName']}")
    print(f"   Final Score: {row['Score']:.6f}")
    print(f"   Submissions: {row['SubmissionCount']}")
    print(f"   Prize: {prize}")

print("\n" + "=" * 70)


print("ğŸ“Š Chart 1: Top 3 Winners Score Comparison\n")

fig1 = go.Figure()

top_3 = df_private.head(3)
colors = ['gold', 'silver', '#CD7F32']  # Gold, Silver, Bronze
medals = ['ğŸ¥‡', 'ğŸ¥ˆ', 'ğŸ¥‰']

for idx, (i, row) in enumerate(top_3.iterrows()):
    fig1.add_trace(go.Bar(
        x=[row['TeamName']],
        y=[row['Score']],
        name=f"{medals[idx]} {row['TeamName']}",
        marker_color=colors[idx],
        text=f"{row['Score']:.6f}",
        textposition='outside',
        textfont=dict(size=14, color='black'),
        hovertemplate=f"<b>{row['TeamName']}</b><br>" +
                      f"Score: {row['Score']:.6f}<br>" +
                      f"Submissions: {row['SubmissionCount']}<br>" +
                      "<extra></extra>"
    ))

# Add score difference annotations
score_diff_2_1 = top_3.iloc[0]['Score'] - top_3.iloc[1]['Score']
score_diff_3_2 = top_3.iloc[1]['Score'] - top_3.iloc[2]['Score']

fig1.add_annotation(
    x=0.5, y=top_3.iloc[0]['Score'] + 0.0005,
    text=f"Î” {score_diff_2_1:.6f}",
    showarrow=False,
    font=dict(size=12, color='red')
)

fig1.add_annotation(
    x=1.5, y=top_3.iloc[1]['Score'] + 0.0005,
    text=f"Î” {score_diff_3_2:.6f}",
    showarrow=False,
    font=dict(size=12, color='red')
)

fig1.update_layout(
    title={
        'text': 'ğŸ�† Division B Top 3 Winners - Score Comparison',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20}
    },
    xaxis_title='Team Name',
    yaxis_title='Final Score',
    showlegend=False,
    height=600,
    width=1000,
    plot_bgcolor='rgba(240, 240, 240, 0.9)',
    yaxis=dict(range=[0.935, 0.943])
)

fig1.show()


print("ğŸ“Š Chart 2: Public vs Private Leaderboard - Top 10\n")

# Merge public and private data
top_10_private = df_private.head(10).copy()
top_10_private['PrivateRank'] = top_10_private['Rank']

# Get public ranks for same teams
public_ranks = []
for team_id in top_10_private['TeamId']:
    public_rank = df_public[df_public['TeamId'] == team_id]['Rank'].values
    public_ranks.append(public_rank[0] if len(public_rank) > 0 else None)

top_10_private['PublicRank'] = public_ranks
top_10_private['RankChange'] = top_10_private['PublicRank'] - top_10_private['PrivateRank']

fig2 = go.Figure()

# Add public ranks
fig2.add_trace(go.Scatter(
    x=top_10_private['TeamName'],
    y=top_10_private['PublicRank'],
    mode='markers+lines',
    name='Public Rank',
    marker=dict(size=12, color='lightblue', line=dict(width=2, color='darkblue')),
    line=dict(color='lightblue', width=2),
    text=[f"Public #{int(r)}" for r in top_10_private['PublicRank']],
    hovertemplate='<b>%{x}</b><br>Public Rank: #%{y}<extra></extra>'
))

# Add private ranks
fig2.add_trace(go.Scatter(
    x=top_10_private['TeamName'],
    y=top_10_private['PrivateRank'],
    mode='markers+lines',
    name='Private Rank',
    marker=dict(size=12, color='lightcoral', line=dict(width=2, color='darkred')),
    line=dict(color='lightcoral', width=2),
    text=[f"Private #{int(r)}" for r in top_10_private['PrivateRank']],
    hovertemplate='<b>%{x}</b><br>Private Rank: #%{y}<extra></extra>'
))

# Add arrows showing movement
for idx, row in top_10_private.iterrows():
    arrow_color = 'green' if row['RankChange'] > 0 else 'red' if row['RankChange'] < 0 else 'gray'
    fig2.add_annotation(
        x=row['TeamName'],
        y=row['PublicRank'],
        ax=row['TeamName'],
        ay=row['PrivateRank'],
        xref='x', yref='y',
        axref='x', ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor=arrow_color,
        opacity=0.4
    )

fig2.update_layout(
    title={
        'text': 'ğŸ“Š Public vs Private Leaderboard - Top 10 Teams<br><sub>Arrows show rank movement (Green=Improved, Red=Dropped)</sub>',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 18}
    },
    xaxis_title='Team Name',
    yaxis_title='Rank',
    yaxis=dict(autorange='reversed'),  # Lower rank numbers at top
    height=600,
    width=1200,
    plot_bgcolor='rgba(240, 240, 240, 0.9)',
    hovermode='closest',
    legend=dict(x=0.01, y=0.99)
)

fig2.show()


print("ğŸ“Š Chart 3: Submissions vs Performance - Top 30 Teams\n")

top_30 = df_private.head(30)

fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=top_30['SubmissionCount'],
    y=top_30['Score'],
    mode='markers+text',
    marker=dict(
        size=12,
        color=top_30['Rank'],
        colorscale='Viridis_r',
        showscale=True,
        colorbar=dict(title="Rank", x=1.15),
        line=dict(width=1, color='white')
    ),
    text=top_30['TeamName'],
    textposition='top center',
    textfont=dict(size=8),
    hovertemplate='<b>%{text}</b><br>' +
                  'Submissions: %{x}<br>' +
                  'Score: %{y:.6f}<br>' +
                  'Rank: %{marker.color}<br>' +
                  '<extra></extra>',
    showlegend=False
))

# Highlight top 3
top_3_highlight = df_private.head(3)
fig3.add_trace(go.Scatter(
    x=top_3_highlight['SubmissionCount'],
    y=top_3_highlight['Score'],
    mode='markers',
    marker=dict(
        size=18,
        color='rgba(255, 215, 0, 0.3)',
        line=dict(width=3, color='gold')
    ),
    hoverinfo='skip',
    showlegend=False
))

fig3.update_layout(
    title={
        'text': 'ğŸ“ˆ Submissions vs Performance - Top 30 Teams<br><sub>Gold outline indicates podium finishers</sub>',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 18}
    },
    xaxis_title='Number of Submissions',
    yaxis_title='Final Score',
    height=700,
    width=1100,
    plot_bgcolor='rgba(240, 240, 240, 0.9)',
    hovermode='closest'
)

fig3.show()


print("ğŸ“Š Chart 4: Score Distribution - All Teams\n")

fig4 = go.Figure()

fig4.add_trace(go.Histogram(
    x=df_private['Score'],
    nbinsx=40,
    marker_color='steelblue',
    opacity=0.7,
    name='Score Distribution'
))

# Add vertical lines for top 3 scores
for idx, row in df_private.head(3).iterrows():
    fig4.add_vline(
        x=row['Score'],
        line_dash="dash",
        line_color=['gold', 'silver', '#CD7F32'][idx],
        line_width=2,
        annotation_text=f"{row['TeamName']}: {row['Score']:.6f}",
        annotation_position="top"
    )

fig4.update_layout(
    title={
        'text': 'ğŸ“Š Final Score Distribution - All Teams<br><sub>Dashed lines show top 3 scores</sub>',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 18}
    },
    xaxis_title='Score',
    yaxis_title='Number of Teams',
    height=600,
    width=1100,
    plot_bgcolor='rgba(240, 240, 240, 0.9)',
    showlegend=False
)

fig4.show()


print("ğŸ“Š Chart 5: Efficiency Analysis - Top 10 Teams\n")

top_10_efficiency = df_private.head(10).copy()
top_10_efficiency['Efficiency'] = top_10_efficiency['Score'] / top_10_efficiency['SubmissionCount']
top_10_efficiency = top_10_efficiency.sort_values('Efficiency', ascending=True)

fig5 = go.Figure()

fig5.add_trace(go.Bar(
    y=top_10_efficiency['TeamName'],
    x=top_10_efficiency['Efficiency'],
    orientation='h',
    marker=dict(
        color=top_10_efficiency['Efficiency'],
        colorscale='Greens',
        showscale=True,
        colorbar=dict(title="Efficiency")
    ),
    text=[f"{eff:.6f}" for eff in top_10_efficiency['Efficiency']],
    textposition='auto',
    hovertemplate='<b>%{y}</b><br>' +
                  'Efficiency: %{x:.6f}<br>' +
                  '<extra></extra>'
))

fig5.update_layout(
    title={
        'text': 'âš¡ Top 10 Teams - Efficiency Score (Score per Submission)',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 18}
    },
    xaxis_title='Efficiency (Score / Submission Count)',
    yaxis_title='Team Name',
    height=600,
    width=1100,
    plot_bgcolor='rgba(240, 240, 240, 0.9)'
)

fig5.show()


print("ğŸ�† GRAND SLAM PRIZE - CROSS-DIVISION PERFORMANCE")
print("=" * 70)

# Division A scores (from previous competition)
division_a_scores = {
    'Raj gupta': 0.942319,
    'Masry1': 0.940755,
    'Sophia_Carter': 0.940387
}

# Division B scores
division_b_scores = {
    'Raj gupta': 0.9418724,
    'Masry1': 0.9398545,
    'Sophia_Carter': 0.9403233
}

# Calculate Grand Slam scores
grand_slam_data = []
for team in division_a_scores.keys():
    if team in division_b_scores:
        avg_score = (division_a_scores[team] + division_b_scores[team]) / 2
        grand_slam_data.append({
            'Team': team,
            'Division_A': division_a_scores[team],
            'Division_B': division_b_scores[team],
            'Grand_Slam_Average': avg_score
        })

df_grand_slam = pd.DataFrame(grand_slam_data)
df_grand_slam = df_grand_slam.sort_values('Grand_Slam_Average', ascending=False).reset_index(drop=True)
df_grand_slam['Rank'] = df_grand_slam.index + 1

print("\nğŸ�¯ GRAND SLAM FINAL STANDINGS")
print("-" * 70)

prizes = ['$1,250', '$750', '$500']
medals = ['ğŸ¥‡', 'ğŸ¥ˆ', 'ğŸ¥‰']

for idx, row in df_grand_slam.iterrows():
    print(f"\n{medals[idx]} Grand Slam Rank {row['Rank']}: {row['Team']}")
    print(f"   Division A Score: {row['Division_A']:.6f}")
    print(f"   Division B Score: {row['Division_B']:.6f}")
    print(f"   Grand Slam Average: {row['Grand_Slam_Average']:.6f}")
    print(f"   Prize: {prizes[idx]}")

print("\n" + "=" * 70)

# Display DataFrame
df_grand_slam


print("ğŸ“Š Chart 6: Grand Slam Cross-Division Performance\n")

fig6 = go.Figure()

x_teams = df_grand_slam['Team']

# Division A bars
fig6.add_trace(go.Bar(
    name='Division A',
    x=x_teams,
    y=df_grand_slam['Division_A'],
    marker_color='lightblue',
    text=[f"{s:.6f}" for s in df_grand_slam['Division_A']],
    textposition='outside'
))

# Division B bars
fig6.add_trace(go.Bar(
    name='Division B',
    x=x_teams,
    y=df_grand_slam['Division_B'],
    marker_color='lightcoral',
    text=[f"{s:.6f}" for s in df_grand_slam['Division_B']],
    textposition='outside'
))

# Grand Slam Average line
fig6.add_trace(go.Scatter(
    name='Grand Slam Average',
    x=x_teams,
    y=df_grand_slam['Grand_Slam_Average'],
    mode='lines+markers',
    line=dict(color='gold', width=4),
    marker=dict(size=15, color='gold', line=dict(width=2, color='darkgoldenrod')),
    text=[f"{s:.6f}" for s in df_grand_slam['Grand_Slam_Average']],
    textposition='top center'
))

fig6.update_layout(
    title={
        'text': 'ğŸ�† Grand Slam Prize - Cross-Division Performance<br><sub>Average of Division A and Division B scores</sub>',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20}
    },
    xaxis_title='Team',
    yaxis_title='Score',
    barmode='group',
    height=700,
    width=1100,
    plot_bgcolor='rgba(240, 240, 240, 0.9)',
    legend=dict(x=0.01, y=0.99),
    yaxis=dict(range=[0.935, 0.945])
)

fig6.show()


print("\nğŸ“ˆ FINAL STATISTICS SUMMARY")
print("=" * 70)
print(f"Winning Score (Division B):     {df_private.iloc[0]['Score']:.6f}")
print(f"Average Score (All Teams):      {df_private['Score'].mean():.6f}")
print(f"Median Score:                   {df_private['Score'].median():.6f}")
print(f"Score Standard Deviation:       {df_private['Score'].std():.6f}")
print(f"Average Submissions per Team:   {df_private['SubmissionCount'].mean():.1f}")
print(f"Max Submissions:                {df_private['SubmissionCount'].max()}")
print(f"Min Submissions:                {df_private['SubmissionCount'].min()}")
print("=" * 70)

print("\nâœ… Report Generation Complete!")
print("ğŸ�† Congratulations to all participants and champions!")

