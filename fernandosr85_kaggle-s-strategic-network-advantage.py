# =============================================================================
# PYTHON LIBRARIES - ORGANIZED IMPORT SECTION
# =============================================================================

# Core data manipulation and analysis
import pandas as pd
import numpy as np

# Statistical analysis and signal processing
from scipy import stats
from scipy.signal import find_peaks
from scipy.stats import pearsonr

# Data visualization - Matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.dates import YearLocator, DateFormatter

# Data visualization - Seaborn
import seaborn as sns

# Interactive visualization - Plotly
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as py
from plotly.subplots import make_subplots

# Network analysis
import networkx as nx

# Date and time handling
from datetime import datetime, timedelta

# Data structures and utilities
from collections import defaultdict

# External data sources
import kagglehub

# System utilities
import os

# Warning control
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================

# Matplotlib styling
plt.style.use('seaborn-v0_8-darkgrid')  
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
#plt.style.use('dark_background')

# Seaborn styling
sns.set_palette("husl")


# Download datasets
meta_kaggle_path = kagglehub.dataset_download("kaggle/meta-kaggle")
meta_kaggle_code_path = kagglehub.dataset_download("kaggle/meta-kaggle-code")

print(f"Meta Kaggle path: {meta_kaggle_path}")
print(f"Meta Kaggle Code path: {meta_kaggle_code_path}")


# View dataset structure
print("=== Meta Kaggle Files ===")
for file in os.listdir(meta_kaggle_path):
    print(f"ğŸ“� {file}")
    
print("\n=== Meta Kaggle Code Files ===")
for file in os.listdir(meta_kaggle_code_path):
    print(f"ğŸ“� {file}")


# Suppress specific pandas warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# Competitions data
print("Loading competitions data...")
competitions = pd.read_csv(f"{meta_kaggle_path}/Competitions.csv")
print(f"Competitions: {competitions.shape}")
print("Columns:", competitions.columns.tolist())
print("\nFirst 5 rows:")
print(competitions.head())

# Kernel/Notebook data  
print("\n" + "="*50)
print("Loading kernels data...")
kernels = pd.read_csv(f"{meta_kaggle_path}/KernelVersions.csv")
print(f"Kernels: {kernels.shape}")
print("Columns:", kernels.columns.tolist())
print("\nFirst 5 rows:")
print(kernels.head())

# Submissions data (if exists)
print("\n" + "="*50)
print("Checking submissions data...")
if os.path.exists(f"{meta_kaggle_path}/Submissions.csv"):
    print("Loading submissions data...")
    # Specify dtype to avoid warning about mixed types
    submissions = pd.read_csv(
        f"{meta_kaggle_path}/Submissions.csv",
        dtype={'PublicScoreLeaderboardDisplay': 'str'},  # Column 7 as string
        low_memory=False  # Load entire file into memory to infer types
    )
    print(f"Submissions: {submissions.shape}")
    print("Columns:", submissions.columns.tolist())
    print("\nFirst 5 rows:")
    print(submissions.head())
else:
    print("Submissions.csv file not found.")
    submissions = None

# Check for NaN values that might cause warnings
print("\n" + "="*50)
print("LOADED DATA SUMMARY:")
print(f"- Competitions: {competitions.shape[0]:,} records, {competitions.shape[1]} columns")
print(f"- Kernels: {kernels.shape[0]:,} records, {kernels.shape[1]} columns")
if submissions is not None:
    print(f"- Submissions: {submissions.shape[0]:,} records, {submissions.shape[1]} columns")

# Check data quality issues that might cause warnings
print("\nData quality verification:")
print("Competitions - NaN values per column (only columns with NaN):")
nan_cols_comp = competitions.isnull().sum()
nan_cols_comp = nan_cols_comp[nan_cols_comp > 0]
if len(nan_cols_comp) > 0:
    print(nan_cols_comp)
else:
    print("No columns with NaN values found.")

print("\nKernels - NaN values per column (only columns with NaN):")
nan_cols_kernels = kernels.isnull().sum()
nan_cols_kernels = nan_cols_kernels[nan_cols_kernels > 0]
if len(nan_cols_kernels) > 0:
    print(nan_cols_kernels)
else:
    print("No columns with NaN values found.")

if submissions is not None:
    print("\nSubmissions - NaN values per column (top 10 columns with most NaN):")
    nan_cols_subs = submissions.isnull().sum().sort_values(ascending=False).head(10)
    nan_cols_subs = nan_cols_subs[nan_cols_subs > 0]
    if len(nan_cols_subs) > 0:
        print(nan_cols_subs)
    else:
        print("No columns with NaN values found.")


# ================================================================================
# VISUALIZATION 1: INTERACTIVE NETWORK GRAPH
# ================================================================================

def create_interactive_network_graph(network_graph, centrality_results, analysis_data):
    """Create interactive network visualization"""
    
    print("ğŸ•¸ï¸� Creating Interactive Network Graph...")
    
    # Sample network for visualization (too many nodes = slow)
    if network_graph.number_of_nodes() > 500:
        # Get top nodes by centrality
        top_nodes = centrality_results.nlargest(500, 'network_influence_score')['user_id'].tolist()
        G_viz = network_graph.subgraph(top_nodes).copy()
        print(f"   Sampled to {G_viz.number_of_nodes()} nodes for visualization")
    else:
        G_viz = network_graph
    
    # Calculate layout
    pos = nx.spring_layout(G_viz, k=1, iterations=50)
    
    # Extract node and edge data
    node_trace = go.Scatter(
        x=[pos[node][0] for node in G_viz.nodes()],
        y=[pos[node][1] for node in G_viz.nodes()],
        mode='markers+text',
        hovertemplate='<b>User ID: %{customdata[0]}</b><br>' +
                      'Connections: %{customdata[1]}<br>' +
                      'Network Score: %{customdata[2]:.3f}<br>' +
                      'Performance Score: %{customdata[3]:.2f}<br>' +
                      '<extra></extra>',
        marker=dict(
            size=[],
            color=[],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Network Influence"),
            line=dict(width=0.5, color='white')
        ),
        customdata=[],
        text=[],
        textposition="middle center"
    )
    
    # Add node data
    node_data = []
    node_sizes = []
    node_colors = []
    node_texts = []
    
    for node in G_viz.nodes():
        node_info = centrality_results[centrality_results['user_id'] == node]
        if len(node_info) > 0:
            network_score = node_info['network_influence_score'].iloc[0]
            connections = node_info['num_connections'].iloc[0]
            
            # Get performance data if available
            perf_info = analysis_data[analysis_data['Id'] == node] if 'analysis_data' in locals() else pd.DataFrame()
            perf_score = perf_info['performance_score'].iloc[0] if len(perf_info) > 0 else 0
            username = perf_info['UserName'].iloc[0] if len(perf_info) > 0 else f"User_{node}"
            
            node_data.append([node, connections, network_score, perf_score])
            node_sizes.append(max(10, min(50, connections * 2)))  # Scale node size by connections
            node_colors.append(network_score)
            node_texts.append(username[:10] if len(username) > 10 else username)
    
    node_trace.marker.size = node_sizes
    node_trace.marker.color = node_colors
    node_trace.customdata = node_data
    node_trace.text = node_texts
    
    # Create edge traces
    edge_x = []
    edge_y = []
    
    for edge in G_viz.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='rgba(255,255,255,0.3)'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=dict(
                           text='ğŸŒ� Kaggle Collaboration Network<br><sub>Node size = connections, Color = influence score</sub>',
                           x=0.5,
                           font=dict(size=20)
                       ),
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=80),
                       annotations=[ dict(
                           text="Interactive network of Kaggle user collaborations. Larger nodes = more connections.",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor="left", yanchor="bottom",
                           font=dict(color="gray", size=12)
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       plot_bgcolor='black',
                       paper_bgcolor='black'
                   ))
    
    return fig


# ================================================================================
# VISUALIZATION 2: NETWORK VS PERFORMANCE CORRELATION
# ================================================================================

def create_correlation_analysis(analysis_data, correlations):
    """Create network-performance correlation visualizations"""
    
    print("ğŸ“ˆ Creating Correlation Analysis...")
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Network Influence vs Performance', 'Correlation Heatmap', 
                       'Network Tiers Performance', 'Collaboration ROI'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Scatter plot of network vs performance
    fig.add_trace(
        go.Scatter(
            x=analysis_data['network_influence_score'],
            y=analysis_data['performance_score'],
            mode='markers',
            marker=dict(
                size=8,
                color=analysis_data['num_connections'],
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title="Connections", x=1.1)
            ),
            text=analysis_data['UserName'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Network Score: %{x:.3f}<br>' +
                         'Performance: %{y:.2f}<br>' +
                         'Connections: %{marker.color}<br>' +
                         '<extra></extra>',
            name='Users'
        ),
        row=1, col=1
    )
    
    # Plot 2: Correlation heatmap
    network_cols = ['degree_centrality', 'betweenness_centrality', 'pagerank', 'network_influence_score']
    performance_cols = ['total_submissions', 'kernels_created', 'performance_score']
    
    corr_subset = correlations.loc[network_cols, performance_cols]
    
    fig.add_trace(
        go.Heatmap(
            z=corr_subset.values,
            x=performance_cols,
            y=network_cols,
            colorscale='RdBu',
            zmid=0,
            text=corr_subset.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='%{y} â†” %{x}<br>Correlation: %{z:.3f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Plot 3: Performance by network tiers
    analysis_data['network_tier'] = pd.qcut(
        analysis_data['network_influence_score'], 
        q=5,
        labels=['Bottom 20%', '20-40%', '40-60%', '60-80%', 'Top 20%']
    )
    
    tier_performance = analysis_data.groupby('network_tier')['performance_score'].agg(['mean', 'std']).reset_index()
    
    fig.add_trace(
        go.Bar(
            x=tier_performance['network_tier'],
            y=tier_performance['mean'],
            error_y=dict(type='data', array=tier_performance['std']),
            marker_color='lightblue',
            name='Avg Performance',
            hovertemplate='Network Tier: %{x}<br>Avg Performance: %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Plot 4: Team vs Solo Performance
    if 'teams_joined' in analysis_data.columns:
        solo_users = analysis_data[analysis_data['teams_joined'] == 0]
        team_users = analysis_data[analysis_data['teams_joined'] > 0]
        
        collaboration_data = pd.DataFrame({
            'Type': ['Solo Players', 'Team Players'],
            'Count': [len(solo_users), len(team_users)],
            'Avg_Performance': [solo_users['performance_score'].mean(), team_users['performance_score'].mean()],
            'Std_Performance': [solo_users['performance_score'].std(), team_users['performance_score'].std()]
        })
        
        fig.add_trace(
            go.Bar(
                x=collaboration_data['Type'],
                y=collaboration_data['Avg_Performance'],
                error_y=dict(type='data', array=collaboration_data['Std_Performance']),
                marker_color=['red', 'green'],
                name='Performance by Type',
                hovertemplate='%{x}<br>Count: %{customdata}<br>Avg Performance: %{y:.2f}<extra></extra>',
                customdata=collaboration_data['Count']
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="ğŸ”— Network-Performance Analysis Dashboard",
        title_x=0.5,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update axes
    fig.update_xaxes(title_text="Network Influence Score", row=1, col=1)
    fig.update_yaxes(title_text="Performance Score", row=1, col=1)
    fig.update_xaxes(title_text="Network Tier", row=2, col=1)
    fig.update_yaxes(title_text="Average Performance", row=2, col=1)
    fig.update_xaxes(title_text="Player Type", row=2, col=2)
    fig.update_yaxes(title_text="Average Performance", row=2, col=2)
    
    return fig


# ================================================================================
# VISUALIZATION 3: TEMPORAL COLLABORATION TRENDS
# ================================================================================

def create_collaboration_trends(team_analysis):
    """Create temporal trends in collaboration patterns"""
    
    print("ğŸ“… Creating Collaboration Trends...")
    
    # Yearly collaboration metrics
    yearly_trends = team_analysis.groupby('year').agg({
        'team_size': ['mean', 'count'],
        'Id_x': 'nunique'  # Unique teams
    }).round(2)
    
    yearly_trends.columns = ['avg_team_size', 'total_teams', 'unique_competitions']
    yearly_trends = yearly_trends.reset_index()
    
    # Filter years with sufficient data
    yearly_trends = yearly_trends[yearly_trends['total_teams'] >= 10]
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Team Size Evolution', 'Collaboration Volume', 
                       'Team Size Distribution', 'Competition Activity'),
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Team size evolution over time
    fig.add_trace(
        go.Scatter(
            x=yearly_trends['year'],
            y=yearly_trends['avg_team_size'],
            mode='lines+markers',
            name='Avg Team Size',
            line=dict(color='blue', width=3),
            marker=dict(size=8),
            hovertemplate='Year: %{x}<br>Avg Team Size: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=yearly_trends['year'],
            y=yearly_trends['total_teams'],
            name='Total Teams',
            marker_color='lightblue',
            opacity=0.7,
            yaxis='y2',
            hovertemplate='Year: %{x}<br>Total Teams: %{y}<extra></extra>'
        ),
        row=1, col=1, secondary_y=True
    )
    
    # Plot 2: Collaboration volume trends
    fig.add_trace(
        go.Scatter(
            x=yearly_trends['year'],
            y=yearly_trends['total_teams'],
            mode='lines+markers',
            name='Teams Formed',
            line=dict(color='green', width=3),
            hovertemplate='Year: %{x}<br>Teams: %{y}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=yearly_trends['year'],
            y=yearly_trends['unique_competitions'],
            mode='lines+markers',
            name='Active Competitions',
            line=dict(color='orange', width=3),
            yaxis='y4',
            hovertemplate='Year: %{x}<br>Competitions: %{y}<extra></extra>'
        ),
        row=1, col=2, secondary_y=True
    )
    
    # Plot 3: Team size distribution
    team_size_dist = team_analysis['team_size'].value_counts().sort_index()
    team_size_dist = team_size_dist[team_size_dist.index <= 10]  # Focus on sizes 1-10
    
    fig.add_trace(
        go.Bar(
            x=team_size_dist.index,
            y=team_size_dist.values,
            marker_color='purple',
            name='Team Size Distribution',
            hovertemplate='Team Size: %{x}<br>Count: %{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Plot 4: Recent activity (last 5 years)
    recent_data = yearly_trends.tail(5) if len(yearly_trends) >= 5 else yearly_trends
    
    fig.add_trace(
        go.Scatter(
            x=recent_data['year'],
            y=recent_data['total_teams'],
            mode='lines+markers+text',
            text=recent_data['total_teams'],
            textposition='top center',
            name='Recent Activity',
            line=dict(color='red', width=4),
            marker=dict(size=12),
            hovertemplate='Year: %{x}<br>Teams: %{y}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="ğŸ“ˆ Kaggle Collaboration Trends Over Time",
        title_x=0.5,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_yaxes(title_text="Average Team Size", row=1, col=1)
    fig.update_yaxes(title_text="Number of Teams", row=1, col=1, secondary_y=True)
    
    fig.update_xaxes(title_text="Year", row=1, col=2)
    fig.update_yaxes(title_text="Teams Formed", row=1, col=2)
    fig.update_yaxes(title_text="Active Competitions", row=1, col=2, secondary_y=True)
    
    fig.update_xaxes(title_text="Team Size", row=2, col=1)
    fig.update_yaxes(title_text="Number of Teams", row=2, col=1)
    
    fig.update_xaxes(title_text="Year", row=2, col=2)
    fig.update_yaxes(title_text="Teams Formed", row=2, col=2)
    
    return fig


# ================================================================================
# VISUALIZATION 4: KEY INSIGHTS DASHBOARD
# ================================================================================

def create_insights_dashboard(analysis_data, centrality_results, correlations):
    """Create executive summary dashboard with key insights"""
    
    print("ğŸ’¡ Creating Key Insights Dashboard...")
    
    # Calculate key metrics
    total_users = len(analysis_data)
    networked_users = len(centrality_results)
    avg_connections = centrality_results['num_connections'].mean()
    max_connections = centrality_results['num_connections'].max()
    
    # Network-performance correlation
    main_correlation = correlations.loc['network_influence_score', 'performance_score']
    
    # Performance advantage of networked users
    high_network = analysis_data[analysis_data['network_influence_score'] > analysis_data['network_influence_score'].quantile(0.8)]
    low_network = analysis_data[analysis_data['network_influence_score'] < analysis_data['network_influence_score'].quantile(0.2)]
    network_advantage = (high_network['performance_score'].mean() / low_network['performance_score'].mean() - 1) * 100
    
    # Create metrics cards visualization
    fig = go.Figure()
    
    # Add metric cards as annotations
    metrics = [
        {"title": "Total Users Analyzed", "value": f"{total_users:,}", "x": 0.15, "y": 0.85},
        {"title": "Users in Network", "value": f"{networked_users:,}", "x": 0.45, "y": 0.85},
        {"title": "Avg Connections", "value": f"{avg_connections:.1f}", "x": 0.75, "y": 0.85},
        {"title": "Max Connections", "value": f"{max_connections:,}", "x": 0.15, "y": 0.65},
        {"title": "Network-Performance Correlation", "value": f"{main_correlation:.3f}", "x": 0.45, "y": 0.65},
        {"title": "Network Advantage", "value": f"+{network_advantage:.1f}%", "x": 0.75, "y": 0.65}
    ]
    
    annotations = []
    for metric in metrics:
        # Add title
        annotations.append(dict(
            x=metric["x"], y=metric["y"] + 0.05,
            text=f"<b>{metric['title']}</b>",
            showarrow=False,
            font=dict(size=14, color="navy"),
            xref="paper", yref="paper"
        ))
        
        # Add value
        annotations.append(dict(
            x=metric["x"], y=metric["y"],
            text=f"<span style='font-size:24px; color:darkblue'>{metric['value']}</span>",
            showarrow=False,
            xref="paper", yref="paper"
        ))
        
        # Add background box
        fig.add_shape(
            type="rect",
            x0=metric["x"] - 0.08, y0=metric["y"] - 0.03,
            x1=metric["x"] + 0.08, y1=metric["y"] + 0.08,
            fillcolor="lightblue",
            opacity=0.3,
            line=dict(width=1, color="navy"),
            xref="paper", yref="paper"
        )
    
    # Add key insights text
    insights_text = """
    <b>ğŸ”� KEY INSIGHTS:</b><br>
    
    â€¢ <b>Network Effect Confirmed:</b> Users with higher network centrality show {:.1f}% better performance<br>
    â€¢ <b>Collaboration ROI:</b> Top 20% networked users outperform bottom 20% by {:.1f}%<br>
    â€¢ <b>Social Capital:</b> Average of {:.1f} connections per active user<br>
    â€¢ <b>Community Structure:</b> Clear evidence of knowledge-sharing networks<br>
    
    <b>ğŸ�¯ RECOMMENDATIONS:</b><br>
    
    â€¢ <b>For Users:</b> Invest in building network connections for competitive advantage<br>
    â€¢ <b>For Kaggle:</b> Facilitate networking features to enhance community value<br>
    â€¢ <b>For Industry:</b> Leverage social dynamics in ML team formation
    """.format(network_advantage, network_advantage, avg_connections)
    
    annotations.append(dict(
        x=0.5, y=0.35,
        text=insights_text,
        showarrow=False,
        font=dict(size=12),
        xref="paper", yref="paper",
        align="left"
    ))
    
    # Add title
    annotations.append(dict(
        x=0.5, y=0.95,
        text="<b style='font-size:24px'>ğŸŒ� The Kaggle Social Graph - Executive Summary</b>",
        showarrow=False,
        font=dict(size=20, color="darkblue"),
        xref="paper", yref="paper"
    ))
    
    fig.update_layout(
        annotations=annotations,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    return fig


# Try to initialize plotly, fallback to matplotlib if fails
try:
    py.init_notebook_mode(connected=True)
    USE_PLOTLY = True
except:
    USE_PLOTLY = False
    print("âš  Plotly nÃ£o disponÃ­vel, usando matplotlib/seaborn")

print("ğŸŒ� THE KAGGLE SOCIAL GRAPH - COMPLETE ANALYSIS (FIXED)")
print("="*60)

# ================================================================================
# STEP 1: LOAD ALL DATA
# ================================================================================

print("\nğŸ“Š STEP 1: Loading All Required Data")
print("-" * 40)

# Load main datasets
meta_kaggle_path = "/kaggle/input/meta-kaggle"

# Core data
competitions = pd.read_csv(f"{meta_kaggle_path}/Competitions.csv")
kernels = pd.read_csv(f"{meta_kaggle_path}/KernelVersions.csv")
submissions = pd.read_csv(f"{meta_kaggle_path}/Submissions.csv", low_memory=False)

print(f"âœ“ Competitions: {competitions.shape[0]:,}")
print(f"âœ“ Kernels: {kernels.shape[0]:,}")
print(f"âœ“ Submissions: {submissions.shape[0]:,}")

# Social network data
teams = pd.read_csv(f"{meta_kaggle_path}/Teams.csv")
team_memberships = pd.read_csv(f"{meta_kaggle_path}/TeamMemberships.csv")
users = pd.read_csv(f"{meta_kaggle_path}/Users.csv")

print(f"âœ“ Teams: {teams.shape[0]:,}")
print(f"âœ“ Team Memberships: {team_memberships.shape[0]:,}")
print(f"âœ“ Users: {users.shape[0]:,}")

# Optional data (load if available)
optional_files = {
    'user_followers': 'UserFollowers.csv',
    'forum_messages': 'ForumMessages.csv',
    'forum_votes': 'ForumMessageVotes.csv',
    'kernel_votes': 'KernelVotes.csv',
    'organizations': 'Organizations.csv',
    'user_organizations': 'UserOrganizations.csv'
}

optional_data = {}
for name, filename in optional_files.items():
    try:
        optional_data[name] = pd.read_csv(f"{meta_kaggle_path}/{filename}")
        print(f"âœ“ {name}: {optional_data[name].shape[0]:,}")
    except:
        print(f"âš  {name}: Not available")
        optional_data[name] = None

# ================================================================================
# STEP 2: BUILD COLLABORATION NETWORK
# ================================================================================

print("\nğŸ¤� STEP 2: Building Collaboration Network")
print("-" * 40)

def build_collaboration_network_fast():
    """Build collaboration network efficiently"""
    
    # Use more data for better network analysis
    recent_teams = teams.tail(500000) if len(teams) > 500000 else teams  # Increased from 100k
    recent_memberships = team_memberships[
        team_memberships['TeamId'].isin(recent_teams['Id'])
    ]
    
    print(f"Processing {len(recent_memberships):,} team memberships...")
    
    # Create user-user edges from team memberships
    edges = []
    
    # Group by team and create combinations
    for team_id, group in recent_memberships.groupby('TeamId'):
        user_list = group['UserId'].tolist()
        if len(user_list) > 1:  # Teams with multiple members
            for i in range(len(user_list)):
                for j in range(i+1, len(user_list)):
                    edges.append({
                        'user1': min(user_list[i], user_list[j]),
                        'user2': max(user_list[i], user_list[j]),
                        'weight': 1
                    })
    
    if edges:
        collab_df = pd.DataFrame(edges)
        # Aggregate multiple collaborations
        collab_summary = collab_df.groupby(['user1', 'user2'])['weight'].sum().reset_index()
        print(f"âœ“ Created {len(collab_summary):,} collaboration edges")
        return collab_summary
    else:
        print("âš  No collaboration edges found")
        return pd.DataFrame()

collaboration_network = build_collaboration_network_fast()

# ================================================================================
# STEP 3: CALCULATE USER PERFORMANCE METRICS
# ================================================================================

print("\nğŸ�† STEP 3: Calculating User Performance")
print("-" * 40)

def calculate_user_performance_fast():
    """Calculate user performance metrics efficiently"""
    
    # Start with user base
    user_metrics = users[['Id', 'UserName', 'RegisterDate']].copy()
    user_metrics['RegisterDate'] = pd.to_datetime(user_metrics['RegisterDate'], errors='coerce')
    
    # Submission metrics (sample for performance)
    if len(submissions) > 1000000:
        submission_sample = submissions.sample(1000000, random_state=42)
        print("Sampling 1M submissions for performance analysis...")
    else:
        submission_sample = submissions
    
    # Calculate submission stats
    sub_stats = submission_sample.groupby('SubmittedUserId').agg({
        'Id': 'count',
        'PublicScoreFullPrecision': 'mean'
    }).round(4)
    sub_stats.columns = ['total_submissions', 'avg_score']
    sub_stats = sub_stats.reset_index().rename(columns={'SubmittedUserId': 'Id'})
    
    # Kernel stats (sample for performance)
    if len(kernels) > 1000000:
        kernel_sample = kernels.sample(1000000, random_state=42)
        print("Sampling 1M kernels for performance analysis...")
    else:
        kernel_sample = kernels
    
    kernel_stats = kernel_sample.groupby('AuthorUserId').agg({
        'Id': 'count',
        'TotalVotes': ['sum', 'mean']
    }).round(2)
    kernel_stats.columns = ['kernels_created', 'total_votes', 'avg_votes']
    kernel_stats = kernel_stats.reset_index().rename(columns={'AuthorUserId': 'Id'})
    
    # Team participation
    team_stats = team_memberships.groupby('UserId').size().reset_index(name='teams_joined')
    team_stats.rename(columns={'UserId': 'Id'}, inplace=True)
    
    # Merge all metrics
    user_metrics = user_metrics.merge(sub_stats, on='Id', how='left')
    user_metrics = user_metrics.merge(kernel_stats, on='Id', how='left')
    user_metrics = user_metrics.merge(team_stats, on='Id', how='left')
    
    # Fill NaN values
    numeric_cols = ['total_submissions', 'avg_score', 'kernels_created', 
                   'total_votes', 'avg_votes', 'teams_joined']
    for col in numeric_cols:
        if col in user_metrics.columns:
            user_metrics[col] = user_metrics[col].fillna(0)
    
    # Calculate composite performance score
    user_metrics['performance_score'] = (
        user_metrics.get('total_submissions', 0) * 0.3 +
        user_metrics.get('kernels_created', 0) * 0.3 +
        user_metrics.get('total_votes', 0) * 0.001 +  # Scale down votes
        user_metrics.get('teams_joined', 0) * 2.0     # Scale up teams
    ).round(2)
    
    print(f"âœ“ Performance calculated for {len(user_metrics):,} users")
    return user_metrics

user_performance = calculate_user_performance_fast()

# ================================================================================
# STEP 4: NETWORK ANALYSIS
# ================================================================================

print("\nğŸ•¸ï¸� STEP 4: Network Analysis")
print("-" * 40)

def analyze_network_fast():
    """Analyze network efficiently"""
    
    if len(collaboration_network) == 0:
        print("âš  No collaboration data available")
        return pd.DataFrame(), None
    
    # Create NetworkX graph
    G = nx.Graph()
    for _, row in collaboration_network.iterrows():
        G.add_edge(row['user1'], row['user2'], weight=row['weight'])
    
    print(f"âœ“ Network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # Calculate centrality for all nodes (or sample if too large)
    if G.number_of_nodes() > 10000:
        top_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)[:5000]
        G_sample = G.subgraph(top_nodes).copy()
        print(f"Analyzing top {len(top_nodes):,} nodes for centrality...")
    else:
        G_sample = G
        print(f"Analyzing all {G.number_of_nodes():,} nodes for centrality...")
    
    # Calculate centrality metrics
    centrality_data = []
    
    degree_cent = nx.degree_centrality(G_sample)
    
    # Use sample for betweenness if network is large
    if G_sample.number_of_nodes() > 1000:
        betweenness_cent = nx.betweenness_centrality(G_sample, k=min(1000, G_sample.number_of_nodes()))
    else:
        betweenness_cent = nx.betweenness_centrality(G_sample)
    
    for node in G_sample.nodes():
        centrality_data.append({
            'user_id': node,
            'degree_centrality': degree_cent.get(node, 0),
            'betweenness_centrality': betweenness_cent.get(node, 0),
            'num_connections': G.degree(node)
        })
    
    centrality_df = pd.DataFrame(centrality_data)
    
    # Add small random noise to break ties in network influence score
    np.random.seed(42)
    noise = np.random.normal(0, 0.0001, len(centrality_df))
    
    # Calculate network influence score with noise to avoid ties
    centrality_df['network_influence_score'] = (
        centrality_df['degree_centrality'] * 0.6 +
        centrality_df['betweenness_centrality'] * 0.4 +
        noise
    ).round(6)  # More precision to avoid ties
    
    print(f"âœ“ Centrality calculated for {len(centrality_df):,} users")
    return centrality_df, G_sample

centrality_results, network_graph = analyze_network_fast()

# ================================================================================
# STEP 5: CORRELATION ANALYSIS (FIXED)
# ================================================================================

print("\nğŸ“ˆ STEP 5: Network-Performance Correlation")
print("-" * 40)

def analyze_correlations():
    """Analyze network-performance correlations"""
    
    if len(centrality_results) == 0:
        print("âš  No centrality data for correlation")
        return pd.DataFrame()
    
    # Merge datasets
    analysis_df = user_performance.merge(
        centrality_results, 
        left_on='Id', 
        right_on='user_id', 
        how='inner'
    )
    
    print(f"âœ“ Analysis dataset: {len(analysis_df):,} users")
    
    # Calculate key correlations
    network_cols = ['degree_centrality', 'betweenness_centrality', 'network_influence_score', 'num_connections']
    performance_cols = ['total_submissions', 'kernels_created', 'total_votes', 'performance_score']
    
    available_cols = [col for col in network_cols + performance_cols if col in analysis_df.columns]
    
    if len(available_cols) > 1:
        correlation_matrix = analysis_df[available_cols].corr()
        
        # Key insights
        if 'network_influence_score' in analysis_df.columns and 'performance_score' in analysis_df.columns:
            main_corr = correlation_matrix.loc['network_influence_score', 'performance_score']
            print(f"ğŸ”— Network-Performance Correlation: {main_corr:.3f}")
            
            # FIXED: Performance by network tiers with duplicate handling
            try:
                # Check if we have enough unique values for 5 tiers
                unique_scores = analysis_df['network_influence_score'].nunique()
                n_tiers = min(5, unique_scores)
                
                if n_tiers > 1:
                    analysis_df['network_tier'] = pd.qcut(
                        analysis_df['network_influence_score'], 
                        q=n_tiers, 
                        labels=[f'Tier {i+1}' for i in range(n_tiers)],
                        duplicates='drop'  # This fixes the duplicate edge error
                    )
                    
                    tier_performance = analysis_df.groupby('network_tier')['performance_score'].mean()
                    print(f"\nğŸ“Š Performance by Network Tier ({n_tiers} tiers):")
                    for tier, perf in tier_performance.items():
                        print(f"   {tier}: {perf:.2f}")
                else:
                    print("\nâš  Not enough unique network scores for tier analysis")
                    analysis_df['network_tier'] = 'Single Tier'
                
            except Exception as e:
                print(f"âš  Tier analysis error: {e}")
                # Create simple binary tiers as fallback
                median_score = analysis_df['network_influence_score'].median()
                analysis_df['network_tier'] = analysis_df['network_influence_score'].apply(
                    lambda x: 'Above Median' if x > median_score else 'Below Median'
                )
            
            # Calculate network advantage
            high_network = analysis_df[analysis_df['network_influence_score'] > analysis_df['network_influence_score'].quantile(0.8)]
            low_network = analysis_df[analysis_df['network_influence_score'] < analysis_df['network_influence_score'].quantile(0.2)]
            
            if len(high_network) > 0 and len(low_network) > 0:
                high_avg = high_network['performance_score'].mean()
                low_avg = low_network['performance_score'].mean()
                if low_avg > 0:
                    advantage = (high_avg / low_avg - 1) * 100
                    print(f"ğŸš€ Top 20% vs Bottom 20% advantage: +{advantage:.1f}%")
    
    return analysis_df

analysis_data = analyze_correlations()

# ================================================================================
# STEP 6: CREATE VISUALIZATIONS (MELHORADAS PARA KAGGLE)
# ================================================================================

print("\nğŸ�¨ STEP 6: Creating Visualizations")
print("-" * 40)

# ================================================================================
# STEP 6: CREATE VISUALIZATIONS (MELHORADAS PARA KAGGLE)
# ================================================================================

print("\nğŸ�¨ STEP 6: Creating Visualizations")
print("-" * 40)

# First, let's explore what columns we actually have
print("ğŸ”� EXPLORING AVAILABLE DATA COLUMNS:")
print("-" * 40)
print(f"ğŸ“Š Competitions columns: {list(competitions.columns)}")
print(f"ğŸ‘¥ Teams columns: {list(teams.columns)}")
print(f"ğŸ¤� Team Memberships columns: {list(team_memberships.columns)}")
print(f"ğŸ‘¤ Users columns: {list(users.columns)}")
print(f"ğŸ“� Kernels columns: {list(kernels.columns)}")
print(f"ğŸ“¤ Submissions columns: {list(submissions.columns)}")

def create_matplotlib_visualizations():
    """Create comprehensive visualizations using matplotlib/seaborn"""
    
    if len(analysis_data) == 0:
        print("âš  No data for visualization")
        return
    
    # Create figure with subplots - AJUSTE NO ESPAÃ‡AMENTO
    fig, axes = plt.subplots(2, 3, figsize=(20, 16))  # Aumentei a altura de 14 para 16
    
    # AJUSTE CRÃ�TICO: TÃ­tulo principal com mais espaÃ§o
    fig.suptitle('ğŸŒ� Kaggle Social Graph Analysis - Comprehensive Dashboard', 
                 fontsize=18, fontweight='bold', y=0.98)  # Mudei de 0.95 para 0.98
    
    # Sample data for visualization if too large
    viz_data = analysis_data.sample(min(2000, len(analysis_data)), random_state=42) if len(analysis_data) > 2000 else analysis_data
    
    # Plot 1: Network vs Performance Scatter (improved)
    if 'network_influence_score' in viz_data.columns and 'performance_score' in viz_data.columns:
        scatter = axes[0,0].scatter(
            viz_data['network_influence_score'], 
            viz_data['performance_score'],
            c=viz_data['num_connections'], 
            cmap='viridis', 
            alpha=0.7, 
            s=80,
            edgecolors='white',
            linewidth=0.5
        )
        axes[0,0].set_xlabel('Network Influence Score', fontweight='bold')
        axes[0,0].set_ylabel('Performance Score', fontweight='bold')
        # AJUSTE: Reduzir o pad do tÃ­tulo
        axes[0,0].set_title('Network Influence vs Performance', fontweight='bold', pad=10)  # Reduzi de 20 para 10
        cbar = plt.colorbar(scatter, ax=axes[0,0])
        cbar.set_label('Number of Connections', fontweight='bold')
        
        # Add correlation line with better styling
        try:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(viz_data['network_influence_score'], viz_data['performance_score'])
            line = slope * viz_data['network_influence_score'] + intercept
            axes[0,0].plot(viz_data['network_influence_score'], line, 
                          'red', linewidth=2, alpha=0.8, 
                          label=f'Correlation: {r_value:.3f}')
            axes[0,0].legend(fontsize=10)
            axes[0,0].grid(True, alpha=0.3)
        except:
            pass
    
    # Plot 2: Connection Distribution (improved)
    if 'num_connections' in viz_data.columns:
        conn_dist = viz_data['num_connections'].value_counts().sort_index().head(15)
        bars = axes[0,1].bar(range(len(conn_dist)), conn_dist.values, 
                           color='skyblue', alpha=0.8, edgecolor='navy', linewidth=0.8)
        axes[0,1].set_xlabel('Number of Connections', fontweight='bold')
        axes[0,1].set_ylabel('Number of Users', fontweight='bold')
        axes[0,1].set_title('User Connection Distribution', fontweight='bold', pad=10)  # Reduzi o pad
        axes[0,1].set_xticks(range(len(conn_dist)))
        axes[0,1].set_xticklabels(conn_dist.index, rotation=0)
        axes[0,1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, conn_dist.values):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(conn_dist.values)*0.01, 
                         str(value), ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Performance by Network Tier (improved)
    if 'network_tier' in viz_data.columns:
        tier_stats = viz_data.groupby('network_tier')['performance_score'].agg(['mean', 'std', 'count']).reset_index()
        bars = axes[0,2].bar(tier_stats['network_tier'], tier_stats['mean'], 
                           yerr=tier_stats['std'], capsize=8, 
                           color='lightgreen', alpha=0.8, edgecolor='darkgreen', linewidth=0.8)
        axes[0,2].set_xlabel('Network Tier', fontweight='bold')
        axes[0,2].set_ylabel('Average Performance Score', fontweight='bold')
        axes[0,2].set_title('Performance by Network Tier', fontweight='bold', pad=10)  # Reduzi o pad
        axes[0,2].tick_params(axis='x', rotation=45)
        axes[0,2].grid(True, alpha=0.3, axis='y')
        
        # Add value labels and sample sizes
        for bar, mean_val, count in zip(bars, tier_stats['mean'], tier_stats['count']):
            axes[0,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                         f'{mean_val:.1f}\n(n={count})', ha='center', va='bottom', 
                         fontweight='bold', fontsize=9)
    
    # Plot 4: Top Contributors (improved)
    if 'performance_score' in viz_data.columns:
        top_users = viz_data.nlargest(10, 'performance_score')
        usernames = [name[:20] + '...' if len(str(name)) > 20 else str(name) for name in top_users['UserName']]
        bars = axes[1,0].barh(range(len(top_users)), top_users['performance_score'], 
                            color='orange', alpha=0.8, edgecolor='darkorange', linewidth=0.8)
        axes[1,0].set_yticks(range(len(top_users)))
        axes[1,0].set_yticklabels(usernames, fontsize=10)
        axes[1,0].set_xlabel('Performance Score', fontweight='bold')
        axes[1,0].set_title('Top 10 Performers', fontweight='bold', pad=10)  # Reduzi o pad
        axes[1,0].grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, top_users['performance_score'])):
            axes[1,0].text(bar.get_width() + max(top_users['performance_score'])*0.01, 
                         bar.get_y() + bar.get_height()/2, 
                         f'{score:.1f}', ha='left', va='center', fontweight='bold')
    
    # Plot 5: Network Metrics Heatmap (improved)
    if len(viz_data) > 1:
        network_cols = ['degree_centrality', 'betweenness_centrality', 'network_influence_score', 'num_connections']
        performance_cols = ['total_submissions', 'kernels_created', 'total_votes', 'performance_score']
        available_cols = [col for col in network_cols + performance_cols if col in viz_data.columns]
        
        if len(available_cols) > 1:
            corr_matrix = viz_data[available_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                       square=True, ax=axes[1,1], cbar_kws={'label': 'Correlation'},
                       fmt='.3f', linewidths=0.5)
            axes[1,1].set_title('Network-Performance Correlations', fontweight='bold', pad=10)  # Reduzi o pad
            axes[1,1].tick_params(axis='both', labelsize=9)
    
    # Plot 6: Activity Distribution (improved)
    if 'teams_joined' in viz_data.columns and 'kernels_created' in viz_data.columns:
        scatter = axes[1,2].scatter(viz_data['teams_joined'], viz_data['kernels_created'], 
                                  alpha=0.7, color='purple', s=50, edgecolors='white', linewidth=0.5)
        axes[1,2].set_xlabel('Teams Joined', fontweight='bold')
        axes[1,2].set_ylabel('Kernels Created', fontweight='bold')
        axes[1,2].set_title('Collaboration vs Creation Activity', fontweight='bold', pad=10)  # Reduzi o pad
        axes[1,2].grid(True, alpha=0.3)
        
        # Add trend line
        if len(viz_data) > 1:
            try:
                z = np.polyfit(viz_data['teams_joined'], viz_data['kernels_created'], 1)
                p = np.poly1d(z)
                axes[1,2].plot(viz_data['teams_joined'], p(viz_data['teams_joined']), 
                             "red", linewidth=2, alpha=0.8, label='Trend')
                axes[1,2].legend()
            except:
                pass
    
    # AJUSTE CRÃ�TICO: Usar tight_layout com padding maior e ajustar o top
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.35, wspace=0.25)  # Ajustei hspace e wspace para mais espaÃ§o
    plt.show()
    
    return fig

def create_data_exploration_dashboard():
    """Create data exploration dashboard based on available columns"""
    
    # MESMO AJUSTE PARA O DASHBOARD DE EXPLORAÃ‡ÃƒO
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))  # Aumentei a altura
    fig.suptitle('ğŸ“Š Kaggle Data Exploration Dashboard', fontsize=16, fontweight='bold', y=0.98)  # Ajustei y
    
    # Plot 1: Competitions over time (if date column exists)
    try:
        if 'DeadlineDate' in competitions.columns:
            comp_dates = pd.to_datetime(competitions['DeadlineDate'], errors='coerce').dropna()
            comp_dates.dt.year.value_counts().sort_index().plot(kind='bar', ax=axes[0,0], color='skyblue')
            axes[0,0].set_title('Competitions by Year', pad=8)  # Reduzi o pad
            axes[0,0].set_xlabel('Year')
            axes[0,0].set_ylabel('Number of Competitions')
            axes[0,0].tick_params(axis='x', rotation=45)
        else:
            axes[0,0].bar(['Total Competitions'], [len(competitions)], color='skyblue')
            axes[0,0].set_title(f'Total Competitions: {len(competitions):,}', pad=8)
    except Exception as e:
        axes[0,0].text(0.5, 0.5, f'Competition data\nnot available\n({str(e)[:50]})', 
                      ha='center', va='center', transform=axes[0,0].transAxes)
        axes[0,0].set_title('Competitions Analysis', pad=8)
    
    # Plot 2: Team size distribution
    try:
        team_sizes = team_memberships.groupby('TeamId').size()
        team_sizes.value_counts().head(10).plot(kind='bar', ax=axes[0,1], color='lightgreen')
        axes[0,1].set_title('Team Size Distribution', pad=8)
        axes[0,1].set_xlabel('Team Size')
        axes[0,1].set_ylabel('Number of Teams')
        axes[0,1].tick_params(axis='x', rotation=0)
    except Exception as e:
        axes[0,1].text(0.5, 0.5, f'Team size data\nnot available\n({str(e)[:50]})', 
                      ha='center', va='center', transform=axes[0,1].transAxes)
        axes[0,1].set_title('Team Sizes', pad=8)
    
    # Plot 3: User registration over time (if date column exists)
    try:
        if 'RegisterDate' in users.columns:
            user_dates = pd.to_datetime(users['RegisterDate'], errors='coerce').dropna()
            if len(user_dates) > 100000:
                user_dates = user_dates.sample(100000, random_state=42)
            user_dates.dt.year.value_counts().sort_index().plot(kind='line', ax=axes[0,2], color='orange', marker='o')
            axes[0,2].set_title('User Registrations by Year', pad=8)
            axes[0,2].set_xlabel('Year')
            axes[0,2].set_ylabel('Number of Users')
        else:
            axes[0,2].bar(['Total Users'], [len(users)], color='orange')
            axes[0,2].set_title(f'Total Users: {len(users):,}', pad=8)
    except Exception as e:
        axes[0,2].text(0.5, 0.5, f'User registration data\nnot available\n({str(e)[:50]})', 
                      ha='center', va='center', transform=axes[0,2].transAxes)
        axes[0,2].set_title('User Registrations', pad=8)
    
    # Plot 4: Submission activity (sample)
    try:
        sub_sample = submissions.sample(min(50000, len(submissions)), random_state=42)
        sub_counts = sub_sample.groupby('SubmittedUserId').size()
        sub_counts.value_counts().head(15).plot(kind='bar', ax=axes[1,0], color='red', alpha=0.7)
        axes[1,0].set_title('Submission Frequency Distribution', pad=8)
        axes[1,0].set_xlabel('Number of Submissions per User')
        axes[1,0].set_ylabel('Number of Users')
        axes[1,0].tick_params(axis='x', rotation=45)
    except Exception as e:
        axes[1,0].text(0.5, 0.5, f'Submission data\nnot available\n({str(e)[:50]})', 
                      ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('Submission Activity', pad=8)
    
    # Plot 5: Kernel votes distribution (sample)
    try:
        kernel_sample = kernels.sample(min(50000, len(kernels)), random_state=42)
        if 'TotalVotes' in kernel_sample.columns:
            vote_ranges = pd.cut(kernel_sample['TotalVotes'], bins=[0, 1, 5, 10, 50, 100, float('inf')], 
                               labels=['0', '1-5', '6-10', '11-50', '51-100', '100+'])
            vote_ranges.value_counts().plot(kind='bar', ax=axes[1,1], color='purple', alpha=0.7)
            axes[1,1].set_title('Kernel Votes Distribution', pad=8)
            axes[1,1].set_xlabel('Vote Range')
            axes[1,1].set_ylabel('Number of Kernels')
            axes[1,1].tick_params(axis='x', rotation=45)
        else:
            axes[1,1].bar(['Total Kernels'], [len(kernels)], color='purple')
            axes[1,1].set_title(f'Total Kernels: {len(kernels):,}', pad=8)
    except Exception as e:
        axes[1,1].text(0.5, 0.5, f'Kernel vote data\nnot available\n({str(e)[:50]})', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Kernel Votes', pad=8)
    
    # Plot 6: Network summary
    try:
        if len(analysis_data) > 0:
            if 'network_tier' in analysis_data.columns:
                analysis_data['network_tier'].value_counts().plot(kind='pie', ax=axes[1,2], autopct='%1.1f%%')
                axes[1,2].set_title('Network Tier Distribution', pad=8)
                axes[1,2].set_ylabel('')
            else:
                network_stats = ['Connected Users', 'Isolated Users']
                connected = len(centrality_results)
                isolated = len(user_performance) - connected
                axes[1,2].pie([connected, isolated], labels=network_stats, autopct='%1.1f%%')
                axes[1,2].set_title('Network Connectivity', pad=8)
        else:
            axes[1,2].text(0.5, 0.5, 'Network analysis\nnot available', 
                          ha='center', va='center', transform=axes[1,2].transAxes)
            axes[1,2].set_title('Network Summary', pad=8)
    except Exception as e:
        axes[1,2].text(0.5, 0.5, f'Network summary\nnot available\n({str(e)[:50]})', 
                      ha='center', va='center', transform=axes[1,2].transAxes)
        axes[1,2].set_title('Network Summary', pad=8)
    
    # AJUSTE DE ESPAÃ‡AMENTO
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.35, wspace=0.25)
    plt.show()
    
    return fig

def create_summary_metrics():
    """Create summary metrics visualization with light theme"""
    
    # Calculate key metrics
    total_users = len(user_performance)
    networked_users = len(centrality_results) if len(centrality_results) > 0 else 0
    total_teams = len(teams)
    total_collaborations = len(collaboration_network) if len(collaboration_network) > 0 else 0
    
    if len(analysis_data) > 0:
        avg_performance = analysis_data['performance_score'].mean()
        correlation = analysis_data[['network_influence_score', 'performance_score']].corr().iloc[0,1] if len(analysis_data) > 1 else 0
        max_connections = analysis_data['num_connections'].max()
        avg_connections = analysis_data['num_connections'].mean()
    else:
        avg_performance = 0
        correlation = 0
        max_connections = 0
        avg_connections = 0
    
    # AJUSTE: Mais espaÃ§o entre mÃ©tricas
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Ajustei a altura
    fig.suptitle('ğŸ“Š Kaggle Social Graph - Key Metrics Dashboard', 
                 fontsize=18, fontweight='bold', y=0.95)  # Mantive y=0.95 para mÃ©tricas
    
    # Define colors for each metric (professional color scheme)
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', 
              '#9b59b6', '#1abc9c', '#34495e', '#95a5a6']
    
    metrics = [
        {"title": "Total Users", "value": total_users, "format": "{:,.0f}", "desc": "Platform Users"},
        {"title": "Networked Users", "value": networked_users, "format": "{:,.0f}", "desc": "In Collaboration Network"},
        {"title": "Teams Formed", "value": total_teams, "format": "{:,.0f}", "desc": "Total Teams Created"},
        {"title": "Collaborations", "value": total_collaborations, "format": "{:,.0f}", "desc": "Active Connections"},
        {"title": "Avg Performance", "value": avg_performance, "format": "{:.1f}", "desc": "Performance Score"},
        {"title": "Network Correlation", "value": correlation, "format": "{:.3f}", "desc": "Network-Performance"},
        {"title": "Max Connections", "value": max_connections, "format": "{:,.0f}", "desc": "Most Connected User"},
        {"title": "Avg Connections", "value": avg_connections, "format": "{:.1f}", "desc": "Per Networked User"}
    ]
    
    for i, metric in enumerate(metrics):
        row = i // 4
        col = i % 4
        
        # Clear the axis
        axes[row, col].clear()
        
        # Create a clean metric card design
        axes[row, col].text(0.5, 0.7, metric["title"], 
                           ha='center', va='center', fontsize=14, fontweight='bold',
                           transform=axes[row, col].transAxes)
        axes[row, col].text(0.5, 0.4, metric["format"].format(metric["value"]), 
                           ha='center', va='center', fontsize=20, fontweight='bold',
                           color=colors[i], transform=axes[row, col].transAxes)
        axes[row, col].text(0.5, 0.15, metric["desc"], 
                           ha='center', va='center', fontsize=10, style='italic',
                           transform=axes[row, col].transAxes)
        
        # Set clean appearance
        axes[row, col].set_xlim(0, 1)
        axes[row, col].set_ylim(0, 1)
        axes[row, col].set_facecolor('#f8f9fa')
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        
        # Add modern border
        for spine in axes[row, col].spines.values():
            spine.set_edgecolor(colors[i])
            spine.set_linewidth(3)
            spine.set_alpha(0.8)
    
    # AJUSTE ESPECÃ�FICO PARA MÃ‰TRICAS
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, hspace=0.4, wspace=0.15)  # Mais espaÃ§o vertical
    plt.show()
    
    return fig

def create_network_graph_viz():
    """Create network graph visualization"""
    
    if network_graph is None or len(centrality_results) == 0:
        print("âš  No network data for graph visualization")
        return None
    
    # Sample nodes for visualization if network is too large
    if network_graph.number_of_nodes() > 100:
        top_nodes = sorted(network_graph.nodes(), key=lambda x: network_graph.degree(x), reverse=True)[:50]
        G_viz = network_graph.subgraph(top_nodes).copy()
        print(f"Visualizing top {len(top_nodes)} nodes from network")
    else:
        G_viz = network_graph
    
    # Create network visualization
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    # Calculate layout
    pos = nx.spring_layout(G_viz, k=1, iterations=50)
    
    # Node sizes based on degree
    node_sizes = [G_viz.degree(node) * 100 for node in G_viz.nodes()]
    
    # Node colors based on centrality
    centrality_dict = dict(zip(centrality_results['user_id'], centrality_results['degree_centrality']))
    node_colors = [centrality_dict.get(node, 0) for node in G_viz.nodes()]
    
    # Draw network
    nx.draw_networkx_edges(G_viz, pos, alpha=0.3, edge_color='gray', ax=ax)
    nodes = nx.draw_networkx_nodes(G_viz, pos, node_size=node_sizes, 
                                  node_color=node_colors, cmap='viridis', 
                                  alpha=0.8, ax=ax)
    
    # Add colorbar
    plt.colorbar(nodes, ax=ax, label='Degree Centrality')
    
    # TÃ�TULO SEM SOBREPOSIÃ‡ÃƒO
    ax.set_title('ğŸ•¸ï¸� Kaggle Collaboration Network\n(Top Connected Users)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Create all visualizations
print("Creating data exploration dashboard...")
exploration_viz = create_data_exploration_dashboard()

print("Creating network analysis visualizations...")
main_viz = create_matplotlib_visualizations()

print("Creating metrics dashboard...")
metrics_viz = create_summary_metrics()

print("Creating network graph...")
network_viz = create_network_graph_viz()

# ================================================================================
# STEP 7: DISPLAY RESULTS
# ================================================================================

print("\nğŸ�‰ STEP 7: Results Summary")
print("-" * 40)

print("ğŸŒ� KAGGLE SOCIAL GRAPH - ANALYSIS COMPLETE! (FIXED)")
print("=" * 55)

# Summary statistics
print(f"ğŸ“Š DATA PROCESSED:")
print(f"   â€¢ Users analyzed: {len(user_performance):,}")
print(f"   â€¢ Collaboration edges: {len(collaboration_network):,}")
print(f"   â€¢ Network nodes: {len(centrality_results):,}")
print(f"   â€¢ Teams: {len(teams):,}")

if len(analysis_data) > 0:
    print(f"\nğŸ”— KEY FINDINGS:")
    if 'network_influence_score' in analysis_data.columns and 'performance_score' in analysis_data.columns:
        corr = analysis_data[['network_influence_score', 'performance_score']].corr().iloc[0,1]
        print(f"   â€¢ Network-Performance Correlation: {corr:.3f}")
        
        avg_connections = analysis_data['num_connections'].mean()
        max_connections = analysis_data['num_connections'].max()
        print(f"   â€¢ Average connections per user: {avg_connections:.1f}")
        print(f"   â€¢ Most connected user: {max_connections:,} connections")
        
        # Performance insights
        print(f"   â€¢ Average performance score: {analysis_data['performance_score'].mean():.2f}")
        print(f"   â€¢ Performance score range: {analysis_data['performance_score'].min():.2f} - {analysis_data['performance_score'].max():.2f}")

print(f"\nâœ… ANALYSIS READY FOR PRESENTATION!")
print("ğŸ”§ FIXES APPLIED:")
print("   â€¢ Fixed duplicate quantile edges with 'duplicates=drop'")
print("   â€¢ Added noise to prevent ties in network scores")
print("   â€¢ Improved error handling for tier analysis")
print("   â€¢ Enhanced visualizations with better formatting")

# Display visualizations
print("\nğŸ“Š Displaying Data Exploration...")
if exploration_viz:
    print("âœ“ Data exploration dashboard created")

print("\nğŸ“ˆ Displaying Network Analysis...")
if main_viz:
    print("âœ“ Comprehensive analysis dashboard created")

print("\nğŸ“Š Displaying Key Metrics...")
if metrics_viz:
    print("âœ“ Metrics dashboard created")

print("\nğŸ•¸ï¸� Displaying Network Graph...")
if network_viz:
    print("âœ“ Network visualization created")

# Enhanced text summary with data validation
print("\nğŸ“‹ ENHANCED SUMMARY REPORT:")
print("=" * 60)

# Data availability summary
print(f"ğŸ“� DATA INVENTORY:")
print(f"   â€¢ Total competitions: {len(competitions):,}")
print(f"   â€¢ Total teams: {len(teams):,}")
print(f"   â€¢ Total users: {len(users):,}")
print(f"   â€¢ Total submissions: {len(submissions):,}")
print(f"   â€¢ Total kernels: {len(kernels):,}")
print(f"   â€¢ Team memberships: {len(team_memberships):,}")

# Available columns summary
print(f"\nğŸ”� KEY DATA COLUMNS:")
print(f"   â€¢ Competitions: {', '.join(competitions.columns[:5])}{'...' if len(competitions.columns) > 5 else ''}")
print(f"   â€¢ Users: {', '.join(users.columns[:5])}{'...' if len(users.columns) > 5 else ''}")
print(f"   â€¢ Teams: {', '.join(teams.columns[:3])}{'...' if len(teams.columns) > 3 else ''}")

if len(analysis_data) > 0:
    print(f"\nğŸ�¯ CORRELATION ANALYSIS:")
    if 'network_influence_score' in analysis_data.columns and 'performance_score' in analysis_data.columns:
        corr = analysis_data[['network_influence_score', 'performance_score']].corr().iloc[0,1]
        print(f"   Network-Performance Correlation: {corr:.4f}")
        
        if abs(corr) > 0.3:
            print(f"   ğŸ”¥ STRONG correlation detected!")
        elif abs(corr) > 0.1:
            print(f"   ğŸ“ˆ MODERATE correlation detected")
        else:
            print(f"   ğŸ“Š WEAK correlation")
    
    print(f"\nğŸ“Š NETWORK STATISTICS:")
    print(f"   Average connections: {analysis_data['num_connections'].mean():.2f}")
    print(f"   Max connections: {analysis_data['num_connections'].max()}")
    print(f"   Users in network: {len(analysis_data):,}")
    print(f"   Network coverage: {len(analysis_data)/len(users)*100:.2f}% of all users")
    
    if len(centrality_results) > 1:
        density = len(collaboration_network) / (len(centrality_results) * (len(centrality_results) - 1) / 2)
        print(f"   Network density: {density:.6f}")
    
    print(f"\nğŸ�† PERFORMANCE INSIGHTS:")
    print(f"   Average performance score: {analysis_data['performance_score'].mean():.2f}")
    print(f"   Top performer score: {analysis_data['performance_score'].max():.2f}")
    print(f"   Performance range: {analysis_data['performance_score'].min():.2f} - {analysis_data['performance_score'].max():.2f}")
    print(f"   Performance std dev: {analysis_data['performance_score'].std():.2f}")
    
    # Network advantage analysis
    if len(analysis_data) > 20:  # Need sufficient data
        high_network = analysis_data[analysis_data['network_influence_score'] > analysis_data['network_influence_score'].quantile(0.8)]
        low_network = analysis_data[analysis_data['network_influence_score'] < analysis_data['network_influence_score'].quantile(0.2)]
        
        if len(high_network) > 0 and len(low_network) > 0:
            high_avg = high_network['performance_score'].mean()
            low_avg = low_network['performance_score'].mean()
            if low_avg > 0:
                advantage = (high_avg / low_avg - 1) * 100
                print(f"   ğŸš€ Network advantage: Top 20% perform {advantage:+.1f}% better than bottom 20%")
    
    # Top performers with network context
    top_5 = analysis_data.nlargest(5, 'performance_score')[['UserName', 'performance_score', 'num_connections', 'network_influence_score']]
    print(f"\nğŸŒŸ TOP 5 PERFORMERS:")
    for idx, row in top_5.iterrows():
        print(f"   {row['UserName']}: Performance={row['performance_score']:.2f}, Connections={row['num_connections']}, Network Score={row['network_influence_score']:.4f}")
    
    # Most connected users with performance context
    top_connected = analysis_data.nlargest(5, 'num_connections')[['UserName', 'num_connections', 'performance_score', 'network_influence_score']]
    print(f"\nğŸ¤� MOST CONNECTED USERS:")
    for idx, row in top_connected.iterrows():
        print(f"   {row['UserName']}: {row['num_connections']} connections, Performance={row['performance_score']:.2f}, Network Score={row['network_influence_score']:.4f}")

else:
    print(f"\nâš  Limited network analysis available")
    print(f"   Processed {len(user_performance):,} users")
    print(f"   Found {len(collaboration_network):,} collaboration edges")
    print(f"   Network nodes: {len(centrality_results):,}")

# Team analysis
if len(team_memberships) > 0:
    team_sizes = team_memberships.groupby('TeamId').size()
    print(f"\nğŸ‘¥ TEAM COLLABORATION INSIGHTS:")
    print(f"   Average team size: {team_sizes.mean():.2f}")
    print(f"   Largest team: {team_sizes.max()} members")
    print(f"   Teams with 1 member: {(team_sizes == 1).sum():,} ({(team_sizes == 1).sum()/len(team_sizes)*100:.1f}%)")
    print(f"   Teams with 2+ members: {(team_sizes > 1).sum():,} ({(team_sizes > 1).sum()/len(team_sizes)*100:.1f}%)")

# ================================================================================
# STEP 8: ADVANCED INSIGHTS & RECOMMENDATIONS
# ================================================================================

print("\nğŸ”® STEP 8: Advanced Insights & Recommendations")
print("-" * 60)

print("\nğŸ’¡ KEY INSIGHTS:")

if len(analysis_data) > 0:
    # Insight 1: Network Effect Analysis
    if 'network_influence_score' in analysis_data.columns and 'performance_score' in analysis_data.columns:
        corr = analysis_data[['network_influence_score', 'performance_score']].corr().iloc[0,1]
        
        if corr > 0.1:
            print("ğŸ”— POSITIVE NETWORK EFFECT:")
            print("   Users with stronger network connections tend to perform better.")
            print("   This suggests collaboration and networking enhance performance.")
        elif corr < -0.1:
            print("ğŸ”— NEGATIVE NETWORK EFFECT:")
            print("   Highly networked users may be spending time on collaboration")
            print("   rather than individual achievement. Quality vs quantity trade-off.")
        else:
            print("ğŸ”— NEUTRAL NETWORK EFFECT:")
            print("   Network connections don't strongly predict performance.")
            print("   Individual skill may be more important than networking.")
    
    # Insight 2: Team Composition Analysis
    if len(team_memberships) > 0:
        team_sizes = team_memberships.groupby('TeamId').size()
        solo_teams = (team_sizes == 1).sum()
        multi_teams = (team_sizes > 1).sum()
        
        print(f"\nğŸ‘¥ TEAM COMPOSITION INSIGHTS:")
        if solo_teams > multi_teams:
            print("   ğŸ�ƒ INDIVIDUAL PREFERENCE: Most teams are solo efforts")
            print("   This suggests either competitive culture or difficulty in collaboration")
        else:
            print("   ğŸ¤� COLLABORATIVE CULTURE: Most teams have multiple members")
            print("   This indicates strong collaboration preferences in the community")
        
        avg_team_size = team_sizes.mean()
        if avg_team_size > 3:
            print(f"   ğŸ“ˆ LARGE TEAMS: Average {avg_team_size:.1f} members suggests complex challenges")
        elif avg_team_size > 2:
            print(f"   ğŸ“Š MEDIUM TEAMS: Average {avg_team_size:.1f} members shows balanced collaboration")
        else:
            print(f"   ğŸ“‰ SMALL TEAMS: Average {avg_team_size:.1f} members indicates individual focus")

# Insight 3: Platform Activity Patterns
print(f"\nğŸ“Š PLATFORM ACTIVITY PATTERNS:")
total_activities = len(submissions) + len(kernels)
if total_activities > 0:
    submission_ratio = len(submissions) / total_activities
    kernel_ratio = len(kernels) / total_activities
    
    if submission_ratio > 0.7:
        print("   ğŸ�¯ COMPETITION-FOCUSED: High submission-to-content ratio")
        print("   Users primarily engage through competitions rather than knowledge sharing")
    elif kernel_ratio > 0.7:
        print("   ğŸ“š LEARNING-FOCUSED: High content-to-submission ratio")
        print("   Users primarily engage through learning and knowledge sharing")
    else:
        print("   âš–ï¸� BALANCED ENGAGEMENT: Even split between competition and learning")

# Insight 4: User Engagement Levels
if len(user_performance) > 0:
    active_users = user_performance[user_performance['performance_score'] > 0]
    engagement_rate = len(active_users) / len(user_performance)
    
    print(f"\nğŸ‘¤ USER ENGAGEMENT ANALYSIS:")
    print(f"   Active users: {len(active_users):,} ({engagement_rate*100:.1f}% of total)")
    
    if engagement_rate > 0.5:
        print("   ğŸ”¥ HIGH ENGAGEMENT: Majority of users are actively participating")
    elif engagement_rate > 0.2:
        print("   ğŸ“ˆ MODERATE ENGAGEMENT: Significant portion of users are active")
    else:
        print("   ğŸ“‰ LOW ENGAGEMENT: Most users are passive or inactive")

print("\nğŸ�¯ STRATEGIC RECOMMENDATIONS:")
print("-" * 40)

# Recommendation 1: Based on network effects
if len(analysis_data) > 0 and 'network_influence_score' in analysis_data.columns:
    corr = analysis_data[['network_influence_score', 'performance_score']].corr().iloc[0,1]
    
    if corr > 0.1:
        print("1. ğŸ¤� FOSTER COLLABORATION:")
        print("   â€¢ Create team-building features and incentives")
        print("   â€¢ Implement mentorship programs pairing experienced with new users")
        print("   â€¢ Develop collaborative competitions with shared rewards")
    elif corr < -0.1:
        print("1. âš–ï¸� BALANCE COLLABORATION:")
        print("   â€¢ Create separate tracks for individual vs team achievements")
        print("   â€¢ Implement time management tools for networked users")
        print("   â€¢ Focus on quality collaborations over quantity")
    else:
        print("1. ğŸ�¯ FOCUS ON INDIVIDUAL EXCELLENCE:")
        print("   â€¢ Emphasize skill development and personal achievement")
        print("   â€¢ Create individual learning paths and challenges")
        print("   â€¢ Network effects are secondary to individual capabilities")

# Recommendation 2: Platform engagement
print("\n2. ğŸ“Š ENHANCE PLATFORM ENGAGEMENT:")
if len(user_performance) > 0:
    active_users = user_performance[user_performance['performance_score'] > 0]
    engagement_rate = len(active_users) / len(user_performance)
    
    if engagement_rate < 0.3:
        print("   â€¢ Implement onboarding programs for new users")
        print("   â€¢ Create beginner-friendly competitions and tutorials")
        print("   â€¢ Develop gamification elements to encourage participation")
    else:
        print("   â€¢ Focus on retaining high-performing users")
        print("   â€¢ Create advanced challenges for experienced participants")
        print("   â€¢ Develop community leadership programs")

# Recommendation 3: Data-driven insights
print("\n3. ğŸ”¬ LEVERAGE DATA INSIGHTS:")
print("   â€¢ Monitor network formation patterns to predict collaboration success")
print("   â€¢ Use performance correlation data to optimize team formation algorithms")
print("   â€¢ Track engagement metrics to identify at-risk user segments")

# Recommendation 4: Community building
print("\n4. ğŸŒŸ BUILD STRONGER COMMUNITY:")
print("   â€¢ Recognize and highlight top performers and connectors")
print("   â€¢ Create forums and discussion spaces for knowledge sharing")
print("   â€¢ Implement user reputation and recognition systems")

# Final statistics (moved to proper indentation level)
if len(analysis_data) > 0:
    print(f"\nğŸ“Š NETWORK STATISTICS:")
    print(f"   Average connections: {analysis_data['num_connections'].mean():.2f}")
    print(f"   Max connections: {analysis_data['num_connections'].max()}")
    
    # Safe network density calculation
    if 'collaboration_network' in globals() and 'centrality_results' in globals():
        if len(centrality_results) > 1:
            network_density = len(collaboration_network) / (len(centrality_results) * (len(centrality_results) - 1) / 2)
            print(f"   Network density: {network_density:.6f}")
        else:
            print("   Network density: N/A")
    else:
        print("   Network density: N/A")
    
    print(f"\nğŸ�† PERFORMANCE INSIGHTS:")
    print(f"   Average performance score: {analysis_data['performance_score'].mean():.2f}")
    print(f"   Top performer score: {analysis_data['performance_score'].max():.2f}")
    print(f"   Performance std dev: {analysis_data['performance_score'].std():.2f}")
    
    # Top performers
    if 'UserName' in analysis_data.columns:
        top_5 = analysis_data.nlargest(5, 'performance_score')[['UserName', 'performance_score', 'num_connections']]
        print(f"\nğŸŒŸ TOP 5 PERFORMERS:")
        for idx, row in top_5.iterrows():
            print(f"   {row['UserName']}: {row['performance_score']:.2f} (connections: {row['num_connections']})")
        
        # Most connected users
        top_connected = analysis_data.nlargest(5, 'num_connections')[['UserName', 'num_connections', 'performance_score']]
        print(f"\nğŸ¤� MOST CONNECTED USERS:")
        for idx, row in top_connected.iterrows():
            print(f"   {row['UserName']}: {row['num_connections']} connections (performance: {row['performance_score']:.2f})")
    else:
        print(f"\nğŸŒŸ TOP 5 PERFORMERS:")
        top_5 = analysis_data.nlargest(5, 'performance_score')[['performance_score', 'num_connections']]
        for idx, row in top_5.iterrows():
            print(f"   User {idx}: {row['performance_score']:.2f} (connections: {row['num_connections']})")
        
        print(f"\nğŸ¤� MOST CONNECTED USERS:")
        top_connected = analysis_data.nlargest(5, 'num_connections')[['num_connections', 'performance_score']]
        for idx, row in top_connected.iterrows():
            print(f"   User {idx}: {row['num_connections']} connections (performance: {row['performance_score']:.2f})")

print("\nğŸš€ SOCIAL GRAPH ANALYSIS COMPLETE!")
print("=" * 60)
print("âœ¨ COMPREHENSIVE ANALYSIS READY FOR PRESENTATION!")
print("ğŸ“ˆ All visualizations and insights generated successfully")
print("ğŸ�¯ Strategic recommendations provided based on data analysis")
print("ğŸ”¬ Ready for research paper writeup and competition submission!")


# ================================================================================
# FINAL DASHBOARD SUMMARY 
# ================================================================================

def show_final_summary():
    """Display final analysis summary with error handling"""
    
    print("\nğŸ�‰ KAGGLE SOCIAL GRAPH ANALYSIS COMPLETE!")
    print("="*50)
    
    print("ğŸ“Š ANALYSIS HIGHLIGHTS:")
    
    # Check if variables exist
    try:
        if 'analysis_data' in globals() and len(analysis_data) > 0:
            total_users = len(user_performance) if 'user_performance' in globals() else 0
            networked_users = len(centrality_results) if 'centrality_results' in globals() else 0
            
            print(f"   â€¢ Total users analyzed: {total_users:,}")
            print(f"   â€¢ Users in collaboration network: {networked_users:,}")
            
            if 'network_influence_score' in analysis_data.columns and 'performance_score' in analysis_data.columns:
                corr = analysis_data[['network_influence_score', 'performance_score']].corr().iloc[0,1]
                print(f"   â€¢ Network-Performance correlation: {corr:.3f}")
                
                high_net = analysis_data[analysis_data['network_influence_score'] > analysis_data['network_influence_score'].quantile(0.8)]
                low_net = analysis_data[analysis_data['network_influence_score'] < analysis_data['network_influence_score'].quantile(0.2)]
                
                if len(high_net) > 0 and len(low_net) > 0:
                    advantage = (high_net['performance_score'].mean() / low_net['performance_score'].mean() - 1) * 100
                    print(f"   â€¢ Network advantage: +{advantage:.1f}%")
            else:
                print("   â€¢ Network metrics not available in dataset")
        
        elif 'user_performance' in globals():
            # Fallback if only basic data is available
            total_users = len(user_performance)
            print(f"   â€¢ Total users processed: {total_users:,}")
            
            if 'centrality_results' in globals():
                networked_users = len(centrality_results)
                print(f"   â€¢ Users in collaboration network: {networked_users:,}")
            
            if 'collaboration_network' in globals():
                total_collaborations = len(collaboration_network)
                print(f"   â€¢ Collaboration edges created: {total_collaborations:,}")
            
            if 'teams' in globals():
                total_teams = len(teams)
                print(f"   â€¢ Teams analyzed: {total_teams:,}")
        
        else:
            print("   â€¢ Analysis data not found. Please run the main analysis first.")
            
    except Exception as e:
        print(f"   â€¢ Error in summary generation: {str(e)}")
        print("   â€¢ Please check that the main analysis completed successfully.")
    
    print("\nğŸš€ READY FOR WRITEUP AND SUBMISSION!")

# Alternative: Quick status check
def quick_status_check():
    """Quick check of what data is available"""
    
    print("\nğŸ“‹ DATA STATUS CHECK:")
    print("-" * 30)
    
    data_status = {
        'user_performance': 'user_performance' in globals(),
        'centrality_results': 'centrality_results' in globals(),
        'analysis_data': 'analysis_data' in globals(),
        'collaboration_network': 'collaboration_network' in globals(),
        'teams': 'teams' in globals(),
        'network_graph': 'network_graph' in globals()
    }
    
    for data_name, exists in data_status.items():
        status = "âœ…" if exists else "â�Œ"
        count = ""
        
        if exists:
            try:
                data_obj = globals()[data_name]
                if hasattr(data_obj, '__len__'):
                    count = f" ({len(data_obj):,} records)"
            except:
                count = " (data exists)"
        
        print(f"{status} {data_name}{count}")
    
    # Show available columns if analysis_data exists
    if data_status['analysis_data']:
        try:
            cols = list(analysis_data.columns)
            print(f"\nğŸ“Š Available columns in analysis_data:")
            print(f"   {', '.join(cols[:10])}{'...' if len(cols) > 10 else ''}")
        except:
            pass

# Run both checks
print("ğŸ”� Checking data availability...")
quick_status_check()

print("\n" + "="*60)
show_final_summary()




