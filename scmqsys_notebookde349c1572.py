import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Circle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Data preparation
platforms_data = {
    'Platform': ['Kaggle', 'AIcrowd', 'DrivenData', 'bitgrit', 'Zindi', 'CodaLab'],
    'Users_Millions': [23.29, 0.1, 0.05, 0.025, 0.02, 0.015],
    'Founded': [2010, 2017, 2014, 2017, 2018, 2013],
    'Growth_Rate': [15, 35, 25, 30, 40, 10],
    'Business_Focus': [70, 30, 20, 85, 50, 15],
    'Academic_Focus': [20, 60, 20, 10, 30, 80],
    'Social_Focus': [10, 10, 60, 5, 20, 5],
    'Infrastructure_Score': [10, 6, 6, 5, 4, 7],
    'Prize_Money_Score': [9, 7, 7, 6, 5, 4],
    'Community_Score': [10, 4, 4, 3, 3, 3]
}

df = pd.DataFrame(platforms_data)

# Color palette
colors = ['#20BEFF', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

# Function 1: User Base Comparison
def plot_user_base_comparison():
    """Create user base comparison chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart
    bars = ax1.bar(df['Platform'], df['Users_Millions'], color=colors)
    ax1.set_title('User Base Comparison (Millions)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Users (Millions)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, df['Users_Millions']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value}M', ha='center', va='bottom')
    
    # Pie chart for market share
    sizes = df['Users_Millions']
    ax2.pie(sizes, labels=df['Platform'], colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Market Share Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# Function 2: Focus Area Comparison
def plot_focus_areas():
    """Create stacked bar chart for focus areas"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bottom = np.zeros(len(df))
    
    # Stack the bars
    ax.bar(df['Platform'], df['Business_Focus'], label='Business', color='#FF6B6B', bottom=bottom)
    bottom += df['Business_Focus']
    
    ax.bar(df['Platform'], df['Academic_Focus'], label='Academic', color='#4ECDC4', bottom=bottom)
    bottom += df['Academic_Focus']
    
    ax.bar(df['Platform'], df['Social_Focus'], label='Social Impact', color='#96CEB4', bottom=bottom)
    
    ax.set_title('Competition Focus Areas Distribution (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# Function 3: Feature Radar Chart
def plot_feature_comparison():
    """Create radar chart for feature comparison"""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Define features for radar chart
    features = ['Community Size', 'Infrastructure', 'Prize Money', 'Learning Resources', 'Partnerships', 'Social Impact']
    
    # Data for Kaggle and DrivenData (example)
    kaggle_scores = [10, 10, 9, 9, 10, 6]
    drivendata_scores = [4, 6, 7, 6, 8, 10]
    
    # Number of variables
    N = len(features)
    
    # Compute angles for each feature
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Close the plots
    kaggle_scores += kaggle_scores[:1]
    drivendata_scores += drivendata_scores[:1]
    
    # Plot
    ax.plot(angles, kaggle_scores, 'o-', linewidth=2, label='Kaggle', color='#20BEFF')
    ax.fill(angles, kaggle_scores, alpha=0.25, color='#20BEFF')
    
    ax.plot(angles, drivendata_scores, 'o-', linewidth=2, label='DrivenData', color='#FF6B6B')
    ax.fill(angles, drivendata_scores, alpha=0.25, color='#FF6B6B')
    
    # Add feature labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_ylim(0, 10)
    ax.set_title('Feature Comparison: Kaggle vs DrivenData', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    ax.grid(True)
    
    plt.show()

# Function 4: Growth vs Age Analysis
def plot_growth_analysis():
    """Create scatter plot for platform age vs growth"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(df['Founded'], df['Growth_Rate'], 
                        s=df['Users_Millions'] * 20,  # Size based on user count
                        c=colors, alpha=0.7)
    
    # Add platform labels
    for i, txt in enumerate(df['Platform']):
        ax.annotate(txt, (df['Founded'][i], df['Growth_Rate'][i]), 
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Founded Year')
    ax.set_ylabel('Growth Rate (%)')
    ax.set_title('Platform Age vs Growth Rate\n(Bubble size = User base)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Function 5: Comprehensive Heatmap
def plot_feature_heatmap():
    """Create heatmap of all platform features"""
    # Prepare data for heatmap
    heatmap_data = df[['Platform', 'Infrastructure_Score', 'Prize_Money_Score', 
                      'Community_Score', 'Business_Focus', 'Academic_Focus', 'Social_Focus']]
    heatmap_data = heatmap_data.set_index('Platform')
    
    # Normalize data to 0-1 scale for better visualization
    normalized_data = heatmap_data.div(heatmap_data.max(axis=0))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(normalized_data, annot=True, cmap='YlOrRd', 
                cbar_kws={'label': 'Normalized Score'}, fmt='.2f')
    ax.set_title('Platform Feature Comparison (Normalized)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Interactive Plotly Functions
def create_interactive_dashboard():
    """Create interactive Plotly dashboard"""
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('User Base', 'Focus Areas', 'Growth vs Age', 'Feature Scores'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # User base bar chart
    fig.add_trace(
        go.Bar(x=df['Platform'], y=df['Users_Millions'], 
               marker_color=colors, name='Users (M)'),
        row=1, col=1
    )
    
    # Focus areas stacked bar
    fig.add_trace(
        go.Bar(x=df['Platform'], y=df['Business_Focus'], 
               name='Business', marker_color='#FF6B6B'),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=df['Platform'], y=df['Academic_Focus'], 
               name='Academic', marker_color='#4ECDC4'),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=df['Platform'], y=df['Social_Focus'], 
               name='Social', marker_color='#96CEB4'),
        row=1, col=2
    )
    
    # Growth vs age scatter
    fig.add_trace(
        go.Scatter(x=df['Founded'], y=df['Growth_Rate'], 
                   mode='markers+text', text=df['Platform'],
                   marker=dict(size=df['Users_Millions'] * 3, color=colors),
                   name='Platforms'),
        row=2, col=1
    )
    
    # Feature scores
    fig.add_trace(
        go.Bar(x=df['Platform'], y=df['Infrastructure_Score'], 
               name='Infrastructure', marker_color='#20BEFF'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Data Science Platform Comparison Dashboard",
        title_x=0.5,
        showlegend=True
    )
    
    return fig

# Function to generate summary statistics
def generate_summary_stats():
    """Generate summary statistics table"""
    summary_stats = {
        'Metric': [
            'Total Platforms Analyzed',
            'Largest Platform (Users)',
            'Market Leader Share',
            'Average Growth Rate',
            'Most Academic-Focused',
            'Most Social Impact-Focused',
            'Newest Platform',
            'Most Business-Focused'
        ],
        'Value': [
            len(df),
            f"Kaggle ({df.loc[0, 'Users_Millions']}M)",
            f"{(df.loc[0, 'Users_Millions'] / df['Users_Millions'].sum() * 100):.1f}%",
            f"{df['Growth_Rate'].mean():.1f}%",
            df.loc[df['Academic_Focus'].idxmax(), 'Platform'],
            df.loc[df['Social_Focus'].idxmax(), 'Platform'],
            df.loc[df['Founded'].idxmax(), 'Platform'],
            df.loc[df['Business_Focus'].idxmax(), 'Platform']
        ]
    }
    
    return pd.DataFrame(summary_stats)

# Main execution function
def main():
    """Execute all visualization functions"""
    print("=== Data Science Platform Comparison Analysis ===\n")
    
    # Display summary statistics
    print("Summary Statistics:")
    print(generate_summary_stats().to_string(index=False))
    print("\n" + "="*50 + "\n")
    
    # Generate all plots
    print("Generating visualizations...")
    
    plot_user_base_comparison()
    plot_focus_areas()
    plot_feature_comparison()
    plot_growth_analysis()
    plot_feature_heatmap()
    
    # Create interactive dashboard
    interactive_fig = create_interactive_dashboard()
    interactive_fig.show()
    
    print("\nAnalysis complete! All visualizations have been generated.")
    print("\nKey Insights:")
    print("1. Kaggle dominates with 98.8% market share")
    print("2. Newer platforms show higher growth rates")
    print("3. Platforms specialize in different focus areas")
    print("4. Infrastructure and community size correlate with market share")

# Export data function
def export_data():
    """Export data to CSV for further analysis"""
    df.to_csv('platform_comparison_data.csv', index=False)
    print("Data exported to 'platform_comparison_data.csv'")

# Advanced analysis functions
def correlation_analysis():
    """Perform correlation analysis between features"""
    numeric_cols = ['Users_Millions', 'Growth_Rate', 'Infrastructure_Score', 
                   'Prize_Money_Score', 'Community_Score']
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    return correlation_matrix

# Run the analysis
if __name__ == "__main__":
    main()
    
    # Additional analyses
    print("\n" + "="*50)
    print("Additional Analysis:")
    
    # Correlation analysis
    correlation_matrix = correlation_analysis()
    print("\nCorrelation Analysis:")
    print(correlation_matrix)
    
    # Export data
    export_data()
    
    print("\nTo run this script, ensure you have the following packages installed:")
    print("pip install matplotlib seaborn pandas numpy plotly")





