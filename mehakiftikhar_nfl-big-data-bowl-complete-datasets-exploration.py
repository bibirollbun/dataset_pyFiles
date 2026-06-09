# Import Basis
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
# Import Plotly.go
import plotly.graph_objects as go
# import Subplots
from plotly.subplots import make_subplots
import plotly.io as pio
import seaborn as sns 
import math
from io import StringIO
from colorama import Fore, Style, init;
# Import necessary libraries
from IPython.core.display import display, HTML
from scipy.stats import skew  

from sklearn.preprocessing import LabelEncoder, MinMaxScaler , StandardScaler , QuantileTransformer
from sklearn.impute import SimpleImputer

# Set the default renderer for both Plotly Express and Graph Objects
pio.renderers.default = 'iframe_connected'
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Set the option to display all columns
pd.set_option('display.max_columns', None)


# Function to load, merge, and display NFL datasets with styled output
def load_and_merge_nfl_data():
    """
    This function loads, renames, and merges multiple NFL datasets:
    - games.csv
    - player_play.csv
    - players.csv
    - plays.csv

    Returns:
        merged_df: The final merged DataFrame containing all relevant information.
    """
    import pandas as pd
    from IPython.display import display, HTML

    # Helper function for styled headings
    def styled_heading(text):
        return f"""
        <div style='
            background-color: #ba8d63; 
            color: #382411; 
            padding: 10px; 
            font-size: 18px; 
            border-radius: 5px; 
            text-align: center;'>
            {text}
        </div>
        """

    # Load the data
    print("Loading datasets...")
    df_games = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/games.csv")
    df_player_play = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/player_play.csv")
    df_players = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/players.csv")
    df_plays = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/plays.csv")
    display(HTML(styled_heading("Datasets loaded successfully!")))

    # common columns
    display(HTML(styled_heading("Checking for duplicate column names...")))
    common_1 = set(df_games.columns).intersection(set(df_player_play.columns))
    common_2 = set(df_player_play.columns).intersection(set(df_plays.columns))
    common_3 = set(df_plays.columns).intersection(set(df_players.columns))
    display(HTML(f"""
        <ul>
            <li><b>Common columns between <code>df_games</code> and <code>df_player_play</code>: </b>{common_1}</li>
            <li><b>Common columns between <code>df_player_play</code> and <code>df_plays</code>: </b>{common_2}</li>
            <li><b>Common columns between <code>df_plays</code> and <code>df_players</code>: </b>{common_3}</li>
        </ul>
    """))

    # Rename conflicting columns
    df_player_play.rename(columns={'penaltyYards': 'penaltyYards_playerPlay'}, inplace=True)
    df_plays.rename(columns={'penaltyYards': 'penaltyYards_plays'}, inplace=True)
    display(HTML(styled_heading("Renaming conflicting columns completed!")))

    # Merge the dataframes
    display(HTML(styled_heading("Merging DataFrames...")))
    merged_df = pd.merge(df_player_play, df_plays, on=['gameId', 'playId'], how='inner')
    merged_df = pd.merge(merged_df, df_games, on='gameId', how='inner')
    
    required_columns = ['season', 'week', 'gameDate', 'gameTimeEastern']
    missing_from_games = [col for col in required_columns if col not in df_games.columns]
    if missing_from_games:
        display(HTML(f"<b style='color: red;'>Warning: Missing columns in <code>df_games</code>: {missing_from_games}</b>"))
    else:
        display(HTML("<b style='color: green;'>All required columns are present in <code>df_games</code>.</b>"))

    merged_df = pd.merge(merged_df, df_players, on='nflId', how='inner')
    display(HTML(styled_heading("Data Merging Completed!")))

    return merged_df


# ===Main Execution===
merged_nfl_data = load_and_merge_nfl_data()


# Function to create styled headings
def styled_heading(text, background_color='#ba8d63', text_color='#382411'):
    return f"""
    <div style="
        text-align: center;
        background: {background_color};
        font-family: 'Montserrat', sans-serif;
        color: {text_color};
        padding: 15px;
        font-size: 30px;
        font-weight: bold;
        line-height: 1.2;
        border-radius: 20px 20px 0 0;
        margin: 20px 0;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        border: 3px dashed {text_color};
    ">
        {text}
    </div>
    """

# Function for displaying data overview with source information
def display_overview_with_source(datasets, heading_bg='#ba8d63', heading_color='#382411', text_bg='#ba8d63', text_color='#382411'):
    """
    Displays a comprehensive overview of multiple datasets with labeled sections.
    
    Args:
        datasets (dict): A dictionary where keys are dataset names and values are DataFrames.
        heading_bg (str): Background color for styled headings.
        heading_color (str): Text color for styled headings.
        text_bg (str): Background color for text content.
        text_color (str): Text color for text content.
    """
    
    try:
        for dataset_name, df in datasets.items():
            # Section heading
            display(HTML(styled_heading(f"Overview of {dataset_name} Dataset", background_color=heading_bg, text_color=heading_color)))

            # Display head, tail, and numerical summary
            sections = [
                ("The Head of the Dataset:", df.head(5)),
                ("The Tail of the Dataset:", df.tail(5)),
                ("Numerical Summary of the Data:", df.describe())
            ]
            
            for heading, df_part in sections:
                display(HTML(styled_heading(heading, background_color=heading_bg, text_color=heading_color)))
                display(HTML(df_part.to_html(index=False).replace(
                    '<table border="1" class="dataframe">',
                    f'<table style="border: 8px solid black; margin-bottom: 20px; background-color: {text_bg}; color: {text_color};">'
                ).replace('<td>', f'<td style="color: {text_color}; background-color: {text_bg};">')))
            
            # Print shape data
            display(HTML(styled_heading("Shape of the Dataset:", background_color=heading_bg, text_color=heading_color)))
            shape_details = f"""
            Rows: {df.shape[0]}  
            Columns: {df.shape[1]}
            """
            display(HTML(f"<p style='color: {text_color}; background-color: {text_bg}; padding: 10px; border: 8px solid black;'>{shape_details}</p>"))
            
            # Display dataset info
            display(HTML(styled_heading("Dataset Information:", background_color=heading_bg, text_color=heading_color)))
            buffer = StringIO()
            df.info(buf=buffer)
            buffer.seek(0)
            info_str = buffer.read()
            display(HTML(f"<pre style='color: {text_color}; background-color: {text_bg}; margin-bottom: 20px; font-family: Courier, monospace; font-size: 14px; padding: 10px; border: 8px solid black;'>{info_str}</pre>"))

            # Display null values
            display(HTML(styled_heading("Null Values in the Dataset:", background_color=heading_bg, text_color=heading_color)))
            null_values = df.isnull().sum()
            display(HTML(f"<pre style='color: {text_color}; background-color: {text_bg}; margin-bottom: 20px; font-family: Courier, monospace; font-size: 14px; padding: 10px; border: 8px solid black;'>{null_values.to_string()}</pre>"))

            # Check for duplicates
            display(HTML(styled_heading("Duplicate Records Check:", background_color=heading_bg, text_color=heading_color)))
            duplicates_exist = df.duplicated().any()
            dup_msg = "Duplicates exist in the dataset." if duplicates_exist else "No duplicate records found."
            display(HTML(f"<p style='color: {text_color}; background-color: {text_bg}; padding: 10px; border: 8px solid black;'>{dup_msg}</p>"))

        # Final heading for merged data overview
        display(HTML(styled_heading("Complete Overview of the Merged Dataset", background_color='#4a2a28', text_color='white')))

    except Exception as e:
        display(HTML(f"<div style='color: red; font-weight: bold;'>Error: {str(e)}</div>"))


# ===Main Execution===
df_games=pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/games.csv")
df_player_play=pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/player_play.csv")
df_players=pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/players.csv")
df_plays=pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/plays.csv")

datasets = {
    "Games Data": df_games,
    "Player Play Data": df_player_play,
    "Players Data": df_players,
    "Plays Data": df_plays,
    "Merged Data": merged_nfl_data
}
display_overview_with_source(datasets)


def styled_heading(text, color="#382411", background="#ba8d63", border="#382411"):
    """
    Display a styled heading using HTML.
    
    Parameters:
    text (str): The heading text to display.
    color (str): Text color (default: #4a2a28).
    background (str): Background color (default: #f7eceb).
    border (str): Border color (default: #4a2a28).
    """
    html_code = f"""
    <div style='background-color: {background}; border: 6px solid {border}; 
                padding: 10px; margin-bottom: 10px; border-radius: 5px;'>
        <h3 style='color: {color}; text-align: center; font-family: Arial, sans-serif;'>
            {text}
        </h3>
    </div>
    """
    display(HTML(html_code))

def impute_missing_values(df):
    """
    Impute missing values in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame with missing values.
    
    Returns:
    pd.DataFrame: The DataFrame with imputed values.
    """
    # Iterate over columns
    for column in df.columns:
        if df[column].isnull().sum() > 0:  
            if df[column].dtype == 'object':  
                most_frequent_value = df[column].mode()[0]
                df[column].fillna(most_frequent_value, inplace=True)
            elif df[column].dtype in ['int64', 'float64']:  
                mean_value = df[column].mean()
                df[column].fillna(mean_value, inplace=True)
            elif df[column].dtype == 'bool':  
                mode_value = df[column].mode()[0]
                df[column].fillna(mode_value, inplace=True)
    return df


# ===Main Execution===
styled_heading("Before Imputation: Missing Values Overview")
display(merged_nfl_data.isnull().sum())

# Perform imputation
merged_nfl_data = impute_missing_values(merged_nfl_data)

styled_heading("After Imputation: Missing Values Overview")
display(merged_nfl_data.isnull().sum())


# Function to display styled headings
def styled_heading(text, color="#382411", background="#ba8d63", border="#382411"):
    """
    Display a styled heading using HTML.
    
    Parameters:
    text (str): The heading text to display.
    color (str): Text color (default: #4a2a28).
    background (str): Background color (default: #f7eceb).
    border (str): Border color (default: #4a2a28).
    """
    html_code = f"""
    <div style='background-color: {background}; border: 6px solid {border}; 
                padding: 10px; margin-bottom: 10px; border-radius: 5px;'>
        <h3 style='color: {color}; text-align: center; font-family: Arial, sans-serif;'>
            {text}
        </h3>
    </div>
    """
    display(HTML(html_code))

# Function to calculate descriptive statistics
def calculate_descriptive_statistics(df, numerical_columns):
    styled_heading("Descriptive Statistics for Numerical Columns")
    numerical_summary = df[numerical_columns].describe()
    display(numerical_summary)

# Function to calculate correlation matrix
def calculate_correlation_matrix(df, numerical_columns):
    styled_heading("Correlation Matrix for Numerical Features")
    correlation_matrix = df[numerical_columns].corr()
    display(correlation_matrix)

# Function to calculate group-wise statistics by categorical feature
def group_statistics_by_category(df, group_by_column, aggregation_dict, title):
    styled_heading(f"Group-wise Statistics by {group_by_column}")
    grouped_stats = df.groupby(group_by_column).agg(aggregation_dict).reset_index()
    display(grouped_stats)

# Function to analyze feature interactions (pairwise correlations)
def analyze_feature_interactions(df, feature_pairs, title):
    styled_heading(title)
    for pair in feature_pairs:
        corr = df[list(pair)].corr()
        display(corr)

# Function to calculate player statistics
def calculate_player_statistics(df, group_by_column, aggregation_dict):
    styled_heading("Player-wise Statistics (Aggregated by NFL ID)")
    player_stats = df.groupby(group_by_column).agg(aggregation_dict).reset_index()
    display(player_stats.head(20))  

# Function for time-based analysis
def time_based_analysis(df, date_column, aggregation_dict):
    styled_heading("Time-based Statistics (By Game Date)")
    df[date_column] = pd.to_datetime(df[date_column])
    game_date_stats = df.groupby(df[date_column]).agg(aggregation_dict).reset_index()
    display(game_date_stats)

# Main function to perform the entire analysis
def perform_analysis(df):
    # Columns of interest
    numerical_columns = ['rushingYards', 'passingYards', 'sackYardsAsOffense', 'receivingYards']
    feature_pairs = [
        ('rushingYards', 'passingYards'),
        ('receivingYards', 'sackYardsAsOffense')
    ]
    
    # Group-by definitions
    team_aggregation = {
        'rushingYards': ['mean', 'max'],
        'passingYards': ['mean', 'max'],
        'sackYardsAsOffense': ['mean', 'max'],
        'receivingYards': ['mean', 'max'],
        'hadRushAttempt': 'sum',
        'hadDropback': 'sum',
        'hadPassReception': 'sum',
        'wasTargettedReceiver': 'sum'
    }
    binary_aggregation = {
        'rushingYards': ['mean', 'max'],
        'passingYards': ['mean', 'max'],
        'sackYardsAsOffense': ['mean', 'max'],
        'receivingYards': ['mean', 'max']
    }
    player_aggregation = {
        'rushingYards': ['sum', 'mean', 'max', 'min'],
        'passingYards': ['sum', 'mean', 'max', 'min'],
        'sackYardsAsOffense': ['sum', 'mean', 'max', 'min'],
        'receivingYards': ['sum', 'mean', 'max', 'min'],
        'hadRushAttempt': 'sum',
        'hadDropback': 'sum',
        'hadPassReception': 'sum',
        'wasTargettedReceiver': 'sum'
    }
    date_aggregation = {
        'rushingYards': ['sum', 'mean'],
        'passingYards': ['sum', 'mean'],
        'sackYardsAsOffense': ['sum', 'mean'],
        'receivingYards': ['sum', 'mean']
    }
    
    # Analysis steps
    calculate_descriptive_statistics(df, numerical_columns)
    calculate_correlation_matrix(df, numerical_columns)
    group_statistics_by_category(df, 'teamAbbr', team_aggregation, "Team Abbreviation")
    group_statistics_by_category(df, 'hadRushAttempt', binary_aggregation, "Rush Attempt")
    group_statistics_by_category(df, 'hadDropback', binary_aggregation, "Dropback")
    analyze_feature_interactions(df, feature_pairs, "Feature Interactions (Pairwise Correlations)")
    calculate_player_statistics(df, 'nflId', player_aggregation)
    time_based_analysis(df, 'gameDate', date_aggregation)


# ===Main Execution===
columns_of_interest = ['gameId', 'playId', 'nflId', 'teamAbbr', 'hadRushAttempt', 'rushingYards', 
                       'hadDropback', 'passingYards', 'sackYardsAsOffense', 'hadPassReception', 
                       'receivingYards', 'wasTargettedReceiver', 'gameDate']
df_selected = merged_nfl_data[columns_of_interest]

# Perform the analysis
perform_analysis(df_selected)


# Function to plot statistics by team and game date
def plot_stats_by_team_and_time(df_selected):
    # Selecting relevant columns
    columns_of_interest = ['gameId', 'playId', 'nflId', 'teamAbbr', 'hadRushAttempt', 'rushingYards', 
                           'hadDropback', 'passingYards', 'sackYardsAsOffense', 'hadPassReception', 
                           'receivingYards', 'wasTargettedReceiver', 'gameDate']
    df_selected = df_selected[columns_of_interest]

    # Group-wise analysis for categorical features (e.g., `teamAbbr`, `hadRushAttempt`)
    grouped_by_team = df_selected.groupby('teamAbbr').agg({
        'rushingYards': 'sum',
        'passingYards': 'sum',
        'sackYardsAsOffense': 'sum',
        'receivingYards': 'sum',
        'hadRushAttempt': 'sum',
        'hadDropback': 'sum',
        'hadPassReception': 'sum',
        'wasTargettedReceiver': 'sum'
    }).reset_index()

    grouped_by_rush_attempt = df_selected.groupby('hadRushAttempt').agg({
        'rushingYards': 'sum',
        'passingYards': 'sum',
        'sackYardsAsOffense': 'sum',
        'receivingYards': 'sum'
    }).reset_index()

    grouped_by_dropback = df_selected.groupby('hadDropback').agg({
        'rushingYards': 'sum',
        'passingYards': 'sum',
        'sackYardsAsOffense': 'sum',
        'receivingYards': 'sum'
    }).reset_index()

    # Time-based analysis by 'gameDate'
    df_selected['gameDate'] = pd.to_datetime(df_selected['gameDate'])
    game_date_stats = df_selected.groupby(df_selected['gameDate']).agg({
        'rushingYards': 'sum',
        'passingYards': 'sum',
        'sackYardsAsOffense': 'sum',
        'receivingYards': 'sum'
    }).reset_index()

    # Plotting
    fig, ax = plt.subplots(4, 2, figsize=(32, 25))  

    # 1. Total Rushing Yards by Team
    sns.barplot(x='teamAbbr', y='rushingYards', data=grouped_by_team, ax=ax[0, 0], palette=['#F4A300', '#382411', '#7d410c', '#b37742', '#755130', '#ba8d63', '#241508'])
    ax[0, 0].set_title('Total Rushing Yards by Team', fontsize=20, weight='bold')
    ax[0, 0].tick_params(axis='x', rotation=45, labelsize=14)
    ax[0, 0].tick_params(axis='y', labelsize=14)
    ax[0, 0].set_xlabel('Team Abbreviation', fontsize=16, weight='bold')
    ax[0, 0].set_ylabel('Rushing Yards', fontsize=16, weight='bold')
    for p in ax[0, 0].patches:
        ax[0, 0].annotate(f'{p.get_height():,.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center', fontsize=10, fontweight='bold', color='black', xytext=(0, 5), textcoords='offset points')

    # 2. Total Passing Yards by Team
    sns.barplot(x='teamAbbr', y='passingYards', data=grouped_by_team, ax=ax[0, 1], palette=['#F4A300', '#382411', '#7d410c', '#b37742', '#755130', '#ba8d63', '#241508'])
    ax[0, 1].set_title('Total Passing Yards by Team', fontsize=20, weight='bold')
    ax[0, 1].tick_params(axis='x', rotation=45, labelsize=14)
    ax[0, 1].tick_params(axis='y', labelsize=14)
    ax[0, 1].set_xlabel('Team Abbreviation', fontsize=16, weight='bold')
    ax[0, 1].set_ylabel('Passing Yards', fontsize=16, weight='bold')
    for p in ax[0, 1].patches:
        ax[0, 1].annotate(f'{p.get_height():,.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center', fontsize=10, fontweight='bold', color='black', xytext=(0, 5), textcoords='offset points')

    # 3. Grouped Statistics for Team (Total Rushing Yards)
    sns.barplot(x='teamAbbr', y='rushingYards', data=grouped_by_team, ax=ax[1, 0], palette=['#F4A300', '#382411', '#7d410c', '#b37742', '#755130', '#ba8d63', '#241508'])
    ax[1, 0].set_title('Total Rushing Yards by Team', fontsize=20, weight='bold')
    ax[1, 0].tick_params(axis='x', rotation=45, labelsize=14)
    ax[1, 0].tick_params(axis='y', labelsize=14)
    ax[1, 0].set_xlabel('Team Abbreviation', fontsize=16, weight='bold')
    ax[1, 0].set_ylabel('Rushing Yards', fontsize=16, weight='bold')
    for p in ax[1, 0].patches:
        ax[1, 0].annotate(f'{p.get_height():,.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center', fontsize=10, fontweight='bold', color='black', xytext=(0, 5), textcoords='offset points')

    # 4. Grouped Statistics for Team (Total Passing Yards)
    sns.barplot(x='teamAbbr', y='passingYards', data=grouped_by_team, ax=ax[1, 1], palette=['#F4A300', '#382411', '#7d410c', '#b37742', '#755130', '#ba8d63', '#241508'])
    ax[1, 1].set_title('Total Passing Yards by Team', fontsize=20, weight='bold')
    ax[1, 1].tick_params(axis='x', rotation=45, labelsize=14)
    ax[1, 1].tick_params(axis='y', labelsize=14)
    ax[1, 1].set_xlabel('Team Abbreviation', fontsize=16, weight='bold')
    ax[1, 1].set_ylabel('Passing Yards', fontsize=16, weight='bold')
    for p in ax[1, 1].patches:
        ax[1, 1].annotate(f'{p.get_height():,.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center', fontsize=10, fontweight='bold', color='black', xytext=(0, 5), textcoords='offset points')

    # 5. Grouped Statistics for Rush Attempt (Total Rushing Yards)
    sns.barplot(x='hadRushAttempt', y='rushingYards', data=grouped_by_rush_attempt, ax=ax[2, 0], palette=['#7C4DFF', '#382411'])
    ax[2, 0].set_title('Rushing Yards by Rush Attempt', fontsize=20, weight='bold')
    ax[2, 0].set_xlabel('Had Rush Attempt (0 = No, 1 = Yes)', fontsize=16, weight='bold')
    ax[2, 0].tick_params(axis='y', labelsize=14)
    for p in ax[2, 0].patches:
        ax[2, 0].annotate(f'{p.get_height():,.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center', fontsize=10, fontweight='bold', color='black', xytext=(0, 5), textcoords='offset points')

    # 6. Grouped Statistics for Dropback (Total Passing Yards)
    sns.barplot(x='hadDropback', y='passingYards', data=grouped_by_dropback, ax=ax[2, 1], palette=['#7C4DFF', '#382411'])
    ax[2, 1].set_title('Passing Yards by Dropback', fontsize=20, weight='bold')
    ax[2, 1].set_xlabel('Had Dropback (0 = No, 1 = Yes)', fontsize=16, weight='bold')
    ax[2, 1].tick_params(axis='y', labelsize=14)
    for p in ax[2, 1].patches:
        ax[2, 1].annotate(f'{p.get_height():,.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center', fontsize=10, fontweight='bold', color='black', xytext=(0, 5), textcoords='offset points')

    # 7. Time-based analysis: Total Rushing Yards over Game Date
    sns.lineplot(data=game_date_stats, x='gameDate', y='rushingYards', ax=ax[3, 0], color='#382411', lw=3)
    ax[3, 0].set_title('Rushing Yards over Game Date', fontsize=20, weight='bold')
    ax[3, 0].set_xlabel('Game Date', fontsize=16, weight='bold')
    ax[3, 0].set_ylabel('Rushing Yards', fontsize=16, weight='bold')
    ax[3, 0].tick_params(axis='x', rotation=45, labelsize=14)
    ax[3, 0].tick_params(axis='y', labelsize=14)

    # 8. Time-based analysis: Total Passing Yards over Game Date
    sns.lineplot(data=game_date_stats, x='gameDate', y='passingYards', ax=ax[3, 1], color='#382411', lw=3)
    ax[3, 1].set_title('Passing Yards over Game Date', fontsize=20, weight='bold')
    ax[3, 1].set_xlabel('Game Date', fontsize=16, weight='bold')
    ax[3, 1].set_ylabel('Passing Yards', fontsize=16, weight='bold')
    ax[3, 1].tick_params(axis='x', rotation=45, labelsize=14)
    ax[3, 1].tick_params(axis='y', labelsize=14)

    plt.tight_layout()
    plt.show()


# ===Main Execution===
plot_stats_by_team_and_time(df_selected)


def plot_total_yards_by_play_type(merged_df):
    # Summing up yards for different play types
    yard_data = merged_df[['rushingYards', 'passingYards', 'receivingYards']].sum()
    yard_data = yard_data.reset_index()
    yard_data.columns = ['Play Type', 'Total Yards']

    # Plotting the bar chart
    fig = px.bar(yard_data, 
                 x='Play Type', 
                 y='Total Yards', 
                 title='Total Yards by Play Type', 
                 color='Play Type', 
                 color_discrete_map={'rushingYards': '#382411', 
                                     'passingYards': '#7d410c',  
                                     'receivingYards': '#b37742'},  
                 text='Total Yards',  
                 labels={'Total Yards': 'Total Yards (in yards)', 
                         'Play Type': 'Play Type'},  
                 template='plotly_white')  

    # Update the layout for a better presentation
    fig.update_layout(
        title_font=dict(size=24, family='Arial', color='black'),  
        xaxis_title_font=dict(size=18, family='Arial', color='black'),  
        yaxis_title_font=dict(size=18, family='Arial', color='black'),  
        plot_bgcolor='white',  
        paper_bgcolor='white',  
        xaxis=dict(
            tickfont=dict(size=14, color='black'),  
        ),
        yaxis=dict(
            tickfont=dict(size=14, color='black'),  
        ),
        bargap=0.3  
    )

    # Add annotations on bars
    for i in range(len(yard_data)):
        fig.add_annotation(
            x=yard_data['Play Type'][i],
            y=yard_data['Total Yards'][i],
            text=f"{yard_data['Total Yards'][i]} yards",  
            font=dict(size=14, color='black'),
            showarrow=False,
            yshift=10  
        )

    # Show the plot
    fig.show()

# ===Main Execution===
plot_total_yards_by_play_type(merged_nfl_data)


# Function to generate all the required visualizations
def generate_visualizations(merged_df):
    # The columns you are interested in:
    columns_of_interest = ['yardageGainedAfterTheCatch', 'fumbles', 'fumbleLost', 'fumbleOutOfBounds',
                           'assistedTackle', 'forcedFumbleAsDefense', 'halfSackYardsAsDefense', 'passDefensed',
                           'quarterbackHit', 'sackYardsAsDefense', 'safetyAsDefense', 'soloTackle', 'tackleAssist',
                           'tackleForALoss', 'tackleForALossYardage', 'teamAbbr', 'nflId']

    # Selecting relevant columns for analysis
    df_selected = merged_df[columns_of_interest]

    # Descriptive statistics for numerical columns
    numerical_summary = df_selected.describe()

    # Group-wise analysis for categorical features (e.g., 'teamAbbr' or 'nflId')
    # Group by `teamAbbr` for analyzing offensive and defensive statistics
    grouped_by_team = df_selected.groupby('teamAbbr').agg({
        'yardageGainedAfterTheCatch': ['mean', 'sum'],
        'fumbles': ['mean', 'sum'],
        'fumbleLost': ['mean', 'sum'],
        'fumbleOutOfBounds': ['mean', 'sum'],
        'assistedTackle': ['mean', 'sum'],
        'forcedFumbleAsDefense': ['mean', 'sum'],
        'halfSackYardsAsDefense': ['mean', 'sum'],
        'passDefensed': ['mean', 'sum'],
        'quarterbackHit': ['mean', 'sum'],
        'sackYardsAsDefense': ['mean', 'sum'],
        'safetyAsDefense': ['mean', 'sum'],
        'soloTackle': ['mean', 'sum'],
        'tackleAssist': ['mean', 'sum'],
        'tackleForALoss': ['mean', 'sum'],
        'tackleForALossYardage': ['mean', 'sum']
    }).reset_index()

    # Group-wise analysis for individual players ('nflId')
    player_stats = df_selected.groupby('nflId').agg({
        'yardageGainedAfterTheCatch': ['sum', 'mean', 'max', 'min'],
        'fumbles': ['sum', 'mean', 'max', 'min'],
        'fumbleLost': ['sum', 'mean', 'max', 'min'],
        'fumbleOutOfBounds': ['sum', 'mean', 'max', 'min'],
        'assistedTackle': ['sum', 'mean', 'max', 'min'],
        'forcedFumbleAsDefense': ['sum', 'mean', 'max', 'min'],
        'halfSackYardsAsDefense': ['sum', 'mean', 'max', 'min'],
        'passDefensed': ['sum', 'mean', 'max', 'min'],
        'quarterbackHit': ['sum', 'mean', 'max', 'min'],
        'sackYardsAsDefense': ['sum', 'mean', 'max', 'min'],
        'safetyAsDefense': ['sum', 'mean', 'max', 'min'],
        'soloTackle': ['sum', 'mean', 'max', 'min'],
        'tackleAssist': ['sum', 'mean', 'max', 'min'],
        'tackleForALoss': ['sum', 'mean', 'max', 'min'],
        'tackleForALossYardage': ['sum', 'mean', 'max', 'min']
    }).reset_index()

    # Advanced Aggregate Statistics: Total tackles, forced fumbles, sack yards
    # Creating aggregated features like total tackles and forced fumbles
    df_selected['totalTackles'] = df_selected['soloTackle'] + df_selected['assistedTackle']
    df_selected['totalFumbles'] = df_selected['fumbles'] + df_selected['fumbleLost']
    df_selected['totalSackYards'] = df_selected['sackYardsAsDefense'] + df_selected['halfSackYardsAsDefense']
    df_selected['totalTacklesForLoss'] = df_selected['tackleForALoss'] + df_selected['tackleForALossYardage']

    # Grouping by 'teamAbbr' for the new features
    grouped_by_team_advanced = df_selected.groupby('teamAbbr').agg({
        'totalTackles': ['mean', 'sum'],
        'totalFumbles': ['mean', 'sum'],
        'totalSackYards': ['mean', 'sum'],
        'totalTacklesForLoss': ['mean', 'sum']
    }).reset_index()

    # Function for team performance visualization with custom unique colors
    def plot_team_performance(grouped_data, metric_name, metric_label, colors):
        plt.figure(figsize=(20, 9))
        ax = sns.barplot(x='teamAbbr', y=(metric_name, 'mean'), data=grouped_data, palette=colors)
        plt.title(f"Average {metric_label} by Team")
        plt.xlabel("Team Abbreviation")
        plt.ylabel(metric_label)
        plt.xticks(rotation=45)
        
        # Adding values above bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.4f}', 
                        (p.get_x() + p.get_width() / 4., p.get_height()),
                        xytext=(0, 5),  # 5 points vertical offset
                        textcoords='offset points',
                        ha='center', va='center', fontsize=8, fontweight='bold', color='black')
        
        plt.tight_layout()
        plt.show()

    # Manually define unique and unusual color palette
    unique_colors = ['#F4A300', '#382411', '#7d410c', '#b37742', '#755130', '#ba8d63', '#241508', 
                     '#a87f32', '#524630', '#787061', '#382411', '#7d410c', '#b37742', '#d49a2f']

    # Plot group-by team performance for different metrics with the new color palette
    plot_team_performance(grouped_by_team, 'yardageGainedAfterTheCatch', "Yardage Gained After The Catch", unique_colors)
    plot_team_performance(grouped_by_team, 'fumbles', "Fumbles", unique_colors)
    plot_team_performance(grouped_by_team, 'sackYardsAsDefense', "Sack Yards as Defense", unique_colors)

    # Advanced aggregate statistics for tackles and forced fumbles
    def plot_advanced_team_stats(grouped_data, metric_name, metric_label, colors):
        plt.figure(figsize=(14, 7))
        ax = sns.barplot(x='teamAbbr', y=(metric_name, 'mean'), data=grouped_data, palette=unique_colors)
        plt.title(f"Average {metric_label} by Team")
        plt.xlabel("Team Abbreviation")
        plt.ylabel(metric_label)
        plt.xticks(rotation=45)
        
        # Adding values above bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.4f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        xytext=(0, 5),  # 5 points vertical offset
                        textcoords='offset points',
                        ha='center', va='center', fontsize=8, fontweight='bold', color='black')
        
        plt.tight_layout()
        plt.show()

    # Plot advanced aggregate statistics with the new color palette
    plot_advanced_team_stats(grouped_by_team_advanced, 'totalTackles', "Total Tackles", unique_colors)
    plot_advanced_team_stats(grouped_by_team_advanced, 'totalFumbles', "Total Fumbles", unique_colors)
    plot_advanced_team_stats(grouped_by_team_advanced, 'totalSackYards', "Total Sack Yards", unique_colors)


# ===Main Execution===
generate_visualizations(merged_nfl_data)


palette = ['#F4A300', '#382411', '#7d410c', '#b37742', '#755130', '#ba8d63', '#241508', 
                     '#a87f32', '#524630', '#787061', '#382411', '#7d410c', '#b37742', '#d49a2f']
# Define the function to create a 3D scatter plot
def plot_3d_yards_by_team(df):
    """
    Creates and displays a 3D scatter plot of rushing yards, passing yards, and receiving yards,
    with different colors for each team.
    
    Parameters:
    - df: DataFrame containing the data with columns 'rushingYards', 'passingYards', 'receivingYards', and 'teamAbbr'.
    """
    # Create the 3D scatter plot
    fig7 = px.scatter_3d(df, 
                         x='rushingYards', 
                         y='passingYards', 
                         z='receivingYards',
                         color='teamAbbr',
                         title='3D Scatter Plot of Yards by Team',
                         color_continuous_scale=palette,  
                         labels={'rushingYards': 'Rushing Yards', 
                                 'passingYards': 'Passing Yards', 
                                 'receivingYards': 'Receiving Yards', 
                                 'teamAbbr': 'Team Abbreviation'},
                         opacity=0.8)  

    # Customize the layout for better readability and aesthetics
    fig7.update_layout(
        title='3D Scatter Plot of Yards by Team',
        title_x=0.5,  
        title_font=dict(size=24, color='black', family='Arial'),
        plot_bgcolor='rgba(0, 0, 0, 0)',  
        paper_bgcolor='rgba(240, 240, 240, 1)',  
        font=dict(family='Arial', size=12, color='black'),  
        scene=dict(
            xaxis_title='Rushing Yards',
            yaxis_title='Passing Yards',
            zaxis_title='Receiving Yards',
            xaxis=dict(showgrid=True, gridcolor='lightgrey'),
            yaxis=dict(showgrid=True, gridcolor='lightgrey'),
            zaxis=dict(showgrid=True, gridcolor='lightgrey'),
        ),
        height=800, width=800,  
        showlegend=True,  
    )

    # Show the updated 3D scatter plot
    fig7.show()


# ===Main Execution=== 
plot_3d_yards_by_team(merged_nfl_data)


# Styled Heading Function
def styled_heading(text, color="#382411", background="#ba8d63", border="#382411"):
    """
    Display a styled heading using HTML.
    
    Parameters:
    text (str): The heading text to display.
    color (str): Text color (default: #382411).
    background (str): Background color (default: #ba8d63).
    border (str): Border color (default: #382411).
    """
    html_code = f"""
    <div style='background-color: {background}; border: 6px solid {border}; 
                padding: 10px; margin-bottom: 10px; border-radius: 5px;'>
        <h3 style='color: {color}; text-align: center; font-family: Arial, sans-serif;'>
            {text}
        </h3>
    </div>
    """
    display(HTML(html_code))

# Analysis Function for Defensive Metrics
def analyze_defensive_metrics(df):
    """
    Analyze and display key defensive metrics from the NFL dataset with styled headings.

    Parameters:
    df (pd.DataFrame): The DataFrame containing defensive NFL data.
    """
    # Descriptive Statistics for Numerical Columns
    styled_heading("ğŸ”� DESCRIPTIVE STATISTICS FOR NUMERICAL COLUMNS")
    numerical_summary = df[['interceptionYards', 'fumbleRecoveryYards', 'penaltyYards_playerPlay',
                            'timeToPressureAsPassRusher', 'getOffTimeAsPassRusher']].describe()
    display(HTML(numerical_summary.to_html()))

    # Total and Average for Key Defensive Metrics
    styled_heading("ğŸ“Š TOTAL AND AVERAGE FOR KEY DEFENSIVE METRICS")
    
    turnover_summary = df[['hadInterception', 'fumbleRecoveries']].sum()
    turnover_avg = df[['hadInterception', 'fumbleRecoveries']].mean()

    penalty_summary = df[['penaltyYards_playerPlay']].sum()
    penalty_avg = df[['penaltyYards_playerPlay']].mean()

    pressure_summary = df[['causedPressure']].sum()
    pressure_avg = df[['causedPressure']].mean()

    time_to_pressure_summary = df[['timeToPressureAsPassRusher']].mean()
    get_off_time_summary = df[['getOffTimeAsPassRusher']].mean()

    # Display Results
    styled_heading("ğŸ�ˆ TOTAL DEFENSIVE ACTIONS SUMMARY")
    display(HTML(turnover_summary.to_frame("Total").to_html(header=True)))

    styled_heading("ğŸ“ˆ AVERAGE DEFENSIVE ACTIONS")
    display(HTML(turnover_avg.to_frame("Average").to_html(header=True)))

    styled_heading("ğŸš© TOTAL PENALTY YARDS SUMMARY")
    display(HTML(penalty_summary.to_frame("Total Penalty Yards").to_html(header=True)))

    styled_heading("ğŸ“‰ PENALTY YARD AVERAGE VALUES")
    display(HTML(penalty_avg.to_frame("Average Penalty Yards").to_html(header=True)))

    styled_heading("ğŸ”¥ TOTAL PRESSURE SUMMARY")
    display(HTML(pressure_summary.to_frame("Total Pressure").to_html(header=True)))

    styled_heading("ğŸ“Œ AVERAGE PRESSURE SUMMARY")
    display(HTML(pressure_avg.to_frame("Average Pressure").to_html(header=True)))

    styled_heading("â�± TIME TO PRESSURE AND GET OFF TIME SUMMARY")
    time_to_pressure_df = pd.DataFrame({
        "Metric": ["Time to Pressure", "Get Off Time"],
        "Average (Seconds)": [time_to_pressure_summary.values[0], get_off_time_summary.values[0]]
    })
    display(HTML(time_to_pressure_df.to_html(index=False)))


# ===Main Execution===
analyze_defensive_metrics(merged_nfl_data)


# Styled Heading Function
def styled_heading(text, color="#382411", background="#ba8d63", border="#382411"):
    """
    Display a styled heading using HTML.
    
    Parameters:
    text (str): The heading text to display.
    color (str): Text color (default: #382411).
    background (str): Background color (default: #ba8d63).
    border (str): Border color (default: #382411).
    """
    html_code = f"""
    <div style='background-color: {background}; border: 6px solid {border}; 
                padding: 10px; margin-bottom: 10px; border-radius: 5px;'>
        <h3 style='color: {color}; text-align: center; font-family: Arial, sans-serif;'>
            {text}
        </h3>
    </div>
    """
    display(HTML(html_code))

# Function to analyze pass rusher performance
def analyze_pass_rusher_performance(df):
    """
    Analyze and display the performance summary for initial and non-initial pass rushers.

    Parameters:
    df (pd.DataFrame): The DataFrame containing NFL player data.
    """
    # Initial Pass Rusher Performance Summary
    styled_heading("âš¡ INITIAL PASS RUSHER PERFORMANCE SUMMARY")
    initial_rusher_stats = df[df['wasInitialPassRusher'] == 1][['causedPressure', 
                                                                 'timeToPressureAsPassRusher',
                                                                 'getOffTimeAsPassRusher']].describe()
    display(HTML(initial_rusher_stats.to_html()))

    # Non-Initial Pass Rusher Performance Summary
    styled_heading("âš”ï¸� NON-PASS RUSHER PERFORMANCE SUMMARY")
    non_rusher_stats = df[df['wasInitialPassRusher'] == 0][['causedPressure', 
                                                              'timeToPressureAsPassRusher',
                                                              'getOffTimeAsPassRusher']].describe()
    display(HTML(non_rusher_stats.to_html()))

# Function to calculate total turnovers and yards recovered
def calculate_turnovers_and_yards(df):
    """
    Calculate and display total turnovers and yards recovered.

    Parameters:
    df (pd.DataFrame): The DataFrame containing NFL player data.
    """
    # Create new columns for total turnovers and total yards recovered
    df['totalTurnovers'] = df['hadInterception'] + df['fumbleRecoveries']
    df['totalYardsRecovered'] = df['interceptionYards'] + df['fumbleRecoveryYards']

    # Aggregate totals
    total_turnovers = df['totalTurnovers'].sum()
    total_yards_recovered = df['totalYardsRecovered'].sum()

    # Display Results
    styled_heading("ğŸ�† TOTAL TURNOVERS AND YARDS RECOVERED")
    turnovers_summary = pd.DataFrame({"Total Turnovers": [total_turnovers]})
    yards_summary = pd.DataFrame({"Total Yards Recovered": [total_yards_recovered]})
    
    display(HTML(turnovers_summary.to_html(header=True, index=False)))
    display(HTML(yards_summary.to_html(header=True, index=False)))

# Main function to call both analyses
def main_analysis(df):
    analyze_pass_rusher_performance(df)
    calculate_turnovers_and_yards(df)


# ===Main Execution===
main_analysis(merged_nfl_data)


# Styled Heading Function
def styled_heading(text, color="#382411", background="#ba8d63", border="#382411"):
    """
    Display a styled heading using HTML.
    
    Parameters:
    text (str): The heading text to display.
    color (str): Text color (default: #382411).
    background (str): Background color (default: #ba8d63).
    border (str): Border color (default: #382411).
    """
    html_code = f"""
    <div style='background-color: {background}; border: 6px solid {border}; 
                padding: 10px; margin-bottom: 10px; border-radius: 5px;'>
        <h3 style='color: {color}; text-align: center; font-family: Arial, sans-serif;'>
            {text}
        </h3>
    </div>
    """
    display(HTML(html_code))

# Function for Motion and Route Analysis
def motion_and_route_analysis(df):
    """
    Analyze and display motion and route running data.

    Parameters:
    df (pd.DataFrame): The DataFrame containing NFL player data.
    """
    # Motion and Route Analysis
    motion_at_snap_percentage = df['inMotionAtBallSnap'].mean() * 100
    total_shifts = df['shiftSinceLineset'].sum()
    total_motions = df['motionSinceLineset'].sum()
    distinct_routes_ran = df['routeRan'].nunique()

    # Display Results
    styled_heading("ğŸ�ƒâ€�â™‚ï¸� MOTION AND ROUTE ANALYSIS")
    motion_stats = pd.DataFrame({
        "Metric": ["Motion at Ball Snap (%)", "Total Shifts", "Total Motions", "Distinct Routes Ran"],
        "Value": [f"{motion_at_snap_percentage:.2f}%", total_shifts, total_motions, distinct_routes_ran]
    })
    display(HTML(motion_stats.to_html(index=False, header=True)))

# Function for Blocked Player Analysis
def blocked_player_analysis(df):
    """
    Analyze and display blocked player data.

    Parameters:
    df (pd.DataFrame): The DataFrame containing NFL player data.
    """
    # Blocked Player Analysis
    blocked_players = df[['blockedPlayerNFLId1', 'blockedPlayerNFLId2', 'blockedPlayerNFLId3']].stack()
    blocked_players_count = blocked_players.value_counts()

    # Display Results
    styled_heading("ğŸš§ BLOCKED PLAYER ANALYSIS")
    blocked_stats = blocked_players_count.head(10).to_frame("Blocked Times")
    display(HTML(blocked_stats.to_html(header=True)))

# Function for Pressure Allowed by Blocker
def pressure_allowed_by_blocker(df):
    """
    Analyze and display pressure allowed by blockers.

    Parameters:
    df (pd.DataFrame): The DataFrame containing NFL player data.
    """
    # Pressure Analysis
    average_pressure_allowed = df['pressureAllowedAsBlocker'].mean()
    average_time_to_pressure = df['timeToPressureAllowedAsBlocker'].mean()

    # Display Results
    styled_heading("ğŸ’¥ PRESSURE ALLOWED BY BLOCKER")
    pressure_stats = pd.DataFrame({
        "Metric": ["Average Pressure Allowed", "Average Time to Pressure (seconds)"],
        "Value": [f"{average_pressure_allowed:.2f}", f"{average_time_to_pressure:.2f}"]
    })
    display(HTML(pressure_stats.to_html(index=False, header=True)))

# Function for Defensive Coverage Assignment Analysis
def defensive_coverage_assignment_analysis(df):
    """
    Analyze and display defensive coverage assignment data.

    Parameters:
    df (pd.DataFrame): The DataFrame containing NFL player data.
    """
    # Defensive Coverage Assignment Counts
    defensive_coverage_counts = df['pff_defensiveCoverageAssignment'].value_counts()

    # Display Results
    styled_heading("ğŸ›¡ DEFENSIVE COVERAGE ASSIGNMENT ANALYSIS")
    display(HTML(defensive_coverage_counts.to_frame("Coverage Assignments").to_html(header=True)))

# Function for Primary and Secondary Defensive Matchup Analysis
def defensive_matchup_analysis(df):
    """
    Analyze and display primary and secondary defensive matchup data.

    Parameters:
    df (pd.DataFrame): The DataFrame containing NFL player data.
    """
    # Defensive Matchup Analysis
    primary_matchup_counts = df['pff_primaryDefensiveCoverageMatchupNflId'].value_counts()
    secondary_matchup_counts = df['pff_secondaryDefensiveCoverageMatchupNflId'].value_counts()

    # Display Results
    styled_heading("ğŸ”’ PRIMARY DEFENSIVE MATCHUP ANALYSIS")
    display(HTML(primary_matchup_counts.head(10).to_frame("Top Primary Matchups").to_html(header=True)))
    
    styled_heading("ğŸ”‘ SECONDARY DEFENSIVE MATCHUP ANALYSIS")
    display(HTML(secondary_matchup_counts.head(10).to_frame("Top Secondary Matchups").to_html(header=True)))

# Main function to call all analyses
def main_analysis(df):
    motion_and_route_analysis(df)
    blocked_player_analysis(df)
    pressure_allowed_by_blocker(df)
    defensive_coverage_assignment_analysis(df)
    defensive_matchup_analysis(df)


# ===Main Execution===
main_analysis(merged_nfl_data)


# Styled Heading Function
def styled_heading(text, color="#382411", background="#ba8d63", border="#382411"):
    """
    Display a styled heading using HTML.
    
    Parameters:
    text (str): The heading text to display.
    color (str): Text color (default: #382411).
    background (str): Background color (default: #ba8d63).
    border (str): Border color (default: #382411).
    """
    html_code = f"""
    <div style='background-color: {background}; border: 6px solid {border}; 
                padding: 10px; margin-bottom: 10px; border-radius: 5px;'>
        <h3 style='color: {color}; text-align: center; font-family: Arial, sans-serif;'>
            {text}
        </h3>
    </div>
    """
    display(HTML(html_code))

# Function for Motion vs Success Rate Analysis
def motion_success_rate_analysis(df):
    """
    Analyze the success rate of plays with and without motion at the ball snap.

    Parameters:
    df (pd.DataFrame): The DataFrame containing NFL player data.
    """
    # Success Rate with Motion at the Ball Snap
    success_with_motion = df[df['inMotionAtBallSnap'] == 1].groupby('wasRunningRoute')['routeRan'].count()
    success_without_motion = df[df['inMotionAtBallSnap'] == 0].groupby('wasRunningRoute')['routeRan'].count()

    # Display Results
    styled_heading("âš–ï¸� SUCCESS RATE WITH AND WITHOUT MOTION AT BALL SNAP")

    success_with_motion_df = success_with_motion.to_frame("Success with Motion").reset_index()
    success_without_motion_df = success_without_motion.to_frame("Success without Motion").reset_index()

    # Merge the two results for better comparison
    success_comparison = pd.merge(success_with_motion_df, success_without_motion_df, 
                                  on='wasRunningRoute', how='outer').fillna(0)

    display(HTML(success_comparison.to_html(index=False, header=True)))

# Main function to call the motion success rate analysis
def main_analysis(df):
    motion_success_rate_analysis(df)


# ===Main Execution===
main_analysis(merged_nfl_data)


# Styled Heading Function
def styled_heading(text, color="#382411", background="#ba8d63", border="#382411"):
    """
    Display a styled heading using HTML.
    
    Parameters:
    text (str): The heading text to display.
    color (str): Text color (default: #382411).
    background (str): Background color (default: #ba8d63).
    border (str): Border color (default: #382411).
    """
    html_code = f"""
    <div style='background-color: {background}; border: 6px solid {border}; 
                padding: 10px; margin-bottom: 10px; border-radius: 5px;'>
        <h3 style='color: {color}; text-align: center; font-family: Arial, sans-serif;'>
            {text}
        </h3>
    </div>
    """
    display(HTML(html_code))

# Function to analyze pressure by game quarter
def pressure_by_quarter_analysis(df):
    """
    Analyze and display the average pressure allowed across game quarters.

    Parameters:
    df (pd.DataFrame): The DataFrame containing NFL player data.
    """
    # Ensure 'routeRan' is numeric
    df['routeRan'] = pd.to_numeric(df['routeRan'], errors='coerce')

    # Group by 'gameTimeEastern' and calculate the mean for 'timeToPressureAllowedAsBlocker'
    pressure_by_quarter = df.groupby('gameTimeEastern')['timeToPressureAllowedAsBlocker'].mean().reset_index()

    # Display Results
    styled_heading("â�± AVERAGE PRESSURE ALLOWED ACROSS GAME QUARTERS")
    display(HTML(pressure_by_quarter.to_html(index=False, header=True)))

# Function to analyze defensive coverage distribution by game quarter
def coverage_by_quarter_analysis(df):
    """
    Analyze and display the defensive coverage assignment distribution by game quarter.

    Parameters:
    df (pd.DataFrame): The DataFrame containing NFL player data.
    """
    # Coverage assignment distribution by quarter
    coverage_by_quarter = df.groupby('gameTimeEastern')['pff_defensiveCoverageAssignment'].value_counts().unstack().fillna(0)

    # Display Results
    styled_heading("ğŸ›¡ DEFENSIVE COVERAGE TRENDS BY GAME QUARTER")
    display(HTML(coverage_by_quarter.to_html(header=True)))

# Main function to call all analyses
def main_analysis(df):
    pressure_by_quarter_analysis(df)
    coverage_by_quarter_analysis(df)


# ===Main Execution===
main_analysis(merged_nfl_data)


def plot_defensive_analyses(merged_df):
    # Increase figure size for better clarity
    fig, axes = plt.subplots(4, 2, figsize=(24, 32))
    
    # Adjust spacing between subplots
    fig.tight_layout(pad=8.0)
    
    # Define unique colors for each plot
    palette = ['#F4A300', '#382411', '#7d410c', '#b37742', '#755130', '#ba8d63', '#241508', 
                     '#a87f32', '#524630', '#787061', '#382411', '#7d410c', '#b37742', '#d49a2f']

    colors = sns.color_palette(palette)  # Generates 10 unique colors
    
    # 1. Total Turnovers
    turnover_labels = ['Total Interceptions', 'Total Fumble Recoveries', 'Total Turnovers']
    turnover_values = [
        merged_df['hadInterception'].sum(),
        merged_df['fumbleRecoveries'].sum(),
        merged_df['totalTurnovers'].sum()
    ]
    axes[0, 0].bar(turnover_labels, turnover_values, color=colors[:3], edgecolor='black')
    axes[0, 0].set_title('Total Turnovers', fontsize=20, fontweight='bold')
    axes[0, 0].set_ylabel('Count', fontsize=15)
    axes[0, 0].tick_params(axis='x', labelsize=12, rotation=45)
    axes[0, 0].tick_params(axis='y', labelsize=12)
    for i, value in enumerate(turnover_values):
        axes[0, 0].text(i, value, f'{value:.0f}', ha='center', va='bottom', fontsize=12)

    # 2. Average Turnovers
    avg_turnover_labels = ['Avg Interceptions', 'Avg Fumble Recoveries', 'Avg Turnovers']
    avg_turnover_values = [
        merged_df['hadInterception'].mean(),
        merged_df['fumbleRecoveries'].mean(),
        merged_df['totalTurnovers'].mean()
    ]
    axes[0, 1].bar(avg_turnover_labels, avg_turnover_values, color=colors[3:6], edgecolor='black')
    axes[0, 1].set_title('Average Turnovers', fontsize=20, fontweight='bold')
    axes[0, 1].set_ylabel('Average', fontsize=15)
    axes[0, 1].tick_params(axis='x', labelsize=12, rotation=45)
    axes[0, 1].tick_params(axis='y', labelsize=12)
    for i, value in enumerate(avg_turnover_values):
        axes[0, 1].text(i, value, f'{value:.2f}', ha='center', va='bottom', fontsize=12)

    # 3. Penalty Yards
    penalty_labels = ['Total Penalty Yards', 'Avg Penalty Yards']
    penalty_values = [
        merged_df['penaltyYards_playerPlay'].sum(),
        merged_df['penaltyYards_playerPlay'].mean()
    ]
    axes[1, 0].bar(penalty_labels, penalty_values, color=colors[6:8], edgecolor='black')
    axes[1, 0].set_title('Penalty Yards', fontsize=20, fontweight='bold')
    axes[1, 0].set_ylabel('Yards', fontsize=15)
    axes[1, 0].tick_params(axis='x', labelsize=12, rotation=45)
    axes[1, 0].tick_params(axis='y', labelsize=12)
    for i, value in enumerate(penalty_values):
        axes[1, 0].text(i, value, f'{value:.2f}', ha='center', va='bottom', fontsize=12)

    # 4. Caused Pressure
    pressure_labels = ['Total Caused Pressure', 'Avg Caused Pressure']
    pressure_values = [
        merged_df['causedPressure'].sum(),
        merged_df['causedPressure'].mean()
    ]
    axes[1, 1].bar(pressure_labels, pressure_values, color=colors[8:10], edgecolor='black')
    axes[1, 1].set_title('Caused Pressure', fontsize=20, fontweight='bold')
    axes[1, 1].set_ylabel('Count', fontsize=15)
    axes[1, 1].tick_params(axis='x', labelsize=12, rotation=45)
    axes[1, 1].tick_params(axis='y', labelsize=12)
    for i, value in enumerate(pressure_values):
        axes[1, 1].text(i, value, f'{value:.2f}', ha='center', va='bottom', fontsize=12)

    # 5. Time to Pressure & Get Off Time
    time_labels = ['Avg Time to Pressure', 'Avg Get Off Time']
    time_values = [
        merged_df['timeToPressureAsPassRusher'].mean(),
        merged_df['getOffTimeAsPassRusher'].mean()
    ]
    axes[2, 0].bar(time_labels, time_values, color=colors[:2], edgecolor='black')
    axes[2, 0].set_title('Time Metrics', fontsize=20, fontweight='bold')
    axes[2, 0].set_ylabel('Seconds', fontsize=15)
    axes[2, 0].tick_params(axis='x', labelsize=12, rotation=45)
    axes[2, 0].tick_params(axis='y', labelsize=12)
    for i, value in enumerate(time_values):
        axes[2, 0].text(i, value, f'{value:.2f}', ha='center', va='bottom', fontsize=12)

    # 6. Top 10 Most Blocked Players
    blocked_players = merged_df[['blockedPlayerNFLId1', 'blockedPlayerNFLId2', 'blockedPlayerNFLId3']].stack()
    blocked_count = blocked_players.value_counts().head(10)
    axes[2, 1].bar(blocked_count.index.astype(str), blocked_count.values, color=colors[2:4], edgecolor='black')
    axes[2, 1].set_title('Top 10 Most Blocked Players', fontsize=20, fontweight='bold')
    axes[2, 1].set_ylabel('Block Count', fontsize=15)
    axes[2, 1].tick_params(axis='x', labelsize=12, rotation=45)
    axes[2, 1].tick_params(axis='y', labelsize=12)
    for i, value in enumerate(blocked_count.values):
        axes[2, 1].text(i, value, f'{value:.0f}', ha='center', va='bottom', fontsize=12)

    # 7. Top Defensive Matchups
    matchup_count = merged_df['pff_primaryDefensiveCoverageMatchupNflId'].value_counts().head(10)
    axes[3, 0].bar(matchup_count.index.astype(str), matchup_count.values, color=colors[4:6], edgecolor='black')
    axes[3, 0].set_title('Top Defensive Matchups', fontsize=20, fontweight='bold')
    axes[3, 0].set_ylabel('Matchup Count', fontsize=15)
    axes[3, 0].tick_params(axis='x', labelsize=12, rotation=45)
    axes[3, 0].tick_params(axis='y', labelsize=12)
    for i, value in enumerate(matchup_count.values):
        axes[3, 0].text(i, value, f'{value:.0f}', ha='center', va='bottom', fontsize=12)

    # Hide the last subplot (4th in last row)
    axes[3, 1].axis('off')

    # Show the plots
    plt.show()



# ===Main Execution===
plot_defensive_analyses(merged_nfl_data)


# Styled Heading Function
def styled_heading(text, color="#382411", background="#ba8d63", border="#382411"):
    """
    Display a styled heading using HTML.
    
    Parameters:
    text (str): The heading text to display.
    color (str): Text color (default: #382411).
    background (str): Background color (default: #ba8d63).
    border (str): Border color (default: #382411).
    """
    html_code = f"""
    <div style='background-color: {background}; border: 6px solid {border}; 
                padding: 10px; margin-bottom: 10px; border-radius: 5px;'>
        <h3 style='color: {color}; text-align: center; font-family: Arial, sans-serif;'>
            {text}
        </h3>
    </div>
    """
    display(HTML(html_code))

# Function for Play Type Analysis
def play_type_analysis(df):
    """
    Categorize play types and analyze average yards for each play type.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing NFL play data.
    """
    # Categorizing play types based on description
    df['playType'] = df['playDescription'].apply(lambda x: 'pass' if 'pass' in x.lower() else ('run' if 'run' in x.lower() else 'kick'))

    # Analyze success rate and average yards for each play type
    play_type_analysis = df.groupby('playType').agg(
        avg_yards=('yardlineNumber', 'mean')
    ).reset_index()

    # Display Results
    styled_heading("ğŸ“Š PLAY TYPE ANALYSIS BASED ON AVERAGE YARDS")
    display(HTML(play_type_analysis.to_html(index=False, header=True)))

# Function for Quarter and Down Analysis
def quarter_down_analysis(df):
    """
    Analyze success rate and average yards by quarter and down.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing NFL play data.
    """
    # Group by quarter and down, and assess the pass success rate and average yards
    quarter_down_analysis = df.groupby(['quarter', 'down']).agg(
        avg_yards_gained=('yardlineNumber', 'mean')
    ).reset_index()

    # Display Results
    styled_heading("ğŸ“‰ PASS SUCCESS RATE AND AVERAGE YARDS BY QUARTER AND DOWN")
    display(HTML(quarter_down_analysis.to_html(index=False, header=True)))

# Function for Score Difference Analysis
def score_difference_analysis(df):
    """
    Analyze how score difference affects pass success.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing NFL play data.
    """
    # Calculate score difference and analyze how it affects pass success
    df['scoreDifference'] = df['preSnapHomeScore'] - df['preSnapVisitorScore']
    score_diff_analysis = df.groupby('scoreDifference').agg(
        avg_pass_length=('passLength', 'mean'),
        avg_yards=('yardlineNumber', 'mean')
    ).reset_index()

    # Display Results
    styled_heading("ğŸ�† SCORE DIFFERENCE AND PASS SUCCESS ANALYSIS")
    display(HTML(score_diff_analysis.to_html(index=False, header=True)))

# Function for Formation Analysis
def formation_analysis(df):
    """
    Analyze performance based on offensive formation.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing NFL play data.
    """
    # Analyze the performance by offensive formation
    formation_analysis = df.groupby('offenseFormation').agg(
        avg_yards_gained=('yardlineNumber', 'mean')
    ).reset_index()

    # Display Results
    styled_heading("ğŸ“Š OFFENSIVE FORMATION AND YARDLINE NUMBER ANALYSIS")
    display(HTML(formation_analysis.to_html(index=False, header=True)))

# Main function to call all analyses
def main_analysis(df):
    play_type_analysis(df)
    quarter_down_analysis(df)
    score_difference_analysis(df)
    formation_analysis(df)


# ===Main Execution===
main_analysis(merged_nfl_data)


# Styled Heading Function
def styled_heading(text, color="#382411", background="#ba8d63", border="#382411"):
    """
    Display a styled heading using HTML.
    
    Parameters:
    text (str): The heading text to display.
    color (str): Text color (default: #382411).
    background (str): Background color (default: #ba8d63).
    border (str): Border color (default: #382411).
    """
    html_code = f"""
    <div style='background-color: {background}; border: 6px solid {border}; 
                padding: 10px; margin-bottom: 10px; border-radius: 5px;'>
        <h3 style='color: {color}; text-align: center; font-family: Arial, sans-serif;'>
            {text}
        </h3>
    </div>
    """
    display(HTML(html_code))

# Function for Game Clock Analysis
def game_clock_analysis(df):
    """
    Analyze game clock seconds, average yards, and play clock at snap.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing NFL play data.
    """
    # Game Clock and Play Execution
    df['gameClockSeconds'] = df['gameClock'].apply(
        lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]) if isinstance(x, str) else 0
    )
    game_clock_analysis = df.groupby('gameClockSeconds').agg(
        avg_yards=('yardlineNumber', 'mean'),
        total_plays=('playDescription', 'count'),
        avg_play_clock_at_snap=('playClockAtSnap', 'mean')
    ).reset_index()

    # Display Results
    styled_heading("â�± GAME CLOCK AND PLAY EXECUTION ANALYSIS")
    display(HTML(game_clock_analysis.head(10).to_html(index=False, header=True)))

# Function for Score & Win Probability Impact
def score_win_probability_analysis(df):
    """
    Analyze the impact of score difference and win probability on plays.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing NFL play data.
    """
    df['scoreDifference'] = df['preSnapHomeScore'] - df['preSnapVisitorScore']
    score_win_probability_analysis = df.groupby(['scoreDifference', 'preSnapHomeTeamWinProbability', 'preSnapVisitorTeamWinProbability']).agg(
        avg_expected_points=('expectedPoints', 'mean'),
        avg_yards_to_go=('yardsToGo', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index()

    # Display Results
    styled_heading("ğŸ�ˆ SCORE AND WIN PROBABILITY IMPACT ANALYSIS")
    display(HTML(score_win_probability_analysis.head(10).to_html(index=False, header=True)))

# Function for Nullified Play Analysis
def nullified_play_analysis(df):
    """
    Analyze the nullified plays based on penalties.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing NFL play data.
    """
    nullified_play_analysis = df.groupby('playNullifiedByPenalty').agg(
        avg_absolute_yardline_number=('absoluteYardlineNumber', 'mean'),
        total_nullified_plays=('playDescription', 'count')
    ).reset_index()

    # Display Results
    styled_heading("ğŸš« NULLIFIED PLAY ANALYSIS")
    display(HTML(nullified_play_analysis.to_html(index=False, header=True)))

# Function for Offensive Formation Analysis
def offensive_formation_analysis(df):
    """
    Analyze the impact of offensive formations on yardage and play success.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing NFL play data.
    """
    formation_analysis = df.groupby('offenseFormation').agg(
        avg_yards_gained=('yardlineNumber', 'mean'),
        total_plays=('playDescription', 'count'),
        avg_pass_length=('passLength', 'mean')
    ).reset_index()

    # Display Results
    styled_heading("ğŸ“Š OFFENSIVE FORMATION IMPACT ANALYSIS")
    display(HTML(formation_analysis.head(10).to_html(index=False, header=True)))

# Function for Receiver Alignment Impact
def receiver_alignment_analysis(df):
    """
    Analyze the impact of receiver alignment on pass length and play success.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing NFL play data.
    """
    receiver_alignment_analysis = df.groupby('receiverAlignment').agg(
        avg_pass_length=('passLength', 'mean'),
        avg_yards_to_go=('yardsToGo', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index()

    # Display Results
    styled_heading("ğŸ§‘â€�ğŸ�« RECEIVER ALIGNMENT IMPACT ANALYSIS")
    display(HTML(receiver_alignment_analysis.to_html(index=False, header=True)))

# Function for Play Action Influence
def play_action_analysis(df):
    """
    Analyze the impact of play action on passing success and total plays.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing NFL play data.
    """
    play_action_analysis = df.groupby('playAction').agg(
        avg_pass_length=('passLength', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index()

    # Display Results
    styled_heading("ğŸ�¬ PLAY ACTION IMPACT ANALYSIS")
    display(HTML(play_action_analysis.to_html(index=False, header=True)))

# Function for Target Location Analysis
def target_location_analysis(df):
    """
    Analyze the passing success based on target locations.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing NFL play data.
    """
    df['targetLocation'] = df.apply(lambda row: (row['targetX'], row['targetY']), axis=1)
    target_location_analysis = df.groupby('targetLocation').agg(
        avg_pass_length=('passLength', 'mean')
    ).reset_index()

    # Display Results
    styled_heading("ğŸ“� TARGET LOCATION IMPACT ANALYSIS")
    display(HTML(target_location_analysis.head(10).to_html(index=False, header=True)))

# Function for Play Clock at Snap Analysis
def play_clock_snap_analysis(df):
    """
    Analyze the impact of play clock at snap on play execution and yardage.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing NFL play data.
    """
    play_clock_snap_analysis = df.groupby('playClockAtSnap').agg(
        avg_yards_to_go=('yardsToGo', 'mean'),
        total_plays=('playDescription', 'count'),
        avg_yards_gained=('yardlineNumber', 'mean')
    ).reset_index()

    # Display Results
    styled_heading("â�° PLAY CLOCK AT SNAP IMPACT ANALYSIS")
    display(HTML(play_clock_snap_analysis.to_html(index=False, header=True)))

# Function for Pass Result Analysis
def pass_result_analysis(df):
    """
    Analyze pass result statistics based on play outcomes.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing NFL play data.
    """
    pass_result_analysis = df.groupby('passResult').agg(
        avg_pass_length=('passLength', 'mean'),
        avg_yards_to_go=('yardsToGo', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index()

    # Display Results
    styled_heading("ğŸ�¯ PASS RESULT ANALYSIS")
    display(HTML(pass_result_analysis.to_html(index=False, header=True)))

# Main function to call all analyses
def main_analysis(df):
    game_clock_analysis(df)
    score_win_probability_analysis(df)
    nullified_play_analysis(df)
    offensive_formation_analysis(df)
    receiver_alignment_analysis(df)
    play_action_analysis(df)
    target_location_analysis(df)
    play_clock_snap_analysis(df)
    pass_result_analysis(df)


# ===Main Execution===
main_analysis(merged_nfl_data)


# Styled Heading Function
def styled_heading(text, color="#382411", background="#ba8d63", border="#382411"):
    """
    Display a styled heading using HTML.
    
    Parameters:
    text (str): The heading text to display.
    color (str): Text color (default: #382411).
    background (str): Background color (default: #ba8d63).
    border (str): Border color (default: #382411).
    """
    html_code = f"""
    <div style='background-color: {background}; border: 6px solid {border}; 
                padding: 10px; margin-bottom: 10px; border-radius: 5px;'>
        <h3 style='color: {color}; text-align: center; font-family: Arial, sans-serif;'>
            {text}
        </h3>
    </div>
    """
    display(HTML(html_code))

# Function for Dropback Type vs Time to Throw and Time to Sack Analysis
def dropback_time_analysis(df):
    dropback_time_analysis = df.groupby('dropbackType').agg(
        avg_time_to_throw=('timeToThrow', 'mean'),
        avg_time_to_sack=('timeToSack', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index()

    styled_heading("ğŸ”„ DROPBACK TYPE ANALYSIS (TIME TO THROW & SACK)")
    display(HTML(dropback_time_analysis.to_html(index=False, header=True)))

# Function for Dropback Distance vs Yards Gained and Time to Throw Analysis
def dropback_distance_analysis(df):
    dropback_distance_analysis = df.groupby('dropbackDistance').agg(
        avg_yards_gained=('yardsGained', 'mean'),
        avg_time_to_throw=('timeToThrow', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index().head(10)

    styled_heading("ğŸ“� DROPBACK DISTANCE ANALYSIS (YARDS GAINED & TIME TO THROW)")
    display(HTML(dropback_distance_analysis.to_html(index=False, header=True)))

# Function for Pass Location Type vs Yards Gained and Completion Rate Analysis
def pass_location_analysis(df):
    pass_location_analysis = df.groupby('passLocationType').agg(
        avg_yards_gained=('yardsGained', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index().head(10)

    styled_heading("ğŸ“� PASS LOCATION TYPE ANALYSIS (YARDS GAINED & COMPLETION RATE)")
    display(HTML(pass_location_analysis.to_html(index=False, header=True)))

# Function for Time in Tackle Box vs Yards Gained Analysis
def tackle_box_analysis(df):
    tackle_box_analysis = df.groupby('timeInTackleBox').agg(
        avg_yards_gained=('yardsGained', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index().head(10)

    styled_heading("â�± TIME IN TACKLE BOX ANALYSIS (YARDS GAINED)")
    display(HTML(tackle_box_analysis.to_html(index=False, header=True)))

# Function for Time to Sack vs Yards Lost and Unblocked Pressure Analysis
def time_to_sack_analysis(df):
    time_to_sack_analysis = df.groupby('timeToSack').agg(
        avg_yards_lost=('yardsGained', lambda x: (x < 0).mean()),
        unblocked_pressure_avg=('unblockedPressure', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index().head(10)

    styled_heading("â�³ TIME TO SACK ANALYSIS (YARDS LOST & UNBLOCKED PRESSURE)")
    display(HTML(time_to_sack_analysis.to_html(index=False, header=True)))

# Function for Pass Tipped at Line vs Pass Length Analysis
def pass_tipped_analysis(df):
    pass_tipped_analysis = df.groupby('passTippedAtLine').agg(
        avg_pass_length=('passLength', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index()

    styled_heading("ğŸ“� PASS TIPPED AT LINE ANALYSIS (PASS LENGTH)")
    display(HTML(pass_tipped_analysis.to_html(index=False, header=True)))

# Main function to call all analyses
def main_analysis(df):
    dropback_time_analysis(df)
    dropback_distance_analysis(df)
    pass_location_analysis(df)
    tackle_box_analysis(df)
    time_to_sack_analysis(df)
    pass_tipped_analysis(df)


# ===Main Execution===
main_analysis(merged_nfl_data)


# Styled Heading Function
def styled_heading(text, color="#382411", background="#ba8d63", border="#382411"):
    """
    Display a styled heading using HTML.
    
    Parameters:
    text (str): The heading text to display.
    color (str): Text color (default: #382411).
    background (str): Background color (default: #ba8d63).
    border (str): Border color (default: #382411).
    """
    html_code = f"""
    <div style='background-color: {background}; border: 6px solid {border}; 
                padding: 10px; margin-bottom: 10px; border-radius: 5px;'>
        <h3 style='color: {color}; text-align: center; font-family: Arial, sans-serif;'>
            {text}
        </h3>
    </div>
    """
    display(HTML(html_code))

# Function for QB Spike, Kneel, and Sneak Analysis by Yards Gained and Situation
def qb_actions_analysis(df):
    qb_actions_analysis = df.groupby(['qbSpike', 'qbKneel', 'qbSneak']).agg(
        avg_yards_gained=('yardsGained', 'mean'),
        total_plays=('playDescription', 'count'),
        avg_score_difference=('scoreDifference', 'mean')
    ).reset_index()

    styled_heading("ğŸ§‘â€�âš–ï¸� QB SPIKE, KNEEL, AND SNEAK ANALYSIS")
    display(HTML(qb_actions_analysis.to_html(index=False, header=True)))

# Function for Rush Location Type vs Yards Gained and Time to Throw
def rush_location_analysis(df):
    rush_location_analysis = df.groupby('rushLocationType').agg(
        avg_yards_gained=('yardsGained', 'mean'),
        avg_time_to_throw=('timeToThrow', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index()

    styled_heading("ğŸ�ƒ RUSH LOCATION TYPE ANALYSIS (YARDS GAINED & TIME TO THROW)")
    display(HTML(rush_location_analysis.to_html(index=False, header=True)))

# Function for Penalty Yards vs Yards Gained and Pre-Penalty Yards
def penalty_analysis(df):
    penalty_analysis = df.groupby('penaltyYards_plays').agg(
        avg_yards_gained=('yardsGained', 'mean'),
        avg_pre_penalty_yards=('prePenaltyYardsGained', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index()

    styled_heading("ğŸš« PENALTY YARDS ANALYSIS (YARDS GAINED & PRE-PENALTY YARDS)")
    display(HTML(penalty_analysis.to_html(index=False, header=True)))

# Function for Detailed Yards Gained Analysis and Success Rate
def yards_gained_success_analysis(df):
    yards_gained_success_analysis = df.groupby('yardsGained').agg(
        success_rate=('passResult', lambda x: (x == 'Complete').mean()),
        total_plays=('playDescription', 'count')
    ).reset_index()

    styled_heading("ğŸ“Š DETAILED YARDS GAINED AND PLAY SUCCESS ANALYSIS")
    display(HTML(yards_gained_success_analysis.to_html(index=False, header=True)))

# Function for Time to Throw Grouping and Play Type Analysis
def time_to_throw_group_analysis(df):
    time_to_throw_groups = df.copy()
    time_to_throw_groups['timeToThrowGroup'] = pd.cut(df['timeToThrow'], bins=[0, 2, 4, 10], labels=['<2s', '2-4s', '>4s'])

    time_to_throw_group_analysis = time_to_throw_groups.groupby('timeToThrowGroup').agg(
        avg_yards_gained=('yardsGained', 'mean'),
        total_plays=('playDescription', 'count'),
        play_type_distribution=('dropbackType', lambda x: x.value_counts().to_dict())
    ).reset_index()

    styled_heading("â�± TIME TO THROW GROUP ANALYSIS")
    display(HTML(time_to_throw_group_analysis.to_html(index=False, header=True)))

# Main function to call all analyses
def main_analysis(df):
    qb_actions_analysis(df)
    rush_location_analysis(df)
    penalty_analysis(df)
    yards_gained_success_analysis(df)
    time_to_throw_group_analysis(df)


# ===Main Execution===
main_analysis(merged_nfl_data)


def create_subplot(ax, data, xlabel, ylabel, title, plot_type='bar', palette='viridis'):
    """
    Create subplots for different analyses.

    Parameters:
    - ax: Axis object to plot on.
    - data: DataFrame containing the data to plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - title: Title for the plot.
    - plot_type: Type of plot ('bar', 'line', 'scatter').
    - palette: Color palette for the plot.
    """
    if xlabel not in data.columns or ylabel not in data.columns:
        print(f"Error: Columns '{xlabel}' or '{ylabel}' do not exist in the data.")
        return

    ax.set_title(title, fontsize=14, fontweight='bold')

    if plot_type == 'bar':
        bar_plot = sns.barplot(x=xlabel, y=ylabel, palette=palette, data=data, ax=ax)
        for p in bar_plot.patches:
            ax.text(
                p.get_x() + p.get_width() / 2., p.get_height(), f'{p.get_height():.2f}',
                ha='center', va='center', fontsize=10, color='black', fontweight='bold',
                verticalalignment='bottom'
            )
    elif plot_type == 'line':
        sns.lineplot(x=xlabel, y=ylabel, data=data, ax=ax, color=palette[0])
    elif plot_type == 'scatter':
        sns.scatterplot(x=xlabel, y=ylabel, data=data, ax=ax, color=palette[0])

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

def plot_analyses():
    # Custom color palettes
    rainbow_palette = ['#F4A300', '#382411', '#7d410c']
    cool_blues_palette = ['#b37742', '#755130', '#ba8d63']
    pastel_palette = ['#241508', '#a87f32', '#524630']
    vibrant_mix_palette = ['#787061', '#382411', '#7d410c']
    sunset_palette = ['#b37742', '#d49a2f']

    # Example datasets (replace these with your actual data)
    play_type_analysis = pd.DataFrame({
        'playType': ['Run', 'Pass', 'Kick'],
        'avg_yards': [4.2, 7.8, 1.5]
    })
    quarter_down_analysis = pd.DataFrame({
        'quarter': [1, 2, 3, 4],
        'avg_yards_gained': [6.4, 5.7, 4.8, 5.1]
    })
    score_diff_analysis = pd.DataFrame({
        'scoreDifference': [-10, -5, 0, 5, 10],
        'avg_pass_length': [7.1, 6.5, 5.9, 6.8, 7.3]
    })
    formation_analysis = pd.DataFrame({
        'offenseFormation': ['I-Form', 'Shotgun', 'Pistol'],
        'avg_yards_gained': [5.1, 6.8, 4.9]
    })
    game_clock_analysis = pd.DataFrame({
        'gameClockSeconds': [900, 600, 300, 100],
        'avg_yards': [5.5, 6.0, 4.2, 5.7]
    })
    play_action_analysis = pd.DataFrame({
        'playAction': ['Yes', 'No'],
        'avg_pass_length': [7.4, 6.1]
    })
    pass_result_analysis = pd.DataFrame({
        'passResult': ['Complete', 'Incomplete', 'Interception'],
        'avg_pass_length': [8.2, 5.7, 3.5]
    })
    time_to_throw_group_analysis = pd.DataFrame({
        'timeToThrowGroup': ['<2.5s', '2.5-3s', '>3s'],
        'avg_yards_gained': [6.7, 7.2, 5.8]
    })

    # Create a 4x2 grid of subplots
    fig, axs = plt.subplots(4, 2, figsize=(18, 24))

    # Example plots for the 4x2 grid with custom palettes
    create_subplot(axs[0, 0], play_type_analysis, 'playType', 'avg_yards', 'Play Type Analysis - Average Yards', plot_type='bar', palette=rainbow_palette)
    create_subplot(axs[0, 1], quarter_down_analysis, 'quarter', 'avg_yards_gained', 'Quarter vs Down - Average Yards Gained', plot_type='line', palette=cool_blues_palette)
    create_subplot(axs[1, 0], score_diff_analysis, 'scoreDifference', 'avg_pass_length', 'Score Difference vs Average Pass Length', plot_type='scatter', palette=pastel_palette)
    create_subplot(axs[1, 1], formation_analysis, 'offenseFormation', 'avg_yards_gained', 'Offensive Formation - Average Yards Gained', plot_type='bar', palette=vibrant_mix_palette)
    create_subplot(axs[2, 0], game_clock_analysis, 'gameClockSeconds', 'avg_yards', 'Game Clock - Average Yards Gained', plot_type='line', palette=sunset_palette)
    create_subplot(axs[2, 1], play_action_analysis, 'playAction', 'avg_pass_length', 'Play Action - Average Pass Length', plot_type='bar', palette=rainbow_palette)
    create_subplot(axs[3, 0], pass_result_analysis, 'passResult', 'avg_pass_length', 'Pass Result - Average Pass Length', plot_type='bar', palette=cool_blues_palette)
    create_subplot(axs[3, 1], time_to_throw_group_analysis, 'timeToThrowGroup', 'avg_yards_gained', 'Time to Throw Group - Average Yards Gained', plot_type='line', palette=pastel_palette)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


# ===Main Execution===
plot_analyses()


def display_attractive_sunburst_plot(merged_df):
    # Create Sunburst plot with customizations
    fig = px.sunburst(
         merged_df, 
        path=['teamAbbr', 'playAction'], 
        values='yardsGained',
        title='Yards Gained by Play Type and Team'
    )

    # Update hover info to display additional columns for interactivity
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>" +
                      "Team: %{parent}<br>" +
                      "Play Action: %{customdata[0]}<br>" +
                      "Yards Gained: %{value}<br>" +
                      "<extra></extra>",
        customdata=merged_df[['teamAbbr', 'playAction']].values
    )
    
    # Customize colors for a more vibrant and unique look
    fig.update_traces(
        marker=dict(
            colorscale=palette,  
            line=dict(color='#000000', width=2)  
        )
    )

    # Update layout for a more attractive appearance
    fig.update_layout(
        title={
            'text': 'Yards Gained by Play Type and Team',
            'x': 0.5,  
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 22, 'color': 'black'}
        },
        paper_bgcolor="white",  
        plot_bgcolor="white",  
        font={'color': 'black'},  
        margin={"t": 50, "b": 50, "l": 50, "r": 50},  
        height=600,  
        template="plotly_white"  
    )
    
    # Show the plot
    fig.show()


# ===Main Execution===
display_attractive_sunburst_plot(merged_nfl_data)


# Styled Heading Function
def styled_heading(text, color="#382411", background="#ba8d63", border="#382411"):
    """
    Display a styled heading using HTML.
    
    Parameters:
    text (str): The heading text to display.
    color (str): Text color (default: #382411).
    background (str): Background color (default: #ba8d63).
    border (str): Border color (default: #382411).
    """
    html_code = f"""
    <div style='background-color: {background}; border: 6px solid {border}; 
                padding: 10px; margin-bottom: 10px; border-radius: 5px;'>
        <h3 style='color: {color}; text-align: center; font-family: Arial, sans-serif;'>
            {text}
        </h3>
    </div>
    """
    display(HTML(html_code))

# Function for Correlation between Home and Visitor Team Win Probabilities
def win_probability_analysis(df):
    win_probability_analysis = df[['homeTeamWinProbabilityAdded', 'visitorTeamWinProbilityAdded']].corr()
    win_probability_analysis_reset = win_probability_analysis.reset_index()
    win_probability_analysis_reset.columns = ['Feature', 'homeTeamWinProbabilityAdded', 'visitorTeamWinProbilityAdded']

    styled_heading("ğŸ“Š WIN PROBABILITY CORRELATION")
    display(HTML(win_probability_analysis_reset.to_html(index=False, header=True)))
    print("===========================================================")

# Function for Expected Points Added vs Home Team Win Probability
def expected_points_home_win_analysis(df):
    expected_points_analysis = df.groupby('homeTeamWinProbabilityAdded').agg(
        avg_expected_points_added=('expectedPointsAdded', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index().head(10)

    styled_heading("ğŸ“Š EXPECTED POINTS ADDED vs HOME TEAM WIN PROBABILITY")
    display(HTML(expected_points_analysis.to_html(index=False, header=True)))
    print("===========================================================")

# Function for Dropback vs Expected Points Added
def dropback_analysis(df):
    dropback_analysis = df.groupby('isDropback').agg(
        avg_expected_points_added=('expectedPointsAdded', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index()

    styled_heading("ğŸ“Š DROPBACK ANALYSIS vs EXPECTED POINTS ADDED")
    display(HTML(dropback_analysis.to_html(index=False, header=True)))
    print("===========================================================")

# Function for Run Concept Primary vs Expected Points Added
def run_concept_primary_analysis(df):
    run_concept_primary_analysis = df.groupby('pff_runConceptPrimary').agg(
        avg_expected_points_added=('expectedPointsAdded', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index().head(10)

    styled_heading("ğŸ“Š RUN CONCEPT PRIMARY vs EXPECTED POINTS ADDED")
    display(HTML(run_concept_primary_analysis.to_html(index=False, header=True)))
    print("===========================================================")

# Function for Run Concept Secondary vs Expected Points Added
def run_concept_secondary_analysis(df):
    run_concept_secondary_analysis = df.groupby('pff_runConceptSecondary').agg(
        avg_expected_points_added=('expectedPointsAdded', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index().head(10)

    styled_heading("ğŸ“Š RUN CONCEPT SECONDARY vs EXPECTED POINTS ADDED")
    display(HTML(run_concept_secondary_analysis.to_html(index=False, header=True)))
    print("===========================================================")

# Function for Run Pass Option vs Expected Points Added
def run_pass_option_analysis(df):
    run_pass_option_analysis = df.groupby('pff_runPassOption').agg(
        avg_expected_points_added=('expectedPointsAdded', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index().head(10)

    styled_heading("ğŸ“Š RUN PASS OPTION vs EXPECTED POINTS ADDED")
    display(HTML(run_pass_option_analysis.to_html(index=False, header=True)))
    print("===========================================================")

# Function for Pass Coverage vs Expected Points Added
def pass_coverage_analysis(df):
    pass_coverage_analysis = df.groupby('pff_passCoverage').agg(
        avg_expected_points_added=('expectedPointsAdded', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index().head(10)

    styled_heading("ğŸ“Š PASS COVERAGE vs EXPECTED POINTS ADDED")
    display(HTML(pass_coverage_analysis.to_html(index=False, header=True)))
    print("===========================================================")

# Function for Man vs Zone Coverage Analysis
def man_zone_coverage_analysis(df):
    man_zone_coverage_analysis = df.groupby('pff_manZone').agg(
        avg_expected_points_added=('expectedPointsAdded', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index()

    styled_heading("ğŸ“Š MAN vs ZONE COVERAGE vs EXPECTED POINTS ADDED")
    display(HTML(man_zone_coverage_analysis.to_html(index=False, header=True)))
    print("===========================================================")

# Function for Team Win Probability vs Expected Points Added
def win_prob_vs_points_analysis(df):
    win_prob_vs_points = df.groupby('homeTeamWinProbabilityAdded').agg(
        avg_expected_points_added=('expectedPointsAdded', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index().head(10)

    styled_heading("ğŸ“Š HOME TEAM WIN PROBABILITY vs EXPECTED POINTS ADDED")
    display(HTML(win_prob_vs_points.to_html(index=False, header=True)))
    print("===========================================================")

# Function for Visitor Team Win Probability vs Expected Points Added
def visitor_win_prob_vs_points_analysis(df):
    visitor_win_prob_vs_points = df.groupby('visitorTeamWinProbilityAdded').agg(
        avg_expected_points_added=('expectedPointsAdded', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index().head(10)

    styled_heading("ğŸ“Š VISITOR TEAM WIN PROBABILITY vs EXPECTED POINTS ADDED")
    display(HTML(visitor_win_prob_vs_points.to_html(index=False, header=True)))
    print("===========================================================")

# Main function to call all analyses
def main_analysis(df):
    win_probability_analysis(df)
    expected_points_home_win_analysis(df)
    dropback_analysis(df)
    run_concept_primary_analysis(df)
    run_concept_secondary_analysis(df)
    run_pass_option_analysis(df)
    pass_coverage_analysis(df)
    man_zone_coverage_analysis(df)
    win_prob_vs_points_analysis(df)
    visitor_win_prob_vs_points_analysis(df)


# ===Main Execution===
main_analysis(merged_nfl_data)


# Styled Heading Function
def styled_heading(text, color="#382411", background="#ba8d63", border="#382411"):
    """
    Display a styled heading using HTML.
    
    Parameters:
    text (str): The heading text to display.
    color (str): Text color (default: #382411).
    background (str): Background color (default: #ba8d63).
    border (str): Border color (default: #382411).
    """
    html_code = f"""
    <div style='background-color: {background}; border: 6px solid {border}; 
                padding: 10px; margin-bottom: 10px; border-radius: 5px;'>
        <h3 style='color: {color}; text-align: center; font-family: Arial, sans-serif;'>
            {text}
        </h3>
    </div>
    """
    display(HTML(html_code))

# 1. Run Concepts vs Pass Coverage Interaction
def run_concept_coverage_interaction_analysis(df):
    run_concept_coverage_interaction = df.groupby(['pff_runConceptPrimary', 'pff_passCoverage']).agg(
        avg_expected_points_added=('expectedPointsAdded', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index().head(10)

    styled_heading("ğŸ“Š RUN CONCEPT and PASS COVERAGE Interaction Analysis")
    display(HTML(run_concept_coverage_interaction.to_html(index=False, header=True)))
    print("===========================================================")

# 2. Win Probability Shift Analysis
def win_probability_shift_analysis(df):
    win_probability_shift = df.copy()
    win_probability_shift['win_prob_shift_home'] = df['homeTeamWinProbabilityAdded'].diff().fillna(0)
    win_probability_shift['win_prob_shift_visitor'] = df['visitorTeamWinProbilityAdded'].diff().fillna(0)

    shift_analysis_home = win_probability_shift.groupby('win_prob_shift_home').agg(
        avg_expected_points_added=('expectedPointsAdded', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index().head(10)

    shift_analysis_visitor = win_probability_shift.groupby('win_prob_shift_visitor').agg(
        avg_expected_points_added=('expectedPointsAdded', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index().head(10)

    styled_heading("ğŸ“Š WIN PROBABILITY SHIFT (Home) vs Expected Points Added")
    display(HTML(shift_analysis_home.to_html(index=False, header=True)))
    styled_heading("ğŸ“Š WIN PROBABILITY SHIFT (Visitor) vs Expected Points Added")
    display(HTML(shift_analysis_visitor.to_html(index=False, header=True)))
    print("===========================================================")

# 3. Is Dropback vs Expected Points Added per Quarter
def dropback_quarter_analysis(df):
    dropback_quarter_analysis = df.groupby(['isDropback', 'quarter']).agg(
        avg_expected_points_added=('expectedPointsAdded', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index()

    styled_heading("ğŸ“Š DROPBACK vs EXPECTED POINTS ADDED per QUARTER")
    display(HTML(dropback_quarter_analysis.to_html(index=False, header=True)))
    print("===========================================================")

# 4. Impact of Offensive Concepts on Points
def offensive_concept_analysis(df):
    offensive_concept_analysis = df.groupby(['pff_runConceptPrimary', 'pff_runPassOption']).agg(
        avg_expected_points_added=('expectedPointsAdded', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index()

    styled_heading("ğŸ“Š IMPACT of OFFENSIVE CONCEPTS (Primary vs Pass Option) on Expected Points Added")
    display(HTML(offensive_concept_analysis.to_html(index=False, header=True)))
    print("===========================================================")

# 5. Pass Coverage vs Expected Points Added by Home Team Win Probability
def pass_coverage_win_prob_analysis(df):
    pass_coverage_win_prob_analysis = df.groupby(['pff_passCoverage', 'homeTeamWinProbabilityAdded']).agg(
        avg_expected_points_added=('expectedPointsAdded', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index().head(10)

    styled_heading("ğŸ“Š PASS COVERAGE vs EXPECTED POINTS ADDED by HOME TEAM WIN PROBABILITY")
    display(HTML(pass_coverage_win_prob_analysis.to_html(index=False, header=True)))
    print("===========================================================")

# 6. Time-based Analysis: Win Probability and Run Concepts
def time_based_analysis(df):
    time_based_analysis = df.groupby(['quarter', 'pff_runConceptPrimary']).agg(
        avg_home_team_win_prob=('homeTeamWinProbabilityAdded', 'mean'),
        avg_expected_points_added=('expectedPointsAdded', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index().head(10)

    styled_heading("ğŸ“Š TIME-BASED ANALYSIS: WIN PROBABILITY and RUN CONCEPTS vs EXPECTED POINTS ADDED")
    display(HTML(time_based_analysis.to_html(index=False, header=True)))
    print("===========================================================")

# 7. Run Concept Comparison: Primary vs Secondary
def run_concept_comparison(df):
    run_concept_comparison = df.groupby(['pff_runConceptPrimary', 'pff_runConceptSecondary']).agg(
        avg_expected_points_added=('expectedPointsAdded', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index().head(10)

    styled_heading("ğŸ“Š PRIMARY vs SECONDARY RUN CONCEPT COMPARISON")
    display(HTML(run_concept_comparison.to_html(index=False, header=True)))
    print("===========================================================")

# 8. Is Dropback Flag vs Pass Coverage Type Analysis
def dropback_vs_coverage_analysis(df):
    dropback_vs_coverage_analysis = df.groupby(['isDropback', 'pff_passCoverage']).agg(
        avg_expected_points_added=('expectedPointsAdded', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index().head(10)

    styled_heading("ğŸ“Š IS DROPBACK FLAG vs PASS COVERAGE TYPE Analysis")
    display(HTML(dropback_vs_coverage_analysis.to_html(index=False, header=True)))
    print("===========================================================")

# 9. Player/Team Strategy Analysis based on Run Concepts and Pass Coverage
def strategy_analysis(df):
    strategy_analysis = df.groupby(['pff_runConceptPrimary', 'pff_passCoverage']).agg(
        avg_expected_points_added=('expectedPointsAdded', 'mean'),
        total_plays=('playDescription', 'count')
    ).reset_index().head(10)

    styled_heading("ğŸ“Š PLAYER/TEAM STRATEGY based on RUN CONCEPTS and PASS COVERAGE")
    display(HTML(strategy_analysis.to_html(index=False, header=True)))
    print("=========================================================================")

# Main function to call all analyses
def main_analysis(df):
    run_concept_coverage_interaction_analysis(df)
    win_probability_shift_analysis(df)
    dropback_quarter_analysis(df)
    offensive_concept_analysis(df)
    pass_coverage_win_prob_analysis(df)
    time_based_analysis(df)
    run_concept_comparison(df)
    dropback_vs_coverage_analysis(df)
    strategy_analysis(df)


# ===Main Execution===
main_analysis(merged_nfl_data)


# Styled Heading Function
def styled_heading(text, color="#382411", background="#ba8d63", border="#382411"):
    """
    Display a styled heading using HTML.
    
    Parameters:
    text (str): The heading text to display.
    color (str): Text color (default: #382411).
    background (str): Background color (default: #ba8d63).
    border (str): Border color (default: #382411).
    """
    html_code = f"""
    <div style='background-color: {background}; border: 6px solid {border}; 
                padding: 10px; margin-bottom: 10px; border-radius: 5px;'>
        <h3 style='color: {color}; text-align: center; font-family: Arial, sans-serif;'>
            {text}
        </h3>
    </div>
    """
    display(HTML(html_code))

# 1. Calculate Home Wins vs Visitor Wins
def home_visitor_win_analysis(df):
    # Calculate home wins
    home_win_percentage = df[df['homeFinalScore'] > df['visitorFinalScore']].groupby('homeTeamAbbr').agg(
        home_win_count=('gameDate', 'count')
    ).reset_index()

    # Calculate visitor wins
    visitor_win_percentage = df[df['visitorFinalScore'] > df['homeFinalScore']].groupby('visitorTeamAbbr').agg(
        visitor_win_count=('gameDate', 'count')
    ).reset_index()

    # Combine data for comparison
    combined_win_data = pd.merge(home_win_percentage, visitor_win_percentage, left_on='homeTeamAbbr', right_on='visitorTeamAbbr', how='outer')

    # Impute missing values for the 'home_win_count' and 'visitor_win_count' columns using median
    combined_win_data['home_win_count'].fillna(combined_win_data['home_win_count'].median(), inplace=True)
    combined_win_data['visitor_win_count'].fillna(combined_win_data['visitor_win_count'].median(), inplace=True)

    # Impute missing values for 'homeTeamAbbr' column
    combined_win_data['homeTeamAbbr'].fillna(combined_win_data['homeTeamAbbr'].mode()[0], inplace=True)

    styled_heading("ğŸ�† HOME vs VISITOR WIN ANALYSIS")
    display(HTML(combined_win_data.to_html(index=False, header=True)))
    print("===========================================================")

# 2. Calculate Player Age and Perform Age vs Position Analysis
def player_age_position_analysis(df):
    # Convert birthDate to datetime with coercion for different formats
    df['birthDate'] = pd.to_datetime(df['birthDate'], errors='coerce', dayfirst=False)

    # Calculate age
    df['age'] = (pd.to_datetime('today') - df['birthDate']).dt.days // 365

    # Perform age-position analysis
    age_position_analysis = df.groupby(['age', 'position']).agg(
        avg_home_final_score=('homeFinalScore', 'mean'),
        avg_visitor_final_score=('visitorFinalScore', 'mean'),
        total_players=('displayName', 'count')
    ).reset_index().head(20)

    styled_heading("ğŸ“Š AGE vs POSITION ANALYSIS")
    display(HTML(age_position_analysis.to_html(index=False, header=True)))
    print("================================================================")

# Main function to execute all analyses
def main_analysis(df):
    home_visitor_win_analysis(df)
    player_age_position_analysis(df)


# ===Main Execution===
main_analysis(merged_nfl_data)


# Styled Heading Function
def styled_heading(text, color="#382411", background="#ba8d63", border="#382411"):
    html_code = f"""
    <div style='background-color: {background}; border: 6px solid {border}; 
                padding: 10px; margin-bottom: 10px; border-radius: 5px;'>
        <h3 style='color: {color}; text-align: center; font-family: Arial, sans-serif;'>
            {text}
        </h3>
    </div>
    """
    display(HTML(html_code))

# 1. Performance by Team Abbreviation (Home vs Visitor) Over Time
def team_performance_analysis_func(df):
    team_performance_analysis = df.groupby(['homeTeamAbbr', 'season']).agg(
        avg_home_score=('homeFinalScore', 'mean'),
        avg_visitor_score=('visitorFinalScore', 'mean')
    ).reset_index()

    styled_heading("ğŸ�ˆ Performance by Team Abbreviation (Home vs Visitor) Over Time")
    display(HTML(team_performance_analysis.to_html(index=False, header=True)))
    print("===========================================================")

    return team_performance_analysis

# 2. Age vs Player Performance (Final Scores)
def age_performance_analysis_func(df):
    df['age'] = (pd.to_datetime('today') - pd.to_datetime(df['birthDate'])).dt.days // 365
    age_performance_analysis = df.groupby('age').agg(
        avg_home_final_score=('homeFinalScore', 'mean'),
        avg_visitor_final_score=('visitorFinalScore', 'mean')
    ).reset_index()

    styled_heading("ğŸ“Š Age vs Player Performance Analysis")
    display(HTML(age_performance_analysis.to_html(index=False, header=True)))
    print("===========================================================")

    return age_performance_analysis

# 3. Weight Distribution for Specific Positions
def weight_distribution_func(df):
    weight_distribution = df.groupby('position').agg(
        avg_weight=('weight', 'mean')
    ).reset_index()

    styled_heading("âš–ï¸� Weight Distribution for Specific Positions")
    display(HTML(weight_distribution.to_html(index=False, header=True)))
    print("===========================================================")
    print("Height Summary Stats\n", df["height"].describe())
    print("===========================================================")

    return weight_distribution

# 4. Position-Specific Performance (Win Percentage by Position)
def position_performance_analysis_func(df):
    df['home_win'] = df['homeFinalScore'] > df['visitorFinalScore']
    df['visitor_win'] = df['visitorFinalScore'] > df['homeFinalScore']

    position_win_percentage = df.groupby('position').agg(
        home_wins=('home_win', 'sum'),
        visitor_wins=('visitor_win', 'sum'),
        total_games=('gameDate', 'count')
    ).reset_index()

    # Calculate win percentages
    position_win_percentage['home_win_percentage'] = position_win_percentage['home_wins'] / position_win_percentage['total_games'] * 100
    position_win_percentage['visitor_win_percentage'] = position_win_percentage['visitor_wins'] / position_win_percentage['total_games'] * 100

    styled_heading("ğŸ“ˆ Position-Specific Performance (Win Percentage by Position)")
    display(HTML(position_win_percentage.to_html(index=False, header=True)))
    print("=================================================================================")

    return position_win_percentage

# Main function to execute all analyses
def main_analysis(df):
    """
    Run all analyses and return their results as individual DataFrames.
    """
    team_performance = team_performance_analysis_func(df)
    age_performance = age_performance_analysis_func(df)
    weight_distribution = weight_distribution_func(df)
    position_performance = position_performance_analysis_func(df)

    # Return results for unpacking
    return age_performance, weight_distribution, team_performance, position_performance


# ===Main Execution===
age_performance_analysis, weight_distribution, team_performance_analysis, position_performance_analysis = main_analysis(merged_nfl_data)


# Function to create subplots for multiple analyses
def plot_analysis_results(
    merged_df, combined_win_data, age_performance_analysis, weight_distribution, team_performance_analysis, position_win_percentage
):
    # Set plot style
    sns.set_theme(style="whitegrid")
    
    # Create a figure with larger size (increased to 35x28)
    fig, axes = plt.subplots(3, 2, figsize=(38, 28))
    fig.suptitle("Analysis Results", fontsize=32, fontweight="bold", color="#2b2b2b")

    # Define custom palettes for better visual appeal
    palette_hist = sns.color_palette("viridis", as_cmap=True)
    palette_line = sns.color_palette("coolwarm", 6)
    palette_bar = sns.color_palette("cubehelix", 6)

    # 1. Histogram of Home Win Count
    sns.histplot(
        data=combined_win_data, 
        x='home_win_count', 
        kde=True, 
        ax=axes[0, 0], 
        color=palette_hist(0.7)
    )
    axes[0, 0].set_title("Histogram of Home Win Count", fontsize=24, fontweight="bold", color="#4a4a4a")
    axes[0, 0].set_xlabel("Home Win Count", fontsize=20, color="#4a4a4a")
    axes[0, 0].set_ylabel("Frequency", fontsize=20, color="#4a4a4a")
    axes[0, 0].tick_params(axis='both', which='major', labelsize=18)
    axes[0, 0].set_facecolor("#f7f7f7")

    # 2. Line Plot: Age vs Final Scores
    sns.lineplot(
        data=age_performance_analysis, 
        x='age', 
        y='avg_home_final_score', 
        ax=axes[0, 1], 
        label='Home Team', 
        color=palette_line[0], 
        linewidth=3
    )
    sns.lineplot(
        data=age_performance_analysis, 
        x='age', 
        y='avg_visitor_final_score', 
        ax=axes[0, 1], 
        label='Visitor Team', 
        color=palette_line[3], 
        linewidth=3
    )
    axes[0, 1].set_title("Age vs Player Performance (Final Scores)", fontsize=24, fontweight="bold", color="#4a4a4a")
    axes[0, 1].set_xlabel("Age", fontsize=20, color="#4a4a4a")
    axes[0, 1].set_ylabel("Average Final Score", fontsize=20, color="#4a4a4a")
    axes[0, 1].legend(fontsize=18)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=18)
    axes[0, 1].set_facecolor("#f0f7fa")

    # 3. Weight Distribution for Specific Positions
    bar_plot = sns.barplot(
        data=weight_distribution, 
        x='position', 
        y='avg_weight', 
        ax=axes[1, 0], 
        palette=palette_bar
    )
    # Annotating the bars with values
    for p in bar_plot.patches:
        bar_plot.annotate(
            f'{p.get_height():.2f}', 
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha='center', va='center', 
            fontsize=16, color='black', 
            xytext=(0, 9), textcoords='offset points'
        )
    axes[1, 0].set_title("Weight Distribution by Position", fontsize=24, fontweight="bold", color="#4a4a4a")
    axes[1, 0].set_xlabel("Position", fontsize=20, color="#4a4a4a")
    axes[1, 0].set_ylabel("Average Weight", fontsize=20, color="#4a4a4a")
    axes[1, 0].tick_params(axis='both', which='major', labelsize=18)
    axes[1, 0].set_facecolor("#fbf3e3")

    # 4. Position-Specific Home Win Percentage
    bar_plot = sns.barplot(
        data=position_win_percentage, 
        x='position', 
        y='home_win_percentage', 
        ax=axes[1, 1], 
        palette="Blues_d"
    )
    # Annotating the bars with values
    for p in bar_plot.patches:
        bar_plot.annotate(
            f'{p.get_height():.2f}%', 
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha='center', va='center', 
            fontsize=16, color='black', 
            xytext=(0, 9), textcoords='offset points'
        )
    axes[1, 1].set_title("Home Win Percentage by Position", fontsize=24, fontweight="bold", color="#4a4a4a")
    axes[1, 1].set_xlabel("Position", fontsize=20, color="#4a4a4a")
    axes[1, 1].set_ylabel("Home Win Percentage (%)", fontsize=20, color="#4a4a4a")
    axes[1, 1].tick_params(axis='both', which='major', labelsize=18)
    axes[1, 1].set_facecolor("#e6f5f2")

    # 5. Win Percentage Comparison (Home vs Visitor)
    sns.barplot(
        data=combined_win_data, 
        x='homeTeamAbbr', 
        y='home_win_count', 
        ax=axes[2, 0], 
        color=palette_hist(0.5), 
        label="Home Wins"
    )
    sns.barplot(
        data=combined_win_data, 
        x='visitorTeamAbbr', 
        y='visitor_win_count', 
        ax=axes[2, 0], 
        color=palette_hist(0.9), 
        label="Visitor Wins"
    )
    axes[2, 0].set_title("Win Percentage: Home vs Visitor", fontsize=24, fontweight="bold", color="#4a4a4a")
    axes[2, 0].set_xlabel("Team Abbreviation", fontsize=20, color="#4a4a4a")
    axes[2, 0].set_ylabel("Win Count", fontsize=20, color="#4a4a4a")
    axes[2, 0].legend(fontsize=18)
    axes[2, 0].tick_params(axis='both', which='major', labelsize=18)
    axes[2, 0].set_facecolor("#f9f4f2")

    # 6. Position-Specific Visitor Win Percentage
    sns.barplot(
        data=position_win_percentage, 
        x='position', 
        y='visitor_win_percentage', 
        ax=axes[2, 1], 
        palette="Reds_d"
    )
    # Annotating the bars with values
    for p in axes[2, 1].patches:
        axes[2, 1].annotate(
            f'{p.get_height():.2f}%', 
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha='center', va='center', 
            fontsize=16, color='black', 
            xytext=(0, 9), textcoords='offset points'
        )
    axes[2, 1].set_title("Visitor Win Percentage by Position", fontsize=24, fontweight="bold", color="#4a4a4a")
    axes[2, 1].set_xlabel("Position", fontsize=20, color="#4a4a4a")
    axes[2, 1].set_ylabel("Visitor Win Percentage (%)", fontsize=20, color="#4a4a4a")
    axes[2, 1].tick_params(axis='both', which='major', labelsize=18)
    axes[2, 1].set_facecolor("#ffe7e7")

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Show the plot
    plt.show()

#Calculate home wins
home_win_percentage = merged_nfl_data[merged_nfl_data['homeFinalScore'] > merged_nfl_data['visitorFinalScore']].groupby('homeTeamAbbr').agg(home_win_count=('gameDate', 'count')).reset_index()

# Calculate visitor wins
visitor_win_percentage = merged_nfl_data[merged_nfl_data['visitorFinalScore'] > merged_nfl_data['homeFinalScore']].groupby('visitorTeamAbbr').agg(visitor_win_count=('gameDate', 'count')).reset_index()

# Combine data for comparison
combined_win_data = pd.merge(home_win_percentage, visitor_win_percentage, left_on='homeTeamAbbr', right_on='visitorTeamAbbr', how='outer')

# Impute missing values for the 'home_win_count' and 'visitor_win_count' columns using median
combined_win_data['home_win_count'].fillna(combined_win_data['home_win_count'].median(), inplace=True)
combined_win_data['visitor_win_count'].fillna(combined_win_data['visitor_win_count'].median(), inplace=True)

# Impute missing values for 'homeTeamAbbr' column
combined_win_data['homeTeamAbbr'].fillna(combined_win_data['homeTeamAbbr'].mode()[0], inplace=True)


# ===Main Execution===
plot_analysis_results(
    merged_nfl_data, 
    combined_win_data, 
    age_performance_analysis, 
    weight_distribution, 
    team_performance_analysis, 
    position_performance_analysis
)


# Function to load, merge, and display NFL datasets with styled output
def load_and_merge_data():
    """
    This function loads, renames, and merges multiple NFL datasets:
    - games.csv
    - player_play.csv
    - players.csv
    - plays.csv

    Returns:
        merged_df: The final merged DataFrame containing all relevant information.
    """
    
    # Helper function for styled headings
    def styled_heading(text):
        return f"""
        <div style='
            background-color: #ba8d63; 
            color: #382411; 
            padding: 10px; 
            font-size: 18px; 
            border-radius: 5px; 
            text-align: center;'>
            {text}
        </div>
        """

    # Load the data (as per your provided code)
    df_track1 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/tracking_week_1.csv")
    df_track2 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/tracking_week_2.csv")
    df_track3 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/tracking_week_3.csv")
    df_track4 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/tracking_week_4.csv")
    df_track5 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/tracking_week_5.csv")
    df_track6 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/tracking_week_6.csv")
    df_track7 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/tracking_week_7.csv")
    df_track8 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/tracking_week_8.csv")
    df_track9 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/tracking_week_9.csv")

    display(HTML(styled_heading("Datasets loaded successfully!")))

    # Concatenate data for all weeks (1-9)
    df_combine_tracks = pd.concat([df_track1, df_track2, df_track3, df_track4, df_track5, df_track6, df_track7, df_track8, df_track9], ignore_index=True)

    display(HTML(styled_heading("Data Merging Completed!")))

    return df_combine_tracks


# ===Main Execution===
df_combine_tracks = load_and_merge_data()


# Function to create styled headings
def styled_heading(text, background_color='#ffabf0', text_color='black'):
    return f"""
    <div style="
        text-align: center;
        background: {background_color};
        font-family: 'Montserrat', sans-serif;
        color: {text_color};
        padding: 15px;
        font-size: 30px;
        font-weight: bold;
        line-height: 1.2;
        border-radius: 20px 20px 0 0;
        margin: 20px 0;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        border: 3px dashed {text_color};
    ">
        {text}
    </div>
    """

# Function for displaying data overview
def display_overview(train_df, heading_bg='#b379ed', heading_color='black', text_bg='white', text_color='black'):
    try:
        # Display head, tail, and numerical summary
        sections = [
            ("The Head of the Dataset:", train_df.head(5)),
            ("The Tail of the Dataset:", train_df.tail(5)),
            ("Numerical Summary of the Data:", train_df.describe())
        ]
        
        for heading, df_part in sections:
            display(HTML(styled_heading(heading, background_color=heading_bg, text_color=heading_color)))
            display(HTML(df_part.to_html(index=False).replace(
                '<table border="1" class="dataframe">',
                f'<table style="border: 8px solid black; margin-bottom: 20px; background-color: {text_bg}; color: {text_color};">'
            ).replace('<td>', f'<td style="color: {text_color}; background-color: {text_bg};">')))
        
        # Print shape data
        display(HTML(styled_heading("Shape of the Dataset:", background_color=heading_bg, text_color=heading_color)))
        shape_details = f"""
        Rows: {train_df.shape[0]}  
        Columns: {train_df.shape[1]}
        """
        display(HTML(f"<p style='color: {text_color}; background-color: {text_bg}; padding: 10px; border: 8px solid black;'>{shape_details}</p>"))
        
        # Display dataset info
        display(HTML(styled_heading("Dataset Information:", background_color=heading_bg, text_color=heading_color)))
        buffer = StringIO()
        train_df.info(buf=buffer)
        buffer.seek(0)
        info_str = buffer.read()
        display(HTML(f"<pre style='color: {text_color}; background-color: {text_bg}; margin-bottom: 20px; font-family: Courier, monospace; font-size: 14px; padding: 10px; border: 8px solid black;'>{info_str}</pre>"))

        # Display categorical columns
        categorical_columns = [col for col in train_df.columns if train_df[col].dtype == 'O']
        display(HTML(styled_heading("Categorical Columns in the Dataset:", background_color=heading_bg, text_color=heading_color)))
        cat_cols_str = f"The categorical columns are: {', '.join(categorical_columns)}" if categorical_columns else "No categorical columns found."
        display(HTML(f"<p style='color: {text_color}; background-color: {text_bg}; padding: 10px; border: 8px solid black;'>{cat_cols_str}</p>"))

        # Display numerical columns
        numerical_columns = [col for col in train_df.columns if train_df[col].dtype in ['float64', 'int64']]
        display(HTML(styled_heading("Numerical Columns in the Dataset:", background_color=heading_bg, text_color=heading_color)))
        num_cols_str = f"The numerical columns are: {', '.join(numerical_columns)}" if numerical_columns else "No numerical columns found."
        display(HTML(f"<p style='color: {text_color}; background-color: {text_bg}; padding: 10px; border: 8px solid black;'>{num_cols_str}</p>"))

        # Display null values
        display(HTML(styled_heading("Null Values in the Dataset:", background_color=heading_bg, text_color=heading_color)))
        null_values = train_df.isnull().sum()
        display(HTML(f"<pre style='color: {text_color}; background-color: {text_bg}; margin-bottom: 20px; font-family: Courier, monospace; font-size: 14px; padding: 10px; border: 8px solid black;'>{null_values.to_string()}</pre>"))

        # Check for duplicates
        display(HTML(styled_heading("Duplicate Records Check:", background_color=heading_bg, text_color=heading_color)))
        duplicates_exist = train_df.duplicated().any()
        dup_msg = "Duplicates exist in the dataset." if duplicates_exist else "No duplicate records found."
        display(HTML(f"<p style='color: {text_color}; background-color: {text_bg}; padding: 10px; border: 8px solid black;'>{dup_msg}</p>"))

    except Exception as e:
        display(HTML(f"<div style='color: red; font-weight: bold;'>Error: {str(e)}</div>"))


# ===Main Execution===
display_overview(df_combine_tracks, heading_bg='#ba8d63', heading_color='#382411', text_bg='#ba8d63', text_color='#382411')


def styled_heading(text, color="#382411", background="#ba8d63", border="#382411"):
    """
    Display a styled heading using HTML.
    
    Parameters:
    text (str): The heading text to display.
    color (str): Text color (default: #382411).
    background (str): Background color (default: #ba8d63).
    border (str): Border color (default: #382411).
    """
    html_code = f"""
    <div style='background-color: {background}; border: 6px solid {border}; 
                padding: 10px; margin-bottom: 10px; border-radius: 5px;'>
        <h3 style='color: {color}; text-align: center; font-family: Arial, sans-serif;'>
            {text}
        </h3>
    </div>
    """
    display(HTML(html_code))

def impute_missing_values(df):
    """
    Impute missing values in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame with missing values.
    
    Returns:
    pd.DataFrame: The DataFrame with imputed values.
    """
    # Iterate over columns
    for column in df.columns:
        if df[column].isnull().sum() > 0:  
            if df[column].dtype == 'object':  
                most_frequent_value = df[column].mode()[0]
                df[column].fillna(most_frequent_value, inplace=True)
            elif df[column].dtype in ['int64', 'float64']:  
                mean_value = df[column].mean()
                df[column].fillna(mean_value, inplace=True)
            elif df[column].dtype == 'bool':  
                mode_value = df[column].mode()[0]
                df[column].fillna(mode_value, inplace=True)
    return df


# ===Main Execution===
styled_heading("Before Imputation: Missing Values Overview")
display(df_combine_tracks.isnull().sum())

# Perform imputation
df_combine_tracks = impute_missing_values(df_combine_tracks)

styled_heading("After Imputation: Missing Values Overview")
display(df_combine_tracks.isnull().sum())


def styled_heading(text, color="#382411", background="#ba8d63", border="#382411"):
    """
    Display a styled heading using HTML.
    
    Parameters:
    text (str): The heading text to display.
    color (str): Text color (default: #382411).
    background (str): Background color (default: #ba8d63).
    border (str): Border color (default: #382411).
    """
    html_code = f"""
    <div style='background-color: {background}; border: 6px solid {border}; 
                padding: 10px; margin-bottom: 10px; border-radius: 5px;'>
        <h3 style='color: {color}; text-align: center; font-family: Arial, sans-serif;'>
            {text}
        </h3>
    </div>
    """
    display(HTML(html_code))
    
def compute_basic_insights(df):
    """Compute basic statistical insights and frequency distribution."""
    styled_heading("Basic Statistical Insights")
    summary = df.describe(include='all').transpose()
    summary['missing_values'] = df.isnull().sum()
    display(summary.style.set_caption("Dataset Summary"))

    styled_heading("Column Frequency Distribution")
    for col in df.select_dtypes(include=['object']).columns:
        styled_heading(f"Frequency Distribution for {col}")
        display(df[col].value_counts().head(10).to_frame(name='Frequency'))

def compute_advanced_insights(df):
    """Perform advanced analysis including temporal and contextual insights."""
    styled_heading("Advanced Insights")

    # Temporal Analysis
    if 'timestamp' in df.columns:
        styled_heading("Temporal Analysis")
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['hour'] = df['timestamp'].dt.hour
        hourly_counts = df['hour'].value_counts().sort_index()
        display(hourly_counts.to_frame(name='Frequency'))

    # Positional Analysis
    if {'x', 'y'}.issubset(df.columns):
        styled_heading("Positional Analysis")
        avg_position = df[['x', 'y']].mean()
        display(avg_position.to_frame(name='Average Position'))

    # Player/Club Analysis
    if {'playerId', 'club'}.issubset(df.columns):
        styled_heading("Player & Club Analysis")
        top_players = df['playerId'].value_counts().head(10)
        display(top_players.to_frame(name='Top Players'))

        top_clubs = df['club'].value_counts().head(10)
        display(top_clubs.to_frame(name='Top Clubs'))

    # Relationship Analysis
    if {'playId', 'gameId'}.issubset(df.columns):
        styled_heading("Relationship Analysis between playId and gameId")
        grouped = df.groupby('gameId')['playId'].nunique()
        display(grouped.to_frame(name='Unique playId per gameId'))

def comprehensive_analysis(df):
    """Perform a combined analysis, including both basic and advanced insights."""
    styled_heading("Comprehensive Dataset Analysis")
    compute_basic_insights(df)
    compute_advanced_insights(df)


# ===Main Execution===
if __name__ == "__main__":
    comprehensive_analysis(df_combine_tracks)


palette = ['#F4A300', '#382411', '#7d410c', '#b37742', '#755130', '#ba8d63', '#241508', 
                     '#a87f32', '#524630', '#787061', '#382411', '#7d410c', '#b37742', '#d49a2f']


# Sample dataset 
df_sample = df_combine_tracks.sample(n=5000, random_state=42)

def display_nfl_tracking_plots(df_sample):
    # Set up the figure with subplots: 5 rows, 2 columns (except last row will have 1 plot)
    fig, axes = plt.subplots(5, 2, figsize=(15, 20))
    fig.suptitle('NFL Tracking Data Visualizations', fontsize=16)

    # 1. Scatter plot of player positions (x, y)
    sns.scatterplot(ax=axes[0, 0], data=df_sample, x='x', y='y', hue='jerseyNumber', palette=palette, s=50)
    axes[0, 0].set_title('Player Positions (x, y)')
    axes[0, 0].set_xlabel('X Position')
    axes[0, 0].set_ylabel('Y Position')
    axes[0, 0].legend().set_visible(False)  
 
    # 2. Distribution of player speed (s)
    sns.histplot(ax=axes[0, 1], data=df_sample, x='s', kde=True, color='#F4A300')
    axes[0, 1].set_title('Speed Distribution (s)')
    axes[0, 1].set_xlabel('Speed (s)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend().set_visible(False)  

    # 3. Distribution of acceleration (a)
    sns.histplot(ax=axes[1, 0], data=df_sample, x='a', kde=True, color='#382411')
    axes[1, 0].set_title('Acceleration Distribution (a)')
    axes[1, 0].set_xlabel('Acceleration (a)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend().set_visible(False)  

    # 4. Direction of movement (o) vs. Speed (s)
    sns.scatterplot(ax=axes[1, 1], data=df_sample, x='o', y='s', hue='playDirection', palette=palette, s=50)
    axes[1, 1].set_title('Direction vs. Speed')
    axes[1, 1].set_xlabel('Direction (o)')
    axes[1, 1].set_ylabel('Speed (s)')
    axes[1, 1].legend().set_visible(False)  

    # 5. Distribution of distance (dis)
    sns.histplot(ax=axes[2, 0], data=df_sample, x='dis', kde=True, color='#7d410c')
    axes[2, 0].set_title('Distance Distribution (dis)')
    axes[2, 0].set_xlabel('Distance (dis)')
    axes[2, 0].set_ylabel('Frequency')
    axes[2, 0].legend().set_visible(False)  

    # 6. Distribution of frame type (frameType)
    sns.countplot(ax=axes[2, 1], data=df_sample, x='frameType', palette=palette)
    axes[2, 1].set_title('Frame Type Distribution')
    axes[2, 1].set_xlabel('Frame Type')
    axes[2, 1].set_ylabel('Count')
    axes[2, 1].legend().set_visible(False)  

    # 7. Player movement over the field (x vs. y)
    sns.scatterplot(ax=axes[3, 0], data=df_sample, x='x', y='y', hue='frameType', palette=palette, s=50)
    axes[3, 0].set_title('Player Movement (x vs. y)')
    axes[3, 0].set_xlabel('X Position')
    axes[3, 0].set_ylabel('Y Position')
    axes[3, 0].legend().set_visible(False)  

    # 8. Distribution of player speed (s)
    sns.histplot(ax=axes[3, 1], data=df_sample, x='s', kde=True, color='#b37742')
    axes[3, 1].set_title('Player Speed Distribution')
    axes[3, 1].set_xlabel('Speed (s)')
    axes[3, 1].set_ylabel('Frequency')
    axes[3, 1].legend().set_visible(False)  

    # 9. Distribution of direction (dir)
    sns.histplot(ax=axes[4, 0], data=df_sample, x='dir', kde=True, color='#755130')
    axes[4, 0].set_title('Direction Distribution (dir)')
    axes[4, 0].set_xlabel('Direction (dir)')
    axes[4, 0].set_ylabel('Frequency')
    axes[4, 0].legend().set_visible(False)  

    # Remove the extra axis in the last column
    axes[4, 1].axis('off')

    # Layout adjustments
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  

    # Show the plots
    plt.show()


# ===Main Execution===
display_nfl_tracking_plots(df_sample)


# Define the function to create a 3D scatter plot
def plot_3d_player_position_speed_direction(df):
    """
    Creates and displays a 3D scatter plot of player position (x, y), speed (s), and direction (o),
    with color representing the direction of movement.
    
    Parameters:
    - df: DataFrame containing the data with columns 'x', 'y', 's', and 'o'.
    """
    # Create the 3D scatter plot
    fig1 = px.scatter_3d(df, 
                         x='x', 
                         y='y', 
                         z='s', 
                         color='o', 
                         color_continuous_scale=palette,  
                         labels={'x': 'X Position', 'y': 'Y Position', 's': 'Speed', 'o': 'Direction'},
                         opacity=0.7) 

    # Customize the layout for better readability and aesthetics
    fig1.update_layout(
        title="3D Scatter Plot: Position, Speed, and Direction",
        title_x=0.5,  
        title_font=dict(size=24, color='black', family='Arial'),
        plot_bgcolor='rgba(0, 0, 0, 0)',  
        paper_bgcolor='rgba(240, 240, 240, 1)',  
        font=dict(family='Arial', size=12, color='black'),  
        scene=dict(
            xaxis_title='X Position',
            yaxis_title='Y Position',
            zaxis_title='Speed',
            xaxis=dict(showgrid=True, gridcolor='lightgrey'),
            yaxis=dict(showgrid=True, gridcolor='lightgrey'),
            zaxis=dict(showgrid=True, gridcolor='lightgrey'),
        ),
        height=800, width=800,  
        showlegend=True,  
    )

    # Show the updated 3D scatter plot
    fig1.show()


# ===Main Execution===
plot_3d_player_position_speed_direction(df_sample)


# Define the function to create and display the scatter matrix plot
def plot_multivariate_distribution(df, dimensions, color_column, color_scale=palette, title="Multivariate Distribution: Speed, Acceleration, and Direction"):
    # Create the scatter matrix plot
    fig = px.scatter_matrix(df, 
                            dimensions=dimensions, 
                            color=color_column, 
                            title=title,
                            color_continuous_scale=palette,  
                            labels={dimensions[0]: 'Speed (m/s)', dimensions[1]: 'Acceleration (m/sÂ²)', dimensions[2]: 'Direction (Â°)', color_column: 'Frame Type'})

    # Customize layout for better readability and aesthetics
    fig.update_layout(
        title=title,
        title_x=0.5,  
        title_font=dict(size=24, color='black', family='Arial'),
        plot_bgcolor='rgba(0, 0, 0, 0)',  
        paper_bgcolor='rgba(240, 240, 240, 1)', 
        font=dict(family='Arial', size=12, color='black'),  
        height=800, width=800,  
        showlegend=True,  
        hovermode='closest',  
    )

    # Customize the axes and gridlines
    fig.update_xaxes(showgrid=True, gridcolor='lightgrey', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='lightgrey', zeroline=False)

    # Show the updated scatter matrix plot
    fig.show()


# Sample of 5000 rows 
df_sample = df_combine_tracks.sample(n=5000, random_state=42)

# ===Main Execution===
plot_multivariate_distribution(df_sample, dimensions=['s', 'a', 'o'], color_column='frameType')

