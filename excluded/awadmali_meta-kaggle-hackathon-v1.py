# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Setup and Imports

# --- Standard Libraries ---
import pandas as pd
import numpy as np
import sqlite3
import re
from collections import Counter
import warnings
from tqdm.notebook import tqdm # <--- ADD THIS LINE TO IMPORT tqdm

# --- Visualization Libraries ---
!pip install plotly -q
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import init_notebook_mode
import matplotlib.pyplot as plt
import seaborn as sns

# --- Notebook Settings ---
init_notebook_mode(connected=True)
warnings.filterwarnings('ignore')
# Use a nice template for plotly figures
px.defaults.template = "plotly_dark"

print("âœ… All libraries imported and settings configured successfully.")




# Loading and Merging the Data

# Define the path to the data directory
data_dir = '/kaggle/input/meta-kaggle/'

# Load the necessary CSV files
try:
    print("Loading datasets...")
    kernels_df = pd.read_csv(data_dir + 'Kernels.csv')
    kernel_versions_df = pd.read_csv(data_dir + 'KernelVersions.csv')
    users_df = pd.read_csv(data_dir + 'Users.csv')
    languages_df = pd.read_csv(data_dir + 'KernelLanguages.csv')
    print("âœ… Main datasets loaded.")
    
    # --- Create the Master DataFrame ---
    master_df = pd.merge(kernels_df, kernel_versions_df, left_on='Id', right_on='ScriptId', suffixes=('', '_version'))
    master_df = pd.merge(master_df, users_df, left_on='AuthorUserId', right_on='Id', suffixes=('', '_user'))
    master_df = pd.merge(master_df, languages_df, left_on='ScriptLanguageId', right_on='Id', suffixes=('', '_lang'))
    
    print("âœ… Dataframes merged successfully.")

except FileNotFoundError as e:
    print(f"â�Œ Error loading data: {e}")
    master_df = pd.DataFrame()
    


# Comparing Language Distribution (All vs. Top-Tier)

if not master_df.empty:
    # --- âœ�ï¸� CORRECTION: Use the 'Name' column for language ---
    all_lang_counts = master_df['Name'].value_counts()

    # --- Language distribution for TOP-TIER notebooks ---
    top_tier_df = master_df[master_df['TotalVotes'] > 500]
    # --- âœ�ï¸� CORRECTION: Use the 'Name' column here as well ---
    top_lang_counts = top_tier_df['Name'].value_counts()

    # --- Visualization ---
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                        subplot_titles=['All Notebooks', 'Top-Tier (>500 Votes)'])

    fig.add_trace(go.Pie(labels=all_lang_counts.index, values=all_lang_counts.values, name="All"), 1, 1)
    fig.add_trace(go.Pie(labels=top_lang_counts.index, values=top_lang_counts.values, name="Top-Tier"), 1, 2)

    fig.update_traces(hole=.4, hoverinfo="label+percent+name")
    fig.update_layout(
        title_text="Language Distribution: All Notebooks vs. Top-Tier Notebooks",
        font=dict(color='white')
    )
    fig.show()




# Analyzing and Visualizing Votes by User Tier

if not master_df.empty:
    # --- Analysis ---
    # Group by Performance Tier and calculate the mean of TotalVotes
    tier_votes = master_df.groupby('PerformanceTier')['TotalVotes'].mean().reset_index()

    # Define the correct order for the tiers for proper visualization
    tier_order = ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster']
    tier_votes['PerformanceTier'] = pd.Categorical(tier_votes['PerformanceTier'], categories=tier_order, ordered=True)
    
    # Sort the dataframe by the defined order
    tier_votes = tier_votes.sort_values('PerformanceTier')

    print("Average votes per user tier:")
    print(tier_votes)

    # --- Visualization ---
    fig = px.bar(
        tier_votes,
        x='PerformanceTier',
        y='TotalVotes',
        title='Average Votes vs. User Performance Tier',
        labels={'PerformanceTier': 'User Tier', 'TotalVotes': 'Average Number of Votes'},
        color='PerformanceTier',
        color_discrete_map={
            'Novice': '#5A5A5A',
            'Contributor': '#1E90FF',
            'Expert': '#2ECC40',
            'Master': '#FF851B',
            'Grandmaster': '#FFDC00'
        }
    )
    
    fig.show()




# Create a CSV Output File

# This cell saves the results of our 'votes by tier' analysis to a CSV file.

# Check if the result dataframe exists and is not empty
if 'tier_votes' in locals() and not tier_votes.empty:
    output_filename = 'tier_vote_analysis.csv'
    
    # Save the dataframe to a CSV file
    tier_votes.to_csv(output_filename, index=False)
    
    print(f"âœ… Successfully created the output file: {output_filename}")
    
    # Optional: Display the first few lines to confirm its content
    print("\n--- File Content Preview ---")
    print(pd.read_csv(output_filename))
else:
    print("âš ï¸� Analysis dataframe 'tier_votes' not found or is empty. Skipping file creation.")
    

