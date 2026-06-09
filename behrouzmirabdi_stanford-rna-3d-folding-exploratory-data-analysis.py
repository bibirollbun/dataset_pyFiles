import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from collections import Counter
from datetime import datetime


train_sequences=pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
train_labels=pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")


print(f"Number of training sequences: {len(train_sequences)}")


train_sequences.head()


train_labels.head()


train_sequences['length'] = train_sequences['sequence'].apply(len)
min_length = train_sequences['length'].min()
max_length = train_sequences['length'].max()
avg_length = train_sequences['length'].mean()



print(f"Sequence length statistics:")
print(f"- Minimum: {min_length}")
print(f"- Maximum: {max_length}")
print(f"- Average: {avg_length:.2f}")


fig = px.histogram(
    train_sequences,
    x='length',
    nbins=30,
    title='Distribution of RNA Sequence Lengths',
    labels={'length': 'Sequence Length', 'count': 'Count'},
    template='plotly_white'
)

# Customize the layout
fig.update_layout(
    width=900,
    height=600,
    title_font_size=20,
    xaxis_title_font_size=16,
    yaxis_title_font_size=16,
    xaxis_title='Sequence Length',
    yaxis_title='Count'
)

# Add grid lines for better readability
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

# Customize histogram appearance
fig.update_traces(
    marker_color='royalblue',
    marker_line_color='darkblue',
    marker_line_width=1.5,
    opacity=0.75
)

# Show the plot
fig.show()


all_nucleotides = ''.join(train_sequences['sequence'].tolist())
nucleotide_counts = Counter(all_nucleotides)
total_nucleotides = sum(nucleotide_counts.values())


print("\nNucleotide distribution:")
for nucleotide, count in nucleotide_counts.most_common():
    percentage = (count / total_nucleotides) * 100
    print(f"{nucleotide}: {count} ({percentage:.2f}%)")


nucleotide_df = pd.DataFrame(nucleotide_counts.items(), columns=['Nucleotide', 'Count'])
nucleotide_df['Percentage'] = (nucleotide_df['Count'] / total_nucleotides) * 100

# Define colors for nucleotides (standard biological coloring)
nucleotide_colors = {
    'A': '#32CD32',  # Green
    'U': '#FFD700',  # Gold
    'G': '#4169E1',  # Blue
    'C': '#FF6347',  # Red
    '-': '#808080',  # Gray
    'X': '#9370DB'   # Purple
}

# Create the bar plot
fig_bar = px.bar(
    nucleotide_df, 
    x='Nucleotide', 
    y='Count', 
    text='Percentage',
 #   labels={'count': 'Count', 'nucleotide': 'Nucleotide'},
    title='Nucleotide Distribution',
    color='Nucleotide',
    color_discrete_map=nucleotide_colors,
    template='plotly_white'
)

# Format the text to show percentages
fig_bar.update_traces(
    texttemplate='%{text:.2f}%', 
    textposition='outside'
)

# Set the size of the bar chart
fig_bar.update_layout(
    title_font_size=24,
    width=800,
    height=600,
    font=dict(size=16)
)



# Create the pie chart (filter out very small values for better visualization)
nucleotide_df_filtered = nucleotide_df[nucleotide_df['Percentage'] > 0.1]

fig_pie = px.pie(
    nucleotide_df_filtered, 
    values='Percentage', 
    names='Nucleotide',
    title='Nucleotide Distribution (%)',
    color='Nucleotide',
    color_discrete_map=nucleotide_colors,
    template='plotly_white'
)

# Update pie chart text format
fig_pie.update_traces(
    textinfo='label+percent', 
    textposition='inside',
    insidetextorientation='radial'
)

# Increase the size of the pie chart
fig_pie.update_layout(
    width=800,    
    height=600,  
    title_font_size=24,  
    font=dict(size=16)   
)


# Display plots one below the other
fig_bar.show()
fig_pie.show()


train_sequences['date'] = pd.to_datetime(train_sequences['temporal_cutoff'])
min_date = train_sequences['date'].min()
max_date = train_sequences['date'].max()
print(f"\nTemporal range: {min_date.date()} to {max_date.date()}")


# Calculate year counts
year_counts = train_sequences['date'].dt.year.value_counts().sort_index()

# Convert to DataFrame for Plotly
year_df = pd.DataFrame({'Year': year_counts.index, 'Count': year_counts.values})

# Create bar chart using Plotly Express
fig = px.bar(
    year_df,
    x='Year',
    y='Count',
    title='Number of RNA Structures by Year',
    labels={'Year': 'Year', 'Count': 'Count'},
    template='plotly_white'
)

# Improve layout
fig.update_layout(
    width=900,
    height=600,
    title_font_size=20,
    xaxis_title_font_size=16,
    yaxis_title_font_size=16
)

# Format x-axis to ensure years display as integers without decimal points
fig.update_xaxes(
    type='category',  # Treat years as categories to maintain order
    tickmode='linear'  # Show all years
)

# Add grid lines for y-axis only
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

# Customize bar appearance
fig.update_traces(
    marker_color='royalblue',
    marker_line_color='darkblue',
    marker_line_width=1.5
)

# Show the plot
fig.show()


pdb_ids = set([target_id.split('_')[0] for target_id in train_sequences['target_id']])
print(f"Number of unique PDB IDs: {len(pdb_ids)}")




