import kagglehub
import os
import json
import codecs
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'iframe'

palette_pink_light = '#EED0CE'
palette_yellow = '#F6B254'
palette_maroon = '#983D4F'


# generic function to create a bar chart.

def create_bar_chart(data, x_col, y_col, valueformat='number', title="Bar Chart", subtitle=None,
                     x_label=None, y_label=None, color=None, 
                     width=800, height=500, show_values=True, 
                     sort_by='y', ascending=False, top_n=None):
    """
    Function to create a reusable bar chart
    
    Parameters:
    -----------
    data : pandas.DataFrame or dict or list
        Data for the chart. Can be:
        - DataFrame with x_col and y_col columns
        - dict with keys as x-values and values as y-values
        - list of tuples [(x1, y1), (x2, y2), ...]

    valueformat: str
        Format in which the values are to be displayed on the chart
        
    x_col : str
        Column name for x-axis (if data is DataFrame) or 'x' for other formats
        
    y_col : str  
        Column name for y-axis (if data is DataFrame) or 'y' for other formats
        
    title : str, default "Bar Chart"
        Chart title
        
    x_label : str, optional
        X-axis label (defaults to x_col if None)
        
    y_label : str, optional
        Y-axis label (defaults to y_col if None)
        
    color : str, optional
        Color for bars (default: Plotly blue)
        
    width : int, default 800
        Chart width in pixels
        
    height : int, default 500
        Chart height in pixels
        
    show_values : bool, default True
        Whether to show values on top of bars
        
    sort_by : str, default 'y'
        Sort bars by 'x' or 'y' values
        
    ascending : bool, default False
        Sort order (True for ascending, False for descending)
        
    top_n : int, optional
        Show only top N values (after sorting)
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    
    # Handle different data input formats
    if isinstance(data, dict):
        # Convert dict to lists
        x_values = list(data.keys())
        y_values = list(data.values())
    elif isinstance(data, list):
        # Assume list of tuples
        x_values = [item[0] for item in data]
        y_values = [item[1] for item in data]
    else:
        # Assume DataFrame
        x_values = data[x_col].tolist()
        y_values = data[y_col].tolist()
    
    # Create DataFrame for sorting and filtering
    chart_data = pd.DataFrame({'x': x_values, 'y': y_values})
    
    # Sort data
    if sort_by == 'y':
        chart_data = chart_data.sort_values('y', ascending=ascending)
    elif sort_by == 'x':
        chart_data = chart_data.sort_values('x', ascending=ascending)
    
    # Filter to top N if specified
    if top_n:
        chart_data = chart_data.head(top_n)
    
    # Set default labels
    if x_label is None:
        x_label = x_col if hasattr(data, 'columns') else "Categories"
    if y_label is None:
        y_label = y_col if hasattr(data, 'columns') else "Values"
    
    # Set default color
    if color is None:
        color = '#EED0CE'  # Plotly default blue

    text_template_formatted = '%{text:,.2f}%' if valueformat == 'percentage' else '%{text:.3s}'
    
    # Create the bar chart using Graph Objects
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=chart_data['x'], 
        y=chart_data['y'],
        marker_color=color,
        text=chart_data['y'] if show_values else None,
        textposition='outside' if show_values else None,
        texttemplate= text_template_formatted if show_values else None
    ))

    title_formatted = f"<span style='display:block; margin-bottom:16px; letter-spacing:-1px; font-size:26px; font-family:Georgia; font-weight: 700;'>{title}</span>"

     # Break the text into lines
    lines = break_text_by_length(subtitle, 80) if subtitle != None else []
    
    # Format each line with the specified HTML styling
    formatted_lines = []
    for line in lines:
        formatted_line = f"<span style='font-size:16px; font-family:Tahoma; font-color: #444; display: block; max-width: 30ch;'>{line}</span>"
        formatted_lines.append(formatted_line)
    
    # Join all formatted lines with <br> tags
    subtitle_formatted =  "<br>".join(formatted_lines)

    title_text = title_formatted if subtitle == None else title_formatted + "<br>" + subtitle_formatted
    
    
    # Update layout
    fig.update_layout(
        title={
            'text': title_text,
            'x': 0.05,
            'y': 0.925,
            'xanchor': 'left',
            'yanchor': 'top'
        },
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=width,
        height=height,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        bargap=0.025,
        margin=dict(t=140, b=20, l=60, r=60)
    )
    
    # Update axes styling
    fig.update_xaxes(
        tickangle=0,
        linecolor='#444',
        linewidth=3,
        title_standoff=50
    )
    
    fig.update_yaxes(
        gridcolor='lightgray',
        linewidth=0.5,
        linecolor='black'
    )
    
    return fig

# Example usage functions for common patterns:

def plot_value_counts(series, title="Value Counts", top_n=10, **kwargs):
    """
    Quick function to plot value counts from a pandas Series
    """
    value_counts = series.value_counts().head(top_n)
    return create_bar_chart(
        data=value_counts.to_dict(),
        x_col='x', 
        y_col='y',
        title=title,
        x_label=series.name or "Categories",
        y_label="Count",
        **kwargs
    )

def plot_groupby_result(grouped_data, title="Grouped Data", **kwargs):
    """
    Quick function to plot results from DataFrame.groupby()
    """
    if hasattr(grouped_data, 'to_dict'):
        data = grouped_data.to_dict()
    else:
        data = grouped_data
    
    return create_bar_chart(
        data=data,
        x_col='x',
        y_col='y', 
        title=title,
        **kwargs
    )


def break_text_by_length(text, max_chars):
    """
    Break text into lines at specified character limit without splitting words.
    
    Args:
        text (str): The input text to break
        max_chars (int): Maximum characters per line
    
    Returns:
        list: List of text lines, each within the character limit
    """
    if not text or max_chars <= 0:
        return []
    
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        # Check if adding this word would exceed the limit
        if current_line and len(current_line) + 1 + len(word) > max_chars:
            # Save current line and start new one
            lines.append(current_line)
            current_line = word
        else:
            # Add word to current line
            if current_line:
                current_line += " " + word
            else:
                current_line = word
    
    # Add the last line if it exists
    if current_line:
        lines.append(current_line)
    
    return lines

def write_csv_from_dict(current_dict, columns=['Id'], file_name='output_file'):
    df = pd.DataFrame(list(current_dict.items()), columns=columns)
    df.to_csv(f'/kaggle/working/{file_name}.csv', index=False)


# SAMPLE CHART
time_buckets_chart = create_bar_chart(
    data={
      'Same Day': 53543,
      'Within 1 Week': 35785,
      '1 Week -<br>1 Month': 50274,
      '1-3 Months': 61057,
      '3 Months - <br>1 Year': 116087,
      '1 Year - 3 Years': 99073,
      'More Than<br> 3 Years': 31548
    },
    x_col='x',
    y_col='y',
    title='Time from Registration to First Kernel Creation',
    subtitle="The number of theatrical releases has reached <b>unprecedented levels in recent years</b>, reflecting Hollywood's response to increased demand for content across streaming and traditional platforms.",
    x_label='Time Period',
    y_label='Number of Users',
    color=['#EED0CE','#EED0CE','#EED0CE','#EED0CE','#983D4F','#F6B254','#EED0CE',],
    width=800,
    height=600,
    sort_by=None
)

time_buckets_chart.show()


meta_kaggle_path = kagglehub.dataset_download("kaggle/meta-kaggle")

competitions = pd.read_csv(f"{meta_kaggle_path}/Competitions.csv")

# kernel data
kernels = pd.read_csv(f"{meta_kaggle_path}/Kernels.csv")
kernel_versions = pd.read_csv(f"{meta_kaggle_path}/KernelVersions.csv")
kernel_version_comp_sources = pd.read_csv(f"{meta_kaggle_path}/KernelVersionCompetitionSources.csv")
kernel_version_dataset_sources = pd.read_csv(f"{meta_kaggle_path}/KernelVersionDatasetSources.csv")

# tag data
tags = pd.read_csv(f"{meta_kaggle_path}/Tags.csv")
competition_tags = pd.read_csv(f"{meta_kaggle_path}/CompetitionTags.csv")
forum_topics = pd.read_csv(f"{meta_kaggle_path}/ForumTopics.csv")

# Additional data sources - currently not using this.
kaggle_survey = pd.read_csv("/kaggle/input/kaggle-survey-2022/kaggle_survey_2022_responses.csv")

# common data prep for the next sections
kernels['CreationDate'] = pd.to_datetime(kernels['CreationDate'], errors='coerce')
kernel_versions['CreationDate'] = pd.to_datetime(kernel_versions['CreationDate'], errors='coerce')
competitions['EnabledDate'] = pd.to_datetime(competitions['EnabledDate'], errors='coerce')
competitions['CreationYear'] = competitions['EnabledDate'].dt.year
competitions['DeadlineDate'] = pd.to_datetime(competitions['DeadlineDate'], errors='coerce')
forum_topics['CreationDate'] = pd.to_datetime(forum_topics['CreationDate'], errors='coerce')


# FIRST COMPETITIONS

# Get first kernels for each user
kernels_clean = kernels[kernels['CreationDate'].notna()].copy()
first_kernel_idx = kernels_clean.groupby('AuthorUserId')['CreationDate'].idxmin()
users_first_kernels = kernels_clean.loc[first_kernel_idx].copy()

# Get kernel versions for first kernels and link to competition sources
first_kernels_versions = (kernel_versions[kernel_versions['ScriptId'].isin(users_first_kernels['Id'])]
                         .sort_values('CreationDate')
                         .drop_duplicates('ScriptId', keep='first')
                         .merge(kernel_version_comp_sources[['KernelVersionId','SourceCompetitionId']],
                               left_on='Id', right_on='KernelVersionId', how='inner'))

# Get competition details
comp_data = first_kernels_versions.merge(
    competitions[['Id', 'Slug', 'Title', 'HostSegmentTitle']], 
    left_on='SourceCompetitionId', 
    right_on='Id', 
    how='inner'
)

# Calculate popularity and get top 300
num_comps = 300
comp_popularity = comp_data['Slug'].value_counts().head(num_comps)

# Create final dataframe with details
comp_popularity_details = (pd.DataFrame(comp_popularity)
                          .reset_index()
                          .merge(competitions[['Slug','Title','HostSegmentTitle']], 
                                on='Slug', how='left'))

# Add segment buckets
segment_buckets_mapping = {
    'Getting Started': 'Learning',
    'Playground': 'Learning', 
    'Featured': 'Competitive',
    'Recruitment': 'Recruitment',
    'Analytics': 'Research and Analytics',
    'Research': 'Research and Analytics',
    'Community': 'Community'
}
comp_popularity_details['SegmentBucket'] = comp_popularity_details['HostSegmentTitle'].map(segment_buckets_mapping)

print(f"Generated popularity data for {len(comp_popularity_details)} competitions")
print(f"Total first kernels with competition data: {len(comp_data)}")

# Save results
comp_popularity_details.to_csv('/kaggle/working/comp_popularity_details.csv', index=False)
print("\nTop 10 most popular competitions for first kernels:")
display(comp_popularity_details.head(10))


# DATA PREPARATION SECTION
# Clean kernels data
kernels_clean = kernels[kernels['CreationDate'].notna()].copy()

# Filter competitions with valid dates and reasonable durations
competitions_clean = competitions[
    (competitions['EnabledDate'].notna()) & 
    (competitions['DeadlineDate'].notna()) &
    (competitions['ForumId'].notna())
].copy()

competitions_clean['DurationDays'] = (
    competitions_clean['DeadlineDate'] - competitions_clean['EnabledDate']
).dt.days

competitions_clean = competitions_clean[
    (competitions_clean['DurationDays'] >= 14) & 
    (competitions_clean['DurationDays'] <= 112)
].copy()

# 1. Kernels created
kernels_with_comps = kernels_clean.merge(
    kernel_versions[['Id', 'ScriptId']],
    left_on='Id',
    right_on='ScriptId',
    how='inner'
).merge(
    kernel_version_comp_sources[['KernelVersionId', 'SourceCompetitionId']],
    left_on='Id_y',
    right_on='KernelVersionId',
    how='inner'
).merge(
    competitions_clean[['Id', 'EnabledDate', 'DeadlineDate']],
    left_on='SourceCompetitionId',
    right_on='Id',
    how='inner'
)

kernels_with_comps['WeeksFromStart'] = (
    kernels_with_comps['CreationDate'] - kernels_with_comps['EnabledDate']
).dt.days // 7

kernels_with_comps = kernels_with_comps[
    (kernels_with_comps['WeeksFromStart'] >= 0) & 
    (kernels_with_comps['WeeksFromStart'] <= 20)
]

# 2. Discussion topics created
topics_clean = forum_topics[forum_topics['CreationDate'].notna()].copy()
topics_with_comps = topics_clean.merge(
    competitions_clean[['Id', 'ForumId', 'EnabledDate', 'DeadlineDate']],
    on='ForumId',
    how='inner'
)

topics_with_comps['WeeksFromStart'] = (
    topics_with_comps['CreationDate'] - topics_with_comps['EnabledDate']
).dt.days // 7

topics_with_comps = topics_with_comps[
    (topics_with_comps['WeeksFromStart'] >= 0) & 
    (topics_with_comps['WeeksFromStart'] <= 20)
]

# 3. Notebooks forked
forked_kernels = kernels_clean[kernels_clean['ForkParentKernelVersionId'].notna()].copy()

forked_with_comps = forked_kernels.merge(
    kernel_versions[['Id', 'ScriptId']],
    left_on='Id',
    right_on='ScriptId',
    how='inner'
).merge(
    kernel_version_comp_sources[['KernelVersionId', 'SourceCompetitionId']],
    left_on='Id_y',
    right_on='KernelVersionId',
    how='inner'
).merge(
    competitions_clean[['Id', 'EnabledDate', 'DeadlineDate']],
    left_on='SourceCompetitionId',
    right_on='Id',
    how='inner'
)

forked_with_comps['WeeksFromStart'] = (
    forked_with_comps['CreationDate'] - forked_with_comps['EnabledDate']
).dt.days // 7

forked_with_comps = forked_with_comps[
    (forked_with_comps['WeeksFromStart'] >= 0) & 
    (forked_with_comps['WeeksFromStart'] <= 20)
]


# Kernels created per week

weekly_kernels = kernels_with_comps.groupby('WeeksFromStart').size()
weeks_range = range(0, 21)
weekly_kernels_complete = {week: weekly_kernels.get(week, 0) * 100 / len(kernels_with_comps) for week in weeks_range}

kernels_chart = create_bar_chart(
    data=weekly_kernels_complete,
    x_col='x',
    y_col='y',
    title='Kernels Created by Week After Competition Start',
    x_label='Weeks from Competition Start',
    y_label='Number of Kernels Created',
    valueformat='percentage',
    color= [palette_maroon] + [palette_pink_light]*20,
    width=900,
    height=550,
    sort_by=None
)
kernels_chart.show()

write_csv_from_dict(weekly_kernels_complete, ['week', 'value'], 'comp_activity_kernels')


# Discussion topics created per week
weekly_topics = topics_with_comps.groupby('WeeksFromStart').size()
weekly_topics_complete = {week: weekly_topics.get(week, 0) * 100 / len(topics_with_comps) for week in weeks_range}

topics_chart = create_bar_chart(
    data=weekly_topics_complete,
    x_col='x',
    y_col='y',
    title='Discussion Topics Created by Week After Competition Start',
    x_label='Weeks from Competition Start',
    y_label='Number of Discussion Topics',
    valueformat='percentage',
    color= [palette_maroon] + [palette_pink_light]*20,
    width=900,
    height=550,
    sort_by=None
)
topics_chart.show()

write_csv_from_dict(weekly_topics_complete, ['week', 'value'], 'comp_activity_topics')


# Notebooks forked per week
weekly_forks = forked_with_comps.groupby('WeeksFromStart').size()
weekly_forks_complete = {week: weekly_forks.get(week, 0) * 100 / len(forked_with_comps) for week in weeks_range}

forks_chart = create_bar_chart(
    data=weekly_forks_complete,
    x_col='x',
    y_col='y',
    title='Notebooks Forked by Week After Competition Start',
    x_label='Weeks from Competition Start',
    y_label='Number of Notebooks Forked',
    valueformat='percentage',
    color= [palette_maroon] + [palette_pink_light]*20,
    width=900,
    height=550,
    sort_by=None
)
forks_chart.show()

write_csv_from_dict(weekly_forks_complete, ['week', 'value'], 'comp_activity_forks')


# Data taken from the Anaconda State of DS Report 2020
time_spent = {
    "Data <br> loading": 19,
    "Data <br> cleaning": 26,
    "Data <br> visualisation": 21,
    "Model <br> selection": 11,
    "Model <br> training and scoring": 12,
    "Deploying <br> models": 11,
}

time_spent_chart = create_bar_chart(
    data=time_spent,
    valueformat='percentage',
    x_col='x',
    y_col='y',
    title='Time spent in the Machine Learning process',
    subtitle='In the Anaconda State of DS Report 2020, respondents were asked the following question - "Thinking about your current job, how much of your time is spent in each of the following tasks?"',
    x_label='User Behavior',
    y_label='Number of Users',
    color=[palette_pink_light, palette_maroon] + [palette_pink_light] * 4,
    width=800,
    height=600,
    sort_by=None
)
time_spent_chart.show()

time_spent_df = pd.DataFrame(list(time_spent.items()), columns=['Activity', 'Time_Percentage'])

# write out csv to use elsewhere
time_spent_df.to_csv('/kaggle/working/ds_time_spent.csv', index=False)


# DATA PREPARATION SECTION

# Get latest version of each kernel
kernel_versions_clean = kernel_versions[kernel_versions['CreationDate'].notna()].copy()
latest_versions_idx = kernel_versions_clean.groupby('ScriptId')['CreationDate'].idxmax()
latest_kernel_versions = kernel_versions_clean.loc[latest_versions_idx].copy()
latest_kernel_versions['CreationYear'] = latest_kernel_versions['CreationDate'].dt.year

# Count datasets used per kernel version
dataset_counts_per_version = kernel_version_dataset_sources.groupby('KernelVersionId').size().reset_index(name='DatasetCount')

# Merge and fill missing values
latest_versions_with_datasets = latest_kernel_versions.merge(
    dataset_counts_per_version,
    left_on='Id',
    right_on='KernelVersionId',
    how='left'
)
latest_versions_with_datasets['DatasetCount'] = latest_versions_with_datasets['DatasetCount'].fillna(0).astype(int)

# RESULTS SECTION

# Filter to significant years (≥100 notebooks) and calculate totals
yearly_counts = latest_versions_with_datasets.groupby('CreationYear').size()
yearly_total_datasets = latest_versions_with_datasets.groupby('CreationYear')['DatasetCount'].sum()

# 
total_datasets_chart = create_bar_chart(
    data=yearly_total_datasets.to_dict(),
    x_col='x',
    y_col='y',
    title='Total Number of Datasets Used in Notebooks by Year',
    subtitle='The number of datasets are increasing with each passing year.',
    x_label='Year',
    y_label='Total Datasets Used',
    color=[palette_maroon]*10 + [palette_pink_light],
    width=800,
    height=550,
    sort_by=None
)
total_datasets_chart.show()

# Create final dataframe and save
yearly_total_datasets_df = pd.DataFrame(list(yearly_total_datasets.items()), columns=['Year', 'Count'])
yearly_total_datasets_df.to_csv('/kaggle/working/dataset_accesses_by_year.csv', index=False)


data_scientists = kaggle_survey[kaggle_survey['Q23'] == 'Data Scientist']

ml_methods_cols = [col for col in data_scientists.columns if 'Q18' in col]
data_scientists[ml_methods_cols]

ml_methods_responses = data_scientists[ml_methods_cols]

# If each column has only one unique non-null value, get that value
new_column_names = {}
for col in ml_methods_responses.columns:
    unique_vals = ml_methods_responses[col].dropna().unique()
    if len(unique_vals) == 1:
        new_column_names[col] = unique_vals[0]

# Rename columns
ml_methods_responses = ml_methods_responses.rename(columns=new_column_names)
ml_knowledge_results = ml_methods_responses.notnull().sum(axis=0).sort_values(ascending=False) / ml_methods_responses.shape[0]

ml_knowledge_results_df = pd.DataFrame(list(ml_knowledge_results.items()), columns=['Method', 'PercentageUsed'])


print('When asked about their familiarity with different machine learning techniques, data scientists said they had used: ')
display(ml_knowledge_results_df)

# write out csv to use elsewhere
ml_knowledge_results_df.to_csv('/kaggle/working/ml_methods_usage.csv', index=False)


# DATA PREPARATION SECTION

# Define tag buckets
tag_buckets = {
    'Tabular Data': ['tabular'],
    'Computer Vision': ['image', 'computer vision'],
    'Natural<br>Language<br>Processing': ['text', 'nlp'],
    'Time Series': ['time series analysis'],
    'Binary<br>Classification': ['binary classification'],
    'Multiclass<br>Classification': ['multiclass classification'],
    'Regression': ['regression'],
    'Other': ['beginner']
}

# Create reverse mapping
tag_to_bucket = {}
for bucket, tag_list in tag_buckets.items():
    for tag in tag_list:
        tag_to_bucket[tag] = bucket

# Link competitions to their tags
comp_tags_with_names = competition_tags.merge(
    tags[['Id', 'Name']],
    left_on='TagId',
    right_on='Id',
    how='left'
)

# Apply bucketing to tags
comp_tags_with_names['TagBucket'] = comp_tags_with_names['Name'].map(tag_to_bucket)
comp_tags_bucketed = comp_tags_with_names[comp_tags_with_names['TagBucket'].notna()].copy()

# Link back to competition details
comp_tags_complete = comp_tags_bucketed.merge(
    competitions[['Id', 'Title', 'EnabledDate', 'CreationYear', 'RewardType', 'TotalCompetitors']],
    left_on='CompetitionId',
    right_on='Id',
    how='left'
)

# Filter to valid years
current_year = 2025
comp_tags_filtered = comp_tags_complete[
    (comp_tags_complete['CreationYear'] >= 2010) & 
    (comp_tags_complete['CreationYear'] <= current_year)
].copy()

# Most common tag buckets overall

bucket_counts = comp_tags_filtered['TagBucket'].value_counts()

bucket_counts_chart = create_bar_chart(
    data=bucket_counts.to_dict(),
    x_col='x',
    y_col='y',
    title='Most Common Competition Tag Buckets',
    subtitle='Charting the most common competition types that Kaggle competitions are tagged with',
    x_label='ML Problem Type',
    y_label='Number of Competition Tags',
    color=palette_maroon,
    width=800,
    height=570,
    sort_by='y',
    ascending=False
)
bucket_counts_chart.show()

# Create final dataframe and save
bucket_counts_df = pd.DataFrame(list(bucket_counts.items()), columns=['Technique', 'Count'])
bucket_counts_df.to_csv('/kaggle/working/tag_bucket_counts.csv', index=False)

