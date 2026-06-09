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


# IMPORTS
meta_kaggle_path = kagglehub.dataset_download("kaggle/meta-kaggle")

users = pd.read_csv(f"{meta_kaggle_path}/Users.csv")

# competitions and datasets
competitions = pd.read_csv(f"{meta_kaggle_path}/Competitions.csv")
datasets = pd.read_csv(f"{meta_kaggle_path}/Datasets.csv")

# forum activity
forums = pd.read_csv(f"{meta_kaggle_path}/Forums.csv")
forum_topics = pd.read_csv(f"{meta_kaggle_path}/ForumTopics.csv")
forum_messages = pd.read_csv(f"{meta_kaggle_path}/ForumMessages.csv")


# Kernel related data
kernels = pd.read_csv(f"{meta_kaggle_path}/Kernels.csv")
kernel_versions = pd.read_csv(f"{meta_kaggle_path}/KernelVersions.csv")
kernel_version_comp_sources = pd.read_csv(f"{meta_kaggle_path}/KernelVersionCompetitionSources.csv")
kernel_version_dataset_sources = pd.read_csv(f"{meta_kaggle_path}/KernelVersionDatasetSources.csv")
kernel_languages = pd.read_csv(f"{meta_kaggle_path}/KernelLanguages.csv")

# kernel_votes = pd.read_csv(f"{meta_kaggle_path}/KernelVotes.csv") # commenting out since this file is empty at the time of re-running this.

# tag data
tags = pd.read_csv(f"{meta_kaggle_path}/Tags.csv")
competition_tags = pd.read_csv(f"{meta_kaggle_path}/CompetitionTags.csv")

teams = pd.read_csv(f"{meta_kaggle_path}/Teams.csv")
team_memberships = pd.read_csv(f"{meta_kaggle_path}/TeamMemberships.csv")
submissions = pd.read_csv(f"{meta_kaggle_path}/Submissions.csv")

# Additional data sources - currently not using this.
kaggle_survey = pd.read_csv("/kaggle/input/kaggle-survey-2022/kaggle_survey_2022_responses.csv")


# common data prep for the next sections
users['RegisterDate'] = pd.to_datetime(users['RegisterDate'], errors='coerce')
kernels['CreationDate'] = pd.to_datetime(kernels['CreationDate'], errors='coerce')
competitions['EnabledDate'] = pd.to_datetime(competitions['EnabledDate'], errors='coerce')
competitions['DeadlineDate'] = pd.to_datetime(competitions['DeadlineDate'], errors='coerce')
forum_topics['CreationDate'] = pd.to_datetime(forum_topics['CreationDate'], errors='coerce')
forum_messages['PostDate'] = pd.to_datetime(forum_messages['PostDate'], errors='coerce')
kernel_versions['CreationDate'] = pd.to_datetime(kernel_versions['CreationDate'], errors='coerce')
team_memberships['RequestDate'] = pd.to_datetime(team_memberships['RequestDate'], errors='coerce')
submissions['SubmissionDate'] = pd.to_datetime(submissions['SubmissionDate'], errors='coerce')


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


# Analysing notebook creation habits of new users.

# --- DATA PREP ---

# Remove users and kernels with missing dates
users_clean = users[users['RegisterDate'].notna()].copy()
kernels_clean = kernels[kernels['CreationDate'].notna()].copy()

total_registered_users = users_clean.shape[0]

users_first_kernel = kernels_clean.groupby('AuthorUserId')['CreationDate'].min().reset_index()

users_who_created_kernels = users_first_kernel.shape[0]
users_who_never_created_kernels = total_registered_users - users_who_created_kernels
percentage_never_created = (users_who_never_created_kernels / total_registered_users) * 100


users_with_kernels = users_clean.merge(users_first_kernel, left_on='Id', right_on='AuthorUserId', how='inner')

# Get the number of days it took them to create their first notebook
users_with_kernels['DaysToFirstKernel'] = (
    users_with_kernels['CreationDate'] - users_with_kernels['RegisterDate']
).dt.days

# --- ANALYSIS AND FINDINGS ---

print(f"Total number of registered users: {total_registered_users:,}")

print("\n\nNote: The following analysis will focus ONLY on users who created at least one kernel")
print("to understand engagement patterns among active notebook creators.\n")

print(f"Users who NEVER created a public kernel: {users_who_never_created_kernels:,}")
print(f"Percentage who never created a public kernel: {percentage_never_created:.1f}%")
print(f"Percentage who created at least one public kernel: {100 - percentage_never_created:.1f}%")

print("\nSummary Statistics: Days to First Kernel")
print(users_with_kernels['DaysToFirstKernel'].describe())


# Kernel creation overview

kernel_creation_overview = {
    'Created Kernels': users_with_kernels.shape[0] * 100 / total_registered_users,
    'Never Created Kernels': users_who_never_created_kernels * 100 / total_registered_users
}

creation_overview_chart = create_bar_chart(
    data=kernel_creation_overview,
    valueformat='percentage',
    x_col='x',
    y_col='y',
    title='Kernel Creation Participation Among All Users',
    subtitle="Around <b>1.8% of Kaggle's userbase</b> drives all of the notebook sharing activity on the platform",
    x_label='User Behavior',
    y_label='Number of Users',
    color=[palette_pink_light, palette_maroon],
    width=800,
    height=600,
    sort_by=None
)
creation_overview_chart.show()


# Distribution of notebook creation over time
time_buckets = [
    (0, 1, "Same Day"),
    (1, 7, "Within 1 Week"),
    (7, 30, "1 Week - 1 Month"),
    (30, 90, "1-3 Months"),
    (90, 365, "3 Months - 1 Year"),
    (365, 365 * 3, "1 Year - 3 Years"),
    (365 * 3, float('inf'), "More Than 3 Years")
]

bucket_counts = {}
bucket_percentages = {}
for min_days, max_days, label in time_buckets:
    count = len(users_with_kernels[
        (users_with_kernels['DaysToFirstKernel'] >= min_days) & 
        (users_with_kernels['DaysToFirstKernel'] < max_days)
    ])
    bucket_counts[label] = count
    bucket_percentages[label] = count / users_with_kernels.shape[0] * 100

time_buckets_chart = create_bar_chart(
    data=bucket_percentages,
    valueformat='percentage',
    x_col='x',
    y_col='y',
    title='Time from Registration to First Public Notebook',
    subtitle='While many users will publish a notebook on their <b>first day of</b> <b>registering</b>, many only feel comfortable publishing their notebooks <b>several months to a few years</b> into their Kaggle journey.',
    x_label='Time Period',
    y_label='Number of Users',
    color=[ palette_yellow,'#EED0CE','#EED0CE','#EED0CE',palette_maroon ,palette_maroon ,'#EED0CE',],
    width=900,
    height=600,
    sort_by=None
)

time_buckets_chart.show()


# Recent vs Older Users comparison
recent_cutoff = users_with_kernels['RegisterDate'].quantile(0.75)  # Top 25% most recent
recent_users = users_with_kernels[users_with_kernels['RegisterDate'] >= recent_cutoff]
older_users = users_with_kernels[users_with_kernels['RegisterDate'] < recent_cutoff]

user_cohort_comparison = {
    f'Users registered \nbefore {recent_cutoff.date()}': older_users['DaysToFirstKernel'].median(),
    f'After {recent_cutoff.date()}': recent_users['DaysToFirstKernel'].median(),
}

cohort_chart = create_bar_chart(
    data=user_cohort_comparison,
    x_col='x',
    y_col='y',
    title='Recent vs Older Users',
    subtitle='When pitting the most recent 25% of Kaggle joiners against the previous sections of the user base, we see how the time it takes them to publish their first kernel significantly drops',
    x_label='User Cohort',
    y_label='Median Days to first public notebook',
    color=[palette_pink_light, palette_maroon],
    width=750,
    height=550,
    sort_by=None
)
cohort_chart.show()


# Get each user's first kernel
kernels_clean = kernels[kernels['CreationDate'].notna()].copy()
first_kernel_idx = kernels_clean.groupby('AuthorUserId')['CreationDate'].idxmin()
users_first_kernels = kernels_clean.loc[first_kernel_idx].copy()

# Get kernel versions for first kernels
first_kernels_versions = (kernel_versions[kernel_versions['ScriptId'].isin(users_first_kernels['Id'])]
                         .sort_values('CreationDate')
                         .drop_duplicates('ScriptId', keep='first')
                         .copy())

# Keep only specific columns
first_kernels_versions = first_kernels_versions[['Id', 'ScriptId', 'ParentScriptVersionId', 'ScriptLanguageId',
       'AuthorUserId', 'CreationDate', 'VersionNumber', 'Title',
       'EvaluationDate', 'TotalLines']]

# Link to competition sources
first_kernels_with_sources = first_kernels_versions.merge(
    kernel_version_comp_sources[['KernelVersionId','SourceCompetitionId']],
    left_on='Id',
    right_on='KernelVersionId',
    how='left'
)

# Add competition details
first_kernels_complete = first_kernels_with_sources.merge(
    competitions[['Id', 'Title', 'Slug', 'HostSegmentTitle', 'RewardType', 'TotalCompetitors']],
    left_on='SourceCompetitionId',
    right_on='Id',
    how='left',
    suffixes=('', '_comp')
)

first_kernels_complete.head()

# Link back to original users_first_kernels dataframe
final_df = first_kernels_complete.merge(
    users_first_kernels[['Id', 'AuthorUserId', 'CreationDate', 'TotalViews', 'TotalVotes', 'CurrentUrlSlug']],
    left_on='ScriptId',
    right_on='Id',
    how='left',
    suffixes=('_version', '_kernel')
)

# Clean and select relevant columns
users_first_kernels_df = final_df[[
    'Id_version',
    'ScriptId', 
    'ParentScriptVersionId', 
    'ScriptLanguageId',
    'AuthorUserId_version', 
    'CreationDate_version', 
    'SourceCompetitionId',
    'Title', 
    'TotalLines', 
    'Title_comp', 
    'Slug', 
    'HostSegmentTitle',
    'RewardType',
    'TotalCompetitors', 
    'CreationDate_kernel', 
    'TotalViews', 
    'TotalVotes', 
    'CurrentUrlSlug'
]].rename(columns={
    'AuthorUserId_version': 'AuthorUserId', 
    'ScriptId': 'KernelId',
    'CreationDate_kernel': 'KernelCreationDate',
    'Title_version': 'KernelTitle',
    'Title': 'CompetitionTitle'
})

print(f"Generated DataFrame with {len(users_first_kernels_df):,} records")
print(f"Unique users: {users_first_kernels_df['AuthorUserId'].nunique():,}")
print(f"Users with competition data: {users_first_kernels_df['SourceCompetitionId'].notna().sum():,}")

# Display sample
print("\nSample data:")
display(users_first_kernels_df.head())


total_users = users_first_kernels_df['AuthorUserId'].nunique()
users_with_comp = users_first_kernels_df['SourceCompetitionId'].notna().sum()
users_without_comp = total_users - users_with_comp

overview_data = {
    'Uses Competition Data': users_with_comp * 100 / total_users,
    'Uses External/General Data': users_without_comp * 100 / total_users
}
overview_chart = create_bar_chart(
    data=overview_data,
    valueformat='percentage',
    x_col='x',
    y_col='y',
    title='Data Source Types in Users\' First Kernels',
    subtitle='Almost three-fourths of new users write their first lines of code in notebooks <b>outside of active competitions</b> which do not provide a dataset',
    x_label='Data Source Type',
    y_label='Number of Users',
    color=[palette_maroon, palette_pink_light],
    width=800,
    height=550
)
overview_chart.show()


# Filter to only users with competition data
comp_data = users_first_kernels_df[users_first_kernels_df['SourceCompetitionId'].notna()].copy()

# Most popular competitions
comp_popularity = comp_data['Slug'].value_counts().head(10)

print('Most popular competitions: ')
comp_popularity_details = pd.DataFrame(comp_popularity).reset_index().merge(competitions[['Slug','Title','HostSegmentTitle','RewardType']], on='Slug', how='left')

# Truncate long competition titles for better display
truncated_titles = {}
for title, count in comp_popularity.items():
    short_title = title[:12] + "..." if len(title) > 10 else title
    truncated_titles[short_title] = count

comp_chart = create_bar_chart(
    data=truncated_titles,
    x_col='x',
    y_col='y',
    title='Top 10 Most Popular Competitions for First Kernels',
    x_label='Competition',
    y_label='Number of Users',
    color=[palette_maroon, palette_maroon, palette_maroon, palette_maroon, palette_maroon, palette_maroon, palette_pink_light, palette_pink_light, palette_pink_light, palette_maroon],
    width=1000,
    height=600
)
comp_chart.show()


comp_popularity_details['count'].describe()


num_comps = 300

# Most popular competitions
comp_popularity = comp_data['Slug'].value_counts().head(num_comps)

print('Most popular competitions: ')
comp_popularity_details = pd.DataFrame(comp_popularity).reset_index().merge(competitions[['Slug','Title','HostSegmentTitle','RewardType']], on='Slug', how='left')
display(comp_popularity_details.groupby('RewardType')['count'].sum().sort_values(ascending=False))

segment_buckets_mapping = {
    'Getting Started': 'Learning',
    'Playground': 'Learning',
    'Featured': 'Competitve',
    'Recruitment': 'Recruitment',
    'Analytics': 'Research and Analytics',
    'Research': 'Research and Analytics',
    'Community': 'Community'
}

comp_popularity_details['SegmentBucket'] = comp_popularity_details['HostSegmentTitle'].map(segment_buckets_mapping)
display(comp_popularity_details['count'].describe())

# write out csv to use elsewhere
comp_popularity_details.to_csv('/kaggle/working/comp_popularity_details.csv', index=False)


# Create size categories
comp_data['SizeCategory'] = pd.cut(
    comp_data['TotalCompetitors'], 
    bins=[0, 100, 500, 1000, 5000, float('inf')], 
    labels=['Very Small\n(≤100)', 'Small\n(101-500)', 'Medium\n(501-1000)', 'Large\n(1001-5000)', 'Very Large\n(>5000)'],
    include_lowest=True
)

size_analysis = comp_data['SizeCategory'].value_counts()
size_chart = create_bar_chart(
    data=size_analysis.to_dict(),
    x_col='x',
    y_col='y',
    title='Competition Size Preferences for First Kernels',
    subtitle='New users tended to prefer larger, more established competitions for their first notebooks rather than smaller community run competitions.',
    x_label='Competition Size Category',
    y_label='Number of Users',
    color= palette_pink_light,
    width=800,
    height=550
)
size_chart.show()


print("--- ANALYZING DISCUSSION ACTIVITY TIMELINE AFTER COMPETITION START ---")

# Filter competitions with valid dates
competitions_clean = competitions[
    (competitions['EnabledDate'].notna()) 
].copy()

print(f"Competitions with valid start dates and forums: {len(competitions_clean):,}")

# Calculate competition duration for context
competitions_clean['DurationDays'] = (
    competitions_clean['DeadlineDate'] - competitions_clean['EnabledDate']
).dt.days
    
# Filter to reasonable competition durations (>1 day)
competitions_clean = competitions_clean[
    (competitions_clean['DurationDays'] >= 1)
].copy()

print(f"Competitions with reasonable durations: {len(competitions_clean):,}")

# So there seems to be an issue where about half of all competitions don't have a ForumId value (currently is NaN)
# This includes some pretty prominent competitions like the Titanic competition. By manually going into a couple of these
# competitons and searching out specific discussions topics in the data, I found that by matching on competition title(yes, not ideal)
# I can use the Id column in Forums.csv to get the ForumId for a few competitions.

# Admittedly unsure how this all works because even still competitions like titanic still have no forum topics in the data
# even searching for titanic forum topics don't turn up results the way other competitions do:
# eg. forum_topics[forum_topics.Title.str.contains('Some writeups', case=False, na=False)] links to: https://www.kaggle.com/competitions/openai-to-z-challenge/discussion/589010
# but forum_topics[forum_topics.Title.str.contains('Titanic Survival Prediction Using XGBoost — Solution Summary', case=False, na=False)] doesn't find: https://www.kaggle.com/competitions/titanic/discussion/586706

# Hopefully this section can still serve to give a broad sense of what competitions look like on Kaggle

# Fill missing ForumId values by matching titles with Forums.csv
missing_forum_ids = competitions_clean['ForumId'].isna()
print(f"Competitions with missing ForumId: {missing_forum_ids.sum()}")

if missing_forum_ids.sum() > 0:
   # Create a mapping from forum title to forum Id
   forum_title_to_id = forums.set_index('Title')['Id'].to_dict()
   
   # Fill missing ForumId values by matching titles
   for idx, row in competitions_clean[missing_forum_ids].iterrows():
       comp_title = row['Title']
       if comp_title in forum_title_to_id:
           competitions_clean.loc[idx, 'ForumId'] = forum_title_to_id[comp_title]
   
   print(f"ForumId values filled. Remaining missing: {competitions_clean['ForumId'].isna().sum()}")

# Now filter to competitions with valid ForumId
competitions_clean = competitions_clean[competitions_clean['ForumId'].notna()].copy()
print(f"Competitions with valid ForumId: {len(competitions_clean):,}")

# Link forum topics to competitions
print("\n--- Linking forum topics to competitions ---")
comp_forum_topics = forum_topics.merge(
    competitions_clean[['Id', 'Slug', 'ForumId', 'Title', 'EnabledDate', 'DeadlineDate', 'DurationDays', 'RewardType', 'TotalCompetitors', 'HostSegmentTitle']],
    left_on='ForumId',
    right_on='ForumId',
    how='inner'
)

print(f"Forum topics linked to competitions: {len(comp_forum_topics):,}")

# Calculate days from competition start to topic creation
comp_forum_topics['DaysFromCompStart'] = (
    comp_forum_topics['CreationDate'] - comp_forum_topics['EnabledDate']
).dt.days

# Filter topics created after competition start (or slightly before for prep discussions)
comp_topics_valid = comp_forum_topics[
    (comp_forum_topics['DaysFromCompStart'] >= -14) &  # Allow 1 week before start
    (comp_forum_topics['DaysFromCompStart'] <= comp_forum_topics['DurationDays'] + 60)  # Allow 2 months after end
].copy()

print(f"Valid competition forum topics: {len(comp_topics_valid):,}")

# Link forum messages to competition topics
print("\n--- Linking forum messages to competition topics ---")
comp_messages = forum_messages.merge(
    comp_topics_valid[['Id_x', 'Id_y', 'Slug', 'Title_y', 'EnabledDate', 'DurationDays', 'RewardType', 'TotalCompetitors']],
    left_on='ForumTopicId',
    right_on='Id_x',
    how='inner',
    suffixes=('_message', '_comp')
)

# Calculate days from competition start to message post
comp_messages['DaysFromCompStart'] = (
    comp_messages['PostDate'] - comp_messages['EnabledDate']
).dt.days

# Filter messages within reasonable timeframe
comp_messages_valid = comp_messages[
    (comp_messages['DaysFromCompStart'] >= -14) &
    (comp_messages['DaysFromCompStart'] <= comp_messages['DurationDays'] + 60)
].copy()

print(f"Valid competition forum messages: {len(comp_messages_valid):,}")


# Discussion activity by time periods from competition start

# Define time periods relative to competition start
time_periods = [
    (-7, 0, "Pre-Competition<br>(1 Week Before)"),
    (0, 1, "Launch Day"),
    (1, 7, "First Week"),
    (7, 30, "Weeks 2-4"),
    (30, 60, "Month 2"),
    (60, 90, "Month 3"),
    (90, float('inf'), "Late Stage<br>(3+ Months)")
]

# Count topics by time period
topic_period_counts = {}
for min_days, max_days, label in time_periods:
    if max_days == float('inf'):
        count = len(comp_topics_valid[comp_topics_valid['DaysFromCompStart'] >= min_days])
    else:
        count = len(comp_topics_valid[
            (comp_topics_valid['DaysFromCompStart'] >= min_days) & 
            (comp_topics_valid['DaysFromCompStart'] < max_days)
        ])
    topic_period_counts[label] = count * 100 / comp_topics_valid.shape[0]

topics_timeline_chart = create_bar_chart(
    data=topic_period_counts,
    valueformat='percentage',
    x_col='x',
    y_col='y',
    title='Discussion Activity after Competition Start',
    subtitle='When looking when discussion topics activity after a competition starts, we notice that almost half of all topics were created <b>between 2 weeks to 2</b> <b>months</b> of the competition start',
    x_label='Time Period',
    y_label='Number of Topics Created',
    color=[palette_pink_light, palette_pink_light, palette_pink_light, palette_maroon, palette_maroon, palette_pink_light, palette_pink_light],
    sort_by=None,
    width=900,
    height=550
)
topics_timeline_chart.show()


# Count messages by time period
message_period_counts = {}
for min_days, max_days, label in time_periods:
    if max_days == float('inf'):
        count = len(comp_messages_valid[comp_messages_valid['DaysFromCompStart'] >= min_days])
    else:
        count = len(comp_messages_valid[
            (comp_messages_valid['DaysFromCompStart'] >= min_days) & 
            (comp_messages_valid['DaysFromCompStart'] < max_days)
        ])
    message_period_counts[label] = count * 100 / comp_messages_valid.shape[0]

messages_timeline_chart = create_bar_chart(
    data=message_period_counts,
    valueformat='percentage',
    x_col='x',
    y_col='y',
    title='Discussion Messages after Competition Start',
    subtitle='Similar to Topic creation, reach highs after 2 weeks into the competitions and stay that way till the 3 month mark',
    x_label='Time Period',
    y_label='Number of Messages Posted',
    color=[palette_pink_light, palette_pink_light, palette_pink_light, palette_maroon, palette_maroon, palette_maroon, palette_pink_light],
    sort_by=None,
    width=1000,
    height=550
)
messages_timeline_chart.show()


# Average discussion activity per competition 'HostSegmentTitle'

# Calculate average topics and messages per competition
comp_discussion_stats = comp_topics_valid.groupby('Id_y').agg({  
    'Id_x': 'count',  
    'TotalMessages': 'first', 
    'RewardType': 'first',
    'TotalCompetitors': 'first'
}).rename(columns={'Id_x': 'TopicsCount'})

# Calculate total messages per competition from our message data
comp_message_counts = comp_messages_valid.groupby('Id_y').size().reset_index(name='MessagesCount') 
comp_discussion_stats = comp_discussion_stats.merge(
    comp_message_counts, 
    left_index=True, 
    right_on='Id_y', 
    how='left'
)
comp_discussion_stats['MessagesCount'] = comp_discussion_stats['MessagesCount'].fillna(0)

# Average by reward type
avg_discussion_by_reward = comp_discussion_stats.groupby('RewardType').agg({
    'TopicsCount': 'mean',
    'MessagesCount': 'mean'
}).round(1)

avg_topics_chart = create_bar_chart(
    data=avg_discussion_by_reward['TopicsCount'].to_dict(),
    x_col='x',
    y_col='y',
    title='Average Discussion Topics per Competition by Reward Type',
    subtitle='From',
    x_label='Reward Type',
    y_label='Average Topics per Competition',
    color= [palette_maroon] + [palette_pink_light] * 8,
    width=900,
    height=550
)
avg_topics_chart.show()


# Average discussion activity per competition 'HostSegmentTitle'

# Calculate average topics and messages per competition
comp_discussion_stats = comp_topics_valid.groupby('Id_y').agg({  
    'Id_x': 'count',  
    'TotalMessages': 'first', 
    'HostSegmentTitle': 'first',
    'TotalCompetitors': 'first'
}).rename(columns={'Id_x': 'TopicsCount'})

# Calculate total messages per competition from our message data
comp_message_counts = comp_messages_valid.groupby('Id_y').size().reset_index(name='MessagesCount') 
comp_discussion_stats = comp_discussion_stats.merge(
    comp_message_counts, 
    left_index=True, 
    right_on='Id_y', 
    how='left'
)
comp_discussion_stats['MessagesCount'] = comp_discussion_stats['MessagesCount'].fillna(0)

# Average by reward type
avg_discussion_by_reward = comp_discussion_stats.groupby('HostSegmentTitle').agg({
    'TopicsCount': 'mean',
    'MessagesCount': 'mean'
}).round(1)

avg_topics_chart = create_bar_chart(
    data=avg_discussion_by_reward['TopicsCount'].to_dict(),
    x_col='x',
    y_col='y',
    title='Average Discussion Topics per Competition by Competition Type',
    x_label='Reward Type',
    y_label='Average Topics per Competition',
    color= [palette_maroon] + [palette_pink_light] * 8,
    width=900,
    height=550
)
avg_topics_chart.show()


# DATA PREPARATION SECTION

# Identify forked notebooks
print("\n--- Identifying forked notebooks ---")
forked_kernels = kernels[kernels['ForkParentKernelVersionId'].notna()].copy()
print(f"Total forked kernels: {len(forked_kernels):,}")

# Clean date data
forked_kernels_clean = forked_kernels[forked_kernels['CreationDate'].notna()].copy()
print(f"Forked kernels with valid dates: {len(forked_kernels_clean):,}")

# Add creation year for analysis
forked_kernels_clean['CreationYear'] = forked_kernels_clean['CreationDate'].dt.year

print(f"Forked kernels in valid year range: {len(forked_kernels_clean):,}")

# Link forked kernels to their parent kernel versions
forked_with_parents = forked_kernels_clean.merge(
    kernel_versions[['Id', 'ScriptId', 'Title', 'AuthorUserId', 'CreationDate']],
    left_on='ForkParentKernelVersionId',
    right_on='Id',
    how='left',
    suffixes=('_fork', '_parent_version')
)

# Link to parent kernel details
forked_with_parent_kernels = forked_with_parents.merge(
    kernels[['Id', 'TotalViews', 'TotalVotes', 'CreationDate']],
    left_on='ScriptId',
    right_on='Id',
    how='left',
    suffixes=('', '_parent_kernel')
)

print(f"Forks linked to parent information: {len(forked_with_parent_kernels):,}")

# Calculate all kernels by year for fork rate analysis
print("\n--- Calculating total kernels by year for fork rates ---")
all_kernels_clean = kernels[kernels['CreationDate'].notna()].copy()
all_kernels_clean['CreationYear'] = all_kernels_clean['CreationDate'].dt.year

yearly_total_kernels = all_kernels_clean.groupby('CreationYear').size()
yearly_forked_kernels = forked_kernels_clean.groupby('CreationYear').size()

print(f"Total kernels by year calculated for {len(yearly_total_kernels)} years")


years_list = yearly_total_kernels.index

# Total forks created by year
yearly_forks = yearly_forked_kernels.reindex(years_list, fill_value=0)

total_forks_chart = create_bar_chart(
    data=yearly_forks.to_dict(),
    x_col='x',
    y_col='y',
    title='Total Number of Notebook Forks Created by Year',
    subtitle='The trend of forked notebooks is steadily increasing year after year. Since the start of the data collection in 2015, the <b>number of forks created each year </b><b>has seen a 17x increase</b>!',
    x_label='Year',
    y_label='Number of Forks Created',
    color=[palette_pink_light]*9 + [palette_maroon]*2,
    width=800,
    height=550,
    sort_by=None
)
total_forks_chart.show()


# Fork rate (percentage of kernels that are forks)

fork_rates = {}
for year in years_list:
    total_kernels = yearly_total_kernels[year]
    forks = yearly_forked_kernels.get(year, 0)
    fork_rate = (forks / total_kernels) * 100
    fork_rates[year] = round(fork_rate, 1)

fork_rate_chart = create_bar_chart(
    data=fork_rates,
    valueformat='percentage',
    x_col='x',
    y_col='y',
    title='Fork Rate over the years',
    subtitle='The percentage of new kernels that are forks by year. Even though the number of forks are increasing, we are also seeing a larger number of original notebooks being created in recent years.',
    x_label='Year',
    y_label='Fork Rate (%)',
    color=[palette_pink_light]*8 + [palette_maroon]*3,
    width=900,
    height=550,
    sort_by=None
)
fork_rate_chart.show()


# 3. Most forked notebooks analysis
print(f"\n--- MOST FORKED NOTEBOOKS ---")

# Count forks per parent kernel
most_forked = forked_with_parent_kernels.groupby(['ScriptId', 'Title']).size().reset_index(name='ForkCount')
most_forked = most_forked.sort_values('ForkCount', ascending=False).head(15)

display(most_forked)


# 5. Fork popularity vs original popularity

# Average metrics for original vs forked notebooks
original_kernels = all_kernels_clean[all_kernels_clean['ForkParentKernelVersionId'].isna()]
forked_kernels_metrics = forked_kernels_clean

# Calculate averages
comparison_data = {
    'Original Notebooks': original_kernels['TotalViews'].mean().round(2),
    'Forked Notebooks': forked_kernels_metrics['TotalViews'].mean().round(2)
}

comparison_chart = create_bar_chart(
    data=comparison_data,
    x_col='x',
    y_col='y',
    title='Average Views: Original vs Forked Notebooks',
    subtitle='Original notebooks tend to receive more engagement when compared with their forked counterparts.',
    x_label='Notebook Type',
    y_label='Average Views',
    color=[palette_maroon, palette_pink_light],
    width=800,
    height=550,
    sort_by=None
)
comparison_chart.show()


# Datasets analysis

# Get latest version of each kernel
print("\n--- Finding latest version of each kernel ---")
kernel_versions_clean = kernel_versions[kernel_versions['CreationDate'].notna()].copy()

# Get latest version by creation date for each kernel
latest_versions_idx = kernel_versions_clean.groupby('ScriptId')['CreationDate'].idxmax()
latest_kernel_versions = kernel_versions_clean.loc[latest_versions_idx].copy()

print(f"Latest kernel versions identified: {len(latest_kernel_versions):,}")

# Add creation year for trend analysis
latest_kernel_versions['CreationYear'] = latest_kernel_versions['CreationDate'].dt.year

print(f"Latest versions with valid years (start-date): {len(latest_kernel_versions):,}")

# Count datasets used per latest kernel version
print("\n--- Counting datasets per kernel version ---")
dataset_counts_per_version = kernel_version_dataset_sources.groupby('KernelVersionId').size().reset_index(name='DatasetCount')

print(f"Kernel versions with dataset counts: {len(dataset_counts_per_version):,}")

# Merge latest versions with dataset counts
latest_versions_with_datasets = latest_kernel_versions.merge(
    dataset_counts_per_version,
    left_on='Id',
    right_on='KernelVersionId',
    how='left'
)

# Fill NaN values with 0 (kernels that don't use any datasets)
latest_versions_with_datasets['DatasetCount'] = latest_versions_with_datasets['DatasetCount'].fillna(0).astype(int)

print(f"Latest versions with dataset usage data: {len(latest_versions_with_datasets):,}")

# Add kernel
latest_versions_complete = latest_versions_with_datasets.merge(
    kernels[['Id', 'TotalViews', 'TotalVotes', 'AuthorUserId']],
    left_on='ScriptId',
    right_on='Id',
    how='left',
    suffixes=('_version', '_kernel')
)

print(f"Complete dataset with kernel metadata: {len(latest_versions_complete):,}")




# Calculate average datasets per notebook by year
yearly_avg_datasets = latest_versions_complete.groupby('CreationYear')['DatasetCount'].mean().round(2)

# Filter to years with sufficient data (at least 100 notebooks)
yearly_counts = latest_versions_complete.groupby('CreationYear').size()
significant_years = yearly_counts[yearly_counts >= 100].index
yearly_avg_filtered = yearly_avg_datasets[yearly_avg_datasets.index.isin(significant_years)]

# Total datasets used by year
yearly_total_datasets = latest_versions_complete[latest_versions_complete['CreationYear'].isin(significant_years)].groupby('CreationYear')['DatasetCount'].sum()

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

yearly_total_datasets_df = pd.DataFrame(list(yearly_total_datasets.items()), columns=['Year', 'Count'])
yearly_total_datasets_df.to_csv('/kaggle/working/dataset_accesses_by_year.csv', index=False)


# Overall distribution of dataset usage

# Create dataset usage categories
def categorize_dataset_usage(count):
    if count == 0:
        return "No Datasets"
    elif count == 1:
        return "1 Dataset"
    elif count == 2:
        return "2 Datasets"
    elif count == 3:
        return "3 Datasets"
    elif count <= 5:
        return "4-5 Datasets"
    else:
        return "6+ Datasets"

latest_versions_complete['DatasetCategory'] = latest_versions_complete['DatasetCount'].apply(categorize_dataset_usage)

# Overall distribution
overall_distribution = latest_versions_complete['DatasetCategory'].value_counts()

# Order categories logically
category_order = ["No Datasets", "1 Dataset", "2 Datasets", "3 Datasets", "4-5 Datasets", "6+ Datasets"]
overall_ordered = {cat: overall_distribution.get(cat, 0) for cat in category_order}

overall_chart = create_bar_chart(
    data=overall_ordered,
    x_col='x',
    y_col='y',
    title='Distribution of Dataset Usage in Latest Notebook Versions',
    subtitle='When looking at all created notebooks, we notice that the vast majority use either none or just a single dataset.',
    x_label='Number of Datasets Used',
    y_label='Number of Notebooks',
    color=[palette_maroon]*2 + [palette_pink_light]*4,
    width=800,
    height=550,
    sort_by=None
)
overall_chart.show()


# Year-over-year trends

yearly_trend_chart = create_bar_chart(
    data=yearly_avg_filtered.to_dict(),
    x_col='x',
    y_col='y',
    title='Average Number of Datasets per Notebook by Year',
    x_label='Year',
    y_label='Average Datasets per Notebook',
    color='#2E86AB',
    width=800,
    height=550,
    sort_by=None
)
yearly_trend_chart.show()


# Multi-dataset notebook trends

# Calculate percentage of notebooks using multiple datasets by year
yearly_multi_dataset = latest_versions_complete[latest_versions_complete['CreationYear'].isin(significant_years)].groupby('CreationYear').apply(
    lambda x: (x['DatasetCount'] > 1).mean() * 100
).round(1)

multi_dataset_chart = create_bar_chart(
    data=yearly_multi_dataset.to_dict(),
    x_col='x',
    y_col='y',
    title='Percentage of Notebooks Using Multiple Datasets by Year',
    x_label='Year',
    y_label='Percentage Using 2+ Datasets',
    color='#4ECDC4',
    width=800,
    height=550,
    sort_by=None
)
multi_dataset_chart.show()


# DATA PREPARATION SECTION

print("--- ANALYZING USER ACTIVITY PATTERNS BY YEAR ---")

# Clean data - remove rows with missing critical dates
users_clean = users[users['RegisterDate'].notna()].copy()
kernels_clean = kernels[kernels['CreationDate'].notna()].copy()
team_memberships_clean = team_memberships[team_memberships['RequestDate'].notna()].copy()

print(f"After cleaning:")
print(f"  Users with valid registration dates: {len(users_clean):,}")
print(f"  Kernels with valid creation dates: {len(kernels_clean):,}")
print(f"  Team memberships with valid request dates: {len(team_memberships_clean):,}")

# Add year columns for analysis
users_clean['RegisterYear'] = users_clean['RegisterDate'].dt.year
kernels_clean['CreationYear'] = kernels_clean['CreationDate'].dt.year
team_memberships_clean['RequestYear'] = team_memberships_clean['RequestDate'].dt.year

# Filter to specified years (start-2025)
current_year = 2025
year_filter = users_clean['RegisterYear'] <= current_year
users_clean = users_clean[year_filter].copy()

year_filter_kernels = kernels_clean['CreationYear'] <= current_year
kernels_clean = kernels_clean[year_filter_kernels].copy()

year_filter_teams = team_memberships_clean['RequestYear'] <= current_year
team_memberships_clean = team_memberships_clean[year_filter_teams].copy()

print(f"After year filtering (start-{current_year}):")
print(f"  Users: {len(users_clean):,}")
print(f"  Kernels: {len(kernels_clean):,}")
print(f"  Team memberships: {len(team_memberships_clean):,}")

# 1. Calculate new users each year
print("\n--- CALCULATING NEW USERS BY YEAR ---")
new_users_by_year = users_clean.groupby('RegisterYear').size()
print(f"New users calculated for {len(new_users_by_year)} years")

# 2. Calculate users who wrote at least one notebook each year
print("\n--- CALCULATING NOTEBOOK WRITERS BY YEAR ---")
notebook_writers_by_year = kernels_clean.groupby('CreationYear')['AuthorUserId'].nunique()
print(f"Notebook writers calculated for {len(notebook_writers_by_year)} years")

# 3. Calculate users who entered at least one competition each year
print("\n--- CALCULATING COMPETITION PARTICIPANTS BY YEAR ---")
competition_participants_by_year = team_memberships_clean.groupby('RequestYear')['UserId'].nunique()
print(f"Competition participants calculated for {len(competition_participants_by_year)} years")

# Create comprehensive year range for analysis
all_years = sorted(set(new_users_by_year.index) | set(notebook_writers_by_year.index) | set(competition_participants_by_year.index))
significant_years = [year for year in all_years if year <= current_year]

print(f"Analyzing {len(significant_years)} years: {min(significant_years)}-{max(significant_years)}")


# New users registered each year

new_users_chart = create_bar_chart(
    data=new_users_by_year.to_dict(),
    x_col='x',
    y_col='y',
    title='New Users Registered Each Year',
    subtitle='The number of users joining the platform <b>keeps increasing each year</b> and is well on track to surpass that number in 2025 as well',
    x_label='Year',
    y_label='Number of New Users',
    color=[palette_pink_light]*14 + [palette_maroon]*2,
    width=800,
    height=550,
    sort_by=None
)
new_users_chart.show()


# Users who wrote at least one notebook each year
notebook_writers_complete = notebook_writers_by_year.reindex(significant_years, fill_value=0)

# There are no notebooks from before 2015, so don't display these
notebook_writers_complete = notebook_writers_complete[notebook_writers_complete.index >= 2015]

notebook_writers_chart = create_bar_chart(
    data=notebook_writers_complete.to_dict(),
    x_col='x',
    y_col='y',
    title='Users Who Wrote At Least One Notebook Each Year',
    subtitle='The number of users actively participating and contributing to the Kaggle scene is increasing year by year',
    x_label='Year',
    y_label='Number of Notebook Writers',
    color=[palette_pink_light]* 9 + [palette_maroon]* 2,
    width=800,
    height=550,
    sort_by=None
)
notebook_writers_chart.show()


# Users who entered at least one competition each year

competition_participants_complete = competition_participants_by_year.reindex(significant_years, fill_value=0)

# There are no notebooks from before 2010, so don't display these
competition_participants_complete = competition_participants_complete[competition_participants_complete.index >= 2010]

competition_participants_chart = create_bar_chart(
    data=competition_participants_complete.to_dict(),
    x_col='x',
    y_col='y',
    title='Users Who Entered At Least One Competition Each Year',
    subtitle='Competition participation reached a record high in 2021, but then started to dip slightly in the following years',
    x_label='Year',
    y_label='Number of Competition Participants',
    color=[palette_pink_light]* 11 + [palette_maroon]* 5,
    width=800,
    height=550,
    sort_by=None
)
competition_participants_chart.show()



# Comparative analysis - Activity rates
new_users_complete = new_users_by_year.reindex(significant_years, fill_value=0)

# Calculate activity rates as percentage of new users
activity_rates = {}
for year in significant_years:
    if year > 2015:
        new_users = new_users_complete[year]
        notebook_writers = notebook_writers_complete[year]
        competition_participants = competition_participants_complete[year]
        
        if new_users > 0:
            notebook_rate = (notebook_writers / new_users) * 100
            competition_rate = (competition_participants / new_users) * 100
        else:
            notebook_rate = 0
            competition_rate = 0
        
        activity_rates[year] = {
            'Notebook Writers': notebook_rate,
            'Competition Participants': competition_rate
        }

# Create separate charts for each activity rate
notebook_rates = {year: rates['Notebook Writers'] for year, rates in activity_rates.items()}
competition_rates = {year: rates['Competition Participants'] for year, rates in activity_rates.items()}

notebook_rate_chart = create_bar_chart(
    data=notebook_rates,
    x_col='x',
    y_col='y',
    title='Notebook Writers as % of New Users Each Year',
    x_label='Year',
    y_label='Percentage of New Users (%)',
    color=[palette_maroon] + [palette_pink_light]* 11,
    width=800,
    height=550,
    sort_by=None
)
notebook_rate_chart.show()


competition_rate_chart = create_bar_chart(
    data=competition_rates,
    x_col='x',
    y_col='y',
    title='Competition Participants as % of New Users Each Year',
    x_label='Year',
    y_label='Percentage of New Users (%)',
    color=[palette_maroon]*2 + [palette_pink_light]* 11,
    width=800,
    height=550,
    sort_by=None
)
competition_rate_chart.show()


# --- DATA PREP ---

# Clean data
kernels_clean = kernels[kernels['CreationDate'].notna()].copy()
kernel_versions_clean = kernel_versions[kernel_versions['CreationDate'].notna()].copy()

# Get latest version details for each kernel
latest_versions_idx = kernel_versions_clean.groupby('ScriptId')['CreationDate'].idxmax()
latest_versions = kernel_versions_clean.loc[latest_versions_idx][['ScriptId', 'Title', 'AuthorUserId']].copy()

# Merge kernels with their latest version details
analysis_data = kernels_clean.merge(
   latest_versions,
   left_on='Id',
   right_on='ScriptId',
   how='left'
)

# Use TotalVotes and TotalComments from kernels table
analysis_data = analysis_data.rename(columns={
   'TotalVotes': 'Upvotes',
   'TotalComments': 'Comments'
})

# Add creation year
analysis_data['CreationYear'] = analysis_data['CreationDate'].dt.year

print(f"Analysis dataset prepared: {len(analysis_data):,} notebooks")

# Top 10 most upvoted notebooks
print("\n=== TOP 10 MOST UPVOTED NOTEBOOKS ===")
top_upvoted = analysis_data.nlargest(10, 'Upvotes')[['Title', 'Upvotes', 'Comments', 'TotalViews', 'CreationYear']]
display(top_upvoted)

# Top 10 most commented notebooks  
print("\n=== TOP 10 MOST COMMENTED NOTEBOOKS ===")
top_commented = analysis_data.nlargest(10, 'Comments')[['Title', 'Comments', 'Upvotes', 'TotalViews', 'CreationYear']]
display(top_commented)


# DATA PREPARATION SECTION

print("--- ANALYZING MOST COMMON COMPETITION TAG BUCKETS ---")

# Convert date columns
competitions['EnabledDate'] = pd.to_datetime(competitions['EnabledDate'], errors='coerce')
competitions['CreationYear'] = competitions['EnabledDate'].dt.year

# Define tag buckets based on our discussion
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

# Create reverse mapping for easier lookup
tag_to_bucket = {}
for bucket, tag_list in tag_buckets.items():
    for tag in tag_list:
        tag_to_bucket[tag] = bucket

for bucket, tags_list in tag_buckets.items():
    print(f"  {bucket}: {tags_list}")

# Link competitions to their tags
comp_tags_with_names = competition_tags.merge(
    tags[['Id', 'Name']],
    left_on='TagId',
    right_on='Id',
    how='left'
)

print(f"Competition-tag links with tag names: {len(comp_tags_with_names):,}")

# Apply bucketing to tags
print("\n--- Applying bucket categorization ---")
comp_tags_with_names['TagBucket'] = comp_tags_with_names['Name'].map(tag_to_bucket)

# Filter to only tags that belong to our defined buckets
comp_tags_bucketed = comp_tags_with_names[comp_tags_with_names['TagBucket'].notna()].copy()

print(f"Competition tags after bucketing: {len(comp_tags_bucketed):,}")

# Link back to competition details
comp_tags_complete = comp_tags_bucketed.merge(
    competitions[['Id', 'Title', 'EnabledDate', 'CreationYear', 'RewardType', 'TotalCompetitors']],
    left_on='CompetitionId',
    right_on='Id',
    how='left'
)

print(f"Complete competition-tag-bucket dataset: {len(comp_tags_complete):,}")

# Filter to valid years for trend analysis
current_year = 2025
comp_tags_filtered = comp_tags_complete[
    (comp_tags_complete['CreationYear'] >= 2010) & 
    (comp_tags_complete['CreationYear'] <= current_year)
].copy()

print(f"Filtered to valid years (2010-{current_year}): {len(comp_tags_filtered):,}")


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
    height=550,
    sort_by='y',
    ascending=False
)
bucket_counts_chart.show()


# Multi-tag competitions analysis

# Count how many buckets each competition has
tags_per_competition = comp_tags_filtered.groupby('CompetitionId')['TagBucket'].nunique().reset_index()
tags_per_competition.columns = ['CompetitionId', 'NumberOfBuckets']

# Distribution of tag bucket counts
bucket_count_distribution = tags_per_competition['NumberOfBuckets'].value_counts().sort_index()

# Create categories for display
bucket_count_categories = {}
for count, freq in bucket_count_distribution.items():
    if count == 1:
        bucket_count_categories['Single Topic'] = freq
    elif count == 2:
        bucket_count_categories['Two Topics'] = freq
    elif count == 3:
        bucket_count_categories['Three Topics'] = freq
    else:
        bucket_count_categories['Four+ Topics'] = freq

multi_tag_chart = create_bar_chart(
    data=bucket_count_categories,
    x_col='x',
    y_col='y',
    title='Number of Topic Buckets per Competition',
    subtitle='Often competitions will include multiple competition tags eg. finding sentiment analysis over time would include nlp and time series analysis',
    x_label='Number of Topic Buckets',
    y_label='Number of Competitions',
    color=[palette_maroon]*2 + [palette_pink_light]*2,
    width=800,
    height=550,
    sort_by=None
)
multi_tag_chart.show()


# --- DATA PREP ---

# Define tag buckets
tag_buckets = {
    'Tabular': ['binary classification', 'multiclass classification', 'regression'],
    'Image/Vision': ['image', 'computer vision'],
    'Text/NLP': ['text', 'nlp'],
    'Time Series': ['time series analysis']
}

# Create reverse mapping for easier lookup
tag_to_bucket = {}
for bucket, tag_list in tag_buckets.items():
    for tag in tag_list:
        tag_to_bucket[tag] = bucket

print("Tag buckets defined:")
for bucket, tags_list in tag_buckets.items():
    print(f"  {bucket}: {tags_list}")

# Link kernel versions to competitions
print("\n--- Linking kernel versions to competitions ---")
kernel_comp_links = kernel_version_comp_sources.merge(
    kernel_versions[['Id', 'ScriptId', 'AuthorUserId', 'CreationDate']],
    left_on='KernelVersionId',
    right_on='Id',
    how='inner'
)

print(f"Kernel versions linked to competitions: {len(kernel_comp_links):,}")

# Link competitions to their tags
print("\n--- Linking competitions to tags ---")
comp_tags_with_names = competition_tags.merge(
    tags[['Id', 'Name']],
    left_on='TagId',
    right_on='Id',
    how='inner'
)

# Apply bucket mapping
comp_tags_with_names['TagBucket'] = comp_tags_with_names['Name'].map(tag_to_bucket)

# Filter to only tags that belong to our defined buckets
comp_tags_bucketed = comp_tags_with_names[comp_tags_with_names['TagBucket'].notna()].copy()

print(f"Competition tags after bucketing: {len(comp_tags_bucketed):,}")

# Link kernel-competition data to tag buckets
print("\n--- Linking kernels to tag buckets ---")
kernel_comp_tags = kernel_comp_links.merge(
    comp_tags_bucketed[['CompetitionId', 'TagBucket']],
    left_on='SourceCompetitionId',
    right_on='CompetitionId',
    how='inner'
)

print(f"Kernel-competition-tag links: {len(kernel_comp_tags):,}")

# Aggregate by user and bucket
print("\n--- Analyzing user bucket diversity ---")
user_bucket_combinations = kernel_comp_tags.groupby(['AuthorUserId', 'TagBucket']).size().reset_index(name='NotebookCount')

print(f"User-bucket combinations: {len(user_bucket_combinations):,}")

# Calculate how many buckets each user has worked with
user_bucket_counts = user_bucket_combinations.groupby('AuthorUserId')['TagBucket'].nunique().reset_index()
user_bucket_counts.columns = ['AuthorUserId', 'BucketCount']

print(f"Users with bucket diversity data: {len(user_bucket_counts):,}")

# Add user information
user_bucket_counts = user_bucket_counts.merge(
    users[['Id', 'UserName', 'RegisterDate']],
    left_on='AuthorUserId',
    right_on='Id',
    how='left'
)

# Filter to users with valid registration dates for cohort analysis
user_bucket_counts_clean = user_bucket_counts[user_bucket_counts['RegisterDate'].notna()].copy()
user_bucket_counts_clean['RegisterYear'] = user_bucket_counts_clean['RegisterDate'].dt.year

# Filter to reasonable years
current_year = 2025
user_bucket_counts_clean = user_bucket_counts_clean[
    (user_bucket_counts_clean['RegisterYear'] <= current_year)
].copy()

print(f"Users with clean registration data: {len(user_bucket_counts_clean):,}")



# Distribution of bucket diversity across users

bucket_diversity_dist = user_bucket_counts_clean['BucketCount'].value_counts().sort_index()

# Create meaningful labels
diversity_labels = {
    1: "Single Topic<br>(1 bucket)",
    2: "Two Topics<br>(2 buckets)", 
    3: "Three Topics<br>(3 buckets)",
    4: "All Topics<br>(4 buckets)"
}

diversity_display = {}
for count, freq in bucket_diversity_dist.items():
    label = diversity_labels.get(count, f"{count} buckets")
    diversity_display[label] = freq * 100 / bucket_diversity_dist.sum()

diversity_chart = create_bar_chart(
    data=diversity_display,
    valueformat='percentage',
    x_col='x',
    y_col='y',
    title='Distribution of Competition Types Users participated in',
    subtitle='We look at all notebooks created and note the tags for the competitions they were created for. While the <b>majority stuck to a single topic</b>, many are willing to dip their toes into <b>different competition types</b>.',
    x_label='Number of Topic Buckets Explored',
    y_label='Number of Users',
    color=[palette_maroon]*2 + [palette_pink_light]*2,
    width=800,
    height=580,
    sort_by=None
)
diversity_chart.show()


# Memory cleanup for all generated analyses - keep initial datasets
import gc

print("--- CLEANING UP ANALYSIS VARIABLES ---")

# Variables to keep (initial datasets)
keep_variables = {
   'users', 'kernels', 'kernel_versions', 'kernel_votes', 'competitions', 
   'competition_tags', 'tags', 'datasets', 'forum_topics', 'forum_messages',
   'teams', 'team_memberships', 'submissions', 'kernel_version_comp_sources',
   'kernel_version_dataset_sources', 'organizations', 'datasources'
}

# Get all current variables
all_variables = set(globals().keys())

# Find analysis variables to delete (exclude built-ins, functions, and kept datasets)
analysis_variables = []
for var_name in all_variables:
   if (not var_name.startswith('_') and  # Not built-in
       var_name not in keep_variables and  # Not a dataset to keep
       var_name not in ['gc', 'pd', 'np', 'plt', 'sns', 'create_bar_chart', 'plot_value_counts', 'plot_groupby_result'] and  # Not functions/imports
       isinstance(globals()[var_name], (pd.DataFrame, pd.Series, dict, list))):  # Is a data structure
       analysis_variables.append(var_name)

print(f"Found {len(analysis_variables)} analysis variables to clean up")

# Delete analysis variables
deleted_count = 0
for var_name in analysis_variables:
   try:
       del globals()[var_name]
       deleted_count += 1
   except:
       pass

print(f"Deleted {deleted_count} variables")

# Force garbage collection
gc.collect()

print("Memory cleanup complete! Initial datasets preserved.")
print("\nRemaining key datasets:")
remaining_datasets = [var for var in keep_variables if var in globals()]
for dataset in sorted(remaining_datasets):
   print(f"  ✓ {dataset}")


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

# Filter to competitions lasting 2-16 weeks for meaningful weekly analysis
competitions_clean = competitions_clean[
    (competitions_clean['DurationDays'] >= 14) & 
    (competitions_clean['DurationDays'] <= 112)
].copy()

print(f"Competitions with valid dates and 2-16 week duration: {len(competitions_clean):,}")

# Prepare data for each activity indicator
print("\n--- Preparing activity data ---")

# 1. Kernels created (link through KernelVersionCompetitionSources)

# Link kernels to competitions through version sources
kernels_with_comps = kernels_clean.merge(
    kernel_versions[['Id', 'ScriptId']],
    left_on='Id',
    right_on='ScriptId',
    how='inner'
).merge(
    kernel_version_comp_sources[['KernelVersionId', 'SourceCompetitionId']],
    left_on='Id_y',  # kernel version Id
    right_on='KernelVersionId',
    how='inner'
).merge(
    competitions_clean[['Id', 'EnabledDate', 'DeadlineDate']],
    left_on='SourceCompetitionId',
    right_on='Id',
    how='inner'
)

# Calculate weeks from competition start
kernels_with_comps['WeeksFromStart'] = (
    kernels_with_comps['CreationDate'] - kernels_with_comps['EnabledDate']
).dt.days // 7

# Filter to reasonable timeframe (0-20 weeks)
kernels_with_comps = kernels_with_comps[
    (kernels_with_comps['WeeksFromStart'] >= 0) & 
    (kernels_with_comps['WeeksFromStart'] <= 20)
]

print(f"Kernels linked to competitions: {len(kernels_with_comps):,}")

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

print(f"Forum topics linked to competitions: {len(topics_with_comps):,}")

# 3. Forum replies
messages_clean = forum_messages[forum_messages['PostDate'].notna()].copy()
messages_with_comps = messages_clean.merge(
    topics_with_comps[['Id_x', 'ForumId', 'EnabledDate']].drop_duplicates(),
    left_on='ForumTopicId',
    right_on='Id_x',
    how='inner'
)

messages_with_comps['WeeksFromStart'] = (
    messages_with_comps['PostDate'] - messages_with_comps['EnabledDate']
).dt.days // 7

messages_with_comps = messages_with_comps[
    (messages_with_comps['WeeksFromStart'] >= 0) & 
    (messages_with_comps['WeeksFromStart'] <= 20)
]

print(f"Forum messages linked to competitions: {len(messages_with_comps):,}")

# 4. Notebooks forked (link through KernelVersionCompetitionSources)
forked_kernels = kernels_clean[kernels_clean['ForkParentKernelVersionId'].notna()].copy()

# Link forked kernels to competitions through version sources
forked_with_comps = forked_kernels.merge(
    kernel_versions[['Id', 'ScriptId']],
    left_on='Id',
    right_on='ScriptId',
    how='inner'
).merge(
    kernel_version_comp_sources[['KernelVersionId', 'SourceCompetitionId']],
    left_on='Id_y',  # kernel version Id
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

print(f"Forked kernels linked to competitions: {len(forked_with_comps):,}")

# 5. Submissions and scores (link through teams)
submissions_clean = submissions[submissions['SubmissionDate'].notna()].copy()

# Link submissions to competitions through teams
submissions_with_comps = submissions_clean.merge(
    teams[['Id', 'CompetitionId']],
    left_on='TeamId',
    right_on='Id',
    how='inner'
).merge(
    competitions_clean[['Id', 'EnabledDate', 'DeadlineDate']],
    left_on='CompetitionId',
    right_on='Id',
    how='inner'
)

submissions_with_comps['WeeksFromStart'] = (
    submissions_with_comps['SubmissionDate'] - submissions_with_comps['EnabledDate']
).dt.days // 7

submissions_with_comps = submissions_with_comps[
    (submissions_with_comps['WeeksFromStart'] >= 0) & 
    (submissions_with_comps['WeeksFromStart'] <= 20)
]

print(f"Submissions linked to competitions: {len(submissions_with_comps):,}")


# Kernels created per week
print(f"\n--- KERNELS CREATED BY WEEK ---")

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
print(f"\n--- DISCUSSION TOPICS BY WEEK ---")

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


weekly_messages = messages_with_comps.groupby('WeeksFromStart').size()
weekly_messages_complete = {week: weekly_messages.get(week, 0) * 100 / len(messages_with_comps) for week in weeks_range}

messages_chart = create_bar_chart(
    data=weekly_messages_complete,
    x_col='x',
    y_col='y',
    title='Forum Replies by Week After Competition Start',
    x_label='Weeks from Competition Start',
    y_label='Number of Forum Replies',
    valueformat='percentage',
    color= [palette_maroon] + [palette_pink_light]*20,
    width=900,
    height=550,
    sort_by=None
)
messages_chart.show()

write_csv_from_dict(weekly_messages_complete, ['week', 'value'], 'comp_activity_messages')


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
    width=1200,
    height=550,
    sort_by=None
)
forks_chart.show()

write_csv_from_dict(weekly_forks_complete, ['week', 'value'], 'comp_activity_forks')


print(f"\n--- ACTIVITY SUMMARY ---")
print(f"PEAK ACTIVITY WEEKS:")

# Find peak weeks for each activity
peak_kernels_week = max(weekly_kernels_complete, key=weekly_kernels_complete.get)
peak_topics_week = max(weekly_topics_complete, key=weekly_topics_complete.get)
peak_messages_week = max(weekly_messages_complete, key=weekly_messages_complete.get)
peak_forks_week = max(weekly_forks_complete, key=weekly_forks_complete.get)

print(f"• Peak kernel creation: Week {peak_kernels_week} ({weekly_kernels_complete[peak_kernels_week]:,} kernels)")
print(f"• Peak discussion topics: Week {peak_topics_week} ({weekly_topics_complete[peak_topics_week]:,} topics)")
print(f"• Peak forum replies: Week {peak_messages_week} ({weekly_messages_complete[peak_messages_week]:,} replies)")
print(f"• Peak notebook forks: Week {peak_forks_week} ({weekly_forks_complete[peak_forks_week]:,} forks)")

# Early vs late activity comparison
early_weeks = list(range(0, 4))  # First month
late_weeks = list(range(8, 12))   # Third month

early_kernels = sum(weekly_kernels_complete.get(w, 0) for w in early_weeks)
late_kernels = sum(weekly_kernels_complete.get(w, 0) for w in late_weeks)

early_topics = sum(weekly_topics_complete.get(w, 0) for w in early_weeks)
late_topics = sum(weekly_topics_complete.get(w, 0) for w in late_weeks)

print(f"\n EARLY VS LATE COMPETITION ACTIVITY:")
print(f"• Kernels - Early (weeks 0-3): {early_kernels:,}, Late (weeks 8-11): {late_kernels:,}")
print(f"• Topics - Early (weeks 0-3): {early_topics:,}, Late (weeks 8-11): {late_topics:,}")

print(f"\n CONCLUSION: Competition activity shows {'early' if early_kernels > late_kernels else 'sustained'}")
print(f"   engagement patterns with peak kernel creation in week {peak_kernels_week}")
print(f"   and peak discussions in week {peak_topics_week}.")

