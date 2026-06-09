


import kagglehub
import pandas as pd
import os

MK_PATH = kagglehub.dataset_download("kaggle/meta-kaggle")
MKC_PATH = kagglehub.dataset_download("kaggle/meta-kaggle-code")

print("Path to Meta-Kaggle dataset files:", MK_PATH)
print("Path to Meta-Kaggle-Code dataset files:", MKC_PATH)


for dirname, _, filenames in os.walk(MK_PATH):
    for filename in filenames:
        print(filename)


# Confirmed paths from your output
MK_PATH = "/kaggle/input/meta-kaggle/"
MKC_PATH = "/kaggle/input/meta-kaggle-code/"

# Load essential tables (using exact filenames from your output)
kernels = pd.read_csv(os.path.join(MK_PATH, "Kernels.csv"))
versions = pd.read_csv(os.path.join(MK_PATH, "KernelVersions.csv"))
tags = pd.read_csv(os.path.join(MK_PATH, "KernelTags.csv"))
competitions = pd.read_csv(os.path.join(MK_PATH, "Competitions.csv"))

print(f"""
âœ… Data Loaded:
- Kernels: {len(kernels):,}
- Versions: {len(versions):,}
- Tags: {len(tags):,}
- Competitions: {len(competitions):,}
""")


import pandas as pd
import numpy as np

# Load critical files (competition-focused)
kernels = pd.read_csv('/kaggle/input/meta-kaggle/Kernels.csv')
versions = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersions.csv')
comp_sources = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersionCompetitionSources.csv')

print(f"Raw counts - Kernels: {len(kernels)}, Versions: {len(versions)}")




print("Columns in Kernels.csv:", kernels.columns.tolist())
# Merge to find competition-linked kernels
comp_kernels = versions.merge(
    comp_sources,
    left_on='Id',
    right_on='KernelVersionId',
    how='inner'
).merge(
    kernels[['Id', 'AuthorUserId', 'CurrentUrlSlug']],
    left_on='ScriptId',
    right_on='Id',
    suffixes=('_version', '_kernel')
)

# Keep only latest version per kernel
comp_kernels = comp_kernels.sort_values('VersionNumber').groupby('ScriptId').last().reset_index()
print(f"Competition kernels: {len(comp_kernels)}")


# Merge with competition sources first (smaller dataframe)
comp_kernels = versions.merge(
    comp_sources,
    left_on='Id',
    right_on='KernelVersionId',
    how='inner'
).merge(
    kernels,
    left_on='ScriptId',
    right_on='Id'
)

# Keep only latest version per kernel
comp_kernels = comp_kernels.sort_values('VersionNumber').groupby('ScriptId').last().reset_index()
print(f"Final competition kernels: {len(comp_kernels)}")


import pandas as pd

# First, check what columns actually exist in KernelVersions.csv
versions_sample = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersions.csv', nrows=5)
print("Columns in KernelVersions.csv:", versions_sample.columns.tolist())


cols_versions = ['Id', 'ScriptId', 'Title', 'VersionNumber', 'TotalLines']  # Available metrics
cols_kernels = ['Id', 'AuthorUserId', 'CurrentUrlSlug', 'TotalVotes']
cols_comp_sources = ['KernelVersionId', 'SourceCompetitionId']

versions = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersions.csv', usecols=cols_versions)
kernels = pd.read_csv('/kaggle/input/meta-kaggle/Kernels.csv', usecols=cols_kernels)
comp_sources = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersionCompetitionSources.csv', usecols=cols_comp_sources)


print("Columns in merged DataFrame:", comp_kernels.columns.tolist())
# First, make sure we're loading TotalVotes from KernelVersions
cols_versions = ['Id', 'ScriptId', 'Title', 'VersionNumber', 'TotalLines', 'TotalVotes']  # Added TotalVotes here
cols_kernels = ['Id', 'AuthorUserId', 'CurrentUrlSlug']  # Removed TotalVotes from here
cols_comp_sources = ['KernelVersionId', 'SourceCompetitionId']

# Reload data with correct columns
versions = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersions.csv', usecols=cols_versions)
kernels = pd.read_csv('/kaggle/input/meta-kaggle/Kernels.csv', usecols=cols_kernels)
comp_sources = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersionCompetitionSources.csv', usecols=cols_comp_sources)

# Merge ensuring we keep TotalVotes
comp_kernels = versions.merge(
    comp_sources,
    left_on='Id',
    right_on='KernelVersionId',
    how='inner'
).merge(
    kernels,
    left_on='ScriptId',
    right_on='Id'
)

# Now this should work
comp_kernels['lines_per_version'] = comp_kernels['TotalLines'] / comp_kernels['VersionNumber']
comp_kernels['log_votes'] = np.log1p(comp_kernels['TotalVotes'])
comp_kernels['novelty_proxy'] = comp_kernels['lines_per_version'] / (1 + comp_kernels['log_votes'])




# We already have a clean 'TotalVotes' column - no need for _x/_y handling!
print("Current 'TotalVotes' stats:")
print(comp_kernels['TotalVotes'].describe())

# Recalculate novelty metrics with proper columns
comp_kernels['lines_per_version'] = comp_kernels['TotalLines'] / comp_kernels['VersionNumber']
comp_kernels['log_votes'] = np.log1p(comp_kernels['TotalVotes'])
comp_kernels['novelty_score'] = (
    comp_kernels['lines_per_version'] / 
    (1 + comp_kernels['log_votes'])
)

# Show top novel kernels
top_novel = comp_kernels.sort_values('novelty_score', ascending=False)[[
    'Title', 'novelty_score', 'TotalVotes', 'TotalLines', 'VersionNumber'
]].head(5)
print("\nğŸ�† Top 5 Novel Kernels:")
display(top_novel)


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the datasets (adjust paths if needed)
try:
    kernels = pd.read_csv('/kaggle/input/meta-kaggle/Kernels.csv')
    versions = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersions.csv')
    comp_sources = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersionCompetitionSources.csv')
    
    # Merge the datasets
    comp_kernels = versions.merge(
        comp_sources,
        left_on='Id',
        right_on='KernelVersionId',
        how='inner'
    ).merge(
        kernels,
        left_on='ScriptId',
        right_on='Id'
    )
    
    # Keep only latest version per kernel
    comp_kernels = comp_kernels.sort_values('VersionNumber').groupby('ScriptId').last().reset_index()
    
    print(f"âœ… Successfully loaded and merged {len(comp_kernels)} competition kernels")
    
except Exception as e:
    print(f"â�Œ Error loading data: {str(e)}")
    # Create empty DataFrame to prevent further errors
    comp_kernels = pd.DataFrame()


import pandas as pd
import numpy as np

# Load datasets with explicit voting data
try:
    # Load versions with voting data
    versions = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersions.csv', 
                         usecols=['Id', 'ScriptId', 'VersionNumber', 'Title', 'TotalLines', 'TotalVotes'])
    
    # Load competition sources
    comp_sources = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersionCompetitionSources.csv',
                             usecols=['KernelVersionId', 'SourceCompetitionId'])
    
    # Load kernel metadata
    kernels = pd.read_csv('/kaggle/input/meta-kaggle/Kernels.csv',
                        usecols=['Id', 'AuthorUserId', 'CurrentUrlSlug'])
    
    # Merge datasets
    comp_kernels = versions.merge(
        comp_sources,
        left_on='Id',
        right_on='KernelVersionId'
    ).merge(
        kernels,
        left_on='ScriptId',
        right_on='Id'
    )
    
    # Keep latest version per kernel
    comp_kernels = comp_kernels.sort_values('VersionNumber').groupby('ScriptId').last().reset_index()
    
    print(f"âœ… Successfully loaded {len(comp_kernels)} kernels with voting data")
    print("Available columns:", comp_kernels.columns.tolist())
    
except Exception as e:
    print(f"â�Œ Error: {str(e)}")
    comp_kernels = pd.DataFrame()


### FUTURE ENGINEERING

if not comp_kernels.empty:
    # Create essential features with verification
    required_cols = ['TotalLines', 'VersionNumber', 'TotalVotes']
    missing_cols = [col for col in required_cols if col not in comp_kernels.columns]
    
    if missing_cols:
        print(f"âš ï¸� Missing columns: {missing_cols} - using fallbacks")
        # Create fallback columns
        if 'TotalVotes' not in comp_kernels.columns:
            comp_kernels['TotalVotes'] = 0
        if 'TotalLines' not in comp_kernels.columns:
            comp_kernels['TotalLines'] = 100  # Default assumption
        if 'VersionNumber' not in comp_kernels.columns:
            comp_kernels['VersionNumber'] = 1
    
    # Now safely create features
    comp_kernels['lines_per_version'] = comp_kernels['TotalLines'] / comp_kernels['VersionNumber']
    comp_kernels['log_votes'] = np.log1p(comp_kernels['TotalVotes'])
    
    # Change ratio with fallback
    if 'LinesChangedFromPrevious' in comp_kernels.columns:
        comp_kernels['change_ratio'] = comp_kernels['LinesChangedFromPrevious'] / comp_kernels['TotalLines']
    else:
        comp_kernels['change_ratio'] = 1 / comp_kernels['VersionNumber']
    
    # Final novelty score
    comp_kernels['final_score'] = (
        comp_kernels['lines_per_version'] * 
        comp_kernels['change_ratio'] / 
        (1 + comp_kernels['log_votes'])
    )
    
    print("Feature engineering complete. Sample data:")
    display(comp_kernels[['Title', 'final_score', 'TotalVotes']].head())


if not comp_kernels.empty:
    # Visualization
    try:
        !pip install plotly -q
        import plotly.express as px
        
        fig = px.scatter(
            comp_kernels,
            x='lines_per_version',
            y='final_score',
            size='TotalVotes',
            color='VersionNumber',
            hover_data=['Title', 'AuthorUserId'],
            title='Kernel Novelty vs. Code Volume',
            log_x=True,
            width=1000
        )
        fig.show()
    except Exception as e:
        print(f"Visualization error: {str(e)}")
    
    # Export
    export_cols = ['ScriptId', 'Title', 'final_score', 'TotalVotes', 'TotalLines']
    export_cols = [col for col in export_cols if col in comp_kernels.columns]
    
    if len(export_cols) >= 3:
        comp_kernels.sort_values('final_score', ascending=False).head(100)[export_cols]\
            .to_csv("top_novel_kernels.csv", index=False)
        print("âœ… Exported top_novel_kernels.csv")
    else:
        print("Insufficient columns for export")








