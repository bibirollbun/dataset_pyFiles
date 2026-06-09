# Essential Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import ipywidgets as widgets
from IPython.display import display
import warnings
import json

# Niche Imports



# Settings
sns.set_style("whitegrid")
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
warnings.filterwarnings("ignore")


# Tools


# View files

def get_comp_files_and_dirs(input_dir):
    file_list = []
    dir_list = []
    try:
        for comp_dir in os.listdir(input_dir):
            comp_path = '/'.join([input_dir, comp_dir])
            print(f"Competition Directory: {comp_path}")
            print("Contains:")
            with os.scandir(comp_path) as entries:
                for entry in entries:
                    if entry.is_file():
                        print(f"- (File) {entry.name}, Size: {entry.stat().st_size} bytes")
                        file_list.append(os.path.join(input_dir, comp_dir, entry))
                    elif entry.is_dir():
                        print(f"- (Folder) {entry.name}")
                        dir_list.append(os.path.join(input_dir, comp_dir, entry))

    except FileNotFoundError:
        print(f"The specified directory '{directory}' does not exist.")
    except PermissionError:
        print(f"Permission error accessing directory '{directory}'.")
    return file_list, dir_list
    
input_dir = '/kaggle/input'
file_list, dir_list = get_comp_files_and_dirs(input_dir)


# Load Data
comp_path = '/kaggle/input/MABe-mouse-behavior-detection'
train_df = pd.read_csv(os.path.join(comp_path, 'train.csv'))
test_df = pd.read_csv(os.path.join(comp_path, 'test.csv'))
ss_df = pd.read_csv(os.path.join(comp_path, 'sample_submission.csv'))
df_dict = {
    "Training data": train_df,
    "Testing data": test_df,
    "Sample submission": ss_df,
}

test_tracking_dir = os.path.join(comp_path, 'test_tracking')
train_annotation_dir = os.path.join(comp_path, 'train_annotation')


# Example data
train_annotation_example = pd.read_parquet('/kaggle/input/MABe-mouse-behavior-detection/train_annotation/AdaptableSnail/1212811043.parquet')
train_tracking_example = pd.read_parquet(r'/kaggle/input/MABe-mouse-behavior-detection/train_tracking/AdaptableSnail/1212811043.parquet')


display(ss_df)


display(train_df)


display(test_df)


print(train_df.iloc[0])


print(f'Example Body parts tracked: \n{train_df.iloc[0]["body_parts_tracked"]}')
print(f'Example Behaviours labelled: \n{train_df.iloc[0]["behaviors_labeled"]}')


display(train_annotation_example)


display(train_tracking_example)


from matplotlib.ticker import ScalarFormatter

# Count unique video_ids per lab_id
videos_per_lab = train_df.groupby('lab_id')['video_id'].nunique().sort_values(ascending=False)

# Create column plot with logarithmic y-axis
plt.figure(figsize=(12, 6))
plt.bar(range(len(videos_per_lab)), videos_per_lab.values, color=color_pal[0])
plt.xlabel('Lab ID', fontsize=12)
plt.ylabel('Number of Unique Videos (log scale)', fontsize=12)
plt.title('Number of Unique Videos per Lab', fontsize=14, fontweight='bold')
plt.xticks(range(len(videos_per_lab)), videos_per_lab.index, rotation=45, ha='right')
plt.yscale('log')

# Format y-axis to show full numbers instead of scientific notation
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.get_major_formatter().set_scientific(False)

# Add gridlines with lighter color
plt.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.show()

print(f'\nSummary Statistics:')
print(f'Total labs: {len(videos_per_lab)}')
print(f'Total videos: {videos_per_lab.sum()}')
print(f'Mean videos per lab: {videos_per_lab.mean():.1f}')
print(f'Median videos per lab: {videos_per_lab.median():.1f}')
print(f'Max videos (lab {videos_per_lab.idxmax()}): {videos_per_lab.max()}')
print(f'Min videos (lab {videos_per_lab.idxmin()}): {videos_per_lab.min()}')


# 5. Heatmap: Action Types by Lab ID
# Create a matrix showing which action types are used by each lab

# Build a dataframe with lab_id and action types
lab_action_data = []

for idx, row in train_df.iterrows():
    lab_id = row['lab_id']
    try:
        behaviors_list = json.loads(row['behaviors_labeled'].replace("'", '"'))
        # Extract action types from behaviors
        for behavior in behaviors_list:
            action = behavior.split(',')[-1]  # Get the action part
            lab_action_data.append({'lab_id': lab_id, 'action': action})
    except:
        pass

# Create dataframe and count occurrences
lab_action_df = pd.DataFrame(lab_action_data)
action_matrix = lab_action_df.groupby(['lab_id', 'action']).size().unstack(fill_value=0)

# Sort columns by total occurrences (descending)
column_sums = action_matrix.sum(axis=0).sort_values(ascending=False)
action_matrix = action_matrix[column_sums.index]

# Sort rows by total occurrences (descending)
row_sums = action_matrix.sum(axis=1).sort_values(ascending=False)
action_matrix = action_matrix.loc[row_sums.index]

# Replace 0 with NaN to make them appear white/empty in the heatmap
action_matrix_masked = action_matrix.replace(0, np.nan)

# Create heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(action_matrix_masked, cmap='viridis', annot=False, fmt='g',
            cbar_kws={'label': 'Number of Occurrences'},
            linewidths=0.5, linecolor='lightgray')
plt.xlabel('Action Type', fontsize=12, fontweight='bold')
plt.ylabel('Lab ID', fontsize=12, fontweight='bold')
plt.title('Action Types by Lab ID', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# 2. Heatmap: Body Parts Tracked by Lab ID
# Create a matrix showing which body parts are tracked by each lab

# Build a dataframe with lab_id and body parts
lab_bodypart_data = []

for idx, row in train_df.iterrows():
    lab_id = row['lab_id']
    try:
        body_parts_list = json.loads(row['body_parts_tracked'].replace("'", '"'))
        # Extract each body part
        for body_part in body_parts_list:
            lab_bodypart_data.append({'lab_id': lab_id, 'body_part': body_part})
    except:
        pass

# Create dataframe and get unique combinations (1 if tracked, 0 if not)
lab_bodypart_df = pd.DataFrame(lab_bodypart_data)
bodypart_matrix = lab_bodypart_df.groupby(['lab_id', 'body_part']).size().unstack(fill_value=0)

# Convert to boolean (True if tracked, False if not)
bodypart_matrix_bool = (bodypart_matrix > 0).astype(int)

# Sort columns by total occurrences (descending)
column_sums = bodypart_matrix_bool.sum(axis=0).sort_values(ascending=False)
bodypart_matrix_bool = bodypart_matrix_bool[column_sums.index]

# Sort rows by total occurrences (descending)
row_sums = bodypart_matrix_bool.sum(axis=1).sort_values(ascending=False)
bodypart_matrix_bool = bodypart_matrix_bool.loc[row_sums.index]

# Create custom colormap for binary data (white = not tracked, blue = tracked)
from matplotlib.colors import ListedColormap
colors = ['#f0f0f0', '#2E86AB']  # Light gray for 0, Blue for 1
cmap = ListedColormap(colors)

# Create heatmap
plt.figure(figsize=(16, 10))
sns.heatmap(bodypart_matrix_bool, cmap=cmap, annot=False, fmt='d',
            cbar_kws={'label': 'Tracked', 'ticks': [0.25, 0.75]},
            linewidths=0.5, linecolor='white', vmin=0, vmax=1)

# Customize colorbar labels
colorbar = plt.gca().collections[0].colorbar
colorbar.set_ticklabels(['Not Tracked', 'Tracked'])

plt.xlabel('Body Part', fontsize=12, fontweight='bold')
plt.ylabel('Lab ID', fontsize=12, fontweight='bold')
plt.title('Body Parts Tracked by Lab ID', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# 1. Heatmap: Tracking Methods by Lab ID
# Create a matrix showing which tracking methods are used by each lab
import json

# Get unique tracking methods per lab
tracking_by_lab = train_df.groupby('lab_id')['tracking_method'].apply(lambda x: x.unique().tolist()).to_dict()

# Create a summary dataframe
tracking_summary = train_df.groupby(['lab_id', 'tracking_method']).size().unstack(fill_value=0)

# Convert to boolean (True if used, False if not)
tracking_summary_bool = (tracking_summary > 0).astype(int)

# Sort columns by total occurrences (descending)
column_sums = tracking_summary_bool.sum(axis=0).sort_values(ascending=False)
tracking_summary_bool = tracking_summary_bool[column_sums.index]

# Sort rows by total occurrences (descending)
row_sums = tracking_summary_bool.sum(axis=1).sort_values(ascending=False)
tracking_summary_bool = tracking_summary_bool.loc[row_sums.index]

# Create custom colormap for binary data (light gray = not used, green = used)
from matplotlib.colors import ListedColormap
colors = ['#f0f0f0', '#27AE60']  # Light gray for 0, Green for 1
cmap = ListedColormap(colors)

# Create heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(tracking_summary_bool, cmap=cmap, annot=False, fmt='d',
            cbar_kws={'label': 'Used', 'ticks': [0.25, 0.75]},
            linewidths=0.5, linecolor='white', vmin=0, vmax=1)

# Customize colorbar labels
colorbar = plt.gca().collections[0].colorbar
colorbar.set_ticklabels(['Not Used', 'Used'])

plt.xlabel('Tracking Method', fontsize=12, fontweight='bold')
plt.ylabel('Lab ID', fontsize=12, fontweight='bold')
plt.title('Tracking Methods by Lab ID', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# 4. Heatmap: Mouse Strains by Lab ID
# Create a matrix showing which mouse strains are used by each lab

# Build a dataframe with lab_id and mouse strains
lab_strain_data = []

for idx, row in train_df.iterrows():
    lab_id = row['lab_id']
    # Collect strains from all mice (mouse1 through mouse4)
    for i in range(1, 5):
        strain_col = f'mouse{i}_strain'
        if strain_col in row and pd.notna(row[strain_col]) and row[strain_col] != '':
            lab_strain_data.append({'lab_id': lab_id, 'strain': row[strain_col]})

# Create dataframe and get unique combinations (1 if used, 0 if not)
lab_strain_df = pd.DataFrame(lab_strain_data)
strain_matrix = lab_strain_df.groupby(['lab_id', 'strain']).size().unstack(fill_value=0)

# Convert to boolean (True if used, False if not)
strain_matrix_bool = (strain_matrix > 0).astype(int)

# Sort columns by total occurrences (descending)
column_sums = strain_matrix_bool.sum(axis=0).sort_values(ascending=False)
strain_matrix_bool = strain_matrix_bool[column_sums.index]

# Sort rows by total occurrences (descending)
row_sums = strain_matrix_bool.sum(axis=1).sort_values(ascending=False)
strain_matrix_bool = strain_matrix_bool.loc[row_sums.index]

# Create custom colormap for binary data (light gray = not used, orange = used)
from matplotlib.colors import ListedColormap
colors = ['#f0f0f0', '#E67E22']  # Light gray for 0, Orange for 1
cmap = ListedColormap(colors)

# Create heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(strain_matrix_bool, cmap=cmap, annot=False, fmt='d',
            cbar_kws={'label': 'Used', 'ticks': [0.25, 0.75]},
            linewidths=0.5, linecolor='white', vmin=0, vmax=1)

# Customize colorbar labels
colorbar = plt.gca().collections[0].colorbar
colorbar.set_ticklabels(['Not Used', 'Used'])

plt.xlabel('Mouse Strain', fontsize=12, fontweight='bold')
plt.ylabel('Lab ID', fontsize=12, fontweight='bold')
plt.title('Mouse Strains by Lab ID', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

