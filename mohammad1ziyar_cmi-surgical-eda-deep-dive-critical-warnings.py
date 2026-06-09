import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from IPython.display import display, Image
from itertools import combinations
import seaborn as sns
import imageio
import random
import re
import os

sns.set()


pip install pandas numpy matplotlib imageio


Dir = "/kaggle/input/cmi-detect-behavior-with-sensor-data"


df_train = pd.read_csv(f"{Dir}/train.csv") 
df_demographic = pd.read_csv(f"{Dir}/train_demographics.csv")


df_demographic.head()


df_train.head()


df_train.shape



df_train.info()


df_train.describe(include='all')


# We know that all records within a single sequence have the same 'sequence_type'.
# This means we don't need to check every row. We can look at just one row per sequence.
unique_sequences_df = df_train.drop_duplicates(subset=['sequence_id'])
# Now, unique_sequences_df contains exactly one row for each of the 8151 sequences.


df_train.gesture.value_counts()


df_train.phase.value_counts()


df_train.behavior.value_counts()


# Specify the columns 
columns_of_interest = ['rot_x', 'thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5', 'tof_1_v10', 'tof_2_v10', 'tof_3_v10', 'tof_4_v10', 'tof_5_v10'] 

# Calculate and print missing value counts for these columns
print("--- Missing values for specific columns ---")
for col in columns_of_interest:
    missing_count = df_train[col].isnull().sum()
    print(f"{col}: {missing_count}")



# 1. Calculate the number of missing values for each column
missing_values_per_column = df_train.isnull().sum()

# 2. Filter and display only the columns that have at least one missing value
columns_with_missing_values = missing_values_per_column[missing_values_per_column > 0].sort_values(ascending=False)

# 3. Calculate the total number of missing values in the entire DataFrame
total_missing_values = missing_values_per_column.sum()

# 4. Calculate the total percentage of missing values relative to the entire dataset
total_cells = np.product(df_train.shape)
percentage_missing = (total_missing_values / total_cells) * 100

# Print the Results
print("--- Missing Values Report ---")
print("")

if total_missing_values > 0:
    print(f"âœ… Total number of missing values in the dataset: {total_missing_values}")
    print(f"ğŸ“Š Total percentage of missing data: {percentage_missing:.2f}%")
    print("---------------------------------------------")
    print("Number of missing values per column (only for columns with missing data):")
    print(columns_with_missing_values)


# --- Sequence count per subject analysis ---
# This analysis shows how many unique sequences each subject has performed.

# Use unique_sequences_df (already created, one row per sequence)
# Group by 'subject' and count the number of occurrences (i.e., sequences) for each subject.
sequences_per_subject = unique_sequences_df['subject'].value_counts().sort_index()

# --- Visualization ---
plt.figure(figsize=(18, 6))

# Barplot: x-axis is subject ID, y-axis is number of sequences for that subject
sns.barplot(x=sequences_per_subject.index, y=sequences_per_subject.values, color='skyblue')

plt.title('Number of Unique Sequences per Subject', fontsize=16, pad=15)
plt.xlabel('Subject ID', fontsize=12)
plt.ylabel('Number of Sequences', fontsize=12)
plt.xticks(rotation=90, fontsize=9)  # Rotate subject IDs for readability

# Add the count above each bar for clarity (optional, remove if plot is crowded)
# for i, val in enumerate(sequences_per_subject.values):
#     plt.text(i, val + 0.5, str(val), ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()



# Randomly select a subject_id from the dataframe
subject_id = random.choice(unique_sequences_df["subject"].unique())

# Filter the dataframe for the selected subject
subject_sequences = unique_sequences_df[unique_sequences_df['subject'] == subject_id]

# Create a pivot table: Rows = gestures, Columns = orientations, Values = sequence count
pivot_table = subject_sequences.pivot_table(
    index='gesture', 
    columns='orientation', 
    values='sequence_id',  # any column with unique values per sequence works
    aggfunc='count',
    fill_value=0
)

# Display the result
print(f" ------------------------------ Sequence Count Table for {subject_id} : ------------------------------ ")
display(pivot_table)


# --- Analysis: Count Unique Sequences ---
# The 'sequence_id' column identifies each unique sequence.
num_unique_sequences = df_train['sequence_id'].nunique()

# --- Display the Result ---
# Print the result in a clear, formatted string.
print(f"Total number of records (rows): {len(df_train)}")
print("-" * 47) # A separator for better readability
print(f"Number of unique sequences in the dataset: {num_unique_sequences}")


# --- Distribution of Sequence Counts per Subject ---
# Group by 'subject' to count the number of unique sequences each subject participated in
sequence_length = df_train.groupby('sequence_id').size()
length_counts = sequence_length.value_counts().sort_index()

# --- Visualization ---
plt.figure(figsize=(20,5))
# Histogram: Number of sequences per user
length_counts.plot(kind='bar')
plt.title("Distribution of Sequence length")
plt.xlabel("Sequence length")
plt.ylabel("Number of Sequence")
plt.grid(True)
plt.tight_layout()
plt.show()


length_counts


length_counts.sort_values()


# --- Analysis: Count Sequence Types (Target vs. Non-Target) ---
# W defined unique_sequences_df befor.
# We can simply count the values in the 'sequence_type' column of this new DataFrame.
sequence_type_counts = unique_sequences_df['sequence_type'].value_counts()


# --- Visualization ---
# A bar chart is a great way to visualize this distribution.
plt.figure(figsize=(8, 6)) # Create a figure with a specific size for better visualization

# Create the bar plot using the calculated counts.
# The index of the Series (e.g., 'Target', 'Non-Target') becomes the x-axis labels.
# The values of the Series become the height of the bars.
bars = plt.bar(sequence_type_counts.index, sequence_type_counts.values, color=['#1a80bb', 'salmon'])

# Add labels and a title to make the plot informative.
plt.title('Distribution of Sequence Types', fontsize=16)
plt.xlabel('Sequence Type', fontsize=12)
plt.ylabel('Number of Unique Sequences', fontsize=12)
plt.xticks(rotation=0) # Keep the x-axis labels horizontal for readability

# Add the exact count on top of each bar for clarity.
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center', fontsize=12) # va='bottom' places text just above the bar

plt.tight_layout() 
plt.show() # Display the plot



# --- Analysis: Count Sequences per Gesture ---
# This part of the code remains the same.
# We still need the counts for all gestures to plot them.

# Step 1: Group by the gesture and count the number of sequences in each group.
# The value_counts() method on the 'gesture' column does this directly and efficiently.
gesture_counts = unique_sequences_df['gesture'].value_counts()

# --- Visualization with Two Colors ---
# Step 2: Create a color mapping for the plot.
# We need to create a list of colors where each color corresponds to a gesture in our gesture_counts series.

# First, get the type for each gesture. We can do this by creating a mapping from gesture to sequence_type.
# drop_duplicates() ensures we have a clean mapping of each gesture to its single type.
gesture_to_type_map = unique_sequences_df.drop_duplicates(subset=['gesture'])[['gesture', 'sequence_type']].set_index('gesture')['sequence_type']

# Now, create the color list based on the order of gestures in 'gesture_counts.index'.
colors = ['#1a80bb' if gesture_to_type_map[gesture] == 'Target' else 'salmon' for gesture in gesture_counts.index]
# We've chosen a standard blue for 'Target' and orange for 'Non-Targett'.

# Step 3: Plot the distribution using the new color mapping.
plt.figure(figsize=(14, 8)) # A slightly wider figure for better readability.

# Using Seaborn's barplot, but passing our custom color list to the 'palette' argument.
ax = sns.barplot(x=gesture_counts.index, y=gesture_counts.values, palette=colors)

# Step 4: Add labels, title, and rotate x-axis labels.
plt.title('Number of Unique Sequences per Gesture (Colored by Type)', fontsize=18, pad=20)
plt.xlabel('Gesture', fontsize=14)
plt.ylabel('Number of Unique Sequences', fontsize=14)
plt.xticks(rotation=45, ha='right') # Rotate labels to prevent them from overlapping.

# Step 5: Add the count value on top of each bar for clarity.
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 9), 
                textcoords='offset points')

# Step 6: Create and add a custom legend to explain the colors.
target_patch = mpatches.Patch(color='#1a80bb', label='Target')
nontarget_patch = mpatches.Patch(color='salmon', label='Non-Target')
plt.legend(handles=[target_patch, nontarget_patch], title="Sequence Type", fontsize='large')

plt.tight_layout() 
plt.show()


# --- Gestures List for choosing as Target for comparison.---
# <Text on phone>          -----   <Neck - scratch>
# <Forehead - scratch>     -----   <Eyebrow - pull hair>
# <Above ear - pull hair>  -----   <Forehead - pull hairline>
# <Neck - pinch skin>      -----   <Feel around in tray and pull out an object>
# <Eyelash - pull hair>    -----   <Pull air toward your face>
# <Cheek - pinch skin>     -----   <Drink from bottle/cup>
# <Wave hello>             -----   <Write name on leg>
# <Write name in air>      -----   <Pinch knee/leg skin>
# <Glasses on/off>         -----   <Scratch knee/leg skin>


def find_comparison_pairs(df):
    """
    Searches the dataframe to find pairs of sequences suitable for Comparison Type 1:
    - Same Gesture
    - Same Orientation
    - Different Subject
    """
    print("Step 1: Searching for suitable sequence pairs...")
    
    # Optimize by using only necessary columns and removing duplicates
    relevant_cols_df = df[['sequence_id', 'subject', 'gesture', 'orientation']].drop_duplicates()
    
    grouped = relevant_cols_df.groupby(['gesture', 'orientation'])
    valid_pairs = []
    
    for (gesture, orientation), group_df in grouped:
        # If there are fewer than 2 unique subjects in this group, it cannot be used for comparison.
        if group_df['subject'].nunique() < 2:
            continue
            
        # Map subjects to their sequences within this group
        subject_to_sequences = group_df.groupby('subject')['sequence_id'].apply(list).to_dict()
        subjects_in_group = list(subject_to_sequences.keys())
        
        # Create all possible pairs of subjects
        subject_pairs = combinations(subjects_in_group, 2)
        
        # For each subject pair, create a sequence pair
        for sub1, sub2 in subject_pairs:
            # For simplicity, we take the first sequence from each subject
            seq1 = subject_to_sequences[sub1][0]
            seq2 = subject_to_sequences[sub2][0]
            
            valid_pairs.append({
                'gesture': gesture, 'orientation': orientation,
                'sequence_1': seq1, 'subject_1': sub1,
                'sequence_2': seq2, 'subject_2': sub2
            })
            
    print(f"Search complete. Found {len(valid_pairs)} suitable pairs for comparison.")
    return pd.DataFrame(valid_pairs)



def plot_single_sequence(ax, sequence_id, df_train, df_demographic):
    """
    Plots a single sequence on a given matplotlib axis (ax).
    """
    sequence_df = df_train[df_train['sequence_id'] == sequence_id].copy()
    sequence_df.sort_values('sequence_counter', inplace=True)
    
    if sequence_df.empty:
        ax.text(0.5, 0.5, f"Sequence ID not found:{sequence_id}", ha='center', va='center')
        return
    
    # Extract metadata for titles and annotations
    subject_id = sequence_df['subject'].iloc[0]
    gesture = sequence_df['gesture'].iloc[0]
    orientation = sequence_df['orientation'].iloc[0]
    
    # Get demographic info for the subject
    subject_info = df_demographic[df_demographic['subject'] == subject_id]
    age = subject_info['age'].iloc[0] if not subject_info.empty else "N/A"
    sex_str = "Male" if not subject_info.empty and subject_info['sex'].iloc[0] == 1 else "Female"
    
    # Plot acceleration data
    sns.lineplot(data=sequence_df, x='sequence_counter', y='acc_x', label='acc_x', ax=ax, alpha=0.9)
    sns.lineplot(data=sequence_df, x='sequence_counter', y='acc_y', label='acc_y', ax=ax, alpha=0.9)
    sns.lineplot(data=sequence_df, x='sequence_counter', y='acc_z', label='acc_z', ax=ax, alpha=0.9)
    
    # Highlight behavior regions with colored backgrounds
    unique_behaviors = sequence_df['behavior'].unique()
    colors = sns.color_palette("husl", len(unique_behaviors))
    behavior_color_map = dict(zip(unique_behaviors, colors))
    
    start_time = sequence_df['sequence_counter'].iloc[0]
    current_behavior = sequence_df['behavior'].iloc[0]
    for i in range(1, len(sequence_df)):
        if sequence_df['behavior'].iloc[i] != current_behavior:
            end_time = sequence_df['sequence_counter'].iloc[i-1]
            ax.axvspan(start_time, end_time, facecolor=behavior_color_map[current_behavior], alpha=0.2)
            current_behavior = sequence_df['behavior'].iloc[i]
            start_time = sequence_df['sequence_counter'].iloc[i]
    # Add the final behavior span
    ax.axvspan(start_time, sequence_df['sequence_counter'].iloc[-1], facecolor=behavior_color_map[current_behavior], alpha=0.2)

    # Set titles and labels
    ax.set_title(f"Gesture: {gesture} | Orientation: {orientation}Subject: {subject_id}", fontsize=14, weight='bold')
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Acceleration', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add demographic info text below the plot
    demographic_text = f"Subject Info: Age={age}, Sex={sex_str}"
    ax.text(0.5, -0.25, demographic_text, ha='center', transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle="round,pad=0.4", fc="lightgray", alpha=0.8))


def compare_gesture_across_subjects(sequence_id_1, sequence_id_2, df_train, df_demographic):
    """
    Compares the same gesture performed by two different subjects.
    Plots are arranged vertically.
    """
    sns.set_theme(style="whitegrid")
    
    # --- MODIFICATION: Changed layout to 2 rows, 1 column and adjusted figsize ---
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18, 12)) # Changed figsize for better vertical aspect
    
    # Plot each sequence on its respective axis
    plot_single_sequence(axes[0], sequence_id_1, df_train, df_demographic)
    plot_single_sequence(axes[1], sequence_id_2, df_train, df_demographic)
    
    # Add a main title to the figure
    gesture = df_train[df_train['sequence_id'] == sequence_id_1]['gesture'].iloc[0]
    fig.suptitle(f"Comparison 1: Same Gesture ('{gesture}') by Different Subjects", fontsize=20, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to make space for suptitle
    plt.show()


# Call the function to find all possible pairs
comparison1_pairs_df = find_comparison_pairs(df_train)

# Check if any pairs were found
if not comparison1_pairs_df.empty:
    print("Step 2: Selecting a pair to plot...")
    
    # Example: Select a pair for the 'Typing' gesture
    # You can replace 'Cheek - pinch skin' with any other gesture.
    target_gesture = 'Cheek - pinch skin'
    selected_pairs = comparison1_pairs_df[comparison1_pairs_df['gesture'] == target_gesture]
    
    if not selected_pairs.empty:
        # Select the first pair found for the target gesture
        pair_to_plot = selected_pairs.iloc[0]
        
        seq_id_1 = pair_to_plot['sequence_1']
        subj_id_1 = pair_to_plot['subject_1']
        seq_id_2 = pair_to_plot['sequence_2']
        subj_id_2 = pair_to_plot['subject_2']
        
        print(f"Plotting the selected pair:")
        print(f"  - Gesture: {pair_to_plot['gesture']}")
        print(f"  - Orientation: {pair_to_plot['orientation']}")
        print(f"  - Sequence 1: {seq_id_1} (from {subj_id_1})")
        print(f"  - Sequence 2: {seq_id_2} (from {subj_id_2})")
        
        print("Step 3: Generating the comparison plot...")
        # Call the plotting function with the selected sequence IDs
        compare_gesture_across_subjects(seq_id_1, seq_id_2, df_train, df_demographic)
        
    else:
        print(f"No pairs were found for the gesture '{target_gesture}'. Please try another gesture.")
        # As a fallback, plot the very first pair from the full list
        print("Plotting the first available pair from the list as a fallback...")
        pair_to_plot = comparison1_pairs_df.iloc[0]
        seq_id_1 = pair_to_plot['sequence_1']
        seq_id_2 = pair_to_plot['sequence_2']
        compare_gesture_across_subjects(seq_id_1, seq_id_2, df_train, df_demographic)
        
else:
    print("No sequence pairs suitable for Comparison Type 1 were found in the dataset.")



def find_comparison_pairs_type2(df):
    """
    Searches the dataframe to find pairs of sequences suitable for Comparison Type 2:
    - Same Subject
    - Same Gesture
    - Different Orientation
    """
    print("Step 1: Searching for suitable sequence pairs for Comparison Type 2...")
    
    # Optimize by using only necessary columns and removing duplicates
    relevant_cols_df = df[['sequence_id', 'subject', 'gesture', 'orientation']].drop_duplicates()
    
    # Group by subject and gesture to find cases with multiple orientations
    grouped = relevant_cols_df.groupby(['subject', 'gesture'])
    valid_pairs = []
    
    for (subject, gesture), group_df in grouped:
        # If a subject has performed a gesture in fewer than 2 orientations, they can't be compared.
        if group_df['orientation'].nunique() < 2:
            continue
            
        # Get all sequences for this subject/gesture combination
        all_sequences_in_group = group_df[['sequence_id', 'orientation']].to_dict('records')
        
        # Create all possible pairs of sequences (which have different orientations by definition of the group)
        sequence_pairs = combinations(all_sequences_in_group, 2)
        
        for seq1_info, seq2_info in sequence_pairs:
            # Ensure the orientations are actually different, just in case of data duplication
            if seq1_info['orientation'] != seq2_info['orientation']:
                valid_pairs.append({
                    'subject': subject,
                    'gesture': gesture,
                    'sequence_1': seq1_info['sequence_id'],
                    'orientation_1': seq1_info['orientation'],
                    'sequence_2': seq2_info['sequence_id'],
                    'orientation_2': seq2_info['orientation'],
                })
            
    print(f"Search complete. Found {len(valid_pairs)} suitable pairs for comparison.")
    return pd.DataFrame(valid_pairs)



def plot_single_sequence(ax, sequence_id, df_train, df_demographic):
    """
    Plots a single sequence on a given matplotlib axis (ax).
    This function remains unchanged from the previous version.
    """
    sequence_df = df_train[df_train['sequence_id'] == sequence_id].copy()
    sequence_df.sort_values('sequence_counter', inplace=True)
    
    if sequence_df.empty:
        ax.text(0.5, 0.5, f"Sequence ID not found:{sequence_id}", ha='center', va='center')
        return
    
    # Extract metadata for titles and annotations
    subject_id = sequence_df['subject'].iloc[0]
    gesture = sequence_df['gesture'].iloc[0]
    orientation = sequence_df['orientation'].iloc[0]
    
    # Get demographic info for the subject
    subject_info = df_demographic[df_demographic['subject'] == subject_id]
    age = subject_info['age'].iloc[0] if not subject_info.empty else "N/A"
    sex_str = "Male" if not subject_info.empty and subject_info['sex'].iloc[0] == 1 else "Female"
    
    # Plot acceleration data
    sns.lineplot(data=sequence_df, x='sequence_counter', y='acc_x', label='acc_x', ax=ax, alpha=0.9)
    sns.lineplot(data=sequence_df, x='sequence_counter', y='acc_y', label='acc_y', ax=ax, alpha=0.9)
    sns.lineplot(data=sequence_df, x='sequence_counter', y='acc_z', label='acc_z', ax=ax, alpha=0.9)
    
    # Highlight behavior regions with colored backgrounds
    unique_behaviors = sequence_df['behavior'].unique()
    colors = sns.color_palette("husl", len(unique_behaviors))
    behavior_color_map = dict(zip(unique_behaviors, colors))
    
    start_time = sequence_df['sequence_counter'].iloc[0]
    current_behavior = sequence_df['behavior'].iloc[0]
    for i in range(1, len(sequence_df)):
        if sequence_df['behavior'].iloc[i] != current_behavior:
            end_time = sequence_df['sequence_counter'].iloc[i-1]
            ax.axvspan(start_time, end_time, facecolor=behavior_color_map[current_behavior], alpha=0.2)
            current_behavior = sequence_df['behavior'].iloc[i]
            start_time = sequence_df['sequence_counter'].iloc[i]
    ax.axvspan(start_time, sequence_df['sequence_counter'].iloc[-1], facecolor=behavior_color_map[current_behavior], alpha=0.2)

    # Set titles and labels
    ax.set_title(f"Gesture: {gesture} | Orientation: {orientation}Subject: {subject_id}", fontsize=14, weight='bold')
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Acceleration', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add demographic info text below the plot
    demographic_text = f"Subject Info: Age={age}, Sex={sex_str}"
    ax.text(0.5, -0.25, demographic_text, ha='center', transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle="round,pad=0.4", fc="lightgray", alpha=0.8))


def compare_gesture_across_orientations(sequence_id_1, sequence_id_2, df_train, df_demographic):
    """
    Compares the same gesture in two different orientations by the same subject.
    Plots are arranged vertically.
    """
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18, 12))
    
    plot_single_sequence(axes[0], sequence_id_1, df_train, df_demographic)
    plot_single_sequence(axes[1], sequence_id_2, df_train, df_demographic)
    
    # Add a main title to the figure
    subject_id = df_train[df_train['sequence_id'] == sequence_id_1]['subject'].iloc[0]
    gesture = df_train[df_train['sequence_id'] == sequence_id_1]['gesture'].iloc[0]
    fig.suptitle(f"Comparison 2: Same Gesture ('{gesture}') by One Subject ('{subject_id}')with Different Orientations", fontsize=20, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust rect to make space for suptitle
    plt.show()


# Call the function to find all possible pairs for Comparison Type 2
comparison2_pairs_df = find_comparison_pairs_type2(df_train)

# Check if any pairs were found
if not comparison2_pairs_df.empty:
    print("Step 2: Selecting a pair to plot...")
    
    # Example: Select a pair for the 'Forehead - scratch' gesture
    target_gesture = 'Eyebrow - pull hair'
    selected_pairs = comparison2_pairs_df[comparison2_pairs_df['gesture'] == target_gesture]
    
    if not selected_pairs.empty:
        # Select the first pair found for the target gesture
        pair_to_plot = selected_pairs.iloc[0]
        
        seq_id_1 = pair_to_plot['sequence_1']
        orientation_1 = pair_to_plot['orientation_1']
        seq_id_2 = pair_to_plot['sequence_2']
        orientation_2 = pair_to_plot['orientation_2']
        
        print(f"Plotting the selected pair:")
        print(f"  - Subject: {pair_to_plot['subject']}")
        print(f"  - Gesture: {pair_to_plot['gesture']}")
        print(f"  - Sequence 1: {seq_id_1} (Orientation: {orientation_1})")
        print(f"  - Sequence 2: {seq_id_2} (Orientation: {orientation_2})")
        
        print("Step 3: Generating the comparison plot...")
        # Call the plotting function with the selected sequence IDs
        compare_gesture_across_orientations(seq_id_1, seq_id_2, df_train, df_demographic)
        
    else:
        print(f"No pairs were found for the gesture '{target_gesture}'. Please try another gesture.")
        # As a fallback, plot the very first pair from the full list
        print("Plotting the first available pair from the list as a fallback...")
        pair_to_plot = comparison2_pairs_df.iloc[0]
        seq_id_1 = pair_to_plot['sequence_1']
        seq_id_2 = pair_to_plot['sequence_2']
        compare_gesture_across_orientations(seq_id_1, seq_id_2, df_train, df_demographic)
        
else:
    print("No sequence pairs suitable for Comparison Type 2 were found in the dataset.")


def find_comparison_pairs_type3(df):
    """
    Searches the dataframe to find pairs of sequences suitable for Comparison Type 3:
    - Same Subject
    - Same Gesture
    - Same Orientation
    (This checks for repeated recordings of the same activity by the same person.)
    """
    print("Step 1: Searching for suitable sequence pairs for Comparison Type 3...")
    
    # Optimize by using only necessary columns and removing duplicates
    relevant_cols_df = df[['sequence_id', 'subject', 'gesture', 'orientation']].drop_duplicates()
    
    # Group by subject, gesture, AND orientation to find repetitions
    grouped = relevant_cols_df.groupby(['subject', 'gesture', 'orientation'])
    valid_pairs = []
    
    for (subject, gesture, orientation), group_df in grouped:
        # If a subject has performed a specific gesture/orientation combo fewer than 2 times,
        # we cannot compare repetitions.
        if len(group_df) < 2:
            continue
            
        # Get all sequence_ids for this specific combination
        sequence_ids = group_df['sequence_id'].tolist()
        
        # Create all possible pairs of these repeated sequences
        sequence_pairs = combinations(sequence_ids, 2)
        
        for seq1, seq2 in sequence_pairs:
            valid_pairs.append({
                'subject': subject,
                'gesture': gesture,
                'orientation': orientation,
                'sequence_1': seq1,
                'sequence_2': seq2,
            })
            
    print(f"Search complete. Found {len(valid_pairs)} suitable pairs for comparison.")
    return pd.DataFrame(valid_pairs)


def plot_single_sequence(ax, sequence_id, df_train, df_demographic):
    """
    Plots a single sequence on a given matplotlib axis (ax).
    This function remains unchanged.
    """
    sequence_df = df_train[df_train['sequence_id'] == sequence_id].copy()
    sequence_df.sort_values('sequence_counter', inplace=True)
    
    if sequence_df.empty:
        ax.text(0.5, 0.5, f"Sequence ID not found:{sequence_id}", ha='center', va='center')
        return
    
    # Extract metadata for titles and annotations
    subject_id = sequence_df['subject'].iloc[0]
    gesture = sequence_df['gesture'].iloc[0]
    orientation = sequence_df['orientation'].iloc[0]
    
    # Get demographic info for the subject
    subject_info = df_demographic[df_demographic['subject'] == subject_id]
    age = subject_info['age'].iloc[0] if not subject_info.empty else "N/A"
    sex_str = "Male" if not subject_info.empty and subject_info['sex'].iloc[0] == 1 else "Female"
    
    # Plot acceleration data
    sns.lineplot(data=sequence_df, x='sequence_counter', y='acc_x', label='acc_x', ax=ax, alpha=0.9)
    sns.lineplot(data=sequence_df, x='sequence_counter', y='acc_y', label='acc_y', ax=ax, alpha=0.9)
    sns.lineplot(data=sequence_df, x='sequence_counter', y='acc_z', label='acc_z', ax=ax, alpha=0.9)
    
    # Highlight behavior regions
    unique_behaviors = sequence_df['behavior'].unique()
    colors = sns.color_palette("husl", len(unique_behaviors))
    behavior_color_map = dict(zip(unique_behaviors, colors))
    
    start_time = sequence_df['sequence_counter'].iloc[0]
    current_behavior = sequence_df['behavior'].iloc[0]
    for i in range(1, len(sequence_df)):
        if sequence_df['behavior'].iloc[i] != current_behavior:
            end_time = sequence_df['sequence_counter'].iloc[i-1]
            ax.axvspan(start_time, end_time, facecolor=behavior_color_map[current_behavior], alpha=0.2)
            current_behavior = sequence_df['behavior'].iloc[i]
            start_time = sequence_df['sequence_counter'].iloc[i]
    ax.axvspan(start_time, sequence_df['sequence_counter'].iloc[-1], facecolor=behavior_color_map[current_behavior], alpha=0.2)

    # Set titles and labels
    ax.set_title(f"Gesture: {gesture} | Orientation: {orientation}Sequence ID: {sequence_id}", fontsize=14, weight='bold')
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Acceleration', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add demographic info text
    demographic_text = f"Subject Info: ID={subject_id}, Age={age}, Sex={sex_str}"
    ax.text(0.5, -0.25, demographic_text, ha='center', transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle="round,pad=0.4", fc="lightgray", alpha=0.8))


def compare_repeated_gestures(sequence_id_1, sequence_id_2, df_train, df_demographic):
    """
    Compares two instances of the same gesture, by the same person, in the same orientation.
    Plots are arranged vertically.
    """
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18, 12))
    
    plot_single_sequence(axes[0], sequence_id_1, df_train, df_demographic)
    plot_single_sequence(axes[1], sequence_id_2, df_train, df_demographic)
    
    # Extract info for the main title
    info = df_train[df_train['sequence_id'] == sequence_id_1].iloc[0]
    subject_id, gesture, orientation = info['subject'], info['gesture'], info['orientation']
    
    fig.suptitle(f"Comparison 3: Repeated Gesture ('{gesture}') by One Subject ('{subject_id}')(Same Orientation: {orientation})", fontsize=20, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# Call the function to find all possible pairs for Comparison Type 3
comparison3_pairs_df = find_comparison_pairs_type3(df_train)

# Check if any pairs were found
if not comparison3_pairs_df.empty:
    print("Step 2: Selecting a RANDOM pair to plot...")
    
    # Instead of picking the first pair with .iloc[0], we select one RANDOM pair.
    pair_to_plot = comparison3_pairs_df.sample(1)
    
    # Extracting sequence IDs from the randomly selected pair's dataframe row.
    # .iloc[0] is used here because pair_to_plot is a dataframe with a single random row.
    seq_id_1 = pair_to_plot['sequence_1'].iloc[0]
    seq_id_2 = pair_to_plot['sequence_2'].iloc[0]
    
    print(f"Plotting the randomly selected pair:")
    print(f"  - Subject: {pair_to_plot['subject'].iloc[0]}")
    print(f"  - Gesture: {pair_to_plot['gesture'].iloc[0]}")
    print(f"  - Orientation: {pair_to_plot['orientation'].iloc[0]}")
    print(f"  - Sequence 1: {seq_id_1}")
    print(f"  - Sequence 2: {seq_id_2}")
    
    print("Step 3: Generating the comparison plot...")
    # Call the plotting function with the selected sequence IDs
    compare_repeated_gestures(seq_id_1, seq_id_2, df_train, df_demographic)


def find_comparison_pairs_type4(df_train, df_demographic):
    """
    Searches for pairs of sequences suitable for Comparison Type 4:
    - Same Gesture
    - Same Orientation
    - Performed by two different subjects
    - Both subjects must be Adults
    - One subject must be Male, the other Female
    """    
    # Merge train and demographic data to have all info in one place
    # We only need sequence_id and the grouping keys from df_train
    merged_df = pd.merge(
        df_train[['sequence_id', 'subject', 'gesture', 'orientation']].drop_duplicates(),
        df_demographic,
        on='subject'
    )
    
    # Filter for adults only, as this is a core requirement
    adults_df = merged_df[merged_df['adult_child'] == 1].copy()
    
    # Separate adults into male and female groups to simplify pairing
    male_adults_df = adults_df[adults_df['sex'] == 1]
    female_adults_df = adults_df[adults_df['sex'] == 0]
    
    valid_pairs = []
    
    # Iterate through each male sequence
    for _, male_row in male_adults_df.iterrows():
        # Find all matching female sequences (same gesture and orientation)
        matching_females = female_adults_df[
            (female_adults_df['gesture'] == male_row['gesture']) &
            (female_adults_df['orientation'] == male_row['orientation'])
        ]
        
        # If matches are found, create a pair for each match
        for _, female_row in matching_females.iterrows():
            valid_pairs.append({
                'gesture': male_row['gesture'],
                'orientation': male_row['orientation'],
                'subject_male': male_row['subject'],
                'sequence_male': male_row['sequence_id'],
                'age_male': male_row['age'],
                'subject_female': female_row['subject'],
                'sequence_female': female_row['sequence_id'],
                'age_female': female_row['age']
            })
            
    result_df = pd.DataFrame(valid_pairs)
    print(f"Search complete. Found {len(result_df)} suitable pairs for comparison.")
    return result_df


def plot_single_sequence(ax, sequence_id, df_train, df_demographic):
    """
    Plots a single sequence on a given matplotlib axis (ax).
    This function is reused from previous comparisons.
    """
    sequence_df = df_train[df_train['sequence_id'] == sequence_id].copy()
    sequence_df.sort_values('sequence_counter', inplace=True)
    
    if sequence_df.empty:
        ax.text(0.5, 0.5, f"Sequence ID not found:\n{sequence_id}", ha='center', va='center')
        return
    
    # Extract metadata for titles
    subject_id = sequence_df['subject'].iloc[0]
    gesture = sequence_df['gesture'].iloc[0]
    orientation = sequence_df['orientation'].iloc[0]
    
    # Get demographic info
    subject_info = df_demographic[df_demographic['subject'] == subject_id]
    age = subject_info['age'].iloc[0] if not subject_info.empty else "N/A"
    sex_str = "Male" if not subject_info.empty and subject_info['sex'].iloc[0] == 1 else "Female"
    
    # Plot acceleration data
    sns.lineplot(data=sequence_df, x='sequence_counter', y='acc_x', label='acc_x', ax=ax, alpha=0.9)
    sns.lineplot(data=sequence_df, x='sequence_counter', y='acc_y', label='acc_y', ax=ax, alpha=0.9)
    sns.lineplot(data=sequence_df, x='sequence_counter', y='acc_z', label='acc_z', ax=ax, alpha=0.9)
    
    # Set titles and labels
    ax.set_title(f"Subject: {subject_id} ({sex_str}, Age: {age})\nSequence ID: {sequence_id}", fontsize=14, weight='bold')
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Acceleration', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)


def compare_male_vs_female_gesture(male_seq_id, female_seq_id, df_train, df_demographic):
    """
    Compares two sequences (one male, one female) for the same gesture and orientation.
    Plots are arranged vertically.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18, 12))
    
    # Plot the two sequences
    plot_single_sequence(axes[0], male_seq_id, df_train, df_demographic)
    plot_single_sequence(axes[1], female_seq_id, df_train, df_demographic)
    
    # Extract info for the main title from one of the sequences (gesture/orientation are the same)
    info = df_train[df_train['sequence_id'] == male_seq_id].iloc[0]
    gesture, orientation = info['gesture'], info['orientation']
    
    fig.suptitle(f"Comparison 4: Male vs. Female Performing Gesture '{gesture}'\n(Same Orientation: {orientation}, Both Adults)", fontsize=20, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# Call the function to find all possible pairs for Comparison Type 4
comparison4_pairs_df = find_comparison_pairs_type4(df_train, df_demographic)

# Check if any pairs were found
if not comparison4_pairs_df.empty:
    print("\nStep 2: Selecting a RANDOM pair to plot...")
    
    # Randomly sample one pair from the DataFrame of all valid pairs
    pair_to_plot = comparison4_pairs_df.sample(1)
    
    # Extract the sequence IDs from the single-row DataFrame
    seq_id_male = pair_to_plot['sequence_male'].iloc[0]
    seq_id_female = pair_to_plot['sequence_female'].iloc[0]
    
    print("Plotting the randomly selected pair:")
    print(f"  - Common Gesture: {pair_to_plot['gesture'].iloc[0]}")
    print(f"  - Common Orientation: {pair_to_plot['orientation'].iloc[0]}")
    print("-" * 20)
    print(f"  - Male Subject: {pair_to_plot['subject_male'].iloc[0]} (Age: {pair_to_plot['age_male'].iloc[0]}) | Sequence: {seq_id_male}")
    print(f"  - Female Subject: {pair_to_plot['subject_female'].iloc[0]} (Age: {pair_to_plot['age_female'].iloc[0]}) | Sequence: {seq_id_female}")
    
    print("\nStep 3: Generating the comparison plot...")
    # Call the plotting function with the selected sequence IDs
    compare_male_vs_female_gesture(seq_id_male, seq_id_female, df_train, df_demographic)


# Step 1: Configuration and Data Preparation
features_to_plot = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
sensor_name = "Rotation (Quaternion)"
target_sequence_id = 'SEQ_035116'
target_sequence_id = random.choice(df_train["sequence_id"].unique())

sequence_df = df_train[df_train['sequence_id'] == target_sequence_id].copy()
sequence_df.sort_values('sequence_counter', inplace=True)

# Step 2: Extract Metadata for Title and Footer
subject_id = sequence_df['subject'].iloc[0]
gesture = sequence_df['gesture'].iloc[0]
orientation = sequence_df['orientation'].iloc[0]

# Look up demographic info for the subject
# Filter the demographic dataframe to find the row for our subject
subject_info = df_demographic[df_demographic['subject'] == subject_id]

# Check if the subject was found and extract the data
if not subject_info.empty:
    age = subject_info['age'].iloc[0]
    sex_code = subject_info['sex'].iloc[0]
    # Convert sex code to a human-readable string
    sex_str = "Male" if sex_code == 1 else "Female"

# Step 3: Create Subplots and Plot the Data -
fig, axes = plt.subplots(
    nrows=len(features_to_plot),
    ncols=1,
    figsize=(20, 5 * len(features_to_plot)),
    sharex=True
)

fig.suptitle(
    f"Sensor: {sensor_name}  |  Subject: {subject_id}  |  Gesture: {gesture}  |  Orientation: {orientation}",
    fontsize=20,
    fontweight='bold'
)

# Plotting Logic (No changes here)
behavior_changes = sequence_df['behavior'].ne(sequence_df['behavior'].shift())
change_indices = sequence_df.index[behavior_changes]
segment_starts = sequence_df.loc[change_indices, ['sequence_counter', 'behavior']].to_dict('records')
last_timestep = sequence_df['sequence_counter'].iloc[-1]
segment_starts.append({'sequence_counter': last_timestep + 1, 'behavior': 'END'})
unique_behaviors = sequence_df['behavior'].unique()
colors = sns.color_palette("viridis", len(unique_behaviors))
behavior_color_map = dict(zip(unique_behaviors, colors))

for i, feature in enumerate(features_to_plot):
    ax = axes[i]
    sns.lineplot(data=sequence_df, x='sequence_counter', y=feature, ax=ax, label=feature)

    for j in range(len(segment_starts) - 1):
        start_time = segment_starts[j]['sequence_counter']
        end_time = segment_starts[j+1]['sequence_counter'] - 1
        behavior_name = segment_starts[j]['behavior']
        ax.axvspan(
            start_time, end_time,
            facecolor=behavior_color_map[behavior_name], alpha=0.2
        )
        text_y_position = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.9
        ax.text(
            start_time + (end_time - start_time) / 2, text_y_position, behavior_name,
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc='white', ec="black", lw=1, alpha=0.7)
        )

    ax.set_ylabel(feature, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True)

axes[-1].set_xlabel('Time Step (sequence_counter)', fontsize=14)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Add the demographic information as a footer text
# Create the text string to display
demographic_text = f"Subject Information ==> Age: {age} | Sex: {sex_str}"

# Add the text to the figure.
# x=0.5 means horizontally centered.
# y=0.01 means slightly above the bottom of the figure.
# ha='center' ensures the text is centered on the x-coordinate.
fig.text(
    0.5, 0.01, demographic_text, color='navy',
    ha='center', fontsize=16, fontweight='bold',
    bbox=dict(boxstyle="round,pad=0.5", fc='lightgray', alpha=0.5)
)
plt.show()



fig.text(
    0.5, 0.01,  # x=center, y=bottom
    demographic_text,
    ha='center',
    fontsize=16,
    fontweight='bold',
    color='navy',
    bbox=dict(boxstyle="round,pad=0.5", fc="lightcyan", ec="black", lw=1)
)


# Step 1: Choose a Sequence ID and Prepare Data
target_sequence_id = 'SEQ_035116'
target_sequence_id = random.choice(df_train["sequence_id"].unique())

# Filter the DataFrame for the chosen sequence
sequence_df = df_train[df_train['sequence_id'] == target_sequence_id].copy()
sequence_df.sort_values('sequence_counter', inplace=True)

print(f"Analyzing sequence: {target_sequence_id}")
print(f"Number of data points in this sequence: {len(sequence_df)}")
print(f"Unique behaviors in this sequence: {sequence_df['behavior'].unique().tolist()}")


#  Step 2: Extract Metadata for Title and Footer
subject_id = sequence_df['subject'].iloc[0]
gesture = sequence_df['gesture'].iloc[0]
orientation = sequence_df['orientation'].iloc[0]
sensor_name = "Accelerometer" # Define the sensor name for the title

# --- Look up demographic info from df_demographic ---
# Find the row corresponding to our subject
subject_info = df_demographic[df_demographic['subject'] == subject_id]

# Extract age and sex, with a fallback for safety
if not subject_info.empty:
    age = subject_info['age'].iloc[0]
    sex_code = subject_info['sex'].iloc[0]
    # Convert the numeric sex code to a descriptive string
    sex_str = "Male" if sex_code == 1 else "Female"

# Step 3: Set up the Plot and Plot Accelerometer Data
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(20, 10)) # Increased height slightly for more space

# Plot all three accelerometer axes
sns.lineplot(data=sequence_df, x='sequence_counter', y='acc_x', label='acc_x', ax=ax)
sns.lineplot(data=sequence_df, x='sequence_counter', y='acc_y', label='acc_y', ax=ax)
sns.lineplot(data=sequence_df, x='sequence_counter', y='acc_z', label='acc_z', ax=ax)


# Step 4: Identify Behavior Changes and Visualize Segments
behavior_changes = sequence_df['behavior'].ne(sequence_df['behavior'].shift())
change_indices = sequence_df.index[behavior_changes]
segment_starts = sequence_df.loc[change_indices, ['sequence_counter', 'behavior']].to_dict('records')
last_timestep = sequence_df['sequence_counter'].iloc[-1]
segment_starts.append({'sequence_counter': last_timestep + 1, 'behavior': 'END'})
unique_behaviors = sequence_df['behavior'].unique()
colors = sns.color_palette("husl", len(unique_behaviors))
behavior_color_map = dict(zip(unique_behaviors, colors))

for i in range(len(segment_starts) - 1):
    start_time = segment_starts[i]['sequence_counter']
    end_time = segment_starts[i+1]['sequence_counter'] - 1
    behavior_name = segment_starts[i]['behavior']
    
    ax.axvspan(
        start_time, end_time,
        facecolor=behavior_color_map[behavior_name], alpha=0.2
    )
    
    text_x_position = start_time + (end_time - start_time) / 2
    text_y_position = ax.get_ylim()[1] * 0.9
    
    ax.text(
        text_x_position, text_y_position, behavior_name,
        ha='center', va='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc='yellow', ec="black", lw=1, alpha=0.7)
    )

# Step 5: Final Touches for Clarity
# Set a comprehensive main title (suptitle) for the entire figure
fig.suptitle(
    f'Sensor: {sensor_name} | Gesture: {gesture} | Orientation: {orientation}',
    fontsize=20,
    fontweight='bold'
)

# Set a specific title for the subplot itself
ax.set_title(
    f'Signals for sequence id: {target_sequence_id} | Subject: {subject_id}',
    fontsize=16,
    y=1.02 
)

ax.set_xlabel('Time Step (sequence_counter)', fontsize=14)
ax.set_ylabel('Acceleration Value', fontsize=14)
ax.legend(title='Sensor Axis')

# --- Add the demographic information below the plot ---
demographic_text = f"Subject information ==> Age: {age} | Sex: {sex_str}"

# Use fig.text to place text in figure coordinates (relative to the whole figure)
fig.text(
    0.5, 0.01,  # x=center, y=bottom
    demographic_text,
    ha='center',
    fontsize=16,
    fontweight='bold',
    color='navy',
    bbox=dict(boxstyle="round,pad=0.5", fc="lightcyan", ec="black", lw=1)
)

# Adjust layout to prevent elements from overlapping
plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Make space for footer and suptitle
plt.show()


# --- Calculate the length of each unique sequence ---
# Step 1: We group by 'sequence_id' and use .size() to count the number of rows (timesteps) for each sequence.
sequence_lengths = df_train.groupby('sequence_id').size().rename('sequence_length')

# We only need the gesture and type, and we'll set the index to 'sequence_id' to easily join it.
sequence_labels = unique_sequences_df.set_index('sequence_id')[['gesture', 'sequence_type']]

# Step 2: Combine sequence lengths with their labels 
df_lengths_with_labels = sequence_lengths.to_frame().join(sequence_labels)

# Step 3: Calculate the required statistics (min, mean, max)
length_stats = df_lengths_with_labels.groupby(['sequence_type', 'gesture']).agg(
    min_length=('sequence_length', 'min'),
    mean_length=('sequence_length', lambda x: x.mean().round(1)),
    max_length=('sequence_length', 'max')
).rename(columns={
    'min_length': 'Min Length',
    'mean_length': 'Mean Length',
    'max_length': 'Max Length'
})

# Step 4: Display the results 
# Displaying the results in a clean format.
print(" ------------ Sequence Length Statistics for Target Gestures ------------ ")
display(length_stats.loc['Target'])

print(" --------------------------- Sequence Length Statistics for Non-Target Gestures --------------------------- ")
display(length_stats.loc['Non-Target'])


    # Define the pattern for ToF column names using a regular expression
    # Pattern: starts with 'tof_', followed by a number from 1-5, then '_v', and finally one or two digits
    tof_pattern = re.compile(r'^tof_[1-5]_v\d{1,2}$')

    # Filter columns that match the defined pattern
    tof_columns = [col for col in df_train.columns if tof_pattern.match(col)]

    num_tof_columns = len(tof_columns)
    num_total_records = len(df_train)

    # Calculate the total number of cells
    total_tof_cells = num_tof_columns * num_total_records

    print("--- ToF Sensor Cell Count Analysis ---")
    print(f"Total number of rows (records) in the dataset: {num_total_records:,}")
    print(f"Number of columns identified for ToF sensors: {num_tof_columns}")
    print(f"Calculation: {num_tof_columns} (columns) * {num_total_records:,} (rows)")
    print("--------------------------------------------------")
    print(f"Total number of cells for ToF sensors: {total_tof_cells:,}")


    # --- Step 1: Identify ToF Columns ---
    tof_pattern = re.compile(r'^tof_[1-5]_v\d{1,2}$')
    tof_columns = [col for col in df_train.columns if tof_pattern.match(col)]
    
    # --- Step 2: Define variables for calculation ---
    num_total_records = len(df_train)
    num_tof_columns = len(tof_columns)
    total_tof_cells = num_tof_columns * num_total_records

    print("--- Analysis of -1 Values in ToF Sensor Columns ---")
    
    # --- Step 3: Efficiently Count -1 Values ---
    # Create a new DataFrame containing only the ToF columns.
    df_tof_only = df_train[tof_columns]

    # Convert the slice to a NumPy array for faster computation and then count the -1s.
    count_of_minus_one = (df_tof_only.values == -1).sum()
    
    # --- Step 4: Calculate Percentage and Display Results ---
    if total_tof_cells > 0:
        percentage_minus_one = (count_of_minus_one / total_tof_cells) * 100
    else:
        percentage_minus_one = 0

    print("--- Calculation Results ---")
    print(f"Total number of cells for ToF sensors: {total_tof_cells:,}")
    print(f"Number of cells with a value of -1: {count_of_minus_one:,}")
    print("--------------------------------------------------")
    print(f"Percentage of ToF cells that are -1: {percentage_minus_one:.4f}%")



total_minus_one_count = 105770214 

print("--- Breakdown of -1 Values per ToF Sensor ---")

# --- Step 1: Create a dictionary to store the results ---
# This is a clean way to organize the counts for each sensor.
sensor_minus_one_counts = {}

# --- Step 2: Loop through each of the 5 sensors ---
for i in range(1, 6): # This will loop through numbers 1, 2, 3, 4, 5
    sensor_id = f"tof_{i}"
    
    # Create a regular expression pattern to find all columns for this specific sensor.
    # e.g., for i=1, pattern becomes '^tof_1_v\d{1,2}$'
    sensor_pattern = re.compile(f'^{sensor_id}_v\d{{1,2}}$')
    
    # Get the list of column names for the current sensor.
    sensor_columns = [col for col in df_train.columns if sensor_pattern.match(col)]
    
    # Check if we actually found any columns for this sensor
    if not sensor_columns:
        print(f"Warning: No columns found for {sensor_id}. Skipping.")
        continue
        
    # --- Step 3: Efficiently count -1s for the current sensor ---
    # Select only the columns for the current sensor.
    df_sensor_only = df_train[sensor_columns]
    
    # Use the same efficient NumPy method to count -1s.
    count = (df_sensor_only.values == -1).sum()
    
    # Store the result in our dictionary.
    sensor_minus_one_counts[sensor_id] = count

# --- Step 4: Display the results in a clear table ---
print("--- Calculation Results ---")
print(f"{'Sensor ID':<15} | {'Count of -1 Values':<25} | {'Percentage of Total -1s':<25}")
print("-" * 75)

if total_minus_one_count > 0:
    for sensor_id, count in sensor_minus_one_counts.items():
        # Calculate what percentage of the *total -1s* comes from this sensor.
        percentage_of_total = (count / total_minus_one_count) * 100
        
        # Print a formatted row.
        print(f"{sensor_id:<15} | {count:<25,} | {percentage_of_total:.2f}%")
    
print("-" * 75)
sum_of_counts = sum(sensor_minus_one_counts.values())
print(f"{'Total (Sum of above)':<15} | {sum_of_counts:<25,}")
print(f"{'Original Total Count':<15} | {total_minus_one_count:<25,}")

if sum_of_counts == total_minus_one_count:
    print("Verification successful: The sum of individual sensor counts matches the overall total.")


tof_pattern = re.compile(r'^tof_[1-5]_v\d{1,2}$')
tof_columns = [col for col in df_train.columns if tof_pattern.match(col)]

print(f"Identified {len(tof_columns)} ToF columns for analysis.")
print("--- Analyzing sequences with 100% invalid ToF data ---")

# --- Check each row to see if ALL its ToF values are -1 ---
# We create a new boolean column 'all_tof_are_minus_one'.
df_train['all_tof_are_minus_one'] = (df_train[tof_columns].values == -1).all(axis=1)

# --- Group by sequence_id and check the condition ---
# We group the DataFrame by 'sequence_id'.
is_sequence_completely_invalid = df_train.groupby('sequence_id')['all_tof_are_minus_one'].all()

# --- Count the sequences that meet the condition and display results ---
# We can simply sum the boolean Series, as True is treated as 1 and False as 0.
num_invalid_sequences = is_sequence_completely_invalid.sum()

# Get the total number of unique sequences for context.
total_unique_sequences = df_train['sequence_id'].nunique()

# Calculate the percentage.
if total_unique_sequences > 0:
    percentage_invalid_sequences = (num_invalid_sequences / total_unique_sequences) * 100
else:
    percentage_invalid_sequences = 0

print("--- Results ---")
print(f"Total number of unique sequences in the dataset: {total_unique_sequences:,}")
print(f"Number of sequences where ALL ToF values are -1: {num_invalid_sequences:,}")
print("---------------------------------------------------------")
print(f"Percentage of sequences with completely invalid ToF data: {percentage_invalid_sequences:.2f}%")

# Optional: Display the IDs of the invalid sequences if there are not too many.
if 0 < num_invalid_sequences < 20:
    invalid_sequence_ids = is_sequence_completely_invalid[is_sequence_completely_invalid].index.tolist()
    print(f"IDs of the invalid sequences: {invalid_sequence_ids}")

# Cleanup: It's good practice to remove the temporary column we created.
df_train.drop(columns=['all_tof_are_minus_one'], inplace=True)


num_tof_columns = len(tof_columns)
print(f"Identified {num_tof_columns} ToF columns for analysis.")

# --- Define the threshold ---
threshold_percentage = 70.0
print(f"Searching for sequences with more than {threshold_percentage}% invalid ToF values...")

# --- Group by sequence_id and calculate stats ---
# First, calculate the count of -1s for each row, but only across ToF columns
df_train['temp_minus_one_count'] = (df_train[tof_columns] == -1).sum(axis=1)

# Now, group by sequence_id and aggregate the results
# '.size' gives the number of rows in each sequence.
# '.sum()' on our temp column gives the total number of -1s in each sequence.
sequence_stats = df_train.groupby('sequence_id').agg(
    row_count=('sequence_id', 'size'),
    total_minus_ones=('temp_minus_one_count', 'sum'))

# Clean up the temporary column as it's no longer needed
df_train.drop(columns=['temp_minus_one_count'], inplace=True)

# --- Calculate percentage and filter ---
# Calculate total ToF cells for each sequence (number of rows * number of ToF columns)
sequence_stats['total_tof_cells'] = sequence_stats['row_count'] * num_tof_columns

# Calculate the percentage of invalid data for each sequence
epsilon = 1e-9
sequence_stats['invalid_percentage'] = \
    (sequence_stats['total_minus_ones'] / (sequence_stats['total_tof_cells'] + epsilon)) * 100

# Filter the DataFrame to keep only the sequences that meet our criteria
sequences_above_threshold = sequence_stats[sequence_stats['invalid_percentage'] > threshold_percentage]

# --- Display the results ---
num_sequences_above_threshold = len(sequences_above_threshold)
total_unique_sequences = df_train['sequence_id'].nunique()

print("--- Results ---")
print(f"Total number of unique sequences: {total_unique_sequences:,}")
print(f"Number of sequences with > {threshold_percentage}% invalid ToF values: {num_sequences_above_threshold:,}")

if total_unique_sequences > 0:
    percentage_of_all_sequences = (num_sequences_above_threshold / total_unique_sequences) * 100
    print(f"This represents {percentage_of_all_sequences:.2f}% of all sequences.")

# Optional: Display the top 10 most invalid sequences
if num_sequences_above_threshold > 0:
    print("--- Top 10 Sequences with the Highest Percentage of Invalid ToF Data ---")
    # Prepare the DataFrame for display
    display_df = sequences_above_threshold.sort_values(by='invalid_percentage', ascending=False).head(10).copy()
    display_df = display_df[['total_minus_ones', 'total_tof_cells', 'invalid_percentage']] # Reorder for clarity

    # Formatting for better readability
    display_df['invalid_percentage'] = display_df['invalid_percentage'].map('{:.2f}%'.format)
    display_df['total_minus_ones'] = display_df['total_minus_ones'].map('{:,}'.format)
    display_df['total_tof_cells'] = display_df['total_tof_cells'].map('{:,}'.format)
    
    # Rename columns for the final display
    display_df.rename(columns={
        'total_minus_ones': 'Invalid Cells (-1)',
        'total_tof_cells': 'Total ToF Cells',
        'invalid_percentage': 'Invalid %'
    }, inplace=True)

    print(display_df)


print(f"Identified {num_tof_columns} ToF columns for analysis.")

# --- Step 2: Define the NEW threshold ---
threshold_percentage_low = 30.0  # New threshold
print(f"Searching for sequences with LESS than {threshold_percentage_low}% invalid ToF values...")

# --- Step 3: Group by sequence_id and calculate stats ---
# Calculate the count of -1s for each row across ToF columns
df_train['temp_minus_one_count'] = (df_train[tof_columns] == -1).sum(axis=1)

# Group by sequence_id and aggregate
sequence_stats = df_train.groupby('sequence_id').agg(
    row_count=('sequence_id', 'size'),
    total_minus_ones=('temp_minus_one_count', 'sum')
)

# Clean up the temporary column
df_train.drop(columns=['temp_minus_one_count'], inplace=True)

# --- Step 4: Calculate percentage and filter (with the key change) ---
# Calculate total ToF cells for each sequence
sequence_stats['total_tof_cells'] = sequence_stats['row_count'] * num_tof_columns

# Calculate the percentage of invalid data
epsilon = 1e-9  # To prevent division by zero
sequence_stats['invalid_percentage'] = \
    (sequence_stats['total_minus_ones'] / (sequence_stats['total_tof_cells'] + epsilon)) * 100

# --- THE CRITICAL CHANGE IS HERE ---
# Filter the DataFrame to keep sequences BELOW the threshold
sequences_below_threshold = sequence_stats[sequence_stats['invalid_percentage'] < threshold_percentage_low]

# --- Step 5: Display the results ---
num_sequences_below_threshold = len(sequences_below_threshold)
total_unique_sequences = df_train['sequence_id'].nunique()

print("--- Results ---")
print(f"Total number of unique sequences: {total_unique_sequences:,}")
print(f"Number of sequences with < {threshold_percentage_low}% invalid ToF values: {num_sequences_below_threshold:,}")

if total_unique_sequences > 0:
    percentage_of_all_sequences = (num_sequences_below_threshold / total_unique_sequences) * 100
    print(f"This represents {percentage_of_all_sequences:.2f}% of all sequences.")

# Optional: Display the top 10 cleanest sequences (those with the lowest percentage of invalid data)
if num_sequences_below_threshold > 0:
    print("--- Top 10 Sequences with the LOWEST Percentage of Invalid ToF Data (Cleanest) ---")
    # Prepare the DataFrame for display
    # Sorting is now ascending to show the best sequences first
    display_df = sequences_below_threshold.sort_values(by='invalid_percentage', ascending=True).head(10).copy()
    display_df = display_df[['total_minus_ones', 'total_tof_cells', 'invalid_percentage']] # Reorder for clarity

    # Formatting for better readability
    display_df['invalid_percentage'] = display_df['invalid_percentage'].map('{:.2f}%'.format)
    display_df['total_minus_ones'] = display_df['total_minus_ones'].map('{:,}'.format)
    display_df['total_tof_cells'] = display_df['total_tof_cells'].map('{:,}'.format)
    
    # Rename columns for the final display
    display_df.rename(columns={
        'total_minus_ones': 'Invalid Cells (-1)',
        'total_tof_cells': 'Total ToF Cells',
        'invalid_percentage': 'Invalid %'
    }, inplace=True)

    print(display_df)


# Step 1: Select a Random Sequence
if not df_train.empty and 'sequence_id' in df_train.columns:
    unique_sequences = df_train['sequence_id'].unique()
    random_sequence_id = random.choice(unique_sequences)
    
    sequence_df = df_train[df_train['sequence_id'] == random_sequence_id].copy()
    sequence_df.sort_values('sequence_counter', inplace=True)
    
    subject = sequence_df.iloc[0]['subject']
    # Placeholders for metadata not present in the main file
    gesture = "N/A"
    orientation = "N/A"

    print("Random sequence selected:")
    print(f"  - Sequence ID: {random_sequence_id}")
    print(f"  - Subject: {subject}")
    print(f"  - Number of frames: {len(sequence_df)}")
else:
    print("DataFrame is empty or 'sequence_id' column is missing. Cannot proceed.")
    sequence_df = pd.DataFrame()


# Function to Create a Physically-Arranged Combined Animation
def create_physical_layout_animation(sequence_data, sequence_id, subject_id, gesture_info, orientation_info, output_dir="tof_animations"):
    """
    Generates a single GIF arranging ToF sensor animations to mimic their
    physical layout on the Helios device (cross-shape).
    """
    num_sensors = 5
    pixel_count = 64
    img_shape = (8, 8)
    
    print("\n--- Preparing data for physical layout animation ---")
    
    # Extract all ToF data and find global min/max for consistent coloring
    tof_columns = [f'tof_{s}_v{p}' for s in range(1, num_sensors + 1) for p in range(pixel_count)]
    existing_tof_columns = [col for col in tof_columns if col in sequence_data.columns]
    
    if not existing_tof_columns:
        print("Error: No ToF columns found in the DataFrame.")
        return None

    min_val = sequence_data[existing_tof_columns].min().min()
    max_val = sequence_data[existing_tof_columns].max().max()
    print(f"ToF data values in this sequence range from {min_val:.2f} to {max_val:.2f}.")

    # Reshape data for each sensor
    sensor_frames_map = {}
    for s_num in range(1, num_sensors + 1):
        s_cols = [f'tof_{s_num}_v{p}' for p in range(pixel_count)]
        if all(col in sequence_data.columns for col in s_cols):
            s_data = sequence_data[s_cols].values
            sensor_frames_map[s_num] = s_data.reshape(-1, img_shape[0], img_shape[1])
        else:
            print(f"Warning: Data for Sensor {s_num} is incomplete. It will not be displayed.")

    if not sensor_frames_map:
        print("Error: No complete sensor data available to create animation.")
        return None
        
    num_frames = next(iter(sensor_frames_map.values())).shape[0]
    combined_images = []
    
    print(f"Generating {num_frames} combined frames for the physical layout...")

    # Define the grid positions for each sensor
    # (row, col) in a 3x3 grid
    sensor_positions = {
        1: (1, 1),  # Center
        2: (0, 1),  # Top
        3: (1, 0),  # Left  (<- Was Right)
        4: (2, 1),  # Bottom
        5: (1, 2)   # Right (<- Was Left)
    }

    for frame_idx in range(num_frames):
        # Create a 3x3 subplot grid. `constrained_layout=True` helps manage spacing.
        fig, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
        
        # Turn off all axes by default. We'll turn them on only for sensors.
        for ax in axes.flat:
            ax.axis('off')

        for sensor_num, pos in sensor_positions.items():
            if sensor_num in sensor_frames_map:
                ax = axes[pos[0], pos[1]] # Get the correct subplot
                ax.axis('on') # Turn it back on
                frame_data = sensor_frames_map[sensor_num][frame_idx]
                
                ax.imshow(frame_data, cmap='viridis', vmin=min_val, vmax=max_val)
                ax.set_title(f"ToF Sensor {sensor_num}")
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Add a shared main title (suptitle) with all common information
        shared_info = (f"Frame: {frame_idx + 1}/{num_frames} | Sequence ID: {sequence_id}\n"
                       f"Subject: {subject_id} | Gesture: {gesture_info} | Orientation: {orientation_info}")
        fig.suptitle(shared_info, fontsize=16)
        
        # Convert the entire figure (with the 3x3 grid) into a single image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        combined_images.append(image)
        plt.close(fig) # Close the figure to free up memory

    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, f'physical_layout_seq_{sequence_id}.gif')
    
    print(f"Saving physical layout animation to '{gif_path}'...")
    imageio.mimsave(gif_path, combined_images, duration=100, loop=0)
    print("âœ… Physical layout animation saved successfully.")
    
    return gif_path


# Step 2: Generate and Display the Physical Layout Animation -
if not sequence_df.empty:
    physical_layout_gif_path = create_physical_layout_animation(
        sequence_data=sequence_df,
        sequence_id=random_sequence_id,
        subject_id=subject,
        gesture_info=gesture,
        orientation_info=orientation
    )
    if physical_layout_gif_path:
        # Display the final GIF in the output cell
        print(f"\nğŸ‘‡ Displaying physical layout animation for sequence {random_sequence_id}:")
        display(Image(filename=physical_layout_gif_path, width=600)) # Adjust width for optimal viewing




