import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# Define the columns to use for comparison (all except 'id' and 'Calories')
comparison_columns = [col for col in train.columns if col != 'id' and col != 'Calories']

print("="*80)
print("COMPREHENSIVE DATA LEAKAGE AND DUPLICATES ANALYSIS")
print("="*80)

print("\nDATASET INFORMATION:")
print(f"Train dataset shape: {train.shape}")
print(f"Test dataset shape: {test.shape}")
print(f"Columns used for duplicate detection: {comparison_columns}")

# 1. ANALYZE DUPLICATES WITHIN TRAIN DATASET
print("\n" + "="*80)
print("1. DUPLICATES WITHIN TRAIN DATASET")
print("="*80)

# Find duplicates in train
train_duplicated_mask = train.duplicated(subset=comparison_columns, keep=False)
train_duplicated_count = train_duplicated_mask.sum()
train_total_count = len(train)
train_duplicated_percentage = (train_duplicated_count / train_total_count) * 100

print(f"Total rows in train: {train_total_count:,}")
print(f"Rows with duplicates: {train_duplicated_count:,}")
print(f"Percentage of rows with duplicates: {train_duplicated_percentage:.2f}%")

# Create a detailed duplicate analysis table for train
train_dup_analysis = []

# Group by comparison columns
train_groups = train[train_duplicated_mask].groupby(comparison_columns)
train_total_groups = len(train_groups)

# Count groups with different Calories values and gather analysis data
different_calories_groups = 0
for name, group in train_groups:
    unique_calories = group['Calories'].unique()
    has_different_calories = len(unique_calories) > 1
    
    if has_different_calories:
        different_calories_groups += 1
    
    min_cal = group['Calories'].min()
    max_cal = group['Calories'].max()
    mean_cal = group['Calories'].mean()
    
    # Calculate percentage difference
    base_value = min_cal if min_cal > 0 else (mean_cal if mean_cal > 0 else 1)
    max_pct_diff = 100 * (max_cal - min_cal) / base_value
    
    # Create a group ID for reference
    group_id = hash(str(name))
    
    # Add each row to the analysis
    for idx, row in group.iterrows():
        pct_diff_from_min = 100 * (row['Calories'] - min_cal) / base_value if base_value > 0 else 0
        
        train_dup_analysis.append({
            'group_id': group_id,
            'row_id': idx,
            'Calories': row['Calories'],
            'min_cal_in_group': min_cal,
            'max_cal_in_group': max_cal,
            'mean_cal_in_group': mean_cal,
            'pct_diff_from_min': pct_diff_from_min,
            'max_pct_diff_in_group': max_pct_diff,
            'group_size': len(group),
            'has_different_calories': has_different_calories,
            **{col: row[col] for col in comparison_columns}
        })

# Create train duplicates DataFrame
train_dup_df = pd.DataFrame(train_dup_analysis)
if not train_dup_df.empty:
    train_dup_df = train_dup_df.sort_values(['group_id', 'Calories'])

train_diff_cal_percentage = (different_calories_groups / train_total_groups) * 100 if train_total_groups > 0 else 0

print(f"Total duplicate groups: {train_total_groups:,}")
print(f"Groups with different Calories values: {different_calories_groups:,}")
print(f"Percentage of groups with different Calories: {train_diff_cal_percentage:.2f}%")

# Print statistics for groups with different Calories
if different_calories_groups > 0:
    diff_cal_stats = train_dup_df[train_dup_df['has_different_calories']][['group_size', 'max_pct_diff_in_group']].drop_duplicates()
    
    print("\nStatistics for groups with different Calories values:")
    print(diff_cal_stats.describe())
    
    # Show examples of most divergent groups
    print("\nTop 5 groups with largest percentage differences in Calories:")
    top_groups = train_dup_df[train_dup_df['has_different_calories']].sort_values('max_pct_diff_in_group', ascending=False)
    top_group_ids = top_groups['group_id'].unique()[:5]
    
    for i, group_id in enumerate(top_group_ids):
        group_data = train_dup_df[train_dup_df['group_id'] == group_id]
        calories_values = sorted(group_data['Calories'].unique())
        
        print(f"Group {i+1}:")
        print(f"  Group size: {group_data['group_size'].iloc[0]}")
        print(f"  Min/Max Calories: {group_data['min_cal_in_group'].iloc[0]}/{group_data['max_cal_in_group'].iloc[0]}")
        print(f"  Percentage difference: {group_data['max_pct_diff_in_group'].iloc[0]:.2f}%")
        print(f"  All Calories values: {calories_values}")

# Visualize distribution of duplicates with same/different Calories
plt.figure(figsize=(10, 6))
plt.pie([different_calories_groups, train_total_groups - different_calories_groups], 
        labels=['Groups with different Calories', 'Groups with same Calories'],
        autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'],
        startangle=90, explode=(0.1, 0))
plt.title('Distribution of Duplicate Groups in Train Dataset')
plt.axis('equal')
plt.tight_layout()
plt.show()

# Display the full duplicates table (first rows)
print("\nTable of duplicates in train dataset (first 10 rows):")
if not train_dup_df.empty:
    display(train_dup_df.head(10))
    print(f"Total rows in duplicates table: {len(train_dup_df):,}")


# 2. ANALYZE DUPLICATES WITHIN TEST DATASET
print("\n" + "="*80)
print("2. DUPLICATES WITHIN TEST DATASET")
print("="*80)

# Find duplicates in test
test_duplicated_mask = test.duplicated(subset=comparison_columns, keep=False)
test_duplicated_count = test_duplicated_mask.sum()
test_total_count = len(test)
test_duplicated_percentage = (test_duplicated_count / test_total_count) * 100

print(f"Total rows in test: {test_total_count:,}")
print(f"Rows with duplicates: {test_duplicated_count:,}")
print(f"Percentage of rows with duplicates: {test_duplicated_percentage:.2f}%")

# Group by comparison columns for test
test_groups = test[test_duplicated_mask].groupby(comparison_columns)
test_total_groups = len(test_groups)

print(f"Total duplicate groups in test: {test_total_groups:,}")

# Create test duplicates analysis table
test_dup_analysis = []
for name, group in test_groups:
    group_id = hash(str(name))
    for idx, row in group.iterrows():
        test_dup_analysis.append({
            'group_id': group_id,
            'row_id': idx,
            'group_size': len(group),
            **{col: row[col] for col in comparison_columns}
        })

# Create test duplicates DataFrame
test_dup_df = pd.DataFrame(test_dup_analysis)
if not test_dup_df.empty:
    test_dup_df = test_dup_df.sort_values(['group_id'])
    
    print("\nTable of duplicates in test dataset (first 10 rows):")
    display(test_dup_df.head(10))
    print(f"Total rows in test duplicates table: {len(test_dup_df):,}")


# 3. ANALYZE DATA LEAKAGE BETWEEN TRAIN AND TEST
print("\n" + "="*80)
print("3. DATA LEAKAGE BETWEEN TRAIN AND TEST")
print("="*80)

# Create copies with source labels
train_subset = train[comparison_columns].copy()
train_subset['source'] = 'train'
train_subset['original_index'] = train.index
train_subset['calories'] = train['Calories']

test_subset = test[comparison_columns].copy()
test_subset['source'] = 'test'
test_subset['original_index'] = test.index

# Combine the datasets
combined = pd.concat([train_subset, test_subset], ignore_index=True)
combined_duplicated_mask = combined.duplicated(subset=comparison_columns, keep=False)
combined_duplicated_rows = combined[combined_duplicated_mask].copy()

# Analyze overlaps between train and test
overlap_analysis = []
train_rows_in_overlap = 0
test_rows_in_overlap = 0
train_test_overlap_groups = 0
train_only_groups = 0
test_only_groups = 0

for name, group in combined_duplicated_rows.groupby(comparison_columns):
    sources = group['source'].unique()
    group_size = len(group)
    
    if len(sources) > 1:  # Has both train and test rows
        train_test_overlap_groups += 1
        train_rows_in_group = sum(group['source'] == 'train')
        test_rows_in_group = sum(group['source'] == 'test')
        
        train_rows_in_overlap += train_rows_in_group
        test_rows_in_overlap += test_rows_in_group
        
        # Get train indices and Calories values
        train_indices = group[group['source'] == 'train']['original_index'].tolist()
        test_indices = group[group['source'] == 'test']['original_index'].tolist()
        calories_values = group[group['source'] == 'train']['calories'].tolist()
        
        overlap_analysis.append({
            'group_id': hash(str(name)),
            'group_size': group_size,
            'train_rows': train_rows_in_group,
            'test_rows': test_rows_in_group,
            'train_indices': train_indices,
            'test_indices': test_indices,
            'calories_values': calories_values,
            'has_different_calories': len(set(calories_values)) > 1,
            **{col: name[i] for i, col in enumerate(comparison_columns)}
        })
    elif 'train' in sources:
        train_only_groups += 1
    elif 'test' in sources:
        test_only_groups += 1

overlap_df = pd.DataFrame(overlap_analysis)
total_overlap_groups = train_test_overlap_groups + train_only_groups + test_only_groups

print(f"Total rows in train: {train_total_count:,}")
print(f"Rows from train with exact matches in test: {train_rows_in_overlap:,}")
print(f"Percentage of train data with matches in test: {(train_rows_in_overlap / train_total_count) * 100:.2f}%")

print(f"\nTotal rows in test: {test_total_count:,}")
print(f"Rows from test with exact matches in train: {test_rows_in_overlap:,}")
print(f"Percentage of test data with matches in train: {(test_rows_in_overlap / test_total_count) * 100:.2f}%")

print(f"\nTotal duplicate groups across both datasets: {total_overlap_groups:,}")
print(f"Groups with overlap between train and test: {train_test_overlap_groups:,}")
print(f"Groups only in train: {train_only_groups:,}")
print(f"Groups only in test: {test_only_groups:,}")

# Count groups with different Calories in overlap
overlap_with_diff_calories = 0
if not overlap_df.empty:
    overlap_with_diff_calories = sum(overlap_df['has_different_calories'])
    
    print(f"\nOverlap groups with different Calories in train: {overlap_with_diff_calories:,}")
    print(f"Percentage of overlap groups with different Calories: {(overlap_with_diff_calories / train_test_overlap_groups) * 100:.2f}%")

# Visualize data leakage
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
labels = ['Train rows without\nmatches in test', 'Train rows with\nmatches in test']
sizes = [train_total_count - train_rows_in_overlap, train_rows_in_overlap]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'],
        startangle=90, explode=(0, 0.1))
plt.title('Train Dataset')
plt.axis('equal')

plt.subplot(1, 2, 2)
labels = ['Test rows without\nmatches in train', 'Test rows with\nmatches in train']
sizes = [test_total_count - test_rows_in_overlap, test_rows_in_overlap]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'],
        startangle=90, explode=(0, 0.1))
plt.title('Test Dataset')
plt.axis('equal')

plt.tight_layout()
plt.show()

# Display the full overlap analysis table
print("\nTable of overlapping groups between train and test (first 10 rows):")
if not overlap_df.empty:
    # Make the indices and calories more readable
    readable_overlap_df = overlap_df.copy()
    readable_overlap_df['train_indices'] = readable_overlap_df['train_indices'].apply(lambda x: str(x[:3]) + ("..." if len(x) > 3 else ""))
    readable_overlap_df['test_indices'] = readable_overlap_df['test_indices'].apply(lambda x: str(x[:3]) + ("..." if len(x) > 3 else ""))
    readable_overlap_df['calories_values'] = readable_overlap_df['calories_values'].apply(lambda x: str(x[:3]) + ("..." if len(x) > 3 else ""))
    
    display(readable_overlap_df.head(10))
    print(f"Total overlapping groups: {len(overlap_df):,}")


# 4. ANALYZE TEST ROWS MATCHING TRAIN GROUPS WITH DIFFERENT CALORIES
print("\n" + "="*80)
print("4. TEST ROWS MATCHING TRAIN GROUPS WITH DIFFERENT CALORIES")
print("="*80)

# Create mapping from feature combinations to their train properties
feature_to_group_properties = {}
for _, group in train_groups:
    calories_values = group['Calories'].unique()
    key = tuple(group.iloc[0][col] for col in comparison_columns)
    feature_to_group_properties[key] = {
        'has_different_calories': len(calories_values) > 1,
        'calories_values': sorted(calories_values),
        'min_cal': min(calories_values),
        'max_cal': max(calories_values),
        'mean_cal': group['Calories'].mean(),
        'max_pct_diff': 100 * (max(calories_values) - min(calories_values)) / min(calories_values) if min(calories_values) > 0 else 0
    }

# Identify test rows matching train groups with different Calories
test_matching_diff_cal_train = []
for idx, row in test.iterrows():
    key = tuple(row[col] for col in comparison_columns)
    if key in feature_to_group_properties and feature_to_group_properties[key]['has_different_calories']:
        test_matching_diff_cal_train.append({
            'test_index': idx,
            'train_calories_values': feature_to_group_properties[key]['calories_values'],
            'min_cal_in_train': feature_to_group_properties[key]['min_cal'],
            'max_cal_in_train': feature_to_group_properties[key]['max_cal'],
            'mean_cal_in_train': feature_to_group_properties[key]['mean_cal'],
            'max_pct_diff_in_train': feature_to_group_properties[key]['max_pct_diff'],
            **{col: row[col] for col in comparison_columns}
        })

# Create DataFrame for test rows matching problematic train groups
test_matching_df = pd.DataFrame(test_matching_diff_cal_train)

if not test_matching_df.empty:
    print(f"Number of test rows matching train groups with different Calories: {len(test_matching_df):,}")
    print(f"Percentage of test data affected: {(len(test_matching_df) / test_total_count) * 100:.2f}%")
    
    print("\nTable of test rows matching train groups with different Calories (first 10 rows):")
    display(test_matching_df.head(10))
    print(f"Total rows in this table: {len(test_matching_df):,}")
    
    # Calculate statistics on the percentage differences
    print("\nStatistics on percentage differences in matching groups:")
    print(test_matching_df['max_pct_diff_in_train'].describe())
else:
    print("No test rows match train groups with different Calories values.")


# 5. SUMMARY OF FINDINGS
print("\n" + "="*80)
print("5. SUMMARY OF FINDINGS")
print("="*80)
print(f"1. In the train dataset, {train_duplicated_count:,} rows ({train_duplicated_percentage:.2f}%) have duplicates.")
print(f"2. Among {train_total_groups:,} duplicate groups in train, {different_calories_groups:,} groups ({train_diff_cal_percentage:.2f}%) have different Calories values for identical feature sets.")
print(f"3. {test_rows_in_overlap:,} rows in test ({(test_rows_in_overlap / test_total_count) * 100:.2f}%) have exact matches in train.")

if not test_matching_df.empty:
    print(f"4. {len(test_matching_df):,} rows in test ({(len(test_matching_df) / test_total_count) * 100:.2f}%) match train groups that have inconsistent Calories values.")
    print(f"   These test rows match train groups where Calories differs by {test_matching_df['max_pct_diff_in_train'].mean():.2f}% on average (max: {test_matching_df['max_pct_diff_in_train'].max():.2f}%).")


