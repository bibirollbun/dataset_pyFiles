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


# Load in the dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


import matplotlib.pyplot as plt

# Compute 15-day rolling mean
rainfall_avg = train.groupby('day', as_index=False)['rainfall'].mean()
rainfall_avg['rolling_rainfall'] = rainfall_avg['rainfall'].rolling(window=15, min_periods=1).mean()

# Identify key trend points
high_rain_day = rainfall_avg.loc[rainfall_avg['rainfall'].idxmax(), 'day']
low_rain_day = rainfall_avg.loc[rainfall_avg['rainfall'].idxmin(), 'day']
peak_roll_day = rainfall_avg.loc[rainfall_avg['rolling_rainfall'].idxmax(), 'day']
low_roll_day = rainfall_avg.loc[rainfall_avg['rolling_rainfall'].idxmin(), 'day']

print(f"Max Avg Rainfall: Day {high_rain_day} ({rainfall_avg['rainfall'].max():.2f} mm)")
print(f"Min Avg Rainfall: Day {low_rain_day} ({rainfall_avg['rainfall'].min():.2f} mm)")
print(f"Peak Rolling Mean: Day {peak_roll_day} ({rainfall_avg['rolling_rainfall'].max():.2f} mm)")
print(f"Lowest Rolling Mean: Day {low_roll_day} ({rainfall_avg['rolling_rainfall'].min():.2f} mm)")

# Plot trend
plt.figure(figsize=(12, 6))
plt.plot(rainfall_avg['day'], rainfall_avg['rolling_rainfall'], label="15-Day Rolling Mean", color="blue")
plt.scatter(rainfall_avg['day'], rainfall_avg['rainfall'], alpha=0.3, color="gray", label="Daily Rainfall")

# Mark anomalies
for day, color, label in zip(
    [high_rain_day, low_rain_day, peak_roll_day, low_roll_day], 
    ['red', 'green', 'purple', 'orange'],
    ['Peak Rainfall', 'Lowest Rainfall', 'Peak Rolling Mean', 'Lowest Rolling Mean']
):
    plt.axvline(day, color=color, linestyle="--", label=f"{label} (Day {day})")

plt.xlabel("Day of the Year")
plt.ylabel("Avg Rainfall (mm)")
plt.title("Rainfall Trends Over the Year")
plt.xticks(range(0, 365, 30))  # Less cluttered labels
plt.legend()
plt.show()



# Count days with 7 entries
count_7 = (train.groupby('day').size() == 7).sum()

# Count days with 5 entries
count_5 = (train.groupby('day').size() == 5).sum()

print(f"Days with 7 entries: {count_7}")
print(f"Days with 5 entries: {count_5}")


# Show raw data for Day 3 before any grouping or modifications
print("Raw records for Day 3 before any processing:")
print(train[train['day'] == 3].sort_values(by='id').to_string(index=False))


# Get lists of days with 7 and 5 entries
days_with_7 = train.groupby('day').size()[train.groupby('day').size() == 7].index
days_with_5 = train.groupby('day').size()[train.groupby('day').size() == 5].index

# Display them
print("Days with 7 entries:", list(days_with_7))
print("Days with 5 entries:", list(days_with_5))

# Get all records for days with 7 entries
extra_entries = train[train['day'].isin(days_with_7)].sort_values(by=['day', 'id'])

# Get all records for days with 5 entries
missing_entries = train[train['day'].isin(days_with_5)].sort_values(by=['day', 'id'])

# Display them side by side
print("Extra entries (Days with 7):")
print(extra_entries.to_string(index=False))

print("\nMissing entries (Days with 5):")
print(missing_entries.to_string(index=False))


# Get ID 1037's data
id_1037 = train.loc[train['id'] == 1037]

# Get Day 308's records
day_308 = train[train['day'] == 308]

# Compute the mean of Day 308’s existing entries (excluding id and day from calculations)
day_308_mean = day_308.drop(columns=["id", "day"]).mean(numeric_only=True)

# Compute ID 1037 values (excluding id and day from calculations)
id_1037_values = id_1037.drop(columns=["id", "day"]).iloc[0]  # Ensure it's a Series

# Compute absolute differences for each column
differences = abs(day_308_mean - id_1037_values)

# Compute total difference (sum of all differences, excluding id and day)
total_difference = differences.sum()

# Print comparison (Keeping ID and Day in the display, but not in calculations)
print("ID 1037 values:")
print(id_1037.to_string(index=False))

print("\nDay 308 Mean Values (excluding ID and Day from calculations):")
print(day_308_mean.to_string())

print(f"\n**Total Difference (Excluding ID & Day):** {total_difference:.2f}")

print("\n**Breakdown of Differences (ID & Day Excluded):**")
print(differences.to_string())


# Dictionary to store differences
day_diffs = {}

for possible_day in days_with_5:
    # Get data for the possible reassignment day
    day_data = train[train['day'] == possible_day]

    # Compute mean values for the day (excluding 'id' and 'day' from calculations)
    day_mean = day_data.drop(columns=["id", "day"]).mean(numeric_only=True)

    # Extract ID 1037's values (excluding 'id' and 'day' from calculations)
    id_1037_values = id_1037.drop(columns=["id", "day"]).iloc[0]  # Ensure it's a Series

    # Calculate absolute differences for each numeric column (excluding ID & Day)
    diff = abs(day_mean - id_1037_values)

    # Compute total difference as the sum of all individual differences
    total_diff = diff.sum()

    # Store results
    day_diffs[possible_day] = (total_diff, diff)

# Sort by best match (lowest total difference)
sorted_days = sorted(day_diffs.items(), key=lambda x: x[1][0])

# Print the top matches with detailed breakdown
print("**Comparison of ID 1037 Against 5-Entry Days (ID & DAY EXCLUDED from Calculation):**")
for day, (total_diff, diff) in sorted_days[:5]:  # Show top 5 matches
    print(f"\nDay {day} | **Total Difference: {total_diff:.2f}**")
    print(diff.to_string())

# Identify the best match
best_day, (best_diff, best_diff_details) = sorted_days[0]

# Print the best match with key differences
print(f"\n**Best Match for ID 1037: Day {best_day}** with **Total Difference: {best_diff:.2f}**")
print("\n**Breakdown of Key Matching Factors:**")
print(best_diff_details.to_string())


import pandas as pd

# Extract all records for Day 3 (7 entries)
day_3_records = train[train['day'] == 3]

# Extract all records for Day 273 (5 entries)
day_273_records = train[train['day'] == 273]

# Display both sets for review
print("Full Records for Day 3 (7 Entries):")
print(day_3_records.to_string(index=False))

print("\nFull Records for Day 273 (5 Entries):")
print(day_273_records.to_string(index=False))


# Find the current assigned day for ID 1367
current_day_1367 = train.loc[train['id'] == 1367, 'day'].values[0]

# Check if this day is in the list of miscounted days (5 or 7 entries)
is_miscounted = current_day_1367 in days_with_5 or current_day_1367 in days_with_7

print(f"ID 1367 is currently assigned to Day {current_day_1367}.")
print(f"Is this day in the miscounted 5/7 entries list? {'YES' if is_miscounted else 'NO'}")


# Identify the erroneous extra entry in each 7-entry day
erroneous_entries = {}

for day in days_with_7:
    # Get all IDs for this day
    day_records = train[train['day'] == day]
    actual_ids = sorted(day_records['id'].tolist())  # Sort IDs for logical comparison
    
    # Generate expected ID sequence (based on the assumption of one per year)
    expected_ids = [actual_ids[0] + (365 * i) for i in range(6)]
    
    # Identify the ID that does not match the expected pattern
    misplaced_id = next((id_value for id_value in actual_ids if id_value not in expected_ids), None)

    # Store the misplaced entry
    erroneous_entries[day] = misplaced_id

# Convert results into DataFrame for clarity
erroneous_entries_df = pd.DataFrame.from_dict(erroneous_entries, orient="index", columns=["Erroneous_ID"])
erroneous_entries_df.index.name = "Day"

# Print results
print("\nErroneous Extra Entries in 7-Entry Days:")
print(erroneous_entries_df.to_string())


# Dictionary to store expected IDs that are missing
missing_entries = {}

# Iterate through each 5-entry day and determine the missing ID
for day in days_with_5:
    # Compute the expected ID sequence for this day using yearly increments
    expected_ids = set([(day - 1) + (365 * i) for i in range(6)])  # Expected IDs for 6 years
    
    # Get actual IDs present for this day
    actual_ids = set(train.loc[train['day'] == day, 'id'])

    # Determine the missing ID
    missing_id = expected_ids - actual_ids

    # Store the result if we find exactly one missing ID
    if len(missing_id) == 1:
        missing_id = list(missing_id)[0]  # Convert set to single value
        missing_entries[day] = missing_id

# Convert results to DataFrame
missing_entries_df = pd.DataFrame.from_dict(missing_entries, orient="index", columns=["Missing_ID"])

# Step 2: Identify where these missing IDs currently are
missing_entries_df["Currently_Assigned_Day"] = missing_entries_df["Missing_ID"].apply(
    lambda x: train.loc[train['id'] == x, 'day'].values[0] if x in train['id'].values else None
)

# Step 3: Check if the currently assigned day is in the 7-entry list
missing_entries_df["Is_Extra_Day?"] = missing_entries_df["Currently_Assigned_Day"].isin(days_with_7)

# Display results
print("Missing Expected Entries in 5-Entry Days:")
print(missing_entries_df.to_string())


print(train[train['day'] == 116])


# Identify all days that currently have 6 entries
days_with_6 = train.groupby('day').size()[train.groupby('day').size() == 6].index

# Dictionary to store misplaced IDs within these "correct" days
misplaced_in_6 = {}

# Iterate through each 6-entry day and verify expected sequence
for day in days_with_6:
    # Compute expected IDs for this day using yearly increments
    expected_ids = set([(day - 1) + (365 * i) for i in range(6)])  
    
    # Get actual IDs present for this day
    actual_ids = set(train.loc[train['day'] == day, 'id'])

    # Determine misplaced IDs (should not be here)
    misplaced_ids = actual_ids - expected_ids

    # Determine missing IDs (should be here but aren't)
    missing_ids = expected_ids - actual_ids

    # Store only if there's a mismatch
    if misplaced_ids or missing_ids:
        misplaced_in_6[day] = {"Misplaced_IDs": list(misplaced_ids), "Missing_IDs": list(missing_ids)}

# Convert results to DataFrame for easier analysis
misplaced_in_6_df = pd.DataFrame.from_dict(misplaced_in_6, orient="index")

# Display results
print("Misplaced Entries in 6-Entry Days:")
print(misplaced_in_6_df)


# Find where ID 1210 is currently assigned
current_day_1210 = train.loc[train['id'] == 1210, 'day'].values[0]
print(f"ID 1210 is currently assigned to Day {current_day_1210}")


# Extract all records for Day 80
day_80_records = train[train['day'] == 80]

# Display the records for review
print("Full Records for Day 80:")
print(day_80_records.to_string(index=False))


# Get records for Day 334
day_334_records = train[train['day'] == 334]

# Compute expected IDs for Day 334
expected_ids_334 = set([(334 - 1) + (365 * i) for i in range(6)])

# Get actual IDs present
actual_ids_334 = set(day_334_records['id'])

# Find the missing ID in Day 334
missing_id_334 = expected_ids_334 - actual_ids_334

# Print the results
print("Full Records for Day 334:")
print(day_334_records.to_string(index=False))

print(f"\nExpected IDs for Day 334: {sorted(expected_ids_334)}")
print(f"Actual IDs for Day 334: {sorted(actual_ids_334)}")
print(f"Missing ID from Day 334: {missing_id_334}")

# If Day 334 is missing an entry, check if it matches 1428
if missing_id_334:
    missing_id = list(missing_id_334)[0]
    if missing_id == 1428:
        print(f"Swap Confirmed: Move ID 1428 to Day 334 to fix the dataset.")
    else:
        print(f"Unexpected: The missing ID in Day 334 is {missing_id}, not 1428.")
else:
    print("No missing ID detected in Day 334. Further investigation needed.")



# Fix the two known misplaced entries
fixes = {1210: 116, 1428: 334}

# Apply the fixes
for id_value, corrected_day in fixes.items():
    train.loc[train['id'] == id_value, 'day'] = corrected_day

# Verify the fix
corrected_1210 = train.loc[train['id'] == 1210, 'day'].values[0]
corrected_1428 = train.loc[train['id'] == 1428, 'day'].values[0]

print(f"ID 1210 is now assigned to Day {corrected_1210} (Expected: 116)")
print(f"ID 1428 is now assigned to Day {corrected_1428} (Expected: 334)")



# Step 1: Identify missing IDs in 5-entry days
missing_entries_v2 = {}

# Iterate through each 5-entry day and determine the missing ID
for day in days_with_5:
    # Compute the expected ID sequence for this day using yearly increments
    expected_ids_v2 = set([(day - 1) + (365 * i) for i in range(6)])  # Expected IDs for 6 years

    # Get actual IDs present for this day
    actual_ids_v2 = set(train.loc[train['day'] == day, 'id'])

    # Determine the missing ID
    missing_id_v2 = expected_ids_v2 - actual_ids_v2

    # Store the result if we find exactly one missing ID
    if len(missing_id_v2) == 1:
        missing_id_v2 = list(missing_id_v2)[0]  # Convert set to single value
        missing_entries_v2[day] = missing_id_v2

# Convert results to DataFrame
missing_entries_df_v2 = pd.DataFrame.from_dict(missing_entries_v2, orient="index", columns=["Missing_ID"])

# Step 2: Identify where these missing IDs currently are
missing_entries_df_v2["Currently_Assigned_Day"] = missing_entries_df_v2["Missing_ID"].apply(
    lambda x: train.loc[train['id'] == x, 'day'].values[0] if x in train['id'].values else None
)

# Step 3: Check if the currently assigned day is in the 7-entry list
missing_entries_df_v2["Is_Extra_Day?"] = missing_entries_df_v2["Currently_Assigned_Day"].isin(days_with_7)

# Display results
print("Updated Missing Expected Entries in 5-Entry Days:")
print(missing_entries_df_v2.to_string())


# Count days with 7 entries
count_7v2 = (train.groupby('day').size() == 7).sum()

# Count days with 5 entries
count_5v2 = (train.groupby('day').size() == 5).sum()

print(f"Days with 7 entries: {count_7v2}")
print(f"Days with 5 entries: {count_5v2}")


# Dictionary mapping each misplaced ID to its correct day
reassignment_map = {
    1132: 38, 1251: 157, 1284: 190, 1290: 196, 1312: 218, 1318: 224, 
    1346: 252, 1352: 258, 1367: 273, 1373: 279, 1380: 286, 1382: 288, 
    1388: 294, 1395: 301, 1400: 306, 1037: 308, 1403: 309, 1404: 310, 
    1406: 312, 1407: 313, 1409: 315, 1414: 320, 1416: 322, 1420: 326, 
    1430: 336, 1438: 344, 1439: 345, 1445: 351, 1452: 358, 1453: 359, 
    1457: 363, 1458: 364, 1459: 365
}

# Apply the reassignments
for misplaced_id, correct_day in reassignment_map.items():
    train.loc[train['id'] == misplaced_id, 'day'] = correct_day

# Verify that all days now have exactly 6 entries
fixed_counts = train.groupby('day').size()
print("Post-Fix Record Counts:", fixed_counts.value_counts())

# Verify that all sequences are correct
incorrect_sequences = []
for day, group in train.groupby('day'):
    expected_ids = [(day - 1) + (365 * i) for i in range(6)]
    actual_ids = sorted(group['id'])
    if expected_ids != actual_ids:
        incorrect_sequences.append(day)

if incorrect_sequences:
    print(f"ERROR: Some days still have incorrect ID sequences: {incorrect_sequences}")
else:
    print("All day ID sequences are correctly aligned.")



# Extract the records for previously miscounted 5-entry days (should now be 6)
fixed_5_entry_days = train[train['day'].isin(missing_entries_df.index)].sort_values(by=['day', 'id'])

# Display the records for visual inspection
print("Fixed 5-Entry Days After Reassignment:")
print(fixed_5_entry_days.to_string(index=False))



import matplotlib.pyplot as plt

# Create a copy of the dataset to preserve the original
train_fixed = train.copy()

# Compute 15-day rolling mean
rainfall_avg_fixed = train_fixed.groupby('day', as_index=False)['rainfall'].mean()
rainfall_avg_fixed['rolling_rainfall'] = rainfall_avg_fixed['rainfall'].rolling(window=15, min_periods=1).mean()

# Identify key trend points
high_rain_day_fixed = rainfall_avg_fixed.loc[rainfall_avg_fixed['rainfall'].idxmax(), 'day']
low_rain_day_fixed = rainfall_avg_fixed.loc[rainfall_avg_fixed['rainfall'].idxmin(), 'day']
peak_roll_day_fixed = rainfall_avg_fixed.loc[rainfall_avg_fixed['rolling_rainfall'].idxmax(), 'day']
low_roll_day_fixed = rainfall_avg_fixed.loc[rainfall_avg_fixed['rolling_rainfall'].idxmin(), 'day']

print(f"Max Avg Rainfall: Day {high_rain_day_fixed} ({rainfall_avg_fixed['rainfall'].max():.2f} mm)")
print(f"Min Avg Rainfall: Day {low_rain_day_fixed} ({rainfall_avg_fixed['rainfall'].min():.2f} mm)")
print(f"Peak Rolling Mean: Day {peak_roll_day_fixed} ({rainfall_avg_fixed['rolling_rainfall'].max():.2f} mm)")
print(f"Lowest Rolling Mean: Day {low_roll_day_fixed} ({rainfall_avg_fixed['rolling_rainfall'].min():.2f} mm)")

# Plot trend
plt.figure(figsize=(12, 6))
plt.plot(rainfall_avg_fixed['day'], rainfall_avg_fixed['rolling_rainfall'], label="15-Day Rolling Mean", color="blue")
plt.scatter(rainfall_avg_fixed['day'], rainfall_avg_fixed['rainfall'], alpha=0.3, color="gray", label="Daily Rainfall")

# Mark anomalies
for day, color, label in zip(
    [high_rain_day_fixed, low_rain_day_fixed, peak_roll_day_fixed, low_roll_day_fixed], 
    ['red', 'green', 'purple', 'orange'],
    ['Peak Rainfall', 'Lowest Rainfall', 'Peak Rolling Mean', 'Lowest Rolling Mean']
):
    plt.axvline(day, color=color, linestyle="--", label=f"{label} (Day {day})")

# Add leaderboard cutoff line
plt.axvline(147, color='black', linestyle="--", label="Leaderboard Cutoff (Day 147)")

plt.xlabel("Day of the Year")
plt.ylabel("Avg Rainfall (mm)")
plt.title("Rainfall Trends Over the Year (Fixed Data)")
plt.xticks(range(0, 365, 30))  # Less cluttered labels
plt.legend()
plt.show()



# Calculate average rainfall before and after day 147
avg_rainfall_before_147 = rainfall_avg_fixed.loc[rainfall_avg_fixed['day'] < 147, 'rainfall'].mean()
avg_rainfall_after_147 = rainfall_avg_fixed.loc[rainfall_avg_fixed['day'] >= 147, 'rainfall'].mean()

print(f"Average Rainfall Before Day 147: {avg_rainfall_before_147:.2f} mm")
print(f"Average Rainfall From Day 147 Onward: {avg_rainfall_after_147:.2f} mm")

