# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import glob
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


def process_file(file_path, file_name):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(',')
            timestamp = parts[0].split('GMT')[0].strip() + ' GMT'  # Clean the timestamp
            result, white_name, white_elo, black_result, black_name, black_elo = parts[1:]
            data.append({
                'timestamp': timestamp,
                'result': result,
                'white_name': white_name,
                'white_elo': int(white_elo),
                'black_result': black_result,
                'black_name': black_name,
                'black_elo': int(black_elo),
                'file_name': file_name  # Add a file_name column to identify the source
            })
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%a %b %d %Y %H:%M:%S %Z', errors='coerce')  # Convert timestamp
    return df


import re

# Function to extract the bot version from the filename
def extract_bot_version(file_path):
    # Use regex to match the bot version (e.g., "v414" from "v414.txt" or "v414-2.txt")
    match = re.match(r'(v\d+)', os.path.basename(file_path))
    if match:
        return match.group(1)
    return "Unknown"  # Default if no match is found


# Function to process a single file and add file_name as a column
def process_files_and_combine(file_paths):
    combined_df = pd.DataFrame()
    
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        clean_name = file_name.removeprefix("final_")
        clean_name = os.path.splitext(clean_name)[:-1][0]
        # Process the file
        df = process_file(file_path, file_name=clean_name)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    return combined_df

# List of file paths (e.g., from a directory or manual entry)
file_paths = glob.glob('/kaggle/input/top3-final/final-games-top-3/*.txt')  # Update the directory path if needed

# Process all files and combine into a single DataFrame
combined_df = process_files_and_combine(file_paths)


combined_df = combined_df.drop_duplicates()


# Extract the ELO based on matching file_name with either white_name or black_name
combined_df["player_elo"] = combined_df.apply(lambda row: row["white_elo"] if row["file_name"] == row["white_name"].lower() else row["black_elo"], axis=1)

# Sort the data by timestamp for proper plotting
combined_df = combined_df.sort_values(by='timestamp')

# Check the DataFrame
print(combined_df.head())


# Plot the ELO over time for each bot version
plt.figure(figsize=(12, 6))

# Remove rows with "Unknown" in the file_name column
clean_df = combined_df[combined_df['file_name'] != "Unknown"].copy()
clean_df['timestamp'] = clean_df['timestamp'].dt.tz_convert(None)

excluded = []

# Loop through each file_name and plot the data
for file_name in clean_df['file_name'].unique():
    if file_name in excluded:
        continue
    bot_df = clean_df[clean_df['file_name'] == file_name]
    plt.plot(bot_df['timestamp'], bot_df['player_elo'], label=file_name, alpha=1)

# Customize the plot
plt.title('ELO Progression Over Time for Each Bot')
plt.xlabel('Timestamp')
plt.ylabel('ELO')
plt.grid(True)
plt.legend(title='Bot Version (File)')
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.tight_layout()

# Show the plot
plt.show()



# Plot the ELO over time for each bot version
plt.figure(figsize=(12, 6))

# Remove rows with "Unknown" in the file_name column
clean_df = combined_df[combined_df['file_name'] != "Unknown"].copy()
clean_df['timestamp'] = clean_df['timestamp'].dt.tz_convert(None)

excluded = ["approvers", "linrock"]

# Loop through each file_name and plot the data
for file_name in clean_df['file_name'].unique():
    if file_name in excluded:
        continue
    bot_df = clean_df[clean_df['file_name'] == file_name]
    plt.plot(bot_df['timestamp'], bot_df['player_elo'], label=file_name, alpha=1)

# Customize the plot
plt.title('ELO Progression Over Time for Each Bot')
plt.xlabel('Timestamp')
plt.ylabel('ELO')
plt.grid(True)
plt.legend(title='Bot Version (File)')
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.tight_layout()

# Show the plot
plt.show()



# Plot the ELO over time for each bot version
plt.figure(figsize=(12, 6))

# Remove rows with "Unknown" in the file_name column
clean_df = combined_df[combined_df['file_name'] != "Unknown"].copy()
clean_df['timestamp'] = clean_df['timestamp'].dt.tz_convert(None)

excluded = ["approvers", "fix_the_bugs"]

# Loop through each file_name and plot the data
for file_name in clean_df['file_name'].unique():
    if file_name in excluded:
        continue
    bot_df = clean_df[clean_df['file_name'] == file_name]
    plt.plot(bot_df['timestamp'], bot_df['player_elo'], label=file_name, alpha=1)

# Customize the plot
plt.title('ELO Progression Over Time for Each Bot')
plt.xlabel('Timestamp')
plt.ylabel('ELO')
plt.grid(True)
plt.legend(title='Bot Version (File)')
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.tight_layout()

# Show the plot
plt.show()



# Plot the ELO over time for each bot version
plt.figure(figsize=(12, 6))

# Remove rows with "Unknown" in the file_name column
clean_df = combined_df[combined_df['file_name'] != "Unknown"].copy()
clean_df['timestamp'] = clean_df['timestamp'].dt.tz_convert(None)

excluded = ["fix_the_bugs", "linrock"]

# Loop through each file_name and plot the data
for file_name in clean_df['file_name'].unique():
    if file_name in excluded:
        continue
    bot_df = clean_df[clean_df['file_name'] == file_name]
    plt.plot(bot_df['timestamp'], bot_df['player_elo'], label=file_name, alpha=1)

# Customize the plot
plt.title('ELO Progression Over Time for Each Bot')
plt.xlabel('Timestamp')
plt.ylabel('ELO')
plt.grid(True)
plt.legend(title='Bot Version (File)')
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.tight_layout()

# Show the plot
plt.show()



# Plot the ELO over time for each bot version
plt.figure(figsize=(12, 6))

# Remove rows with "Unknown" in the file_name column
clean_df = combined_df[combined_df['file_name'] != "Unknown"].copy()
clean_df['timestamp'] = clean_df['timestamp'].dt.tz_convert(None)

clean_df['timestamp'] = clean_df['timestamp'].dt.floor('5T')  # 'T' = minute

# Group by 'file_name' and 'timestamp', keeping only the max 'player_elo'
clean_df = clean_df.groupby(['file_name', 'timestamp'], as_index=False)['player_elo'].max()

# Loop through each file_name and plot the data
for file_name in clean_df['file_name'].unique():
    bot_df = clean_df[clean_df['file_name'] == file_name]
    plt.plot(bot_df['timestamp'], bot_df['player_elo'], label=file_name)

# Customize the plot
plt.title('ELO Progression Over Time for Each Bot')
plt.xlabel('Timestamp')
plt.ylabel('ELO')
plt.grid(True)
plt.legend(title='Bot Version (File)')
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.tight_layout()

# Show the plot
plt.show()



# Plot the ELO over time for each bot version
plt.figure(figsize=(12, 6))

# Remove rows with "Unknown" in the file_name column
clean_df = combined_df[combined_df['file_name'] != "Unknown"].copy()
clean_df['timestamp'] = clean_df['timestamp'].dt.tz_convert(None)

excluded = ["approvers", "fix_the_bugs", "linrock", "niboshi", "ascalon", "ymg_aq"]

# Loop through each file_name and plot the data
for file_name in clean_df['file_name'].unique():
    if file_name in excluded:
        continue
    bot_df = clean_df[clean_df['file_name'] == file_name]
    plt.plot(bot_df['timestamp'], bot_df['player_elo'], label=file_name)

# Customize the plot
plt.title('ELO Progression Over Time for Each Bot')
plt.xlabel('Timestamp')
plt.ylabel('ELO')
plt.grid(True)
plt.legend(title='Bot Version (File)')
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.tight_layout()
# plt.ylim(2000, 2100)

# Show the plot
plt.show()



# Plot the ELO over time for each bot version
plt.figure(figsize=(12, 6))

# Remove rows with "Unknown" in the file_name column
clean_df = combined_df[combined_df['file_name'] != "Unknown"].copy()
clean_df['timestamp'] = clean_df['timestamp'].dt.tz_convert(None)

excluded = ["approvers", "fix_the_bugs", "linrock"]

# Loop through each file_name and plot the data
for file_name in clean_df['file_name'].unique():
    if file_name in excluded:
        continue
    bot_df = clean_df[clean_df['file_name'] == file_name]
    plt.plot(bot_df['timestamp'], bot_df['player_elo'], label=file_name)

# Customize the plot
plt.title('ELO Progression Over Time for Each Bot')
plt.xlabel('Timestamp')
plt.ylabel('ELO')
plt.grid(True)
plt.legend(title='Bot Version (File)')
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.tight_layout()
# plt.ylim(2000, 2100)

# Show the plot
plt.show()




