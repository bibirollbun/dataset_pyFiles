# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#load in datasets
PlayerPlay = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/player_play.csv')
Play=pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/plays.csv')
PlayerInfo= pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/players.csv')



import matplotlib.pyplot as plt

# Updated data for the pie chart
routes = ['GO', 'HITCH', 'IN', 'CROSS', 'OUT', 'POST', 'SLANT', 'CORNER', 'FLAT', 'SCREEN', 'ANGLE']
percentages = [22.877784, 15.291993, 12.703191, 11.077664, 9.512342, 8.127634, 7.826610, 5.779651, 3.913305, 2.769416, 0.120409]

# Create the pie chart
fig, ax = plt.subplots()
ax.pie(percentages, labels=[f'{route} ({percentage:.1f}%)' for route, percentage in zip(routes, percentages)], 
       autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)

# Set chart title
ax.set_title("Distribution of Routes Run by WRs in 3x0, 3x1, 3x2 Alignment on Redzone pass plays")

# Display the chart
plt.show()



routes = [
    'GO', 'HITCH', 'IN', 'OUT', 'CROSS', 'POST', 
    'SLANT', 'CORNER', 'FLAT', 'SCREEN', 'ANGLE'
]
percentages = [
    5.679012, 4.979424, 4.609053, 4.320988, 
    3.950617, 3.909465, 3.374486, 2.962963, 
    1.769547, 1.522634, 0.082305
]

# Create the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(routes, percentages, color='skyblue')

# Set chart title and labels
plt.title('Probability of each route being ran by a WR in a 3x0, 3x1, 3x2 Alignment on a Redzone pass Play')
plt.xlabel('Route Names')
plt.ylabel('Percentage')
plt.ylim(0, 100)  # Set y-axis limit to 100%
plt.yticks(range(0, 101, 10))  # Set y-axis ticks every 10 percent

# Add percentage values on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2, 
        yval + 1,  # Position above the bar
        f'{yval:.1f}%',  # Format to one decimal place
        ha='center', va='bottom'
    )

# Show the chart
plt.show()


 #Combine PlayerPlay and Play datasets
CombinedData = pd.merge(Play, PlayerPlay, on='playId', how='inner')
#Combine The other datasets with PlayerInfo Dataset
CombinedData= pd.merge(CombinedData,PlayerInfo, on='nflId', how='inner')
#filter out for plays that meet the conditions
CombinedData = CombinedData[(CombinedData['absoluteYardlineNumber'] <= 20) & (CombinedData['receiverAlignment'].isin(['3x0', '3x1', '3x2'])) & 
(CombinedData['isDropback'] == True) & (CombinedData['position'] == 'WR')]
# Count how many times each route was ran 
Count_Routes = CombinedData['routeRan'].value_counts()
# Total number of routes
Total_Routes = Count_Routes.sum()
# Formula to calculate route percentages
Route_Percentages = (Count_Routes / Total_Routes) * 100
#show route percentages
print(Route_Percentages,Count_Routes)



# Combine PlayerPlay dataset with the filtered Play dataset
CombinedData = pd.merge(Play, PlayerPlay, on='playId', how='inner')
#Combine PlayerInfo dataset with others
CombinedData= pd.merge(CombinedData,PlayerInfo, on='nflId', how='inner')
#filter out for plays that meet the conditions
CombinedData = CombinedData[(CombinedData['absoluteYardlineNumber'] <= 20) & 
(CombinedData['receiverAlignment'].isin(['3x0', '3x1', '3x2'])) & (CombinedData['isDropback'] == True) & (CombinedData['position'] == 'WR')]
# Drop duplicate routes within each play
UniqueRoutes_PerPlay = CombinedData[['playId', 'routeRan']].drop_duplicates()
# Count each route once everytime it appeared in a play
Route_Appearance_Count = UniqueRoutes_PerPlay['routeRan'].value_counts()
# Total number of plays that meet the requirements
Total_Plays = len(CombinedData)
# Calculate the probability of each route appearing in a play
Route_Probabilities = (Route_Appearance_Count / Total_Plays)*100
# Show the probabilities of each route appearing
print('Route appearnce %')
print(Route_Probabilities)
print('Route appearnce count')
print(Route_Appearance_Count)


# Data
routes = [
    'CROSS', 'GO', 'OUT', 'HITCH', 'IN', 'SLANT', 
    'SCREEN', 'CORNER', 'POST', 'FLAT', 'ANGLE'
]
percentages = [
    14.244186, 13.953488, 13.372093, 12.500000, 11.046512,
    10.755814, 6.976744, 6.976744, 5.813953, 4.069767, 0.290698
]

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(percentages, labels=routes, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)

# Set chart title
plt.title(' Distribution of Route Targets in a 3x0, 3x1, 3x2 Alignment on a Redzone Pass Play')

# Show the chart
plt.show()


# Combine PlayerPlay dataset with the filtered Play dataset
CombinedData = pd.merge(Play, PlayerPlay, on='playId', how='inner')
#Combine PlayerInfo dataset with others
CombinedData= pd.merge(CombinedData,PlayerInfo, on='nflId', how='inner')
#filter out for plays that meet the conditions
CombinedData = CombinedData[(CombinedData['absoluteYardlineNumber'] <= 20) & 
(CombinedData['receiverAlignment'].isin(['3x0', '3x1', '3x2'])) & (CombinedData['isDropback'] == True) & (CombinedData['position'] == 'WR') & (CombinedData['wasTargettedReceiver']== 1)]
# # Count each route everytime it was targeted
Count_Routes= CombinedData['routeRan'].value_counts()
# Count all targeted routes together
Total_Routes=Count_Routes.sum()
# get route target percentages
Target_Percentages=(Count_Routes/Total_Routes)*100
#show route Target Percentages
print("Route Target percentages")
print(Target_Percentages)
print("Route Target Count")
print(Count_Routes)


import matplotlib.pyplot as plt

# Data
routes = [
    'CROSS', 'GO', 'OUT', 'HITCH', 'IN', 
    'SLANT', 'SCREEN', 'CORNER', 'POST', 'FLAT', 'ANGLE'
]
probabilities = [
    2.016461, 1.975309, 1.893004, 1.769547, 1.563786, 
    1.522634, 0.987654, 0.987654, 0.823045, 0.576132, 0.041152
]

# Create the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(routes, probabilities, color='skyblue')

# Set chart title and labels
plt.title('Probability of Each Route Being Targeted in a 3x0, 3x1, 3x2 Alignment on a Redzone pass Play')
plt.xlabel('Route Names')
plt.ylabel('Probability (%)')
plt.ylim(0, 2.5)  # Set y-axis limit for better visualization
plt.yticks(range(0, 3, 1))  # Set y-axis ticks

# Add probability values on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2, 
        yval + 0.05,  # Position above the bar
        f'{yval:.2f}%',  # Format to two decimal places
        ha='center', va='bottom'
    )

# Show the chart
plt.show()


# Combine PlayerPlay dataset with the filtered Play dataset
CombinedData = pd.merge(Play, PlayerPlay, on='playId', how='inner')
#Combine PlayerInfo dataset with others
CombinedData= pd.merge(CombinedData,PlayerInfo, on='nflId', how='inner')
#filter out for plays that meet the conditions
CombinedData = CombinedData[(CombinedData['absoluteYardlineNumber'] <= 20) & 
(CombinedData['receiverAlignment'].isin(['3x0', '3x1', '3x2'])) & (CombinedData['isDropback'] == True) & (CombinedData['position'] == 'WR')]
# Calculate the targeted occurrences for each route
Targeted_Route_Count = CombinedData[CombinedData['wasTargettedReceiver'] == 1]['routeRan'].value_counts()
# Calculate the probability of each route being targeted in all plays
Route_Probability = (Targeted_Route_Count / len(CombinedData)) * 100
# show route target the probabilities
print("Probability of Each Route Being Targeted %")
print(Route_Probability)
print("Route Target count")
print(Targeted_Route_Count)


# Data
routes = ['OUT', 'HITCH', 'CROSS', 'SLANT', 'GO', 'IN', 'POST', 'SCREEN', 'FLAT', 'CORNER', 'ANGLE']
percentages = [16.725979, 16.370107, 14.946619, 13.879004, 13.523132, 7.829181, 5.338078, 4.626335, 3.202847, 2.491103, 1.067616]

# Create formatted labels combining route names and percentages
labels = [f'{route} ({percentage:.1f}%)' for route, percentage in zip(routes, percentages)]

# Create the pie chart with both route names and percentages inside each slice
plt.figure(figsize=(10, 7))
plt.pie(percentages, labels=labels, autopct='%1.1f%%', startangle=140)

# Set chart title
plt.title('Distribution of Routes Run by WRs in 2x0, 2x1, 2x2 Alignment on Redzone pass plays')

# Show the chart
plt.show()


# Combine PlayerPlay dataset with the filtered Play dataset
CombinedData = pd.merge(Play, PlayerPlay, on='playId', how='inner')
#Combine PlayerInfo dataset with others
CombinedData= pd.merge(CombinedData,PlayerInfo, on='nflId', how='inner')
#filter out for plays that meet the conditions
CombinedData = CombinedData[(CombinedData['absoluteYardlineNumber'] <= 20) & 
(CombinedData['receiverAlignment'].isin(['2x0', '2x1', '2x2'])) & (CombinedData['isDropback'] == True) & (CombinedData['position'] == 'WR') & (CombinedData['wasTargettedReceiver']== 1)]
# # Count each route everytime it was targeted
Count_Routes= CombinedData['routeRan'].value_counts()
# Count all targeted routes together
Total_Routes=Count_Routes.sum()
# get route target percentages
Target_Percentages=(Count_Routes/Total_Routes)*100
#show route Target Percentages
print("Route Target Distrubution")
print(Target_Percentages)
print("Route Target Count")
print(Count_Routes)


import matplotlib.pyplot as plt

# Data
routes = [
    'GO', 'HITCH', 'CROSS', 'OUT', 'IN', 'POST',
    'SLANT', 'CORNER', 'FLAT', 'SCREEN', 'ANGLE'
]
percentages = [
    6.589492, 5.520926, 5.075690, 4.363313, 4.096171,
    3.650935, 3.561888, 2.359751, 2.048085, 1.380232, 0.133571
]

# Create the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(routes, percentages, color='lightcoral')

# Set chart title and labels
plt.title('Probability of each route being ran by a WR in a 2x0, 2x1, 2x2 Alignment on a Redzone pass Play')
plt.xlabel('Route Names')
plt.ylabel('Percentage')
plt.ylim(0, 100)  # Set y-axis limit to 100%
plt.yticks(range(0, 101, 10))  # Set y-axis ticks every 10 percent

# Add percentage values on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 1,  # Position above the bar
        f'{yval:.1f}%',  # Format to one decimal place
        ha='center', va='bottom'
    )

# Show the chart
plt.show()


#Combined datasets
CombinedData= pd.merge(Play, PlayerPlay, on='playId', how='inner')
#Combined PlayerInfo dataset with the others
CombinedData=pd.merge(CombinedData, PlayerInfo, on='nflId', how='inner')
#Filter the combined Datasets
CombinedData=CombinedData[(CombinedData['absoluteYardlineNumber'] <= 20)&(CombinedData['receiverAlignment'].isin(['2x2','2x1','2x0'])) & 
(CombinedData['isDropback']==True)& (CombinedData['position'] == 'WR')]
# drop duplicate routes within each play
UniqueRoutes_PerPlay= CombinedData[['playId','routeRan']].drop_duplicates()
#Count each route once everytime it appeared in a play
Route_Appearance= UniqueRoutes_PerPlay['routeRan'].value_counts()
## Total number of plays that meet the requirements
Total_Plays=len(CombinedData)
## Calculate the probability of each route appearing in a play
Route_probabilities= (Route_Appearance/Total_Plays)*100
# Show the probabilities of each route appearing
print("Route apperance probability")
print(Route_probabilities)
print("Route Apperance Count")
print(Route_Appearance)


# Data
routes = [
    'OUT', 'HITCH', 'CROSS', 'SLANT', 'GO', 'IN', 
    'POST', 'SCREEN', 'FLAT', 'CORNER', 'ANGLE'
]
probabilities = [
    2.092609, 2.048085, 1.869991, 1.736420, 1.691897,
    0.979519, 0.667854, 0.578807, 0.400712, 0.311665, 0.133571
]

# Create the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(routes, probabilities, color='lightcoral')

# Set chart title and labels
plt.title('Probability of Each Route Being Targeted in a 2x0, 2x1, 2x2 Alignment on a Redzone pass Play')
plt.xlabel('Route Names')
plt.ylabel('Probability (%)')
plt.ylim(0, 3)  # Set y-axis limit based on the data
plt.yticks(range(0, 4, 1))  # Set y-axis ticks every 1 percent

# Add probability values on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 0.05,  # Position above the bar
        f'{yval:.2f}%',  # Format to two decimal places
        ha='center', va='bottom'
    )

# Show the chart
plt.show()


# Combine PlayerPlay dataset with the filtered Play dataset
CombinedData = pd.merge(Play, PlayerPlay, on='playId', how='inner')
#Combine PlayerInfo dataset with others
CombinedData= pd.merge(CombinedData,PlayerInfo, on='nflId', how='inner')
#filter out for plays that meet the conditions
CombinedData = CombinedData[(CombinedData['absoluteYardlineNumber'] <= 20) & 
(CombinedData['receiverAlignment'].isin(['2x0', '2x1', '2x2'])) & (CombinedData['isDropback'] == True) & (CombinedData['position'] == 'WR')]
# Calculate the targeted occurrences for each route
Targeted_Route_Count = CombinedData[CombinedData['wasTargettedReceiver'] == 1]['routeRan'].value_counts()
# Calculate the probability of each route being targeted in all plays
Route_Probability = (Targeted_Route_Count / len(CombinedData)) * 100
# show route target the probabilities
print("Probability of Each Route Being Targeted %")
print(Route_Probability)
print("Route Target count")
print(Targeted_Route_Count)



# Example data from your input
data = {
    'x': [87.70, 84.16, 45.74, 45.37, 63.59, 9.77, 83.28, 80.99, 107.82, 88.55, 69.93, 60.06, 50.19, 43.84, 22.50, 47.23, 73.48, 48.98, 19.15, 78.89, 76.26, 78.86, 49.47, 80.26, 80.93, 36.91, 29.04, 65.90, 60.74, 90.55, 111.91, 32.62, 50.02, 78.90, 67.64, 36.33, 61.49, 41.83, 40.53, 44.56, 16.10, 17.90, 76.53],
    'y': [23.75, 36.47, 28.22, 21.89, 30.66, 37.55, 27.02, 32.20, 31.86, 25.90, 35.50, 22.43, 28.33, 28.58, 24.55, 32.47, 24.06, 26.58, 29.27, 30.48, 29.23, 29.67, 26.32, 21.59, 25.44, 28.29, 26.37, 26.97, 29.13, 25.28, 29.63, 22.56, 39.43, 26.42, 35.84, 23.81, 25.54, 30.31, 27.14, 27.45, 27.25, 34.01, 28.42],
    'position': ['ILB', 'CB', 'ILB', 'SS', 'ILB', 'FS', 'ILB', 'ILB', 'ILB', 'ILB', 'ILB', 'ILB', 'ILB', 'ILB', 'ILB', 'CB', 'SS', 'FS', 'ILB', 'FS', 'ILB', 'ILB', 'ILB', 'ILB', 'FS', 'ILB', 'ILB', 'SS', 'FS', 'CB', 'SS', 'ILB', 'CB', 'SS', 'SS', 'ILB', 'ILB', 'CB', 'CB', 'ILB', 'CB', 'SS', 'CB']
}
# Convert to DataFrame
df = pd.DataFrame(data)
# Field dimensions
field_length = 120
field_width = 53.3
# Color mapping for positions
color_mapping = {'CB': 'blue', 'ILB': 'purple', 'FS': 'green', 'SS': 'orange'}
# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
# Draw the field outline
field_outline = patches.Rectangle((0, 0), field_length, field_width, linewidth=2, edgecolor='black', facecolor='green')
ax.add_patch(field_outline)
# Add end zones
end_zone_left = patches.Rectangle((0, 0), 10, field_width, linewidth=1, edgecolor='black', facecolor='lightblue')
end_zone_right = patches.Rectangle((110, 0), 10, field_width, linewidth=1, edgecolor='black', facecolor='lightblue')
ax.add_patch(end_zone_left)
ax.add_patch(end_zone_right)
# Add 10-yard lines
for x in range(10, 120, 10):
    ax.plot([x, x], [0, field_width], color='white', linestyle='--', linewidth=1)
# Highlight 20-yard lines
ax.plot([20, 20], [0, field_width], color='yellow', linestyle='-', linewidth=3, label='20-yard line')
ax.plot([100, 100], [0, field_width], color='yellow', linestyle='-', linewidth=3)
# Plot the player positions
for _, row in df.iterrows():
    ax.scatter(row['x'], row['y'], color=color_mapping[row['position']], label=row['position'], s=50)
# Avoid duplicate legends
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())
# Set axis limits and labels
ax.set_xlim(0, field_length)
ax.set_ylim(0, field_width)
ax.set_aspect('equal')
ax.set_xticks(range(0, 121, 10))
ax.set_yticks(range(0, 54, 5))
ax.set_xticklabels(range(0, 121, 10))
ax.set_yticklabels(range(0, 54, 5))
ax.grid(False)
# Add labels
ax.set_title("Football Field with Player Positions")
ax.set_xlabel("X-axis (yards)")
ax.set_ylabel("Y-axis (yards)")

# Show the plot
plt.show()



Play2 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/plays.csv', usecols=['gameId', 'absoluteYardlineNumber','playDescription'])
PlayerPlay2 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/player_play.csv', usecols=['gameId','nflId','playId','wasInitialPassRusher','causedPressure']  )
PlayerInfo2= pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/players.csv', usecols=['nflId','position'] ) 
TrackdataWeek9=pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_9.csv',  usecols=['nflId','playId','gameId','x','y','displayName','frameType','playDirection'])

PlayerPlay2=PlayerPlay2[(PlayerPlay2['causedPressure']==True)& PlayerPlay2['wasInitialPassRusher']==1]
PlayerInfo2=PlayerInfo2[PlayerInfo2['position'].isin(['SS','CB','FS','ILB',])]
Play2=Play2[Play2['absoluteYardlineNumber'] <=20]
TrackdataWeek9 = TrackdataWeek9[(TrackdataWeek9['frameType'] == 'BEFORE_SNAP') ]

# Grouping with gameId
TrackdataWeek9 = TrackdataWeek9.groupby(['nflId', 'gameId', 'playId']).first().reset_index()

# Merge operations
CombinedData = pd.merge(PlayerPlay2, TrackdataWeek9, on=['nflId', 'gameId', 'playId'], how='inner')
CombinedData = pd.merge(CombinedData, PlayerInfo2, on='nflId', how='inner')
CombinedData = pd.merge(CombinedData, Play2, on='gameId', how='inner')

# Remove duplicates if necessary
CombinedData = CombinedData.drop_duplicates(subset=['nflId', 'gameId', 'playId'])
BlitzCount= CombinedData['position'].value_counts()
#print(len(CombinedData))
print(CombinedData[['x','y','position']])
print('Blitz count for each position')
print(BlitzCount)

