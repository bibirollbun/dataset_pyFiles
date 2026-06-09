### Loading Relevant Packages

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps


path = "/kaggle/input/stanford-rna-3d-folding/"
os.chdir(path)


train_label_raw = pd.read_csv("train_labels.csv")
train_label_raw.describe()


train_label_raw.describe()


train_label_raw.tail()


### Let's focus on calculating the Euclidean distances between 2 consecutive C-1 atoms



### Define a function

def euclidean_distance(df):

    j = 0

    x1_first = df["x_1"].iloc[j]
    y1_first = df["y_1"].iloc[j]
    z1_first = df["z_1"].iloc[j]
    
    j = 1
    
    x1_second = df["x_1"].iloc[j]
    y1_second = df["y_1"].iloc[j]
    z1_second = df["z_1"].iloc[j]
    
    d_Euclidean = np.sqrt(np.power( (x1_second - x1_first), 2) + np.power( (y1_second - y1_first), 2) + np.power( (z1_second - z1_first), 2) )

    return d_Euclidean






euclidean_distance(train_label_raw)


def two_letter_identification(df, i):
    return df["resname"][i] + df["resname"][i+1]

two_letter_identification(train_label_raw, 0)


L_index = list()
L_2_Letters = list()
L_2_distance = list()
L_data_distance = list()


# Go through the entire dataset row by row
for i in range(0, len(train_label_raw) - 1):

    # Check that the 2 consecutive C-1 atoms are part of the same RNA chain
    if train_label_raw["resid"].iloc[i] == (train_label_raw["resid"].iloc[(i+1)] - 1):

        # Store the index
        L_index.append(i)

        # Store the 2 consecutive letters 
        L_2_Letters.append(two_letter_identification(train_label_raw[i:(i+2)], i))

        # Store the Euclidean distance between the 2 consecutive letters
        L_2_distance.append(euclidean_distance(train_label_raw[i:(i+2)]))

        # Make a list of list containing the index, the 2 consecutive letters and the Euclidean distance
        L_data_distance.append([i, two_letter_identification(train_label_raw[i:(i+2)], i), euclidean_distance(train_label_raw[i:(i+2)])])



### Check the list of list

L_data_distance[:10]


## Create a dataframe

df_data_distances_raw = pd.DataFrame(L_data_distance)
df_data_distances_raw.columns = ["index","pair","distance"]
df_data_distances_raw


# Remove the NaN

df_data_distances = df_data_distances_raw[df_data_distances_raw["distance"] == df_data_distances_raw["distance"]]
df_data_distances


# Calculate the mean distance in Angstroms between pairs

grouped = df_data_distances.groupby("pair")["distance"].mean(numeric_only=True)
grouped = grouped.sort_values(ascending=False)
grouped


# Plot the mean distance in Angstroms between pairs

plt.figure(figsize=(12, 6))
num_pairs = len(grouped)
cmap = colormaps['Blues']
colors = [cmap(0.6 + 0.4 * i / max(num_pairs - 1, 1)) for i in range(num_pairs)]  # dark blues only

# Plot
plt.figure(figsize=(12, 6))
plt.bar(grouped.index, grouped.values, color=colors)

# Styling
plt.title("Average Distance by Pair", fontsize=16)
plt.xlabel("Pair", fontsize=12)
plt.ylabel("Average Distance", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()


# Plot the mean standard deviation for the distance (Angstroms) between pairs

df_data_distances.groupby("pair")["distance"].std(numeric_only=True)


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps

# Calculate mean and standard deviation
mean_dist = df_data_distances.groupby("pair")["distance"].mean(numeric_only=True)
std_dist = df_data_distances.groupby("pair")["distance"].std(numeric_only=True)

# Sort by mean for consistency
mean_dist = mean_dist.sort_values(ascending=False)
std_dist = std_dist[mean_dist.index]  # reindex to match order

# Set up distinct colors
num_pairs = len(mean_dist)
cmap = colormaps['Blues']
colors = [cmap(0.6 + 0.4 * i / max(num_pairs - 1, 1)) for i in range(num_pairs)]  # dark blues only

# Plot
plt.figure(figsize=(12, 6))
plt.bar(mean_dist.index, mean_dist.values, yerr=std_dist.values, color=colors, capsize=5)

# Styling
plt.title("Average Distance by Pair (with Standard Deviation)", fontsize=16)
plt.xlabel("Pair", fontsize=12)
plt.ylabel("Average Distance", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()



# Calculate a histogram of the distances for each pair

import matplotlib.pyplot as plt
import numpy as np

# Define number of bins
num_bins = 35

# Compute global min and max for distances
xmin = df_data_distances["distance"].min()
xmax = df_data_distances["distance"].max()

# Generate consistent bin edges
bins = np.linspace(xmin, xmax, num_bins + 1)

# Loop through each pair
for pair in df_data_distances["pair"].unique():
    subset = df_data_distances[df_data_distances["pair"] == pair]["distance"].dropna()
    
    if subset.empty:
        continue

    plt.figure(figsize=(8, 4))
    plt.hist(subset, bins=bins, color='skyblue', edgecolor='black')
    plt.xlim(xmin, xmax)
    plt.title(f"Histogram of Distances for Pair: {pair}")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()



### These are very asymetric distriubtions. Some of them appear to have very large distances on the tails.


df_data_distances[df_data_distances["distance"] >20]



# This sounds off. Either a data problem, an experimental problem or an exception.


# There are some large distances of over 20 Angstroms. This is surprising. Let's have a sanity chec on one


train_label_raw.iloc[12182:12186]


# Yes, there is a large jump from 12183 to 12184.


# Now, let's focus on the angles (theta) formed by the 3 consecutive C-1 atoms


### Define a function

def angle_theta(df):

    j = 0

    x1_first = df["x_1"].iloc[j]
    y1_first = df["y_1"].iloc[j]
    z1_first = df["z_1"].iloc[j]
    
    j = 1
    
    x1_second = df["x_1"].iloc[j]
    y1_second = df["y_1"].iloc[j]
    z1_second = df["z_1"].iloc[j]

    j = 2

    x1_third = df["x_1"].iloc[j]
    y1_third = df["y_1"].iloc[j]
    z1_third = df["z_1"].iloc[j]
    
        
    # Define vectors to the 3 consecutive C-1 atoms
    v_c1 = np.array([x1_first, y1_first, z1_first])
    v_c2 = np.array([x1_second, y1_second, z1_second])
    v_c3 = np.array([x1_third, y1_third, z1_third])
    
    # vector connecting the C-1 and C-2 as well as the vector connecting C-3 and C-2
    v1 = v_c1 - v_c2
    v2 = v_c3 - v_c2
    
    # Dot product between the two vectors
    dot_product = np.dot(v1, v2)
    
    # Norms (magnitudes)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Angle in radians
    theta_rad = np.arccos(dot_product / (norm_v1 * norm_v2))
    
    # Angle in degrees
    theta_deg = np.degrees(theta_rad)
    
    return theta_deg
    



angle_theta(train_label_raw)


def three_letter_identification(df, i):
    return df["resname"][i] + df["resname"][i+1] + df["resname"][i+2]

three_letter_identification(train_label_raw, 0)


L_index = list()
L_3_Letters = list()
L_angle = list()
L_data_angle = list()


# Go through the entire dataset row by row
for i in range(0, len(train_label_raw) - 2):

    # Check that the 3 consecutive C-1 atoms are part of the same RNA chain
    condition_1 = train_label_raw["resid"].iloc[i] == (train_label_raw["resid"].iloc[(i+1)] - 1)
    condition_2 = train_label_raw["resid"].iloc[(i+1)] == (train_label_raw["resid"].iloc[(i+2)] - 1)
    
    if (condition_1 and condition_2):

        # Store the index
        L_index.append(i)

        # Store the 3 consecutive letters 
        L_3_Letters.append(three_letter_identification(train_label_raw[i:(i+3)], i))

        # Store the Euclidean distance between the 3 consecutive letters
        L_angle.append(angle_theta(train_label_raw[i:(i+3)]))

        # Make a list of list containing the index, the 2 consecutive letters and the Euclidean distance
        L_data_angle.append([i, three_letter_identification(train_label_raw[i:(i+3)], i), angle_theta(train_label_raw[i:(i+3)])])



L_data_angle[:10]


## Create a dataframe

df_data_angles_raw = pd.DataFrame(L_data_angle)
df_data_angles_raw.columns = ["index","3-Letters","angle"]
df_data_angles_raw


# Remove the NaN

df_data_angles = df_data_angles_raw[df_data_angles_raw["angle"] == df_data_angles_raw["angle"]]
df_data_angles


grouped = df_data_angles.groupby("3-Letters")["angle"].mean(numeric_only=True)
grouped = grouped.sort_values(ascending=False)
grouped


plt.figure(figsize=(12, 6))
num_pairs = len(grouped)
cmap = colormaps['Blues']
colors = [cmap(0.6 + 0.4 * i / max(num_pairs - 1, 1)) for i in range(num_pairs)]  # dark blues only

# Plot
plt.figure(figsize=(12, 6))
plt.bar(grouped.index, grouped.values, color=colors)

# Styling
plt.title("Average Angle (degrees) between 3 consecutive bases", fontsize=16)
plt.xlabel("3 consecutive bases", fontsize=12)
plt.ylabel("Average Angle (degrees)", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()


# There is a distribution of angles. Most of the angles are between 100 and 160 degres.


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps

# Calculate mean and standard deviation
mean_dist = df_data_angles.groupby("3-Letters")["angle"].mean(numeric_only=True)
std_dist = df_data_angles.groupby("3-Letters")["angle"].std(numeric_only=True)

# Sort by mean for consistency
mean_dist = mean_dist.sort_values(ascending=False)
std_dist = std_dist[mean_dist.index]  # reindex to match order

# Set up distinct colors
num_pairs = len(mean_dist)
cmap = colormaps['Blues']
colors = [cmap(0.6 + 0.4 * i / max(num_pairs - 1, 1)) for i in range(num_pairs)]  # dark blues only

# Plot
plt.figure(figsize=(12, 6))
plt.bar(mean_dist.index, mean_dist.values, yerr=std_dist.values, color=colors, capsize=5)

# Styling
plt.title("Average Angle (degrees) between 3 consecutive bases", fontsize=16)
plt.xlabel("3 consecutive bases", fontsize=12)
plt.ylabel("Average Angle (degrees)", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()



import matplotlib.pyplot as plt
import numpy as np

# Define number of bins
num_bins = 35

# Compute global min and max for distances
xmin = df_data_angles["angle"].min()
xmax = df_data_angles["angle"].max()

# Generate consistent bin edges
bins = np.linspace(xmin, xmax, num_bins + 1)

# Loop through each pair
for pair in df_data_angles["3-Letters"].unique():
    subset = df_data_angles[df_data_angles["3-Letters"] == pair]["angle"].dropna()
    
    if subset.empty:
        continue

    plt.figure(figsize=(8, 4))
    plt.hist(subset, bins=bins, color='skyblue', edgecolor='black')
    plt.xlim(xmin, xmax)
    plt.title(f"Histogram of Angles (degrees) for 3 consecutive C-1 atoms for a specific sequence: {pair}")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()



# The specific consecutive sequence of 3 nitrogenous bases leads to a distribution of angles.
# The distributions are often left-skewed. The mean, median and mode changes depending on the specific 3 nitrogenous base pairs.
# It is interesting that most histogram have on peak like the GGG histogram.
# However, there are histogram with two peaks like the GAA histogram.




