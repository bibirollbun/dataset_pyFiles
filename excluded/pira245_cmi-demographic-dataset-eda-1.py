# General use default library
import numpy as np
import pandas as pd
import time
# visualisation library
import IPython.display
from IPython.display import Image, display
import matplotlib
from matplotlib import pyplot as plt
import itertools
from itertools import cycle
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
sns.set(rc={'figure.figsize':(12,14)})
sns.set_theme()


import cmi_comp_utils_functions as my_utils


filepath = my_utils.Path('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
raw_train_demographics_df = pd.read_csv(filepath, sep=',', encoding='utf-8', na_filter=True)


raw_train_demographics_df.shape


raw_train_demographics_df.head(10)


train_shape = raw_train_demographics_df.shape
train_dtypes = raw_train_demographics_df.dtypes
missing_counts = raw_train_demographics_df.isna().sum()
missing_percent = ((raw_train_demographics_df.isnull().sum() / len(raw_train_demographics_df)) * 100).sort_values()

summary = f"""
Train Demographics Dataset Summary:
- Shape: {train_shape[0]} rows, {train_shape[1]} columns
- Data types:
{train_dtypes.to_string()}
- Missing values (count):
{missing_counts.to_string()}
- Missing values (percentage):
{missing_percent.to_string()}

"""
print(summary)


TRAIN_DEMO_COLUMNS = [
    'subject',
    'adult_child',       
    'age',    
    'sex',     
    'handedness',     
    'height_cm',        
    'shoulder_to_wrist_cm', 
    'elbow_to_wrist_cm', 
]
print(TRAIN_DEMO_COLUMNS)


sns.pairplot(raw_train_demographics_df[TRAIN_DEMO_COLUMNS], corner=True)


# Plot 3 histograms
my_utils.display_3x_histoplot(raw_train_demographics_df, ['sex', 'adult_child', 'handedness'], figsize=(18, 5))


sns.histplot(data=raw_train_demographics_df, x="age", hue="sex", stat="percent")


sns.histplot(data=raw_train_demographics_df, x="height_cm", hue="sex", stat="percent")


# Scatter plot: shoulder_to_wrist_cm vs elbow_to_wrist_cm
fig, axes = plt.subplots(1, 1, figsize=(6, 6))
sns.scatterplot(
    data=raw_train_demographics_df,
    x="shoulder_to_wrist_cm",
    y="elbow_to_wrist_cm"
)
plt.title("Scatter Plot: Shoulder to Wrist vs Elbow to Wrist")
plt.xlabel("Shoulder to Wrist (cm)")
plt.ylabel("Elbow to Wrist (cm)")
plt.show()

# Boxplot for both variables
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(y=raw_train_demographics_df["shoulder_to_wrist_cm"], ax=axes[0])
axes[0].set_title("Boxplot: Shoulder to Wrist (cm)")
sns.boxplot(y=raw_train_demographics_df["elbow_to_wrist_cm"], ax=axes[1])
axes[1].set_title("Boxplot: Elbow to Wrist (cm)")
plt.tight_layout()
plt.show()


# Calculate IQR for 'shoulder_to_wrist_cm'

q1 = raw_train_demographics_df['shoulder_to_wrist_cm'].quantile(0.25)
q3 = raw_train_demographics_df['shoulder_to_wrist_cm'].quantile(0.75)
iqr = q3 - q1

# Define outlier bounds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Find outlier rows
outliers = raw_train_demographics_df[
    (raw_train_demographics_df['shoulder_to_wrist_cm'] < lower_bound) |
    (raw_train_demographics_df['shoulder_to_wrist_cm'] > upper_bound)
]

outliers


# Calculate IQR for 'elbow_to_wrist_cm'

q1 = raw_train_demographics_df['elbow_to_wrist_cm'].quantile(0.25)
q3 = raw_train_demographics_df['elbow_to_wrist_cm'].quantile(0.75)
iqr = q3 - q1

# Define outlier bounds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Find outlier rows
outliers = raw_train_demographics_df[
    (raw_train_demographics_df['elbow_to_wrist_cm'] < lower_bound) |
    (raw_train_demographics_df['elbow_to_wrist_cm'] > upper_bound)
]

outliers


filepath = my_utils.Path('/kaggle/usr/lib/cmi_comp_utils_functions/cmi-competition-data/raw-data/test_demographics.csv')
raw_test_demographics_df = pd.read_csv(filepath, sep=',', encoding='utf-8', na_filter=True)


raw_test_demographics_df.shape


raw_test_demographics_df.head(10)

