import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


Dir = "/kaggle/input/cmi-detect-behavior-with-sensor-data"


df_Demographic  = pd.read_csv(f"{Dir}/train_demographics.csv") # "dmgc" means demographics
df_Demographic.head()


df_Demographic.shape


df_Demographic.info()


df_Demographic.describe()


df_Demographic['adult_child'] = df_Demographic['adult_child'].astype('object')
df_Demographic['sex'] = df_Demographic['sex'].astype('object')
df_Demographic['handedness'] = df_Demographic['handedness'].astype('object')


df_Demographic.info()


df_Demographic.describe(include='all')


# Reverting demographic columns to integer type for further analysis.

df_Demographic['adult_child'] = df_Demographic['adult_child'].astype('int64')
df_Demographic['sex'] = df_Demographic['sex'].astype('int64')
df_Demographic['handedness'] = df_Demographic['handedness'].astype('int64')


# Checking
df_Demographic[['adult_child', 'sex', 'handedness']].dtypes


# Choose Only numeric columns
numeric_columns = df_Demographic.select_dtypes(include='number').columns

# Set number of rows and columns for subplots
num_cols = 3
num_rows = (len(numeric_columns) + num_cols) - 1 // num_cols

# subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize = (15, 4 * num_rows))
axes = axes.flatten()

# Plot for each numeric columns
for i, col in enumerate(numeric_columns):
    axes[i].hist(df_Demographic[col], bins=20, color='lightblue', edgecolor='black')
    axes[i].set_title(f'Histogram of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Absolute Frequency')

# Remove unused subplots if any
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# Correlation Matrix
cor_mat = df_Demographic.corr(numeric_only=True)

plt.figure(figsize=(10, 8))

sns.heatmap(cor_mat, annot=True, cmap='coolwarm', fmt='.2f', square=True)

plt.title('Correlation Matrix')
plt.show()


plt.scatter(df_Demographic.age, df_Demographic.height_cm)


plt.scatter(df_Demographic.shoulder_to_wrist_cm, df_Demographic.height_cm)


plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    df_Demographic['age'],
    df_Demographic['height_cm'],
    c = df_Demographic['sex'],
    cmap = 'coolwarm',
    edgecolor = 'k',
    s = 100,
    alpha = 0.8
)
plt.xlabel('Age (year)', fontsize = 14)
plt.ylabel('Height (centimetre)', fontsize = 14)

plt.title('Age vs Height Colored by Sex', fontsize = 16)

cbar = plt.colorbar(scatter)
cbar.set_label('Sex (Blue=Female, Red=Male)', fontsize =13)

plt.tight_layout()
plt.show()




