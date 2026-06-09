import tensorflow as tf
import numpy as np
import pandas as pd

#visual
import matplotlib.pyplot as plt
import seaborn as sns


path = '../input/stanford-rna-3d-folding/'

df_train_seq = pd.read_csv(path +'train_sequences.csv')
df_train_labels = pd.read_csv(path + 'train_labels.csv')

df_valid_seq = pd.read_csv(path + 'validation_sequences.csv')
df_validation_labels = pd.read_csv(path + 'validation_labels.csv')


shapes = {
    'train_sequences': df_train_seq.shape,
    'train_labels': df_train_labels.shape,
    'validation_sequences': df_valid_seq.shape,
    'validation_labels': df_validation_labels.shape
}

# Print the sizes for each DataFrame
for key, value in shapes.items():
    print(f"{key}: {value}")


df_train_seq.head()


df_train_seq.columns


df_train_labels.head()


df_train_labels.columns


df_valid_seq.head()


df_valid_seq.columns


df_validation_labels.head()


df_validation_labels.columns


df_train_labels.describe()


df_train_labels.resname.unique()


df_train_labels.resname.value_counts()


df_train_labels.isnull().values.any()


df_train_labels.isnull()


coord_cols = ['x_1', 'y_1', 'z_1']
for col in coord_cols:
    df_train_labels[col] = df_train_labels[col].fillna(df_train_labels[col].mean())


df_validation_labels.isnull().values.any()


# Compute and plot the distribution of sequence lengths
df_train_seq['seq_length'] = df_train_seq['sequence'].apply(len)
plt.figure(figsize=(5, 3))
sns.histplot(df_train_seq['seq_length'], bins=20, kde=True)
plt.title("Distribution of RNA Sequence Lengths (Train)")
plt.xlabel("Sequence Length")
plt.ylabel("Frequency")
plt.show()


x_coords = df_train_labels['x_1']
y_coords = df_train_labels['y_1']
z_coords = df_train_labels['z_1']

# Create a 3D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
residue_ids = df_train_labels['resid']
        
# Create a normalized color mapping based on residue IDs
normalized_ids = (residue_ids - residue_ids.min()) / (residue_ids.max() - residue_ids.min())
colors = plt.cm.viridis(normalized_ids)
ax.scatter(x_coords, y_coords, z_coords, marker='o', c=colors, s=20)

# Labeling the axes and giving a title
ax.set_xlabel("X coordinate")
ax.set_ylabel("Y coordinate")
ax.set_zlabel("Z coordinate")
ax.set_title("3D Scatter Plot of Training Labels Coordinates")

plt.show()




