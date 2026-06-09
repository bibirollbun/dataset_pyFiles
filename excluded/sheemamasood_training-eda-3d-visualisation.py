import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load train sequences
train_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")

# Load train labels (3D coordinates)
train_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")

# Load validation sequences
validation_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv")

# Load validation labels (3D coordinates)
validation_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_labels.csv")

# Load test sequences
test_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")

# Display basic info
print("Train Sequences Sample:")
print(train_sequences.head())

print("\nTrain Labels Sample:")
print(train_labels.head())

print("\nValidation Sequences Sample:")
print(validation_sequences.head())




import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# 1️⃣ Load and Preprocess Data
# -----------------------------

# Load train_labels.csv
train_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")  # Replace with actual path

# Extract `target_id` from `ID` (Remove residue numbering)
train_labels["target_id"] = train_labels["ID"].apply(lambda x: "_".join(x.split("_")[:2]))

# Sort data by `target_id` and `resid` for proper ordering
train_labels = train_labels.sort_values(by=["target_id", "resid"])

# -----------------------------
# 2️⃣ Select 3 Sample RNA Sequences for Visualization
# -----------------------------

# Select first 3 unique target_ids
sample_rnas = train_labels["target_id"].unique()[:9]

# Color mapping for nucleotides (A, C, G, U)
nucleotide_colors = {"A": "red", "C": "blue", "G": "green", "U": "purple"}

# -----------------------------
# 3️⃣ 3D Plotting Function
# -----------------------------

def plot_rna_structure_with_edges(ax, rna_data, target_id):
    """Plots the 3D structure of an RNA sequence with edges between consecutive nucleotides."""
    for resname, color in nucleotide_colors.items():
        subset = rna_data[rna_data["resname"] == resname]
        ax.scatter(subset["x_1"], subset["y_1"], subset["z_1"], c=color, label=resname, s=30)
        
    # Draw edges between consecutive nucleotides
    for i in range(len(rna_data) - 1):
        x_vals = [rna_data.iloc[i]["x_1"], rna_data.iloc[i+1]["x_1"]]
        y_vals = [rna_data.iloc[i]["y_1"], rna_data.iloc[i+1]["y_1"]]
        z_vals = [rna_data.iloc[i]["z_1"], rna_data.iloc[i+1]["z_1"]]
        ax.plot(x_vals, y_vals, z_vals, color="gray", linewidth=0.5)  # Edge line between consecutive residues

    ax.set_xlabel("X Coordinate (Å)")
    ax.set_ylabel("Y Coordinate (Å)")
    ax.set_zlabel("Z Coordinate (Å)")
    ax.set_title(f"3D Structure of RNA: {target_id}")
    ax.legend()

# -----------------------------
# 4️⃣ Plot 3D Structures with Edges for Selected RNAs
# -----------------------------


# Define the number of rows and columns for the grid
num_rows, num_cols = 5, 5

# Select the first 9 unique target_ids for visualization
sample_rnas = train_labels["target_id"].unique()[: num_rows * num_cols]

# Create a figure for 9 subplots (3x3 layout)
fig = plt.figure(figsize=(15, 15))

for i, target_id in enumerate(sample_rnas):
    ax = fig.add_subplot(num_rows, num_cols, i + 1, projection="3d")
    
    # Filter data for the selected RNA sequence
    rna_data = train_labels[train_labels["target_id"] == target_id]
    
    # Plot the RNA structure with edges
    plot_rna_structure_with_edges(ax, rna_data, target_id)

plt.tight_layout()
plt.show()



train_labels


print("\nValidation Labels Sample:")
print(validation_labels.head())

print("\nTest Sequences Sample:")
print(test_sequences.head())




# Visualizing RNA sequence lengths distribution in training data
train_sequences["sequence_length"] = train_sequences["sequence"].apply(len)

plt.figure(figsize=(10, 5))
sns.histplot(train_sequences["sequence_length"], bins=30, kde=True)
plt.xlabel("Sequence Length")
plt.ylabel("Frequency")
plt.title("Distribution of RNA Sequence Lengths in Training Set")
plt.show()




# Visualizing the 3D structure of a sample RNA (plotting C1' coordinates)
sample_rna = train_labels[train_labels["ID"].str.contains("1RHT_A")]

print(sample_rna.head())




fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sample_rna["x_1"], sample_rna["y_1"], sample_rna["z_1"], c='b', marker='o')

ax.set_xlabel("X Coordinate (Å)")
ax.set_ylabel("Y Coordinate (Å)")
ax.set_zlabel("Z Coordinate (Å)")
ax.set_title("3D Structure of Sample RNA (C1' Atom Coordinates)")

plt.show()



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.impute import KNNImputer
from collections import Counter

# -------------------------
# 1️⃣ Sequence & Length Analysis
# -------------------------

# Compute sequence lengths
train_sequences["sequence_length"] = train_sequences["sequence"].apply(len)

# Plot sequence length distribution
plt.figure(figsize=(10, 5))
sns.histplot(train_sequences["sequence_length"], bins=30, kde=True)
plt.xlabel("Sequence Length")
plt.ylabel("Frequency")
plt.title("Distribution of RNA Sequence Lengths in Training Set")
plt.show()

# GC-content calculation (proportion of G and C nucleotides)
def gc_content(seq):
    return (seq.count("G") + seq.count("C")) / len(seq)

train_sequences["GC_content"] = train_sequences["sequence"].apply(gc_content)

# Scatter plot of sequence length vs GC content
plt.figure(figsize=(10, 5))
sns.scatterplot(x=train_sequences["sequence_length"], y=train_sequences["GC_content"])
plt.xlabel("Sequence Length")
plt.ylabel("GC Content")
plt.title("GC Content vs Sequence Length")
plt.show()

# Nucleotide frequency analysis
nucleotide_counts = Counter("".join(train_sequences["sequence"]))
plt.figure(figsize=(8, 5))
sns.barplot(x=list(nucleotide_counts.keys()), y=list(nucleotide_counts.values()))
plt.xlabel("Nucleotide")
plt.ylabel("Frequency")
plt.title("Nucleotide Frequency in RNA Sequences")
plt.show()






import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load train_sequences again (after reset)
train_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")

# Compute sequence lengths
train_sequences["sequence_length"] = train_sequences["sequence"].apply(len)

# Separate outliers (length > 200)
outliers = train_sequences[train_sequences["sequence_length"] > 200]
filtered_sequences = train_sequences[train_sequences["sequence_length"] <= 200]

# Plot sequence length distribution without outliers
plt.figure(figsize=(10, 5))
sns.histplot(filtered_sequences["sequence_length"], bins=30, kde=True)
plt.xlabel("Sequence Length")
plt.ylabel("Frequency")
plt.title("Distribution of RNA Sequence Lengths (<= 1000 residues)")
plt.show()

# Plot outliers separately
plt.figure(figsize=(10, 4))
sns.histplot(outliers["sequence_length"], bins=10, color='red')
plt.xlabel("Sequence Length")
plt.ylabel("Frequency")
plt.title("Outliers: RNA Sequences with Length > 1000")
plt.show()

# Display the outlier data
outliers.reset_index(drop=True, inplace=True)
outliers.head(10)



# -------------------------
# 3️⃣ Missing Data Handling (Imputation)
# -------------------------

# Checking missing values
missing_values = train_labels.isnull().sum()
print("Missing Values:\n", missing_values)

# Using KNN Imputer to fill missing 3D coordinates
from sklearn.impute import SimpleImputer

# Creating a mean imputer
mean_imputer = SimpleImputer(strategy="mean")

# Apply mean imputation for missing values
train_labels[["x_1", "y_1", "z_1"]] = mean_imputer.fit_transform(train_labels[["x_1", "y_1", "z_1"]])

# Check if missing values are handled
print("Missing Values After Imputation:\n", train_labels.isnull().sum())



# -------------------------
# 4️⃣ Multi-Conformation Handling (Grouping RNA Structures)
# -------------------------

# Count number of conformations per RNA sequence
conformation_counts = train_labels["ID"].value_counts()

plt.figure(figsize=(10, 5))
sns.histplot(conformation_counts, bins=30, kde=True)
plt.xlabel("Number of Conformations")
plt.ylabel("Frequency")
plt.title("Distribution of Multiple Conformations Per RNA Sequence")
plt.show()

# Print some examples of RNA sequences with multiple conformations
multi_conformations = conformation_counts[conformation_counts > 1].index[:5]
print("Examples of RNA sequences with multiple conformations:", multi_conformations.tolist())



import tensorflow as tf
import numpy as np
import pandas as pd

# -----------------------------
# Load and Process Data
# -----------------------------

train_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")
train_labels["target_id"] = train_labels["ID"].apply(lambda x: "_".join(x.split("_")[:2]))
train_labels = train_labels.sort_values(by=["target_id", "resid"])

MAX_SEQ_LEN = 20
NUCLEOTIDE_MAP = {"A": [1, 0, 0, 0], "C": [0, 1, 0, 0], "G": [0, 0, 1, 0], "U": [0, 0, 0, 1]}


# Updated version of the create_sequence_chunks function with sliding window support
def create_sequence_chunks_with_sliding(data, max_seq_len=20, stride=5):
    """
    Split RNA sequences into chunks of `max_seq_len` residues using sliding windows.
    
    Parameters:
    - data: DataFrame with RNA structural info.
    - max_seq_len: Number of residues per chunk.
    - stride: Number of residues to move the window each time.
    
    Returns:
    - A list of chunks, each containing a (max_seq_len, 5) matrix [resname, resid, x, y, z].
    """
    sequences = []
    grouped = data.groupby("target_id")

    for target_id, group in grouped:
        coords = group[["resname", "resid", "x_1", "y_1", "z_1"]].values

        for i in range(0, len(coords), stride):
            chunk = coords[i: i + max_seq_len]
            if len(chunk) < max_seq_len:
                padding = np.array([["N", -1, np.nan, np.nan, np.nan]] * (max_seq_len - len(chunk)))
                chunk = np.vstack((chunk, padding))
            sequences.append(chunk)

    return sequences




class RNATensorflowDataset(tf.keras.utils.Sequence):
    def __init__(self, sequences, batch_size=32):
        self.sequences = sequences
        self.batch_size = batch_size

    def __len__(self):
        return int(90)


    def __getitem__(self, idx):
        batch_chunks = self.sequences[idx * self.batch_size:(idx + 1) * self.batch_size]

        node_features_batch = []
        positions_batch = []
        mask_indices = []
        targets = []

        for chunk in batch_chunks:
            node_features = []
            positions = []

            # Choose a valid index to mask
            valid_indices = [i for i, row in enumerate(chunk) if row[0] in NUCLEOTIDE_MAP]
            masked_idx = np.random.choice(valid_indices) if valid_indices else 0
            target = chunk[masked_idx, 2:5].astype(np.float32)

            for i, row in enumerate(chunk):
                nucleotide = row[0]
                pos_index = i / MAX_SEQ_LEN
                is_masked = 1.0 if i == masked_idx else 0.0
                coords = np.array([0.0, 0.0, 0.0], dtype=np.float32) if i == masked_idx else row[2:5].astype(np.float32)
            
                one_hot = NUCLEOTIDE_MAP.get(nucleotide, [0, 0, 0, 0])
                features = one_hot + [pos_index, is_masked] + coords.tolist()
                node_features.append(features)
                positions.append(coords)


            node_features_batch.append(node_features)
            positions_batch.append(positions)
            mask_indices.append(masked_idx)
            targets.append(target)

        return (
            tf.convert_to_tensor(node_features_batch, dtype=tf.float32),  # shape (B, 20, 9)
            tf.convert_to_tensor(positions_batch, dtype=tf.float32),      # shape (B, 20, 3)
            tf.convert_to_tensor(mask_indices, dtype=tf.int32),           # shape (B,)
            tf.convert_to_tensor(targets, dtype=tf.float32),              # shape (B, 3)
        )


# Create sequence chunks and dataset
# Create sliding window sequences using stride=5
sliding_sequence_chunks = create_sequence_chunks_with_sliding(train_labels, max_seq_len=20, stride=5)

# Check how many samples are now created
num_samples = len(sliding_sequence_chunks)
print(f"Total training samples with sliding window: {num_samples}")
sliding_sequence_chunks[:1]  # Show the first chunk for inspection





def create_sequence_chunks(data):
    """Split RNA sequences into chunks of MAX_SEQ_LEN residues."""
    sequences = []
    grouped = data.groupby("target_id")

    for target_id, group in grouped:
        coords = group[["resname", "resid", "x_1", "y_1", "z_1"]].values

        for i in range(0, len(coords), MAX_SEQ_LEN):
            chunk = coords[i: i + MAX_SEQ_LEN]
            if len(chunk) < MAX_SEQ_LEN:
                padding = np.array([["N", -1, np.nan, np.nan, np.nan]] * (MAX_SEQ_LEN - len(chunk)))
                chunk = np.vstack((chunk, padding))
            sequences.append(chunk)

    return sequences


# Create sequence chunks and dataset
sequence_chunks = create_sequence_chunks(train_labels)
rna_tf_dataset = RNATensorflowDataset(sequence_chunks, batch_size=128)

# Preview a batch
sample_batch = rna_tf_dataset[0]


# Convert tensors to numpy
node_features_np = sample_batch[0].numpy()
positions_np = sample_batch[1].numpy()
masked_indices_np = sample_batch[2].numpy()
targets_np = sample_batch[3].numpy()

# Flatten for display
import pandas as pd

flat_batch = {
    "Masked Index": masked_indices_np.tolist(),
    "Target Coordinates": targets_np.tolist()
}

# Show as DataFrame
pd.DataFrame(flat_batch)



import tensorflow as tf

class EGNNLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim):
        super(EGNNLayer, self).__init__()
        self.hidden_dim = hidden_dim

        # MLP to compute messages based on feature differences and squared distance
        self.message_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(hidden_dim)
        ])

        # Coordinate update MLP
        self.coord_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(1, activation='relu'),  # One scalar to scale directional vector
        ])

        # Node feature update MLP
        self.feature_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(hidden_dim)
        ])

    def call(self, node_features, positions, edge_index=None):
        """
        Arguments:
        - node_features: (B, N, F)
        - positions: (B, N, 3)
        Returns:
        - updated node_features: (B, N, F)
        - updated positions: (B, N, 3)
        """
        B, N, F = node_features.shape

        # Step 1: Create all pairwise differences (broadcasted)
        pos_i = tf.expand_dims(positions, 2)  # (B, N, 1, 3)
        pos_j = tf.expand_dims(positions, 1)  # (B, 1, N, 3)
        diff = pos_i - pos_j  # (B, N, N, 3)
        dist2 = tf.reduce_sum(tf.square(diff), axis=-1, keepdims=True)  # (B, N, N, 1)

        # Step 2: Message computation
        h_i = tf.expand_dims(node_features, 2)  # (B, N, 1, F)
        h_j = tf.expand_dims(node_features, 1)  # (B, 1, N, F)
        message_input = tf.concat([h_i - h_j, dist2], axis=-1)  # (B, N, N, F+1)

        messages = self.message_mlp(message_input)  # (B, N, N, F)

        # Step 3: Aggregate messages
        agg_messages = tf.reduce_sum(messages, axis=2)  # (B, N, F)

        # Step 4: Update node features
        updated_features = self.feature_mlp(agg_messages)  # (B, N, F)

        # Step 5: Update coordinates
        coord_weights = self.coord_mlp(messages)  # (B, N, N, 1)
        coord_update = tf.reduce_sum(coord_weights * diff, axis=2)  # (B, N, 3)
        updated_positions = positions + coord_update  # (B, N, 3)

        return updated_features, updated_positions

import tensorflow as tf

class RNAGNN(tf.keras.Model):
    def __init__(self, hidden_dim=64, num_layers=3):
        super(RNAGNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Initial embedding layer for node features (input_dim = 9 → hidden_dim)
        self.embed = tf.keras.layers.Dense(hidden_dim)

        # Stack multiple EGNN layers
        self.egnn_layers = [EGNNLayer(hidden_dim) for _ in range(num_layers)]

        # MLP head to predict (x, y, z)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(3)  # output 3D coordinates
        ])

    def call(self, node_features, positions, masked_idx):
        """
        Arguments:
        - node_features: (B, N, 9) including one-hot, pos index, mask flag, coords
        - positions: (B, N, 3)
        - masked_idx: (B,) index of masked residue per sequence

        Returns:
        - predicted_coords: (B, 3) predicted (x, y, z) for each masked residue
        """
        x = self.embed(node_features)  # shape (B, N, hidden_dim)

        for egnn in self.egnn_layers:
            x, positions = egnn(x, positions)

        # Gather masked embeddings for each sample in the batch
        # masked_idx: (B,) → convert to (B, 1) → gather along dim=1
        masked_repr = tf.gather(x, tf.expand_dims(masked_idx, axis=1), batch_dims=1)
        masked_repr = tf.squeeze(masked_repr, axis=1)  # shape: (B, hidden_dim)

        # Predict (x, y, z)
        predicted_coords = self.mlp(masked_repr)  # shape: (B, 3)

        return predicted_coords



# Define and create the RNA GNN model object
hidden_dim = 64
num_layers = 10

# Instantiate the model
rna_model = RNAGNN(hidden_dim=hidden_dim, num_layers=num_layers)

# Dummy input to build the model and show summary
B, N, F = 4, 20, 9  # batch size, number of nodes, feature dimension
dummy_node_features = tf.random.normal((B, N, F))
dummy_positions = tf.random.normal((B, N, 3))
dummy_masked_idx = tf.constant([3, 7, 12, 5], dtype=tf.int32)  # one masked node per sequence

# Call the model once to build it
_ = rna_model(dummy_node_features, dummy_positions, dummy_masked_idx)

# Show the model summary
rna_model.summary()



from sklearn.model_selection import train_test_split

# Shuffle and split the data
train_chunks, val_chunks = train_test_split(sliding_sequence_chunks, test_size=0.1, random_state=42)

# Re-initialize data loaders
train_dataset = RNATensorflowDataset(train_chunks, batch_size=256)
val_dataset = RNATensorflowDataset(val_chunks, batch_size=256)



print(len(val_dataset))  # Number of chunks (i.e., batches)



from tqdm import tqdm
import tensorflow as tf
import pandas as pd
import numpy as np

# Re-initialize model
rna_model = RNAGNN(hidden_dim=64, num_layers=30)

loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

epochs = 50
train_logs = []

for epoch in range(epochs):
    total_loss = 0.0
    total_l2_error = 0.0
    steps = 0

    print(f"\nEpoch {epoch + 1}/{epochs}")
    

    for batch in tqdm(train_dataset, desc=f"Training Epoch {epoch+1}", total=len(train_dataset), leave=False):

        node_features, positions, masked_idx, targets = batch
        if node_features.shape[0] == 0:
            continue

        with tf.GradientTape() as tape:
            predictions = rna_model(node_features, positions, masked_idx)
            loss = loss_fn(targets, predictions)

        l2_error = tf.reduce_mean(tf.norm(targets - predictions, axis=1))
        grads = tape.gradient(loss, rna_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, rna_model.trainable_variables))

        total_loss += loss.numpy()
        total_l2_error += l2_error.numpy()
        steps += 1

    epoch_loss = total_loss / steps
    epoch_l2_error = total_l2_error / steps

    # Validation phase
    val_loss = 0.0
    val_l2_error = 0.0
    val_steps = 0
    print("training completed")
    for val_batch in tqdm(val_dataset,desc=f"Validation Epoch {epoch+1}", leave=False):
        print("validation loop")
        node_features, positions, masked_idx, targets = val_batch
        # if node_features.shape[0] == 0:
        #     continue

        val_preds = rna_model(node_features, positions, masked_idx)
        val_batch_loss = loss_fn(targets, val_preds)
        val_l2 = tf.reduce_mean(tf.norm(targets - val_preds, axis=1))

        val_loss += val_batch_loss.numpy()
        val_l2_error += val_l2.numpy()
        val_steps += 1

    avg_val_loss = val_loss / val_steps
    avg_val_l2 = val_l2_error / val_steps

    train_logs.append({
        "epoch": epoch + 1,
        "train_loss": epoch_loss,
        "train_l2_error": epoch_l2_error,
        "val_loss": avg_val_loss,
        "val_l2_error": avg_val_l2
    })

    print(f"Epoch {epoch+1}/{epochs} - "
          f"Train Loss: {epoch_loss:.4f}, L2: {epoch_l2_error:.4f} | "
          f"Val Loss: {avg_val_loss:.4f}, L2: {avg_val_l2:.4f}")





