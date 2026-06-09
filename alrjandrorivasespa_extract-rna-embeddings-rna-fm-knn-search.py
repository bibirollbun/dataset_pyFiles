!pip install multimolecule -q


import pandas as pd
import numpy as np 
import torch
from tqdm import tqdm
import plotly.graph_objects as go



# Extraction of base Dataframes
train_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')
test_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_labels.csv')
train_df = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')
test_df = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')

# Extraction of RNA sequences from DataFrames
train_rna, test_rna = train_df['sequence'].to_list(), test_df['sequence'].to_list()

# Create a unique identifier by extracting the base ID from the full ID
# (removes the last part after the last underscore)
train_labels['unique_id'] = train_labels['ID'].str.rsplit('_', n=1).str[0]
test_labels['unique_id'] = test_labels['ID'].str.rsplit('_', n=1).str[0]


from multimolecule import RnaTokenizer, RnaFmModel

# Define constants
MAX_SEQ_LEN = 720
embed_dim = 640  # Size of the RNA-FM output tensor
batch_size = 32

# Make sure device is defined
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnafm")
model = RnaFmModel.from_pretrained("multimolecule/rnafm").to(device)



def extract_embeddings(rna_seq, batch_size=32, MAX_SEQ_LEN=720):
    """
    Extracts embeddings from RNA sequences using a pre-trained language model.
    
    Parameters:
    - rna_seq: List of RNA sequences (strings of nucleotides)
    - batch_size: Number of sequences to process at once (default: 32)
    - MAX_SEQ_LEN: Maximum sequence length for tokenization (default: 720)
    
    Returns:
    - last_hidden_states_stack: NumPy array of shape (n_sequences, sequence_length, hidden_size)
      containing contextual embeddings for each token in each sequence
    - pooler_outputs_stack: NumPy array of shape (n_sequences, hidden_size)
      containing aggregated sequence-level embeddings
    """
    # To process multiple sequences from a DataFrame
    last_hidden_states = []
    pooler_outputs = []
    
    # Using batch processing for faster inference
    for idx in tqdm(range(0, len(rna_seq), batch_size)):
    
        seq = rna_seq[idx:idx+batch_size]
        
        # Tokenize and get model outputs
        inputs = tokenizer(
            seq, 
            return_tensors="pt",
            padding="max_length",  # Add padding
            truncation=True,       # Enable truncation
            max_length=MAX_SEQ_LEN # Set the maximum length that the model expects
        ).to(device)    
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract and store tensors as NumPy arrays
        last_hidden_states.extend(outputs.last_hidden_state.detach().cpu().numpy())
        pooler_outputs.extend(outputs.pooler_output.detach().cpu().numpy())
    
    # Convert lists to NumPy arrays
    last_hidden_states_stack = np.array(last_hidden_states)
    pooler_outputs_stack = np.array(pooler_outputs)
    
    return last_hidden_states_stack, pooler_outputs_stack
    

# Use of the function for train and test sequences
hidden_states_train, pooler_outputs_train = extract_embeddings(train_rna, batch_size=16)
hidden_states_test, pooler_outputs_test = extract_embeddings(test_rna, batch_size=16)


from sklearn.neighbors import NearestNeighbors

def k_similars_embeddings(base_embedding, embeddings_list, k=5):
    """
    Find k most similar embeddings to the base embedding using cosine similarity.
    
    Args:
        base_embedding: The reference embedding to compare against
        embeddings_list: List of embeddings to search through
        k: Number of similar embeddings to return (default: 5)
    
    Returns:
        indices: Indices of the k most similar embeddings
        distances: Corresponding similarity distances
    """

    # Reshape base_embedding if it's only one sample
    if base_embedding.shape[0] == 640:
        base_embedding = base_embedding.reshape(1, -1)

    
    # Cosine similarity is prefered for embeddings as it measures angular distance
    knn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='cosine')
    knn.fit(embeddings_list)
    
    # Find the k most similar embeddings and their distances
    distances, indices = knn.kneighbors(base_embedding)
    return indices, distances


# Finding 5 most similar proteins found in train data for the test sequences
test_indices, test_distances = k_similars_embeddings(pooler_outputs_test, pooler_outputs_train, k=10)
train_indices, train_distances = k_similars_embeddings(pooler_outputs_train, pooler_outputs_train, k=10)



import plotly.graph_objects as go


def plot_structure(similar_idx, distances, n_similars=3, max_sizes=720) -> None:
    """
    Plot 3D structures of RNA molecules with similar sequences.
    
    Args:
        similar_idx: Indices of similar RNA sequences
        distances: Distance metrics between sequences
        n_similars: Number of similar structures to display (default: 3)
        max_sizes: Maximum size parameter (default: 720)
    """
    # Limit the number of structures to display
    similar_idx, distances = similar_idx[:n_similars], distances[:n_similars]
    
    # Get base IDs, sequences and labels from the training data
    base_ids = train_df.iloc[similar_idx]['target_id'].values
    sequences = train_df.iloc[similar_idx]['sequence'].values
    similar_df_labels = train_labels[train_labels['unique_id'].isin(base_ids)]
    
    # Create a dictionary mapping sequences to their 3D coordinates
    coordinates = {}
    for curr_id, data in similar_df_labels.groupby('unique_id'):
        coordinates[train_df[train_df['target_id'] == curr_id]['sequence'].values[0]] = data[['x_1', 'y_1', 'z_1']].values
    
    # Define colors for nucleotides
    nucleotide_colors = {"A": "red", "G": "blue", "C": "green", "U": "orange"}
    
    # Define colors for each sequence backbone
    backbone_colors = ["rgba(255,0,0,0.7)", "rgba(0,0,255,0.7)", "rgba(0,255,0,0.7)", 
                      "rgba(255,165,0,0.7)", "rgba(128,0,128,0.7)", "rgba(0,128,128,0.7)"]
    
    fig = go.Figure()
    
    # Preprocess coordinates to center them
    processed_coordinates = {}
    centroids = []
    
    # First, center each structure on its own centroid
    for i, sequence in enumerate(sequences):
        if sequence in coordinates:
            x, y, z = coordinates[sequence][:, 0], coordinates[sequence][:, 1], coordinates[sequence][:, 2]
            
            # Check that lists have the same length
            if not (len(x) == len(y) == len(z) == len(sequence)):
                print(f"Warning: Lists for sequence {i+1} don't have the same length. Adjusting...")
                min_len = min(len(x), len(y), len(z), len(sequence))
                x, y, z = x[:min_len], y[:min_len], z[:min_len]
                sequence = sequence[:min_len]
            
            # Calculate the centroid of this structure
            centroid = np.array([np.mean(x), np.mean(y), np.mean(z)])
            centroids.append(centroid)
            
            # Center the coordinates
            centered_coords = np.column_stack((x, y, z)) - centroid
            processed_coordinates[sequence] = centered_coords
    
    # Calculate small displacements for each structure
    max_radius = max([np.max(np.sqrt(np.sum(coords**2, axis=1))) for coords in processed_coordinates.values()])
    spacing = max_radius * 0.5  # Space between structures
    
    # Iterate over all available sequences
    for i, sequence in enumerate(sequences):
        if sequence in coordinates:
            sizes = len(sequence)
            
            # Get centered coordinates
            centered_coords = processed_coordinates[sequence]
            
            # Apply a small radial displacement to visualize structures together
            # but not completely superimposed
            angle = 2 * np.pi * i / len(sequences)  # Distribute in circle
            offset = np.array([spacing * np.cos(angle), spacing * np.sin(angle), 0])
            
            x = centered_coords[:, 0] + offset[0]
            y = centered_coords[:, 1] + offset[1]
            z = centered_coords[:, 2] + offset[2]
            
            # Select color for the backbone (rotating if there are more sequences than colors)
            backbone_color = backbone_colors[i % len(backbone_colors)]
            
            # Add points by nucleotide type for this sequence
            for resname, color in nucleotide_colors.items():
                indices = [j for j, res in enumerate(sequence) if res == resname]
                if indices:
                    fig.add_trace(go.Scatter3d(
                        x=[x[j] for j in indices],
                        y=[y[j] for j in indices],
                        z=[z[j] for j in indices],
                        mode='markers',
                        marker=dict(size=4, color=color),
                        name=f'{resname} (Seq {i+1})',
                        # Only show in legend for the first sequence to avoid duplicates
                        showlegend=(i == 0)
                    ))
                    
            # Add line for the RNA backbone
            fig.add_trace(go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='lines',
                line=dict(color=backbone_color, width=8),
                name=f'RNA Backbone (Seq {i+1} Similarity:{distances[i]*1000:.4f})'
            ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title=f'Comparison of RNA 3D structures ({len(sequences)} sequences)',
        legend=dict(
            itemsizing='constant',
            itemwidth=30
        ),
    )
    
    # Display the figure
    fig.show()


# plot sequences from most similar embeddings for train sample 0 
plot_structure(train_indices[0], train_distances[0], n_similars=3)


# plot sequences from most similar embeddings for test sample 8
plot_structure(test_indices[8], test_distances[8], n_similars=2)






















