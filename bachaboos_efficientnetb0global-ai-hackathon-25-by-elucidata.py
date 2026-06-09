import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from spektral.layers import GATConv
from spektral.data import Graph, BatchLoader
import tensorflow_hub as hub

# Step 1: Load the .h5 file
file_path = '/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5'

with h5py.File(file_path, 'r') as f:
    # Corrected paths based on the actual structure
    valid_keys = [key for key in ['S_1', 'S_2', 'S_3', 'S_4', 'S_5', 'S_6'] if f'spots/Train/{key}' in f]
    
    # Train Spots
    train_spots = {key: pd.DataFrame(f[f'spots/Train/{key}'][:]) for key in valid_keys}
    
    # Train Images
    train_images = {key: np.array(f[f'images/Train/{key}']) for key in valid_keys}
    
    # Test Spots
    test_spots = pd.DataFrame(f['spots/Test/S_7'][:])
    
    # Test Image
    test_image = np.array(f['images/Test/S_7'])

# Step 2: Normalize Cell Type Abundances
scaler_y = MinMaxScaler()
for key in train_spots:
    train_spots[key].iloc[:, 2:] = scaler_y.fit_transform(train_spots[key].iloc[:, 2:])

# Step 3: Extract Image Features Using Vision Transformer (ViT) or EfficientNetB0
try:
    # Try loading the Vision Transformer (ViT)
    vit_model = hub.KerasLayer("https://tfhub.dev/sayakpaul/vit_base_patch16_224_fe/1", trainable=False)

    def extract_image_features(image, spots):
        features = []
        for _, row in spots.iterrows():
            x, y = int(row['x']), int(row['y'])
            patch = image[y-32:y+32, x-32:x+32]  # Example: 64x64 patch around the spot
            patch = tf.image.resize(patch, (224, 224))  # Resize to match ViT input size
            embedding = vit_model(tf.expand_dims(patch, axis=0))  # Extract ViT embedding
            features.append(embedding.numpy().flatten())
        return np.vstack(features)

except Exception as e:
    print(f"Failed to load ViT model: {e}")
    print("Falling back to EfficientNetB0...")
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

    base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

    def extract_image_features(image, spots):
        features = []
        for _, row in spots.iterrows():
            x, y = int(row['x']), int(row['y'])
            patch = image[y-32:y+32, x-32:x+32]  # Example: 64x64 patch around the spot
            patch = tf.image.resize(patch, (224, 224))  # Resize to match EfficientNet input size
            patch = efficientnet_preprocess(patch)  # Preprocess for EfficientNet
            embedding = base_model.predict(tf.expand_dims(patch, axis=0))  # Extract features
            features.append(embedding.flatten())
        return np.vstack(features)

# Extract features for all training and test images
train_image_features = {key: extract_image_features(img, train_spots[key]) for key, img in train_images.items()}
test_image_features = extract_image_features(test_image, test_spots)

# Step 4: Build Graph with Edge Features
def build_graph(spots, features):
    num_nodes = len(spots)
    node_features = features
    
    # Create adjacency matrix and edge features
    edge_index, edge_features = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dist = np.sqrt((spots.iloc[i]['x'] - spots.iloc[j]['x'])**2 + 
                               (spots.iloc[i]['y'] - spots.iloc[j]['y'])**2)
                if dist < 50:  # Threshold for connecting nodes
                    edge_index.append([i, j])
                    edge_features.append(dist)
    edge_index = np.array(edge_index).T  # Shape: (2, num_edges)
    edge_features = np.array(edge_features)
    
    # Create Graph object
    graph = Graph(x=node_features, a=edge_index, e=edge_features, y=spots.iloc[:, 2:].values)
    return graph

# Build graphs for training and test data
train_graphs = [build_graph(train_spots[key], train_image_features[key]) for key in train_spots]
test_graph = build_graph(test_spots, test_image_features)

# Step 5: Prepare Training and Validation Data
# Split the list of graphs into training and validation sets
train_graphs_split, val_graphs_split = train_test_split(train_graphs, test_size=0.2, random_state=42)

# Combine all training graphs into a single dataset
X_train, y_train = [], []
for graph in train_graphs_split:
    X_train.append(graph.x)  # Node features
    y_train.append(graph.y)  # Target values (cell type abundances)

X_train = np.vstack(X_train)  # Stack all node features
y_train = np.vstack(y_train)  # Stack all target values

# Combine all validation graphs into a single dataset
X_val, y_val = [], []
for graph in val_graphs_split:
    X_val.append(graph.x)
    y_val.append(graph.y)

X_val = np.vstack(X_val)
y_val = np.vstack(y_val)

# Step 6: Define GAT Model with Multi-Task Learning
class GATModel(tf.keras.Model):
    def __init__(self, hidden_dim, output_dim):
        super(GATModel, self).__init__()
        self.gat1 = GATConv(hidden_dim, activation='relu', attn_heads=8)
        self.gat2 = GATConv(output_dim)  # Output layer for multi-task learning
        self.dropout = Dropout(0.5)
    
    def call(self, inputs):
        x, a, e = inputs
        x = self.gat1([x, a, e])
        x = self.dropout(x)
        x = self.gat2([x, a, e])
        return x

# Step 7: Self-Supervised Pretraining (Optional)
def self_supervised_pretrain(model, unlabeled_graphs):
    # Define a pretext task (e.g., predicting distances between nodes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    for epoch in range(10):  # Pretrain for a few epochs
        total_loss = 0
        for graph in unlabeled_graphs:
            with tf.GradientTape() as tape:
                x, a, e = graph.x, graph.a, graph.e
                predictions = model([x, a, e])
                loss = tf.reduce_mean(tf.square(predictions - e))  # Example: Predict edge features
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += loss.numpy()
        print(f"Pretraining Epoch {epoch+1}, Loss: {total_loss / len(unlabeled_graphs):.4f}")

# Optional: Pretrain on unlabeled data
# self_supervised_pretrain(model, unlabeled_graphs)

# Step 8: Train the GAT Model
model = GATModel(hidden_dim=128, output_dim=35)
model.compile(optimizer='adam', loss='mse')

# Create DataLoader for training and validation
train_loader = BatchLoader(train_graphs_split, batch_size=32, shuffle=True)
val_loader = BatchLoader(val_graphs_split, batch_size=32, shuffle=False)

# Training loop
for epoch in range(50):
    total_loss = 0
    for batch in train_loader:
        x, a, e, y = batch.x, batch.a, batch.e, batch.y
        loss = model.train_on_batch([x, a, e], y)
        total_loss += loss
    print(f"Training Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# Step 9: Predict on Test Data
x_test, a_test, e_test = test_graph.x, test_graph.a, test_graph.e
predictions = model.predict([x_test, a_test, e_test])

# Step 10: Save Predictions to CSV
submission = pd.DataFrame(predictions, columns=[f'C{i+1}' for i in range(35)])
submission.insert(0, 'ID', test_spots.index)  # Add ID column
submission.to_csv('submission.csv', index=False)







