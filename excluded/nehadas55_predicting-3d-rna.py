!pip install /kaggle/input/spektral/pytorch/default/1/spektral-1.3.1-py3-none-any.whl > /dev/null 2>&1
!pip install /kaggle/input/bio/pytorch/default/1/biopython-1.85-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl > /dev/null 2>&1


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.optimizers import AdamW
import spektral.layers as gnn_layers
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import gc
import pickle
from Bio import SeqIO
import subprocess
import warnings
import logging


warnings.filterwarnings('ignore')
set_global_policy('mixed_float16')


# Set up logging
logging.basicConfig(filename='submission.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Set seeds
np.random.seed(42)
tf.random.set_seed(42)


# ==============================
# GPU Configuration
# ==============================

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


class ReshapeLayer(Layer):
    def __init__(self, target_shape, **kwargs):
        super(ReshapeLayer, self).__init__(**kwargs)
        self.target_shape = target_shape  # e.g., (num_nodes, 3)
    
    def call(self, inputs):
        return tf.reshape(inputs, [-1] + list(self.target_shape))
    
    def compute_output_shape(self, input_shape):
        return (None,) + self.target_shape


# Custom callback to check for NaN gradients
class GradientCheckCallback(tf.keras.callbacks.Callback):
    def __init__(self, training_data):
        super(GradientCheckCallback, self).__init__()
        self.training_data = training_data

    def on_batch_end(self, batch, logs=None):
        # Fetch the current batch (requires access to the training dataset)
        # This is tricky because `batch` is just an index, and we need the actual data
        # For simplicity, this would need a custom training loop or batch access
        batch_data = next(iter(self.training_data.take(1)))
        inputs, targets = batch_data

        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.model.compiled_loss(targets, predictions, regularization_losses=self.model.losses)

        grads = tape.gradient(loss, self.model.trainable_weights)
        for i, g in enumerate(grads):
            if g is not None and (tf.reduce_any(tf.math.is_nan(g)) or tf.reduce_any(tf.math.is_inf(g))):
                print(f"NaN/Inf gradient detected in weight {i}")


# ==============================
# Data Loading
# ==============================

DATA_PATH = "/kaggle/input/stanford-rna-3d-folding/"
MSA_DIR = os.path.join(DATA_PATH, "MSA")
MAX_NODES = 1000
FEATURES = [
    'res_pos', 'pairing_prob',
    'freq_A', 'freq_C', 'freq_G', 'freq_U',
    'resname_A', 'resname_C', 'resname_G', 'resname_U', 'resname_-',
    'prev_resname_A', 'prev_resname_C', 'prev_resname_G', 'prev_resname_U', 'prev_resname_-',
    'next_resname_A', 'next_resname_C', 'next_resname_G', 'next_resname_U', 'next_resname_-'
]
COORDINATE_COLUMNS = ['x_1', 'y_1', 'z_1']
COORDINATE_COLUMNS_SUBMISSION = [
    'x_1', 'y_1', 'z_1', 'x_2', 'y_2', 'z_2', 'x_3', 'y_3', 'z_3',
    'x_4', 'y_4', 'z_4', 'x_5', 'y_5', 'z_5'
]


def safe_read_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path} with shape {df.shape}")
        logging.info(f"Successfully loaded {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        print(f"Error: Failed to load {file_path}. Exception: {str(e)}")
        logging.error(f"Failed to load {file_path}. Exception: {str(e)}")
        return None

def safe_read_fasta(file_path):
    try:
        return {record.id: str(record.seq) for record in SeqIO.parse(file_path, "fasta")}
    except:
        return {}

def load_msa_subset(msa_dir, seq_ids):
    msa_data = {}
    for seq_id in seq_ids:
        msa_file = os.path.join(msa_dir, f"{seq_id}.MSA.fasta")
        if os.path.exists(msa_file):
            msa_data[seq_id] = safe_read_fasta(msa_file)
    return msa_data

def compute_msa_features_for_sequence(msa_dict, target_sequence, max_resid):
    if not msa_dict or not target_sequence:
        return pd.DataFrame({
            'resid': range(1, max_resid + 1),
            'freq_A': 0, 'freq_C': 0, 'freq_G': 0, 'freq_U': 0
        })
    
    seq_length = min(len(target_sequence), max_resid)
    msa_sequences = [s[:seq_length] for s in msa_dict.values()]
    
    # Convert to TensorFlow tensor and move to GPU
    msa_array = tf.convert_to_tensor([list(s.ljust(seq_length, '-')) for s in msa_sequences], dtype=tf.string)
    total_seqs = tf.cast(len(msa_sequences), tf.float32)
    
    features = {'resid': np.arange(1, seq_length + 1)}
    for base in ['A', 'C', 'G', 'U']:
        freq = tf.reduce_sum(tf.cast(tf.equal(msa_array, base), tf.float32), axis=0) / total_seqs
        features[f'freq_{base}'] = freq.numpy()
    
    return pd.DataFrame(features)

# Top-level function for multiprocessing
def process_msa_sequence(args):
    seq_id, group, msa_df = args
    msa_df['sequence_id'] = seq_id
    return msa_df

def compute_pairing_probabilities(sequence, max_resid):
    if not isinstance(sequence, str) or not sequence or max_resid <= 0:
        return np.zeros(max_resid)
    seq_len = min(len(sequence), max_resid)
    if seq_len < 2:
        return np.zeros(max_resid)
    probs = np.zeros(max_resid)
    for i in range(seq_len):
        base = sequence[i]
        if base in ['G', 'C', 'A', 'U']:
            probs[i] = 0.1
    return probs

# Top-level function for multiprocessing
def process_secondary_structure(args):
    seq_id, group, seq_lookup = args
    sequence = seq_lookup.get(seq_id, '')
    max_resid = group['resid'].max()
    probs = compute_pairing_probabilities(sequence, max_resid)
    return seq_id, probs[:len(group)]
    
def normalize_coordinates(df, coord_cols):
    for col in coord_cols:
        stats = df.groupby('sequence_id')[col].agg(['mean', 'std']).reset_index()
        stats['std'] = stats['std'].replace(0, 1.0)
        df = df.merge(stats, on='sequence_id')
        df[col] = (df[col] - df['mean']) / df['std']
        df[col] = df[col].fillna(0).clip(lower=-10, upper=10)
        df = df.drop(columns=['mean', 'std'])
    return df

def check_data_quality(df, cols):
    valid_cols = [col for col in cols if col in df.columns]
    print(f"Checking columns: {valid_cols}")
    print(f"NaNs in {valid_cols}:", df[valid_cols].isna().sum())
    print(f"Infs in {valid_cols}:", np.isinf(df[valid_cols]).sum())
    df[valid_cols] = df[valid_cols].replace([np.inf, -np.inf], 0).fillna(0)
    return df

def add_msa_features(df, seq_df, msa_data):
    seq_lookup = dict(zip(seq_df['ID'], seq_df['sequence']))
    msa_features = []
    
    # Compute all MSA features on GPU in the main process
    msa_results = []
    for seq_id, group in df.groupby('sequence_id'):
        max_resid = group['resid'].max()
        target_seq = seq_lookup.get(seq_id, '')
        msa_dict = msa_data.get(seq_id, {})
        msa_df = compute_msa_features_for_sequence(msa_dict, target_seq, max_resid)
        msa_results.append((seq_id, group, msa_df))
    
    # Parallelize DataFrame operations across CPU cores
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        msa_features = list(executor.map(process_msa_sequence, msa_results))
    
    msa_features_df = pd.concat(msa_features)
    return df.merge(msa_features_df, on=['sequence_id', 'resid'], how='left').fillna(0)

def add_secondary_structure_features(df, seq_df):
    seq_lookup = dict(zip(seq_df['ID'], seq_df['sequence']))
    df['pairing_prob'] = 0.0
    
    # Prepare arguments for multiprocessing
    groups = [(seq_id, group, seq_lookup) for seq_id, group in df.groupby('sequence_id')]
    
    # Parallelize across CPU cores
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(executor.map(process_secondary_structure, groups))
    
    # Update DataFrame in one go
    for seq_id, probs in results:
        df.loc[df['sequence_id'] == seq_id, 'pairing_prob'] = probs
    
    return df

def add_sequence_context(df):
    grouped = df.groupby('sequence_id')
    df['prev_resname'] = grouped['resname'].shift(1).fillna('-')
    df['next_resname'] = grouped['resname'].shift(-1).fillna('-')
    return df

def create_adj_matrix(pairing_probs, threshold=0.05):
    adj = (pairing_probs[:, None] + pairing_probs[None, :]) > threshold
    return adj.astype(float)

def prepare_sequence_data(df, features, max_nodes=MAX_NODES, include_adj=False):
    node_features = []
    adj_matrices = []
    coords = []
    seq_ids = []
    res_counts = []
    for seq_id, group in df.groupby('sequence_id'):
        feat = group[features].values
        pairing = group['pairing_prob'].values
        coord = group[COORDINATE_COLUMNS].values if all(c in df.columns for c in COORDINATE_COLUMNS) else np.zeros((len(group), 3))
        num_nodes = len(feat)
        res_counts.append(num_nodes)
        seq_ids.append(seq_id)
        if num_nodes > max_nodes:
            feat = feat[:max_nodes]
            pairing = pairing[:max_nodes]
            coord = coord[:max_nodes]
            num_nodes = max_nodes
        padded_feat = np.pad(feat, ((0, max_nodes - num_nodes), (0, 0)), mode='constant')
        padded_coord = np.pad(coord, ((0, max_nodes - num_nodes), (0, 0)), mode='constant')
        if include_adj:
            adj = create_adj_matrix(pairing)
            padded_adj = np.pad(adj, ((0, max_nodes - num_nodes), (0, max_nodes - num_nodes)), mode='constant')
            adj_matrices.append(padded_adj)
        node_features.append(padded_feat)
        coords.append(padded_coord)
    node_features = np.array(node_features, dtype=np.float32)
    coords = np.array(coords, dtype=np.float32)
    # Validate inputs
    print(f"node_features shape: {node_features.shape}")
    print(f"coords shape: {coords.shape}")
    if np.any(np.isnan(node_features)) or np.any(np.isinf(node_features)):
        raise ValueError("NaNs or Infs detected in node_features")
    if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
        raise ValueError("NaNs or Infs detected in coords")
    if include_adj:
        adj_matrices = np.array(adj_matrices, dtype=np.float32)
        print(f"adj_matrices shape: {adj_matrices.shape}")
        if np.any(np.isnan(adj_matrices)) or np.any(np.isinf(adj_matrices)):
            raise ValueError("NaNs or Infs detected in adj_matrices")
        return node_features, adj_matrices, coords, seq_ids, res_counts
    return node_features, coords, seq_ids, res_counts

# Build & Model Training with Dense Model (Single GPU)
def build_gnn_model(num_features, num_nodes=MAX_NODES):
    node_input = Input(shape=(num_nodes, num_features), name='node_features', dtype=tf.float32)
    x = Dense(256, activation='relu')(node_input)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = tf.keras.layers.Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    output = Dense(3 * num_nodes)(x)
    output = ReshapeLayer((num_nodes, 3))(output)
    return Model(inputs=node_input, outputs=output)

# Update predict_with_dropout to ensure varied predictions
def predict_with_dropout(model, X_nodes, num_samples=5, batch_size=4):
    predictions = []
    for i in range(num_samples):
        batch_preds = []
        for batch_start in range(0, len(X_nodes), batch_size):
            batch_end = min(batch_start + batch_size, len(X_nodes))
            batch_X = X_nodes[batch_start:batch_end]
            tf.random.set_seed(None)
            np.random.seed(None)
            pred = model(batch_X, training=True)
            batch_preds.append(pred.numpy())
        predictions.append(np.concatenate(batch_preds, axis=0))
    return np.stack(predictions, axis=1)

def validate_submission(submission, sample_submission):
    assert submission.shape == sample_submission.shape, f"Shape mismatch: {submission.shape} vs {sample_submission.shape}"
    assert all(col in submission.columns for col in sample_submission.columns), "Column mismatch"
    assert not submission[COORDINATE_COLUMNS_SUBMISSION].isna().any().any(), "NaNs in submission"
    assert not np.isinf(submission[COORDINATE_COLUMNS_SUBMISSION]).any().any(), "Infs in submission"


train_seq = safe_read_csv(os.path.join(DATA_PATH, "train_sequences.v2.csv")).rename(columns={'target_id': 'ID'})
test_seq = safe_read_csv(os.path.join(DATA_PATH, "test_sequences.csv")).rename(columns={'target_id': 'ID'})
train_labels = safe_read_csv(os.path.join(DATA_PATH, "train_labels.v2.csv"))
validation_seq = safe_read_csv(os.path.join(DATA_PATH, "validation_sequences.csv")).rename(columns={'target_id': 'ID'})
validation_labels = safe_read_csv(os.path.join(DATA_PATH, "validation_labels.csv"))
sample_submission = safe_read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))


# ==============================
# Data Preprocessing
# ==============================

train_labels['sequence_id'] = train_labels['ID'].apply(lambda x: x.rsplit('_', 1)[0])
validation_labels['sequence_id'] = validation_labels['ID'].apply(lambda x: x.rsplit('_', 1)[0])
sample_submission['sequence_id'] = sample_submission['ID'].apply(lambda x: x.rsplit('_', 1)[0])

train_data = train_labels.merge(train_seq[['ID', 'sequence']], left_on='sequence_id', right_on='ID', how='left', suffixes=('_label', '_seq'))
val_data = validation_labels.merge(validation_seq[['ID', 'sequence']], left_on='sequence_id', right_on='ID', how='left', suffixes=('_label', '_seq'))
test_data = sample_submission.merge(test_seq[['ID', 'sequence']], left_on='sequence_id', right_on='ID', how='left', suffixes=('_label', '_seq'))

train_data = train_data.rename(columns={'ID_label': 'ID_x', 'ID_seq': 'ID_y'})
val_data = val_data.rename(columns={'ID_label': 'ID_x', 'ID_seq': 'ID_y'})
test_data = test_data.rename(columns={'ID_label': 'ID_x', 'ID_seq': 'ID_y'})

def add_residue_features(df):
    if 'resid' not in df.columns:
        df['resid'] = df['ID_x'].str.extract(r'_r(\d+)$').astype(float)
    if 'resname' not in df.columns:
        df['resname'] = df['sequence'].str.split('').str[df['resid'].astype(int)]
    if 'res_pos' not in df.columns:
        df['res_pos'] = df.groupby('sequence_id')['resid'].transform(lambda x: x / x.max() if x.max() > 0 else 0)
    return df

train_data = add_residue_features(train_data)
val_data = add_residue_features(val_data)
test_data = add_residue_features(test_data)

for df in [train_data, val_data, test_data]:
    df['sequence'] = df['sequence'].fillna('')
    df['resname'] = df['resname'].fillna('-')
    df['resid'] = df['resid'].fillna(0)
    df['res_pos'] = df['res_pos'].fillna(0)

print("train_data columns:", train_data.columns)
print("Missing values in train_data:", train_data.isna().sum())


# ==============================
# Down-Sample training data
# ==============================

train_data = train_data.sample(n=300000, random_state=42).reset_index(drop=True)


# ==============================
# Load and cache MSA features
# ==============================

seq_ids = set(train_seq['ID']).union(validation_seq['ID'], test_seq['ID'])
msa_feature_cache = '/kaggle/working/msa_features.pkl'
if os.path.exists(msa_feature_cache):
    with open(msa_feature_cache, 'rb') as f:
        msa_data = pickle.load(f)
else:
    msa_data = load_msa_subset(MSA_DIR, set(train_data['sequence_id']).union(set(val_data['sequence_id']), set(test_data['sequence_id'])))
    with open(msa_feature_cache, 'wb') as f:
        pickle.dump(msa_data, f)


# ==============================
# Add features
# ==============================

# Add MSA features
print("Adding MSA features...")
train_data = add_msa_features(train_data, train_seq, msa_data)
val_data = add_msa_features(val_data, validation_seq, msa_data)
test_data = add_msa_features(test_data, test_seq, msa_data)

# Add secondary structure features
print("Adding secondary structure features...")
train_data = add_secondary_structure_features(train_data, train_seq)
val_data = add_secondary_structure_features(val_data, validation_seq)
test_data = add_secondary_structure_features(test_data, test_seq)

# Add sequence context
print("Adding sequence context...")
train_data = add_sequence_context(train_data)
val_data = add_sequence_context(val_data)
test_data = add_sequence_context(test_data)

print("Feature addition completed.")


# ==============================
# Encode categorical features
# ==============================

resname_categories = ['A', 'C', 'G', 'U', '-']
for df in [train_data, val_data, test_data]:
    df['resname'] = pd.Categorical(df['resname'], categories=resname_categories)
    df['prev_resname'] = pd.Categorical(df['prev_resname'], categories=resname_categories)
    df['next_resname'] = pd.Categorical(df['next_resname'], categories=resname_categories)
    # Create one-hot encoded columns
    resname_dummies = pd.get_dummies(df['resname'], prefix='resname', dtype=np.uint8)
    prev_resname_dummies = pd.get_dummies(df['prev_resname'], prefix='prev_resname', dtype=np.uint8)
    next_resname_dummies = pd.get_dummies(df['next_resname'], prefix='next_resname', dtype=np.uint8)
    # Add new columns to df
    df[resname_dummies.columns] = resname_dummies
    df[prev_resname_dummies.columns] = prev_resname_dummies
    df[next_resname_dummies.columns] = next_resname_dummies


# ==============================
# Normalize coordinates
# ==============================

train_data = normalize_coordinates(train_data, COORDINATE_COLUMNS)
val_data = normalize_coordinates(val_data, COORDINATE_COLUMNS)


# ==============================
# Check Data Quality
# ==============================

train_data = check_data_quality(train_data, FEATURES + COORDINATE_COLUMNS)
val_data = check_data_quality(val_data, FEATURES + COORDINATE_COLUMNS)
test_data = check_data_quality(test_data, FEATURES)

print("train_data columns after feature engineering:", train_data.columns)
print("Missing values in train_data:", train_data.isna().sum())


# ==============================
# Prepare GNN data
# ==============================

X_train_nodes, y_train, train_seq_ids, train_res_counts = prepare_sequence_data(train_data, FEATURES, include_adj=False)
X_val_nodes, y_val, val_seq_ids, val_res_counts = prepare_sequence_data(val_data, FEATURES, include_adj=False)
X_test_nodes, _, test_seq_ids, test_res_counts = prepare_sequence_data(test_data, FEATURES, include_adj=False)


# Debug NaNs before training
print("NaNs in X_train_nodes:", np.isnan(X_train_nodes).sum())

print("NaNs in y_train:", np.isnan(y_train).sum())


print(f"X_train_nodes min/max: {X_train_nodes.min()}, {X_train_nodes.max()}")
print(f"y_train min/max: {y_train.min()}, {y_train.max()}")
print(f"NaNs in X_train_nodes: {np.isnan(X_train_nodes).sum()}")
print(f"NaNs in y_train: {np.isnan(y_train).sum()}")


# Normalize Inputs and Targets
scaler_nodes = StandardScaler()
X_train_nodes_reshaped = X_train_nodes.reshape(-1, X_train_nodes.shape[-1])
X_val_nodes_reshaped = X_val_nodes.reshape(-1, X_val_nodes.shape[-1])
X_test_nodes_reshaped = X_test_nodes.reshape(-1, X_test_nodes.shape[-1])

X_train_nodes_scaled = scaler_nodes.fit_transform(X_train_nodes_reshaped).reshape(X_train_nodes.shape)
X_val_nodes_scaled = scaler_nodes.transform(X_val_nodes_reshaped).reshape(X_val_nodes.shape)
X_test_nodes_scaled = scaler_nodes.transform(X_test_nodes_reshaped).reshape(X_test_nodes.shape)

scaler_targets = StandardScaler()
y_train_reshaped = y_train.reshape(-1, y_train.shape[-1])
y_val_reshaped = y_val.reshape(-1, y_val.shape[-1])

y_train_scaled = scaler_targets.fit_transform(y_train_reshaped).reshape(y_train.shape)
y_val_scaled = scaler_targets.transform(y_val_reshaped).reshape(y_val.shape)


# Debug normalized data
print(f"X_train_nodes_scaled min/max: {X_train_nodes_scaled.min()}, {X_train_nodes_scaled.max()}")
print(f"y_train_scaled min/max: {y_train_scaled.min()}, {y_train_scaled.max()}")
print(f"NaNs in X_train_nodes_scaled: {np.isnan(X_train_nodes_scaled).sum()}")
print(f"NaNs in y_train_scaled: {np.isnan(y_train_scaled).sum()}")


# Convert training and validation data to tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_nodes_scaled, y_train_scaled)).batch(16).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val_nodes_scaled, y_val_scaled)).batch(16).prefetch(tf.data.AUTOTUNE)


# ==============================
# Build & Model Training
# ==============================

model = build_gnn_model(num_features=len(FEATURES))
model.compile(optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-4, clipnorm=1.0), 
              loss=lambda y_true, y_pred: tf.keras.losses.mse(y_true, y_pred) + 1e-6)

lr_scheduler = ReduceLROnPlateau(patience=5, factor=0.8)

# Check initial predictions for NaNs
test_batch_nodes = X_train_nodes_scaled[:8]
initial_pred = model.predict(test_batch_nodes, batch_size=8)
print(f"Initial predictions min/max: {initial_pred.min()}, {initial_pred.max()}")
print(f"Initial predictions NaNs: {np.isnan(initial_pred).sum()}")

# Train the model using tf.data.Dataset
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[lr_scheduler, GradientCheckCallback(train_dataset)],
    verbose=2
)


# ==============================
# Generate Predictions
# ==============================

y_test_pred_scaled = predict_with_dropout(model, X_test_nodes_scaled, num_samples=5, batch_size=4)

# Inverse transform predictions to original scale
y_test_pred_reshaped = y_test_pred_scaled.reshape(-1, y_test_pred_scaled.shape[-1])
y_test_pred = scaler_targets.inverse_transform(y_test_pred_reshaped).reshape(y_test_pred_scaled.shape)

# Handle NaNs/Infs and clip coordinates
y_test_pred = np.nan_to_num(y_test_pred, nan=0.0, posinf=0.0, neginf=0.0)
y_test_pred = np.clip(y_test_pred, -100, 100)

print(f"y_test_pred shape: {y_test_pred.shape}")
print(f"y_test_pred min/max: {np.nanmin(y_test_pred)}, {np.nanmax(y_test_pred)}")
print(f"y_test_pred NaNs: {np.isnan(y_test_pred).sum()}")
print(f"y_test_pred[0, :, 0, :]: {y_test_pred[0, :, 0, :]}")


# Clean up
gc.collect()
tf.keras.backend.clear_session()


# ==============================
# Prepare Submission
# ==============================

submission = sample_submission.copy()
pred_dict = {}

for seq_idx, seq_id in enumerate(test_seq_ids):
    num_residues = test_res_counts[seq_idx]
    print(f"Test sequence {seq_id}: num_residues={num_residues}")  # Debug hidden test set
    coords = y_test_pred[seq_idx, :, :min(num_residues, MAX_NODES), :]
    print(f"Sequence {seq_id}: num_residues={num_residues}, coords shape={coords.shape}")
    for sample_idx in range(coords.shape[0]):
        for res_idx in range(num_residues):
            res_id = f"{seq_id}_{res_idx + 1}"
            if res_idx < coords.shape[1]:
                pred_dict[(res_id, sample_idx)] = coords[sample_idx, res_idx, :]
            else:
                mean_coords = np.mean(coords[sample_idx, :num_residues, :], axis=0)
                pred_dict[(res_id, sample_idx)] = mean_coords

# Assign predictions to submission
for sample_idx in range(5):
    for coord_idx, coord_name in enumerate(['x', 'y', 'z']):
        col = f"{coord_name}_{sample_idx + 1}"
        submission[col] = submission['ID'].apply(
            lambda id: pred_dict.get((id, sample_idx), [0, 0, 0])[coord_idx]
        )


# Ensure no NaNs or Infs
submission[COORDINATE_COLUMNS_SUBMISSION] = submission[COORDINATE_COLUMNS_SUBMISSION].fillna(0)
submission[COORDINATE_COLUMNS_SUBMISSION] = np.nan_to_num(
    submission[COORDINATE_COLUMNS_SUBMISSION].values, nan=0.0, posinf=0.0, neginf=0.0
)


# Debug submission coordinates
print(f"Submission coordinates min/max: {submission[COORDINATE_COLUMNS_SUBMISSION].min().min()}, {submission[COORDINATE_COLUMNS_SUBMISSION].max().max()}")


# ==============================
# Validate and save submission
# ==============================

validate_submission(submission, sample_submission)
submission = submission[['ID', 'resname', 'resid'] + COORDINATE_COLUMNS_SUBMISSION]
submission.to_csv('submission.csv', index=False)


submission.head()


print(f"Submission shape: {submission.shape}")
print(f"Submission NaNs: {submission[COORDINATE_COLUMNS_SUBMISSION].isna().sum()}")


# ==============================
# Compute TM Score
# ==============================

# Function to compute d0 based on L_ref
def compute_d0(L_ref):
    if L_ref < 12:
        return 0.3
    elif 12 <= L_ref <= 15:
        return 0.4
    elif 16 <= L_ref <= 19:
        return 0.5
    elif 20 <= L_ref <= 23:
        return 0.6
    elif 24 <= L_ref <= 29:
        return 0.7
    else:
        return 1.24 * (L_ref - 15) ** (1/3) - 1.8

# Kabsch algorithm for rigid-body alignment (simplified)
def kabsch_align(pred_coords, ref_coords):
    # Center the coordinates
    pred_centroid = np.mean(pred_coords, axis=0)
    ref_centroid = np.mean(ref_coords, axis=0)
    pred_centered = pred_coords - pred_centroid
    ref_centered = ref_coords - ref_centroid
    
    # Compute the covariance matrix
    H = np.dot(pred_centered.T, ref_centered)
    
    # Singular Value Decomposition
    U, _, Vt = np.linalg.svd(H)
    
    # Rotation matrix
    R = np.dot(Vt.T, U.T)
    
    # Ensure a right-handed coordinate system
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    # Translate and rotate predicted coordinates
    pred_aligned = np.dot(pred_centered, R) + ref_centroid
    return pred_aligned

# Function to compute TM-score
def compute_tm_score(pred_coords, ref_coords):
    # Remove rows with NaN in ref_coords
    mask = ~np.isnan(ref_coords).any(axis=1)
    pred_coords = pred_coords[mask]
    ref_coords = ref_coords[mask]
    
    L_ref = len(ref_coords)
    if L_ref == 0:
        return 0.0
    
    # Align the structures
    pred_aligned = kabsch_align(pred_coords, ref_coords)
    
    # Compute distances between aligned residues
    distances = np.sqrt(np.sum((pred_aligned - ref_coords) ** 2, axis=1))
    
    # Compute d0
    d0 = compute_d0(L_ref)
    
    # Compute TM-score
    tm_score = np.sum(1 / (1 + (distances / d0) ** 2)) / L_ref
    return tm_score


# Evaluate TM-score using validation set
# Map validation sequences to submission predictions
submission['sequence_id'] = submission['ID'].apply(lambda x: x.split('_')[0])

# Map validation sequences to submission predictions
val_submission = submission[submission['sequence_id'].isin(val_data['sequence_id'].unique())].copy()
tm_scores = []

# For each sequence in the validation set
for seq_id in val_data['sequence_id'].unique():
    # Get reference coordinates from validation_labels
    ref_df = val_data[val_data['sequence_id'] == seq_id][['x_1', 'y_1', 'z_1']].values
    
    # Get predicted coordinates (5 structures) from submission
    pred_df = val_submission[val_submission['sequence_id'] == seq_id]
    
    # Compute TM-score for each of the 5 predictions
    seq_tm_scores = []
    for sample_idx in range(5):
        pred_coords = pred_df[[f'x_{sample_idx+1}', f'y_{sample_idx+1}', f'z_{sample_idx+1}']].values
        tm_score = compute_tm_score(pred_coords, ref_df)
        seq_tm_scores.append(tm_score)
    
    # Take the best TM-score for this sequence
    best_tm_score = max(seq_tm_scores)
    tm_scores.append(best_tm_score)
    print(f"Sequence {seq_id}: Best TM-score = {best_tm_score:.4f}")

# Compute the average TM-score across all sequences
average_tm_score = np.mean(tm_scores)
print(f"Average TM-score across validation sequences: {average_tm_score:.4f}")

