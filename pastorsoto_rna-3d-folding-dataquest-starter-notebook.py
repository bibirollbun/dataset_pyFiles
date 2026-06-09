import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler  # Using MinMaxScaler instead
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load data
def load_data():
    train_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')
    train_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')
    test_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')
    return train_sequences, train_labels, test_sequences

# Preprocessing function with fixed sequence length and scaled data
def preprocess_data(train_sequences, train_labels, test_sequences, fixed_seq_length=50):
    # One-hot encoding mapping
    nucleotides = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'U': [0,0,0,1]}
    default_nuc = [0.25, 0.25, 0.25, 0.25]  # For non-standard nucleotides
    
    # Process training data
    X_train = []
    y_train = []
    valid_counts = 0
    
    for idx, row in train_sequences.iterrows():
        target_id = row['target_id']
        seq = row['sequence']
        
        # Skip sequences longer than our fixed length to avoid truncation issues
        if len(seq) > fixed_seq_length * 2:
            continue
            
        # Get labels for this sequence
        target_labels = train_labels[train_labels['ID'].str.startswith(target_id + '_')]
        
        if len(target_labels) > 0:
            # Check if target_labels has valid numeric data
            has_valid_coordinates = True
            for _, label_row in target_labels.iterrows():
                if (pd.isna(label_row['x_1']) or pd.isna(label_row['y_1']) or pd.isna(label_row['z_1'])):
                    has_valid_coordinates = False
                    break
            
            if not has_valid_coordinates:
                continue
                
            # Create fixed-length sequence representation
            seq_encoded = np.zeros((fixed_seq_length, 4))
            for i in range(min(len(seq), fixed_seq_length)):
                seq_encoded[i] = nucleotides.get(seq[i], default_nuc)
            
            # Create fixed-length coordinate array
            coords = np.zeros((fixed_seq_length, 3))
            for _, label_row in target_labels.iterrows():
                resid = label_row['resid']
                if 1 <= resid <= fixed_seq_length:
                    coords[resid-1] = [
                        float(label_row['x_1']), 
                        float(label_row['y_1']), 
                        float(label_row['z_1'])
                    ]
            
            # Skip if all coordinates are zero (would cause training issues)
            if np.all(coords == 0):
                continue
                
            X_train.append(seq_encoded)
            y_train.append(coords)
            valid_counts += 1
    
    print(f"Using {valid_counts} valid training examples")
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Use MinMaxScaler instead of StandardScaler for better numerical stability
    scaler = MinMaxScaler(feature_range=(-1, 1))  # Range from -1 to 1
    y_train_reshaped = y_train.reshape(-1, 3)
    y_train_normalized = scaler.fit_transform(y_train_reshaped)
    y_train = y_train_normalized.reshape(y_train.shape)
    
    # Check for any remaining NaN values and replace them
    X_train = np.nan_to_num(X_train)
    y_train = np.nan_to_num(y_train)
    
    # Process test data
    X_test = []
    test_ids = []
    test_seq_lengths = []
    
    for idx, row in test_sequences.iterrows():
        target_id = row['target_id']
        seq = row['sequence']
        test_seq_lengths.append(len(seq))
        
        # Create fixed-length sequence representation
        seq_encoded = np.zeros((fixed_seq_length, 4))
        for i in range(min(len(seq), fixed_seq_length)):
            seq_encoded[i] = nucleotides.get(seq[i], default_nuc)
        
        X_test.append(seq_encoded)
        test_ids.append(target_id)
    
    X_test = np.array(X_test)
    
    return X_train, y_train, X_test, test_ids, test_seq_lengths, scaler

# Extremely simple model to avoid NaN issues
def build_simple_model(seq_length):
    model = models.Sequential([
        layers.InputLayer(input_shape=(seq_length, 4)),
        layers.Flatten(),
        layers.Dense(128, activation='relu', 
                    kernel_initializer='he_normal',
                    kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Dense(256, activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Dense(seq_length * 3),
        layers.Reshape((seq_length, 3))
    ])
    
    # Use a more robust optimizer with gradient clipping
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0001,  # Reduced learning rate
        clipnorm=1.0  # Gradient clipping
    )
    
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Generate 5 different predictions
def generate_predictions(model, X_test, scaler, test_seq_lengths):
    # Base prediction
    base_pred = model.predict(X_test)
    
    # Create 5 different predictions
    all_preds = []
    
    # First is the base prediction
    all_preds.append(base_pred)
    
    # Add 4 variations with small noise
    for i in range(4):
        noise = np.random.normal(0, 0.02 * (i+1), base_pred.shape)
        noisy_pred = base_pred + noise
        all_preds.append(noisy_pred)
    
    # Denormalize
    all_denorm = []
    for pred in all_preds:
        pred_flat = pred.reshape(-1, 3)
        denorm_flat = scaler.inverse_transform(pred_flat)
        denorm = denorm_flat.reshape(pred.shape)
        all_denorm.append(denorm)
    
    return all_denorm

# Create submission file
def create_submission(predictions, test_sequences, test_seq_lengths):
    sample_submission = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/sample_submission.csv')
    submission_rows = []
    
    for idx, target_id in enumerate(test_sequences['target_id']):
        seq = test_sequences.loc[test_sequences['target_id'] == target_id, 'sequence'].values[0]
        seq_length = len(seq)
        
        for i in range(seq_length):
            row = {
                'ID': f"{target_id}_{i+1}",
                'resname': seq[i],
                'resid': i+1
            }
            
            # Add coordinates for all 5 predictions
            for model_idx in range(5):
                pred = predictions[model_idx][idx]
                if i < len(pred):
                    row[f'x_{model_idx+1}'] = pred[i, 0]
                    row[f'y_{model_idx+1}'] = pred[i, 1] 
                    row[f'z_{model_idx+1}'] = pred[i, 2]
                else:
                    # Use last prediction for positions beyond model's sequence length
                    row[f'x_{model_idx+1}'] = pred[-1, 0]
                    row[f'y_{model_idx+1}'] = pred[-1, 1]
                    row[f'z_{model_idx+1}'] = pred[-1, 2]
            
            submission_rows.append(row)
    
    # Create DataFrame with same columns as sample submission
    submission = pd.DataFrame(submission_rows)
    submission = submission[sample_submission.columns]
    
    return submission



# Fixed sequence length - using smaller value
FIXED_SEQ_LENGTH = 50

# 1. Load data
train_sequences, train_labels, test_sequences = load_data()

# 2. Preprocess with fixed length
X_train, y_train, X_test, test_ids, test_seq_lengths, scaler = preprocess_data(
    train_sequences, train_labels, test_sequences, FIXED_SEQ_LENGTH
)

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Test data shape: {X_test.shape}")

# 3. Build and train model
model = build_simple_model(FIXED_SEQ_LENGTH)
model.summary()

# Use early stopping to prevent overfitting and detect NaN
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=3,
    restore_best_weights=True
)

# Reduced epochs and batch size
history = model.fit(
    X_train, y_train, 
    epochs=3, 
    batch_size=4, 
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

# Check if training was successful
if np.isnan(history.history['loss'][-1]):
    print("Warning: NaN loss detected. Using fallback prediction method.")
    # Create a fallback prediction based on average coordinates
    avg_coords = np.mean(y_train, axis=0)
    base_pred = np.tile(avg_coords, (len(X_test), 1, 1))
    
    # Manually create 5 predictions with slight variations
    all_preds = [base_pred]
    for i in range(4):
        noise = np.random.normal(0, 0.05 * (i+1), base_pred.shape)
        noisy_pred = base_pred + noise
        all_preds.append(noisy_pred)
        
    # Denormalize
    predictions = []
    for pred in all_preds:
        pred_flat = pred.reshape(-1, 3)
        denorm_flat = scaler.inverse_transform(pred_flat)
        denorm = denorm_flat.reshape(pred.shape)
        predictions.append(denorm)
else:
    # Normal prediction if training succeeded
    predictions = generate_predictions(model, X_test, scaler, test_seq_lengths)

# 5. Create submission
submission = create_submission(predictions, test_sequences, test_seq_lengths)

# 6. Save submission
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")


submission

