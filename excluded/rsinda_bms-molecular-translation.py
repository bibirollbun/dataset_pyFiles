!pip install levenshtein



import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print('Libraries imported successfully')


# Step 2: Data Loading
# Load train_labels.csv and create image paths with nested folder structure

train_labels = pd.read_csv('/kaggle/input/bms-molecular-translation/train_labels.csv')
print(f'Train labels loaded: {len(train_labels)} samples')
print(train_labels.head())

# Generate image file paths with nested folder structure
def get_image_path(image_id):
    return f'/kaggle/input/bms-molecular-translation/train/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.png'

train_labels['image_path'] = train_labels['image_id'].apply(get_image_path)

# Verify paths exist (check first few)
print('\nChecking if image paths exist (first 5):')
for i in range(min(5, len(train_labels))):
    path = train_labels.iloc[i]['image_path']
    exists = os.path.exists(path)
    print(f'{path}: {exists}')

print(f'\nDataFrame shape: {train_labels.shape}')
print(f'Columns: {list(train_labels.columns)}')



# Step 3: Build Character-Level Tokenizer for InChI
# Create character vocabulary from all InChI strings

all_inchi = train_labels['InChI'].tolist()

# Build character set
char_set = set()
for inchi in all_inchi:
    char_set.update(inchi)

# Create character to index mapping (add special tokens)
char_to_idx = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
for idx, char in enumerate(sorted(char_set), start=4):
    char_to_idx[char] = idx

idx_to_char = {v: k for k, v in char_to_idx.items()}
vocab_size = len(char_to_idx)

print(f'Vocabulary size: {vocab_size}')
print(f'First 20 characters: {list(char_to_idx.keys())[:20]}')

# Function to encode InChI strings for TEACHER FORCING
def encode_inchi_input(inchi, max_length=275):
    """Encode input sequence (starts with <START>, no <END>)"""
    encoded = [char_to_idx['<START>']]
    for char in inchi:
        encoded.append(char_to_idx.get(char, char_to_idx['<UNK>']))
    
    # Pad or truncate
    if len(encoded) < max_length:
        encoded.extend([char_to_idx['<PAD>']] * (max_length - len(encoded)))
    else:
        encoded = encoded[:max_length]
    
    return encoded

def encode_inchi_target(inchi, max_length=275):
    """Encode target sequence (shifted by one, ends with <END>)"""
    encoded = []
    for char in inchi:
        encoded.append(char_to_idx.get(char, char_to_idx['<UNK>']))
    encoded.append(char_to_idx['<END>'])
    
    # Pad or truncate
    if len(encoded) < max_length:
        encoded.extend([char_to_idx['<PAD>']] * (max_length - len(encoded)))
    else:
        encoded = encoded[:max_length]
    
    return encoded

# Function to decode sequences back to InChI
def decode_inchi(encoded_seq):
    decoded = []
    for idx in encoded_seq:
        if idx == char_to_idx['<END>'] or idx == char_to_idx['<PAD>']:
            break
        if idx != char_to_idx['<START>']:
            decoded.append(idx_to_char.get(idx, '<UNK>'))
    return ''.join(decoded)

# Test encoding and decoding
test_inchi = train_labels.iloc[0]['InChI']
encoded_input = encode_inchi_input(test_inchi)
encoded_target = encode_inchi_target(test_inchi)
decoded = decode_inchi(encoded_target)

print(f'\nOriginal InChI: {test_inchi[:100]}...')
print(f'Encoded input length: {len(encoded_input)}')
print(f'Encoded target length: {len(encoded_target)}')
print(f'Decoded InChI: {decoded[:100]}...')
print(f'Match: {test_inchi == decoded}')



# Step 4: Image Preprocessing and Data Preparation
# Prepare a smaller subset for faster training (use first 10000 samples)
# For full training, remove the subset limitation

MAX_SAMPLES = 100  # Reduce for testing, set to None for full training
IMAGE_SIZE = (224, 224)
MAX_INCHI_LENGTH = 275

if MAX_SAMPLES:
    train_labels_subset = train_labels.head(MAX_SAMPLES).copy()
else:
    train_labels_subset = train_labels.copy()

print(f'Using {len(train_labels_subset)} samples for training')

# Encode all InChI strings
train_labels_subset['encoded_inchi_input'] = train_labels_subset['InChI'].apply(
    lambda x: encode_inchi_input(x, MAX_INCHI_LENGTH)
)
train_labels_subset['encoded_inchi_target'] = train_labels_subset['InChI'].apply(
    lambda x: encode_inchi_target(x, MAX_INCHI_LENGTH)
)

# Split data: 90% train, 10% validation
train_df, val_df = train_test_split(
    train_labels_subset, 
    test_size=0.1, 
    random_state=42
)

print(f'Train samples: {len(train_df)}')
print(f'Validation samples: {len(val_df)}')

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def preprocess_image(image_path):
    """Load and preprocess image with ImageNet normalization"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return img
    except:
        return None

print('\nData preparation complete!')




# Step 5: Create Data Generator with Teacher Forcing
class DataGenerator(keras.utils.Sequence):
    def __init__(self, dataframe, batch_size=32, shuffle=True):
        self.dataframe = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataframe))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))
    
    def __getitem__(self, index):
        # Get batch indexes
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Get batch data
        images = []
        decoder_inputs = []
        targets = []
        
        for idx in batch_indexes:
            row = self.dataframe.iloc[idx]
            img = preprocess_image(row['image_path'])
            if img is not None:
                images.append(img)
                decoder_inputs.append(row['encoded_inchi_input'])
                targets.append(row['encoded_inchi_target'])
        
        if len(images) == 0:
            print('error loading the data.')
            # Return dummy batch if all images failed to load
            return ({
                'image_input': np.zeros((1, 224, 224, 3), dtype=np.float32),
                'decoder_input': np.zeros((1, MAX_INCHI_LENGTH), dtype=np.int32)
            }, np.zeros((1, MAX_INCHI_LENGTH), dtype=np.int32))
        
        return ({
            'image_input': np.array(images, dtype=np.float32),
            'decoder_input': np.array(decoder_inputs, dtype=np.int32)
        }, np.array(targets, dtype=np.int32))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

print('Data generator created successfully')




# Step 6: Build PROPER Encoder-Decoder Model with Teacher Forcing
def build_model(vocab_size, max_length, learning_rate=1e-4):
    # IMAGE ENCODER: EfficientNet-B0 pretrained on ImageNet
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    
    # Fine-tune the last layers
    base_model.trainable = True
    
    # Image input
    image_input = layers.Input(shape=(224, 224, 3), name='image_input')
    
    # Extract image features
    image_features = base_model(image_input)
    image_features = layers.Dense(512, activation='relu', name='image_dense')(image_features)
    image_features = layers.Dropout(0.3)(image_features)
    
    # DECODER INPUT: Previous tokens (for teacher forcing)
    decoder_input = layers.Input(shape=(max_length,), name='decoder_input')
    
    # Embedding layer for decoder input
    decoder_embedding = layers.Embedding(
        input_dim=vocab_size,
        output_dim=256,
        mask_zero=True,
        name='decoder_embedding'
    )(decoder_input)
    
    # Initialize decoder state with image features
    # Repeat image features for each LSTM unit
    initial_state_h = layers.Dense(512, name='init_state_h')(image_features)
    initial_state_c = layers.Dense(512, name='init_state_c')(image_features)
    
    # LSTM Decoder with initial state from image
    lstm_out = layers.LSTM(
        512,
        return_sequences=True,
        return_state=False,
        name='decoder_lstm_1'
    )(decoder_embedding, initial_state=[initial_state_h, initial_state_c])
    
    lstm_out = layers.Dropout(0.3)(lstm_out)
    
    # Second LSTM layer
    lstm_out = layers.LSTM(
        512,
        return_sequences=True,
        name='decoder_lstm_2'
    )(lstm_out)
    
    lstm_out = layers.Dropout(0.3)(lstm_out)
    
    # Output layer
    outputs = layers.Dense(vocab_size, activation='softmax', name='output')(lstm_out)

    # Build model
    model = keras.Model(
        inputs=[image_input, decoder_input],
        outputs=outputs,
        name='image_to_inchi_encoder_decoder'
    )
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', LevenshteinDistanceMetric(name='mean_levenshtein_distance')]
    )
    
    return model

print('Model architecture defined successfully')



# Step 7: Levenshtein Distance for Evaluation
import Levenshtein

class LevenshteinDistanceMetric(keras.metrics.Metric):
    """
    Custom Keras metric to calculate mean Levenshtein distance
    This will be used in model.compile() for automatic tracking
    """
    def __init__(self, name='mean_levenshtein_distance', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_distance = self.add_weight(name='total_distance', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update metric state with batch predictions
        
        Note: This is a simplified version that works with token-level accuracy.
        For exact Levenshtein distance, we need the callback (which does full decoding).
        This metric provides a proxy that's correlated with Levenshtein distance.
        """
        # Get predicted tokens (argmax over vocabulary dimension)
        y_pred_tokens = tf.argmax(y_pred, axis=-1)
        
        # Compare with true tokens (element-wise)
        # This gives us a per-position accuracy, which correlates with Levenshtein
        matches = tf.cast(tf.equal(y_pred_tokens, tf.cast(y_true, tf.int64)), tf.float32)
        
        # Calculate error rate (1 - accuracy) as proxy for edit distance
        # Higher error rate ≈ higher Levenshtein distance
        errors_per_sequence = tf.reduce_sum(1.0 - matches, axis=-1)
        
        # Update running totals
        batch_distance = tf.reduce_sum(errors_per_sequence)
        self.total_distance.assign_add(batch_distance)
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
    
    def result(self):
        """Return mean distance"""
        return tf.math.divide_no_nan(self.total_distance, self.count)
    
    def reset_state(self):
        """Reset metric state"""
        self.total_distance.assign(0.0)
        self.count.assign(0.0)

print('Levenshtein distance metric class defined')


# Step 7: Custom Callback for TRUE Levenshtein Distance Validation
class MeanLevenshteinCallback(keras.callbacks.Callback):
    """
    Custom callback to calculate TRUE mean Levenshtein distance on validation set
    This does full autoregressive decoding and calculates actual edit distance
    
    This is more accurate than the compiled metric (which is a proxy)
    Use this for model selection and early stopping
    """
    def __init__(self, validation_data, val_df, max_length=275):
        super().__init__()
        self.validation_data = validation_data
        self.val_df = val_df.reset_index(drop=True)
        self.max_length = max_length
        self.levenshtein_history = []
        self.best_distance = float('inf')
        
    def on_epoch_end(self, epoch, logs=None):
        # Sample a subset of validation data for speed (use 10 samples)
        # For full validation, remove the sampling
        sample_size = min(10, len(self.val_df))
        sample_indices = np.random.choice(len(self.val_df), sample_size, replace=False)
        
        predictions = []
        ground_truths = []
        
        for idx in sample_indices:
            row = self.val_df.iloc[idx]
            img = preprocess_image(row['image_path'])
            
            if img is not None:
                # Generate prediction autoregressively
                decoder_input = np.zeros((1, self.max_length), dtype=np.int32)
                decoder_input[0, 0] = char_to_idx['<START>']
                img_batch = np.expand_dims(img, axis=0)
                
                for i in range(1, self.max_length):
                    preds = self.model.predict([img_batch, decoder_input], verbose=0)
                    next_token = np.argmax(preds[0, i-1, :])
                    
                    if next_token == char_to_idx['<END>'] or next_token == char_to_idx['<PAD>']:
                        break
                    
                    decoder_input[0, i] = next_token
                
                pred_str = decode_inchi(decoder_input[0])
                predictions.append(pred_str)
                ground_truths.append(row['InChI'])
        
        # Calculate TRUE average Levenshtein distance
        if len(predictions) > 0:
            distances = [Levenshtein.distance(pred, gt) for pred, gt in zip(predictions, ground_truths)]
            avg_distance = np.mean(distances)
            self.levenshtein_history.append(avg_distance)
            
            # Update logs with TRUE Levenshtein distance (overrides proxy metric)
            # Use 'val_mean_levenshtein' to match the validation metric name
            logs['val_mean_levenshtein'] = avg_distance
            
            # Track best distance
            if avg_distance < self.best_distance:
                self.best_distance = avg_distance
            
            print(f'\n  TRUE Mean Levenshtein Distance: {avg_distance:.2f} (best: {self.best_distance:.2f})')

print('Mean Levenshtein callback defined')



# Step 8: Hyperparameter Grid Search Training
# Grid search over learning rate and batch size

# Define hyperparameter grid
# Grid size can be increased when we have more compute.
param_grid = {
    'learning_rate': [1e-3],
    'batch_size': [32]
}

best_score = float('inf')
best_params = None
best_model = None

print('Starting hyperparameter grid search...')
print(f'Grid: {param_grid}')
print(f'\nTesting {len(param_grid["learning_rate"]) * len(param_grid["batch_size"])} configurations')

for lr in param_grid['learning_rate']:
    for bs in param_grid['batch_size']:
        print(f'\n=== Training with lr={lr}, batch_size={bs} ===')
        
        # Build model
        model = build_model(vocab_size, MAX_INCHI_LENGTH, learning_rate=lr)
        
        # Create data generators
        train_gen = DataGenerator(train_df, batch_size=bs, shuffle=True)
        val_gen = DataGenerator(val_df, batch_size=bs, shuffle=False)
        
        # Callbacks with TRUE Mean Levenshtein distance monitoring
        mean_levenshtein_callback = MeanLevenshteinCallback(
            validation_data=val_gen,
            val_df=val_df,
            max_length=MAX_INCHI_LENGTH
        )
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_mean_levenshtein',  # Monitor Levenshtein distance instead of loss
            patience=3,
            restore_best_weights=True,
            mode='min'  # Lower distance is better
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_mean_levenshtein',  # Monitor Levenshtein distance
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            mode='min'
        )
        
        # Train model - INCREASED EPOCHS
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=2,  # Increased from 2
            callbacks=[mean_levenshtein_callback, early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate on validation set using Levenshtein distance
        best_distance = min(history.history['val_mean_levenshtein'])
        print(f'Best Levenshtein distance: {best_distance:.2f}')
        
        # Update best configuration based on Levenshtein distance
        if best_distance < best_score:
            best_score = best_distance
            best_params = {'learning_rate': lr, 'batch_size': bs}
            best_model = model
            print(f'New best configuration found!')


print(f'\n=== Grid Search Complete ===')
print(f'Best parameters: {best_params}')
print(f'Best validation distance/loss: {best_score:.4f}')





len(train_labels_subset)


# Step 8.5: Final Retraining on Combined Train + Validation Data
print('\n=== Step 8.5: Final Retraining on Combined Data ===')
print('Retraining best model on combined train + validation data for maximum performance...')

# Combine train and validation data
full_df = train_labels_subset.copy()
print(f'Combined dataset size: {len(full_df)} samples')

# Build fresh model with best hyperparameters
final_model = build_model(
    vocab_size, 
    MAX_INCHI_LENGTH, 
    learning_rate=best_params['learning_rate']
)

# Create data generator for combined data
full_df_gen = DataGenerator(full_df, batch_size=best_params['batch_size'], shuffle=True)

# Train on combined data (no validation split)
# Use fewer epochs since we already validated the hyperparameters
print(f'Training with best hyperparameters: {best_params}')

final_history = final_model.fit(
    full_df_gen,
    epochs=10,  # Same number of epochs as before
    verbose=1
)

print('\nFinal retraining complete!')
print(f'Final mean_levenshtein: {final_history.history["mean_levenshtein_distance"][-1]:.4f}')

# Use the final model for predictions
best_model = final_model
print('Updated best_model to final retrained model')



# Step 9: Generate Predictions with AUTOREGRESSIVE DECODING
# Load test data
test_df = pd.read_csv('/kaggle/input/bms-molecular-translation/sample_submission.csv')
print(f'Test samples: {len(test_df)}')

if MAX_SAMPLES:
    test_df = test_df.head(MAX_SAMPLES).copy()


# Generate test image paths
def get_test_image_path(image_id):
    return f'/kaggle/input/bms-molecular-translation/test/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.png'

test_df['image_path'] = test_df['image_id'].apply(get_test_image_path)

# Verify a few test paths
print('\nVerifying test image paths (first 3):')
for i in range(min(3, len(test_df))):
    path = test_df.iloc[i]['image_path']
    exists = os.path.exists(path)
    print(f'{path}: {exists}')

# AUTOREGRESSIVE PREDICTION FUNCTION
def predict_inchi_autoregressive(model, image, max_length=275):
    """
    Generate InChI string autoregressively (one token at a time)
    [SLOW - Use for single predictions only]
    """
    # Start with <START> token
    decoder_input = np.zeros((1, max_length), dtype=np.int32)
    decoder_input[0, 0] = char_to_idx['<START>']
    
    # Expand image dimensions
    img_batch = np.expand_dims(image, axis=0)
    
    # Generate tokens one by one
    for i in range(1, max_length):
        # Predict next token
        predictions = model.predict([img_batch, decoder_input], verbose=0)
        
        # Get the token at position i-1 (we're predicting position i)
        next_token_probs = predictions[0, i-1, :]
        next_token = np.argmax(next_token_probs)
        
        # If we predict <END> or <PAD>, stop
        if next_token == char_to_idx['<END>'] or next_token == char_to_idx['<PAD>']:
            break
        
        # Add predicted token to decoder input for next iteration
        decoder_input[0, i] = next_token
    
    # Decode the sequence
    return decode_inchi(decoder_input[0])


def predict_inchi_batch_fast(model, images, max_length=275):
    """
    OPTIMIZED: Batch prediction for multiple images
    
    Speed improvements:
    - Processes multiple images simultaneously
    - Reduces model.predict() calls from N*max_length to max_length
    - 5-10x faster than sequential prediction
    
    Args:
        model: Trained Keras model
        images: List or array of preprocessed images
        max_length: Maximum sequence length
    
    Returns:
        List of decoded InChI strings
    """
    batch_size = len(images)
    
    # Initialize decoder inputs for entire batch
    decoder_inputs = np.zeros((batch_size, max_length), dtype=np.int32)
    decoder_inputs[:, 0] = char_to_idx['<START>']
    
    # Stack images into batch
    img_batch = np.array(images)
    
    # Track which sequences are still generating (not ended)
    active_seqs = np.ones(batch_size, dtype=bool)
    
    # Generate tokens autoregressively for entire batch
    for i in range(1, max_length):
        # Early exit if all sequences have ended
        if not np.any(active_seqs):
            break
        
        # Predict next tokens for ALL images in batch simultaneously
        predictions = model.predict([img_batch, decoder_inputs], verbose=0)
        
        # Get next token for each sequence (argmax over vocabulary)
        next_tokens = np.argmax(predictions[:, i-1, :], axis=-1)
        
        # Update each sequence
        for j in range(batch_size):
            if active_seqs[j]:
                # Check if this sequence should end
                if (next_tokens[j] == char_to_idx['<END>'] or 
                    next_tokens[j] == char_to_idx['<PAD>']):
                    active_seqs[j] = False
                else:
                    decoder_inputs[j, i] = next_tokens[j]
    
    # Decode all sequences
    decoded_results = []
    for j in range(batch_size):
        decoded_results.append(decode_inchi(decoder_inputs[j]))
    
    return decoded_results

# Make predictions on test set
print('\nGenerating predictions on test set with BATCHED autoregressive decoding...')

# Step 1: Load all test images first
print('Loading test images...')
test_images = []
valid_indices = []
failed_indices = []

for idx in tqdm(range(len(test_df)), desc="Loading images"):
    image_path = test_df.iloc[idx]['image_path']
    img = preprocess_image(image_path)
    
    if img is not None:
        test_images.append(img)
        valid_indices.append(idx)
    else:
        failed_indices.append(idx)

print(f'Loaded {len(test_images)} images successfully, {len(failed_indices)} failed')

# Step 2: Predict in batches (MUCH faster!)
PREDICTION_BATCH_SIZE = 64  
print(f'\nPredicting in batches of {PREDICTION_BATCH_SIZE}...')

predictions = []
num_batches = int(np.ceil(len(test_images) / PREDICTION_BATCH_SIZE))

for batch_idx in tqdm(range(num_batches), desc="Predicting batches"):
    start_idx = batch_idx * PREDICTION_BATCH_SIZE
    end_idx = min(start_idx + PREDICTION_BATCH_SIZE, len(test_images))
    
    batch_images = test_images[start_idx:end_idx]
    
    # Predict entire batch at once
    batch_predictions = predict_inchi_batch_fast(
        best_model, 
        batch_images, 
        MAX_INCHI_LENGTH
    )
    
    predictions.extend(batch_predictions)

# Debug first few predictions
print('\nFirst 5 predictions:')
for i in range(min(5, len(predictions))):
    pred = predictions[i]
    print(f'  {i}: {pred[:100]}{"..." if len(pred) > 100 else ""}')
    if len(pred) == 0:
        print(f'    WARNING: Empty prediction!')

# Step 3: Handle failed images and create full prediction list
full_predictions = []
valid_idx_set = set(valid_indices)

prediction_pointer = 0
for idx in range(len(test_df)):
    if idx in valid_idx_set:
        pred = predictions[prediction_pointer]
        # Fallback for empty predictions
        if len(pred) == 0:
            pred = 'InChI=1S/C'
        full_predictions.append(pred)
        prediction_pointer += 1
    else:
        # Use fallback for failed images
        full_predictions.append('InChI=1S/C')

predictions = full_predictions

# Create submission dataframe
submission = pd.DataFrame({
    'image_id': test_df['image_id'],
    'InChI': predictions
})

print(f'\nSubmission shape: {submission.shape}')
print(submission.head(10))

# Save submission file
submission.to_csv('submission.csv', index=False)
print('\nSubmission file saved: submission.csv')




# pred_indices = np.argmax(pred[0], axis=0)
# decode_inchi(pred_indices)




