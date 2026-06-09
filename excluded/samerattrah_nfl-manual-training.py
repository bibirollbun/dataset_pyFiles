import tensorflow as tf
import os

# Detect hardware
try:
    # Check for TPU
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print('Running on TPU ', tpu.master())
except ValueError:
    # Check for GPU(s)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f'Running on {len(gpus)} GPUs')
    else:
        strategy = tf.distribute.get_strategy()
        print('Running on single GPU or CPU')

print("Number of accelerators: ", strategy.num_replicas_in_sync)

# Configure for Kaggle
if os.path.exists('/kaggle'):
    print("Running in Kaggle environment")



!pip install polars
!pip install keras-tuner

import polars as pl
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence


class UnsupervisedNFLDataLoader:
    """Loads NFL data for unsupervised learning (no trajectory labels needed).
    
    This loader processes ALL player sequences (player_to_predict=True and False)
    to maximize the amount of training data for representation learning.
    """
    
    def __init__(self):
        self.input_sequences = None
        
    def load_files(self, directories, include_labeled=True, include_unlabeled=True, normalize=False):
        """Load input files from specified directories.
        
        Args:
            directories (list): List of directory paths to load from
            include_labeled (bool): Include player_to_predict=True sequences
            include_unlabeled (bool): Include player_to_predict=False sequences
            normalize (bool): Whether to normalize features (Z-score)
        """
        input_dfs = []
        
        print(f"Loading unsupervised data from {len(directories)} directories...")
        print(f"Include labeled: {include_labeled}, Include unlabeled: {include_unlabeled}")
        
        for d in directories:
            if not os.path.exists(d):
                print(f"Warning: Directory not found: {d}")
                continue
                
            input_files = sorted([f for f in os.listdir(d) if f.startswith('input') and f.endswith('.csv')])
            print(f"  Found {len(input_files)} input files in {d}")
            
            for f in input_files:
                try:
                    df = pl.read_csv(os.path.join(d, f), infer_schema_length=10000)
                    
                    initial_rows = len(df)
                    
                    # Filter based on player_to_predict flag
                    if "player_to_predict" in df.columns:
                        if include_labeled and not include_unlabeled:
                            # Only labeled
                            if df["player_to_predict"].dtype == pl.Boolean:
                                df = df.filter(pl.col("player_to_predict") == True)
                            else:
                                df = df.filter(pl.col("player_to_predict").cast(pl.Utf8).str.to_lowercase() == "true")
                        elif include_unlabeled and not include_labeled:
                            # Only unlabeled
                            if df["player_to_predict"].dtype == pl.Boolean:
                                df = df.filter(pl.col("player_to_predict") == False)
                            else:
                                df = df.filter(pl.col("player_to_predict").cast(pl.Utf8).str.to_lowercase() == "false")
                        # If both True, include all (no filtering)
                    
                    if len(df) > 0:
                        input_dfs.append(df)
                        print(f"    {f}: {initial_rows} -> {len(df)} rows")
                        
                except Exception as e:
                    print(f"Error loading {f}: {e}")
        
        if not input_dfs:
            print("No data found.")
            self.input_sequences = pl.DataFrame()
            return
        
        # Concatenate all dataframes
        print("Concatenating dataframes...")
        full_input = pl.concat(input_dfs, how="vertical_relaxed")
        
        # Deduplicate
        full_input = full_input.unique(subset=["game_id", "play_id", "nfl_id", "frame_id"])
        
        # Process features
        print("Processing features...")
        id_cols = ["game_id", "play_id", "nfl_id", "frame_id", "player_to_predict", "time"]
        feature_cols = [c for c in full_input.columns if c not in id_cols]
        
        expressions = []
        for col in feature_cols:
            if full_input[col].dtype == pl.Utf8:
                expr = (
                    pl.when(pl.col(col).str.to_lowercase() == "true").then(1.0)
                    .when(pl.col(col).str.to_lowercase() == "false").then(0.0)
                    .when(pl.col(col).str.to_lowercase() == "left").then(0.0)
                    .when(pl.col(col).str.to_lowercase() == "right").then(1.0)
                    .when(pl.col(col).str.to_lowercase() == "defense").then(0.0)
                    .when(pl.col(col).str.to_lowercase() == "offense").then(1.0)
                    .otherwise(
                        pl.col(col).cast(pl.Float64, strict=False).fill_null(
                            pl.col(col).hash() % 10000
                        )
                    ).cast(pl.Float64).alias(col)
                )
                expressions.append(expr)
            else:
                expressions.append(pl.col(col).cast(pl.Float64).alias(col))
        
        full_input = full_input.with_columns(expressions)
        
        # Sort by frame_id
        if "frame_id" in full_input.columns:
            full_input = full_input.sort(["game_id", "play_id", "nfl_id", "frame_id"])
            
        # Normalize features if requested
        if normalize:
            print("Normalizing features (Z-score)...")
            norm_exprs = []
            for col in feature_cols:
                # Check for constant columns to avoid division by zero
                n_unique = full_input.select(pl.col(col).n_unique()).item()
                if n_unique > 1:
                    norm_exprs.append(
                        ((pl.col(col) - pl.col(col).mean()) / (pl.col(col).std() + 1e-8)).alias(col)
                    )
            
            if norm_exprs:
                full_input = full_input.with_columns(norm_exprs)
        
        # Group into sequences
        agg_exprs = [pl.col(c) for c in feature_cols]
        self.input_sequences = full_input.group_by(
            ["game_id", "play_id", "nfl_id"], 
            maintain_order=True
        ).agg(agg_exprs)
        
        print(f"Total sequences: {len(self.input_sequences)}")
        
        # Debug sequence lengths
        if len(self.input_sequences) > 0:
            lengths = self.input_sequences.select(pl.col(feature_cols[0]).list.len()).to_series()
            print(f"Sequence length stats: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.2f}")

        
    def get_sequences(self):
        """Convert sequences to numpy arrays.
        
        Returns:
            np.ndarray: Array of input sequences (object array)
        """
        if self.input_sequences is None or self.input_sequences.is_empty():
            return np.array([])
        
        print("Converting to NumPy arrays...")
        
        # Get feature columns (exclude keys)
        input_cols = [c for c in self.input_sequences.columns 
                     if c not in ["game_id", "play_id", "nfl_id"]]
        
        # Convert to sequences
        input_col_indices = [self.input_sequences.columns.index(c) for c in input_cols]
        rows = self.input_sequences.iter_rows()
        
        X_list = []
        for row in rows:
            feature_seqs = [row[i] for i in input_col_indices]
            # Convert to numpy array to ensure consistent shape (T, F)
            X_seq = np.array(list(zip(*feature_seqs)), dtype=np.float32)
            if len(X_seq) > 0:
                X_list.append(X_seq)
        
        X = np.array(X_list, dtype=object)
        print(f"Loaded {len(X)} sequences")
        
        return X


class UnsupervisedNFLSequence(Sequence):
    """Keras Sequence for unsupervised learning on NFL data.
    
    For autoencoder: input and output are the same (reconstruction)
    For next-step prediction: input is sequence[:-n], output is sequence[n:]
    """
    
    def __init__(self, X, batch_size=32, maxlen=None, shuffle=True, 
                 task='autoencoder', prediction_steps=1):
        """Initialize the sequence.
        
        Args:
            X: Input sequences
            batch_size: Batch size
            maxlen: Maximum sequence length (auto-detect if None)
            shuffle: Whether to shuffle
            task: 'autoencoder' or 'next_step'
            prediction_steps: For next_step, how many steps ahead to predict
        """
        self.X = X
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.task = task
        self.prediction_steps = prediction_steps
        self.indices = np.arange(len(self.X))
        
        # Determine max length
        if maxlen is None:
            self.maxlen = max(len(seq) for seq in X)
        else:
            self.maxlen = maxlen
        
        print(f"UnsupervisedNFLSequence initialized:")
        print(f"  Samples: {len(self.X)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Max length: {self.maxlen}")
        print(f"  Task: {task}")
        
        if self.task == 'next_step':
            if self.maxlen <= self.prediction_steps:
                raise ValueError(
                    f"Invalid configuration: maxlen ({self.maxlen}) must be greater than "
                    f"prediction_steps ({self.prediction_steps}) for 'next_step' task.\n"
                    f"Please increase maxlen or decrease prediction_steps."
                )
            
            # Ensure we don't produce 0-length sequences even if maxlen is technically larger but close
            effective_len = self.maxlen - self.prediction_steps
            if effective_len < 1:
                 raise ValueError(f"Effective sequence length (maxlen - prediction_steps) is {effective_len}, must be >= 1.")

        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        # Filter out empty sequences to prevent pad_sequences errors
        batch_X = [self.X[i] for i in batch_indices if len(self.X[i]) > 0]
        
        if not batch_X:
            # Return dummy batch with correct shape to avoid crashing the training loop
            # Shape: (0, maxlen, features)
            if len(self.X) > 0 and len(self.X[0]) > 0:
                 feat_dim = self.X[0].shape[1]
                 return np.zeros((0, self.maxlen, feat_dim)), np.zeros((0, self.maxlen, feat_dim))
            return np.array([]), np.array([])
        
        if self.task == 'autoencoder':
            # Input and output are the same (reconstruction task)
            X_padded = pad_sequences(
                batch_X,
                maxlen=self.maxlen,
                dtype='float32',
                padding='post',
                truncating='post',
                value=0.0
            )
            return X_padded, X_padded
            
        elif self.task == 'next_step':
            # Input: sequence up to -prediction_steps
            # Output: last prediction_steps frames
            batch_X_input = []
            batch_y_output = []
            
            for seq in batch_X:
                if len(seq) > self.prediction_steps:
                    batch_X_input.append(seq[:-self.prediction_steps])
                    batch_y_output.append(seq[-self.prediction_steps:])
                else:
                    # If sequence too short, use full sequence for both
                    batch_X_input.append(seq)
                    batch_y_output.append(seq)
            
            X_padded = pad_sequences(
                batch_X_input,
                maxlen=self.maxlen - self.prediction_steps,
                dtype='float32',
                padding='post',
                truncating='post',
                value=0.0
            )
            
            y_padded = pad_sequences(
                batch_y_output,
                maxlen=self.prediction_steps,
                dtype='float32',
                padding='post',
                truncating='post',
                value=0.0
            )
            
            return X_padded, y_padded
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


if __name__ == "__main__":
    # Test the loader
    PREDICTION_TRAIN_DIR = '/kaggle/input/nfl-big-data-bowl-2026-prediction/train'
    
    print("=== Testing Unsupervised Data Loader ===\n")
    
    # Test 1: Load only unlabeled data
    print("Test 1: Loading UNLABELED data only")
    loader = UnsupervisedNFLDataLoader()
    loader.load_files([PREDICTION_TRAIN_DIR], include_labeled=False, include_unlabeled=True)
    X_unlabeled = loader.get_sequences()
    print(f"Unlabeled sequences: {len(X_unlabeled)}\n")
    
    # Test 2: Load ALL data
    print("Test 2: Loading ALL data (labeled + unlabeled)")
    loader_all = UnsupervisedNFLDataLoader()
    loader_all.load_files([PREDICTION_TRAIN_DIR], include_labeled=True, include_unlabeled=True)
    X_all = loader_all.get_sequences()
    print(f"Total sequences: {len(X_all)}\n")
    
    if len(X_all) > 0:
        print(f"Sample sequence length: {len(X_all[0])}")
        print(f"Sample features per timestep: {len(X_all[0][0])}")
        
        # Test sequence generators
        print("\n=== Testing Sequence Generators ===")
        
        print("\nAutoencoder sequence:")
        ae_seq = UnsupervisedNFLSequence(X_all[:1000], batch_size=32, task='autoencoder')
        x_batch, y_batch = ae_seq[0]
        print(f"Input shape: {x_batch.shape}")
        print(f"Output shape: {y_batch.shape}")
        print(f"Are input and output same? {np.array_equal(x_batch, y_batch)}")
        
        print("\nNext-step prediction sequence:")
        ns_seq = UnsupervisedNFLSequence(X_all[:1000], batch_size=32, task='next_step', prediction_steps=5)
        x_batch, y_batch = ns_seq[0]
        print(f"Input shape: {x_batch.shape}")
        print(f"Output shape: {y_batch.shape}")

    # Test 3: Load with normalization
    print("\nTest 3: Loading with NORMALIZATION")
    loader_norm = UnsupervisedNFLDataLoader()
    loader_norm.load_files([PREDICTION_TRAIN_DIR], include_labeled=True, include_unlabeled=True, normalize=True)
    X_norm = loader_norm.get_sequences()
    
    if len(X_norm) > 0:
        print(f"Sample normalized sequence (first feature of first step): {X_norm[0][0][0]}")



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


class LSTMAutoencoder:
    """LSTM Autoencoder for unsupervised representation learning on NFL sequences.
    
    The encoder learns to compress player movement sequences into a latent representation,
    and the decoder reconstructs the original sequence. The encoder can then be used
    to initialize supervised models.
    """
    
    def __init__(self, input_shape, latent_dim=128, lstm_units=[512, 256, 128, 64, 32]):
        """Initialize the LSTM Autoencoder.
        
        Args:
            input_shape: Shape of input (timesteps, features)
            latent_dim: Dimension of latent representation
            lstm_units: List of LSTM units for encoder layers
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        
    def build_encoder(self):
        """Build the encoder network."""
        inputs = layers.Input(shape=self.input_shape, name='encoder_input')
        
        x = inputs
        # Stack LSTM layers
        for i, units in enumerate(self.lstm_units[:-1]):
            x = layers.LSTM(
                units, 
                return_sequences=True,
                name=f'encoder_lstm_{i+1}'
            )(x)
            x = layers.Dropout(0.2)(x)
        
        # Last LSTM layer doesn't return sequences
        x = layers.LSTM(
            self.lstm_units[-1],
            return_sequences=False,
            name=f'encoder_lstm_{len(self.lstm_units)}'
        )(x)
        x = layers.Dropout(0.2)(x)
        
        # Latent representation
        latent = layers.Dense(self.latent_dim, activation='relu', name='latent')(x)
        
        self.encoder = Model(inputs, latent, name='encoder')
        return self.encoder
    
    def build_decoder(self):
        """Build the decoder network."""
        # Decoder input is the latent vector
        latent_inputs = layers.Input(shape=(self.latent_dim,), name='decoder_input')
        
        # Repeat the latent vector for each timestep
        x = layers.RepeatVector(self.input_shape[0])(latent_inputs)
        
        # Stack LSTM layers in reverse
        for i, units in enumerate(reversed(self.lstm_units)):
            x = layers.LSTM(
                units,
                return_sequences=True,
                name=f'decoder_lstm_{i+1}'
            )(x)
            x = layers.Dropout(0.2)(x)
        
        # Output layer to reconstruct features
        outputs = layers.TimeDistributed(
            layers.Dense(self.input_shape[1], activation='linear'),
            name='reconstruction'
        )(x)
        
        self.decoder = Model(latent_inputs, outputs, name='decoder')
        return self.decoder
    
    def build_autoencoder(self):
        """Build the complete autoencoder."""
        if self.encoder is None:
            self.build_encoder()
        if self.decoder is None:
            self.build_decoder()
        
        # Connect encoder and decoder
        inputs = layers.Input(shape=self.input_shape, name='autoencoder_input')
        latent = self.encoder(inputs)
        outputs = self.decoder(latent)
        
        self.autoencoder = Model(inputs, outputs, name='autoencoder')
        return self.autoencoder
    
    def compile(self, learning_rate=0.001):
        """Compile the autoencoder."""
        if self.autoencoder is None:
            self.build_autoencoder()
        
        self.autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
    def get_summary(self):
        """Print model summaries."""
        if self.autoencoder:
            print("\n=== Autoencoder Summary ===")
            self.autoencoder.summary()
        if self.encoder:
            print("\n=== Encoder Summary ===")
            self.encoder.summary()
        if self.decoder:
            print("\n=== Decoder Summary ===")
            self.decoder.summary()


class NextStepPredictor:
    """LSTM model for self-supervised next-step prediction.
    
    Predicts future timesteps given past timesteps, which can be used
    as a pre-training task for the supervised trajectory prediction.
    """
    
    def __init__(self, input_shape, output_steps=5, lstm_units=[256, 128], output_features=None):
        """Initialize the next-step predictor.
        
        Args:
            input_shape: Shape of input (timesteps, features)
            output_steps: Number of future steps to predict
            lstm_units: List of LSTM units
            output_features: Number of output features (if None, same as input features)
        """
        self.input_shape = input_shape
        self.output_steps = output_steps
        self.lstm_units = lstm_units
        self.output_features = output_features or input_shape[1]
        self.model = None
        
    def build(self):
        """Build the next-step prediction model."""
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        x = inputs
        # Stack LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_seq = (i < len(self.lstm_units) - 1)
            x = layers.LSTM(
                units,
                return_sequences=return_seq,
                name=f'lstm_{i+1}'
            )(x)
            x = layers.Dropout(0.2)(x)
        
        # Prediction head
        # Expand to output_steps timesteps
        x = layers.RepeatVector(self.output_steps)(x)
        x = layers.LSTM(128, return_sequences=True, name='prediction_lstm')(x)
        
        # Output for each timestep
        outputs = layers.TimeDistributed(
            layers.Dense(self.output_features, activation='linear'),
            name='predictions'
        )(x)
        
        self.model = Model(inputs, outputs, name='next_step_predictor')
        return self.model
    
    def compile(self, learning_rate=0.001):
        """Compile the model."""
        if self.model is None:
            self.build()
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='mse',
            metrics=['mae']
        )
    
    def get_summary(self):
        """Print model summary."""
        if self.model:
            self.model.summary()


def create_training_callbacks(model_path, patience=10):
    """Create standard callbacks for training.
    
    Args:
        model_path: Path to save best model
        patience: Patience for early stopping
        
    Returns:
        List of callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    return callbacks


def transfer_encoder_weights(pretrained_encoder, supervised_model, freeze_encoder=False):
    """Transfer weights from pretrained encoder to supervised model.
    
    Args:
        pretrained_encoder: The pretrained encoder model
        supervised_model: The supervised model to transfer weights to
        freeze_encoder: Whether to freeze the transferred weights
        
    Returns:
        The supervised model with transferred weights
    """
    print("\n=== Transferring Encoder Weights ===")
    
    # Get encoder layers from pretrained model
    encoder_layer_names = [layer.name for layer in pretrained_encoder.layers]
    
    # Transfer weights to matching layers in supervised model
    transferred_count = 0
    for layer in supervised_model.layers:
        if layer.name in encoder_layer_names:
            try:
                pretrained_layer = pretrained_encoder.get_layer(layer.name)
                layer.set_weights(pretrained_layer.get_weights())
                
                if freeze_encoder:
                    layer.trainable = False
                
                transferred_count += 1
                print(f"Transferred weights for layer: {layer.name} (frozen={freeze_encoder})")
            except Exception as e:
                print(f"Could not transfer weights for {layer.name}: {e}")
    
    print(f"\nTransferred weights for {transferred_count} layers")
    return supervised_model


if __name__ == "__main__":
    print("=== Testing Unsupervised Models ===\n")
    
    # Test parameters
    timesteps = 28
    features = 18
    latent_dim = 64
    
    print("1. Testing LSTM Autoencoder")
    print("-" * 50)
    ae = LSTMAutoencoder(
        input_shape=(timesteps, features),
        latent_dim=latent_dim,
        lstm_units=[128, 64]
    )
    ae.build_autoencoder()
    ae.compile()
    ae.get_summary()
    
    print("\n2. Testing Next-Step Predictor")
    print("-" * 50)
    predictor = NextStepPredictor(
        input_shape=(timesteps, features),
        output_steps=5,
        lstm_units=[128, 64],
        output_features=features
    )
    predictor.build()
    predictor.compile()
    predictor.get_summary()
    
    # Test with dummy data
    print("\n3. Testing with dummy data")
    print("-" * 50)
    dummy_input = tf.random.normal((32, timesteps, features))
    
    print("Autoencoder forward pass:")
    ae_output = ae.autoencoder(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {ae_output.shape}")
    
    print("\nNext-step predictor forward pass:")
    ns_output = predictor.model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {ns_output.shape}")
    
    print("\nEncoder output (latent representation):")
    latent = ae.encoder(dummy_input)
    print(f"Latent shape: {latent.shape}")



"""
Unsupervised Pre-training Script for NFL Player Trajectory Prediction

This script performs unsupervised pre-training using LSTM autoencoders on all available
NFL player sequences (both labeled and unlabeled). The pretrained encoder can then be
used to initialize supervised models for better performance.

Usage:
    python unsupervised_pretraining.py --task autoencoder --epochs 50
    python unsupervised_pretraining.py --task next_step --epochs 50
"""

import argparse
import os
import sys
from datetime import datetime
from tensorflow.keras import layers, metrics, models, losses



# Add parent directory to path
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# from unsupervised_data_loader import UnsupervisedNFLDataLoader, UnsupervisedNFLSequence
# from unsupervised_models import (
#     LSTMAutoencoder, 
#     NextStepPredictor, 
#     create_training_callbacks
# )


def train_autoencoder(train_seq, val_seq, epochs=100, latent_dim=128, model_save_path='autoencoder.keras'):
    """Train LSTM autoencoder for representation learning.
    
    Args:
        train_seq: Training data sequence
        val_seq: Validation data sequence
        epochs: Number of training epochs
        latent_dim: Dimension of latent space
        model_save_path: Path to save the trained model
    """
    print("\n" + "="*70)
    print("TRAINING LSTM AUTOENCODER")
    print("="*70)
    
    # Get input shape from first batch
    x_sample, _ = train_seq[0]
    input_shape = (x_sample.shape[1], x_sample.shape[2])
    
    print(f"\nInput shape: {input_shape}")
    print(f"Latent dimension: {latent_dim}")
    
    # Build autoencoder
    # ae = LSTMAutoencoder(
    #     input_shape=input_shape,
    #     latent_dim=latent_dim,
    #     lstm_units=[512, 256, 128, 64, 32]
    # )
    # ae.build_autoencoder()
    # ae.compile(learning_rate=0.0001)
    
    # print("\n" + "-"*70)
    # ae.get_summary()
    
    # Create callbacks
    callbacks = create_training_callbacks(model_save_path, patience=10)
    
    # Train
    print("\n" + "-"*70)
    print("Starting training...")
    print("-"*70)
    
    pretrained_ae = keras.models.load_model('/kaggle/working/best_hyperband_unsupervised_model_second_run.keras')
    
    pretrained_ae.summary()

    cosine_decay = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.0005780335476190155,
    decay_steps=140000,
    alpha=1e-5,
    )

    pretrained_ae.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=cosine_decay, global_clipnorm=1.0),
        loss=losses.MeanSquaredError(),
        metrics=[metrics.MeanSquaredError(), metrics.MeanAbsoluteError()]
    )
    

    history = pretrained_ae.fit(
        train_seq,
        validation_data=val_seq,
        epochs=epochs,
        shuffle=False,
        initial_epoch=20,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*70)
    print("Training completed!")
    print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
    print(f"Model saved to: {model_save_path}")
    print("="*70)
    
    # Save encoder separately
    encoder_path = model_save_path.replace('.keras', '_encoder.keras')
    ae.encoder.save(encoder_path)
    print(f"Encoder saved to: {encoder_path}")
    
    return ae, history


def train_next_step_predictor(train_seq, val_seq, epochs=50, prediction_steps=5, 
                               model_save_path='next_step_predictor.keras'):
    """Train next-step predictor for self-supervised learning.
    
    Args:
        train_seq: Training data sequence
        val_seq: Validation data sequence
        epochs: Number of training epochs
        prediction_steps: Number of steps to predict ahead
        model_save_path: Path to save the trained model
    """
    print("\n" + "="*70)
    print("TRAINING NEXT-STEP PREDICTOR")
    print("="*70)
    
    # Get input shape from first batch
    x_sample, y_sample = train_seq[0]
    input_shape = (x_sample.shape[1], x_sample.shape[2])
    output_features = y_sample.shape[2]
    
    print(f"\nInput shape: {input_shape}")
    print(f"Output steps: {prediction_steps}")
    print(f"Output features: {output_features}")
    
    # Build model
    predictor = NextStepPredictor(
        input_shape=input_shape,
        output_steps=prediction_steps,
        lstm_units=[256, 128],
        output_features=output_features
    )
    predictor.build()
    predictor.compile(learning_rate=0.001)
    
    print("\n" + "-"*70)
    predictor.get_summary()
    
    # Create callbacks
    callbacks = create_training_callbacks(model_save_path, patience=10)
    
    # Train
    print("\n" + "-"*70)
    print("Starting training...")
    print("-"*70)
    
    history = predictor.model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*70)
    print("Training completed!")
    print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
    print(f"Model saved to: {model_save_path}")
    print("="*70)
    
    return predictor, history




def main():
    
    PREDICTION_TRAIN_DIR = '/kaggle/input/nfl-big-data-bowl-2026-prediction/train'
    ANALYTICS_TRAIN_DIR = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train'
    
    print("\n" + "="*70)
    print("UNSUPERVISED PRE-TRAINING FOR NFL PLAYER TRAJECTORY PREDICTION")
    print("="*70)
    # print(f"\nTask: {args.task}")
    # print(f"Epochs: {args.epochs}")
    # print(f"Batch size: {args.batch_size}")
    # print(f"Include labeled: {args.include_labeled}")
    # print(f"Include unlabeled: {args.include_unlabeled}")
    # print(f"Validation split: {args.val_split}")
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    loader = UnsupervisedNFLDataLoader()
    loader.load_files(
        [PREDICTION_TRAIN_DIR, ANALYTICS_TRAIN_DIR],
        include_labeled=True,
        include_unlabeled=True
    )
    X = loader.get_sequences()
    
    if len(X) == 0:
        print("ERROR: No data loaded!")
        return
    
    print(f"\nTotal sequences loaded: {len(X)}")
    print(f"Sample sequence length: {len(X[0])}")
    print(f"Sample features: {len(X[0][0])}")
    
    # Split into train/val
    from sklearn.model_selection import train_test_split
    
    X_train, X_val = train_test_split(
        X, 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"\nTraining sequences: {len(X_train)}")
    print(f"Validation sequences: {len(X_val)}")
    
    # Create data sequences based on task
    print("\n" + "="*70)
    print("CREATING DATA GENERATORS")
    print("="*70)
    
    train_seq = UnsupervisedNFLSequence(
        X_train,
        batch_size=64,
        maxlen=10,
        shuffle=False,
        task="autoencoder",
        prediction_steps=10
    )
    
    val_seq = UnsupervisedNFLSequence(
        X_val,
        batch_size=64,
        maxlen=10,
        shuffle=False,
        task="autoencoder",
        prediction_steps=10
    )
    
    # Generate timestamp for model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Train based on task
    model_path = os.path.join("/kaggle/working/", f'autoencoder_{timestamp}.keras')
    model, history = train_autoencoder(
        train_seq, 
        val_seq, 
        epochs=100,
        latent_dim=256,
        model_save_path=model_path
    )
    
    # model_path = os.path.join(args.output_dir, f'next_step_{timestamp}.keras')
    # model, history = train_next_step_predictor(
    #     train_seq,
    #     val_seq,
    #     epochs=args.epochs,
    #     prediction_steps=args.prediction_steps,
    #     model_save_path=model_path
    # )
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
    print(f"\nModel saved to: {model_path}")
    
    encoder_path = model_path.replace('.keras', '_encoder.keras')
    print(f"Encoder saved to: {encoder_path}")
    print("\nTo use the pretrained encoder in your supervised model:")
    print(f"  from tensorflow import keras")
    print(f"  from unsupervised_models import transfer_encoder_weights")
    print(f"  pretrained_encoder = keras.models.load_model('{encoder_path}')")
    print(f"  supervised_model = transfer_encoder_weights(pretrained_encoder, supervised_model)")

    print("="*70)
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    main()



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics, models, losses
import os
import sys

# Add the manual_data_processing directory to the path
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'manual_data_processing'))

# from csv_to_numpy import NFLDataLoader, create_tf_datasets

def build_seq2seq_model(input_seq_length, input_features, output_seq_length, output_features, lstm_units=256):
    """
    Builds a sequence-to-sequence model with LSTM layers.

    Args:
        input_seq_length (int): The length of input sequences (time steps).
        input_features (int): The number of input features per timestep.
        output_seq_length (int): The length of output sequences (time steps).
        output_features (int): The number of output features per timestep.
        lstm_units (int): The number of units in the LSTM layers.

    Returns:
        keras.Model: The compiled Keras model.
    """
    dropout_rate = 0.2
    SEED = 42

    encoder_inputs = layers.Input(shape=(input_seq_length, input_features), name='encoder_inputs')
    
    enc_lstm_1 = layers.LSTM(lstm_units, return_sequences=True, name='enc_lstm_1')(encoder_inputs)
    
    enc_lstm_2 = layers.LSTM(lstm_units, return_sequences=True, name='enc_lstm_2')(enc_lstm_1)
    
    enc_lstm_3 = layers.LSTM(lstm_units, return_sequences=True, name='enc_lstm_3')(enc_lstm_2)
    
    enc_lstm_4 = layers.LSTM(lstm_units, return_sequences=True, name='enc_lstm_4')(enc_lstm_3)
    
    enc_lstm_5 = layers.LSTM(lstm_units, return_sequences=True, name='enc_lstm_5')(enc_lstm_4)
    
    latent = layers.LSTM(lstm_units, return_sequences=True, name='latent')(enc_lstm_5)

    outputs = layers.TimeDistributed(
        layers.Dense(2, activation='linear'),
        name='trajectory_output'
    )(latent)
    
    model = models.Model(inputs=encoder_inputs, outputs=outputs, name='encoder')

    cosine_decay = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.003054455019851368,
    decay_steps=11500,
    alpha=1e-5,
    )

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=cosine_decay, global_clipnorm=1.0),
        loss=losses.MeanSquaredError(),
        metrics=[metrics.MeanSquaredError(), metrics.MeanAbsoluteError()]
    )
    
    return model

def train_model(model, train_sequence, val_sequence, epochs=100, callbacks=None):
    """
    Trains the Keras model using Keras Sequence objects.
    
    Args:
        model: The Keras model to train
        train_sequence: Training data sequence (NFLDataSequence)
        val_sequence: Validation data sequence (NFLDataSequence)
        epochs (int): Number of training epochs
        callbacks: List of Keras callbacks
    
    Returns:
        history: Training history object
    """
    pretrained_encoder = keras.models.load_model('/kaggle/working/best_hyperband_unsupervised_model_fifth_run.keras')
    pretrained_encoder.summary()
    supervised_model = transfer_encoder_weights(pretrained_encoder, model)
    print("pretrained encoder")
    pretrained_encoder.summary()
    print("supervised model")
    supervised_model.summary()
    if callbacks is None:
        callbacks = []
    
    # Add early stopping and model checkpoint callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    callbacks.extend([early_stopping, model_checkpoint])
    
    print("Starting model training...")
    history = supervised_model.fit(
        train_sequence,
        epochs=epochs,
        initial_epoch=24,
        shuffle=False,
        validation_data=val_sequence,
        callbacks=model_checkpoint,
        verbose=1
    )
    print("Model training finished.")
    return history

def main():
    """
    Main function to load data, build, and train the model.
    """
    # Configuration
    train_dir = '/kaggle/input/nfl-big-data-bowl-2026-prediction/train'
    batch_size = 64
    epochs = 100
    test_size = 0.2
    
    print("="*60)
    print("NFL Big Data Bowl 2026 - Predictor Training")
    print("="*60)
    
    # Load and prepare data
    print("\n[1/4] Loading data from CSV files...")
    loader = NFLDataLoader(train_dir)
    X, y = loader.get_aligned_data()
    
    if len(X) == 0:
        print("Error: No data loaded. Please check the data directory.")
        return
    
    print(f"\nData Summary:")
    print(f"  Total sequences: {len(X)}")
    print(f"  Sample input sequence length: {len(X[0])}")
    print(f"  Sample output sequence length: {len(y[0])}")
    print(f"  Input features per timestep: {len(X[0][0]) if len(X[0]) > 0 else 0}")
    print(f"  Output features per timestep: {len(y[0][0]) if len(y[0]) > 0 else 0}")
    
    # Create Keras Sequences with padding
    print(f"\n[2/4] Creating training and validation sequences (test_size={test_size})...")
    train_seq, val_seq = create_tf_datasets(X, y, test_size=test_size, batch_size=batch_size)
    
    if train_seq is None:
        print("Error: Failed to create training sequences.")
        return
    
    # Get one batch to determine shapes
    x_sample, y_sample = train_seq[0]
    input_seq_length = x_sample.shape[1]
    input_features = x_sample.shape[2]
    output_seq_length = y_sample.shape[1]
    output_features = y_sample.shape[2]
    
    print(f"\nSequence Shapes:")
    print(f"  Input: (batch_size, {input_seq_length}, {input_features})")
    print(f"  Output: (batch_size, {output_seq_length}, {output_features})")
    
    # Build model
    print(f"\n[3/4] Building sequence-to-sequence model...")
    model = build_seq2seq_model(
        input_seq_length=input_seq_length,
        input_features=input_features,
        output_seq_length=output_seq_length,
        output_features=output_features,
        lstm_units=256
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    # Train model
    print(f"\n[4/4] Training model for {epochs} epochs...")
    history = train_model(model, train_seq, val_seq, epochs=epochs)
    
    # Save the final model
    final_model_path = 'nfl_predictor_final.keras'
    model.save(final_model_path)
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: best_model.keras")
    print(f"{'='*60}")
    
    # Print training summary
    print(f"\nTraining Summary:")
    print(f"  Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"  Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print(f"  Final training MAE: {history.history['mae'][-1]:.4f}")
    print(f"  Final validation MAE: {history.history['val_mae'][-1]:.4f}")
    print(f"  Best validation loss: {min(history.history['val_loss']):.4f}")

if __name__ == '__main__':
    main()



import csv
import numpy as np
import os

class NFLDataLoader:
    """
    Loads and processes NFL Big Data Bowl 2026 data from CSV files.
    Filters input data for 'player_to_predict' == True and aligns with output data.
    
    Selected Input Features: ['x', 'y', 's', 'a', 'dir', 'o']
    Selected Output Features: ['x', 'y']
    """
    def __init__(self, train_dir):
        self.train_dir = train_dir
        self.input_sequences = {}
        self.output_sequences = {}
        self.input_header = []
        self.output_header = []

    def process_value(self, val):
        """
        Converts a CSV string value into the appropriate type.
        """
        val_lower = val.lower()

        # Handle Booleans
        if val_lower == 'true':
            return 1.0
        if val_lower == 'false':
            return 0.0
        
        # Handle Direction (left/right)
        if val_lower == 'left':
            return 0.0
        if val_lower == 'right':
            return 1.0

        # Handle Player Side (defense/offense)
        if val_lower == 'defense':
            return 0.0
        if val_lower == 'offense':
            return 1.0
        
        # Handle Numbers (Integers and Floats)
        try:
            return float(val)
        except ValueError:
            pass
            
        # Handle Strings (Object type)
        # Use a deterministic hash for strings to ensure consistency
        import zlib
        return float(zlib.adler32(val.encode('utf-8')) % 10000)

    def load_input_files(self):
        """
        Loads and filters input CSV files.
        Filters for 'player_to_predict' == True.
        Selects all features matching the unsupervised loader format (18 features).
        """
        input_files = sorted([f for f in os.listdir(self.train_dir) if f.startswith('input') and f.endswith('.csv')])
        print(f"Loading and filtering {len(input_files)} Input files...")
        
        # Use the same features as unsupervised loader for consistency
        # These match the 18 features from the unsupervised dataloader
        id_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id', 'player_to_predict', 'time']

        for input_file in input_files:
            input_path = os.path.join(self.train_dir, input_file)
            with open(input_path, 'r') as f:
                reader = csv.reader(f)
                first_row = True
                
                # Indices for ID columns
                player_to_predict_idx = -1
                game_id_idx = -1
                play_id_idx = -1
                nfl_id_idx = -1
                
                # Indices for feature columns (all non-ID columns)
                feature_indices = []

                for row in reader:
                    if first_row:
                        if not self.input_header:
                            self.input_header = row
                        
                        try:
                            player_to_predict_idx = row.index('player_to_predict')
                            game_id_idx = row.index('game_id')
                            play_id_idx = row.index('play_id')
                            nfl_id_idx = row.index('nfl_id')
                            
                            # Get all feature columns (exclude ID columns)
                            feature_indices = [i for i, col in enumerate(row) if col not in id_cols]
                            
                        except ValueError as e:
                            print(f"Error finding columns in {input_file}: {e}")
                            break
                        
                        first_row = False
                        continue
                    
                    # Filter: Only keep rows where player_to_predict is True
                    if player_to_predict_idx != -1:
                        val = row[player_to_predict_idx].lower()
                        if val != 'true':
                            continue 

                    # Extract Key
                    key = (row[game_id_idx], row[play_id_idx], row[nfl_id_idx])
                    
                    if key not in self.input_sequences:
                        self.input_sequences[key] = []
                    
                    # Append all feature columns (should be 18 features like unsupervised)
                    self.input_sequences[key].append([self.process_value(row[i]) for i in feature_indices])

    def load_output_files(self):
        """
        Loads output CSV files.
        Selects specific features: ['x', 'y'].
        """
        output_files = sorted([f for f in os.listdir(self.train_dir) if f.startswith('output') and f.endswith('.csv')])
        print(f"Loading {len(output_files)} Output files...")
        
        features_to_keep = ['x', 'y']

        for output_file in output_files:
            output_path = os.path.join(self.train_dir, output_file)
            with open(output_path, 'r') as f:
                reader = csv.reader(f)
                first_row = True
                
                # Indices for ID columns
                game_id_idx = -1
                play_id_idx = -1
                nfl_id_idx = -1
                
                # Indices for feature columns
                feature_indices = []

                for row in reader:
                    if first_row:
                        if not self.output_header:
                            self.output_header = row
                        
                        try:
                            game_id_idx = row.index('game_id')
                            play_id_idx = row.index('play_id')
                            nfl_id_idx = row.index('nfl_id')
                            
                            # Find indices for the features we want to keep
                            feature_indices = [row.index(feat) for feat in features_to_keep]
                            
                        except ValueError as e:
                            print(f"Error finding columns in {output_file}: {e}")
                            break

                        first_row = False
                        continue
                    
                    # Extract Key
                    key = (row[game_id_idx], row[play_id_idx], row[nfl_id_idx])
                    
                    if key not in self.output_sequences:
                        self.output_sequences[key] = []
                    
                    # Append only the selected features (x, y)
                    self.output_sequences[key].append([float(row[i]) for i in feature_indices])

    def normalize_sequences(self, sequences):
        """
        Normalizes the sequences (Z-score) feature-wise.
        """
        print("Normalizing features...")
        # Flatten to compute stats
        all_values = []
        for seq in sequences:
            all_values.extend(seq)
        
        if not all_values:
            return sequences, {}

        data_np = np.array(all_values, dtype=np.float32)
        mean = np.mean(data_np, axis=0)
        std = np.std(data_np, axis=0)
        
        # Avoid division by zero
        std[std == 0] = 1.0
        
        normalized_sequences = []
        for seq in sequences:
            seq_np = np.array(seq, dtype=np.float32)
            norm_seq = (seq_np - mean) / std
            normalized_sequences.append(norm_seq.tolist())
            
        print(f"Normalization complete. Mean shape: {mean.shape}, Std shape: {std.shape}")
        return normalized_sequences, {'mean': mean, 'std': std}

    def get_aligned_data(self, normalize=False):
        """
        Aligns input and output sequences and returns NumPy arrays.
        Returns:
            X (np.ndarray): Input sequences with features ['x', 'y', 's', 'a', 'dir', 'o']
            y (np.ndarray): Output sequences with features ['x', 'y']
        """
        self.load_input_files()
        self.load_output_files()

        print("Aligning Input and Output sequences...")
        common_keys = sorted(list(set(self.input_sequences.keys()).intersection(set(self.output_sequences.keys()))))

        aligned_X = []
        aligned_y = []

        for key in common_keys:
            aligned_X.append(self.input_sequences[key])
            aligned_y.append(self.output_sequences[key])

        if normalize and aligned_X:
            aligned_X, self.stats = self.normalize_sequences(aligned_X)

        print(f"Processing complete.")
        print(f"Total Unique Sequences (Matches): {len(common_keys)}")

        if not aligned_X:
            print("No matching data found.")
            return np.array([]), np.array([])

        # Convert to NumPy arrays
        # Using dtype=object to handle potential variable lengths or mixed types safely
        try:
            X = np.array(aligned_X, dtype=object)
            print(f"Initial X shape: {X.shape}")
        except Exception as e:
            print(f"Error creating X array: {e}")
            X = np.array([])

        try:
            y = np.array(aligned_y, dtype=object)
            print(f"Initial y shape: {y.shape}")
        except Exception as e:
            print(f"Error creating y array: {e}")
            y = np.array([])
            
        return X, y

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence

class NFLDataSequence(Sequence):
    """
    Keras Sequence for NFL data with automatic padding of variable-length sequences.
    """
    def __init__(self, X, y, batch_size=64, maxlen_x=10, maxlen_y=10, shuffle=False):
        """
        Args:
            X (list): List of input sequences (each sequence is a list of time steps)
            y (list): List of output sequences (each sequence is a list of time steps)
            batch_size (int): Batch size
            maxlen_x (int, optional): Maximum length for input sequences. If None, uses max length in data.
            maxlen_y (int, optional): Maximum length for output sequences. If None, uses max length in data.
            shuffle (bool): Whether to shuffle data at the end of each epoch
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        
        # Determine max lengths if not provided
        if maxlen_x is None:
            self.maxlen_x = max(len(seq) for seq in X)
        else:
            self.maxlen_x = maxlen_x
            
        if maxlen_y is None:
            self.maxlen_y = max(len(seq) for seq in y)
        else:
            self.maxlen_y = maxlen_y
        
        print(f"NFLDataSequence initialized: {len(self.X)} samples, batch_size={batch_size}")
        print(f"Max sequence lengths - X: {self.maxlen_x}, y: {self.maxlen_y}")
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        """
        Generate one batch of data
        """
        # Get batch indices
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Get batch data
        batch_X = [self.X[i] for i in batch_indices]
        batch_y = [self.y[i] for i in batch_indices]
        
        # Process X sequences: handle mixed types
        # The data from process_value() should already have numeric types as floats
        # and strings as strings. We need to filter out or encode string columns.
        batch_X_numeric = []
        for seq in batch_X:
            seq_numeric = []
            for frame in seq:
                frame_numeric = []
                for item in frame:
                    # If item is already a float or int (from process_value), keep it
                    if isinstance(item, (int, float)):
                        frame_numeric.append(float(item))
                    # If it's a string, we need to handle it
                    # For now, let's use a hash or skip it
                    # Better approach: filter these columns out or use proper encoding
                    elif isinstance(item, str):
                        # Try to convert to float, if fails, use hash or 0
                        try:
                            frame_numeric.append(float(item))
                        except ValueError:
                            # For non-numeric strings, use a simple hash-based encoding
                            # This is a simple placeholder - ideally use proper categorical encoding
                            frame_numeric.append(float(hash(item) % 10000))
                    else:
                        frame_numeric.append(0.0)
                seq_numeric.append(frame_numeric)
            batch_X_numeric.append(seq_numeric)
        
        # Use pad_sequences for both X and y
        # pad_sequences expects sequences of shape (n_samples, n_timesteps) for 2D
        # For 3D (n_samples, n_timesteps, n_features), we need to pad manually or use padding='post'
        
        # Method: Pad each sequence to maxlen, filling with zeros
        X_padded = pad_sequences(
            batch_X_numeric, 
            maxlen=self.maxlen_x, 
            dtype='float32',
            padding='post',
            truncating='post',
            value=0.0
        )
        
        y_padded = pad_sequences(
            batch_y,
            maxlen=self.maxlen_y,
            dtype='float32',
            padding='post',
            truncating='post',
            value=0.0
        )
        
        return X_padded, y_padded
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)


def create_tf_datasets(X, y, test_size=0.2, batch_size=64, maxlen_x=10, maxlen_y=10):
    """
    Splits X and y into training and validation sets and creates Keras Sequence datasets.
    Uses keras.utils.Sequence with padding to handle variable-length sequences.
    
    Args:
        X (np.ndarray): Input data (object array of variable-length sequences).
        y (np.ndarray): Output data (object array of variable-length sequences).
        test_size (float): Proportion of the dataset to include in the validation split.
        batch_size (int): Batch size for the datasets.
        maxlen_x (int, optional): Maximum length for input sequences. If None, auto-detects.
        maxlen_y (int, optional): Maximum length for output sequences. If None, auto-detects.
        
    Returns:
        train_sequence (NFLDataSequence): Training data sequence.
        val_sequence (NFLDataSequence): Validation data sequence.
    """
    print("\n--- Creating Keras Sequence Datasets with Padding ---")
    
    try:
        # Convert object arrays to lists
        X_list = X.tolist()
        y_list = y.tolist()
        
        # Split into train and validation
        print(f"Splitting data (test_size={test_size})...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_list, y_list, 
            test_size=test_size, 
            random_state=42
        )
        
        print(f"Train size: {len(X_train)}")
        print(f"Val size: {len(X_val)}")
        
        # Create Sequence objects
        print("Creating Training Sequence...")
        train_sequence = NFLDataSequence(
            X_train, y_train, 
            batch_size=batch_size,
            maxlen_x=maxlen_x,
            maxlen_y=maxlen_y,
            shuffle=True
        )
        
        print("Creating Validation Sequence...")
        val_sequence = NFLDataSequence(
            X_val, y_val,
            batch_size=batch_size,
            maxlen_x=train_sequence.maxlen_x,  # Use same max lengths as training
            maxlen_y=train_sequence.maxlen_y,
            shuffle=False
        )
        
        print("Sequences created successfully.")
        print(f"Training batches per epoch: {len(train_sequence)}")
        print(f"Validation batches per epoch: {len(val_sequence)}")
        
        return train_sequence, val_sequence

    except Exception as e:
        print(f"Error creating Keras sequences: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    TRAIN_DIR = '/kaggle/input/nfl-big-data-bowl-2026-prediction/train'
    
    loader = NFLDataLoader(TRAIN_DIR)
    # Enable normalization
    X, y = loader.get_aligned_data(normalize=True)

    print("\n--- Final Data Shapes ---")
    print(f"X (Input) Shape: {X.shape}")
    print(f"y (Output) Shape: {y.shape}")

    if len(X) > 0:
        print(f"Sample Input Sequence Length: {len(X[0])}")
        print(f"Sample Output Sequence Length: {len(y[0])}")

    # Create Keras Sequences with padding
    train_seq, val_seq = create_tf_datasets(X, y, batch_size=32)
    
    if train_seq:
        print("\nVerifying Sequence Element:")
        # Get one batch to verify shapes
        x_batch, y_batch = train_seq[0]
        print(f"Batch X shape: {x_batch.shape}")
        print(f"Batch y shape: {y_batch.shape}")
        print(f"Max sequence lengths - X: {train_seq.maxlen_x}, y: {train_seq.maxlen_y}")

    print("\nData loading, alignment, and sequence creation complete.")



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import sys
import keras_tuner

# Add the manual_data_processing directory to the path
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'manual_data_processing'))

# from csv_to_numpy import NFLDataLoader, create_tf_datasets


def build_model(hp):
    """
    Builds a compiled Keras LSTM model with hyperparameters to be experimented on.

    This function defines the architecture of the LSTM model for sequence-to-sequence prediction.
    It incorporates hyperparameter search spaces for key model parameters like learning rate,
    number of LSTM units, kernel regularization, and activation functions.

    Args:
        hp (keras_tuner.HyperParameters): An instance of Keras Tuner's HyperParameters class,
                                          used to define the search space for hyperparameters.

    Returns:
        keras.Model: The compiled Keras LSTM model with hyperparameters set by Keras Tuner.
    """
    k_init = keras.initializers.RandomNormal(mean=1503.17, 
    stddev=2755.38, 
    seed=42)
    SEED = 42
    # Define hyperparameter search spaces for tuning
    learning_rate = hp.Float("lr", min_value=1e-7, max_value=1e-3, sampling="log")
    layer_u = hp.Int("lu", min_value=160, max_value=1024, step=8)
    kernel_r = hp.Float("kr", min_value=1e-10, max_value=1e-5, sampling="log")
    acti_f = hp.Choice("af", ["sigmoid", "hard_sigmoid", "tanh", "relu", "softmax", "linear"])
    weight_d = hp.Float("wd", min_value=1e-10, max_value=0.0009, sampling="log")

    # Define the model structure using Keras Sequential API
    model = keras.Sequential([
        # Input layer
        keras.layers.Input(shape=(input_seq_length, input_features)),
        
        # Encoder LSTM layers
        keras.layers.LSTM(
            units=layer_u,
            activation=acti_f,
            kernel_initializer=k_init,
            return_sequences=True,
            # kernel_regularizer=keras.regularizers.L2(l2=kernel_r),
            seed=SEED,
        ),
        keras.layers.LSTM(
            units=layer_u // 2,
            activation=acti_f,
            kernel_initializer=k_init,
            return_sequences=True,
            # kernel_regularizer=keras.regularizers.L2(l2=kernel_r),
            seed=SEED,
        ),
        # keras.layers.LSTM(
        #     units=layer_u // 2,
        #     activation=acti_f,
        #     kernel_initializer=k_init,
        #     return_sequences=True,
        #     # kernel_regularizer=keras.regularizers.L2(l2=kernel_r),
        #     seed=SEED,
        # ),
        # keras.layers.LSTM(
        #     units=layer_u // 2,
        #     activation=acti_f,
        #     kernel_initializer=k_init,
        #     return_sequences=True,
        #     # kernel_regularizer=keras.regularizers.L2(l2=kernel_r),
        #     seed=SEED,
        # ),
        # keras.layers.LSTM(
        #     units=layer_u // 2,
        #     activation=acti_f,
        #     kernel_initializer=k_init,
        #     return_sequences=True,
        #     # kernel_regularizer=keras.regularizers.L2(l2=kernel_r),
        #     seed=SEED,
        # ),
        # layers.RepeatVector(output_seq_length),
        keras.layers.LSTM(
            units=32,
            activation="sigmoid",
            kernel_initializer=k_init,
            return_sequences=True,
            # kernel_regularizer=keras.regularizers.L2(l2=0.00000195),
            seed=SEED,
        ),
        # Crop or slice to match output sequence length
        # layers.Lambda(lambda x: x[:, :output_seq_length, :]),
        # TimeDistributed dense layer for output features
        layers.TimeDistributed(
        keras.layers.Dense(units=output_features, activation="linear")
        ),
    ])

    # Compile the model with a tunable optimizer and metrics
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
            global_clipnorm=1,
            amsgrad=False,
            weight_decay=weight_d, # Tunable weight decay
        ),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )

    return model


def experimenting(training_dataset, validation_data):
    """
    Runs Keras Tuner experiments for the LSTM model using the RandomSearch algorithm.

    This function initializes a `RandomSearch` tuner with the `build_model` function,
    configures the search objective (minimizing validation loss), and then executes
    the hyperparameter search across the defined search spaces. It prints summaries
    of the search space and the results.

    Args:
        training_dataset: NFLDataSequence object for training data
        validation_data: NFLDataSequence object for validation data

    """

    hp = keras_tuner.HyperParameters()
    
    # Get a batch from the sequence to determine shapes
    x_batch, y_batch = training_dataset[0]
    global input_features, input_seq_length, output_seq_length, output_features
    input_seq_length = x_batch.shape[1]
    input_features = x_batch.shape[2]
    output_seq_length = y_batch.shape[1]
    output_features = y_batch.shape[2]
    
    print(f"\nDetected shapes:")
    print(f"  Input: ({input_seq_length}, {input_features})")
    print(f"  Output: ({output_seq_length}, {output_features})")
    
    build_model(hp) # Instantiate a dummy model to build the search space

    # Initialize Keras Tuner's RandomSearch algorithm
    tuner = keras_tuner.RandomSearch(
        hypermodel=build_model,
        max_trials=10, # Maximum number of hyperparameter combinations to try
        objective=keras_tuner.Objective("val_loss", "min"),   # Objective is to minimize validation loss
        executions_per_trial=1, # Number of models to train for each trial (1 for efficiency)
        overwrite=True, # Overwrite previous results in the directory
        directory=os.getenv("KERAS_TUNER_EXPERIMENTS_DIR", "./tuner_results"), # Directory to save experiment logs and checkpoints
        project_name="nfl_prediction", # Name of the Keras Tuner project
        seed = 42,
        max_consecutive_failed_trials=5,
    )

    tuner.search_space_summary() # Print a summary of the hyperparameter search space

    # NFLDataSequence is already batched, no need to call batch() again
    # Run the hyperparameter search experiments
    tuner.search(
        training_dataset, 
        validation_data=validation_data, 
        epochs=5
    )

    tuner.results_summary() # Print a summary of the best performing trials


if __name__ == "__main__":
    train_dir = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train'
    batch_size = 32
    epochs = 50
    test_size = 0.2
    
    print("="*60)
    print("NFL Big Data Bowl 2026 - Predictor Training")
    print("="*60)
    
    # Load and prepare data
    print("\n[1/4] Loading data from CSV files...")
    loader = NFLDataLoader(train_dir)
    X, y = loader.get_aligned_data()
    
    if len(X) == 0:
        print("Error: No data loaded. Please check the data directory.")
    
    print(f"\nData Summary:")
    print(f"  Total sequences: {len(X)}")
    print(f"  Sample input sequence length: {len(X[0])}")
    print(f"  Sample output sequence length: {len(y[0])}")
    print(f"  Input features per timestep: {len(X[0][0]) if len(X[0]) > 0 else 0}")
    print(f"  Output features per timestep: {len(y[0][0]) if len(y[0]) > 0 else 0}")
    
    # Create Keras Sequences with padding
    print(f"\n[2/4] Creating training and validation sequences (test_size={test_size})...")
    train_seq, val_seq = create_tf_datasets(X, y, test_size=test_size, batch_size=batch_size)
    
    # Run the hyperparameter experimentation
    experimenting(train_seq, val_seq)



import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, losses, metrics
import matplotlib.pyplot as plt

def build_seq2seq_model(hp, 
                        input_seq_length, input_features,
                        output_seq_length, output_features):
    """
    Returns a compiled Keras model.
    hp – HyperParameters object supplied by Keras‑Tuner.
    """
    # ---------- Hyper‑parameters ----------
    # Number of LSTM layers (encoder + decoder)
    n_encoder_layers = hp.Int('enc_layers', 2, 6, step=1)
    n_decoder_layers = hp.Int('dec_layers', 2, 6, step=1)

    # LSTM units per layer (same for all layers for simplicity)
    lstm_units = hp.Choice('lstm_units', [64, 128, 256, 384, 512])

    # Kernel regularizer
    # kernel_reg = hp.Float('kernel_reg', 1e-8, 1e-4, sampling='log')
    
    # Dropout rate
    dropout_rate = hp.Float('dropout', 0.0, 0.3, step=0.05)

    # Learning‑rate schedule
    init_lr = hp.Float('init_lr', 1e-5, 5e-3, sampling='log')
    
    # ---------- Model ----------
    # Encoder
    encoder_inputs = layers.Input(shape=(input_seq_length, input_features),
                                  name='encoder_inputs')
    x = encoder_inputs
    for i in range(n_encoder_layers):
        # Residual LSTM block
        lstm_out = layers.LSTM(lstm_units,
                               return_sequences=True,
                               seed=42,
                               dropout=dropout_rate,
                               # kernel_regularizer=kernel_reg,
                               name=f'enc_lstm_{i+1}')(x)
        # lstm_out = layers.Dropout(dropout_rate,
        #                           name=f'enc_dropout_{i+1}')(lstm_out)
        # # Add residual connection (if dimensions match)
        # if lstm_out.shape[-1] == x.shape[-1]:
        #     lstm_out = layers.Add(name=f'enc_res_{i+1}')([x, lstm_out])
        # # Normalise
        # lstm_out = layers.LayerNormalization(name=f'enc_norm_{i+1}')(lstm_out)
        x = lstm_out

    # Grab the final hidden state as the latent vector
    latent = layers.LSTM(lstm_units,
                         return_sequences=False,
                         seed=42,
                         name='latent')(x)

    # Decoder – repeat latent vector for each output timestep
    decoder_inputs = layers.RepeatVector(output_seq_length,
                                         name='repeat_latent')(latent)
    y = decoder_inputs
    for i in range(n_decoder_layers):
        lstm_out = layers.LSTM(lstm_units,
                               return_sequences=True,
                               seed=42,
                               dropout=dropout_rate,
                               # kernel_regularizer=kernel_reg,
                               name=f'dec_lstm_{i+1}')(y)
        # lstm_out = layers.Dropout(dropout_rate,
        #                           name=f'dec_dropout_{i+1}')(lstm_out)
        # # Residual connection (again only when shapes match)
        # if lstm_out.shape[-1] == y.shape[-1]:
        #     lstm_out = layers.Add(name=f'dec_res_{i+1}')([y, lstm_out])
        # lstm_out = layers.LayerNormalization(name=f'dec_norm_{i+1}')(lstm_out)
        y = lstm_out

    # Final TimeDistributed dense layer
    decoder_outputs = layers.TimeDistributed(
        layers.Dense(output_features, activation='linear'),
        name='decoder_output')(y)

    model = models.Model(inputs=encoder_inputs, outputs=decoder_outputs,
                         name='tunable_seq2seq')

    # ---------- Learning‑rate schedule ----------
    # Simplified to just CosineDecay to avoid TypeError
    total_steps = hp.Int('total_steps', 1, 100000, step=100)
    learning_rate = optimizers.schedules.CosineDecay(
        initial_learning_rate=init_lr,
        decay_steps=total_steps,
        alpha=1e-5)

    optimizer = optimizers.AdamW(learning_rate=learning_rate, global_clipnorm=1.0)

    model.compile(optimizer=optimizer,
                  loss=losses.MeanSquaredError(),
                  metrics=[metrics.MeanSquaredError(), metrics.MeanAbsoluteError()])
    return model



        
def tuner_search(train_seq, val_seq,
                 input_seq_len, input_feat,
                 output_seq_len, output_feat,
                 max_trials=30, epochs_per_trial=5):
    """
    Runs Hyperband and returns the best model + history.
    """
    # Define the hypermodel function
    def make_model(hp):
        return build_seq2seq_model(
            hp,
            input_seq_length=input_seq_len,
            input_features=input_feat,
            output_seq_length=output_seq_len,
            output_features=output_feat)
    
    # Check for distribution strategy
    if 'strategy' in globals() and strategy is not None:
        print(f"Using distribution strategy: {strategy}")
        distribution = strategy
    else:
        print("No distribution strategy found, using default.")
        distribution = tf.distribute.get_strategy()
        
    # Initialize tuner with distribution strategy
    tuner = kt.Hyperband(
        hypermodel=make_model,
        objective='val_loss',
        max_epochs=epochs_per_trial,
        factor=3,
        directory='kt_tuner',
        project_name='nfl_seq2seq',
        overwrite=True,
        distribution_strategy=distribution)

    # Early-stopping inside each trial
    stop_early = callbacks.EarlyStopping(monitor='val_loss',
                                         patience=3,
                                         restore_best_weights=True)

    tuner.search(train_seq,
                 validation_data=val_seq,
                 callbacks=[stop_early],
                 verbose=1)

    # Retrieve the best hyper-parameters & model
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]

    # Train the best model a little longer (optional)
    # Ensure training also uses the strategy if needed (model is already compiled with it)
    final_history = best_model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=epochs_per_trial * 2,   # give it more epochs now that we know the arch.
        callbacks=[callbacks.EarlyStopping(monitor='val_loss',
                                           patience=5,
                                           restore_best_weights=True)],
        verbose=1)

    return best_model, final_history, best_hp


def save_encoder_from_model(model, path):
    """
    Extracts and saves the encoder part of the seq2seq model.
    """
    try:
        # The encoder input is the model input
        encoder_inputs = model.input
        
        # The latent vector is the output of the layer named 'latent'
        latent_layer = model.get_layer('latent')
        latent_output = latent_layer.output
        
        # Create encoder model
        encoder_model = models.Model(inputs=encoder_inputs, outputs=latent_output, name='encoder')
        
        # Save
        encoder_model.save(path)
        print(f"Encoder model saved to {path}")
        return encoder_model
    except Exception as e:
        print(f"Error saving encoder: {e}")
        return None

def main():
    # ------------------------------------------------------------------
    # 1️⃣  Load Unsupervised Data & Prepare Sequences
    # ------------------------------------------------------------------
    PREDICTION_TRAIN_DIR = '/kaggle/input/nfl-big-data-bowl-2026-prediction/train'
    ANALYTICS_TRAIN_DIR = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train'

    print("Loading unsupervised data...")
    # Initialize loader
    loader = UnsupervisedNFLDataLoader()
    # Load from both directories
    loader.load_files(
        [PREDICTION_TRAIN_DIR, ANALYTICS_TRAIN_DIR],
        include_labeled=True,
        include_unlabeled=True,
        normalize=True,
    )
    X_unsupervised = loader.get_sequences()

    if len(X_unsupervised) == 0:
        print("ERROR: No unsupervised data loaded!")
        return
    
    print(f"Total unsupervised sequences: {len(X_unsupervised)}")
    
    # Split into train/val
    from sklearn.model_selection import train_test_split
    X_train, X_val = train_test_split(X_unsupervised, test_size=0.2, random_state=42)
    
    # Create sequences for Next-Step Prediction (Self-Supervised)
    task = 'autoencoder'
    prediction_steps = 10  # Predict next 5 steps
    
    train_seq = UnsupervisedNFLSequence(
        X_train,
        batch_size=64,
        maxlen=10, 
        shuffle=False,
        task=task,
        prediction_steps=prediction_steps
    )
    
    val_seq = UnsupervisedNFLSequence(
        X_val,
        batch_size=64,
        maxlen=10,
        shuffle=False,
        task=task,
        prediction_steps=prediction_steps
    )

    # Get shapes from a batch to configure the model
    x_batch, y_batch = train_seq[0]
    input_seq_len = x_batch.shape[1]
    input_feat = x_batch.shape[2]
    print(x_batch.shape)
    output_seq_len = y_batch.shape[1]
    output_feat = y_batch.shape[2]
    print(y_batch.shape)
    print(f"Input shape: ({input_seq_len}, {input_feat})")
    print(f"Output shape: ({output_seq_len}, {output_feat})")

    # ------------------------------------------------------------------
    # 2️⃣  Launch the tuner
    # ------------------------------------------------------------------
    best_model, best_history, best_hp = tuner_search(
            train_seq=train_seq,
            val_seq=val_seq,
            input_seq_len=input_seq_len,
            input_feat=input_feat,
            output_seq_len=output_seq_len,
            output_feat=output_feat,
            max_trials=30,          # increase if you have more time
            epochs_per_trial=12)    # short trials for speed

    print("\n=== Best hyper‑parameters ===")
    for name, value in best_hp.values.items():
        print(f"{name}: {value}")
        
    # Save best model
    best_model.save('best_hyperband_unsupervised_model_third_run.keras')
    print("Best model saved to best_hyperband_unsupervised_model.keras")

    # Save encoder separately
    save_encoder_from_model(best_model, 'best_hyperband_encoder_third_run.keras')


if __name__ == "__main__":
    main()


plt.figure(figsize=(8,5))
plt.plot(best_history.history['loss'], label='Training loss')
plt.plot(best_history.history['val_loss'], label='Validation loss')
plt.title('Best model – Training & Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import sys

# Add the manual_data_processing directory to the path
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'manual_data_processing'))

# from csv_to_numpy import NFLDataLoader, create_tf_datasets

def build_seq2seq_model(input_seq_length, input_features, output_seq_length, output_features, lstm_units=128):
    """
    Builds a sequence-to-sequence model with LSTM layers.

    Args:
        input_seq_length (int): The length of input sequences (time steps).
        input_features (int): The number of input features per timestep.
        output_seq_length (int): The length of output sequences (time steps).
        output_features (int): The number of output features per timestep.
        lstm_units (int): The number of units in the LSTM layers.

    Returns:
        keras.Model: The compiled Keras model.
    """

    SEED = 42
    # Encoder-decoder architecture for sequence-to-sequence prediction
    model = keras.Sequential([
        # Input layer
        keras.layers.Input(shape=(input_seq_length, input_features)),
        
        # Encoder LSTM layers
        keras.layers.LSTM(
            units=696,
            activation="sigmoid",
            return_sequences=True,
            kernel_regularizer=keras.regularizers.L2(l2=3.7001e-06),
            seed=SEED,
        ),
        keras.layers.LSTM(
            units=696 // 2,
            activation="sigmoid",
            return_sequences=True,
            kernel_regularizer=keras.regularizers.L2(l2=3.7001e-06),
            seed=SEED,
        ),
        keras.layers.LSTM(
            units=696 // 2,
            activation="sigmoid",
            return_sequences=True,
            kernel_regularizer=keras.regularizers.L2(l2=3.7001e-06),
            seed=SEED,
        ),
        keras.layers.LSTM(
            units=696 // 2,
            activation="sigmoid",
            return_sequences=True,
            kernel_regularizer=keras.regularizers.L2(l2=3.7001e-06),
            seed=SEED,
        ),
        keras.layers.LSTM(
            units=696 // 2,
            activation="sigmoid",
            return_sequences=True,
            kernel_regularizer=keras.regularizers.L2(l2=3.7001e-06),
            seed=SEED,
        ),
        # layers.RepeatVector(output_seq_length),
        keras.layers.LSTM(
            units=32,
            activation="sigmoid",
            return_sequences=True,
            # kernel_regularizer=keras.regularizers.L2(l2=0.00000195),
            seed=SEED,
        ),
        # Crop or slice to match output sequence length
        # layers.Lambda(lambda x: x[:, :output_seq_length, :]),
        # TimeDistributed dense layer for output features
        layers.TimeDistributed(
            keras.layers.Dense(units=output_features, activation="linear")
        ),
    ])

    cosine_decay = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=415000,
    alpha=1e-5,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00081926),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_model(model, train_sequence, val_sequence, epochs=10, callbacks=None):
    """
    Trains the Keras model using Keras Sequence objects.
    
    Args:
        model: The Keras model to train
        train_sequence: Training data sequence (NFLDataSequence)
        val_sequence: Validation data sequence (NFLDataSequence)
        epochs (int): Number of training epochs
        callbacks: List of Keras callbacks
    
    Returns:
        history: Training history object
    """
    if callbacks is None:
        callbacks = []
    
    # Add early stopping and model checkpoint callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    callbacks.extend([early_stopping, model_checkpoint])
    
    print("Starting model training...")
    history = model.fit(
        train_sequence,
        epochs=epochs,
        validation_data=val_sequence,
        callbacks=model_checkpoint,
        verbose=1
    )
    # -------------------------------------------------
    # Visualize training & validation loss
    # -------------------------------------------------
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'],      label='Training loss')
    plt.plot(history.history['val_loss'],  label='Validation loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("Model training finished.")
    return history

def main():
    """
    Main function to load data, build, and train the model.
    """
    # Configuration
    train_dir = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train'
    batch_size = 32
    epochs = 20
    test_size = 0.2
    
    print("="*60)
    print("NFL Big Data Bowl 2026 - Predictor Training")
    print("="*60)
    
    # Load and prepare data
    print("\n[1/4] Loading data from CSV files...")
    loader = NFLDataLoader(train_dir)
    X, y = loader.get_aligned_data()
    
    if len(X) == 0:
        print("Error: No data loaded. Please check the data directory.")
        return
    
    print(f"\nData Summary:")
    print(f"  Total sequences: {len(X)}")
    print(f"  Sample input sequence length: {len(X[0])}")
    print(f"  Sample output sequence length: {len(y[0])}")
    print(f"  Input features per timestep: {len(X[0][0]) if len(X[0]) > 0 else 0}")
    print(f"  Output features per timestep: {len(y[0][0]) if len(y[0]) > 0 else 0}")
    
    # Create Keras Sequences with padding
    print(f"\n[2/4] Creating training and validation sequences (test_size={test_size})...")
    train_seq, val_seq = create_tf_datasets(X, y, test_size=test_size, batch_size=batch_size)
    
    if train_seq is None:
        print("Error: Failed to create training sequences.")
        return
    
    # Get one batch to determine shapes
    x_sample, y_sample = train_seq[0]
    input_seq_length = x_sample.shape[1]
    input_features = x_sample.shape[2]
    output_seq_length = y_sample.shape[1]
    output_features = y_sample.shape[2]
    
    print(f"\nSequence Shapes:")
    print(f"  Input: (batch_size, {input_seq_length}, {input_features})")
    print(f"  Output: (batch_size, {output_seq_length}, {output_features})")
    
    # Build model
    print(f"\n[3/4] Building sequence-to-sequence model...")
    model = build_seq2seq_model(
        input_seq_length=input_seq_length,
        input_features=input_features,
        output_seq_length=output_seq_length,
        output_features=output_features,
        lstm_units=128
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    # Train model
    print(f"\n[4/4] Training model for {epochs} epochs...")
    history = train_model(model, train_seq, val_seq, epochs=epochs)
    
    # Save the final model
    final_model_path = 'nfl_predictor_final.keras'
    model.save(final_model_path)
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: best_model.keras")
    print(f"{'='*60}")
    
    # Print training summary
    print(f"\nTraining Summary:")
    print(f"  Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"  Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print(f"  Final training MAE: {history.history['mae'][-1]:.4f}")
    print(f"  Final validation MAE: {history.history['val_mae'][-1]:.4f}")
    print(f"  Best validation loss: {min(history.history['val_loss']):.4f}")

if __name__ == '__main__':
    main()

