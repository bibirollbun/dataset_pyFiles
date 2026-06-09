import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Embedding, Bidirectional, 
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D,
    Dropout, Add, concatenate, Layer
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint,  TensorBoard
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')


class SequenceFeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates and encodes sequence position features"""
    def __init__(self, max_seq_len=206):
        self.max_seq_len = max_seq_len
        self.residues = ['A','C','G','U']
        self.combinations = [a+b for a in self.residues for b in self.residues]
        self.vocab = {'A':0, 'C':1, 'G':2, 'U':3}
        self.encoder = LabelEncoder()
        self.expected_features = None

    def fit(self, X, y=None):
        # Create temporary features for encoding
        X_temp = X.copy()
        X_temp['start_res'] = X_temp['sequence'].str[0]
        X_temp['end_res'] = X_temp['sequence'].str[-1]
        
        # Keep all original columns except specific ones
        base_features = [col for col in X.columns 
                        if col not in ['temporal_cutoff', 'description', 'all_sequences']]
        
        new_features = [
            'start_res', 'end_res', 'sequence',
            *[f'{r}_count' for r in self.residues],
            *[f'{combo}_count' for combo in self.combinations]
        ]
        
        self.expected_features = list(set(base_features + new_features))
        self.encoder.fit(pd.concat([X_temp['start_res'], X_temp['end_res']]))
        return self
    
    def transform(self, X):
        df = X.copy()

        # Preserve original sequence column
        if 'sequence' not in df.columns:
            raise ValueError("Missing required 'sequence' column")
        
        # Create position features
        df['start_res'] = df['sequence'].str[0]
        df['end_res'] = df['sequence'].str[-1]
        
        # Encode start/end residues
        df['start_res'] = self.encoder.transform(df['start_res'])
        df['end_res'] = self.encoder.transform(df['end_res'])
        
        # Add residue counts
        for r in self.residues:
            df[f'{r}_count'] = df['sequence'].str.count(r).fillna(0)
        
        # Add combination counts
        for combo in self.combinations:
            df[f'{combo}_count'] = df['sequence'].str.count(combo).fillna(0)
            
        # Add encoded sequence
        df['encoded'] = df['sequence'].apply(
            lambda s: [self.vocab.get(b.upper(), 0) for b in s[:self.max_seq_len]] +
                     [0]*(self.max_seq_len - len(s)))

        # Ensure consistent feature set
        for col in self.expected_features:
            if col not in df.columns:
                df[col] = 0
                
        return df[self.expected_features + ['encoded']]


# Enhanced Feature Engineering
class EnhancedFeatureEngineer(SequenceFeatureEngineer):
    def transform(self, X):
        df = super().transform(X)
        
        # Add features while preserving identifiers
        df['gc_content'] = (df['G_count'] + df['C_count']) / df['sequence'].str.len()
        
        # Calculate dinucleotide frequencies
        for combo in self.combinations:
            df[f'{combo}_freq'] = df[f'{combo}_count'] / (df['sequence'].str.len() - 1)
            
        # Proper structural feature calculation
        df['stem_potential'] = df['sequence'].apply(lambda s: s.count('G') + s.count('C'))
        df['loop_potential'] = df['sequence'].apply(lambda s: s.count('A') + s.count('U'))
        
        return df

    def add_structural_features(self, df, sequence):
        # Placeholder for actual structural calculations
        df['stem_potential'] = sequence.count('G') + sequence.count('C')
        df['loop_potential'] = sequence.count('A') + sequence.count('T')

# Enhanced Data Augmentation
def augment_sequences(features_df, labels_df):
    """Proper data augmentation with coordinate list handling"""
    # Store original feature columns
    feature_cols = features_df.columns.tolist()
    
    # Ensure that the labels DataFrame has an 'ID' column
    if 'ID' not in labels_df.columns:
        raise ValueError("Labels DataFrame does not have an 'ID' column.")
    
    # Extract join key from labels: extract target_id from ID (format: targetid_chainid_resid)
    labels_df = labels_df.copy()  # Avoid modifying original DataFrame
    labels_df['sequence_id'] = labels_df['ID'].str.split('_').apply(lambda x: '_'.join(x[:2]))
    
    # Debug: Print unique join keys from both DataFrames
    # print("Unique target_id in features:", features_df['target_id'].unique())
    # print("Unique sequence_id in labels:", labels_df['sequence_id'].unique())
    
    # Group labels by the extracted sequence_id
    labels_processed = (
        labels_df.groupby('sequence_id')
        .agg({
            'x_1': list,
            'y_1': list,
            'z_1': list,
            'ID': 'first',   
            'resname': 'first', 
            'resid': 'first' 
        })
        .reset_index()
    )
    
    # Merge using features_df.target_id and labels_processed.sequence_id
    merged = pd.merge(
        features_df,
        labels_processed,
        left_on='target_id',
        right_on='sequence_id',
        how='inner'
    )
    
    if merged.empty:
        raise ValueError("No matching records found between features and labels. "
                         "Please check that your join keys (target_id and sequence_id) match.")
    
    augmented = []
    for _, row in merged.iterrows():
        # Original entry
        augmented.append(row.to_dict())
        
        # Reverse complement with coordinate reversal
        rev_entry = row.to_dict()
        rev_entry['sequence'] = rev_entry['sequence'][::-1].translate(str.maketrans('ACGU', 'UGCA'))
        rev_entry['x_1'] = rev_entry['x_1'][::-1]
        rev_entry['y_1'] = rev_entry['y_1'][::-1]
        rev_entry['z_1'] = rev_entry['z_1'][::-1]
        # Update both target_id and ID so that grouping later treats this as a separate sample
        rev_entry['target_id'] = f"{rev_entry['target_id']}_rev"
        rev_entry['ID'] = f"{rev_entry['ID']}_rev"
        augmented.append(rev_entry)

    
    augmented_df = pd.DataFrame(augmented)
    
    # Ensure all original feature columns are still present
    missing = [col for col in feature_cols if col not in augmented_df.columns]
    if missing:
        raise ValueError(f"Missing columns after augmentation: {missing}")
    
    return augmented_df[feature_cols], augmented_df[['x_1', 'y_1', 'z_1']]


# Enhanced Training Configuration
def configure_training(model):
    return [
        EarlyStopping(patience=15, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6),
        TerminateOnNaN(),
        ModelCheckpoint('best_model.h5', save_best_only=True)
    ]


# Define the GPU device
device_name = tf.test.gpu_device_name()


class RNAHybridModel:
    """Combines sequence embedding with engineered features"""
    def __init__(self, max_seq_len, feature_dim):
        self.max_seq_len = max_seq_len
        self.feature_dim = feature_dim
        self.model = self.build_model()

    # Use the GPU device for model creation and training
    def build_model(self):
        # Use the GPU device for model creation
        with tf.device(device_name):
            # Enhanced sequence processing
            seq_input = Input(shape=(self.max_seq_len,), name='seq_input')
            x = Embedding(4, 256, mask_zero=False)(seq_input)
            
            # Stacked BiLSTMs with dropout
            x = Bidirectional(LSTM(512, return_sequences=True, dropout=0.3))(x)
            x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2))(x)
            
            # Transformer block with positional encoding
            pos_enc = self.positional_encoding(self.max_seq_len, 512)
            x = Add()([x, pos_enc])
            x = MultiHeadAttention(num_heads=12, key_dim=64)(x, x)
            x = GlobalAveragePooling1D()(x)
            
            # Feature fusion
            feat_input = Input(shape=(self.feature_dim,), name='feat_input')
            merged = concatenate([x, feat_input])
            
            # Enhanced dense processing
            merged = Dense(1024, activation='swish', kernel_regularizer='l2')(merged)
            merged = Dropout(0.5)(merged)
            
            # Create named output heads
            x_head = self.create_coord_head(merged, 'x_out')
            y_head = self.create_coord_head(merged, 'y_out')
            z_head = self.create_coord_head(merged, 'z_out')
            
            return Model(inputs=[seq_input, feat_input], 
                       outputs=[x_head, y_head, z_head])

    def create_coord_head(self, x, name):
        x = Dense(768, activation='swish')(x)
        x = Dense(384, activation='swish')(x)
        return Dense(self.max_seq_len, activation='linear', name=name)(x)

    def positional_encoding(self, length, depth):
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :]/depth
        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates
        pos_encoding = np.concatenate(
            [np.sin(angle_rads[:, ::2]), np.cos(angle_rads[:, ::2])],
            axis=-1
        )
        return tf.cast(pos_encoding[np.newaxis, ...], tf.float32)

    def compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss={'x_out': 'huber', 'y_out': 'huber', 'z_out': 'huber'},
            loss_weights=[0.35, 0.35, 0.3],
            metrics={'x_out': ['mae'], 'y_out': ['mae'], 'z_out': ['mae']} 
        )


class DataProcessor:
    """Handles temporal features and data validation"""
    def __init__(self, max_seq_len=206):
        self.max_seq_len = max_seq_len
        self.feature_engineer = SequenceFeatureEngineer(max_seq_len)
        
    def add_temporal_features(self, df):
        processed_df = df.copy()
        # Ensure target_id is preserved
        if 'target_id' not in processed_df.columns:
            raise ValueError("target_id column missing in input data")
            
        # Existing temporal feature calculations
        processed_df['cutoff_year'] = pd.to_datetime(processed_df['temporal_cutoff']).dt.year
        processed_df['seq_length'] = processed_df['sequence'].str.len()
        
        # Yearly averages
        yearly_avg = processed_df.groupby('cutoff_year')['seq_length'].mean().to_dict()
        processed_df['yearly_avg_length'] = processed_df['cutoff_year'].map(yearly_avg)
        
        return processed_df
    
    def preprocess_labels(self, label_df):
        """Process labels at the sequence level"""
        coord_arrays = []
        
        for _, row in label_df.iterrows():
            # Convert coordinate lists to numpy arrays
            x = np.array(row['x_1'], dtype='float32')
            y = np.array(row['y_1'], dtype='float32')
            z = np.array(row['z_1'], dtype='float32')
            
            # Pad each coordinate sequence individually
            x_pad = pad_sequences([x], maxlen=self.max_seq_len, padding='post')[0]
            y_pad = pad_sequences([y], maxlen=self.max_seq_len, padding='post')[0]
            z_pad = pad_sequences([z], maxlen=self.max_seq_len, padding='post')[0]
            
            # Combine into single array (x1, x2..., y1, y2..., z1, z2...)
            coord_arrays.append(np.concatenate([x_pad, y_pad, z_pad]))
            
        return np.array(coord_arrays)


def predict_structures(model, inputs, num_samples=5):
    """Returns predictions in (num_samples, batch_size, 3, max_seq_len) shape"""
    mc_model = Model(inputs=model.inputs, outputs=model.outputs)
    ensemble_preds = []
    
    for _ in range(num_samples):
        preds = mc_model(inputs, training=True)
        # Stack predictions as (batch_size, max_seq_len, 3)
        stacked = np.stack(preds, axis=-1)
        ensemble_preds.append(stacked)
    
    # Combine to (num_samples, batch_size, max_seq_len, 3)
    combined = np.stack(ensemble_preds, axis=0)
    # Reshape to (num_samples, batch_size, 3, max_seq_len)
    return combined.transpose(0, 1, 3, 2)

def create_submission(model, test_df, sample_path, feature_engineer, max_seq_len):
    """Create submission using pre-fitted feature engineer"""
    # Preserve original IDs and sequences
    original_ids = test_df['target_id'].copy()
    original_sequences = test_df['sequence'].copy()
    
    # Process test data with existing feature engineer
    test_processed = feature_engineer.transform(test_df)
    
    # Prepare features
    X_seq = np.array(test_processed['encoded'].tolist())
    feature_columns = [col for col in test_processed.columns 
                      if col not in ['target_id', 'sequence', 'encoded', 
                                    'temporal_cutoff', 'description', 'all_sequences']]
    X_feat = test_processed[feature_columns].values.astype('float32')
    
    # Generate predictions
    ensemble_preds = predict_structures(model, [X_seq, X_feat])
    
    # Format predictions using original IDs and sequences
    submission_rows = []
    for idx in range(len(test_df)):
        target_id = original_ids.iloc[idx]
        sequence = original_sequences.iloc[idx]
        seq_len = len(sequence)
        
        # Get predictions (num_samples, 3, max_seq_len)
        mc_samples = ensemble_preds[:, idx, :, :]
        
        for pos in range(seq_len):
            pos_idx = min(pos, max_seq_len-1)
            entry = {
                'ID': f"{target_id}_{pos+1}",
                'resname': sequence[pos].upper(),  # Ensure uppercase
                'resid': pos+1
            }
            
            # Add all 5 samples
            for sample_num in range(5):
                entry.update({
                    f'x_{sample_num+1}': mc_samples[sample_num, 0, pos_idx],
                    f'y_{sample_num+1}': mc_samples[sample_num, 1, pos_idx],
                    f'z_{sample_num+1}': mc_samples[sample_num, 2, pos_idx]
                })
            
            submission_rows.append(entry)
    
    submission_df = pd.DataFrame(submission_rows)

    print(submission_df.columns)
    print(submission_df.shape)
    print(submission_df.head())
    
    return submission_df

def compute_calibration_factors(true_x, true_y, true_z, pred_x, pred_y, pred_z):
    """
    Compute calibration factors for each coordinate axis.
    For each axis, the calibration factor is defined as:
    
        factor = mean(true_coordinate) / mean(predicted_coordinate)
    
    If the predicted mean is zero (to avoid division by zero), a factor of 1.0 is used.
    
    Parameters:
        true_x, true_y, true_z: Arrays of true coordinates (flattened or over all residues)
        pred_x, pred_y, pred_z: Arrays of predicted coordinates (flattened or over all residues)
    
    Returns:
        A dictionary with keys 'x', 'y', 'z' and corresponding calibration factors.
    """
    factors = {}
    factors['x'] = np.mean(true_x) / np.mean(pred_x) if np.mean(pred_x) != 0 else 1.0
    factors['y'] = np.mean(true_y) / np.mean(pred_y) if np.mean(pred_y) != 0 else 1.0
    factors['z'] = np.mean(true_z) / np.mean(pred_z) if np.mean(pred_z) != 0 else 1.0
    return factors

def apply_calibration(submission_df, calibration_factors):
    """
    Adjust the submission DataFrame coordinates by applying calibration factors.
    
    For every coordinate column (x, y, and z) and for each sample (x_1, x_2, …, x_5),
    multiply the predicted value by the corresponding calibration factor.
    
    Parameters:
        submission_df (DataFrame): The submission DataFrame with columns like 'x_1', 'y_1', 'z_1', etc.
        calibration_factors (dict): A dictionary containing calibration factors for 'x', 'y', and 'z'.
    
    Returns:
        The adjusted submission DataFrame.
    """
    for axis in ['x', 'y', 'z']:
        for sample_num in range(1, 6):  # For samples 1 through 5
            col = f'{axis}_{sample_num}'
            submission_df[col] = submission_df[col] * calibration_factors.get(axis, 1.0)
    return submission_df



# Define a TensorBoard callback (log directory can be adjusted)
log_dir = "./logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)


callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
    TerminateOnNaN(),
    tensorboard_cb
]


def main():
    # Configuration
    # MAX_SEQ_LEN = 15
    
    # Load raw data with original columns
    train_seq = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')
    train_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')

    # Handle Missing Values
    train_labels['x_1'] = train_labels['x_1'].fillna(train_labels['x_1'].median())
    train_labels['y_1'] = train_labels['y_1'].fillna(train_labels['y_1'].median())
    train_labels['z_1'] = train_labels['z_1'].fillna(train_labels['z_1'].median())

    # Compute lengths of sequences
    seq_lengths = train_seq['sequence'].str.len().values

    MAX_SEQ_LEN = int(np.percentile(seq_lengths, 95))  # Covers 95% of sequences
    

    # After loading data
    # print("Original features:", train_seq.columns.tolist())
    
    # 1. Add temporal features (requires original columns)
    processor = DataProcessor(MAX_SEQ_LEN)
    train_seq = processor.add_temporal_features(train_seq)
    # print("After temporal processing:", train_seq.columns.tolist())
    
    # 2. Add sequence features while preserving temporal features
    feature_engineer = EnhancedFeatureEngineer(MAX_SEQ_LEN)
    feature_engineer.fit(train_seq) 
    train_seq = feature_engineer.transform(train_seq)
    # print("After feature engineering:", train_seq.columns.tolist())

    # Data augmentation (2x dataset size)
    # print("Features going into augmentation:", train_seq.columns.tolist())
    # print("Applying data augmentation...")
    augmented_features, augmented_labels = augment_sequences(train_seq, train_labels)
    # print("Feature columns after augmentation:", train_seq.columns.tolist())
    
    # After augmentation
    X_seq = np.stack(augmented_features['encoded'].values)
    feature_columns = [col for col in augmented_features.columns 
                      if col not in ['encoded', 'target_id', 'ID', 'sequence_id', 'sequence']]  # Added 'sequence'
    X_feat_array = augmented_features[feature_columns].values.astype('float32')
    
    # Process labels and split coordinates
    padded_labels = processor.preprocess_labels(augmented_labels)
    y = [
        padded_labels[:, :MAX_SEQ_LEN],         # x coordinates
        padded_labels[:, MAX_SEQ_LEN:2*MAX_SEQ_LEN],  # y coordinates
        padded_labels[:, 2*MAX_SEQ_LEN:]        # z coordinates
    ]
    
    # Add validation before training
    # print(f"Feature shapes: {X_seq.shape}, {X_feat_array.shape}")
    # print(f"Label shapes: {[arr.shape for arr in y]}")
    
    # Build and train model (assuming EnhancedRNAHybridModel is defined)
    model_wrapper = RNAHybridModel(MAX_SEQ_LEN, X_feat_array.shape[1])
    model_wrapper.compile_model()

    # After model initialization
    # print(model_wrapper.model.output_names)  # Should show ['x_out', 'y_out', 'z_out']
    
    history = model_wrapper.model.fit(
        [X_seq, X_feat_array], y,
        epochs=50,
        batch_size=128,
        validation_split=0.2,
        callbacks=callbacks
    )

    # Predict on your validation set (or a subset of training data)
    preds = model_wrapper.model.predict([X_seq, X_feat_array])
    # preds is a list: [pred_x, pred_y, pred_z] each of shape (num_samples, MAX_SEQ_LEN)
    
    # Flatten predictions (ignoring padded zeros if needed)
    pred_x = preds[0].flatten()
    pred_y = preds[1].flatten()
    pred_z = preds[2].flatten()
    
    # Similarly, flatten the true coordinates from padded_labels
    true_x = padded_labels[:, :MAX_SEQ_LEN].flatten()
    true_y = padded_labels[:, MAX_SEQ_LEN:2*MAX_SEQ_LEN].flatten()
    true_z = padded_labels[:, 2*MAX_SEQ_LEN:].flatten()
    
    calibration_factors = compute_calibration_factors(true_x, true_y, true_z, pred_x, pred_y, pred_z)
    print("Calibration factors:", calibration_factors)

    
    # Create submission with the fitted feature engineer
    test_df = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')
    submission = create_submission(
        model_wrapper.model,
        test_df,
        '/kaggle/input/stanford-rna-3d-folding/sample_submission.csv',
        feature_engineer,
        MAX_SEQ_LEN
    )
    # Post-prediction calibration if applicable
    submission = apply_calibration(submission, calibration_factors)
    submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()

