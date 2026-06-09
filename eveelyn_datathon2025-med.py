# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_csv = pd.read_csv("/kaggle/input/asl-signs/train.csv")


train_csv


critical_emergency_signs = [
    "bad", "boy", "child", "dad", "fireman", "girl", "go", "home", 
    "hot", "listen", "look", "man", "mom", "no", "outside", "owie", 
    "person", "police", "sick", "there", "wait", "water", "where", "yes",
    "airplane", "all", "arm", "black", "blow", "blue", "boat", "car", 
    "close", "cry", "cut", "down", "dry", "ear", "eye", "face", "fall", 
    "fast", "feet", "finger", "green", "happy", "head", "hear", 
    "helicopter", "later", "loud", "mad", "many", "mouth", "nose", 
    "now", "open", "quiet", "red", "sad", "say", "see", "talk", 
    "touch", "up", "wet", "white", "who", "why", "yellow", "alligator", "animal", "backyard", "bed", "because", "after", "another", "any", "bedroom", "bee", "before", "beside", "bug", "can", "cat", "cheek", "chin", "dog", "drop", "find", 
    "for", "give", "glasswindow", "grandma", "grandpa", "hair", "have", "haveto", "hesheit", "jump","if", 
    "into", "hide", "high","like",  "night", "noisy", 
    "not", "please", "pool", "room", "stairs", "stuck", "think", "that", "time", "tree", "will", 
]

# filter out critical emergency signs
df_filtered = train_csv[train_csv['sign'].isin(critical_emergency_signs)]
df_filtered


df_filtered.loc[df_filtered["participant_id"] == 32319]


parquet_files = df_filtered.loc[df_filtered["participant_id"] == 32319]["path"].to_list()


folder_path = "/kaggle/input/asl-signs/"

# Read, label with path, and concatenate
df_parquet = pd.concat(
    [
        pd.read_parquet(os.path.join(folder_path, path)).assign(path=path)
        for path in parquet_files
    ],
    ignore_index=True
)


df_parquet.head()


merged_df = df_parquet.merge(df_filtered, on="path", how="left")


merged_df


# merged_df.drop(columns="participant_id").to_csv("train_landmark_files_2018.csv", index=False)


merged_df.info()


import os
import warnings

# Suppress TensorFlow warnings and GPU messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Explicitly set GPU devices for T4 x2

import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split

# Additional GPU configuration for T4 x2
def configure_gpu():
    """Configure GPU settings for optimal performance on T4 x2"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid OOM errors
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set up multi-GPU strategy for T4 x2
            strategy = tf.distribute.MirroredStrategy()
            print(f"Number of replicas: {strategy.num_replicas_in_sync}")
            return strategy
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            return None
    else:
        print("No GPUs found, using CPU")
        return None

# Configure GPU before importing the rest
strategy = configure_gpu()

# Constants
ROWS_PER_FRAME = 543
MAX_LEN = 384
PAD = -100.0

NOSE = [1, 2, 98, 327]
LNOSE = [98]
RNOSE = [327]
LIP = [0, 61, 185, 40, 39, 37, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LLIP = [84, 181, 91, 146, 61, 185, 40, 39, 37, 87, 178, 88, 95, 78, 191, 80, 81, 82]
RLIP = [314, 405, 321, 375, 291, 409, 270, 269, 267, 317, 402, 318, 324, 308, 415, 310, 311, 312]
POSE = [500, 502, 504, 501, 503, 505, 512, 513]
LPOSE = [513, 505, 503, 501]
RPOSE = [512, 504, 502, 500]
REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
LHAND = list(range(468, 489))
RHAND = list(range(522, 543))
POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE
NUM_NODES = len(POINT_LANDMARKS)
CHANNELS = 6 * NUM_NODES

class SignLanguageProcessor:
    def __init__(self):
        self.sign_to_index = {}
        self.index_to_sign = {}
        self.sign_count = 0
        
    def analyze_data_structure(self, df: pd.DataFrame):
        """Analyze the structure of the input data to understand format"""
        print("=== Data Structure Analysis ===")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Shape: {df.shape}")
        
        # Analyze row_id format
        sample_row_ids = df['row_id'].head(20).tolist()
        print(f"Sample row_ids: {sample_row_ids}")
        
        # Check if row_ids are numeric or string-based
        numeric_count = 0
        string_count = 0
        
        for row_id in sample_row_ids:
            try:
                int(row_id)
                numeric_count += 1
            except (ValueError, TypeError):
                string_count += 1
        
        print(f"Numeric row_ids: {numeric_count}, String-based row_ids: {string_count}")
        
        # Analyze frame structure
        print(f"Frame range: {df['frame'].min()} to {df['frame'].max()}")
        print(f"Total frames: {df['frame'].nunique()}")
        print(f"Unique signs: {df['sign'].nunique()}")
        print(f"Sample signs: {df['sign'].unique()[:10]}")
        
        # Check for missing values
        print(f"Missing values per column:")
        for col in ['x', 'y', 'z']:
            missing = df[col].isna().sum()
            print(f"  {col}: {missing} ({missing/len(df)*100:.1f}%)")
        
        # Additional debugging info
        print(f"Unique row_id count: {df['row_id'].nunique()}")
        print(f"Records per frame (avg): {len(df) / df['frame'].nunique():.1f}")
        
        print("=" * 30)
        
        return {
            'has_string_row_ids': string_count > numeric_count,
            'frame_range': (df['frame'].min(), df['frame'].max()),
            'num_signs': df['sign'].nunique(),
            'missing_coords': {col: df[col].isna().sum() for col in ['x', 'y', 'z']}
        }
    def build_sign_vocabulary(self, df: pd.DataFrame) -> List[str]:
        unique_signs = df['sign'].unique()
        for idx, sign in enumerate(unique_signs):
            self.sign_to_index[sign] = idx
            self.index_to_sign[idx] = sign
        self.sign_count = len(unique_signs)
        return unique_signs.tolist()
    
    def group_by_sequence(self, df: pd.DataFrame) -> Dict:
        print("\n=== Grouping by Sequence ===")
        # Check actual column names and adapt
        if 'participant_id' in df.columns and 'sequence_id' in df.columns:
            print("Using participant_id + sequence_id for grouping")
            df['seq_key'] = df['participant_id'].astype(str) + '_' + df['sequence_id'].astype(str)
        elif 'path' in df.columns:
            print("Using path column for grouping")
            # Extract participant and sequence from path
            df['seq_key'] = df['path'].str.replace('.parquet', '').str.replace('train_landmark_files/', '')
        else:
            print("Using frame-based grouping (fallback)")
            # Fallback: use frame groups if available
            df['seq_key'] = df.index // 1000  # Rough grouping
        
        grouped = df.groupby('seq_key')
        print(f"Created {len(grouped)} sequences")
        
        # Debug: Show sample sequence info
        sample_keys = list(grouped.groups.keys())[:5]
        print(f"Sample sequence keys: {sample_keys}")
        
        for key in sample_keys[:2]:  # Show details for first 2 sequences
            seq_data = grouped.get_group(key)
            print(f"  Sequence '{key}': {len(seq_data)} records, frames {seq_data['frame'].min()}-{seq_data['frame'].max()}, sign: {seq_data['sign'].iloc[0]}")
        
        return grouped
    
    def create_landmark_mapping(self, df: pd.DataFrame):
        """Create a mapping from string-based row_ids to numeric indices"""
        print("\n=== Creating Landmark Mapping ===")
        unique_row_ids = df['row_id'].unique()
        print(f"Found {len(unique_row_ids)} unique row_ids")
        
        # Create mapping dictionary
        self.row_id_mapping = {}
        
        # Sort unique row_ids to ensure consistent mapping
        sorted_row_ids = sorted(unique_row_ids, key=str)
        
        for idx, row_id in enumerate(sorted_row_ids):
            if idx < ROWS_PER_FRAME:  # Only map if within our expected range
                self.row_id_mapping[row_id] = idx
        
        print(f"Created mapping for {len(self.row_id_mapping)} landmark types")
        print(f"Sample mapping: {dict(list(self.row_id_mapping.items())[:5])}")
        
        # Show mapping statistics
        if len(unique_row_ids) > ROWS_PER_FRAME:
            print(f"WARNING: {len(unique_row_ids)} unique row_ids found, but only {ROWS_PER_FRAME} slots available")
            print(f"Unmapped row_ids: {len(unique_row_ids) - len(self.row_id_mapping)}")
        
        return self.row_id_mapping
    
    def reshape_to_frames_robust(self, sequence_df: pd.DataFrame) -> np.ndarray:
        """Robust version that handles different row_id formats"""
        print(f"\n--- Processing sequence with {len(sequence_df)} records ---")
        frames = []
        
        # Create mapping if not exists
        if not hasattr(self, 'row_id_mapping'):
            self.create_landmark_mapping(sequence_df)
        
        frame_numbers = sorted(sequence_df['frame'].unique())
        print(f"Processing {len(frame_numbers)} frames: {frame_numbers[:5]}{'...' if len(frame_numbers) > 5 else ''}")
        
        successful_mappings = 0
        failed_mappings = 0
        
        for frame_num in frame_numbers:
            frame_data = sequence_df[sequence_df['frame'] == frame_num]
            coordinates = np.full((ROWS_PER_FRAME, 3), np.nan)
            
            frame_successful = 0
            
            for _, row in frame_data.iterrows():
                row_id = row['row_id']
                
                # Try different approaches to get index
                idx = None
                
                # Approach 1: Use mapping if available
                if hasattr(self, 'row_id_mapping') and row_id in self.row_id_mapping:
                    idx = self.row_id_mapping[row_id]
                
                # Approach 2: Try parsing as number
                elif isinstance(row_id, (int, float)):
                    idx = int(row_id)
                
                # Approach 3: Extract number from string
                else:
                    try:
                        idx = self.parse_row_id(row_id)
                    except:
                        failed_mappings += 1
                        continue
                
                # Store coordinates if valid index
                if idx is not None and 0 <= idx < ROWS_PER_FRAME:
                    coordinates[idx] = [row['x'], row['y'], row['z']]
                    successful_mappings += 1
                    frame_successful += 1
                else:
                    failed_mappings += 1
            
            frames.append(coordinates)
            
            # Debug info for first few frames
            if len(frames) <= 3:
                non_nan_count = np.sum(~np.isnan(coordinates[:, 0]))
                print(f"  Frame {frame_num}: {frame_successful} landmarks mapped, {non_nan_count} non-NaN coordinates")
        
        print(f"Total mapping results: {successful_mappings} successful, {failed_mappings} failed")
        result = np.array(frames)
        print(f"Final frames shape: {result.shape}")
        
        return result
    def parse_row_id(self, row_id):
        """Parse row_id which can be either integer or string format like '18-face-0'"""
        if isinstance(row_id, (int, float)):
            return int(row_id)
        
        # Handle string format like '18-face-0', 'left_hand_21', etc.
        if isinstance(row_id, str):
            # Try to extract the last number after the last dash or underscore
            import re
            numbers = re.findall(r'\d+', row_id)
            if numbers:
                # Use the last number found
                return int(numbers[-1])
            else:
                # If no numbers found, try to map landmark types to indices
                return self.map_landmark_type_to_index(row_id)
        
        return 0  # fallback
    
    def map_landmark_type_to_index(self, row_id):
        """Map landmark type strings to indices"""
        # Create a mapping for different landmark types
        landmark_mapping = {}
        
        # Face landmarks (0-467)
        if 'face' in row_id.lower():
            return 0  # Start of face landmarks
        
        # Left hand landmarks (468-488)  
        elif 'left_hand' in row_id.lower():
            return 468
        
        # Right hand landmarks (489-509)
        elif 'right_hand' in row_id.lower():
            return 489
        
        # Pose landmarks (510-542)
        elif 'pose' in row_id.lower():
            return 510
        
        return 0  # fallback
    
    def reshape_to_frames(self, sequence_df: pd.DataFrame) -> np.ndarray:
        """Use the robust version by default"""
        return self.reshape_to_frames_robust(sequence_df)
    
    def filter_nans(self, frames: np.ndarray) -> np.ndarray:
        print(f"\n--- Filtering NaN frames ---")
        print(f"Input frames shape: {frames.shape}")
        
        valid_frames = []
        for i, frame in enumerate(frames):
            ref_points = frame[POINT_LANDMARKS]
            if not np.all(np.isnan(ref_points[:, :2])):
                valid_frames.append(frame)
            elif i < 5:  # Debug first few frames
                nan_count = np.sum(np.isnan(ref_points[:, :2]))
                print(f"  Frame {i}: {nan_count}/{len(ref_points)*2} NaN values in reference points")
        
        result = np.array(valid_frames) if valid_frames else np.array([])
        print(f"Filtered from {len(frames)} to {len(result)} valid frames")
        
        return result
    
    def tf_nan_mean(self, x, axis=0, keepdims=False):
        return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis, keepdims=keepdims) / \
               tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis, keepdims=keepdims)
    
    def tf_nan_std(self, x, center=None, axis=0, keepdims=False):
        if center is None:
            center = self.tf_nan_mean(x, axis=axis, keepdims=True)
        d = x - center
        return tf.math.sqrt(self.tf_nan_mean(d * d, axis=axis, keepdims=keepdims))
    
    def preprocess_sequence(self, frames: np.ndarray) -> np.ndarray:
        print(f"--- Preprocessing sequence with {len(frames)} frames ---")
        
        if len(frames) == 0:
            print("Empty frames array, returning empty result")
            return np.array([])
        
        x = tf.constant(frames, dtype=tf.float32)
        if tf.rank(x) == 3:
            x = x[None, ...]
        
        print(f"Input tensor shape: {x.shape}")
        
        mean = self.tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1, 2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5, x.dtype), mean)
        
        x = tf.gather(x, POINT_LANDMARKS, axis=2)
        print(f"After gathering landmarks: {x.shape}")
        
        std = self.tf_nan_std(x, center=mean, axis=[1, 2], keepdims=True)
        x = (x - mean) / std
        
        if x.shape[1] > MAX_LEN:
            print(f"Trimming sequence from {x.shape[1]} to {MAX_LEN} frames")
            x = x[:, :MAX_LEN]
        
        length = tf.shape(x)[1]
        x = x[..., :2]
        
        dx = tf.cond(tf.shape(x)[1] > 1,
                    lambda: tf.pad(x[:, 1:] - x[:, :-1], [[0, 0], [0, 1], [0, 0], [0, 0]]),
                    lambda: tf.zeros_like(x))
        
        dx2 = tf.cond(tf.shape(x)[1] > 2,
                     lambda: tf.pad(x[:, 2:] - x[:, :-2], [[0, 0], [0, 2], [0, 0], [0, 0]]),
                     lambda: tf.zeros_like(x))
        
        x = tf.concat([
            tf.reshape(x, (-1, length, 2 * len(POINT_LANDMARKS))),
            tf.reshape(dx, (-1, length, 2 * len(POINT_LANDMARKS))),
            tf.reshape(dx2, (-1, length, 2 * len(POINT_LANDMARKS))),
        ], axis=-1)
        
        x = tf.where(tf.math.is_nan(x), tf.constant(0., x.dtype), x)
        result = x[0].numpy()
        print(f"Final preprocessed shape: {result.shape}")
        
        return result
    
    def crop_or_pad(self, sequence: np.ndarray, max_len: int = MAX_LEN) -> np.ndarray:
        if len(sequence) >= max_len:
            return sequence[:max_len]
        
        padding = np.full((max_len - len(sequence), CHANNELS), PAD)
        return np.vstack([sequence, padding])
    
    def one_hot_encode(self, sign_label: str) -> np.ndarray:
        index = self.sign_to_index.get(sign_label, 0)
        one_hot = np.zeros(self.sign_count)
        one_hot[index] = 1
        return one_hot
    
    def process_sequence(self, sequence_df: pd.DataFrame, sign: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        print(f"\n=== Processing sequence for sign: '{sign}' ===")
        print(f"Sequence data shape: {sequence_df.shape}")
        
        frames = self.reshape_to_frames(sequence_df)
        if len(frames) == 0:
            print("â�Œ No frames generated, skipping sequence")
            return None
        
        filtered_frames = self.filter_nans(frames)
        if len(filtered_frames) == 0:
            print("â�Œ No valid frames after filtering, skipping sequence")
            return None
        
        processed_features = self.preprocess_sequence(filtered_frames)
        if len(processed_features) == 0:
            print("â�Œ No features after preprocessing, skipping sequence")
            return None
        
        final_sequence = self.crop_or_pad(processed_features)
        one_hot_sign = self.one_hot_encode(sign)
        
        print(f"âœ… Successfully processed sequence: {final_sequence.shape}")
        
        return final_sequence, one_hot_sign
    
    def process_dataset(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        print("\nğŸš€ Starting dataset processing...")
        
        # First analyze the data structure
        structure_info = self.analyze_data_structure(df)
        
        print("\nğŸ“� Building sign vocabulary...")
        self.build_sign_vocabulary(df)
        print(f"Built vocabulary with {self.sign_count} signs: {list(self.sign_to_index.keys())[:10]}{'...' if self.sign_count > 10 else ''}")
        
        grouped = self.group_by_sequence(df)
        
        X, y = [], []
        successful_sequences = 0
        failed_sequences = 0
        
        print(f"\nğŸ”„ Processing {len(grouped)} sequences...")
        
        for i, (seq_key, sequence_df) in enumerate(grouped):
            if i < 5 or i % 100 == 0:  # Show progress for first 5 and every 100th
                print(f"\nProcessing sequence {i+1}/{len(grouped)}: {seq_key}")
            
            sign = sequence_df['sign'].iloc[0]
            result = self.process_sequence(sequence_df, sign)
            
            if result is not None:
                features, label = result
                X.append(features)
                y.append(label)
                successful_sequences += 1
            else:
                failed_sequences += 1
                if failed_sequences <= 5:  # Show first few failures
                    print(f"â�Œ Failed to process sequence {seq_key}")
        
        final_X = np.array(X) if X else np.array([])
        final_y = np.array(y) if y else np.array([])
        
        print(f"\nğŸ“Š Dataset processing complete!")
        print(f"âœ… Successfully processed: {successful_sequences} sequences")
        print(f"â�Œ Failed to process: {failed_sequences} sequences")
        print(f"ğŸ“ˆ Success rate: {successful_sequences/(successful_sequences+failed_sequences)*100:.1f}%")
        
        return final_X, final_y

# Model Components
class ECA(tf.keras.layers.Layer):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same", use_bias=False)

    def call(self, inputs, mask=None):
        nn = tf.keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)
        nn = tf.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        nn = nn[:,None,:]
        return inputs * nn

class LateDropout(tf.keras.layers.Layer):
    def __init__(self, rate, noise_shape=None, start_step=0, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.rate = rate
        self.start_step = start_step
        self.dropout = tf.keras.layers.Dropout(rate, noise_shape=noise_shape)
      
    def build(self, input_shape):
        super().build(input_shape)
        agg = tf.VariableAggregation.ONLY_FIRST_REPLICA
        self._train_counter = tf.Variable(0, dtype="int64", aggregation=agg, trainable=False)

    def call(self, inputs, training=False):
        x = tf.cond(self._train_counter < self.start_step, lambda:inputs, lambda:self.dropout(inputs, training=training))
        if training:
            self._train_counter.assign_add(1)
        return x

class CausalDWConv1D(tf.keras.layers.Layer):
    def __init__(self, kernel_size=17, dilation_rate=1, use_bias=False, depthwise_initializer='glorot_uniform', name='', **kwargs):
        super().__init__(name=name,**kwargs)
        self.causal_pad = tf.keras.layers.ZeroPadding1D((dilation_rate*(kernel_size-1),0),name=name + '_pad')
        self.dw_conv = tf.keras.layers.DepthwiseConv1D(
                            kernel_size, strides=1, dilation_rate=dilation_rate, padding='valid',
                            use_bias=use_bias, depthwise_initializer=depthwise_initializer, name=name + '_dwconv')
        self.supports_masking = True
        
    def call(self, inputs):
        x = self.causal_pad(inputs)
        x = self.dw_conv(x)
        return x

def Conv1DBlock(channel_size, kernel_size, dilation_rate=1, drop_rate=0.0, expand_ratio=2, activation='swish', name=None):
    if name is None:
        name = str(tf.keras.backend.get_uid("mbblock"))
    
    def apply(inputs):
        channels_in = tf.keras.backend.int_shape(inputs)[-1]
        channels_expand = channels_in * expand_ratio
        skip = inputs

        x = tf.keras.layers.Dense(channels_expand, use_bias=True, activation=activation, name=name + '_expand_conv')(inputs)
        x = CausalDWConv1D(kernel_size, dilation_rate=dilation_rate, use_bias=False, name=name + '_dwconv')(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.95, name=name + '_bn')(x)
        x = ECA()(x)
        x = tf.keras.layers.Dense(channel_size, use_bias=True, name=name + '_project_conv')(x)

        if drop_rate > 0:
            x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1), name=name + '_drop')(x)

        if (channels_in == channel_size):
            x = tf.keras.layers.add([x, skip], name=name + '_add')
        return x

    return apply

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.scale = self.dim ** -0.5
        self.num_heads = num_heads
        self.qkv = tf.keras.layers.Dense(3 * dim, use_bias=False)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.proj = tf.keras.layers.Dense(dim, use_bias=False)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        qkv = self.qkv(inputs)
        qkv = tf.keras.layers.Permute((2, 1, 3))(tf.keras.layers.Reshape((-1, self.num_heads, self.dim * 3 // self.num_heads))(qkv))
        q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)

        attn = tf.matmul(q, k, transpose_b=True) * self.scale

        if mask is not None:
            mask = mask[:, None, None, :]

        attn = tf.keras.layers.Softmax(axis=-1)(attn, mask=mask)
        attn = self.drop1(attn)

        x = attn @ v
        x = tf.keras.layers.Reshape((-1, self.dim))(tf.keras.layers.Permute((2, 1, 3))(x))
        x = self.proj(x)
        return x

def TransformerBlock(dim=256, num_heads=4, expand=4, attn_dropout=0.2, drop_rate=0.2, activation='swish'):
    def apply(inputs):
        x = inputs
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = MultiHeadSelfAttention(dim=dim,num_heads=num_heads,dropout=attn_dropout)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
        x = tf.keras.layers.Add()([inputs, x])
        attn_out = x

        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = tf.keras.layers.Dense(dim*expand, use_bias=False, activation=activation)(x)
        x = tf.keras.layers.Dense(dim, use_bias=False)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
        x = tf.keras.layers.Add()([attn_out, x])
        return x
    return apply

def get_model(max_len=384, dropout_step=0, dim=192, num_classes=None):
    inp = tf.keras.Input((max_len, CHANNELS))
    x = tf.keras.layers.Masking(mask_value=PAD, input_shape=(max_len, CHANNELS))(inp)
    ksize = 17
    x = tf.keras.layers.Dense(dim, use_bias=False, name='stem_conv')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.95, name='stem_bn')(x)

    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = TransformerBlock(dim, expand=2)(x)

    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = TransformerBlock(dim, expand=2)(x)

    if dim == 384:
        x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
        x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
        x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
        x = TransformerBlock(dim, expand=2)(x)

        x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
        x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
        x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
        x = TransformerBlock(dim, expand=2)(x)

    x = tf.keras.layers.Dense(dim*2, activation=None, name='top_conv')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = LateDropout(0.8, start_step=dropout_step)(x)
    x = tf.keras.layers.Dense(num_classes, name='classifier')(x)
    return tf.keras.Model(inp, x)

class CFG:
    n_splits = 5
    save_output = True
    output_dir = './models'
    
    seed = 42
    verbose = 1  # Reduced verbosity to minimize output
    
    max_len = 384
    replicas = 2  # Updated for T4 x2
    lr = 5e-4 * replicas
    weight_decay = 0.1
    lr_min = 1e-6
    epoch = 50
    warmup = 0
    batch_size = 32 * replicas  # Increased batch size for dual GPUs
    
    fp16 = True
    dropout_start_epoch = 15
    resume = 0
    decay_type = 'cosine'
    dim = 192
    comment = f'islr-fp16-192-seed{seed}-t4x2'

def create_dataset(X, y, batch_size=32, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def train_model(data_input, config: CFG = None):
    """
    Train the sign language model.
    
    Args:
        data_input: Either a string (CSV file path) or pandas DataFrame
        config: Configuration object
    """
    if config is None:
        config = CFG()
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load and check data structure
    print("Loading and processing data...")
    
    # Handle both DataFrame and file path inputs
    if isinstance(data_input, pd.DataFrame):
        df = data_input.copy()
        print("Using provided DataFrame")
    elif isinstance(data_input, str):
        df = pd.read_csv(data_input)
        print(f"Loaded CSV from: {data_input}")
    else:
        raise ValueError("data_input must be either a pandas DataFrame or a file path string")
    
    print(f"Dataset columns: {df.columns.tolist()}")
    print(f"Dataset shape: {df.shape}")
    
    # Check for required columns
    required_cols = ['frame', 'row_id', 'x', 'y', 'z', 'sign']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        print("Please ensure your CSV has these columns: frame, row_id, x, y, z, sign")
        return None, None, None
    
    processor = SignLanguageProcessor()
    
    # Create training function within strategy scope for multi-GPU
    def train_step():
        print("\nğŸ”¥ Starting data processing within training scope...")
        X, y = processor.process_dataset(df)
        
        print(f"\nğŸ“Š Final processed dataset:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Number of classes: {processor.sign_count}")
        print(f"  Samples per class: {[np.sum(y.argmax(axis=1) == i) for i in range(min(5, processor.sign_count))]}")
        
        if len(X) == 0:
            print("â�Œ No valid sequences processed! Cannot train model.")
            return None, None
        
        # Split data
        print(f"\nâœ‚ï¸� Splitting data (80% train, 20% validation)...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=config.seed, stratify=y.argmax(axis=1)
        )
        
        print(f"  Training set: {X_train.shape}")
        print(f"  Validation set: {X_val.shape}")
        
        # Create datasets
        print(f"\nğŸ“¦ Creating TensorFlow datasets (batch size: {config.batch_size})...")
        train_ds = create_dataset(X_train, y_train, config.batch_size, shuffle=True)
        val_ds = create_dataset(X_val, y_val, config.batch_size, shuffle=False)
        
        # Build model
        print(f"\nğŸ�—ï¸� Building model (dim: {config.dim}, max_len: {config.max_len})...")
        model = get_model(
            max_len=config.max_len,
            dropout_step=config.dropout_start_epoch,
            dim=config.dim,
            num_classes=processor.sign_count
        )
        
        print(f"Model parameters: {model.count_params():,}")
        
        # Compile model
        print(f"\nâš™ï¸� Compiling model (lr: {config.lr}, weight_decay: {config.weight_decay})...")
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=config.lr,
            weight_decay=config.weight_decay
        )
        
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'{config.output_dir}/{config.comment}-best.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=config.lr_min,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Train model
        print(f"\nğŸš€ Starting training ({config.epoch} epochs)...")
        print("=" * 50)
        history = model.fit(
            train_ds,
            epochs=config.epoch,
            callbacks=callbacks,
            validation_data=val_ds,
            verbose=config.verbose
        )
        
        # Load best weights
        print(f"\nğŸ“¥ Loading best weights...")
        model.load_weights(f'{config.output_dir}/{config.comment}-best.h5')
        
        # Evaluate
        print(f"\nğŸ“ˆ Final evaluation...")
        val_loss, val_acc = model.evaluate(val_ds, verbose=0)
        print(f"ğŸ�¯ Final validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        
        return model, historyprint(f"Number of classes: {processor.sign_count}")
        print(f"Number of classes: {processor.sign_count}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=config.seed, stratify=y.argmax(axis=1)
        )
        
        # Create datasets
        train_ds = create_dataset(X_train, y_train, config.batch_size, shuffle=True)
        val_ds = create_dataset(X_val, y_val, config.batch_size, shuffle=False)
        
        # Build model
        model = get_model(
            max_len=config.max_len,
            dropout_step=config.dropout_start_epoch,
            dim=config.dim,
            num_classes=processor.sign_count
        )
        
        # Compile model
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=config.lr,
            weight_decay=config.weight_decay
        )
        
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'{config.output_dir}/{config.comment}-best.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=0  # Suppress checkpoint messages
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=config.lr_min,
                verbose=0
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=0
            )
        ]
        
        # Train model
        print("Starting training...")
        history = model.fit(
            train_ds,
            epochs=config.epoch,
            callbacks=callbacks,
            validation_data=val_ds,
            verbose=config.verbose
        )
        
        # Load best weights
        model.load_weights(f'{config.output_dir}/{config.comment}-best.h5')
        
        # Evaluate
        val_loss, val_acc = model.evaluate(val_ds, verbose=0)
        print(f"Final validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        
        return model, history
    
    # Execute training within strategy scope if available
    print(f"\nğŸ�® GPU Strategy: {'Multi-GPU' if strategy else 'Single GPU/CPU'}")
    
    if strategy:
        print(f"Running with {strategy.num_replicas_in_sync} replicas")
        with strategy.scope():
            model, history = train_step()
    else:
        print("Running without distributed strategy")
        model, history = train_step()
    
    if model is None:
        print("â�Œ Training failed - no model returned")
        return None, None, None
    
    print(f"\nğŸ�‰ Training completed successfully!")
    return model, processor, history

# Usage example with GPU optimization:
if __name__ == "__main__":
    # Print GPU information
    print("=" * 50)
    print("GPU Configuration:")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs available: {len(tf.config.experimental.list_physical_devices('GPU'))}")
    for i, gpu in enumerate(tf.config.experimental.list_physical_devices('GPU')):
        print(f"  GPU {i}: {gpu}")
    print("=" * 50)
    
    # Train the model with optimized configuration
    config = CFG()
    
    # Example usage with DataFrame (your case):
    # model, processor, history = train_model(merged_df, config)
    
    # Example usage with CSV file:
    # model, processor, history = train_model('train_landmark_files_2018.csv', config)
    
    # For demonstration, using CSV file path:
    model, processor, history = train_model(merged_df, config)
    
    # Save processor for inference
    import pickle
    with open(f'{config.output_dir}/processor.pkl', 'wb') as f:
        pickle.dump(processor, f)
    
    print(f"Model and processor saved to {config.output_dir}/")




