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


FAST =  True

# %%
if FAST:
    TRAINING_EPOCHS = 2
    FOLDS = 2
    MAX_SAMPLES = 100
    BEAM_WIDTH = 2
    LENGTH_PENALTY = 0.8
    param_grid = {
    'learning_rate': [1e-4],  # [1e-4,5e-4]
    'batch_size': [32]
}
    best_params = param_grid
else:  
    # Full training
    TRAINING_EPOCHS = 20
    FOLDS = 5
    MAX_SAMPLES = None
    BEAM_WIDTH = 10
    LENGTH_PENALTY = 0.8
    param_grid = {
    'learning_rate': [1e-4,5e-4],  # [1e-4,5e-4]
    'batch_size': [32]
}


# Enable mixed precision for faster training
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')



# %%
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


# %%



# Hugging Face tokenizers (fast Rust-backed BPE)
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import numpy as np

# Prepare corpus: list of InChI strings without the prefix
INCHI_PREFIX = "InChI=1S/"
all_inchi = train_labels['InChI'].tolist()
all_inchi_no_prefix = [s[len(INCHI_PREFIX):] if s.startswith(INCHI_PREFIX) else s for s in all_inchi]

# ------------- Train tokenizer -------------
BPE_VOCAB_TARGET = 800

# Initialize a BPE tokenizer (Rust-backed)
tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
# Byte-level pre-tokenizer works well for continuous chemical strings
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

# Trainer: specify vocab size and special tokens you want present
trainer = trainers.BpeTrainer(
    vocab_size=BPE_VOCAB_TARGET,
    special_tokens=["<PAD>", "<START>", "<END>", "<UNK>"]
)

# Train from the in-memory iterator (very fast)
tokenizer.train_from_iterator(all_inchi_no_prefix, trainer=trainer)

# Save tokenizer for reuse
tokenizer.save("inchi_bpe.json")
print("Tokenizer trained and saved to inchi_bpe.json")




# ------------- Vocab and ids -------------
vocab = tokenizer.get_vocab()          # dict: token -> id (fast)
# Get ids for special tokens (guaranteed to exist because trainer added them)
PAD_ID   = tokenizer.token_to_id("<PAD>")
START_ID = tokenizer.token_to_id("<START>")
END_ID   = tokenizer.token_to_id("<END>")
UNK_ID   = tokenizer.token_to_id("<UNK>")

print("Vocab size (from tokenizer):", len(vocab))
print("PAD/START/END/UNK ids:", PAD_ID, START_ID, END_ID, UNK_ID)

# ------------- Fast statistics (batch) -------------
# Encode corpus in batch (very fast). We disable adding special tokens here.

BATCH_SIZE = 2048
encodings = []            # will hold list-of-lists: one list of token ids per string

for start in tqdm(range(0, len(all_inchi_no_prefix), BATCH_SIZE), desc="Encoding batches"):
    batch = all_inchi_no_prefix[start:start + BATCH_SIZE]
    # encode_batch returns a list of Encoding objects (each has .ids)
    encs = tokenizer.encode_batch(batch, add_special_tokens=False)
    encodings.extend([enc for enc in encs])

token_lengths = [len(enc.ids) for enc in encodings]
percentile_95_length = int(np.percentile(token_lengths, 95))
print(f"95th percentile token length: {percentile_95_length}")
print(f"min/max/mean: {min(token_lengths)}/{max(token_lengths)}/{np.mean(token_lengths):.1f}")

# ------------- Helpers for training (teacher forcing) -------------
# We will produce lists of token ids compatible with tokenizer.decode(ids)
def remove_inchi_prefix(inchi):
    return inchi[len(INCHI_PREFIX):] if inchi.startswith(INCHI_PREFIX) else inchi

def encode_inchi_input(inchi, max_length):
    """
    Returns a list of token ids representing: <START> + tokens (no <END>), padded/truncated to max_length.
    """
    s = remove_inchi_prefix(inchi)
    enc = tokenizer.encode(s, add_special_tokens=False)
    ids = [START_ID] + enc.ids  # teacher forcing input starts with <START>
    # pad / truncate
    if len(ids) < max_length:
        ids = ids + [PAD_ID] * (max_length - len(ids))
    else:
        ids = ids[:max_length]
    return ids

def encode_inchi_target(inchi, max_length):
    """
    Returns a list of token ids: tokens + <END>, padded/truncated to max_length.
    """
    s = remove_inchi_prefix(inchi)
    enc = tokenizer.encode(s, add_special_tokens=False)
    ids = enc.ids + [END_ID]
    if len(ids) < max_length:
        ids = ids + [PAD_ID] * (max_length - len(ids))
    else:
        ids = ids[:max_length]
    return ids

def decode_inchi(id_list):
    """
    Decode id_list to string (without InChI prefix).
    Removes PAD and START, stops at END (if present).
    """
    # trim at END if present
    if END_ID in id_list:
        id_list = id_list[: id_list.index(END_ID)]
    # remove start/pad tokens if present
    id_list = [i for i in id_list if i not in (PAD_ID, START_ID)]
    # tokenizer.decode expects ids referencing the tokenizer's vocab
    return tokenizer.decode(id_list, skip_special_tokens=True)

def add_inchi_prefix(s_no_prefix):
    return s_no_prefix if s_no_prefix.startswith(INCHI_PREFIX) else INCHI_PREFIX + s_no_prefix

# ------------- Example sanity check -------------
test_inchi = train_labels.iloc[0]['InChI']
max_len = percentile_95_length + 2   # a small safety margin
inp_ids = encode_inchi_input(test_inchi, max_len)
tgt_ids = encode_inchi_target(test_inchi, max_len)
decoded = decode_inchi(tgt_ids)
decoded_with_prefix = add_inchi_prefix(decoded)

print("Original:", test_inchi[:120], "...")
print("Decoded(with prefix):", decoded_with_prefix[:120], "...")
print("Match (no prefix):", remove_inchi_prefix(test_inchi) == decoded)
print("Match (with prefix):", test_inchi == decoded_with_prefix)



IMAGE_SIZE = (384, 384)
MAX_INCHI_LENGTH = percentile_95_length  # Use 95th percentile length (post-prefix removal)

if MAX_SAMPLES:
    train_labels_subset = train_labels.head(MAX_SAMPLES).copy()
else:
    train_labels_subset = train_labels.copy()


if START_ID is None or END_ID is None or PAD_ID is None:
    raise ValueError("Make sure the tokenizer JSON contains PAD/START/END tokens or pass them when loading.")

# ensure subset is a copy to avoid pandas SettingWithCopyWarning
df = train_labels_subset.copy()

# Remove prefix quickly (vectorized)
INCHI_PREFIX = "InChI=1S/"
texts = df['InChI'].str.replace(INCHI_PREFIX, '', regex=False).tolist()

# Safety check
if MAX_INCHI_LENGTH < 2:
    raise ValueError("MAX_INCHI_LENGTH must be at least 2 to allow adding START/END tokens.")

# We will encode tokens with length (MAX_INCHI_LENGTH - 1), then prepend START (for input)
# and append END (for target) to reach MAX_INCHI_LENGTH.
enc_max_len = MAX_INCHI_LENGTH - 1

# Batch encode once: add_special_tokens=False because we will add START/END manually.
enc = tokenizer(
    texts,
    add_special_tokens=False,
    padding="max_length",
    truncation=True,
    max_length=enc_max_len,
    return_tensors="np"   # returns numpy arrays for fast vector ops
)

# enc["input_ids"] shape == (N, enc_max_len)
input_ids_base = enc["input_ids"].astype(np.int32)   # shape (N, L-1)
batch_size = input_ids_base.shape[0]

# Create inputs: prepend START_ID (column of shape (N,1)) -> shape (N, MAX_INCHI_LENGTH)
start_col = np.full((batch_size, 1), START_ID, dtype=np.int32)
encoded_inputs_np = np.concatenate([start_col, input_ids_base], axis=1)

# Create targets: append END_ID -> shape (N, MAX_INCHI_LENGTH)
end_col = np.full((batch_size, 1), END_ID, dtype=np.int32)
encoded_targets_np = np.concatenate([input_ids_base, end_col], axis=1)

# Optional: sanity - ensure PAD_ID is used where tokenizer padded earlier (it used pad_token_id)
# If you want to guarantee pad token value, you can replace tokenizer pad id if needed.

# Assign back into DataFrame as lists (fast)
df['encoded_inchi_input'] = list(encoded_inputs_np.tolist())
df['encoded_inchi_target'] = list(encoded_targets_np.tolist())

# If you want the arrays instead of lists:
# df['encoded_inchi_input_np'] = list(encoded_inputs_np)  # stores numpy arrays per cell - but lists are typically easier

# Replace original variable if you want
train_labels_subset = df



# %%
# Step 4: Image Preprocessing and Data Preparation
# Prepare a smaller subset for faster training (use first 10000 samples)
# For full training, remove the subset limitation

print(f'Using {len(train_labels_subset)} samples for training')

# Add InChI length (without prefix) for length-bucketed sampling
train_labels_subset['inchi_length'] = train_labels_subset['InChI'].apply(
    lambda x: len(remove_inchi_prefix(x))
)

# Prepare stratification bins by length for K-Fold
num_bins = min(20, max(2, int(np.sqrt(len(train_labels_subset)))))
train_labels_subset['length_bin'] = pd.qcut(train_labels_subset['inchi_length'], q=num_bins, labels=False, duplicates='drop')

# For compatibility with earlier code paths, also create a single split preview (fold 0)
train_df, val_df = train_test_split(
    train_labels_subset,
    test_size=0.1,
    random_state=42,
    stratify=train_labels_subset['length_bin']
)

print(f'Train samples (preview split): {len(train_df)}')
print(f'Validation samples (preview split): {len(val_df)}')
print(f'InChI length distribution (train preview):')
print(f'  Min: {train_df["inchi_length"].min()}, Max: {train_df["inchi_length"].max()}')
print(f'  Mean: {train_df["inchi_length"].mean():.1f}, Median: {train_df["inchi_length"].median():.1f}')

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def letterbox_resize(img, target_size):
    """
    Resize image with letterbox (preserve aspect ratio, pad to square)
    Args:
        img: Input image (H, W, C)
        target_size: Tuple (target_height, target_width)
    Returns:
        Resized and padded image
    """
    h, w = img.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scale to fit within target while preserving aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(img, (new_w, new_h))
    
    # Create padded image (gray padding)
    padded = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    
    # Calculate padding offsets to center the image
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    # Place resized image in center
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded

def to_tri_channel(img_rgb):
    """
    Build tri-channel line-art stack:
      Ch1: grayscale original
      Ch2: adaptive-binarized map (OTSU)
      Ch3: edge map (Canny)
    Output: HxWx3 uint8
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # OTSU binarization
    _, bin_map = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Canny edges
    edges = cv2.Canny(gray, 50, 150)
    stacked = np.stack([gray, bin_map, edges], axis=-1)
    return stacked

def _random_erasure(img, max_h_frac=0.1, max_w_frac=0.1):
    h, w = img.shape[:2]
    erase_h = int(np.random.uniform(0.02, max_h_frac) * h)
    erase_w = int(np.random.uniform(0.02, max_w_frac) * w)
    y0 = np.random.randint(0, max(1, h - erase_h))
    x0 = np.random.randint(0, max(1, w - erase_w))
    img[y0:y0+erase_h, x0:x0+erase_w] = 128
    return img

def augment_image(img, augment=True):
    """
    Apply safe augmentations: rotate (±3°), scale (±5%), translate (±4%), 
    brightness/contrast jitter. No flips.
    """
    if not augment:
        return img
    
    h, w = img.shape[:2]
    
    # Random rotation (±3 degrees)
    angle = np.random.uniform(-3, 3)
    
    # Random scale (±5%)
    scale = np.random.uniform(0.95, 1.05)
    
    # Random translation (±4%)
    tx = np.random.uniform(-0.04, 0.04) * w
    ty = np.random.uniform(-0.04, 0.04) * h
    
    # Rotation and scale matrix
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    
    # Apply affine transformation
    img = cv2.warpAffine(img, M, (w, h), borderValue=(128, 128, 128))

    # Light morphological line-width jitter (low probability)
    if np.random.rand() < 0.2:
        k = np.random.choice([3, 5])
        kernel = np.ones((k, k), np.uint8)
        if np.random.rand() < 0.5:
            img = cv2.dilate(img, kernel, iterations=1)
        else:
            img = cv2.erode(img, kernel, iterations=1)

    # Micro-erase small random patches (very low probability)
    if np.random.rand() < 0.2:
        img = _random_erasure(img)
    
    # Brightness and contrast jitter
    brightness = np.random.uniform(0.9, 1.1)
    contrast = np.random.uniform(0.9, 1.1)
    
    img = img.astype(np.float32)
    img = img * contrast + (brightness - 1) * 128
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Slight blur or sharpen (low probability)
    if np.random.rand() < 0.2:
        if np.random.rand() < 0.5:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        else:
            # Simple unsharp masking
            blur = cv2.GaussianBlur(img, (3, 3), 0)
            img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    
    return img

def preprocess_image(image_path, augment=False):
    """Load and preprocess image to 384x384 tri-channel with optional augmentation"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Letterbox resize (preserve aspect ratio)
        img = letterbox_resize(img, IMAGE_SIZE)

        # Build tri-channel stack (gray, binarized, edges)
        img = to_tri_channel(img)

        # Apply augmentations if training
        img = augment_image(img, augment=augment)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return img
    except:
        return None

print('\nData preparation complete!')

# %%


# %%
# Step 5: Create Data Generator with Teacher Forcing
class LengthBucketedDataGenerator(keras.utils.Sequence):
    """
    Data generator with length-bucketed sampling
    Groups sequences of similar lengths together to minimize padding waste
    """
    def __init__(self, dataframe, batch_size=32, shuffle=True, augment=False, num_buckets=10):
        self.dataframe = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.num_buckets = num_buckets
        
        # Create length buckets
        self._create_buckets()
        self.on_epoch_end()
    
    def _create_buckets(self):
        """Create buckets based on InChI length"""
        # Sort by length
        self.dataframe = self.dataframe.sort_values('inchi_length').reset_index(drop=True)
        
        # Calculate bucket boundaries
        lengths = self.dataframe['inchi_length'].values
        self.bucket_boundaries = np.percentile(
            lengths, 
            np.linspace(0, 100, self.num_buckets + 1)
        )
        
        # Assign each sample to a bucket
        self.dataframe['bucket'] = pd.cut(
            self.dataframe['inchi_length'], 
            bins=self.bucket_boundaries, 
            labels=False, 
            include_lowest=True
        )
        
        print(f'Created {self.num_buckets} length buckets:')
        for bucket_id in range(self.num_buckets):
            bucket_data = self.dataframe[self.dataframe['bucket'] == bucket_id]
            if len(bucket_data) > 0:
                print(f'  Bucket {bucket_id}: {len(bucket_data)} samples, '
                      f'length range [{bucket_data["inchi_length"].min():.0f}, '
                      f'{bucket_data["inchi_length"].max():.0f}]')
        
        # Create batches from buckets
        self._create_batches()
    
    def _create_batches(self):
        """Create batches from buckets"""
        self.batches = []
        
        for bucket_id in range(self.num_buckets):
            bucket_indices = self.dataframe[self.dataframe['bucket'] == bucket_id].index.tolist()
            
            # Create batches from this bucket
            for i in range(0, len(bucket_indices), self.batch_size):
                batch_indices = bucket_indices[i:i + self.batch_size]
                self.batches.append(batch_indices)
        
        self.batches = np.array(self.batches, dtype=object)
    
    def __len__(self):
        return len(self.batches)
    
    def __getitem__(self, index):
        # Get batch indexes
        batch_indexes = self.batches[index]
        
        # Get batch data
        images = []
        decoder_inputs = []
        targets = []
        sample_weights = []
        
        for idx in batch_indexes:
            row = self.dataframe.iloc[idx]
            img = preprocess_image(row['image_path'], augment=self.augment)
            if img is not None:
                images.append(img)
                di = row['encoded_inchi_input']
                tg = row['encoded_inchi_target']
                decoder_inputs.append(di)
                targets.append(tg)
                # PAD mask: 1 for non-PAD, 0 for PAD
                pad_id = char_to_idx['<PAD>']
                mask = (np.array(tg, dtype=np.int32) != pad_id).astype(np.float32)
                sample_weights.append(mask)
        
        if len(images) == 0:
            print('error loading the data.')
            # Return dummy batch if all images failed to load
            return ({
                'image_input': np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32),
                'decoder_input': np.zeros((1, MAX_INCHI_LENGTH), dtype=np.int32)
            }, np.zeros((1, MAX_INCHI_LENGTH), dtype=np.int32), np.ones((1, MAX_INCHI_LENGTH), dtype=np.float32))
        
        return ({
            'image_input': np.array(images, dtype=np.float32),
            'decoder_input': np.array(decoder_inputs, dtype=np.int32)
        }, np.array(targets, dtype=np.int32), np.array(sample_weights, dtype=np.float32))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.batches)


# Standard Data Generator (for validation/test - no bucketing needed)
class DataGenerator(keras.utils.Sequence):
    def __init__(self, dataframe, batch_size=32, shuffle=False, augment=False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
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
        sample_weights = []
        
        for idx in batch_indexes:
            row = self.dataframe.iloc[idx]
            img = preprocess_image(row['image_path'], augment=self.augment)
            if img is not None:
                images.append(img)
                di = row['encoded_inchi_input']
                tg = row['encoded_inchi_target']
                decoder_inputs.append(di)
                targets.append(tg)
                pad_id = char_to_idx['<PAD>']
                mask = (np.array(tg, dtype=np.int32) != pad_id).astype(np.float32)
                sample_weights.append(mask)
        
        if len(images) == 0:
            print('error loading the data.')
            # Return dummy batch if all images failed to load
            return ({
                'image_input': np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32),
                'decoder_input': np.zeros((1, MAX_INCHI_LENGTH), dtype=np.int32)
            }, np.zeros((1, MAX_INCHI_LENGTH), dtype=np.int32), np.ones((1, MAX_INCHI_LENGTH), dtype=np.float32))
        
        return ({
            'image_input': np.array(images, dtype=np.float32),
            'decoder_input': np.array(decoder_inputs, dtype=np.int32)
        }, np.array(targets, dtype=np.int32), np.array(sample_weights, dtype=np.float32))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

print('Data generators created successfully (with length-bucketed sampling for training)')





# Step 6: Build PROPER Encoder-Decoder Model with Teacher Forcing
def build_model(vocab_size, max_length, learning_rate=1e-4, weight_decay=1e-2):
    # IMAGE ENCODER: EfficientNet-B0 pretrained on ImageNet
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(256, 256, 3),
        pooling='avg'
    )
    
    # Fine-tune the last layers
    base_model.trainable = True
    
    # Image input
    image_input = layers.Input(shape=(256, 256, 3), name='image_input')
    
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
    
    # Compile model with AdamW optimizer and gradient clipping
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        clipnorm=1.0  # Gradient clipping
    )
    
    model.compile(
        optimizer=optimizer,
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
                
                # Decode and add prefix back for fair comparison
                pred_str = decode_inchi(decoder_input[0])
                pred_str_with_prefix = add_inchi_prefix(pred_str)
                predictions.append(pred_str_with_prefix)
                ground_truths.append(row['InChI'])  # Original InChI with prefix
        
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


# Step 7.5: Custom Learning Rate Schedule with Warmup + Cosine Decay
class WarmupCosineDecaySchedule(keras.callbacks.Callback):
    """
    Learning rate schedule with warmup and cosine decay
    - Warmup: Linear increase for warmup_epochs
    - Cosine decay: After warmup, decay to min_lr using cosine schedule
    """
    def __init__(self, initial_lr, warmup_epochs, total_epochs, min_lr=1e-7):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay after warmup
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        # Set learning rate
        self.model.optimizer.learning_rate.assign(lr)
        print(f'\n  Learning Rate: {lr:.6f}')


# Teacher Forcing Schedule Callback
class TeacherForcingSchedule(keras.callbacks.Callback):
    """
    Gradually reduce teacher forcing ratio from 1.0 to target_ratio
    Schedule: 1.0 → 0.7 by 60% of training
    """
    def __init__(self, total_epochs, target_ratio=0.7, decay_point=0.6):
        super().__init__()
        self.total_epochs = total_epochs
        self.target_ratio = target_ratio
        self.decay_point = decay_point
        self.current_ratio = 1.0
        
    def on_epoch_begin(self, epoch, logs=None):
        progress = epoch / self.total_epochs
        
        if progress <= self.decay_point:
            # Linear decay to target_ratio by decay_point
            self.current_ratio = 1.0 - (1.0 - self.target_ratio) * (progress / self.decay_point)
        else:
            # Stay at target_ratio after decay_point
            self.current_ratio = self.target_ratio
        
        print(f'\n  Teacher Forcing Ratio: {self.current_ratio:.2f}')


print('Custom learning rate and teacher forcing schedules defined')




# Step 8: Hyperparameter Grid Search Training
# Grid size can be increased when we have more compute.

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
        
        # Create data generators with augmentation for training
        # Use length-bucketed sampling for training (better efficiency)
        train_gen = LengthBucketedDataGenerator(train_df, batch_size=bs, shuffle=True, augment=True, num_buckets=10)
        val_gen = DataGenerator(val_df, batch_size=bs, shuffle=False, augment=False)
        
        # Callbacks with TRUE Mean Levenshtein distance monitoring
        mean_levenshtein_callback = MeanLevenshteinCallback(
            validation_data=val_gen,
            val_df=val_df,
            max_length=MAX_INCHI_LENGTH
        )
        
        # Early stopping with patience 4 as per improvement.md
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_mean_levenshtein',  # Monitor Levenshtein distance instead of loss
            patience=4,  # Changed from 3 to 4
            restore_best_weights=True,
            mode='min'  # Lower distance is better
        )
        
        # ModelCheckpoint to save top-3 checkpoints
        checkpoint_dir = 'checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f'model_lr{lr}_bs{bs}_epoch{{epoch:02d}}_lev{{val_mean_levenshtein:.2f}}.h5'),
            monitor='val_mean_levenshtein',
            mode='min',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        
        # Warmup + Cosine Decay LR Schedule (3-epoch warmup as per improvement.md)
        lr_schedule = WarmupCosineDecaySchedule(
            initial_lr=lr,
            warmup_epochs=3,
            total_epochs=TRAINING_EPOCHS,
            min_lr=1e-7
        )
        
        # Teacher Forcing Schedule (1.0 → 0.7 by 60% of training)
        teacher_forcing_schedule = TeacherForcingSchedule(
            total_epochs=TRAINING_EPOCHS,
            target_ratio=0.7,
            decay_point=0.6
        )
        
        # Train model with all callbacks
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=TRAINING_EPOCHS,
            callbacks=[
                mean_levenshtein_callback,
                early_stopping,
                model_checkpoint,
                lr_schedule,
                teacher_forcing_schedule
            ],
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

# Create data generator for combined data with augmentation
# Use length-bucketed sampling for better efficiency
full_df_gen = LengthBucketedDataGenerator(full_df, batch_size=best_params['batch_size'], shuffle=True, augment=True, num_buckets=10)

# ModelCheckpoint for final training
final_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, f'final_model_epoch{{epoch:02d}}_loss{{loss:.4f}}.h5'),
    monitor='loss',
    mode='min',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# Learning rate schedule for final training
final_lr_schedule = WarmupCosineDecaySchedule(
    initial_lr=best_params['learning_rate'],
    warmup_epochs=3,
    total_epochs=TRAINING_EPOCHS,
    min_lr=1e-7
)

# Teacher forcing schedule for final training
final_teacher_forcing = TeacherForcingSchedule(
    total_epochs=TRAINING_EPOCHS,
    target_ratio=0.7,
    decay_point=0.6
)

# Train on combined data (no validation split)
print(f'Training with best hyperparameters: {best_params}')

final_history = final_model.fit(
    full_df_gen,
    epochs=TRAINING_EPOCHS,
    callbacks=[final_checkpoint, final_lr_schedule, final_teacher_forcing],
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

# BEAM SEARCH PREDICTION FUNCTION
def beam_search_decode(model, image, beam_width=5, max_length=275, length_penalty=0.7, add_prefix=True):
    """
    Generate InChI string using beam search decoding
    
    Args:
        model: Trained Keras model
        image: Preprocessed image
        beam_width: Number of beams to maintain (default: 5)
        max_length: Maximum sequence length
        length_penalty: Length penalty factor (default: 0.7)
        add_prefix: Whether to prepend "InChI=1S/" to decoded sequence (default: True)
        
    Returns:
        Decoded InChI string (best beam) with prefix prepended
    """
    # Expand image dimensions
    img_batch = np.expand_dims(image, axis=0)
    
    # Initialize beams: each beam is (sequence, score)
    beams = [(np.array([char_to_idx['<START>']]), 0.0)]
    completed_beams = []
    
    for step in range(1, max_length):
        all_candidates = []
        
        for seq, score in beams:
            # Skip if sequence ended
            if len(seq) > 0 and (seq[-1] == char_to_idx['<END>'] or seq[-1] == char_to_idx['<PAD>']):
                completed_beams.append((seq, score))
                continue
            
            # Prepare decoder input
            decoder_input = np.zeros((1, max_length), dtype=np.int32)
            decoder_input[0, :len(seq)] = seq
            
            # Get predictions
            predictions = model.predict([img_batch, decoder_input], verbose=0)
            next_token_probs = predictions[0, len(seq) - 1, :]
            
            # Get top k tokens
            top_k_indices = np.argsort(next_token_probs)[-beam_width:]
            
            for token_idx in top_k_indices:
                # Calculate score with log probability
                token_prob = next_token_probs[token_idx]
                token_score = np.log(token_prob + 1e-10)
                
                # Apply length penalty: score / (length ** length_penalty)
                new_seq = np.append(seq, token_idx)
                new_score = score + token_score
                
                all_candidates.append((new_seq, new_score))
        
        # Select top beam_width candidates
        if len(all_candidates) == 0:
            break
        
        # Sort by score with length penalty
        all_candidates = sorted(all_candidates, 
                               key=lambda x: x[1] / (len(x[0]) ** length_penalty), 
                               reverse=True)
        beams = all_candidates[:beam_width]
        
        # Early stopping if all beams completed
        if len(completed_beams) >= beam_width:
            break
    
    # Add remaining beams to completed
    completed_beams.extend(beams)
    
    # Select best beam
    if len(completed_beams) == 0:
        return INCHI_PREFIX if add_prefix else ''
    
    best_beam = max(completed_beams, 
                    key=lambda x: x[1] / (len(x[0]) ** length_penalty))
    
    # Decode and prepend prefix
    decoded = decode_inchi(best_beam[0])
    if add_prefix:
        decoded = add_inchi_prefix(decoded)
    
    return decoded


def beam_search_batch(model, images, beam_width=5, max_length=275, length_penalty=0.7, add_prefix=True):
    """
    Batch beam search decoding for multiple images
    
    Args:
        model: Trained Keras model
        images: List or array of preprocessed images
        beam_width: Number of beams (default: 5)
        max_length: Maximum sequence length
        length_penalty: Length penalty factor (default: 0.7)
        add_prefix: Whether to prepend "InChI=1S/" to decoded sequences (default: True)
        
    Returns:
        List of decoded InChI strings (with prefix prepended)
    """
    results = []
    for img in images:
        result = beam_search_decode(model, img, beam_width, max_length, length_penalty, add_prefix)
        results.append(result)
    return results

# Make predictions on test set
print('\nGenerating predictions on test set with BEAM SEARCH decoding...')
print('Beam Search Parameters: beam_width=5, length_penalty=0.7')
print('NOTE: "InChI=1S/" prefix will be prepended to all decoded sequences')

# Step 1: Load all test images first
print('Loading test images...')
test_images = []
valid_indices = []
failed_indices = []

for idx in tqdm(range(len(test_df)), desc="Loading images"):
    image_path = test_df.iloc[idx]['image_path']
    img = preprocess_image(image_path, augment=False)  # No augmentation for test
    
    if img is not None:
        test_images.append(img)
        valid_indices.append(idx)
    else:
        failed_indices.append(idx)

print(f'Loaded {len(test_images)} images successfully, {len(failed_indices)} failed')

# Step 2: Predict with beam search
PREDICTION_BATCH_SIZE = 8  # Smaller batch for beam search (more memory intensive)
print(f'\nPredicting with beam search (batch size {PREDICTION_BATCH_SIZE})...')

predictions = []
num_batches = int(np.ceil(len(test_images) / PREDICTION_BATCH_SIZE))

for batch_idx in tqdm(range(num_batches), desc="Predicting batches"):
    start_idx = batch_idx * PREDICTION_BATCH_SIZE
    end_idx = min(start_idx + PREDICTION_BATCH_SIZE, len(test_images))
    
    batch_images = test_images[start_idx:end_idx]
    
    # Predict with beam search (beam=5, length_penalty=0.7)
    batch_predictions = beam_search_batch(
        best_model, 
        batch_images,
        beam_width=BEAM_WIDTH,
        max_length=MAX_INCHI_LENGTH,
        length_penalty=LENGTH_PENALTY,
        add_prefix=True  # Prepend "InChI=1S/" to decoded sequences
    )
    
    predictions.extend(batch_predictions)

# Debug first few predictions
print('\nFirst 5 predictions (with prefix):')
for i in range(min(5, len(predictions))):
    pred = predictions[i]
    print(f'  {i}: {pred[:100]}{"..." if len(pred) > 100 else ""}')
    if len(pred) == 0 or pred == INCHI_PREFIX:
        print(f'    WARNING: Empty or prefix-only prediction!')

# Step 3: Handle failed images and create full prediction list
full_predictions = []
valid_idx_set = set(valid_indices)

prediction_pointer = 0
for idx in range(len(test_df)):
    if idx in valid_idx_set:
        pred = predictions[prediction_pointer]
        # Fallback for empty predictions (ensure prefix is included)
        if len(pred) == 0 or pred == INCHI_PREFIX:
            pred = INCHI_PREFIX + 'C'  # Minimal valid InChI
        full_predictions.append(pred)
        prediction_pointer += 1
    else:
        # Use fallback for failed images (with prefix)
        full_predictions.append(INCHI_PREFIX + 'C')

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




