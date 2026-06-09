# ==============================================================================
# 0. SETUP AND IMPORTS
# ==============================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import random
from glob import glob
from tqdm import tqdm
import math

print(f"TensorFlow Version: {tf.__version__}")

# Ensure GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is available and memory growth is enabled.")
else:
    print("GPU not available, training will be on CPU.")

# ==============================================================================
# 1. CONFIGURATION PARAMETERS (MODIFIED FOR WEIGHTS-ONLY SAVING)
# ==============================================================================
class Config:
    # --- Control Flow ---
    # SET THIS TO TRUE TO RESUME TRAINING FROM SAVED WEIGHTS
    RESUME_TRAINING = True
    # PROVIDE THE PATH TO THE .weights.h5 FILE YOU WANT TO RESUME FROM
    # NOTE: This should now be a weights file, not a full model file.
    MODEL_PATH_TO_RESUME = "/kaggle/input/full_imagenet_tit_v5_e1/tensorflow2/default/6/tit_model_epoch_03.weights.h5"

    # --- Data and Paths (UPDATED FOR KAGGLE IMAGENET DATASET) ---
    IMAGENET_PATH = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/"
    TRAIN_DIR = os.path.join(IMAGENET_PATH, "train")
    VAL_DIR = os.path.join(IMAGENET_PATH, "val")

    SOLUTION_PATH = "/kaggle/input/imagenet-object-localization-challenge/"
    TRAIN_SOLUTION_CSV = os.path.join(SOLUTION_PATH, "LOC_train_solution.csv")
    VAL_SOLUTION_CSV = os.path.join(SOLUTION_PATH, "LOC_val_solution.csv")
    SYNSET_MAPPING_FILE = os.path.join(SOLUTION_PATH, "LOC_synset_mapping.txt")

    IMG_SIZE = 224
    NUM_CLASSES = 1000  # ImageNet has 1000 classes

    # --- Output Directories (UPDATED FOR WEIGHTS-ONLY) ---
    OUTPUT_DIR = "/kaggle/working/"
    # Checkpoints will now save only weights
    MODEL_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "tit_model_epoch_{epoch:02d}.weights.h5")
    # Final model will also be saved as weights
    FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, "tit_final_model.weights.h5")
    CLASS_MAPPING_SAVE_PATH = os.path.join(OUTPUT_DIR, "synset_to_int_mapping.json")
    TRAINING_LOG_PATH = os.path.join(OUTPUT_DIR, "trained_images_log.json")
    HISTORY_SAVE_PATH = os.path.join(OUTPUT_DIR, "training_history.json")

    # --- Training Hyperparameters ---
    BATCH_SIZE = 64
    EPOCHS = 3
    RANDOM_SEED = 42
    LEARNING_RATE = 1e-4
    DROPOUT_RATE = 0.1

    # ===============================================================
    # --- TiT Model Architecture (UNCHANGED) ---
    # ===============================================================
    INNER_PATCH_SIZE = 4
    INNER_PROJECTION_DIM = 192
    INNER_NUM_HEADS = 3
    INNER_FF_DIM = 768
    INNER_DEPTH = 2
    INNER_OUTPUT_DIM = 768
    OUTER_PATCH_SIZE = 32
    OUTER_INPUT_TOKEN_DIM = INNER_OUTPUT_DIM
    OUTER_NUM_HEADS = 6
    OUTER_FF_DIM = 3072
    OUTER_DEPTH = 4

config = Config()

# Set seeds for reproducibility
np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)
random.seed(config.RANDOM_SEED)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 2. DATA PREPARATION (UNCHANGED)
# ==============================================================================
def _load_class_mappings(config):
    """
    Loads the synset to integer mapping.
    """
    print("--- Preparing Class Mappings ---")
    synset_to_int = {}
    int_to_synset = {}
    with open(config.SYNSET_MAPPING_FILE, 'r') as f:
        for i, line in enumerate(f):
            synset = line.strip().split(' ')[0]
            synset_to_int[synset] = i
            int_to_synset[i] = synset

    if len(synset_to_int) != config.NUM_CLASSES:
        raise ValueError(f"Expected {config.NUM_CLASSES} classes, but found {len(synset_to_int)} in mapping file.")

    with open(config.CLASS_MAPPING_SAVE_PATH, 'w') as f:
        json.dump(synset_to_int, f, indent=4)
    print(f"Synset to integer mapping created with {len(synset_to_int)} classes.")
    return synset_to_int

def _load_image_lists(config, synset_to_int):
    """
    Parses the solution CSVs to create lists of image paths and their corresponding integer labels.
    """
    print("\n--- Loading File Paths and Labels from CSVs ---")
    train_df = pd.read_csv(config.TRAIN_SOLUTION_CSV)
    train_df['synset'] = train_df['PredictionString'].apply(lambda x: x.split(' ')[0])
    train_df = train_df[['ImageId', 'synset']].drop_duplicates().reset_index(drop=True)
    train_paths = [os.path.join(config.TRAIN_DIR, row.synset, f"{row.ImageId}.JPEG") for _, row in train_df.iterrows()]
    train_labels = [synset_to_int[row.synset] for _, row in train_df.iterrows()]

    val_df = pd.read_csv(config.VAL_SOLUTION_CSV)
    val_df['synset'] = val_df['PredictionString'].apply(lambda x: x.split(' ')[0])
    val_paths = [os.path.join(config.VAL_DIR, f"{row.ImageId}.JPEG") for _, row in val_df.iterrows()]
    val_labels = [synset_to_int[row.synset] for _, row in val_df.iterrows()]

    print(f"Found {len(train_paths)} training images.")
    print(f"Found {len(val_paths)} validation images.")
    return train_paths, train_labels, val_paths, val_labels

class ImageNetDataGenerator(Sequence):
    """
    Custom Keras Sequence to efficiently load, preprocess, and batch ImageNet data.
    """
    def __init__(self, image_paths, labels, batch_size, img_size, num_classes, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.is_training = is_training
        self.n = len(self.image_paths)
        self.indices = np.arange(self.n)
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(self.n / self.batch_size)

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_indices = self.indices[start_idx:end_idx]
        batch_paths = [self.image_paths[i] for i in batch_indices]
        batch_labels = [self.labels[i] for i in batch_indices]
        batch_images = np.array([self._load_and_preprocess_image(p) for p in batch_paths])
        batch_labels_one_hot = tf.keras.utils.to_categorical(batch_labels, num_classes=self.num_classes)
        return batch_images, batch_labels_one_hot

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indices)

    def _load_and_preprocess_image(self, file_path):
        img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [self.img_size, self.img_size])
        img = tf.cast(img, tf.float32) / 255.0
        return img

if os.path.exists(config.IMAGENET_PATH):
    synset_to_int_map = _load_class_mappings(config)
    train_paths, train_labels, val_paths, val_labels = _load_image_lists(config, synset_to_int_map)
    print("\n--- Creating TensorFlow Data Generators ---")
    train_generator = ImageNetDataGenerator(
        image_paths=train_paths, labels=train_labels, batch_size=config.BATCH_SIZE,
        img_size=config.IMG_SIZE, num_classes=config.NUM_CLASSES, is_training=True
    )
    val_generator = ImageNetDataGenerator(
        image_paths=val_paths, labels=val_labels, batch_size=config.BATCH_SIZE,
        img_size=config.IMG_SIZE, num_classes=config.NUM_CLASSES, is_training=False
    )
    data_available = True
else:
    print("ImageNet dataset directory not found. Skipping data preparation.")
    train_generator, val_generator = None, None
    data_available = False

# ==============================================================================
# 3. TiT MODEL DEFINITION (UNCHANGED)
# ==============================================================================
# --- Custom Layers with get_config for Serialization ---
class OuterPatches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images, sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1], padding='VALID'
        )
        patch_dims = patches.shape[-1]
        num_patches = patches.shape[1] * patches.shape[2]
        return tf.reshape(patches, [batch_size, num_patches, self.patch_size, self.patch_size, 3])

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class InnerPatches(layers.Layer):
    def __init__(self, inner_size, **kwargs):
        super().__init__(**kwargs)
        self.inner_size = inner_size

    def call(self, patch):
        batch_size = tf.shape(patch)[0]
        H, W, C = patch.shape[1], patch.shape[2], patch.shape[3]
        s = self.inner_size
        inner = tf.image.extract_patches(
            images=patch, sizes=[1, s, s, 1], strides=[1, s, s, 1],
            rates=[1, 1, 1, 1], padding='VALID'
        )
        return tf.reshape(inner, [batch_size, -1, s * s * C])

    def get_config(self):
        config = super().get_config()
        config.update({"inner_size": self.inner_size})
        return config

class ClassToken(layers.Layer):
    def __init__(self, token_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_dim = token_dim
        self.cls_token = self.add_weight(
            shape=[1, 1, self.token_dim], initializer='random_normal',
            trainable=True, name='cls_token'
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_token_broadcast = tf.tile(self.cls_token, [batch_size, 1, 1])
        return layers.Concatenate(axis=1)([cls_token_broadcast, inputs])

    def get_config(self):
        config = super().get_config()
        config.update({"token_dim": self.token_dim})
        return config

class SelectClassToken(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return inputs[:, 0, :]

    def get_config(self):
        return super().get_config()

# --- Model Builder Functions ---
def build_inner_vit(patch_size, inner_size, projection_dim, num_heads, ff_dim, depth, output_dim):
    inputs = layers.Input(shape=(patch_size, patch_size, 3))
    x = InnerPatches(inner_size)(inputs)
    x = layers.Dense(projection_dim)(x)
    x = ClassToken(projection_dim)(x)
    num_inner_tokens = ((patch_size // inner_size) ** 2) + 1
    pos_emb = layers.Embedding(input_dim=num_inner_tokens, output_dim=projection_dim)(tf.range(num_inner_tokens))
    x = x + pos_emb
    for _ in range(depth):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim // num_heads, dropout=0.1)(x1, x1)
        x2 = layers.Add()([x, attn_out])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        ff = layers.Dense(ff_dim, activation='gelu')(x3)
        ff = layers.Dense(projection_dim)(ff)
        x = layers.Add()([x2, ff])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = SelectClassToken()(x)
    outputs = layers.Dense(output_dim, name="inner_projection_to_outer_token")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="InnerViT_with_CLSToken")

def build_tit_model():
    inputs = layers.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    outer_patches = OuterPatches(config.OUTER_PATCH_SIZE)(inputs)
    inner_vit = build_inner_vit(
        patch_size=config.OUTER_PATCH_SIZE, inner_size=config.INNER_PATCH_SIZE,
        projection_dim=config.INNER_PROJECTION_DIM, num_heads=config.INNER_NUM_HEADS,
        ff_dim=config.INNER_FF_DIM, depth=config.INNER_DEPTH, output_dim=config.INNER_OUTPUT_DIM
    )
    patch_embeddings = layers.TimeDistributed(inner_vit)(outer_patches)
    x = ClassToken(config.OUTER_INPUT_TOKEN_DIM)(patch_embeddings)
    num_outer_tokens = ((config.IMG_SIZE // config.OUTER_PATCH_SIZE) ** 2) + 1
    pos_emb_outer = layers.Embedding(
        input_dim=num_outer_tokens, output_dim=config.OUTER_INPUT_TOKEN_DIM
    )(tf.range(num_outer_tokens))
    x = x + pos_emb_outer
    for _ in range(config.OUTER_DEPTH):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attn_out = layers.MultiHeadAttention(num_heads=config.OUTER_NUM_HEADS, key_dim=config.OUTER_INPUT_TOKEN_DIM // config.OUTER_NUM_HEADS, dropout=0.1)(x1, x1)
        x2 = layers.Add()([x, attn_out])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        ff = layers.Dense(config.OUTER_FF_DIM, activation='gelu')(x3)
        ff = layers.Dense(config.OUTER_INPUT_TOKEN_DIM)(ff)
        x = layers.Add()([x2, ff])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = SelectClassToken()(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation='softmax')(x)
    return keras.Model(inputs=inputs, outputs=outputs, name=f"TiT_CLS_Token_ImageNet_{config.IMG_SIZE}")

# ==============================================================================
# 4. BUILD MODEL AND LOAD WEIGHTS (REWRITTEN LOGIC)
# ==============================================================================
# This section is completely rewritten to follow the "build then load weights" strategy.
# This avoids the `load_model` serialization error.

print("\n--- Building TiT Model Architecture ---")
model = build_tit_model()

# If resuming, load the saved weights into the newly built model structure.
if config.RESUME_TRAINING:
    print(f"\n--- Path Selected: Resuming Training by Loading Weights from {config.MODEL_PATH_TO_RESUME} ---")
    if not os.path.exists(config.MODEL_PATH_TO_RESUME):
        raise FileNotFoundError(f"Model weights file not found at '{config.MODEL_PATH_TO_RESUME}'. Please check the path.")
    
    # Load only the weights. This is robust and avoids architecture deserialization.
    model.load_weights(config.MODEL_PATH_TO_RESUME)
    print("Model weights loaded successfully.")
else:
    print("\n--- Path Selected: Training from Scratch ---")

# Compile the model after it has been built and (optionally) had weights loaded.
optimizer = tf.keras.optimizers.AdamW(learning_rate=config.LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=[
        keras.metrics.CategoricalAccuracy(name="accuracy"),
        keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)

model.summary()

# ==============================================================================
# 5. DEFINE CALLBACKS AND START TRAINING (MODIFIED FOR WEIGHTS-ONLY)
# ==============================================================================
print("\n--- Defining Callbacks ---")

# CORRECTED: The ModelCheckpoint callback now saves *only weights*.
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=config.MODEL_CHECKPOINT_PATH,
    save_weights_only=True,  # This is the key change to fix the issue.
    save_freq='epoch',
    verbose=1
)

early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# --- START TRAINING ---
if data_available:
    print(f"\n--- Starting/Resuming Training for {config.EPOCHS} Epochs ---")
    history = model.fit(
        train_generator,
        epochs=config.EPOCHS,
        validation_data=val_generator,
        callbacks=[checkpoint_callback, early_stopping_callback],
        verbose=1,
    )

    # --- SAVE FINAL RESULTS ---
    print("\n--- Training Finished ---")
    
    # CORRECTED: Save the final model's weights instead of the full model.
    print(f"Saving final model weights to: {config.FINAL_MODEL_PATH}")
    model.save_weights(config.FINAL_MODEL_PATH)

    with open(config.HISTORY_SAVE_PATH, 'w') as f:
        serializable_history = {k: [float(v) for v in val] for k, val in history.history.items()}
        json.dump(serializable_history, f)
    print(f"Training history saved to: {config.HISTORY_SAVE_PATH}")
else:
    print("\n--- Skipping Training: No data generators created ---")
    history = None

# ==============================================================================
# 6. VISUALIZATION OF TRAINING RESULTS (UNCHANGED)
# ==============================================================================
if history:
    print("\n--- Plotting Training and Validation Metrics ---")
    acc, val_acc = history.history['accuracy'], history.history['val_accuracy']
    loss, val_loss = history.history['loss'], history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.savefig(os.path.join(config.OUTPUT_DIR, 'training_validation_plots.png'))
    plt.show()
else:
    print("\n--- Skipping Visualization: No training was performed ---")

# ==============================================================================
# 7. INFERENCE AND TESTING SECTION (REWRITTEN FOR WEIGHTS-ONLY)
# ==============================================================================
print("\n--- Starting Inference/Testing Example ---")

if os.path.exists(config.FINAL_MODEL_PATH) and data_available:
    try:
        # CORRECTED: Rebuild the model architecture first.
        print(f"Building model architecture for evaluation...")
        inference_model = build_tit_model()

        # CORRECTED: Load the saved weights into the new model instance.
        print(f"Loading final weights from {config.FINAL_MODEL_PATH} for evaluation...")
        inference_model.load_weights(config.FINAL_MODEL_PATH)
        
        # CORRECTED: The model must be compiled before evaluation.
        print("Compiling model for evaluation...")
        inference_model.compile(
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=[
                keras.metrics.CategoricalAccuracy(name="accuracy"),
                keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
        )

        print("Evaluating model on the validation set...")
        results = inference_model.evaluate(val_generator)
        print(f"\nValidation Loss: {results[0]:.4f}")
        print(f"Validation Accuracy: {results[1]:.4f}")
        print(f"Validation Top-5 Accuracy: {results[2]:.4f}")

    except Exception as e:
        print(f"An error occurred during the inference step: {e}")
else:
    print("\n--- Skipping Inference: Final model weights or data not available ---")

print("\n--- SCRIPT EXECUTION COMPLETE ---")


