# Grand X-Ray Slam: Division B - EfficientNetB0 Model + TPU
# An advanced transfer learning approach for image-only classification.
# VERSION: Reverted to image-only model, paths consolidated.

import numpy as np
import pandas as pd
import cv2
import os
import sys
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Lambda, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# --- Global Settings ---
DEBUG = False
EPOCHS = 8 if not DEBUG else 3 

# --- Path Definitions ---
BASE_PATH = '/kaggle/input/grand-xray-slam-division-b/'
TRAIN_CSV_PATH = os.path.join(BASE_PATH, 'train2.csv')
TRAIN_IMAGE_DIR = os.path.join(BASE_PATH, 'train2/')
TEST_IMAGE_DIR = os.path.join(BASE_PATH, 'test2/')
SAMPLE_SUBMISSION_PATH = os.path.join(BASE_PATH, 'sample_submission_2.csv')
SUBMISSION_PATH = '/kaggle/working/submission.csv'

print(f"TensorFlow version: {tf.__version__}")
if DEBUG:
    print("ğŸ”¥ğŸ”¥ğŸ”¥ RUNNING IN DEBUG MODE ON A SMALL SUBSET OF DATA ğŸ”¥ğŸ”¥ğŸ”¥")

# --- Step 1: Detect TPU, Define Strategy, and Set Mixed Precision ---
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu='local')
    print('âœ… Running on TPU ', tpu.master())
    strategy = tf.distribute.TPUStrategy(tpu)
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
    print('âœ… Mixed precision (mixed_bfloat16) enabled for TPU.')
except ValueError:
    print('â�Œ No TPU found, using CPU/GPU strategy')
    tpu = None
    strategy = tf.distribute.get_strategy()

print(f"REPLICAS: {strategy.num_replicas_in_sync}")

# --- Step 2: Load and Prepare Data ---
print("\n--- Loading and Preparing Data ---")
try:
    df_train = pd.read_csv(TRAIN_CSV_PATH)
    print(f"Loaded train2.csv with {len(df_train)} rows")
except FileNotFoundError:
    print(f"Error: {TRAIN_CSV_PATH} not found. Please ensure the Kaggle dataset is attached.")
    sys.exit()

if DEBUG:
    df_train = df_train.sample(n=100, random_state=42)
    print(f"DEBUG MODE: Sliced training data to {len(df_train)} rows.")

conditions = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
    'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
    'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
]

train_set, val_set = train_test_split(
    df_train, test_size=0.15, random_state=42, stratify=None if DEBUG else df_train['No Finding']
)
print(f"Training samples: {len(train_set)}, Validation samples: {len(val_set)}")


# --- Step 3: Calculate Class Weights for Weighted Loss ---
print("\n--- Calculating Class Weights for Imbalance ---")
def calculate_class_weights(df, max_weight_cap=20.0):
    pos_weights = {}
    neg_weights = {}
    total_samples = len(df)
    for i, condition in enumerate(conditions):
        pos_counts = df[condition].sum()
        neg_counts = total_samples - pos_counts
        pos_w = total_samples / (2 * pos_counts) if pos_counts > 0 else 1.0
        neg_w = total_samples / (2 * neg_counts) if neg_counts > 0 else 1.0
        pos_weights[i] = min(pos_w, max_weight_cap)
        neg_weights[i] = neg_w
    return pos_weights, neg_weights

pos_weights, neg_weights = calculate_class_weights(train_set)
pos_weights_tensor = tf.constant([pos_weights[i] for i in range(len(conditions))], dtype=tf.float32)
neg_weights_tensor = tf.constant([neg_weights[i] for i in range(len(conditions))], dtype=tf.float32)

def get_weighted_loss(pos_weights, neg_weights):
    def weighted_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        loss = pos_weights * y_true * tf.math.log(y_pred) + neg_weights * (1 - y_true) * tf.math.log(1 - y_pred)
        return -tf.reduce_mean(loss)
    return weighted_loss


# --- Step 4: Create High-Performance tf.data Pipeline ---
print("\n--- Creating tf.data Pipelines with Augmentation ---")

BATCH_SIZE_PER_REPLICA = 16
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
IMG_SIZE = (512, 512)
AUTOTUNE = tf.data.AUTOTUNE

def augment(image, label):
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    
    # Random small rotation using rot90 (approximate)
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    
    # Random brightness & contrast
    image = tf.image.random_brightness(image, max_delta=0.05)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    
    # Add Gaussian noise
    # noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.01)
    # image = tf.clip_by_value(image + noise, 0.0, 1.0)
    
    return image, label

def parse_function(filename, labels):
    """Loads and preprocesses an image file."""
    image_string = tf.io.read_file(filename)
    is_not_empty = tf.strings.length(image_string) > 0
    
    def process_normal():
        image = tf.io.decode_image(image_string, channels=3, expand_animations=False)
        is_valid_image = tf.greater(tf.size(image), 0)
        def resize_and_prep():
            return efficientnet.preprocess_input(tf.image.resize(image, IMG_SIZE))
        def empty_img():
             return tf.zeros((*IMG_SIZE, 3), dtype=tf.float32)
        return tf.cond(is_valid_image, resize_and_prep, empty_img)

    def process_empty():
        return tf.zeros((*IMG_SIZE, 3), dtype=tf.float32)

    processed_image = tf.cond(is_not_empty, process_normal, process_empty)
    processed_image.set_shape((*IMG_SIZE, 3))
    
    return processed_image, labels

def create_dataset(dataframe, image_dir, is_training=False, is_test=False):
    """Creates a high-performance tf.data.Dataset."""
    filepaths = image_dir + dataframe['Image_name']
    
    if is_test:
        labels = np.zeros((len(dataframe), len(conditions)), dtype=np.float32)
    else:
        labels = dataframe[conditions].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.map(parse_function, num_parallel_calls=AUTOTUNE)
    
    if is_training:
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
        ds = ds.repeat()

    ds = ds.batch(GLOBAL_BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

train_ds = create_dataset(train_set, TRAIN_IMAGE_DIR, is_training=True)
val_ds = create_dataset(val_set, TRAIN_IMAGE_DIR, is_training=False)
print(f"Global batch size: {GLOBAL_BATCH_SIZE}")


# --- Step 5: Build EfficientNetB0 Model ---
print("\n--- Building and Compiling EfficientNetB0 Model with Fine-Tuning ---")

def build_efficientnet_model(num_classes=14):
    """Builds an EfficientNetB0 model and fine-tunes the top layers."""
    image_input = Input(shape=(*IMG_SIZE, 3), name='image_input')
    base_model = efficientnet.EfficientNetB0(weights='imagenet', include_top=False, input_tensor=image_input)
    base_model.trainable = True
    
    fine_tune_at = 100 
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='sigmoid', dtype='float32')(x)
    
    model = Model(inputs=image_input, outputs=outputs, name="EfficientNetB0_Image_Only")
    return model

with strategy.scope():
    model = build_efficientnet_model(num_classes=len(conditions))
    scaled_lr = 1e-4 * strategy.num_replicas_in_sync 
    weighted_binary_crossentropy = get_weighted_loss(pos_weights_tensor, neg_weights_tensor)
    model.compile(optimizer=Adam(learning_rate=scaled_lr), loss=weighted_binary_crossentropy, metrics=['AUC'])

print("Model Architecture: EfficientNetB0")
model.summary()


# --- Step 6: Train the Model ---
print("\n--- Starting Model Training ---")
early_stopper = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, verbose=1, min_lr=1e-6)
steps_per_epoch = max(1, len(train_set) // GLOBAL_BATCH_SIZE)
validation_steps = max(1, len(val_set) // GLOBAL_BATCH_SIZE)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[early_stopper, lr_reducer],
    verbose=1
)

val_auc = max(history.history.get('val_AUC', [0.0]))
print(f"Best Validation AUC-ROC achieved: {val_auc:.4f}")


# --- Step 7: Analyze Validation Predictions ---
print("\n--- Analyzing Validation Predictions and Displaying Examples ---")

def plot_per_label_auc(true_labels, predictions):
    plt.figure(figsize=(10, 8))
    auc_scores = [roc_auc_score(true_labels[:, i], predictions[:, i]) for i in range(len(conditions)) if len(np.unique(true_labels[:, i])) > 1]
    valid_conditions = [c for i, c in enumerate(conditions) if len(np.unique(true_labels[:, i])) > 1]
    auc_df = pd.DataFrame({'Condition': valid_conditions, 'AUC': auc_scores})
    auc_df = auc_df.sort_values(by='AUC', ascending=True)
    bars = plt.barh(auc_df['Condition'], auc_df['AUC'], color='skyblue')
    plt.xlabel('AUC Score'); plt.title('AUC per Condition'); plt.xlim(0, 1)
    for bar in bars: plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.3f}', va='center', ha='left')
    plt.tight_layout(); plt.show()

def plot_prediction_examples(indices, predictions, ground_truth_df, image_dir, status):
    print(f"\n--- Displaying 5 {status.capitalize()} Prediction Examples ---")
    for i, idx in enumerate(indices):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1.2]})
        img_name = ground_truth_df.iloc[idx]['Image_name']
        img_path = os.path.join(image_dir, img_name)
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        ax1.imshow(image); ax1.set_title(f"Example #{i+1}\n{img_name}"); ax1.axis('off')
        
        ax2.axis('off')
        pred_labels = predictions[idx]
        true_labels = ground_truth_df.iloc[idx][conditions].values
        table_data = [["Condition", "Actual", "Predicted"]]
        table_data.extend([[c, int(t), f"{p:.3f}"] for c, t, p in zip(conditions, true_labels, pred_labels)])
        table = ax2.table(cellText=table_data, colWidths=[0.5, 0.2, 0.2], cellLoc='left', loc='center')
        table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.0, 1.2)
        
        for (row, col), cell in table.get_celld().items():
            if row == 0: cell.set_text_props(weight='bold')
            if col == 1 and row > 0: cell.set_facecolor("#90EE90" if table_data[row][1] == 1 else "#FFB6C1")
            if col == 2 and row > 0: cell.set_facecolor("#90EE90" if float(table_data[row][2]) >= 0.5 else "#FFB6C1")

        plt.tight_layout(); plt.show()

val_predictions = model.predict(val_ds, verbose=1)
val_predictions = val_predictions[:len(val_set)]
true_labels_val = val_set[conditions].values

plot_per_label_auc(true_labels_val, val_predictions)
mae = np.mean(np.abs(true_labels_val - val_predictions), axis=1)
sorted_indices = np.argsort(mae)
plot_prediction_examples(sorted_indices[:5], val_predictions, val_set, TRAIN_IMAGE_DIR, "good")
plot_prediction_examples(sorted_indices[-5:], val_predictions, val_set, TRAIN_IMAGE_DIR, "wrong")


# --- Step 8: Generate Submission with Test-Time Augmentation (TTA) ---
print("\n--- Generating Submission File with Test-Time Augmentation (TTA) ---")
try:
    submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    print(f"Loaded sample_submission_2.csv with {len(submission_df)} rows")
except FileNotFoundError:
    print(f"Error: {SAMPLE_SUBMISSION_PATH} not found."); sys.exit()

if DEBUG:
    submission_df = submission_df.head(100)

# Create a dataset for the original test images
print("ğŸš€ Predicting on original test images...")
test_ds_orig = create_dataset(submission_df, TEST_IMAGE_DIR, is_test=True)
predictions_orig = model.predict(test_ds_orig, verbose=1)
predictions_orig = predictions_orig[:len(submission_df)]

# Create a dataset for the flipped test images
print("ğŸ”„ Predicting on horizontally flipped test images...")
def flip_parse_function(filename, labels):
    """Loads and preprocesses an image file and flips it."""
    image, labels = parse_function(filename, labels)
    image = tf.image.flip_left_right(image)
    return image, labels

def create_flipped_dataset(dataframe, image_dir, is_test=True):
    filepaths = image_dir + dataframe['Image_name']
    labels = np.zeros((len(dataframe), len(conditions)), dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.map(flip_parse_function, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(GLOBAL_BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

test_ds_flipped = create_flipped_dataset(submission_df, TEST_IMAGE_DIR, is_test=True)
predictions_flipped = model.predict(test_ds_flipped, verbose=1)
predictions_flipped = predictions_flipped[:len(submission_df)]

# Average the predictions from both original and flipped images
print("ğŸ“Š Averaging predictions...")
final_predictions = (predictions_orig + predictions_flipped) / 2.0

# Generate the final submission file
submission_df[conditions] = final_predictions
submission_df.to_csv(SUBMISSION_PATH, index=False)

print(f"\nSubmission file created with TTA: {SUBMISSION_PATH}")
print("Script finished successfully!")



