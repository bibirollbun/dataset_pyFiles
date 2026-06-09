import pandas as pd
import numpy as np
import os
import sys
import math
import gc
from PIL import Image
import cv2
import ast
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
import keras_cv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
# Suppress specific TensorFlow logging for a cleaner output
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


print("Available devices: \n")
# List and print all logical devices configured for TensorFlow
for device in tf.config.list_logical_devices():
    print(device.name, device.device_type)


def get_strategy():
    """
    Detects and returns the best TensorFlow distribution strategy.
    - TPUStrategy for TPU(s)
    - MirroredStrategy for GPU(s)
    - Default strategy for CPU
    """
    try:
        # Try TPU first: Initialize and connect to the TPU cluster
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Using TPU strategy:", type(strategy).__name__)
    except Exception:
        # If TPU not available, try GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Use MirroredStrategy for distributed training across multiple GPUs
            strategy = tf.distribute.MirroredStrategy()
            print("Using GPU strategy:", type(strategy).__name__)
        else:
            # Fallback to CPU/default strategy
            strategy = tf.distribute.get_strategy()
            print("No TPU/GPU found. Using CPU strategy:", type(strategy).__name__)

    # Report the number of replicas, which equals the number of devices used for training
    print("REPLICAS:", strategy.num_replicas_in_sync)
    return strategy

# Execute the function to set the global distribution strategy
strategy = get_strategy()


# Print the confirmed number of synchronous replicas (devices) being utilized
print("REPLICAS:", strategy.num_replicas_in_sync)
# Print the TensorFlow version for reproducibility documentation
print("TensorFlow version:", tf.__version__)


SEED = 28
def seed_everything(SEED):
    # Set seed for the standard 'random' library
    random.seed(SEED)
    # Set seed for TensorFlow's global random operations
    tf.random.set_seed(SEED)
    # Set seed for NumPy's random operations
    np.random.seed(SEED)
    print('For reproducing purposes, everything seeded !')

# Execute the seeding function with the defined constant
seed_everything(SEED)


# Define base directory and paths for all resources
DATA_DIR = '/kaggle/input/global-wheat-detection'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
CSV_PATH = os.path.join(DATA_DIR, 'train.csv')


# Count the number of files in the train and test directories
num_train_images = len(os.listdir(TRAIN_DIR))
num_test_images = len(os.listdir(TEST_DIR))
print(f'Number of total images on Train directory: {num_train_images}')
print(f'Number of test images on Test directory: {num_test_images}')


# Load a sample image to check the default image resolution
img_path = os.path.join(TRAIN_DIR, os.listdir(TRAIN_DIR)[0])
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
print(img.shape)


# Load the annotation CSV file into a pandas DataFrame
df = pd.read_csv(CSV_PATH)
# Display the first few rows of the DataFrame
df.head()


# Check the total number of bounding box annotations
df.shape


# Calculate the average number of bounding boxes per image
averaged_bbox_per_img = df.groupby('image_id').size().mean()
print(f'Average Bounding boxes exists in an image: {int(averaged_bbox_per_img)}')


# Get detailed statistics on the count of wheat heads per image
bbox_counts = df.groupby('image_id').size()
print('Statistics of wheat head per image:')
print(bbox_counts.describe().T)


plt.figure(figsize= (12, 6))
# Create a histogram to visualize the distribution of wheat head counts
sns.histplot(bbox_counts, bins= 30, kde= True, color= 'purple')
plt.title('Number of Bounding Boxes per Image')
plt.xlabel('Number of Bounding Boxes')
plt.ylabel('Number of images')

plt.show()


# Identify all image IDs that have at least one annotation
annonated_ids = set(df['image_id'].unique())
print(f'Number of images with Wheat: {len(annonated_ids)}')


# Compare all image files with annotated IDs to find empty images
all_images = [f.replace('.jpg', '') for f in os.listdir(TRAIN_DIR)]
empty_images = [f for f in all_images if f not in annonated_ids]
print(f'Number of images without annonation(Wheat): {len(empty_images)}')
print(f'Example of empty image: {empty_images[0]}')


# Calculate the percentage of annotated and unannotated images
empty_img_frac = len(empty_images) / len(os.listdir(TRAIN_DIR))
annonated_img_frac = len(annonated_ids) / len(os.listdir(TRAIN_DIR))

print(f'Empty images percentage: {empty_img_frac:.4f}')
print(f'Annonated images percentage: {annonated_img_frac:.4f}')
print("Empty images aren't dominated, no problem with them at all!")


img_path = os.path.join(TRAIN_DIR, empty_images[0] + '.jpg')
img = Image.open(img_path)

plt.imshow(img)
plt.axis('off')
plt.title(f'Example of empty: {empty_images[0]}.jpg')
plt.show()


def show_images(num_images= 6, cols= 3):
    # Determine the list of files to display
    files = os.listdir(TRAIN_DIR)[:num_images]
    rows = (num_images + cols - 1) // cols

    fig = plt.figure(figsize= (cols* 4, rows* 4))
    
    for i, fname in enumerate(files):
        img_path = os.path.join(TRAIN_DIR, fname)
        img = Image.open(img_path)
        img = img.resize((256, 256)) # Resize for consistent display

        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(fname)
        
    plt.tight_layout()
    plt.show()


# Display 6 sample images
show_images(num_images= 6, cols= 3)


# Convert the string representation of the list in 'bbox' column to an actual list
df['bbox'] = df['bbox'].apply(ast.literal_eval)
# Extract coordinates from the list: [x_min, y_min, x_max, y_max]
df['x_min'] = df['bbox'].apply(lambda b: b[0])
df['y_min'] = df['bbox'].apply(lambda b: b[1])
df['x_max'] = df['bbox'].apply(lambda b: b[0] + b[2])
df['y_max'] = df['bbox'].apply(lambda b: b[1] + b[3])


# Display the modified DataFrame structure
df.head()


# Calculate the actual width and height of each bounding box
df['width'] = df['x_max'] - df['x_min']
df['height'] = df['y_max'] - df['y_min']
print(df[['width' ,'height']].describe().T)


df.head()


fig, ax = plt.subplots(1, 2, figsize= (12, 6))
for i, col in enumerate(['width', 'height']):
    sns.histplot(df[col], bins= 50, kde= True, ax= ax[i])
    ax[i].set_title(f'Bounding Boxes {col} distribution')
    ax[i].set_xlim((0, 250))
    ax[i].set_xlabel(f'{col} pixels')
    ax[i].set_ylabel('Count')


def show_images_with_bboxes(df, image_dir, nrows, ncols):
    # Pick random images from the train dir
    files = os.listdir(image_dir)
    selected_files = random.sample(files, nrows * ncols)

    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))

    for ax, fname in zip(axs.flatten(), selected_files):
        image_id = fname.replace('.jpg', '')

        # Load image
        img_path = os.path.join(image_dir, fname)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get bboxes if exists
        if image_id in df['image_id'].values:
            bboxes = df[df['image_id'] == image_id][['x_min', 'y_min', 'x_max', 'y_max']].values
            for (x_min, y_min, x_max, y_max) in bboxes:
                start_point = (int(x_min), int(y_min))
                end_point = (int(x_max), int(y_max))
                color = (255, 0, 0)
                thickness = 2
                cv2.rectangle(img, start_point, end_point, color, thickness)

        # Show image
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(fname, fontsize=8)

    plt.tight_layout()
    plt.show()


show_images_with_bboxes(df, TRAIN_DIR, 2, 2)


# Group bounding box coordinates by image_id, resulting in a list of bboxes per image
grouped = df.groupby('image_id')[['x_min', 'y_min', 'x_max', 'y_max']].apply(
    lambda x: x.values.tolist()
)


data_dicts = []
for image_id, bboxes in grouped.items():
    img_path = os.path.join(TRAIN_DIR, f'{image_id}.jpg')
    # Convert the list of bboxes into a float32 NumPy array with shape (N, 4)
    bboxes = np.array(bboxes, dtype=np.float32).reshape(-1, 4)
    data_dicts .append({
        'image_path': img_path,
        # 'bboxes' is the key expected by KerasCV's preprocessors
         'bboxes': bboxes
    })


# Split the data_dicts into train and validation sets (80/20 split)
train_dicts, val_dicts = train_test_split(
    data_dicts,
    test_size= 0.2,
    random_state= SEED,
    shuffle= True
)
print('Train and Validation dicts created successfully! 20% of data stored for validation')


for fname in empty_images:
    img_path = os.path.join(TRAIN_DIR, f'{fname}.jpg')
    # Create an empty bounding box array for images without wheat heads
    bboxes = np.zeros((0, 4), dtype=np.float32)
    train_dicts.append({
        'image_path': img_path,
        'bboxes': bboxes
    })

# Randomly shuffle the final training dictionary list after adding negative samples
random.shuffle(train_dicts)


# --- Model and Input Configuration ---
IMG_SIZE = (1024, 1024) # Target input resolution for the model
NUM_CLASSES = 1 # Only one class: 'wheat head'
GLOBAL_CLIPNORM = 10.0 # Gradient clipping value for training stability (prevents exploding gradients)

# --- Learning Rate Configuration (Three-Phase Strategy) ---
WARMUP_LR= 1e-3 # Learning rate for the initial Warmup phase
FINE_TUNE_BB_LR = 1e-4 # Learning rate for bounding box head during Fine-Tune (if separate training is desired)
FINE_TUNE_MODEL_LR = 1e-5 # Very low final learning rate for subtle refinement during Fine-Tune phase

# --- Epoch Configuration (Three-Phase Strategy) ---
WARMUP_EPOCH = 10 # Duration of the Warmup phase
INTERMEDIATE_EPOCH = WARMUP_EPOCH + 20 # Total epochs up to the end of the Mid-Tune phase
FINAL_EPOCH = INTERMEDIATE_EPOCH + 50 # Total epochs up to the end of the Fine-Tune phase

# --- tf.data Pipeline Configuration ---
AUTO = tf.data.AUTOTUNE # Optimal setting for parallel execution
BATCH_SIZE_PER_REPLICA = 4 # Batch size allocated to each available device (GPU/TPU core)
BUFFER_SHUFFLE_SIZE = 512 # Size of the buffer used for shuffling the dataset

# Global Batch Size calculation: crucial for scaling learning rates later
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync 
print(f'Global Batch size: {BATCH_SIZE}')


def prepare_inputs(dicts):
    # Convert list of image paths into a Ragged Tensor of strings
    image_paths = tf.ragged.constant(
        [s["image_path"] for s in dicts], dtype=tf.string
    )

    bbox_list = [
        np.array(s["bboxes"], dtype=np.float32).reshape(-1, 4)
        for s in dicts
    ]

    # Assign a class ID of 0 (since NUM_CLASSES=1, representing "wheat head")
    classes_list = [
        np.zeros((len(b)), dtype=np.float32) for b in bbox_list
    ]

    # Convert bounding box and class lists into Ragged Tensors
    bboxes  = tf.ragged.constant(bbox_list, ragged_rank=1, dtype=tf.float32)
    classes = tf.ragged.constant(classes_list, ragged_rank=1, dtype=tf.float32)

    return image_paths, classes, bboxes



# Load and decode JPEG image
def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


# Package into the dictionary format expected by KerasCV
def load_dataset(image_path, classes_rt, boxes_rt):
    image = load_image(image_path)
    bounding_boxes = {"boxes": boxes_rt, "classes": classes_rt}
    return {"images": image, "bounding_boxes": bounding_boxes}


# Strong augmentations: Aggressive data diversification for early training
augmenter_strong = tf.keras.Sequential([
    # JitteredResize: Randomly scales the image before resizing to introduce scale variation
    keras_cv.layers.JitteredResize(
        target_size=IMG_SIZE, scale_factor=(0.9, 1.1), bounding_box_format="xyxy"
    ),
    # Mosaic: Combines 4 images into 1, dramatically increasing batch size and context diversity
    keras_cv.layers.Mosaic(bounding_box_format="xyxy", name= 'mosaic'),
    # Standard horizontal flipping
    keras_cv.layers.RandomFlip(
        mode="horizontal", bounding_box_format="xyxy"
    ),
    # Strong color distortion
    keras_cv.layers.RandomColorJitter(
        value_range=(0.0, 255.0),
        brightness_factor=0.2, contrast_factor=0.2,
        saturation_factor=0.2, hue_factor=0.1
    ),
    # Randomly desaturates colors (simulating sensor noise/weather)
    keras_cv.layers.RandomColorDegeneration(
        factor=(0.2, 0.7), seed=SEED
    ),
])

# Light augmentations: Milder transformations for stable convergence in later phases
augmenter_light = tf.keras.Sequential([
    # Reduced jitter scale
    keras_cv.layers.JitteredResize(
        target_size=IMG_SIZE, scale_factor=(0.95, 1.05), bounding_box_format="xyxy"
    ),
    keras_cv.layers.RandomFlip(
        mode="horizontal", bounding_box_format="xyxy"
    ),
    # Reduced color distortion
    keras_cv.layers.RandomColorJitter(
        value_range=(0.0, 255.0),
        brightness_factor=0.1, contrast_factor=0.1,
        saturation_factor=0.1, hue_factor=0.05
    ),
    keras_cv.layers.RandomColorDegeneration(
        factor=(0.1, 0.4), seed=SEED
    ),
])

# Validation (deterministic) resizing: Only standard resizing for evaluation
augmenter_val = tf.keras.Sequential([
    # Fixed resize scale
    keras_cv.layers.JitteredResize(
        target_size=IMG_SIZE, scale_factor=(1.0, 1.0), bounding_box_format="xyxy"
    )
])


def dict_to_tuple(inputs):
    # Convert KerasCV dictionary output to standard Keras (x, y) tuple
    return inputs['images'], inputs['bounding_boxes']


def create_strong_dataset(dict_list, batch_size=BATCH_SIZE):
    
    image_paths, classes, bboxes = prepare_inputs(dict_list)

    # 1. Start with tensor slices
    ds = tf.data.Dataset.from_tensor_slices((image_paths, classes, bboxes))
    # 2. Shuffle paths
    ds = ds.shuffle(BUFFER_SHUFFLE_SIZE)
    # 3. Load image from path
    ds = ds.map(load_dataset, num_parallel_calls=AUTO)
    # 4. Batch before augmentation (required for Mosaic and efficient augmentation)
    ds = ds.ragged_batch(batch_size, drop_remainder=True)
    # 5. Apply strong augmentation
    ds = ds.map(augmenter_strong, num_parallel_calls=AUTO)
    # 6. Convert to tuple format
    ds = ds.map(dict_to_tuple, num_parallel_calls=AUTO)

    # Pre-fetch data to overlap CPU work (loading/augmenting) with GPU work (training)
    return ds.prefetch(AUTO)


def create_light_dataset(dict_list, batch_size=BATCH_SIZE, is_training= False):

    image_paths, classes, bboxes = prepare_inputs(dict_list)
    
    ds = tf.data.Dataset.from_tensor_slices((image_paths, classes, bboxes))

    # 1. Load image from path
    ds = ds.map(load_dataset, num_parallel_calls=AUTO)

    if is_training:
        ds = ds.shuffle(BUFFER_SHUFFLE_SIZE)
        ds = ds.ragged_batch(batch_size, drop_remainder=True)
        # Apply light training augmentation
        ds = ds.map(augmenter_light, num_parallel_calls=AUTO)
    else:
        # No shuffle for validation
        ds = ds.ragged_batch(batch_size, drop_remainder=True)
        # Apply deterministic validation augmentation
        ds = ds.map(augmenter_val, num_parallel_calls=AUTO)
    
    # 2. Convert to tuple format
    ds = ds.map(dict_to_tuple, num_parallel_calls=AUTO)
    
    # Pre-fetch for efficiency
    return ds.prefetch(AUTO)


# Initialize the datasets for the three training phases
train_strong_dataset = create_strong_dataset(train_dicts)
# Validation set is consistent across all phases
val_dataset = create_light_dataset(val_dicts, is_training= False)
# Light training set for Mid-Tune and Fine-Tune
train_light_dataset = create_light_dataset(train_dicts, is_training= True)

print('âœ… Train and Validation datasets are ready !')
print('Light Augmented dataset for Mid-Tune and Fine-Tune phases created !')


# Check the shapes of the output tensors from the data pipeline
for images, bounding_boxes in train_strong_dataset.take(1):
    bboxes = bounding_boxes["boxes"]
    classes = bounding_boxes["classes"]

    # Image shape should be (BATCH_SIZE, 1024, 1024, 3)
    print("Images shape:", images.shape)
    # Boxes shape will be RaggedTensor, showing (BATCH_SIZE, None, 4)
    print("Boxes shape:", bboxes.shape)
    # Classes shape will be RaggedTensor, showing (BATCH_SIZE, None)
    print("Classes shape:", classes.shape)


def visualize_dataset(dataset, rows=2, cols=2, 
                      value_range=(0, 255), bounding_box_format="xyxy"):
    # Take a single batch for visualization
    batch = next(iter(dataset.take(1)))
    # Our dataset is already (images, bounding_boxes) from the final map function
    images, bounding_boxes = batch
    
    num_images = rows * cols

    # 1. Plot raw augmented images
    fig, axs = plt.subplots(rows, cols, figsize= (4* cols, 4* rows))
    axs = axs.flatten()
    for i in range(num_images):
        # Convert tensor to NumPy array for plotting
        img = images[i].numpy().astype('uint8')

        axs[i].imshow(img)
        axs[i].set_title('Raw Image')
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()
         
    # 2. Plot images with ground-truth bounding boxes
    keras_cv.visualization.plot_bounding_box_gallery(
        images,                 # images
        y_pred= bounding_boxes, # y_pred is used here as the input format is (images, y_true)
        value_range=value_range,# range of image values
        rows=rows,
        cols=cols,
        scale=5,
        font_scale=0.7,
        bounding_box_format=bounding_box_format,
    )

    plt.tight_layout()
    plt.show()

# Usage
visualize_dataset(train_strong_dataset, rows=2, cols=2)
visualize_dataset(val_dataset, rows=2, cols=2)


# Get the total number of samples in each split
NUM_TRAIN_IMAGES = len(train_dicts)
NUM_VAL_IMAGES   = len(val_dicts)

# Calculate the number of batches (steps) per training epoch
steps_per_epoch  = math.ceil(NUM_TRAIN_IMAGES / BATCH_SIZE)
# Calculate the number of batches (steps) for validation
validation_steps = math.ceil(NUM_VAL_IMAGES / BATCH_SIZE)

print(f"Steps per Epoch: {steps_per_epoch}")
print(f"Validation Steps: {validation_steps}")


def create_model():
    # Instantiate YOLOv8-M backbone with COCO pre-trained weights
    backbone = keras_cv.models.YOLOV8Backbone.from_preset(
        'yolo_v8_m_backbone_coco',
        name= 'yolov8_backbone'
    )
    # Instantiate the full YOLOv8 Detector model
    model = keras_cv.models.YOLOV8Detector(
        num_classes= NUM_CLASSES,
        bounding_box_format= 'xyxy',
        fpn_depth= 3, # Standard Feature Pyramid Network (FPN) depth
        backbone= backbone,
        name= 'yolov8_detector'
    )
    return model


with strategy.scope():
    
    model = create_model()

    # Initially freeze the backbone weights
    model.backbone.trainable = False

    # Explicitly freeze Batch Normalization layers across the backbone
    for layer in model.backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.summary()
    
    # Define the AdamW optimizer with the Warmup learning rate
    optimizer = tf.keras.optimizers.AdamW(
    learning_rate= WARMUP_LR,
    weight_decay= 1e-3, # Decoupled weight decay
    beta_1= 0.9,
    beta_2= 0.999,
    global_clipnorm= GLOBAL_CLIPNORM) # Apply gradient clipping

    classification_loss = keras_cv.losses.FocalLoss() # Used for classification

    # Compile the model with specified loss functions
    model.compile(
        optimizer= optimizer,
        classification_loss= classification_loss,
        box_loss= 'ciou',
        # Steps per execution: Increase performance on TPU/XLA by compiling a larger training graph
        steps_per_execution= 32 if isinstance(strategy, tf.distribute.TPUStrategy) else 1
    )


class EvaluateCOCOMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, data, save_path):
        super().__init__()
        self.data = data
        self.metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format="xyxy",
            evaluate_freq=1e9,  # Set to a high number as evaluation is triggered manually in on_epoch_end
        )
        self.save_path = save_path
        self.best_map = -1.0 # Tracks the highest MaP achieved

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.metrics.reset_state()

        # ---- START: MODIFIED SECTION ----
        # 1. Create lists to hold all ground truth and prediction data
        y_true_list = []
        y_pred_list = []

        # 2. Iterate through the entire validation dataset to collect data
        for images, y_true in self.data:
            # Predict on the current batch
            y_pred = self.model.predict(images, verbose=0)
            y_true_list.append(y_true)
            y_pred_list.append(y_pred)

        # 3. Concatenate all batches into single, large ragged tensors
        # Concatenate ground truth (boxes and classes)
        y_true_concat = {
            'boxes': tf.concat([item['boxes'] for item in y_true_list], axis=0),
            'classes': tf.concat([item['classes'] for item in y_true_list], axis=0)
        }
        # Concatenate predictions (boxes, classes, and confidence)
        y_pred_concat = {
            'boxes': tf.concat([item['boxes'] for item in y_pred_list], axis=0),
            'classes': tf.concat([item['classes'] for item in y_pred_list], axis=0),
            'confidence': tf.concat([item['confidence'] for item in y_pred_list], axis=0)
        }
        # ---- END: MODIFIED SECTION ----

        # 4. Update the metric's state ONCE with the full dataset for accurate COCO metrics
        self.metrics.update_state(y_true_concat, y_pred_concat)

        # 5. Get the final results
        metrics = self.metrics.result(force=True)
        logs.update(metrics)

        current_map = metrics["MaP"]
        
        # Manually print the validation metrics for visibility
        print(f"\nEpoch {epoch+1}: Validation Metrics")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        # Model checkpointing logic  
        if current_map > self.best_map:
            self.best_map = current_map
            # Save the model to the Kaggle working directory
            self.model.save(self.save_path)
            print(f"âœ… Validation MaP improved to {current_map:.4f}. Model saved to {self.save_path}")

        return logs


# Define the save path for the best model of the Warmup phase(you can use your own path)
phase1_saved_path = "/kaggle/working/warmup_best_model.keras"
# Instantiate Callbacks
coco_cb = EvaluateCOCOMetricsCallback(val_dataset, 
                                      save_path= phase1_saved_path,
                                      )
early_stopping_cb = EarlyStopping(
    monitor= 'MaP',
    patience= 3, # Stop if MaP doesn't improve for 3 epochs
    restore_best_weights= True,
    mode= 'max'
)

reduce_lr_cb = ReduceLROnPlateau(
    monitor= 'MaP',
    patience= 3,
    factor= 0.66,
    min_lr= WARMUP_LR * 0.1,
    verbose= 1
)

# TensorBoard Object if you want to use
tb_cb = TensorBoard(
    log_dir= '/kaggle/working/logs',
    histogram_freq= 1
)

callbacks = [
    coco_cb,
    early_stopping_cb,
    reduce_lr_cb,
    tb_cb
]


print("--- Starting Phase 1: Warmup Training ---")
# # Train the Warmup Model
history = model.fit(train_strong_dataset.repeat(), 
                    validation_data= val_dataset.repeat(),
                    epochs= WARMUP_EPOCH,
                    callbacks= [callbacks],
                    steps_per_epoch= steps_per_epoch,
                    validation_steps= validation_steps)


START_UNFREEZE_LAYER_NAME = 'stack4_downsample_conv'
with strategy.scope():
    print("Loading model from warmup phase...")
    # Load the best model from the previous Warmup phase
    model = tf.keras.models.load_model(
        # The model is loaded from a pre-uploaded Kaggle Model
        '/kaggle/input/wheat-detection/keras/warmup/1/warmup_best_model.keras',
            custom_objects = {
                'YOLOV8Detector': keras_cv.models.YOLOV8Detector,
                'YOLOV8Backbone': keras_cv.models.YOLOV8Backbone
            }
    )
    print("Model loaded successfully. Ready for Mid-Tune phase !")
    
    # 1. Unfreeze the entire backbone
    model.backbone.trainable = True
    unfreeze_checkpoint = False

    # 2. Implement partial unfreezing (freeze the first few stacks)
    for layer in model.backbone.layers:
        if layer.name == START_UNFREEZE_LAYER_NAME:
            unfreeze_checkpoint = True
        if unfreeze_checkpoint:
            layer.trainable = True
        else:
            layer.trainable = False

        # Ensure BN layers remain frozen regardless of backbone trainable status
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # Also check and freeze BN layers in the Neck and Head of the detector
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    
    # Configure Cosine Decay Learning Rate schedule
    num_phase2_epochs = INTERMEDIATE_EPOCH - WARMUP_EPOCH
    decay_steps = int(steps_per_epoch * num_phase2_epochs)
    learning_rate = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=FINE_TUNE_BB_LR,
        decay_steps=decay_steps,
        alpha=0.1 # End LR will be 10% of initial LR (1e-5)
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate = learning_rate,
        weight_decay = 1e-4,
        beta_1 = 0.9,
        beta_2 = 0.999,
        global_clipnorm = GLOBAL_CLIPNORM
    )

    classification_loss = keras_cv.losses.FocalLoss()
    
    # Recompile the model with the new, scheduled optimizer
    model.compile(
        optimizer = optimizer,
        classification_loss = classification_loss,
        box_loss = 'ciou',
        steps_per_execution= 32 if isinstance(strategy, tf.distribute.TPUStrategy) else 1
    )
    print("\n--- Model configured for Phase 2: Mid-Tune ---")


# Define Phase 2 callbacks
phase2_saved_path = "/kaggle/working/midtune_best_model.keras"
coco_cb = EvaluateCOCOMetricsCallback(val_dataset, 
                                       phase2_saved_path)
early_stopping_cb = EarlyStopping(
    monitor= 'MaP',
    patience= 5, # Increased patience for finer tuning
    restore_best_weights= True,
    mode= 'max'
)

tb_cb = TensorBoard(
    log_dir= '/kaggle/working/logs',
    histogram_freq= 1
)

callbacks = [
    coco_cb,
    early_stopping_cb,
    tb_cb
]


print("--- Starting Phase 2: Mid-Tune Training ---")
# Train the model for Mid-Tune, starting from WARMUP_EPOCH
final_history = model.fit(
    train_light_dataset.repeat(),
    epochs= INTERMEDIATE_EPOCH,
    initial_epoch= WARMUP_EPOCH,
    validation_data= val_dataset.repeat(),
    steps_per_epoch= steps_per_epoch,
    validation_steps= validation_steps,
    callbacks= callbacks
)


with strategy.scope():
    print("Loading model from mid-tune phase...")
    # Load the best model from the previous Mid-Tune phase
    model = tf.keras.models.load_model(
        '/kaggle/input/wheat-detection/keras/warmup/2/midtune_best_model.keras',
            custom_objects = {
                'YOLOV8Detector': keras_cv.models.YOLOV8Detector,
                'YOLOV8Backbone': keras_cv.models.YOLOV8Backbone
            }
    )
    print("Model loaded successfully. Ready for Fine-Tune phase !")
    
    # Set all layers in the backbone to trainable (Full Unfreezing)
    model.backbone.trainable = True

    # Ensure BN layers are frozen across the entire detector model (backbone, neck, head)
    for layer in model.backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    for layer in model.layers: # Iterate through all layers of the detector model
    # Note: We re-check for BN to catch those in the Neck and Head
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    
    # Configure Cosine Decay Learning Rate schedule with a very low initial LR
    num_phase3_epochs = FINAL_EPOCH - INTERMEDIATE_EPOCH
    decay_steps = int(steps_per_epoch * num_phase3_epochs)
    learning_rate = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=FINE_TUNE_MODEL_LR,
        decay_steps=decay_steps,
        alpha=0.1 # End LR will be 10% of initial LR (5e-6)
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate = learning_rate,
        weight_decay = 1e-4,
        beta_1 = 0.9,
        beta_2 = 0.999,
        global_clipnorm = GLOBAL_CLIPNORM
    )

    classification_loss = keras_cv.losses.FocalLoss()
    
    # Recompile the model with the final optimizer settings
    model.compile(
        optimizer = optimizer,
        classification_loss = classification_loss,
        box_loss = 'ciou',
        steps_per_execution= 32 if isinstance(strategy, tf.distribute.TPUStrategy) else 1
    )
    print("\n--- Model configured for Phase 3: Fine-Tune ---")


# Define Phase 3 callbacks
phase3_saved_path = "/kaggle/working/finetune_best_model.keras"
coco_cb = EvaluateCOCOMetricsCallback(val_dataset, 
                                       phase3_saved_path)
early_stopping_cb = EarlyStopping(
    monitor= 'MaP',
    patience= 8, # Highest patience to allow for slow, subtle improvements
    restore_best_weights= True,
    mode= 'max'
)

tb_cb = TensorBoard(
    log_dir= '/kaggle/working/logs',
    histogram_freq= 1
)

callbacks = [
    coco_cb,
    early_stopping_cb,
    tb_cb
]


print("--- Starting Phase 3: Fine-Tune Training ---")
# Train the model, continuing from INTERMEDIATE_EPOCH, Final Model.
final_history = model.fit(
    train_light_dataset.repeat(),
    epochs= FINAL_EPOCH,
    initial_epoch= INTERMEDIATE_EPOCH,
    validation_data= val_dataset.repeat(),
    steps_per_epoch= steps_per_epoch,
    validation_steps= validation_steps,
    callbacks= callbacks
)




