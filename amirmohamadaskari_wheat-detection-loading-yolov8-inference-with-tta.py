import ast
import cv2
import keras_cv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf
# Suppress TensorFlow warning messages for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


print("Available devices: \n")
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
        # Try TPU first
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Using TPU strategy:", type(strategy).__name__)
    except Exception:
        # If TPU not available, try GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            strategy = tf.distribute.MirroredStrategy()
            print("Using GPU strategy:", type(strategy).__name__)
        else:
            # Fallback CPU
            strategy = tf.distribute.get_strategy()
            print("No TPU/GPU found. Using CPU strategy:", type(strategy).__name__)

    print("REPLICAS:", strategy.num_replicas_in_sync)
    return strategy

# Call it
strategy = get_strategy()


# --- Configuration for Offline Libraries ---
# Path to the dataset containing the 'ensemble_boxes' library
PACKAGE_DIR = '/kaggle/input/wheat-detection-ensemble-boxes/isolated_ensemble_boxes/'
if PACKAGE_DIR not in sys.path:
    sys.path.append(PACKAGE_DIR)

print(f"Added isolated package path to sys.path: {PACKAGE_DIR}")

import ensemble_boxes
print("âœ… ensemble_boxes imported successfully.")


try:
    from ensemble_boxes import weighted_boxes_fusion
except ImportError:
    print("WARNING: The 'ensemble_boxes' library is required for WBF.")
    print("Please run: pip install ensemble-boxes")
    # Placeholder to prevent crash, you must install the library to proceed
    weighted_boxes_fusion = None 


# --- Inference Hyperparameters ---
AUTO = tf.data.AUTOTUNE  # Auto from TensorFlow
IMG_SIZE = (1024, 1024)  # Input size for the YOLOv8 model
NUM_CLASSES = 1          # Wheat heads
NUM_TTA = 8              # Number of Test Time Augmentations
CONF_THRESHOLD = 0.40    # Minimum confidence score to keep a box
IOU_THRESHOLD = 0.50     # IoU threshold for Weighted Boxes Fusion
BATCH_SIZE_PER_REPLICA = 1
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

# --- Data & Model Paths ---
# Root directory for competition data
DATA_DIR = '/kaggle/input/global-wheat-detection' 
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
CSV_PATH = os.path.join(DATA_DIR, 'train.csv')

# Path to the fine-tuned KerasCV model (ensure this matches your uploaded model dataset)
MODEL_DIR = '/kaggle/input/wheat-detection/keras/warmup/3/finetune_best_model.keras'

print(f"Global Batch size: {BATCH_SIZE}")


with strategy.scope():
    print("Loading model ...")
    # Load the model providing necessary custom objects from KerasCV
    yolo_model = tf.keras.models.load_model(MODEL_DIR,
            custom_objects = {
                'YOLOV8Detector': keras_cv.models.YOLOV8Detector,
                'YOLOV8Backbone': keras_cv.models.YOLOV8Backbone
            }
    )
    print("Model loaded successfully. Ready for Visualization and Test Time !")


df = pd.read_csv(CSV_PATH)
# Convert the string representation of the list in 'bbox' column to an actual list
df['bbox'] = df['bbox'].apply(ast.literal_eval)
# Extract coordinates from the list: [x_min, y_min, x_max, y_max]
df['x_min'] = df['bbox'].apply(lambda b: b[0])
df['y_min'] = df['bbox'].apply(lambda b: b[1])
df['x_max'] = df['bbox'].apply(lambda b: b[0] + b[2])
df['y_max'] = df['bbox'].apply(lambda b: b[1] + b[3])


def show_images_with_bboxes(df, image_dir, nrows, ncols):
    # Pick random images from the train dir
    files = os.listdir(image_dir)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))

    for ax, fname in zip(axs.flatten(), files):
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


def load_single_image(image_path):
    """Loads and resizes a single image for prediction."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    return image


def detections(model, image_path, bounding_box_format="xyxy", 
                         confidence_threshold=CONF_THRESHOLD):
    files = os.listdir(image_path)[:4]
    paths = [os.path.join(image_path, f) for f in files]
    num_images = len(paths)

    image_list = [load_single_image(path) for path in paths]
    images = tf.stack(image_list)
    print(f"Loaded {num_images} images into a batch of shape {images.shape}")
    
    # Run model inference on the batch
    y_pred = model.predict(images)

    # y_pred is a dictionary: {'boxes': ..., 'confidence': ..., 'classes': ...}
    # Filter low-confidence boxes manually (since keras_cv.bounding_box has no filter)
    conf_mask = y_pred["confidence"] > confidence_threshold

    # Create filtered prediction dict
    y_pred_filtered = {
        "boxes": tf.ragged.boolean_mask(y_pred["boxes"], conf_mask),
        "classes": tf.ragged.boolean_mask(y_pred["classes"], conf_mask),
        "confidence": tf.ragged.boolean_mask(y_pred["confidence"], conf_mask),
    }

    # Visualize with KerasCV utility
    keras_cv.visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_pred=y_pred_filtered,
        scale=4,
        rows=2,
        cols=2,
        show=True,
        font_scale=0.7,
    )

    return y_pred



y_pred = detections(yolo_model, TRAIN_DIR, bounding_box_format="xyxy")


print(y_pred)


y_pred = detections(yolo_model, TEST_DIR)


def reverse_h_flip(boxes, image_width= IMG_SIZE[1]):
    """Reverses a horizontal flip for pixel-space [xmin, ymin, xmax, ymax] boxes."""
    xmin = image_width - boxes[:, 2]
    ymin = boxes[:, 1]
    xmax = image_width - boxes[:, 0]
    ymax = boxes[:, 3]
    return tf.stack([xmin, ymin, xmax, ymax], axis=-1)



def reverse_v_flip(boxes, image_height= IMG_SIZE[0]):
    """Reverses a vertical flip for normalized [xmin, ymin, xmax, ymax] boxes."""
    # ymin_new = 1 - ymax_old; ymax_new = 1 - ymin_old
    xmin = boxes[:, 0]     # Use xmin index (0)
    ymin = image_height - boxes[:, 3] # Use ymax index (3)
    xmax = boxes[:, 2]     # Use xmax index (2)
    ymax = image_height - boxes[:, 1] # Use ymin index (1)
    
    # Must stack in the original [xmin, ymin, xmax, ymax] order
    return tf.stack([xmin, ymin, xmax, ymax], axis=-1)


# Function to apply multiple, slightly random color/light transforms
def random_color_jitter(img):
    # Apply small random hue shift
    img = tf.image.random_hue(img, max_delta=0.04) 
    # Apply small random contrast
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15) 
    # Apply small random saturation
    img = tf.image.random_saturation(img, lower=0.85, upper=1.15)
    # Apply small random brightness
    img = tf.image.random_brightness(img, max_delta=0.04)
    return img


test_img_path = os.path.join(TEST_DIR, os.listdir(TEST_DIR)[3])
test_img = load_single_image(test_img_path)


def predict_and_filter(model, image_tensor, conf_threshold):
    """
    Runs model prediction on a single image, filters results by confidence.
    
    Args:
        model: The trained Keras model.
        image_tensor: The input image (H, W, 3).
        conf_threshold: Minimum confidence score to keep a detection.
        
    Returns:
        dict: A dictionary of filtered predictions {"boxes", "classes", "confidence"}
              in KerasCV RaggedTensor format, with batch dimension [1, None, ...].
    """
    # Add batch dimension: [1, H, W, 3]
    input_tensor = tf.expand_dims(image_tensor, 0)
    
    # Predict
    y_pred = model.predict(input_tensor, verbose=0)

    # Filter by confidence
    conf_mask = y_pred["confidence"] > conf_threshold
    
    # Apply mask to all prediction components
    y_pred_filtered = {
        "boxes": tf.ragged.boolean_mask(y_pred["boxes"], conf_mask),
        "classes": tf.ragged.boolean_mask(y_pred["classes"], conf_mask),
        "confidence": tf.ragged.boolean_mask(y_pred["confidence"], conf_mask),
    }
    
    return y_pred_filtered


def visualize_prediction(image_tensor, predictions, title, **kwargs):
    """
    Visualizes the prediction on the given image tensor.
    
    Args:
        image_tensor: The image to plot on (H, W, 3).
        predictions: The prediction dictionary from predict_and_filter.
        title: Title to print before visualization.
        **kwargs: Optional keyword arguments for plot_bounding_box_gallery.
    """
    print(f"\n--- {title} ---")
    
    keras_cv.visualization.plot_bounding_box_gallery(
        tf.expand_dims(image_tensor, 0),
        value_range=(0, 255),
        bounding_box_format="xyxy",
        y_pred=predictions,
        scale=4,
        rows=1,
        cols=1,
        show=True,
        font_scale=0.7,
        **kwargs
    )
    print(f"âœ… {title} visualized.")


# --- STEP 1: Original Image Prediction ---
y_pred_original = predict_and_filter(
    yolo_model, 
    test_img, 
    CONF_THRESHOLD
)

visualize_prediction(
    test_img, 
    y_pred_original, 
    "Original Prediction"
)


# --- STEP 2: Vertical Flip, Predict, and Visualize Flipped ---
vflipped_img = tf.image.flip_up_down(test_img)

y_pred_flip = predict_and_filter(
    yolo_model, 
    vflipped_img, 
    CONF_THRESHOLD
)

visualize_prediction(
    vflipped_img, 
    y_pred_flip, 
    "Flipped Image Prediction"
)


# --- STEP 3: Reverse Boxes and Visualize on Original Image ---
print("\nReversing flipped boxes to original coordinates...")

# Extract boxes from the filtered RaggedTensor (get the dense tensor at index 0)
boxes_augmented = y_pred_flip["boxes"][0] 

# Note: Your original reverse_v_flip function requires the image dimension
# We assume it takes IMG_SIZE[0] (1024) as the image_height argument.
# We must ensure reverse_v_flip is passed the image size as defined in your setup.
boxes_reversed = reverse_v_flip(boxes_augmented)

# Reformat the reversed boxes into the prediction dictionary for plotting
y_pred_reversed = {
    "boxes": tf.RaggedTensor.from_tensor(tf.expand_dims(boxes_reversed, axis=0)),
    "classes": y_pred_flip["classes"],
    "confidence": y_pred_flip["confidence"],
}

visualize_prediction(
    test_img, 
    y_pred_reversed, 
    "Reversed Flipped Boxes on Original Image"
)


# --- TTA Sequence Definition (for [xmin, ymin, xmax, ymax] format) ---
TTA_SEQUENCES = [
    # 0. Baseline (Identity)
    (lambda img: img, lambda boxes: boxes),  
    
    # 1. Horizontal Flip
    (tf.image.flip_left_right, reverse_h_flip),
    
    # 2. Vertical Flip
    (tf.image.flip_up_down, reverse_v_flip),
    
    # 3-6. Four different, random photometric variants (No box reversal needed)
    (lambda img: random_color_jitter(img), lambda boxes: boxes), 
    (lambda img: random_color_jitter(img), lambda boxes: boxes), 
    (lambda img: random_color_jitter(img), lambda boxes: boxes), 
    (lambda img: random_color_jitter(img), lambda boxes: boxes), 
]


def tta(model, image_tensor, tta_sequences, conf_threshold):
    """
    Applies TTA sequences, collects predictions, reverses boxes, and aggregates results 
    for a single image. **Note: Relies on external global variable IMG_SIZE[0] for dimensions
    if not passed to reverse_boxes_fn.**

    Args:
        model: The trained Keras model.
        image_tensor: The input image tensor (e.g., test_img, shape [H, W, 3]).
        tta_sequences: The list of (augment_fn, reverse_fn) tuples.
        conf_threshold: The minimum confidence for filtering individual TTA predictions.
        
    Returns:
        tuple: (all_boxes_np, all_confidences_np, all_classes_np) as concatenated NumPy arrays.
    """
    all_tta_predictions = []

    print(f"--- Running TTA for 1 image ({len(tta_sequences)} sequences) ---")

    for i, (augment_image_fn, reverse_boxes_fn) in enumerate(TTA_SEQUENCES):
        # 1. Augment and Prepare Input
        augmented_img = augment_image_fn(image_tensor)
        input_tensor = tf.expand_dims(augmented_img, 0)
        
        # 2. Predict (verbose=0 suppresses Keras logging)
        y_pred = model.predict(input_tensor, verbose=0)

        # 3. Filter by confidence
        conf_mask = y_pred["confidence"] > conf_threshold
        
        # Extract and un-batch filtered tensors
        boxes_augmented = tf.ragged.boolean_mask(y_pred["boxes"], conf_mask)[0]
        classes = tf.ragged.boolean_mask(y_pred["classes"], conf_mask)[0]
        confidence = tf.ragged.boolean_mask(y_pred["confidence"], conf_mask)[0]

        n_detections = tf.shape(boxes_augmented)[0].numpy()
        
        if n_detections > 0:
            # Note: We must now pass the image_dim back, as reverse functions typically need it
            # unless they were hard-coded to 1024 (which is poor practice).
            # Assuming reverse_boxes_fn still expects the dimension for safety:
            boxes_reversed = reverse_boxes_fn(boxes_augmented) 
            all_tta_predictions.append((boxes_reversed, confidence, classes))
        
        # New: Log every single TTA sequence
        print(f"  > Sequence {i+1}/{len(tta_sequences)} ({n_detections} boxes) processed.")


    if not all_tta_predictions:
        print("--- Aggregation: No detections found above confidence threshold. ---")
        return None, None, None

    # 4. Aggregate all predictions into single NumPy arrays
    list_of_boxes = [p[0].numpy() for p in all_tta_predictions]
    list_of_confidences = [p[1].numpy() for p in all_tta_predictions]
    list_of_classes = [p[2].numpy() for p in all_tta_predictions]

    all_boxes_np = np.concatenate(list_of_boxes, axis=0)
    all_confidences_np = np.concatenate(list_of_confidences, axis=0)
    # Corrected potential bug: Use list_of_classes instead of all_classes_np
    all_classes_np = np.concatenate(list_of_classes, axis=0)

    print(f"--- Aggregation Complete. Total boxes for WBF: {len(all_boxes_np)} ---")
    
    return all_boxes_np, all_confidences_np, all_classes_np


all_boxes_np, all_confidences_np, all_classes_np = tta(yolo_model, test_img, 
                                                       TTA_SEQUENCES, CONF_THRESHOLD)


def wbf(all_boxes_np, all_confidences_np, 
        all_classes_np, img_size, conf_threshold, iou_threshold):
    """
    Performs WBF on aggregated predictions and converts results to Kaggle submission string.

    Returns:
        str: Space-delimited prediction string in "conf x y w h" format.
        dict: Fused prediction dictionary for visualization (KerasCV format).
    """
    image_dim = img_size[0]

    # 1. Normalize pixel coordinates to [0.0, 1.0] for WBF
    normalized_boxes = all_boxes_np / image_dim
    
    # WBF expects list of lists
    boxes = [normalized_boxes.tolist()]
    scores = [all_confidences_np.tolist()]
    labels = [all_classes_np.astype(float).tolist()] 

    # 2. Apply Weighted Boxes Fusion (WBF)
    fused_boxes_norm, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes,
        scores,
        labels,
        weights=None,
        iou_thr=iou_threshold,
        conf_type='max',
        skip_box_thr=0.0001
    )
    print(f"--- WBF Applied. Boxes reduced to {len(fused_boxes_norm)} before final filter. ---")
    
    # 3. Final Filtering and Denormalization
    final_mask = fused_scores >= conf_threshold
    final_boxes_norm = fused_boxes_norm[final_mask]
    final_scores = fused_scores[final_mask]
    final_labels_filtered = fused_labels[final_mask] # Capture filtered labels
    
    # Denormalize boxes back to [0, 1024] pixel space (xyxy format)
    final_boxes_denorm_xyxy = final_boxes_norm * image_dim
    
    print(f"--- Final {len(final_boxes_denorm_xyxy)} boxes remain after final confidence filter. ---")

    # 4. Format Conversion to Kaggle "conf x y w h"
    prediction_strings = []
    
    # Iterate over the fused, filtered results
    for box, score in zip(final_boxes_denorm_xyxy, final_scores):
        # Convert to integer pixel values (Kaggle standard)
        xmin, ymin, xmax, ymax = box.astype(np.int32)
        
        # Convert xyxy to xywh
        w = xmax - xmin
        h = ymax - ymin
        
        # Format: confidence xmin ymin w h (space-delimited)
        box_string = f"{score:.4f} {xmin} {ymin} {w} {h}"
        prediction_strings.append(box_string)
        
    submission_string = " ".join(prediction_strings)
    
    # Optional: Show example submission string
    if submission_string:
        print(f"--- Submission String Example: {submission_string[:100]}... ---")
    else:
        print("--- Submission String is empty (No confident detections). ---")
    
    # 5. Prepare KerasCV format for optional visualization
    final_boxes_denorm_batched = tf.expand_dims(
        tf.convert_to_tensor(final_boxes_denorm_xyxy, dtype=tf.float32), axis=0
    )
    final_scores_batched = tf.expand_dims(
        tf.convert_to_tensor(final_scores, dtype=tf.float32), axis=0
    )
    
    y_pred_tta_fused = {
        "boxes": tf.RaggedTensor.from_tensor(final_boxes_denorm_batched),
        # Use the filtered labels, converted to int32, and batched
        "classes": tf.RaggedTensor.from_tensor(
            tf.expand_dims(tf.convert_to_tensor(final_labels_filtered, dtype=tf.int32), axis=0)
        ),
        "confidence": tf.RaggedTensor.from_tensor(final_scores_batched),
    }

    return submission_string, y_pred_tta_fused


submission_string, y_pred_fused = wbf(all_boxes_np, 
                                      all_confidences_np, 
                                      all_classes_np,
                                      IMG_SIZE,
                                      CONF_THRESHOLD,
                                      IOU_THRESHOLD)


# Assuming you have the fused result from wbf_and_format:
# submission_string, y_pred_fused = wbf_and_format(...)

visualize_prediction(
    image_tensor=test_img, 
    predictions=y_pred_fused, 
    title="Final WBF-Fused TTA Result"
)


def preprocess_for_inference(image_path):
    """Loads and resizes a single image for model prediction."""
    image = load_single_image(image_path)
    image_id = tf.strings.split(image_path, os.sep)[-1]       # filename
    image_id = tf.strings.regex_replace(image_id, ".jpg$", "") # remove extension
    return image, image_id


def test_dataset(TEST_DIR):
    '''Create dataset with loading all images in Test directory'''
    test_images_path = [
        os.path.join(TEST_DIR, fname) for fname in os.listdir(TEST_DIR)
    ]
    test_ds = tf.data.Dataset.from_tensor_slices(test_images_path)
    test_ds = (test_ds.map(preprocess_for_inference, num_parallel_calls= AUTO)
                      .batch(BATCH_SIZE)
                      .prefetch(AUTO))
    
    print('Test Dataset created successfully !')
    return test_ds


def create_submission_file(model, test_ds, tta_sequences, img_size, conf_threshold, iou_threshold):
    """
    Processes the entire test dataset using TTA/WBF and creates the submission DataFrame.
    
    Args:
        model: The Keras model.
        test_ds: The tf.data.Dataset for the test images.
        ... all other necessary parameters
        
    Returns:
        pd.DataFrame: The final submission dataframe.
    """
    submission_data = {"image_id": [], "PredictionString": []}
    
    print("\n=======================================================")
    print(">>> Starting Inference on Test Dataset with TTA/WBF <<<")
    print("=======================================================")
    
    # Iterate through the test dataset
    for step, (images, image_ids) in enumerate(test_ds):
        
        # Loop through images in the current batch (usually batch_size=1 for TTA)
        for image_tensor, image_id in zip(images, image_ids):
            
            image_id_str = image_id.numpy().decode('utf-8')
            print(f"\n[IMAGE {step+1}] Processing image: {image_id_str}")

            # 1. Run TTA and Aggregate
            all_boxes, all_confidences, all_classes = tta(
                model, image_tensor, tta_sequences, conf_threshold
            )
            
            if all_boxes is None:
                # No detections, prediction string is empty
                submission_string = ""
                y_pred_fused = None
            else:
                # 2. Run WBF and Format
                submission_string, y_pred_fused = wbf(
                    all_boxes, all_confidences, all_classes, 
                    img_size, conf_threshold, iou_threshold
                )
            
            # 3. Collect Result
            submission_data["image_id"].append(image_id_str)
            submission_data["PredictionString"].append(submission_string)

            # 4. Optional: Visualize the first image result
            if step == 1 and y_pred_fused is not None:
                visualize_prediction(image_tensor, 
                                     predictions=y_pred_fused, 
                                     title="Final WBF-Fused TTA Result")

    print("\n=======================================================")
    print(">>> TTA/WBF Inference Complete. Creating Submission File <<<")
    
    submission_df = pd.DataFrame(submission_data)
    return submission_df


test_ds = test_dataset(TEST_DIR)


submission_df = create_submission_file(yolo_model,
                       test_ds,
                       TTA_SEQUENCES,
                       IMG_SIZE,
                       CONF_THRESHOLD,
                       IOU_THRESHOLD)


submission_df.head()


# Save the final submission file
submission_df.to_csv('submission.csv', index=False)

print('Submission file created successfully!')




