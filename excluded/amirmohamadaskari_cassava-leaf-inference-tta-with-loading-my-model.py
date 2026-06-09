import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import random


# Some configuration based on training
AUTO = tf.data.AUTOTUNE
IMAGE_SIZE = (512, 512)
BATCH_SIZE_PER_REPLICA = 8
NUM_CLASSES = 5
BATCH_SIZE = 8


DATA_DIR = '/kaggle/input/cassava-leaf-disease-classification'
MODEL_DIR = '/kaggle/input/cassava-leaf-model/tensorflow2/default/1/final_model_cassava.keras'


model = load_model(MODEL_DIR)


# This function decodes examples from the test TFRecord files.
# Note that test files contain an 'image_name' instead of a 'target' label.
def decode_test_example(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_name': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, feature_description)
    
    # Decode and process the image
    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    
    # Return the image and its ID
    return image, example['image_name']


from tensorflow.keras.applications.efficientnet import preprocess_input

def preprocess(image, label):
    # Apply the specific preprocessing required by the EfficientNet model
    image = preprocess_input(image)
    return image, label


data_augmentation = tf.keras.Sequential([
    # Geometric Transformations
    tf.keras.layers.RandomRotation(40/ 360), # Randomly rotate images
    tf.keras.layers.RandomTranslation(0.2, 0.2), # Randomly shift images horizontally and vertically
    tf.keras.layers.RandomZoom(0.2, 0.2), # Randomly zoom into images
    tf.keras.layers.RandomFlip('horizontal'), # Randomly flip images horizontally
    tf.keras.layers.RandomFlip('vertical') # Randomly flip images vertically
], name="data_augmentation")


# Create the test dataset pipeline
TEST_DIR = os.path.join(DATA_DIR, 'test_tfrecords')
test_files = tf.io.gfile.glob(os.path.join(TEST_DIR, '*.tfrec'))
test_dataset = tf.data.TFRecordDataset(test_files, num_parallel_reads= AUTO)
test_dataset = (test_dataset
                .map(decode_test_example, num_parallel_calls= AUTO)
                .map(lambda image, image_id: (preprocess_input(image), image_id), num_parallel_calls= AUTO)
                .batch(BATCH_SIZE)
                .prefetch(AUTO))

print('Test dataset created Successfully !')


# Function to get TTA predictions for a full batch of images
def tta_predict_batch(model, images, ids, num_tta):
    """
    Run Test-Time Augmentation (TTA) on a batch of images.

    Args:
        model: Trained keras model
        images: Batch of test images
        ids: Batch of image ids (filenames)
        num_tta: Number of augmentations per image

    Returns:
        preds: Averaged predictions for this batch
        names: List of image ids corresponding to preds
    """
    all_preds = []

    for _ in range(num_tta):
        # Apply augmentation to the *whole batch*
        augmented = data_augmentation(images, training=True)

        # Preprocess for EfficientNet
        preprocessed = preprocess_input(augmented)

        # Run inference on GPU (much faster in batch mode)
        preds = model.predict(preprocessed, verbose=1)

        all_preds.append(preds)

    # Average predictions across TTA rounds
    mean_preds = np.mean(all_preds, axis=0)

    # Convert ids from tf.Tensors to python strings
    names = [img_id.numpy().decode("utf-8") for img_id in ids]

    return mean_preds, names



import matplotlib.pyplot as plt
import pandas as pd

# Full Name Mapping for Cassava Leaf Diseases
LABEL_MAP = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy"
}

def check_model_on_train_images(num_samples=3, tta_runs=5):
    """
    Loads random images from train_images, runs your TTA function, 
    and compares predictions to ground truth with large visualizations.
    """
    # 1. Load the training CSV to get IDs and Labels
    train_csv_path = os.path.join(DATA_DIR, 'train.csv')
    train_df = pd.read_csv(train_csv_path)
    
    # 2. Sample random rows
    samples = train_df.sample(num_samples)
    
    # Lists to store batch data
    batch_images = []
    batch_ids = []
    ground_truths = []
    
    print(f"Loading {num_samples} training images for inference check...")
    
    # 3. Load and prepare images
    for idx, row in samples.iterrows():
        img_id = row['image_id']
        label = row['label']
        ground_truths.append(label)
        
        # Load image file
        img_path = os.path.join(DATA_DIR, 'train_images', img_id)
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE) 
        
        batch_images.append(img)
        batch_ids.append(img_id)

    # Convert to Tensor Batch
    batch_images = tf.stack(batch_images)
    batch_ids = tf.constant(batch_ids)

    # 4. Run Inference using YOUR defined TTA function
    print(f"Running TTA ({tta_runs} augmentations per image)...")
    preds, names = tta_predict_batch(model, batch_images, batch_ids, tta_runs)
    
    # 5. Visualize Results (Individual large plots)
    for i in range(num_samples):
        # Create a large figure for each image
        plt.figure(figsize=(10, 10))
        
        # Get labels and confidence
        pred_idx = np.argmax(preds[i])
        true_idx = ground_truths[i]
        confidence = preds[i][pred_idx]
        
        # Determine Color: Green if correct, Red if wrong
        color_code = 'green' if pred_idx == true_idx else 'red'
        status = "CORRECT" if pred_idx == true_idx else "WRONG"
        
        # Display image (normalized 0-1 for matplotlib)
        img_disp = batch_images[i].numpy() / 255.0
        plt.imshow(img_disp)
        
        # Create detailed title
        title_text = (
            f"Image ID: {batch_ids[i].numpy().decode('utf-8')}\n"
            f"Prediction: {LABEL_MAP[pred_idx]} ({confidence:.2%})\n"
            f"Ground Truth: {LABEL_MAP[true_idx]}\n"
            f"[{status}]"
        )
        
        plt.title(title_text, color=color_code, fontsize=16, fontweight='bold', loc='left')
        plt.axis("off")
        plt.show()

# --- Run the check ---
# I reduced num_samples to 3 so it doesn't clutter your notebook too much,
# but since they are individual plots, you can increase it if you like.
check_model_on_train_images(num_samples=6, tta_runs=5)


# Iterate through the test dataset and generate TTA predictions for each image
tta_num_augmentations = 10  # Number of augmented images to create per test image
tta_predictions = []
tta_image_names = []

for images, ids in test_dataset:
    # Run TTA for this *batch* (instead of each image individually)
    mean_preds, names = tta_predict_batch(model, images, ids, tta_num_augmentations)

    # Save results
    tta_predictions.extend(mean_preds)
    tta_image_names.extend(names)


tta_predictions = np.array(tta_predictions)
# Find the index of the highest probability for each prediction to get the final label
final_tta_labels = tf.argmax(tta_predictions, axis=1)
pred_labels = final_tta_labels.numpy()


# Create a new DataFrame with the correct order
submission_df = pd.DataFrame({
    'image_id': tta_image_names,
    'label': pred_labels
})

# Save the final submission file
submission_df.to_csv('submission.csv', index=False)

print('Submission file created successfully!')




