!unzip -o -q -u "/kaggle/input/tgs-salt-identification-challenge/train.zip" -d "/kaggle/working/train/"
!unzip -o -q -u "/kaggle/input/tgs-salt-identification-challenge/test.zip" -d "/kaggle/working/test/"


import os
import numpy as np
import cv2
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split


def build_unet(input_shape=(128, 128, 3)):
    # Encoder
    inputs = Input(input_shape)
    
    # Contracting Path (Encoder)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # Bridge
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    # Expansive Path (Decoder)
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model




def load_data(image_path, mask_path):
    images = []
    masks = []
    
    # Load images and masks
    for img_name in os.listdir(image_path):
        if img_name.endswith('.png'):
            # Load image
            img = cv2.imread(os.path.join(image_path, img_name))
            img = cv2.resize(img, (128, 128))
            img = img / 255.0  # Normalize
            images.append(img)
            
            # Load corresponding mask
            mask = cv2.imread(os.path.join(mask_path, img_name), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (128, 128))
            mask = mask / 255.0  # Normalize
            masks.append(mask)
    
    return np.array(images), np.array(masks)[..., np.newaxis]

def dice_coefficient(y_true, y_pred):
    smooth = 1e-15
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) + smooth)

def train_model(train_images, train_masks, val_images, val_masks):
    # Create a MirroredStrategy for multi-GPU training
    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')
    
    # Create and compile the model within the strategy scope
    with strategy.scope():
        model = build_unet()
        model.compile(optimizer=Adam(learning_rate=1e-4),
                     loss='binary_crossentropy',
                     metrics=['accuracy', dice_coefficient])
    
    # Define callbacks
    callbacks = [
        # Reduce learning rate when validation loss plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            verbose=1,
            min_delta=1e-4,
            min_lr=1e-8
        ),
        # Early stopping if validation loss doesn't improve
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        ),
        # Save best model based on validation loss
        tf.keras.callbacks.ModelCheckpoint(
            filepath='/kaggle/working/models/best_model.h5.keras',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]
    
    history = model.fit(train_images, train_masks,
                       batch_size=32,
                       epochs=50,
                       validation_data=(val_images, val_masks),
                       callbacks=callbacks)
    
    return model, history



# Set paths
if not os.path.exists("/kaggle/working/models"):
    os.makedirs("/kaggle/working/models")
    
data_dir = "/kaggle/working/train"
image_dir = os.path.join(data_dir, "images")
mask_dir = os.path.join(data_dir, "masks")

# Load data
images, masks = load_data(image_dir, mask_dir)

# Split data
train_images, val_images, train_masks, val_masks = train_test_split(
    images, masks, test_size=0.2, random_state=42
)

# Train model
model, history = train_model(train_images, train_masks, val_images, val_masks)

# Save model

model.save("/kaggle/working/models/unet_salt.h5")


import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def calculate_metrics(y_true, y_pred, threshold=0.5):
    """Calculate precision and recall for different thresholds"""
    # Flatten the arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Convert to binary values for sklearn metrics
    y_true_binary = (y_true > 0.5).astype(np.int32)
    
    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true_binary, y_pred)
    
    # Calculate average precision
    ap = average_precision_score(y_true_binary, y_pred)
    
    return precisions, recalls, thresholds, ap

def plot_precision_recall_curve(precisions, recalls, ap):
    """Plot precision-recall curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, color='blue', label=f'AP={ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('/kaggle/working/results/precision_recall_curve.png')
    plt.close()

def find_optimal_threshold(precisions, recalls, thresholds):
    """Find optimal threshold using F1 score"""
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-15)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx]

def display_random_samples(title,images, masks,predictions, num_samples=4):
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    # Display training samples
    print(title)
    for idx in indices:
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(images[idx])
        axs[0].set_title('Seismic Image')
        
        axs[1].imshow(masks[idx].squeeze(), cmap='viridis')
        axs[1].set_title('Expected Salt')
        
        axs[2].imshow(predictions[idx].squeeze(), cmap='viridis', vmin=0, vmax=1)
        axs[2].set_title('Predicted Salt (Float)')
        
        binary_pred = (predictions[idx] > 0.5).astype(np.float32)
        axs[3].imshow(binary_pred.squeeze(), cmap='viridis')
        axs[3].set_title('Predicted Salt (Binary)')
        
        for ax in axs:
            ax.axis('off')
        plt.show()   
        
# Function to evaluate and display results for a given dataset
def evaluate_and_display(title,images, masks, dataset_name):
    print(f'Predicting on {dataset_name} dataset')
    predictions = model.predict(images)
    
    print(f'Calculating metrics for {dataset_name} dataset')
    precisions, recalls, thresholds, ap = calculate_metrics(masks, predictions)
    
    if not os.path.exists("/kaggle/working/results"):
        os.makedirs("/kaggle/working/results")
        
    plot_precision_recall_curve(precisions, recalls, ap)
    
    optimal_threshold = find_optimal_threshold(precisions, recalls, thresholds)
    print(f"Optimal threshold for {dataset_name}: {optimal_threshold:.3f}")
    
    binary_preds = (predictions > optimal_threshold).astype(np.float32)
    final_precision = np.mean(precisions)
    final_recall = np.mean(recalls)
    
    print(f"Final Precision for {dataset_name}: {final_precision:.3f}")
    print(f"Final Recall for {dataset_name}: {final_recall:.3f}")
    print(f"Average Precision for {dataset_name}: {ap:.3f}")
    
    # Display random samples
    display_random_samples(title,images, masks,predictions)

# Evaluate on training data
evaluate_and_display("Training Samples:",train_images, train_masks, "Training")

# Evaluate on validation data
evaluate_and_display("Validation Samples:",val_images, val_masks, "Validation")

