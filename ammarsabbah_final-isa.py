import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Updated dataset paths for Kaggle
train_csv_path = "/kaggle/input/ai-dl-multiclass-segmentation/train.csv"
test_class_csv_path = "/kaggle/input/ai-dl-multiclass-segmentation/test_class.csv"
train_images_path = "/kaggle/input/ai-dl-multiclass-segmentation/TrainImages"
test_images_path = "/kaggle/input/ai-dl-multiclass-segmentation/TestImages"

# Load the CSV files
train_df = pd.read_csv(train_csv_path)
test_class_df = pd.read_csv(test_class_csv_path)

# RLE Decoding Function
def rle_decode(mask_rle, shape):
    height, width = shape
    mask = np.zeros(height * width, dtype=np.uint8)
    if not mask_rle or mask_rle == "nan":
        return mask.reshape(height, width)
    rle_pairs = np.array(mask_rle.split(), dtype=int).reshape(-1, 2)
    for start, length in rle_pairs:
        start -= 1  # Convert to zero indexing
        mask[start:start + length] = 1
    return mask.reshape(height, width)

# Testing the decoding function
sample_rle = train_df.iloc[0]['Encoding']
sample_height = train_df.iloc[0]['ImageHeight']
sample_width = train_df.iloc[0]['ImageWidth']

decoded_mask = rle_decode(sample_rle, (sample_height, sample_width))


# Visualize the decoded mask to verify it
plt.figure(figsize=(6,6))
plt.title(f"Decoded Mask for Image: {train_df.iloc[0]['ImageName']}")
plt.imshow(decoded_mask, cmap='gray')
plt.axis('off')
plt.show()



from sklearn.model_selection import train_test_split

# Splitting the dataset into train and validation sets (80% train, 20% validation)
train_df_split, val_df_split = train_test_split(train_df, test_size=0.2, random_state=42)

# Displaying the number of samples in each split
len(train_df_split), len(val_df_split)



import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Paths for Kaggle (update these as needed)
train_images_path = "/kaggle/input/ai-dl-multiclass-segmentation/TrainImages"
train_masks_path = "/kaggle/working/TrainMasks"
val_masks_path = "/kaggle/working/ValMasks"

# Create directories if they don't exist
os.makedirs(train_masks_path, exist_ok=True)
os.makedirs(val_masks_path, exist_ok=True)

# Function to decode RLE
def rle_decode(mask_rle, shape):
    height, width = shape
    mask = np.zeros(height * width, dtype=np.uint8)
    if not mask_rle or mask_rle == "nan":
        return mask.reshape(height, width)
    rle_pairs = np.array(mask_rle.split(), dtype=int).reshape(-1, 2)
    for start, length in rle_pairs:
        start -= 1  # Convert to zero indexing
        mask[start:start + length] = 1
    return mask.reshape(height, width)

# Function to generate and save masks ensuring correct dimensions
def generate_and_save_masks_with_check(df, save_path, images_path):
    """
    Generate binary masks from RLE and save them as .png files only if they match the image size.
    """
    mismatched_masks = 0
    for index, row in df.iterrows():
        img_name = row['ImageName']
        height, width = row['ImageHeight'], row['ImageWidth']
        
        # Load the corresponding image to check dimensions
        img_path = os.path.join(images_path, f"{img_name}.jpg")
        if not os.path.exists(img_path):
            print(f"Image {img_name} not found in {images_path}")
            continue
            
        image = Image.open(img_path)
        image_width, image_height = image.size

        # Decode the mask
        mask = rle_decode(row['Encoding'], (height, width))

        # Check if the image size matches the mask size
        if (height, width) != (image_height, image_width):
            print(f"Size mismatch for {img_name}: Image ({image_height}, {image_width}) vs Mask ({height}, {width})")
            mismatched_masks += 1
            continue

        # Save the mask only if dimensions match
        mask_file_path = os.path.join(save_path, f"{img_name}_mask.png")
        plt.imsave(mask_file_path, mask, cmap='gray')

    print(f"Finished generating masks. {mismatched_masks} mismatched masks were found.")

# Generate and save masks for both training and validation sets with checks
generate_and_save_masks_with_check(train_df_split, train_masks_path, train_images_path)
generate_and_save_masks_with_check(val_df_split, val_masks_path, train_images_path)


import tensorflow as tf
from tensorflow.keras import layers, models

def unet_model(input_shape=(256, 256, 3), num_classes=4):
    """
    Build a U-Net model for multiclass segmentation.
    
    Parameters:
    - input_shape: Shape of the input image (height, width, channels)
    - num_classes: Number of classes to predict (for multiclass segmentation)
    
    Returns:
    - TensorFlow U-Net model
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

    # Decoder
    u1 = layers.UpSampling2D((2, 2))(c4)
    u1 = layers.Concatenate()([u1, c3])
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u1)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    u2 = layers.UpSampling2D((2, 2))(c5)
    u2 = layers.Concatenate()([u2, c2])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u3 = layers.UpSampling2D((2, 2))(c6)
    u3 = layers.Concatenate()([u3, c1])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u3)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    # Output layer for multiclass segmentation
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c7)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Initialize and compile the model
unet = unet_model(input_shape=(256, 256, 3), num_classes=4)

# Display model summary
unet.summary()


!pip install tensorflow-addons


import tensorflow as tf
from tensorflow.keras import layers, models
import random

# Custom NT-Xent Loss (Contrastive Loss) without TensorFlow Addons
def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    Custom NT-Xent loss without TensorFlow Addons.
    """
    # Normalize embeddings to unit length
    z_i = tf.math.l2_normalize(z_i, axis=1)
    z_j = tf.math.l2_normalize(z_j, axis=1)
    
    # Cosine similarity matrix
    logits = tf.matmul(z_i, z_j, transpose_b=True) / temperature
    batch_size = tf.shape(z_i)[0]

    # Labels for positive pairs
    labels = tf.range(batch_size)
    labels = tf.concat([labels, labels], axis=0)

    # Create positive pairs from z_i and z_j
    logits = tf.concat([logits, tf.transpose(logits)], axis=0)
    
    # Compute the loss using sparse categorical crossentropy
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return tf.reduce_mean(loss)

# Testing the custom NT-Xent loss
z_i = tf.random.normal([32, 128])  # Simulated embeddings for batch size 32
z_j = tf.random.normal([32, 128])
loss_value = nt_xent_loss(z_i, z_j)
print(f"Sample NT-Xent Loss: {loss_value.numpy():.4f}")


import tensorflow as tf
from tensorflow.keras import layers, models

def get_simclr_model(input_shape=(256, 256, 3), projection_dim=128):
    inputs = layers.Input(shape=input_shape)

    # Simple CNN Backbone
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Projection Head (2-layer MLP)
    x = layers.Dense(512, activation="relu")(x)
    outputs = layers.Dense(projection_dim)(x)

    return models.Model(inputs, outputs)

# Initialize the model
simclr_model = get_simclr_model()
simclr_model.summary()


# Custom NT-Xent Loss (Contrastive Loss) without TensorFlow Addons
def nt_xent_loss(z_i, z_j, temperature=0.5):
    z_i = tf.math.l2_normalize(z_i, axis=1)
    z_j = tf.math.l2_normalize(z_j, axis=1)
    logits = tf.matmul(z_i, z_j, transpose_b=True) / temperature
    batch_size = tf.shape(z_i)[0]
    labels = tf.range(batch_size)
    labels = tf.concat([labels, labels], axis=0)
    logits = tf.concat([logits, tf.transpose(logits)], axis=0)
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return tf.reduce_mean(loss)


# Training step using the custom NT-Xent loss
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def train_step(images):
    with tf.GradientTape() as tape:
        z_i = simclr_model(images[0], training=True)
        z_j = simclr_model(images[1], training=True)
        loss = nt_xent_loss(z_i, z_j)
    gradients = tape.gradient(loss, simclr_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, simclr_model.trainable_variables))
    return loss


import tensorflow as tf

# Define SimCLR augmentations for your real dataset
def simclr_augmentations(image):
    image = tf.image.random_crop(image, size=[256, 256, 3])  
    image = tf.image.random_flip_left_right(image)         
    image = tf.image.random_brightness(image, max_delta=0.5) 
    image = tf.image.random_contrast(image, lower=0.2, upper=0.8)  
    image = tf.clip_by_value(image, 0.0, 1.0)                
    return image

def load_and_preprocess_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256]) / 255.0
    return image

def prepare_simclr_batch(file_path):
    # Load image and apply two different augmentations for contrastive learning
    image = load_and_preprocess_image(file_path)
    return simclr_augmentations(image), simclr_augmentations(image)

# Load the actual test dataset from Kaggle
test_image_files = tf.data.Dataset.list_files("/kaggle/input/ai-dl-multiclass-segmentation/TestImages/*.jpg")
test_dataset = test_image_files.map(prepare_simclr_batch).batch(32).shuffle(buffer_size=100)

# Confirm the dataset structure
for batch in test_dataset.take(1):
    print(f"Batch shape: {batch[0].shape}, Augmented Pair Shape: {batch[1].shape}")


# Optimizer for SimCLR pretraining
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Custom SimCLR training step using NT-Xent loss
@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        z_i = simclr_model(batch[0], training=True)  # First augmented view
        z_j = simclr_model(batch[1], training=True)  # Second augmented view
        loss = nt_xent_loss(z_i, z_j)
    
    gradients = tape.gradient(loss, simclr_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, simclr_model.trainable_variables))
    return loss

# SimCLR Training Loop (10 epochs for demonstration)
EPOCHS = 10
for epoch in range(EPOCHS):
    total_loss = 0.0
    for step, batch in enumerate(test_dataset):
        loss = train_step(batch)
        total_loss += loss
    avg_loss = total_loss / (step + 1)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss.numpy():.4f}")


# Save the pretrained SimCLR model for future fine-tuning
simclr_model.save("/kaggle/working/simclr_pretrained_model.h5")
print("SimCLR Pretraining Completed and Model Saved!")


import tensorflow as tf
from tensorflow.keras import layers, models

# Load the pretrained SimCLR model
pretrained_simclr = tf.keras.models.load_model("/kaggle/working/simclr_pretrained_model.h5")

# Freeze the pretrained layers
for layer in pretrained_simclr.layers:
    layer.trainable = False


def build_unet_with_simclr(pretrained_simclr, num_classes=4):
    """
    Build a U-Net model with a frozen SimCLR backbone and dynamic feature map size.
    """
    inputs = pretrained_simclr.input
    x = pretrained_simclr(inputs, training=False)

    # ✅ Ensure the output is a single tensor (not a list)
    if isinstance(x, list):
        x = x[0]  # Take the first tensor if multiple outputs exist

    # ✅ Check the shape after ensuring it's a tensor
    print(f"Corrected SimCLR Output Shape: {x.shape}")

    # ✅ Ensure the output is flattened
    x = layers.Flatten()(x) if len(x.shape) > 2 else x

    # ✅ Expand the feature vector with a Dense layer before reshaping
    x = layers.Dense(8 * 8 * 64, activation='relu')(x)
    
    # ✅ Reshape into a spatial feature map
    x = layers.Reshape((8, 8, 64))(x)
    
    # ✅ U-Net Decoder with upsampling
    x = layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)

    # ✅ Output Layer (4 classes with softmax)
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# ✅ Initialize the corrected U-Net with the pretrained SimCLR backbone
unet_model = build_unet_with_simclr(pretrained_simclr)
unet_model.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
unet_model.summary()


import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Function to load images and masks together for fine-tuning
def preprocess_image_mask(image_path, mask_path):
    # Load the image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256]) / 255.0

    # Load the mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [256, 256], method='nearest')
    mask = tf.cast(mask, tf.int32)
    mask = tf.squeeze(mask)
    mask = to_categorical(mask, num_classes=4)  # Convert to categorical format

    return image, mask

# Prepare the training and validation datasets
train_image_paths = tf.data.Dataset.list_files("/kaggle/input/ai-dl-multiclass-segmentation/TrainImages/*.jpg")
train_mask_paths = tf.data.Dataset.list_files("/kaggle/working/TrainMasks/*.png")

train_dataset = tf.data.Dataset.zip((train_image_paths, train_mask_paths))
train_dataset = train_dataset.map(preprocess_image_mask).batch(32).shuffle(buffer_size=100)

val_image_paths = tf.data.Dataset.list_files("/kaggle/input/ai-dl-multiclass-segmentation/TrainImages/*.jpg")
val_mask_paths = tf.data.Dataset.list_files("/kaggle/working/ValMasks/*.png")

val_dataset = tf.data.Dataset.zip((val_image_paths, val_mask_paths))
val_dataset = val_dataset.map(preprocess_image_mask).batch(32)


# Custom Intersection over Union (IoU) metric
def mean_iou(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)
    
    # Intersection and union calculation
    intersection = tf.reduce_sum(tf.cast(y_pred * y_true, tf.float32))
    union = tf.reduce_sum(tf.cast(y_pred + y_true, tf.float32)) - intersection
    return intersection / (union + tf.keras.backend.epsilon())

# Adding IoU to the model's compilation
unet_model.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy', mean_iou])


def build_unet_with_simclr(pretrained_simclr, num_classes=4):
    """
    Build a U-Net model with a frozen SimCLR backbone and dynamic feature map size.
    """
    inputs = pretrained_simclr.input
    x = pretrained_simclr(inputs, training=False)

    # Ensure output is a single tensor
    if isinstance(x, list):
        x = x[0]

    # Expand feature vector
    x = layers.Dense(8 * 8 * 64, activation='relu')(x)
    x = layers.Reshape((8, 8, 64))(x)

    # ✅ Additional Upsampling Layers to Match 256x256 Output Size
    x = layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)  # 16x16

    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)  # 32x32

    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)  # 64x64

    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)  # 128x128

    # ✅ Final Upsampling to Match 256x256
    x = layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)  # 256x256

    # ✅ Output Layer with Correct Size
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# ✅ Reinitialize and Compile the Corrected U-Net
unet_model = build_unet_with_simclr(pretrained_simclr)
unet_model.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
unet_model.summary()


# Fine-tune the U-Net model with the labeled dataset
EPOCHS = 20
history = unet_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    verbose=1
)

# Save the fine-tuned model for future use
unet_model.save("/kaggle/working/fine_tuned_unet_model.h5")
print("✅ Fine-tuning completed and model saved!")


# If using RLE encoding, ensure masks are generated correctly
def rle_decode(mask_rle, shape=(256, 256)):
    '''
    Decode RLE encoded mask into a 2D numpy array.
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1
    return mask.reshape(shape)

# Apply the fix while loading the mask
def preprocess_image_mask_fixed(image_path, mask_rle):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256]) / 255.0

    # Decode the RLE mask
    mask = rle_decode(mask_rle)
    mask = tf.cast(mask, tf.int32)
    return image, mask


# Test again for unique values in the corrected mask
sample_image, sample_mask = next(iter(train_dataset))
unique_values = np.unique(sample_mask.numpy())
print(f"✅ Fixed Unique Values in the Mask: {unique_values}")


# Fine-tune the U-Net model with the corrected masks
EPOCHS = 20
history = unet_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    verbose=1
)

# Save the fine-tuned model for future use
unet_model.save("/kaggle/working/fine_tuned_unet_model.h5")
print("✅ Fine-tuning completed and model saved!")




