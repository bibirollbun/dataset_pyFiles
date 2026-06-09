import numpy as np
import cv2 as cv
import csv
import matplotlib.pyplot as plt


!ls /kaggle/input/



!ls /kaggle/input/ai-dl-multiclass-segmentation/



def decode_rle_to_mask(rle, height, width, viz=False):
    '''
    rle : run-length as string formated (start value, count)
    height : height of the mask 
    width : width of the mask
    returns binary mask
    '''
    rle = np.array(rle.split(' ')).reshape(-1, 2)
    mask = np.zeros((height*width, 1, 3))
    if viz:
        color = np.random.rand(3)
    else:
        color = [1,1,1]
    for i in rle:
        mask[int(i[0]):int(i[0])+int(i[1]), :, :] = color

    return mask.reshape(height, width, 3)


# Open train file
with open('/kaggle/input/ai-dl-multiclass-segmentation/train.csv', 'r', encoding='utf-8') as train:
    reader = csv.reader(train)
    # skip headers row
    next(reader)

    # pull the masks for the first image
    row = next(iter(reader))
    print(row[0])
    mask1 = decode_rle_to_mask(row[4],int(row[1]), int(row[2]))
    row = next(iter(reader))
    mask2 = decode_rle_to_mask(row[4],int(row[1]), int(row[2]))
    row = next(iter(reader))
    mask3 = decode_rle_to_mask(row[4],int(row[1]), int(row[2]))    


# Head mask
plt.imshow(mask1)


# Body mask
plt.imshow(mask2)


# Legs mask
plt.imshow(mask3)


# combine masks for training
mixed_mask = mask1*1+mask2*2+mask3*3 #+mask4*4

# This is an image multi-segmentation task. The classes are:
    # 1: Head
    # 2: Body
    # 3: Legs
    # 4: Tail
# As seen below (notice this image does not include a tail in the image so no class 4)
print(np.unique(mixed_mask))

## For visualization purposes only, we can see the different segments colorcoded in the mixed mask by dividing my the sum of the class values (normalization)
## However, for submission, you should split the mask into 4 binary masks and RLE encode them individually as shown below.

plt.imshow(mixed_mask/6) # normalized just to show different segmentations (divide by sum of values)


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import cv2


# Load and resize images from a directory
def load_images_from_directory(directory_path, target_size=(64, 64)):
    images = []
    for img_name in os.listdir(directory_path):
        img_path = os.path.join(directory_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            images.append(img)
    images = np.array(images, dtype=np.float32) / 255.0  # Normalize
    print(f"Loaded {len(images)} images with shape {images.shape}")
    return images

# Load train images
train_images_path = "/kaggle/input/ai-dl-multiclass-segmentation/TrainImages"
X_train = load_images_from_directory(train_images_path)


# Apply strong augmentations for contrastive pairs
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2)
])


# Create a contrastive learning model using MobileNetV2
def create_pretraining_model(input_shape=(64, 64, 3)):
    # Base encoder (MobileNetV2 is lighter than ResNet50)
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=input_shape)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(128, activation='relu')(x)  # Projection head
    representation = layers.Dense(128)(x)

    # Dual-input for contrastive learning
    input_view_1 = tf.keras.Input(shape=input_shape)
    input_view_2 = tf.keras.Input(shape=input_shape)

    # Encode both views
    z_i = tf.keras.Model(inputs=base_model.input, outputs=representation)(input_view_1)
    z_j = tf.keras.Model(inputs=base_model.input, outputs=representation)(input_view_2)

    return tf.keras.Model(inputs=[input_view_1, input_view_2], outputs=[z_i, z_j])

# Initialize the model
pretraining_model = create_pretraining_model()
pretraining_model.summary()


# Convert the dataset into a TensorFlow Dataset for memory efficiency
batch_size = 32  # Reduced batch size for memory optimization
train_dataset = tf.data.Dataset.from_tensor_slices(X_train).batch(batch_size).shuffle(buffer_size=100)


# Compile the model with contrastive loss and Adam optimizer
loss_fn = ContrastiveLoss()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


# Define a single training step
@tf.function
def train_step(images):
    # Generate contrastive pairs
    augmented_views_1, augmented_views_2 = generate_contrastive_pairs(images)

    with tf.GradientTape() as tape:
        # Forward pass with both views
        z_i, z_j = pretraining_model([augmented_views_1, augmented_views_2], training=True)
        # Calculate contrastive loss
        loss = loss_fn(z_i, z_j)

    # Backpropagation and gradient descent
    gradients = tape.gradient(loss, pretraining_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, pretraining_model.trainable_variables))
    return loss


# Train the model with mini-batch gradient descent
EPOCHS = 10
for epoch in range(EPOCHS):
    total_loss = 0
    num_batches = 0
    for batch in train_dataset:
        loss = train_step(batch)
        total_loss += loss
        num_batches += 1
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / num_batches:.4f}")

# Save the pretrained model
pretraining_model.save("/kaggle/working/contrastive_pretrained_model.h5")
print("✅ Pretraining Completed and Model Saved!")


class NTXentLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def call(self, z_i, z_j):
        # Normalize embeddings
        z_i = tf.math.l2_normalize(z_i, axis=1)
        z_j = tf.math.l2_normalize(z_j, axis=1)

        # Combine positive pairs for contrastive comparison
        batch_size = tf.shape(z_i)[0]
        projections = tf.concat([z_i, z_j], axis=0)

        # Compute similarity matrix
        similarity_matrix = tf.matmul(projections, projections, transpose_b=True) / self.temperature

        # Mask self-similar pairs (diagonal masking)
        mask = tf.eye(2 * batch_size)
        mask = tf.cast(mask, dtype=tf.bool)
        logits = tf.where(mask, -1e9 * tf.ones_like(similarity_matrix), similarity_matrix)

        # Correct positive pair labeling
        labels = tf.range(batch_size)
        labels = tf.concat([labels, labels], axis=0)

        # Apply softmax and compute cross-entropy loss
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        return tf.reduce_mean(loss)


# Adjusting the temperature and optimizer settings
loss_fn = NTXentLoss(temperature=0.1)  # Reduced temperature for stability
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)  # Reduced LR to prevent gradient explosion


# Updated Training Step with Gradient Clipping
@tf.function
def train_step(images):
    augmented_views_1, augmented_views_2 = generate_contrastive_pairs(images)
    with tf.GradientTape() as tape:
        z_i, z_j = pretraining_model([augmented_views_1, augmented_views_2], training=True)
        z_j = tf.stop_gradient(z_j)  # Prevent collapsing gradients
        loss = loss_fn(z_i, z_j)

    # Apply gradient clipping to avoid exploding gradients
    gradients = tape.gradient(loss, pretraining_model.trainable_variables)
    gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients]  # Gradient clipping
    optimizer.apply_gradients(zip(gradients, pretraining_model.trainable_variables))
    return loss


# Train with corrected settings and loss function
EPOCHS = 10
for epoch in range(EPOCHS):
    total_loss = 0
    num_batches = 0
    for batch in train_dataset:
        loss = train_step(batch)
        total_loss += loss
        num_batches += 1
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / num_batches:.4f}")

# Save the corrected contrastive learning model
pretraining_model.save("/kaggle/working/fixed_contrastive_model_stable.h5")
print("✅ Fixed Training Completed with Stable Loss!")


def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


mask1.shape


# IMPORTANT || Convert mask to grayscale
mask1 = mask1[:,:,0] # or using other methods depending on mask
mask1.shape


rle = encode_mask_to_rle(mask1)
rle







