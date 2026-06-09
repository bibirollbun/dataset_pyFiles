import numpy as np
import cv2 as cv
import csv
import matplotlib.pyplot as plt


import numpy as np
import csv

def decode_rle_to_mask(rle, height, width, viz=False):
    '''
    rle : run-length encoding string formatted as "start count start count ..."
    height : height of the mask
    width : width of the mask
    returns binary mask
    '''
    if not rle or rle.strip() == '':
        print(f"Skipping empty RLE: {rle}")
        return np.zeros((height, width, 3), dtype=np.float32)

    try:
        rle = np.array(rle.split(), dtype=int).reshape(-1, 2)
    except ValueError:
        print(f"Invalid RLE format: {rle}")
        return np.zeros((height, width, 3), dtype=np.float32)

    mask = np.zeros((height * width, 3), dtype=np.float32)
    color = np.random.rand(3) if viz else [1, 1, 1]

    for start, count in rle:
        mask[start:start + count, :] = color

    return mask.reshape(height, width, 3)

# Open train file
with open('/kaggle/input/ai-dl-multiclass-segmentation/train.csv', 'r', encoding='utf-8') as train:
    reader = csv.reader(train)
    # Skip headers row
    next(reader)

    # Pull the masks for the first image
    row = next(iter(reader))
    print(row[0])
    mask1 = decode_rle_to_mask(row[4], int(row[1]), int(row[2]))

    row = next(iter(reader))
    mask2 = decode_rle_to_mask(row[4], int(row[1]), int(row[2]))

    row = next(iter(reader))
    mask3 = decode_rle_to_mask(row[4], int(row[1]), int(row[2]))



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


!pip install tensorflow opencv-python matplotlib numpy



import os
import numpy as np
import csv
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Define paths
data_paths = {
    "images": "/kaggle/input/ai-dl-multiclass-segmentation/TrainImages",
    "masks": "/kaggle/working/GeneratedMasks"
}

# Create directory for masks if not present
os.makedirs(data_paths["masks"], exist_ok=True)

# Generate masks from CSV data
with open('/kaggle/input/ai-dl-multiclass-segmentation/train.csv', 'r', encoding='utf-8') as train_file:
    reader = csv.reader(train_file)
    next(reader)  # Skip header

    for row in reader:
        image_name, height, width, class_number, rle = row[0], int(row[1]), int(row[2]), int(row[3]), row[4]
        mask = decode_rle_to_mask(rle, height, width)
        mask_path = os.path.join(data_paths["masks"], f"{os.path.splitext(image_name)[0]}_mask.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

print(f"Masks saved in {data_paths['masks']}")

def load_data(images_path, masks_path, target_size=(256, 256)):
    image_files = sorted(os.listdir(images_path))
    mask_files = sorted(os.listdir(masks_path))

    images = []
    masks = []

    for img_file, mask_file in zip(image_files, mask_files):
        img = load_img(os.path.join(images_path, img_file), target_size=target_size)
        img = img_to_array(img) / 255.0

        mask = load_img(os.path.join(masks_path, mask_file), target_size=target_size, color_mode='grayscale')
        mask = img_to_array(mask).astype(np.uint8)

        # Ensure mask values map to valid class indices (0–3)
        mask = np.clip(mask, 0, 3)  # Adjust values if necessary

        images.append(img)
        masks.append(mask)

    images = np.array(images)
    masks = np.array(masks).squeeze(-1)  # Remove extra channel dimension

    # Convert masks to categorical (4 classes)
    masks = to_categorical(masks, num_classes=4)

    return images, masks


# Load dataset
images, masks = load_data(data_paths["images"], data_paths["masks"], target_size=(256, 256))

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")



# U-Net model
def unet_model(input_size=(256, 256, 3), num_classes=4):
    inputs = layers.Input(input_size)

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

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = layers.UpSampling2D((2, 2))(c5)
    u6 = layers.concatenate([u6, c4], axis=-1)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.UpSampling2D((2, 2))(c6)
    u7 = layers.concatenate([u7, c3], axis=-1)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.UpSampling2D((2, 2))(c7)
    u8 = layers.concatenate([u8, c2], axis=-1)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.UpSampling2D((2, 2))(c8)
    u9 = layers.concatenate([u9, c1], axis=-1)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c9)

    return models.Model(inputs=[inputs], outputs=[outputs])

# Build and compile the model
model = unet_model(input_size=(256, 256, 3), num_classes=4)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# Train the model
epochs = 20
batch_size = 16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import tensorflow as tf


# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Compile the model with IoU as a metric
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[early_stopping]
)


# Save using the recommended `.keras` extension
model.save("unet_model.keras")
print("Model saved in Keras format: 'unet_model.keras'")



import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Define paths
test_csv_path = '/kaggle/input/ai-dl-multiclass-segmentation/test_class.csv'
test_images_path = '/kaggle/input/ai-dl-multiclass-segmentation/TestImages'
model_path = '/kaggle/working/unet_model.keras'
submission_path = '/kaggle/working/submission.csv'

# Load the saved model
model = load_model(model_path)

def preprocess_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def postprocess_mask(pred, target_size=(256, 256)):
    pred = np.argmax(pred, axis=-1)
    pred = np.squeeze(pred)
    pred = cv2.resize(pred, target_size, interpolation=cv2.INTER_NEAREST)
    return pred

def encode_rle(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# Load test images
test_df = pd.read_csv(test_csv_path)
submission_data = []

for _, row in test_df.iterrows():
    image_name = row['ImageName']
    class_number = row['ClassNumber']
    original_height, original_width = int(row['ImageHeight']), int(row['ImageWidth'])
    
    # Preprocess image
    image_path = os.path.join(test_images_path, image_name + '.jpg')
    img = preprocess_image(image_path)
    
    # Predict mask
    pred = model.predict(img)[0]
    mask = postprocess_mask(pred, target_size=(original_height, original_width))
    binary_mask = (mask == class_number).astype(np.uint8)
    
    # Encode mask
    if binary_mask.sum() > 0:
        rle = encode_rle(binary_mask)
    else:
        rle = ''  # Empty RLE for empty masks
    
    submission_data.append({'ImageId_ClassId': f"{image_name}_{class_number}", 'EncodedPixels': rle})

# Save submission
submission_df = pd.DataFrame(submission_data)
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")



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


import matplotlib.pyplot as plt

def visualize_predictions(image_path, model_path):
    model = tf.keras.models.load_model(model_path, custom_objects={'dice_loss': dice_loss})

    img = load_img(image_path, target_size=(256, 256))  # Assuming H, W = 256, 256
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0]
    pred_mask = np.argmax(pred, axis=-1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(load_img(image_path))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask, cmap='viridis')
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.show()

# Example usage
image_path = '/kaggle/input/ai-dl-multiclass-segmentation/TestImages/02.jpg' 
model_path = '/kaggle/working/unet_model.keras'
visualize_predictions(image_path, model_path)



def generate_submission(test_csv_path, test_images_path, model_path, submission_path):
    test_df = pd.read_csv(test_csv_path)
    model = tf.keras.models.load_model(model_path, custom_objects={'dice_loss': dice_loss})

    submission_data = []
    for _, row in test_df.iterrows():
        image_name = row['ImageName']
        class_number = row['ClassNumber']
        original_height, original_width = int(row['ImageHeight']), int(row['ImageWidth'])

        # Preprocess image
        image_path = os.path.join(test_images_path, image_name + '.jpg')
        img = load_img(image_path, target_size=(H, W))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict and process mask
        pred = model.predict(img)[0]
        pred_mask = np.argmax(pred, axis=-1)
        binary_mask = (pred_mask == class_number).astype(np.uint8)
        binary_mask = cv2.resize(binary_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        # Debugging output
        print(f"Image: {image_name}, Class: {class_number}")
        print(f"Predicted Mask Unique Values: {np.unique(pred_mask)}")
        print(f"Binary Mask Unique Values: {np.unique(binary_mask)}")

        # Encode mask if not empty
        if binary_mask.sum() > 0:
            pixels = binary_mask.flatten()
            pixels = np.concatenate([[0], pixels, [0]])
            runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            rle = ' '.join(map(str, runs))
        else:
            rle = ''  # Empty RLE for empty masks

        submission_data.append({'ImageName': f"{image_name}_{class_number}", 'Encoding': rle})

    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


# Paths for submission generation
test_csv_path = '/kaggle/input/ai-dl-multiclass-segmentation/test_class.csv'
test_images_path = '/kaggle/input/ai-dl-multiclass-segmentation/TestImages'
model_path = 'best_model.keras'
submission_path = '/kaggle/working/submission3.csv'

# Generate submission
generate_submission(test_csv_path, test_images_path, model_path, submission_path)



import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Paths
model_path = '/kaggle/working/best_model.keras'  # Path to your saved model
test_images_path = '/kaggle/input/ai-dl-multiclass-segmentation/TestImages'  # Path to test images
submission_path = '/kaggle/working/sample_submission1.csv'  # Output submission file

# Load trained model
model = load_model(model_path)

# Constants
IMG_HEIGHT, IMG_WIDTH = 256, 256  # Model input size
NUM_CLASSES = 4  # Number of classes (1=head, 2=body, 3=legs, 4=tail)

# Function to encode mask to RLE
def encode_mask_to_rle(mask):
    '''
    mask : binary numpy array of shape (height, width)
    Returns RLE as string
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])  # Add padding to handle edge cases
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(map(str, runs))

# Function to preprocess test images
def preprocess_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size)  # Resize image
    img = img_to_array(img) / 255.0  # Normalize
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Prepare submission DataFrame
submission = []

# List of test images
test_image_files = sorted(os.listdir(test_images_path))  # List of test image filenames

# Predict masks and generate RLE
for img_file in test_image_files:
    # Preprocess the test image
    img_path = os.path.join(test_images_path, img_file)
    preprocessed_img = preprocess_image(img_path)

    # Predict the mask using the trained model
    pred_mask = model.predict(preprocessed_img, verbose=0)[0]  # Output shape: (256, 256, NUM_CLASSES)
    pred_mask = np.argmax(pred_mask, axis=-1)  # Convert probabilities to class indices

    # Iterate over each class (1 to 4)
    for class_number in range(1, NUM_CLASSES + 1):
        binary_mask = (pred_mask == class_number).astype(np.uint8)  # Generate binary mask for the class
        rle = encode_mask_to_rle(binary_mask)  # Encode the binary mask to RLE
        
        # Prepare the submission entry
        submission.append([f"{os.path.splitext(img_file)[0]}_{class_number}", rle])

# Create the DataFrame
submission_df = pd.DataFrame(submission, columns=['ImageId_ClassId', 'EncodedPixels'])

# Save to CSV
submission_df.to_csv(submission_path, index=False)
print(f"Sample submission file saved successfully at {submission_path}!")

# Display a sample of the submission DataFrame
print(submission_df.head())
print(f"Total rows in submission: {len(submission_df)}")


import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Paths
test_csv_path = '/kaggle/input/ai-dl-multiclass-segmentation/test_class.csv'  # Path to test CSV
test_images_path = '/kaggle/input/ai-dl-multiclass-segmentation/TestImages'  # Path to test images
model_path = '/kaggle/working/best_model.keras'  # Path to trained model
output_csv_path = '/kaggle/working/sample_submission.csv'  # Output submission file

# Constants
IMG_HEIGHT, IMG_WIDTH = 256, 256  # Model input size
NUM_CLASSES = 4  # Number of classes (head, body, legs, tail)

# Load trained model
model = load_model(model_path)
from tensorflow.keras.losses import Loss

# Custom Dice Loss function
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# Load trained model with custom loss function
model = load_model(model_path, custom_objects={'dice_loss': dice_loss})


# Function to encode mask to RLE
def encode_mask_to_rle(mask):
    '''
    mask : binary numpy array of shape (height, width)
    Returns RLE as string
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])  # Add padding to handle edge cases
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(map(str, runs))

# Function to preprocess test images
def preprocess_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size)  # Resize image
    img = img_to_array(img) / 255.0  # Normalize
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Load test CSV
test_csv = pd.read_csv(test_csv_path)

# Predict and fill the Encoding column
encodings = []

for _, row in test_csv.iterrows():
    image_name = row['ImageName']
    class_number = row['ClassNumber']
    original_height = int(row['ImageHeight'])
    original_width = int(row['ImageWidth'])

    # Preprocess the image
    image_path = os.path.join(test_images_path, image_name + '.jpg')
    preprocessed_img = preprocess_image(image_path)

    # Predict the mask using the trained model
    pred_mask = model.predict(preprocessed_img, verbose=0)[0]  # Output shape: (256, 256, NUM_CLASSES)
    pred_mask = np.argmax(pred_mask, axis=-1)  # Convert probabilities to class indices

    # Generate binary mask for the current class
    binary_mask = (pred_mask == class_number).astype(np.uint8)  # Binary mask for the class

    # Resize the binary mask back to original dimensions
    binary_mask = np.expand_dims(binary_mask, axis=-1)  # Add channel dimension
    binary_mask = tf.image.resize(binary_mask, (original_height, original_width), method='nearest')
    binary_mask = tf.squeeze(binary_mask).numpy().astype(np.uint8)  # Remove channel dimension

    # Encode the binary mask to RLE
    rle = encode_mask_to_rle(binary_mask)
    encodings.append(rle)

# Update the Encoding column
test_csv['Encoding'] = encodings

# Save the updated test CSV
test_csv.to_csv(output_csv_path, index=False)
print(f"Updated test CSV saved to {output_csv_path}")

# Display a sample of the updated CSV
print(test_csv.head())



# Load the correct test CSV file
test_csv_path = '/kaggle/input/ai-dl-multiclass-segmentation/Sample Submission.csv'
test_csv = pd.read_csv(test_csv_path)

# Now proceed with your processing steps
print(test_csv.head())  # Check the first few rows to understand the structure


