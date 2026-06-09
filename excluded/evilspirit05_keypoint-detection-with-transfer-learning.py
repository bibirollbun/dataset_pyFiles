import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import Sequence
from imgaug.augmentables.kps import KeypointsOnImage, Keypoint
import imgaug.augmenters as iaa
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import keras
from keras import layers
import tensorflow as tf
from keras import backend as K
%matplotlib inline


df=pd.read_csv("/kaggle/input/facial-keypoints-detection/training.zip",compression="zip")


df.head()


df.shape


df.isnull().sum()


cols=df.select_dtypes(include=["number"]).columns
df[cols]=df[cols].fillna(df[cols].mean())


df.isnull().sum()


df['ImageId'] = df.index + 1 


idlookup=pd.read_csv("/kaggle/input/facial-keypoints-detection/IdLookupTable.csv")


idlookup.head()


plt.figure(figsize=(15,10))
sns.countplot(x="FeatureName",data=idlookup,palette="cool")
plt.xticks(rotation=90)
plt.show()


idlookup.drop(columns=["Location"],axis=1,inplace=True)


idlookup.shape


idlookup.isnull().sum()


import numpy as np
import matplotlib.pyplot as plt

df["Image"] = df["Image"].apply(lambda x: np.array(x.split(), dtype=np.uint8).reshape(96, 96) if isinstance(x, str) else x)

keypoint_columns_x = [col for col in df.columns if 'x' in col]
keypoint_columns_y = [col.replace('x', 'y') for col in keypoint_columns_x]

def visualize_samples(df, num_samples=6):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for i, ax in enumerate(axes.flat):
        if i >= num_samples:
            break
        sample_image = df.iloc[i]["Image"]
        sample_keypoints_x = df.iloc[i][keypoint_columns_x].values
        sample_keypoints_y = df.iloc[i][keypoint_columns_y].values

        keypoints = np.column_stack((sample_keypoints_x, sample_keypoints_y))
        
        ax.imshow(sample_image, cmap="gray")
        for (x, y) in keypoints:
            if not np.isnan(x) and not np.isnan(y):
                ax.scatter(x, y, c="red", marker="x", s=30)
        
        ax.axis("off")

    plt.tight_layout()
    plt.show()

visualize_samples(df)



df_merge= pd.merge(df, idlookup, on='ImageId', how='left')

df_merge.head()



df_merge.columns


len(df_merge.columns[: -4])


df_merge['RowId'] = range(len(df_merge))
# Fill FeatureName with facial keypoint locations based on column names
feature_names = [col.rsplit('_', 1)[0] for col in df_merge.columns if col.endswith('_x')]
df_merge['FeatureName'] = df_merge.apply(lambda row: feature_names[row.name % len(feature_names)], axis=1)


df_merge.isnull().sum()


df_merge["FeatureName"].value_counts()


IMG_SIZE = 96
BATCH_SIZE = 32
NUM_KEYPOINTS=30
EPOCH=15


def visualize_keypoints(images, keypoints, titles=None):
    # Set up the figure
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 25))
    [ax.axis('off') for ax in axes.flatten()]
    
    # Define feature groups and their properties
    feature_groups = {
        'eyes': {
            'color': 'red',
            'indices': list(range(0, 12)),  # first 12 coordinates (6 points) are eye-related
            'label': 'Eyes'
        },
        'eyebrows': {
            'color': 'blue',
            'indices': list(range(12, 20)),  # next 8 coordinates (4 points) are eyebrow-related
            'label': 'Eyebrows'
        },
        'nose': {
            'color': 'green',
            'indices': list(range(20, 22)),  # next 2 coordinates (1 point) is nose
            'label': 'Nose'
        },
        'mouth': {
            'color': 'yellow',
            'indices': list(range(22, 30)),  # last 8 coordinates (4 points) are mouth-related
            'label': 'Mouth'
        }
    }
    
    # Process each image
    for idx, (image, keypoint) in enumerate(zip(images, keypoints)):
        if idx >= len(axes):
            break
            
        # Show original image
        axes[idx, 0].imshow(image, cmap='gray')
        axes[idx, 0].set_title(f'Original Image {idx+1}')
        
        # Show image with keypoints
        axes[idx, 1].imshow(image, cmap='gray')
        axes[idx, 1].set_title(f'Facial Keypoints {idx+1}')
        
        # Reshape keypoints to (x,y) pairs
        keypoint_coords = keypoint.reshape(-1, 2)
        
        # Plot each feature group
        for group_name, group_info in feature_groups.items():
            start_idx = group_info['indices'][0] // 2
            end_idx = (group_info['indices'][-1] // 2) + 1
            
            # Get keypoints for this feature group
            group_keypoints = keypoint_coords[start_idx:end_idx]
            
            # Plot points
            for x, y in group_keypoints:
                # Convert normalized coordinates to pixel coordinates
                x_px = x * IMG_SIZE
                y_px = y * IMG_SIZE
                
                # Plot point
                axes[idx, 1].scatter(
                    x_px, y_px,
                    c=group_info['color'],
                    marker='x',
                    s=100,
                    linewidth=2,
                    label=group_info['label']
                )
                
                # Add circle for visibility
                circle = plt.Circle(
                    (x_px, y_px),
                    2,
                    color=group_info['color'],
                    fill=False,
                    linewidth=1
                )
                axes[idx, 1].add_artist(circle)
        
        # Add legend to first image only
        if idx == 0:
            # Create legend handles
            handles = [plt.Line2D(
                [0], [0],
                marker='x',
                color='w',
                markerfacecolor=group_info['color'],
                markersize=10,
                label=group_info['label'],
                markeredgecolor=group_info['color']
            ) for group_info in feature_groups.values()]
            
            # Add legend
            axes[idx, 1].legend(
                handles=handles,
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                borderaxespad=0.
            )
    
    plt.tight_layout()
    plt.show()




class KeyPointsDataset:
    def __init__(self, data, aug_pipeline, batch_size=32):
        self.data = data.copy()
        self.aug_pipeline = aug_pipeline
        self.batch_size = batch_size
        
        # Define feature columns in exact order
        self.feature_columns = [
            'left_eye_center_x', 'left_eye_center_y',
            'right_eye_center_x', 'right_eye_center_y',
            'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
            'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
            'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
            'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
            'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
            'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
            'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
            'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
            'nose_tip_x', 'nose_tip_y',
            'mouth_left_corner_x', 'mouth_left_corner_y',
            'mouth_right_corner_x', 'mouth_right_corner_y',
            'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
            'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y'
        ]
        
        self.image_ids = self.data['ImageId'].values
        self.indexes = np.arange(len(self.image_ids))
        np.random.shuffle(self.indexes)

    def __len__(self):
        
        total_batches= int(np.floor(len(self.image_ids) / self.batch_size))
        print(f"Total batches : {total_batches}")
        return total_batches
        

    def get_image(self, image_id):
        image_data = self.data[self.data["ImageId"] == image_id]["Image"].iloc[0]
        
        # Convert string of pixels to numpy array
        if isinstance(image_data, str):
            image = np.array([int(pixel) for pixel in image_data.split()], dtype=np.uint8)
        else:
            image = np.array(image_data, dtype=np.uint8)
            
        # Reshape to correct dimensions
        image = image.reshape(IMG_SIZE, IMG_SIZE)
        
        # Convert to RGB (3 channels)
        image = np.repeat(image[:, :, np.newaxis], 3, axis=-1)
        return image

    def get_keypoints(self, image_id):
        row = self.data[self.data['ImageId'] == image_id].iloc[0]
        keypoints = []
        
        # Get keypoints in exact order of feature_columns
        for col in self.feature_columns:
            value = row[col]
            # Normalize the coordinate value
            if not np.isnan(value):
                value = value / IMG_SIZE  # Normalize to [0,1]
            else:
                value = 0.5  # Center point for missing values
            keypoints.append(value)
            
        return np.array(keypoints, dtype=np.float32)

    def __getitem__(self, index):
        # Get batch indexes
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        indexes = self.indexes[start_idx:end_idx]
        
        # Get batch image IDs
        batch_image_ids = [self.image_ids[k] for k in indexes]
        
        batch_images = []
        batch_keypoints = []
        
        for image_id in batch_image_ids:
            image = self.get_image(image_id)
            keypoints = self.get_keypoints(image_id)
            
            if len(keypoints) > 0:
                batch_images.append(image)
                batch_keypoints.append(keypoints)
        
        if len(batch_images) == 0:
            return np.empty((0, IMG_SIZE, IMG_SIZE, 3)), np.empty((0, len(self.feature_columns)))
            
        return self.__data_generation(batch_images, batch_keypoints)

    def __data_generation(self, batch_images, batch_keypoints):
        batch_size = len(batch_images)
        batch_images_array = np.empty((batch_size, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        batch_keypoints_array = np.empty((batch_size, len(self.feature_columns)), dtype=np.float32)

        for i, (image, keypoint) in enumerate(zip(batch_images, batch_keypoints)):
            # Create keypoints for augmentation
            keypoint_pairs = keypoint.reshape(-1, 2)  # Reshape to (x,y) pairs
            keypoints_list = [
                Keypoint(x=x * IMG_SIZE, y=y * IMG_SIZE)
                for x, y in keypoint_pairs
            ]
            kps = KeypointsOnImage(keypoints_list, shape=image.shape)

            # Apply augmentation
            augmented_image, augmented_kps = self.aug_pipeline(image=image, keypoints=kps)

            # Fallback if augmentation produces invalid image
            if np.mean(augmented_image) < 10:
                augmented_image = image
                augmented_kps = kps

            batch_images_array[i] = augmented_image

            # Process augmented keypoints
            flattened_keypoints = []
            for kp in augmented_kps.keypoints:
                x_norm = np.clip(kp.x / IMG_SIZE, 0, 1)
                y_norm = np.clip(kp.y / IMG_SIZE, 0, 1)
                flattened_keypoints.extend([x_norm, y_norm])

            # Fallback if augmentation fails
            if len(flattened_keypoints) == 0:
                flattened_keypoints = keypoint

            batch_keypoints_array[i] = np.array(flattened_keypoints)

        return batch_images_array, batch_keypoints_array




train_aug = iaa.Sequential(
    [
        iaa.Resize(IMG_SIZE, interpolation="linear"),
        iaa.Fliplr(0.3),
        # `Sometimes()` applies a function randomly to the inputs with
        # a given probability (0.3, in this case).
        iaa.Sometimes(0.3, iaa.Affine(rotate=10, scale=(0.5, 0.7))),
    ]
)

test_aug = iaa.Sequential([iaa.Resize(IMG_SIZE, interpolation="linear")])



train_dataset = KeyPointsDataset(data=df_merge, aug_pipeline=train_aug, batch_size=BATCH_SIZE)
valid_dataset = KeyPointsDataset(data=df_merge, aug_pipeline=test_aug, batch_size=BATCH_SIZE)

# Test the dataset
sample_images, sample_keypoints = train_dataset.__getitem__(0)
print(f"Sample batch shapes:")
print(f"Images: {sample_images.shape}")
print(f"Keypoints: {sample_keypoints.shape}")

# Verify keypoint ranges
print(f"\nKeypoint value ranges:")
print(f"Min: {sample_keypoints.min():.3f}")
print(f"Max: {sample_keypoints.max():.3f}")
visualize_keypoints(sample_images, sample_keypoints)


def generator(dataset):
    for img, label in dataset:
        for i in range(len(img)):
            yield img[i], label[i]

# Create the TensorFlow dataset for training
train_tf_dataset = tf.data.Dataset.from_generator(
    lambda: generator(train_dataset),
    output_signature=(
        tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.uint8),  # Individual image
        tf.TensorSpec(shape=(len(train_dataset.feature_columns),), dtype=tf.float32),  # Individual keypoints
    )
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)  

# Create the TensorFlow dataset for validation
valid_tf_dataset = tf.data.Dataset.from_generator(
    lambda: generator(valid_dataset),
    output_signature=(
        tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.uint8),  # Individual image
        tf.TensorSpec(shape=(len(valid_dataset.feature_columns),), dtype=tf.float32),  # Individual keypoints
    )
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



def get_model():
    backbone = keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    backbone.trainable = False

    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = backbone(x)
    
    # Reduce feature map dimensions
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(NUM_KEYPOINTS, activation="sigmoid")(x)  # Output keypoints

    return keras.Model(inputs, outputs, name="keypoint_detector")

model = get_model()

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))
    
model.compile(loss="mse",optimizer=keras.optimizers.Adam(1e-4),metrics=[rmse])
model.summary()


keras.utils.plot_model(model, show_shapes=True)


steps_per_epoch = len(train_dataset) // BATCH_SIZE
validation_steps = len(valid_dataset) // BATCH_SIZE



history = model.fit(train_tf_dataset,validation_data=valid_tf_dataset,epochs=EPOCH,steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps)



loss = history.history['loss']
val_loss = history.history['val_loss']
rmse = history.history['rmse']
val_rmse = history.history['val_rmse']

# Plot loss and RMSE
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot loss
axes[0].plot(range(1, len(loss) + 1), loss, label='Training Loss')
axes[0].plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
axes[0].set_title('Loss Over Epochs')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].legend()

# Plot RMSE
axes[1].plot(range(1, len(rmse) + 1), rmse, label='Training RMSE')
axes[1].plot(range(1, len(val_rmse) + 1), val_rmse, label='Validation RMSE')
axes[1].set_title('RMSE Over Epochs')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('RMSE')
axes[1].legend()

plt.tight_layout()
plt.show()




sample_val_images, sample_val_keypoints = next(iter(valid_tf_dataset))

sample_val_images_np = sample_val_images.numpy()
sample_val_keypoints_np = sample_val_keypoints.numpy()

predictions = model.predict(sample_val_images_np)

IMG_SIZE = 96
num_features_per_point = 2
num_points = sample_val_keypoints_np.shape[1] // num_features_per_point

predictions_reshaped = predictions.reshape(-1, num_points, num_features_per_point) * IMG_SIZE
sample_val_keypoints_reshaped = sample_val_keypoints_np.reshape(-1, num_points, num_features_per_point) * IMG_SIZE

feature_groups = {
    'eyes': {'color': 'red', 'indices': list(range(0, 6)), 'label': 'Eyes'},  # First 6 keypoints
    'eyebrows': {'color': 'blue', 'indices': list(range(6, 10)), 'label': 'Eyebrows'},  # Next 4 keypoints
    'nose': {'color': 'green', 'indices': list(range(10, 12)), 'label': 'Nose'},  # 2 keypoints
    'mouth': {'color': 'yellow', 'indices': list(range(12,15)), 'label': 'Mouth'}  # Last 3 keypoints
}

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 20))

for i in range(5):
    axes[i, 0].imshow(sample_val_images_np[i], cmap="gray")
    axes[i, 0].set_title(f"Original {i+1}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(sample_val_images_np[i], cmap="gray")
    
    for group in feature_groups.values():
        indices = np.array(group['indices'])
        axes[i, 1].scatter(predictions_reshaped[i, indices, 0], predictions_reshaped[i, indices, 1], c=group['color'], marker='x')
        axes[i, 1].scatter(sample_val_keypoints_reshaped[i, indices, 0], sample_val_keypoints_reshaped[i, indices, 1], c=group['color'], marker='o')

    axes[i, 1].set_title(f"Predicted vs GT {i+1}")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()



test_data=pd.read_csv("/kaggle/input/facial-keypoints-detection/test.zip",compression="zip")


test_data.head()


RowId=test_data.ImageId


test_data.head()


test_data.drop(columns=["ImageId"],axis=1,inplace=True)




def preprocess_image(row):
    pixels = np.array(row.split(), dtype=np.float32)
    pixels = pixels.reshape(96, 96, 1)
    pixels = np.repeat(pixels, 3, axis=-1)
    pixels = pixels / 255.0
    return pixels

processed_images = np.array([preprocess_image(row) for row in test_data["Image"]])

print("Processed Image Shape:", processed_images.shape)

predictions = model.predict(processed_images)

lookid_list = list(idlookup['FeatureName'])
imageId = list(idlookup['ImageId'] - 1)
rowid = list(idlookup['RowId'])

feature = [lookid_list.index(f) for f in lookid_list]

location = [predictions[x][y] for x, y in zip(imageId, feature)]

submission = pd.DataFrame({'RowId': rowid, 'Location': location})

submission.to_csv('facial_keypoints_detection_submission.csv', index=False)

print("Submission file saved successfully!")



submission.head()




