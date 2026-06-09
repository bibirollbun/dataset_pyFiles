# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import rotate as scipy_rotate
import pickle
from tqdm import tqdm
import pandas as pd
import pickle

# 2. Load train.pkl
with open("/kaggle/input/mnist-rotation/train.pkl", "rb") as f:
    train_data = pickle.load(f)

print("\nğŸ”¹ train.pkl loaded.")
print(f"Type: {type(train_data)}")

# Check keys if it's a dictionary
if isinstance(train_data, dict):
    print("Keys in train_data:", train_data.keys())

# Example info
print("Training data image shape:", train_data['image'][0].shape)
print("First 5 labels:", train_data['label'][:5])

# 3. Load test.pkl
with open("/kaggle/input/mnist-rotation/test.pkl", "rb") as f:
    test_data = pickle.load(f)

print("\nğŸ”¹ test.pkl loaded.")
print(f"Type: {type(test_data)}")

# Example info
print("Test data image shape:", test_data['image'][0].shape)
print("Number of test samples:", len(test_data['image']))



import matplotlib.pyplot as plt

# Show 5 images from train and 5 from test in a 2x5 subplot
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

# First row: train images
for i in range(5):
    axes[0, i].imshow(train_data['image'][i], cmap='gray')
    axes[0, i].set_title(f"Train\nAngle: {train_data['label'][i]}")
    axes[0, i].axis('off')

# Second row: test images
for i in range(5):
    axes[1, i].imshow(test_data['image'][i], cmap='gray')
    axes[1, i].set_title(f"Test\nId: {i}")
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()



# Check the type and shape of the first image in train and test datasets
train_image_shape = train_data['image'][0].shape
test_image_shape = test_data['image'][0].shape

print(f"âœ… Train image shape: {train_image_shape}")
print(f"âœ… Test image shape: {test_image_shape}")
# Check unique shapes in train images
unique_train_shapes = set(img.shape for img in train_data['image'])
unique_test_shapes = set(img.shape for img in test_data['image'])

print(f"ğŸ”� Unique shapes in train images: {unique_train_shapes}")
print(f"ğŸ”� Unique shapes in test images: {unique_test_shapes}")


import numpy as np
import pandas as pd

# Show first 10 labels
print("ğŸ”¹ First 10 labels:")
print(train_data['label'][:10])

# Total number of labels
print(f"\nğŸ“¦ Total labels: {len(train_data['label'])}")

# Unique label values and their counts
label_series = pd.Series(train_data['label'])
print("\nğŸ”� Label value counts:")
print(label_series.value_counts().sort_index())

# Unique labels only
unique_labels = np.unique(train_data['label'])
print(f"\nâœ… Unique level in training data: {unique_labels}")


# Show first 10 labels
print("ğŸ”¹ First 10 labels:")
print(test_data['label'][:10])

# Total number of labels
print(f"\nğŸ“¦ Total labels: {len(test_data['label'])}")

# Unique label values and their counts
label_series = pd.Series(test_data['label'])
print("\nğŸ”� Label value counts:")
print(label_series.value_counts().sort_index())

# Unique labels only
unique_labels = np.unique(test_data['label'])
print(f"\nâœ… Unique level in training data: {unique_labels}")


print("Keys in test_data:", test_data.keys())
print("Keys in train_data:", train_data.keys())
print("Unique angles:", test_data['angle'].unique())
print("\nAngle counts:\n", test_data['angle'].value_counts().sort_index())
print(test_data['angle'].describe())
print("Sample label values:", test_data['label'].unique())
print("Sample angle values:", test_data['angle'].unique())
print("Number of train images:", len(train_data['image']))
print("Number of test images:", len(test_data['image']))


import numpy as np

# Convert first image to array
train_img = train_data['image'][0]
test_img = test_data['image'][0]

# Check pixel value stats
print("ğŸ”¹ Train image pixel stats:")
print(f"Min: {train_img.min()}, Max: {train_img.max()}, Mean: {train_img.mean():.2f}, Std: {train_img.std():.2f}")

print("\nğŸ”¹ Test image pixel stats:")
print(f"Min: {test_img.min()}, Max: {test_img.max()}, Mean: {test_img.mean():.2f}, Std: {test_img.std():.2f}")


# Normalize if needed
train_norm = train_img / 255.0
test_norm = test_img / 255.0

# Compute absolute pixel difference
diff_img = np.abs(train_norm - test_norm)

plt.imshow(diff_img, cmap='hot')
plt.colorbar()
plt.title("Pixel-wise Difference (Train vs Test)")
plt.axis('off')
plt.show()


# Filter train images where label == 1
train_label_1 = train_data[train_data['label'] == 1]

# Filter test images where label == 1
test_label_1 = test_data[test_data['label'] == 1]

print(f"Number of train images with label 1: {len(train_label_1)}")
print(f"Number of test images with label 1: {len(test_label_1)}")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(12, 5))

# Plot 5 train images with label=1
for i in range(5):
    axes[0, i].imshow(train_label_1['image'].iloc[i], cmap='gray')
    axes[0, i].set_title(f"Train Label=1\nAngle: {train_label_1['angle'].iloc[i]}")
    axes[0, i].axis('off')

# Plot 5 test images with label=1
for i in range(5):
    axes[1, i].imshow(test_label_1['image'].iloc[i], cmap='gray')
    axes[1, i].set_title(f"Test Label=1\nAngle: {test_label_1['angle'].iloc[i]}")
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()


class NoizeGenerator:
    def __init__(
        self,
        discrete_noise_proba=0.2,
        beta_alpha=0.3,
        beta_beta=0.3,
        gaussian_sigma=40,
        shift_prob=1.0,
        seed=1,
    ):
        self.discrete_noise_proba = discrete_noise_proba
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.gaussian_sigma = gaussian_sigma
        self.shift_prob = shift_prob
        self.rng = np.random.default_rng(seed)

    def apply_beta_noise(self, img):
        mask = self.rng.random(img.shape) < self.discrete_noise_proba
        beta_noise = (
            self.rng.beta(self.beta_alpha, self.beta_beta, size=img.shape) * 255
        )
        noisy_img = img.copy()
        noisy_img[mask] = beta_noise[mask]
        return noisy_img, mask, beta_noise

    def apply_gaussian_noise(self, img):
        if self.gaussian_sigma > 0:
            noise = self.rng.normal(loc=0.0, scale=self.gaussian_sigma, size=img.shape)
            img = img + noise
            img = np.clip(img, 0, 255)
        else:
            noise = np.zeros_like(img)
        return img, noise

    def apply_random_shift(self, img):
        direction = self.rng.choice(["up", "down", "left", "right"])
        shifted = np.zeros_like(img)
        if direction == "up":
            shifted[:-1, :] = img[1:, :]
        elif direction == "down":
            shifted[1:, :] = img[:-1, :]
        elif direction == "left":
            shifted[:, :-1] = img[:, 1:]
        elif direction == "right":
            shifted[:, 1:] = img[:, :-1]
        return shifted, direction

    def transform_image_with_debug(self, image):
        tmp_image = image.astype(np.float32)

        # Apply beta noise and get mask
        tmp_image, beta_mask, beta_noise = self.apply_beta_noise(tmp_image)

        # Apply Gaussian noise
        tmp_image, gaussian_noise = self.apply_gaussian_noise(tmp_image)

        # Shift image
        shift_direction = None
        if self.rng.random() < self.shift_prob:
            tmp_image, shift_direction = self.apply_random_shift(tmp_image)

        return tmp_image.astype(np.uint8), beta_mask, beta_noise, gaussian_noise, shift_direction



noize_gen = NoizeGenerator(
    discrete_noise_proba=0.2,
    beta_alpha=0.3,
    beta_beta=0.3,
    gaussian_sigma=40,
    shift_prob=1.0,
    seed=1
)

level_0_indices = np.where(np.array(train_data['label']) == 0)[0]
img = train_data['image'][level_0_indices[0]]

# Apply debug transform
noisy_img, beta_mask, beta_noise, gaussian_noise, shift_dir = noize_gen.transform_image_with_debug(img)

# Plot à¦¦à§‡à¦–à¦¾à¦“
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(noisy_img, cmap='gray')
plt.title(f"Noisy Image\nShift: {shift_dir}")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(beta_mask, cmap='gray')
plt.title("Beta Noise Mask")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(beta_noise, cmap='viridis')
plt.title("Beta Noise Values")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(gaussian_noise, cmap='viridis')
plt.title("Gaussian Noise Values")
plt.axis('off')

plt.tight_layout()
plt.show()



import numpy as np
import cv2
from scipy.stats import beta

class NoizeGenerator:
    def __init__(self, discrete_noise_proba=0.2, beta_alpha=0.3, beta_beta=0.3,
                 gaussian_sigma=40, shift_prob=1.0, seed=None):
        self.discrete_noise_proba = discrete_noise_proba
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.gaussian_sigma = gaussian_sigma
        self.shift_prob = shift_prob
        self.random_state = np.random.RandomState(seed)

    def transform_image_with_debug(self, image):
        image = image.copy().astype(np.float32)
        beta_mask = self.random_state.rand(*image.shape) < self.discrete_noise_proba
        beta_noise = beta.rvs(self.beta_alpha, self.beta_beta, size=image.shape, random_state=self.random_state)
        gaussian_noise = self.random_state.normal(0, self.gaussian_sigma, size=image.shape)
        shift_direction = self.random_state.choice(['up', 'down', 'left', 'right'])

        transformed = image.copy()

        # Apply beta noise
        transformed[beta_mask] = beta_noise[beta_mask] * 255

        # Apply gaussian noise
        transformed += gaussian_noise
        transformed = np.clip(transformed, 0, 255)

        return transformed.astype(np.uint8), beta_mask, beta_noise, gaussian_noise, shift_direction
# à¦§à¦°à§‡ à¦¨à¦¿à¦‡ train_data/test_data à¦�à¦•à¦Ÿà¦¿ dict à¦¯à§‡à¦–à¦¾à¦¨à§‡ 'image' à¦“ 'label' à¦†à¦›à§‡

level0_train_indices = np.where(np.array(train_data['label']) == 0)[0]
level0_test_indices = np.where(np.array(test_data['label']) == 0)[0]
def compute_noise_scores(images, noize_gen, num_images=5):
    scores = []

    for i in range(min(num_images, len(images))):
        img = images[i]
        _, beta_mask, beta_noise, gaussian_noise, _ = noize_gen.transform_image_with_debug(img)

        beta_score = np.sum(beta_mask) / beta_mask.size
        gaussian_score = np.mean(np.abs(gaussian_noise))

        scores.append((beta_score, gaussian_score))

    return scores
noize_gen = NoizeGenerator(
    discrete_noise_proba=0.2,
    beta_alpha=0.3,
    beta_beta=0.3,
    gaussian_sigma=40,
    seed=42
)

train_imgs_level0 = [train_data['image'][idx] for idx in level0_train_indices]
test_imgs_level0 = [test_data['image'][idx] for idx in level0_test_indices]

train_scores = compute_noise_scores(train_imgs_level0, noize_gen, num_images=5)
test_scores = compute_noise_scores(test_imgs_level0, noize_gen, num_images=5)

print("\nğŸ§ª Train Level 0 Noise Scores (Beta%, Gaussian Mean):")
for i, (beta, gauss) in enumerate(train_scores):
    print(f"Image {i+1}: Beta Noise = {beta:.2%}, Gaussian Mean = {gauss:.2f}")

print("\nğŸ§ª Test Level 0 Noise Scores (Beta%, Gaussian Mean):")
for i, (beta, gauss) in enumerate(test_scores):
    print(f"Image {i+1}: Beta Noise = {beta:.2%}, Gaussian Mean = {gauss:.2f}")



# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import rotate as scipy_rotate
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import StratifiedKFold
# from tensorflow.keras import layers, models
# import tensorflow as tf
# # âœ… Save test IDs for submission
# test_ids = test_data['ID'].values
# test_data = test_data.drop(columns=['ID'])

# # âœ… Drop 'angle' from both train and test datasets if exists
# if 'angle' in train_data.columns:
#     train_data = train_data.drop(columns=['angle'])

# if 'angle' in test_data.columns:
#     test_data = test_data.drop(columns=['angle'])
# def rotate(img: np.ndarray, angle: int) -> np.ndarray:
#     if not (-120 <= angle <= 120):
#         raise ValueError("Angle must be between -120 and 120 degrees.")
#     rotated_img = scipy_rotate(
#         img,
#         angle=angle,
#         reshape=False,
#         order=1,
#         mode='constant',
#         cval=0.0
#     )
#     return rotated_img

# class NoizeGenerator:
#     def __init__(
#         self,
#         discrete_noise_proba=0.2,
#         beta_alpha=0.3,
#         beta_beta=0.3,
#         gaussian_sigma=40,
#         shift_prob=1.0,
#         seed=1,
#     ):
#         self.discrete_noise_proba = discrete_noise_proba
#         self.beta_alpha = beta_alpha
#         self.beta_beta = beta_beta
#         self.gaussian_sigma = gaussian_sigma
#         self.shift_prob = shift_prob
#         self.rng = np.random.default_rng(seed)

#     def apply_beta_noise(self, img):
#         mask = self.rng.random(img.shape) < self.discrete_noise_proba
#         beta_noise = (
#             self.rng.beta(self.beta_alpha, self.beta_beta, size=img.shape) * 255
#         )
#         noisy_img = img.copy()
#         noisy_img[mask] = beta_noise[mask]
#         return noisy_img

#     def apply_gaussian_noise(self, img):
#         if self.gaussian_sigma > 0:
#             noise = self.rng.normal(loc=0.0, scale=self.gaussian_sigma, size=img.shape)
#             img = img + noise
#             img = np.clip(img, 0, 255)
#         return img

#     def apply_random_shift(self, img):
#         direction = self.rng.choice(["up", "down", "left", "right"])
#         shifted = np.zeros_like(img)
#         if direction == "up":
#             shifted[:-1, :] = img[1:, :]
#         elif direction == "down":
#             shifted[1:, :] = img[:-1, :]
#         elif direction == "left":
#             shifted[:, :-1] = img[:, 1:]
#         elif direction == "right":
#             shifted[:, 1:] = img[:, :-1]
#         return shifted

#     def transform_image(self, image):
#         tmp_image = image.astype(np.float32)
#         tmp_image = self.apply_beta_noise(tmp_image)
#         tmp_image = self.apply_gaussian_noise(tmp_image)
#         if self.rng.random() < self.shift_prob:
#             tmp_image = self.apply_random_shift(tmp_image)
#         return tmp_image.astype(np.uint8)

# # Parameters
# angles = [-120, -90, -60, -30, 0, 30, 60, 90, 120]
# noize_gen = NoizeGenerator()

# def normalize(img):
#     return img.astype(np.float32) / 255.0

# # Data Augmentation: Rotation + Noise
# augmented_images = []
# augmented_labels = []

# for img in train_data['image']:
#     for angle in angles:
#         rotated_img = rotate(img, angle)
#         noisy_img = noize_gen.transform_image(rotated_img)
#         normalized_img = normalize(noisy_img)
#         augmented_images.append(normalized_img)
#         augmented_labels.append(angle)

# X_all = np.array(augmented_images).reshape(-1, 28, 28, 1)
# y_all = np.array(augmented_labels)

# # Encode labels
# label_encoder = LabelEncoder()
# label_encoder.fit(angles)
# y_all_encoded = label_encoder.transform(y_all)

# # ----------------- Build CNN model ----------------- #
# def build_model():
#     model = models.Sequential([
#         layers.Input(shape=(28, 28, 1)),
#         layers.Conv2D(32, (3, 3), activation='relu'),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D((2, 2)),
#         layers.Dropout(0.25),
#         layers.Conv2D(64, (3, 3), activation='relu'),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D((2, 2)),
#         layers.Dropout(0.25),
#         layers.Flatten(),
#         layers.Dense(128, activation='relu'),
#         layers.Dropout(0.5),
#         layers.Dense(len(angles), activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     return model

# # ----------------- Cross Validation ----------------- #
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# histories = []
# fold_no = 1

# for train_index, val_index in skf.split(X_all, y_all_encoded):
#     print(f"\nTraining fold {fold_no} ...")
#     X_train, X_val = X_all[train_index], X_all[val_index]
#     y_train, y_val = y_all_encoded[train_index], y_all_encoded[val_index]
    
#     model = build_model()
#     history = model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=15,
#         batch_size=64,
#         verbose=2
#     )
#     histories.append(history)
#     fold_no += 1

# # ----------------- Final Evaluation on Test Data ----------------- #
# X_test = np.array([normalize(img) for img in test_data['image']]).reshape(-1, 28, 28, 1)
# y_pred_probs = model.predict(X_test)
# y_pred_classes = np.argmax(y_pred_probs, axis=1)
# predicted_angles = label_encoder.inverse_transform(y_pred_classes)

# print("\nğŸ”� Predicted Angles on Test Data (First 10):")
# print(predicted_angles[:10])

# if 'angle' in test_data:
#     true_angles = test_data['angle']
#     print("\nâœ… True angles available. First 10 True vs Predicted:")
#     for i in range(10):
#         print(f"True: {true_angles[i]}, Predicted: {predicted_angles[i]}")



# import matplotlib.pyplot as plt

# def plot_histories(histories):
#     """
#     Plot training and validation accuracy and loss for each fold.
#     """
#     num_folds = len(histories)
#     plt.figure(figsize=(16, 6))

#     # Plot Loss
#     plt.subplot(1, 2, 1)
#     for i, history in enumerate(histories):
#         plt.plot(history.history['loss'], label=f'Train Fold {i+1}')
#         plt.plot(history.history['val_loss'], linestyle='--', label=f'Val Fold {i+1}')
#     plt.title('Training and Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()

#     # Plot Accuracy
#     plt.subplot(1, 2, 2)
#     for i, history in enumerate(histories):
#         plt.plot(history.history['accuracy'], label=f'Train Fold {i+1}')
#         plt.plot(history.history['val_accuracy'], linestyle='--', label=f'Val Fold {i+1}')
#     plt.title('Training and Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

# # ğŸ”½ Call the function
# plot_histories(histories)



# from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt
# all_y_true = []
# all_y_pred = []

# for i, (train_index, val_index) in enumerate(skf.split(X_all, y_all_encoded)):
#     print(f"\nğŸ“Š Evaluating Fold {i+1} ...")
    
#     X_train, X_val = X_all[train_index], X_all[val_index]
#     y_train, y_val = y_all_encoded[train_index], y_all_encoded[val_index]
    
#     model = build_model()
#     model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=15,
#         batch_size=64,
#         verbose=0  # keep output clean
#     )
    
#     y_val_pred = np.argmax(model.predict(X_val), axis=1)
    
#     all_y_true.extend(y_val)
#     all_y_pred.extend(y_val_pred)



# report = classification_report(all_y_true, all_y_pred, target_names=[str(a) for a in angles])
# print(report)



# cm = confusion_matrix(all_y_true, all_y_pred)
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=angles, yticklabels=angles)
# plt.xlabel("Predicted Angle")
# plt.ylabel("True Angle")
# plt.title("Confusion Matrix")
# plt.show()


# import pandas as pd

# cm = confusion_matrix(all_y_true, all_y_pred)
# per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

# plt.figure(figsize=(10, 6))
# sns.barplot(x=angles, y=per_class_accuracy)
# plt.ylim(0, 1)
# plt.xlabel("Angle Class")
# plt.ylabel("Accuracy")
# plt.title("Per-class Accuracy")
# plt.show()





