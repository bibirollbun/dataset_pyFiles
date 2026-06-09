import pandas as pd

# Load the dataset
data = pd.read_csv('/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/icml_face_data.csv')

# Display the first 5 rows
data.head()


import matplotlib.pyplot as plt
import seaborn as sns

# Total samples
total_samples = len(data)
print(f"ğŸ“Š Total number of samples: {total_samples}")

# Samples per class
class_counts = data['emotion'].value_counts().sort_index()
emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}
emotion_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}
# Print class counts
print("\nğŸ“Š Samples per class:")
for idx, count in class_counts.items():
    print(f"{emotion_labels[idx]} ({idx}): {count} samples")

# Plot distribution
plt.figure(figsize=(10,6))
sns.barplot(x=[emotion_labels[i] for i in class_counts.index], y=class_counts.values, palette="Set2")
plt.title("Emotion Class Distribution")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.grid(axis='y')
plt.show()




print("Column names:", list(data.columns))

# Strip leading/trailing whitespace from column names
data.columns = data.columns.str.strip()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV
data = pd.read_csv('/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/icml_face_data.csv')

# Clean column names
data.columns = data.columns.str.strip()

# Summary
print("ğŸ“Š Total samples:", len(data))
print("\nğŸ“ˆ Samples per emotion class:\n", data['emotion'].value_counts().sort_index())

# Emotion label map
emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

# Get image shape
pixels = np.fromstring(data.iloc[0]['pixels'], sep=' ', dtype=int)
image_shape = (48, 48)
print("\nğŸ–¼ï¸� Image shape:", image_shape)

# Visualize 1 image per class
plt.figure(figsize=(14, 6))
for emotion in range(7):
    example = data[data['emotion'] == emotion].iloc[0]
    pixel_array = np.fromstring(example['pixels'], sep=' ', dtype=int).reshape(image_shape)
    plt.subplot(2, 4, emotion + 1)
    plt.imshow(pixel_array, cmap='gray')
    plt.title(emotion_labels[emotion])
    plt.axis('off')

plt.tight_layout()
plt.show()



import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/icml_face_data.csv')
data.columns = ['emotion', 'Usage', 'pixels']  # Fix column names

# Parse and validate pixel arrays
def parse_pixels(pixels):
    try:
        arr = np.fromstring(pixels, sep=' ')
        if arr.size == 2304:  # 48x48 = 2304
            return arr
        else:
            return np.nan
    except:
        return np.nan

# Apply and drop invalid rows
data['pixels_array'] = data['pixels'].apply(parse_pixels)
data.dropna(subset=['pixels_array'], inplace=True)

# Stack pixel arrays into X and normalize
X = np.stack(data['pixels_array'].values) / 255.0
X = X.reshape(-1, 48, 48, 1)  # Add channel dimension

# Encode target labels
encoder = LabelBinarizer()
y = encoder.fit_transform(data['emotion'])

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

print("âœ… Shapes â€” X_train:", X_train.shape, "| y_train:", y_train.shape)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Build the CNN model
model_1 = Sequential([
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.0001), input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(7, activation='softmax')  # 7 emotion classes
])

# Compile the model
model_1.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model_1.summary()



import time
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

class EpochLogger(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        print(f"Epoch [{epoch+1}/{self.params['epochs']}] "
              f"â�±ï¸� {epoch_time:.2f}s  "
              f"Loss: {loss:.4f}  "
              f"Val Loss: {val_loss:.4f}  "
              f"Accuracy: {acc:.4f}  "
              f"Val Accuracy: {val_acc:.4f}")
# Define callbacks
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True,
    verbose=1
)
# âœ… Train the model
history = model_1.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=64,
    callbacks=[EpochLogger(), reduce_lr, early_stop],
    verbose=0
)



# Evaluate final model
test_loss, test_acc = model_1.evaluate(X_val, y_val, verbose=0)
print(f"âœ… Final Test Accuracy: {test_acc:.4f}")
print(f"âœ… Final Test Loss: {test_loss:.4f}")



import numpy as np

# Get predicted probabilities and class predictions
y_pred_probs = model_1.predict(X_val)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_val, axis=1)  # If y_val is one-hot encoded


from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Print classification report
print("\nğŸ“Š Classification Report:\n")
print(classification_report(y_true_classes, y_pred_classes, target_names=list(emotion_dict.values())))

# Confusion matrix
conf_mat = confusion_matrix(y_true_classes, y_pred_classes)

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=emotion_dict.values(),
            yticklabels=emotion_dict.values())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()




y_train_labels = np.argmax(y_train, axis=1)


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Convert one-hot labels to class indices
y_train_labels = np.argmax(y_train, axis=1)

# Group X and y by class
X_by_class = {i: X_train[y_train_labels == i] for i in range(7)}
y_by_class = {i: y_train[y_train_labels == i] for i in range(7)}

# Choose target size (maximum class count)
target_count = max(len(X_by_class[i]) for i in range(7))

# Define augmentation generator
augmenter = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Augmentation function
def augment_class(X, y, target_size):
    n_needed = target_size - len(X)
    batches = augmenter.flow(X, y, batch_size=64, shuffle=True)
    X_aug, y_aug = [], []
    while len(X_aug) < n_needed:
        X_batch, y_batch = next(batches)
        X_aug.extend(X_batch)
        y_aug.extend(y_batch)
    return np.array(X_aug[:n_needed]), np.array(y_aug[:n_needed])

# Create balanced dataset
X_balanced, y_balanced = [], []

for i in range(7):
    X_class = X_by_class[i]
    y_class = y_by_class[i]

    if len(X_class) < target_count:
        X_aug, y_aug = augment_class(X_class, y_class, target_count)
        X_class = np.concatenate([X_class, X_aug])
        y_class = np.concatenate([y_class, y_aug])

    X_balanced.append(X_class)
    y_balanced.append(y_class)

# Final arrays
X_balanced = np.concatenate(X_balanced)
y_balanced = np.concatenate(y_balanced)

print("âœ… New training shape:", X_balanced.shape, y_balanced.shape)

# Convert one-hot labels back to class indices for counting
y_balanced_labels = np.argmax(y_balanced, axis=1)

# Count samples per class
unique, counts = np.unique(y_balanced_labels, return_counts=True)
print("\nğŸ“Š Sample count per emotion class (after augmentation):")
for label, count in zip(unique, counts):
    print(f"Class {label}: {count} samples")




# Re-train model on the balanced dataset
history = model_1.fit(
    X_balanced, y_balanced,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[EpochLogger(), reduce_lr, early_stop],
    verbose=0
)



from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# Predict on validation set
y_pred_probs = model_1.predict(X_val)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_val, axis=1)

# Print accuracy
accuracy = np.mean(y_pred_classes == y_true_classes)
print(f"âœ… Final Validation Accuracy: {accuracy:.4f}")

# Print classification report
print("\nğŸ“Š Classification Report:\n")
print(classification_report(
    y_true_classes,
    y_pred_classes,
    target_names=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
))

# Plot confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title("ğŸ§© Confusion Matrix")
plt.tight_layout()
plt.show()



import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# â”€â”€â”€ 0) BUILD YOUR VGG16 CLASSIFIER â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
base_model = VGG16(
    include_top=False,
    weights=None,               # or 'imagenet' if you want pretrained weights
    input_shape=(48,48,3)
)

model_vgg16 = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax')
])

# â”€â”€â”€ 1) PREPARE TF.DATA PIPELINE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
AUTOTUNE   = tf.data.AUTOTUNE
num_classes = y_train.shape[1]

def make_dataset(X, y, batch_size, shuffle=False):
    def gen():
        for img, lbl in zip(X, y):
            yield img.astype(np.float32), lbl.astype(np.float32)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(48,48,1), dtype=tf.float32),
            tf.TensorSpec(shape=(num_classes,), dtype=tf.float32)
        )
    )
    ds = ds.map(lambda img, lbl: (tf.image.grayscale_to_rgb(img), lbl),
                num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(2048)
    return ds.batch(batch_size).prefetch(AUTOTUNE)

train_ds = make_dataset(X_train, y_train, batch_size=64, shuffle=True)
val_ds   = make_dataset(X_val,   y_val,   batch_size=64, shuffle=False)

# â”€â”€â”€ 2) COMPILE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
model_vgg16.compile(
    optimizer=Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# â”€â”€â”€ 3) CALLBACKS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
                              min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=10,
                           restore_best_weights=True, verbose=1)

# â”€â”€â”€ 4) TRAIN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
history = model_vgg16.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[reduce_lr, early_stop],
    verbose=2
)



import numpy as np

# Recreate X_val_rgb on the CPU as float32:
X_val_rgb = np.repeat(X_val.astype(np.float32), 3, axis=-1)

# Now predict on the whole validation array:
y_pred_probs   = model_vgg16.predict(X_val_rgb, batch_size=64)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_val,           axis=1)




from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Classification report
print("\nğŸ“Š Classification Report:\n")
print(classification_report(
    y_true_classes,
    y_pred_classes,
    target_names=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
))



# Confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title("VGG16 Confusion Matrix")
plt.tight_layout()
plt.show()



# Accuracy and loss curves
plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# â”€â”€â”€ PREPARE RGB INPUT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Convert your grayscale [N,48,48,1] arrays into [N,48,48,3] float32 on CPU
X_train_rgb = np.repeat(X_train.astype(np.float32), 3, axis=-1)
X_val_rgb   = np.repeat(X_val.astype(np.float32),   3, axis=-1)

# â”€â”€â”€ 1) LOAD VGG16 & FREEZE ALL CONV LAYERS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
base_model = VGG16(
    include_top=False,
    weights='/kaggle/input/vggggg/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
    input_shape=(48, 48, 3)
)
for layer in base_model.layers:
    layer.trainable = False

# â”€â”€â”€ 2) BUILD THE FULL MODEL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
model_vgg = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# â”€â”€â”€ 3) COMPILE FOR HEAD TRAINING â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
model_vgg.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# â”€â”€â”€ 4) TRAIN JUST THE HEAD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
history_head = model_vgg.fit(
    X_train_rgb, y_train,
    validation_data=(X_val_rgb, y_val),
    epochs=150,
    batch_size=64,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=2
)

# â”€â”€â”€ 5) UNFREEZE TOP 4 CONV LAYERS & FINE-TUNE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
for layer in base_model.layers[-4:]:
    layer.trainable = True

model_vgg.compile(
    optimizer=Adam(learning_rate=1e-6),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_finetune = model_vgg.fit(
    X_train_rgb, y_train,
    validation_data=(X_val_rgb, y_val),
    epochs=200,
    batch_size=64,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=2
)



from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Predict
y_pred_probs = model_vgg.predict(X_val_rgb)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_val, axis=1)

# Classification Report
print("\nğŸ“Š Classification Report:\n")
print(classification_report(
    y_true_classes,
    y_pred_classes,
    target_names=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
))

# Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
    'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'
])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title("ğŸ§© Confusion Matrix")
plt.tight_layout()
plt.show()



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# === Load pretrained ResNet18 weights manually ===
resnet18 = models.resnet18(weights=None)
weights_path = '/kaggle/input/pytorch-pretrained-models/resnet18-5c106cde.pth'
state_dict = torch.load(weights_path, weights_only=False)
resnet18.load_state_dict(state_dict)

# === Freeze feature extractor layers ===
for param in resnet18.parameters():
    param.requires_grad = False

# === Replace FC layer for 7 emotion classes ===
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 7)

# === Move model to device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet18 = resnet18.to(device)

# === Define custom dataset ===
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx].squeeze()
        label = self.y[idx]
        image = self.transform(image)
        return image, label

# === Prepare Data ===
if y_train.ndim > 1:
    y_train = np.argmax(y_train, axis=1)
    y_val = np.argmax(y_val, axis=1)

train_dataset = EmotionDataset(X_train, y_train)
val_dataset = EmotionDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# === Loss and optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet18.fc.parameters(), lr=1e-4)

# === Tracking & early stopping ===
train_acc_list, val_acc_list = [], []
train_loss_list, val_loss_list = [], []

best_val_loss = float('inf')
best_model_wts = copy.deepcopy(resnet18.state_dict())
patience = 5
trigger_times = 0

# === Training loop ===
for epoch in range(50):
    resnet18.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet18(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = correct / total
    train_loss = running_loss / len(train_loader)

    # === Validation ===
    resnet18.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = resnet18(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = val_correct / val_total
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Track metrics
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)

    # === Early stopping logic ===
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(resnet18.state_dict())
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# === Restore best model ===
resnet18.load_state_dict(best_model_wts)

# === Plot Accuracy & Loss Curves ===
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(val_acc_list, label='Validation Accuracy')
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Validation Loss')
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# === Evaluation ===
resnet18.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = resnet18(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# === Classification Report & Confusion Matrix ===
print("\nğŸ“Š Classification Report:\n")
print(classification_report(
    all_labels,
    all_preds,
    target_names=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
))

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix - ResNet18")
plt.show()



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# â”€â”€â”€ 1) MODEL & WEIGHTS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# init architecture
resnet18 = models.resnet18(weights=None)

# load legacy .pth onto CPU
weights_path = "/kaggle/input/pytorch-pretrained-models/resnet18-5c106cde.pth"
state_dict   = torch.load(weights_path, map_location="cpu", weights_only=False)
resnet18.load_state_dict(state_dict)

# move model to device
resnet18.to(device)

# â”€â”€â”€ 2) FREEZE & UNFREEZE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# freeze everything
for p in resnet18.parameters():
    p.requires_grad = False

# unfreeze layer3, layer4, and the final fc
for name, module in resnet18.named_children():
    if name in ("layer3", "layer4", "fc"):
        for p in module.parameters():
            p.requires_grad = True

# replace the head for 7 emotions
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 7).to(device)

# â”€â”€â”€ 3) DATASET & DATALOADER â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        img = self.X[idx].squeeze().astype(np.uint8)  # ensure uint8 HÃ—W
        return self.tf(img), int(self.y[idx])

# if one-hot, convert
if y_train.ndim > 1:
    y_train = np.argmax(y_train, axis=1)
    y_val   = np.argmax(y_val,   axis=1)

train_ds = EmotionDataset(X_train, y_train)
val_ds   = EmotionDataset(X_val,   y_val)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# â”€â”€â”€ 4) TRAINING SETUP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet18.parameters()), lr=1e-4)

train_accs, val_accs = [], []
train_losses, val_losses = [], []
best_val_loss = float("inf")
best_weights = copy.deepcopy(resnet18.state_dict())
patience, wait = 5, 0

for epoch in range(50):
    # â€” train â€”
    resnet18.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = resnet18(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_losses.append(running_loss / len(train_loader))
    train_accs.append(correct/total)

    # â€” validate â€”
    resnet18.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = resnet18(imgs)
            loss = criterion(out, labels)
            v_loss += loss.item()
            preds = out.argmax(1)
            v_correct += (preds == labels).sum().item()
            v_total += labels.size(0)
    val_losses.append(v_loss/len(val_loader))
    val_accs.append(v_correct/v_total)

    print(f"Epoch {epoch+1} | "
          f"Train Acc: {train_accs[-1]:.4f} | Val Acc: {val_accs[-1]:.4f} | "
          f"Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

    # early stop
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        best_weights = copy.deepcopy(resnet18.state_dict())
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# restore best
resnet18.load_state_dict(best_weights)

# â”€â”€â”€ 5) PLOT METRICS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs,   label="Val Acc")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Accuracy")

plt.subplot(1,2,2)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses,   label="Val Loss")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.tight_layout()
plt.show()

# â”€â”€â”€ 6) FINAL EVAL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
resnet18.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        out  = resnet18(imgs)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())

print(classification_report(all_labels, all_preds,
      target_names=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']))
cm = confusion_matrix(all_labels, all_preds)
Disp = ConfusionMatrixDisplay(cm,
       display_labels=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral'])
Disp.plot(cmap='Blues', xticks_rotation=45)
plt.show()



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# â”€â”€â”€ 1) MODEL & WEIGHTS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# init architecture
resnet18 = models.resnet18(weights=None)

# load legacy .pth onto CPU
weights_path = "/kaggle/input/pytorch-pretrained-models/resnet18-5c106cde.pth"
state_dict   = torch.load(weights_path, map_location="cpu", weights_only=False)
resnet18.load_state_dict(state_dict)

# now move to GPU
resnet18.to(device)

# â”€â”€â”€ 2) FREEZE & UNFREEZE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# freeze everything
for p in resnet18.parameters():
    p.requires_grad = False

# unfreeze layer2, layer3, layer4, and the final fc
for name, module in resnet18.named_children():
    if name in ("layer2", "layer3", "layer4", "fc"):
        for p in module.parameters():
            p.requires_grad = True

# replace the head for 7 emotion classes
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 7).to(device)

# â”€â”€â”€ 3) DATASET & DATALOADER â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        img = (self.X[idx].squeeze() * 255).astype(np.uint8)
        return self.tf(img), int(self.y[idx])

# if y is one-hot, convert back to labels
if y_train.ndim > 1:
    y_train = np.argmax(y_train, axis=1)
    y_val   = np.argmax(y_val,   axis=1)

train_ds = EmotionDataset(X_train, y_train)
val_ds   = EmotionDataset(X_val,   y_val)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# â”€â”€â”€ 4) OPTIMIZER & SCHEDULER â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# differential LR: backbone @1e-5, head @1e-4
backbone_params = []
head_params     = []
for name, p in resnet18.named_parameters():
    if p.requires_grad:
        if name.startswith("fc."):
            head_params.append(p)
        else:
            backbone_params.append(p)

optimizer = optim.Adam([
    {"params": backbone_params, "lr": 1e-5},
    {"params": head_params,     "lr": 1e-4}
], weight_decay=1e-4)

criterion = nn.CrossEntropyLoss()

# â”€â”€â”€ 5) TRAINING LOOP w/ EARLY STOPPING â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
train_accs, val_accs = [], []
train_losses, val_losses = [], []
best_val_loss = float("inf")
best_weights = copy.deepcopy(resnet18.state_dict())
patience, wait = 7, 0

for epoch in range(50):
    # â€” train â€”
    resnet18.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = resnet18(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_losses.append(running_loss / len(train_loader))
    train_accs.append(correct/total)

    # â€” validate â€”
    resnet18.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = resnet18(imgs)
            loss = criterion(out, labels)
            v_loss += loss.item()
            preds = out.argmax(1)
            v_correct += (preds == labels).sum().item()
            v_total += labels.size(0)
    val_losses.append(v_loss/len(val_loader))
    val_accs.append(v_correct/v_total)

    print(f"Epoch {epoch+1} | "
          f"Train Acc: {train_accs[-1]:.4f} | Val Acc: {val_accs[-1]:.4f} | "
          f"Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

    # early stop
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        best_weights = copy.deepcopy(resnet18.state_dict())
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# load best
resnet18.load_state_dict(best_weights)

# â”€â”€â”€ 6) PLOT METRICS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs,   label="Val Acc")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Accuracy")

plt.subplot(1,2,2)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses,   label="Val Loss")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.tight_layout()
plt.show()

# â”€â”€â”€ 7) FINAL EVAL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
resnet18.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        out  = resnet18(imgs)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())

print(classification_report(all_labels, all_preds,
      target_names=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']))

cm = confusion_matrix(all_labels, all_preds)
Disp = ConfusionMatrixDisplay(cm,
       display_labels=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral'])
Disp.plot(cmap='Blues', xticks_rotation=45)
plt.show()



import torch, copy
import torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) architecture + load legacy .pth on CPU
resnet18 = models.resnet18(weights=None)
sd = torch.load("/kaggle/input/pytorch-pretrained-models/resnet18-5c106cde.pth",
               map_location="cpu", weights_only=False)
resnet18.load_state_dict(sd)
resnet18.to(device)

# 2) freeze all, then unfreeze layer1â€“4 & head
for p in resnet18.parameters():
    p.requires_grad = False

for name, module in resnet18.named_children():
    if name in ("layer1", "layer2", "layer3", "layer4", "fc"):
        for p in module.parameters():
            p.requires_grad = True

# replace the head
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 7).to(device)

# 3) dataset & dataloader
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        img = (self.X[idx].squeeze()*255).astype(np.uint8)
        return self.tf(img), int(self.y[idx])

# if your yâ€™s are one-hot:
if y_train.ndim>1:
    y_train = y_train.argmax(1)
    y_val   = y_val.argmax(1)

train_ds = EmotionDataset(X_train, y_train)
val_ds   = EmotionDataset(X_val,   y_val)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# 4) optimizer: super-low LR on backbone, higher on head
backbone, head = [], []
for n,p in resnet18.named_parameters():
    if p.requires_grad:
        if n.startswith("fc."): head.append(p)
        else:                   backbone.append(p)

optimizer = optim.Adam([
    {"params": backbone, "lr": 1e-6},
    {"params": head,     "lr": 1e-4}
], weight_decay=1e-4)

criterion = nn.CrossEntropyLoss()

# 5) train loop with early-stop
best_wts = copy.deepcopy(resnet18.state_dict())
best_val = float("inf")
patience = 7; wait = 0

train_accs = []; val_accs = []
train_losses = []; val_losses = []

for epoch in range(50):
    # â€” train â€”
    resnet18.train()
    running_loss = correct = total = 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out = resnet18(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = out.argmax(1)
        correct += (preds==lbls).sum().item()
        total += lbls.size(0)
    train_losses.append(running_loss/len(train_loader))
    train_accs.append(correct/total)

    # â€” validate â€”
    resnet18.eval()
    v_loss = v_correct = v_total = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = resnet18(imgs)
            l = criterion(out, lbls)
            v_loss += l.item()
            p = out.argmax(1)
            v_correct += (p==lbls).sum().item()
            v_total += lbls.size(0)
    val_losses.append(v_loss/len(val_loader))
    val_accs.append(v_correct/v_total)

    print(f"Epoch {epoch+1} | "
          f"Train Acc: {train_accs[-1]:.4f} | Val Acc: {val_accs[-1]:.4f} | "
          f"Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

    # early-stop
    if val_losses[-1] < best_val:
        best_val = val_losses[-1]
        best_wts  = copy.deepcopy(resnet18.state_dict())
        wait = 0
    else:
        wait += 1
        if wait>=patience:
            print("Early stopping at epoch", epoch+1)
            break

# 6) restore best & plot
resnet18.load_state_dict(best_wts)
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_accs,label="Train Acc"); plt.plot(val_accs,label="Val Acc")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Acc")
plt.subplot(1,2,2)
plt.plot(train_losses,label="Train Loss"); plt.plot(val_losses,label="Val Loss")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.tight_layout()
plt.show()

# 7) final report
resnet18.eval()
all_p, all_l = [], []
with torch.no_grad():
    for imgs,lbls in val_loader:
        imgs = imgs.to(device)
        out  = resnet18(imgs)
        all_p.extend(out.argmax(1).cpu().numpy())
        all_l.extend(lbls.numpy())

print(classification_report(all_l, all_p,
      target_names=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']))
cm = confusion_matrix(all_l, all_p)
Disp = ConfusionMatrixDisplay(cm,
      display_labels=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral'])
Disp.plot(cmap='Blues', xticks_rotation=45)
plt.show()



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import matplotlib.pyplot as plt
import os

# ==== VAE Model ====
class ImprovedVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(ImprovedVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # 48 -> 24
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 24 -> 12
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 12 -> 6
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),  # 6 -> 6
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(256 * 6 * 6, latent_dim)
        self.fc_logvar = nn.Linear(256 * 6 * 6, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 256 * 6 * 6)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 1, 1),  # -> 6x6
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # -> 12x12
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # -> 24x24
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),     # -> 48x48
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z).view(-1, 256, 6, 6)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ==== VAE Loss Function ====
def vae_loss(x, x_recon, mu, logvar, beta=0.1):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div


# ==== FER-2013 Dataset ====
class FERDataset(Dataset):
    def __init__(self, X):
        self.X = X.astype(np.float32) / 255.0
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx].squeeze()
        return self.transform(image)


# ==== Training Function ====
def train_vae(model, dataloader, epochs=30, lr=1e-3, beta=0.1, device='cuda'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(batch, recon, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")
    return model


# ==== Face Generation ====
def generate_faces(model, latent_dim, num_samples=64, device='cuda'):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.decode(z)
    return samples.cpu()


# ==== Visualization ====
def visualize_generated_faces(images, title="Generated Faces from Improved VAE"):
    grid = utils.make_grid(images, nrow=8, normalize=True, pad_value=1)
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title(title)
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
    plt.show()


# ==== Run Full Pipeline ====

# === Load and prepare your real data ===
# Replace this with actual FER-2013 data loading
# Example:
# X_train = np.load("X_train.npy")  # Shape: [N, 48, 48, 1]

# Dummy data (for testing only):
# X_train = np.random.randint(0, 255, size=(1000, 48, 48, 1), dtype=np.uint8)

# === Step 1: Dataset and DataLoader ===
dataset = FERDataset(X_train)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# === Step 2: Train the VAE ===
vae = ImprovedVAE(latent_dim=128)
vae = train_vae(vae, dataloader, epochs=150, lr=1e-3, beta=0.1, device='cuda')

# === Step 3: Generate and Show ===
generated = generate_faces(vae, latent_dim=128, num_samples=64, device='cuda')
visualize_generated_faces(generated)







import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
import numpy as np
import matplotlib.pyplot as plt
import os

# ==== Improved VAE Model with BatchNorm & Dropout ====
class ImprovedVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),    # 48â†’24
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1),   # 24â†’12
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),  # 12â†’6
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3, 1, 1), # 6â†’6
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten()
        )
        self.fc_mu     = nn.Linear(256*6*6, latent_dim)
        self.fc_logvar = nn.Linear(256*6*6, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 256*6*6)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 1, 1),  # 6â†’6
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 6â†’12
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 12â†’24
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 1, 4, 2, 1),     # 24â†’48
            nn.Sigmoid()  # output in [0,1]
        )

    def encode(self, x):
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        x = self.fc_decode(z).view(-1,256,6,6)
        return self.decoder(x)

    def forward(self, x, beta=1.0):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# ==== VAE Loss with BCE + KL ====
def vae_loss_bce(x, recon, mu, logvar, beta):
    # BCE over pixels
    bce = F.binary_cross_entropy(recon, x, reduction='sum')
    # KL divergence
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + beta*kl, bce, kl

# ==== Dataset Wrapper ====
class FERDataset(Dataset):
    def __init__(self, X_array):
        self.X = X_array.astype(np.float32) / 255.0
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx].squeeze()         # [48,48]
        return self.transform(img)         # [1,48,48]

# ==== Training + Validation Loop ====
def train_validate_vae(model, train_loader, val_loader, 
                       epochs=100, lr=1e-3, 
                       beta_start=0.0, beta_end=1.0, warmup_epochs=10,
                       device='cuda'):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.to(device)

    train_losses, val_losses = [], []
    best_val = float('inf')
    patience, wait = 10, 0

    for epoch in range(1, epochs+1):
        # Î²-annealing schedule
        beta = min(beta_end, beta_start + (beta_end-beta_start)*epoch/warmup_epochs)

        # â€”â€”â€” Training â€”â€”â€”
        model.train()
        running_loss = 0
        for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x, beta)
            loss, _, _ = vae_loss_bce(x, recon, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train = running_loss / len(train_loader.dataset)
        train_losses.append(avg_train)

        # â€”â€”â€” Validation â€”â€”â€”
        model.eval()
        running_val = 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                recon, mu, logvar = model(x, beta)
                loss, _, _ = vae_loss_bce(x, recon, mu, logvar, beta)
                running_val += loss.item()

        avg_val = running_val / len(val_loader.dataset)
        val_losses.append(avg_val)

        print(f"Epoch {epoch:3d} | Î²={beta:.2f} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        # â€”â€”â€” Early Stopping â€”â€”â€”
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), "best_vae.pth")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    # Plot loss curves
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses,   label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (per image)")
    plt.legend()
    plt.title("VAE Train vs. Validation Loss")
    plt.show()

    # Load best model
    model.load_state_dict(torch.load("best_vae.pth"))
    return model

# ==== Generate & Visualize ====
def generate_and_show(model, latent_dim, n_samples=64, device='cuda'):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim).to(device)
        imgs = model.decode(z).cpu()
    grid = utils.make_grid(imgs, nrow=8, normalize=True, pad_value=1)
    plt.figure(figsize=(8,8))
    plt.imshow(grid.permute(1,2,0).squeeze(), cmap='gray')
    plt.axis("off")
    plt.title("Generated Faces from Improved VAE")
    plt.show()

# ==== Main Pipeline ====
if __name__ == "__main__":
    # --- Load your FER-2013 arrays here ---
    # X = np.load("fer2013_images.npy")  # shape [N,48,48,1]
    X = X_train  # replace with your loaded data

    # Split into train/validation
    n_val = int(0.1 * len(X))
    n_train = len(X) - n_val
    dataset = FERDataset(X)
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=4)

    # Initialize & train
    vae = ImprovedVAE(latent_dim=128)
    vae = train_validate_vae(
        vae, train_loader, val_loader,
        epochs=150, lr=1e-3,
        beta_start=0.0, beta_end=1.0, warmup_epochs=20,
        device="cuda"
    )

    # Generate samples
    generate_and_show(vae, latent_dim=128, n_samples=64, device="cuda")



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt

# â€” assume X_train, X_val are numpy arrays of shape [N,48,48,1], normalized to [0,1] â€”
# â€” e.g. from your earlier code: X_train, X_val = train_test_split(â€¦) â€”

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Dataset + DataLoader
class FERDataset(Dataset):
    def __init__(self, X):
        self.X = X.astype('float32')  # already in [0,1]
        self.tf = transforms.ToTensor()  # turns (H,W,1) â†’ (1,H,W)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.tf(self.X[i])

batch_size = 128
train_loader = DataLoader(FERDataset(X_train), batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(FERDataset(X_val),   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# 2) Improved VAE with BatchNorm & Dropout
class ImprovedVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # 48â†’24
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, 4, 2, 1), # 24â†’12
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, 4, 2, 1),# 12â†’6
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3, 1, 1),# 6â†’6
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten()
        )
        self.fc_mu     = nn.Linear(256*6*6, latent_dim)
        self.fc_logvar = nn.Linear(256*6*6, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 256*6*6)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 1, 1), # 6â†’6
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 6â†’12
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 12â†’24
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),

            nn.ConvTranspose2d(32, 1, 4, 2, 1),    # 24â†’48
            nn.Sigmoid()  # output in [0,1]
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = self.fc_dec(z).view(-1,256,6,6)
        return self.decoder(h)

    def forward(self, x, beta=1.0):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# 3) Loss + training loop (200 epochs, no earlyâ€�stop) with KL warmup
def vae_loss(x, recon, mu, logvar, beta):
    bce = F.binary_cross_entropy(recon, x, reduction='sum')
    kl  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + beta*kl, bce, kl

model = ImprovedVAE(latent_dim=128).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)

train_losses, val_losses = [], []
warmup_epochs = 20

for epoch in range(1, 201):
    beta = min(1.0, epoch / warmup_epochs)  # linear Î² warmup from 0â†’1

    # â€” Train â€”
    model.train()
    running_train = 0
    for xb in train_loader:
        xb = xb.to(device)
        opt.zero_grad()
        recon, mu, logvar = model(xb, beta)
        loss, _, _ = vae_loss(xb, recon, mu, logvar, beta)
        loss.backward()
        opt.step()
        running_train += loss.item()
    avg_train = running_train / len(train_loader.dataset)
    train_losses.append(avg_train)

    # â€” Validate â€”
    model.eval()
    running_val = 0
    with torch.no_grad():
        for xb in val_loader:
            xb = xb.to(device)
            recon, mu, logvar = model(xb, beta)
            loss, _, _ = vae_loss(xb, recon, mu, logvar, beta)
            running_val += loss.item()
    avg_val = running_val / len(val_loader.dataset)
    val_losses.append(avg_val)

    sched.step()
    print(f"Epoch {epoch:3d} | Î²={beta:.2f} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

# 4) Plot loss curves
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train loss")
plt.plot(val_losses,   label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss per image")
plt.legend()
plt.title("VAE Train vs. Val Loss (200 epochs)")
plt.show()

# 5) Generate & visualize 64 new faces
model.eval()
with torch.no_grad():
    z = torch.randn(64, 128, device=device)
    samples = model.decode(z).cpu()
grid = utils.make_grid(samples, nrow=8, normalize=True, pad_value=1)
plt.figure(figsize=(8,8))
plt.imshow(grid.permute(1,2,0).squeeze(), cmap='gray')
plt.axis('off')
plt.title("Generated Faces from Improved VAE")
plt.show()



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import numpy as np

# â”€â”€â”€ ASSUME X_train, X_val ARE ALREADY DEFINED â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# X_train, X_val: numpy arrays of shape [N,48,48,1], float32 in [0,1]

device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size   = 128
latent_dim   = 128
epochs       = 200
lr           = 1e-3
beta_warmup  = 20     # epochs to ramp Î² from 0â†’1
sched_step   = 50     # halve LR every 50 epochs
clip_norm    = 1.0    # gradient clipping norm

# â”€â”€â”€ 1) DATASET & DATALOADERS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class FERDataset(Dataset):
    def __init__(self, X, augment=False):
        # X: [N,48,48,1], values âˆˆ [0,1]
        self.X = X
        self.augment = augment
        self.tf_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor()
        ])
        self.tf_plain = transforms.ToTensor()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = (self.X[idx] * 255).astype(np.uint8).squeeze()  # [48,48]
        if self.augment:
            return self.tf_aug(img)   # returns [1,48,48], float32 [0,1]
        else:
            return self.tf_plain(img)

train_loader = DataLoader(
    FERDataset(X_train, augment=True),
    batch_size=batch_size, shuffle=True,
    num_workers=4, pin_memory=True
)
val_loader   = DataLoader(
    FERDataset(X_val,   augment=False),
    batch_size=batch_size, shuffle=False,
    num_workers=4, pin_memory=True
)

# â”€â”€â”€ 2) IMPROVED VAE DEFINITION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class ImprovedVAE(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1,  32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2), nn.Dropout2d(0.1),
            nn.Conv2d(64,128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128,256, 3, 1, 1),nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        self.fc_mu     = nn.Linear(256*6*6, z_dim)
        self.fc_logvar = nn.Linear(256*6*6, z_dim)

        # Decoder
        self.fc_dec = nn.Linear(z_dim, 256*6*6)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256,128,3,1,1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1),  nn.BatchNorm2d(64),  nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1),   nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.ConvTranspose2d(32, 1,4,2,1),   nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z).view(-1,256,6,6)
        return self.decoder(h)

    def forward(self, x, beta=1.0):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# â”€â”€â”€ 3) LOSS & TRAIN/VAL LOOP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def vae_loss(x, recon, mu, logvar, beta):
    bce = F.binary_cross_entropy(recon, x, reduction='sum')
    kl  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + beta * kl

model = ImprovedVAE(latent_dim).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=sched_step, gamma=0.5)

train_losses, val_losses = [], []

for epoch in range(1, epochs+1):
    beta = min(1.0, epoch / beta_warmup)

    # â€” Training â€”
    model.train()
    running_train = 0.0
    for xb in train_loader:
        xb = xb.to(device)
        opt.zero_grad()
        recon, mu, logvar = model(xb, beta)
        loss = vae_loss(xb, recon, mu, logvar, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        opt.step()
        running_train += loss.item()
    avg_train = running_train / len(train_loader.dataset)
    train_losses.append(avg_train)

    # â€” Validation â€”
    model.eval()
    running_val = 0.0
    with torch.no_grad():
        for xb in val_loader:
            xb = xb.to(device)
            recon, mu, logvar = model(xb, beta)
            running_val += vae_loss(xb, recon, mu, logvar, beta).item()
    avg_val = running_val / len(val_loader.dataset)
    val_losses.append(avg_val)

    sched.step()
    print(f"Epoch {epoch:3d} | Î²={beta:.2f} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

# â”€â”€â”€ 4) PLOT LOSS CURVES â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train loss")
plt.plot(val_losses,   label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss per image")
plt.legend()
plt.title(f"VAE Train vs. Val Loss ({epochs} epochs)")
plt.show()

# â”€â”€â”€ 5) GENERATE & VISUALIZE NEW FACES â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
model.eval()
with torch.no_grad():
    z = torch.randn(64, latent_dim, device=device)
    samples = model.decode(z).cpu()

grid = utils.make_grid(samples, nrow=8, normalize=True, pad_value=1)
plt.figure(figsize=(8,8))
plt.imshow(grid.permute(1,2,0).squeeze(), cmap='gray')
plt.axis('off')
plt.title("Generated Faces from Improved VAE")
plt.show()



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, utils
import matplotlib.pyplot as plt

# â”€â”€â”€ ASSUME PREPROCESSING IS DONE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# You should already have:
#   X_train, X_val: numpy arrays [N,48,48,1], float32 in [0,1]
#   y_train, y_val: numpy arrays [N,num_classes] one-hot labels

device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size  = 128
latent_dim  = 128
epochs      = 100
lr          = 1e-3
beta_warmup = 20     # epochs to ramp Î² 0â†’1
sched_step  = 50     # halve LR every 50 epochs
clip_norm   = 1.0    # gradient clipping norm
num_synth   = 64     # number of faces to generate

# â”€â”€â”€ 1) DEFINE DATASET & DATALOADERS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class FERDataset(Dataset):
    def __init__(self, X, augment=False):
        self.X = X  # [N,48,48,1]
        self.augment = augment
        self.tf_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor()
        ])
        self.tf_plain = transforms.ToTensor()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = (self.X[idx] * 255).astype(np.uint8).squeeze()  # [48,48]
        return self.tf_aug(img) if self.augment else self.tf_plain(img)

train_loader = DataLoader(
    FERDataset(X_train, augment=True),
    batch_size=batch_size, shuffle=True,
    num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    FERDataset(X_val, augment=False),
    batch_size=batch_size, shuffle=False,
    num_workers=4, pin_memory=True
)

# â”€â”€â”€ 2) IMPROVED VAE MODEL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class ImprovedVAE(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,  32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2), nn.Dropout2d(0.1),
            nn.Conv2d(64,128, 4, 2, 1), nn.BatchNorm2d(128),nn.LeakyReLU(0.2),
            nn.Conv2d(128,256, 3, 1, 1),nn.BatchNorm2d(256),nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        self.fc_mu     = nn.Linear(256*6*6, z_dim)
        self.fc_logvar = nn.Linear(256*6*6, z_dim)
        self.fc_dec    = nn.Linear(z_dim, 256*6*6)
        self.decoder   = nn.Sequential(
            nn.ConvTranspose2d(256,128,3,1,1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1),  nn.BatchNorm2d(64),  nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1),   nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.ConvTranspose2d(32,1, 4,2,1),   nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = self.fc_dec(z).view(-1,256,6,6)
        return self.decoder(h)

    def forward(self, x, beta=1.0):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# â”€â”€â”€ 3) LOSS & TRAINâ€�VALIDATION LOOP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def vae_loss(x, recon, mu, logvar, beta):
    bce = F.binary_cross_entropy(recon, x, reduction='sum')
    kl  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + beta * kl

vae   = ImprovedVAE(latent_dim).to(device)
opt   = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=sched_step, gamma=0.5)

train_losses, val_losses = [], []
for epoch in range(1, epochs+1):
    beta = min(1.0, epoch / beta_warmup)

    # â€”â€” Train â€”â€” 
    vae.train()
    running = 0.0
    for xb in train_loader:
        xb = xb.to(device)
        opt.zero_grad()
        recon, mu, logvar = vae(xb, beta)
        loss = vae_loss(xb, recon, mu, logvar, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), clip_norm)
        opt.step()
        running += loss.item()
    train_losses.append(running / len(train_loader.dataset))

    # â€”â€” Val â€”â€” 
    vae.eval()
    running = 0.0
    with torch.no_grad():
        for xb in val_loader:
            xb = xb.to(device)
            recon, mu, logvar = vae(xb, beta)
            running += vae_loss(xb, recon, mu, logvar, beta).item()
    val_losses.append(running / len(val_loader.dataset))

    sched.step()
    print(f"Epoch {epoch:3d} | Î²={beta:.2f} | Train: {train_losses[-1]:.4f} | Val: {val_losses[-1]:.4f}")

# plot losses
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train loss")
plt.plot(val_losses,   label="Val loss")
plt.xlabel("Epoch"); plt.ylabel("Loss/image")
plt.legend(); plt.title("VAE Train vs. Val Loss")
plt.show()

# â”€â”€â”€ 4) GENERATE SYNTHETIC FACES â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
vae.eval()
with torch.no_grad():
    z       = torch.randn(num_synth, latent_dim, device=device)
    samples = vae.decode(z).cpu()   # [num_synth,1,48,48]

# (Optional) visualize them
grid = utils.make_grid(samples, nrow=8, normalize=True, pad_value=1)
plt.figure(figsize=(8,8))
plt.imshow(grid.permute(1,2,0).squeeze(), cmap='gray')
plt.axis('off'); plt.title("Generated Faces")
plt.show()





import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# â”€â”€â”€ ASSUMPTIONS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# X_train, X_val: np.arrays of shape [N,48,48,1], float32 in [0,1]
# y_train, y_val: either 1D int labels [N] or one-hot [N,C]

# 1) Fix labels & get num_classes
if y_train.ndim == 1:
    num_classes = int(y_train.max()) + 1
    y_train = np.eye(num_classes, dtype=np.float32)[y_train]
    y_val   = np.eye(num_classes, dtype=np.float32)[y_val]
else:
    num_classes = y_train.shape[1]

# 2) Hyperparams & device
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim  = 128
batch_size  = 128
lr          = 1e-3
epochs      = 100
beta_warmup = 20
sched_step  = 50
clip_norm   = 1.0

# 3) Prepare CPU tensors & DataLoader (no .to(device) here)
X_t = torch.from_numpy(X_train).permute(0,3,1,2).float()  # [N,1,48,48] on CPU
y_t = torch.from_numpy(y_train).float()                   # [N,C] on CPU
loader = DataLoader(
    TensorDataset(X_t, y_t),
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# 4) Define Conditional VAE
class ConditionalVAE(nn.Module):
    def __init__(self, z_dim, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32,4,2,1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32,64,4,2,1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64,128,4,2,1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128,256,3,1,1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        flat_dim = 256*6*6
        self.fc_mu     = nn.Linear(flat_dim + num_classes, z_dim)
        self.fc_logvar = nn.Linear(flat_dim + num_classes, z_dim)
        self.fc_dec    = nn.Linear(z_dim + num_classes, flat_dim)
        self.decoder   = nn.Sequential(
            nn.ConvTranspose2d(256,128,3,1,1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1),  nn.BatchNorm2d(64),  nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1),   nn.BatchNorm2d(32),  nn.ReLU(),
            nn.ConvTranspose2d(32,1,4,2,1),    nn.Sigmoid()
        )
    def encode(self, x, y):
        h = self.encoder(x)
        h_cat = torch.cat([h, y], dim=1)
        return self.fc_mu(h_cat), self.fc_logvar(h_cat)
    def reparameterize(self, mu, logvar):
        std = (0.5*logvar).exp()
        return mu + torch.randn_like(std)*std
    def decode(self, z, y):
        z_cat = torch.cat([z, y], dim=1)
        h     = self.fc_dec(z_cat).view(-1,256,6,6)
        return self.decoder(h)
    def forward(self, x, y, beta=1.0):
        mu, logvar = self.encode(x, y)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decode(z, y)
        return recon, mu, logvar

# 5) Training loop
def cvae_loss(x, recon, mu, logvar, beta):
    bce = F.binary_cross_entropy(recon, x, reduction='sum')
    kl  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + beta * kl

cvae      = ConditionalVAE(latent_dim, num_classes).to(device)
optimizer = torch.optim.Adam(cvae.parameters(), lr=lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=0.5)
loss_hist = []

for epoch in range(1, epochs+1):
    beta = min(1.0, epoch/beta_warmup)
    cvae.train()
    running = 0.0
    for xb_cpu, yb_cpu in loader:
        xb, yb = xb_cpu.to(device, non_blocking=True), yb_cpu.to(device, non_blocking=True)
        optimizer.zero_grad()
        recon, mu, logvar = cvae(xb, yb, beta)
        loss = cvae_loss(xb, recon, mu, logvar, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cvae.parameters(), clip_norm)
        optimizer.step()
        running += loss.item()
    scheduler.step()
    avg = running / len(loader.dataset)
    loss_hist.append(avg)
    print(f"Epoch {epoch:3d} | Î²={beta:.2f} | Loss: {avg:.4f}")

plt.plot(loss_hist)
plt.xlabel("Epoch"); plt.ylabel("Loss/image"); plt.title("cVAE Training")
plt.show()

# 6) Generate & balance
counts = y_train.sum(axis=0).astype(int)
max_c  = counts.max()
need   = max_c - counts
print("Current counts:", counts)
print("To add per class:", need)

cvae.eval()
X_syn_list, y_syn_list = [], []
with torch.no_grad():
    for cls, n in enumerate(need):
        if n <= 0: continue
        z      = torch.randn(n, latent_dim, device=device)
        y_cond = torch.zeros(n, num_classes, device=device); y_cond[:,cls] = 1
        imgs   = cvae.decode(z, y_cond).cpu().numpy().transpose(0,2,3,1)
        X_syn_list.append(imgs)
        lbls = np.zeros((n, num_classes), dtype=y_train.dtype); lbls[:,cls] = 1
        y_syn_list.append(lbls)

X_synth = np.concatenate(X_syn_list, axis=0)
y_synth = np.concatenate(y_syn_list, axis=0)
print(f"Generated synthetic images: {X_synth.shape[0]}")

X_train_bal = np.concatenate([X_train, X_synth], axis=0)
y_train_bal = np.concatenate([y_train, y_synth], axis=0)
print("Balanced shapes:", X_train_bal.shape, y_train_bal.shape)



import matplotlib.pyplot as plt
import numpy as np

# â€” Plot synthetic faces (up to 64) â€”
n_plot = min(64, X_synth.shape[0])
fig, axes = plt.subplots(8, 8, figsize=(8,8),
                         gridspec_kw=dict(wspace=0.01, hspace=0.01))
for i, ax in enumerate(axes.flatten()):
    ax.axis('off')
    if i < n_plot:
        # X_synth: [M,48,48,1]
        ax.imshow(X_synth[i].squeeze(), cmap='gray')
plt.suptitle("Generated Synthetic Faces", fontsize=16)
plt.show()




# X_train:    original real set, shape [N,48,48,1]
# X_synth:    cVAEâ€�generated, shape [M,48,48,1]
# X_train_bal: concatenation of the two, shape [N+M,48,48,1]

import numpy as np

N = X_train.shape[0]
M = X_synth.shape[0]
total = N + M

print(f"Original real images:     {N}")
print(f"Synthetic (cVAE) images:  {M}")
print(f"Fully balanced total:     {total}")



import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# 1) Predict on X_val via your tf.data val_ds (or rebuild X_val_rgb)
y_pred_probs = model_vgg.predict(val_ds)              # shape [V,7]
y_pred = np.argmax(y_pred_probs, axis=1)             # predicted class indices
y_true = np.argmax(y_val,           axis=1)          # true class indices

# 2) Print classification report
class_names = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
print(classification_report(y_true, y_pred, target_names=class_names))

# 3) Compute & plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
import seaborn as sns; import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names,
            yticklabels=class_names, cmap='Blues')
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Val Confusion Matrix")
plt.show()



from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred,
      target_names=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=emotion_labels).plot(cmap='Blues')




import matplotlib.pyplot as plt
import numpy as np

# â€” Plot synthetic faces (up to 64) â€”
n_plot = min(64, X_synth.shape[0])
# randomly pick n_plot samples
idxs = np.random.choice(X_synth.shape[0], size=n_plot, replace=False)

# set up the grid: up to 8Ã—8
cols = 8
rows = int(np.ceil(n_plot / cols))

plt.figure(figsize=(cols * 2, rows * 2))
for i, idx in enumerate(idxs):
    img = X_synth[idx].squeeze()  # shape (48,48)
    ax = plt.subplot(rows, cols, i+1)
    ax.imshow(img, cmap='gray')
    ax.axis('off')

plt.suptitle("Example Synthetic Faces", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()



# If you have a dict mapping classâ€�indices â†’ names:
emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

# Turn that into an ordered list (0â†’6)
emotion_names = [emotion_labels[i] for i in range(len(emotion_labels))]

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print("Classification Report:")
print(classification_report(
    y_true_classes,
    y_pred_classes,
    target_names=emotion_names
))

cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(cm, display_labels=emotion_names)

fig, ax = plt.subplots(figsize=(8,6))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()




from sklearn.metrics import roc_auc_score
# If you still have y_val as one-hot and y_pred_probs:
auc = roc_auc_score(y_val, y_pred_probs, average="macro", multi_class="ovr")
print("Macro ROC-AUC:", auc)



X_train_bal, y_train_bal, X_val, y_val



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# â”€â”€â”€ 1) Prepare labels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# If your balanced labels are one-hot, convert to ints:
if y_train_bal.ndim > 1:
    y_train_int = np.argmax(y_train_bal, axis=1)
else:
    y_train_int = y_train_bal.copy()
if y_val.ndim > 1:
    y_val_int = np.argmax(y_val, axis=1)
else:
    y_val_int = y_val.copy()

# â”€â”€â”€ 2) Dataset & DataLoaders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        # X: [N,48,48,1] in [0,1], y: [N] int
        self.X = (X * 255).astype(np.uint8)  # to PIL
        self.y = y.astype(np.int64)
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        img = self.X[idx].squeeze()        # (48,48)
        lbl = self.y[idx]
        return self.tf(img), lbl

batch_size = 64
train_ds = EmotionDataset(X_train_bal, y_train_int)
val_ds   = EmotionDataset(X_val,       y_val_int)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# â”€â”€â”€ 3) Load ResNet-18 & freeze / unfreeze â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# a) instantiate and load legacy .pth
resnet = models.resnet18(weights=None)
state_dict = torch.load("/kaggle/input/pytorch-pretrained-models/resnet18-5c106cde.pth",
                        map_location="cpu", weights_only=False)
resnet.load_state_dict(state_dict)

# b) replace final layer
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 7)

# c) freeze all, then unfreeze layer2/3/4 + fc
for p in resnet.parameters(): p.requires_grad = False
for name, module in resnet.named_children():
    if name in ("layer2","layer3","layer4","fc"):
        for p in module.parameters(): p.requires_grad = True

resnet = resnet.to(device)

# â”€â”€â”€ 4) Loss & optimizer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()), lr=1e-4)

# â”€â”€â”€ 5) Training loop w/ early stopping â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
num_epochs = 50
best_val_loss = float("inf")
best_wts = None
patience, wait = 5, 0

train_losses, val_losses = [], []
train_accs, val_accs     = [], []

for epoch in range(1, num_epochs+1):
    # â€” train â€”
    resnet.train()
    running_loss = running_correct = total = 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        opt.zero_grad()
        out = resnet(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        opt.step()

        running_loss   += loss.item() * imgs.size(0)
        preds = out.argmax(dim=1)
        running_correct+= (preds==lbls).sum().item()
        total          += imgs.size(0)

    train_loss = running_loss/total
    train_acc  = running_correct/total

    # â€” validate â€”
    resnet.eval()
    val_loss = val_correct = val_total = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = resnet(imgs)
            loss = criterion(out, lbls)
            val_loss    += loss.item() * imgs.size(0)
            preds = out.argmax(dim=1)
            val_correct += (preds==lbls).sum().item()
            val_total   += imgs.size(0)

    val_loss /= val_total
    val_acc   = val_correct/val_total

    print(f"Epoch {epoch}/{num_epochs}  "
          f"Train loss: {train_loss:.4f} acc: {train_acc:.4f}  "
          f"Val   loss: {val_loss:.4f} acc: {val_acc:.4f}")

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    # early stop
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_wts      = copy.deepcopy(resnet.state_dict())
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# restore best
resnet.load_state_dict(best_wts)

# â”€â”€â”€ 6) Plot metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_accs, label="Train Acc"); plt.plot(val_accs, label="Val Acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
plt.subplot(1,2,2)
plt.plot(train_losses, label="Train Loss"); plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.tight_layout()
plt.show()

# â”€â”€â”€ 7) Final report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
resnet.eval()
all_preds, all_lbls = [], []
with torch.no_grad():
    for imgs, lbls in val_loader:
        imgs = imgs.to(device)
        out  = resnet(imgs)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_lbls.extend(lbls.numpy())

print("\nClassification Report:")
print(classification_report(all_lbls, all_preds,
      target_names=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']))

cm = confusion_matrix(all_lbls, all_preds)
disp = ConfusionMatrixDisplay(cm,
      display_labels=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral'])
fig, ax = plt.subplots(figsize=(8,6))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("ResNet-18 Confusion Matrix")
plt.show()



import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# â”€â”€â”€ 1) MODEL SETUP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
resnet = models.resnet18(weights=None)
state = torch.load(
    "/kaggle/input/pytorch-pretrained-models/resnet18-5c106cde.pth",
    map_location="cpu", weights_only=False
)
resnet.load_state_dict(state)

# freeze everything
for p in resnet.parameters():
    p.requires_grad = False

# unfreeze layers 2,3,4 + later fc
for name, mod in resnet.named_children():
    if name in ("layer2","layer3","layer4"):
        for p in mod.parameters():
            p.requires_grad = True

# replace head: dropoutâ†’fc
num_ftrs     = resnet.fc.in_features
resnet.fc    = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 7)
)
for p in resnet.fc.parameters():
    p.requires_grad = True

resnet.to(device)

class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
        self.tf = transforms.Compose([
            transforms.ToPILImage(),                        # â†’ PIL 'L'
            transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.Grayscale(num_output_channels=3),    # â†� convert to RGB-like 3-ch
            transforms.ToTensor(),                          # â†’ [3Ã—224Ã—224]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # if X is in [0,1], map to [0,255] uint8
        img = (self.X[idx].squeeze() * 255).astype(np.uint8)
        label = int(self.y[idx])
        return self.tf(img), label


# assume X_train, y_train, X_val, y_val are already balanced / one-hot â†’ int
if y_train.ndim>1:
    y_train = np.argmax(y_train, axis=1)
    y_val   = np.argmax(y_val,   axis=1)

train_ds = EmotionDataset(X_train, y_train)
val_ds   = EmotionDataset(X_val,   y_val)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# â”€â”€â”€ 3) OPTIM & SCHED â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# differential LR & AdamW
backbone_params = [p for n,p in resnet.named_parameters() if p.requires_grad and not n.startswith("fc.")]
head_params     = [p for n,p in resnet.named_parameters() if n.startswith("fc.")]

optimizer = optim.AdamW([
    {"params": backbone_params, "lr": 1e-5},
    {"params": head_params,     "lr": 1e-4}
], weight_decay=1e-4)

scheduler = ReduceLROnPlateau(
    optimizer, mode="min",
    factor=0.5, patience=5,
    min_lr=1e-6, verbose=True
)

criterion = nn.CrossEntropyLoss()

# â”€â”€â”€ 4) TRAIN LOOP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
max_epochs    = 100
stop_patience = 5   # sooner early-stop on val_acc

best_val_acc = 0.0
best_wts     = copy.deepcopy(resnet.state_dict())
wait         = 0

history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}

for epoch in range(1, max_epochs+1):
    # â€” train â€”
    resnet.train()
    tl, tc, tt = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = resnet(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        tl  += loss.item() * xb.size(0)
        preds = out.argmax(1)
        tc  += (preds==yb).sum().item()
        tt  += xb.size(0)

    train_loss = tl/tt
    train_acc  = tc/tt

    # â€” validate â€”
    resnet.eval()
    vl, vc, vt = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = resnet(xb)
            loss = criterion(out, yb)

            vl += loss.item() * xb.size(0)
            preds = out.argmax(1)
            vc += (preds==yb).sum().item()
            vt += xb.size(0)

    val_loss = vl/vt
    val_acc  = vc/vt

    history["train_loss"].append(train_loss)
    history["train_acc" ].append(train_acc)
    history["val_loss"  ].append(val_loss)
    history["val_acc"   ].append(val_acc)

    print(f"Epoch {epoch:3d} | "
          f"Train {train_loss:.4f}/{train_acc:.4f} | "
          f" Val {val_loss:.4f}/{val_acc:.4f}")

    # LR scheduler
    scheduler.step(val_loss)

    # early-stop on val_acc
    if val_acc > best_val_acc + 1e-4:
        best_val_acc = val_acc
        best_wts     = copy.deepcopy(resnet.state_dict())
        wait = 0
    else:
        wait += 1
        if wait >= stop_patience:
            print(f"â†’ Early stopping on val_acc at epoch {epoch}")
            break

# â”€â”€â”€ 5) RESTORE & PLOT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
resnet.load_state_dict(best_wts)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["val_acc"],   label="Val   Acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

plt.subplot(1,2,2)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"],   label="Val   Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

plt.tight_layout()
plt.show()



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import torch

# 1) Run evaluation to collect predictions & labels
resnet.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        out  = resnet(imgs)
        preds = out.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

# 2) Compute overall accuracy
acc = accuracy_score(y_true, y_pred)
print(f"Overall accuracy: {acc:.4%}\n")

# 3) Print classification report
class_names = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
print("Classification report:\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
))

# 4) Compute & plot confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=class_names)

fig, ax = plt.subplots(figsize=(8,8))
disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# 5) (Optional) raw counts
print("Confusion matrix (rows=true, cols=pred):")
print(cm)



import pandas as pd
import matplotlib.pyplot as plt

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1) Enter your results here (replace with your actual numbers)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
results = [
    {
        "Model": "Simple CNN (scratch)",
        "Val Acc": 0.453,      # ~0.453 at epoch ~44
        "Val Loss": 1.428,     # final val loss
        "Macro F1": 0.37       # (approx macro avg f1)
    },
    {
        "Model": "ResNet-18 (fc only)",
        "Val Acc": 0.460,      # 0.460 from classification_report
        "Val Loss": 1.428,     # ~1.4279
        "Macro F1": 0.37       # macro avg f1 â‰ˆ0.37
    },
    {
        "Model": "ResNet-18 (unfreeze l3+4+fc)",
        "Val Acc": 0.261,      # 0.26
        "Val Loss": 1.872,     # peak val loss
        "Macro F1": 0.12       # macro avg f1 â‰ˆ0.12
    },
    {
        "Model": "ResNet-18 (unfreeze l2+3+4+fc)",
        "Val Acc": 0.641,      # 0.6414
        "Val Loss": 1.020,     # ~1.0200
        "Macro F1": 0.59       # macro avg f1 â‰ˆ0.59
    },
    {
        "Model": "ResNet-18 (unfreeze l1+2+3+4+fc)",
        "Val Acc": 0.605,      # 0.6049
        "Val Loss": 1.082,     # ~1.0820
        "Macro F1": 0.53       # macro avg f1 â‰ˆ0.53
    },
    {
        "Model": "VGG16 + VAE augment",
        "Val Acc": 0.480,      # 0.48
        "Val Loss": 1.312,     # ~1.3121
        "Macro F1": 0.37       # macro avg f1 â‰ˆ0.37
    },
    {
        "Model": "ResNet-18 + VAE (balanced)",
        "Val Acc": 0.550,      # 0.55
        "Val Loss": None,      # not tracked
        "Macro F1": 0.53       # macro avg f1 â‰ˆ0.53
    },
    {
        "Model": "ResNet-18 + VAE + Dropout head",
        "Val Acc": 0.680,      # 0.680
        "Val Loss": 0.909,     # best val loss
        "Macro F1": 0.66       # macro avg f1 â‰ˆ0.66
    },
]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2) Build DataFrame & print
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
df = pd.DataFrame(results)
print(df)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3) Plot comparison charts
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fig, axes = plt.subplots(1, 2, figsize=(14,5))

# Validation Accuracy
df.plot(
    kind='bar',
    x='Model', y='Val Acc',
    ax=axes[0], legend=False
)
axes[0].set_ylabel("Validation Accuracy")
axes[0].set_title("Model Comparison: Val Accuracy")
axes[0].set_xticklabels(df['Model'], rotation=45, ha='right')

# Macro F1 Score
df.plot(
    kind='bar',
    x='Model', y='Macro F1',
    ax=axes[1], legend=False,
    color='C1'
)
axes[1].set_ylabel("Macro-Fâ‚� Score")
axes[1].set_title("Model Comparison: Macro-Fâ‚�")
axes[1].set_xticklabels(df['Model'], rotation=45, ha='right')

plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1) Paste your final metrics for each experiment below
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
results = [
    {
        "Model": "ResNet-18 (l3+4+fc)",
        "Val Acc": 0.2700,
        "Val Loss": 1.8724,
        "Macro F1": 0.12,
        "Weighted F1": 0.15,
        "Best Epoch": 8
    },
    {
        "Model": "ResNet-18 (l2+3+4+fc)",
        "Val Acc": 0.6414,
        "Val Loss": 1.0200,
        "Macro F1": 0.59,
        "Weighted F1": 0.63,
        "Best Epoch": 12
    },
    {
        "Model": "ResNet-18 (l1+2+3+4+fc)",
        "Val Acc": 0.6049,
        "Val Loss": 1.0820,
        "Macro F1": 0.53,
        "Weighted F1": 0.59,
        "Best Epoch": 28
    },
    {
        "Model": "VGG16 + VAE-aug",
        "Val Acc": 0.4800,
        "Val Loss": 1.3121,
        "Macro F1": 0.37,
        "Weighted F1": 0.44,
        "Best Epoch": None
    },
    {
        "Model": "ResNet-18 + VAE balanced",
        "Val Acc": 0.5500,
        "Val Loss": None,
        "Macro F1": 0.53,
        "Weighted F1": 0.55,
        "Best Epoch": None
    },
    {
        "Model": "ResNet-18 + VAE + dropout head",
        "Val Acc": 0.6787,
        "Val Loss": 0.680,
        "Macro F1": 0.66,
        "Weighted F1": 0.67,
        "Best Epoch": 6
    },
]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2) Build DataFrame and highlight best performers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
df = pd.DataFrame(results)

# flag the best
df['Best Val Acc'] = df['Val Acc'] == df['Val Acc'].max()
df['Best Macro F1'] = df['Macro F1'] == df['Macro F1'].max()

# display
print(df.to_string(index=False))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3) Plot Val Accuracy vs Macro-Fâ‚�
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))

# Val Acc
df.plot.bar(
    x='Model', y='Val Acc', ax=ax1,
    legend=False
)
ax1.set_title("Validation Accuracy by Model")
ax1.set_ylabel("Val Accuracy")
ax1.set_xticklabels(df['Model'], rotation=45, ha='right')

# Macro-Fâ‚�
df.plot.bar(
    x='Model', y='Macro F1', ax=ax2,
    color='C1', legend=False
)
ax2.set_title("Macro-Fâ‚� Score by Model")
ax2.set_ylabel("Macro-Fâ‚�")
ax2.set_xticklabels(df['Model'], rotation=45, ha='right')

plt.tight_layout()
plt.show()



import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support

# â”€â”€â”€ CONFIG â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class_names = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# â”€â”€â”€ 1) GROUNDâ€�TRUTH â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# convert one-hot y_val back to ints if needed
if y_val.ndim > 1:
    y_true = y_val.argmax(axis=1)
else:
    y_true = y_val.copy().astype(int)

# â”€â”€â”€ 2) â€œBEFOREâ€� PREDICTIONS (Keras CNN) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
y_pred_before_probs = model_1.predict(X_val, batch_size=64, verbose=0)
y_pred_before       = np.argmax(y_pred_before_probs, axis=1)

# â”€â”€â”€ 3) â€œAFTERâ€� PREDICTIONS (PyTorch resnet on balanced data) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
resnet.to(device).eval()

# build a simple PyTorch Dataset/Loader over X_val, y_true
transform_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])
class ValDataset(Dataset):
    def __init__(self, X, y, tf):
        self.X, self.y, self.tf = X, y, tf
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        img = (self.X[idx].squeeze() * 255).astype(np.uint8)
        return self.tf(img), int(self.y[idx])

val_ds    = ValDataset(X_val, y_true, transform_val)
val_loader= DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

y_pred_after = []
with torch.no_grad():
    for imgs, _ in val_loader:
        imgs = imgs.to(device)
        out  = resnet(imgs)
        y_pred_after.extend(out.argmax(dim=1).cpu().numpy())
y_pred_after = np.array(y_pred_after)

# â”€â”€â”€ 4) PER-CLASS METRICS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(
    y_true, y_pred_before, labels=range(len(class_names)), zero_division=0
)
prec_a, rec_a, f1_a, _ = precision_recall_fscore_support(
    y_true, y_pred_after,  labels=range(len(class_names)), zero_division=0
)

df = pd.DataFrame({
    'P_before':  prec_b,
    'R_before':  rec_b,
    'F1_before': f1_b,
    'P_after':   prec_a,
    'R_after':   rec_a,
    'F1_after':  f1_a,
}, index=class_names)

print("ğŸ“Š  Per-class performance **before** vs. **after** balancing:\n")
display(df.style.format("{:.3f}"))

# â”€â”€â”€ 5) TOP-3 & BOTTOM-3 BY F1_after â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
best3  = df.sort_values('F1_after', ascending=False).head(3)
worst3 = df.sort_values('F1_after', ascending=True ).head(3)

print("\nâœ…  Top 3 classes by Fâ‚� (after balancing):")
for cls, row in best3.iterrows():
    print(f"   â€¢ {cls:>8s}   F1_after = {row['F1_after']:.3f}")

print("\nâš ï¸�  3 classes needing most attention (lowest Fâ‚�_after):")
for cls, row in worst3.iterrows():
    print(f"   â€¢ {cls:>8s}   F1_after = {row['F1_after']:.3f}")


