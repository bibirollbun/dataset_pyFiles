import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import shutil
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping


# Original dataset paths for V2
test_dir = "/kaggle/input/edge-forest-fire-challenge-2025/Forest-Fire-Cleaned-Processed/test"
train_dir = "/kaggle/input/edge-forest-fire-challenge-2025/Forest-Fire-Cleaned-Processed/train"
val_dir = "/kaggle/input/edge-forest-fire-challenge-2025/Forest-Fire-Cleaned-Processed/val"


IMG_SIZE = (224, 224)
BATCH_SIZE = 32


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest")

val_test_datagen = ImageDataGenerator(rescale=1./255)


# --- 2. Create the Generators, now pointing to your new, clean directories ---
train_generator = train_datagen.flow_from_directory(
    train_dir,  # Now uses the correct training directory
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical")


val_generator = val_test_datagen.flow_from_directory(
    val_dir,    # Now uses the correct validation directory
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical")


test_generator = val_test_datagen.flow_from_directory(
    test_dir,   # Now uses the correct test directory
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False)


class_names = list(train_generator.class_indices.keys())
print("Classes:", class_names)


class_names = list(test_generator.class_indices.keys())
print("Classes:", class_names)


class_names = list(val_generator.class_indices.keys())
print("Classes:", class_names)


print("VAL CLASSES:", os.listdir(val_dir))


base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))


base_model.trainable =False #freeze base layer


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)


preds = Dense(len(class_names), activation = 'softmax')(x)


model = Model(inputs = base_model.input, outputs = preds)


model.compile(optimizer=Adam(learning_rate=0.0001),
             loss='categorical_crossentropy',
             metrics=["accuracy"])


model.summary()


# Define No of epochs here
EPOCHS = 10
INIT_LR = 1e-2  # initial learning rate


# Define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=INIT_LR)


# Custom callback to log learning rate at the end of each epoch
class LrLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = float(K.get_value(self.model.optimizer.learning_rate))
        print(f"\nEpoch {epoch+1} â†’ Learning Rate: {lr:.6f}")


# Early stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor="val_loss",   # stop when validation loss stops improving
    patience=3,           # wait 3 epochs before stopping
    restore_best_weights=True
)


# Compile with all metrics
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]
)


# Train
history = model.fit(
    train_generator,
    steps_per_epoch=math.ceil(train_generator.samples / BATCH_SIZE),
    validation_data=val_generator,
    validation_steps=math.ceil(val_generator.samples / BATCH_SIZE),
    epochs=EPOCHS,
    callbacks=[LrLogger(), early_stop],
    verbose=1
)


def plot_training_history(history):
    metrics = ["accuracy", "loss", "auc", "precision", "recall"]
    n = len(metrics)

    plt.figure(figsize=(15, 10))

    for i, metric in enumerate(metrics, 1):
        plt.subplot(3, 2, i)  # 3 rows Ã— 2 cols grid
        plt.plot(history.history[metric], label=f"Train {metric.capitalize()}")
        plt.plot(history.history[f"val_{metric}"], label=f"Val {metric.capitalize()}")
        plt.title(f"Model {metric.capitalize()}")
        plt.xlabel("Epochs")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()

# Call function
plot_training_history(history)


base_model.trainable = True
for layer in base_model.layers[:-30]:  # unfreeze last 30 layers
    layer.trainable = False


model.compile(optimizer=Adam(learning_rate=1e-5),
              loss="categorical_crossentropy",
              metrics=["accuracy"])


fine_tune_epochs = 15
history_fine = model.fit(
    train_generator,
    steps_per_epoch=math.ceil(train_generator.samples / BATCH_SIZE),
    validation_data=val_generator,
    validation_steps=math.ceil(val_generator.samples / BATCH_SIZE),
    epochs=fine_tune_epochs
)


plot_training_history(history_fine)


test_loss, test_acc = model.evaluate(test_generator, steps=math.ceil(test_generator.samples / BATCH_SIZE))
print(f"âœ… Test Accuracy: {test_acc*100:.2f}%")


# Classification Report
Y_pred = model.predict(test_generator, steps=math.ceil(test_generator.samples / BATCH_SIZE))
y_pred = np.argmax(Y_pred, axis=1)

print("Classification Report:\n")
print(classification_report(test_generator.classes, y_pred, target_names=class_names))


# Confusion Matrix
cm = confusion_matrix(test_generator.classes, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()



# Get the test predictions
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Get the true labels from the test generator
y_true = test_generator.classes

# Find the indices of the misclassified images
misclassified_indices = np.where(y_true != y_pred_classes)[0]
print(f"Total misclassified images: {len(misclassified_indices)}")

# Map class indices back to class names
class_names = list(test_generator.class_indices.keys())

# Display a few of the misclassified images
plt.figure(figsize=(15, 15))
for i, index in enumerate(misclassified_indices[:25]): # Display up to 25 images
    plt.subplot(5, 5, i + 1)
    
    # Get the image and true/predicted labels
    img = test_generator[index // test_generator.batch_size][0][index % test_generator.batch_size]
    true_label = class_names[y_true[index]]
    predicted_label = class_names[y_pred_classes[index]]
    
    # Show the image
    plt.imshow(img)
    plt.title(f"True: {true_label}\nPred: {predicted_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()


model.save("/kaggle/working/mobilenetv2_fire_detection.h5")
print("âœ… Model saved in /kaggle/working/")


loaded_model = tf.keras.models.load_model("/kaggle/working/mobilenetv2_fire_detection.h5")
loss, acc = loaded_model.evaluate(test_generator, steps=math.ceil(test_generator.samples / BATCH_SIZE))
print(f"Reloaded Model Test Accuracy: {acc*100:.2f}%")


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load model
model = tf.keras.models.load_model("/kaggle/working/mobilenetv2_fire_detection.h5")

# Prepare test dataset (adjust path)
test_dir = '/kaggle/input/edge-forest-fire-challenge-2025/Forest-Fire-Cleaned-Processed/test'

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# Evaluate
results = model.evaluate(test_generator)
print("\nğŸ“Š Evaluation Results:")
for metric, value in zip(model.metrics_names, results):
    print(f"{metric}: {value:.4f}")


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

# Load the saved model
model = tf.keras.models.load_model("/kaggle/working/mobilenetv2_fire_detection.h5")
print("âœ… Model loaded successfully!")

# Define your class labels (update if you find a 3rd class)
class_labels = ["Fire", "No Fire"]

# Function to preprocess image and predict
def predict_image(url, target_size=(224, 224)):
    try:
        # Load image from URL
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        
        # Resize and preprocess
        img_resized = img.resize(target_size)
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Prediction
        preds = model.predict(img_array)
        pred_class = np.argmax(preds, axis=1)[0]
        confidence = np.max(preds)

        # Show image + prediction
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Prediction: {class_labels[pred_class]} ({confidence:.2f})")
        plt.show()

    except Exception as e:
        print("â�Œ Error processing image:", e)

# Example test images from Google (replace with your own)
test_urls = [
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTBoCPC2CA6n0QjqOyT6gKI-xMT2EMLWEhDQA&s",  # Fire
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTSSBhbrWQi_2CyKMpVxfSf6qfgFTfz-To4bA&"        # No Fire
]

# Run predictions
for url in test_urls:
    predict_image(url)


!ls -lh /kaggle/working/mobilenetv2_fire_detection.h5


!zip -r mobilenetv2_fire_detection.zip mobilenetv2_fire_detection.h5

