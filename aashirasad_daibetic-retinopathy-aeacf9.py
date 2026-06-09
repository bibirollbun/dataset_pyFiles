import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# --- 1. DATA LOADING ---
try:
    train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
    img_dir = '../input/aptos2019-blindness-detection/train_images/'
except:
    print("â�Œ Error: Dataset path nahi mila!")

# Extensions & Type Conversion
train_df['id_code'] = train_df['id_code'].apply(lambda x: x + ".png")
train_df['diagnosis'] = train_df['diagnosis'].astype(str)

# Splitting
train_set, val_set = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['diagnosis'])

# --- 2. GENERATORS (FAST VERSION) ---
# Hum EfficientNet ki built-in preprocessing use kar rahe hain jo GPU friendly hai
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    zoom_range=0.2
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_set,
    directory=img_dir,
    x_col="id_code",
    y_col="diagnosis",
    batch_size=32, # P100 ke liye optimized
    class_mode="categorical",
    target_size=(255, 255)
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_set,
    directory=img_dir,
    x_col="id_code",
    y_col="diagnosis",
    batch_size=32,
    class_mode="categorical",
    target_size=(255, 255)
)

# --- 3. MODEL BUILDING ---
def create_model():
    # Weights download honge, thoda time lag sakta hai
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(255, 255, 3))
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(5, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
    return model

model = create_model()

# --- 4. TRAINING ---
# === NEW TRAINING CELL (WITH CLASS WEIGHTS) ===
from sklearn.utils import class_weight
import numpy as np

# 1. Weights Calculate Karna
train_classes = train_generator.classes
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_classes),
    y=train_classes
)

# Dictionary banana
class_weights_dict = dict(enumerate(class_weights))
print("âš–ï¸� Class Weights active hain!")
print(class_weights_dict)

# --- 4. TRAINING ---
print("ğŸš€ Training Starting... Please wait 1-2 mins for Epoch 1")

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=[early_stop, reduce_lr]
)

# --- 5. SAVE ---
model.save('final_model.h5')
print("âœ… Done!")



# === FIXING THE EVALUATION (Shuffle Error) ===

# 1. Validation Generator ko dubara banayein (Shuffle=False ke sath)
# Yeh bohot zaroori hai taake predictions aur labels match karein
test_generator = val_datagen.flow_from_dataframe(
    dataframe=val_set,
    directory='../input/aptos2019-blindness-detection/train_images/',
    x_col="id_code",
    y_col="diagnosis",
    batch_size=32,
    class_mode="categorical",
    target_size=(255, 255),
    shuffle=False  # <--- YEH HAI MAGIC FIX
)

# 2. Ab Predict karein
print("Sahi tareeqay se Testing shuru... (Wait karein)")
test_generator.reset()
preds = model.predict(test_generator, verbose=1)

# 3. Report Generate karein
y_pred = np.argmax(preds, axis=1)
y_true = test_generator.classes  # Ab ye order match karega

# 4. Print Report
print("\n--- REAL REPORT ---")
print(classification_report(y_true, y_pred, target_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']))

# 5. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Corrected Confusion Matrix')
plt.show()


import matplotlib.pyplot as plt

def plot_training_history(history):
    # Data nikalna
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(15, 5))

    # --- Plot 1: Accuracy ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', linestyle='--')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.grid(True)

    # --- Plot 2: Loss ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss', linestyle='--')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.grid(True)

    plt.show()

# Graph Show karein
if 'history' in globals():
    plot_training_history(history)
else:
    print("History variable nahi mila. Kya aapne training wala cell run kiya tha?")


# === GRAD-CAM VISUALIZATION ===
import matplotlib.cm as cm

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, alpha=0.4):
    # Image load & Resize
    img = cv2.imread(img_path)
    img = cv2.resize(img, (255, 255)) # Humara target size
    
    # Heatmap ko color mein badalna
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))
    jet_heatmap = np.uint8(jet_heatmap * 255)

    # Overlay
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = np.uint8(superimposed_img)

    # Plotting
    plt.figure(figsize=(4, 4))
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Model Focus (Red Area)")
    plt.show()

# --- RUNNING ON 3 SAMPLE IMAGES ---
# EfficientNetB3 ki last layer ka naam usually 'top_activation' hota hai
last_conv_layer_name = "top_activation"

# Hum Validation set se 3 random images uthayenge
import random
random_indices = random.sample(range(len(test_generator.filenames)), 3)

print("Generating Heatmaps... (Red area = Disease Location)")

for i in random_indices:
    img_path = "../input/aptos2019-blindness-detection/train_images/" + test_generator.filenames[i]
    
    # Image Preprocess
    original = cv2.imread(img_path)
    original = cv2.resize(original, (255, 255))
    img_array = preprocess_input(original)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Generate Heatmap
    try:
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        display_gradcam(img_path, heatmap)
        
        # Asal bimari bhi print karte hain
        print(f"Actual Class: {test_generator.classes[i]}")
    except Exception as e:
        print(f"Error: {e}. Shayad layer ka naam ghalat hai.")

