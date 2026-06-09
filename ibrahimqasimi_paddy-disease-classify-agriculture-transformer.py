# ===============================
# TensorFlow / Keras
# ===============================
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, AveragePooling2D, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l1
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ===============================
# Basic Libraries
# ===============================
import numpy as np
import pandas as pd
import os
import cv2
import random
from IPython.display import display, HTML

# ===============================
# Ignore Warnings
# ===============================
import warnings
warnings.filterwarnings('ignore')

# ===============================
# Visualization
# ===============================
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Data Augmentation
# ===============================
import albumentations as A

# ===============================
# Scikit-learn (metrics)
# ===============================
from sklearn.metrics import classification_report, confusion_matrix

colors = ["#2e2905", "#806320", "#ad821d", "#362603", "#f2e47e"]
# ===============================
# Reproducibility: Set Seeds
# ===============================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# ===============================
# Check GPU Availability
# ===============================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"âœ… GPU Available: {gpus[0].name}")
else:
    print("âš ï¸� GPU not found, using CPU.")


# Read training metadata file
csv_path = "/kaggle/input/paddy-disease-classification/train.csv"
train = pd.read_csv(csv_path)

# Display first few rows
train.head()


# Count samples per class
label_counts = train['label'].value_counts()
print("Class distribution:\n", label_counts)

# Check unique classes
num_classes = train['label'].nunique()
print(f"\nTotal unique classes: {num_classes}")


# Normalization layer to scale pixel values between 0â€“1
rescale = tf.keras.layers.Rescaling(1./255)


train_ds = keras.utils.image_dataset_from_directory(
    directory="/kaggle/input/paddy-disease-classification/train_images",
    batch_size=32,
    image_size=(224, 224),
    validation_split=0.2,
    subset="training",
    seed=123
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory="/kaggle/input/paddy-disease-classification/train_images",
    batch_size=32,
    image_size=(224, 224),
    validation_split=0.2,
    subset="validation",
    seed=123
)


test_ds = keras.utils.image_dataset_from_directory(
    directory="/kaggle/input/paddy-disease-classification/test_images",
    batch_size=32,
    image_size=(224, 224),
    label_mode=None,   
    shuffle=False
)


plt.figure(figsize=(12,6))
train['label'].value_counts().plot(kind='bar', color=colors)
plt.title("Class Distribution of Paddy Diseases", fontsize=14, fontweight="bold")
plt.xlabel("Disease Label")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.show()


plt.figure(figsize=(12,6))
sns.boxplot(data=train, x="variety", y="age", palette=colors)
plt.title("Age Distribution across Varieties", fontsize=14, fontweight="bold")
plt.xlabel("Rice Variety")
plt.ylabel("Age")
plt.xticks(rotation=45, ha="right")
plt.show()


plt.figure(figsize=(14,6))
sns.violinplot(data=train, x="label", y="age", palette=colors)
plt.title("Age Distribution across Disease Classes", fontsize=14, fontweight="bold")
plt.xlabel("Disease Label")
plt.ylabel("Age")
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(8,6))
corr = train.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap=colors, fmt=".2f", cbar=True)
plt.title("Correlation Heatmap", fontsize=14, fontweight="bold")
plt.show()


cross_tab = pd.crosstab(train['variety'], train['label'])

cross_tab.plot(
    kind="bar",
    stacked=True,
    figsize=(14, 6),
    color=colors   
)

plt.title("Variety vs Disease Labels (Stacked)", fontsize=14, fontweight="bold")
plt.xlabel("Rice Variety")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Disease Label", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()



def show_classwise_samples(data_dir, classes, samples_per_class=3, img_size=(120, 120)):
    """
    Show 3 sample images for each class with styled bordered heading above.
    """
    for cls in classes:
        # Stylish bordered + background heading
        display(HTML(
            f"""
            <div style='
                text-align:center; 
                background-color:#f2e47e; 
                border:2px solid #2e2905; 
                border-radius:10px; 
                padding:8px; 
                margin:10px 0;'>
                <h2 style='color:#2e2905; font-weight:bold; margin:0;'>ğŸŒ¾ {cls}</h2>
            </div>
            """
        ))
        
        # Randomly select images
        img_names = random.sample(os.listdir(os.path.join(data_dir, cls)), samples_per_class)
        
        # Plot images
        plt.figure(figsize=(12, 4))
        for i, img_name in enumerate(img_names):
            img_path = os.path.join(data_dir, cls, img_name)
            
            # Read and resize
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            
            plt.subplot(1, samples_per_class, i+1)
            plt.imshow(img)
            plt.axis("off")
        
        plt.show()


# Example usage
data_dir = "/kaggle/input/paddy-disease-classification/train_images"
classes = os.listdir(data_dir)

show_classwise_samples(data_dir, classes, samples_per_class=3)


# AUTOTUNE
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)


# ğŸ”¹ EfficientNetB4 Model Build + Training
efficientnet_base = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(224,224,3))
efficientnet_base.trainable = False

model_effnet = Sequential([
    efficientnet_base,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model_effnet.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

es = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

history_effnet = model_effnet.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[es])


try:
    class_names = validation_ds.class_names
except Exception:
    try:
        class_names = train_ds.class_names
    except Exception:
        # fallback: infer from train folder
        train_dir = "/kaggle/input/paddy-disease-classification/train_images"
        class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

print("Class names:", class_names)

# --- Evaluate model (loss & acc) ---
loss_effnet, acc_effnet = model_effnet.evaluate(val_ds, verbose=1)
print(f"\nEfficientNet - Validation Loss: {loss_effnet:.4f}, Accuracy: {acc_effnet:.4f}\n")

# --- Collect predictions and true labels from val dataset ---
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model_effnet.predict(images, verbose=0)
    y_true.extend(labels.numpy().tolist())                
    y_pred.extend(np.argmax(preds, axis=1).tolist())     

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt="d", cmap=colors,
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - EfficientNet", fontsize=14)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# --- Classification Report ---
print("Classification Report - EfficientNet:\n")
print(classification_report(y_true, y_pred, target_names=class_names))


# ğŸ”¹ Save EfficientNetB4 Model
model_effnet.save("efficientnet_model.h5")


# ğŸ”¹ Xception Model Build + Training
xception_base = Xception(weights='imagenet', include_top=False, input_shape=(224,224,3))
xception_base.trainable = False

model_xception = Sequential([
    xception_base,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model_xception.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

history_xception = model_xception.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[es])


# --- Xception Evaluation ---
loss_xcep, acc_xcep = model_xception.evaluate(val_ds, verbose=1)
print(f"\nXception - Validation Loss: {loss_xcep:.4f}, Accuracy: {acc_xcep:.4f}\n")

# Predictions
y_true, y_pred = [], []
for images, labels in val_ds:
    preds = model_xception.predict(images, verbose=0)
    y_true.extend(labels.numpy().tolist())
    y_pred.extend(np.argmax(preds, axis=1).tolist())

y_true, y_pred = np.array(y_true), np.array(y_pred)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt="d", cmap=colors,
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - Xception", fontsize=14)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Classification Report
print("Classification Report - Xception:\n")
print(classification_report(y_true, y_pred, target_names=class_names))


# ğŸ”¹ Save Xception Model
model_xception.save("xception_model.h5")


# ğŸ”¹ ResNet50 Model Build + Training
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
resnet_base.trainable = False

model_resnet = Sequential([
    resnet_base,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model_resnet.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

history_resnet = model_resnet.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[es])


# --- ResNet Evaluation ---
loss_resnet, acc_resnet = model_resnet.evaluate(val_ds, verbose=1)
print(f"\nResNet50 - Validation Loss: {loss_resnet:.4f}, Accuracy: {acc_resnet:.4f}\n")

# Predictions
y_true, y_pred = [], []
for images, labels in val_ds:
    preds = model_resnet.predict(images, verbose=0)
    y_true.extend(labels.numpy().tolist())
    y_pred.extend(np.argmax(preds, axis=1).tolist())

y_true, y_pred = np.array(y_true), np.array(y_pred)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt="d", cmap=colors,
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - ResNet50", fontsize=14)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Classification Report
print("Classification Report - ResNet50:\n")
print(classification_report(y_true, y_pred, target_names=class_names))


# ğŸ”¹ Save ResNet50 Model
model_resnet.save("resnet_model.h5")


def plot_advanced_history(histories, model_names):
    colors = ["#2e2905", "#806320", "#ad821d"]  
    
    # Accuracy Plot
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    for (history, name, color) in zip(histories, model_names, colors):
        plt.plot(history.history['accuracy'], linestyle='--', color=color, alpha=0.7, label=f"{name} Train Acc")
        plt.plot(history.history['val_accuracy'], color=color, linewidth=2, label=f"{name} Val Acc")
    plt.title("Training vs Validation Accuracy", fontsize=14, fontweight='bold')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Loss Plot
    plt.subplot(1,2,2)
    for (history, name, color) in zip(histories, model_names, colors):
        plt.plot(history.history['loss'], linestyle='--', color=color, alpha=0.7, label=f"{name} Train Loss")
        plt.plot(history.history['val_loss'], color=color, linewidth=2, label=f"{name} Val Loss")
    plt.title("Training vs Validation Loss", fontsize=14, fontweight='bold')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Call plotting function
histories = [history_effnet, history_xception, history_resnet]
model_names = ["EfficientNetB4", "Xception", "ResNet50"]

plot_advanced_history(histories, model_names)


# Collect final results
results = {
    "Model": model_names,
    "Final Train Acc": [h.history['accuracy'][-1] for h in histories],
    "Final Val Acc": [h.history['val_accuracy'][-1] for h in histories],
    "Final Train Loss": [h.history['loss'][-1] for h in histories],
    "Final Val Loss": [h.history['val_loss'][-1] for h in histories]
}

results_df = pd.DataFrame(results)

# Sort by best validation accuracy
results_df.sort_values(by="Final Val Acc", ascending=False, inplace=True)
results_df.reset_index(drop=True, inplace=True)

# Add Rank column
results_df.index = results_df.index + 1
results_df.index.name = "Rank"

# Display with gradient
import IPython.display as display
display.display(
    results_df.style.background_gradient(cmap="YlGnBu").set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#2e2105'),
                                     ('color', 'white'),
                                     ('font-weight', 'bold'),
                                     ('text-align', 'center')]}
    ]).set_properties(**{'text-align': 'center'})
)


# ğŸ�† Identify the Best Model
best_index = results_df["Final Val Acc"].astype(float).idxmax()
best_model_name = results_df.loc[best_index, "Model"]

print(f"ğŸ�† Best Model Identified: {best_model_name}")

# âœ… Save the Best Model
if best_model_name == "EfficientNetB4":
    model_effnet.save("best_model.h5")
elif best_model_name == "Xception":
    model_xception.save("best_model.h5")
elif best_model_name == "ResNet50":
    model_resnet.save("best_model.h5")

print("âœ… Best model saved as best_model.h5")

