import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_DIR = os.path.join(DATA_DIR, "train_images")
TEST_DIR = os.path.join(DATA_DIR, "test_images")

# Load training labels
df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_train['label'] = df_train['label'].astype('str')  # For flow_from_dataframe

# Class names
class_names = {
    '0': 'Cassava Bacterial Blight (CBB)',
    '1': 'Cassava Brown Streak Disease (CBSD)',
    '2': 'Cassava Green Mottle (CGM)',
    '3': 'Cassava Mosaic Disease (CMD)',
    '4': 'Healthy'
}



IMG_SIZE = 380
BATCH_SIZE = 16

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)


train_gen = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    directory=TRAIN_DIR,
    x_col="image_id",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

valid_gen = valid_datagen.flow_from_dataframe(
    dataframe=df_train,
    directory=TRAIN_DIR,
    x_col="image_id",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Class weights
class_counts = df_train['label'].value_counts()
total = sum(class_counts)
class_weights = {i: total/(len(class_counts)*count) for i, count in enumerate(class_counts)}


# ======================
# 5. MODEL ARCHITECTURE (CORRECTED)
# ======================
def build_model():
    base_model = EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)  # Removed invalid parameter
    )
    
    # Freeze first 150 layers
    for layer in base_model.layers[:150]:
        layer.trainable = False
    
    # Add custom dropout instead
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),  # Explicit dropout layer
        layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),  # Additional dropout
        layers.Dense(5, activation='softmax')
    ])
    
    return model

model = build_model()


model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)



callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
]


# ======================
history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.n//train_gen.batch_size,
    validation_data=valid_gen,
    validation_steps=valid_gen.n//valid_gen.batch_size,
    epochs=30,
    callbacks=callbacks,
    class_weight=class_weights
)



def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Predict on validation set
y_pred = model.predict(valid_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = valid_gen.classes

# Classification report
print(classification_report(y_true, y_pred_classes, target_names=class_names.values()))

# Confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_true, y_pred_classes), 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names.values(),
            yticklabels=class_names.values())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


