

# ## Cell 1: Imports and Setup
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU(s) available: {len(gpus)}")
else:
    print("No GPU detected.")



# ## Cell 2: Configuration

# Paths to your Kaggle data
TRAIN_CSV = '/kaggle/input/aptos2019-blindness-detection/train.csv'
TRAIN_IMG_DIR = '/kaggle/input/aptos2019-blindness-detection/train_images'

# Model and training hyperparameters
IMG_SIZE = (384, 384)
BATCH_SIZE = 16
SEED = 42
EPOCHS = 25  # Increased from 20 to allow more time for augmentation
NUM_CLASSES = 5

# Set seeds for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ## Cell 3: Load and Prepare Data

# Load the CSV and ensure the diagnosis column is a string for the generator
df = pd.read_csv(TRAIN_CSV)
df['id_code'] = df['id_code'].astype(str) + '.png'

# This line fixes the error by converting the diagnosis labels to strings
df['diagnosis'] = df['diagnosis'].astype(str)

print("Data loaded successfully.")
print(f"Total samples: {len(df)}")


# ## Cell 4: Quick EDA
# Visualize the class distribution
plt.figure(figsize=(10, 6))
df['diagnosis'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Class Distribution of Diabetic Retinopathy')
plt.xlabel('Diagnosis Level')
plt.ylabel('Number of Images')
plt.xticks(rotation=0)
plt.show()



# ## Cell 5: Train/Validation Split
# Create a stratified split to maintain class distribution in both sets
train_df, val_df = train_test_split(
    df,
    test_size=0.15,
    random_state=SEED,
    stratify=df['diagnosis']
)

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")



# ## Cell 6: Data Generators (with Augmentations)

# Add basic augmentations to the training generator to combat overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    horizontal_flip=True,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# The validation generator should NOT have augmentations
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=TRAIN_IMG_DIR,
    x_col='id_code',
    y_col='diagnosis',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=True,
    seed=SEED
)

val_gen = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=TRAIN_IMG_DIR,
    x_col='id_code',
    y_col='diagnosis',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)


# ## Cell 7: Calculate Class Weights
# Correctly compute class weights on the integer labels to handle imbalance
classes = np.unique(train_df['diagnosis'])
class_weights = compute_class_weight(
    'balanced',
    classes=classes,
    y=train_df['diagnosis'].values
)
class_weight_dict = {c: w for c, w in zip(classes, class_weights)}

print("Class weights calculated:")
print(class_weight_dict)


# ## Cell 8: Build the Model
def build_simple_model(input_shape=IMG_SIZE + (3,), n_classes=NUM_CLASSES):
    """Builds a simple, robust EfficientNetB0 model for single-phase training."""
    # Base model
    base = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base.trainable = True # Train the whole model

    # Model architecture
    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    model = Model(inputs, outputs)

    # Compile the model with the correct loss function
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy', # The correct loss for 'sparse' mode
        metrics=['accuracy']
    )
    return model

model = build_simple_model()
model.summary()



# ## Cell 9: Define Callbacks (with Gentler LR Reduction)
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5, # Stop if no improvement for 5 epochs
    restore_best_weights=True,
    mode='max',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # Changed to 0.5 for a gentler reduction
    patience=2,
    verbose=1,
    min_lr=1e-7
)




# ## Cell 10: Train the Model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=[checkpoint, early_stopping, reduce_lr],
    verbose=1
)



# ## Cell 11: Plot Training History
def plot_history(history):
    """Plots accuracy and loss curves for training and validation."""
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Plot accuracy
    ax[0].plot(history.history['accuracy'], label='Train Accuracy')
    ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[0].set_title('Model Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()

    # Plot loss
    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Validation Loss')
    ax[1].set_title('Model Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    plt.show()

plot_history(history)




# ## Cell 12: Evaluate the Model
# Load the best performing model
model.load_weights('best_model.h5')

# Make predictions on the validation set
preds = model.predict(val_gen)
pred_classes = np.argmax(preds, axis=1)

# Get true labels directly from the generator for robust evaluation
true_classes = val_gen.classes

# Calculate Quadratic Weighted Kappa
qwk = cohen_kappa_score(true_classes, pred_classes, weights='quadratic')
print(f"\nðŸ“ˆ Validation Quadratic Weighted Kappa (QWK): {qwk:.4f}\n")

# Print Classification Report
print("ðŸ“Š Classification Report:\n")
print(classification_report(true_classes, pred_classes, target_names=[str(i) for i in classes]))

# Display Confusion Matrix
cm = confusion_matrix(true_classes, pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()




