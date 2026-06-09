import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S, DenseNet201
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, concatenate, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import os

# Use GPU if available, otherwise use CPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Using GPU")
    except RuntimeError as e:
        print(e)
else:
    print("Using CPU")

# Load and preprocess data
train_df = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/train.csv')
train_df['diagnosis'] = train_df['diagnosis'].astype(str)
train_df['id_code'] = train_df['id_code'].apply(lambda x: f"{x}.png")

# Label encoding
le = LabelEncoder()
train_df['diagnosis'] = le.fit_transform(train_df['diagnosis'])

def preprocess_image(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (380, 380))
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.2)
    return image, label

def create_dataset(dataframe, batch_size, is_training=True):
    image_paths = tf.constant([f"/kaggle/input/aptos2019-blindness-detection/train_images/{filename}" for filename in dataframe['id_code']])
    labels = tf.constant(dataframe['diagnosis'].values, dtype=tf.int32)
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=len(dataframe))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def create_model(input_shape=(380, 380, 3), num_classes=5):
    input_tensor = Input(shape=input_shape)
    
    base_model_1 = EfficientNetV2S(weights='imagenet', include_top=False, input_tensor=input_tensor)
    x1 = GlobalAveragePooling2D()(base_model_1.output)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Dense(512, activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    
    base_model_2 = DenseNet201(weights='imagenet', include_top=False, input_tensor=input_tensor)
    x2 = GlobalAveragePooling2D()(base_model_2.output)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.5)(x2)
    x2 = Dense(512, activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.3)(x2)
    
    combined = concatenate([x1, x2])
    combined = Dense(256, activation='relu')(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.3)(combined)
    output = Dense(num_classes, activation='softmax')(combined)
    
    model = Model(inputs=input_tensor, outputs=output)
    return model

# Implement k-fold cross-validation
n_splits = 2
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['diagnosis']), 1):
    print(f"Training Fold {fold}")
    
    train_fold = train_df.iloc[train_idx].reset_index(drop=True)
    val_fold = train_df.iloc[val_idx].reset_index(drop=True)
    
    model = create_model()
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True),
        ModelCheckpoint(f'best_model_fold_{fold}.keras', save_best_only=True),
        ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6)
    ]
    
    # Create datasets
    batch_size = 16  # Adjust based on your GPU memory
    train_dataset = create_dataset(train_fold, batch_size)
    val_dataset = create_dataset(val_fold, batch_size, is_training=False)
    
    # Train model
    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=20,
            callbacks=callbacks,
            steps_per_epoch=len(train_fold) // batch_size,
            validation_steps=len(val_fold) // batch_size,
            verbose=2
        )
        
        # Print fold results
        print(f"Fold {fold} - Best validation accuracy: {max(history.history['val_accuracy'])}")
    except Exception as e:
        print(f"An error occurred during training fold {fold}: {str(e)}")
        continue

print("Training completed for all folds.")


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import os
from PIL import Image
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path, target_size=(380, 380)):
    """
    Load and preprocess a single image for prediction
    """
    # Read the image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    
    # Resize to match the model's expected input
    img = tf.image.resize(img, target_size)
    
    # Normalize the pixel values
    img = tf.cast(img, tf.float32) / 255.0
    
    # Add batch dimension
    img = tf.expand_dims(img, 0)
    
    return img

def predict_image(model_path, image_path):
    """
    Make a prediction for a single image using the saved model
    """
    # Load the model
    try:
        # First try loading the Keras model
        model = load_model(model_path)
        print(f"Loaded Keras model from {model_path}")
    except:
        # If that fails, try loading the pickle model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Loaded pickle model from {model_path}")
    
    # Load and preprocess the image
    img = load_and_preprocess_image(image_path)
    
    # Make the prediction
    prediction = model.predict(img)
    
    # Get the predicted class
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    
    # Map the class index to its meaning (adjust these labels based on your dataset)
    severity_map = {
        0: "No DR (No diabetic retinopathy)",
        1: "Mild DR",
        2: "Moderate DR",
        3: "Severe DR", 
        4: "Proliferative DR"
    }
    
    severity = severity_map.get(predicted_class, f"Unknown class {predicted_class}")
    
    return {
        "class": predicted_class,
        "severity": severity,
        "confidence": confidence,
        "raw_probabilities": prediction[0]
    }

def display_prediction(image_path, prediction_result):
    """
    Display the image with its prediction result
    """
    # Load the image for display
    img = Image.open(image_path)
    
    # Display the image and prediction
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title(f"Prediction: {prediction_result['severity']}\nConfidence: {prediction_result['confidence']:.2f}%")
    plt.axis('off')
    plt.show()
    
    # Print detailed results
    print(f"Prediction details:")
    print(f"Class: {prediction_result['class']}")
    print(f"Severity: {prediction_result['severity']}")
    print(f"Confidence: {prediction_result['confidence']:.2f}%")
    print("Probabilities for each class:")
    for i, prob in enumerate(prediction_result['raw_probabilities']):
        severity_map = {
            0: "No DR",
            1: "Mild DR",
            2: "Moderate DR",
            3: "Severe DR", 
            4: "Proliferative DR"
        }
        print(f"  {severity_map.get(i, f'Class {i}')}: {prob*100:.2f}%")

# Example usage
if __name__ == "__main__":
    # Update these paths to match your environment
    model_path = "/kaggle/input/model/best_model_fold_2 (1).keras"  # or .pkl if you're using the pickle version
    
    # You can test with a single image
    image_path = "/kaggle/input/aptos2019-blindness-detection/test_images/003f0afdcd15.png"  # Update this path
    
    # Make prediction
    result = predict_image(model_path, image_path)
    
    # Display the result
    display_prediction(image_path, result)
    
    # If you want to run prediction on multiple images in a folder
    """
    test_folder = "/path/to/test_images_folder"
    for filename in os.listdir(test_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(test_folder, filename)
            print(f"\nProcessing: {filename}")
            result = predict_image(model_path, image_path)
            display_prediction(image_path, result)
    """

