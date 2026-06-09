import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import Xception, InceptionV3, MobileNet
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Define the data paths - UPDATED FOLDER NAME
train_data_dir = '/kaggle/input/deepfake-detection-challenge/train_sample_videos'
test_data_dir = '/kaggle/input/deepfake-detection-challenge/test_videos'

# Image preprocessing settings
img_width, img_height = 256, 256
batch_size = 64

def visualize_sample_images(data_dir, num_samples=5):
    """Visualize sample images from each class in the dataset"""
    classes = os.listdir(data_dir)
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        images = os.listdir(class_dir)
        sample_images = random.sample(images, min(num_samples, len(images)))
        
        plt.figure(figsize=(10, 5))
        for i, image_name in enumerate(sample_images):
            image_path = os.path.join(class_dir, image_name)
            image = load_img(image_path)
            plt.subplot(1, len(sample_images), i+1)
            plt.imshow(image)
            plt.title(class_name)
            plt.axis('off')
        plt.show()

# Visualize sample images from all sets
print("Training Set Sample Images:")
visualize_sample_images(train_data_dir)
print("Validation Set Sample Images:")
visualize_sample_images(valid_data_dir)
print("Test Set Sample Images:")
visualize_sample_images(test_data_dir)

def plot_class_distribution(data_dir, title):
    """Plot the distribution of classes in the dataset"""
    classes = os.listdir(data_dir)
    num_samples_per_class = [len(os.listdir(os.path.join(data_dir, class_name))) for class_name in classes]
    
    plt.figure(figsize=(8, 5))
    plt.bar(classes, num_samples_per_class)
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Plot class distribution for all sets
plot_class_distribution(train_data_dir, 'Training Set Class Distribution')
plot_class_distribution(valid_data_dir, 'Validation Set Class Distribution')
plot_class_distribution(test_data_dir, 'Test Set Class Distribution')

# Data generators with augmentation for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20
)

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = valid_datagen.flow_from_directory(
    valid_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

def visualize_augmented_images(data_generator, num_samples=5):
    """Visualize augmented images from the data generator"""
    plt.figure(figsize=(12, 6))
    batch_x, batch_y = next(data_generator)
    
    for i in range(min(num_samples, len(batch_x))):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(batch_x[i])
        plt.axis('off')
        plt.title(f'Label: {int(batch_y[i])}')
    plt.show()

# Visualize augmented images from the training generator
print("Augmented Training Images:")
visualize_augmented_images(train_generator)

def visualize_original_and_augmented(image_path, data_generator, num_augmented=5):
    """Show original image alongside its augmented versions"""
    original_image = load_img(image_path)
    image_array = img_to_array(original_image)
    image_array = np.expand_dims(image_array, axis=0)
    
    aug_iter = data_generator.flow(image_array, batch_size=1)
    
    plt.figure(figsize=(15, 3))
    plt.subplot(1, num_augmented + 1, 1)
    plt.imshow(original_image)
    plt.title("Original")
    plt.axis('off')
    
    for i in range(num_augmented):
        augmented_image = next(aug_iter)[0]
        plt.subplot(1, num_augmented + 1, i + 2)
        plt.imshow(array_to_img(augmented_image))
        plt.title(f"Aug {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def get_random_image_paths(root_dir, num_samples=10, extensions=('.jpg', '.png', '.jpeg')):
    """Get random image paths from the directory"""
    image_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith(extensions):
                image_paths.append(os.path.join(dirpath, file))
    
    return random.sample(image_paths, min(num_samples, len(image_paths)))

# Get random sample paths and visualize augmentations - UPDATED PATH
sample_paths = get_random_image_paths('/kaggle/input/faceforensic1600-videospreprocess', num_samples=3)
print("Original vs Augmented Images:")
for image_path in sample_paths:
    visualize_original_and_augmented(image_path, train_datagen, num_augmented=5)

# Calculate steps per epoch
train_steps_per_epoch = train_generator.samples // batch_size
valid_steps_per_epoch = validation_generator.samples // batch_size

print(f"Training steps per epoch: {train_steps_per_epoch}")
print(f"Validation steps per epoch: {valid_steps_per_epoch}")

# 1. CNN Model
def create_cnn_model():
    """Create a simple CNN model"""
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

model_cnn = create_cnn_model()
model_cnn.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(learning_rate=0.001),
    metrics=['accuracy']
)

print("CNN Model Summary:")
model_cnn.summary()

# Train CNN model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("Training CNN Model...")
history_cnn = model_cnn.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=valid_steps_per_epoch,
    verbose=1,
    callbacks=[early_stopping]
)

# Plot CNN training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history_cnn.history['accuracy'])
plt.plot(history_cnn.history['val_accuracy'])
plt.title('CNN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history_cnn.history['loss'])
plt.plot(history_cnn.history['val_loss'])
plt.title('CNN Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# 2. Xception Model
def create_xception_model():
    """Create Xception-based model"""
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

xception_model = create_xception_model()
xception_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Training Xception Model...")
history_xception = xception_model.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=valid_steps_per_epoch,
    callbacks=[early_stopping]
)

# Plot Xception training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history_xception.history['accuracy'])
plt.plot(history_xception.history['val_accuracy'])
plt.title('Xception Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history_xception.history['loss'])
plt.plot(history_xception.history['val_loss'])
plt.title('Xception Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# 3. InceptionV3 Model
def create_inception_model():
    """Create InceptionV3-based model"""
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

inception_model = create_inception_model()
inception_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Training InceptionV3 Model...")
history_inception = inception_model.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=valid_steps_per_epoch,
    callbacks=[early_stopping]
)

# Plot InceptionV3 training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history_inception.history['accuracy'])
plt.plot(history_inception.history['val_accuracy'])
plt.title('InceptionV3 Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history_inception.history['loss'])
plt.plot(history_inception.history['val_loss'])
plt.title('InceptionV3 Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# 4. MobileNet Model
def create_mobilenet_model():
    """Create MobileNet-based model"""
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

mobilenet_model = create_mobilenet_model()
mobilenet_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Training MobileNet Model...")
history_mobilenet = mobilenet_model.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=valid_steps_per_epoch,
    callbacks=[early_stopping]
)

# Plot MobileNet training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history_mobilenet.history['accuracy'])
plt.plot(history_mobilenet.history['val_accuracy'])
plt.title('MobileNet Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history_mobilenet.history['loss'])
plt.plot(history_mobilenet.history['val_loss'])
plt.title('MobileNet Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluation Function
def evaluate_model(model, model_name, test_gen):
    """Evaluate a single model and display results"""
    print(f"\nEvaluating {model_name} Model...")
    
    # Get predictions
    y_pred = model.predict(test_gen, steps=len(test_gen), verbose=1)
    y_pred_binary = np.round(y_pred)
    
    # Get true labels
    y_true = test_gen.classes
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred_binary)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    
    # Classification report
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_true, y_pred_binary, target_names=['fake', 'real']))
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred_binary)
    plt.figure(figsize=(7, 5))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', 
                xticklabels=['fake', 'real'], yticklabels=['fake', 'real'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()
    
    return y_pred_binary, accuracy

# Evaluate all models
print("="*50)
print("MODEL EVALUATION")
print("="*50)

# Reset test generator
test_generator.reset()
cnn_pred, cnn_acc = evaluate_model(model_cnn, "CNN", test_generator)

test_generator.reset()
xception_pred, xception_acc = evaluate_model(xception_model, "Xception", test_generator)

test_generator.reset()
inception_pred, inception_acc = evaluate_model(inception_model, "InceptionV3", test_generator)

test_generator.reset()
mobilenet_pred, mobilenet_acc = evaluate_model(mobilenet_model, "MobileNet", test_generator)

# Ensemble Method
def ensemble_predict(models, test_gen, threshold=0.5):
    """Create ensemble predictions using majority voting"""
    test_gen.reset()
    predictions = []
    
    for i, model in enumerate(models):
        print(f"Getting predictions from model {i+1}/{len(models)}...")
        pred = (model.predict(test_gen, verbose=0) > threshold).astype('int32').flatten()
        predictions.append(pred)
        test_gen.reset()  # Reset for next model
    
    predictions = np.array(predictions)
    # Majority voting
    ensemble_predictions = np.apply_along_axis(
        lambda x: Counter(x).most_common(1)[0][0], 
        axis=0, 
        arr=predictions
    )
    return ensemble_predictions

# Create ensemble
print("\n" + "="*50)
print("ENSEMBLE MODEL EVALUATION")
print("="*50)

models_list = [xception_model, inception_model, mobilenet_model]
model_names = ["Xception", "InceptionV3", "MobileNet"]

ensemble_pred = ensemble_predict(models_list, test_generator)
y_true = test_generator.classes

# Ensemble evaluation
ensemble_acc = accuracy_score(y_true, ensemble_pred)
print(f"Ensemble Accuracy: {ensemble_acc:.4f}")

print("\nEnsemble Classification Report:")
print(classification_report(y_true, ensemble_pred, target_names=['fake', 'real']))

# Ensemble confusion matrix
conf_matrix_ensemble = confusion_matrix(y_true, ensemble_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix_ensemble, annot=True, cmap='Blues', fmt='g', 
            xticklabels=['fake', 'real'], yticklabels=['fake', 'real'])
plt.title('Ensemble Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# Summary of all model performances
print("\n" + "="*50)
print("SUMMARY OF MODEL PERFORMANCES")
print("="*50)
print(f"CNN Accuracy:        {cnn_acc:.4f}")
print(f"Xception Accuracy:   {xception_acc:.4f}")
print(f"InceptionV3 Accuracy: {inception_acc:.4f}")
print(f"MobileNet Accuracy:  {mobilenet_acc:.4f}")
print(f"Ensemble Accuracy:   {ensemble_acc:.4f}")

# Find best performing model
accuracies = [cnn_acc, xception_acc, inception_acc, mobilenet_acc, ensemble_acc]
model_names_all = ["CNN", "Xception", "InceptionV3", "MobileNet", "Ensemble"]
best_model_idx = np.argmax(accuracies)
print(f"\nBest performing model: {model_names_all[best_model_idx]} with accuracy: {accuracies[best_model_idx]:.4f}")

