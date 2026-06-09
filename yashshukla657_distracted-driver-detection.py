#  Import required libraries
import os
import random
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


#  Function to create training/testing folders and move images
def prepare_data(base_input_path, base_output_path, train_ratio=0.8):
    labels = [f'c{i}' for i in range(10)]
    for label in labels:
        input_folder = os.path.join(base_input_path, label)
        images = os.listdir(input_folder)
        random.shuffle(images)
        split_point = int(len(images) * train_ratio)

        # Create class folders for train and test
        train_class_dir = os.path.join(base_output_path, 'training', label)
        test_class_dir = os.path.join(base_output_path, 'testing', label)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        for i, image in enumerate(images):
            src_path = os.path.join(input_folder, image)
            if i < split_point:
                dest_path = os.path.join(train_class_dir, image)
            else:
                dest_path = os.path.join(test_class_dir, image)
            shutil.copy(src_path, dest_path)

# ðŸš€ Prepare data (run once)
prepare_data(
    base_input_path="/kaggle/input/state-farm-distracted-driver-detection/imgs/train",
    base_output_path="/kaggle/working/master_data"
)


# ðŸ”„ Load data using ImageDataGenerator
train_path = "/kaggle/working/master_data/training"
test_path = "/kaggle/working/master_data/testing"

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path, target_size=(224, 224), batch_size=32, class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_path, target_size=(224, 224), batch_size=32, class_mode='categorical'
)



#  Build transfer learning model using MobileNetV2
def build_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
model.summary()


#  EarlyStopping to avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# ðŸ”§ Train model
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10,
    callbacks=[early_stop]
)



#  Unfreeze top layers for fine-tuning
model.layers[0].trainable = True  # Unfreeze base model

# Recompile with lower learning rate
model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

#  Retrain
fine_tune_history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=5,
    callbacks=[early_stop]
)


#  Save model
model.save("driver_distraction_model.h5")
print("Model saved.")

#  Plot training performance
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.plot(fine_tune_history.history['accuracy'], label='Fine-tune Acc')
plt.plot(fine_tune_history.history['val_accuracy'], label='Fine-tune Val Acc')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# Load trained model
model = load_model('driver_distraction_model.h5')
print("model loaded")

# Path to test images
test_path = "/kaggle/input/state-farm-distracted-driver-detection/imgs/test"
test_images = os.listdir(test_path)

# Collect rows in a list
submission_rows = []

# Process each image
for img_name in test_images:
    print("haha")
    img_path = os.path.join(test_path, img_name)
    
    # Load and preprocess image
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict probabilities
    preds = model.predict(img_array, verbose=0)[0]
    
    # Create row dictionary
    row = {'img': img_name}
    for i in range(10):
        row[f'c{i}'] = preds[i]
    submission_rows.append(row)

# Convert to DataFrame
submission = pd.DataFrame(submission_rows)

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("âœ… submission.csv generated successfully!")

