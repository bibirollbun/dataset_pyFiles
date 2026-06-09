import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Paths
df = pd.read_csv("/kaggle/input/cidaut-ai-fake-scene-classification-2024/train.csv")
train_dir = "/kaggle/input/cidaut-ai-fake-scene-classification-2024/Train"
test_dir = "/kaggle/input/cidaut-ai-fake-scene-classification-2024/Test"
sample_submission_csv = "/kaggle/input/cidaut-ai-fake-scene-classification-2024/sample_submission.csv"


df['label'].value_counts()


# Encode labels (real/editada -> 1/0)
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])
print(f"Label Mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")



df


train_data,val_data=train_test_split(df,test_size=0.20,random_state=42)


# Save split datasets to temporary CSV files (optional)
train_data.to_csv("train_split.csv", index=False)
val_data.to_csv("val_split.csv", index=False)


# Image data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)




val_datagen = ImageDataGenerator(rescale=1.0/255)


# Convert 'label' column values to strings
train_data['label'] = train_data['label'].astype(str)
val_data['label'] = val_data['label'].astype(str)

train_generator = train_datagen.flow_from_dataframe(
    train_data,
    directory=train_dir,
    x_col='image',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)



val_generator = val_datagen.flow_from_dataframe(
    val_data,
    directory=train_dir,
    x_col='image',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)


# Step 3: Define the Model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)  # Binary classification
model = Model(inputs=base_model.input, outputs=output)


# Freeze base model initially
base_model.trainable = False


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['AUC'])



early_stopping = EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_auc', mode='max', save_best_only=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=5, min_lr=1e-6, verbose=1)



base_model.trainable = True


# Recompile with a lower learning rate
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['AUC'])



# Fine-tune the model
fine_tune_history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[early_stopping, model_checkpoint, lr_scheduler]
)


def preprocess_test_images(test_dir):
    test_images = []
    test_filenames = []
    for img_name in sorted(os.listdir(test_dir)):
        img_path = os.path.join(test_dir, img_name)
        if os.path.isfile(img_path):
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            test_images.append(img_array)
            test_filenames.append(img_name)
    return np.array(test_images), test_filenames



test_images, test_filenames = preprocess_test_images(test_dir)



# Generate predictions
test_predictions = model.predict(test_images)
test_labels = (test_predictions >= 0.5).astype(int).ravel()



# Step 9: Create Submission File
submission = pd.DataFrame({'image': test_filenames, 'label': test_labels})
submission.to_csv('submission_csv', index=False)



submission

