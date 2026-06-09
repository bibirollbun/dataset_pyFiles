import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.regularizers import l2
from sklearn.metrics import roc_auc_score # Import roc_auc_score
from tensorflow.keras.metrics import AUC



df = pd.read_csv("/kaggle/input/cidaut-ai-fake-scene-classification-2024/train.csv")  


image_folder = "/kaggle/input/cidaut-ai-fake-scene-classification-2024/Train" 
image_size = 224


images = []
labels = []

for index, row in df.iterrows():
    image_path = os.path.join(image_folder, row['image'])
    img = load_img(image_path, target_size=(image_size, image_size))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    images.append(img_array)
    labels.append(row['label'])

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)


images.shape


label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)


X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)


class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))


datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)


unique_counts = df['label'].value_counts()  # Count occurrences of each unique value
print(unique_counts)



from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

IMG_SIZE = (224, 224)
# Load pre-trained InceptionV3 (or other model)
base_model = keras.applications.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Freeze the initial layers
for layer in base_model.layers[:100]:
    layer.trainable = False

# Add custom layers for classification
inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.7)(x) # Increased dropout
x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.02))(x) # Increased L2 regularization
x = layers.Dropout(0.4)(x)  # Increased dropout
x = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.02))(x) # Increased L2 regularization
x = layers.Dropout(0.4)(x)  # Increased dropout

outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)
# Print the model summary
model.summary()



train_ds = datagen.flow(X_train, y_train, batch_size=32)  # Adjust batch_size as needed
val_ds = datagen.flow(X_val, y_val, batch_size=32)  # Adjust batch_size as needed



import tensorflow as tf

# Learning rate schedule and optimizer
initial_learning_rate = 1e-4
optimizer = keras.optimizers.AdamW(learning_rate=initial_learning_rate)

def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return (lr * tf.math.exp(-0.1)).numpy()

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.4, patience=4, min_lr=1e-7, mode='max') # Adjusted parameters
es = tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True) # Adjusted parameters

# compile
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[keras.metrics.AUC(name='auc')])
# train the model
EPOCHS = 50 # Increased Epoch
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[callback, reduce_lr,es])



y_pred_probs = model.predict(X_val)

# Calculate and print ROC AUC
roc_auc = roc_auc_score(y_val, y_pred_probs)
print(f"ROC AUC: {roc_auc}")


test_image_folder = "/kaggle/input/cidaut-ai-fake-scene-classification-2024/Test" 
test_images = os.listdir(test_image_folder)

submission = []


for test_image in test_images:
    img_path = os.path.join(test_image_folder, test_image)
    img = load_img(img_path, target_size=(224, 224)) 
    img_array = img_to_array(img) / 255.0           
    img_array = np.expand_dims(img_array, axis=0)    
    

    prediction = model.predict(img_array)
    label = 1 if prediction[0][0] > 0.5 else 0 
    

    submission.append({"image": test_image, "label": label})


submission_df = pd.DataFrame(submission)

# save to CSV
submission_csv_path = "submission.csv"
submission_df.to_csv(submission_csv_path, index=False)

print(f"Submission file saved to {submission_csv_path}")
submission_df.head()





