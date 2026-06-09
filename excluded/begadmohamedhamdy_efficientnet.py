import efficientnet.keras as efn

# Standard ImageNet weights
model = efn.EfficientNetB3(weights="imagenet", include_top=False, input_shape=(300,300,3))

# Or Noisy Student weights
base = efn.EfficientNetB3(weights="noisy-student", include_top=False, input_shape=(300,300,3))


import efficientnet.keras as efn
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
import keras.backend as K
import os

# -------------------------------
# 1. Fix Keras backend issue
# -------------------------------
if not hasattr(K, "sigmoid"):
    K.sigmoid = tf.nn.sigmoid

# -------------------------------
# 2. Load EfficientNetB3 backbone
# -------------------------------
base = efn.EfficientNetB3(weights="noisy-student", include_top=False, input_shape=(300,300,3))
base.trainable = False   # freeze base for stage 1

# -------------------------------
# 3. Add custom classification head (5 classes for DR)
# -------------------------------
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.5)(x)
out = Dense(5, activation="softmax")(x)

model = Model(inputs=base.input, outputs=out)

# -------------------------------
# 4. Compile model
# -------------------------------
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# -------------------------------
# 5. Option A: REAL TRAINING (if you have dataset)
# -------------------------------
DATASET_PATH = "/kaggle/input/aptos2019-blindness-detection/train_images"
CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/train.csv"

if os.path.exists(DATASET_PATH):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Load CSV
    # Load CSV
    df = pd.read_csv(CSV_PATH)
    df['id_code'] = df['id_code'].apply(lambda x: f"{x}.png")

    # Convert diagnosis to string for flow_from_dataframe
    df['diagnosis'] = df['diagnosis'].astype(str)

    # Split train/val
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)


    # Generators
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=DATASET_PATH,
        x_col="id_code",
        y_col="diagnosis",
        target_size=(300, 300),
        batch_size=16,
        class_mode="categorical"
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=DATASET_PATH,
        x_col="id_code",
        y_col="diagnosis",
        target_size=(300, 300),
        batch_size=16,
        class_mode="categorical"
    )


    history = model.fit(train_generator,
                        validation_data=val_generator,
                        epochs=5)

# -------------------------------
# 5. Option B: MOCK TRAINING (using single image duplicated)
# -------------------------------
else:
    print("âš ï¸� No dataset found, running mock training with a single image...")

    # Load your retina image
    img_path = r"C:\Users\ghane\Downloads\1bb0ddfe753a.png"
    img = image.load_img(img_path, target_size=(300, 300))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Create fake dataset (duplicate same image with random labels)
    X_fake = np.vstack([img_array for _ in range(20)])
    y_fake = tf.keras.utils.to_categorical(np.random.randint(0, 5, 20), num_classes=5)

    # Train only on fake data (to test pipeline)
    history = model.fit(X_fake, y_fake, epochs=3, batch_size=4)

# -------------------------------
# 6. Fine-tuning (Stage 2)
# -------------------------------
print("\nğŸ”“ Unfreezing EfficientNet base for fine-tuning...")
base.trainable = True
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Run one fine-tuning epoch (mock or real)
if os.path.exists(DATASET_PATH):
    history_fine = model.fit(train_generator,
                             validation_data=val_generator,
                             epochs=3)
else:
    history_fine = model.fit(X_fake, y_fake, epochs=2, batch_size=4)

# -------------------------------
# 7. Test prediction on a single image
# -------------------------------
preds = model.predict(img_array)
print("Raw probabilities:", preds)
print("Predicted class index:", np.argmax(preds))



!pip install image




