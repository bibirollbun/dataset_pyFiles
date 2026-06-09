import os
import numpy as np
import pandas as pd
import random
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)




df = pd.read_csv('/kaggle/input/open-data-day-2025-dates-types-classification/train_labels.csv')  

image_folder = r"//kaggle/input/open-data-day-2025-dates-types-classification/train"

df["exists"] = df["filename"].apply(lambda x: os.path.exists(os.path.join(image_folder, x)))
df = df[df["exists"]]

num_rows, num_cols = 3, 5
sample_df = df.sample(n=num_rows * num_cols, random_state=42).reset_index(drop=True)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 9))
for i in range(num_rows * num_cols):
    row, col = divmod(i, num_cols)
    image_name = sample_df.loc[i, "filename"]
    label = sample_df.loc[i, "label"]
    image_path = os.path.join(image_folder, image_name)
    image = Image.open(image_path)
    axes[row, col].imshow(image)
    axes[row, col].set_title(label, fontsize=10)
    axes[row, col].axis("off")
plt.suptitle("Sample Images with Labels", fontsize=16)
plt.tight_layout()
plt.show()

label_counts = df["label"].value_counts()
plt.figure(figsize=(8, 5))
plt.bar(label_counts.index, label_counts.values, color='skyblue')
plt.title("Label Distribution", fontsize=14)
plt.xlabel("Categories", fontsize=12)
plt.ylabel("Number of Images", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

def load_and_preprocess_image(image_path, target_size=(64, 64)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    return img_array

df["image_path"] = df["filename"].apply(lambda x: os.path.join(image_folder, x))
images = np.array([load_and_preprocess_image(path) for path in df['image_path']])

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df['label'])

# Save processed data
np.save("images.npy", images)
np.save("labels.npy", labels)
with open("label_encoder.pkl", "wb") as f:
    import pickle
    pickle.dump(label_encoder, f)

print("Data saved successfully!")




df_limited = df.groupby("label").head(44).reset_index(drop=True)

print("New Dataset Size:", len(df_limited))
print(df_limited["label"].value_counts())  


df_limited.to_csv("filtered_train_labels.csv", index=False)

print("Filtered dataset saved successfully!")



invalid_exists = df_limited[~df_limited['exists'].isin([True, False])]

if not invalid_exists.empty:
    print("Rows with invalid values in 'exists' column:")
    print(invalid_exists)
else:
    print("All values in the 'exists' column are valid (True or False).")



df_limited = df_limited.drop(columns=['filename', 'exists'])

df_limited


df_limited["label"] = df_limited["label"].str.strip()

label_mapping = {
    "Ajwa": 0,
    "Medjool": 1,
    "Meneifi": 2,
    "Nabtat Ali": 3,
    "Shaishe": 4,
    "Sokari": 5,
    "Sugaey": 6
}

df_limited["label"] = df_limited["label"].map(label_mapping)

print(df_limited[df_limited["label"].isna()])



df_limited["label"] = df_limited["label"].replace(label_mapping)
df_limited


np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

X = df_limited['image_path']
y = df_limited['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.14, 
    stratify=y,
    random_state=43

)

print(f"Training Data: {len(X_train)} samples")
print(f"Test Data: {len(X_test)} samples")

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

y_train_onehot = to_categorical(y_train_encoded)
y_test_onehot = to_categorical(y_test_encoded)

def load_and_process_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  
    return img_array

X_train_images = np.array([load_and_process_image(image_path) for image_path in X_train])
X_test_images = np.array([load_and_process_image(image_path) for image_path in X_test])

model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train_images, 
    y_train_onehot, 
    epochs=25, 
    validation_data=(X_test_images, y_test_onehot),
    callbacks=[early_stopping]
)

loss, accuracy = model.evaluate(X_test_images, y_test_onehot)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

model.save("dates_classifier_model.h5")

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)




absl.logging.set_verbosity(absl.logging.ERROR)

np.random.seed(42)
tf.random.set_seed(42)

image_path = "/kaggle/input/open-data-day-2025-dates-types-classification/test/03ab6acd.jpg"

model = load_model("dates_classifier_model.h5")

label_mapping = {
    "Ajwa": 0,
    "Medjool": 1,
    "Meneifi": 2,
    "Nabtat Ali": 3,
    "Shaishe": 4,
    "Sokari": 5,
    "Sugaey": 6
}

reverse_label_mapping = {v: k for k, v in label_mapping.items()}

def load_and_process_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    return img_array

image = load_and_process_image(image_path)

image_expanded = np.expand_dims(image, axis=0)

y_pred = model.predict(image_expanded)

predicted_label = np.argmax(y_pred, axis=1)

predicted_label_name = reverse_label_mapping[predicted_label[0]]

plt.imshow(image)
plt.title(f"Predicted Label: {predicted_label_name}")
plt.axis('off')
plt.show()

print(f"Image: {image_path} --> Predicted Label: {predicted_label_name}")




absl.logging.set_verbosity(absl.logging.ERROR)

np.random.seed(42)
tf.random.set_seed(42)

image_path = "/kaggle/input/open-data-day-2025-dates-types-classification/test/0be87456.jpg"

model = load_model("dates_classifier_model.h5")

label_mapping = {
    "Ajwa": 0,
    "Medjool": 1,
    "Meneifi": 2,
    "Nabtat Ali": 3,
    "Shaishe": 4,
    "Sokari": 5,
    "Sugaey": 6
}

reverse_label_mapping = {v: k for k, v in label_mapping.items()}

def load_and_process_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    return img_array

image = load_and_process_image(image_path)

image_expanded = np.expand_dims(image, axis=0)

y_pred = model.predict(image_expanded)

predicted_label = np.argmax(y_pred, axis=1)

predicted_label_name = reverse_label_mapping[predicted_label[0]]

plt.imshow(image)
plt.title(f"Predicted Label: {predicted_label_name}")
plt.axis('off')
plt.show()

print(f"Image: {image_path} --> Predicted Label: {predicted_label_name}")




absl.logging.set_verbosity(absl.logging.ERROR)

np.random.seed(42)
tf.random.set_seed(42)

image_path = "/kaggle/input/open-data-day-2025-dates-types-classification/test/10f11e1e.jpg"

model = load_model("dates_classifier_model.h5")

label_mapping = {
    "Ajwa": 0,
    "Medjool": 1,
    "Meneifi": 2,
    "Nabtat Ali": 3,
    "Shaishe": 4,
    "Sokari": 5,
    "Sugaey": 6
}

reverse_label_mapping = {v: k for k, v in label_mapping.items()}

def load_and_process_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    return img_array

image = load_and_process_image(image_path)

image_expanded = np.expand_dims(image, axis=0)

y_pred = model.predict(image_expanded)

predicted_label = np.argmax(y_pred, axis=1)

predicted_label_name = reverse_label_mapping[predicted_label[0]]

plt.imshow(image)
plt.title(f"Predicted Label: {predicted_label_name}")
plt.axis('off')
plt.show()

print(f"Image: {image_path} --> Predicted Label: {predicted_label_name}")




absl.logging.set_verbosity(absl.logging.ERROR)

np.random.seed(42)
tf.random.set_seed(42)

image_path = "/kaggle/input/open-data-day-2025-dates-types-classification/test/179f6419.jpg"

model = load_model("dates_classifier_model.h5")

label_mapping = {
    "Ajwa": 0,
    "Medjool": 1,
    "Meneifi": 2,
    "Nabtat Ali": 3,
    "Shaishe": 4,
    "Sokari": 5,
    "Sugaey": 6
}

reverse_label_mapping = {v: k for k, v in label_mapping.items()}

def load_and_process_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    return img_array

image = load_and_process_image(image_path)

image_expanded = np.expand_dims(image, axis=0)

y_pred = model.predict(image_expanded)

predicted_label = np.argmax(y_pred, axis=1)

predicted_label_name = reverse_label_mapping[predicted_label[0]]

plt.imshow(image)
plt.title(f"Predicted Label: {predicted_label_name}")
plt.axis('off')
plt.show()

print(f"Image: {image_path} --> Predicted Label: {predicted_label_name}")




absl.logging.set_verbosity(absl.logging.ERROR)

np.random.seed(42)
tf.random.set_seed(42)

image_path = "/kaggle/input/open-data-day-2025-dates-types-classification/test/0bd19b28.jpg"

model = load_model("dates_classifier_model.h5")

label_mapping = {
    "Ajwa": 0,
    "Medjool": 1,
    "Meneifi": 2,
    "Nabtat Ali": 3,
    "Shaishe": 4,
    "Sokari": 5,
    "Sugaey": 6
}

reverse_label_mapping = {v: k for k, v in label_mapping.items()}

def load_and_process_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    return img_array

image = load_and_process_image(image_path)

image_expanded = np.expand_dims(image, axis=0)

y_pred = model.predict(image_expanded)

predicted_label = np.argmax(y_pred, axis=1)

predicted_label_name = reverse_label_mapping[predicted_label[0]]

plt.imshow(image)
plt.title(f"Predicted Label: {predicted_label_name}")
plt.axis('off')
plt.show()

print(f"Image: {image_path} --> Predicted Label: {predicted_label_name}")




absl.logging.set_verbosity(absl.logging.ERROR)

np.random.seed(42)
tf.random.set_seed(42)

image_path = "/kaggle/input/open-data-day-2025-dates-types-classification/test/21c321ea.png"

model = load_model("dates_classifier_model.h5")

label_mapping = {
    "Ajwa": 0,
    "Medjool": 1,
    "Meneifi": 2,
    "Nabtat Ali": 3,
    "Shaishe": 4,
    "Sokari": 5,
    "Sugaey": 6
}

reverse_label_mapping = {v: k for k, v in label_mapping.items()}

def load_and_process_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    return img_array

image = load_and_process_image(image_path)

image_expanded = np.expand_dims(image, axis=0)

y_pred = model.predict(image_expanded)

predicted_label = np.argmax(y_pred, axis=1)

predicted_label_name = reverse_label_mapping[predicted_label[0]]

plt.imshow(image)
plt.title(f"Predicted Label: {predicted_label_name}")
plt.axis('off')
plt.show()

print(f"Image: {image_path} --> Predicted Label: {predicted_label_name}")




absl.logging.set_verbosity(absl.logging.ERROR)

np.random.seed(42)
tf.random.set_seed(42)

image_path = "/kaggle/input/open-data-day-2025-dates-types-classification/test/28728d90.jpg"

model = load_model("dates_classifier_model.h5")

label_mapping = {
    "Ajwa": 0,
    "Medjool": 1,
    "Meneifi": 2,
    "Nabtat Ali": 3,
    "Shaishe": 4,
    "Sokari": 5,
    "Sugaey": 6
}

reverse_label_mapping = {v: k for k, v in label_mapping.items()}

def load_and_process_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    return img_array

image = load_and_process_image(image_path)

image_expanded = np.expand_dims(image, axis=0)

y_pred = model.predict(image_expanded)

predicted_label = np.argmax(y_pred, axis=1)

predicted_label_name = reverse_label_mapping[predicted_label[0]]

plt.imshow(image)
plt.title(f"Predicted Label: {predicted_label_name}")
plt.axis('off')
plt.show()

print(f"Image: {image_path} --> Predicted Label: {predicted_label_name}")


