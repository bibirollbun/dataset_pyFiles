import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix


print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)


img_size = 224
batch_size = 32
csv_path = "/kaggle/input/aptos2019-blindness-detection/train.csv"
img_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
save_dir = "/kaggle/working/preprocessed_images" #to save preprocessed images


os.makedirs(save_dir, exist_ok=True)


df = pd.read_csv(csv_path)
df["id_code"] = df["id_code"].apply(lambda x: os.path.join(img_dir, x + ".png"))


def apply_clahe_and_save(image_path, save_dir):
    # Read and resize the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image at path: {image_path}")
    image = cv2.resize(image, (img_size, img_size))
    
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge the LAB channels and convert back to RGB
    merged_lab = cv2.merge((cl, a, b))
    final_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    
    # Save the preprocessed image
    save_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
    
    return final_image / 255.0


print("Preprocessing images and saving to disk...")
for image_path in tqdm(df["id_code"], desc="Processing Images"):  # Add tqdm here
    apply_clahe_and_save(image_path, save_dir)
print("Preprocessing complete!")


def load_preprocessed_image(image_path, label):
    image = tf.io.read_file(image_path)
    
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [img_size, img_size])
    image = tf.cast(image, tf.float32) / 255.0
    
    return image, label


image_paths = [os.path.join(save_dir, os.path.basename(path)) for path in df["id_code"]]
labels = df["diagnosis"].values
dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.shuffle(len(df)).map(load_preprocessed_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def visualize_clahe_effect(df_sample):
    fig, axes = plt.subplots(len(df_sample), 2, figsize=(10, 5 * len(df_sample)))
    
    for i, row in enumerate(df_sample.itertuples()):
        img_path = row.id_code
        original = cv2.imread(img_path, cv2.IMREAD_COLOR)
        original = cv2.resize(original, (img_size, img_size))
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        processed_path = os.path.join(save_dir, os.path.basename(img_path))
        processed = cv2.imread(processed_path, cv2.IMREAD_COLOR)
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        axes[i, 0].imshow(original)
        axes[i, 0].set_title(f"Original - {row.diagnosis}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(processed)
        axes[i, 1].set_title(f"CLAHE Processed - {row.diagnosis}")
        axes[i, 1].axis("off")
    
    plt.show()


df_sample = df.sample(5)
visualize_clahe_effect(df_sample)


def dataset_split(ds, train=0.7, val=0.15, test=0.15):
    ds_size = len(ds)
    train_size = int(ds_size * train)
    val_size = int(ds_size * val)
    
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size + val_size)
    
    return train_ds, val_ds, test_ds


train_ds, val_ds, test_ds = dataset_split(dataset)
train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)


from tensorflow.keras.applications import DenseNet201,ResNet50
from sklearn.utils.class_weight import compute_class_weight


base1 = DenseNet201(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
base2 = ResNet50(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))


base1.trainable = True
base2.trainable = True


for layer in base1.layers[:-10]:  # Unfreeze the last 10 layers of EfficientNetB0
    layer.trainable = False
for layer in base2.layers[:-10]:  # Unfreeze the last 10 layers of ResNet50
    layer.trainable = False


inputs = tf.keras.Input(shape=(img_size, img_size, 3))
x1 = layers.GlobalAveragePooling2D()(base1(inputs))
x2 = layers.GlobalAveragePooling2D()(base2(inputs))
merged = layers.Concatenate()([x1, x2])
x = layers.Dense(256, activation="relu")(merged)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(5, activation="softmax")(x)


multibranch_model_1 = Model(inputs, outputs)
multibranch_model_1.summary()


multibranch_model_1.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)


class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(class_weights))


def lr_scheduler(epoch, lr):
    if epoch > 0 and epoch % 10 == 0:  # Reduce LR every 10 epochs
        return lr * 0.1
    return lr

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)


# لحفظ أفضل Multibranch CNN
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_multibranch = ModelCheckpoint(
    "best_multibranch_model.keras",  # اسم الملف
    monitor="val_accuracy",          # المراقبة على دقة validation
    verbose=1,                       # طباعة التقدم
    save_best_only=True,             # حفظ أفضل موديل فقط
    mode="max"                       # لأننا نريد أعلى قيمة
)


# تدريب الموديل مع حفظ أفضل نسخة
multibranch_history = multibranch_model_1.fit(
    train_ds,
    epochs=50,
    batch_size=batch_size,
    validation_data=val_ds,
    class_weight=class_weights,
    callbacks=[lr_callback,checkpoint_multibranch],
    verbose=1)



multibranch_model_1.save("multibranch_model_1.h5")


test_loss, test_acc = multibranch_model_1.evaluate(test_ds)
print(f"Test Accuracy: {test_acc * 100:.2f}%")


def build_2d_cnn(input_shape=(224, 224, 3), num_classes=5):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Convolutional layers
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    # Build the model
    cnn_model = Model(inputs, outputs)
    return cnn_model


cnn_model = build_2d_cnn(input_shape=(img_size, img_size, 3), num_classes=5)
cnn_model.summary()


cnn_model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)


cnn_history = cnn_model.fit(
    train_ds,
    epochs=40,
    batch_size=batch_size,
    verbose=1,
    class_weight=class_weights,
    validation_data=val_ds,
    callbacks=[lr_callback]
)


cnn_model.save("cnn_model_1.keras")
cnn_model.save("cnn_model_1.h5")


test_loss, test_acc = cnn_model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc * 100:.2f}%")


import matplotlib.pyplot as plt

# ✅ Convert only if still a History object
if hasattr(multibranch_history, 'history'):
    multibranch_history = multibranch_history.history

if hasattr(cnn_history, 'history'):
    cnn_history = cnn_history.history

# ✅ Plot Accuracy and Loss Comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Accuracy ---
axes[0].plot(multibranch_history['accuracy'], label='Multi-Branch CNN Training Accuracy')
axes[0].plot(multibranch_history['val_accuracy'], label='Multi-Branch CNN Validation Accuracy')
axes[0].plot(cnn_history['accuracy'], label='2D CNN Training Accuracy')
axes[0].plot(cnn_history['val_accuracy'], label='2D CNN Validation Accuracy')
axes[0].set_title('Accuracy Comparison')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# --- Loss ---
axes[1].plot(multibranch_history['loss'], label='Multi-Branch CNN Training Loss')
axes[1].plot(multibranch_history['val_loss'], label='Multi-Branch CNN Validation Loss')
axes[1].plot(cnn_history['loss'], label='2D CNN Training Loss')
axes[1].plot(cnn_history['val_loss'], label='2D CNN Validation Loss')
axes[1].set_title('Loss Comparison')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()



multibranch_predictions = multibranch_model_1.predict(test_ds)
cnn_predictions = cnn_model.predict(test_ds)


multibranch_class_labels = np.argmax(multibranch_predictions, axis=1) 
cnn_class_labels = np.argmax(cnn_predictions, axis=1)
true_labels = np.concatenate([y for x, y in test_ds], axis=0)


multibranch_cm = confusion_matrix(true_labels, multibranch_class_labels)
cnn_cm = confusion_matrix(true_labels, cnn_class_labels)


plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.heatmap(multibranch_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Multi-Branch CNN Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

plt.subplot(1, 2, 2)
sns.heatmap(cnn_cm, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title('2D CNN Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

plt.tight_layout()
plt.show()


multibranch_model=tf.keras.models.load_model("/kaggle/working/multibranch_model_1.h5")
cnn_model = tf.keras.models.load_model("/kaggle/working/cnn_model_1.h5")


weight_multibranch = 0.7
weight_cnn = 0.3


ensemble_predictions = (weight_multibranch * multibranch_predictions) + (weight_cnn * cnn_predictions)
ensemble_class_labels = np.argmax(ensemble_predictions, axis=1)
true_labels = np.concatenate([y for x, y in test_ds], axis=0)
ensemble_accuracy = np.mean(ensemble_class_labels == true_labels)
print(f"Ensemble Accuracy: {ensemble_accuracy * 100:.2f}%")


plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.hist(multibranch_class_labels, bins=5, range=(0, 5), alpha=0.7, color='blue')
plt.title('Multi-Branch CNN Predictions')
plt.xlabel('Class')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(cnn_class_labels, bins=5, range=(0, 5), alpha=0.7, color='green')
plt.title('2D CNN Predictions')
plt.xlabel('Class')
plt.ylabel('Frequency')

# Ensemble predictions
plt.subplot(1, 3, 3)
plt.hist(ensemble_class_labels, bins=5, range=(0, 5), alpha=0.7, color='red')
plt.title('Ensemble Predictions')
plt.xlabel('Class')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


from sklearn.metrics import classification_report

# افترضنا إن ensemble_class_labels و true_labels موجودين عندك بالفعل
# لو عندك أسماء الـ classes
class_labels = ["Class1", "Class2", "Class3", "Class4", "Class5"]  # عدّل حسب أسماء الفئات عندك

report = classification_report(true_labels, ensemble_class_labels, target_names=class_labels)
print(report)



from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# أسماء الفئات (عدّل حسب بياناتك)
class_labels = ["Class1", "Class2", "Class3", "Class4", "Class5"]

# --- Multibranch Model ---
multibranch_cm = confusion_matrix(true_labels, multibranch_class_labels)
print("Multibranch Classification Report:\n")
print(classification_report(true_labels, multibranch_class_labels, target_names=class_labels))

plt.figure(figsize=(8,6))
sns.heatmap(multibranch_cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap="Blues")
plt.title("Multibranch Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# --- CNN Model ---
cnn_cm = confusion_matrix(true_labels, cnn_class_labels)
print("CNN Classification Report:\n")
print(classification_report(true_labels, cnn_class_labels, target_names=class_labels))

plt.figure(figsize=(8,6))
sns.heatmap(cnn_cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap="Greens")
plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# --- Ensemble Model ---
ensemble_cm = confusion_matrix(true_labels, ensemble_class_labels)
print("Ensemble Classification Report:\n")
print(classification_report(true_labels, ensemble_class_labels, target_names=class_labels))

plt.figure(figsize=(8,6))
sns.heatmap(ensemble_cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap="Oranges")
plt.title("Ensemble Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()



import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def preprocess_image(image_path, img_size=224):
    """
    Preprocesses the image using CLAHE and resizes it.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image at path: {image_path}")

    image = cv2.resize(image, (img_size, img_size))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    merged_lab = cv2.merge((l_clahe, a, b))
    final_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)

    final_image = final_image / 255.0  # Normalize to [0, 1]
    return final_image


def saliency_map(model, img_array):
    """
    Generates a Saliency Map for a given model and image.
    """
    img_tensor = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        top_pred = tf.argmax(predictions[0])
        loss = predictions[:, top_pred]

    grads = tape.gradient(loss, img_tensor)[0]
    saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()
    return saliency


def predict_with_explanations(image_path, multibranch_model, cnn_model, img_size=224, weight_multibranch=0.7, weight_cnn=0.3):
    """
    Predicts the class probabilities for an image using multi-branch CNN, 2D CNN, and ensemble.
    Also generates Saliency Map explanations for each model.

    Args:
        image_path (str): Path to the input image.
        multibranch_model (tf.keras.Model): Trained multi-branch CNN model.
        cnn_model (tf.keras.Model): Trained 2D CNN model.
        img_size (int): Size of the input image (default: 224).
        weight_multibranch (float): Weight for multi-branch CNN in the ensemble (default: 0.7).
        weight_cnn (float): Weight for 2D CNN in the ensemble (default: 0.3).

    Returns:
        dict: A dictionary containing the predictions, confidence percentages, and Saliency Maps.
    """
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path, img_size)
    img_array = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension

    # Get predictions from multi-branch CNN
    multibranch_probs = multibranch_model.predict(img_array, verbose=0)[0]
    multibranch_class = np.argmax(multibranch_probs)
    multibranch_confidence = float(multibranch_probs[multibranch_class])

    # Get predictions from 2D CNN
    cnn_probs = cnn_model.predict(img_array, verbose=0)[0]
    cnn_class = np.argmax(cnn_probs)
    cnn_confidence = float(cnn_probs[cnn_class])

    # Combine predictions using weighted average for ensemble
    ensemble_probs = (weight_multibranch * multibranch_probs) + (weight_cnn * cnn_probs)
    ensemble_class = np.argmax(ensemble_probs)
    ensemble_confidence = float(ensemble_probs[ensemble_class])

    # Generate Saliency Map for multi-branch CNN
    multibranch_saliency = saliency_map(multibranch_model, img_array)

    # Generate Saliency Map for 2D CNN
    cnn_saliency = saliency_map(cnn_model, img_array)

    # Return results as a dictionary
    results = {
        "multi_branch_cnn": {
            "class": int(multibranch_class),
            "confidence": multibranch_confidence,
            "probabilities": [float(prob) for prob in multibranch_probs],
            "saliency_map": multibranch_saliency,
        },
        "2d_cnn": {
            "class": int(cnn_class),
            "confidence": cnn_confidence,
            "probabilities": [float(prob) for prob in cnn_probs],
            "saliency_map": cnn_saliency,
        },
        "ensemble": {
            "class": int(ensemble_class),
            "confidence": ensemble_confidence,
            "probabilities": [float(prob) for prob in ensemble_probs],
        },
    }
    return results


# Example usage
image_path = "/kaggle/input/aptos2019-blindness-detection/test_images/006efc72b638.png"
results = predict_with_explanations(image_path, multibranch_model_1, cnn_model)

# Print predictions
print("Multi-Branch CNN Predictions:")
print(f"Class: {results['multi_branch_cnn']['class']}")
print(f"Confidence: {results['multi_branch_cnn']['confidence'] * 100:.2f}%")
print(f"Probabilities: {results['multi_branch_cnn']['probabilities']}")

print("\n2D CNN Predictions:")
print(f"Class: {results['2d_cnn']['class']}")
print(f"Confidence: {results['2d_cnn']['confidence'] * 100:.2f}%")
print(f"Probabilities: {results['2d_cnn']['probabilities']}")

print("\nEnsemble Predictions:")
print(f"Class: {results['ensemble']['class']}")
print(f"Confidence: {results['ensemble']['confidence'] * 100:.2f}%")
print(f"Probabilities: {results['ensemble']['probabilities']}")

# Visualize results
original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
visualize_results(results, original_image)


import tensorflow as tf
from tensorflow.keras import layers, Model

# افترض إن الموديلات جاهزة ومحمّلة
multibranch_model = tf.keras.models.load_model("/kaggle/working/multibranch_model_1.h5")
cnn_model = tf.keras.models.load_model("/kaggle/working/cnn_model_1.h5")

# نعمل inputs جديدة
inputs = tf.keras.Input(shape=(224, 224, 3))

# نعمل forward pass لكل موديل
pred1 = multibranch_model(inputs)
pred2 = cnn_model(inputs)

# ندمج النتائج بالـ weights
ensemble_output = layers.Lambda(lambda x: weight_multibranch*x[0] + weight_cnn*x[1])([pred1, pred2])

# نختار الـ class النهائي
ensemble_class = layers.Lambda(lambda x: tf.argmax(x, axis=1))(ensemble_output)

# نبني الموديل
ensemble_model = Model(inputs=inputs, outputs=ensemble_class)

# نحفظ الموديل
ensemble_model.save("ensemble_model.keras")



import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Concatenate
from tensorflow.keras.models import Model

# تحميل الموديلات الأصلية
multibranch_model = tf.keras.models.load_model("/kaggle/working/multibranch_model_1.h5")
cnn_model = tf.keras.models.load_model("/kaggle/working/cnn_model_1.h5")

# تثبيت أوزان ensemble
weight_multibranch = 0.7
weight_cnn = 0.3

# إنشاء مدخل واحد للموديل
inputs = Input(shape=(224, 224, 3))

# إخراج الموديلات
pred1 = multibranch_model(inputs)
pred2 = cnn_model(inputs)

# دمج المخرجات بالوزن
ensemble_output = Lambda(lambda x: weight_multibranch*x[0] + weight_cnn*x[1])([pred1, pred2])

# إنشاء موديل جديد
ensemble_model = Model(inputs, ensemble_output)
ensemble_model.save("ensemble_model_for_tflite.keras")



import tensorflow as tf

class WeightedEnsemble(tf.keras.layers.Layer):
    def __init__(self, weight1=0.7, weight2=0.3, **kwargs):
        super(WeightedEnsemble, self).__init__(**kwargs)
        self.weight1 = weight1
        self.weight2 = weight2

    def call(self, inputs):
        return self.weight1*inputs[0] + self.weight2*inputs[1]

    def get_config(self):
        config = super(WeightedEnsemble, self).get_config()
        config.update({"weight1": self.weight1, "weight2": self.weight2})
        return config

# استخدام الـ Custom Layer
inputs = tf.keras.Input(shape=(224, 224, 3))
pred1 = multibranch_model(inputs)
pred2 = cnn_model(inputs)
ensemble_output = WeightedEnsemble()( [pred1, pred2] )
ensemble_model = tf.keras.Model(inputs, ensemble_output)

# حفظ الموديل
ensemble_model.save("ensemble_model_for_tflite.keras")



import tensorflow as tf

# تعريف الـ Custom Layer
class WeightedEnsemble(tf.keras.layers.Layer):
    def __init__(self, weight1=0.7, weight2=0.3, **kwargs):
        super(WeightedEnsemble, self).__init__(**kwargs)
        self.weight1 = weight1
        self.weight2 = weight2

    def call(self, inputs):
        return self.weight1*inputs[0] + self.weight2*inputs[1]

    def get_config(self):
        config = super(WeightedEnsemble, self).get_config()
        config.update({"weight1": self.weight1, "weight2": self.weight2})
        return config

# إعادة تحميل الموديلات الفرعية
multibranch_model = tf.keras.models.load_model("/kaggle/working/multibranch_model_1.h5")
cnn_model = tf.keras.models.load_model("/kaggle/working/cnn_model_1.h5")

# بناء موديل الـ Ensemble
inputs = tf.keras.Input(shape=(224, 224, 3))
pred1 = multibranch_model(inputs)
pred2 = cnn_model(inputs)
ensemble_output = WeightedEnsemble()( [pred1, pred2] )
ensemble_model = tf.keras.Model(inputs, ensemble_output)

# حفظ الموديل بصيغة .keras
ensemble_model.save("ensemble_model_for_tflite.keras")



# إنشاء Converter
converter = tf.lite.TFLiteConverter.from_keras_model(ensemble_model)

# اختياري: تفعيل تحسينات الحجم والأداء
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# تحويل الموديل
tflite_model = converter.convert()

# حفظ الموديل النهائي
with open("ensemble_model.tflite", "wb") as f:
    f.write(tflite_model)

print("تم حفظ الموديل بصيغة TFLite بنجاح ✅")



import os
import shutil
import pandas as pd

# Paths
train_images_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
csv_path = "/kaggle/input/aptos2019-blindness-detection/train.csv"
sample_dir = "/kaggle/working/Sample"

# Read labels CSV
df = pd.read_csv(csv_path)

# Create Sample directory
os.makedirs(sample_dir, exist_ok=True)

# Create class folders (0 to 4)
for cls in range(5):
    os.makedirs(os.path.join(sample_dir, str(cls)), exist_ok=True)

# For each class, copy 10 images
for cls in range(5):
    class_images = df[df["diagnosis"] == cls]["id_code"].values[:10]

    for img_id in class_images:
        src = os.path.join(train_images_dir, img_id + ".png")
        dst = os.path.join(sample_dir, str(cls), img_id + ".png")

        if os.path.exists(src):
            shutil.copy(src, dst)

print("✅ Sample folder created successfully!")



import shutil
import os

sample_dir = "/kaggle/working/Sample"
zip_path = "/kaggle/working/Sample.zip"

# Remove zip if exists
if os.path.exists(zip_path):
    os.remove(zip_path)

# Create zip
shutil.make_archive(
    base_name=zip_path.replace(".zip", ""),
    format="zip",
    root_dir=sample_dir
)

print("✅ Sample.zip created successfully!")



import os
import cv2
import numpy as np
from tqdm import tqdm

# Image size
img_size = 224

# Paths
input_root = "/kaggle/working/Sample"
output_root = "/kaggle/working/Sample_Preprocessed"

os.makedirs(output_root, exist_ok=True)

def apply_clahe_and_save(image_path, save_dir):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image at path: {image_path}")

    # Resize
    image = cv2.resize(image, (img_size, img_size))

    # BGR → LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE on L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge + LAB → RGB
    merged_lab = cv2.merge((cl, a, b))
    final_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)

    # Save
    save_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

    return final_image / 255.0


# Loop over classes
for cls in range(5):
    class_input_dir = os.path.join(input_root, str(cls))
    class_output_dir = os.path.join(output_root, str(cls))
    os.makedirs(class_output_dir, exist_ok=True)

    images = os.listdir(class_input_dir)

    for img_name in tqdm(images, desc=f"Processing class {cls}"):
        img_path = os.path.join(class_input_dir, img_name)
        apply_clahe_and_save(img_path, class_output_dir)

print("✅ Preprocessing completed successfully!")


import shutil
import os

sample_dir = "/kaggle/working/Sample_Preprocessed"
zip_path = "/kaggle/working/Sample_Preprocessed.zip"

# Remove zip if exists
if os.path.exists(zip_path):
    os.remove(zip_path)

# Create zip
shutil.make_archive(
    base_name=zip_path.replace(".zip", ""),
    format="zip",
    root_dir=sample_dir
)

print("✅ Sample.zip created successfully!")


from keras import config
config.enable_unsafe_deserialization()

import tensorflow as tf

model_path = "/kaggle/working/multibranch_model_1.h5"
model = tf.keras.models.load_model(model_path, compile=False)

print("Model loaded successfully!")
model.summary()



import tensorflow as tf
import cv2
import os
import numpy as np
from tqdm import tqdm

img_size = 224
batch_size = 32

sample_dir = "/kaggle/working/Sample_Preprocessed"
def apply_clahe_to_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    
    image = cv2.resize(image, (img_size, img_size))
    
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    merged_lab = cv2.merge((cl, a, b))
    final_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    
    return final_image / 255.0
image_paths = []
labels = []

for class_label in os.listdir(sample_dir):
    class_path = os.path.join(sample_dir, class_label)
    if os.path.isdir(class_path):
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image_paths.append(img_path)
            labels.append(int(class_label))

print(f"Total images: {len(image_paths)}")



def load_preprocessed_image(image_path, label):
    image = apply_clahe_to_image(image_path)
    return image, label

# استخدم tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

def tf_preprocess(path, label):
    image = tf.py_function(func=lambda p: apply_clahe_to_image(p.numpy().decode()), 
                           inp=[path], 
                           Tout=tf.float32)
    image.set_shape([img_size, img_size, 3])
    return image, label

dataset = dataset.shuffle(len(image_paths))
dataset = dataset.map(tf_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)



preds = model.predict(dataset)
pred_classes = np.argmax(preds, axis=1)

print("Predictions done!")
print(pred_classes[:10])



import matplotlib.pyplot as plt
import math
import cv2

# عدد الصور في كل صف
cols = 5

# لو dataset كبير — نقدر نعرض أول 50 صورة فقط مثلاً
num_images_to_show = len(image_paths)  # ممكن تغيّر الرقم
rows = math.ceil(num_images_to_show / cols)

plt.figure(figsize=(20, 4*rows))

for i, img_path in enumerate(image_paths[:num_images_to_show]):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pred_label = pred_classes[i]
    
    plt.subplot(rows, cols, i+1)
    plt.imshow(img)
    plt.title(f"Predicted: {pred_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()



import tensorflow as tf

# فعل unsafe deserialization لو فيه Lambda layers
from keras import config
config.enable_unsafe_deserialization()

h5_model_path = "/kaggle/working/multibranch_model_1.h5"
model = tf.keras.models.load_model(h5_model_path, compile=False)

print("Model loaded successfully!")



import tensorflow as tf
from keras import config
config.enable_unsafe_deserialization()

h5_model_path = "/kaggle/working/multibranch_model_1.h5"
model = tf.keras.models.load_model(h5_model_path, compile=False)

tflite_model_path = "/kaggle/working/multibranch_model_1.tflite"

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved at: {tflite_model_path}")



interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TFLite interpreter ready!")


import tensorflow as tf
import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

img_size = 224
sample_dir = "/kaggle/working/Sample_Preprocessed"
tflite_model_path = "/kaggle/working/multibranch_model_1.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    
    image = cv2.resize(image, (img_size, img_size))
    
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    merged_lab = cv2.merge((cl, a, b))
    final_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    
    # normalize
    return final_image.astype(np.float32) / 255.0
image_paths = []
labels = []

for class_label in os.listdir(sample_dir):
    class_path = os.path.join(sample_dir, class_label)
    if os.path.isdir(class_path):
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image_paths.append(img_path)
            labels.append(int(class_label))

print(f"Total images: {len(image_paths)}")
pred_classes = []

for img_path in tqdm(image_paths, desc="Predicting"):
    img = preprocess_image(img_path)
    img = np.expand_dims(img, axis=0)  # إضافة batch dimension
    
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    
    pred = interpreter.get_tensor(output_details[0]['index'])
    pred_class = np.argmax(pred, axis=1)[0]
    pred_classes.append(pred_class)
df_preds = pd.DataFrame({
    "image_path": image_paths,
    "true_label": labels,
    "predicted_class": pred_classes
})

csv_save_path = "/kaggle/working/sample_tflite_predictions.csv"
df_preds.to_csv(csv_save_path, index=False)
print(f"Predictions saved to: {csv_save_path}")



import pandas as pd

df_preds = pd.DataFrame({
    "image_path": image_paths,
    "predicted_class": pred_classes
})

df_preds.to_csv("/kaggle/working/sample_predictions.csv", index=False)
print("Saved predictions to sample_predictions.csv")



import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import config
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

# === تمكين unsafe deserialization لو فيه Lambda layers ===
config.enable_unsafe_deserialization()

# === تحميل الموديل ===
model_path = "/kaggle/working/multibranch_model_1.h5"
model = tf.keras.models.load_model(model_path, compile=False)
print("Model loaded successfully!")

# === إعداد المتغيرات ===
img_size = 224
batch_size = 32
sample_dir = "/kaggle/working/Sample_Preprocessed"

# === دالة preprocessing (CLAHE + normalization) ===
def apply_clahe_to_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    
    image = cv2.resize(image, (img_size, img_size))
    
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    merged_lab = cv2.merge((cl, a, b))
    final_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    
    return final_image / 255.0

# === جمع كل الصور والكلاسات ===
image_paths = []
labels = []

for class_label in os.listdir(sample_dir):
    class_path = os.path.join(sample_dir, class_label)
    if os.path.isdir(class_path):
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image_paths.append(img_path)
            labels.append(int(class_label))

print(f"Total images: {len(image_paths)}")

# === إنشاء dataset مع tf.data ===
dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

def tf_preprocess(path, label):
    image = tf.py_function(func=lambda p: apply_clahe_to_image(p.numpy().decode()), 
                           inp=[path], 
                           Tout=tf.float32)
    image.set_shape([img_size, img_size, 3])
    return image, label

dataset = dataset.map(tf_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# === عمل Predictions ===
preds = model.predict(dataset)
pred_classes = np.argmax(preds, axis=1)
print("Predictions done!")

# === حفظ النتائج في CSV ===
df_preds = pd.DataFrame({
    "image_path": image_paths,
    "true_label": labels,
    "predicted_class": pred_classes
})

csv_save_path = "/kaggle/working/sample_predictions1.csv"
df_preds.to_csv(csv_save_path, index=False)
print(f"Predictions saved to CSV: {csv_save_path}")

# === عرض الصور مع التنبؤات ===
cols = 5
num_images_to_show = len(image_paths)  # يمكن تقليل العدد لو الصور كثيرة
rows = math.ceil(num_images_to_show / cols)

plt.figure(figsize=(20, 4*rows))

for i, img_path in enumerate(image_paths[:num_images_to_show]):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pred_label = pred_classes[i]
    true_label = labels[i]
    
    plt.subplot(rows, cols, i+1)
    plt.imshow(img)
    plt.title(f"True: {true_label} | Pred: {pred_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()



import numpy as np

# تحويل القوائم إلى numpy arrays
true_labels = np.array(labels)
pred_labels = np.array(pred_classes)

# حساب عدد الصور الصحيحة
correct = np.sum(true_labels == pred_labels)
total = len(true_labels)

accuracy = (correct / total) * 100
print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")




