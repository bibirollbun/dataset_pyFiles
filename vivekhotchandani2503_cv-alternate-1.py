import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Paths to the dataset
train_dir = "/kaggle/input/state-farm-distracted-driver-detection/imgs/train"
labels_csv = "/kaggle/input/state-farm-distracted-driver-detection/driver_imgs_list.csv"

# Load metadata
labels_df = pd.read_csv(labels_csv)

# Prepare image paths and labels
image_paths = []
labels = []

for class_name in os.listdir(train_dir):
    class_dir = os.path.join(train_dir, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        image_paths.append(img_path)
        labels.append(class_name)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split data into training and validation sets
train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels_encoded, test_size=0.2, random_state=42)





from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential

# Load a pre-trained MobileNetV2 model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Add custom layers
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Preprocess images
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0
    return image

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))

train_dataset = train_dataset.map(lambda x, y: (load_and_preprocess_image(x), y))
val_dataset = val_dataset.map(lambda x, y: (load_and_preprocess_image(x), y))

train_dataset = train_dataset.batch(32)
val_dataset = val_dataset.batch(32)

# Train the model
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)


history


import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import cv2

# Load a pre-trained Faster R-CNN model
detection_model = fasterrcnn_resnet50_fpn(pretrained=True)
detection_model.eval()

# Function to detect objects
def detect_objects(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image).unsqueeze(0)

    with torch.no_grad():
        predictions = detection_model(image_tensor)

    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    return image, boxes, labels, scores


# Original class names (from your dataset)
CLASS_NAMES = {
    0: "Safe driving",
    1: "Texting - right",
    2: "Talking on the phone - right",
    3: "Texting - left",
    4: "Talking on the phone - left",
    5: "Operating the radio",
    6: "Drinking",
    7: "Reaching behind",
    8: "Hair and makeup",
    9: "Talking to passenger"
}


def visualize_results(image_path):
    # Perform classification
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    classification_prediction = model.predict(image_array)
    predicted_class = np.argmax(classification_prediction)
    predicted_class_name = CLASS_NAMES[predicted_class]  # Map to class name

    # Perform object detection
    image, boxes, labels, scores = detect_objects(image_path)

    # Display the image with bounding boxes
    plt.imshow(image)
    ax = plt.gca()
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # Filter out weak detections
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            rect = plt.Rectangle((x1, y1), width, height, fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            # Map COCO class ID to name
            class_name = COCO_CLASSES.get(label, f"Class: {label}")
            ax.text(x1, y1, f'{class_name} ({score:.2f})', color='white', backgroundcolor='red')

    plt.title(f'Classification: {predicted_class_name} (Confidence: {np.max(classification_prediction):.2f})')
    plt.axis('off')
    plt.show()


def real_time_inference():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide a video file path

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform classification
        resized_frame = cv2.resize(frame, (224, 224))
        resized_frame = tf.keras.applications.mobilenet_v2.preprocess_input(resized_frame)
        resized_frame = np.expand_dims(resized_frame, axis=0)
        classification_prediction = model.predict(resized_frame)
        predicted_class = np.argmax(classification_prediction)

        # Perform object detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = F.to_tensor(frame_rgb).unsqueeze(0)
        with torch.no_grad():
            predictions = detection_model(frame_tensor)
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()

        # Draw bounding boxes and classification result
        for box, label in zip(boxes, labels):
            if scores[0] > 0.5:  # Filter out weak detections
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'Class: {label}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f'Classification: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Real-Time Inference', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

# Run real-time inference
real_time_inference()







