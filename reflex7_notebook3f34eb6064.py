!pip install pennylane


# =============================================================================
# 1. Import Libraries and Define Global Variables
# =============================================================================
import os
import cv2
import csv
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ExifTags
import pennylane as qml

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Directories for training data and test set
TRAIN_DIRS = {
    "Cover": "data/Cover",         # Clean images
    "JMiPOD": "data/JMiPOD",       # Stego images via JMiPOD
    "JUNIWARD": "data/JUNIWARD",   # Stego images via JUNIWARD
    "UERD": "data/UERD"            # Stego images via UERD
}
TEST_DIR = "data/Test"             # Test images for prediction


# =============================================================================
# 2. Feature Extraction Functions
# =============================================================================
def extract_features(image_path):
    """
    Extract features from an image:
      - Pixel values (normalized grayscale image).
      - Entropy computed from the intensity histogram.
      - Frequency components using Discrete Cosine Transform (DCT).
      - Metadata extraction via PIL (if available).
    """
    # Read image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or unable to load: {image_path}")
    # Convert to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # --- Pixel Values ---
    pixel_values = image_gray.flatten() / 255.0

    # --- Entropy Calculation ---
    hist, _ = np.histogram(image_gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    
    # --- Frequency Components using DCT ---
    dct = cv2.dct(np.float32(image_gray))
    
    # --- Metadata Extraction ---
    try:
        pil_img = Image.open(image_path)
        exif_data = pil_img._getexif()
        metadata = {}
        if exif_data is not None:
            for tag, value in exif_data.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                metadata[decoded] = value
        else:
            metadata = {}
    except Exception as e:
        metadata = {}
    
    return {"pixel_values": pixel_values, "entropy": entropy, "dct": dct, "metadata": metadata}




# =============================================================================
# 3. Quantum-Inspired CNN for Stego Detection (Binary Classification)
# =============================================================================
def build_quantum_inspired_cnn(input_shape):
    """
    Builds a CNN model (using Keras) that mimics quantum-inspired feature extraction.
    This model is trained to perform binary classification:
      - 0: Cover (Clean)
      - 1: Stego (hidden message embedded)
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Output: probability of stego
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model




# =============================================================================
# 4. Variational Quantum Classifier (VQC) for Steganography Algorithm Classification
# =============================================================================
def create_vqc(num_qubits=4, num_layers=2):
    """
    Creates a Variational Quantum Classifier using Pennylane.
    This circuit encodes a classical 4-dimensional input into a quantum state,
    applies variational rotations (with RX) and entangling CNOTs, and finally
    measures expectation values which are passed to a classical dense layer.
    
    The VQC outputs a probability distribution over the three steganography algorithms:
      - 0: JMiPOD
      - 1: JUNIWARD
      - 2: UERD
    """
    dev = qml.device("default.qubit", wires=num_qubits)
    
    @qml.qnode(dev, interface='tf')
    def circuit(inputs, weights):
        # Data encoding: use RY rotations
        for i in range(num_qubits):
            qml.RY(inputs[i], wires=i)
        # Variational layers: apply RX and entangle using CNOT in a ring
        for layer in range(num_layers):
            for i in range(num_qubits):
                qml.RX(weights[layer, i], wires=i)
            for i in range(num_qubits):
                qml.CNOT(wires=[i, (i + 1) % num_qubits])
        # Return expectation values for each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
    
    weight_shapes = {"weights": (num_layers, num_qubits)}
    qlayer = qml.qnn.KerasLayer(circuit, weight_shapes, output_dim=num_qubits)
    
    # Build the full VQC model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_qubits, activation='relu'),
        qlayer,
        tf.keras.layers.Dense(3, activation='softmax')  # Three stego algorithm classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model




# =============================================================================
# 5. Data Loader Functions
# =============================================================================
def load_image(image_path, target_size=(64, 64)):
    """
    Loads an image in grayscale, resizes it to target_size, and normalizes pixel values.
    Returns an image array with shape (target_size[0], target_size[1], 1).
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image_resized = cv2.resize(image, target_size)
    image_resized = image_resized.astype("float32") / 255.0
    return np.expand_dims(image_resized, axis=-1)

def get_file_paths_and_labels():
    """
    Retrieves file paths and labels for training.
    - For the binary stego detection: label 0 for Cover images; label 1 for any stego image.
    - For stego algorithm classification, a separate mapping is provided:
         0: JMiPOD, 1: JUNIWARD, 2: UERD.
    Returns two lists: one for binary detection and one for stego algorithm classification.
    """
    binary_file_paths = []
    binary_labels = []  # 0: Cover, 1: Stego
    
    stego_file_paths = []
    stego_algo_labels = []  # 0: JMiPOD, 1: JUNIWARD, 2: UERD
    
    # Cover images (clean)
    cover_paths = glob.glob(os.path.join(TRAIN_DIRS["Cover"], "*.jpg"))
    for path in cover_paths:
        binary_file_paths.append(path)
        binary_labels.append(0)
    
    # Stego images from each algorithm
    for algo, label in zip(["JMiPOD", "JUNIWARD", "UERD"], [0, 1, 2]):
        algo_paths = glob.glob(os.path.join(TRAIN_DIRS[algo], "*.jpg"))
        for path in algo_paths:
            binary_file_paths.append(path)
            binary_labels.append(1)
            stego_file_paths.append(path)
            stego_algo_labels.append(label)
    
    return (binary_file_paths, binary_labels), (stego_file_paths, stego_algo_labels)




# =============================================================================
# 6. Stego Detector Class: Integrates the Two-Stage Pipeline
# =============================================================================
class StegoDetector:
    def _init_(self):
        """
        Initializes the detection system by building:
          - The quantum-inspired CNN for binary stego detection.
          - The VQC for classifying the steganography algorithm.
        """
        self.cnn_input_shape = (64, 64, 1)
        self.cnn_model = build_quantum_inspired_cnn(self.cnn_input_shape)
        self.vqc_model = create_vqc(num_qubits=4, num_layers=2)
    
    def preprocess_image(self, image_path):
        """
        Loads and preprocesses an image for the CNN.
        """
        return load_image(image_path, target_size=(self.cnn_input_shape[0], self.cnn_input_shape[1]))
    
    def detect_stego(self, image_path):
        """
        Uses the CNN to detect steganography.
          Returns: "Cover" if clean, "Stego" if hidden message is detected.
        """
        img = self.preprocess_image(image_path)
        img = np.expand_dims(img, axis=0)
        pred = self.cnn_model.predict(img)
        # Threshold of 0.5 (can be tuned)
        return "Stego" if pred[0][0] > 0.5 else "Cover"
    
    def classify_stego_algorithm(self, image_path):
        """
        For images detected as stego, use the VQC to classify the algorithm used.
        A feature vector is constructed using entropy and DCT mean from the image.
        Returns one of: "JMiPOD", "JUNIWARD", "UERD".
        """
        features = extract_features(image_path)
        entropy = features["entropy"]
        dct_mean = np.mean(features["dct"])
        # Construct a 4-dimensional feature vector
        input_vector = np.array([entropy, dct_mean, entropy * dct_mean, entropy + dct_mean], dtype="float32")
        input_vector = np.expand_dims(input_vector, axis=0)
        algo_pred = self.vqc_model.predict(input_vector)
        algo_index = np.argmax(algo_pred, axis=1)[0]
        algo_labels = {0: "JMiPOD", 1: "JUNIWARD", 2: "UERD"}
        return algo_labels.get(algo_index, "Unknown")
    
    def run_detection(self, image_path):
        """
        Runs the full detection pipeline:
          1. Detects whether the image is Cover or Stego.
          2. If Stego, classifies the stego algorithm used.
        Prints the final result.
        """
        result = self.detect_stego(image_path)
        if result == "Cover":
            print(f"Image: {os.path.basename(image_path)} --> Classified as: COVER (no hidden message)")
        else:
            algo = self.classify_stego_algorithm(image_path)
            print(f"Image: {os.path.basename(image_path)} --> Classified as: STEGO (Algorithm: {algo})")
    
    def predict_test_set(self, submission_file="sample_submission.csv"):
        """
        Processes all images in the Test/ directory and writes predictions in the required
        CSV format. For each test image:
          - If detected as Cover, assign label 0.
          - If detected as Stego, assign label 1.
        Optionally, you may include the algorithm classification for further analysis.
        """
        test_files = glob.glob(os.path.join(TEST_DIR, "*.jpg"))
        predictions = []
        for path in test_files:
            base_name = os.path.basename(path)
            detection = self.detect_stego(path)
            # For competition, only the binary label is needed:
            label = 0 if detection == "Cover" else 1
            predictions.append((base_name, label))
        
        # Write predictions to CSV in the format of sample_submission.csv
        with open(submission_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["filename", "label"])
            for fname, lbl in predictions:
                writer.writerow([fname, lbl])
        print(f"Submission file saved as {submission_file}")




# =============================================================================
# 7. (Optional) Training Pipeline for the Models
# =============================================================================
def train_models(detector, epochs=5, batch_size=32):
    """
    Trains the CNN for stego detection and the VQC for stego algorithm classification.
    Note: This is a simplified training loop. In practice, you may use Keras generators,
    callbacks, and more sophisticated data augmentation.
    """
    # Retrieve file paths and labels
    (binary_paths, binary_labels), (stego_paths, stego_algo_labels) = get_file_paths_and_labels()
    
    # Prepare training data for the CNN
    X_cnn = []
    for path in binary_paths:
        try:
            img = load_image(path, target_size=(64, 64))
            X_cnn.append(img)
        except Exception as e:
            continue
    X_cnn = np.array(X_cnn)
    y_cnn = np.array(binary_labels[:len(X_cnn)]).reshape(-1, 1)
    
    print(f"Training CNN on {len(X_cnn)} images...")
    detector.cnn_model.fit(X_cnn, y_cnn, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    
    # Prepare training data for the VQC (only stego images)
    X_vqc = []
    y_vqc = []
    for path, label in zip(stego_paths, stego_algo_labels):
        try:
            features = extract_features(path)
            entropy = features["entropy"]
            dct_mean = np.mean(features["dct"])
            input_vector = [entropy, dct_mean, entropy * dct_mean, entropy + dct_mean]
            X_vqc.append(input_vector)
            y_vqc.append(label)
        except Exception as e:
            continue
    X_vqc = np.array(X_vqc, dtype="float32")
    # Convert labels to one-hot encoding for three classes
    y_vqc = tf.keras.utils.to_categorical(y_vqc, num_classes=3)
    
    print(f"Training VQC on {len(X_vqc)} stego images...")
    detector.vqc_model.fit(X_vqc, y_vqc, epochs=epochs, batch_size=batch_size, validation_split=0.1)



# =============================================================================
# 8. Main Execution: Train Models and/or Run Predictions
# =============================================================================
if __name__ == "__main__":
    # Initialize the detection system
    detector = StegoDetector()
    
    # Uncomment the following line to train both models.
    # Ensure that the dataset directories exist and contain images.
    # train_models(detector, epochs=5, batch_size=32)
    
    # Example: Run detection on a single image (for demo purposes)
    test_image = "/kaggle/input/alaska2-image-steganalysis/JUNIWARD/00008.jpg"  # Replace with an actual test image path
    try:
        detector.run_detection(test_image)
    except Exception as e:
        print(f"Error during detection: {e}")
    
    # Uncomment the following line to predict the entire test set and generate a submission CSV.
    # detector.predict_test_set(submission_file="sample_submission.csv")

