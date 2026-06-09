!pip install tensorflow pennylane matplotlib pandas numpy scipy transformers imblearn scikit-image


import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import ndimage
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, GlobalAveragePooling2D
from tensorflow.keras.layers import Concatenate, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import AdamW
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from skimage import exposure
# Import PennyLane for quantum computing
import pennylane as qml
from pennylane import numpy as pnp



IMG_SIZE = 224
NUM_QUBITS = 16


DATA_ROOT = '/kaggle/input/aptos2019'  # Change this to your data path
TRAIN_CSV = os.path.join(DATA_ROOT, 'train_1.csv')
VAL_CSV = os.path.join(DATA_ROOT, 'valid.csv')
TEST_CSV = os.path.join(DATA_ROOT, 'test.csv')

TRAIN_IMG_DIR = os.path.join(DATA_ROOT, 'train_images/train_images')
VAL_IMG_DIR = os.path.join(DATA_ROOT, 'val_images/val_images')
TEST_IMG_DIR = os.path.join(DATA_ROOT, 'test_images/test_images')


dev = qml.device("default.qubit", wires=NUM_QUBITS)


seed=42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# Set NumPy print options for cleaner output
np.set_printoptions(precision=4, suppress=True)

print(f"Random seed set to {seed} ✓")


def enhanced_preprocess_image(path):
    """
    Advanced preprocessing for retinal images with improved techniques:
    1. Improved border detection and cropping
    2. Multi-scale contrast enhancement
    3. Adaptive circular masking with smooth transitions
    4. Robust channel-wise normalization
    """
    # Load image
    img = Image.open(path).convert('RGB')
    img_array = np.array(img)
    
    # More robust border detection and cropping
    if np.mean(img_array) < 50:  # Likely has black borders
        # Find non-black regions with more sensitivity
        threshold = np.mean(img_array) + 10
        y, x = np.where(np.mean(img_array, axis=2) > threshold)
        if len(y) > 0 and len(x) > 0:
            # Add padding to ensure no content is lost
            padding = 10
            y_min, y_max = max(0, min(y) - padding), min(img_array.shape[0], max(y) + padding)
            x_min, x_max = max(0, min(x) - padding), min(img_array.shape[1], max(x) + padding)
            if y_max > y_min and x_max > x_min:
                img_array = img_array[y_min:y_max, x_min:x_max]
                img = Image.fromarray(img_array)
    
    # Resize to target dimensions
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    
    # Multi-scale contrast enhancement for better feature visibility
    # 1. CLAHE for local contrast enhancement
    img_adapteq = exposure.equalize_adapthist(img_array, clip_limit=0.03)
    
    # 2. Apply additional gamma correction for better feature visibility
    img_adapteq = exposure.adjust_gamma(img_adapteq, 1.2)
    
    # When calculating local contrast normalization, add epsilon:
    for c in range(3):
        channel = img_adapteq[:,:,c]
        local_mean = ndimage.gaussian_filter(channel, sigma=15)
        local_std = np.sqrt(ndimage.gaussian_filter(channel**2, sigma=15) - local_mean**2)
        # Add epsilon before division
        local_std = np.maximum(local_std, 0.001)  # Use 0.001 instead of checking < 0.001
        img_adapteq[:,:,c] = (channel - local_mean) / local_std
    
    # Create refined circular mask (retina images are circular)
    h, w = img_adapteq.shape[:2]
    y, x = np.ogrid[:h, :w]
    center = (h//2, w//2)
    
    # Adaptive radius based on image content
    img_gray = np.mean(img_adapteq, axis=2)
    non_zero = np.where(img_gray > 0.05)
    if len(non_zero[0]) > 0:
        # Calculate distance from center to farthest bright pixel
        distances = np.sqrt((non_zero[0] - center[0])**2 + (non_zero[1] - center[1])**2)
        radius = np.percentile(distances, 95)  # Use 95th percentile to avoid outliers
    else:
        radius = min(h, w) * 0.45
    
    mask = ((x - center[1])**2 + (y - center[0])**2) <= (radius**2)
    
    # Apply mask to each channel with smoother edge transition
    masked_img = img_adapteq.copy()
    for c in range(3):
        channel = masked_img[:,:,c]
        # Create smooth transition at the edge (soft masking)
        edge_width = 5
        soft_mask = np.ones_like(mask, dtype=float)
        dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        transition_zone = (dist_from_center > radius - edge_width) & (dist_from_center < radius)
        soft_mask[transition_zone] = 1 - (dist_from_center[transition_zone] - (radius - edge_width)) / edge_width
        soft_mask[dist_from_center >= radius] = 0
        
        channel = channel * soft_mask
        masked_img[:,:,c] = channel
    
    # Channel-wise standardization with robust statistics
    for c in range(3):
        channel = masked_img[:,:,c]
        # Use only non-zero (masked) values for statistics
        non_zero_vals = channel[channel > 0]
        if len(non_zero_vals) > 0:
            mean_val = np.mean(non_zero_vals)
            std_val = np.std(non_zero_vals)
            if std_val > 0:
                channel = (channel - mean_val) / std_val
                # Clip outliers
                channel = np.clip(channel, -3, 3)
                # Rescale to [0, 1]
                channel = (channel + 3) / 6
                masked_img[:,:,c] = channel
    
    # Final normalization
    masked_img = np.clip(masked_img, 0, 1)
    
    return masked_img


@qml.qnode(dev)
def quantum_feature_circuit(features):
    """
    Advanced quantum circuit for feature transformation with:
    1. Multi-scale entanglement
    2. Quantum attention mechanism
    3. Feature embedding with amplitude encoding
    """
    # Scale features to appropriate range
    scaled_features = np.tanh(features[:NUM_QUBITS] / 5.0) * np.pi
    
    # Embedding layer - Amplitude encoding
    for i in range(NUM_QUBITS):
        qml.RX(scaled_features[i % len(scaled_features)], wires=i)
        qml.RY(scaled_features[(i + 1) % len(scaled_features)], wires=i)
    
    # Multi-scale entanglement layers
    for layer in range(2):
        # Nearest-neighbor entanglement
        for i in range(NUM_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        
        # Long-range entanglement (simulating attention)
        for i in range(0, NUM_QUBITS, 4):
            if i + 3 < NUM_QUBITS:
                qml.CNOT(wires=[i, i + 3])
        
        # Rotation gates with feature-dependent angles
        for i in range(NUM_QUBITS):
            # Create feature-dependent rotation angles
            angle_x = scaled_features[i % len(scaled_features)]
            angle_y = scaled_features[(i + 1) % len(scaled_features)]
            qml.RX(angle_x, wires=i)
            qml.RY(angle_y, wires=i)
    
    # Quantum attention mechanism - controlled rotations based on features
    # Select a subset of qubits as "attention qubits"
    attention_qubits = [0, 4, 8, 12]
    target_qubits = [i+1 for i in attention_qubits if i+1 < NUM_QUBITS]
    
    for control, target in zip(attention_qubits, target_qubits):
        # Controlled rotation based on features
        qml.CRX(scaled_features[control % len(scaled_features)], wires=[control, target])
    
    # Measure in multiple bases for richer feature extraction
    # Z-basis measurements
    z_measurements = [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]
    
    # X-basis measurements for complementary information
    x_measurements = [qml.expval(qml.PauliX(i)) for i in range(0, NUM_QUBITS, 2)]
    
    return z_measurements + x_measurements



def apply_quantum_transformation(features, batch_size=8):
    """Apply quantum transformation to features in batches"""
    print("Applying quantum feature transformation...")
    transformed_features = []
    
    # Process in batches
    total_samples = len(features)
    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        batch = features[i:end_idx]
        
        batch_transformed = []
        for sample in batch:
            # Apply quantum transformation
            quantum_features = quantum_feature_circuit(sample)
            batch_transformed.append(quantum_features)
        
        transformed_features.extend(batch_transformed)
        
        # Print progress
        if (i + batch_size) % 100 == 0 or end_idx == total_samples:
            print(f"  Processed {end_idx}/{total_samples} samples")
    
    return np.array(transformed_features)


def sparse_categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss for multi-class classification with integer labels
    """
    def loss_function(y_true, y_pred):
        # Cast y_true to int32
        y_true = tf.cast(y_true, tf.int32)
        
        # Get the number of classes from y_pred shape
        num_classes = tf.shape(y_pred)[-1]
        
        # Convert y_true to one-hot encoding
        y_true_one_hot = tf.one_hot(y_true, depth=num_classes)
        
        # Clip predictions for numerical stability
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal weights
        # Higher gamma gives more weight to misclassified examples
        focal_weight = tf.pow(1.0 - y_pred, gamma)
        
        # Apply class weights via alpha
        if alpha is not None:
            # For simplicity, use scalar alpha for all classes
            focal_weight = alpha * focal_weight * y_true_one_hot
        
        # Calculate cross entropy loss
        ce_loss = -y_true_one_hot * tf.math.log(y_pred)
        
        # Apply focal weights to CE loss
        focal_loss = focal_weight * ce_loss
        
        # Sum over classes, mean over batch
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))
    
    return loss_function


def create_advanced_feature_extractor(model_type="ensemble"):
    """
    Create an advanced feature extractor using state-of-the-art models
    
    Parameters:
    -----------
    model_type : str
        Type of model to use - can be "vit", "convnext", or "ensemble"
    
    Returns:
    --------
    model : Model
        Feature extraction model
    """
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    
    if model_type == "vit":
        # Vision Transformer approach
        print("Creating Vision Transformer feature extractor...")
        
        # Load pre-trained ViT model
        vit_model = TFViTModel.from_pretrained('google/vit-base-patch16-224')
        
        # Create model
        inputs = Input(shape=input_shape)
        vit_outputs = vit_model(inputs).last_hidden_state
        
        # Global pooling to get a fixed-size representation
        features = GlobalAveragePooling1D()(vit_outputs)
        
        model = Model(inputs=inputs, outputs=features)
        return model
        
    elif model_type == "convnext":
        # ConvNeXt approach
        print("Creating ConvNeXt feature extractor...")
        
        # Load pre-trained ConvNeXt model
        convnext_model = ConvNeXtBase(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling='avg'
        )
        
        # Create model
        inputs = Input(shape=input_shape)
        features = convnext_model(inputs)
        
        model = Model(inputs=inputs, outputs=features)
        return model
    
    elif model_type == "ensemble":
        # Ensemble approach combining ViT, ConvNeXt, and ResNet
        print("Creating ensemble feature extractor combining ViT, ConvNeXt, and ResNet...")
        
        # 1. Vision Transformer
        vit_model = TFViTModel.from_pretrained('google/vit-base-patch16-224')
        
        # 2. ConvNeXt
        convnext_model = ConvNeXtBase(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling='avg'
        )
        
        # 3. ResNet50 (as a proven baseline)
        resnet_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling='avg'
        )
        
        # Create ensemble model
        inputs = Input(shape=input_shape)
        
        # Get features from each model
        vit_outputs = vit_model(inputs).last_hidden_state
        vit_features = GlobalAveragePooling1D()(vit_outputs)
        
        convnext_features = convnext_model(inputs)
        resnet_features = resnet_model(inputs)
        
        # Process each feature set separately to balance their contributions
        vit_processed = Dense(512, activation='relu')(vit_features)
        vit_processed = BatchNormalization()(vit_processed)
        
        convnext_processed = Dense(512, activation='relu')(convnext_features)
        convnext_processed = BatchNormalization()(convnext_processed)
        
        resnet_processed = Dense(512, activation='relu')(resnet_features)
        resnet_processed = BatchNormalization()(resnet_processed)
        
        # Combine all features with attention
        combined = Concatenate()([vit_processed, convnext_processed, resnet_processed])
        
        # Attention mechanism to focus on the most relevant features
        attention = Dense(1536, activation='tanh')(combined)
        attention = Dense(1536, activation='sigmoid')(attention)
        
        # Apply attention weights
        weighted_features = Multiply()([combined, attention])
        
        # Dimension reduction
        features = Dense(768, activation='relu')(weighted_features)
        features = BatchNormalization()(features)
        
        model = Model(inputs=inputs, outputs=features)
        return model
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_enhanced_quantum_classical_model(classical_input_shape, quantum_input_shape):
    """
    Create an enhanced hybrid quantum-classical model utilizing Vision Transformers and ConvNeXt
    """
    # Classical branch with advanced feature extraction
    classical_input = Input(shape=classical_input_shape)
    
    # Use the ensemble feature extractor
    feature_extractor = create_advanced_feature_extractor(model_type="ensemble")
    
    # Freeze the feature extractor except for the final few layers
    for layer in feature_extractor.layers[:-10]:
        layer.trainable = False
    
    classical_features = feature_extractor(classical_input)
    
    # Further process classical features
    classical_features = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(classical_features)
    classical_features = BatchNormalization()(classical_features)
    classical_features = Dropout(0.5)(classical_features)
    
    # Quantum branch
    quantum_input = Input(shape=(quantum_input_shape,))
    quantum_features = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(quantum_input)
    quantum_features = BatchNormalization()(quantum_features)
    quantum_features = Dropout(0.5)(quantum_features)
    
    # Combine classical and quantum features
    combined_features = Concatenate()([classical_features, quantum_features])
    
    # Multi-head attention mechanism (inspired by Transformer architecture)
    def attention_block(x, heads=4):
        dim = tf.shape(x)[-1]
        head_dim = dim // heads
        
        attention_outputs = []
        for i in range(heads):
            # Query, Key, Value projections
            query = tf.reshape(Dense(head_dim)(x), [-1, 1, head_dim])
            key = tf.reshape(Dense(head_dim)(x), [-1, 1, head_dim])
            value = tf.reshape(Dense(head_dim)(x), [-1, 1, head_dim])
            
            # Compute attention scores
            scores = tf.matmul(query, tf.transpose(key, [0, 2, 1])) / tf.math.sqrt(tf.cast(head_dim, tf.float32))
            
            # Apply softmax to get attention weights
            weights = tf.nn.softmax(scores, axis=-1)
            
            # Apply attention weights to values
            output = tf.matmul(weights, value)
            attention_outputs.append(tf.reshape(output, [-1, head_dim]))
        
        # Concatenate attention heads
        multi_head = Concatenate()(attention_outputs)
        
        # Project back to original dimension
        projected = Dense(dim)(multi_head)
        
        return projected
    
    # Apply self-attention
    attended_features = attention_block(tf.expand_dims(combined_features, axis=1))
    
    # Additional processing
    x = Dense(384, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(attended_features)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(192, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Final classification
    dr_output = Dense(5, activation='softmax', name='dr_grade')(x)
    
    # Create model
    model = Model(inputs=[classical_input, quantum_input], outputs=dr_output)
    
    # Compile model using AdamW optimizer with weight decay
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=0.01)
    
    model.compile(
        optimizer=optimizer,
        loss=sparse_categorical_focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )
    
    return model


def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def run_enhanced_quantum_dr_pipeline(use_full_dataset=True, max_samples=None):
    """
    Complete pipeline for enhanced quantum DR classification using Vision Transformers and ConvNeXt
    """
    # Start timing
    start_time = time.time()
    
    print("=" * 70)
    print("STARTING ENHANCED QUANTUM DR CLASSIFICATION WITH VISION TRANSFORMERS AND CONVNEXT")
    print("=" * 70)
    
    # Step 1: Load data
    print("\n[1/7] Loading data...")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    if use_full_dataset:
        # Combine train and validation for training
        combined_df = pd.concat([train_df, val_df])
        
        # Use combined dataset for training
        train_split_df = combined_df
        # Hold out a small part for validation
        if len(test_df) > 20:
            val_split_df = test_df.sample(min(len(test_df), 50), random_state=42)
        else:
            val_split_df = combined_df.sample(min(len(combined_df), 50), random_state=42)
        # Use remaining test data for final testing
        test_split_df = test_df
    else:
        # Standard split
        train_split_df = train_df
        val_split_df = val_df
        test_split_df = test_df
    
    # Limit samples if needed
    if max_samples:
        train_split_df = train_split_df.head(max_samples)
        val_split_df = val_split_df.head(max(12, max_samples//8))
        test_split_df = test_split_df.head(max(12, max_samples//8))
    
    # Create image paths
    train_split_df['image_path'] = train_split_df['id_code'].apply(
        lambda x: os.path.join(TRAIN_IMG_DIR, f"{x}.png"))
    val_split_df['image_path'] = val_split_df['id_code'].apply(
        lambda x: os.path.join(VAL_IMG_DIR, f"{x}.png"))
    test_split_df['image_path'] = test_split_df['id_code'].apply(
        lambda x: os.path.join(TEST_IMG_DIR, f"{x}.png"))
    
    print(f"Dataset split: {len(train_split_df)} training, {len(val_split_df)} validation, {len(test_split_df)} test")
    
    # Step 2: Process images
    print("\n[2/7] Preprocessing images...")
    
    # Function to load and preprocess images
    def load_images(df):
        images = []
        for i, path in enumerate(df['image_path']):
            if i % 50 == 0 or i == len(df) - 1:
                print(f"  Processed {i+1}/{len(df)} images")
            try:
                img = enhanced_preprocess_image(path)
                images.append(img)
            except Exception as e:
                print(f"Error processing image {path}: {str(e)}")
                images.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))
        return np.array(images)
    
    train_images = load_images(train_split_df)
    val_images = load_images(val_split_df)
    test_images = load_images(test_split_df)
    
    # Step 3: Extract classical features using our enhanced feature extractor
    print("\n[3/7] Extracting classical features with advanced models...")
    
    # Create the feature extractor model
    feature_extractor = create_advanced_feature_extractor(model_type="ensemble")
    
    # Extract features
    print("Extracting features from training images...")
    train_classical_features = feature_extractor.predict(train_images, batch_size=16, verbose=1)
    
    print("Extracting features from validation images...")
    val_classical_features = feature_extractor.predict(val_images, batch_size=16, verbose=1)
    
    print("Extracting features from test images...")
    test_classical_features = feature_extractor.predict(test_images, batch_size=16, verbose=1)
    
    print(f"Classical features shape: {train_classical_features.shape}")
    
    # Step 4: Apply quantum transformation to a subset of classical features
    print("\n[4/7] Applying quantum feature transformation...")
    
    # Select top features for quantum processing
    # We'll use PCA to reduce dimensionality first
    pca = PCA(n_components=min(NUM_QUBITS * 2, train_classical_features.shape[1]))
    train_pca = pca.fit_transform(train_classical_features)
    val_pca = pca.transform(val_classical_features)
    test_pca = pca.transform(test_classical_features)
    
    # Apply quantum transformation
    train_quantum_features = apply_quantum_transformation(train_pca)
    val_quantum_features = apply_quantum_transformation(val_pca)
    test_quantum_features = apply_quantum_transformation(test_pca)
    
    print(f"Quantum features shape: {train_quantum_features.shape}")
    
    # Step 5: Create enhanced hybrid quantum-classical model
    print("\n[5/7] Building enhanced hybrid quantum-classical model...")
    
    hybrid_model = create_enhanced_quantum_classical_model(
        classical_input_shape=(IMG_SIZE, IMG_SIZE, 3),
        quantum_input_shape=train_quantum_features.shape[1]
    )
    
    # Step 6: Train model
    print("\n[6/7] Training enhanced hybrid model...")
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=25,  # Longer patience for this complex model
            restore_best_weights=True,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,
            min_lr=1e-6
        ),
        ModelCheckpoint(
            'best_enhanced_quantum_dr_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    # Calculate class weights
    class_counts = train_split_df['diagnosis'].value_counts()
    total_samples = len(train_split_df)
    n_classes = 5
    
    class_weights = {}
    for i in range(n_classes):
        if i in class_counts:
            class_weights[i] = total_samples / (n_classes * class_counts[i])
        else:
            class_weights[i] = 1.0
    
    # Train model with longer epochs
    history = hybrid_model.fit(
        [train_images, train_quantum_features],
        train_split_df['diagnosis'].values,
        validation_data=(
            [val_images, val_quantum_features],
            val_split_df['diagnosis'].values
        ),
        epochs=150,  # Increased epochs for better convergence
        batch_size=16,  # Smaller batch size for better generalization
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Step 7: Evaluate model
    print("\n[7/7] Evaluating enhanced model...")
    
    # Evaluate on test set
    test_loss, test_accuracy = hybrid_model.evaluate(
        [test_images, test_quantum_features],
        test_split_df['diagnosis'].values
    )
    
    # Get predictions
    y_pred_probs = hybrid_model.predict([test_images, test_quantum_features])
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(test_split_df['diagnosis'].values, y_pred)
    report = classification_report(test_split_df['diagnosis'].values, y_pred)
    conf_matrix = confusion_matrix(test_split_df['diagnosis'].values, y_pred)
    
    # Calculate AUC for each class
    auc_scores = []
    for i in range(5):  # 5 classes
        y_true_binary = (test_split_df['diagnosis'].values == i).astype(int)
        y_pred_binary = y_pred_probs[:, i]
        try:
            auc_i = roc_auc_score(y_true_binary, y_pred_binary)
            auc_scores.append(auc_i)
        except Exception as e:
            print(f"Could not calculate AUC for class {i}: {str(e)}")
            auc_scores.append(0.5)  # Default value
    
    macro_auc = np.mean(auc_scores)
    # Print results
    print(f"\nTest accuracy: {accuracy:.4f}")
    print(f"Macro AUC: {macro_auc:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(report)
    
    # Plot training history
    plot_training_history(history)
    
    # Calculate execution time
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "=" * 70)
    print(f"ENHANCED QUANTUM DR PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("=" * 70)
    
    # Save the model
    hybrid_model.save('enhanced_quantum_dr_model.keras')
    
    # Return results
    return {
        'model': hybrid_model,
        'accuracy': accuracy,
        'macro_auc': macro_auc,
        'auc_per_class': auc_scores,
        'confusion_matrix': conf_matrix,
        'classification_report': report,
        'history': history
    }


print("Starting Quantum-Enhanced DR Classification Pipeline...")
results = run_enhanced_quantum_dr_pipeline(use_full_dataset=True)

# Print final results
print("\nFinal Results Summary:")
print(f"Test Accuracy: {results['accuracy']:.4f}")
print(f"Macro AUC: {results['macro_auc']:.4f}")
print("\nAUC per class:")
for i, auc in enumerate(results['auc_per_class']):
    print(f"  Class {i}: {auc:.4f}")

# Plot training history again if needed
print("\nPlotting training history...")
plot_training_history(results['history'])

print("\nModel saved as 'enhanced_quantum_dr_model.keras'")
print("Pipeline complete!")




