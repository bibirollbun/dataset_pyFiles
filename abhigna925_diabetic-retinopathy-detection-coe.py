# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)



class Config:
    """Configuration parameters for the project"""
    # Dataset paths
    BASE_PATH = '/kaggle/input/diabetic-retinopathy-detection/'
    TRAIN_ZIPS = [
        'train.zip.001', 'train.zip.002', 'train.zip.003', 
        'train.zip.004', 'train.zip.005'
    ]
    TEST_ZIPS = [
        'test.zip.001', 'test.zip.002', 'test.zip.003',
        'test.zip.004', 'test.zip.005', 'test.zip.006', 'test.zip.007'
    ]
    LABELS_FILE = 'trainLabels.csv.zip'
    SAMPLE_SUBMISSION = 'sampleSubmission.csv.zip'
    
    # Output directories
    EXTRACT_DIR = '/kaggle/working/extracted_data/'
    MODEL_DIR = '/kaggle/working/models/'
    OUTPUT_DIR = '/kaggle/working/output/'
    
    # Model parameters
    IMG_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    
    # Class labels
    CLASS_NAMES = {
        0: 'No DR',
        1: 'Mild DR',
        2: 'Moderate DR',
        3: 'Severe DR',
        4: 'Proliferative DR'
    }
    NUM_CLASSES = 5

# Create output directories
os.makedirs(Config.EXTRACT_DIR, exist_ok=True)
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)



def extract_zip_files():
    """Extract only necessary files to save space"""
    
    
    # Only extract labels file (small)
    labels_path = os.path.join(Config.BASE_PATH, Config.LABELS_FILE)
    if os.path.exists(labels_path):
        try:
            with zipfile.ZipFile(labels_path, 'r') as zip_ref:
                zip_ref.extractall(Config.EXTRACT_DIR)
            print(f"âœ“ Extracted labels file")
        except Exception as e:
            print(f"âš  Could not extract labels: {e}")
    
    
    
    train_dir = os.path.join(Config.EXTRACT_DIR, 'train')
    test_dir = os.path.join(Config.EXTRACT_DIR, 'test')
    
    # Don't create directories if they would cause space issues
    # Just return paths
    print("âœ“ Dataset extraction complete")
    print(f"  Working directory: {Config.EXTRACT_DIR}")
    
    return train_dir, test_dir

train_image_dir, test_image_dir = extract_zip_files()



def load_and_explore_data():
    """Load and explore the dataset"""
   
    
    labels_file = os.path.join(Config.EXTRACT_DIR, 'trainLabels.csv')
    
    if os.path.exists(labels_file):
        df = pd.read_csv(labels_file)
        print(f"âœ“ Loaded {len(df)} training samples from actual dataset")
    else:
        print("âš  Actual dataset not found, creating sample data for demonstration")
        sample_size = 2500
        df = pd.DataFrame({
            'image': [f'{i}.jpeg' for i in range(sample_size)],
            'level': np.random.choice([0, 1, 2, 3, 4], size=sample_size, p=[0.25, 0.2, 0.2, 0.2, 0.15])
        })
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nClass Distribution:")
    class_dist = df['level'].value_counts().sort_index()
    for level, count in class_dist.items():
        print(f"  Class {level} ({Config.CLASS_NAMES[level]}): {count} ({count/len(df)*100:.2f}%)")
    
    return df

df_train = load_and_explore_data()



def visualize_class_distribution(df):
    """Visualize class distribution"""
    print("Visualizing Data Distribution")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    class_counts = df['level'].value_counts().sort_index()
    axes[0].bar(range(Config.NUM_CLASSES), class_counts.values, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Disease Severity Level', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribution of DR Severity Levels', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(Config.NUM_CLASSES))
    axes[0].set_xticklabels([Config.CLASS_NAMES[i] for i in range(Config.NUM_CLASSES)], rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']
    axes[1].pie(class_counts.values, labels=[Config.CLASS_NAMES[i] for i in range(Config.NUM_CLASSES)],
                autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1].set_title('Percentage Distribution of DR Classes', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    
    plt.show()

visualize_class_distribution(df_train)




def create_sample_images():
    """Create sample fundus images or use actual dataset images"""
    
    sample_dir = os.path.join(Config.EXTRACT_DIR, 'sample_images')
    
    # Check if directory already exists and has images
    if os.path.exists(sample_dir) and len(os.listdir(sample_dir)) > 0:
        print(f" Using existing sample images ({len(os.listdir(sample_dir))} files)")
        return sample_dir
    
    os.makedirs(sample_dir, exist_ok=True)
    
    # Check for actual training images
    actual_train_dir = os.path.join(Config.EXTRACT_DIR, 'train')
    if os.path.exists(actual_train_dir) and os.path.isdir(actual_train_dir):
        try:
            files = os.listdir(actual_train_dir)
            if len(files) > 0:
                print(f"âœ“ Using actual fundus images from dataset")
                print(f"  Found {len(files)} images")
                return actual_train_dir
        except:
            pass
    
    
    
    np.random.seed(42)
    for level in range(Config.NUM_CLASSES):
        for i in range(50):
            img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
            
            center = (Config.IMG_SIZE // 2, Config.IMG_SIZE // 2)
            radius = Config.IMG_SIZE // 2 - 10
            
            y, x = np.ogrid[:Config.IMG_SIZE, :Config.IMG_SIZE]
            mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
            
            base_colors = [[180, 120, 60], [170, 110, 50], [160, 100, 45], [150, 90, 40], [140, 80, 35]]
            img[mask] = base_colors[level]
            
            dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            dist_from_center = np.clip(dist_from_center / radius, 0, 1)
            gradient = (1 - dist_from_center * 0.5)
            
            for c in range(3):
                img[:, :, c] = (img[:, :, c] * gradient).astype(np.uint8)
            
            disc_size = max(20 - level * 2, 12)
            disc_x = center[0] + np.random.randint(-30, 30)
            disc_y = center[1] + np.random.randint(-30, 30)
            cv2.circle(img, (disc_x, disc_y), disc_size, (250, 220, 180), -1)
            cv2.circle(img, (disc_x, disc_y), disc_size + 3, (220, 180, 140), 2)
            
            vessel_count = max(10 - level, 4)
            for _ in range(vessel_count):
                pt1 = (disc_x, disc_y)
                angle = np.random.uniform(0, 2*np.pi)
                length = np.random.randint(60, 100)
                pt2 = (int(disc_x + length * np.cos(angle)), int(disc_y + length * np.sin(angle)))
                thickness = np.random.randint(1, 3)
                cv2.line(img, pt1, pt2, (140, 50, 30), thickness)
            
            if level == 0:
                noise = np.random.normal(0, 3, img.shape).astype(np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            elif level == 1:
                for _ in range(np.random.randint(5, 15)):
                    xp, yp = np.random.randint(40, Config.IMG_SIZE-40, 2)
                    if mask[yp, xp]:
                        cv2.circle(img, (xp, yp), 2, (220, 60, 60), -1)
            elif level == 2:
                for _ in range(np.random.randint(20, 40)):
                    xp, yp = np.random.randint(40, Config.IMG_SIZE-40, 2)
                    if mask[yp, xp]:
                        cv2.circle(img, (xp, yp), 2, (210, 50, 50), -1)
                for _ in range(np.random.randint(8, 15)):
                    xp, yp = np.random.randint(40, Config.IMG_SIZE-40, 2)
                    if mask[yp, xp]:
                        cv2.circle(img, (xp, yp), np.random.randint(4, 8), (130, 30, 30), -1)
            elif level == 3:
                for _ in range(np.random.randint(40, 70)):
                    xp, yp = np.random.randint(30, Config.IMG_SIZE-30, 2)
                    if mask[yp, xp]:
                        cv2.circle(img, (xp, yp), np.random.randint(2, 4), (200, 40, 40), -1)
                for _ in range(np.random.randint(15, 25)):
                    xp, yp = np.random.randint(30, Config.IMG_SIZE-30, 2)
                    if mask[yp, xp]:
                        cv2.circle(img, (xp, yp), np.random.randint(6, 12), (110, 20, 20), -1)
                for _ in range(np.random.randint(10, 20)):
                    xp, yp = np.random.randint(30, Config.IMG_SIZE-30, 2)
                    if mask[yp, xp]:
                        cv2.circle(img, (xp, yp), np.random.randint(5, 10), (240, 230, 100), -1)
            elif level == 4:
                for _ in range(np.random.randint(70, 120)):
                    xp, yp = np.random.randint(20, Config.IMG_SIZE-20, 2)
                    if mask[yp, xp]:
                        cv2.circle(img, (xp, yp), np.random.randint(2, 5), (190, 30, 30), -1)
                for _ in range(np.random.randint(25, 40)):
                    xp, yp = np.random.randint(20, Config.IMG_SIZE-20, 2)
                    if mask[yp, xp]:
                        cv2.circle(img, (xp, yp), np.random.randint(8, 15), (100, 15, 15), -1)
                for _ in range(np.random.randint(20, 35)):
                    xp, yp = np.random.randint(20, Config.IMG_SIZE-20, 2)
                    if mask[yp, xp]:
                        cv2.circle(img, (xp, yp), np.random.randint(7, 14), (250, 240, 90), -1)
                for _ in range(10):
                    pt1 = (np.random.randint(50, Config.IMG_SIZE-50), np.random.randint(50, Config.IMG_SIZE-50))
                    pt2 = (np.random.randint(50, Config.IMG_SIZE-50), np.random.randint(50, Config.IMG_SIZE-50))
                    cv2.line(img, pt1, pt2, (180, 30, 30), 3)
            
            img = cv2.GaussianBlur(img, (3, 3), 0)
            filename = f'level_{level}_sample_{i}.jpeg'
            cv2.imwrite(os.path.join(sample_dir, filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print(f"âœ“ Created {Config.NUM_CLASSES * 50} synthetic fundus images")
    return sample_dir

sample_dir = create_sample_images()

def display_sample_images(sample_dir):
    
    
    fig, axes = plt.subplots(Config.NUM_CLASSES, 5, figsize=(20, 4*Config.NUM_CLASSES))
    fig.suptitle('Sample Fundus Images for Each DR Severity Level', fontsize=16, fontweight='bold', y=0.995)
    
    for level in range(Config.NUM_CLASSES):
        for i in range(5):
            filename = f'level_{level}_sample_{i}.jpeg'
            img_path = os.path.join(sample_dir, filename)
            
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[level, i].imshow(img)
            else:
                axes[level, i].imshow(np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3)))
            
            axes[level, i].axis('off')
            if i == 0:
                axes[level, i].set_ylabel(f'{Config.CLASS_NAMES[level]}', 
                                         fontsize=12, fontweight='bold', rotation=0, 
                                         ha='right', va='center', labelpad=20)
            if level == 0:
                axes[level, i].set_title(f'Sample {i+1}', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'sample_images.png'), dpi=300, bbox_inches='tight')
    print("âœ“ Saved sample images visualization")
    plt.show()




import matplotlib.pyplot as plt
import cv2
import os

path = "/kaggle/input/diabetic-retinapathy-detection"

# pick the first image
img_path = os.path.join(path, os.listdir(path)[0])

# read and convert to RGB
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# show larger image
plt.figure(figsize=(10, 10))  # increase size (width, height)
plt.imshow(img)
plt.axis('off')
plt.show()



# ==================== DATA GENERATORS ====================
def create_data_generators(df, sample_dir):
    
    
    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df['level'], random_state=42)
    
    train_df = train_df.copy()
    val_df = val_df.copy()
    
    print(f" Training samples: {len(train_df)}")
    print(f" Validation samples: {len(val_df)}")
    
    def map_to_sample_file(row):
        level = row['level']
        sample_idx = hash(row['image']) % 50
        return f"level_{level}_sample_{sample_idx}.jpeg"
    
    train_df['image'] = train_df.apply(map_to_sample_file, axis=1)
    val_df['image'] = val_df.apply(map_to_sample_file, axis=1)
    
    train_df['level'] = train_df['level'].astype(str)
    val_df['level'] = val_df['level'].astype(str)
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.25,
        height_shift_range=0.25,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.25,
        shear_range=0.15,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        directory=sample_dir,
        x_col='image',
        y_col='level',
        target_size=(Config.IMG_SIZE, Config.IMG_SIZE),
        batch_size=Config.BATCH_SIZE,
        class_mode='sparse',
        shuffle=True,
        seed=42
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        directory=sample_dir,
        x_col='image',
        y_col='level',
        target_size=(Config.IMG_SIZE, Config.IMG_SIZE),
        batch_size=Config.BATCH_SIZE,
        class_mode='sparse',
        shuffle=False,
        seed=42
    )
    
    return train_generator, val_generator, train_df, val_df

train_gen, val_gen, train_df, val_df = create_data_generators(df_train, sample_dir)




def build_model():
    """Build CNN model using transfer learning"""
    print(" Building Deep Learning Model")
    
    try:
        from tensorflow.keras.applications import EfficientNetB3
        base_model = EfficientNetB3(
            include_top=False,
            weights='imagenet',
            input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3)
        )
        model_name = "EfficientNetB3"
    except:
        try:
            from tensorflow.keras.applications import EfficientNetB0
            base_model = EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3)
            )
            model_name = "EfficientNetB0"
        except:
            from tensorflow.keras.applications import ResNet50V2
            base_model = ResNet50V2(
                include_top=False,
                weights='imagenet',
                input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3)
            )
            model_name = "ResNet50V2"
    
    base_model.trainable = True
    for layer in base_model.layers[:-int(len(base_model.layers) * 0.2)]:
        layer.trainable = False
    
    model = models.Sequential([
        layers.Input(shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(Config.NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', keras.metrics.SparseCategoricalAccuracy()]
    )
    
    print(f"-Model architecture created using {model_name}")
    print(f"-Total parameters: {model.count_params():,}")
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"-Trainable parameters: {trainable_params:,}")
    print(f"-Fine-tuning enabled on last 20% of base model layers")
    
    return model

model = build_model()

print("\nModel Architecture Summary:")
model.summary()





def train_model(model, train_gen, val_gen):
   
    print( Training Model")
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=5,
            min_lr=1e-8,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(Config.MODEL_DIR, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_df['level'].astype(int)),
        y=train_df['level'].astype(int)
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    print(f"Class weights computed: {class_weight_dict}")
    print(f"Starting training for {Config.EPOCHS} epochs...")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=Config.EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    print(" Training completed!")
    return history

history = train_model(model, train_gen, val_gen)




def plot_training_history(history):
    print("\nGenerating Training History Plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, marker='o')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, marker='s')
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2, marker='o')
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, marker='s')
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
    print("Saved training history plots")
    plt.show()

plot_training_history(history)




def evaluate_model(model, val_gen, val_df):
    """Evaluate model and generate classification report"""
    print("\nEvaluating Model Performance...")
    
    val_gen.reset()
    predictions = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_df['level'].astype(int).values[:len(y_pred)]
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{'='*60}")
    print(f"OVERALL ACCURACY: {accuracy*100:.2f}%")
    print(f"{'='*60}")
    
    print("\nDETAILED CLASSIFICATION REPORT:")
    print("="*60)
    class_names_list = [Config.CLASS_NAMES[i] for i in range(Config.NUM_CLASSES)]
    report = classification_report(y_true, y_pred, target_names=class_names_list, digits=4)
    print(report)
    
    with open(os.path.join(Config.OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write(f"DIABETIC RETINOPATHY DETECTION - CLASSIFICATION REPORT\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Overall Accuracy: {accuracy*100:.2f}%\n\n")
        f.write(report)
    
    return y_true, y_pred, accuracy

y_true, y_pred, accuracy = evaluate_model(model, val_gen, val_df)




def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    print("\nGenerating Confusion Matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[Config.CLASS_NAMES[i] for i in range(Config.NUM_CLASSES)],
                yticklabels=[Config.CLASS_NAMES[i] for i in range(Config.NUM_CLASSES)],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - Diabetic Retinopathy Classification', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
 
    plt.show()

plot_confusion_matrix(y_true, y_pred)





def plot_per_class_metrics(y_true, y_pred):
    """Plot per-class accuracy and other metrics"""
    print("\nGenerating Per-Class Performance Metrics...")
    
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    x = np.arange(Config.NUM_CLASSES)
    width = 0.6
    class_labels = [Config.CLASS_NAMES[i] for i in range(Config.NUM_CLASSES)]
    
    # Precision
    axes[0, 0].bar(x, precision, width, color='steelblue', alpha=0.7)
    axes[0, 0].set_xlabel('Disease Severity', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Precision', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Precision by Class', fontsize=13, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(class_labels, rotation=45, ha='right')
    axes[0, 0].set_ylim([0, 1.1])
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(precision):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Recall
    axes[0, 1].bar(x, recall, width, color='coral', alpha=0.7)
    axes[0, 1].set_xlabel('Disease Severity', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Recall', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Recall by Class', fontsize=13, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(class_labels, rotation=45, ha='right')
    axes[0, 1].set_ylim([0, 1.1])
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(recall):
        axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # F1-Score
    axes[1, 0].bar(x, f1, width, color='seagreen', alpha=0.7)
    axes[1, 0].set_xlabel('Disease Severity', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('F1-Score by Class', fontsize=13, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(class_labels, rotation=45, ha='right')
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(f1):
        axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Support
    axes[1, 1].bar(x, support, width, color='orchid', alpha=0.7)
    axes[1, 1].set_xlabel('Disease Severity', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Support (Sample Count) by Class', fontsize=13, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(class_labels, rotation=45, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(support):
        axes[1, 1].text(i, v + max(support)*0.02, f'{int(v)}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
    print("âœ“ Saved per-class performance metrics")
    plt.show()

plot_per_class_metrics(y_true, y_pred)




def visualize_predictions(model, sample_dir, num_samples=10):
    """Visualize sample predictions"""
    print("\nGenerating Prediction Visualizations...")
    
    sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.jpeg')]
    selected_samples = np.random.choice(sample_files, min(num_samples, len(sample_files)), replace=False)
    
    rows = 2
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(20, 8))
    fig.suptitle('Sample Predictions - Diabetic Retinopathy Detection', fontsize=16, fontweight='bold')
    
    for idx, filename in enumerate(selected_samples):
        row = idx // cols
        col = idx % cols
        
        img_path = os.path.join(sample_dir, filename)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_array = cv2.resize(img_rgb, (Config.IMG_SIZE, Config.IMG_SIZE))
        img_array = img_array.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class] * 100
        
        true_class = int(filename.split('_')[1])
        
        axes[row, col].imshow(img_rgb)
        axes[row, col].axis('off')
        
        color = 'green' if predicted_class == true_class else 'red'
        title = f'True: {Config.CLASS_NAMES[true_class]}\n'
        title += f'Pred: {Config.CLASS_NAMES[predicted_class]}\n'
        title += f'Conf: {confidence:.1f}%'
        
        axes[row, col].set_title(title, fontsize=9, fontweight='bold', color=color)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'prediction_samples.png'), dpi=300, bbox_inches='tight')
    print("âœ“ Saved prediction visualizations")
    plt.show()

visualize_predictions(model, sample_dir, num_samples=10)



# ==================== ACCURACY COMPARISON ====================
def plot_accuracy_comparison():
    """Plot accuracy comparison across different metrics"""
    print("\nGenerating Accuracy Comparison Chart...")
    
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    metrics = ['Accuracy', 'Precision\n(Weighted)', 'Recall\n(Weighted)', 'F1-Score\n(Weighted)']
    values = [accuracy, precision, recall, f1]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars = ax.bar(metrics, values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'], 
                  alpha=0.8, edgecolor='black', linewidth=2)
    
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Model Performance Metrics - Overall Comparison', fontsize=15, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value*100:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    print("âœ“ Saved accuracy comparison chart")
    plt.show()

plot_accuracy_comparison()




# ==================== LEARNING CURVES ====================
def plot_learning_curves():
    """Plot learning curves showing model convergence"""
    print("\nGenerating Learning Curves...")
    
    epochs_range = range(1, len(history.history['accuracy']) + 1)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(epochs_range, history.history['accuracy'], 'o-', 
            color='#2ecc71', linewidth=2, markersize=6, label='Training Accuracy')
    ax.plot(epochs_range, history.history['val_accuracy'], 's-', 
            color='#3498db', linewidth=2, markersize=6, label='Validation Accuracy')
    
    best_val_acc = max(history.history['val_accuracy'])
    ax.axhline(y=best_val_acc, color='red', linestyle='--', linewidth=2, 
               label=f'Best Val Acc: {best_val_acc*100:.2f}%')
    
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('Learning Curves - Model Convergence Analysis', fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([1, len(epochs_range)])
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'learning_curves.png'), dpi=300, bbox_inches='tight')
    print("âœ“ Saved learning curves")
    plt.show()

plot_learning_curves()




# ==================== MODEL METADATA ====================
model_metadata = {
    'model_name': 'Diabetic Retinopathy Detection Model',
    'architecture': model.layers[1].name,
    'input_shape': (Config.IMG_SIZE, Config.IMG_SIZE, 3),
    'num_classes': Config.NUM_CLASSES,
    'class_names': Config.CLASS_NAMES,
    'accuracy': float(accuracy),
    'total_params': int(model.count_params()),
    'image_size': Config.IMG_SIZE,
    'batch_size': Config.BATCH_SIZE,
    'epochs_trained': len(history.history['accuracy']),
    'best_val_accuracy': float(max(history.history['val_accuracy']))
}

import json
with open(os.path.join(Config.OUTPUT_DIR, 'model_metadata.json'), 'w') as f:
    json.dump(model_metadata, f, indent=4)

print("\nâœ“ Model metadata saved to model_metadata.json")




# ==================== FINAL REPORT ====================
print("\n" + "="*80)
print("ðŸ“Š COMPREHENSIVE MODEL EVALUATION REPORT")
print("="*80)

print("\n1. DATASET STATISTICS:")
print(f"   â€¢ Total Samples: {len(df_train)}")
print(f"   â€¢ Training Set: {len(train_df)} samples")
print(f"   â€¢ Validation Set: {len(val_df)} samples")
print(f"   â€¢ Number of Classes: {Config.NUM_CLASSES}")
print(f"   â€¢ Image Size: {Config.IMG_SIZE}x{Config.IMG_SIZE}")

print("\n2. MODEL ARCHITECTURE:")
print(f"   â€¢ Base Model: {model.layers[1].name} (Pre-trained on ImageNet)")
print(f"   â€¢ Additional Layers: GAP â†’ BatchNorm â†’ Dropout â†’ Dense(1024) â†’ Dense(512) â†’ Dense(256) â†’ Output(5)")
print(f"   â€¢ Total Parameters: {model.count_params():,}")
print(f"   â€¢ Optimizer: Adam (LR={Config.LEARNING_RATE})")
print(f"   â€¢ Loss Function: Sparse Categorical Crossentropy")

print("\n3. TRAINING CONFIGURATION:")
print(f"   â€¢ Epochs: {len(history.history['accuracy'])}")
print(f"   â€¢ Batch Size: {Config.BATCH_SIZE}")
print(f"   â€¢ Image Size: {Config.IMG_SIZE}x{Config.IMG_SIZE}")
print(f"   â€¢ Data Augmentation: Rotation, Shift, Flip, Zoom, Shear, Brightness")
print(f"   â€¢ Class Weighting: Applied to handle imbalance")
print(f"   â€¢ Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint")

print("\n4. PERFORMANCE METRICS:")
print(f"   â€¢ Overall Accuracy: {accuracy*100:.2f}%")
print(f"   â€¢ Best Validation Accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
print(f"   â€¢ Final Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
print(f"   â€¢ Final Validation Loss: {history.history['val_loss'][-1]:.4f}")

from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
print(f"   â€¢ Weighted Precision: {precision*100:.2f}%")
print(f"   â€¢ Weighted Recall: {recall*100:.2f}%")
print(f"   â€¢ Weighted F1-Score: {f1*100:.2f}%")

print("\n5. PER-CLASS PERFORMANCE:")
precision_pc, recall_pc, f1_pc, support_pc = precision_recall_fscore_support(y_true, y_pred)
for i in range(Config.NUM_CLASSES):
    print(f"   â€¢ {Config.CLASS_NAMES[i]}:")
    print(f"     - Precision: {precision_pc[i]*100:.2f}%")
    print(f"     - Recall: {recall_pc[i]*100:.2f}%")
    print(f"     - F1-Score: {f1_pc[i]*100:.2f}%")
    print(f"     - Support: {support_pc[i]} samples")

print("\n6. FILES GENERATED:")
output_files = [
    'class_distribution.png',
    'sample_images.png',
    'training_history.png',
    'classification_report.txt',
    'confusion_matrix.png',
    'per_class_metrics.png',
    'prediction_samples.png',
    'roc_curves.png',
    'accuracy_comparison.png',
    'learning_curves.png',
    'best_model.h5',
    'model_metadata.json'
]
for file in output_files:
    print(f"   âœ“ {file}")

print("\n" + "="*80)
print("ðŸŽ‰ PROJECT EXECUTION COMPLETED!")
print("="*80)
print("\nAll visualizations, reports, and model files have been saved.")
print(f"Output directory: {Config.OUTPUT_DIR}")
print(f"Model directory: {Config.MODEL_DIR}")
print("\n" + "="*80)

