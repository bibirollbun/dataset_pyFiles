import os 
import pandas as pd

def print_isic_tree(base_path, prefix="", limit_subfiles=3):
    """Print ISIC 2024 directory tree.
    - Print all files in the base directory.
    - Print subdirectories recursively.
    - For train-image/image, show only first `limit_subfiles` files then '...'.
    """
    try:
        items = sorted(os.listdir(base_path))
    except PermissionError:
        print(prefix + "[Permission Denied]")
        return

    files = [f for f in items if os.path.isfile(os.path.join(base_path, f))]
    dirs = [d for d in items if os.path.isdir(os.path.join(base_path, d))]

    # Print all files in this directory
    for f in files:
        print(prefix + "├── " + f)

    for d in dirs:
        print(prefix + "├── " + d)
        sub_path = os.path.join(base_path, d)

        # Special case: train-image/image
        if os.path.basename(sub_path) == "image":
            try:
                imgs = sorted(os.listdir(sub_path))
                for i, img in enumerate(imgs):
                    if i < limit_subfiles:
                        print(prefix + "│   " + "├── " + img)
                    elif i == limit_subfiles:
                        print(prefix + "│   " + "├── ...")
                        break
            except PermissionError:
                print(prefix + "│   [Permission Denied]")
        else:
            print_isic_tree(sub_path, prefix + "│   ", limit_subfiles)
base_path = "/kaggle/input/isic-2024-challenge"
print_isic_tree(base_path)


images = "/kaggle/input/isic-2024-challenge/train-image/image"
metadata = "/kaggle/input/isic-2024-challenge/train-metadata.csv"
metadata = pd.read_csv(metadata)
metadata.head(5)


metadata.info()


# missing values
metadata.isnull().sum()


!pip install pydot graphviz


# --- Core Libraries ---
import os
import numpy as np
import pandas as pd
import warnings
from PIL import Image, ImageEnhance, ImageOps
from tqdm.auto import tqdm

# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- Scikit-learn ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# --- TensorFlow and Keras ---
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121, VGG16, ResNet50
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomBrightness
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

# --- Configuration ---
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define AUTOTUNE for tf.data performance
AUTOTUNE = tf.data.AUTOTUNE

# --- File Paths and Parameters ---
BASE_PATH = "/kaggle/input/isic-2024-challenge"
IMAGE_SIZE = (224, 224)
# Add this new line to define a writable output directory
AUG_SAVE_DIR = "/kaggle/working/augmented_images/"

BATCH_SIZE = 64 # Adjusted for potentially better gradient stability
EPOCHS = 25 # Increased for more training time on the balanced dataset
PATIENCE = 5
MODELS_TO_TRAIN = ['DenseNet121', 'ResNet50', 'VGG16']

print("TensorFlow Version:", tf.__version__)
print("Setup Complete. All libraries loaded.")


import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from PIL import Image, ImageEnhance, ImageOps
from tqdm.auto import tqdm
import joblib # NEW IMPORT
import json   # NEW IMPORT

class ISICBalancedDataPreparer:
    """
    Handles the end-to-end process of sampling, augmenting, and balancing the
    ISIC 2024 dataset to prepare it for efficient deep learning training.
    """
    def __init__(self, metadata_path, image_dir, augmentation_save_path):
        self.metadata_path = metadata_path
        self.image_dir = image_dir
        self.aug_save_dir = augmentation_save_path

        if not os.path.exists(self.aug_save_dir):
            os.makedirs(self.aug_save_dir)
            print(f"Created directory for augmented images: {self.aug_save_dir}")
        self.scaler = StandardScaler()

    def _apply_augmentations(self, image):
        # ... (this method is unchanged)
        augmented_images = []
        augmented_images.append(ImageOps.mirror(image))
        augmented_images.append(ImageOps.flip(image))
        for angle in [60, 120, 180]:
            augmented_images.append(image.rotate(angle))
        for factor in [0.7, 1.3]:
            enhancer = ImageEnhance.Brightness(image)
            augmented_images.append(enhancer.enhance(factor))
        w, h = image.size
        for factor in [0.7, 0.8]:
            crop_w, crop_h = int(w * factor), int(h * factor)
            left, top = (w - crop_w) // 2, (h - crop_h) // 2
            right, bottom = (w + crop_w) // 2, (h + crop_h) // 2
            cropped = image.crop((left, top, right, bottom))
            augmented_images.append(cropped.resize((w, h)))
        return augmented_images

    def _preprocess_metadata(self, df_train, df_val, df_test):
        # ... (this method is unchanged)
        print("\nPreprocessing and scaling metadata...")
        df_train, df_val, df_test = df_train.copy(), df_val.copy(), df_test.copy()
        cols_to_remove = ['patient_id', 'lesion_id', 'copyright_license', 'attribution',
                          'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5', 'mel_mitotic_index', 'mel_thick_mm']
        df_train.drop(columns=cols_to_remove, inplace=True, errors='ignore')
        df_val.drop(columns=cols_to_remove, inplace=True, errors='ignore')
        df_test.drop(columns=cols_to_remove, inplace=True, errors='ignore')
        numerical_cols = df_train.select_dtypes(include=np.number).columns
        categorical_cols = df_train.select_dtypes(include=['object', 'category']).columns
        for col in numerical_cols:
            median_val = df_train[col].median()
            df_train[col].fillna(median_val, inplace=True)
            df_val[col].fillna(median_val, inplace=True)
            df_test[col].fillna(median_val, inplace=True)
        for col in categorical_cols:
            mode_val = df_train[col].mode()[0]
            df_train[col].fillna(mode_val, inplace=True)
            df_val[col].fillna(mode_val, inplace=True)
            df_test[col].fillna(mode_val, inplace=True)
        df_train = pd.get_dummies(df_train, columns=categorical_cols, drop_first=True)
        df_val = pd.get_dummies(df_val, columns=categorical_cols, drop_first=True)
        df_test = pd.get_dummies(df_test, columns=categorical_cols, drop_first=True)
        shared_cols = list(set(df_train.columns) & set(df_val.columns) & set(df_test.columns))
        df_train, df_val, df_test = df_train[shared_cols], df_val[shared_cols], df_test[shared_cols]
        df_train_scaled = self.scaler.fit_transform(df_train)
        df_val_scaled = self.scaler.transform(df_val)
        df_test_scaled = self.scaler.transform(df_test)
        return df_train_scaled, df_val_scaled, df_test_scaled, shared_cols

    def run(self, num_positives_target=5000, num_negatives_target=5000):
        # ... (The first part of this method is unchanged)
        print("="*80 + "\nSTARTING ISIC 2024 DATA PREPARATION PIPELINE\n" + "="*80)
        df = pd.read_csv(self.metadata_path)
        pos_df = df[df['target'] == 1].copy()
        neg_df = df[df['target'] == 0].copy()
        print(f"Loaded {len(pos_df)} positive and {len(neg_df)} negative samples.")

        print(f"Subsampling negative class to {num_negatives_target}...")
        neg_df_sampled = neg_df.sample(n=num_negatives_target, random_state=42)

        print("Splitting original data (60/20/20) BEFORE augmentation...")
        pos_train, pos_temp = train_test_split(pos_df, test_size=0.4, random_state=42)
        pos_val, pos_test = train_test_split(pos_temp, test_size=0.5, random_state=42)
        neg_train, neg_temp = train_test_split(neg_df_sampled, test_size=0.4, random_state=42)
        neg_val, neg_test = train_test_split(neg_temp, test_size=0.5, random_state=42)

        final_dfs = {}
        for split_name, pos_split_df in [('train', pos_train), ('val', pos_val), ('test', pos_test)]:
            print(f"\nAugmenting positive samples for '{split_name}' split...")
            # ... (augmentation loop is unchanged) ...
            augmented_records = []
            if len(pos_split_df) == 0: continue
            target_split_size = int(num_positives_target * (len(pos_split_df) / len(pos_df)))
            aug_per_image = max(1, round(target_split_size / len(pos_split_df)))

            for _, row in tqdm(pos_split_df.iterrows(), total=len(pos_split_df)):
                original_image_path = os.path.join(self.image_dir, f"{row['isic_id']}.jpg")
                record = row.to_dict(); record['image_path'] = original_image_path
                augmented_records.append(record)
                
                try:
                    with Image.open(original_image_path) as img:
                        generated_augs = self._apply_augmentations(img)
                        num_augs_to_generate = aug_per_image - 1
                        
                        if num_augs_to_generate > 0:
                            chosen_indices = np.random.choice(len(generated_augs), size=num_augs_to_generate, replace=True)
                            for i, idx in enumerate(chosen_indices):
                                aug_img = generated_augs[idx]
                                aug_path = os.path.join(self.aug_save_dir, f"{row['isic_id']}_aug_{i}.jpg")
                                aug_img.save(aug_path)
                                aug_record = row.to_dict(); aug_record['image_path'] = aug_path
                                augmented_records.append(aug_record)
                except FileNotFoundError:
                    print(f"Warning: Image not found for {row['isic_id']}, skipping.")
            final_dfs[split_name] = pd.DataFrame(augmented_records)

        train_df = pd.concat([final_dfs['train'], neg_train]).sample(frac=1, random_state=42).reset_index(drop=True)
        val_df = pd.concat([final_dfs['val'], neg_val]).sample(frac=1, random_state=42).reset_index(drop=True)
        test_df = pd.concat([final_dfs['test'], neg_test]).sample(frac=1, random_state=42).reset_index(drop=True)

        for df_split in [train_df, val_df, test_df]:
            df_split['image_path'] = df_split.apply(lambda r: r['image_path'] if pd.notna(r['image_path']) else os.path.join(self.image_dir, f"{r['isic_id']}.jpg"), axis=1)
        
        X_train_meta, X_val_meta, X_test_meta, final_cols = self._preprocess_metadata(
            train_df.drop(columns=['target', 'isic_id', 'image_path']),
            val_df.drop(columns=['target', 'isic_id', 'image_path']),
            test_df.drop(columns=['target', 'isic_id', 'image_path'])
        )

        # --- START: NEW CODE TO ADD ---
        # Save the fitted scaler and the list of columns for later use in prediction
        scaler_path = '/kaggle/working/metadata_scaler.joblib'
        cols_path = '/kaggle/working/training_columns.json'

        joblib.dump(self.scaler, scaler_path)
        print(f"\nMetadata scaler saved to: {scaler_path}")

        with open(cols_path, 'w') as f:
            json.dump(final_cols, f)
        print(f"Training columns saved to: {cols_path}")
        # --- END: NEW CODE TO ADD ---

        print("\n" + "="*80 + "\nDATA PREPARATION COMPLETE\n" + "="*80)
        # ... (rest of the method is unchanged) ...
        print(f"Training set:   {len(train_df)} samples (Pos: {(train_df.target==1).sum()}, Neg: {(train_df.target==0).sum()})")
        print(f"Validation set: {len(val_df)} samples (Pos: {(val_df.target==1).sum()}, Neg: {(val_df.target==0).sum()})")
        print(f"Test set:       {len(test_df)} samples (Pos: {(test_df.target==1).sum()}, Neg: {(test_df.target==0).sum()})")
        print(f"Metadata has {X_train_meta.shape[1]} features after preprocessing.")
        print("="*80)
        
        return {
            'X_train_meta': X_train_meta, 'X_val_meta': X_val_meta, 'X_test_meta': X_test_meta,
            'y_train': train_df['target'].values, 'y_val': val_df['target'].values, 'y_test': test_df['target'].values,
            'train_image_paths': train_df['image_path'].tolist(), 'val_image_paths': val_df['image_path'].tolist(), 'test_image_paths': test_df['image_path'].tolist(),
            'metadata_feature_names': final_cols
        }

print("Data Preparer class defined.")


# Define specific paths for clarity
METADATA_PATH = os.path.join(BASE_PATH, "train-metadata.csv")
IMAGE_DIR = os.path.join(BASE_PATH, "train-image/image")

# Instantiate and run the data preparation pipeline with the corrected paths
data_preparer = ISICBalancedDataPreparer(
    metadata_path=METADATA_PATH,
    image_dir=IMAGE_DIR,
    augmentation_save_path=AUG_SAVE_DIR  # Pass the new writable path here
)
prepared_data = data_preparer.run(num_positives_target=5000, num_negatives_target=5000)

# Unpack the data for easier access
X_train_meta = prepared_data['X_train_meta']
X_val_meta = prepared_data['X_val_meta']
X_test_meta = prepared_data['X_test_meta']
y_train = prepared_data['y_train']
y_val = prepared_data['y_val']
y_test = prepared_data['y_test']
train_image_paths = prepared_data['train_image_paths']
val_image_paths = prepared_data['val_image_paths']
test_image_paths = prepared_data['test_image_paths']


# Optional: on-the-fly augmentation for further regularization
data_augmentation_layer = keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    RandomZoom(height_factor=0.2, width_factor=0.2),
    RandomBrightness(factor=0.2),
], name="on_the_fly_augmentation")

def load_and_preprocess_image(image_path):
    """Loads and preprocesses a single image file."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = img / 255.0  # Normalize to [0, 1]
    return img

def create_dataset(image_paths, metadata, targets, augment=False):
    """Creates a tf.data.Dataset for multi-input training."""
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    
    meta_ds = tf.data.Dataset.from_tensor_slices(metadata.astype('float32'))
    target_ds = tf.data.Dataset.from_tensor_slices(targets.reshape(-1, 1).astype('float32'))
    
    # Zip the components into the required structure: ((image, metadata), target)
    ds = tf.data.Dataset.zip(((image_ds, meta_ds), target_ds))
    
    if augment:
        ds = ds.map(lambda x, y: ((data_augmentation_layer(x[0], training=True), x[1]), y), num_parallel_calls=AUTOTUNE)
        ds = ds.shuffle(buffer_size=BATCH_SIZE * 10)
        
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

# Create the final data generators
train_generator = create_dataset(train_image_paths, X_train_meta, y_train, augment=True)
val_generator = create_dataset(val_image_paths, X_val_meta, y_val)
test_generator = create_dataset(test_image_paths, X_test_meta, y_test)

print("tf.data pipelines created successfully.")
print("Train generator spec:", train_generator.element_spec)


class MultiInputModelBuilder:
    """Builds multi-input models with a specified CNN backbone."""
    def __init__(self, image_size, metadata_dim):
        self.image_size = image_size
        self.metadata_dim = metadata_dim

    def build(self, backbone_name='DenseNet121', trainable_backbone=False):
        # Image Branch
        image_input = layers.Input(shape=(*self.image_size, 3), name='image_input')
        if backbone_name == 'DenseNet121':
            backbone = DenseNet121(weights='imagenet', include_top=False, input_tensor=image_input)
        elif backbone_name == 'ResNet50':
            backbone = ResNet50(weights='imagenet', include_top=False, input_tensor=image_input)
        # --- ADD THIS BLOCK ---
        elif backbone_name == 'VGG16':
            backbone = VGG16(weights='imagenet', include_top=False, input_tensor=image_input)
        # ----------------------
        else:
            raise ValueError(f"Backbone '{backbone_name}' not supported.")
        backbone.trainable = trainable_backbone
        
        x1 = layers.GlobalAveragePooling2D()(backbone.output)
        x1 = layers.Dense(128, activation='relu')(x1)
        x1 = layers.Dropout(0.5)(x1)

        # Metadata Branch
        metadata_input = layers.Input(shape=(self.metadata_dim,), name='metadata_input')
        x2 = layers.Dense(64, activation='relu')(metadata_input)
        x2 = layers.Dropout(0.3)(x2)

        # Combined Branch
        combined = layers.Concatenate()([x1, x2])
        z = layers.Dense(64, activation='relu')(combined)
        z = layers.Dropout(0.4)(z)
        output = layers.Dense(1, activation='sigmoid', name='output')(z)
        
        model = models.Model(inputs=[image_input, metadata_input], outputs=output)
        return model

class ModelTrainer:
    """Handles model compilation, training, and evaluation."""
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.history = None

    def train(self, train_gen, val_gen, epochs, patience, class_weights=None):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.F1Score(threshold=0.5, name='f1_score')]
        )
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=patience-2, min_lr=1e-7)
        ]
        self.history = self.model.fit(
            train_gen, validation_data=val_gen, epochs=epochs,
            callbacks=callbacks, class_weight=class_weights, verbose=1
        )
        return self.history

    def evaluate(self, test_gen):
        y_true = np.concatenate([y for x, y in test_gen], axis=0).flatten()
        y_pred_proba = self.model.predict(test_gen).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        print("\n" + "="*50 + f"\nTest Results for {self.model_name}\n" + "="*50)
        print(classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1']))
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred)
        }
        
    def plot_history(self):
        history_dict = self.history.history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Training History - {self.model_name}', fontsize=16)
        
        ax1.plot(history_dict['loss'], label='Train Loss')
        ax1.plot(history_dict['val_loss'], label='Val Loss')
        ax1.set_title('Model Loss'); ax1.set_xlabel('Epoch'); ax1.legend()
        
        ax2.plot(history_dict['accuracy'], label='Train Accuracy')
        ax2.plot(history_dict['val_accuracy'], label='Val Accuracy')
        ax2.set_title('Model Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend()
        plt.show()

print("Model builder and trainer classes defined.")


# Although balanced, calculating weights is still good practice.
class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: w for i, w in enumerate(class_weights_array)}
print("Calculated Class Weights:", class_weights)

all_results = {}
# --- START: NEW CODE TO ADD ---
# Variables to track the best model
best_model_path = ""
best_val_f1_score = 0.0
# --- END: NEW CODE TO ADD ---

for model_name in MODELS_TO_TRAIN:
    print("\n" + "#"*80)
    print(f"##  STARTING TRAINING FOR: {model_name.upper()}")
    print("#"*80)
    
    keras.backend.clear_session()
    
    builder = MultiInputModelBuilder(image_size=IMAGE_SIZE, metadata_dim=X_train_meta.shape[1])
    model = builder.build(backbone_name=model_name)
    display(plot_model(model, to_file=f'{model_name}_horizontal.png', show_shapes=True, rankdir='LR'))
    
    trainer = ModelTrainer(model, model_name)
    history = trainer.train(train_generator, val_generator, EPOCHS, PATIENCE, class_weights)
    
    # --- START: NEW CODE TO ADD ---
    # Save the model after training is complete
    model_save_path = f'/kaggle/working/{model_name}_model.keras'
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")
    
    # Check if this is the best model so far based on max validation F1 score
    current_max_val_f1 = max(history.history['val_f1_score'])
    if current_max_val_f1 > best_val_f1_score:
        best_val_f1_score = current_max_val_f1
        best_model_path = model_save_path
        print(f"*** New best model found: {model_name} with validation F1-score: {best_val_f1_score:.4f} ***")
    # --- END: NEW CODE TO ADD ---

    # Evaluate Model on the balanced test set
    test_metrics = trainer.evaluate(test_generator)
    all_results[model_name] = test_metrics
    
    trainer.plot_history()

# ... (the final comparison part of the cell is unchanged) ...
results_df = pd.DataFrame(all_results).T
results_df.index.name = 'Model'
print("\n" + "="*60)
print("           FINAL MODEL PERFORMANCE COMPARISON")
print("="*60)
print(results_df)

results_df.plot(kind='bar', figsize=(14, 7), rot=0)
plt.title('Model Performance Comparison on Test Set', fontsize=16)
plt.ylabel('Score')
plt.ylim(0, 1)
plt.legend(title='Metric', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.show()


# Create a DataFrame from the results
results_df = pd.DataFrame(all_results).T
results_df.index.name = 'Model'

print("\n" + "="*60)
print("           FINAL MODEL PERFORMANCE COMPARISON")
print("="*60)
print(results_df)

# Plotting the results
results_df.plot(kind='bar', figsize=(14, 7), rot=0)
plt.title('Model Performance Comparison on Test Set', fontsize=16)
plt.ylabel('Score')
plt.ylim(0, 1)
plt.legend(title='Metric', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.show()


# The path to the dataset containing your saved models
TRAINING_OUTPUT_PATH = '/kaggle/input/isic-2024-models' 

print("--- Starting Final Sanity Check from Saved Assets ---")
print(f"Loading assets from: {TRAINING_OUTPUT_PATH}")

# 1. Load the shared preprocessing assets
print("\nLoading preprocessing assets...")
try:
    scaler_path = os.path.join(TRAINING_OUTPUT_PATH, 'metadata_scaler.joblib')
    cols_path = os.path.join(TRAINING_OUTPUT_PATH, 'training_columns.json')
    scaler = joblib.load(scaler_path)
    with open(cols_path, 'r') as f:
        training_columns = json.load(f)
    print("Assets loaded successfully.")
except Exception as e:
    print(f"Could not load preprocessing assets. Error: {e}. Skipping sanity check.")
    training_columns = None

if training_columns:
    # 2. Create the imbalanced test set from original data
    print("\nCreating a new 5000-sample test set from the original data...")
    original_df = pd.read_csv(METADATA_PATH)
    sanity_check_df = original_df.sample(n=5000, random_state=123).copy()
    
    # 3. Preprocess this new test set
    print("Preprocessing the new test set...")
    sanity_check_ids = sanity_check_df['isic_id']
    sanity_check_targets = sanity_check_df['target']
    df_to_process = sanity_check_df.copy()
    
    # --- START: NEWLY ADDED CODE ---
    # Drop the same irrelevant/heavily-missing columns as in the original training
    cols_to_remove = ['patient_id', 'lesion_id', 'copyright_license', 'attribution',
                      'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5', 'mel_mitotic_index', 'mel_thick_mm']
    df_to_process.drop(columns=[col for col in cols_to_remove if col in df_to_process.columns], inplace=True)
    # --- END: NEWLY ADDED CODE ---

    # Now, proceed with imputation on the remaining columns
    numerical_cols = df_to_process.select_dtypes(include=np.number).columns
    categorical_cols = df_to_process.select_dtypes(include=['object', 'category']).columns
    for col in numerical_cols:
        df_to_process[col] = df_to_process[col].fillna(df_to_process[col].median())
    for col in categorical_cols:
        df_to_process[col] = df_to_process[col].fillna(df_to_process[col].mode()[0])
        
    df_to_process = pd.get_dummies(df_to_process, columns=categorical_cols, drop_first=True)
    df_aligned = df_to_process.reindex(columns=training_columns, fill_value=0)
    X_meta_scaled = scaler.transform(df_aligned)

    # 4. Create the shared tf.data pipeline
    print("Creating shared tf.data pipeline for the sanity check...")
    sanity_check_paths = [os.path.join(IMAGE_DIR, f"{id}.jpg") for id in sanity_check_ids]
    sanity_check_generator = create_dataset(sanity_check_paths, X_meta_scaled, sanity_check_targets.values, augment=False)
    y_true_sanity = sanity_check_targets.values

    # 5. Loop through each model, load it, and evaluate
    print("\n--- Evaluating each model on the sanity check set ---")
    for model_name in MODELS_TO_TRAIN:
        model_path = os.path.join(TRAINING_OUTPUT_PATH, f'{model_name}_model.keras')
        
        print("\n" + "="*60)
        print(f"EVALUATING: {model_name}")
        print(f"Model path: {model_path}")
        print("="*60)
        
        try:
            keras.backend.clear_session()
            model = tf.keras.models.load_model(model_path)
            
            y_pred_proba_sanity = model.predict(sanity_check_generator).flatten()
            y_pred_sanity = (y_pred_proba_sanity > 0.5).astype(int)
            
            print(f"\n--- Sanity Check Evaluation Report for {model_name} ---")
            print(classification_report(y_true_sanity, y_pred_sanity, target_names=['Benign (Class 0)', 'Malignant (Class 1)']))

        except Exception as e:
            print(f"Could not evaluate model {model_name}. Error: {e}")




