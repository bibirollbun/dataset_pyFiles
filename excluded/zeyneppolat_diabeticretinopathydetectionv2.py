# Import necessary libraries
import numpy as np
import pandas as pd
import os
import cv2 # OpenCV for image processing
import matplotlib.pyplot as plt # For plotting
import seaborn as sns # For enhanced visualizations
import time # To measure training times

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D, Flatten, Activation
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


BASE_DRIVE_DIR = '/kaggle/input/aptos2019-blindness-detection/'

TRAIN_CSV_PATH = os.path.join(BASE_DRIVE_DIR, 'train.csv')
TRAIN_IMG_PATH = os.path.join(BASE_DRIVE_DIR, 'train_images/')

print(f"TensorFlow Version: {tf.__version__}")

# --- 2. Verify GPU Availability ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), " GPU found,", len(logical_gpus), "logic GPU will be used.")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory set for the dynamic increase.")
    except RuntimeError as e:
        print(f"GPU settings error: {e}")
else:
    print("GPU couldn't found. CPU will be used.")

# Project parameters
IMG_SIZE = 224
BATCH_SIZE = 32  # Suitable for GPU.
print(f"Batch Size to be used: {BATCH_SIZE}")

EPOCHS = 50 # Adjust based on training time and convergence
NUM_CLASSES = 5 # Diabetic retinopathy stages (0, 1, 2, 3, 4)
RANDOM_STATE = 42

# Check if data paths exist
if not os.path.exists(TRAIN_CSV_PATH):
    print(f"ERROR: Training CSV not found at {TRAIN_CSV_PATH}")
    print("Please verify the BASE_DRIVE_DIR path.")
if not os.path.exists(TRAIN_IMG_PATH):
    print(f"ERROR: Training images folder not found at {TRAIN_IMG_PATH}")
    print("Please verify the BASE_DRIVE_DIR path.")


# --- Data Loading and Preprocessing ---
print("\n--- 1. Data Loading and Preprocessing ---") 
train_df = pd.read_csv(TRAIN_CSV_PATH)
train_df['image_path'] = train_df['id_code'].apply(lambda x: os.path.join(TRAIN_IMG_PATH, x + '.png'))
# Convert diagnosis to string for ImageDataGenerator's class_mode='categorical'
train_df['diagnosis'] = train_df['diagnosis'].astype(str)

print(f"Training dataset size: {train_df.shape}")
print(train_df.head())

# Check class distribution
print("\nClass Distribution in Training Data:")
print(train_df['diagnosis'].value_counts().sort_index())
sns.countplot(x='diagnosis', data=train_df)
plt.title('Class Distribution in Training Data')
plt.show()

# Split data into training and validation sets
train_data_df, val_data_df = train_test_split(train_df,
                                        test_size=0.2, # 20% for validation
                                        random_state=RANDOM_STATE,
                                        stratify=train_df['diagnosis']) # Preserve class distribution

print(f"\nSplit Training Data Size: {train_data_df.shape}")
print(f"Validation Data Size: {val_data_df.shape}")

# Image data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalize pixel values to [0,1]
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255) # Only rescale for validation

# Create data generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data_df,
    x_col='image_path',
    y_col='diagnosis',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical', # For multi-class classification
    shuffle=True
)

validation_generator = val_datagen.flow_from_dataframe(
    dataframe=val_data_df,
    x_col='image_path',
    y_col='diagnosis',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False # No need to shuffle validation data
)

# Calculate class weights for imbalanced datasets (optional, but can be helpful)
from sklearn.utils import class_weight
try:
    class_labels_int = np.unique(train_df['diagnosis'].astype(int))
    class_counts = np.bincount(train_df['diagnosis'].astype(int)) # Ensures all classes from 0 to N-1 are counted
    
    # Compute class weights using sklearn.utils.class_weight
    # Filter out classes with zero samples if any, though stratify should prevent this for train_df
    active_classes = [i for i, count in enumerate(class_counts) if count > 0]
    active_class_labels = train_df[train_df['diagnosis'].astype(int).isin(active_classes)]['diagnosis'].astype(int)
    
    if len(active_class_labels) > 0:
        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(active_class_labels),
            y=active_class_labels
        )
        class_weights_dict = dict(zip(np.unique(active_class_labels), weights))
        print("\nCalculated Class Weights:", class_weights_dict)
    else:
        print("\nCould not calculate class weights: No active classes found.")
        class_weights_dict = None

except Exception as e:
    print(f"\nError calculating class weights: {e}. Class weights will not be used.")
    class_weights_dict = None


# --- CustomAlexNet Definition ---
def CustomAlexNet(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    model = Sequential([
        Conv2D(96, (11, 11), strides=(4,4), padding='same', input_shape=input_shape), Activation('relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(256, (5, 5), padding='same'), Activation('relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(384, (3, 3), padding='same'), Activation('relu'),
        Conv2D(384, (3, 3), padding='same'), Activation('relu'),
        Conv2D(256, (3, 3), padding='same'), Activation('relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Flatten(),
        Dense(4096), Activation('relu'), Dropout(0.5),
        Dense(4096), Activation('relu'), Dropout(0.5),
        Dense(num_classes, activation='softmax') # Softmax for multi-class
    ], name="AlexNet_Custom")
    return model

# --- 2. Model Definition Function ---
def build_model(model_name, input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """
    Builds the specified model: AlexNet (custom), VGG16, or ResNet50.
    """
    if model_name.lower() == 'alexnet':
        model = CustomAlexNet(input_shape=input_shape, num_classes=num_classes)
    elif model_name.lower() == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x) # Regularization
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions, name=model_name)
    elif model_name.lower() == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x) # Regularization
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions, name=model_name)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    print(f"{model_name} model defined.")
    return model


# --- 3. Training and Comparing Models ---
print("\n--- 3. Training and Comparing Models ---")

model_names_list = ['AlexNet', 'VGG16', 'ResNet50'] # Models to compare
trained_models_dict = {}
histories_dict = {}
training_times_dict = {}
evaluation_results_dict = {}

for current_model_name in model_names_list:
    print(f"\n===== Processing Model: {current_model_name} (Expecting GPU usage) =====")
    start_time_model_processing = time.time()

    # Build and compile the model
    print(f"Building and compiling {current_model_name}...")
    keras_model = build_model(current_model_name,
                              input_shape=(IMG_SIZE, IMG_SIZE, 3),
                              num_classes=NUM_CLASSES)
    
    optimizer = Adam(learning_rate=1e-4) # Start with a common learning rate
    
    keras_model.compile(optimizer=optimizer,
                        loss='categorical_crossentropy', # For multi-class classification
                        metrics=['accuracy'])
    print(f"{current_model_name} model compiled.")
    keras_model.summary() #print model summary

    # Callbacks
    early_stopping_cb = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    # Save model checkpoints to Drive (ensure the path is writable)
    kaggle_output_dir = "/kaggle/working/"
    if not os.path.exists(kaggle_output_dir):
        os.makedirs(kaggle_output_dir)
    checkpoint_save_path = os.path.join(kaggle_output_dir, f'{current_model_name}_best_weights_gpu.keras')
    print(f"Model checkpoints will be saved to: {checkpoint_save_path}")
    model_checkpoint_cb = ModelCheckpoint(filepath=checkpoint_save_path,
                                          monitor='val_accuracy',
                                          save_best_only=True,
                                          save_weights_only=False, # Save the full model
                                          mode='max',
                                          verbose=1)
    reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    callbacks_to_use = [early_stopping_cb, model_checkpoint_cb, reduce_lr_cb]

    print(f"\n--- Training {current_model_name} Model ---")
    
    steps_per_epoch_val = max(1, train_generator.samples // BATCH_SIZE)
    validation_steps_val = max(1, validation_generator.samples // BATCH_SIZE)

    history_obj = keras_model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch_val,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_steps_val,
        callbacks=callbacks_to_use,
        class_weight=class_weights_dict # Use calculated class weights if available
    )

    end_time_model_processing = time.time()
    model_training_time = end_time_model_processing - start_time_model_processing
    print(f"Total processing time for {current_model_name}: {model_training_time:.2f} seconds")

    trained_models_dict[current_model_name] = keras_model
    histories_dict[current_model_name] = history_obj
    training_times_dict[current_model_name] = model_training_time

    # --- 4. Plotting Model Performance ---
    if history_obj and history_obj.history:
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        if 'accuracy' in history_obj.history and 'val_accuracy' in history_obj.history:
            plt.plot(history_obj.history['accuracy'], label='Training Accuracy')
            plt.plot(history_obj.history['val_accuracy'], label='Validation Accuracy')
            plt.title(f'{current_model_name} Accuracy')
            plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
        else:
            plt.text(0.5, 0.5, 'Accuracy data not available', ha='center', va='center')

        plt.subplot(1, 2, 2)
        if 'loss' in history_obj.history and 'val_loss' in history_obj.history:
            plt.plot(history_obj.history['loss'], label='Training Loss')
            plt.plot(history_obj.history['val_loss'], label='Validation Loss')
            plt.title(f'{current_model_name} Loss')
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
        else:
            plt.text(0.5, 0.5, 'Loss data not available', ha='center', va='center')
        plt.suptitle(f'Training History for {current_model_name}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
        plt.show()
    else:
        print(f"No training history found for {current_model_name} to plot.")

    # --- 5. Enhanced Model Evaluation (on Validation Set) ---
    print(f"\n--- Evaluating {current_model_name} on Validation Set ---")
    # Ensure the generator is reset before prediction
    validation_generator.reset()
    # If ModelCheckpoint restored best weights, current_tf_model has them.
    # Otherwise, load the best model:
    print(f"Loading best weights for {current_model_name} from {checkpoint_save_path}")
    best_model = tf.keras.models.load_model(checkpoint_save_path) # This might be needed if restore_best_weights=False or if evaluating later

    y_pred_probabilities = keras_model.predict(validation_generator,
                                             steps=max(1, (validation_generator.samples // BATCH_SIZE) + 1),
                                             verbose=1)
    y_pred_classes_indices = np.argmax(y_pred_probabilities, axis=1)
    y_true_indices = validation_generator.classes # True class indices
    
    # Ensure lengths match for metrics calculation, taking the shortest length
    num_samples_to_evaluate = min(len(y_pred_classes_indices), len(y_true_indices), validation_generator.samples)
    y_pred_classes_indices = y_pred_classes_indices[:num_samples_to_evaluate]
    y_true_indices = y_true_indices[:num_samples_to_evaluate]

    class_labels_str = [str(k) for k, v in sorted(validation_generator.class_indices.items(), key=lambda item: item[1])]

    # Accuracy
    overall_accuracy = accuracy_score(y_true_indices, y_pred_classes_indices)
    print(f"\nOverall Accuracy for {current_model_name}: {overall_accuracy:.4f}")

    # Classification Report (Precision, Recall/Sensitivity, F1-score)
    print(f"\nClassification Report for {current_model_name}:")
    cls_report = classification_report(y_true_indices, y_pred_classes_indices, target_names=class_labels_str, zero_division=0)
    print(cls_report)
    
    # Confusion Matrix and Specificity Calculation
    print(f"Confusion Matrix for {current_model_name}:")
    cm = confusion_matrix(y_true_indices, y_pred_classes_indices)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels_str, yticklabels=class_labels_str)
    plt.title(f'{current_model_name} Confusion Matrix'); plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.show()

    specificities = []
    for i in range(NUM_CLASSES):
        tn = 0
        fp = 0
        # Calculate TN and FP for class i
        tp_i = cm[i, i]
        fp_i = np.sum(cm[:, i]) - tp_i
        fn_i = np.sum(cm[i, :]) - tp_i
        tn_i = np.sum(cm) - (tp_i + fp_i + fn_i)

        specificity_i = tn_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else 0.0
        specificities.append(specificity_i)
        print(f"Specificity for class {class_labels_str[i]} ({current_model_name}): {specificity_i:.4f}")
    
    avg_specificity = np.mean(specificities)
    print(f"Average Specificity for {current_model_name}: {avg_specificity:.4f}")

    evaluation_results_dict[current_model_name] = {
        'accuracy': overall_accuracy,
        'classification_report_str': cls_report, # Storing as string
        'confusion_matrix_obj': cm,
        'specificities_list': specificities,
        'average_specificity': avg_specificity
    }



# --- 6. Final Results Comparison ---
print("\n--- Final Results Comparison ---")
print("\nTraining Times:")
for model_n, time_val in training_times_dict.items():
    print(f"{model_n}: {time_val:.2f} seconds")

print("\n--- Detailed Metrics per Model (Recalculating Recall from Confusion Matrix) ---")

class_labels_for_table = [f"Class {i}" for i in range(NUM_CLASSES)] 
if model_names_list and evaluation_results_dict.get(model_names_list[0]):
    first_model_eval_data = evaluation_results_dict[model_names_list[0]]
    if 'confusion_matrix_obj' in first_model_eval_data:
        try:
            class_labels_for_table = [str(k) for k, v in sorted(validation_generator.class_indices.items(), key=lambda item: item[1])]
        except NameError: # validation_generator might not be in scope if this cell is run alone
             print("Warning: validation_generator not in scope to get class_labels_str. Using generic labels for table.")


for model_n, results_data in evaluation_results_dict.items():
    print(f"\nMetrics for Model: {model_n}")
    print(f"  Overall Accuracy: {results_data.get('accuracy', 'N/A'):.4f}")
    print(f"  Average Specificity: {results_data.get('average_specificity', 'N/A'):.4f}")

    cm = results_data.get('confusion_matrix_obj')
    if cm is not None and isinstance(cm, np.ndarray):
        per_class_recall = []
        class_supports = []
        for i in range(NUM_CLASSES):
            tp_i = cm[i, i]
            fn_i = np.sum(cm[i, :]) - tp_i # Sum of row i minus TP_i
            
            recall_i = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0.0
            per_class_recall.append(recall_i)
            class_supports.append(tp_i + fn_i) # Support for class i is TP_i + FN_i

        if per_class_recall:
            macro_avg_recall = np.mean(per_class_recall)
            weighted_avg_recall = np.average(per_class_recall, weights=class_supports if sum(class_supports) > 0 else None)
            
            print(f"  Macro Avg Recall (Sensitivity) (recalculated): {macro_avg_recall:.4f}")
            print(f"  Weighted Avg Recall (Sensitivity) (recalculated): {weighted_avg_recall:.4f}")
            
            print("  Per-Class Recall (Sensitivity) (recalculated):")
            for i in range(NUM_CLASSES):
                class_label_name = class_labels_for_table[i] if i < len(class_labels_for_table) else f"Class {i}"
                print(f"    {class_label_name}: {per_class_recall[i]:.4f}")
            
            # Store these recalculated recalls for the summary table
            results_data['recalculated_macro_avg_recall'] = macro_avg_recall
            results_data['recalculated_weighted_avg_recall'] = weighted_avg_recall
            results_data['recalculated_per_class_recall'] = per_class_recall
        else:
            print("  Could not recalculate recall: Per-class recall list is empty.")
            results_data['recalculated_macro_avg_recall'] = 0.0
            results_data['recalculated_weighted_avg_recall'] = 0.0
            results_data['recalculated_per_class_recall'] = [0.0] * NUM_CLASSES

    else:
        print(f"  Confusion matrix not found for model {model_n}, cannot recalculate recall.")
        results_data['recalculated_macro_avg_recall'] = 'N/A'
        results_data['recalculated_weighted_avg_recall'] = 'N/A'
        results_data['recalculated_per_class_recall'] = ['N/A'] * NUM_CLASSES


# Create a more detailed comparison DataFrame
comparison_data = []
for model_n in model_names_list:
    if model_n in evaluation_results_dict:
        res = evaluation_results_dict[model_n]
        
        # Using recalculated recall values
        macro_recall = res.get('recalculated_macro_avg_recall', 0.0)
        weighted_recall = res.get('recalculated_weighted_avg_recall', 0.0)
        per_class_recalls = res.get('recalculated_per_class_recall', [0.0] * NUM_CLASSES)

        # Prepare data for DataFrame, handling 'N/A' for missing recall
        row_data = {
            'Model': model_n,
            'Training Time (s)': training_times_dict.get(model_n, 0),
            'Accuracy': res.get('accuracy', 0.0),
            'Avg. Specificity': res.get('average_specificity', 0.0)
        }
        
        if macro_recall != 'N/A':
            row_data['Macro Recall (Sensitivity)'] = macro_recall
        else:
            row_data['Macro Recall (Sensitivity)'] = 0.0 # Or np.nan

        if weighted_recall != 'N/A':
            row_data['Weighted Recall (Sensitivity)'] = weighted_recall
        else:
            row_data['Weighted Recall (Sensitivity)'] = 0.0 # Or np.nan
            
        # Add per-class recalls to the table
        for i in range(NUM_CLASSES):
            class_label_name = class_labels_for_table[i] if i < len(class_labels_for_table) else f"Class {i}"
            recall_value = per_class_recalls[i] if per_class_recalls[i] != 'N/A' else 0.0 # Or np.nan
            row_data[f'Recall {class_label_name}'] = recall_value
            
        comparison_data.append(row_data)

if comparison_data:
    comparison_df = pd.DataFrame(comparison_data)
    print("\nComparison Summary Table (with Recalculated Recall):")
    pd.set_option('display.max_columns', None) 
    pd.set_option('display.width', 1000)
    # Format float columns for better readability
    float_cols = [col for col in comparison_df.columns if comparison_df[col].dtype == 'float64' or comparison_df[col].dtype == 'float32']
    for col in float_cols:
        comparison_df[col] = comparison_df[col].map(lambda x: f"{x:.4f}" if isinstance(x, (float, np.float_)) else x)

    print(comparison_df)
else:
    print("\nNo evaluation results to display in summary table.")

print("\nProject finished.")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 


try:
    class_labels_for_table = [str(k) for k, v in sorted(validation_generator.class_indices.items(), key=lambda item: item[1])]
except NameError:
    print("Warning: validation_generator not in scope for class_labels_for_table. Using generic labels for plotting.")
    class_labels_for_table = [str(i) for i in range(NUM_CLASSES)] 

model_names_plot = list(evaluation_results_dict.keys())
accuracies_plot = [evaluation_results_dict[m].get('accuracy', 0) for m in model_names_plot]
avg_specificities_plot = [evaluation_results_dict[m].get('average_specificity', 0) for m in model_names_plot]

macro_recalls_plot = []

per_class_recalls_plot = {f"Recall {label}": [] for label in class_labels_for_table}

for model_n in model_names_plot:
    res = evaluation_results_dict[model_n]
    macro_recalls_plot.append(res.get('recalculated_macro_avg_recall', 0))
    
    recalls_for_model = res.get('recalculated_per_class_recall', [0.0] * NUM_CLASSES)
    for i in range(NUM_CLASSES):
        if i < len(class_labels_for_table):
            class_label_name = class_labels_for_table[i]
            key_name = f"Recall {class_label_name}"
            if key_name in per_class_recalls_plot: 
                 per_class_recalls_plot[key_name].append(recalls_for_model[i])
            else:
                print(f"Warning: Key '{key_name}' not found in per_class_recalls_plot. Skipping for model {model_n}, class index {i}.")
        else:
             print(f"Warning: Index {i} out of bounds for class_labels_for_table. Skipping for model {model_n}.")


metrics_to_plot = {
    'Accuracy': accuracies_plot,
    'Avg. Specificity': avg_specificities_plot,
    'Macro Recall (Sensitivity)': macro_recalls_plot
}

df_general_metrics = pd.DataFrame(metrics_to_plot, index=model_names_plot)

df_general_metrics.plot(kind='bar', figsize=(15, 7), colormap='viridis')
plt.title('General Performance Metrics Comparison per Model', fontsize=16)
plt.ylabel('Score', fontsize=14)
plt.xlabel('Model', fontsize=14)
plt.xticks(rotation=0)
plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()



valid_per_class_data = {k: v for k, v in per_class_recalls_plot.items() if len(v) == len(model_names_plot)}
if not valid_per_class_data:
    print("Warning: Not enough data to plot per-class recall. All lists might be empty or have inconsistent lengths.")
else:
    df_per_class_recall = pd.DataFrame(valid_per_class_data, index=model_names_plot)
    # Sütun adlarını düzeltelim (Recall 0 -> 0)
    df_per_class_recall.columns = [col.replace("Recall ", "") for col in df_per_class_recall.columns]

    df_per_class_recall.plot(kind='bar', figsize=(18, 8), colormap='plasma')
    plt.title('Per-Class Recall (Sensitivity) Comparison per Model', fontsize=16)
    plt.ylabel('Recall (Sensitivity)', fontsize=14)
    plt.xlabel('Model', fontsize=14)
    plt.xticks(rotation=0)
    plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()



if len(model_names_plot) > 0:
    num_models = len(model_names_plot)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    if num_models == 1:
        axes = [axes]
        
    for i, model_n in enumerate(model_names_plot):
        cm = evaluation_results_dict[model_n].get('confusion_matrix_obj')
        if cm is not None and isinstance(cm, np.ndarray):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                        xticklabels=class_labels_for_table, yticklabels=class_labels_for_table)
            axes[i].set_title(f'{model_n}\nConfusion Matrix', fontsize=14)
            axes[i].set_xlabel('Predicted Label')
            axes[i].set_ylabel('True Label')
        else:
            axes[i].text(0.5, 0.5, 'CM not available', ha='center', va='center')
            axes[i].set_title(f'{model_n}\nConfusion Matrix', fontsize=14)

    plt.tight_layout()
    plt.show()




val_data_ordered = val_data_df.reset_index(drop=True)

results_df = pd.DataFrame({
    'image_path': val_data_ordered['image_path'],
    'true_label': val_data_ordered['diagnosis'],
    'predicted_label': y_pred_classes_indices
})

import cv2
import matplotlib.pyplot as plt

unique_classes = results_df['true_label'].unique()
unique_classes = sorted([int(cls) for cls in unique_classes])

plt.figure(figsize=(15, 3 * len(unique_classes)))

for i, class_id in enumerate(unique_classes):
    idx = results_df[results_df['true_label'].astype(int) == class_id].index[0]
    image_path = results_df.iloc[idx]['image_path']
    true_label = results_df.iloc[idx]['true_label']
    pred_label = results_df.iloc[idx]['predicted_label']

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.subplot(len(unique_classes), 1, i+1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Image: {os.path.basename(image_path)} | True: {true_label} | Predicted: {pred_label}", fontsize=14)

plt.tight_layout()
plt.show()


