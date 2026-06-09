# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
tif_files = []

for dirname, _, filenames in os.walk('/kaggle/input'):
    print(f"\nos path: [{dirname}]")
    tif_file_count =0
    for filename in filenames:
        if filename.endswith('.tif'):
            tif_file_count +=1
            if tif_file_count < 3:
                print(os.path.join(dirname, filename))
            elif tif_file_count == 3:
                print("...")
        else:
            print(os.path.join(dirname, filename))  
    if tif_file_count > 0:
        print(f"\nTotal number of .tif files in {dirname}: {tif_file_count}")
        tif_file_count =0

        
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import os 

train_labels = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/train_labels.csv')
print("shape:",train_labels.shape)
print("-- Label head ---")
print("################")
print(train_labels.head())
print("-----------------")
print("Missing values: ")
print("################")
print(train_labels.isnull().sum())
print("-----------------")
print(f"Sum of duplicated labels: {train_labels.duplicated().sum()}.")


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='label', data=train_labels)
plt.xticks([0, 1], ['Non-Cancer (0)', 'Cancer (1)'])
plt.title("Distribution of Cancer vs Non-Cancer Images")
plt.ylabel("Image Count")
plt.xlabel("Label")
plt.show()



### show sample image files
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# add 'train_filepath' to dataframe
train_dir = Path("/kaggle/input/histopathologic-cancer-detection/train")
train_labels = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/train_labels.csv')
train_labels['train_filepath'] = train_labels['id'].apply(lambda x: str(train_dir / f'{x}.tif'))
# # convert 'label' field to string.
# train_labels['label'] = train_labels['label'].astype(str)

# show sample file
def show_samples(label, df=train_labels, num_images=5):
    subset = df[df['label'] == label].sample(num_images, random_state=42)
    plt.figure(figsize=(12, 4))
    for i, row in enumerate(subset.itertuples()):
        img = cv2.imread(row.train_filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        center = (32, 32, 64, 64)  
        cv2.rectangle(img, (center[0], center[1]), (center[2], center[3]), (255, 0, 0), 1)
        plt.subplot(1, num_images, i+1)
        plt.imshow(img)
        plt.title(f'Label: {row.label}')
        plt.axis('off')
    plt.show()


show_samples(0)  # Non-cancerous
show_samples(1)  # Cancerous

# check image size and channel
img_path = f"/kaggle/input/histopathologic-cancer-detection/train/{train_labels['id'].iloc[0]}.tif"
img = Image.open(img_path)
print(f"image size: {img.size}") 
print(f"image mode: {img.mode}") 


import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow_io as tfio
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


from sklearn.model_selection import train_test_split
import cv2


# Split the DataFrame into training and validation sets
train_df, val_df = train_test_split(
    train_labels,
    test_size=0.2,
    stratify=train_labels['label'],
    random_state=42
)

def load_image(path, label):
    def _load_image_py(path_tensor):
        path_str = path_tensor.numpy()
        if isinstance(path_str, bytes):
            path_str = path_str.decode("utf-8")
        img = cv2.imread(path_str)
        if img is None:
            raise ValueError(f"Failed to load image at path: {path_str}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.uint8)

    image = tf.py_function(func=_load_image_py, inp=[path], Tout=tf.uint8)
    image.set_shape([96, 96, 3])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.int32)  
    return image, label

# Create training and validation datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_df['train_filepath'], train_df['label']))
train_dataset = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE)

validation_dataset = tf.data.Dataset.from_tensor_slices((val_df['train_filepath'], val_df['label']))
validation_dataset = validation_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.batch(64).prefetch(tf.data.AUTOTUNE)

print("Data Pre-processing complete..")


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.metrics import AUC

#input shape = 96x 96x 3 (width x height x RGB)
cnn_model = Sequential([
    # First CNN block 
    Conv2D(32, kernel_size=5, strides=1, padding='same', input_shape=(96, 96, 3)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=2, strides=2), 
    
    # Second CNN block
    Conv2D(64, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=2, strides=2), 
    
    # Third CNN block
    Conv2D(128, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=2, strides=2), 
    
    # Output layer
    Flatten(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid') 
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', AUC(name='auroc')])
cnn_model.summary()
three_layer_history = cnn_model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

print("\nTraining Metrics:")
print("---------------------------")
for metric in ['loss', 'accuracy', 'auroc']:
    print(f"{metric.capitalize()}: {three_layer_history.history[metric][-1]:.4f}")

print("\nValidation Metrics:")
print("---------------------------")
for metric in ['val_loss', 'val_accuracy', 'val_auroc']:
    print(f"{metric.replace('val_', 'Validation ').capitalize()}: {three_layer_history.history[metric][-1]:.4f}")



# show confusion matrix
y_true = []
y_pred = []
for images, labels in validation_dataset:
    preds = cnn_model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend((preds > 0.5).astype(int).flatten())
y_true = np.array(y_true)
y_pred = np.array(y_pred)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])

plt.title('Confusion Matrix (Validation DataSet)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("\nConfusion Matrix Metrics:")
print(f"True Negatives (TN): {cm[0,0]}")
print(f"False Positives (FP): {cm[0,1]}")
print(f"False Negatives (FN): {cm[1,0]}")
print(f"True Positives (TP): {cm[1,1]}")


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


# Training loss vs Validation Loss
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(three_layer_history.history['loss'], label='Training Loss')
plt.plot(three_layer_history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Training Accuracy vs Validation Accuracy
plt.subplot(1, 3, 2)
plt.plot(three_layer_history.history['accuracy'], label='Training Accuracy')
plt.plot(three_layer_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# AUC 
plt.subplot(1, 3, 3)
plt.plot(three_layer_history.history['auroc'], label='Training AUC')
plt.plot(three_layer_history.history['val_auroc'], label='Validation AUC')
plt.title('Training vs Validation AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.metrics import AUC

#Two convolutional block architecture
two_layer_cnn_model = Sequential([
    Conv2D(32, kernel_size=5, strides=1, padding='same', input_shape=(96, 96, 3)),
    BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),

    Conv2D(64, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),

    # Output layer
    Flatten(),Dropout(0.5),
    Dense(256, activation='relu'),Dropout(0.3), Dense(1, activation='sigmoid') 
])

two_layer_cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', AUC(name='auroc')])
two_layer_history = two_layer_cnn_model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

print("\nTraining Metrics:")
print("---------------------------")
for metric in ['loss', 'accuracy', 'auroc']:
    print(f"{metric.capitalize()}: {two_layer_history.history[metric][-1]:.4f}")

print("\nValidation Metrics:")
print("---------------------------")
for metric in ['val_loss', 'val_accuracy', 'val_auroc']:
    print(f"{metric.replace('val_', 'Validation ').capitalize()}: {two_layer_history.history[metric][-1]:.4f}")



import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.metrics import AUC

#input shape = 96x 96x 3 (width x height x RGB)
five_layer_cnn_model = Sequential([
    Conv2D(32, kernel_size=5, strides=1, padding='same', input_shape=(96, 96, 3)),
    BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),

    Conv2D(64, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),

    Conv2D(128, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),

    Conv2D(256, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),

    Conv2D(512, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),
        
    # Output layer
    Flatten(),Dropout(0.5),
    Dense(256, activation='relu'),Dropout(0.3), Dense(1, activation='sigmoid') 
])

five_layer_cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', AUC(name='auroc')])
five_layer_history = five_layer_cnn_model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

print("\nTraining Metrics:")
print("---------------------------")
for metric in ['loss', 'accuracy', 'auroc']:
    print(f"{metric.capitalize()}: {five_layer_history.history[metric][-1]:.4f}")

print("\nValidation Metrics:")
print("---------------------------")
for metric in ['val_loss', 'val_accuracy', 'val_auroc']:
    print(f"{metric.replace('val_', 'Validation ').capitalize()}: {five_layer_history.history[metric][-1]:.4f}")



import matplotlib.pyplot as plt

histories = {
    "2-layer CNN": two_layer_history,
    "3-layer CNN": three_layer_history,
    "5-layer CNN": five_layer_history
}

metrics = ['accuracy', 'val_accuracy']
titles = {
    'accuracy': 'Training Accuracy',
    'val_accuracy': 'Validation Accuracy'
}

for metric in ['accuracy', 'val_accuracy']:    
    plt.figure(figsize=(6, 4))
    for label, history in histories.items():
        plt.plot(history.history[metric], label=label)
    plt.title(f"{titles[metric]} Comparison")
    plt.xlabel("Epoch")
    plt.ylabel(metric.split('_')[-1].capitalize())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

records = []
for name, hist in histories.items():
    max_train_accuracy = max(hist.history['accuracy'])
    max_val_accuracy = max(hist.history['val_accuracy'])
    last_train_accuracy = hist.history['accuracy'][-1]
    last_val_accuracy = hist.history['val_accuracy'][-1]
    train_loss = hist.history['loss'][-1]
    val_loss = hist.history['val_loss'][-1]
    max_auroc = max(hist.history['val_auroc'])
    last_auroc = hist.history['val_auroc'][-1]    
    records.append({
        "Model": name,
        "Max Training Accuracy": round(max_train_accuracy, 4),
        "Max Validation Accuracy": round(max_val_accuracy, 4),
        "Last Train Acc": round(last_train_accuracy, 4),
        "Last Val Acc": round(last_val_accuracy, 4),
        "Train Loss": round(train_loss, 4),
        "Val Loss": round(val_loss, 4),
        "Max Val AUC": round(max_auroc, 4),
        "Last Val AUC": round(last_auroc, 4)         
    })
metrics_df = pd.DataFrame(records)
metrics_df


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Training loss vs Validation Loss
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(five_layer_history.history['loss'], label='Training Loss')
plt.plot(five_layer_history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss \n(Best Model Architecture)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Training Accuracy vs Validation Accuracy
plt.subplot(1, 3, 2)
plt.plot(five_layer_history.history['accuracy'], label='Training Accuracy')
plt.plot(five_layer_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy \n(Best Model Architecture)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# AUC 
plt.subplot(1, 3, 3)
plt.plot(five_layer_history.history['auroc'], label='Training AUC')
plt.plot(five_layer_history.history['val_auroc'], label='Validation AUC')
plt.title('Training vs Validation AUC \n(Best Model Architecture)')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.metrics import AUC


filter_config = [(16, 32, 64, 128, 256), (24, 48, 96, 192, 384)]
# add previous model result
history_dict = {} 
history_dict[(32, 64, 128, 256, 512)] = five_layer_history
best_validation_accuracy = five_layer_history.history['val_accuracy'][-1]
best_filter_config = (32, 64, 128, 256, 512)
best_filter_history = five_layer_history

for idx, (f1, f2, f3, f4, f5) in enumerate(filter_config):
    filter_label = (f1, f2, f3, f4, f5)
    print(f"\nTraining Model with Filters: {filter_label}...")
    # best model architecture (5 layers)
    model_filter_tuning = Sequential([
        Conv2D(f1, kernel_size=5, strides=1, padding='same', input_shape=(96, 96, 3)),
        BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),
    
        Conv2D(f2, kernel_size=3, strides=1, padding='same'),
        BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),
    
        Conv2D(f3, kernel_size=3, strides=1, padding='same'),
        BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),
    
        Conv2D(f4, kernel_size=3, strides=1, padding='same'),
        BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),
    
        Conv2D(f5, kernel_size=3, strides=1, padding='same'),
        BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),
            
        # Output layer
        Flatten(),Dropout(0.5),
        Dense(256, activation='relu'),Dropout(0.3), Dense(1, activation='sigmoid') 
    ])

        
    model_filter_tuning.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', AUC(name='auroc')])
    history_filter_tuning = model_filter_tuning.fit(train_dataset, validation_data=validation_dataset, epochs=10, verbose=1)
    history_dict[filter_label] = history_filter_tuning
    final_validation_accuracy =  history_filter_tuning.history['val_accuracy'][-1]
    print(f"Final Validation Accuracy for {filter_label}: {final_validation_accuracy:.4f}")

    # Update best if current model is better
    if final_validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = final_validation_accuracy
        best_filter_config = filter_label
        best_filter_history = history_filter_tuning

print("Best Filter Configuration:")
print(f"Filters: {best_filter_config}")
print(f"Validation Accuracy: {best_validation_accuracy:.4f}")


# Table of performance comparison
records_filter = []
for name, hist in history_dict.items():
    max_train_accuracy = max(hist.history['accuracy'])
    max_val_accuracy = max(hist.history['val_accuracy'])
    last_train_accuracy = hist.history['accuracy'][-1]
    last_val_accuracy = hist.history['val_accuracy'][-1]
    train_loss = hist.history['loss'][-1]
    val_loss = hist.history['val_loss'][-1]
    max_auroc = max(hist.history['val_auroc'])
    last_auroc = hist.history['val_auroc'][-1]
    
    records_filter.append({
        "name": name,
        "Max Train Accuracy": round(max_train_accuracy, 4),
        "Max Validation Accuracy": round(max_val_accuracy, 4),
        "Last Train Acc": round(last_train_accuracy, 4),
        "Last Validation Acc": round(last_val_accuracy, 4),
        "Train Loss": round(train_loss, 4),
        "Validation Loss": round(val_loss, 4),
        "Max Train AUC": round(max_auroc, 4),
        "Last Validation AUC": round(last_auroc, 4)        
    })
metrics_df_filter = pd.DataFrame(records_filter)
metrics_df_filter


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


# Training loss vs Validation Loss
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(best_filter_history.history['loss'], label='Training Loss')
plt.plot(best_filter_history.history['val_loss'], label='Validation Loss')
plt.title(f"Training vs Validation Loss  \n(Best Num of Filters {best_filter_config}")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Training Accuracy vs Validation Accuracy
plt.subplot(1, 3, 2)
plt.plot(best_filter_history.history['accuracy'], label='Training Accuracy')
plt.plot(best_filter_history.history['val_accuracy'], label='Validation Accuracy')
plt.title(f"Training vs Validation Accuracy \n(Best Num of Filters {best_filter_config}")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# AUC 
plt.subplot(1, 3, 3)
plt.plot(best_filter_history.history['auroc'], label='Training AUC')
plt.plot(best_filter_history.history['val_auroc'], label='Validation AUC')
plt.title(f"Training vs Validation AUC \n(Best Num of Filters {best_filter_config}")
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



import tensorflow as tf

# default lr = 0.001
optimizer = tf.keras.optimizers.Adam()
default_lr = optimizer.learning_rate.numpy()
print(f"Default learning rate: {default_lr}")


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

# increase number of epoch to 50 as learning rate decreases
learning_rates_to_epoch = {
    5e-4: 20,   # (lower learning rate: 0.0005)
    5e-3: 10    # (higher learning rate=0.005)
}

history_dict_lr = {} 
history_dict_lr[default_lr] = five_layer_history
best_lr_validation_accuracy = five_layer_history.history['val_accuracy'][-1]
best_lr = default_lr
best_lr_history = five_layer_history

for lr, num_epochs in learning_rates_to_epoch.items():
    print(f"\nTraining Model with Learning Rate: {lr}...")
    
    model_lr_tuning = Sequential([
        Conv2D(f1, kernel_size=5, strides=1, padding='same', input_shape=(96, 96, 3)),
        BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),
    
        Conv2D(f2, kernel_size=3, strides=1, padding='same'),
        BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),
    
        Conv2D(f3, kernel_size=3, strides=1, padding='same'),
        BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),
    
        Conv2D(f4, kernel_size=3, strides=1, padding='same'),
        BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),
    
        Conv2D(f5, kernel_size=3, strides=1, padding='same'),
        BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),
            
        # Output layer
        Flatten(),Dropout(0.5),
        Dense(256, activation='relu'),Dropout(0.3), Dense(1, activation='sigmoid') 
    ])


    model_lr_tuning.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy', AUC(name='auroc')])

    # Early stopping and keep best result
    early_stopping = EarlyStopping(monitor='val_auroc', patience=3, mode='max', restore_best_weights=True)
    history_lr_turning = model_lr_tuning.fit(train_dataset, validation_data=validation_dataset,epochs=num_epochs, callbacks=[early_stopping])
    history_dict_lr[lr] = history_lr_turning
    final_lr_validation_accuracy =  history_lr_turning.history['val_accuracy'][-1]
    print(f"Final Validation Accuracy for {lr}: {final_lr_validation_accuracy:.4f}")
    
    if final_lr_validation_accuracy > best_lr_validation_accuracy:
        best_lr_validation_accuracy = final_lr_validation_accuracy
        best_lr = lr
        best_lr_history = history_lr_turning

print("Best Learing Rate Configuration:")
print(f"Learning Rate: {best_lr}")
print(f"Validation Accuracy: {best_lr_validation_accuracy:.4f}")


# Table of performance comparison
records_lr = []
for name, hist in history_dict_lr.items():
    max_train_accuracy = max(hist.history['accuracy'])
    max_val_accuracy = max(hist.history['val_accuracy'])
    last_train_accuracy = hist.history['accuracy'][-1]
    last_val_accuracy = hist.history['val_accuracy'][-1]
    train_loss = hist.history['loss'][-1]
    val_loss = hist.history['val_loss'][-1]
    max_auroc = max(hist.history['val_auroc'])
    last_auroc = hist.history['val_auroc'][-1]    
    records_lr.append({
        "Learning Rate": name,
        "Max Training Accuracy": round(max_train_accuracy, 4),
        "Max Validation Accuracy": round(max_val_accuracy, 4),
        "Last Train Acc": round(last_train_accuracy, 4),
        "Last Val Acc": round(last_val_accuracy, 4),
        "Train Loss": round(train_loss, 4),
        "Val Loss": round(val_loss, 4),
        "Max Val AUC": round(max_auroc, 4),
        "Last Val AUC": round(last_auroc, 4)                
    })
metrics_df_lr = pd.DataFrame(records_lr)
metrics_df_lr


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


#putting everything together for kaggle submission
# Model Architecture: 5 layers
# Learning rate: default (0.001)
# Best filter configuration: 32, 64, 128, 256, 512
model_best = Sequential([
    Conv2D(32, kernel_size=5, strides=1, padding='same', input_shape=(96, 96, 3)),
    BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),

    Conv2D(64, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),

    Conv2D(128, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),

    Conv2D(256, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),

    Conv2D(512, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),
        
    # Output layer
    Flatten(),Dropout(0.5),
    Dense(256, activation='relu'),Dropout(0.3), Dense(1, activation='sigmoid') 
])

# default Learning rate =0.001
model_best.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', AUC(name='auroc')])
checkpoint = ModelCheckpoint('model_best.h5', monitor='val_auroc', save_best_only=True, mode='max')

early_stopping = EarlyStopping(monitor='val_auroc', patience=3, mode='max', restore_best_weights=True)
history_best_model = model_best.fit(train_dataset, validation_data=validation_dataset,epochs=num_epochs, callbacks=[early_stopping, checkpoint])



from tensorflow.keras.models import load_model
import pandas as pd
import os
#model_loaded = load_model("model_best.h5", compile=False)

# Load test dataset and generate predictions

test_dir = Path("/kaggle/input/histopathologic-cancer-detection/test")

test_df = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/sample_submission.csv')  
test_df['test_filepath'] = test_df['id'].apply(lambda x: str(test_dir / f'{x}.tif'))

test_dataset = tf.data.Dataset.from_tensor_slices((test_df['test_filepath'], test_df['label']))
test_dataset = test_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(64).prefetch(tf.data.AUTOTUNE)


predictions = model_best.predict(test_dataset, verbose=1)
test_df['label'] = predictions.flatten()

# Save submission file
submission_path = 'submission.csv'
test_df[['id', 'label']].to_csv(submission_path, index=False)
print(f"Submission file saved to {submission_path}")



submission_path = '/kaggle/working/submission.csv'
test_df[['id', 'label']].to_csv(submission_path, index=False)
print(f"Submission file saved to {submission_path}")


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix

y_true_best_model = []
for _, labels in validation_dataset:
    y_true_best_model.extend(labels.numpy())
y_true_best_model = np.array(y_true_best_model)
y_pred_prob_best_model = model_best.predict(validation_dataset).flatten()
y_pred_best_model = (y_pred_prob_best_model >= 0.5).astype(int)

cm = confusion_matrix(y_true_best_model, y_pred_best_model)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])

plt.title('Confusion Matrix (Validation DataSet)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("\nConfusion Matrix Metrics:")
print(f"True Negatives (TN): {cm[0,0]}")
print(f"False Positives (FP): {cm[0,1]}")
print(f"False Negatives (FN): {cm[1,0]}")
print(f"True Positives (TP): {cm[1,1]}")


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix


# Training loss vs Validation Loss
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(history_best_model.history['loss'], label='Training Loss')
plt.plot(history_best_model.history['val_loss'], label='Validation Loss')
plt.title(f"Training vs Validation Loss (Best Model)")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Training Accuracy vs Validation Accuracy
plt.subplot(1, 3, 2)
plt.plot(history_best_model.history['accuracy'], label='Training Accuracy')
plt.plot(history_best_model.history['val_accuracy'], label='Validation Accuracy')
plt.title(f"Training vs Validation Accuracy (Best Model)")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# AUC 
plt.subplot(1, 3, 3)
plt.plot(history_best_model.history['auroc'], label='Training AUC')
plt.plot(history_best_model.history['val_auroc'], label='Validation AUC')
plt.title(f"Training vs Validation AUC (Best Model)")
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


