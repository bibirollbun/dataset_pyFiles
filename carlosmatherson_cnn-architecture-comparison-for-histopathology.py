# Basic Imports
import os
import numpy as np
import pandas as pd

# Plotting Tools
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks

# sklearn stuff
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


# setting random seed for use throughout the notebook
RANDOM_SEED = 19
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# check for available GPUs
gpus = tf.config.list_physical_devices('GPU')
print(f"Num GPUs Available: {len(gpus)}")


# set up strategy for use with two NVIDIA T4s, if available
if gpus:

    # configure memory growth to avoid memory allocation issues
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # setup MirroredStrategy on the T4s
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} T4 GPUs")
    
    # enable mixed precision for optimzied training
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

else: # use defaults
    strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    print("No GPUs available, using Kaggle standard CPU")
    tf.keras.mixed_precision.set_global_policy('float32')


# load data
TRAIN_DIR = '/kaggle/input/histopathologic-cancer-detection/train/'
TEST_DIR = '/kaggle/input/histopathologic-cancer-detection/test/'
TRAIN_LABELS_PATH = '/kaggle/input/histopathologic-cancer-detection/train_labels.csv'

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
print(f"Training labels shape: {train_labels.shape}")
print(train_labels.head())


# check class distribution
class_distribution = train_labels['label'].value_counts(normalize=True) * 100
print(f"\nClass distribution (%):\n{class_distribution}")


# sample images from each class
pos_samples = train_labels[train_labels['label'] == 1].sample(2)
neg_samples = train_labels[train_labels['label'] == 0].sample(2)
sample_df = pd.concat([pos_samples, neg_samples])

# add file extension to path & convert to string for Keras
sample_df['filename'] = sample_df['id'] + '.tif'
sample_df['label_str'] = sample_df['label'].astype(str)

# plot images
plt.figure(figsize=(15, 8))
for i, (_, row) in enumerate(sample_df.iterrows()):
    img_path = os.path.join(TRAIN_DIR, row['filename'])
    img = plt.imread(img_path)/255   
    plt.subplot(2, 4, i+1)
    plt.imshow(img)
    plt.title(f"Label: {row['label']}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# plot dist
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=train_labels)
plt.title('Training Labels Distribution')
plt.xlabel('Label (0 = No Cancer, 1 = Cancer)')
plt.ylabel('Count')
plt.show()


train_data = train_labels.copy() # uncomment for full run

# add file extension to path & convert to string for Keras
train_data['filename'] = train_data['id'] + '.tif'
train_data['label_str'] = train_data['label'].astype(str)


# set batch size to match strategy
if gpus:
    replicas = strategy.num_replicas_in_sync
    BATCH_SIZE = 128 * replicas # should be fine
    print(f"Using batch size of {BATCH_SIZE} with {replicas} GPU(s)")
else:
    BATCH_SIZE = 64
    print(f"Using default batch size of {BATCH_SIZE}")

# size of the images
IMG_SIZE = (32, 32)


def extract_center(img):
    # center coordinates
    h, w = img.shape[0], img.shape[1]
    center_h, center_w = h // 2, w // 2
    offset = 16  # 32/2 pixels
    
    # get center 32x32 region
    center_img = img[center_h-offset:center_h+offset, center_w-offset:center_w+offset, :]
    return center_img


datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=extract_center,
    validation_split=0.2
)


train_generator = datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=TRAIN_DIR,
    x_col='filename',
    y_col='label_str',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

print(f"Training generator batches: {len(train_generator)}")


validation_generator = datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=TRAIN_DIR,
    x_col='filename',
    y_col='label_str',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)


print(f"Validation generator batches: {len(validation_generator)}")


# calculate steps for training
steps_per_epoch = int(np.ceil(train_generator.samples / train_generator.batch_size))
validation_steps = int(np.ceil(validation_generator.samples / validation_generator.batch_size))

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")


with strategy.scope():
    def create_baseline_model():
        model = models.Sequential([
            
            # 1st convolutional block
            layers.Input(shape=(32, 32, 3)),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            
            # 2nd convolutional block
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            
            # 3rd convolutional block
            layers.Conv2D(256, (3, 3), padding='same'),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            
            # flatten and dense layers
            layers.Flatten(),
            layers.Dense(256),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            
            layers.Dense(1, activation='sigmoid', dtype='float32')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
    
    baseline_model = create_baseline_model()
    baseline_model.name = "Baseline"  

print("Baseline CNN Model Summary:")
baseline_model.summary()


with strategy.scope():
    def create_batchnorm_model():
        model = models.Sequential([
            
            # 1st convolutional block with batch normalization
            layers.Input(shape=(32, 32, 3)),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            
            # 2nd convolutional block with batch normalization
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            
            # 3rd convolutional block with batch normalization
            layers.Conv2D(256, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            
            # flatten and dense layers
            layers.Flatten(),
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),

            layers.Dense(1, activation='sigmoid', dtype='float32')  
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
    
    batchnorm_model = create_batchnorm_model()
    batchnorm_model.name = "BatchNorm"  

print("BatchNorm CNN Model Summary:")
batchnorm_model.summary()


# callbacks 
callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    ),
    # save best model
    callbacks.ModelCheckpoint(
        filepath='best_baseline_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# callbacks for the BatchNorm model
batchnorm_callbacks = callbacks_list.copy()

# ModelCheckpoint for BatchNorm
batchnorm_callbacks[-1] = callbacks.ModelCheckpoint(
    filepath='best_batchnorm_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)


# train baseline model 
print("\nTraining Baseline CNN Model...")
baseline_history = baseline_model.fit(
    train_generator,
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks_list,
    verbose=1
)


# train batch normalization model
print("\nTraining BatchNorm CNN Model...")
batchnorm_history = batchnorm_model.fit(
    train_generator,
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=batchnorm_callbacks,
    verbose=1
)


# load models & evaluate on validation data
baseline_model = tf.keras.models.load_model('/kaggle/working/best_baseline_model.keras')
batchnorm_model = tf.keras.models.load_model('/kaggle/working/best_batchnorm_model.keras')

baseline_results = baseline_model.evaluate(validation_generator, verbose=1)
batchnorm_results = batchnorm_model.evaluate(validation_generator, verbose=1)

metric_names = baseline_model.metrics_names
metric_names[1] = 'accuracy'


# vizz
def visualize_model_performance(model, generator, results):
        
    generator.reset()
    prediction_batch_size = BATCH_SIZE * 2
    prediction_steps = (generator.samples + prediction_batch_size - 1) // prediction_batch_size
    
    # new generator with larger batch size for prediction
    temp_generator = datagen.flow_from_dataframe(
        dataframe=train_data,
        directory=TRAIN_DIR,
        x_col='filename',
        y_col='label_str',
        target_size=IMG_SIZE,
        batch_size=prediction_batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    
    # predict
    y_pred = model.predict(temp_generator, steps=prediction_steps, verbose=1)
    y_pred_classes = (y_pred > 0.5).astype(int).flatten()
    
    # get labels
    y_true = generator.classes[:len(y_pred_classes)]
    
    # confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Non-tumor', 'Tumor'],
        yticklabels=['Non-tumor', 'Tumor']
    )
    plt.title(f'Confusion Matrix - {model.name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # ROC AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model.name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes))
    
    return roc_auc


# Get additional metrics (AUC)
baseline_auc = visualize_model_performance(baseline_model, validation_generator, baseline_results)
batchnorm_auc = visualize_model_performance(batchnorm_model, validation_generator, batchnorm_results)


# results table
comparison_df = pd.DataFrame({
    'Model': ['Baseline CNN', 'BatchNorm CNN'],
    'Validation Loss': [baseline_results[0], batchnorm_results[0]],
    'Validation Accuracy': [baseline_results[1], batchnorm_results[1]],
    #'AUC': [baseline_results[2], batchnorm_results[2]],
    'AUC': [baseline_auc, batchnorm_auc]
})

print("\nModel Performance Comparison:")
print(comparison_df)


# compare metrics
metrics = ['Validation Loss', 'Validation Accuracy', 'AUC']
plt.figure(figsize=(12, 5))

for i, metric in enumerate(metrics):
    plt.subplot(1, 3, i+1)
    plt.bar(['Baseline', 'BatchNorm'], comparison_df[metric])
    plt.title(metric)
    plt.ylim(0 if metric == 'Validation Loss' else 0.5, 1.0)
    
    for j, v in enumerate(comparison_df[metric]):
        plt.text(j, v + 0.02, f"{v}", ha='center')

plt.tight_layout()
plt.show()


if baseline_results[1] > batchnorm_results[1]:
    best_model = baseline_model
    print(f"Using Baseline model for submission (accuracy: {baseline_results[1]})")
else:
    best_model = batchnorm_model
    print(f"Using BatchNorm model for submission (accuracy: {batchnorm_results[1]})")


# test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

# test file names DF
test_files = os.listdir(TEST_DIR)
test_df = pd.DataFrame({
    'id': [os.path.splitext(file)[0] for file in test_files],
    'filename': test_files
})


# make test data gen
submission_batch_size = BATCH_SIZE * 2 

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=TEST_DIR,
    x_col='filename',
    y_col=None,  # no labels for test data
    target_size=IMG_SIZE,
    batch_size=submission_batch_size,
    class_mode=None,
    shuffle=False
)


# predict
print("\nGenerating predictions for submission...")
predictions = best_model.predict(
    test_generator,
    verbose=1
)
predicted_classes = (predictions > 0.5).astype(int).flatten()


# save
submission_df = pd.DataFrame({
    'id': test_df['id'][:len(predicted_classes)],
    'label': predicted_classes
})

submission_path = 'submission.csv'
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(f"Sample of submission file:\n{submission_df.head()}")

